# Architecture

High-level mental model for `forge-alloc` — the design source-of-truth.
Companion to [`COMPOSITION_RECIPES.md`](COMPOSITION_RECIPES.md)
(caller-facing examples).

## Three layers, separable

```
                  ┌─────────────────────────────┐
   user code  →   │  Wrapped allocator type     │   ←  zero-cost composition
                  └──────┬──────────────────────┘
                         │
                         ↓
   Layer 3:   hardening wrappers (Canary, PoisonOnFree, Quarantine,
   forge-hardening    Statistics, Watermark, GuardPage, CacheJitter,
                   HugePageAligned, NumaLocal, SplitMetadata) +
                   Faulty (test-only fault injection)
                         │   wrap any Layer-2 (or each other)
                         ↓
   Layer 2:   layout primitives (BumpArena, SharedBumpArena, Slab,
   forge-layout       SizeClassed, StackAlloc, ExtendableSlab,
                   GenerationalSlab, SlabOwner/SlabRemote, WithFallback)
                         │   organize memory from a Layer-1 backing
                         ↓
   Layer 1:   backings (InlineBacked, MmapBacked, System)
   forge-backing      "where do bytes come from"
                         │
                         ↓
   Layer 0:   traits (Allocator, Deallocator, OsBacked, FixedRange,
   forge-core         FreelistProtection, ProtectFlags), NonZeroLayout,
                   StdCompat<A>
```

The crate boundaries match the layers. A user who only needs
`BumpArena<InlineBacked<N>>` depends on `forge-core` + `forge-backing` +
`forge-layout` and pays no compile-time cost for the hardening surface.
The `forge-alloc` meta-crate re-exports everything and is the
default entry point.

## Why three layers?

The spec calls this "pay-for-what-you-use security hardening":

- **Layer 1 backings** are the only place where memory provenance
  enters the system. A backing is responsible for the OS-level
  semantics (mmap'ing a region, owning the stack buffer's storage,
  forwarding to the global heap).
- **Layer 2 layout primitives** turn an opaque region of bytes into
  a structured allocator. They consume a `B: Allocator` (typically a
  backing) and impose discipline: bump, slab, LIFO stack, etc. Layout
  primitives have correctness behaviour but no security behaviour.
- **Layer 3 hardening wrappers** add ONE specific protection or
  observability behaviour. They forward `Allocator` / `Deallocator`
  to the wrapped allocator with their own side-effect interposed.
  Absent wrappers cost nothing — the cost shows up in the type.

A `Canary<Slab<T, MmapBacked>>` has a Canary in the type and pays for
its 8 bytes of pre/post sentinel per allocation. A plain
`Slab<T, MmapBacked>` does not, and there is no runtime check or
branch that says "if hardening then …".

## Key trait choices

### `Allocator` / `Deallocator` split

`core::alloc::Allocator` bundles allocation and deallocation. That
creates a structural problem for arena allocators: a
`Box<T, &Arena>` must carry the arena reference to satisfy the
deallocator bound at drop time, even though `Arena::deallocate` is a
no-op. `forge-alloc` splits the two:

- `Allocator` extends `Deallocator`. Every allocator can also free.
- A type can implement *only* `Deallocator` — typically as a ZST
  token (`BumpDeallocator<'a>`) tied to an arena's borrow. The token
  lives in `Box<T, BumpDeallocator<'a>>`'s allocator slot at zero
  runtime cost.

The arena lifetime is enforced by the borrow checker: `'a` ties the
token to the arena's borrow, so `Box::drop` can't outlive the arena.

### `NonZeroLayout`

The library's internal layout type. `size > 0`, `align` a power of
two. ZSTs are absorbed at the `StdCompat<A>` boundary — the
`allocator_api2::Allocator` adapter returns a properly-aligned
dangling pointer for ZST requests without ever calling into the inner
allocator. This eliminates a recurring class of "what does
`allocate(size=0)` even mean" bugs at the primitive layer.

### `FixedRange`

A marker trait for allocators whose entire address range is fixed at
construction. Implementing `FixedRange` enables two things:

- **Provenance-based routing** in `WithFallback<P, S>`. The router's
  `deallocate(ptr)` checks `primary.contains(ptr)` and dispatches
  accordingly.
- **Watermark threshold computation**. A `Watermark<I, H>` over a
  `FixedRange` allocator knows the capacity (the address range size)
  and can compute "fraction used" without asking the inner.

### `OsBacked`

A separate trait for allocators that own an OS-level mapping. Adds
three operations the hardening layer needs: `base_ptr()` +
`region_size()` (to install guard pages around a region),
`release_pages(ptr, size)` (to drop physical pages while keeping the
virtual reservation), and `protect(ptr, size, flags)` (to change
page permissions). Higher layers like `HugePageAligned` and `NumaLocal`
require `OsBacked` because they operate on the underlying mapping
directly, not on the slab/arena structure imposed by Layer 2.

### `FreelistProtection`

Pluggable integrity policy on `Slab` freelist links. With `M = NoProtection`
(the default), `sign` returns `0` and `verify` always succeeds — the
optimizer eliminates the calls entirely. With `M = SipHashMAC` (under
the `siphasher` feature) every link's `(next_idx, slot_addr)` pair is
authenticated, so freelist corruption from a heap disclosure causes a
verification failure on the next pop.

## Concurrency model

Most Layer-2 primitives are `!Sync`. The reason: a single-threaded
allocator can use `UnsafeCell` for the cursor / freelist-head /
frame-stack, letting `Allocator::allocate(&self)` mutate without
internal locking. `!Sync` enforces single-thread access at the type
level — no atomic ops, no contention, no lock fairness questions.

Cross-thread allocation lives in three primitives:

- `SharedBumpArena<B>`. `Send + Sync` because the cursor is an
  `AtomicUsize` with a CAS loop. No `reset` (would need exclusive
  access; getting it through `Arc::get_mut` requires waiting for all
  Arcs to drop, at which point you may as well rebuild).
- `SlabOwner<T, B>` + `SlabRemote<T, B>`. The owner is `!Sync` (lock-
  free local freelist via `UnsafeCell`); the remote handle is
  `Send + Sync` and routes frees through a `Mutex<VecDeque>` queue
  (lock-free MPSC ring in v1.0) that the owner drains on its next
  allocate. Batch policy is `Fixed(N)` or `Adaptive` (5-level stepped
  threshold).
- `GenerationalSlab<T, B>` for handle-based access where the underlying
  slot may be recycled — the generation counter on each handle catches
  use-after-recycle.

## Why the funny build conventions?

### `[lints] workspace = true` everywhere

The workspace root registers `cfg(kani)` as a known cfg name via
`check-cfg`. Each crate opts in with `[lints] workspace = true` so the
registration reaches every `#[cfg(kani)]` block. Without this, the
default-on `unexpected_cfgs` lint (since Rust 1.80) plus CI's
`RUSTFLAGS=-D warnings` would break on any crate that adds a Kani
proof harness.

### `default = ["std"]` with `no_std` fallback

The Layer-0 / Layer-2 surfaces are `no_std`-compatible. The Layer-1
`MmapBacked` / `System` and several Layer-3 wrappers (`GuardPage`,
`HugePageAligned`, `NumaLocal`, `SplitMetadata`) require the OS, so
they're `#[cfg(feature = "std")]`-gated. CI runs a separate
`cargo check --no-default-features` job per crate to catch accidental
`use std::...` slipping into the no-std subset.

### `cfg(target_has_atomic = "ptr")`

`SharedBumpArena` and `Watermark` use `AtomicUsize` cursors / counters,
which require pointer-sized atomic support. Single-core no-std targets
without those (most microcontrollers) skip the affected types
entirely; users on such targets must use `BumpArena` with explicit
ownership discipline rather than the atomic-cursor variant. All
observability counters across the family (`Slab::corruption_events`,
`AllocStats`, `ExtendableSlab::routing_failures`, …) are `AtomicUsize`
rather than `AtomicU64` so the crates compile on 32-bit bare-metal
targets that lack native 64-bit atomics.

## Where the unsafe is

The trait contracts (`unsafe trait Allocator`, `unsafe trait
Deallocator`, `unsafe trait OsBacked`) push the safety burden to
**implementors** — anyone writing a new primitive has to prove the
listed invariants. Callers of the trait methods see safe-looking
`allocate(&self, layout) -> Result<NonNull<[u8]>, AllocError>` and
`unsafe fn deallocate(&self, ptr, layout)` — the `unsafe` on
`deallocate` makes the caller-side ownership/discipline obligation
explicit.

The unsafe blocks themselves cluster around a few patterns:

- **`UnsafeCell` reads/writes** of the cursor / freelist-head /
  frame-stack. Sound because the wrapping type is `!Sync`.
- **`NonNull::add` and pointer arithmetic** in slot-pointer math.
  Sound because the slab/arena tracks capacity and rejects out-of-
  range indices before computing the pointer.
- **`read_unaligned` / `write_unaligned`** on `FreeLink` storage in
  free slots. Sound because slot alignment is at least
  `align_of::<FreeLink>()` and the slot's user payload is gone (this
  is a free slot, not a live one).
- **Platform syscalls** (`mmap`, `munmap`, `madvise`, `mprotect`,
  `mbind`, `getcpu`, `VirtualAlloc`, `VirtualFree`, `VirtualProtect`).
  Sound because the wrapping `MmapBacked` / `NumaLocal` enforces the
  invariants in their constructors.

MIRI runs on `forge-core` (full features) and the `no_std` subsets of
the higher layers in CI. Kani proofs live at the per-crate level
(currently `BumpArena` + `Slab`) and verify symbolic-input properties
of the unsafe arithmetic.
