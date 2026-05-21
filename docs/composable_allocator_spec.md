# Composable Allocator Primitives — Design Specification

**Project working title:** `forge-alloc`  
**Language:** Rust (stable via `allocator-api2`; nightly for M13 verification CI only)
**Status:** Pre-implementation specification  
**Revision:** 1.5

---

## 1. Problem Statement

General-purpose allocators (mimalloc, jemalloc, tcmalloc) optimize for the average case across all workloads. Application hot paths have *specific* allocation patterns — known object sizes, bounded lifetimes, typed ownership — that a general allocator cannot exploit. The result is unnecessary heap round-trips, fragmentation, and uniform security overhead applied regardless of whether a given allocation is sensitive.

The alternative — writing one-off custom allocators per project — produces duplicated, untested code with no shared safety guarantees.

This library occupies the middle ground: a small set of well-tested, composable allocation primitives that snap together via the Rust `Allocator` trait to produce application-specific allocators with zero runtime dispatch overhead, compile-time-enforced guarantees, and pay-for-what-you-use security hardening.

**Immediate target use cases driving the design:**

- Hot-path slab pool for typed trading objects (bars, ticks, snapshots) in a latency-sensitive Rust trading engine
- Query-scoped bump arena for key encoding scratch buffers in a database storage engine
- Hardened PHI-handling buffers in a medical billing SaaS (poison-on-free, quarantine)
- Per-parse arena for short-lived AST nodes in a tree-sitter rewrite

---

## 2. Design Principles

**P1 — Zero-cost at the type level.** Composition is expressed through generic type parameters. In release builds, the optimizer sees through wrapper layers and eliminates abstraction overhead. There is no virtual dispatch.

**P2 — Pay for what you use.** Security hardening primitives (canaries, guard pages, freelist MAC, quarantine) impose exactly zero cost when absent from the type. The hot path trading allocator and the hardened PHI allocator are different types, not different runtime configurations.

**P3 — Primitives are narrow.** Each primitive does one thing. A bump arena does not manage free lists. A slab does not know about canaries. Hardening wrappers do not know about layout. Composition is the user's responsibility, not the library's.

**P4 — Backing is explicit.** Where memory comes from is always visible in the type. There is no implicit global heap fallback unless `WithFallback<Inner, System>` is explicitly chosen.

**P5 — Safe Rust surface, unsafe internals.** Public APIs are safe. Unsafe blocks are contained, documented, and justified per the standard Rust safety model.

**P6 — Platform-aware but not platform-dependent.** Hardware features (MTE on ARM, MPK on x86-64, PAC on arm64e) are expressed as backing variants and are conditionally compiled. The composition model works identically across platforms; only the backing primitive differs.

**P7 — `no_std` by default where possible.** Primitives with no OS dependency (`InlineBacked`, `BumpArena`, `Slab`, `Canary`, `PoisonOnFree`, `FreelistMAC`, `Quarantine`) compile under `#![no_std]` with only the `alloc` crate. OS-dependent primitives (`MmapBacked`, `GuardPage`, `NumaLocal`, hardware-backed variants) are gated behind the `std` feature flag. This enables the same crate to be used in kernel modules, WASM, and UEFI firmware without forking.

**P8 — Allocation failure is always explicit.** Every allocation returns `Result<_, AllocError>`. No primitive panics on OOM. Callers choose their failure policy — fail-fast, fallback, backpressure, or graceful degradation — at the composition site, not inside the primitive.

---

## 3. Architecture: Three-Layer Model

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — Hardening Wrappers                               │
│  Canary │ GuardPage │ PoisonOnFree │ Quarantine             │
│  Statistics │ Watermark │ SplitMetadata                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — Layout Primitives                                │
│  Slab<T, B, M=NoProtection> │ BumpArena │ SharedBumpArena   │
│  ExtendableSlab │ GenerationalSlab │ StackAlloc │ SizeClassed│
│  SlabOwner/SlabRemote (MessagePassing)                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — Backing Primitives                               │
│  MmapBacked │ InlineBacked<N> │ System                      │
│  HugePageAligned │ NumaLocal │ CacheJitter                  │
│  MteBacked (ARM) │ MpkBacked (x86)                         │
├─────────────────────────────────────────────────────────────┤
│  Foundation — Trait Contracts                               │
│  Allocator │ Deallocator │ NonZeroLayout                    │
│  OsBacked │ FixedRange │ FreelistProtection                  │
└─────────────────────────────────────────────────────────────┘
```

**Key architectural decisions reflected here:**
- `FreelistMAC` is a `FreelistProtection` policy parameter on `Slab`, not a wrapper
- `BumpArena` (non-atomic) and `SharedBumpArena` (atomic) are distinct types
- `MessagePassingSlab` is replaced by the `SlabOwner`/`SlabRemote` pair
- `GenerationalSlab` uses interleaved `GenerationalSlot<T>` storage
- `AllocContext` is deferred to v2.0; all v1.0 allocators use `NoContext`

Each layer consumes the layer below via the `Allocator` trait. A complete allocator is a type expression that traverses all three layers:

```rust
// Trading hot path: inline-backed slab, no hardening
type BarPool = Slab<Bar, InlineBacked<65536>>;

// Database query scratch: bump arena, fallback to global on overflow
type QueryScratch = WithFallback<BumpArena<InlineBacked<65536>>, System>;

// PHI buffer: mmap-backed, poison on free, quarantine 16 epochs
type ClaimBuffer = PoisonOnFree<Quarantine<Slab<ClaimRecord, MmapBacked>, 16>>;

// Debug-only canary wrapping in debug builds
#[cfg(debug_assertions)]
type ParseArena = Canary<BumpArena<InlineBacked<131072>>>;
#[cfg(not(debug_assertions))]
type ParseArena = BumpArena<InlineBacked<131072>>;

// ARM production: MTE-tagged backing, slab on top
#[cfg(all(target_arch = "aarch64", feature = "mte"))]
type SecureSlab<T> = Slab<T, MteBacked>;
```

---

## 4. Core Trait Contracts

### 4.1 `Allocator` / `Deallocator` Split

The standard `Allocator` trait bundles allocation and deallocation into a single trait. This creates a structural problem for arena-style allocators: a `Box<T, &BumpArena>` must carry the arena reference to satisfy the `Deallocator` bound at drop time, even though `BumpArena::deallocate` is a no-op. For non-ZST allocators this inflates every allocated container by one pointer.

The library defines a split:

```rust
/// Implemented by all allocators. Satisfies drop-time deallocation.
/// For arenas, this is a no-op — reclaim happens via reset().
/// For slabs, this pushes the block onto the internal free list.
pub unsafe trait Deallocator {
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout);
}

/// Extends Deallocator with allocation. All layout primitives implement this.
pub unsafe trait Allocator: Deallocator {
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError>;

    // Provided methods
    fn allocate_zeroed(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> { ... }
    unsafe fn grow(&self, ptr: NonNull<u8>, old: NonZeroLayout, new: NonZeroLayout)
        -> Result<NonNull<[u8]>, AllocError> { ... }
    unsafe fn shrink(&self, ptr: NonNull<u8>, old: NonZeroLayout, new: NonZeroLayout)
        -> Result<NonNull<[u8]>, AllocError> { ... }

    fn reset(&mut self) -> Result<(), AllocError> { Err(AllocError) }
    /// # Safety
    /// `ptr` must come from this allocator with `layout` as its original allocation
    /// layout — the implementor must trust the (ptr, layout) pair to avoid UB.
    unsafe fn usable_size(&self, ptr: NonNull<u8>, layout: NonZeroLayout) -> Option<usize> { None }
    fn capacity_bytes(&self) -> Option<usize> { None }
}
```

**`reset()` design note:** Placing `reset()` on `Allocator` with a default `Err(AllocError)` is a pragmatic compromise. The semantically correct design is a separate `Reset` trait implemented only by arenas:

```rust
pub trait Reset {
    fn reset(&mut self);  // infallible for BumpArena; not exposed on Slab at all
}
```

This would prevent callers from attempting `slab.reset()` (which has no meaning) and make BumpArena's reset correctly infallible rather than returning `Result`. The `Reset` trait is specced here as a v2.0 candidate. v1.0 keeps `reset()` on `Allocator` with `Err` default to minimize trait proliferation during the initial stabilization cycle. Implementors of arena-style allocators should implement `Reset` directly once it exists; the `Allocator::reset` default should not be relied upon.

**Interior mutability decision (Gate 2):** `deallocate` takes `&self` to match `std::alloc::Allocator` and allow deallocation through shared references (required by `Box`, `Arc`, `WithFallback`). `Slab` achieves mutation through `UnsafeCell<*mut FreeBlock>` for `free_head`. `Slab` is therefore `!Sync` — concurrent access to the free list without synchronization is UB. This is enforced at the type level. Multi-threaded deallocation uses `SlabRemote` (§6.7), which pushes to a separate lock-free queue rather than touching `free_head` directly.

**`BumpArena` implements both:** `Deallocator::deallocate` is a no-op; `Allocator::allocate` bumps the cursor. A `Box<T, BumpDeallocator>` carries a ZST and compiles to a single pointer:

```rust
/// Zero-sized deallocator token for arenas. Holds a lifetime only.
/// Prevents the arena from being dropped while any Box<T, BumpDeallocator<'_>> is live.
pub struct BumpDeallocator<'a>(PhantomData<&'a ()>);

unsafe impl Deallocator for BumpDeallocator<'_> {
    unsafe fn deallocate(&self, _: NonNull<u8>, _: NonZeroLayout) {}
}
```

**`Slab<T>` implements both meaningfully:** `deallocate` pushes to the free list via `UnsafeCell`; `allocate` pops or carves.

This split mirrors the proposal in the Rust allocator working group (wg-allocators issue #112) and the Linux kernel allocator design.

---

### 4.2 `NonZeroLayout`

Every allocator implementation must handle zero-sized allocations because `std::alloc::Layout` permits size zero. The handling is always the same — return a dangling non-null pointer, skip the deallocation — but every implementor writes this boilerplate independently, which is both tedious and a source of subtle bugs (e.g., incorrectly freeing a dangling ZST pointer).

The library adopts `NonZeroLayout` as its internal contract, with a shim at the `std::alloc::Allocator` boundary:

```rust
/// Layout guaranteed to have size > 0 and power-of-two alignment.
#[derive(Copy, Clone)]
pub struct NonZeroLayout {
    size: NonZeroUsize,
    align: NonZeroUsize,  // invariant: align.is_power_of_two()
}

impl NonZeroLayout {
    pub fn new(size: NonZeroUsize, align: NonZeroUsize) -> Result<Self, LayoutError>;
    pub fn for_type<T>() -> Option<Self>;  // None for ZSTs
    pub fn size(&self) -> NonZeroUsize;
    pub fn align(&self) -> NonZeroUsize;
    pub fn pad_to_align(&self) -> Self;
}
```

The `std::alloc::Allocator` compatibility shim intercepts zero-sized layouts at the trait boundary before they reach any primitive:

```rust
/// Blanket impl: wraps any forge-alloc Allocator as a std::alloc::Allocator.
/// Handles ZSTs at the boundary so primitives never see size == 0.
unsafe impl<A: crate::Allocator> std::alloc::Allocator for StdCompat<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match NonZeroLayout::try_from(layout) {
            Ok(nzl) => self.inner.allocate(nzl),
            Err(_)  => Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0)),
        }
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if let Ok(nzl) = NonZeroLayout::try_from(layout) {
            self.inner.deallocate(ptr, nzl);
        }
        // ZST: no-op, ptr is dangling
    }
}
```

All standard library collections (`Vec`, `HashMap`, `Box`, etc.) use `StdCompat<A>` as the allocator parameter, gaining ZST safety automatically.

---

### 4.3 `AllocContext` — Deferred to v2.0

Per-call allocation context (urgency, NUMA preference, compliance flags) was evaluated for v1.0. The extraction mechanism requires a `HasXxxContext` trait per context type with blanket tuple implementations — sound but adds implementation complexity not justified by v1.0 use cases.

**Decision:** All v1.0 allocators use an implicit `NoContext`. The `Allocator` trait has no `Context` associated type in v1.0. `NumaLocal` sets NUMA binding at construction time (covers 95% of the trading engine use case). `Watermark` fires on all allocations regardless of urgency — the handler decides how to respond.

**v2.0 extension point:** The `Allocator` trait will gain `type Context: Copy + Default` and `fn allocate_with_context(...)` as a non-breaking addition. Existing implementations gain the default via `type Context = NoContext`. Library-provided contexts: `NumaContext`, `UrgencyContext`. Composition via extraction traits: `HasNumaContext`, `HasUrgencyContext` with blanket tuple impls.

This is a known limitation for v1.0: a single `Watermark`-wrapped allocator cannot distinguish hot-path from cold-path allocations at call time. Workaround: use separate allocator instances for hot and cold paths, each with their own `Watermark` thresholds.

---

### 4.4 `std::alloc::Allocator` Interoperability

**`allocator-api2` as the stable bridge (Gate 10 sub-decision):** Rather than gating per-collection allocator usage behind a `nightly` feature flag, the library depends on the `allocator-api2` crate. This crate mirrors the nightly `std::alloc::Allocator` types and traits verbatim. When `allocator_api` stabilizes in the standard library, `allocator-api2` automatically re-exports from `core`/`alloc` with no API change on the library's side. This gives stable Rust users full `Vec<T, A>`, `Box<T, A>`, `HashMap<K, V, A>` support today, removes the `nightly` feature flag entirely, and provides a zero-friction migration path to the standard when stabilization lands.

```
forge_alloc::Allocator  ──(via StdCompat<A>)──►  allocator_api2::Allocator
                                                          │
                                              Vec<T, StdCompat<A>>      (stable)
                                              Box<T, StdCompat<A>>      (stable)
                                              HashMap<K, V, StdCompat<A>> (stable)
```

When `allocator_api` stabilizes:
```
allocator_api2::Allocator  ──re-exports──►  core::alloc::Allocator
                                            (zero changes to forge-alloc)
```

The `nightly` feature flag is removed from the library entirely. Users on stable get full functionality. The `StdCompat<A>` shim applies to `allocator_api2::Allocator` instead of `std::alloc::Allocator` — same ZST handling, same blanket impl pattern.

---

### 4.5 Foundation Traits: `OsBacked`, `FixedRange`, `FreelistProtection`

Three additional traits referenced throughout the spec are defined here.

**`OsBacked`** — marks allocators that manage OS-level memory mappings. Required by `GuardPage`, `HugePageAligned`, `NumaLocal`, and `MpkBacked`.

```rust
pub unsafe trait OsBacked: Allocator {
    fn base_ptr(&self) -> NonNull<u8>;
    fn region_size(&self) -> usize;

    /// Release physical pages in [ptr, ptr+size) back to the OS.
    /// Virtual address range remains reserved.
    unsafe fn release_pages(&self, ptr: NonNull<u8>, size: usize);

    /// Change memory protection flags on [ptr, ptr+size).
    unsafe fn protect(&self, ptr: NonNull<u8>, size: usize, flags: ProtectFlags);
}

/// `#[non_exhaustive]` so future hardware-protection bits (PROT_MTE,
/// PROT_GROWSDOWN, MPK key index) can be added without an API break.
/// Callers must construct via the provided constants or via the `RW`/`READ`-
/// style builders — never with a struct literal.
#[non_exhaustive]
pub struct ProtectFlags {
    pub read: bool,
    pub write: bool,
    pub exec: bool,
}

impl ProtectFlags {
    pub const NONE: Self  = Self { read: false, write: false, exec: false };
    pub const RW: Self    = Self { read: true,  write: true,  exec: false };
    pub const READ: Self  = Self { read: true,  write: false, exec: false };
    pub const RX: Self    = Self { read: true,  write: false, exec: true  };
}
```

Implemented by: `MmapBacked`, `MteBacked`, `MpkBacked`, `HugePageAligned`. Not implemented by `InlineBacked` or `System` — applying `GuardPage<InlineBacked<N>>` is a compile error.

---

**`FixedRange`** — marks allocators whose address range does not change after construction. Required by `WithFallback<P: FixedRange, S>` to enable correct deallocation routing. Bounded on `Allocator` because every concrete `FixedRange` type in this library is also an `Allocator` — the supertrait makes pointer-provenance routing express both "where the range is" and "how to deallocate within it" in one bound.

```rust
pub trait FixedRange: Allocator {
    fn base(&self) -> NonNull<u8>;
    fn size(&self) -> usize;

    /// Implemented as `(p - base) < size` using wrapping subtraction so the
    /// check remains correct when the region's end address wraps past
    /// `usize::MAX` (rare on 64-bit, but possible on 16-/32-bit no_std targets).
    fn contains(&self, ptr: NonNull<u8>) -> bool {
        let base = self.base().as_ptr() as usize;
        let p    = ptr.as_ptr() as usize;
        p.wrapping_sub(base) < self.size()
    }
}
```

Implemented by: `BumpArena<InlineBacked<N>>`, `BumpArena<MmapBacked>` (fixed at construction), non-growing `Slab`. Not implemented by growing `Slab`, `ExtendableSlab`, or `System`.

---

**`FreelistProtection`** — policy parameter on `Slab` controlling freelist integrity. Zero-overhead when `M = NoProtection`.

**Freelist design:** Free list links use 1-based slot indices and a separate MAC field, avoiding the pointer-bits encoding that conflates the zero sentinel with signed values.

```rust
/// A free list link stored inside each free slot.
/// next_idx: 1-based index of the next free slot; 0 means end of list.
/// mac:      integrity check computed by FreelistProtection::sign().
///           NoProtection always writes 0. Field exists for alignment;
///           optimizer may elide in release builds when M = NoProtection.
struct FreeLink {
    next_idx: u32,
    mac: u32,
}

pub trait FreelistProtection {
    /// Compute a 32-bit integrity tag for a freelist link.
    /// `next_idx` is the 1-based index being stored; `slot_addr` is the
    /// virtual address of the slot holding the link (used as nonce).
    fn sign(&self, next_idx: u32, slot_addr: usize) -> u32;

    /// Verify the stored tag. Returns Err(FreelistCorruption) on mismatch.
    fn verify(&self, next_idx: u32, stored_mac: u32, slot_addr: usize)
        -> Result<(), FreelistCorruption>;
}

pub struct NoProtection;     // ZST — sign returns 0; verify always Ok(())

pub struct SipHashMAC {
    key: [u8; 16],           // initialized from OsRng at Slab construction
}
// sign:   SipHash13(key, next_idx as u64 || slot_addr as u64)[0..4] as u32
// verify: recompute and compare

#[cfg(all(target_arch = "aarch64", feature = "pac"))]
pub struct PacMAC;           // sign: PACIB(next_idx as *mut _, slot_addr); verify: AUTIB
```

**Push/pop with 1-based indices:**
```rust
// Push slot i (0-based) onto free list:
let next_idx = *free_head;                           // current head (1-based, 0 if empty)
let mac      = self.mac.sign(next_idx, slot_ptr);    // sign the index we're storing
slot_ptr.write(FreeLink { next_idx, mac });
*free_head   = i as u32 + 1;                        // store 1-based index

// Pop from free list:
let i = *free_head;           // 1-based
if i == 0 { return None; }
let link = slot_ptr_for(i - 1).read::<FreeLink>();
self.mac.verify(link.next_idx, link.mac, slot_ptr_for(i - 1))?;
*free_head = link.next_idx;
return Some(ptr_to_slot(i - 1));
```

`NoProtection` is the default: `Slab<T, B>` expands to `Slab<T, B, NoProtection>`.

---

## 5. Layer 1 — Backing Primitives

### 5.1 `InlineBacked<const N: usize>`

Fixed-size inline storage. No heap involvement. Suitable for stack-allocated arenas where N is known at compile time.

```rust
#[repr(C, align(16))]
pub struct InlineBacked<const N: usize> {
    storage: UnsafeCell<MaybeUninit<[u8; N]>>,
    cursor: UnsafeCell<usize>,
}
// Send is auto-derived (both UnsafeCell payloads are Send).
// NOT Sync — UnsafeCell on cursor prevents concurrent &self allocation.
```

**Guarantees:** allocation fails if `allocated + request > N`. No external memory. Drop is a no-op.

**Constraints:** `N` must be a multiple of `align_of::<usize>()`. Max alignment supported is 16 bytes (covers all standard Rust types). For larger alignments, use `MmapBacked`.

**Platform notes:** Works identically on all targets. On Apple Silicon, `N` should be a multiple of 16KB if the backing will be promoted to huge pages by the OS.

---

### 5.2 `MmapBacked`

Anonymous `mmap` (Linux/macOS) or `VirtualAlloc` (Windows). Returned to the OS on drop. Suitable for large arenas or any allocator where lifetime extends beyond the stack frame.

```rust
pub struct MmapBacked {
    ptr: NonNull<u8>,
    len: usize,
    cursor: UnsafeCell<usize>,
}
// Send: unsafe-impl (NonNull is !Send by default; the underlying mapping is
// thread-portable since munmap keys on (ptr, len), not thread identity).
// NOT Sync — UnsafeCell on cursor prevents concurrent &self allocation.
// Cross-thread bump allocation belongs to SharedBumpArena (§6.1).
```

After every failing OS-syscall (mmap / munmap / madvise / mprotect on
Unix, VirtualAlloc / VirtualFree / VirtualProtect on Windows), the
platform's errno / `GetLastError` is captured into a per-thread slot;
operators read it via `mmap_last_os_error() -> Option<io::Error>` to
distinguish ENOMEM / EACCES / EOVERFLOW / ERROR_NOT_ENOUGH_MEMORY etc.
without needing strace / ETW.

**Construction:**

```rust
impl MmapBacked {
    pub fn new(size: usize) -> Result<Self, AllocError>;
    pub fn with_huge_pages(size: usize) -> Result<Self, AllocError>;
    pub fn with_flags(size: usize, flags: MmapFlags) -> Result<Self, AllocError>;
}
```

**`MmapFlags`:**

```rust
pub struct MmapFlags {
    pub huge_pages: bool,          // MAP_HUGETLB / MAP_ANONYMOUS large pages
    pub populate: bool,            // MAP_POPULATE — fault all pages at alloc time
    pub numa_node: Option<u32>,    // mbind() to specific NUMA node post-alloc
    pub guard_at_end: bool,        // append one unmapped guard page
}
```

**Platform notes:**
- Linux: `mmap(MAP_ANONYMOUS | MAP_PRIVATE)`, optional `MAP_HUGETLB` for 2MB pages
- macOS/Apple Silicon: `mmap`, no `MAP_HUGETLB`; 16KB page granularity; 32MB block alignment for huge page promotion
- Windows: `VirtualAlloc(MEM_RESERVE | MEM_COMMIT)`, `MEM_LARGE_PAGES` for huge pages (requires privilege)
- NUMA: `mbind()` post-mmap on Linux; not available on macOS (UMA, irrelevant); `VirtualAllocExNuma` on Windows

---

### 5.3 `System`

Thin newtype over `std::alloc::System`. Used as a fallback backing in `WithFallback`. Not intended for direct use on hot paths.

---

### 5.4 `MteBacked` *(ARM, feature = "mte")*

`MmapBacked` variant with `PROT_MTE` flag set. Every 16-byte granule gets a hardware tag. The allocator assigns and rotates tags on each allocation; the hardware checks tags on every load/store.

```rust
#[cfg(all(target_arch = "aarch64", feature = "mte"))]
pub struct MteBacked {
    inner: MmapBacked,
    tag_counter: u8,  // cycles through 1..=15; tag 0 is reserved by the architecture
}
```

**Tag assignment formula:** `tag = (tag_counter % 15) + 1` — yields values 1 through 15, never 0. Increment `tag_counter` on each allocation. Sequential rotation ensures adjacent allocations always carry different tags, catching linear overflows deterministically. Random mode (using the ARM `IRG` instruction) gives ~93% single-execution UAF detection per Android's characterization.

**Compile-time gate:** Falls back to `MmapBacked` on non-MTE targets with no API change.

---

### 5.5 `MpkBacked` *(x86-64, feature = "mpk")*

`MmapBacked` variant that assigns an Intel MPK protection key to the region via a shared `MpkPool`. The pool manages key lifecycle with LRU rotation and generation tracking. See §9.6 for the full pool design including fork safety.

```rust
#[cfg(all(target_arch = "x86_64", feature = "mpk"))]
pub struct MpkBacked {
    inner: MmapBacked,
    pkey: i32,
    generation: u64,
    pool: Arc<Mutex<MpkPool>>,
}
```

**Constraint:** x86 provides 16 protection keys system-wide (2 reserved by OS = 14 available). Keys are leased from `MpkPool` — see §9.6 for exhaustion behavior and pool construction.

**Use case:** Appropriate for a small number of high-value allocation regions (e.g., the PHI buffer in ClaimMatch). The 14-key ceiling limits practical deployment to O(10) `MpkBacked` instances per process.

---

## 6. Layer 2 — Layout Primitives

### 6.1 `BumpArena<Backing>` and `SharedBumpArena<Backing>`

Two distinct types based on the thread-safety requirement. The distinction is at the type level — users choose once at the declaration site.

**`BumpArena<B>` — single-threaded, non-atomic cursor**

```rust
pub struct BumpArena<B: FixedRange> {
    backing: B,
    base: NonNull<u8>,
    end: NonNull<u8>,
    cursor: UnsafeCell<usize>,   // interior mutability for &self alloc
}
// impl Send for BumpArena<B> where B: Send
// NOT Sync — UnsafeCell on cursor blocks concurrent &self racing.
```

The bound is `B: FixedRange` (not just `Allocator`) — the arena
sub-allocates the entire address range directly and never calls
`B::allocate`, so it needs `base()` / `size()`. `B: FixedRange` already
implies `B: Allocator`.

**Allocation:** `O(1)` — increment `cursor`, align, bounds check.

**Free:** No-op. Individual deallocations are accepted (trait compliance) but reclaim nothing. This is fundamental to arena allocation — objects are not freed individually.

**Reset:** `fn reset(&mut self)` — restores cursor to 0, reclaiming all memory in O(1). Requires `&mut self`, enforcing exclusive ownership at call time. No runtime live-allocation check is possible: `deallocate` is a no-op, so `Statistics` tracking `allocations - deallocations` will always show a nonzero live count for an arena (every deallocation is silently discarded). **Correctness is the caller's responsibility.** Two enforcement options:
- **Borrow checker (Box-style):** Use `Box<T, BumpDeallocator<'a>>` — the `'a` lifetime ties the box to the arena's borrow, so the arena cannot be reset (or dropped) while any `Box` is live. The borrow checker enforces this at compile time.
- **Discipline (raw alloc):** For raw `allocate` calls that return `NonNull<u8>`, the caller must ensure all issued pointers are dead before calling `reset()`. No runtime check is available without external tracking.

```rust
let mut arena = BumpArena::<InlineBacked<65536>>::new();
// ... allocate scratch objects ...
arena.reset();  // O(1) — reclaims all memory at once
```

---

**`SharedBumpArena<B>` — multi-reader, atomic cursor**

```rust
/// Only available on targets with pointer-sized atomics.
/// Single-core no_std targets without atomics: use BumpArena exclusively.
#[cfg(target_has_atomic = "ptr")]
pub struct SharedBumpArena<B: Allocator> {
    backing: B,
    base: NonNull<u8>,
    end: NonNull<u8>,
    cursor: AtomicUsize,    // allows concurrent &self allocation
}
// impl Send + Sync for SharedBumpArena<B> where B: Send.
// The backing is sealed inside the arena — no `backing()` accessor — so its
// !Sync interior mutability (e.g. InlineBacked, MmapBacked's per-backing
// cursor) cannot be raced on. SharedBumpArena's own atomic cursor is the
// only mutable state reachable through &self.
```

**Allocation:** `O(1)` — atomic `fetch_add` plus alignment rounding.

**No `reset()`:** `SharedBumpArena` does not implement `reset()`. Getting `&mut self` through an `Arc<SharedBumpArena>` requires `Arc::get_mut()` which only succeeds when there are no other clones — at which point the arena can simply be dropped and recreated. Callers that need periodic reset should use `BumpArena` with explicit ownership discipline.

**Use cases:** Shared parse arenas where multiple tasks allocate concurrently; test scaffolding; any `Arc<SharedBumpArena>` pattern.

Both types implement `FixedRange` (address range is fixed at construction).

---

### 6.2 `Slab<T, Backing, M = NoProtection>`

Fixed-size typed block allocator. Maintains an offset-based free list of `T`-sized blocks. Allocation pops the free list; deallocation pushes. No fragmentation. `O(1)` both directions.

```rust
pub struct Slab<T, B: Allocator, M: FreelistProtection = NoProtection> {
    backing: B,
    mac: M,
    base: NonNull<u8>,
    free_head: UnsafeCell<u32>,  // 1-based slot index; 0 = list empty
    block_stride: usize,         // = max(size_of::<T>(), size_of::<FreeLink>())
    capacity: usize,             // number of T slots
    next_uncarved: u32,          // 0-based index of first never-used slot
    _phantom: PhantomData<T>,
}

// Interior mutability for &self deallocate — safe because Slab is !Sync.
// !Sync is the root mechanism: &Slab cannot cross thread boundaries,
// preventing concurrent calls to deallocate that would race on free_head.
unsafe impl<T: Send, B: Send + Allocator, M: Send + FreelistProtection> Send for Slab<T, B, M> {}
// NOT Sync
```

**1-based freelist index (fixes slot-0 sentinel collision):** `free_head` stores 1-based slot indices. Slot 0 (byte offset 0 from `base`) is stored as index 1, not 0. The value 0 is unambiguously "list empty" — no collision with any valid slot. `FreeLink.mac` is a separate field from `FreeLink.next_idx`, so MAC signing never encodes its output in pointer bits, eliminating the problem where a signed value with zero low bits would lose the link.

**Freelist protection (Gate 1 decision):** `mac: M` signs the `(next_idx, slot_address)` pair before writing and verifies before following. With `M = NoProtection`, sign returns 0 and verify is always Ok — zero overhead, optimizer eliminates them.

```rust
// Free list push (in deallocate) — slot_idx is 0-based:
let slot_idx   = (ptr.as_ptr() as usize - self.base.as_ptr() as usize) / self.block_stride;
let slot_addr  = ptr.as_ptr() as usize;
let old_head   = unsafe { *self.free_head.get() };
let mac        = self.mac.sign(old_head, slot_addr);
unsafe { ptr.cast::<FreeLink>().as_ptr().write(FreeLink { next_idx: old_head, mac }) };
unsafe { *self.free_head.get() = slot_idx as u32 + 1 };   // store 1-based

// Free list pop (in allocate):
let head = unsafe { *self.free_head.get() };
if head == 0 { /* carve from next_uncarved or return AllocError */ }
let slot_idx  = head - 1;   // convert to 0-based
let slot_ptr  = base + slot_idx * block_stride;
let link      = unsafe { slot_ptr.cast::<FreeLink>().as_ptr().read() };
self.mac.verify(link.next_idx, link.mac, slot_ptr as usize)?;
unsafe { *self.free_head.get() = link.next_idx };
```

**Construction:**

```rust
impl<T, B: Allocator, M: FreelistProtection> Slab<T, B, M> {
    /// Fixed capacity. Will not grow. Suitable for real-time and verified contexts.
    pub fn new(capacity: usize, backing: B, mac: M) -> Result<Self, AllocError>;
}

// Convenience constructors with NoProtection default
impl<T, B: Allocator> Slab<T, B> {
    pub fn unprotected(capacity: usize, backing: B) -> Result<Self, AllocError> {
        Self::new(capacity, backing, NoProtection)
    }
}
```

**`grow()` and `shrink()`:** Both return `AllocError`. `Slab` does not support in-place resize. For growable typed allocation use `ExtendableSlab` (§6.8).

**`FixedRange` implementation:** `Slab` implements `FixedRange` (base address fixed at construction) making it valid as `Primary` in `WithFallback`.

**`capacity_bytes()`:** Returns `Some(capacity * block_stride)`, enabling `Watermark` to compute correct percentages in byte units.

---

### 6.3 `SizeClassed<Backing, const CLASSES: usize>`

Array of `CLASSES` untyped slabs with geometrically increasing block sizes. Routes allocation requests to the smallest slab whose block size satisfies the request. Falls back to `Backing` directly for requests exceeding all size classes.

```rust
pub struct SizeClassed<B: Allocator, const CLASSES: usize> {
    slabs: [UntypedSlab; CLASSES],
    class_sizes: [usize; CLASSES],
    backing: B,
}

/// An erased Slab<u8, MmapBacked, NoProtection> with a runtime-specified block size.
/// Used internally by SizeClassed. Not part of the public API.
struct UntypedSlab {
    base: NonNull<u8>,
    free_head: UnsafeCell<u32>,  // 1-based slot index; 0 = list empty (matches Slab design)
    block_stride: usize,
    capacity: usize,
    next_uncarved: u32,
}
```

`UntypedSlab` is identical to `Slab<u8>` but with runtime block stride instead of a type-derived size. It is an internal implementation detail — users interact with `SizeClassed`, not `UntypedSlab` directly.

**Default size classes (8 classes):** 8, 16, 32, 64, 128, 256, 512, 1024 bytes, exported as `forge_layout::DEFAULT_CLASS_SIZES_8`.

**Use case:** Drop-in replacement for a general allocator where object sizes are bounded but not fully predictable. Less efficient than a typed `Slab<T>` but more efficient than `System`.

**v0.1 implementation note:** Construction allocates each class region from the backing with alignment equal to that class's stride. Backings with a smaller `MAX_ALIGN` (`InlineBacked` caps at 16) reject the request and `with_class_sizes` returns `Err(AllocError)` — use `MmapBacked` / `BumpArena<MmapBacked>` / `System` for class sizes exceeding 16 bytes. Construction also validates that class sizes are strictly increasing, powers of two, and ≥ `size_of::<FreeLink>() = 4` bytes.

---

### 6.4 `StackAlloc<Backing>`

LIFO discipline allocator. Tracks the last allocation; deallocation is only valid for the most recently allocated block. Panics in debug builds if out-of-order free is attempted.

```rust
pub struct StackAlloc<B: FixedRange> {
    backing: B,
    base: NonNull<u8>,
    capacity: usize,
    cursor: UnsafeCell<usize>,
    /// Frame stack: (aligned_off, prev_cursor) per live allocation. On
    /// deallocate we validate the supplied pointer matches the top of the
    /// stack and restore cursor to prev_cursor. Holds heap-Vec for the
    /// bookkeeping; user allocations still flow through `backing`.
    frames: UnsafeCell<Vec<(usize, usize)>>,
}
// Send if B: Send; NOT Sync (UnsafeCell on cursor + frames).
```

A single-frame `last_alloc` sentinel is **not** sufficient — nested
alloc/free with arbitrary depth requires a per-frame stack, otherwise
popping the innermost allocation also drops the bookkeeping for outer
allocations. The current implementation uses `Vec<(off, prev_cursor)>`.

**Use case:** Nested scope allocations where teardown order is the inverse of allocation order. Slightly cheaper than a bump arena for patterns that do reclaim memory but always in LIFO order.

---

### 6.5 `WithFallback<Primary, Secondary>`

Attempts allocation from `Primary`. On failure, falls back to `Secondary`. Deallocation is routed to whichever allocator owns the pointer via the `FixedRange` contract.

```rust
pub struct WithFallback<P: Allocator + FixedRange, S: Allocator> {
    primary: P,
    secondary: S,
}

unsafe impl<P: Allocator + FixedRange, S: Allocator> Deallocator for WithFallback<P, S> {
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        if self.primary.contains(ptr) {
            self.primary.deallocate(ptr, layout)
        } else {
            self.secondary.deallocate(ptr, layout)
        }
    }
}
```

**`FixedRange` bound (Gate 8 decision):** `Primary` must implement `FixedRange` — its address range cannot change after construction. This prevents the routing bug where a growing primary allocates outside its tracked range. Growing allocators (`ExtendableSlab`) cannot be used as `Primary`. If a growing primary is needed, use separate allocators and route at the application level.

**`Secondary` has no `FixedRange` requirement.** If a pointer belongs to neither allocator (a bug — e.g., a pointer from a third allocator), behavior in release builds is UB; debug builds with `Statistics` can catch this via allocation tracking.

**Primary use case:** `WithFallback<BumpArena<InlineBacked<N>>, System>` — stack-fast for the common case, global heap for overflow.

---

### 6.6 `GenerationalSlab<T, Backing>`

A typed allocator that returns `Handle<T>` values instead of raw pointers. Handles encode a slot index and a generation counter. Accessing a slot via a stale handle (generation mismatch) returns `None` rather than undefined behavior.

**Interleaved layout (Gate 4 decision):** Generation and state are colocated in a single `GenerationalSlot<T>` struct, backed by a single contiguous allocation. This avoids the dual-ownership problem of two separate `Vec`s and improves cache behavior for handle validation (one cache line fetch covers both the generation check and the value).

```rust
pub struct GenerationalSlab<T, B: Allocator> {
    backing: B,
    slots: NonNull<GenerationalSlot<T>>,
    capacity: usize,
    len: usize,
    free_head: Option<u32>,    // index into slots array
}

/// Single allocation holds generation + state contiguously.
struct GenerationalSlot<T> {
    generation: u32,
    state: SlotState<T>,
}

enum SlotState<T> {
    Occupied(T),
    Free { next_free: Option<u32> },
}

/// Stable, non-pointer handle. Cheap to copy, compare, store.
/// Does not keep the slab alive.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _phantom: PhantomData<*const T>,
}

impl<T, B: Allocator> GenerationalSlab<T, B> {
    pub fn new(capacity: usize, backing: B) -> Result<Self, AllocError>;

    /// Returns Err if slab is at capacity and backing cannot provide more memory.
    pub fn insert(&mut self, value: T) -> Result<Handle<T>, AllocError>;

    pub fn get(&self, handle: Handle<T>) -> Option<&T>;
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T>;
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T>;
    pub fn contains(&self, handle: Handle<T>) -> bool;
}
```

**ABA prevention:** On `remove`, `generation` at the slot is incremented. A stale handle with the old generation fails `slots[index].generation == handle.generation` and returns `None`.

**Generation width:** `u32` by default — wraps after 2³² reuses per slot. For long-running server processes with high-churn slabs, a `u64` generation counter eliminates wrap risk at the cost of doubling the handle size (8 bytes → 12 bytes with index). Exposed as a public type parameter:

```rust
pub struct GenerationalSlab<T, B: Allocator, G: GenerationInt = u32> { ... }
pub struct Handle<T, G: GenerationInt = u32> {
    index: u32,
    generation: G,
    _phantom: PhantomData<*const T>,
}

// Sealed trait: implemented for u32 and u64 only.
pub trait GenerationInt: Copy + Eq + sealed::Sealed { ... }

// Convenience aliases:
pub type GenerationalSlab32<T, B> = GenerationalSlab<T, B, u32>;  // default
pub type GenerationalSlab64<T, B> = GenerationalSlab<T, B, u64>;  // long-running servers
```

This is preferable to a separate type alias with a hidden internal parameter — the generation width is visible in the type and `Handle<T, u64>` is incompatible with `Handle<T, u32>` at compile time, preventing accidental cross-width handle usage.

**`insert()` return type correction:** Returns `Result<Handle<T>, AllocError>`, not `Handle<T>`. No infallible allocation exists in a bounded allocator. This is a hard requirement, not a design option.

---

### 6.7 `SlabOwner<T, B>` and `SlabRemote<T, B>`

Cross-thread typed allocation via the ownership-return model. Replaces the previously named `MessagePassingSlab`.

**Problem:** `Slab<T, B>` is `!Sync`. A thread other than the owner cannot call `deallocate` directly — it would race on `free_head`. The TCMalloc solution (global locked pool) reintroduces contention. The snmalloc/mimalloc solution: route freed pointers back to the owner via a lock-free queue.

**Design (Gate 3 decision):** Two types backed by the same `Arc<SlabInner<T, B>>`.

```rust
/// Owns the slab. Has exclusive allocate access.
/// Send but !Sync — can be moved between threads, not shared.
/// Implements Allocator (&self via UnsafeCell) so it composes with
/// Watermark, Statistics, and other Layer 3 wrappers.
pub struct SlabOwner<T, B: Allocator> {
    inner: Arc<SlabInner<T, B>>,
    batch_policy: BatchPolicy,
}

/// Remote deallocation handle. Send + Sync — freely cloneable across threads.
/// Implements Deallocator only — cannot allocate.
#[derive(Clone)]
pub struct SlabRemote<T, B: Allocator> {
    inner: Arc<SlabInner<T, B>>,
}

struct SlabInner<T, B: Allocator> {
    // Accessed only by SlabOwner via &self (UnsafeCell).
    // Safety: SlabOwner is !Sync — only one thread at a time holds
    // a &SlabOwner, so concurrent allocate calls are impossible.
    slab: UnsafeCell<Slab<T, B>>,
    remote_queue: RemoteFreeQueue,
}

/// Lock-free MPSC ring buffer for cross-thread freed 1-based slot indices.
/// Fixed capacity: 1024 entries by default, configurable at construction.
struct RemoteFreeQueue {
    slots: Box<[AtomicU32]>,   // ring buffer of 1-based slot indices
    head: AtomicU32,
    tail: AtomicU32,
}

impl<T, B: Allocator> SlabOwner<T, B> {
    pub fn new(capacity: usize, backing: B) -> Result<Self, AllocError>;
    pub fn with_batch_policy(capacity: usize, backing: B, policy: BatchPolicy)
        -> Result<Self, AllocError>;

    /// Create a remote handle. Cheap clone — increments Arc refcount only.
    pub fn remote(&self) -> SlabRemote<T, B>;

    /// Drain remote-free queue into local free list.
    /// Called automatically by allocate() when local list is empty.
    /// Can also be called explicitly at the top of allocation loops.
    pub fn drain(&self);   // &self via UnsafeCell access to inner slab
}

// SlabOwner implements Allocator via &self (UnsafeCell).
// !Sync prevents concurrent callers — the safety invariant is upheld by the type system.
unsafe impl<T, B: Allocator> Deallocator for SlabOwner<T, B> {
    // Owner-side deallocation: push directly to local free list (no queue needed).
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) { ... }
}

unsafe impl<T, B: Allocator> Allocator for SlabOwner<T, B> {
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: !Sync ensures no concurrent calls to allocate or owner-side deallocate.
        let slab = unsafe { &mut *self.inner.slab.get() };
        // Pull any remote-freed slots into the local free list before allocating.
        // maybe_drain consults self.batch_policy and is a no-op when the policy
        // threshold isn't met; the unconditional call on the empty-list path is
        // handled inside Slab::allocate_local. Both helpers are private.
        self.maybe_drain();
        slab.allocate_local(layout)
    }
}

/// BatchPolicy controls when drain() is called automatically.
pub enum BatchPolicy {
    /// Drain every N remote frees. Current snmalloc/mimalloc behavior.
    Fixed(usize),
    /// Drain based on queue depth ratio + cross-thread free rate over a sliding window.
    /// v1.0 ships Fixed(64); Adaptive ships in v2.0 after benchmark validation.
    /// v1.0 implementation uses stepped threshold (5 levels: 8/16/32/64/128).
    Adaptive,
}

unsafe impl<T, B: Allocator> Deallocator for SlabRemote<T, B> {
    /// Spins until a queue slot is available. For latency-sensitive callers,
    /// use try_deallocate instead and handle overflow explicitly.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        while self.try_deallocate(ptr, layout).is_err() {
            core::hint::spin_loop();
        }
    }
}

impl<T, B: Allocator> SlabRemote<T, B> {
    /// Non-spinning deallocation. Returns Err(ptr) if the queue is full.
    /// On Err, caller owns ptr and is responsible for eventual deallocation.
    /// Use this on latency-sensitive paths (e.g., trading engine network thread)
    /// where spinning is unacceptable.
    ///
    /// # Safety
    /// ptr must have been allocated from the corresponding SlabOwner.
    /// On Err, ptr remains valid and must be deallocated before being dropped.
    pub unsafe fn try_deallocate(
        &self,
        ptr: NonNull<u8>,
        layout: NonZeroLayout,
    ) -> Result<(), NonNull<u8>>;
}
```

**Thread safety model:**
- `SlabOwner` is `Send` (can move to another thread) but `!Sync` (exclusive `allocate` access)
- `SlabRemote` is `Send + Sync` (multiple threads can hold and use remote handles concurrently)
- `UnsafeCell<Slab<T, B>>` is only accessed by `SlabOwner` — the single owner is enforced by `!Sync`
- `RemoteFreeQueue` is accessed by multiple threads — protected by atomic operations

**Drain policy:** `SlabOwner::allocate` calls `drain()` lazily when the local free list is empty. Applications can also call `drain()` explicitly at the top of their allocation loop for predictable latency. `BatchPolicy::Fixed(64)` is the v1.0 default.

**Owner drop & queue closing:** `SlabOwner::Drop` performs one final drain of the remote-free queue and then sets a `closed: AtomicBool` flag on the shared inner state. After the flag is set:

- `SlabRemote::try_deallocate` observes `closed == true` and returns `Err(ptr)` immediately instead of pushing — the queue would otherwise grow indefinitely with no drainer.
- `SlabRemote::deallocate` (the spinning impl) reads the flag inside its retry loop and bails out with the pointer un-pushed, so a remote thread does not spin forever after owner drop.

The flag is set under the queue mutex (v0.1) so that any push that observed `closed == false` is also visible in the queue snapshot the owner drains during drop — no remote free is silently lost. The slab itself (and any still-live `T: Drop` in slots that were never routed through the queue) is torn down when the last `Arc<SlabInner>` clone drops, which is outside the owner's control; callers are responsible for draining handles they cared about before dropping the owner.

**`BatchPolicy::Adaptive` — v1.0 implementation (stepped threshold):** Ships in v1.0 as a stepped threshold with cooldown — 5 levels (8, 16, 32, 64, 128), stepping down when queue depth ratio exceeds 0.75 and stepping up when below 0.25, with a cooldown period between steps to prevent oscillation. No floating point; no tuning parameters beyond step levels and cooldown.

**`BatchPolicy::Adaptive` — v2.0 upgrade (EMA):** After benchmark validation, v2.0 upgrades to an EMA-based control law using two signals: queue depth ratio (weight 0.6) and cross-thread free rate (weight 0.4). The EMA smooths noise; a hysteresis band [0.25, 0.75] prevents rapid threshold oscillation. Drain latency (Signal C) is dropped — it is a lagging indicator and adds timer overhead.

**`RemoteFreeQueue` overflow:** The `Deallocator` trait impl spins until a slot is available. Latency-sensitive callers use `try_deallocate` (returns `Err(ptr)` on overflow) and maintain a per-thread pending list, retrying on the next cycle — zero spin, zero block. Queue capacity defaults to 1024 entries; configurable at `SlabOwner` construction.

---

### 6.8 `ExtendableSlab<T>`

A growable typed allocator. Maintains a list of fixed-capacity `Slab<T, MmapBacked>` segments. Growth adds a new segment rather than reallocating existing ones — freelist offsets within each segment remain valid forever.

```rust
pub struct ExtendableSlab<T, M: FreelistProtection = NoProtection> {
    segments: Vec<Slab<T, MmapBacked, M>>,
    segment_capacity: usize,    // number of T slots per segment
    mac_factory: fn() -> M,     // constructs M for each new segment
}

impl<T, M: FreelistProtection> ExtendableSlab<T, M> {
    pub fn new(segment_capacity: usize, mac_factory: fn() -> M) -> Self;
    pub fn with_initial_segments(count: usize, capacity: usize, mac_factory: fn() -> M)
        -> Result<Self, AllocError>;
}
```

**Does not implement `FixedRange`** — address range grows with each segment. Cannot be used as `Primary` in `WithFallback`.

**Memory return:** When a segment becomes fully free (all slots deallocated), `ExtendableSlab` can return it to the OS. This is the primary advantage over `Slab` with offset-based growth: each segment's `MmapBacked` drops independently.

**Use case:** Long-running services where the pool size is unpredictable at startup — ClaimMatch's claim pool on a busy day, the knowledge base document pool during indexing.

---

## 7. Layer 3 — Hardening Wrappers

### 7.1 `Canary<Inner>`

Writes a sentinel value immediately before and after each allocation. Checks canary integrity on deallocation. Detects linear overflows and underflows.

```rust
pub struct Canary<I: Allocator> {
    inner: I,
    value: u64,
}
```

**Construction:**

```rust
impl<I: Allocator> Canary<I> {
    /// std builds: seeds from OsRng automatically.
    #[cfg(feature = "std")]
    pub fn new(inner: I) -> Self;

    /// no_std builds: caller provides seed (e.g., from hardware RNG or boot-time entropy).
    pub fn new_with_seed(inner: I, seed: u64) -> Self;
}
```

**Overhead:** 16 bytes per allocation (8 pre-canary + 8 post-canary). One read + compare on each free. Zero overhead when the type alias selects `Inner` directly in release builds:

```rust
#[cfg(debug_assertions)]
type ParseArena = Canary<BumpArena<InlineBacked<131072>>>;
#[cfg(not(debug_assertions))]
type ParseArena = BumpArena<InlineBacked<131072>>;
```

Note: zero overhead requires the user to write the cfg type alias. `Canary<Inner>` in a release build still carries `value: u64` and calls sign/verify on every free. The cfg alias pattern is the correct usage for zero-cost release builds.

**Detection:** Linear buffer overflows, heap corruption by adjacent writes. Does not detect out-of-bounds reads or UAF.

**On-free zeroize:** After verifying the canary words on `deallocate`, `Canary` overwrites both canary words with zero via volatile writes (paired with a `SeqCst` `compiler_fence`) before forwarding to `self.inner.deallocate`. The seed is a per-process secret; if it persisted in deallocated memory until OS reclaim, any code that later borrows the freed region (slab freelist reuse, `BumpArena::reset`, `mmap` remap) could lift it via a UAF read and use it to forge canaries elsewhere in the process. The `Canary` struct's `value` field is itself volatile-zeroed on `Drop` for the same reason — preventing the seed from lingering in a deallocated stack frame.

---

### 7.2 `GuardPage<Inner>`

Inserts an unmapped virtual memory page between each allocation region (at `MmapBacked` granularity). Overflow into a guard page triggers an immediate segfault rather than silent corruption.

```rust
pub struct GuardPage<I: Allocator> {
    inner: I,
    page_size: usize,
}
```

**Overhead:** One unmapped page (virtual address space only, no physical memory) between each allocation. On Apple Silicon, one 16KB page. On x86/Linux, one 4KB page.

**Constraint:** Only meaningful when `Inner` is `MmapBacked` or similar OS-backed allocator. Applying to `InlineBacked` is a compile error (enforced by trait bound `Inner: OsBacked`).

---

### 7.3 `PoisonOnFree<Inner>`

Overwrites freed memory with a known poison pattern (`0xDE` by default, configurable) before returning control to the inner allocator. Prevents freed data from being read back via UAF or from persisting in freed memory for information disclosure.

```rust
pub struct PoisonOnFree<I: Allocator> {
    inner: I,
    pattern: u8,
}
```

**Security property:** Freed PHI, key material, or sensitive data cannot be recovered from a subsequent allocation that reuses the same address. Complements `Quarantine` — poison destroys content, quarantine delays reuse.

**Overhead:** `memset` over the freed region. For small fixed-size objects this is a handful of instructions. For large allocations this cost is non-trivial and should be weighed against the sensitivity of the data.

---

### 7.4 `Quarantine<Inner, const EPOCHS: usize>`

Holds freed blocks in a ring buffer for `EPOCHS` allocation cycles before returning them to the inner allocator for reuse.

```rust
pub struct Quarantine<I: Allocator, const EPOCHS: usize> {
    inner: I,
    queue: [Option<QuarantinedBlock>; EPOCHS],
    head: usize,
}

// Compile-time enforcement: EPOCHS = 0 is nonsensical
const _: () = assert!(EPOCHS >= 1, "Quarantine requires at least 1 epoch");

struct QuarantinedBlock {
    ptr: NonNull<u8>,
    layout: NonZeroLayout,   // NonZeroLayout, not Layout — Quarantine never sees ZSTs
}
```

**Security property:** With `EPOCHS = 16`, a dangling pointer must survive 16 allocation cycles on the same allocator instance before the slot can be reused.

**Typed vs. mixed quarantine:** The security claim "attacker must spray 16 allocations of the correct size" holds only when `Quarantine` wraps a typed allocator (`Slab<T, _>`), where every slot is the same size and all 16 slots are of the target type. When `Quarantine` wraps a size-classed allocator (`SizeClassed<_>`), all size classes share the same EPOCHS ring — effective per-class quarantine depth is `EPOCHS / active_size_classes`. **Recommended composition:** place `Quarantine` *inside* `SizeClassed` rather than outside it: `SizeClassed<Quarantine<Slab<T, _>, 16>, N>` gives per-class quarantine. `Quarantine<SizeClassed<_, N>, 16>` gives shared quarantine across classes.

**Interaction with `PoisonOnFree`:** Compose as `PoisonOnFree<Quarantine<Inner, N>>` — poison on free, then quarantine the poisoned block.

---

### 7.5 `FreelistProtection` — Now a `Slab` Type Parameter

**Architecture change from earlier revisions:** `FreelistMAC` is no longer a Layer 3 hardening wrapper. It is a `FreelistProtection` policy parameter on `Slab<T, B, M>`, defined in §4.5. This change was required because a wrapper cannot access the private `free_head` field inside `Slab`.

The three implementations are:

```rust
pub struct NoProtection;         // default — zero overhead

pub struct SipHashMAC {
    key: [u8; 16],               // OsRng at construction
}

#[cfg(all(target_arch = "aarch64", feature = "pac"))]
pub struct PacMAC;               // hardware PACIB/AUTIB instructions
```

Usage:

```rust
// No protection (default)
type BarPool = Slab<Bar, InlineBacked<65536>>;

// SipHash MAC
type SecurePool = Slab<ClaimRecord, MmapBacked, SipHashMAC>;

// Hardware PAC on arm64e
#[cfg(all(target_arch = "aarch64", feature = "pac"))]
type HardwarePool = Slab<ClaimRecord, MmapBacked, PacMAC>;
```

`FreelistProtection` is defined in §4.5. The `sign`/`verify` contract and the `SipHash13` key derivation are specified there.

---

### 7.6 `Statistics<Inner>`

Tracks allocation counts, peak usage, current usage, and failure counts. Zero-overhead in release builds via `#[cfg(debug_assertions)]` gate or explicit feature flag.

```rust
pub struct Statistics<I: Allocator> {
    inner: I,
    #[cfg(feature = "stats")]
    stats: AllocStats,
}

pub struct AllocStats {
    pub total_allocations: AtomicU64,
    pub total_deallocations: AtomicU64,
    pub bytes_allocated: AtomicU64,
    pub bytes_peak: AtomicU64,
    pub failures: AtomicU64,
}
```

**Use case:** Wrap any allocator during development to understand allocation patterns before choosing which hardening to apply in production.

### 7.7 `Watermark<Inner, H: WatermarkHandler>`

Monitors allocation utilization in bytes and fires callbacks at configurable thresholds.

```rust
/// Atomic variant — targets with pointer-sized atomics (x86, ARM, Apple Silicon).
#[cfg(target_has_atomic = "ptr")]
pub struct Watermark<I: Allocator, H: WatermarkHandler> {
    inner: I,
    handler: H,
    thresholds: WatermarkThresholds,
    capacity_bytes: usize,        // from inner.capacity_bytes() at construction
    warn_threshold_bytes: usize,  // pre-computed warn_pct * capacity_bytes / 100;
                                  // allocate's hot path skips the (#[cold])
                                  // check_and_fire call entirely while
                                  // new_bytes < warn_threshold_bytes. Set to
                                  // usize::MAX for unbounded inners (gate
                                  // then never fires).
    allocated: AtomicUsize,
    fired: AtomicUsize,           // bit 0 = warn fired, bit 1 = critical fired —
                                  // rising-edge latches that prevent re-firing
                                  // on every allocate. `rearm()` clears them.
}

// Non-atomic variant deferred — single-core no_std builds without
// `target_has_atomic = "ptr"` get the wrapper unimplemented today.

pub struct WatermarkThresholds {
    pub warn_pct: u8,        // default: 75
    pub critical_pct: u8,    // default: 90
    // at 100%: allocation returns AllocError, on_oom() called
}

pub trait WatermarkHandler: Send + Sync {
    fn on_warn(&self, event: WatermarkEvent);
    fn on_critical(&self, event: WatermarkEvent);
    fn on_oom(&self, event: WatermarkEvent);
}

/// Passed to every handler callback.
#[derive(Copy, Clone, Debug)]
pub struct WatermarkEvent {
    pub level: WatermarkLevel,
    pub allocated_bytes: usize,
    pub capacity_bytes: usize,
    pub requested_layout: Option<NonZeroLayout>,  // Some on oom, None on warn/critical
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WatermarkLevel { Warn, Critical, Oom }
```

**Capacity normalization:** `Watermark` calls `inner.capacity_bytes()` on every threshold check rather than caching at construction. This costs one additional method call per allocation but is correct for growing inners — `ExtendableSlab.capacity_bytes()` increases as segments are added, and a cached value would produce stale (too-aggressive) threshold firing. For fixed-capacity inners this is a constant function call that the optimizer typically inlines to a register load.

**Provided handlers:**

```rust
pub struct LogHandler;
pub struct ChannelHandler(Sender<WatermarkEvent>);
pub struct FnHandler<F: Fn(WatermarkEvent) + Send + Sync>(F);
pub struct NullHandler;  // zero overhead; for disabling in release builds
```

**Overhead:** One `AtomicUsize::fetch_add` on allocation, one `AtomicUsize::fetch_sub` on deallocation, one threshold compare. Handler invocation is synchronous on the allocation path — handlers must be fast (set a flag, send on a non-blocking channel).

---

## 8. Composition Examples

### 8.1 Trading Engine Hot Path

```rust
use forge_alloc::*;

// Per-type slabs backed by inline storage — entirely on the stack/BSS
// No heap involvement, no OS calls, no locks
type BarPool    = Slab<Bar,    InlineBacked<{ 1024 * mem::size_of::<Bar>() }>>;
type TickPool   = Slab<Tick,   InlineBacked<{ 4096 * mem::size_of::<Tick>() }>>;
type SignalPool = Slab<Signal, InlineBacked<{ 256  * mem::size_of::<Signal>() }>>;

// Per-tick scratch arena — reset at the end of each tick
type TickScratch = BumpArena<InlineBacked<65536>>;

// Usage — NonZeroLayout, not Layout
let bars = BarPool::unprotected(1024, InlineBacked::new()).unwrap();
let layout = NonZeroLayout::for_type::<Bar>().unwrap();
let bar_ptr = bars.allocate(layout).unwrap();
// ...process...
unsafe { bars.deallocate(bar_ptr.cast(), layout) };
```

### 8.2 Database Query Key Encoding

```rust
// Before: 2x Vec<u8> allocation per call through RtlAllocateHeap
// After: bump arena, reset at query boundary, zero heap ops on hot path

type QueryArena = WithFallback<BumpArena<InlineBacked<65536>>, System>;

fn execute_query(plan: &Plan, arena: &mut QueryArena) {
    for cell in plan.cells() {
        // encode_key now allocates from arena bump pointer, not heap
        let lower = cell.encode_lower_bound(arena);
        let upper = cell.encode_upper_bound(arena);
        // ... lookup ...
    }
    arena.reset();  // O(1), reclaims all key encoding scratch
}
```

### 8.3 ClaimMatch PHI Buffers

```rust
// PHI data: poison freed bytes, quarantine 16 epochs, MTE on ARM
#[cfg(all(target_arch = "aarch64", feature = "mte"))]
type ClaimBacking = MteBacked;
#[cfg(not(all(target_arch = "aarch64", feature = "mte")))]
type ClaimBacking = MmapBacked;

type ClaimBuffer = PoisonOnFree<
    Quarantine<
        Slab<ClaimRecord, ClaimBacking>,
        16
    >
>;

// Hardening is in the type — no runtime configuration, no accidental bypass
let mut claims = ClaimBuffer::new(256).unwrap();
```

### 8.4 Tree-Sitter Parser Arena

```rust
// All AST nodes allocated from arena, entire arena dropped at parse end
// Debug builds add canary checking; release builds are zero-overhead bump
#[cfg(debug_assertions)]
type ParseArena = Canary<BumpArena<MmapBacked>>;
#[cfg(not(debug_assertions))]
type ParseArena = BumpArena<MmapBacked>;

fn parse(source: &str) -> Ast {
    let arena = ParseArena::new(MmapBacked::new(4 * 1024 * 1024).unwrap());
    let ast = parse_into(source, &arena);
    // arena dropped here — single OS call frees entire parse tree
    ast.into_owned()  // moves non-arena data out before drop
}
```

---

## 9. Additional Primitives from Open Research Opportunities

The following primitives address the design opportunities identified in the allocator research literature that are not covered by the base composition model above. They are first-class components, not afterthoughts.

---

### 9.1 `HugePageAligned<Inner>` — Huge Page Alignment as a Layout Contract

**Problem:** Requesting huge pages via `MAP_HUGETLB` or `madvise(MADV_HUGEPAGE)` is not the same as guaranteeing huge page alignment. The OS promotes pages post-hoc when it can. A purge or decommit operation that releases memory below the 2MiB boundary boundary breaks the huge page, forcing the OS to split it into 4KB pages, negating the TLB benefit.

**Solution:** `HugePageAligned` enforces 2MiB alignment at allocation time and refuses to purge below a full 2MiB boundary. This is a layout contract, not an OS hint.

```rust
pub struct HugePageAligned<I: OsBacked> {
    inner: I,
    huge_page_size: usize,   // 2MiB on x86/ARM Linux, 32MiB on Apple Silicon
    min_purge_size: usize,   // = huge_page_size; purges smaller than this are no-ops
}
```

**Platform mapping:**
- x86-64 Linux: 2MiB alignment, `MAP_HUGETLB | MAP_HUGE_2MB`
- ARM Linux (4KB granule): 2MiB alignment, same flags
- Apple Silicon (16KB granule): 32MiB alignment (16KB × 2048 = 32MiB block); explicit huge pages not available via hugetlbfs, relies on THP promotion which requires 32MiB-aligned regions
- Windows: 2MiB alignment, `MEM_LARGE_PAGES` (requires `SeLockMemoryPrivilege`)

**Key invariant:** The allocator tracks the minimum live address within each huge-page-aligned region. A purge call is silently ignored unless the entire 2MiB (or 32MiB) region is free. This avoids the mimalloc failure mode where a single small live allocation prevents THP promotion of an entire segment.

---

### 9.2 `CacheJitter<Inner>` — Cache Set Associativity Poisoning Defense

**Problem:** Allocators that always return base addresses at fixed alignments concentrate metadata into a small number of cache associativity sets. On an 8-way set-associative L1 cache with 64-byte lines, metadata structures at 4KB-aligned boundaries all hash to the same set. This was discovered empirically in snmalloc development — a microbenchmark that appeared to test allocator throughput was actually testing cache associativity contention.

**Solution:** Apply a small randomized displacement to each backing allocation's base address so that metadata and object headers are spread across associativity sets.

```rust
pub struct CacheJitter<I: Allocator> {
    inner: I,
    cache_line_size: usize,     // 64 on x86/ARM, 128 on Apple Silicon
    associativity: usize,       // 1..=MAX_ASSOCIATIVITY (= 2^16 - 1)
    line_shift: u32,            // cached `trailing_zeros(cache_line_size)`
    mac_key: u64,               // per-instance secret keying the header MAC
    rng: Cell<u64>,             // xorshift64, seeded from OS or caller
}
```

**Construction validates** that `cache_line_size` is a power of two and `>= 8` (so the 8-byte displacement header fits within a line), that `associativity` lies in `1..=MAX_ASSOCIATIVITY` (the 16-bit displacement field caps `associativity * cache_line_size` displacement at well under `usize::MAX`), and that `cache_line_size * associativity` does not overflow. `CacheJitter::with_params` returns `None` on any violation; `CacheJitter::new` (std-only) seeds from OS entropy.

**Displacement calculation:** Per allocation, compute a displacement `d = (rng_next() * associativity >> 64) * cache_line_size` (unbiased multiply-shift reduction — `% associativity` introduces a small bias when `associativity` is not a power of two and would weaken cache-set spreading). The displacement is recorded in an 8-byte header at `user_ptr - 8`, authenticated by a 48-bit MAC keyed by the per-instance `mac_key` and bound to `user_ptr` (so a header copied from one allocation cannot be replayed against another).

**MAC over the header:** The 8-byte header packs `(displacement_in_lines | MAC48)`. On deallocate, `CacheJitter` verifies the MAC in constant time before trusting the embedded displacement — without this, an attacker who can blind-write the prefix (linear underflow from an adjacent allocation, UAF prefix write into a freed slot) would have an arbitrary-free primitive against the inner allocator. The MAC mixer is a keyed SplitMix-style avalanche, not a cryptographic MAC; it raises blind forgery to 2⁻⁴⁸ per attempt but is not key-recovery-hard against an observed-pair attacker who can read many `(user_ptr, header)` pairs (use a cryptographic MAC like SipHash if your threat model includes that adversary).

**Overhead:** One xorshift64 step + one keyed mix per `allocate`; one keyed mix + constant-time compare per `deallocate`. `cache_line_size + (associativity - 1) * cache_line_size` bytes of prefix per allocation (header + displacement room).

**Platform note:** Cache line size must be detected at runtime or compile time. On Apple Silicon it is 128 bytes; using 64 bytes would halve the jitter range and leave half the associativity sets unpopulated.

**Thread safety:** `!Sync` — the `rng: Cell<u64>` state forces single-thread use of any single `CacheJitter` instance. For cross-thread use, give each thread its own instance (each gets an independent `mac_key`, which actually improves the threat model) or external `Mutex<CacheJitter<_>>`.

---

### 9.3 `SlabOwner<T, B>` / `SlabRemote<T, B>` — Ownership-Return Cross-Thread Free

This primitive addresses the open research opportunity of a message-passing free model with workload-adaptive batching. The full specification is in §6.7. Key decisions incorporated:

- Replaces the TCMalloc shared-pool model with an ownership-return queue
- `SlabOwner` holds exclusive allocate access; `SlabRemote` handles cross-thread deallocation only
- `try_deallocate` is in the v1.0 API: non-spinning path for latency-sensitive callers; spinning `Deallocator` impl for standard usage
- **`BatchPolicy::Adaptive` v1.0:** Stepped threshold (5 levels: 8/16/32/64/128), steps down at queue depth > 75%, steps up at < 25%, cooldown period between steps. No floating point.
- **`BatchPolicy::Adaptive` v2.0:** EMA-based, two signals: queue depth ratio (0.6 weight) + cross-thread free rate (0.4 weight). Drain latency signal dropped — lagging indicator. Requires benchmark validation before shipping.
- `BatchPolicy::Adaptive` remains the only allocator component with no equivalent in any production allocator

---

### 9.4 `NumaLocal<Inner>` — Topology-Aware Slab Placement

**Problem:** `MmapFlags.numa_node` is a static hint set at construction time. In a multi-threaded system where threads migrate between NUMA nodes, a slab constructed on node 0 serves allocations for threads that may later run on node 1, paying remote memory latency on every access.

**Solution:** `NumaLocal` wraps any `OsBacked` allocator and binds new backing regions to the NUMA node of the allocating thread at the time the region is requested.

```rust
pub struct NumaLocal<I: OsBacked> {
    inner: I,
    policy: NumaPolicy,
}

pub enum NumaPolicy {
    LocalAtRequest,         // mbind() to current thread's node when backing is requested
    LocalAtConstruction,    // mbind() to constructing thread's node (= MmapFlags behavior)
    Interleaved,            // mbind(MPOL_INTERLEAVE) across all nodes — maximizes bandwidth
    Split(u32, u32),        // split evenly between two nodes — for bandwidth-bound workloads
}
```

**Platform availability:**
- Linux: `mbind()` + `getcpu()` — fully supported
- macOS/Apple Silicon: UMA — `NumaLocal` is a no-op, compiles to zero overhead via conditional implementation
- Windows: `VirtualAllocExNuma()` — supported, requires node ID at allocation time; `LocalAtRequest` requires `GetCurrentProcessorNumberEx()` to determine current node

**Design note:** `NumaLocal` only operates at the *backing* level — it affects which physical memory backs a slab or arena, not which objects go where. For true NUMA-local object placement you'd need application-level partitioning (one slab instance per NUMA node, threads use the local instance). `NumaLocal` handles the infrastructure; the application handles the routing.

---

### 9.5 `SplitMetadata<Inner>` — Hot/Cold Metadata Isolation

**Problem:** In a standard slab, the free list head and block count live in the same struct as (or adjacent to) the user data region. Every allocation touches both metadata and the object, polluting the same cache lines. For large slabs this means metadata and hot user data compete for L1 cache.

**Security dimension:** Metadata adjacent to user data is vulnerable to linear overflows. A one-byte overflow past a user allocation can corrupt the adjacent free list pointer. Separating metadata from data eliminates this attack surface structurally — no overflow of user data can reach allocator metadata regardless of size.

**Solution:** `SplitMetadata` maintains two distinct `MmapBacked` regions — one for allocator metadata (free list, block states, canaries), one for user data. The metadata region is sized for hot access and can be pinned to L1-friendly sizes.

```rust
pub struct SplitMetadata<I: Allocator> {
    meta_region: MmapBacked,    // free list, block headers, canaries
    data_region: I,             // user-visible allocation space
}

// OsBacked is forwarded only when the data region is itself OsBacked
// (i.e. SplitMetadata wraps an MmapBacked directly). Higher-layer
// inners (e.g. Slab) still satisfy the struct-level `Allocator` bound
// but don't expose `release_pages` / `protect` at this layer.
unsafe impl<I: Allocator + OsBacked> OsBacked for SplitMetadata<I> { /* ... */ }
```

**Cache behavior:** For a slab of 64-byte objects with 8 bytes of metadata per object, the metadata region is 1/8 the size of the data region. All metadata for a 4MB slab fits in 512KB — within L2 on most architectures. Allocation becomes: touch metadata (L2 hit) + touch user data (L3 or RAM miss). Without split metadata both accesses go to RAM together.

**Security behavior:** The metadata and data regions are at unrelated virtual addresses. A buffer overflow past any user allocation reaches unmapped memory (if `GuardPage` is composed) or unrelated user data — never the allocator's free list. This is the structural property that GrapheneOS hardened_malloc achieves, here expressed as a composable wrapper.

**Composition with `GuardPage` — coverage model (Gate 13 decision):**

`SplitMetadata` segregates the metadata and data regions into disjoint virtual-address mappings — that is its entire structural contribution. It does **not** install any guard pages itself; callers compose `GuardPage<MmapBacked>` (or rely on the `HardenedSlab` alias, which wraps the data mmap in `GuardPage`) for an unmapped barrier on either region. The "two layers together" full-coverage pattern requires *both* `GuardPage` *and* `SplitMetadata` explicitly in the type.

```rust
// Data region only guarded (SplitMetadata internal):
// (Note: SplitMetadata does not itself install guard pages; "data-only
// guarding" means GuardPage wraps the data side of the split.)
type DataGuarded<T> = Slab<T, SplitMetadata<GuardPage<MmapBacked>>>;

// Metadata region only guarded — not yet expressible without exposing the
// meta region; deferred. The HardenedSlab alias below guards the data
// region; full coverage of meta is tracked in a separate spec gate.

// Recommended composition for security-critical use — guard pages on the
// data side, separate (unprotected) metadata mmap. The hardening wrappers
// sit on the OsBacked side; Slab consumes the protected region from the
// outside.
type FullyGuarded<T> = Slab<T, GuardPage<SplitMetadata<MmapBacked>>>;
```

`SplitMetadata` is annotated to surface the composition gap at use time:

```rust
#[must_use = "SplitMetadata guards the data region only. \
              Compose with GuardPage<_> for metadata region coverage: \
              GuardPage<SplitMetadata<_>>"]
pub struct SplitMetadata<I: Allocator> { ... }
```

**`HardenedSlab<T, M>` — convenience alias for the recommended maximum-hardening composition:**

```rust
/// Slab with split metadata, guard pages on both regions, and freelist protection.
/// Equivalent to Slab<T, GuardPage<SplitMetadata<MmapBacked>>, M>.
/// (Spec v1.0 listed the wrappers as outermost; that composition doesn't
/// compile because SplitMetadata/GuardPage require OsBacked inners which
/// Slab isn't. The in-tree alias swaps the nesting so the OsBacked-
/// requiring wrappers sit on the OS-mapped side.)
/// For use in security-critical contexts: PHI handling, key material, audit logs.
pub type HardenedSlab<T, M = NoProtection> =
    Slab<T, GuardPage<SplitMetadata<MmapBacked>>, M>;

// Usage:
type ClaimPool = HardenedSlab<ClaimRecord>;                    // guard pages only
#[cfg(feature = "siphasher")]
type AuditPool = HardenedSlab<AuditEntry, SipHashMAC>;         // + SipHash freelist MAC
#[cfg(all(target_arch = "aarch64", feature = "pac"))]
type KeyPool   = HardenedSlab<KeyMaterial, PacMAC>;            // + hardware PAC on ARM
```

`HardenedSlab` is a type alias only — it introduces no new behavior. It exists to make the recommended composition discoverable and to give security-auditors a single named type to look for in code review.

**v0.1 default:** `M = NoProtection`. The spec previously listed `SipHashMAC` as the default; that was reverted to `NoProtection` so the alias compiles without the `siphasher` feature. Security-critical callers should opt in explicitly via `HardenedSlab<T, SipHashMAC>` (gated `#[cfg(feature = "siphasher")]`) or `HardenedSlab<T, PacMAC>` on aarch64.

---

### 9.6 `MpkPool` — MPK Domain Rotation for x86-64

**Problem:** `MpkBacked` as specified in §5.5 allocates one protection key per instance. x86-64 provides only 16 keys system-wide (2 reserved by OS = 14 available). A naive implementation exhausts keys quickly.

**Solution:** A shared key pool with LRU rotation and generation tracking. Keys are leased to `MpkBacked` instances, revoked on free, and returned to the pool for reuse with a bumped generation counter.

```rust
pub struct MpkPool {
    available: ArrayVec<i32, 14>,
    generation: [u64; 14],      // bumped each time a key is recycled
}

pub struct MpkBacked {
    inner: MmapBacked,
    pkey: i32,
    generation: u64,            // generation at time of lease
    pool: Arc<Mutex<MpkPool>>,
}
```

**Fork safety (Gate 14 decision):** `MpkPool` registers a `pthread_atfork` child handler at construction. The handler reinitializes the pool in the child process before any child code runs, closing the window between `fork()` and first allocation. A one-time registration guard prevents duplicate handler registration if `MpkPool` is constructed multiple times (e.g., in tests):

```rust
static MPK_ATFORK_REGISTERED: AtomicBool = AtomicBool::new(false);

impl MpkPool {
    fn new() -> Result<Self, AllocError> {
        let pool = Self { ... };
        // Register atfork handler exactly once per process
        if MPK_ATFORK_REGISTERED
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            // SAFETY: handler is async-signal-safe: no allocation, no locks.
            unsafe { libc::pthread_atfork(None, None, Some(mpk_child_atfork)); }
        }
        Ok(pool)
    }
}

extern "C" fn mpk_child_atfork() {
    // Called in child after fork(), before any child code runs.
    // Reinitializes the pool: releases inherited keys, acquires fresh ones.
    // Must be lock-free and alloc-free — process is in a partial fork state.
    // Uses pkey_free() for each inherited key, then pkey_alloc() for fresh keys.
    // A debug-mode eprintln confirms reinitialization for diagnostics:
    // "MpkPool: fork detected, reinitializing protection keys in child"
}
```

**Windows:** `pthread_atfork` is not available. On Windows, `MpkBacked` is unavailable (`cfg(target_os = "windows")` excludes the module) since Windows uses a different memory protection mechanism (`VirtualProtect`/`SetProcessMitigationPolicy`) and does not support MPK.

---

## 10. Rust Ownership as Verification

The StarMalloc work required Steel separation logic, SMT solvers, and dependent types to verify in C what Rust's type system provides structurally.

**What the type system eliminates without proof effort:**

- **Use-after-free at the allocator level (slab):** `Slab::deallocate` pushes to the free list. If the caller holds the slab by value and drops it, outstanding raw pointers become dangling — but `BumpDeallocator<'a>` ties the deallocation token's lifetime to the arena's, preventing the arena from being dropped while any live `Box<T, BumpDeallocator<'_>>` exists. The borrow checker enforces this at compile time.

- **Double-free via `GenerationalSlab`:** `remove()` returns `Option<T>` and increments the generation. A second `remove()` with the same handle returns `None` — the slot is either empty or holds a different generation. No double-free is possible through the `Handle` API.

- **Metadata/data aliasing in `SplitMetadata`:** Two separate `MmapBacked` regions with separate lifetimes. The type system prevents a reference to user data from aliasing a reference to metadata.

**What the arena pattern cannot provide via safe Rust:**

Arenas fundamentally cannot return `&mut T` from `&self`. Two calls to `alloc(&self)` with the same lifetime would produce two `&mut T` references — violating Rust's aliasing rules. The correct arena API returns `NonNull<T>` (unsafe) or uses a scoped callback pattern:

```rust
// Incorrect (unsound): would allow two &mut T with same lifetime
// fn alloc<'a>(&'a self) -> &'a mut T { ... }

// Correct option 1: raw pointer, caller manages aliasing
fn alloc(&self, layout: NonZeroLayout) -> Result<NonNull<u8>, AllocError>;

// Correct option 2: scoped callback prevents escaping references
fn with_alloc<T, F: FnOnce(&mut T) -> R, R>(&mut self, f: F) -> R;
```

`BumpArena` uses option 1 — the `Allocator` trait returns `NonNull<[u8]>`. Safety obligations are on the caller. `BumpDeallocator<'a>` provides the lifetime enforcement for `Box`-style usage.

**What still requires runtime checks or formal verification:**

- Buffer overflows: the borrow checker does not prevent writes past the end of a correctly-owned buffer. `Canary`, `GuardPage`, and MTE/EMTE remain necessary.
- Freelist integrity under adversarial memory corruption from FFI or unsafe code: `FreelistProtection` (`SipHashMAC`, `PacMAC`) defends this.
- Correctness of `unsafe` blocks: Kani and MIRI for M13.

---

## 11. Crate Structure

```
forge-alloc/
├── rust-toolchain.toml
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── traits/
│   │   ├── mod.rs
│   │   ├── allocator.rs          # Allocator, Deallocator traits
│   │   ├── non_zero_layout.rs    # NonZeroLayout, StdCompat<A>
│   │   ├── os_backed.rs          # OsBacked, ProtectFlags
│   │   ├── fixed_range.rs        # FixedRange
│   │   └── freelist_protection.rs # FreelistProtection, NoProtection, SipHashMAC, PacMAC
│   ├── backing/
│   │   ├── mod.rs
│   │   ├── inline.rs             # InlineBacked<N>
│   │   ├── mmap.rs               # MmapBacked, MmapFlags
│   │   ├── system.rs             # System
│   │   ├── huge_page.rs          # HugePageAligned<I: OsBacked>
│   │   ├── numa.rs               # NumaLocal<I: OsBacked>
│   │   ├── cache_jitter.rs       # CacheJitter<I>
│   │   ├── mte.rs                # MteBacked (cfg aarch64 + mte)
│   │   └── mpk.rs                # MpkBacked, MpkPool (cfg x86_64 + mpk)
│   ├── layout/
│   │   ├── mod.rs
│   │   ├── bump.rs               # BumpArena<B>, SharedBumpArena<B>, BumpDeallocator<'a>
│   │   ├── slab.rs               # Slab<T, B, M=NoProtection>, FreeLink
│   │   ├── slab_generational.rs  # GenerationalSlab<T, B>, GenerationalSlot<T>, Handle<T>
│   │   ├── slab_owner.rs         # SlabOwner<T, B>, SlabRemote<T, B>, RemoteFreeQueue
│   │   ├── slab_extendable.rs    # ExtendableSlab<T, M>
│   │   ├── size_classed.rs       # SizeClassed<B, CLASSES>, UntypedSlab (private)
│   │   ├── stack.rs              # StackAlloc<B>
│   │   └── fallback.rs           # WithFallback<P: FixedRange, S>
│   └── hardening/
│       ├── mod.rs
│       ├── canary.rs             # Canary<I>
│       ├── guard_page.rs         # GuardPage<I: OsBacked>
│       ├── poison.rs             # PoisonOnFree<I>
│       ├── quarantine.rs         # Quarantine<I, EPOCHS>
│       ├── split_metadata.rs     # SplitMetadata<I>
│       ├── watermark.rs          # Watermark<I, H>, WatermarkEvent, WatermarkHandler
│       └── statistics.rs         # Statistics<I>, AllocStats
├── fuzz/
│   ├── Cargo.toml
│   ├── fuzz_bump_arena.rs
│   ├── fuzz_slab.rs
│   ├── fuzz_with_fallback.rs
│   └── fuzz_generational_slab.rs
├── tests/
│   ├── correctness/
│   ├── security/
│   ├── numa/                     # Linux only (cfg)
│   └── no_std/                   # Compile-only no_std compatibility tests
├── benches/
│   └── allocators.rs             # Criterion; baseline committed
└── examples/
    ├── trading_pool.rs
    ├── query_arena.rs
    ├── phi_buffer.rs
    ├── generational_handles.rs
    ├── cross_thread_slab.rs      # SlabOwner + SlabRemote
    └── numa_aware_slab.rs
```

---

## 12. Feature Flags

```toml
[features]
default = ["std"]
std     = []        # MmapBacked, GuardPage, NumaLocal, OsRng for Canary::new
mte     = []        # ARM MTE (aarch64 only)
mpk     = []        # Intel MPK (x86_64 only)
pac     = []        # ARM PAC for PacMAC (aarch64e only)
stats   = []        # Statistics counters in release builds
                    # (always available in debug builds)
# No 'nightly' feature flag. Per-collection allocator usage (Vec<T,A> etc.)
# is available on stable Rust via the allocator-api2 dependency.
```

**`no_std`-compatible primitives** (compile without `std` feature):
`InlineBacked`, `BumpArena`, `Slab`, `Canary::new_with_seed`, `PoisonOnFree`, `Quarantine`, `SplitMetadata`, `GenerationalSlab`, `SizeClassed`, `StackAlloc`, `WithFallback`, `Statistics`.

**Requires `std`:** `MmapBacked`, `MmapFlags`, `GuardPage`, `HugePageAligned`, `NumaLocal`, `CacheJitter`, `MteBacked`, `MpkBacked`, `Canary::new` (uses OsRng).

**Atomic gating:** `SharedBumpArena` and `Watermark` (atomic variant) require `cfg(target_has_atomic = "ptr")`. Non-atomic `Watermark` (single-core `no_std`) is available unconditionally but is `!Sync`.

---

## 13. Safety Contract

**Invariants upheld by all primitives:**

1. Returned pointers are valid, non-null, and correctly aligned for the requested layout.
2. The allocated region is at least `layout.size()` bytes.
3. Two live allocations from the same allocator instance never overlap.
4. After `reset()`, all previously issued pointers are invalid. Accessing them is undefined behavior. No runtime check is available — `BumpArena::deallocate` is a no-op, so `Statistics` tracking `allocations - deallocations` will always report nonzero live count for an arena regardless of whether objects are actually live. Correctness is the caller's responsibility. `BumpDeallocator<'a>` lifetimes enforce this at compile time for `Box`-style usage; raw `allocate` calls have no check.
5. After `deallocate(ptr, layout)`, `ptr` is invalid. `PoisonOnFree` and `Quarantine` add defense-in-depth but do not substitute for correct ownership discipline.
6. Calling `deallocate` with a pointer not issued by this allocator is undefined behavior in release builds. Debug builds with `Statistics` track allocation addresses and panic on foreign-pointer deallocation.

**Thread safety:**

- `BumpArena` is `Send` but `!Sync`. Single-threaded use only.
- `SharedBumpArena` is `Send + Sync`. Atomic cursor. No `reset()`.
- `Slab` is `Send` but `!Sync`. `UnsafeCell<free_head>` — concurrent deallocation is UB. Use `SlabRemote` for cross-thread deallocation.
- `SlabOwner` is `Send + !Sync`. `SlabRemote` is `Send + Sync`.
- `MteBacked` and `MpkBacked` — OS mappings are per-process, safe across threads. Tag/key assignment at construction; no per-allocation synchronization required.

---

## 14. Testing Strategy

**Correctness (proptest):**
- Arbitrary allocation sequences: verify no overlap, correct alignment, size satisfied
- Interleaved alloc/free: verify free list integrity after arbitrary patterns
- `BumpArena` reset: verify cursor returns to base, subsequent allocations succeed
- `GenerationalSlab`: stale handle returns `None`; fresh handle after reuse returns new value
- `SlabOwner`/`SlabRemote`: concurrent remote frees followed by owner drain; verify no lost objects

**Hardening invariants:**
- `Canary`: write past allocation boundary → canary check fires on free
- `GuardPage`: write to guard page → SIGSEGV (test via `fork` + signal handler)
- `PoisonOnFree`: allocate, free, read freed memory → verify poison pattern present
- `Quarantine`: free then immediately re-allocate same size → verify different address for `EPOCHS` cycles
- `SipHashMAC`: corrupt a freelist offset → verify detection on next allocation
- `Watermark`: fill to each threshold → verify correct callback level fires

**Fuzzing (cargo-fuzz — part of M4):**
- `BumpArena`: arbitrary alloc sequences, arbitrary reset points; verify invariant 3 (no overlap)
- `Slab`: arbitrary alloc/free interleaving including double-free attempts; verify free list integrity
- `WithFallback`: arbitrary pointer provenance; verify correct deallocation routing
- `GenerationalSlab`: arbitrary handle reuse sequences; verify no ABA, no UB on stale access
- Fuzz targets run in CI on every PR; failures block merge

**Benchmarks (Criterion):**
- `BumpArena` alloc throughput vs `System`, vs `mimalloc`
- `Slab<T>` alloc/free throughput vs `System`, vs `mimalloc`
- `SizeClassed` throughput vs `System` under random size distribution
- Hardening wrappers: overhead per layer (`Canary`, `SipHashMAC`, `Quarantine<_, 16>`)
- `SlabOwner`/`SlabRemote`: producer-consumer throughput vs `Mutex<Slab>`

**Performance regression gate:** Criterion baseline is committed to the repo. A PR that regresses any benchmark by >5% requires explicit override with documented justification.

**Benchmark workloads representing real use cases:**
- Trading bar/tick allocation pattern (typed slab, high frequency, single-threaded)
- Query key encoding pattern (bump arena, reset per query)
- Parser AST pattern (bump arena, bulk-free per parse)
- PHI claim processing (hardened slab: `PoisonOnFree<Quarantine<Slab<_>, 16>>`)

---

## 15. Milestones

| # | Deliverable | Scope |
|---|---|---|
| M1 | `NonZeroLayout`, `Deallocator`/`Allocator` split, `StdCompat<A>`, `OsBacked`, `FixedRange`, `FreelistProtection` traits | Foundation |
| M2 | `InlineBacked`, `MmapBacked`, `System` | Layer 1 baseline |
| M3 | `BumpArena` + `SharedBumpArena` + `BumpDeallocator`, `Slab<T,B,M>`, `WithFallback<P: FixedRange, S>` | Layer 2 core |
| M4 | Correctness test suite, proptest, cargo-fuzz targets, Criterion benchmarks, `no_std` compile tests | Validation gate |
| M5 | `Canary` (with seeded + OsRng constructors), `PoisonOnFree`, `Statistics`, `Watermark` | Layer 3 — hardening + pressure |
| M6 | `Quarantine`, `SplitMetadata` | Layer 3 — UAF + metadata isolation |
| M7 | `GuardPage`, `SizeClassed`, `StackAlloc`, `ExtendableSlab` | Remaining layout + guard |
| M8 | `GenerationalSlab<T, B>`, `Handle<T>`, `SlabOwner`/`SlabRemote`, `RemoteFreeQueue` | Handle safety + cross-thread |
| M9 | `HugePageAligned`, `CacheJitter`, `NumaLocal` | Performance hardening |
| M10 | `MteBacked` (ARM MTE) | Hardware hardening — ARM |
| M11 | `MpkBacked`, `MpkPool` (x86 MPK) | Hardware hardening — x86 |
| M12 | `BatchPolicy::Adaptive` in `SlabOwner` (pending benchmark validation) | Novel contribution |
| M13 | MIRI for all primitives in CI; Kani for `BumpArena` + `Slab` unsafe blocks | Formal verification |
| M14 | API stabilization, crates.io publish | Release |

**v0.1.0 status (snapshot of in-tree implementation):** M1–M9 and M12 are shipped. M10 (ARM MTE) and M11 (x86 MPK) are hardware-blocked. M13 ships the verification CI scaffolding (`.github/workflows/ci.yml` runs `cargo miri test` on the no-std subsets and `cargo kani` on the workspace, advisory); the initial Kani proof harnesses live in `BumpArena::kani_proofs` and `Slab::kani_proofs`. M14 release artifacts (`README.md`, `CHANGELOG.md`, `LICENSE-MIT`, `LICENSE-APACHE`) are in place. `BatchPolicy::Adaptive` ships the v1.0 stepped-threshold law; v2.0 EMA upgrade remains gated on the benchmark validation harness in `crates/forge-bench/benches/adaptive_batch.rs`. Deferred-then-shipped: `SizeClassed<B, CLASSES>` (M3 backlog → done) and `SplitMetadata<I>` (M6 backlog → done).

**Critical path for immediate projects:** M1–M5. Unblocks trading engine hot path, database key encoding, and ClaimMatch PHI.

**Novel contributions vs existing art:** M8 (`GenerationalSlab` + `SlabOwner`/`SlabRemote`) and M12 (`BatchPolicy::Adaptive`) have no direct equivalent in the Rust ecosystem. M1 (`NonZeroLayout`, `Deallocator` split) contributes to the wg-allocators stabilization discussion.

---

## 16. Cargo.toml and Project Configuration

```toml
[package]
name = "forge-alloc"
version = "0.1.0"
edition = "2021"
rust-version = "1.70"    # MSRV: NonNull::slice_from_raw_parts is stable since 1.70
description = "Composable memory allocator primitives for Rust"
license = "MIT OR Apache-2.0"
repository = "https://github.com/..."

[features]
default = ["std"]
std     = []        # Enables MmapBacked, GuardPage, NumaLocal, OsRng for Canary::new
mte     = []        # ARM MTE: aarch64 only
mpk     = []        # Intel MPK: x86_64 only
pac     = []        # ARM PAC: aarch64e only
stats   = []        # Statistics counters in release builds
# No 'nightly' feature. Per-collection allocator usage is on stable via allocator-api2.

[dependencies]
# Provides Allocator trait on stable Rust.
# Auto-upgrades to std re-export when allocator_api stabilizes.
allocator-api2 = "0.2"

[dependencies.siphasher]
version  = "1.0"
optional = true     # Used by SipHashMAC

[dev-dependencies]
proptest   = "1"
criterion  = { version = "0.5", features = ["html_reports"] }
mimalloc   = "0.1"    # benchmark baseline

[[bench]]
name    = "allocators"
harness = false
```

**`rust-toolchain.toml` (main build — stable):**

```toml
[toolchain]
channel    = "stable"
components = ["rustfmt", "clippy", "rust-src"]
```

**`rust-toolchain.toml` (M13 verification CI — nightly):**

```toml
# .github/workflows/verification.toml uses a separate override:
[toolchain]
channel    = "nightly"
components = ["miri", "rust-src"]
```

MIRI and Kani run on nightly in a dedicated CI job. The main build, tests, and benchmarks all run on stable. `generic_const_exprs` is intentionally avoided — known soundness issues. Const assertions on `InlineBacked<N>` use `const _: () = assert!(...)` in `new()`.

---

## 17. Resolved Decisions

All decisions from both decision gates documents are incorporated.

| Gate | Decision |
|---|---|
| 1 — `FreelistMAC` | Type parameter `M: FreelistProtection = NoProtection` on `Slab<T, B, M>` |
| 2 — `Slab` interior mutability | `UnsafeCell<u32>` for `free_head` (1-based index); `Slab` is `!Sync` |
| 3 — Cross-thread dealloc | `SlabOwner<T,B>` + `SlabRemote<T,B>`; `try_deallocate` + spinning `Deallocator` impl |
| 4 — `GenerationalSlab` layout | Interleaved `GenerationalSlot<T>`; single backing allocation |
| 4b — `insert()` return type | `Result<Handle<T>, AllocError>` |
| 5 — `BumpArena` thread safety | Two distinct types: `BumpArena` (non-atomic) + `SharedBumpArena` (atomic, `cfg(target_has_atomic)`) |
| 6 — `AllocContext` | Deferred to v2.0; `NoContext` implicit in all v1.0 allocators |
| 7 — `Slab` growth | Fixed `Slab`; `ExtendableSlab` as a separate growable type |
| 8 — `WithFallback` routing | `FixedRange` trait bound on `Primary` |
| 9 — `OsBacked` definition | Trait: `base_ptr`, `region_size`, `release_pages`, `protect` |
| 10 — WG alignment | Diverge under library trait names; `allocator-api2` as stable bridge; no `nightly` flag |
| 11 — Queue overflow | `try_deallocate` (returns `Err(ptr)`) + spinning `Deallocator` impl; both in M8 API |
| 12a — Adaptive signals | Queue depth ratio + cross-thread free rate; drain latency dropped |
| 12b — Adaptive control law | Stepped threshold (5 levels) for v1.0; EMA + hysteresis for v2.0 |
| 13 — `SplitMetadata` + `GuardPage` | Document coverage model; `#[must_use]` on `SplitMetadata`; add `HardenedSlab<T,M>` alias. **v0.1 correction**: the spec's outer-wrappers composition `GuardPage<SplitMetadata<Slab<...>>>` could not compile because `GuardPage` / `SplitMetadata` require `OsBacked` inners and `Slab` is not `OsBacked`; the in-tree alias swaps the nesting to `Slab<T, GuardPage<SplitMetadata<MmapBacked>>, M>` so the `OsBacked`-requiring wrappers sit on the OS-mapped side. |
| 14 — `MpkPool` fork safety | `pthread_atfork` with one-time `AtomicBool` registration guard |
| 15 — `no_std` atomics | `cfg(target_has_atomic = "ptr")` gate; non-atomic `Watermark` for single-core `no_std` |
| 16 — Verification tools | MIRI in CI (stable); Kani for `BumpArena` + `Slab<T,B,NoProtection>` (nightly CI job) |

**No remaining open questions.** All items from §17 are now resolved and incorporated into the spec.
