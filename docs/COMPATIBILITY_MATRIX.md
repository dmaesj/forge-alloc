# Compatibility Matrix

A reference for which `forge-alloc` compositions work, which are silently
restricted at runtime, and which are footguns to avoid. Read this *before*
wiring an unfamiliar combination — most of the items listed have a
better-shaped alternative documented in
[`COMPOSITION_RECIPES.md`](COMPOSITION_RECIPES.md).

## How to read this

Each row is `(combination → outcome → what to do instead)`. Outcomes fall
into five categories:

| Category | Detection | What it means |
|---|---|---|
| **CE — compile error** | At type-checking time | The Rust type system rejects the wiring; no runtime path exists. |
| **CT — const assert** | At monomorphisation | A `const _: () = assert!(...)` or `panic!(...)` in a const context fires when the type is instantiated. Stops the build at the user's call site. |
| **CX — construction error** | At `new()`/`with_*()` | Constructor returns `Err(AllocError)` or `None`. Wiring compiles; runtime initialisation rejects. |
| **FG — footgun** | At runtime, silently | Compiles AND initialises. Will work, but with a hidden cost or surprising semantics. |
| **V2 — deferred** | Not yet a hard check | Today's behaviour is suboptimal; the constraint is documented but the enforcement is planned for v2.0. |

---

## Compile-time rejections (CE / CT)

These never produce a runtime failure — the build stops at the call site.
All are pinned by `compile_fail` doctests so a regression that accepts
the misuse is itself a test failure.

### 1. `Slab<(), B>` — ZST T

- **Status**: CT
- **Mechanism**: `Slab::ASSERT_T_NON_ZST` in [`crates/forge-alloc/src/layout/slab.rs`](../crates/forge-alloc/src/layout/slab.rs) — a const-time `assert!(size_of::<T>() != 0)` that fails monomorphisation when T is a ZST. Force-evaluated inside `Slab::with_protection` so the build halts before any test runs.
- **Why**: `Slab` issues one `block_stride` slot per `T`. A ZST has zero size, so capacity and freelist arithmetic collapse to nonsense.
- **Instead**: if you need a "tag" allocator with no payload, use `BumpArena<InlineBacked<N>>` and allocate `[u8; 1]` slots — the type system still tracks count, and the wasted byte is the cost of explicitness.
- **Pinned by**: `compile_fail` doctest in `slab.rs`.

### 2. `InlineBacked<N>` where `N % align_of::<usize>() != 0`

- **Status**: CT
- **Mechanism**: `InlineBacked::ASSERT_N_ALIGNED` const in [`crates/forge-alloc/src/backing/inline.rs`](../crates/forge-alloc/src/backing/inline.rs).
- **Why**: `InlineBacked<N>` is `#[repr(align(16))]` and stores `[u8; N]`. The storage's *end* must also be aligned for slot strides to land correctly; that requires N to be a multiple of the alignment.
- **Instead**: pick `N` as a multiple of 16 — typical choices are 256, 1024, 4096, 65536.
- **Pinned by**: `compile_fail` doctest in `inline.rs`.

### 3. `SlabOwner<T, B>: Sync`

- **Status**: CE
- **Mechanism**: `SlabOwner` carries `_not_sync: PhantomData<Cell<()>>` ([`crates/forge-alloc/src/layout/slab_owner.rs`](../crates/forge-alloc/src/layout/slab_owner.rs)) — auto-trait `Sync` is unimplementable. Any code that tries `assert_sync::<SlabOwner<_, _>>()` fails to compile.
- **Why**: `SlabOwner` is the *owner-thread* half of the cross-thread typed allocator. The owner alone touches the inner `Slab`; workers communicate via `SlabRemote::deallocate`. Making the owner `Sync` would invite multi-thread allocation, which the !Sync `Slab` underneath does not support.
- **Instead**: clone the `SlabRemote` (which IS `Send + Sync`) and ship it to workers.
- **Pinned by**: `compile_fail` doctest in `slab_owner.rs`.

### 4. `SlabRemote<Rc<T>, B>: Send`

- **Status**: CE
- **Mechanism**: `unsafe impl<T: Send, B: ... + Send> Send for SlabRemote<T, B>` requires `T: Send`. `Rc<T>` is `!Send`, so the bound rejects.
- **Why**: `SlabRemote::try_deallocate` enqueues a pointer for the owner thread to free. The deallocation crosses the thread boundary; if `T` isn't `Send`, the pointer cannot safely cross either.
- **Instead**: use `Arc<T>` if you need cross-thread reference counting inside slab slots, or restructure to keep the non-Send type on the owner thread.
- **Pinned by**: `compile_fail` doctest in `slab_owner.rs`.

### 5. `WithFallback<P, S>` where `P: !FixedRange`

- **Status**: CE
- **Mechanism**: `impl Allocator for WithFallback<P, S> where P: Allocator + FixedRange + Deallocator, S: ...` rejects any `P` that lacks `FixedRange`.
- **Why**: `WithFallback::deallocate` routes by pointer provenance — "is `ptr` in the primary's address range?" — which requires the primary to expose `FixedRange::base()` and `size()`. The secondary may be unbounded (e.g. `System`), and is therefore the catch-all.
- **Instead**: any in-tree primary that holds a `FixedRange`-implementing backing satisfies this (BumpArena, Slab, SizeClassed, etc.). System cannot be a primary; it can only be a secondary.
- **Pinned by**: `compile_fail` doctest in `fallback.rs`.

### 6. `Statistics<I>: Sync` where `I: !Sync`

- **Status**: CE
- **Mechanism**: `Sync` is auto-derived through inner; `Statistics<Slab<...>>` is `!Sync` because `Slab` is.
- **Why**: `Statistics` is a transparent observability wrapper — it inherits the inner's thread-safety profile. If the inner can't be shared across threads, neither can the statistics view.
- **Instead**: wrap a `Sync` inner (e.g. `SharedBumpArena`), or run the `Statistics`-wrapped allocator on a single thread.
- **Pinned by**: `compile_fail` doctest in `statistics.rs`.

---

## Construction errors (CX)

These compile but reject at `new()` / `with_*()` time. The error is
returned, not panicked — the caller gets a chance to fall back.

### 7. `Slab::new(0, _)` — zero capacity

- **Status**: CX → `Err(AllocError)`
- **Why**: a zero-slot slab is degenerate; every allocate would fail. Caught at construction to surface the bug at wiring time.

### 8. `NonZeroLayout::from_size_align(0, _)` — zero size

- **Status**: CX → `Err(LayoutError)`
- **Why**: per the type name, `NonZeroLayout` excludes ZST layouts. Use the `for_type::<T>()` constructor (which returns `Option`, with `None` for ZST T) when generic over T.

### 9. `NonZeroLayout::from_size_align(_, align)` where `align` is not a power of two

- **Status**: CX → `Err(LayoutError)`
- **Why**: standard layout invariant. Use `align_of::<T>()` or a known power-of-two constant.

### 10. `SizeClassed<InlineBacked<N>, _>` with class sizes > 16

- **Status**: CX → `Err(AllocError)` from the inner `backing.allocate()` when the class region needs alignment ≥ 32.
- **Why**: `InlineBacked` has `MAX_ALIGN = 16` (the alignment of its `#[repr(align(16))]` storage). `SizeClassed` allocates per-class regions whose alignment equals the class's largest slot size. If a class needs ≥ 32-byte alignment, the inline-backed region cannot supply it.
- **Instead**: use `MmapBacked` or `System` for `SizeClassed` backings when class sizes exceed 16 bytes. The default 8-class layout (8/16/32/64/128/256/512/1024) requires page-aligned backing.
- **V2 escalation**: see [§Deferred to v2.0](#deferred-to-v20) — a `MAX_ALIGN` associated const on `FixedRange` would promote this from CX to CT.

### 11. `CacheJitter::with_params` — invalid parameters

- **Status**: CX → `None`
- **Cases**:
  - `cache_line_size == 0`
  - `cache_line_size` not a power of two
  - `associativity == 0`
  - `associativity > MAX_ASSOCIATIVITY` (= `(1<<16) - 1` = 65535)
- **Instead**: typical values are `cache_line_size = 64`, `associativity = 8` (matches x86-64 L1 cache geometry).

### 12. `HardenedSlab::new(...)` with insufficient backing for split metadata

- **Status**: CX → `Err(AllocError)` from `SplitMetadata::new` if the metadata partition is larger than the backing.

---

## Footguns (FG) — works, but be aware

These compile, initialise, and run. They don't fail at any well-defined
point — they degrade quietly. Each entry says what the quiet degradation is.

### 13. `Statistics<ExtendableSlab>` / `Statistics<SizeClassed>` on hot paths

- **Status**: FG — observable overhead
- **Cost**: `corruption_events()` on these allocators walks segments/classes under a mutex per call, adding ~5–10 ns + 1 ns/segment per allocate. `Statistics` polls `inner.corruption_events()` on every allocate/deallocate.
- **Severity**: LOW at v0.1 scale; deferrable to v2.0 when the inner allocators cache a per-instance corruption counter.
- **Workaround**: for high-throughput paths, wrap `Statistics` around `Slab` directly (which has an O(1) cached counter), not around `ExtendableSlab` / `SizeClassed`.

### 14. `SharedBumpArena<InlineBacked<N>>`

- **Status**: FG — soundness rests on a documented `FixedRange` contract clause
- **What works**: in-tree backings (`InlineBacked`, `MmapBacked`) where `FixedRange::base()` and `size()` are pure reads.
- **What's risky**: a *user-implemented* `B: FixedRange` whose `base()` uses non-atomic interior mutability (e.g. lazy-init `Cell<NonNull<u8>>`) would race when wrapped here. `SharedBumpArena: Sync` is widened beyond `B: Sync` to `B: Send` for compatibility with `!Sync` backings like `InlineBacked`, which relies on the `FixedRange` contract clause that `base()`/`size()` be safe to call concurrently.
- **Severity**: LOW. The contract is documented; the widening is sound under it.

### 15. `SlabOwner` used dealloc-only

- **Status**: FIXED (was FG)
- **Pre-fix behaviour**: a `SlabOwner` whose owner thread never `allocate`s but does deallocs (locally or via accumulated `SlabRemote` pushes) would never call `maybe_drain` — the remote queue grew unbounded.
- **Post-fix**: `SlabOwner::deallocate` now calls `maybe_drain` (`try_lock`-gated, ~5–10 ns when empty). A v2.0 cached `AtomicUsize` is the next optimisation.

### 16. `Quarantine<BumpArena>`

- **Status**: FG — composition is structurally pointless
- **Why**: `Quarantine` delays slot reuse for EPOCHS deallocates before returning the slot to the inner. `BumpArena` never reuses slots anyway — its `deallocate` is a no-op (reclaim is via `reset()`). The quarantine ring becomes pure overhead; the inner sees no benefit.
- **Instead**: pair `Quarantine` with `Slab` (typed pool with freelist reuse) or `SizeClassed` (class-based reuse). Pairing with arena-style allocators is a category error.

### 17. `Canary<Statistics<I>>` vs `Statistics<Canary<I>>` ordering

- **Status**: FG — wrong ordering produces misleading metrics
- **Why**: `Canary` inflates each allocation by `2 * 8 = 16` bytes for pre/post sentinels. If `Canary` is on the *outside*, `Statistics`'s `bytes_allocated` counts the inflated layout, not the user request size. Operators see ~16 bytes extra per allocation in their dashboards.
- **Instead**: put `Statistics` *outside* `Canary` (i.e. `Statistics<Canary<I>>`) if you want metrics tracking the user-visible request size. Or accept the inflation if you want metrics tracking actual memory pressure.

### 18. `HugePageAligned<MmapBacked>` without kernel huge pages

- **Status**: FG on Linux; near-no-op on macOS / Windows
- **What works**: alignment enforcement to the huge-page boundary (2 MiB / 32 MiB).
- **What's opportunistic**: actual huge-page *promotion* requires `vm.nr_hugepages` to be configured AND the kernel's transparent-huge-page policy to enable it. Today's wrapper doesn't request `MAP_HUGETLB` explicitly — promotion is best-effort.
- **macOS / Windows**: NUMA bind via `NumaLocal` is a no-op at this revision; production NUMA on Windows belongs to a future `MmapBacked::with_numa_node`.
- **See**: `COMPOSITION_RECIPES.md` "NUMA-local + huge-page-aligned arena" for the production wiring caveats.

### 19. `GenerationalSlab` handle across two pools

- **Status**: FG — type system doesn't catch the misuse
- **Why**: `Handle<T, G>` is typed only by T and G, not by *which pool* issued it. A `Handle<Session, u32>` obtained from pool A, passed to pool B's `get(handle)`, may succeed (probability `1 / G::MAX`) and return a different value.
- **Instead**: don't share handles across distinct pools of the same `(T, G)`. Treat each pool as a closed namespace.
- **V2 escalation**: see [§Deferred to v2.0](#deferred-to-v20) — `generativity`-style invariant-lifetime branding would catch this at compile time.

### 20. `PacMAC` in production

- **Status**: FG — runtime panic
- **Why**: `PacMAC` is a stub for the ARM Pointer Authentication keyed MAC. The instruction-level body (PACIB/AUTIB) lands in M11; today's impl `panic!`s on the first `sign` / `verify` call.
- **Gating**: feature-flagged behind `pac-stub`, and the type is `#[deprecated]` so accidental use surfaces as a build warning.
- **Instead**: until M11, use `SipHashMAC` (`siphasher` feature) for keyed freelist authentication on aarch64.

### 21. `lazy_commit` `MmapBacked` under `SharedBumpArena` or `GuardPage`

- **Status**: FG — faults at first write (Windows only)
- **What works**: `BumpArena` / `StackAlloc` directly over a `lazy_commit` `MmapBacked` (`MmapBacked::new_lazy`), plus any pass-through `FixedRange` wrapper interposed between them (`Statistics`, `PoisonOnFree`, `Quarantine`, `Watermark`, `Canary`, `CacheJitter`, `Faulty`, `HugePageAligned`, `NumaLocal`, `SplitMetadata`) — these forward `FixedRange::commit`. `Slab` / `SizeClassed` / direct `Allocator::allocate` are also safe: they commit the block at allocation time (eager — no demand-paging benefit, but no fault).
- **What faults**: a lazy mapping reserves Windows pages without committing them, and writing one before `commit` raises an access violation. `SharedBumpArena` is `Sync` and cannot drive the `!Sync` commit watermark, so it deliberately does not call `commit`. `GuardPage` offsets its usable range past a guard page and its inner bound is only `OsBacked` (no `commit` to forward). Both leave the pages uncommitted, so the first write crashes.
- **Severity**: LOW — opt-in and Windows-only; the contract is spelled out on `MmapFlags::lazy_commit`. Inert on Unix (`mmap` is already demand-paged, so `commit` is a no-op and the flag changes nothing).
- **Instead**: use `MmapBacked::new` (eager commit) under `SharedBumpArena` / `GuardPage`, or keep the `lazy_commit` mapping directly under `BumpArena` / `StackAlloc`.

---

## Deferred to v2.0

These are known restrictions where today's enforcement is weaker than it
should be. Each requires an API-breaking change to enforce properly,
which we won't do mid-v0.1.

### 22. `SizeClassed<B, _>` backing-alignment as a compile-time check

- **Today**: runtime construction error (item 10).
- **V2**: a `MAX_ALIGN` associated const on the `FixedRange` trait would let `SizeClassed` `const_assert!(B::MAX_ALIGN >= max(class_strides))` at monomorphisation, turning the runtime error into a compile error.
- **Why deferred**: adds a required associated const to a public trait — API-breaking for any external `FixedRange` impl.

### 23. `GenerationalSlab` handle branding

- **Today**: cross-pool handle confusion is a runtime hazard (item 19).
- **V2**: `Handle<'pool, T, G>` carrying an invariant lifetime from its issuing pool (the `generativity` crate's pattern) would make cross-pool use a compile error. The `Handle: Copy` ergonomics would survive.
- **Why deferred**: every `Handle<T, G>` consumer signature gains a lifetime parameter — API-breaking.

### 24. `FixedRange::base()` / `size()` concurrent-call guarantee

- **Today**: implicit clause in the trait docs (item 14's footgun depends on this).
- **V2**: make the clause explicit via either (a) requiring `Self: Sync` to widen `SharedBumpArena: Sync`, or (b) a marker super-trait like `FixedRange + ConcurrentReadFixedRange`.
- **Why deferred**: option (a) breaks the documented `SharedBumpArena<InlineBacked>` recipe; option (b) adds a trait split. Both are API-breaking.

### 25. Bare-metal targets without native 64-bit atomics — RESOLVED in v0.1

- **Today**: all observability counters in `forge-alloc`'s `layout`
  module (`Slab::corruption_events`, `ExtendableSlab::routing_failures`,
  and `SizeClassed`'s per-class counters) and its `hardening` module
  (every `AllocStats` field — `total_allocations`, `bytes_allocated`,
  `corruption_events`, etc.) use `AtomicUsize`, not `AtomicU64`. The
  cross-compile CI matrix exercises `forge-alloc-core` and `forge-alloc`
  against `thumbv7em-none-eabihf` (Cortex-M4, 32-bit
  atomics only) and `wasm32-unknown-unknown` (no atomics by default —
  works because `AtomicUsize` lowers to non-atomic stores on
  single-threaded wasm).
- **Trait surface**: [`Allocator::corruption_events`] still returns
  `u64`; widening at the trait boundary is lossless on all targets.
  The same applies to [`AllocStats::current_bytes`] / [`peak_bytes`].
- **Practical caps on 32-bit hosts** (where `AtomicUsize` is 32 bits):
  - `bytes_allocated` / `bytes_peak` cap at `usize::MAX = 4 GiB`,
    which equals the address-space ceiling anyway.
  - `total_allocations` / `total_deallocations` / `failures` cap at
    `u32::MAX ≈ 4.3 B` ops. Advisory counters wrap silently after
    that.
  - `corruption_events` caps at `u32::MAX`. Even at one event per
    microsecond (already unrealistic), that is ~71 minutes of
    sustained corruption before wrap — well past any realistic
    operator response window.
- **Carve-outs**: no caller-visible API changed; the `AllocStats`
  struct retains `#[non_exhaustive]` and the helper methods retain
  their `u64`/`i64` return types.

### 26. `AllocStats` field additions

- **Today**: `#[non_exhaustive]` — additional observability counters can be added without breaking, but no enforcement mechanism for *which* fields a downstream `Statistics`-wrapper-equivalent must provide.
- **V2**: extracting `AllocStats` into a trait would let third-party wrappers conform. Today's monolith ships fine but isn't extensible from outside the crate.
- **Why deferred**: speculative — no third-party `Statistics` reimplementation has been requested.

---

## Detection summary table

| Item | Combination | Detected at | Severity |
|---|---|---|---|
| 1 | `Slab<(), _>` | Compile (const_assert) | CE |
| 2 | `InlineBacked<N>`, N % 16 ≠ 0 | Compile (const_assert) | CE |
| 3 | `SlabOwner: Sync` claim | Compile (auto-trait) | CE |
| 4 | `SlabRemote<!Send, _>: Send` claim | Compile (auto-trait) | CE |
| 5 | `WithFallback<P, _>` where `P: !FixedRange` | Compile (bound) | CE |
| 6 | `Statistics<!Sync>: Sync` claim | Compile (auto-trait) | CE |
| 7 | `Slab::new(0, _)` | Runtime construction | CX |
| 8 | `NonZeroLayout::from_size_align(0, _)` | Runtime construction | CX |
| 9 | Non-power-of-two alignment in `NonZeroLayout` | Runtime construction | CX |
| 10 | `SizeClassed<InlineBacked, _>` with > 16-byte classes | Runtime construction | CX → V2 |
| 11 | `CacheJitter::with_params` invalid | Runtime construction | CX |
| 12 | `HardenedSlab` undersized | Runtime construction | CX |
| 13 | `Statistics<ExtendableSlab>` / `Statistics<SizeClassed>` hot path | Runtime (overhead) | FG (perf) |
| 14 | `SharedBumpArena<custom !Sync B>` with impure `base()` | Runtime (data race) | FG (soundness) |
| 15 | `SlabOwner` dealloc-only | FIXED | — |
| 16 | `Quarantine<BumpArena>` | Runtime (pointless) | FG (design) |
| 17 | `Canary<Statistics<_>>` ordering | Runtime (misleading metrics) | FG (semantics) |
| 18 | `HugePageAligned` without kernel huge pages | Runtime (best-effort) | FG (degraded) |
| 19 | `GenerationalSlab` cross-pool handle | Runtime (silent wrong value) | FG → V2 |
| 20 | `PacMAC` in production | Runtime panic | FG |
| 21 | `lazy_commit` MmapBacked under `SharedBumpArena` / `GuardPage` | Runtime (fault on write, Windows) | FG (soundness) |
| 22 | `SizeClassed` backing alignment | V2 |
| 23 | `GenerationalSlab` handle branding | V2 |
| 24 | `FixedRange` concurrent-call guarantee | V2 |
| 25 | Bare-metal targets without 64-bit atomics | V2 |
| 26 | `AllocStats` extension trait | V2 |

---

## Cross-references

- [`COMPOSITION_RECIPES.md`](COMPOSITION_RECIPES.md) — recommended wirings with worked examples.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the three-layer mental model.
- Per-type `compile_fail` doctests pin items 1–6.
