# Composition Recipes

A field guide for composing `forge-alloc` primitives into application-
specific allocators. Each recipe shows the type signature, when to use it,
and what each wrapper layer buys you. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the layer model; this doc
is for *callers* assembling the pieces.

## Index

- [Stack-local scratch](#stack-local-scratch)
- [Cross-thread bump arena](#cross-thread-bump-arena)
- [Typed object pool](#typed-object-pool)
- [Multi-size general allocator](#multi-size-general-allocator)
- [Hardened slab for security-critical data](#hardened-slab-for-security-critical-data)
- [Bounded heap with overflow fallback](#bounded-heap-with-overflow-fallback)
- [Observable production allocator](#observable-production-allocator)
- [NUMA-local + huge-page-aligned arena](#numa-local--huge-page-aligned-arena)
- [Cross-thread typed allocator with adaptive batching](#cross-thread-typed-allocator-with-adaptive-batching)
- [Generational-handle slab (ABA-safe)](#generational-handle-slab-aba-safe)
- [Fault injection for OOM-path testing](#fault-injection-for-oom-path-testing)

---

## Stack-local scratch

```rust
type Scratch = BumpArena<InlineBacked<{ 64 * 1024 }>>;
```

64 KiB on the stack, bump-allocate scratch objects, `reset()` between
work units. Zero heap touches, zero atomic ops, `no_std` compatible.

**Use when**: per-request parsing scratch, formatter buffers, anything
with a clear "lifetime ends at this point" boundary.

**Pitfalls**: stack frames have OS-imposed limits (Linux default 8 MiB,
Windows 1 MiB per thread). Keep N reasonable.

---

## Cross-thread bump arena

```rust
type Shared = Arc<SharedBumpArena<MmapBacked>>;
```

`Arc<SharedBumpArena<...>>` lets multiple threads `allocate(&self)` via
an atomic-cursor CAS loop. No `reset` — drop the Arc and rebuild
when you want to reclaim.

**Use when**: a parser farm where worker threads each grab a slice of a
shared scratch region.

**Pitfalls**: `MmapBacked` is the right backing — `InlineBacked` is
`!Sync` because its own cursor uses `UnsafeCell` (a leftover from
single-thread BumpArena composition). Even though `SharedBumpArena`
doesn't *use* that cursor, the seal-by-design `Sync where B: Send`
holds because the backing is no longer reachable through `&self`.

---

## Typed object pool

```rust
type Pool<T> = Slab<T, MmapBacked>;
```

Fixed-stride O(1) alloc/dealloc with an in-band freelist. Capacity
fixed at construction; `grow` returns `AllocError`.

**Use when**: per-message objects with bounded lifetimes
(connections, sessions, audit entries) where you want to size the
pool once at startup.

**Pitfalls**: `T` must not be a ZST. The returned slice's length is
the block stride, not `layout.size()`. Caller must run `T::drop`
before `deallocate` — `Slab::drop` does NOT iterate live slots.

---

## Multi-size general allocator

```rust
type GP = SizeClassed<MmapBacked, 8>;
let gp = SizeClassed::with_default_classes(MmapBacked::new(4 * 1024 * 1024)?, 64)?;
```

Eight class slabs at the spec defaults (8/16/32/64/128/256/512/1024
bytes), 64 slots each. Routes requests to the smallest fitting class
or falls back to the inner backing for oversized requests.

**Use when**: a general allocator for an application with bounded but
unpredictable object sizes — better fragmentation behaviour than a
single bump arena, simpler than a full malloc.

**Pitfalls**: each class region requires backing alignment up to that
class's stride. `InlineBacked` (`MAX_ALIGN = 16`) caps at 16-byte
classes; use `MmapBacked` or `System` for larger.

---

## Hardened slab for security-critical data

```rust
type ClaimPool = HardenedSlab<ClaimRecord>;
let pool = Slab::<ClaimRecord, _>::new(
    1024,
    GuardPage::new(
        SplitMetadata::new(MmapBacked::new(4 * 1024 * 1024)?, 64 * 1024)?,
        4096,
    )?,
)?;
```

`HardenedSlab<T, M = NoProtection>` expands to
`Slab<T, GuardPage<SplitMetadata<MmapBacked>>, M>`. Three lines of
defense:

- **Guard pages** trap linear overflows on either side of the
  metadata + data regions.
- **Split metadata** lives at an unrelated virtual address, so even
  an in-bounds-of-data overflow can't reach allocator bookkeeping.
- (Optional) **Freelist MAC** (`M = SipHashMAC` with the
  `siphasher` feature) authenticates freelist links; a forged link
  is rejected at allocation time.

**Use when**: PHI handling, key material, audit logs, anything where
the security property "no overflow can corrupt the allocator" is
worth the per-allocation overhead.

---

## Bounded heap with overflow fallback

```rust
type Fast = WithFallback<BumpArena<InlineBacked<{ 1024 * 1024 }>>, System>;
```

1 MiB bump arena for the common path, system allocator for overflow.
Deallocation routes by pointer-range provenance.

**Use when**: a hot path with a predictable working set, but where
*occasional* large allocations must still succeed without aborting.

**Pitfalls**: the `Primary` must implement `FixedRange` (so the
router can resolve deallocations). `System` is the natural
secondary because it accepts any pointer.

**Alternate constructor — `WithFallback::try_new`**: when *both*
halves implement `FixedRange` (e.g. two independent `MmapBacked`
regions), prefer `try_new` over `new`. `try_new` verifies the
primary's and secondary's address ranges are disjoint at
construction and returns `Err(AllocError)` on overlap.
Overlapping ranges with the unchecked `new` silently misroute
secondary-issued pointers through `primary.deallocate`, producing
a freelist corruption that's hard to diagnose after the fact. The
default secondary `System` is not `FixedRange`, so the
common `WithFallback<_, System>` wiring stays on `new`.

---

## Observable production allocator

```rust
type Prod<T> = Watermark<
    Statistics<
        PoisonOnFree<Slab<T, MmapBacked>>,
    >,
    LogHandler,
>;
```

Layered from innermost:

- **Slab over MmapBacked** — the actual allocation.
- **PoisonOnFree** — overwrites freed memory with a sentinel pattern
  so use-after-free reads are visible.
- **Statistics** — atomic counters for alloc / dealloc / failure /
  peak-bytes.
- **Watermark<_, LogHandler>** — fires `log::warn!` / `error!` when
  utilization crosses configurable thresholds.

**Use when**: production deployments where operators need observability
on top of a fast typed slab.

---

## NUMA-local + huge-page-aligned arena

```rust
let backing = MmapBacked::new(64 * 1024 * 1024)?; // 64 MiB
let huge    = HugePageAligned::new(backing).ok_or(/* … */)?;
let numa    = NumaLocal::new(huge, NumaPolicy::Bind(NodeSet::single(0).unwrap()))?;
let arena   = BumpArena::new(numa)?;
```

64 MiB OS region, 2 MiB-aligned for THP promotion, bound to NUMA
node 0, served by a single-thread bump arena.

**Use when**: latency-bound work that must run on a known socket and
benefits from large pages (database buffer pools, ML inference state).

**Pitfalls**: huge-page promotion is opportunistic on Linux unless
`MAP_HUGETLB` is requested. `HugePageAligned` enforces *alignment*
but doesn't bypass `vm.nr_hugepages` — the kernel still needs free
huge pages. NUMA bind on macOS / Windows is a no-op at this
revision; production NUMA on Windows belongs to a future
`MmapBacked::with_numa_node`.

---

## Cross-thread typed allocator with adaptive batching

```rust
let owner: SlabOwner<RequestCtx, MmapBacked> =
    SlabOwner::with_batch_policy(
        4096,
        MmapBacked::new(4 * 1024 * 1024)?,
        BatchPolicy::Adaptive,
        1024,
    )?;
let remote = owner.remote(); // Send + Sync — ship to worker threads
```

Owner thread allocates locally (fast); worker threads enqueue
deallocations via `remote.deallocate` / `remote.try_deallocate`. The
adaptive policy steps through batch thresholds {8, 16, 32, 64, 128}
based on observed queue depth.

**Use when**: a producer/consumer pattern where one thread allocates
and many others free (e.g., async runtime task bodies).

**Pitfalls**: the owner is `!Sync`; clone the remote handle, not the
owner. v0.1 uses a `Mutex<VecDeque>` queue; the lock-free MPSC
upgrade ships in v1.0 without API change.

---

## Generational-handle slab (ABA-safe)

```rust
let pool: GenerationalSlab<Session, _> =
    GenerationalSlab::new(1024, MmapBacked::new(64 * 1024)?)?;
let handle: Handle<Session, u32> = pool.insert(Session::new())?;

// Later, possibly after the underlying slot has been freed and reused:
if let Some(session) = pool.get(handle) {
    // … safe to use; generation matches.
}
```

`Handle<T, G>` carries both a slot index and a generation counter;
`get(handle)` returns `Some` only if the slot still holds the
generation that was current at allocate time. ABA-safe replacement
for raw `Box<T>`-style references.

**Use when**: long-lived references to slab entries where the
underlying slot may legitimately be recycled, and you need to
distinguish "still valid" from "recycled into a different value".

**Pitfalls**: `Handle<T, G>` is `Copy`, but its validity is
per-slab. Don't share a handle across two `GenerationalSlab`
instances of the same type — the second one's generation counter
is independent.

---

## Fault injection for OOM-path testing

```rust
// Test-only: force the WithFallback secondary path on every request.
type ChaosTest =
    WithFallback<Faulty<BumpArena<MmapBacked>, AlwaysFail>, System>;
```

`Faulty<I, P>` wraps any allocator and consults an `AllocFaultPolicy`
before each `allocate`, returning `AllocError` when the policy votes to
fail. The inner allocator is never touched on a faulted request, so an
injected failure is observationally identical to a genuine OOM. This
turns the normally-unreachable out-of-memory branch of every allocator
and composition into something a unit test, a `proptest` case, or a
fuzz target can drive deterministically. Built-in policies: `NeverFail`,
`AlwaysFail`, `FailAfter`, `FailEveryNth`, `FailOnSize`.

The example pairs `Faulty` with the *Bounded heap with overflow
fallback* recipe above: an `AlwaysFail` primary forces every request
down the `System` fallback, so that branch (almost never hit in a real
run) gets exercised on purpose.

**Use when**: testing the error paths that production runs rarely
reach, such as `AllocError` handling, `WithFallback` secondary routing,
and graceful degradation.

**Pitfalls**: test and debug builds only. A `Faulty` left in a shipped
allocator stack is an allocator that fails for no reason. Place it just
above the allocator whose OOM you want to simulate and below any
observability wrappers, e.g. `Statistics<Faulty<Slab<T>>>`, so the
injected failure is counted exactly as a real one would be.
