# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-25

### Changed (BREAKING - `forge-alloc` only; `forge-alloc-core` unchanged)
- `CachePadded<T>` is now target-aware. On `x86_64`, `aarch64`, and
  `powerpc64` its alignment grew from 64 bytes to **128 bytes** (Apple
  Silicon uses 128-byte cache coherency, and on x86_64 the adjacent-line
  prefetcher pulls cache lines in pairs, so a 64-byte pad still allows
  false sharing across the prefetched neighbor). Per-arch values: 128 on
  x86_64/aarch64/powerpc64, 32 on arm/mips/mips64/sparc/hexagon, 16 on
  m68k, 256 on s390x, 64 fallback. This is a layout-breaking change for
  any consumer that asserts the size of `CachePadded<T>` directly; the
  public path `forge_alloc::CachePadded` is unchanged.
- `CachePadded` moved from `forge_alloc::hardening` (Layer 3) to the
  crate root so it can be used by Layer 2 (`layout/`) without violating
  the bottom-up layer DAG. The public path `forge_alloc::CachePadded`
  still works; the secondary path `forge_alloc::hardening::CachePadded`
  no longer resolves.

### Added
- `forge_alloc::CACHE_LINE`: target-specific cache-line size constant
  surfaced for downstream layout pins.
- `Watermark::allocated` and `Watermark::fired` are now wrapped in
  `CachePadded` so the rising-edge `fetch_or` on `fired` does not
  invalidate the `allocated` counter's line on every concurrent
  allocate.
- `SharedBumpArena::cursor` is now wrapped in `CachePadded` so the
  contended CAS on the cursor does not invalidate the read-only
  `backing` and `capacity` fields on every retry.
- Compile-time `LAYOUT_PIN` assertions on `AllocStats`, `SlabInner`,
  `Watermark`, and `SharedBumpArena` lock in the cache-line separation
  invariants. A future refactor that reshuffles fields back onto the
  same line fails the build with a clear error pointing at the
  affected struct.

## [0.1.0] - 2026-05-21

### Changed (crate consolidation)
- The workspace was consolidated from five published crates to two.
  `forge-backing`, `forge-layout`, and `forge-hardening` are now modules
  (`backing`, `layout`, `hardening`) of the `forge-alloc` crate; the
  trait-contract crate is published as `forge-alloc-core` (renamed from
  `forge-core`, which was unavailable on crates.io). The public API is
  unchanged — `use forge_alloc::*` exposes exactly the same surface, with
  `forge-alloc-core` re-exported.

### Added (M9 — performance hardening)
- `CacheJitter<I>` — randomized per-allocation displacement (xorshift64-derived)
  to spread metadata across cache associativity sets. Header at `user_ptr - 8`
  stores displacement so deallocate recovers `inner_ptr`. Pass-through for
  layouts with `align > cache_line_size`. (`forge-hardening`)
- `HugePageAligned<I: OsBacked>` — enforces 2 MiB (32 MiB on Apple Silicon)
  allocation alignment and rounds `release_pages` requests inward to whole
  huge pages so partial purges can't demote a promoted huge page back to
  4 KiB. `default_huge_page_size()` exposes the platform default.
  (`forge-hardening`)
- `NumaLocal<I: OsBacked>` — applies a NUMA placement policy to the inner
  region once at construction via `mbind()` on Linux (no-op on macOS /
  Windows). `NumaPolicy::{Bind, Preferred, Interleaved}` over a `NodeSet`
  bitmask (up to 64 nodes). `current_numa_node()` detects the calling
  thread's node via `getcpu(2)`. (`forge-hardening`)

### Added (M12 — adaptive batch policy)
- `BatchPolicy::Adaptive` — stepped-threshold control law with 5 levels
  (8, 16, 32, 64, 128), 25% / 75% hysteresis bands, and a 16-tick cooldown
  between adjustments. All-integer arithmetic; no floating point. Initial
  step = 3 (threshold = 64) matches `Fixed(64)`. (`forge-layout`)
- `SlabOwner::adaptive_threshold_snapshot()` — telemetry accessor for the
  current adaptive threshold (returns `None` under `Fixed`).
- `crates/forge-bench/benches/adaptive_batch.rs` — Criterion bench harness
  comparing Fixed(8) / Fixed(64) / Fixed(128) / Adaptive under a 2-sender
  remote-free workload. Used as the validation gate before the v2.0 EMA
  control law ships.

### Added (backlog burn-down)
- `SizeClassed<B, CLASSES>` — array of `CLASSES` untyped slabs with
  geometrically increasing block sizes. Routes allocate requests to the
  smallest fitting class (size + alignment); oversized / over-aligned
  requests fall through to the backing. `DEFAULT_CLASS_SIZES_8` ships
  the spec's 8 / 16 / 32 / 64 / 128 / 256 / 512 / 1024 set. (`forge-layout`)
- `SplitMetadata<I: Allocator>` — wraps any allocator with a
  separate `MmapBacked` metadata region (forwards `OsBacked` only
  when `I: OsBacked`). Data + metadata live at
  unrelated virtual addresses so a linear overflow past any user
  allocation cannot reach allocator bookkeeping. `#[must_use]` annotation
  reminds callers that full coverage requires `GuardPage<SplitMetadata<_>>`.
  (`forge-hardening`)
- `mmap_last_os_error()` / `mmap_clear_last_os_error()` — per-thread slot
  capturing errno / `GetLastError` on the most recent failing mmap-layer
  syscall (`os_map` / `os_unmap` / `os_release_pages` / `os_protect`).
  Distinguishes ENOMEM vs EACCES vs EOVERFLOW etc. without recompiling.
  (`forge-backing`)
- `pac-stub` feature flag in `forge-core` and `forge-alloc` — separates
  the panicking-stub PacMAC type from the future M11 `pac` feature so
  production builds can't silently pick up a panicking codepath.

### Added (M13 — verification + observability)
- `Allocator::corruption_events() -> u64` — default-impl trait method
  exposing the count of detected freelist / metadata corruption events
  (MAC-verify failures, out-of-range `next_idx` tripwires, wrong-pointer
  deallocations). `Slab`, `ExtendableSlab`, and `SizeClassed` override it
  with real counters; every Layer-3 wrapper forwards it; `AllocStats`
  gains a mirrored `corruption_events` field. Makes silent
  defense-in-depth disarms observable to operators. (`forge-core`,
  `forge-layout`, `forge-hardening`)
- `Faulty<I, P>` + the `AllocFaultPolicy` seam — a **test/debug-only**
  Layer-3 wrapper that forces allocations to fail per a policy, making
  every allocator's out-of-memory `Err` path reachable from tests,
  proptest, fuzz, MIRI, and Kani. Ships five built-in policies:
  `NeverFail`, `AlwaysFail`, `FailAfter`, `FailEveryNth`, `FailOnSize`.
  The trait is dependency-free so a seeded/replayable policy can be
  implemented downstream without `forge-*` gaining a dependency.
  (`forge-core`, `forge-hardening`)

### Changed (adversarial review — passes #1–#4)
- `Statistics::deallocate` and `Watermark::deallocate` now use `fetch_sub`
  (single `lock xadd` on x86_64) instead of the previous `fetch_update`
  CAS loop. The CAS loop saturated at zero to defend against UB caller
  bugs, but under contention from `N` threads each conflicting RMW
  triggered another retry — dealloc cost scaled with thread count. The
  `Deallocator` contract already guarantees `prev >= size` for a correct
  caller; a `debug_assert!` catches the UB caller bug. Allocate retains
  `saturating_add` on `bytes_allocated` so a (UB-induced) wraparound
  doesn't promote into a debug-mode panic on the next allocate.
  (`forge-hardening`)
- `Watermark` now pre-computes `warn_threshold_bytes` at construction and
  gates the (`#[cold]`) `check_and_fire` call with a `new_bytes >= gate`
  comparison on the hot allocate path — sub-threshold allocates skip
  the call entirely. Equivalent gate at `usize::MAX` for unbounded
  inners. (`forge-hardening`)
- `SipHashMAC::Clone` rewritten as a per-byte `read_volatile` /
  `write_volatile` loop (with `SeqCst` `compiler_fence`) instead of
  the derived memcpy. The derived `Clone` lowered to a vectorizable
  memcpy that could leave transient byte-aligned copies of the key in
  caller stack frames outside the original `key` slot — which `Drop`
  could then not zeroize. `ZeroingHasher` (a new internal wrapper)
  also volatile-zeros the `SipHasher13`'s key-derived internal state
  (`v0..v3`, `m`, `tail`, `ntail`) on drop so key-equivalent bits
  don't linger on the `mac()`-callee's stack frame. (`forge-core`)
- `Canary::deallocate` now volatile-zeros the pre- and post-canary words
  after verification, paired with a `SeqCst` `compiler_fence`. The seed
  is a per-process secret; without on-free clearing, code that later
  borrowed the freed region (slab freelist reuse, `BumpArena::reset`,
  `mmap` remap) could lift the seed via a UAF read and use it to forge
  canaries elsewhere in the process. (`forge-hardening`)
- `CacheJitter` now stores a per-instance `mac_key` (derived from the
  caller seed via two SplitMix64 steps) and computes a 48-bit keyed MAC
  over the `(user_ptr, displacement_in_lines)` header. `deallocate`
  verifies the MAC in constant time before trusting the embedded
  displacement — without this, an attacker who can blind-write the
  prefix bytes (linear underflow from an adjacent allocation, UAF
  prefix write into a freed slot) would have an arbitrary-free
  primitive against the inner allocator. `associativity` capped at
  `MAX_ASSOCIATIVITY = 2^16 - 1` to fit the 16-bit displacement field.
  Key and RNG state are volatile-zeroed on `Drop`. (`forge-hardening`)
- `WithFallback::try_new` added (gated on `P: FixedRange + S: FixedRange`)
  — verifies that the primary and secondary address ranges are disjoint
  at construction. Overlapping ranges with `WithFallback::new` silently
  misroute secondary-issued pointers through the primary's `deallocate`,
  producing freelist corruption that's hard to debug after the fact. The
  default secondary (`forge_backing::System`) does not implement
  `FixedRange`, so the common `WithFallback<_, System>` wiring continues
  to use `new`. (`forge-layout`)
- `SlabOwner::Drop` now performs a final drain of the remote-free
  queue and sets a `closed: AtomicBool` flag on the shared inner state.
  `SlabRemote::try_deallocate` observes the flag and returns
  `Err(ptr)` immediately instead of pushing; `SlabRemote::deallocate`
  (the spinning impl) reads the flag inside its retry loop and bails
  out with the pointer un-pushed. Without this, a remote sender could
  spin forever after owner drop, and an unbounded queue would grow
  with no drainer. (`forge-layout`)

### Changed
- `SharedBumpArena<B>` no longer exposes a `backing()` accessor.
  Previously the wrapper claimed `Sync where B: Send` while
  `backing() -> &B` would let multiple threads call `B`'s `&self`
  methods concurrently; `InlineBacked` / `MmapBacked` are `!Sync` via
  `UnsafeCell<usize>` cursors, so the leaked `&B` could race on
  `B`'s interior mutability. The backing is sealed inside the
  arena. (`forge-layout`)
- macOS / BSD `os_release_pages` now uses `MADV_FREE` instead of
  `MADV_DONTNEED` — the latter is hint-only on macOS and may be
  ignored. `MADV_FREE` provides the documented "kernel may reclaim
  under pressure" semantics. Linux retains `MADV_DONTNEED` for the
  immediate zero-fill-on-read semantics. (`forge-backing`)
- Windows `os_protect` now `debug_assert!`s when a write-without-read
  or exec-without-read flag combination is requested. Windows page
  protection can't express "no read", so the implementation silently
  upgrades to `PAGE_READWRITE` / `PAGE_EXECUTE_READ`; the assertion
  catches accidental misuse in dev. (`forge-backing`)
- Windows `os_release_pages` now `debug_assert!`s page-aligned ptr +
  size. `VirtualAlloc(MEM_RESET)` rejects misaligned ranges with
  `ERROR_INVALID_PARAMETER`; observability via `mmap_last_os_error`
  was already there but the debug-build assertion catches misuse
  at the call site.
- `Canary` panic messages no longer leak the expected canary value.
  The seed is per-process entropy used to authenticate freelist links;
  panic strings end up in log scrapers and crash reporters, so the
  release-build message now reports only the corruption-site label.
  Debug builds additionally include the observed corrupting bytes for
  debugging. (`forge-hardening`)
- `Deallocator` / `Allocator` / `OsBacked` `# Safety` docs tightened
  to a multi-clause invariant list — same-allocation-domain pointers,
  exact-layout match at deallocate, no-read-after-free including
  pointer-value compare, grow/shrink in-place-vs-fresh
  disambiguation, page-alignment portability notes. (`forge-core`)
- `NonZeroLayout::array<T>` calls `pad_to_align` before the size
  multiply — matches `core::alloc::Layout::array`'s behaviour exactly.
  For sized `T` this is a no-op (the trailing-pad rule is already
  encoded in `size_of`); the change is future-defence. (`forge-core`)

### Changed (M13 — portability + hot-path)
- All observability counters across the family switched from
  `AtomicU64` to `AtomicUsize` — `Slab::corruption_events`,
  `ExtendableSlab::routing_failures`, `SizeClassed`'s per-class
  counters, and every `AllocStats` field. `Allocator::corruption_events`
  keeps its `u64` return; the widen happens at the trait boundary. This
  lets `forge-core` / `forge-layout` / `forge-hardening` compile on
  32-bit bare-metal targets (`thumbv7em-none-eabihf`, Cortex-M4) and
  pre-atomics `wasm32-unknown-unknown`, which lack native 64-bit atomic
  ops. The cross-compile CI matrix now covers all three crates on those
  targets. (`forge-core`, `forge-layout`, `forge-hardening`)
- `SlabOwner::deallocate` fast path: a cached `AtomicUsize`
  queue-length mirror replaces an uncontended `try_lock` +
  `VecDeque::len()` on every owner-side deallocate, collapsing the
  no-pending-work common case to a single relaxed load. (`forge-layout`)

### Removed
- `stats` feature in `forge-hardening` and `forge-alloc` — it was
  declared but never gated on anywhere. The real opt-in is wrapping
  with `Statistics<I>`, which costs zero for any unwrapped allocator.

### Earlier milestones (M1 – M8) — initial scope
- **M1** — trait foundation: `Allocator` / `Deallocator` split,
  `NonZeroLayout`, `StdCompat<A>`, `OsBacked`, `FixedRange`,
  `FreelistProtection`. (`forge-core`)
- **M2** — Layer 1 backings: `InlineBacked<N>`, `MmapBacked`,
  `System`. (`forge-backing`)
- **M3** — Layer 2 layout: `BumpArena<B>`, `SharedBumpArena<B>`,
  `BumpDeallocator<'_>`, `Slab<T, B, M>`, `WithFallback<P, S>`.
  (`forge-layout`)
- **M4** — validation gate: proptest correctness suite, Criterion
  alloc-throughput benchmarks, cargo-fuzz scaffolding (`forge-fuzz`),
  `no_std` compile tests.
- **M5** — Layer 3 hardening: `Canary<I>` (seeded + OS-RNG
  constructors), `PoisonOnFree<I>`, `Statistics<I>`,
  `Watermark<I, H>`. (`forge-hardening`)
- **M6** — UAF + metadata isolation: `Quarantine<I, EPOCHS>`.
  (`SplitMetadata` deferred to backlog burn-down above.)
- **M7** — guard + remaining layout: `GuardPage<I>`, `StackAlloc<B>`
  (frame-stack LIFO), `ExtendableSlab<T, M>`. (`SizeClassed` deferred
  to backlog burn-down above.)
- **M8** — handle safety + cross-thread frees: `GenerationalSlab<T, B>`,
  `Handle<T, G>`, `SlabOwner<T, B>` / `SlabRemote<T, B>` /
  `RemoteFreeQueue`.

[Unreleased]: https://github.com/dmaesj/forge-alloc/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dmaesj/forge-alloc/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dmaesj/forge-alloc/releases/tag/v0.1.0
