# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.7] - 2026-06-04

`forge-alloc` 0.3.7 (`forge-alloc-core` unchanged at `0.2.3`). An additive
release rounding out the `BumpArena` API and adding arena pooling. No breaking
API changes.

### Added — `forge-alloc`
- **`ArenaPool<B, F>`** — recycle reset `BumpArena`s across a per-commit /
  per-branch workload. `checkout` hands one out (reusing a reset idle arena or
  minting via the factory); `give_back` resets in O(1) and retains it up to a
  cap, so in steady state the same mappings are reused with **zero**
  `munmap` / `mmap` / re-fault. `prewarm` pre-mints, `clear` drops all idle, and
  `release_idle` (when `B: OsBacked`) drops idle arenas' physical pages via
  `madvise(DONTNEED)` / `MEM_RESET` while keeping the virtual reservation warm —
  bounding resident memory without leaving the pool. Needs only `alloc`.
- **Typed `BumpArena` allocation** — `alloc_uninit::<T>()` (compile-time
  size/align so the bounds + alignment math fold; ZST returns a dangling-aligned
  pointer), plus `alloc::<T>(value)`, `alloc_slice_copy::<T: Copy>(&[T])`, and
  `alloc_str(&str)` convenience methods.
- **`BumpArena::scope()` → `Scope`** — a nestable, panic-safe RAII scratch
  checkpoint. Allocations through the scope (`alloc_uninit` / `alloc` /
  `alloc_slice_copy` / `alloc_str`, plus raw `allocate`) borrow the guard, so the
  borrow checker forbids them from outliving the cursor rewind on `Drop` (which
  runs on a panic unwind too). Also exposes `rewind_to(mark)`, the underlying
  checkpoint primitive.
- **In-place `grow`** — `BumpArena`'s `Allocator::grow` now resizes the
  most-recent allocation (the one ending at the cursor) by a cursor advance with
  no copy; other blocks relocate as before. Speeds arena-backed `Vec`/`String`
  building.

## [0.3.6] - 2026-06-02

`forge-alloc` 0.3.6 (`forge-alloc-core` unchanged at `0.2.3`). A small additive
release exposing `release_pages` through `BumpArena`. No breaking API changes.

### Added — `forge-alloc`
- **`BumpArena` now implements `OsBacked`** (forwarding `base_ptr` /
  `region_size` / `release_pages` / `protect` to its backing) when the backing
  is `OsBacked`. This lets an arena pool reclaim a reset arena's physical pages
  via `release_pages` (`madvise(DONTNEED)` / `MEM_RESET`) on pool overflow —
  bounding resident memory without `munmap`/re-`mmap` churn or the demand-zero
  re-fault storm a fresh mapping would incur — while keeping the virtual
  reservation warm for reuse.

## [0.3.5] - 2026-05-31

`forge-alloc` 0.3.5 (`forge-alloc-core` unchanged at `0.2.3`). An additive
release completing the "crypto allocator" composition: a RAM-locking backing
plus a first-class `CryptoSlab` alias that snaps it together with the hardened
slab and scrub-on-free. No breaking API changes.

### Added — `forge-alloc`
- **`LockedMmapBacked`**: an `MmapBacked` whose pages are pinned in RAM with
  `mlock` / `VirtualLock` so secret data never pages to swap. Construction
  **fails closed** — if the lock cannot be taken (`RLIMIT_MEMLOCK`, missing
  privilege) `new` returns `Err` rather than silently leaving the region
  swappable. On Linux it additionally applies `MADV_DONTDUMP` (best-effort) to
  exclude the pages from core dumps. Unlocks before unmapping on drop. Forwards
  the full `OsBacked` surface; `release_pages` is a documented no-op (purging a
  locked region would defeat the lock).
- **`CryptoSlab<T, M>`** type alias — the recommended composition for
  cryptographic key material:
  `ZeroizeOnFree<Slab<T, GuardPage<SplitMetadata<LockedMmapBacked>>, M>>`. It
  layers the two crypto guarantees (no-swap lock + core-dump exclusion, and
  non-elidable scrub-on-free) on top of the existing `HardenedSlab` stack (guard
  pages, split metadata, optional freelist MAC). Documents its threat-model
  boundary: the lock prevents swap, not hibernation / `fork()` COW / `ptrace`,
  and only the data region is locked (metadata and the MAC key are not, by
  design). Proven end-to-end by a new integration test that exercises the lock
  and verifies a freed secret is scrubbed past the freelink.

## [0.3.4] - 2026-05-31

`forge-alloc` 0.3.4 and `forge-alloc-core` 0.2.3. A security and correctness
release: the additive `ZeroizeOnFree` wrapper plus a batch of fixes from an
exhaustive, looped adversarial review of the entire workspace (backing →
layout → hardening). No breaking API changes. The new `forge-alloc-capi` C ABI
crate (`0.1.0`) shipped in the same cycle and is released separately.

### Added
- **`forge-alloc-capi`** (new crate, `0.1.0`): a C ABI over
  `forge-alloc` for C/C++, aimed at embedded users. Exposes a hardened bump
  arena over a caller-provided buffer (`BumpArena<StaticBacked>`) — no global
  allocator, no syscalls, `#![no_std]`-capable down to Cortex-M / wasm.
  `forge_bump_free` scrubs freed blocks with the poison pattern. Ships a C
  header, C and C++ examples, and an FFI integration test. The ABI was
  verified end-to-end against MSVC `cl` (C and C++). See `docs/C_API.md`.

### Added — `forge-alloc`
- `ZeroizeOnFree<I>`: a crypto-grade hardening wrapper that volatile-zeroes
  freed memory. Unlike `PoisonOnFree` (plain `write_bytes`, which the optimizer
  may dead-store-eliminate), the scrub uses `write_volatile` + a compiler fence
  so it cannot be elided, and it uses the trait-default `grow`/`shrink` so a
  moved-from block and `shrink`'s discarded tail are erased too. No new
  dependency. Re-exported from the crate root alongside `PoisonOnFree`.

### Added — CI / project hygiene
- Supply-chain gate (`deny.toml` + cargo-deny CI job): advisories, license
  policy, and source vetting on every push.
- MSRV CI job pinned to Rust 1.84 for the published crates.
- `[package.metadata.docs.rs] all-features = true` on both library crates.
- `SECURITY.md` vulnerability-disclosure policy.
- `forge-alloc-capi` CI coverage: a `no_std` rlib check in the `no_std` job,
  plus a `capi_ffi` job that compiles and runs the C and C++ examples against
  the built shared library and builds the embedded `staticlib-rt` static
  library for `thumbv7em-none-eabihf` — making the previously-manual MSVC
  verification CI-enforced.

### Security — `forge-alloc`
- **Freed-secret slack scrub.** Root allocators that round an allocation up to
  a slot/class (`Slab`, `SizeClassed`, `ExtendableSlab`, `SlabOwner`) now report
  their true usable extent via `usable_size`, and layout-transparent wrappers
  (`Quarantine`, `Statistics`, `Watermark`, `NumaLocal`, `SplitMetadata`,
  `WithFallback`, `HugePageAligned`) forward it. Previously these returned
  `None`, so an outer `PoisonOnFree`/`ZeroizeOnFree` scrubbed only the requested
  size and left the rounding slack (freed secret bytes) un-scrubbed.
- **`GenerationalSlab` ABA horizon closed.** A slot whose generation wraps is
  now permanently retired instead of being re-issued at generation 0, closing
  the use-after-free window that the wraparound previously left open. Added a
  freelist-pop out-of-bounds guard.
- **`Canary` scrub was UB.** The on-free canary wipe used a single wide volatile
  store that is undefined behavior at sub-8-byte alignment; it is now a
  byte-wise `write_volatile`, sound at any alignment and still non-elidable.

### Fixed — `forge-alloc`
- **`Slab` freelist-MAC move-safety.** The SipHash freelist MAC was keyed on the
  free-time *address*; a `Slab` moved between a free and a later alloc (legal
  with move-relative backings like `InlineBacked`) would then false-fail
  verification, leak the slot, bump `corruption_events`, and debug-panic on
  otherwise-valid code. The MAC is now keyed on the move-invariant slot index.
- **`MmapBacked`**: `page_size()` now rejects a non-power-of-two value before it
  reaches the round-up masks; on Windows, `release_pages` clamps the
  `MEM_RESET` range to the committed high-water mark so it cannot fail on
  reserved-but-uncommitted tail pages.
- **`HugePageBacked`**: on macOS x86_64 the mapping length is rounded to the
  fixed 2 MiB superpage size, so a sub-2-MiB `huge_page_size` no longer makes
  the superpage `mmap` reject the request; `huge_page_size()` reports the
  effective granularity.
- Numerous documentation corrections surfaced by the review — most notably the
  `Slab` double-free note (no protection level, including `SipHashMAC`, detects
  a base-of-slot double-free) and tightened scrub/non-elision wording.

### Changed — `forge-alloc-core` (0.2.3)
- Corrected the `corruption_events` doc (Canary/CacheJitter do not keep a
  counter) and documented the `FixedRange::base`/`size` concurrency contract.
- `NonZeroLayout::pad_to_align` gained a `debug_assert` pinning its
  `isize::MAX` layout-bound invariant; overflow-justification comments made
  target-width-independent. No API change.

## [0.3.3] - 2026-05-28

Internal performance fix. No API or behavior changes; `forge-alloc-core`
is unchanged and stays at 0.2.2.

### Changed — `forge-alloc`
- `backing::mmap::page_size()` now caches its result in an `AtomicUsize`.
  On Windows the demand-commit path (`MmapBacked` lazy commit) consulted
  `GetSystemInfo` on every commit; it is now fetched once, removing a
  per-commit syscall from the hot path. The Unix `sysconf` path is cached
  the same way for symmetry.

## [0.3.2] - 2026-05-28 (includes forge-alloc-core 0.2.2)

Additive release: opt-in Windows demand-commit for `MmapBacked`. No
breaking changes.

### Added — `forge-alloc`
- `MmapFlags::lazy_commit` and `MmapBacked::new_lazy(size)`: reserve a
  Windows region (`VirtualAlloc(MEM_RESERVE)`) without charging the
  system commit limit up front, committing pages on demand as a
  `BumpArena` / `StackAlloc` cursor advances. This avoids the
  full-reservation commit charge for a large, mostly-untouched arena —
  Windows does not overcommit, so the previous eager
  `MEM_RESERVE | MEM_COMMIT` charged the entire region even if only a
  fraction was ever written. Inert on Unix (`mmap` is already
  demand-paged), where `commit` is a no-op.
- Commit is fallible and runs before a consumer publishes its cursor, so
  a declined reservation surfaces as `AllocError` rather than a hard
  access violation on first write.
- The pass-through `FixedRange` wrappers (`Statistics`, `PoisonOnFree`,
  `Quarantine`, `Watermark`, `Canary`, `CacheJitter`, `Faulty`,
  `HugePageAligned`, `NumaLocal`, `SplitMetadata`) forward `commit`, so a
  lazy mapping stays correct when wrapped. `Slab` / `SizeClassed` /
  direct `Allocator::allocate` commit the block at allocation time
  (safe, but eager). `SharedBumpArena` and `GuardPage` remain unsupported
  over a lazy mapping and are documented as such.

### Added — `forge-alloc-core` 0.2.2
- `FixedRange::commit(offset, len) -> Result<(), AllocError>`: a
  default-no-op hook that lets a cursor-advancing consumer commit a
  backing's pages just in time. Backwards-compatible — existing
  `FixedRange` implementations inherit the no-op.

## [0.3.1] - 2026-05-25 (includes forge-alloc-core 0.2.1)

Additive release: three new backing primitives, a calloc variant on
`HeapBytes`, and a public `forge_alloc_core::testing` module of
conformance helpers for downstream allocator authors. No breaking
changes. The 0.3.0 trait-decoupling release stands.

### Added — `forge-alloc`
- `forge_alloc::StaticBacked<'a>`: borrows an external `&'a mut [u8]`
  as a `FixedRange`. The most no_std-friendly backing in the family:
  no `alloc`, no syscalls. Use with linker-provided buffers,
  `static mut` arrays, or `Box::leak`ed slices. Pair with
  `BumpArena<StaticBacked<'_>>` for typed allocation in bare-metal
  contexts.
- `forge_alloc::HugePageBacked`: OS-mapped anonymous region backed
  by huge / large pages. Linux `MAP_HUGETLB | MAP_HUGE_<size>`,
  macOS x86_64 `VM_FLAGS_SUPERPAGE_SIZE_ANY`, Windows
  `MEM_LARGE_PAGES`. Returns `AllocError` on platforms without a
  userspace huge-page API (aarch64 macOS, iOS, Android, other
  Unix) — compose with `WithFallback<HugePageBacked, MmapBacked>`
  for graceful degradation. Implements `Allocator + FixedRange +
  OsBacked` as a drop-in for `MmapBacked`.
- `HeapBytes::new_zeroed` / `HeapBytes::with_align_zeroed`: calloc
  variants that route through `Global::allocate_zeroed`. With the
  default `System` allocator, large allocations typically get fresh
  zero pages from the kernel without a userspace memset.

### Added — `forge-alloc-core` 0.2.1
- New `forge_alloc_core::testing` module with conformance helpers
  for downstream `Allocator` / `FixedRange` implementers:
  `assert_fixed_range_invariants`,
  `assert_allocator_basic_round_trip`,
  `assert_allocator_respects_alignment`,
  `assert_combined_invariants`. Each is `#[track_caller]` so
  failures point at the calling test, not at the helper. Alignment
  probe covers `1..=512` so `CACHE_LINE`-sized requests are
  exercised on every supported target.

### Changed
- `forge_alloc::backing::mmap`'s `capture_synthetic_einval` and the
  `unix_prot_from_flags` / `win32_prot_from_flags` helpers are now
  `pub(super)` so `huge_page_backed` can reuse them. Single source
  of truth for the security-relevant prot-mapping table — the two
  backings can no longer silently diverge on a future revision.
- `MmapBacked`, `HugePageBacked`, `ExtendableSlab`, `GuardPage`,
  `HugePageAligned`, `NumaLocal`, `SplitMetadata`, and the
  `HardenedSlab` type alias are now gated on
  `cfg(all(feature = "std", any(unix, windows)))` (previously just
  `feature = "std"`). The narrower gate fixes a pre-existing compile
  failure on `wasm32-wasip1` and other std-capable-but-non-unix-non-
  windows targets — the syscall helpers (`mmap`, `VirtualAlloc`,
  `mbind`, etc.) those types depend on don't exist there. Such
  targets still get the rest of the crate (`InlineBacked`,
  `StaticBacked`, `HeapBytes`, `BumpArena`, `Slab`, etc.).

### Tests
- New `crates/forge-alloc/tests/conformance.rs` integration test
  binary that exercises the `forge_alloc_core::testing`
  conformance helpers against every in-crate backing
  (`InlineBacked`, `StaticBacked`, `HeapBytes`,
  `BumpArena<HeapBytes>`, `BumpArena<StaticBacked<'_>>`,
  `MmapBacked`, `HugePageBacked`, `System`). Doubles as the
  regression gate that any new backing added to the family ships
  with a contract-conforming impl.

## [0.3.0] - 2026-05-25 (includes forge-alloc-core 0.2.0)

### Changed (BREAKING - `forge-alloc-core` trait surface)
- `forge_alloc_core::FixedRange` no longer has `Allocator` as a
  supertrait. The two concerns are independent: a type can own a
  contiguous block of bytes without itself being able to carve
  allocations out of that block, and conversely an allocator does not
  need a fixed address range. This decoupling unblocks pure region-
  owner types such as the new `HeapBytes`.

  **Migration:** anywhere your code took `T: FixedRange` and called
  `T::allocate(...)` (the `Allocator` trait method), change the bound
  to `T: Allocator + FixedRange`. Code that only called
  `T::base()` / `T::size()` / `T::contains()` (the `FixedRange`-native
  methods) needs no change. Internal forge-alloc audit turned up one
  such site (`Quarantine`'s `FixedRange` impl); every other in-tree
  bound was already `Allocator + FixedRange` explicitly.

  `forge-alloc-core` bumps to `0.2.0` for this trait relaxation;
  `forge-alloc` bumps to `0.3.0` because the relaxed trait surfaces
  through its re-exports.

### Added
- `forge_alloc::HeapBytes`: a `FixedRange`-only owner of a single
  global-allocator block. Pair with `BumpArena<HeapBytes>` for bump
  semantics, or `Slab<T, BumpArena<HeapBytes>>` for typed slots, when
  you need a contiguous bounded region but the mmap-level isolation
  of `MmapBacked` isn't worth the ~10-50 µs `mmap` / `VirtualAlloc`
  syscall cost. Deliberately FixedRange-only (no internal bump
  cursor) — bump semantics live in `BumpArena<B>`. Not std-gated;
  uses `allocator_api2::alloc::Global`.
- `MmapBacked` rustdoc gains a `# See also` cross-link to `HeapBytes`
  so a reader picking between the two finds the syscall-cost /
  VM-isolation trade-off immediately.

## [forge-alloc-core 0.1.2] - 2026-05-25 (docs-only)

Patch release of `forge-alloc-core` only. `forge-alloc` stays at 0.2.1
and continues to accept `forge-alloc-core ^0.1.1`, so this version is
picked up automatically by `cargo update`.

### Documentation
- Added a "Use in a layout pin" example to the `CACHE_LINE` rustdoc
  showing how to combine it with `core::mem::offset_of!` to build a
  compile-time assertion that two contended fields in a downstream
  struct never share a cache line. Mirrors the pattern forge-alloc
  uses internally on `AllocStats`, `Watermark`, `SharedBumpArena`,
  and `SlabInner`. The example is a doctest so it runs in CI as a
  regression guard against the idiom breaking.

## [0.2.1] - 2026-05-25

### Changed (internal refactor; non-breaking)
- `CachePadded` and `CACHE_LINE` moved from `forge-alloc` to
  `forge-alloc-core` (bumped 0.1.0 → 0.1.1). Both remain re-exported
  at `forge_alloc::CachePadded` and `forge_alloc::CACHE_LINE` via the
  existing `pub use forge_alloc_core::*`, so the public path that
  consumers use is unchanged. The new canonical path is
  `forge_alloc_core::CachePadded`, which lets downstream crates that
  depend only on `forge-alloc-core` use the primitive without pulling
  in the full `forge-alloc` surface.

## [0.2.0] - 2026-05-25

> **Note:** 0.2.0 was tagged in git but never published to crates.io.
> [0.2.1](#021---2026-05-25) is the first release containing these
> changes on the registry; the entry below is preserved for git-history
> readers.

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

[Unreleased]: https://github.com/dmaesj/forge-alloc/compare/v0.3.7...HEAD
[0.3.7]: https://github.com/dmaesj/forge-alloc/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/dmaesj/forge-alloc/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/dmaesj/forge-alloc/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/dmaesj/forge-alloc/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/dmaesj/forge-alloc/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/dmaesj/forge-alloc/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/dmaesj/forge-alloc/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/dmaesj/forge-alloc/compare/forge-alloc-core-v0.1.2...v0.3.0
[forge-alloc-core 0.1.2]: https://github.com/dmaesj/forge-alloc/compare/v0.2.1...forge-alloc-core-v0.1.2
[0.2.1]: https://github.com/dmaesj/forge-alloc/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/dmaesj/forge-alloc/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dmaesj/forge-alloc/releases/tag/v0.1.0
