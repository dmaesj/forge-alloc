# Performance Tradeoffs

A reader's guide to what each `forge-alloc` wrapper costs you, what
security or observability property it gives in return, and how to
interpret the Criterion numbers from
[`composition_tradeoffs`](../crates/forge-bench/benches/composition_tradeoffs.rs).

The thesis: **you pay only for the layers you compose in.** An
unwrapped `BumpArena<MmapBacked>` runs at the same per-op cost as a
hand-rolled bump allocator. Wrapping it with `Statistics<...>` adds
exactly the cost of five atomic counters per op — nothing more.
`Canary<...>` adds two volatile 8-byte sentinel writes. The wrappers
compose at the type level with zero runtime dispatch, so the
post-wrapping cost is *exactly* the sum of layer costs, not a vtable
indirection on top.

---

## The tiers

Each tier adds a category of property. The compositions in each tier
are roughly comparable in cost to each other; the cost *between*
adjacent tiers is the marginal cost of adding that category.

| Tier | What it buys you | What it costs (per allocate + deallocate round-trip) |
|---|---|---|
| **0 — bare** | typed pool / arena / size-classed routing | ~5–15 ns; the actual allocator's work |
| **1 — + observability** | atomic counters, threshold checks, operator dashboards | + ~3–8 ns; atomic fetches |
| **2 — + UAF soft** | freed-byte poisoning, delayed slot reuse, post-mortem visibility | + ~10–20 ns; sentinel writes + ring management |
| **3 — + corruption detection** | guard sentinels, freelist MAC, cache-line randomisation; panic on detected corruption | + ~15–25 ns; verify + MAC compute |
| **4 — fully hardened** | guard pages + split metadata + freelist MAC stacked together | + ~30–60 ns; multiple page-fault-trapping protections |

Numbers are order-of-magnitude estimates from informed reasoning over
the code paths. **For actual numbers on your hardware, run the bench**
(see [§Running the bench](#running-the-bench)). See
[§Sample numbers](#sample-numbers-windows-11-x86_64) for one author
machine's measurements, with the usual caveats about benchmark
methodology and hardware variance.

---

## Why "pay for what you use" actually holds

Three design choices make the curve real, not aspirational:

1. **Zero-cost when unwrapped**. The wrappers are generic structs with
   `#[inline]` methods. If you don't wrap with `Statistics`, you don't
   compile a `Statistics::allocate` body into your binary. There's no
   `dyn Allocator` indirection.

2. **Hot-path branches are cold-side-skipped**. Corruption-detection
   sites bump their counters and panic / disarm in *else* branches
   that the compiler will static-predict as not-taken. The happy path
   on `Slab<u64, _, SipHashMAC>::allocate` is one MAC compute + one
   in-range check; the `else { bump_counter; abandon_freelist; }`
   block adds no branch-prediction penalty on the success path.

3. **Statistics' polling is fast-pathed**. `Statistics::allocate`
   polls `inner.corruption_events()` on every call and mirrors it into
   `AllocStats.corruption_events`. The `mirror_corruption` helper does
   `if inner_val > mirror.load(Relaxed) { fetch_max(...) }` — in the
   steady-state no-corruption case the inner is 0, the load is one
   `mov`, and the `fetch_max`'s `lock cmpxchg` is skipped entirely.
   ~5 ns saved per op vs an unconditional `fetch_max`.

---

## The compositions (what the bench measures)

All compositions use `MmapBacked` (1 MiB) as the backing, so the
InlineBacked-vs-MmapBacked cache-profile confounder is excluded. Layout
is `u64` (8 bytes, 8-align). The workload is 256 alloc+dealloc
round-trips per iteration. The slab freelist surfaces the just-freed
slot on each subsequent allocate, so steady-state per-op cost is what
gets measured.

### Tier 0 — bare

| name | composition | what it does |
|---|---|---|
| `tier0a_slab_bare` | `Slab<u64, MmapBacked>` | typed pool, LIFO freelist, one push + one pop per round-trip |
| `tier0b_bump_arena` | `BumpArena<MmapBacked>` | bump cursor; `reset()` per iteration since BumpArena's `deallocate` is a no-op |
| `tier0c_size_classed` | `SizeClassed<MmapBacked, 8>` | 8 class slabs, routes by size to the smallest fitting; falls back to inner backing for oversized |
| `tier0d_system_baseline` | `System` (libc malloc/free) | baseline for reference; what *not* having a forge-alloc allocator costs |

### Tier 1 — + observability

| name | composition | what it adds |
|---|---|---|
| `tier1a_statistics_slab` | `Statistics<Slab>` | total alloc/dealloc/failure counters, bytes_allocated, bytes_peak, mirrored corruption_events |
| `tier1b_watermark_stats_slab` | `Watermark<Statistics<Slab>, NullHandler>` | + threshold checks (warn / critical) on capacity utilisation, fire handler when crossed |

### Tier 2 — + UAF soft

| name | composition | what it adds |
|---|---|---|
| `tier2a_poison_on_free_slab` | `PoisonOnFree<Slab>` | overwrites freed bytes with a sentinel pattern (`0xAA` repeated); UAF reads return obvious-garbage |
| `tier2b_quarantine_slab` | `Quarantine<Slab, 4>` | delays slot return to the inner by 4 frees; an attacker who races dealloc + alloc hoping to reclaim the same slot is foiled |

### Tier 3 — + corruption detection

| name | composition | what it adds |
|---|---|---|
| `tier3a_canary_slab` | `Canary<Slab>` | pre/post 8-byte sentinels at user-pointer boundaries; panic on detected modification |
| `tier3b_slab_siphash_mac` *(siphasher)* | `Slab<u64, MmapBacked, SipHashMAC>` | keyed MAC on freelist `next_idx`; corrupted link is detected at allocation time |
| `tier3c_cache_jitter_slab` | `CacheJitter<Slab>` | randomises cache-line offset per allocation + 48-bit keyed MAC header; cache-side-channel mitigation |

### Tier 4 — fully hardened

| name | composition | what it adds |
|---|---|---|
| `tier4_hardened_slab_siphash` *(siphasher)* | `Slab<u64, GuardPage<SplitMetadata<MmapBacked>>, SipHashMAC>` | guard pages around the data region + metadata at a separate virtual address + freelist MAC. Pinned by the `HardenedSlab<u64, SipHashMAC>` type alias in `forge-alloc`. |

---

## Running the bench

```bash
# Default: tiers 0–3 except `tier3b_slab_siphash_mac` and tier 4
cargo bench -p forge-bench --bench composition_tradeoffs

# Full: enables SipHashMAC (tier 3b) + HardenedSlab with SipHash (tier 4)
cargo bench -p forge-bench --bench composition_tradeoffs --features siphasher
```

Criterion writes HTML reports to `target/criterion/composition_tradeoffs/`.
The `report.html` index renders all bench results side-by-side as bars,
ordered by tier prefix so the curve is immediately legible.

To compare against a saved baseline (e.g. before a refactor):

```bash
cargo bench -p forge-bench --bench composition_tradeoffs --features siphasher -- --save-baseline before
# ... make changes ...
cargo bench -p forge-bench --bench composition_tradeoffs --features siphasher -- --baseline before
```

### Continuous regression detection (CI)

The repo wires up [CodSpeed](https://codspeed.io) for automated
regression detection on every pull request — see
[`.github/workflows/codspeed.yml`](../.github/workflows/codspeed.yml).
CodSpeed runs the criterion benches under a deterministic
instruction-counting harness instead of a wall-clock timer, so the
result is reproducible regardless of which CI runner executed it (no
shared-infrastructure noise). The `forge-bench` crate depends on the
`codspeed-criterion-compat` drop-in; a plain local `cargo bench` is
unaffected and still produces the wall-clock HTML reports above.

---

## What the numbers DO mean

- **Steady-state per-op cost** on a warm allocator (freelist non-empty,
  cache lines hot). This is the cost you pay for the 1000th allocate
  in a request handler that allocates 1024 scratch objects per request.

- **Apples-to-apples across tiers**. All compositions use the same
  backing, same layout, same workload. The delta between
  `tier0a_slab_bare` and `tier3a_canary_slab` is *exactly* the cost
  Canary adds to a Slab — no other factor changed.

- **What an unwrapped composition costs** if you remove every layer.
  `tier0a` is the floor — any wrapping adds on top.

## What the numbers DON'T mean

- **They aren't head-to-head vs jemalloc / mimalloc**. The
  `tier0d_system_baseline` row is a reference point for "what does
  vanilla System malloc cost on this hardware," not a comparative
  claim. `forge-alloc` does not compete with jemalloc on raw
  throughput; it competes on observable composability.

- **Cold-cache cost isn't here**. The first allocate in a fresh
  process pays a cold-cache miss; this bench amortises that across
  256 ops per iteration. If your workload allocates rarely (once per
  minute) and the cache between calls turns over, your real per-op
  cost is higher.

- **Tail latency isn't here**. Criterion reports the mean / median
  with confidence intervals. P99 + P99.9 for the corruption-detection
  tiers (3, 4) depends on whether your workload triggers the cold
  branch, which the bench does not exercise.

- **Multi-threaded contention isn't here**. The bench is
  single-threaded. `SharedBumpArena`'s CAS loop and `SlabOwner` +
  `SlabRemote`'s remote-dealloc queue have separate concurrency-
  specific characteristics that would need a contention-spectrum bench
  to surface. ([`crates/forge-layout/tests/loom_concurrency.rs`](../crates/forge-layout/tests/loom_concurrency.rs)
  pins their *correctness* under permutation; perf under contention is
  a future bench.)

---

## How to pick a tier for your workload

The right tier depends on what you're protecting and what you can spend.

- **Compute-bound, no security requirement** (parser scratch, internal
  buffer pools, batch processing): tier 0. Wrap with `Statistics` only
  if you genuinely want to observe; the dashboard query cost is
  separate from the alloc cost.

- **Production service, want operator visibility**: tier 1.
  `Statistics<Watermark<Inner>>` is the common shape. Cost is small
  (~10 ns per op), value is high (dashboard alerts).

- **Auth / session / financial data** where UAF would be a security
  bug: tier 2. `PoisonOnFree<Quarantine<Slab>>` makes silent UAF very
  hard.

- **Anti-forge / anti-tamper on the allocator itself**: tier 3 or 4.
  `Canary<Slab>` catches naive overflow corruption; `SipHashMAC` on
  the slab catches forged freelist links; `HardenedSlab` stacks
  multiple layers.

- **PHI / cryptographic key material**: tier 4. Pay the ~50 ns/op for
  the maximum-defence stack. Bonus: pair with `MmapBacked` + Linux
  `mlock` (separate concern) to keep keys out of swap.

If the workload alloc rate is low enough that ~50 ns/op of overhead is
negligible (most application workloads — anything below 10 M allocs/sec),
just pick the tier that matches your threat model. The perf savings of
"choose tier 1 instead of tier 3" only matter at >100 M allocs/sec.

---

## Sample numbers (Windows 11, x86_64)

Author machine: x86_64 Windows 11, debug-info release build, single
threaded, sccache + Windows Defender running (noisy environment —
treat as order-of-magnitude, not authoritative). 100 samples, 1 s
warm-up + 3 s measurement per row.

| Composition | Time (256 ops) | Per-op cost | Marginal cost vs tier 0a |
|---|---|---|---|
| `tier0b_bump_arena` | ~568 ns | **~2.2 ns** | -50.8 ns (Bump is much cheaper than Slab — no freelist) |
| `tier0a_slab_bare` | ~13.5 µs | **~53 ns** | baseline |
| `tier0d_system_baseline` | ~13.7 µs | ~54 ns | +1 ns (System is competitive here at small sizes) |
| `tier0c_size_classed` | ~16.6 µs | ~65 ns | +12 ns (class routing) |
| `tier1a_statistics_slab` | ~17.4 µs | ~68 ns | +15 ns (5 atomic counters per op) |
| `tier1b_watermark_stats_slab` | ~21.2 µs | ~83 ns | +30 ns (+ threshold check) |
| `tier2a_poison_on_free_slab` | ~12.2 µs | ~48 ns | -5 ns (within noise — poison write may overlap with cold dealloc) |
| `tier2b_quarantine_slab` | ~13.3 µs | ~52 ns | -1 ns (also within noise) |
| `tier3a_canary_bump` | ~6.5 µs | **~26 ns** | (compared against tier0b Bump: +24 ns for canary verify) |
| `tier3b_slab_siphash_mac` | ~37 µs | **~145 ns** | +92 ns (SipHash compute per op is the dominant cost) |
| `tier3c_cache_jitter_bump` | ~54.5 µs | **~213 ns** | (compared against tier0b Bump: +211 ns; cache-line random + 48-bit MAC header) |
| `tier4_hardened_slab_siphash` | ~43 µs | **~168 ns** | +115 ns over tier 0a (GuardPage + SplitMetadata + SipHashMAC stacked) |

Observations from these numbers:

- **BumpArena is genuinely fast** (~2 ns/op). Use it when you have a
  per-request or per-frame scratch lifetime — the cost difference
  between a Bump and a Slab is dramatic.
- **Statistics adds about 15 ns/op**. For a service doing 1 M
  allocs/sec, that's 1.5% CPU on observability. Usually worth it.
- **PoisonOnFree and Quarantine cost less than expected**. The
  measurements landed within Slab's noise floor (~50 ns ± 10 ns). The
  poison write is a single non-temporal store that the CPU can
  overlap with the slab's freelist push; Quarantine's ring index
  bump is also cheap. The compose-on-Slab cost is dominated by the
  Slab itself.
- **SipHashMAC is the most expensive single hardening primitive**.
  Computing a keyed hash per freelist link costs ~90 ns over a bare
  Slab. The security property — corrupted next_idx is detected at
  allocation time, not at use time — is the high-value reason to
  pay it.
- **CacheJitter is more expensive than HardenedSlab**. Because
  HardenedSlab's protections (guard pages + split metadata) are
  *one-time setup* costs at construction; per-op only the SipHashMAC
  verify fires. CacheJitter, by contrast, does cache-line random
  selection + MAC header packing on *every* op.

Run on your own hardware before making sizing decisions — these
numbers are indicative, not authoritative. Windows is a particularly
noisy benchmarking environment; a Linux box with isolated cores and
performance-governor pinned will give much tighter intervals.

## Cross-references

- [`COMPOSITION_RECIPES.md`](COMPOSITION_RECIPES.md) — concrete wirings
  for stack-local scratch, cross-thread bump arena, hardened slab,
  etc.
- [`COMPATIBILITY_MATRIX.md`](COMPATIBILITY_MATRIX.md) — which
  combinations don't work or shouldn't be tried.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the three-layer mental model.
- Regression detection runs automatically in CI via
  [CodSpeed](https://codspeed.io) — the `composition_tradeoffs`,
  `alloc_throughput`, and `adaptive_batch` benches are all gated.
  CodSpeed maintains the baseline against `main` on its own service;
  there are no baseline files committed to the repo (committed
  wall-clock numbers would only be valid for the exact machine that
  produced them).
