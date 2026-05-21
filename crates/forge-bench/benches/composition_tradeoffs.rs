//! Composition-tradeoffs benchmark.
//!
//! Measures the per-op cost added by each hardening / observability layer
//! on top of a baseline `Slab<u64, MmapBacked>`. The goal is the
//! "pay-for-what-you-use" curve documented in `PERFORMANCE_TRADEOFFS.md` —
//! readers can see at a glance what every wrapper costs them.
//!
//! Methodology:
//! - Every composition exercises the same workload: 256 allocate +
//!   deallocate round-trips per iteration (LIFO; the slab freelist
//!   surfaces the just-freed slot on the next allocate, so steady-state
//!   per-op cost is what we measure).
//! - Layout is `NonZeroLayout::for_type::<u64>()` (8 bytes, 8-align) for
//!   every composition that takes a typed slab; for `BumpArena` and
//!   `SizeClassed` we use the equivalent `from_size_align(8, 8)`.
//! - The allocator is constructed once per iteration BATCH (LargeInput).
//!   This means each batch's first allocate pays a cold-cache cost
//!   that's amortised across the 256 ops; subsequent batches are warm.
//!   Mixing cold + warm in the same group is intentional: the
//!   "pay-for-what-you-use" question is about steady-state per-op
//!   cost on a hot allocator, which is what `iter_batched`'s
//!   amortisation surfaces.
//! - All compositions sit on `MmapBacked` to remove the
//!   InlineBacked-vs-MmapBacked cache-profile confounder.
//!
//! To regenerate the curve documented in `PERFORMANCE_TRADEOFFS.md`:
//!
//! ```text
//! cargo bench -p forge-bench --bench composition_tradeoffs
//! cargo bench -p forge-bench --bench composition_tradeoffs --features siphasher
//! ```
//!
//! The siphasher run adds tier 3b (`Slab<_, _, SipHashMAC>`) and tier 4
//! (`HardenedSlab<_, SipHashMAC>`) — the freelist-authenticated and
//! fully-hardened variants. Without the feature those rows are absent
//! from the report.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkGroup, Criterion, Throughput};
use criterion::measurement::WallTime;

use forge_backing::{MmapBacked, System};
use forge_core::{Allocator, NonZeroLayout};
use forge_hardening::{
    Canary, CacheJitter, NullHandler, PoisonOnFree, Quarantine, Statistics, Watermark,
};

#[cfg(feature = "siphasher")]
use forge_hardening::{GuardPage, SplitMetadata};
use forge_layout::{BumpArena, SizeClassed, Slab};

#[cfg(feature = "siphasher")]
use forge_core::SipHashMAC;

// 256 round-trips per iteration. Chosen so:
// - the slab's LIFO freelist reaches steady state by iteration ~4
//   (one slot bouncing between freelist-empty and freelist-head)
// - the measurement absorbs alloc + dealloc cost in equal proportions
// - batch construction cost is small relative to per-iter work
const OPS_PER_ITER: usize = 256;

// Slab capacity bigger than OPS_PER_ITER so we can run 256 sequential
// allocs followed by deallocs without exhausting (some compositions —
// Quarantine, e.g. — delay slot reuse for EPOCHS frees). 1024 slots
// over 8 bytes = 8 KiB live region, well under the 1 MiB backing.
const SLAB_CAPACITY: usize = 1024;
const MMAP_BYTES: usize = 1 << 20; // 1 MiB

/// Per-iteration workload. Allocate + immediately deallocate, OPS_PER_ITER
/// times. The LIFO freelist returns the same slot every time after the
/// first, so this measures steady-state alloc/dealloc cost.
#[inline(always)]
fn round_trip<A: Allocator>(allocator: &A, layout: NonZeroLayout) {
    for _ in 0..OPS_PER_ITER {
        let p = allocator.allocate(black_box(layout)).unwrap();
        // SAFETY: the pointer was just returned from allocate, never
        // dereferenced (only the address is consumed), so the layout
        // matches and the pointer is live.
        unsafe { allocator.deallocate(p.cast(), layout) };
        black_box(p);
    }
}

/// Variant for arena-style allocators (no deallocate; `reset` clears).
#[inline(always)]
fn arena_burst<A: Allocator>(allocator: &mut A, layout: NonZeroLayout) {
    for _ in 0..OPS_PER_ITER {
        let p = allocator.allocate(black_box(layout)).unwrap();
        black_box(p);
    }
    let _ = allocator.reset();
}

fn bench_tradeoffs(c: &mut Criterion) {
    let mut g = c.benchmark_group("composition_tradeoffs");
    g.throughput(Throughput::Elements(OPS_PER_ITER as u64));
    let layout = NonZeroLayout::for_type::<u64>().unwrap();

    // ============================================================================
    // Tier 0 — bare allocators, no observability, no hardening.
    // ============================================================================

    // 0a: Slab<u64, MmapBacked> — typed pool, freelist LIFO
    bench_round_trip(&mut g, "tier0a_slab_bare", layout, || {
        Slab::<u64, MmapBacked>::new(SLAB_CAPACITY, MmapBacked::new(MMAP_BYTES).unwrap()).unwrap()
    });

    // 0b: BumpArena<MmapBacked> — bump cursor, reset per batch
    bench_arena(&mut g, "tier0b_bump_arena", layout, || {
        BumpArena::new(MmapBacked::new(MMAP_BYTES).unwrap()).unwrap()
    });

    // 0c: SizeClassed<MmapBacked, 8> — general allocator, 8 class buckets
    bench_round_trip(&mut g, "tier0c_size_classed", layout, || {
        SizeClassed::<_, 8>::with_default_classes(MmapBacked::new(MMAP_BYTES).unwrap(), 64).unwrap()
    });

    // 0d: System — libc malloc/free, baseline for reference
    bench_round_trip(&mut g, "tier0d_system_baseline", layout, || System);

    // ============================================================================
    // Tier 1 — + observability (atomic counters / threshold checks)
    // ============================================================================

    // 1a: Statistics<Slab> — adds ~4 relaxed atomic counter updates per allocate, ~3 per dealloc
    bench_round_trip(&mut g, "tier1a_statistics_slab", layout, || {
        Statistics::new(
            Slab::<u64, MmapBacked>::new(SLAB_CAPACITY, MmapBacked::new(MMAP_BYTES).unwrap())
                .unwrap(),
        )
    });

    // 1b: Watermark<Statistics<Slab>, NullHandler> — + threshold-fire check
    bench_round_trip(&mut g, "tier1b_watermark_stats_slab", layout, || {
        Watermark::new(
            Statistics::new(
                Slab::<u64, MmapBacked>::new(SLAB_CAPACITY, MmapBacked::new(MMAP_BYTES).unwrap())
                    .unwrap(),
            ),
            NullHandler,
        )
    });

    // ============================================================================
    // Tier 2 — + UAF-soft hardening (visibility + delayed reuse)
    // ============================================================================

    // 2a: PoisonOnFree<Slab> — writes a sentinel pattern over freed bytes
    bench_round_trip(&mut g, "tier2a_poison_on_free_slab", layout, || {
        PoisonOnFree::new(
            Slab::<u64, MmapBacked>::new(SLAB_CAPACITY, MmapBacked::new(MMAP_BYTES).unwrap())
                .unwrap(),
        )
    });

    // 2b: Quarantine<Slab, EPOCHS=4> — delays slot return to inner by 4 frees
    bench_round_trip(&mut g, "tier2b_quarantine_slab", layout, || {
        Quarantine::<_, 4>::new(
            Slab::<u64, MmapBacked>::new(SLAB_CAPACITY, MmapBacked::new(MMAP_BYTES).unwrap())
                .unwrap(),
        )
    });

    // ============================================================================
    // Tier 3 — + corruption-detection hardening (panic on detection)
    // ============================================================================

    // 3a: Canary<BumpArena> — pre/post 8-byte sentinels per allocation.
    // Wraps BumpArena rather than Slab because Canary's per-alloc layout
    // inflation (size + 2 * align worth of pad bytes) does not fit
    // Slab's fixed block_stride for an 8-byte payload.
    bench_arena(&mut g, "tier3a_canary_bump", layout, || {
        Canary::new(BumpArena::new(MmapBacked::new(MMAP_BYTES).unwrap()).unwrap())
    });

    // 3b: Slab<u64, MmapBacked, SipHashMAC> — keyed freelist authentication
    #[cfg(feature = "siphasher")]
    bench_round_trip(&mut g, "tier3b_slab_siphash_mac", layout, || {
        Slab::<u64, MmapBacked, SipHashMAC>::with_protection(
            SLAB_CAPACITY,
            MmapBacked::new(MMAP_BYTES).unwrap(),
            SipHashMAC::new(),
        )
        .unwrap()
    });

    // 3c: CacheJitter<BumpArena> — cache-line random prefix + 48-bit keyed
    // MAC header. Same reason as 3a: layout inflation (cache_line_size +
    // displacement padding) does not fit Slab's fixed stride for small
    // payloads. The documented recipes pair CacheJitter with BumpArena.
    bench_arena(&mut g, "tier3c_cache_jitter_bump", layout, || {
        CacheJitter::new(
            BumpArena::new(MmapBacked::new(MMAP_BYTES).unwrap()).unwrap(),
            64, // cache_line_size
            8,  // associativity
        )
        .expect("CacheJitter::new (cache=64, assoc=8) cannot fail with valid params")
    });

    // ============================================================================
    // Tier 4 — fully hardened (guard pages + split metadata + freelist MAC)
    // ============================================================================

    #[cfg(feature = "siphasher")]
    bench_round_trip(&mut g, "tier4_hardened_slab_siphash", layout, || {
        Slab::<u64, GuardPage<SplitMetadata<MmapBacked>>, SipHashMAC>::with_protection(
            SLAB_CAPACITY,
            GuardPage::new(
                SplitMetadata::new(MmapBacked::new(MMAP_BYTES).unwrap(), 64 * 1024).unwrap(),
                4096,
            )
            .unwrap(),
            SipHashMAC::new(),
        )
        .unwrap()
    });

    g.finish();
}

/// Helper: bench an allocator constructed by `make_alloc` over the
/// `round_trip` workload. Uses `iter_batched` so a fresh allocator is
/// constructed per batch (preventing cumulative exhaustion across
/// iterations), but the per-iter work is the round-trip itself
/// (LargeInput amortises construction).
fn bench_round_trip<A: Allocator, F: FnMut() -> A>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    name: &'static str,
    layout: NonZeroLayout,
    mut make_alloc: F,
) {
    g.bench_function(name, |b| {
        b.iter_batched(
            &mut make_alloc,
            |alloc| round_trip(&alloc, layout),
            BatchSize::LargeInput,
        );
    });
}

/// Helper: bench an arena allocator (BumpArena-style) with reset between
/// iterations. The `&mut` access is required for `reset()`.
fn bench_arena<A: Allocator, F: FnMut() -> A>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    name: &'static str,
    layout: NonZeroLayout,
    mut make_alloc: F,
) {
    g.bench_function(name, |b| {
        b.iter_batched_ref(
            &mut make_alloc,
            |alloc| arena_burst(alloc, layout),
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, bench_tradeoffs);
criterion_main!(benches);
