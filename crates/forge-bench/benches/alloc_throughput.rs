//! Baseline throughput benchmarks for the core allocator primitives.
//!
//! Baselines are tracked by CodSpeed on its hosted service (not committed
//! files). PRs that regress any benchmark by >5% require explicit override.
//! See `.github/workflows/codspeed.yml` for the gate configuration.
//!
//! Workloads (see `docs/PERFORMANCE_TRADEOFFS.md` for context):
//! - Trading bar allocation: typed slab, high frequency, single-threaded
//! - Query key encoding: bump arena, reset per query
//! - Parser AST: bump arena, bulk-free per parse

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use forge_alloc::{Allocator, Deallocator, NonZeroLayout};
use forge_alloc::{BumpArena, Slab, WithFallback};
use forge_alloc::{InlineBacked, MmapBacked, System};

// Per-batch allocation count for the inline-backed bump benches. Kept
// modest so the `InlineBacked<N>` const-generic array stays small: a
// very large inline array routed through criterion's generic
// `iter_batched` machinery crashes the rustc optimizer. 256 round-trips
// is ample for a steady-state per-op measurement.
const BURST: usize = 256;
const INLINE_BYTES: usize = BURST * 64 + 4096;

#[repr(C)]
struct Bar {
    _open: f64,
    _high: f64,
    _low: f64,
    _close: f64,
    _volume: u64,
    _ts: u64,
}

fn bench_bump_arena_inline(c: &mut Criterion) {
    // Measure a *batch* of `BURST` allocations from a fresh arena per
    // iteration. Constructing the arena once outside `iter` lets it
    // exhaust and the bench would then measure only the `AllocError`
    // return path (hot-path numbers must reflect the success path).
    // `iter_batched` with `LargeInput` resets the arena per batch and
    // amortizes the construction cost.
    let mut g = c.benchmark_group("bump_arena_inline");
    g.throughput(Throughput::Elements(BURST as u64));
    let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
    g.bench_function("alloc_64", |b| {
        b.iter_batched(
            || BumpArena::new(InlineBacked::<INLINE_BYTES>::new()).unwrap(),
            |arena| {
                for _ in 0..BURST {
                    let r = arena.allocate(black_box(layout)).unwrap();
                    black_box(r);
                }
            },
            BatchSize::LargeInput,
        );
    });
    g.finish();
}

fn bench_bump_arena_reset_cycle(c: &mut Criterion) {
    let mut g = c.benchmark_group("bump_arena_reset_cycle");
    g.throughput(Throughput::Elements(64));
    let mut arena = BumpArena::new(InlineBacked::<8192>::new()).unwrap();
    let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
    g.bench_function("alloc64_then_reset", |b| {
        b.iter(|| {
            for _ in 0..64 {
                let r = arena.allocate(black_box(layout)).unwrap();
                black_box(r);
            }
            arena.reset();
        });
    });
    g.finish();
}

fn bench_slab_typed(c: &mut Criterion) {
    // Slab LIFO freelist: every iteration after the first allocates the same
    // slot the previous dealloc just pushed. Steady-state cost is one push +
    // one pop on the freelist.
    let mut g = c.benchmark_group("slab_typed");
    g.throughput(Throughput::Elements(1));
    let s: Slab<Bar, MmapBacked> = Slab::new(8192, MmapBacked::new(1 << 20).unwrap()).unwrap();
    let layout = NonZeroLayout::for_type::<Bar>().unwrap();
    g.bench_function("alloc_then_free", |b| {
        b.iter(|| {
            let p = s.allocate(black_box(layout)).unwrap();
            unsafe { s.deallocate(p.cast(), layout) };
            black_box(p);
        });
    });
    g.finish();
}

fn bench_slab_typed_no_free(c: &mut Criterion) {
    // Constructing the slab once and never freeing exhausts it; subsequent
    // iterations measure only the AllocError path and make the reported
    // throughput meaningless. Recreate the slab per batch and measure a
    // `BURST`-alloc burst from empty.
    let mut g = c.benchmark_group("slab_typed_no_free");
    g.throughput(Throughput::Elements(BURST as u64));
    let layout = NonZeroLayout::for_type::<Bar>().unwrap();
    g.bench_function("alloc_only", |b| {
        b.iter_batched(
            || Slab::<Bar, MmapBacked>::new(1 << 14, MmapBacked::new(1 << 22).unwrap()).unwrap(),
            |s| {
                for _ in 0..BURST {
                    let p = s.allocate(black_box(layout)).unwrap();
                    black_box(p);
                }
            },
            BatchSize::LargeInput,
        );
    });
    g.finish();
}

fn bench_with_fallback(c: &mut Criterion) {
    // Primary is an inline bump arena sized for `BURST` 64-byte allocs, so
    // every iteration in the batch hits the bump fast-path before the arena
    // would exhaust and route to System. Construct a fresh WithFallback per
    // batch so "primary_path" reflects the bump fast-path only.
    let mut g = c.benchmark_group("with_fallback");
    g.throughput(Throughput::Elements(BURST as u64));
    let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
    g.bench_function("primary_path", |b| {
        b.iter_batched(
            || WithFallback::new(InlineBacked::<INLINE_BYTES>::new(), System),
            |wf| {
                for _ in 0..BURST {
                    let r = wf.allocate(black_box(layout)).unwrap();
                    black_box(r);
                }
            },
            BatchSize::LargeInput,
        );
    });
    g.finish();
}

fn bench_system_baseline(c: &mut Criterion) {
    // Baseline: vanilla System allocator for size-64 allocations.
    let mut g = c.benchmark_group("system_baseline");
    g.throughput(Throughput::Elements(1));
    let s = System;
    let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
    g.bench_function("alloc_then_free", |b| {
        b.iter(|| {
            let p = s.allocate(black_box(layout)).unwrap();
            unsafe { s.deallocate(p.cast(), layout) };
            black_box(p);
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_bump_arena_inline,
    bench_bump_arena_reset_cycle,
    bench_slab_typed,
    bench_slab_typed_no_free,
    bench_with_fallback,
    bench_system_baseline,
);
criterion_main!(benches);
