//! Baseline throughput benchmarks for the four M1-M3 primitives. Per spec
//! §14, a Criterion baseline is committed; PRs that regress any benchmark by
//! >5% require explicit override.
//!
//! Workloads mirror spec §14 "Benchmark workloads representing real use cases":
//! - Trading bar allocation: typed slab, high frequency, single-threaded
//! - Query key encoding: bump arena, reset per query
//! - Parser AST: bump arena, bulk-free per parse

use forge_backing::{InlineBacked, MmapBacked, System};
use forge_core::{Allocator, Deallocator, NonZeroLayout};
use forge_layout::{BumpArena, Slab, WithFallback};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

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
    // Measure a *batch* of 1024 allocations from a fresh arena per iteration.
    // Constructing the arena once outside `iter` lets it exhaust after a few
    // thousand iterations and the bench would then measure only the
    // `AllocError` return path (spec §14 hot-path numbers must reflect the
    // success path). `iter_batched` with `LargeInput` resets the arena per
    // batch and amortizes the construction cost.
    let mut g = c.benchmark_group("bump_arena_inline");
    g.throughput(Throughput::Elements(1024));
    let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
    g.bench_function("alloc_64", |b| {
        b.iter_batched(
            || BumpArena::new(InlineBacked::<{ 1024 * 64 + 4096 }>::new()).unwrap(),
            |arena| {
                for _ in 0..1024 {
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
    let mut arena = BumpArena::new(InlineBacked::<65536>::new()).unwrap();
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
    // Constructing the slab once and never freeing exhausts it in 1<<14
    // iterations; subsequent iterations measure only the AllocError path and
    // make the reported throughput meaningless. Recreate the slab per
    // batch and measure a 1024-alloc burst from empty.
    let mut g = c.benchmark_group("slab_typed_no_free");
    g.throughput(Throughput::Elements(1024));
    let layout = NonZeroLayout::for_type::<Bar>().unwrap();
    g.bench_function("alloc_only", |b| {
        b.iter_batched(
            || {
                Slab::<Bar, MmapBacked>::new(1 << 14, MmapBacked::new(1 << 22).unwrap()).unwrap()
            },
            |s| {
                for _ in 0..1024 {
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
    // Primary is InlineBacked<65536> over 64-byte allocs = 1024 successful
    // iterations before the bump arena exhausts and we start routing to
    // System. We want "primary_path" to reflect the bump fast-path only —
    // construct a fresh WithFallback per batch.
    let mut g = c.benchmark_group("with_fallback");
    g.throughput(Throughput::Elements(1024));
    let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
    g.bench_function("primary_path", |b| {
        b.iter_batched(
            || WithFallback::new(InlineBacked::<{ 1024 * 64 + 4096 }>::new(), System),
            |wf| {
                for _ in 0..1024 {
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
