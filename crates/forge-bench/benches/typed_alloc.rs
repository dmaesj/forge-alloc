//! Typed fast-path (`alloc_uninit::<T>`) vs the runtime-`Layout` `allocate`.
//!
//! `alloc_uninit::<T>()` feeds `size_of::<T>()` / `align_of::<T>()` as
//! compile-time constants, so the alignment-rounding mask and bounds arithmetic
//! fold to a tight branch; `allocate(layout)` carries the size/align in a
//! runtime `NonZeroLayout`. This measures the gap honestly under fat LTO (see
//! `[profile.bench]` in the workspace `Cargo.toml`), allocating a *batch* from a
//! fresh arena per iteration so we time the success path, not exhaustion.
//!
//! Three variants per type:
//! - `layout_blackbox`  — `allocate(black_box(layout))`: a genuinely runtime
//!   layout (the optimizer cannot fold size/align away). The pessimistic case.
//! - `layout_const`     — `allocate(layout)`: layout is a known local; shows how
//!   much LLVM already folds the existing path without help.
//! - `typed_uninit`     — `alloc_uninit::<T>()`: the new const path.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use forge_alloc::{Allocator, BumpArena, MmapBacked, NonZeroLayout};

const BURST: usize = 256;
// Headroom for alignment padding across the batch; biggest element is 48 bytes.
const REGION: usize = BURST * 64 + 4096;

#[repr(C)]
struct Bar {
    _open: f64,
    _high: f64,
    _low: f64,
    _close: f64,
    _volume: u64,
    _ts: u64,
}

// Persistent arena over a small (pointer-sized) `MmapBacked` handle,
// reset per iteration. This isolates the allocation fast path: no per-iter
// arena construction, no giant inline array to move/cache-thrash. The `reset()`
// is an O(1) cursor store, identical overhead across all three variants.
//
// Note: none of the variants write *through* the returned pointer — this
// measures the allocation bookkeeping only, not store throughput (which would
// add identical memcpy cost to all three and dilute the signal). All three
// `black_box` their result, so no allocation is dead-code-eliminated.
fn bench_type<T: 'static>(c: &mut Criterion, name: &str) {
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(BURST as u64));
    let layout = NonZeroLayout::for_type::<T>().expect("non-ZST");
    let mut arena = BumpArena::new(MmapBacked::new(REGION).unwrap()).unwrap();

    g.bench_function("layout_blackbox", |b| {
        b.iter(|| {
            for _ in 0..BURST {
                black_box(arena.allocate(black_box(layout)).unwrap());
            }
            arena.reset();
        });
    });

    g.bench_function("layout_const", |b| {
        b.iter(|| {
            for _ in 0..BURST {
                black_box(arena.allocate(layout).unwrap());
            }
            arena.reset();
        });
    });

    g.bench_function("typed_uninit", |b| {
        b.iter(|| {
            for _ in 0..BURST {
                black_box(arena.alloc_uninit::<T>().unwrap());
            }
            arena.reset();
        });
    });

    g.finish();
}

fn bench_u64(c: &mut Criterion) {
    bench_type::<u64>(c, "typed_alloc_u64");
}

fn bench_bar(c: &mut Criterion) {
    bench_type::<Bar>(c, "typed_alloc_bar");
}

criterion_group!(benches, bench_u64, bench_bar);
criterion_main!(benches);
