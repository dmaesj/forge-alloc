//! Multi-thread contention benchmarks.
//!
//! Two purposes:
//!
//! 1. **`false_sharing_micro` group** — apples-to-apples demo that the
//!    cache-line padding actually buys throughput on this hardware.
//!    Modeled on cong-or's "The Slowdown That Doesn't Show Up in
//!    Profiles" experiment: two threads, each writing a distinct
//!    `AtomicUsize`, with the atomics either co-located on one cache
//!    line (`contended`) or split apart by an explicit padding array
//!    (`padded`). The padded variant should be substantially faster
//!    on any platform where the L1 coherency granularity matters,
//!    which is essentially every desktop and server target.
//!
//! 2. **`allocator_contention` group** — absolute throughput of the
//!    real `SharedBumpArena` and `Statistics<SharedBumpArena>` under
//!    concurrent allocates. Both wrappers wrap their contended atomics
//!    in `CachePadded` after the v0.2.0 layout change. Tracked on
//!    CodSpeed as a regression gate so a future refactor that
//!    accidentally unwraps a counter is caught at PR time. There is no
//!    "unpadded" comparison variant in this group on purpose: the
//!    structural change is permanent, and reintroducing the unpadded
//!    shape just to bench against it would be more code than the
//!    `false_sharing_micro` group already demonstrates.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use forge_alloc::{Allocator, InlineBacked, NonZeroLayout, SharedBumpArena, Statistics};

/// Per-thread iteration count for the contention micro-bench. Smaller
/// than the article's 50M so a criterion run completes in seconds
/// rather than minutes; large enough that loop-bookkeeping overhead
/// is amortized below the atomic op cost.
const MICRO_OPS_PER_THREAD: usize = 200_000;

/// Same workload definition for both `contended` and `padded` variants.
fn run_pair<T: Send + Sync + 'static>(data: Arc<T>, f0: fn(&T), f1: fn(&T)) {
    let d0 = Arc::clone(&data);
    let d1 = Arc::clone(&data);
    let h0 = thread::spawn(move || f0(&d0));
    let h1 = thread::spawn(move || f1(&d1));
    h0.join().unwrap();
    h1.join().unwrap();
}

/// Two atomics co-located on (almost certainly) the same cache line.
#[repr(C)]
struct Contended {
    x: AtomicUsize,
    y: AtomicUsize,
}

/// Same two atomics separated by an explicit padding array large
/// enough to cross any reasonable cache-line boundary on tier-1
/// targets (128 covers x86_64 adjacent-line prefetch pairs and
/// Apple Silicon's 128-byte coherency).
#[repr(C)]
struct Padded {
    x: AtomicUsize,
    _pad: [u8; 120],
    y: AtomicUsize,
}

fn bench_false_sharing_micro(c: &mut Criterion) {
    let mut g = c.benchmark_group("false_sharing_micro");
    g.throughput(Throughput::Elements((MICRO_OPS_PER_THREAD * 2) as u64));

    g.bench_function("contended", |b| {
        b.iter(|| {
            let data = Arc::new(Contended {
                x: AtomicUsize::new(0),
                y: AtomicUsize::new(0),
            });
            run_pair(
                data,
                |d| {
                    for _ in 0..MICRO_OPS_PER_THREAD {
                        d.x.fetch_add(1, Ordering::Relaxed);
                    }
                },
                |d| {
                    for _ in 0..MICRO_OPS_PER_THREAD {
                        d.y.fetch_add(1, Ordering::Relaxed);
                    }
                },
            );
        });
    });

    g.bench_function("padded", |b| {
        b.iter(|| {
            let data = Arc::new(Padded {
                x: AtomicUsize::new(0),
                _pad: [0; 120],
                y: AtomicUsize::new(0),
            });
            run_pair(
                data,
                |d| {
                    for _ in 0..MICRO_OPS_PER_THREAD {
                        d.x.fetch_add(1, Ordering::Relaxed);
                    }
                },
                |d| {
                    for _ in 0..MICRO_OPS_PER_THREAD {
                        d.y.fetch_add(1, Ordering::Relaxed);
                    }
                },
            );
        });
    });

    g.finish();
}

/// Per-thread allocation count for the SharedBumpArena bench. Sized so
/// 4 threads * this count fits inside the inline backing without OOM.
const ALLOC_OPS_PER_THREAD: usize = 256;
const ALLOC_SIZE: usize = 64;
const N_THREADS: usize = 4;
/// 4 threads * 256 allocs * 64 bytes = 64 KiB, plus headroom.
const BACKING_BYTES: usize = N_THREADS * ALLOC_OPS_PER_THREAD * ALLOC_SIZE + 4096;

fn bench_shared_bump_contention(c: &mut Criterion) {
    let layout = NonZeroLayout::from_size_align(ALLOC_SIZE, 8).unwrap();
    let mut g = c.benchmark_group("allocator_contention");
    g.throughput(Throughput::Elements(
        (N_THREADS * ALLOC_OPS_PER_THREAD) as u64,
    ));

    g.bench_function("shared_bump_4_threads", |b| {
        b.iter_batched(
            || Arc::new(SharedBumpArena::new(InlineBacked::<BACKING_BYTES>::new()).unwrap()),
            |arena| {
                let handles: Vec<_> = (0..N_THREADS)
                    .map(|_| {
                        let a = Arc::clone(&arena);
                        thread::spawn(move || {
                            for _ in 0..ALLOC_OPS_PER_THREAD {
                                let r = a.allocate(black_box(layout)).unwrap();
                                black_box(r);
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.bench_function("statistics_over_shared_bump_4_threads", |b| {
        b.iter_batched(
            || {
                Arc::new(Statistics::new(
                    SharedBumpArena::new(InlineBacked::<BACKING_BYTES>::new()).unwrap(),
                ))
            },
            |stats| {
                let handles: Vec<_> = (0..N_THREADS)
                    .map(|_| {
                        let s = Arc::clone(&stats);
                        thread::spawn(move || {
                            for _ in 0..ALLOC_OPS_PER_THREAD {
                                let r = s.allocate(black_box(layout)).unwrap();
                                black_box(r);
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_false_sharing_micro,
    bench_shared_bump_contention,
);
criterion_main!(benches);
