//! Adaptive vs Fixed batch-policy throughput benchmark.
//!
//! Workload: a single owner thread that allocates+frees on its own
//! local freelist plus 2 remote senders that push frees into the
//! remote queue. The benchmark measures owner throughput (allocate
//! latency on the hot path) under each policy.
//!
//! This benchmark is the validation gate for the Adaptive policy:
//! a new batch-size control law lands only if it consistently beats
//! the Fixed-policy baselines measured here.
//!
//! # Measurement design
//!
//! The earlier version of this bench built the owner + spawned remote
//! threads inside each Criterion iter — measurement was dominated by
//! thread-spawn jitter (~tens of µs per iter vs ~hundreds of ns for
//! the alloc/free loop). The current design:
//!
//! 1. Build the owner and remote threads ONCE per `bench_with_input`
//!    invocation. Remotes run a steady-state push loop until told to
//!    stop. The owner does NOT participate in the warmup.
//! 2. `iter` calls the owner's allocate/free loop directly. Criterion
//!    handles sampling; we don't use `iter_custom`.
//! 3. After the benchmark group finishes we signal the remotes to
//!    stop and join.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use forge_alloc::MmapBacked;
use forge_alloc::{Allocator, Deallocator, NonZeroLayout};
use forge_alloc::{BatchPolicy, SlabOwner};

/// Run a single policy with a long-lived owner + 2 remote senders.
fn run_policy(c: &mut Criterion, group_name: &str, policy: BatchPolicy) {
    const SLAB_CAP: usize = 4096;
    const QUEUE_CAP: usize = 1024;
    const PER_THREAD_POOL: usize = 128;

    let backing = MmapBacked::new(SLAB_CAP * 64).unwrap();
    let owner: SlabOwner<u64, MmapBacked> =
        SlabOwner::with_batch_policy(SLAB_CAP, backing, policy, QUEUE_CAP).unwrap();
    let layout = NonZeroLayout::for_type::<u64>().unwrap();

    // Pre-allocate a pool of pointers per remote thread; remotes ship
    // frees onto the queue continuously during the bench window.
    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    for _ in 0..2 {
        let chunk: Vec<usize> = (0..PER_THREAD_POOL)
            .map(|_| {
                owner
                    .allocate(layout)
                    .expect("bench setup: pool alloc failed; raise SLAB_CAP")
                    .cast::<u8>()
                    .as_ptr() as usize
            })
            .collect();
        let remote = owner.remote();
        let stop = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                for &addr in &chunk {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    // SAFETY: addr came from owner.allocate above with
                    // matching layout.
                    let p = unsafe { core::ptr::NonNull::new_unchecked(addr as *mut u8) };
                    let _ = unsafe { remote.try_deallocate(p, layout) };
                }
                std::hint::spin_loop();
            }
        }));
    }

    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);
    group.throughput(Throughput::Elements(1));
    group.bench_with_input(
        BenchmarkId::from_parameter(format!("{policy:?}")),
        &policy,
        |b, _| {
            b.iter(|| {
                // Single alloc + local free. Criterion samples this in
                // batches; measurement excludes thread-spawn overhead.
                if let Ok(block) = owner.allocate(layout) {
                    // SAFETY: block came from owner.allocate above.
                    unsafe { owner.deallocate(block.cast(), layout) };
                }
            });
        },
    );
    group.finish();

    // Signal remotes and join.
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }
}

fn owner_throughput(c: &mut Criterion) {
    // Each policy runs in its own group so the long-lived setup
    // doesn't conflate metrics across policies.
    run_policy(c, "slab_owner_fixed_8", BatchPolicy::Fixed(8));
    run_policy(c, "slab_owner_fixed_64", BatchPolicy::Fixed(64));
    run_policy(c, "slab_owner_fixed_128", BatchPolicy::Fixed(128));
    run_policy(c, "slab_owner_adaptive", BatchPolicy::Adaptive);
}

criterion_group!(adaptive_batch_benches, owner_throughput);
criterion_main!(adaptive_batch_benches);
