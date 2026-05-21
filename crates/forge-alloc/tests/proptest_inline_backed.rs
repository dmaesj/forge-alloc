//! Property-based tests for `InlineBacked`-backed allocators and
//! their hardening compositions.
//!
//! Skipped entirely under miri: the proptest runner internally calls
//! `std::env::current_dir()` for failure-persistence-file resolution,
//! which miri's isolation refuses to shim. Running this file under
//! `cargo miri test` is therefore not the goal — the per-crate `lib`
//! tests already exercise every wrapper's allocate/deallocate flow
//! against miri's UB detector. The property tests stay running under
//! `cargo test` for randomized coverage.
#![cfg(not(miri))]
//!
//! The NRVO regression battery in `nrvo_defeat.rs` exercises a small set of *fixed*
//! scenarios under stack pressure. This file complements it with
//! `proptest`-generated random sequences that probe the same invariants
//! across thousands of randomly-shaped histories.
//!
//! Every test in this file follows the same shape:
//!
//!   1. Build the wrapper in an `#[inline(never)]` helper that returns
//!      by value — the standard NRVO-defeating recipe.
//!   2. Apply random stack pressure between the constructor return and
//!      first use; this means even if the optimizer collapses caller
//!      and callee slots for one input, the next random input is
//!      likely to lay out differently.
//!   3. Generate a random sequence of `Alloc(size, align) / Dealloc(idx)`
//!      operations and apply them to the wrapper.
//!   4. After every operation, assert the resulting pointer lies inside
//!      the LIVE backing range (`backing().base() .. base() + size()`).
//!
//! The properties tested fall into four buckets:
//!
//! - **No double-issuance** — no two live pointers overlap.
//! - **Capacity-respecting** — bytes-allocated never exceeds the
//!   wrapper's reported capacity.
//! - **LIFO / FIFO contract** — Slab + StackAlloc dealloc-then-realloc
//!   returns the most-recently-freed slot.
//! - **Wrapper invariants** — Statistics counter accuracy, Watermark
//!   allocated-bytes accuracy, Quarantine FIFO eviction, GenerationalSlab
//!   generation monotonicity, WithFallback provenance routing.
//!
//! Each test runs ~256 random sequences by default; sequence length is
//! capped at 100 to keep runtime reasonable.

#![cfg(feature = "std")]
#![allow(clippy::too_many_lines)]

use core::hint::black_box;
use core::ptr::NonNull;

use forge_alloc::{
    Allocator, BumpArena, Deallocator, FixedRange, GenerationalSlab, InlineBacked, NonZeroLayout,
    NullHandler, Quarantine, Slab, StackAlloc, Statistics, System, Watermark, WithFallback,
};

use proptest::prelude::*;

// ============================================================================
// Shared helpers (mirroring the nrvo_defeat.rs NRVO-defeat shape)
// ============================================================================

/// Apply heavy stack pressure between the constructor-return and the
/// first use so NRVO / RVO cannot collapse the caller's slot onto the
/// constructor's slot. Returns its argument unchanged through `black_box`.
#[inline(never)]
fn stack_pressure<T>(t: T) -> T {
    let arr = [0u64; 1024];
    black_box(&arr);
    black_box(t)
}

/// Apply a variable amount of stack pressure (proptest-controlled). The
/// closure receives the wrapper *after* the random-sized array has been
/// allocated, so any structure-relative backing must have survived the
/// shifted stack frame.
#[inline(never)]
fn with_random_stack_pressure<T, R>(t: T, pressure_words: usize, f: impl FnOnce(T) -> R) -> R {
    // Bound stack frame at 16 KiB so we don't blow the stack under
    // proptest's many-iteration regime.
    let words = pressure_words.min(2048);
    // A heap-allocated Vec under proptest control: gives us address
    // diversity without risking stack overflow on Windows where the
    // default thread stack is ~1 MiB.
    let heap_pressure: Vec<u64> = vec![0u64; words];
    black_box(&heap_pressure);
    // Also a fixed stack scratch the optimizer can't elide.
    let arr = [0u64; 512];
    black_box(&arr);
    f(black_box(t))
}

/// Assert pointer `ptr` falls inside `[base, base + size)`.
#[track_caller]
fn assert_in_range(ptr: NonNull<u8>, base: NonNull<u8>, size: usize, what: &str) {
    let p = ptr.as_ptr() as usize;
    let b = base.as_ptr() as usize;
    assert!(
        p >= b && p < b + size,
        "{what}: pointer {p:#x} not in [{b:#x}, {:#x}) — STALE BASE POINTER",
        b + size,
    );
}

// ============================================================================
// Operation generators
// ============================================================================

/// Slab / StackAlloc operation: random alloc / dealloc with arbitrary
/// dealloc target indices. The size is fixed by the test (slab is a
/// single-size pool); the `Dealloc(idx)` mode picks from the *live*
/// vector and is interpreted modulo `live.len()` by the test harness.
#[derive(Debug, Clone, Copy)]
enum SlabOp {
    Alloc,
    /// Dealloc the LIFO top (most recent live ptr).
    DeallocTop,
    /// Dealloc a random live ptr (swap_remove in the harness).
    DeallocAny(usize),
}

fn arb_slab_op() -> impl Strategy<Value = SlabOp> {
    prop_oneof![
        4 => Just(SlabOp::Alloc),
        2 => Just(SlabOp::DeallocTop),
        2 => any::<usize>().prop_map(SlabOp::DeallocAny),
    ]
}

fn arb_slab_op_seq() -> impl Strategy<Value = Vec<SlabOp>> {
    proptest::collection::vec(arb_slab_op(), 0..100)
}

/// StackAlloc requires strict LIFO discipline; out-of-order frees would
/// be UB. So this generator only emits `Alloc` and `DeallocTop`.
#[derive(Debug, Clone, Copy)]
enum StackOp {
    Alloc,
    Dealloc,
}

fn arb_stack_op() -> impl Strategy<Value = StackOp> {
    prop_oneof![
        3 => Just(StackOp::Alloc),
        2 => Just(StackOp::Dealloc),
    ]
}

fn arb_stack_op_seq() -> impl Strategy<Value = Vec<StackOp>> {
    proptest::collection::vec(arb_stack_op(), 0..100)
}

/// BumpArena layout-only op (BumpArena's `deallocate` is a no-op).
fn arb_bump_layout_seq() -> impl Strategy<Value = Vec<NonZeroLayout>> {
    proptest::collection::vec(
        (1usize..=128, 0u32..=4).prop_map(|(size, align_log)| {
            NonZeroLayout::from_size_align(size, 1usize << align_log).unwrap()
        }),
        0..100,
    )
}

// ============================================================================
// 1. Slab post-move random sequence — pointers stay in the live range
//    and no two live pointers ever overlap.
// ============================================================================

#[inline(never)]
fn build_slab() -> Slab<u64, InlineBacked<2048>> {
    Slab::new(64, InlineBacked::<2048>::new()).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Random alloc/dealloc sequence on a moved Slab — every pointer
    /// returned must lie inside `[backing.base(), base + size)` and no
    /// two simultaneously-live pointers may overlap.
    #[test]
    fn pin_slab_random_seq_pointers_stay_in_live_backing_post_move(
        ops in arb_slab_op_seq(),
        pressure in 0usize..=2048,
    ) {
        let s = stack_pressure(build_slab());
        with_random_stack_pressure(s, pressure, |s| {
            let backing_base = s.backing().base();
            let backing_size = s.backing().size();
            let layout = NonZeroLayout::for_type::<u64>().unwrap();
            let mut live: Vec<NonNull<u8>> = Vec::new();

            for op in ops {
                match op {
                    SlabOp::Alloc => {
                        if let Ok(block) = s.allocate(layout) {
                            let p = block.cast::<u8>();
                            assert_in_range(p, backing_base, backing_size, "Slab alloc post-move");
                            // No overlap with any other live ptr.
                            for q in &live {
                                prop_assert_ne!(p.as_ptr(), q.as_ptr(), "Slab returned a duplicate live ptr");
                            }
                            live.push(p);
                        }
                    }
                    SlabOp::DeallocTop => {
                        if let Some(p) = live.pop() {
                            unsafe { s.deallocate(p, layout) };
                        }
                    }
                    SlabOp::DeallocAny(raw_idx) => {
                        if !live.is_empty() {
                            let idx = raw_idx % live.len();
                            let p = live.swap_remove(idx);
                            unsafe { s.deallocate(p, layout) };
                        }
                    }
                }
            }
            // Drain.
            for p in live {
                unsafe { s.deallocate(p, layout) };
            }
            Ok::<(), proptest::test_runner::TestCaseError>(())
        })?;
    }

    /// Slab capacity invariant: total live pointers never exceeds the
    /// slab's configured capacity. We use a slab with 64 slots and assert
    /// `live.len() <= 64` after every successful alloc.
    #[test]
    fn pin_slab_capacity_never_exceeded(ops in arb_slab_op_seq()) {
        let s = stack_pressure(build_slab());
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let mut live: Vec<NonNull<u8>> = Vec::new();
        for op in ops {
            match op {
                SlabOp::Alloc => {
                    if let Ok(block) = s.allocate(layout) {
                        live.push(block.cast());
                        prop_assert!(
                            live.len() <= s.capacity(),
                            "Slab handed out {} live blocks but capacity is {}",
                            live.len(),
                            s.capacity(),
                        );
                    }
                }
                SlabOp::DeallocTop => {
                    if let Some(p) = live.pop() {
                        unsafe { s.deallocate(p, layout) };
                    }
                }
                SlabOp::DeallocAny(raw_idx) => {
                    if !live.is_empty() {
                        let idx = raw_idx % live.len();
                        let p = live.swap_remove(idx);
                        unsafe { s.deallocate(p, layout) };
                    }
                }
            }
        }
        for p in live {
            unsafe { s.deallocate(p, layout) };
        }
    }

    /// LIFO contract: after `dealloc(p)` immediately followed by
    /// `allocate(...)`, the new pointer equals `p`. Tested across random
    /// sequences of allocs interspersed with dealloc-realloc pairs.
    #[test]
    fn pin_slab_dealloc_then_alloc_returns_most_recent_slot(n in 1usize..=32) {
        let s = stack_pressure(build_slab());
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let mut live: Vec<NonNull<u8>> = Vec::new();
        for _ in 0..n {
            live.push(s.allocate(layout).unwrap().cast());
        }
        // Pop one, dealloc, then immediately re-alloc — must return the
        // exact same slot (Slab freelist is LIFO).
        for _ in 0..n {
            let popped = live.pop().unwrap();
            unsafe { s.deallocate(popped, layout) };
            let next = s.allocate(layout).unwrap().cast::<u8>();
            prop_assert_eq!(
                popped.as_ptr(),
                next.as_ptr(),
                "Slab violated LIFO: dealloc-then-alloc must return the freed slot",
            );
            // Put it back at the top so the next iteration pops it again.
            live.push(next);
        }
        for p in live {
            unsafe { s.deallocate(p, layout) };
        }
    }
}

// ============================================================================
// 2. BumpArena post-move random layout sequence
// ============================================================================

#[inline(never)]
fn build_bump() -> BumpArena<InlineBacked<4096>> {
    BumpArena::new(InlineBacked::<4096>::new()).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Bump arena no-overlap + in-range: random layout sequence after a
    /// forced move. Every successful alloc must lie inside the live
    /// backing range and not overlap any prior live range.
    #[test]
    fn pin_bump_arena_random_layouts_no_overlap_in_range_post_move(
        layouts in arb_bump_layout_seq(),
        pressure in 0usize..=2048,
    ) {
        let arena = stack_pressure(build_bump());
        with_random_stack_pressure(arena, pressure, |arena| {
            let backing_base = arena.backing().base();
            let capacity = arena.capacity();
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            for layout in layouts {
                if let Ok(block) = arena.allocate(layout) {
                    let p = block.cast::<u8>();
                    assert_in_range(p, backing_base, capacity, "BumpArena post-move alloc");
                    let start = p.as_ptr() as usize;
                    let end = start + block.len();
                    for (s, e) in &ranges {
                        let disjoint = end <= *s || start >= *e;
                        prop_assert!(
                            disjoint,
                            "bump arena overlap: new [{start:#x},{end:#x}) vs live [{:#x},{:#x})",
                            *s,
                            *e,
                        );
                    }
                    ranges.push((start, end));
                }
            }
            Ok::<(), proptest::test_runner::TestCaseError>(())
        })?;
    }

    /// Bump arena capacity invariant: `allocated() + remaining() ==
    /// capacity()` after any sequence of allocates. The accounting
    /// invariant breaks if the cursor overshoots, which would be a UB.
    #[test]
    fn pin_bump_arena_allocated_plus_remaining_equals_capacity(
        layouts in arb_bump_layout_seq(),
    ) {
        let arena = stack_pressure(build_bump());
        for layout in layouts {
            let _ = arena.allocate(layout);
            prop_assert_eq!(arena.allocated() + arena.remaining(), arena.capacity());
            prop_assert!(arena.allocated() <= arena.capacity(),
                "BumpArena cursor exceeded capacity: allocated={}, capacity={}",
                arena.allocated(), arena.capacity());
        }
    }
}

// ============================================================================
// 3. StackAlloc post-move random LIFO sequence
// ============================================================================

#[inline(never)]
fn build_stack() -> StackAlloc<InlineBacked<4096>> {
    StackAlloc::new(InlineBacked::<4096>::new()).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// StackAlloc respects strict LIFO: random Alloc / Dealloc sequence
    /// (Dealloc only ever pops the top frame). Every alloc must lie in
    /// the live backing; after a full drain, allocated() == 0.
    #[test]
    fn pin_stack_alloc_random_lifo_sequence_post_move(
        ops in arb_stack_op_seq(),
        pressure in 0usize..=2048,
    ) {
        let s = stack_pressure(build_stack());
        with_random_stack_pressure(s, pressure, |s| {
            let backing_base = s.backing().base();
            let backing_size = s.backing().size();
            let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
            let mut live: Vec<NonNull<u8>> = Vec::new();

            for op in ops {
                match op {
                    StackOp::Alloc => {
                        if let Ok(block) = s.allocate(layout) {
                            let p = block.cast::<u8>();
                            assert_in_range(p, backing_base, backing_size, "StackAlloc post-move");
                            live.push(p);
                        }
                    }
                    StackOp::Dealloc => {
                        if let Some(p) = live.pop() {
                            unsafe { s.deallocate(p, layout) };
                        }
                    }
                }
            }
            // Drain LIFO.
            while let Some(p) = live.pop() {
                unsafe { s.deallocate(p, layout) };
            }
            prop_assert_eq!(s.allocated(), 0, "StackAlloc cursor not back at 0 after full drain");
            Ok::<(), proptest::test_runner::TestCaseError>(())
        })?;
    }

    /// StackAlloc LIFO-reuse: after popping the top, the next alloc
    /// returns the same address (cursor restored to prev_cursor).
    #[test]
    fn pin_stack_alloc_lifo_reuse(
        size in 1usize..=128,
        align_log in 0u32..=3,
        n in 1usize..=16,
    ) {
        let s = stack_pressure(build_stack());
        let align = 1usize << align_log;
        let layout = NonZeroLayout::from_size_align(size, align).unwrap();
        let mut ptrs = Vec::new();
        for _ in 0..n {
            if let Ok(p) = s.allocate(layout) {
                ptrs.push(p.cast::<u8>());
            } else {
                break;
            }
        }
        // Pop the top, then immediately reallocate — must equal the popped ptr.
        if let Some(top) = ptrs.pop() {
            unsafe { s.deallocate(top, layout) };
            let again = s.allocate(layout).unwrap().cast::<u8>();
            prop_assert_eq!(
                top.as_ptr(),
                again.as_ptr(),
                "StackAlloc LIFO reuse returned a different address",
            );
            unsafe { s.deallocate(again, layout) };
        }
        for p in ptrs.into_iter().rev() {
            unsafe { s.deallocate(p, layout) };
        }
    }
}

// ============================================================================
// 4. Statistics<Slab<...>> counter accuracy
// ============================================================================

#[inline(never)]
fn build_stats_slab() -> Statistics<Slab<u64, InlineBacked<2048>>> {
    Statistics::new(Slab::new(64, InlineBacked::<2048>::new()).unwrap())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Statistics counter accuracy: after any sequence of alloc/dealloc,
    /// the wrapper's reported totals must match the harness-tracked
    /// totals exactly. `live_count() == total_alloc - total_dealloc`,
    /// `current_bytes() == live_count * size`, `peak >= current` at all
    /// times.
    #[test]
    fn pin_statistics_counter_accuracy(ops in arb_slab_op_seq()) {
        let s = stack_pressure(build_stats_slab());
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Counters became `AtomicUsize` for 32-bit portability — match
        // the harness types to `usize`.
        let size = layout.size().get();
        let mut live: Vec<NonNull<u8>> = Vec::new();
        let mut harness_alloc: usize = 0;
        let mut harness_dealloc: usize = 0;

        for op in ops {
            match op {
                SlabOp::Alloc => {
                    if let Ok(block) = s.allocate(layout) {
                        live.push(block.cast());
                        harness_alloc += 1;
                    }
                }
                SlabOp::DeallocTop => {
                    if let Some(p) = live.pop() {
                        unsafe { s.deallocate(p, layout) };
                        harness_dealloc += 1;
                    }
                }
                SlabOp::DeallocAny(raw_idx) => {
                    if !live.is_empty() {
                        let idx = raw_idx % live.len();
                        let p = live.swap_remove(idx);
                        unsafe { s.deallocate(p, layout) };
                        harness_dealloc += 1;
                    }
                }
            }
            // After every op: invariants hold.
            let stats = s.stats();
            let wrapper_alloc = stats.total_allocations.load(core::sync::atomic::Ordering::Relaxed);
            let wrapper_dealloc = stats.total_deallocations.load(core::sync::atomic::Ordering::Relaxed);
            prop_assert_eq!(wrapper_alloc, harness_alloc, "alloc counter drift");
            prop_assert_eq!(wrapper_dealloc, harness_dealloc, "dealloc counter drift");

            let expected_live = live.len() as i64;
            prop_assert_eq!(stats.live_count(), expected_live, "live_count drift");

            let current = stats.current_bytes();
            prop_assert_eq!(current, (live.len() as u64) * (size as u64), "current_bytes drift");

            let peak = stats.peak_bytes();
            prop_assert!(peak >= current, "peak < current ({peak} < {current})");
        }
        for p in live {
            unsafe { s.deallocate(p, layout) };
        }
    }
}

// ============================================================================
// 5. Watermark<Slab<...>> peak monotonicity
// ============================================================================

#[inline(never)]
fn build_watermark_slab() -> Watermark<Slab<u64, InlineBacked<2048>>, NullHandler> {
    Watermark::new(
        Slab::new(64, InlineBacked::<2048>::new()).unwrap(),
        NullHandler,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Watermark `allocated_bytes()` matches the harness-tracked live
    /// byte total exactly, never goes negative under valid sequences,
    /// and never exceeds the wrapper's reported capacity.
    #[test]
    fn pin_watermark_allocated_bytes_matches_harness(ops in arb_slab_op_seq()) {
        let w = stack_pressure(build_watermark_slab());
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let size = layout.size().get();
        let mut live: Vec<NonNull<u8>> = Vec::new();
        let capacity = w.capacity_bytes().unwrap_or(usize::MAX);

        for op in ops {
            match op {
                SlabOp::Alloc => {
                    if let Ok(block) = w.allocate(layout) {
                        live.push(block.cast());
                    }
                }
                SlabOp::DeallocTop => {
                    if let Some(p) = live.pop() {
                        unsafe { w.deallocate(p, layout) };
                    }
                }
                SlabOp::DeallocAny(raw_idx) => {
                    if !live.is_empty() {
                        let idx = raw_idx % live.len();
                        let p = live.swap_remove(idx);
                        unsafe { w.deallocate(p, layout) };
                    }
                }
            }
            let allocated = w.allocated_bytes();
            prop_assert_eq!(
                allocated,
                live.len() * size,
                "Watermark allocated_bytes drift",
            );
            prop_assert!(allocated <= capacity,
                "Watermark allocated_bytes ({allocated}) exceeded capacity ({capacity})");
        }
        for p in live {
            unsafe { w.deallocate(p, layout) };
        }
    }
}

// ============================================================================
// 6. Quarantine<Slab> FIFO eviction order
// ============================================================================

#[inline(never)]
fn build_quarantine_slab<const E: usize>() -> Quarantine<Slab<u64, InlineBacked<2048>>, E> {
    Quarantine::new(Slab::new(64, InlineBacked::<2048>::new()).unwrap())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Quarantine FIFO: items entering at position i are evicted at
    /// position i + EPOCHS. Verify by allocating 2*EPOCHS slots,
    /// freeing them in order, and then realloc'ing — the realloc'd
    /// addresses must be the *first* EPOCHS we freed (Slab is LIFO on
    /// freed slots, so the last evicted shows up first, and eviction
    /// order is the freeing order, which means we get evicted blocks in
    /// reverse-eviction order — i.e. the most recent eviction first,
    /// which is freed-index `EPOCHS` last in the eviction stream).
    ///
    /// Easier invariant: after EPOCHS frees the ring is full. The next
    /// free evicts the *first* freed slot. Therefore that slot must be
    /// available again on the next alloc.
    #[test]
    fn pin_quarantine_evicts_in_fifo_order(n_extra in 1usize..=8) {
        // EPOCHS = 4: we free `EPOCHS + n_extra` slots and expect each
        // free past the EPOCHS-th to evict the oldest still-held block.
        const EPOCHS: usize = 4;
        let q = stack_pressure(build_quarantine_slab::<EPOCHS>());
        let layout = NonZeroLayout::for_type::<u64>().unwrap();

        // Pre-allocate (EPOCHS + n_extra) slots so we can free them in a
        // known order.
        let total = EPOCHS + n_extra;
        let mut allocs: Vec<NonNull<u8>> = Vec::with_capacity(total);
        for _ in 0..total {
            allocs.push(q.allocate(layout).unwrap().cast());
        }

        // Free in order allocs[0], allocs[1], ...:
        //   - The first EPOCHS go into the ring.
        //   - Subsequent frees evict allocs[0], allocs[1], ... in order
        //     back to the Slab. The Slab itself is LIFO, so the most-
        //     recently-evicted slot is the next one handed out by alloc.
        for p in &allocs {
            unsafe { q.deallocate(*p, layout) };
        }

        // After freeing `total = EPOCHS + n_extra` blocks:
        //   - allocs[0..n_extra] were evicted to the Slab (in that
        //     order). Slab LIFO returns the *last* evicted = allocs[n_extra - 1].
        //   - allocs[n_extra..total] are still in the ring.
        let next = q.allocate(layout).unwrap().cast::<u8>();
        prop_assert_eq!(
            next.as_ptr(),
            allocs[n_extra - 1].as_ptr(),
            "Quarantine eviction violated FIFO: expected allocs[{}]={:?}, got {:?}",
            n_extra - 1,
            allocs[n_extra - 1].as_ptr(),
            next.as_ptr(),
        );

        // Don't deallocate the rest — they're still held in the ring,
        // and the Quarantine Drop impl drains them back to Slab.
        // But do dealloc the one we just got back so the Slab is fully
        // accounted for; Quarantine's Drop will then drain the ring.
        unsafe { q.deallocate(next, layout) };
    }

    /// Quarantine cycle count: `deallocate_count()` increments by 1 per
    /// deallocate. Invariant after N frees: `deallocate_count() == N`.
    #[test]
    fn pin_quarantine_deallocate_count_matches_frees(n in 1usize..=32) {
        let q = stack_pressure(build_quarantine_slab::<4>());
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let mut live: Vec<NonNull<u8>> = Vec::new();
        for _ in 0..n {
            live.push(q.allocate(layout).unwrap().cast());
        }
        for (i, p) in live.iter().enumerate() {
            unsafe { q.deallocate(*p, layout) };
            prop_assert_eq!(q.deallocate_count(), i + 1);
        }
    }
}

// ============================================================================
// 7. GenerationalSlab generation monotonicity per slot
// ============================================================================

#[inline(never)]
fn build_gen_slab() -> GenerationalSlab<u64, InlineBacked<2048>> {
    GenerationalSlab::new(32, InlineBacked::<2048>::new()).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// GenerationalSlab generation monotonicity: for any sequence of
    /// insert/remove pairs on the *same slot*, the slot's generation
    /// strictly increases (modulo wrap, which u32 won't reach inside a
    /// test). Stale handles must always return None.
    #[test]
    fn pin_generational_slab_stale_handles_always_reject(
        cycles in 1usize..=16,
    ) {
        let mut s = stack_pressure(build_gen_slab());
        let mut stale_handles = Vec::new();

        for i in 0..cycles {
            let h = s.insert(i as u64).unwrap();
            // Read back fresh handle — must succeed.
            prop_assert_eq!(s.get(h).copied(), Some(i as u64));
            // Remove — handle now stale.
            let removed = s.remove(h).unwrap();
            prop_assert_eq!(removed, i as u64);
            // Stale handle must now return None.
            prop_assert!(s.get(h).is_none(), "stale handle returned a value after remove");
            stale_handles.push(h);
        }
        // After many cycles, every prior stale handle must STILL be stale
        // (generation moved forward, so the original handle's generation
        // no longer matches).
        for h in &stale_handles {
            prop_assert!(
                s.get(*h).is_none(),
                "Cross-cycle stale handle was incorrectly accepted",
            );
        }
    }

    /// GenerationalSlab handle uniqueness: any two outstanding handles
    /// returned by `insert` must compare unequal (different index OR
    /// different generation).
    #[test]
    fn pin_generational_slab_handles_unique(n in 1usize..=32) {
        let mut s = stack_pressure(build_gen_slab());
        use std::collections::HashSet;
        let mut seen: HashSet<_> = HashSet::new();
        // First fill phase: every handle has a unique index.
        for i in 0..n {
            let h = s.insert(i as u64).unwrap();
            prop_assert!(seen.insert(h), "duplicate handle on initial fill");
        }
        // Snapshot — drain then re-fill, the new handles must STILL be
        // unique vs the old (different generation).
        let old: Vec<_> = seen.iter().copied().collect();
        for h in &old {
            let _ = s.remove(*h);
        }
        seen.clear();
        for i in 0..n {
            let h = s.insert(i as u64 + 1000).unwrap();
            prop_assert!(seen.insert(h), "duplicate handle after refill");
            prop_assert!(!old.contains(&h),
                "GenerationalSlab handed back the same handle after a remove");
        }
    }
}

// ============================================================================
// 8. WithFallback provenance routing
// ============================================================================

#[inline(never)]
fn build_with_fallback() -> WithFallback<InlineBacked<256>, System> {
    WithFallback::new(InlineBacked::<256>::new(), System)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// WithFallback provenance routing: every successful primary
    /// allocation reports `primary().contains(ptr) == true`. Every
    /// allocation that the primary refuses comes back via the
    /// secondary and reports `primary().contains(ptr) == false`.
    #[test]
    fn pin_with_fallback_provenance_routing(
        layouts in proptest::collection::vec(
            (1usize..=200, 0u32..=3).prop_map(|(size, align_log)| {
                NonZeroLayout::from_size_align(size, 1usize << align_log).unwrap()
            }),
            0..32,
        ),
    ) {
        let wf = stack_pressure(build_with_fallback());
        let mut secondary_ptrs: Vec<(NonNull<u8>, NonZeroLayout)> = Vec::new();

        for layout in layouts {
            if let Ok(block) = wf.allocate(layout) {
                let p = block.cast::<u8>();
                let in_primary = wf.primary().contains(p);
                let primary_base = wf.primary().base().as_ptr() as usize;
                let primary_end = primary_base + wf.primary().size();
                let p_addr = p.as_ptr() as usize;
                let in_primary_range = p_addr >= primary_base && p_addr < primary_end;
                prop_assert_eq!(in_primary, in_primary_range,
                    "primary.contains() disagrees with manual range check");

                if !in_primary {
                    // It came from System. Track for cleanup.
                    secondary_ptrs.push((p, layout));
                }
            }
        }

        // Clean up only secondary allocations. (Primary is BumpArena-like
        // inline; dealloc is a no-op there.)
        for (p, layout) in secondary_ptrs {
            // Verify the routing: deallocate-time `primary.contains` must
            // be false, so the secondary is hit. We can't directly assert
            // the secondary call happened, but we can pre-check the
            // routing predicate just like the wrapper does.
            prop_assert!(!wf.primary().contains(p),
                "deallocate-time primary.contains() returned true for a secondary-issued ptr");
            unsafe { wf.deallocate(p, layout) };
        }
    }
}

// ============================================================================
// 9. Cross-wrapper composition: Statistics<Watermark<Slab>>
// ============================================================================

#[inline(never)]
fn build_stats_watermark_slab(
) -> Statistics<Watermark<Slab<u64, InlineBacked<2048>>, NullHandler>> {
    Statistics::new(Watermark::new(
        Slab::new(64, InlineBacked::<2048>::new()).unwrap(),
        NullHandler,
    ))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Cross-wrapper composition: Statistics' counters must agree with
    /// the inner Watermark's counters at all times, because both
    /// observe the same allocator history through the same allocate /
    /// deallocate path.
    #[test]
    fn pin_stats_over_watermark_counters_agree(ops in arb_slab_op_seq()) {
        let s = stack_pressure(build_stats_watermark_slab());
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let size = layout.size().get();
        let mut live: Vec<NonNull<u8>> = Vec::new();

        for op in ops {
            match op {
                SlabOp::Alloc => {
                    if let Ok(block) = s.allocate(layout) {
                        live.push(block.cast());
                    }
                }
                SlabOp::DeallocTop => {
                    if let Some(p) = live.pop() {
                        unsafe { s.deallocate(p, layout) };
                    }
                }
                SlabOp::DeallocAny(raw_idx) => {
                    if !live.is_empty() {
                        let idx = raw_idx % live.len();
                        let p = live.swap_remove(idx);
                        unsafe { s.deallocate(p, layout) };
                    }
                }
            }
            // Cross-wrapper agreement: Statistics' current_bytes equals
            // Watermark's allocated_bytes (Watermark is INSIDE Statistics
            // and both count the same outer-caller layout sizes).
            let stat_current = s.stats().current_bytes() as usize;
            let wm_current = s.inner().allocated_bytes();
            prop_assert_eq!(stat_current, wm_current,
                "Statistics current_bytes={} != Watermark allocated_bytes={}",
                stat_current, wm_current);
            // Both also agree with the harness count.
            prop_assert_eq!(stat_current, live.len() * size);
        }
        for p in live {
            unsafe { s.deallocate(p, layout) };
        }
    }
}

// ============================================================================
// 10. Cross-wrapper composition: WithFallback<Statistics<...>, Statistics<...>>
//     — each half's counters reflect only the allocs it served.
// ============================================================================

#[inline(never)]
fn build_split_stats_fallback() -> WithFallback<
    Statistics<BumpArena<InlineBacked<256>>>,
    Statistics<System>,
> {
    WithFallback::new(
        Statistics::new(BumpArena::new(InlineBacked::<256>::new()).unwrap()),
        Statistics::new(System),
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Per-half statistics on a WithFallback: counts on each half must
    /// agree with the harness-tracked counts of which half served each
    /// request.
    #[test]
    fn pin_with_fallback_per_half_stats_agree(
        layouts in proptest::collection::vec(
            (1usize..=200, 0u32..=3).prop_map(|(size, align_log)| {
                NonZeroLayout::from_size_align(size, 1usize << align_log).unwrap()
            }),
            0..16,
        ),
    ) {
        let wf = stack_pressure(build_split_stats_fallback());
        // Match Statistics counters' AtomicUsize width (was u64 before
        // the 32-bit-portability refactor).
        let mut harness_primary = 0usize;
        let mut harness_secondary = 0usize;
        let mut secondary_ptrs: Vec<(NonNull<u8>, NonZeroLayout)> = Vec::new();

        for layout in layouts {
            if let Ok(block) = wf.allocate(layout) {
                let p = block.cast::<u8>();
                if wf.primary().inner().backing().base().as_ptr() as usize
                    <= p.as_ptr() as usize
                    && (p.as_ptr() as usize)
                        < wf.primary().inner().backing().base().as_ptr() as usize
                            + wf.primary().inner().backing().size()
                {
                    harness_primary += 1;
                } else {
                    harness_secondary += 1;
                    secondary_ptrs.push((p, layout));
                }
            }
        }

        let prim_stats = wf.primary().stats();
        let sec_stats = wf.secondary().stats();
        prop_assert_eq!(
            prim_stats.total_allocations.load(core::sync::atomic::Ordering::Relaxed),
            harness_primary,
            "primary Statistics undercount",
        );
        prop_assert_eq!(
            sec_stats.total_allocations.load(core::sync::atomic::Ordering::Relaxed),
            harness_secondary,
            "secondary Statistics undercount",
        );

        // Clean up — only secondary (heap) allocs need deallocation.
        for (p, layout) in secondary_ptrs {
            unsafe { wf.deallocate(p, layout) };
        }
    }
}

// ============================================================================
// 11. Full-state invariant: post-move random sequence on the full
//     composition `PoisonOnFree<Quarantine<Slab>>` — every allocation
//     stays inside the live backing.
// ============================================================================

#[inline(never)]
fn build_poison_quarantine_slab() -> forge_alloc::PoisonOnFree<
    forge_alloc::Quarantine<Slab<u64, InlineBacked<2048>>, 4>,
> {
    forge_alloc::PoisonOnFree::new(forge_alloc::Quarantine::new(
        Slab::new(64, InlineBacked::<2048>::new()).unwrap(),
    ))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// PoisonOnFree<Quarantine<Slab>> random sequence: every allocated
    /// pointer lies inside the LIVE inner-most backing range, after the
    /// whole chain has been moved through stack_pressure.
    #[test]
    fn pin_poison_quarantine_slab_random_sequence_in_range_post_move(
        ops in arb_slab_op_seq(),
        pressure in 0usize..=2048,
    ) {
        let p = stack_pressure(build_poison_quarantine_slab());
        with_random_stack_pressure(p, pressure, |p| {
            let backing_base = p.inner().inner().backing().base();
            let backing_size = p.inner().inner().backing().size();
            let layout = NonZeroLayout::for_type::<u64>().unwrap();
            let mut live: Vec<NonNull<u8>> = Vec::new();

            for op in ops {
                match op {
                    SlabOp::Alloc => {
                        if let Ok(block) = p.allocate(layout) {
                            let q = block.cast::<u8>();
                            assert_in_range(q, backing_base, backing_size,
                                "PoisonOnFree<Quarantine<Slab>> alloc post-move");
                            live.push(q);
                        }
                    }
                    SlabOp::DeallocTop => {
                        if let Some(q) = live.pop() {
                            unsafe { p.deallocate(q, layout) };
                        }
                    }
                    SlabOp::DeallocAny(raw_idx) => {
                        if !live.is_empty() {
                            let idx = raw_idx % live.len();
                            let q = live.swap_remove(idx);
                            unsafe { p.deallocate(q, layout) };
                        }
                    }
                }
            }
            // Drain
            for q in live {
                unsafe { p.deallocate(q, layout) };
            }
            Ok::<(), proptest::test_runner::TestCaseError>(())
        })?;
    }
}

// ============================================================================
// 12. WithFallback try_new disjoint-ranges invariant: any two
//     InlineBacked instances really are disjoint, and the try_new
//     constructor succeeds.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Two independently-allocated `InlineBacked<N>` instances are
    /// disjoint and `try_new` accepts them.
    #[test]
    fn pin_with_fallback_try_new_accepts_disjoint_inline_backeds(
        which in 0u8..3,
    ) {
        // Three different sizes to vary the layout enough that any
        // adjacency bug would surface with one of them.
        let result = match which {
            0 => WithFallback::try_new(
                InlineBacked::<64>::new(),
                InlineBacked::<64>::new(),
            ).is_ok(),
            1 => WithFallback::try_new(
                InlineBacked::<256>::new(),
                InlineBacked::<256>::new(),
            ).is_ok(),
            _ => WithFallback::try_new(
                InlineBacked::<1024>::new(),
                InlineBacked::<1024>::new(),
            ).is_ok(),
        };
        prop_assert!(result, "try_new spuriously rejected disjoint InlineBacked pair");
    }
}
