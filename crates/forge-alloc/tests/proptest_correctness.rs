//! Property-based correctness tests for `BumpArena`, `Slab`, and
//! `WithFallback`. These exercise the invariants:
//!
//! Skipped under miri: proptest's runner calls `std::env::current_dir`
//! for failure-persistence-file resolution, which miri's isolation
//! refuses to shim. The per-module miri pass covers the same UB
//! invariants on a deterministic test set.
#![cfg(not(miri))]

//!
//! 1. Pointers are valid, non-null, aligned for the requested layout.
//! 2. The allocated region is at least `layout.size()` bytes.
//! 3. Two live allocations from the same allocator never overlap.
//! 4. After `reset()`, the cursor is at 0 (the spec says all previously
//!    issued pointers are invalid; we verify the cursor-reset side here
//!    and rely on the borrow-checker to enforce no live pointers).
//! 5. (Slab) After `deallocate`, the slot is on the free list and the next
//!    `allocate` returns it (LIFO).
//!
//! Property-based, not example-based — proptest generates the layouts,
//! allocation sequences, and reset points.

use forge_alloc::InlineBacked;
use forge_alloc::{Allocator, Deallocator, NoProtection, NonZeroLayout};
use forge_alloc::{BumpArena, Slab};
use proptest::prelude::*;

// A layout generator: size in [1, 256], alignment in {1,2,4,8,16}.
prop_compose! {
    fn arb_layout()(
        size in 1usize..=256,
        align_log in 0u32..=4,
    ) -> NonZeroLayout {
        let align = 1usize << align_log;
        NonZeroLayout::from_size_align(size, align).unwrap()
    }
}

// A "sequence of allocation requests" — a Vec of layouts.
fn arb_layout_seq() -> impl Strategy<Value = Vec<NonZeroLayout>> {
    proptest::collection::vec(arb_layout(), 0..32)
}

// ============================================================================
// BumpArena<InlineBacked<1024>>
// ============================================================================

proptest! {
    #[test]
    fn bump_arena_no_overlapping_allocations(layouts in arb_layout_seq()) {
        let arena = BumpArena::new(InlineBacked::<8192>::new()).unwrap();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for layout in layouts {
            let result = arena.allocate(layout);
            if let Ok(block) = result {
                let start = block.cast::<u8>().as_ptr() as usize;
                let end = start + block.len();
                // Verify no overlap with any prior live allocation.
                for (s, e) in &ranges {
                    let no_overlap = end <= *s || start >= *e;
                    prop_assert!(
                        no_overlap,
                        "overlap: new range [{:#x}, {:#x}) overlaps live [{:#x}, {:#x}) for layout size={} align={}",
                        start, end, *s, *e,
                        layout.size().get(),
                        layout.align().get(),
                    );
                }
                // Verify alignment.
                prop_assert_eq!(
                    start % layout.align().get(),
                    0,
                    "ptr {:#x} not aligned to {}",
                    start,
                    layout.align().get(),
                );
                // Verify size.
                prop_assert!(
                    end - start >= layout.size().get(),
                    "block size {} < requested {}",
                    end - start,
                    layout.size().get(),
                );
                ranges.push((start, end));
            }
        }
    }

    #[test]
    fn bump_arena_reset_returns_cursor_to_zero(layouts in arb_layout_seq()) {
        let mut arena = BumpArena::new(InlineBacked::<8192>::new()).unwrap();
        // Force at least one successful allocation so we actually exercise
        // reset's reclaim path — proptest may shrink to an empty `layouts`
        // Vec, in which case `allocated()` is already 0 and reset is a no-op.
        let seed = NonZeroLayout::from_size_align(1, 1).unwrap();
        arena.allocate(seed).expect("seed alloc must succeed on fresh 8 KiB arena");
        prop_assert!(arena.allocated() > 0, "seed alloc did not advance cursor");
        for layout in layouts {
            let _ = arena.allocate(layout);
        }
        // Some additional allocations may have failed; that's fine. The
        // cursor is at whatever it advanced to. Reset returns it to 0.
        arena.reset();
        prop_assert_eq!(arena.allocated(), 0, "reset did not zero the cursor");
        // Subsequent fresh allocation must succeed for any small layout.
        let small = NonZeroLayout::from_size_align(1, 1).unwrap();
        prop_assert!(arena.allocate(small).is_ok(), "post-reset alloc failed");
    }

    #[test]
    fn bump_arena_allocated_plus_remaining_equals_capacity(
        layouts in arb_layout_seq(),
    ) {
        let arena = BumpArena::new(InlineBacked::<8192>::new()).unwrap();
        for layout in layouts {
            let _ = arena.allocate(layout);
        }
        prop_assert_eq!(arena.allocated() + arena.remaining(), arena.capacity());
    }
}

// ============================================================================
// Slab<u64, InlineBacked<8192>, NoProtection>
// ============================================================================

#[derive(Debug, Clone, Copy)]
enum SlabOp {
    Alloc,
    Free(usize), // index into "live allocations" Vec
}

fn arb_slab_op_seq(max_live: usize) -> impl Strategy<Value = Vec<SlabOp>> {
    let op = prop_oneof![Just(SlabOp::Alloc), (0..max_live).prop_map(SlabOp::Free),];
    proptest::collection::vec(op, 0..64)
}

proptest! {
    #[test]
    fn slab_alloc_dealloc_no_overlap_no_leak(ops in arb_slab_op_seq(64)) {
        let s: Slab<u64, InlineBacked<8192>, NoProtection> =
            Slab::new(64, InlineBacked::<8192>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let mut live: Vec<core::ptr::NonNull<u8>> = Vec::new();

        for op in ops {
            match op {
                SlabOp::Alloc => {
                    if let Ok(block) = s.allocate(layout) {
                        let p = block.cast::<u8>();
                        // No overlap with any other live ptr.
                        for q in &live {
                            prop_assert_ne!(p.as_ptr(), q.as_ptr());
                        }
                        live.push(p);
                    }
                }
                SlabOp::Free(idx) => {
                    if idx < live.len() {
                        let p = live.swap_remove(idx);
                        unsafe { s.deallocate(p, layout) };
                    }
                }
            }
        }
    }

    #[test]
    fn slab_lifo_reuse(n in 1usize..32) {
        let s: Slab<u64, InlineBacked<8192>, NoProtection> =
            Slab::new(64, InlineBacked::<8192>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let mut ptrs = Vec::new();
        for _ in 0..n {
            ptrs.push(s.allocate(layout).unwrap());
        }
        // Free in reverse order.
        for p in ptrs.iter().rev() {
            unsafe { s.deallocate(p.cast(), layout) };
        }
        // Re-allocate — LIFO order means we get them back in original order.
        for p in ptrs.iter() {
            let q = s.allocate(layout).unwrap();
            prop_assert_eq!(p.cast::<u8>().as_ptr(), q.cast::<u8>().as_ptr());
        }
    }

    #[test]
    fn slab_capacity_never_exceeded(n in 1usize..200) {
        // Slab capacity = 64; never hand out more than 64 distinct slots
        // even under heavy alloc.
        let s: Slab<u64, InlineBacked<8192>, NoProtection> =
            Slab::new(64, InlineBacked::<8192>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let mut count = 0;
        for _ in 0..n {
            if s.allocate(layout).is_ok() {
                count += 1;
            }
        }
        prop_assert!(count <= 64);
    }
}

// ============================================================================
// StackAlloc<InlineBacked<8192>>
// ============================================================================

proptest! {
    #[test]
    fn stack_alloc_lifo_nested_frees_clean_state(n in 1usize..=16) {
        use forge_alloc::StackAlloc;
        let s = StackAlloc::new(InlineBacked::<8192>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let mut ptrs = Vec::new();
        for _ in 0..n {
            ptrs.push(s.allocate(layout).unwrap());
        }
        // LIFO free: most-recent first.
        for p in ptrs.iter().rev() {
            unsafe { s.deallocate(p.cast(), layout) };
        }
        // After LIFO drain, cursor must be back at 0.
        prop_assert_eq!(s.allocated(), 0);
        // Subsequent alloc must succeed and return the original first slot.
        let first = ptrs.first().unwrap().cast::<u8>().as_ptr() as usize;
        let p2 = s.allocate(layout).unwrap().cast::<u8>().as_ptr() as usize;
        prop_assert_eq!(p2, first);
    }

    #[test]
    fn stack_alloc_no_overlap(layouts in arb_layout_seq()) {
        use forge_alloc::StackAlloc;
        let s = StackAlloc::new(InlineBacked::<8192>::new()).unwrap();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for layout in layouts {
            if let Ok(block) = s.allocate(layout) {
                let start = block.cast::<u8>().as_ptr() as usize;
                let end = start + block.len();
                for (s_, e_) in &ranges {
                    let no_overlap = end <= *s_ || start >= *e_;
                    prop_assert!(no_overlap, "stack-alloc overlap detected");
                }
                ranges.push((start, end));
            }
        }
    }
}

// ============================================================================
// ExtendableSlab<u64, NoProtection> (std-only)
// ============================================================================

#[cfg(feature = "std")]
mod extendable_slab_tests {
    use super::*;
    use forge_alloc::ExtendableSlab;

    proptest! {
        #[test]
        fn extendable_slab_grows_under_pressure(n in 1usize..=200) {
            let s: ExtendableSlab<u64, NoProtection> = ExtendableSlab::new(8);
            let layout = NonZeroLayout::for_type::<u64>().unwrap();
            let mut ptrs = Vec::new();
            for _ in 0..n {
                ptrs.push(s.allocate(layout).unwrap());
            }
            // All allocs distinct.
            let mut addrs: Vec<_> = ptrs
                .iter()
                .map(|p| p.cast::<u8>().as_ptr() as usize)
                .collect();
            addrs.sort();
            for w in addrs.windows(2) {
                prop_assert_ne!(w[0], w[1]);
            }
            for p in ptrs {
                unsafe { s.deallocate(p.cast(), layout) };
            }
        }
    }
}

// ============================================================================
// SizeClassed<BumpArena<MmapBacked>, 4>
// ============================================================================

#[cfg(feature = "std")]
mod size_classed_tests {
    use super::*;
    use forge_alloc::MmapBacked;
    use forge_alloc::{BumpArena, SizeClassed};

    /// Build a SizeClassed with classes [8, 16, 32, 64] and 32 slots
    /// each. Total class storage = (8+16+32+64)*32 = 3840 bytes plus
    /// alignment slack; 16 KiB backing has plenty of room and a
    /// fallback budget.
    fn build_sc() -> SizeClassed<BumpArena<MmapBacked>, 4> {
        SizeClassed::with_class_sizes(
            BumpArena::new(MmapBacked::new(16 * 1024).unwrap()).unwrap(),
            [8, 16, 32, 64],
            32,
        )
        .unwrap()
    }

    /// Layouts whose size + align combinations exercise multiple
    /// classes and the fallback path.
    fn arb_sc_layout() -> impl Strategy<Value = NonZeroLayout> {
        (1usize..=200, 0u32..=4).prop_map(|(size, align_log)| {
            let align = 1usize << align_log;
            NonZeroLayout::from_size_align(size, align).unwrap()
        })
    }

    proptest! {
        #[test]
        fn size_classed_alignment_respected(layouts in
            proptest::collection::vec(arb_sc_layout(), 0..32))
        {
            let sc = build_sc();
            let mut live: Vec<(NonZeroLayout, *mut u8)> = Vec::new();
            for layout in layouts {
                if let Ok(block) = sc.allocate(layout) {
                    let p = block.cast::<u8>().as_ptr();
                    prop_assert_eq!(
                        p as usize % layout.align().get(),
                        0,
                        "ptr {:?} not aligned to {}",
                        p,
                        layout.align().get(),
                    );
                    prop_assert!(
                        block.len() >= layout.size().get(),
                        "block too small for layout",
                    );
                    live.push((layout, p));
                }
            }
            for (layout, p) in live {
                unsafe {
                    sc.deallocate(core::ptr::NonNull::new_unchecked(p), layout)
                };
            }
        }

        #[test]
        fn size_classed_pick_picks_smallest_fitting_class(
            size in 1usize..=64,
            align_log in 0u32..=3,
        ) {
            let sc = build_sc();
            let align = 1usize << align_log;
            let layout = NonZeroLayout::from_size_align(size, align).unwrap();
            let block = sc.allocate(layout).unwrap();
            let stride = block.len();
            // Expected class: smallest c in [8, 16, 32, 64] s.t.
            //   c >= size AND c >= align.
            let needed = core::cmp::max(size, align);
            let expected = [8, 16, 32, 64]
                .iter()
                .copied()
                .find(|&c| c >= needed)
                .expect("test inputs always fit");
            prop_assert_eq!(stride, expected,
                "size={} align={} → expected stride {}, got {}",
                size, align, expected, stride);
            unsafe {
                sc.deallocate(
                    block.cast(),
                    layout,
                )
            };
        }

        #[test]
        fn size_classed_freelist_recycles_within_class(
            iters in 1usize..32,
        ) {
            let sc = build_sc();
            let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
            let first = sc.allocate(layout).unwrap().cast::<u8>().as_ptr() as usize;
            unsafe { sc.deallocate(
                core::ptr::NonNull::new_unchecked(first as *mut u8),
                layout,
            ) };
            for _ in 0..iters {
                let p = sc.allocate(layout).unwrap().cast::<u8>().as_ptr() as usize;
                prop_assert_eq!(p, first,
                    "freed slot must be reused on next alloc");
                unsafe { sc.deallocate(
                    core::ptr::NonNull::new_unchecked(p as *mut u8),
                    layout,
                ) };
            }
        }
    }
}

// ============================================================================
// WithFallback<InlineBacked<256>, forge_alloc::System>
// ============================================================================

#[cfg(feature = "std")]
mod with_fallback_tests {
    use super::*;
    use forge_alloc::FixedRange;
    use forge_alloc::System;
    use forge_alloc::WithFallback;

    proptest! {
        #[test]
        fn fallback_serves_when_primary_exhausted(
            primary_drain in 1usize..=256,
            tail_alloc in 1usize..=128,
        ) {
            let wf = WithFallback::new(InlineBacked::<256>::new(), System);
            let prim = NonZeroLayout::from_size_align(primary_drain, 1).unwrap();
            let _ = wf.allocate(prim).unwrap();
            // After primary is partially or fully consumed, this tail_alloc
            // must succeed (either from remaining primary, or from secondary).
            let tail = NonZeroLayout::from_size_align(tail_alloc, 1).unwrap();
            let block = wf.allocate(tail).unwrap();
            // Clean up any secondary allocation.
            let ptr = block.cast::<u8>();
            if !wf.primary().contains(ptr) {
                unsafe { forge_alloc::Deallocator::deallocate(&wf, ptr, tail) };
            }
        }
    }
}
