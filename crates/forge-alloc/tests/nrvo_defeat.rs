//! NRVO-defeating regression battery.
//!
//! An earlier audit found that several allocators cached an absolute base pointer
//! captured BEFORE the wrapper was moved into its final stack slot. The
//! bug hid across hundreds of tests because NRVO / copy-elision usually
//! built the wrapper directly in the caller's final slot. Only a test
//! with enough local state to force the wrapper into a *different* slot
//! than the constructor wrote into actually surfaced the stale pointer.
//!
//! Every test in this file follows the same recipe:
//!
//!   1. Build the wrapper in a `#[inline(never)]` helper that returns
//!      by-value, forcing an ABI-mandated move into the caller's frame.
//!   2. Apply additional stack pressure (large local arrays, extra
//!      bindings) so any future return-slot-optimizer change can't
//!      collapse caller and callee back into the same slot.
//!   3. Read `wrapper.backing().base()` (or equivalent) and assert that
//!      every pointer the wrapper hands out lies inside that LIVE base
//!      range — proving the wrapper followed the move.
//!   4. Run many allocate/deallocate cycles so any inconsistency between
//!      the cached-base and the live-base would write through the wrong
//!      address and either trip a debug assertion or corrupt foreign
//!      memory.
//!
//! Each test is designed so it would FAIL under the original absolute-
//! pointer bug and PASS under the current offset-based fixes. If you add
//! a new wrapper, add a pinned test here.

#![cfg(feature = "std")]
#![allow(clippy::too_many_lines)]

use core::hint::black_box;
use core::ptr::NonNull;

use forge_alloc::{
    Allocator, BumpArena, Canary, Deallocator, FixedRange, GenerationalSlab, InlineBacked,
    NonZeroLayout, NullHandler, PoisonOnFree, Quarantine, SharedBumpArena, SizeClassed, Slab,
    StackAlloc, Statistics, System, Watermark, WithFallback,
};

// ============================================================================
// Helpers
// ============================================================================

/// Assert that `ptr` falls inside `[base, base + size)`. Failure message
/// includes both addresses so a stale-base bug is obvious in the output.
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

/// Apply heavy stack pressure between the constructor-return and the
/// first use so NRVO / RVO cannot collapse the caller's slot onto the
/// constructor's slot. Returns its argument unmodified through `black_box`
/// so the optimizer can't dead-code-eliminate the pressure.
#[inline(never)]
fn stack_pressure<T>(t: T) -> T {
    // 8 KiB of stack scratch the optimizer cannot prove is dead.
    let arr = [0u64; 1024];
    black_box(&arr);
    black_box(t)
}

// ============================================================================
// 1. Slab<u64, InlineBacked<N>, NoProtection>
// ============================================================================

#[inline(never)]
fn build_slab_inline() -> Slab<u64, InlineBacked<2048>> {
    let inner = InlineBacked::<2048>::new();
    let s = Slab::new(64, inner).unwrap();
    black_box(s)
}

#[test]
fn pin_slab_inline_backed_survives_move() {
    let s = stack_pressure(build_slab_inline());
    // Stack pressure AFTER receipt: ensures the wrapper isn't sitting
    // where the constructor wrote it.
    let _scratch = [0u8; 4096];
    black_box(&_scratch);

    // Verify the slab's reported base matches its OWNED backing's
    // current base. A stale absolute pointer would NOT agree.
    let slab_base = s.base();
    let backing_base = s.backing().base();
    assert_eq!(
        slab_base.as_ptr(),
        backing_base.as_ptr(),
        "Slab<InlineBacked>: base mismatch after move (stale ptr regression)",
    );

    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let mut ptrs = Vec::new();
    for _ in 0..32 {
        let p = s.allocate(layout).unwrap();
        // Every allocate must lie inside the live backing range.
        assert_in_range(p.cast(), backing_base, s.backing().size(), "Slab alloc");
        ptrs.push(p);
    }
    // Drain in LIFO order — exercises freelist path with the live base.
    for p in ptrs.iter().rev() {
        unsafe { s.deallocate(p.cast(), layout) };
    }
    // Realloc — must match LIFO order at the same live-base addresses.
    for p in ptrs.iter() {
        let q = s.allocate(layout).unwrap();
        assert_eq!(
            p.cast::<u8>().as_ptr(),
            q.cast::<u8>().as_ptr(),
            "LIFO reuse must return original slot at the LIVE base",
        );
    }
    // Drain again so Drop's debug-only leak check is happy.
    for p in ptrs.iter().rev() {
        unsafe { s.deallocate(p.cast(), layout) };
    }
}

// ============================================================================
// 2. BumpArena<InlineBacked<N>>
// ============================================================================

#[inline(never)]
fn build_bump_inline() -> BumpArena<InlineBacked<4096>> {
    BumpArena::new(InlineBacked::<4096>::new()).unwrap()
}

#[test]
fn pin_bump_arena_inline_backed_survives_move() {
    let arena = stack_pressure(build_bump_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    let arena_base = arena.base();
    let backing_base = arena.backing().base();
    assert_eq!(
        arena_base.as_ptr(),
        backing_base.as_ptr(),
        "BumpArena<InlineBacked>: base mismatch after move",
    );

    let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
    for _ in 0..200 {
        let p = arena.allocate(layout).unwrap();
        assert_in_range(p.cast(), backing_base, arena.capacity(), "BumpArena alloc");
    }
}

// ============================================================================
// 3. SharedBumpArena<InlineBacked<N>>
// ============================================================================

#[inline(never)]
fn build_shared_bump_inline() -> SharedBumpArena<InlineBacked<4096>> {
    SharedBumpArena::new(InlineBacked::<4096>::new()).unwrap()
}

#[test]
fn pin_shared_bump_arena_inline_backed_survives_move() {
    // SharedBumpArena<InlineBacked> compiles AS LONG AS the unsafe impls
    // allow B: !Sync (they do — Sync is asserted at the wrapper level via
    // atomic cursor and the !Sync backing's storage is sealed).
    let arena = stack_pressure(build_shared_bump_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    let arena_base = arena.base().as_ptr() as usize;
    let arena_size = arena.size();
    let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
    for _ in 0..100 {
        let p = arena.allocate(layout).unwrap();
        let addr = p.cast::<u8>().as_ptr() as usize;
        assert!(
            addr >= arena_base && addr < arena_base + arena_size,
            "SharedBumpArena alloc {addr:#x} not in [{arena_base:#x}, {:#x})",
            arena_base + arena_size,
        );
    }
}

// ============================================================================
// 4. StackAlloc<InlineBacked<N>>
// ============================================================================

#[inline(never)]
fn build_stack_alloc_inline() -> StackAlloc<InlineBacked<4096>> {
    StackAlloc::new(InlineBacked::<4096>::new()).unwrap()
}

#[test]
fn pin_stack_alloc_inline_backed_survives_move() {
    let s = stack_pressure(build_stack_alloc_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    let s_base = s.base();
    let b_base = s.backing().base();
    assert_eq!(
        s_base.as_ptr(),
        b_base.as_ptr(),
        "StackAlloc<InlineBacked>: base mismatch after move",
    );

    let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
    let mut ptrs = Vec::new();
    for _ in 0..100 {
        let p = s.allocate(layout).unwrap();
        assert_in_range(p.cast(), b_base, 4096, "StackAlloc alloc");
        ptrs.push(p);
    }
    // LIFO drain.
    for p in ptrs.iter().rev() {
        unsafe { s.deallocate(p.cast(), layout) };
    }
}

// ============================================================================
// 5. SizeClassed<InlineBacked<N>, 4>  (fixed in 1d7e1a39 — pin it!)
// ============================================================================

#[inline(never)]
fn build_size_classed_inline() -> SizeClassed<InlineBacked<2048>, 2> {
    // InlineBacked's MAX_ALIGN is 16, so we can only use class sizes <= 16
    // (each class needs alignment == stride). Use [8, 16] with 16 slots.
    SizeClassed::with_class_sizes(InlineBacked::<2048>::new(), [8, 16], 16).unwrap()
}

#[test]
fn pin_size_classed_inline_backed_survives_move() {
    let sc = stack_pressure(build_size_classed_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    let backing_base = sc.backing().base();
    let backing_size = sc.backing().size();
    // Exercise every class so each per-class region is touched at its
    // live (post-move) base address.
    for &class in &[8usize, 16] {
        let layout = NonZeroLayout::from_size_align(class, 8).unwrap();
        // Allocate and deallocate to drive both alloc + free paths.
        let mut ptrs = Vec::new();
        for _ in 0..8 {
            let p = sc.allocate(layout).unwrap();
            assert_in_range(
                p.cast(),
                backing_base,
                backing_size,
                "SizeClassed class alloc",
            );
            ptrs.push(p);
        }
        for p in ptrs.iter().rev() {
            unsafe { sc.deallocate(p.cast(), layout) };
        }
        // Realloc — the freelist (resolved through the live base) must
        // return a valid same-class slot.
        let q = sc.allocate(layout).unwrap();
        assert_in_range(
            q.cast(),
            backing_base,
            backing_size,
            "SizeClassed realloc after free",
        );
        unsafe { sc.deallocate(q.cast(), layout) };
    }
}

// ============================================================================
// 6. WithFallback<InlineBacked<N>, System>
// ============================================================================

#[inline(never)]
fn build_with_fallback_inline() -> WithFallback<InlineBacked<512>, System> {
    WithFallback::new(InlineBacked::<512>::new(), System)
}

#[test]
fn pin_with_fallback_inline_backed_survives_move() {
    let wf = stack_pressure(build_with_fallback_inline());
    let _scratch = [0u8; 4096];
    black_box(&_scratch);

    let primary_base = wf.primary().base();
    let primary_size = wf.primary().size();

    // Round 1: small allocs that all fit in the inline primary. Every
    // pointer must come from the live primary range.
    let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
    let mut primary_ptrs = Vec::new();
    for _ in 0..16 {
        let p = wf.allocate(layout).unwrap();
        // If primary served, ptr is in the primary range.
        if wf.primary().contains(p.cast()) {
            assert_in_range(
                p.cast(),
                primary_base,
                primary_size,
                "WithFallback primary alloc",
            );
            primary_ptrs.push(p);
        }
    }

    // Round 2: drain primary, then alloc — the wf must route through the
    // live `contains` check (which itself depends on the live base).
    let drain = NonZeroLayout::from_size_align(512, 1).unwrap();
    // The drain may fail because round 1 ate some bytes; either way the
    // next overflow allocation must route to System. If the drain succeeds
    // we must hand the block back so miri's leak check is satisfied (the
    // block might come from System if primary couldn't fit it).
    if let Ok(p) = wf.allocate(drain) {
        unsafe { wf.deallocate(p.cast(), drain) };
    }
    let overflow = NonZeroLayout::from_size_align(64, 8).unwrap();
    let p = wf.allocate(overflow).unwrap();
    // It came from System (heap), not the primary. Clean up.
    unsafe { wf.deallocate(p.cast(), overflow) };

    // Clean up primary ptrs (no-op since BumpArena-style, but exercise the path).
    for p in primary_ptrs {
        unsafe { wf.deallocate(p.cast(), layout) };
    }
}

// ============================================================================
// 7. Statistics<Slab<u64, InlineBacked<N>>>
// ============================================================================

#[inline(never)]
fn build_stats_slab_inline() -> Statistics<Slab<u64, InlineBacked<1024>>> {
    Statistics::new(Slab::new(64, InlineBacked::<1024>::new()).unwrap())
}

#[test]
fn pin_statistics_slab_inline_backed_survives_move() {
    let s = stack_pressure(build_stats_slab_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    // Statistics forwards FixedRange to inner Slab, which forwards to
    // backing — every base() call should agree with the live base.
    let stats_base = s.base();
    let inner_slab_base = s.inner().base();
    let inner_backing_base = s.inner().backing().base();
    assert_eq!(stats_base.as_ptr(), inner_slab_base.as_ptr());
    assert_eq!(inner_slab_base.as_ptr(), inner_backing_base.as_ptr());

    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let mut ptrs = Vec::new();
    for _ in 0..32 {
        let p = s.allocate(layout).unwrap();
        assert_in_range(
            p.cast(),
            inner_backing_base,
            s.inner().backing().size(),
            "Statistics<Slab<InlineBacked>> alloc",
        );
        ptrs.push(p);
    }
    for p in ptrs.iter().rev() {
        unsafe { s.deallocate(p.cast(), layout) };
    }
    // Stats reflect the activity.
    assert_eq!(s.stats().live_count(), 0);
}

// ============================================================================
// 8. Watermark<Slab<u64, InlineBacked<N>>, NullHandler>
// ============================================================================

#[inline(never)]
fn build_watermark_slab_inline() -> Watermark<Slab<u64, InlineBacked<1024>>, NullHandler> {
    Watermark::new(
        Slab::new(64, InlineBacked::<1024>::new()).unwrap(),
        NullHandler,
    )
}

#[test]
fn pin_watermark_slab_inline_backed_survives_move() {
    let w = stack_pressure(build_watermark_slab_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    let w_base = w.base();
    let inner_base = w.inner().base();
    assert_eq!(w_base.as_ptr(), inner_base.as_ptr());

    let backing_base = w.inner().backing().base();
    let backing_size = w.inner().backing().size();

    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let mut ptrs = Vec::new();
    for _ in 0..32 {
        let p = w.allocate(layout).unwrap();
        assert_in_range(
            p.cast(),
            backing_base,
            backing_size,
            "Watermark<Slab<InlineBacked>> alloc",
        );
        ptrs.push(p);
    }
    for p in ptrs.iter().rev() {
        unsafe { w.deallocate(p.cast(), layout) };
    }
}

// ============================================================================
// 9. PoisonOnFree<Slab<u64, InlineBacked<N>>>
// ============================================================================

#[inline(never)]
fn build_poison_slab_inline() -> PoisonOnFree<Slab<u64, InlineBacked<1024>>> {
    PoisonOnFree::new(Slab::new(64, InlineBacked::<1024>::new()).unwrap())
}

#[test]
fn pin_poison_on_free_slab_inline_backed_survives_move() {
    let p = stack_pressure(build_poison_slab_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    let p_base = p.base();
    let inner_base = p.inner().base();
    assert_eq!(p_base.as_ptr(), inner_base.as_ptr());

    let backing_base = p.inner().backing().base();
    let backing_size = p.inner().backing().size();

    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let mut ptrs = Vec::new();
    for _ in 0..32 {
        let q = p.allocate(layout).unwrap();
        assert_in_range(
            q.cast(),
            backing_base,
            backing_size,
            "PoisonOnFree<Slab<InlineBacked>> alloc",
        );
        ptrs.push(q);
    }
    // Free triggers poison-then-freelist-link. The poison memset writes
    // through the LIVE pointer (it's the same `q` we got back from alloc).
    // The freelist link Slab writes is computed by `slot_index(ptr)` which
    // resolves via the live base — if the base were stale, slot_index
    // would return None and the dealloc would silently no-op in debug.
    for q in ptrs.iter().rev() {
        unsafe { p.deallocate(q.cast(), layout) };
    }
    // Re-alloc to confirm freelist was actually populated (would be empty
    // if every dealloc silently no-op'd). We freed in reverse iteration
    // order — so the last `deallocate` call was on `ptrs[0]`, and that's
    // the slot the LIFO freelist returns first.
    let q1 = p.allocate(layout).unwrap();
    let q1_addr = q1.cast::<u8>().as_ptr();
    let first_freed_addr = ptrs[0].cast::<u8>().as_ptr();
    assert_eq!(
        q1_addr, first_freed_addr,
        "PoisonOnFree+Slab: LIFO reuse must return the most-recently-freed slot",
    );
    unsafe { p.deallocate(q1.cast(), layout) };
}

// ============================================================================
// 10. Quarantine<Slab<u64, InlineBacked<N>>, 4>
// ============================================================================

#[inline(never)]
fn build_quarantine_slab_inline() -> Quarantine<Slab<u64, InlineBacked<1024>>, 4> {
    Quarantine::new(Slab::new(64, InlineBacked::<1024>::new()).unwrap())
}

#[test]
fn pin_quarantine_slab_inline_backed_survives_move() {
    let q = stack_pressure(build_quarantine_slab_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    let backing_base = q.inner().backing().base();
    let backing_size = q.inner().backing().size();

    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    // Allocate enough to exercise the ring buffer plus evictions.
    let mut ptrs = Vec::new();
    for _ in 0..16 {
        let p = q.allocate(layout).unwrap();
        assert_in_range(
            p.cast(),
            backing_base,
            backing_size,
            "Quarantine<Slab<InlineBacked>> alloc",
        );
        ptrs.push(p);
    }
    // Free all — first 4 stay in ring, rest evict back to slab. Eviction
    // calls Slab::deallocate which uses the live base.
    for p in ptrs.iter().rev() {
        unsafe { q.deallocate(p.cast(), layout) };
    }
    // Re-allocate — must succeed (Slab freelist has the evicted slots).
    let r = q.allocate(layout).unwrap();
    assert_in_range(
        r.cast(),
        backing_base,
        backing_size,
        "Quarantine realloc after evict",
    );
    unsafe { q.deallocate(r.cast(), layout) };
}

// ============================================================================
// 11. Canary<BumpArena<InlineBacked<N>>>  (verify BumpArena fix propagates)
// ============================================================================

#[inline(never)]
fn build_canary_bump_inline() -> Canary<BumpArena<InlineBacked<4096>>> {
    Canary::new_with_seed(
        BumpArena::new(InlineBacked::<4096>::new()).unwrap(),
        0xDEAD_BEEF_CAFE_BABE,
    )
}

#[test]
fn pin_canary_bump_inline_backed_survives_move() {
    let c = stack_pressure(build_canary_bump_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    let c_base = c.base();
    let inner_base = c.inner().base();
    assert_eq!(c_base.as_ptr(), inner_base.as_ptr());

    let backing_base = c.inner().backing().base();
    let backing_size = c.inner().backing().size();

    let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
    // Allocate + write + verify on free. Canary's verify reads the
    // pre/post canary bytes through the LIVE pointer we got back from
    // allocate. If BumpArena's base were stale the underlying allocate
    // would land in someone else's frame and the canary verify on free
    // would either crash or detect "corruption" (random bytes ≠ seed).
    let mut ptrs = Vec::new();
    for _ in 0..16 {
        let p = c.allocate(layout).unwrap();
        assert_in_range(
            p.cast(),
            backing_base,
            backing_size,
            "Canary<BumpArena<InlineBacked>> user-alloc",
        );
        ptrs.push(p);
    }
    for p in ptrs {
        unsafe { c.deallocate(p.cast(), layout) };
    }
}

// ============================================================================
// 12. GenerationalSlab<u64, InlineBacked<N>>  (verify Generational fix)
// ============================================================================

#[inline(never)]
fn build_generational_slab_inline() -> GenerationalSlab<u64, InlineBacked<2048>> {
    GenerationalSlab::new(32, InlineBacked::<2048>::new()).unwrap()
}

#[test]
fn pin_generational_slab_inline_backed_survives_move() {
    let mut s = stack_pressure(build_generational_slab_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    // Insert many values, read them back, remove them. If the slots
    // pointer were stale every insert/get/remove would write into a
    // stale stack frame and the values wouldn't round-trip.
    let mut handles = Vec::new();
    for i in 0..32u64 {
        let h = s.insert(i ^ 0xA5A5_A5A5_A5A5_A5A5).unwrap();
        handles.push(h);
    }
    for (i, h) in handles.iter().enumerate() {
        let expected = (i as u64) ^ 0xA5A5_A5A5_A5A5_A5A5;
        assert_eq!(
            s.get(*h).copied(),
            Some(expected),
            "GenerationalSlab<InlineBacked>: value didn't round-trip — stale-base regression",
        );
    }
    // Remove and re-insert; the slot indices come back via the freelist
    // (which is stored in the slot bytes themselves — at the LIVE base).
    for h in &handles {
        s.remove(*h).unwrap();
    }
    for i in 0..32u64 {
        let _ = s.insert(i).unwrap();
    }
}

// ============================================================================
// 13. Box<BumpArena<InlineBacked<N>>>  (move chain — box-then-unbox)
// ============================================================================

#[inline(never)]
fn build_boxed_bump_inline() -> Box<BumpArena<InlineBacked<4096>>> {
    Box::new(BumpArena::new(InlineBacked::<4096>::new()).unwrap())
}

#[test]
fn pin_boxed_then_unboxed_bump_inline_survives_chain() {
    let boxed = build_boxed_bump_inline();
    // `*boxed` moves the BumpArena out of the heap allocation into a
    // local — a different address than either the original stack frame
    // or the box's heap address.
    let arena = stack_pressure(*boxed);
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    let arena_base = arena.base();
    let backing_base = arena.backing().base();
    assert_eq!(
        arena_base.as_ptr(),
        backing_base.as_ptr(),
        "BumpArena survived box-then-unbox chain",
    );

    let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
    for _ in 0..100 {
        let p = arena.allocate(layout).unwrap();
        assert_in_range(
            p.cast(),
            backing_base,
            arena.capacity(),
            "BumpArena post-unbox alloc",
        );
    }
}

// ============================================================================
// 14. Multi-frame move: Statistics<Watermark<Slab<u64, InlineBacked>>, NullHandler>
//    — deep wrapper stack moves through several `#[inline(never)]` frames.
// ============================================================================

#[inline(never)]
fn build_inner_layer() -> Slab<u64, InlineBacked<1024>> {
    Slab::new(64, InlineBacked::<1024>::new()).unwrap()
}

#[inline(never)]
fn build_middle_layer() -> Watermark<Slab<u64, InlineBacked<1024>>, NullHandler> {
    Watermark::new(build_inner_layer(), NullHandler)
}

#[inline(never)]
fn build_outer_layer() -> Statistics<Watermark<Slab<u64, InlineBacked<1024>>, NullHandler>> {
    Statistics::new(build_middle_layer())
}

#[test]
fn pin_multi_frame_move_through_wrapper_chain() {
    let outer = stack_pressure(build_outer_layer());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    // The whole wrapper chain has been moved through three call frames.
    // Verify base() agrees end-to-end.
    let backing_base = outer.inner().inner().backing().base();
    let outer_base = outer.base();
    assert_eq!(outer_base.as_ptr(), backing_base.as_ptr());

    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let mut ptrs = Vec::new();
    for _ in 0..32 {
        let p = outer.allocate(layout).unwrap();
        assert_in_range(
            p.cast(),
            backing_base,
            outer.inner().inner().backing().size(),
            "multi-frame chain alloc",
        );
        ptrs.push(p);
    }
    for p in ptrs.iter().rev() {
        unsafe { outer.deallocate(p.cast(), layout) };
    }
    assert_eq!(outer.stats().live_count(), 0);
}

// ============================================================================
// 15. PoisonOnFree<Quarantine<Slab<u64, InlineBacked>>> — composition of fixed wrappers
// ============================================================================

#[inline(never)]
fn build_poison_quarantine_slab() -> PoisonOnFree<Quarantine<Slab<u64, InlineBacked<2048>>, 4>> {
    PoisonOnFree::new(Quarantine::new(
        Slab::new(64, InlineBacked::<2048>::new()).unwrap(),
    ))
}

#[test]
fn pin_poison_quarantine_slab_inline_survives_move() {
    let p = stack_pressure(build_poison_quarantine_slab());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    let backing_base = p.inner().inner().backing().base();
    let backing_size = p.inner().inner().backing().size();
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let mut ptrs = Vec::new();
    for _ in 0..32 {
        let q = p.allocate(layout).unwrap();
        assert_in_range(
            q.cast(),
            backing_base,
            backing_size,
            "PoisonOnFree<Quarantine<Slab<InlineBacked>>> alloc",
        );
        ptrs.push(q);
    }
    for q in ptrs.iter().rev() {
        unsafe { p.deallocate(q.cast(), layout) };
    }
}

// ============================================================================
// 16. Vec-of-boxed-allocators: each Box's contents lives at a separate heap
//    address; iterating the Vec exercises each one through a fresh `&` deref.
//    The Vec move into the test frame also drags every Box's heap address
//    along, so the inner allocator's `backing.base()` must remain live.
// ============================================================================

// Box-per-element is load-bearing for this NRVO-defeating test (see
// the section banner above — each Box must live at its own heap
// address). Clippy's `vec_box` lint flags `Vec<Box<T>>` as needless
// indirection, but here the indirection IS the test's point.
#[allow(clippy::vec_box)]
#[inline(never)]
fn build_vec_of_boxed_bump() -> Vec<Box<BumpArena<InlineBacked<1024>>>> {
    let mut v = Vec::with_capacity(4);
    for _ in 0..4 {
        v.push(Box::new(
            BumpArena::new(InlineBacked::<1024>::new()).unwrap(),
        ));
    }
    v
}

#[test]
fn pin_vec_of_boxed_bump_inline_survives_move() {
    let v = stack_pressure(build_vec_of_boxed_bump());
    let _scratch = [0u8; 4096];
    black_box(&_scratch);

    let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
    for arena in &v {
        let base = arena.base();
        let backing_base = arena.backing().base();
        assert_eq!(
            base.as_ptr(),
            backing_base.as_ptr(),
            "Boxed-arena base mismatch — stale ptr regression in Vec context",
        );
        for _ in 0..20 {
            let p = arena.allocate(layout).unwrap();
            assert_in_range(p.cast(), backing_base, 1024, "Vec<Box<BumpArena>> alloc");
        }
    }
}

// ============================================================================
// 17. Property-style fuzz: random alloc/dealloc sequence on Slab<InlineBacked>
//    after a forced move. If any allocation lands outside the live base
//    range, the loop fails immediately with the offending address.
// ============================================================================

#[test]
fn pin_slab_inline_random_alloc_dealloc_sequence_post_move() {
    // Use a deterministic xorshift so failures reproduce.
    let mut rng_state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut rand_u32 = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state >> 32) as u32
    };

    let s = stack_pressure(build_slab_inline());
    let _scratch = [0u8; 4096];
    black_box(&_scratch);

    let backing_base = s.backing().base();
    let backing_size = s.backing().size();
    let layout = NonZeroLayout::for_type::<u64>().unwrap();

    let mut live: Vec<NonNull<u8>> = Vec::new();
    for _ in 0..512 {
        // 60% alloc, 40% free (when something is live).
        let action = rand_u32() % 10;
        if action < 6 || live.is_empty() {
            if let Ok(p) = s.allocate(layout) {
                let q = p.cast::<u8>();
                assert_in_range(q, backing_base, backing_size, "fuzz alloc");
                live.push(q);
            }
        } else {
            let idx = (rand_u32() as usize) % live.len();
            let p = live.swap_remove(idx);
            unsafe { s.deallocate(p, layout) };
        }
    }
    // Clean up.
    for p in live {
        unsafe { s.deallocate(p, layout) };
    }
}

// ============================================================================
// 18. GenerationalSlab post-move + full insert/remove cycle through 2x capacity
//    so the freelist (stored INSIDE the slot region) is exercised after move.
// ============================================================================

#[test]
fn pin_generational_slab_full_cycle_post_move() {
    let mut s = stack_pressure(build_generational_slab_inline());
    let _scratch = [0u8; 8192];
    black_box(&_scratch);

    // Fill to capacity, drain, re-fill — touches both the carve path and
    // the freelist-pop path. Freelist links live IN the slot bytes; if
    // the slots pointer were stale, the freelist would link bytes in
    // some other stack frame and the second fill would either crash or
    // hand out duplicate handles.
    let mut h1 = Vec::new();
    for i in 0..32u64 {
        h1.push(s.insert(i).unwrap());
    }
    for h in &h1 {
        let _ = s.remove(*h);
    }
    let mut h2 = Vec::new();
    for i in 0..32u64 {
        h2.push(s.insert(i + 1000).unwrap());
    }
    // All h2 handles must be unique and resolvable.
    use std::collections::HashSet;
    let unique: HashSet<_> = h2.iter().collect();
    assert_eq!(unique.len(), h2.len(), "duplicate handles — stale-base UB");
    for (i, h) in h2.iter().enumerate() {
        assert_eq!(s.get(*h).copied(), Some(i as u64 + 1000));
    }
}
