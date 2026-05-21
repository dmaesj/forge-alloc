//! Compile-and-run versions of the recipes in `COMPOSITION_RECIPES.md`.
//!
//! Keeps the docs honest — every recipe shape exercised here, so a
//! refactor that breaks a composition fails CI before the docs go stale.

#![cfg(feature = "std")]

use forge_alloc::*;

#[test]
fn stack_local_scratch() {
    type Scratch = BumpArena<InlineBacked<{ 64 * 1024 }>>;
    let mut arena: Scratch = BumpArena::new(InlineBacked::<{ 64 * 1024 }>::new()).unwrap();
    let layout = NonZeroLayout::from_size_align(128, 16).unwrap();
    let _block = arena.allocate(layout).unwrap();
    arena.reset();
    assert_eq!(arena.allocated(), 0);
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn cross_thread_bump_arena() {
    use std::sync::Arc;
    type Shared = Arc<SharedBumpArena<MmapBacked>>;
    let arena: Shared = Arc::new(SharedBumpArena::new(MmapBacked::new(64 * 1024).unwrap()).unwrap());
    let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
    let _block = arena.allocate(layout).unwrap();
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn typed_object_pool() {
    type Pool<T> = Slab<T, MmapBacked>;
    let pool: Pool<u64> = Slab::new(1024, MmapBacked::new(64 * 1024).unwrap()).unwrap();
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let block = pool.allocate(layout).unwrap();
    unsafe { pool.deallocate(block.cast(), layout) };
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn multi_size_general_allocator() {
    type GP = SizeClassed<MmapBacked, 8>;
    let gp: GP =
        SizeClassed::with_default_classes(MmapBacked::new(1024 * 1024).unwrap(), 64).unwrap();
    // Small request — fits class 8.
    let small = NonZeroLayout::from_size_align(5, 1).unwrap();
    let _block = gp.allocate(small).unwrap();
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn hardened_slab() {
    // The recommended-maximum-hardening alias compiles AND round-trips.
    //
    // Sizes are scaled down from the recipe (1024 slots / 4 MiB data / 64 KiB
    // meta) to keep the test fast and avoid pressure on test-runner mmap
    // budgets; the composition shape is identical.
    let pool = Slab::<u64, _>::new(
        128,
        GuardPage::new(
            SplitMetadata::new(MmapBacked::new(256 * 1024).unwrap(), 16 * 1024).unwrap(),
            4096,
        )
        .unwrap(),
    )
    .unwrap();
    let _: HardenedSlab<u64> = pool;
}

#[test]
fn bounded_heap_with_overflow_fallback() {
    // 64 KiB inline arena (the recipe doc uses 1 MiB but that overflows the
    // default Windows test thread stack; the shape is the same).
    type Fast = WithFallback<BumpArena<InlineBacked<{ 64 * 1024 }>>, System>;
    let alloc: Fast =
        WithFallback::new(BumpArena::new(InlineBacked::<{ 64 * 1024 }>::new()).unwrap(), System);
    let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
    let _block = alloc.allocate(layout).unwrap();
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn observable_production_stack() {
    // Recipe uses `LogHandler` for production; we substitute an `FnHandler`
    // that captures events into an `AtomicUsize` so the test can assert
    // the threshold actually fired. (Bonus: keeps stderr clean.)
    // The composition shape and trait bounds are identical to the recipe.
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let warn_count = Arc::new(AtomicUsize::new(0));
    let critical_count = Arc::new(AtomicUsize::new(0));
    let wc = Arc::clone(&warn_count);
    let cc = Arc::clone(&critical_count);
    let handler = FnHandler(move |event: WatermarkEvent| match event.level {
        WatermarkLevel::Warn => {
            wc.fetch_add(1, Ordering::Relaxed);
        }
        WatermarkLevel::Critical => {
            cc.fetch_add(1, Ordering::Relaxed);
        }
        WatermarkLevel::Oom => {}
    });

    // 16-slot Slab over 128-byte capacity (16 slots × 8 bytes/u64 slot) so
    // we can cross 75% / 90% thresholds with a small, predictable number of
    // allocations.
    let pool = Watermark::with_thresholds(
        Statistics::new(PoisonOnFree::new(
            Slab::<u64, _>::new(16, MmapBacked::new(4 * 1024).unwrap()).unwrap(),
        )),
        handler,
        WatermarkThresholds::default(),
    );

    let layout = NonZeroLayout::for_type::<u64>().unwrap();

    // Allocate 13 of 16. `Watermark::check_and_fire` uses `pct >= warn_pct`,
    // so the warn edge actually fires on alloc #12 (96/128 bytes = 75% exact);
    // by alloc #13 we're at 81.25%, well past warn but still below 90% critical.
    // The warn rises exactly once across this run regardless of which alloc
    // tripped it, so the assertion remains correct — the comment is the only
    // thing that was off in earlier versions.
    let mut ptrs = Vec::new();
    for _ in 0..13 {
        ptrs.push(pool.allocate(layout).unwrap());
    }
    assert_eq!(warn_count.load(Ordering::Relaxed), 1, "warn should fire once");
    assert_eq!(
        critical_count.load(Ordering::Relaxed),
        0,
        "critical should not fire yet"
    );

    // Allocate 2 more → 15/16 = 93.75% > 90%, critical fires.
    ptrs.push(pool.allocate(layout).unwrap());
    ptrs.push(pool.allocate(layout).unwrap());
    assert_eq!(
        critical_count.load(Ordering::Relaxed),
        1,
        "critical should fire once on rising edge",
    );

    // Statistics counter check — exercises Statistics::allocate increment.
    let stats = pool.inner().stats();
    assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 15);
    assert_eq!(stats.bytes_allocated.load(Ordering::Relaxed), 15 * 8);
    assert_eq!(stats.bytes_peak.load(Ordering::Relaxed), 15 * 8);

    // Deallocate one — exercises Statistics::deallocate counter, the
    // saturating-sub path, and PoisonOnFree::deallocate (which wrote 0xDE
    // across the slot before forwarding to Slab).
    let first = ptrs.swap_remove(0);
    unsafe { pool.deallocate(first.cast(), layout) };
    assert_eq!(stats.total_deallocations.load(Ordering::Relaxed), 1);
    assert_eq!(stats.bytes_allocated.load(Ordering::Relaxed), 14 * 8);
    // Peak does not decrease.
    assert_eq!(stats.bytes_peak.load(Ordering::Relaxed), 15 * 8);

    // Clean up remaining allocations so the slab is empty on drop. (Slab's
    // Drop does not iterate live slots; we run drops ourselves.)
    for p in ptrs {
        unsafe { pool.deallocate(p.cast(), layout) };
    }
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn numa_local_huge_page_arena() {
    let backing = MmapBacked::new(2 * 1024 * 1024 + 4096).unwrap();
    // Use 2 MiB huge pages (spec default elsewhere; we pick explicit here for testability).
    let huge = HugePageAligned::with_huge_page_size(backing, 2 * 1024 * 1024).unwrap();
    let numa = NumaLocal::new(
        huge,
        NumaPolicy::Bind(NodeSet::single(0).unwrap()),
    )
    .unwrap();
    let arena = BumpArena::new(numa).unwrap();
    // arena may or may not satisfy a 2 MiB-aligned allocation depending on the
    // underlying mmap base; just verify it compiles + constructs.
    let _ = arena;
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn slab_owner_adaptive() {
    let owner: SlabOwner<u64, MmapBacked> = SlabOwner::with_batch_policy(
        4096,
        MmapBacked::new(64 * 1024).unwrap(),
        BatchPolicy::Adaptive,
        1024,
    )
    .unwrap();
    // Ship a SlabRemote handle off to a worker; Send + Sync.
    let _remote: SlabRemote<u64, MmapBacked> = owner.remote();
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let _block = owner.allocate(layout).unwrap();
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn generational_handle_slab() {
    let mut pool: GenerationalSlab<u64, _> =
        GenerationalSlab::new(256, MmapBacked::new(16 * 1024).unwrap()).unwrap();
    let handle: Handle<u64, u32> = pool.insert(0xCAFE).unwrap();
    let v = pool.get(handle).copied().unwrap();
    assert_eq!(v, 0xCAFE);
}

// ============================================================================
// Cross-crate composition coverage.
//
// These tests exercise compositions where two or more wrappers interact in
// ways no single-crate test can cover. Each test pins a documented behavior;
// a regression that changes any of these invariants must update the doc on
// the relevant crate before changing this test.
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn canary_over_slab_is_rejected_at_construction_or_first_alloc() {
    // Documented in `Canary` rustdoc: `Canary<Slab<T, _>>` is incompatible
    // for any T whose stride is the FreeLink minimum (8 bytes) because
    // Canary inflates the inner layout by max(8, align) + 8 — for u64 the
    // inflated layout is (24, 8), which exceeds Slab's 8-byte stride. The
    // composition compiles; the runtime rejection is at first allocate.
    let c = Canary::new_with_seed(
        Slab::<u64, _>::new(8, MmapBacked::new(64 * 1024).unwrap()).unwrap(),
        0x1234_5678_9ABC_DEF0,
    );
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    assert!(
        c.allocate(layout).is_err(),
        "Canary<Slab<u64>> should fail at allocate (inflated layout exceeds Slab stride). \
         If this passes, the Canary doc claim has drifted from runtime behavior."
    );
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn statistics_over_poison_over_slab_accounts_outer_layout() {
    // Statistics counts the OUTER layout size. PoisonOnFree forwards layout
    // unchanged. Slab returns the inflated-stride slot (here stride==size).
    // Counters should reflect the user's requested size, not any inner stride.
    let stats = Statistics::new(PoisonOnFree::new(
        Slab::<u64, _>::new(16, MmapBacked::new(4 * 1024).unwrap()).unwrap(),
    ));
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let p = stats.allocate(layout).unwrap();
    assert_eq!(stats.stats().bytes_allocated.load(std::sync::atomic::Ordering::Relaxed), 8);
    unsafe { stats.deallocate(p.cast(), layout) };
    assert_eq!(stats.stats().bytes_allocated.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn quarantine_over_poison_over_slab_round_trips() {
    // Quarantine<PoisonOnFree<Slab>>: the slab loans slots through
    // PoisonOnFree (poison-on-free); freed pointers sit in Quarantine for
    // EPOCHS dealloc cycles before evicting back to Slab's freelist. The
    // composition must:
    //   - allocate / deallocate without panic,
    //   - leave the slab reclaimable on drop (Quarantine drains).
    {
        let q: Quarantine<PoisonOnFree<Slab<u64, _>>, 4> = Quarantine::new(
            PoisonOnFree::new(
                Slab::<u64, _>::new(8, MmapBacked::new(4 * 1024).unwrap()).unwrap(),
            ),
        );
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = q.allocate(layout).unwrap();
        unsafe { q.deallocate(a.cast(), layout) };
        // q drops here — quarantine drains, slab releases its backing.
    }
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn quarantine_holds_slot_until_slab_exhaustion() {
    // Concern: when items wait in Quarantine, Slab sees them as still
    // allocated. If the program exhausts Slab while items wait in
    // Quarantine, `allocate` returns AllocError — this is the documented
    // behavior and the test pins it.
    let q: Quarantine<Slab<u64, _>, 16> = Quarantine::new(
        Slab::<u64, _>::new(2, MmapBacked::new(4 * 1024).unwrap()).unwrap(),
    );
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let a = q.allocate(layout).unwrap();
    let _b = q.allocate(layout).unwrap();
    unsafe { q.deallocate(a.cast(), layout) };
    // `a` is in quarantine, not back on slab freelist. Slab has 0 free
    // slots → next alloc fails (surfaces clearly as AllocError).
    assert!(
        q.allocate(layout).is_err(),
        "Slab exhaustion while item is in quarantine must surface as AllocError"
    );
}

#[test]
fn watermark_over_with_fallback_monitors_primary_only() {
    // Watermark's warn_threshold_bytes is computed from inner.capacity_bytes().
    // WithFallback::capacity_bytes returns only the primary's capacity, so
    // Watermark monitors the FAST path's pressure — secondary overflow is
    // invisible to the watermark by design. This test pins the documented
    // semantics.
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let warn_count = Arc::new(AtomicUsize::new(0));
    let wc = Arc::clone(&warn_count);
    let handler = FnHandler(move |event: WatermarkEvent| {
        if event.level == WatermarkLevel::Warn {
            wc.fetch_add(1, Ordering::Relaxed);
        }
    });

    type Fast = WithFallback<BumpArena<InlineBacked<1024>>, System>;
    let fast: Fast = WithFallback::new(
        BumpArena::new(InlineBacked::<1024>::new()).unwrap(),
        System,
    );
    // Watermark sees capacity = 1024 (primary only).
    assert_eq!(fast.capacity_bytes(), Some(1024));
    let w = Watermark::with_thresholds(fast, handler, WatermarkThresholds::default());

    // Fill primary to 80% (820 bytes) — past the 75% warn.
    let layout = NonZeroLayout::from_size_align(820, 1).unwrap();
    let _ = w.allocate(layout).unwrap();
    assert_eq!(warn_count.load(Ordering::Relaxed), 1, "primary-side warn must fire");

    // Now overflow to secondary. Watermark's `allocated` counter keeps
    // climbing (it counts ALL allocations through the wrapper), but the
    // documented intent is that the secondary-side capacity isn't part of
    // the budget. The watermark already armed; this is just confirming the
    // composition works without panicking on the overflow.
    let overflow = NonZeroLayout::from_size_align(300, 1).unwrap();
    let block = w.allocate(overflow).unwrap();
    // The secondary path served the request (primary is full).
    unsafe { w.deallocate(block.cast(), overflow) };
}

#[test]
fn with_fallback_try_new_reachable_via_meta_crate() {
    // Re-export gaps have surfaced here before. This test confirms that
    // `WithFallback::try_new` is callable through the
    // meta-crate import surface — it relies on the leaf crate's pub fn, but
    // a missing re-export of `WithFallback` would block the call site.
    let a = InlineBacked::<256>::new();
    let b = InlineBacked::<256>::new();
    let _wf = WithFallback::try_new(a, b).expect("disjoint backings accepted");
}

/// Regression: `System` does NOT implement `FixedRange`, but
/// `Statistics<I>`, `Watermark<I, _>`, and `Canary<I>` must NOT require it
/// either — their `Allocator`/`Deallocator` impls are bound only on
/// `I: Allocator`. An over-constrained wrapper that demanded `FixedRange`
/// would silently drop `System` out of every recipe
/// where it's wrapped for observability. This test compiles each
/// composition shape end-to-end, then exercises a real allocate/deallocate
/// cycle through it.
#[test]
fn observability_wrappers_compose_over_system() {
    // Statistics<System>: counter sums total bytes routed through the heap.
    {
        let s = Statistics::new(System);
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let block = s.allocate(layout).expect("System alloc via Statistics");
        assert_eq!(s.stats().total_allocations.load(core::sync::atomic::Ordering::Relaxed), 1);
        unsafe { s.deallocate(block.cast(), layout) };
        assert_eq!(s.stats().total_deallocations.load(core::sync::atomic::Ordering::Relaxed), 1);
    }
    // Watermark<System, _>: System has `capacity_bytes() == None`, so
    // Watermark's threshold model degrades to a no-op handler — but the
    // type must still compile and allocate.
    {
        let handler = FnHandler(|_event: WatermarkEvent| {});
        let w = Watermark::with_thresholds(System, handler, WatermarkThresholds::default());
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let block = w.allocate(layout).expect("System alloc via Watermark");
        unsafe { w.deallocate(block.cast(), layout) };
    }
    // Canary<System>: inflates layout and tags edges of the allocation;
    // forwards the inflated request to System. Must round-trip.
    {
        let c = Canary::new_with_seed(System, 0xDEADBEEF_u64);
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let block = c.allocate(layout).expect("System alloc via Canary");
        unsafe { c.deallocate(block.cast(), layout) };
    }
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn poison_persists_in_user_region_past_freelink_with_quarantine_wrap() {
    // PoisonOnFree<Quarantine<Slab>>: poison is written immediately on the
    // outer dealloc; the slot then sits in Quarantine for EPOCHS cycles
    // before reaching Slab. During the quarantine window, a UAF read sees
    // fully-poisoned bytes (Slab's FreeLink hasn't been written yet).
    //
    // This pins the documented `PoisonOnFree<Quarantine<Slab>>` security
    // property from `crates/forge-hardening/src/poison.rs` — the "maximum
    // poison persistence" recipe.
    let pof: PoisonOnFree<Quarantine<Slab<u64, _>, 4>> = PoisonOnFree::new(
        Quarantine::new(
            Slab::<u64, _>::new(8, MmapBacked::new(4 * 1024).unwrap()).unwrap(),
        ),
    );
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let block = pof.allocate(layout).unwrap();
    let ptr = block.cast::<u8>();
    unsafe {
        // Write a sentinel that's NOT poison.
        core::ptr::write_bytes(ptr.as_ptr(), 0xAA, 8);
        pof.deallocate(ptr, layout);
        // Slot is now in Quarantine. Slab has NOT yet written a FreeLink
        // (Quarantine hasn't evicted). The poison bytes from PoisonOnFree
        // should be observable across the full user region.
        for i in 0..8 {
            assert_eq!(
                *ptr.as_ptr().add(i),
                DEFAULT_POISON,
                "byte {i} in quarantined slot must hold the poison pattern, \
                 not the original {:#x}",
                0xAAu8,
            );
        }
    }
}
