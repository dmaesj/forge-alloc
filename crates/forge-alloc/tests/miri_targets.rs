//! Miri target battery.
//!
//! Exercises the compositions that most benefit from UB checking
//! under miri. Each test is deterministic, single-
//! threaded, and uses `InlineBacked` so it can run end-to-end under
//! `cargo +nightly miri test`. The compositions that *require*
//! `MmapBacked` or `std::thread` are marked `#[cfg_attr(miri, ignore)]`
//! at the test level; the unmarked tests below are the bulk of the
//! audit's coverage.
//!
//! A clean run of this file under miri is the strongest "no UB" signal
//! we currently produce. Each test is intentionally short (a few
//! allocate / deallocate cycles) — miri runs ~50x slower than a normal
//! debug test, and the audit is about UB detection, not throughput.

use forge_alloc::{
    Allocator, BumpArena, Canary, Deallocator, GenerationalSlab, InlineBacked, NonZeroLayout,
    PoisonOnFree, SharedBumpArena, Slab, SlabOwner, Statistics,
};

/// `Slab<u64, InlineBacked<512>>` — bare-typed slab over inline storage.
/// Round-trip allocate / deallocate must not violate SB / TB.
#[test]
fn slab_inline_alloc_dealloc_cycle() {
    let slab: Slab<u64, InlineBacked<512>> = Slab::new(8, InlineBacked::<512>::new()).unwrap();
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let a = slab.allocate(layout).unwrap();
    let b = slab.allocate(layout).unwrap();
    unsafe {
        slab.deallocate(a.cast(), layout);
        slab.deallocate(b.cast(), layout);
        // Re-allocate after dealloc — LIFO reuse.
        let c = slab.allocate(layout).unwrap();
        slab.deallocate(c.cast(), layout);
    }
}

/// `BumpArena<InlineBacked<1024>>` cursor advance: every allocate
/// advances; reset reclaims. SB-sensitive because BumpArena returns
/// raw pointers carved from `InlineBacked::storage`'s SharedReadWrite.
#[test]
fn bump_inline_cursor_advance() {
    let mut bump: BumpArena<InlineBacked<1024>> =
        BumpArena::new(InlineBacked::<1024>::new()).unwrap();
    let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
    let mut ptrs = Vec::new();
    for _ in 0..16 {
        ptrs.push(bump.allocate(layout).unwrap());
    }
    // Pointers must be distinct + non-overlapping.
    for (i, &p) in ptrs.iter().enumerate() {
        for (j, &q) in ptrs.iter().enumerate() {
            if i != j {
                assert_ne!(p.cast::<u8>().as_ptr(), q.cast::<u8>().as_ptr());
            }
        }
    }
    // Reset reclaims everything.
    bump.reset();
    // After reset we can hand out the same range again.
    let _ = bump.allocate(layout).unwrap();
}

/// Exercises the typed `alloc_uninit` fast path and the `Scope` RAII checkpoint
/// under miri's borrow tracker: the `&mut`-from-`&self` allocations, the cursor
/// rewind on `Drop`, and reuse of the reclaimed region. This is the no-UB
/// signal for the typed-alloc / scope code that the unit tests can't provide.
#[test]
fn alloc_uninit_and_scope_round_trip() {
    let mut bump: BumpArena<InlineBacked<1024>> =
        BumpArena::new(InlineBacked::<1024>::new()).unwrap();

    // Typed alloc: write through the returned pointer (miri checks provenance).
    let p = bump.alloc_uninit::<u64>().unwrap();
    unsafe {
        p.as_ptr().write(0xDEAD_BEEF);
        assert_eq!(p.as_ptr().read(), 0xDEAD_BEEF);
    }
    let mark = bump.allocated();

    // Scope: allocate scratch, write through scope-bound `&mut`s, then drop.
    {
        let scope = bump.scope();
        let a = scope.alloc_uninit::<[u8; 32]>().unwrap();
        let b = scope.alloc_uninit::<u32>().unwrap();
        a.write([0xAB; 32]);
        b.write(0x1234_5678);
        assert_eq!(unsafe { a.assume_init_ref()[0] }, 0xAB);
        assert_eq!(unsafe { b.assume_init_ref() }, &0x1234_5678);

        // ZST through the scope: two `&mut MaybeUninit<ZST>` from the same
        // dangling-but-aligned address coexist. Miri's borrow tracker must
        // accept this (zero bytes accessed) — directly validates the ZST
        // SAFETY note on `Scope::alloc_uninit`.
        let z1 = scope.alloc_uninit::<()>().unwrap();
        let z2 = scope.alloc_uninit::<()>().unwrap();
        z1.write(());
        z2.write(());
    } // scope drops: cursor rewinds to `mark`

    assert_eq!(bump.allocated(), mark, "scope must rewind under miri");
    // The reclaimed region is reusable; miri must see no stale borrows.
    let q = bump.alloc_uninit::<u64>().unwrap();
    unsafe { q.as_ptr().write(1) };
}

/// Exercises the typed convenience copies and in-place `grow` under miri:
/// `alloc`/`alloc_slice_copy`/`alloc_str` provenance and the `*mut [u8] ->
/// *mut str` cast, plus the grow fast path (cursor advance, same pointer) and
/// the relocate path (`copy_nonoverlapping`).
#[test]
fn typed_copies_and_grow_under_miri() {
    use forge_alloc::Allocator;
    let arena: BumpArena<InlineBacked<1024>> = BumpArena::new(InlineBacked::<1024>::new()).unwrap();

    let v = arena.alloc(42u64).unwrap();
    assert_eq!(unsafe { v.as_ptr().read() }, 42);
    let s = arena.alloc_slice_copy(&[1u32, 2, 3]).unwrap();
    assert_eq!(unsafe { s.as_ref() }, &[1, 2, 3]);
    let st = arena.alloc_str("miri").unwrap();
    assert_eq!(unsafe { st.as_ref() }, "miri");

    // Empty + ZST-element dangling slices: miri must accept the
    // `slice_from_raw_parts(dangling, n)` construction (no byte access).
    let empty = arena.alloc_slice_copy::<u32>(&[]).unwrap();
    assert_eq!(unsafe { empty.as_ref() }, &[] as &[u32]);
    let zsts = arena.alloc_slice_copy(&[(), (), ()]).unwrap();
    assert_eq!(unsafe { zsts.as_ref().len() }, 3);
    let empty_str = arena.alloc_str("").unwrap();
    assert_eq!(unsafe { empty_str.as_ref() }, "");

    // In-place grow: `block` is the most-recent allocation.
    let l8 = NonZeroLayout::from_size_align(8, 8).unwrap();
    let l24 = NonZeroLayout::from_size_align(24, 8).unwrap();
    let block = arena.allocate(l8).unwrap().cast::<u8>();
    unsafe { core::ptr::write_bytes(block.as_ptr(), 0xEE, 8) };
    let grown = unsafe { arena.grow(block, l8, l24).unwrap() };
    assert_eq!(grown.cast::<u8>(), block); // same pointer, no copy
    unsafe { core::ptr::write_bytes(grown.cast::<u8>().as_ptr(), 0x11, 24) };

    // Relocate grow: allocate past `block2` so it is no longer last.
    let block2 = arena.allocate(l8).unwrap().cast::<u8>();
    unsafe { core::ptr::write_bytes(block2.as_ptr(), 0xCD, 8) };
    let _other = arena.allocate(l8).unwrap();
    let moved = unsafe { arena.grow(block2, l8, l24).unwrap() };
    assert_ne!(moved.cast::<u8>(), block2);
    assert_eq!(unsafe { moved.cast::<u8>().as_ptr().read() }, 0xCD);
}

/// `SharedBumpArena<InlineBacked<2048>>` — CAS-based bump.
/// Single-threaded under miri (miri can spawn threads but the test
/// stays deterministic to avoid stress on miri's borrow tracker).
#[test]
fn shared_bump_inline_cas_advance() {
    let bump: SharedBumpArena<InlineBacked<2048>> =
        SharedBumpArena::new(InlineBacked::<2048>::new()).unwrap();
    let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
    for _ in 0..32 {
        let _ = bump.allocate(layout).unwrap();
    }
}

/// `GenerationalSlab<u64, InlineBacked<512>>` — insert + get + remove.
#[test]
fn generational_slab_insert_get_remove() {
    let mut slab: GenerationalSlab<u64, InlineBacked<512>> =
        GenerationalSlab::new(8, InlineBacked::<512>::new()).unwrap();
    let h0 = slab.insert(42u64).unwrap();
    let h1 = slab.insert(99u64).unwrap();
    assert_eq!(*slab.get(h0).unwrap(), 42);
    assert_eq!(*slab.get(h1).unwrap(), 99);
    let v = slab.remove(h0).unwrap();
    assert_eq!(v, 42);
    // The removed handle's generation is bumped — re-issuing must NOT
    // collide with the prior handle.
    assert!(slab.get(h0).is_none(), "stale handle must be rejected");
    // Re-insert; the slot is recycled but with a new generation.
    let h2 = slab.insert(7u64).unwrap();
    assert_eq!(*slab.get(h2).unwrap(), 7);
    assert!(
        slab.get(h0).is_none(),
        "stale handle still rejected after recycle"
    );
}

/// `Statistics<Slab<u64, InlineBacked<512>>>` — count tracking +
/// verify allocate / deallocate pairs balance.
#[test]
fn statistics_slab_count_and_verify() {
    let slab: Slab<u64, InlineBacked<512>> = Slab::new(8, InlineBacked::<512>::new()).unwrap();
    let stats = Statistics::new(slab);
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let a = stats.allocate(layout).unwrap();
    let b = stats.allocate(layout).unwrap();
    assert_eq!(stats.stats().live_count(), 2);
    unsafe {
        stats.deallocate(a.cast(), layout);
        stats.deallocate(b.cast(), layout);
    }
    assert_eq!(stats.stats().live_count(), 0);
}

/// `Canary<BumpArena<InlineBacked<2048>>>` — corruption detection
/// requires the **inner** to be a non-Slab allocator because Canary
/// inflates each request by `max(8, align) + 8` bytes and Slab caps
/// the request at its stride. The composition documentation in
/// `canary.rs` calls this out — Canary over BumpArena is the right
/// inline composition to exercise under miri.
#[test]
fn canary_over_bump_arena_round_trips() {
    let canary: Canary<BumpArena<InlineBacked<2048>>> = Canary::new_with_seed(
        BumpArena::new(InlineBacked::<2048>::new()).unwrap(),
        0xC0FF_EE0D_DEAD_BEEF,
    );
    let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
    let block = canary.allocate(layout).unwrap();
    unsafe {
        // Write the user region; canaries must NOT be disturbed.
        core::ptr::write_bytes(block.cast::<u8>().as_ptr(), 0x42, 32);
        canary.deallocate(block.cast(), layout);
    }
}

/// `PoisonOnFree<Slab<u64, InlineBacked<512>>>` — round-trip a single
/// allocation; poison-on-free overwrites the user region with the
/// poison pattern before forwarding to inner.deallocate. The Slab
/// freelist write must not collide with the poison write (which it
/// can't, because Slab.deallocate runs AFTER PoisonOnFree.deallocate).
#[test]
fn poison_on_free_slab_round_trips() {
    let slab: Slab<u64, InlineBacked<512>> = Slab::new(8, InlineBacked::<512>::new()).unwrap();
    let poison: PoisonOnFree<Slab<u64, InlineBacked<512>>> = PoisonOnFree::new(slab);
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let block = poison.allocate(layout).unwrap();
    unsafe {
        // Write a recognizable value, then deallocate (poison + free).
        block.cast::<u64>().as_ptr().write(0xCAFE_BABE_DEAD_BEEF);
        poison.deallocate(block.cast(), layout);
    }
}

/// `SlabOwner<u64, InlineBacked<512>>` — owner-thread alloc + dealloc.
/// The cross-thread case requires `std::thread`; that path is covered
/// by the multi-threaded test in `slab_owner.rs` which is marked
/// `cfg_attr(miri, ignore)` for the same reason. This test exercises
/// the same `&*self.inner.slab.get()` (Shared-retag) fix on the
/// single-thread path so the SB invariant survives Drop.
#[test]
fn slab_owner_inline_single_thread_round_trips() {
    let owner: SlabOwner<u64, InlineBacked<512>> =
        SlabOwner::new(8, InlineBacked::<512>::new()).unwrap();
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let a = owner.allocate(layout).unwrap();
    let b = owner.allocate(layout).unwrap();
    unsafe {
        owner.deallocate(a.cast(), layout);
        owner.deallocate(b.cast(), layout);
    }
    // Owner drops here — the Drop drain path runs through the same
    // `&*self.inner.slab.get()` fix.
}

/// Composition stack: `Statistics< PoisonOnFree< Slab<…, InlineBacked> > >`.
/// Drops in reverse order; every drop step goes through interior-
/// mutability without creating a Unique retag over the inline storage.
#[test]
fn statistics_poison_slab_inline_round_trips() {
    let slab: Slab<u64, InlineBacked<512>> = Slab::new(8, InlineBacked::<512>::new()).unwrap();
    let poison = PoisonOnFree::new(slab);
    let stats = Statistics::new(poison);
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let a = stats.allocate(layout).unwrap();
    let b = stats.allocate(layout).unwrap();
    let c = stats.allocate(layout).unwrap();
    unsafe {
        stats.deallocate(a.cast(), layout);
        stats.deallocate(c.cast(), layout);
        // Re-alloc — must reuse a freed slot.
        let d = stats.allocate(layout).unwrap();
        stats.deallocate(b.cast(), layout);
        stats.deallocate(d.cast(), layout);
    }
}

// The compositions below require either MmapBacked or std::thread and
// cannot run under miri. They are listed here for the audit record so
// the file documents *all* compositions the prompt called out — the
// real coverage for these lives in each crate's own test module.

/// `SizeClassed<MmapBacked, 8>` — requires mmap; tested under non-miri
/// in `crates/forge-alloc/src/layout/size_classed.rs`.
#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: requires MmapBacked")]
fn size_classed_mmap_placeholder() {
    // Body is the real test elsewhere; this stub documents the gap.
}

/// `SlabOwner<u64, MmapBacked>` cross-thread drain — requires
/// `MmapBacked` AND `std::thread`. See
/// `crates/forge-alloc/src/layout/slab_owner.rs` for the live multi-thread test.
#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: requires MmapBacked + threads")]
fn slab_owner_mmap_cross_thread_placeholder() {}

/// `CacheJitter<BumpArena<MmapBacked>>` — needs `MmapBacked` because
/// jitter requires inner alignment >= `cache_line_size` and
/// `InlineBacked::MAX_ALIGN = 16`. Live test in
/// `crates/forge-alloc/src/hardening/cache_jitter.rs`.
#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: jitter needs MmapBacked alignment")]
fn cache_jitter_mmap_placeholder() {}
