//! End-to-end test for the `HardenedSlab` type alias.
//!
//! Verifies that `Slab<T, GuardPage<SplitMetadata<MmapBacked>>>` composes
//! correctly and can round-trip live allocations through write/read/dealloc.
//! This is the security-critical composition (PHI / keys / audit logs);
//! breakage here would silently degrade the protection model.

#![cfg(feature = "std")]

use forge_alloc::{
    Allocator, Deallocator, GuardPage, HardenedSlab, MmapBacked, NonZeroLayout, Slab, SplitMetadata,
};

/// Build a HardenedSlab over 256 u64 slots.
fn build_pool() -> HardenedSlab<u64> {
    // Construction follows the alias expansion:
    //   Slab<T, GuardPage<SplitMetadata<MmapBacked>>, NoProtection>
    //
    // 1. Raw OS mapping for the data region.
    let data_mmap = MmapBacked::new(256 * 1024).expect("mmap for slab data region");
    // 2. Separate metadata region.
    let split = SplitMetadata::new(data_mmap, 16 * 1024).expect("meta region");
    // 3. Guard pages around the data region.
    let guarded = GuardPage::new(split, 4096).expect("guard page wrap");
    // 4. Typed slab on top.
    Slab::<u64, _>::new(128, guarded).expect("slab construction")
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn round_trip_one_alloc() {
    let pool = build_pool();
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let block = pool.allocate(layout).expect("alloc");
    let p = block.cast::<u64>();
    unsafe {
        core::ptr::write(p.as_ptr(), 0xDEADBEEFu64);
        assert_eq!(core::ptr::read(p.as_ptr()), 0xDEADBEEFu64);
        pool.deallocate(p.cast(), layout);
    }
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn many_round_trips() {
    let pool = build_pool();
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    // Repeated alloc/free cycles to exercise the freelist push/pop path
    // through all three wrapper layers.
    for i in 0..1024u64 {
        let block = pool.allocate(layout).expect("alloc");
        let p = block.cast::<u64>();
        unsafe {
            core::ptr::write(p.as_ptr(), i);
            assert_eq!(core::ptr::read(p.as_ptr()), i);
            pool.deallocate(p.cast(), layout);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
fn distinct_live_allocations_dont_overlap() {
    let pool = build_pool();
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let mut ptrs = Vec::new();
    for _ in 0..64 {
        ptrs.push(pool.allocate(layout).expect("alloc"));
    }
    // All addresses distinct.
    let mut sorted: Vec<usize> = ptrs
        .iter()
        .map(|p| p.cast::<u8>().as_ptr() as usize)
        .collect();
    sorted.sort();
    for w in sorted.windows(2) {
        assert_ne!(w[0], w[1], "HardenedSlab returned duplicate pointer");
        // u64 = 8 bytes; Slab's block_stride ≥ 8.
        assert!(w[1] - w[0] >= 8, "addresses too close together");
    }
    for p in ptrs {
        unsafe { pool.deallocate(p.cast(), layout) };
    }
}
