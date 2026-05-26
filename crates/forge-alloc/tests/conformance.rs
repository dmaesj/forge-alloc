//! Smoke-tests the [`forge_alloc_core::testing`] conformance helpers
//! against this crate's own in-tree impls. Doubles as a regression
//! gate: any new backing or allocator added to forge-alloc should be
//! exercised here so the helpers themselves keep working and the new
//! impl is confirmed contract-compliant.

use forge_alloc::{
    BumpArena, HeapBytes, HugePageBacked, InlineBacked, MmapBacked, NonZeroLayout, StaticBacked,
    System,
};
use forge_alloc_core::testing::{
    assert_allocator_basic_round_trip, assert_allocator_respects_alignment,
    assert_combined_invariants, assert_fixed_range_invariants,
};
use forge_alloc_core::Allocator;

#[test]
fn inline_backed_meets_contract() {
    let b = InlineBacked::<4096>::new();
    assert_combined_invariants(&b);
    let b = InlineBacked::<4096>::new();
    assert_allocator_respects_alignment(&b);
}

#[test]
fn heap_bytes_meets_fixed_range_contract() {
    let h = HeapBytes::new(4096).unwrap();
    assert_fixed_range_invariants(&h);
}

#[test]
fn static_backed_meets_fixed_range_contract() {
    let mut buf = [0u8; 4096];
    let s = StaticBacked::new(&mut buf);
    assert_fixed_range_invariants(&s);
}

#[test]
fn bump_arena_over_heap_bytes_meets_contract() {
    let arena = BumpArena::new(HeapBytes::new(8192).unwrap()).unwrap();
    assert_combined_invariants(&arena);
    let arena = BumpArena::new(HeapBytes::new(8192).unwrap()).unwrap();
    assert_allocator_respects_alignment(&arena);
}

#[test]
fn bump_arena_over_static_backed_meets_contract() {
    let mut buf = [0u8; 8192];
    let arena = BumpArena::new(StaticBacked::new(&mut buf)).unwrap();
    assert_combined_invariants(&arena);
}

#[test]
fn mmap_backed_meets_contract() {
    let m = MmapBacked::new(16 * 1024).unwrap();
    assert_combined_invariants(&m);
    let m = MmapBacked::new(16 * 1024).unwrap();
    assert_allocator_respects_alignment(&m);
}

/// `HugePageBacked` exercise gated on platform support: when the
/// kernel can't satisfy a 2 MiB huge-page request (CI runners with
/// no `nr_hugepages` reservation, Apple Silicon, Android, etc.),
/// silently skip. When it can, validate the same contract every
/// other backing meets.
#[test]
fn huge_page_backed_meets_contract_when_supported() {
    if let Ok(h) = HugePageBacked::new(2 * 1024 * 1024) {
        assert_combined_invariants(&h);
    }
    if let Ok(h) = HugePageBacked::new(2 * 1024 * 1024) {
        assert_allocator_respects_alignment(&h);
    }
}

#[test]
fn system_meets_allocator_contract() {
    let s = System;
    assert_allocator_basic_round_trip(&s);
    assert_allocator_respects_alignment(&s);
    // System is intentionally NOT FixedRange (unbounded global), so
    // we skip assert_combined_invariants here. Verify that decision
    // hasn't silently regressed:
    let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
    let p = s.allocate(layout).unwrap();
    assert!(p.len() >= 32);
    // SAFETY: just allocated.
    unsafe {
        use forge_alloc_core::Deallocator;
        s.deallocate(p.cast(), layout);
    }
}
