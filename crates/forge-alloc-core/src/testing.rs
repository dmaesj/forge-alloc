//! Conformance helpers for downstream impls of [`FixedRange`] and
//! [`Allocator`].
//!
//! Drop these into a `#[test]` to validate that a custom
//! implementation meets the trait contracts that wrappers in the
//! `forge-alloc` family rely on. Helpers panic on failure with a
//! human-readable diagnostic, so they compose with the standard
//! `#[test]` harness without ceremony.
//!
//! # Example
//!
//! ```ignore
//! use forge_alloc_core::testing::{
//!     assert_allocator_basic_round_trip,
//!     assert_fixed_range_invariants,
//! };
//!
//! #[test]
//! fn my_allocator_meets_contract() {
//!     let a = MyAllocator::new(4096);
//!     assert_fixed_range_invariants(&a);
//!     assert_allocator_basic_round_trip(&a);
//! }
//! ```
//!
//! These helpers cover the **structural and basic behavioral**
//! parts of the contract — pointer non-null-ness, alignment,
//! non-overlap, address-range stability, single-threaded
//! write-then-read round-trip, `contains` membership for issued
//! pointers. They do NOT cover:
//!
//! - Concurrent-access invariants (Inv 3 — "allocator doesn't
//!   concurrently write user-visible bytes"). Verifying that
//!   requires a multi-threaded probe with a happens-before
//!   discipline, which is allocator-specific.
//! - `grow` / `shrink` semantics (Inv 5). Add helpers for these as
//!   they're needed.
//! - Statistical / performance properties (no benchmarking) or
//!   hardening-specific behavior (no MAC verification, no canary
//!   checks). Use crate-specific tests for those.
//!
//! On failure, helpers panic at the call site (each is
//! `#[track_caller]`) so the test report points at your test, not
//! at this file.

use core::ptr::NonNull;

use crate::traits::{Allocator, FixedRange, NonZeroLayout};

/// Verify the structural invariants of [`FixedRange`]:
///
/// 1. [`base`](FixedRange::base) returns the same address on every
///    call.
/// 2. [`size`](FixedRange::size) returns the same value on every
///    call.
/// 3. If `size > 0`, [`contains`](FixedRange::contains) reports
///    `true` for `base` and for `base + size - 1`.
/// 4. `contains(base + size)` is `false` (one-past-the-end is out
///    of range).
///
/// Panics on the first violated invariant with a message naming the
/// failed check.
#[track_caller]
pub fn assert_fixed_range_invariants<R: FixedRange>(r: &R) {
    let b1 = r.base();
    let b2 = r.base();
    assert_eq!(
        b1, b2,
        "FixedRange::base() must return the same address on every call",
    );

    let s1 = r.size();
    let s2 = r.size();
    assert_eq!(
        s1, s2,
        "FixedRange::size() must return the same value on every call",
    );

    if s1 > 0 {
        assert!(
            r.contains(b1),
            "FixedRange::contains(base()) must be true when size() > 0",
        );

        // SAFETY: s1 > 0 so base() + (s1 - 1) is the last byte of
        // the range, in-bounds for pointer arithmetic.
        let last_byte = unsafe { NonNull::new_unchecked(b1.as_ptr().add(s1 - 1)) };
        assert!(
            r.contains(last_byte),
            "FixedRange::contains(base() + size() - 1) must be true",
        );
    }

    // One-past-the-end: must NOT be reported as contained.
    // Use wrapping_add so an end address that overflows usize::MAX
    // produces a defined (though semantically "past the end") value
    // rather than UB.
    let past_addr = (b1.as_ptr() as usize).wrapping_add(s1);
    if let Some(past_end) = NonNull::new(past_addr as *mut u8) {
        assert!(
            !r.contains(past_end),
            "FixedRange::contains(base() + size()) must be false (one past end)",
        );
    }
}

/// Verify the basic round-trip invariants of [`Allocator`]:
///
/// - [`allocate`](Allocator::allocate) returns a non-null,
///   appropriately aligned pointer to at least `layout.size()`
///   writable bytes.
/// - Two successive allocations from the same instance never
///   overlap.
/// - [`allocate_zeroed`](Allocator::allocate_zeroed) yields a
///   block whose `layout.size()` bytes are all zero.
/// - [`deallocate`](crate::traits::Deallocator::deallocate) accepts
///   every issued pointer without panicking.
///
/// Uses a `64`-byte, `8`-aligned layout. Allocators that legitimately
/// reject this size should not be tested with this helper — use
/// [`assert_allocator_respects_alignment`] for a finer-grained
/// alignment probe.
///
/// Panics on the first violated invariant.
#[track_caller]
pub fn assert_allocator_basic_round_trip<A: Allocator>(a: &A) {
    let layout =
        NonZeroLayout::from_size_align(64, 8).expect("64/8 is a valid NonZeroLayout");

    let p1 = a
        .allocate(layout)
        .expect("first basic allocation (64 bytes, align 8) must succeed");
    assert_eq!(
        p1.cast::<u8>().as_ptr() as usize % 8,
        0,
        "Allocator::allocate must return a pointer aligned to layout.align() ({:p})",
        p1.cast::<u8>().as_ptr(),
    );
    assert!(
        p1.len() >= layout.size().get(),
        "Allocator::allocate must return a slice of at least layout.size() bytes",
    );

    let p2 = a
        .allocate(layout)
        .expect("second basic allocation must succeed");

    let p1_start = p1.cast::<u8>().as_ptr() as usize;
    let p1_end = p1_start + 64;
    let p2_start = p2.cast::<u8>().as_ptr() as usize;
    let p2_end = p2_start + 64;
    assert!(
        p1_end <= p2_start || p2_end <= p1_start,
        "two live allocations must not overlap (got p1=[{p1_start:#x}, {p1_end:#x}), \
         p2=[{p2_start:#x}, {p2_end:#x}))",
    );

    // Write-then-read round trip — surfaces non-writable mappings
    // (PROT_NONE / PAGE_NOACCESS) immediately as SIGSEGV instead of
    // silently passing. Use a non-zero base pattern (`0xCC ^ i`) so
    // byte 0 is `0xCC` — if the allocator silently returned a
    // zeroed region and dropped our writes, byte 0 would no longer
    // match and the test fails. A `0..64` write pattern would
    // happen to put zero at byte 0 and miss this class of bug.
    // SAFETY: 64 bytes we just allocated; in-bounds writes / reads.
    unsafe {
        for i in 0..64_u8 {
            *p1.cast::<u8>().as_ptr().add(i as usize) = 0xCC ^ i;
        }
        for i in 0..64_u8 {
            assert_eq!(
                *p1.cast::<u8>().as_ptr().add(i as usize),
                0xCC ^ i,
                "byte {i} of fresh allocation must round-trip through write/read",
            );
        }
    }

    let p3 = a
        .allocate_zeroed(layout)
        .expect("allocate_zeroed must succeed for 64 bytes, align 8");
    // SAFETY: 64 bytes we just allocated, guaranteed zero by the contract.
    unsafe {
        for i in 0..64 {
            assert_eq!(
                *p3.cast::<u8>().as_ptr().add(i),
                0,
                "Allocator::allocate_zeroed byte {i} must be zero",
            );
        }
    }

    // SAFETY: each of p1/p2/p3 came from this allocator with the
    // same layout we'll pass to deallocate.
    unsafe {
        a.deallocate(p1.cast(), layout);
        a.deallocate(p2.cast(), layout);
        a.deallocate(p3.cast(), layout);
    }
}

/// Verify that [`allocate`](Allocator::allocate) returns pointers
/// aligned to the requested alignment, across every power-of-two
/// alignment from `1` to `512`. (Covers up through
/// [`CACHE_LINE`](crate::cache_padded::CACHE_LINE) `= 128` on
/// every supported target plus a 4-KiB-page-aligned slot.)
///
/// Allocators that legitimately reject high alignments (e.g.
/// [`InlineBacked`](https://docs.rs/forge-alloc/latest/forge_alloc/struct.InlineBacked.html)
/// rejects anything above `MAX_ALIGN = 16`) are not flagged as
/// failing: the helper only asserts that *successful* returns are
/// correctly aligned.
///
/// Panics if a successful allocation returns an under-aligned
/// pointer.
#[track_caller]
pub fn assert_allocator_respects_alignment<A: Allocator>(a: &A) {
    for align_pow in 0..=9_u32 {
        let align: usize = 1 << align_pow;
        let layout = NonZeroLayout::from_size_align(64, align)
            .expect("64/(power-of-two) is always a valid NonZeroLayout");
        match a.allocate(layout) {
            Ok(p) => {
                assert_eq!(
                    p.cast::<u8>().as_ptr() as usize % align,
                    0,
                    "Allocator::allocate with align={align} returned an under-aligned pointer ({:p})",
                    p.cast::<u8>().as_ptr(),
                );
                // SAFETY: we just allocated p with this layout.
                unsafe {
                    a.deallocate(p.cast(), layout);
                }
            }
            Err(_) => {
                // Allocator legitimately rejected the alignment;
                // not a contract violation.
            }
        }
    }
}

/// Verify that an allocator implements **both** [`FixedRange`] and
/// [`Allocator`] consistently:
///
/// - Runs [`assert_fixed_range_invariants`] and
///   [`assert_allocator_basic_round_trip`].
/// - Additionally checks that every pointer issued by `allocate`
///   lies inside `[base(), base() + size())` (i.e. the allocator
///   only hands out memory from its declared range).
///
/// Panics on the first violated invariant.
#[track_caller]
pub fn assert_combined_invariants<A: Allocator + FixedRange>(a: &A) {
    assert_fixed_range_invariants(a);

    let layout =
        NonZeroLayout::from_size_align(64, 8).expect("64/8 is a valid NonZeroLayout");
    let p = a
        .allocate(layout)
        .expect("64/8 allocation must succeed for combined check");
    assert!(
        a.contains(p.cast::<u8>()),
        "FixedRange::contains() must return true for any pointer issued by allocate()",
    );

    // Inverse: a pointer far outside the range must NOT report as
    // contained. We construct it via integer arithmetic and
    // `NonNull::new` rather than `.add()` (which has in-bounds
    // preconditions). 1 TiB past base is far enough outside any
    // realistic range that it can't accidentally be in-range.
    let outside_addr = (a.base().as_ptr() as usize).wrapping_add(1 << 40);
    if let Some(outside) = NonNull::new(outside_addr as *mut u8) {
        assert!(
            !a.contains(outside),
            "FixedRange::contains() must return false for a pointer far outside the range",
        );
    }

    // SAFETY: we just allocated p with this layout.
    unsafe {
        a.deallocate(p.cast(), layout);
    }

    assert_allocator_basic_round_trip(a);
}
