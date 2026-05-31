//! `Faulty<I, P>` — fault-injection wrapper for out-of-memory path testing.
//!
//! Wraps any inner [`Allocator`] and consults an
//! [`AllocFaultPolicy`] on every allocation. When the policy votes to
//! fail, `Faulty` synthesises an [`AllocError`] instead of forwarding
//! to the inner allocator — turning the otherwise-unreachable OOM
//! branches of every allocator and composition in the family into
//! something a test, a `proptest` case, a `cargo fuzz` target, a MIRI
//! run, or a Kani proof can drive deterministically.
//!
//! See [`forge_alloc_core::AllocFaultPolicy`] for the policy seam and the
//! built-in policies ([`NeverFail`](forge_alloc_core::NeverFail),
//! [`AlwaysFail`](forge_alloc_core::AlwaysFail),
//! [`FailAfter`](forge_alloc_core::FailAfter),
//! [`FailEveryNth`](forge_alloc_core::FailEveryNth),
//! [`FailOnSize`](forge_alloc_core::FailOnSize)).

use core::ptr::NonNull;

use forge_alloc_core::{
    AllocError, AllocFaultPolicy, Allocator, Deallocator, FixedRange, NonZeroLayout,
};

/// Fault-injection wrapper: forces allocations to fail according to a
/// [`AllocFaultPolicy`].
///
/// # ⚠ Test / debug use only
///
/// `Faulty` exists to *break* allocation on purpose. It has no place in
/// a production composition — a `Faulty` left in a shipped allocator
/// stack is an allocator that fails for no reason. The type name is the
/// guardrail: you cannot get fault injection without writing `Faulty`,
/// so keep it confined to `#[cfg(test)]`, `tests/`, `benches/`, and
/// fuzz targets.
///
/// # What it intercepts
///
/// `Faulty` overrides exactly one method, [`Allocator::allocate`]: it
/// asks `P::should_fail` and either returns `Err(AllocError)` or
/// forwards to the inner allocator. `allocate_zeroed`, `grow`, and
/// `shrink` are deliberately **left as their `Allocator` trait
/// defaults**, which are implemented in terms of `allocate` — so they
/// route through the policy too, with no extra code. `deallocate` is
/// forwarded unconditionally: deallocation has no error channel to
/// inject into, and a faulted allocation never produced a pointer to
/// free.
///
/// Side effect of that choice: a `Faulty`-wrapped allocator always
/// takes the **allocate-copy-free** path for `grow` / `shrink`, even if
/// the bare inner allocator implements a native in-place resize —
/// `Faulty::grow` resolves to the trait default and never reaches
/// `I::grow`. A test that measures resize behaviour through `Faulty`
/// therefore sees allocate-copy-free semantics; unwrap to the bare
/// allocator if you need to exercise its in-place resize.
///
/// # Faulted requests consume nothing
///
/// The policy is consulted *before* the inner allocator is touched, so
/// a faulted request leaves the inner allocator's capacity, freelist,
/// and cursor completely unchanged. A `Faulty`-injected failure is
/// therefore observationally identical to a genuine OOM: wrappers above
/// it (`Statistics`, `WithFallback`, …) see exactly what they would see
/// if the inner allocator had really run out.
///
/// # Composition position
///
/// Place `Faulty` **just above the allocator whose OOM you want to
/// simulate**, below any hardening/observability wrappers — e.g.
/// `Statistics<Faulty<Slab<T>>>`. That way `Statistics::failures`
/// increments on an injected failure, exactly as it would for a real
/// one. Wrapping the other way (`Faulty<Statistics<Slab<T>>>`) hides
/// the injected failure from `Statistics`.
///
/// # `WithFallback` synergy
///
/// `WithFallback`'s whole reason to exist is its secondary path, and
/// that path is hard to reach in a test because the primary rarely
/// fails. `WithFallback<Faulty<Primary, AlwaysFail>, Secondary>` forces
/// *every* request onto the secondary — the cleanest way to exercise
/// the fallback branch deterministically.
///
/// # Send / Sync
///
/// `Faulty<I, P>` is `Send` / `Sync` exactly when both `I` and `P` are
/// — there is no interior mutability of its own. A policy built from
/// atomics (`FailAfter`, `FailEveryNth`) is `Sync`, so it never
/// downgrades a `Sync` inner allocator.
pub struct Faulty<I, P> {
    inner: I,
    policy: P,
}

impl<I, P> Faulty<I, P> {
    /// Wrap `inner`, injecting failures according to `policy`.
    #[inline]
    pub const fn new(inner: I, policy: P) -> Self {
        Self { inner, policy }
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }

    /// Borrow the fault policy.
    #[inline]
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Decompose into the inner allocator and the policy.
    #[inline]
    pub fn into_parts(self) -> (I, P) {
        (self.inner, self.policy)
    }
}

unsafe impl<I: Allocator, P: AllocFaultPolicy> Deallocator for Faulty<I, P> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // Deallocation is forwarded unconditionally: there is no error
        // channel to inject into, and because the policy is consulted
        // before the inner allocator is ever touched, a faulted
        // allocation never produced a pointer — so every `ptr` that
        // reaches here originated from `self.inner`.
        //
        // SAFETY: forwarded; the caller upholds the `Deallocator`
        // contract on `self`, which (per the above) is the same as
        // upholding it on `self.inner`.
        unsafe { self.inner.deallocate(ptr, layout) }
    }
}

unsafe impl<I: Allocator, P: AllocFaultPolicy> Allocator for Faulty<I, P> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        // Consult the policy BEFORE touching the inner allocator: a
        // faulted request must not consume any inner capacity, so the
        // injected failure is observationally identical to a genuine
        // OOM for any wrapper sitting above this one.
        if self.policy.should_fail(layout) {
            return Err(AllocError);
        }
        self.inner.allocate(layout)
    }

    // `allocate_zeroed`, `grow`, and `shrink` are intentionally NOT
    // overridden — their `Allocator` trait defaults are written in
    // terms of `allocate`, so they route through the fault policy via
    // the override above. Overriding them to forward straight to the
    // inner allocator would let those entry points bypass injection.

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        self.inner.capacity_bytes()
    }

    #[inline]
    unsafe fn usable_size(&self, ptr: NonNull<u8>, layout: NonZeroLayout) -> Option<usize> {
        // SAFETY: forwarded; the caller upholds the `usable_size`
        // contract, and `ptr` (being a live allocation) came from
        // `self.inner`.
        unsafe { self.inner.usable_size(ptr, layout) }
    }

    #[inline]
    fn reset(&mut self) -> Result<(), AllocError> {
        // `reset` reclaims everything; it is not an allocation, so the
        // policy does not apply. Forward so `Faulty<BumpArena>` and
        // other arenas remain resettable.
        self.inner.reset()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        self.inner.corruption_events()
    }
}

// `FixedRange: Allocator`, so this impl needs the same `P` bound the
// `Allocator` impl carries.
impl<I: FixedRange, P: AllocFaultPolicy> FixedRange for Faulty<I, P> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.inner.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Pass-through forward so a `commit`-aware consumer reaches the inner
    /// backing when this wrapper sits over a `lazy_commit` `MmapBacked`.
    #[inline]
    fn commit(&self, offset: usize, len: usize) -> Result<(), AllocError> {
        self.inner.commit(offset, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;
    use crate::layout::{BumpArena, Slab};
    use forge_alloc_core::{AlwaysFail, FailAfter, FailEveryNth, FailOnSize, NeverFail};

    fn layout() -> NonZeroLayout {
        NonZeroLayout::for_type::<u64>().unwrap()
    }

    #[test]
    fn never_fail_is_transparent() {
        let f: Faulty<BumpArena<InlineBacked<256>>, NeverFail> = Faulty::new(
            BumpArena::new(InlineBacked::<256>::new()).unwrap(),
            NeverFail,
        );
        // Every allocation proceeds to the inner arena.
        assert!(f.allocate(layout()).is_ok());
        assert!(f.allocate(layout()).is_ok());
    }

    #[test]
    fn always_fail_fails_every_allocation() {
        let f: Faulty<BumpArena<InlineBacked<256>>, AlwaysFail> = Faulty::new(
            BumpArena::new(InlineBacked::<256>::new()).unwrap(),
            AlwaysFail,
        );
        assert!(f.allocate(layout()).is_err());
        assert!(f.allocate(layout()).is_err());
    }

    #[test]
    fn fail_after_permits_then_fails() {
        let f: Faulty<BumpArena<InlineBacked<512>>, FailAfter> = Faulty::new(
            BumpArena::new(InlineBacked::<512>::new()).unwrap(),
            FailAfter::new(2),
        );
        assert!(f.allocate(layout()).is_ok());
        assert!(f.allocate(layout()).is_ok());
        assert!(f.allocate(layout()).is_err());
        assert!(f.allocate(layout()).is_err());
    }

    #[test]
    fn fail_on_size_discriminates_by_request_size() {
        let f: Faulty<BumpArena<InlineBacked<4096>>, FailOnSize> = Faulty::new(
            BumpArena::new(InlineBacked::<4096>::new()).unwrap(),
            FailOnSize::new(256),
        );
        // Small request succeeds, large request faults.
        let small = NonZeroLayout::from_size_align(64, 8).unwrap();
        let large = NonZeroLayout::from_size_align(512, 8).unwrap();
        assert!(f.allocate(small).is_ok());
        assert!(f.allocate(large).is_err());
    }

    /// A faulted allocation must not consume any inner capacity — the
    /// injected failure is observationally identical to a genuine OOM.
    /// Drive a `FailEveryNth(2)` over a `Slab` with capacity 2: if
    /// faulted requests consumed slots, the third *real* allocation
    /// would fail prematurely.
    #[test]
    fn faulted_allocation_consumes_no_inner_capacity() {
        let f: Faulty<Slab<u64, InlineBacked<512>>, FailEveryNth> = Faulty::new(
            Slab::new(2, InlineBacked::<512>::new()).unwrap(),
            FailEveryNth::new(2),
        );
        // Call 1: succeeds — slab slot 1 of 2 consumed.
        assert!(f.allocate(layout()).is_ok());
        // Call 2: faulted — must NOT touch the slab.
        assert!(f.allocate(layout()).is_err());
        // Call 3: succeeds — slab slot 2 of 2. Only reachable if call 2
        // consumed nothing.
        assert!(f.allocate(layout()).is_ok());
        // Call 4: faulted.
        assert!(f.allocate(layout()).is_err());
        // Call 5: genuine OOM — both slab slots are now live.
        assert!(f.allocate(layout()).is_err());
    }

    /// `deallocate` is forwarded unconditionally, even under a failing
    /// policy: the pointer being freed came from a *successful* earlier
    /// allocation, and freeing it must return the slot to the inner
    /// allocator.
    #[test]
    fn deallocate_forwards_under_failing_policy() {
        let f: Faulty<Slab<u64, InlineBacked<512>>, FailAfter> = Faulty::new(
            Slab::new(1, InlineBacked::<512>::new()).unwrap(),
            FailAfter::new(1),
        );
        // One success, then the policy fails forever.
        let p = f.allocate(layout()).unwrap();
        assert!(f.allocate(layout()).is_err());
        // Free the live pointer — forwarded to the slab despite the
        // policy being in its "fail" state.
        unsafe { f.deallocate(p.cast(), layout()) };
    }

    /// `allocate_zeroed` reaches the policy through its trait-default
    /// implementation (which calls `allocate`).
    #[test]
    fn allocate_zeroed_routes_through_policy() {
        let f: Faulty<BumpArena<InlineBacked<256>>, AlwaysFail> = Faulty::new(
            BumpArena::new(InlineBacked::<256>::new()).unwrap(),
            AlwaysFail,
        );
        assert!(f.allocate_zeroed(layout()).is_err());
    }

    /// `reset` is forwarded so a `Faulty`-wrapped arena stays usable.
    #[test]
    fn reset_forwards_to_inner_arena() {
        let mut f: Faulty<BumpArena<InlineBacked<256>>, NeverFail> = Faulty::new(
            BumpArena::new(InlineBacked::<256>::new()).unwrap(),
            NeverFail,
        );
        let _ = f.allocate(layout()).unwrap();
        assert!(f.reset().is_ok());
        // After reset the arena has full capacity again.
        assert!(f.allocate(layout()).is_ok());
    }

    /// `Faulty` with a `Sync` inner and an atomic-backed policy is
    /// `Sync` — it adds no interior mutability of its own.
    #[test]
    fn faulty_is_sync_when_parts_are_sync() {
        fn assert_sync<T: Sync>() {}
        // `SharedBumpArena` is the family's `Sync` allocator; `FailAfter`
        // is atomic-backed and therefore `Sync`.
        assert_sync::<Faulty<crate::layout::SharedBumpArena<InlineBacked<256>>, FailAfter>>();
    }

    /// `Faulty` drives `WithFallback`'s secondary branch: an
    /// `AlwaysFail` primary forces every request onto the fallback.
    #[test]
    #[cfg(feature = "std")]
    fn faulty_forces_withfallback_onto_secondary() {
        use crate::backing::System;
        use crate::layout::WithFallback;

        let primary: Faulty<Slab<u64, InlineBacked<512>>, AlwaysFail> = Faulty::new(
            Slab::new(8, InlineBacked::<512>::new()).unwrap(),
            AlwaysFail,
        );
        let wf = WithFallback::new(primary, System);

        // Primary always faults ⇒ this pointer is System-issued.
        let p = wf.allocate(layout()).unwrap();
        // Routed back to System on free (outside the primary's range).
        unsafe { wf.deallocate(p.cast(), layout()) };
    }

    /// `grow` is inherited (default allocate-copy-free). Under a failing
    /// policy the internal `self.allocate(new)` faults *before* the old block
    /// is freed, so `grow` returns `Err` and the original allocation is left
    /// intact — `Allocator::grow` contract item 5. Locks in the documented
    /// "never reaches `I::grow`" routing.
    #[test]
    fn grow_under_failing_policy_preserves_old_block() {
        let f: Faulty<BumpArena<InlineBacked<512>>, FailAfter> = Faulty::new(
            BumpArena::new(InlineBacked::<512>::new()).unwrap(),
            FailAfter::new(1),
        );
        let old = NonZeroLayout::from_size_align(16, 8).unwrap();
        let new = NonZeroLayout::from_size_align(64, 8).unwrap();
        let block = f.allocate(old).unwrap(); // call 1: ok
        let ptr = block.cast::<u8>();
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0x55, 16);
            // call 2 (grow's internal allocate) faults ⇒ grow returns Err.
            assert!(
                f.grow(ptr, old, new).is_err(),
                "grow must fail when the policy fails the new allocation",
            );
            // Old block untouched and still valid/deallocatable.
            for i in 0..16 {
                assert_eq!(*ptr.as_ptr().add(i), 0x55, "grow corrupted the old block");
            }
            f.deallocate(ptr, old);
        }
    }

    /// `capacity_bytes` is forwarded to inner, not the trait default `None` —
    /// a dropped forward would make this `None != Some(16)` and fail.
    /// (`corruption_events` is asserted as a smoke check only: a fresh slab and
    /// the unforwarded default both read 0, so this can't regression-proof that
    /// forward without a non-zero corruption source, which the family can't
    /// easily inject here.)
    #[test]
    fn forwards_capacity_bytes() {
        let f: Faulty<Slab<u64, InlineBacked<512>>, NeverFail> =
            Faulty::new(Slab::new(2, InlineBacked::<512>::new()).unwrap(), NeverFail);
        let bare = Slab::<u64, InlineBacked<512>>::new(2, InlineBacked::<512>::new()).unwrap();
        assert_eq!(
            f.capacity_bytes(),
            bare.capacity_bytes(),
            "capacity_bytes must forward to inner, not be None",
        );
        assert!(
            bare.capacity_bytes().is_some(),
            "bare slab must report a capacity"
        );
        assert_eq!(f.corruption_events(), 0); // smoke check (see doc)
    }
}
