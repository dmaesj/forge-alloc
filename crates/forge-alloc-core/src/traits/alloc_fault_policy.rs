//! `AllocFaultPolicy` — the allocation fault-injection seam.
//!
//! Real allocators almost never hit their out-of-memory path in a test
//! suite, so the `Err(AllocError)` branches threaded through every
//! allocator and every composition in this family — `BumpArena`
//! exhaustion, `Slab` capacity, the `WithFallback` fallback decision,
//! a caller's own `?`-propagation — are chronically under-exercised.
//!
//! This module defines the *policy* half of a fix: a trait that decides,
//! per request, whether an allocation should be forced to fail. The
//! `hardening` module's `Faulty<I, P>` wrapper is the *mechanism* half — it
//! consults a `P: AllocFaultPolicy` and synthesises an `AllocError`
//! when the policy says so, turning those OOM paths into something a
//! unit test, a `proptest` case, a `cargo fuzz` target, a MIRI run, or
//! a Kani proof can drive deterministically.

use super::non_zero_layout::NonZeroLayout;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Decides whether a pending allocation request should be forced to
/// fail.
///
/// A `hardening` module `Faulty<I, P>` wrapper consults a
/// `P: AllocFaultPolicy` on the allocation path and, when
/// [`should_fail`](Self::should_fail) returns `true`, returns
/// [`AllocError`](crate::AllocError) instead of forwarding to the inner
/// allocator `I`.
///
/// # The seam
///
/// This trait lives in `forge-alloc-core` and is intentionally
/// **dependency-free**: its single method takes only a [`NonZeroLayout`]
/// (a `forge-alloc-core` type) and returns `bool`. Nothing from a
/// determinism / replay layer crosses it. That keeps the dependency
/// arrow pointing one way — a *seeded, replayable* policy is
/// implemented downstream (it depends on `forge-alloc-core` to obtain this
/// trait), and neither `forge-alloc-core` nor the `hardening` module ever gains a
/// dependency on that layer. The forge-* crates ship only the trivial
/// built-in policies in this module ([`NeverFail`], [`AlwaysFail`],
/// [`FailAfter`], [`FailEveryNth`], [`FailOnSize`]); anything
/// stateful-and-reproducible is a downstream concern.
///
/// # Statefulness
///
/// [`should_fail`](Self::should_fail) takes `&self`, not `&mut self` —
/// matching every allocator in the family, whose methods also take
/// `&self` and reach their mutable state through interior mutability.
/// A policy that counts calls or carries PRNG state holds it in a
/// [`Cell`](core::cell::Cell) (single-threaded) or an atomic (shared).
/// A policy built only from atomics is `Sync`, so it does not constrain
/// the `Sync`-ness of the `Faulty<I, P>` it parameterises.
pub trait AllocFaultPolicy {
    /// Return `true` to force the pending allocation of `layout` to
    /// fail with [`AllocError`](crate::AllocError); return `false` to
    /// let it proceed to the inner allocator.
    ///
    /// Called once per allocation on the `Faulty` wrapper. `Faulty`
    /// overrides only `allocate`; `allocate_zeroed`, `grow`, and
    /// `shrink` reach the policy *through* their default trait
    /// implementations (which route to `allocate`), so a single
    /// `should_fail` verdict covers whichever entry point the caller
    /// used.
    ///
    /// Implementations must not panic and should be cheap — this runs
    /// on the allocation hot path whenever a `Faulty` wrapper is
    /// composed in.
    fn should_fail(&self, layout: NonZeroLayout) -> bool;
}

/// Policy that never injects a failure — every allocation proceeds to
/// the inner allocator unchanged.
///
/// `Faulty<I, NeverFail>` is a transparent passthrough; useful as the
/// "injection disabled" arm of generic code that is parameterised over
/// an `AllocFaultPolicy`.
#[derive(Copy, Clone, Debug, Default)]
pub struct NeverFail;

impl AllocFaultPolicy for NeverFail {
    #[inline]
    fn should_fail(&self, _layout: NonZeroLayout) -> bool {
        false
    }
}

/// Policy that fails every allocation.
///
/// `Faulty<I, AlwaysFail>` is an allocator that is permanently out of
/// memory — useful for exercising a caller's OOM handling in isolation,
/// or as the primary of a `WithFallback` so that every request is
/// forced onto the fallback path.
#[derive(Copy, Clone, Debug, Default)]
pub struct AlwaysFail;

impl AllocFaultPolicy for AlwaysFail {
    #[inline]
    fn should_fail(&self, _layout: NonZeroLayout) -> bool {
        true
    }
}

/// Policy that permits the first `successes` allocations and fails
/// every allocation thereafter.
///
/// Models a fixed-capacity allocator that runs out: the OOM cliff lands
/// at a precise, reproducible request number regardless of request
/// sizes. Stateful — the success counter is an [`AtomicUsize`], so the
/// policy is `Send + Sync` and a `Faulty` parameterised by it can wrap
/// a multi-threaded allocator.
#[derive(Debug)]
pub struct FailAfter {
    /// Allocations still permitted to succeed before failure begins.
    /// Decremented (saturating at zero) by each `should_fail` call.
    successes_left: AtomicUsize,
}

impl FailAfter {
    /// Construct a policy that allows `successes` allocations through,
    /// then fails all subsequent ones.
    ///
    /// `FailAfter::new(0)` is equivalent to [`AlwaysFail`].
    #[inline]
    pub const fn new(successes: usize) -> Self {
        Self {
            successes_left: AtomicUsize::new(successes),
        }
    }

    /// Success credits still remaining (advisory snapshot — may be
    /// stale the instant it is read under concurrent allocation).
    #[inline]
    pub fn remaining(&self) -> usize {
        self.successes_left.load(Ordering::Relaxed)
    }
}

impl AllocFaultPolicy for FailAfter {
    #[inline]
    fn should_fail(&self, _layout: NonZeroLayout) -> bool {
        // Atomically consume one success credit. `checked_sub(1)`
        // returns `None` once the counter has reached zero, which makes
        // `fetch_update` return `Err` and leaves the counter at zero —
        // so the policy never wraps and never "recovers" credits.
        // `Err` is exactly the "fail" verdict.
        self.successes_left
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_err()
    }
}

/// Policy that fails every `period`-th allocation and permits the rest.
///
/// Models intermittent OOM (transient pressure): with `period = 3`,
/// allocations 3, 6, 9, … fail and the rest succeed. Stateful — the
/// call counter is an [`AtomicUsize`], so the policy is `Send + Sync`.
#[derive(Debug)]
pub struct FailEveryNth {
    /// Fail on every allocation whose 1-based index is a multiple of
    /// this. Always `>= 1` (enforced by [`FailEveryNth::new`]).
    period: usize,
    /// Count of `should_fail` calls observed so far.
    count: AtomicUsize,
}

impl FailEveryNth {
    /// Construct a policy that fails every `period`-th allocation.
    ///
    /// # Panics
    ///
    /// Panics if `period == 0`. A period of `1` fails every allocation
    /// (equivalent to [`AlwaysFail`]).
    #[inline]
    pub const fn new(period: usize) -> Self {
        assert!(period >= 1, "FailEveryNth period must be >= 1");
        Self {
            period,
            count: AtomicUsize::new(0),
        }
    }
}

impl AllocFaultPolicy for FailEveryNth {
    #[inline]
    fn should_fail(&self, _layout: NonZeroLayout) -> bool {
        // `fetch_add` returns the previous (0-based) count: the k-th
        // call (1-based) observes `prev == k - 1`. The k-th allocation
        // should fail iff `k` is a multiple of `period`, i.e. iff
        // `prev % period == period - 1`.
        //
        // The counter wraps after `usize::MAX` calls, which phase-shifts
        // the cadence unless `period` divides 2^usize::BITS — irrelevant
        // for any realistic test run.
        let prev = self.count.fetch_add(1, Ordering::Relaxed);
        prev % self.period == self.period - 1
    }
}

/// Policy that fails any allocation whose requested size is at least
/// `min_fail_bytes`, and permits smaller ones.
///
/// Models an allocator that can satisfy small requests but not large
/// ones — exercising the size-dependent OOM branch (a slab that rejects
/// over-stride requests, a fixed arena that has room for scratch but
/// not a bulk buffer). Stateless and `Copy`.
#[derive(Copy, Clone, Debug)]
pub struct FailOnSize {
    /// Smallest request size, in bytes, that is forced to fail
    /// (inclusive).
    pub min_fail_bytes: usize,
}

impl FailOnSize {
    /// Construct a policy that fails allocations of `min_fail_bytes`
    /// bytes or larger.
    #[inline]
    pub const fn new(min_fail_bytes: usize) -> Self {
        Self { min_fail_bytes }
    }
}

impl AllocFaultPolicy for FailOnSize {
    #[inline]
    fn should_fail(&self, layout: NonZeroLayout) -> bool {
        layout.size().get() >= self.min_fail_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec::Vec;

    fn layout(size: usize) -> NonZeroLayout {
        NonZeroLayout::from_size_align(size, 8).unwrap()
    }

    #[test]
    fn never_fail_never_fails() {
        let p = NeverFail;
        assert!(!p.should_fail(layout(8)));
        assert!(!p.should_fail(layout(1 << 20)));
    }

    #[test]
    fn always_fail_always_fails() {
        let p = AlwaysFail;
        assert!(p.should_fail(layout(8)));
        assert!(p.should_fail(layout(1 << 20)));
    }

    #[test]
    fn fail_after_permits_then_fails() {
        let p = FailAfter::new(3);
        assert_eq!(p.remaining(), 3);
        // First three succeed.
        assert!(!p.should_fail(layout(8)));
        assert!(!p.should_fail(layout(8)));
        assert!(!p.should_fail(layout(8)));
        assert_eq!(p.remaining(), 0);
        // Everything after fails — and stays failed (no wrap/recover).
        assert!(p.should_fail(layout(8)));
        assert!(p.should_fail(layout(8)));
        assert_eq!(p.remaining(), 0);
    }

    #[test]
    fn fail_after_zero_is_always_fail() {
        let p = FailAfter::new(0);
        assert!(p.should_fail(layout(8)));
        assert!(p.should_fail(layout(8)));
    }

    #[test]
    fn fail_every_nth_fails_on_period_multiples() {
        let p = FailEveryNth::new(3);
        // Calls 1,2 succeed; 3 fails; 4,5 succeed; 6 fails.
        let verdicts: Vec<bool> = (0..6).map(|_| p.should_fail(layout(8))).collect();
        assert_eq!(verdicts, [false, false, true, false, false, true]);
    }

    #[test]
    fn fail_every_nth_period_one_is_always_fail() {
        let p = FailEveryNth::new(1);
        assert!(p.should_fail(layout(8)));
        assert!(p.should_fail(layout(8)));
    }

    #[test]
    #[should_panic(expected = "period must be >= 1")]
    fn fail_every_nth_rejects_zero_period() {
        let _ = FailEveryNth::new(0);
    }

    #[test]
    fn fail_on_size_threshold_is_inclusive() {
        let p = FailOnSize::new(64);
        assert!(!p.should_fail(layout(63)));
        assert!(p.should_fail(layout(64)));
        assert!(p.should_fail(layout(65)));
    }

    /// The stateful policies must be `Send + Sync` so a `Faulty`
    /// parameterised by them can wrap a multi-threaded allocator.
    #[test]
    fn stateful_policies_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FailAfter>();
        assert_send_sync::<FailEveryNth>();
        assert_send_sync::<NeverFail>();
        assert_send_sync::<AlwaysFail>();
        assert_send_sync::<FailOnSize>();
    }
}
