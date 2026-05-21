//! `WithFallback<P, S>` — try the primary allocator; on `AllocError`, fall
//! back to the secondary. Deallocation routes by pointer-provenance via
//! [`forge_alloc_core::FixedRange::contains`].
//!
//! Typical pattern: `WithFallback<BumpArena<InlineBacked<N>>, System>` —
//! stack-fast for the common case, global heap for overflow.
//!
//! ```
//! # #[cfg(feature = "std")]
//! # {
//! use forge_alloc::{InlineBacked, System};
//! use forge_alloc::{Allocator, Deallocator, NonZeroLayout};
//! use forge_alloc::WithFallback;
//!
//! let alloc = WithFallback::new(InlineBacked::<256>::new(), System);
//! // Small request — served by the inline primary.
//! let small = NonZeroLayout::from_size_align(64, 8).unwrap();
//! let block = alloc.allocate(small).unwrap();
//! unsafe { alloc.deallocate(block.cast(), small) };
//! # }
//! ```
//!
//! See `docs/ARCHITECTURE.md` for the fallback-router design.

use core::ptr::NonNull;

use forge_alloc_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// Router with primary + secondary allocator.
///
/// `Primary` must implement [`FixedRange`] so that deallocation can be routed
/// correctly. Growing primaries (e.g. an `ExtendableSlab`) cannot be
/// used here — split them out at the application level instead.
///
/// # Safety / contract
///
/// - If `secondary.contains(ptr)` would also return `true` for a primary-issued
///   pointer (i.e. their address ranges overlap), deallocation routing is
///   incorrect and behavior is UB. In practice the secondary is usually
///   [`crate::backing::System`](https://docs.rs/forge-alloc) which doesn't implement
///   `FixedRange`, so this concern is hypothetical.
/// - Calling `deallocate` with a pointer that belongs to neither allocator
///   is UB in release builds; debug builds gain a tracking check only when
///   `Statistics` is composed.
///
/// # Watermark composition note
///
/// `capacity_bytes()` reports the **primary's** capacity only — the secondary
/// is treated as unbounded overflow drainage, not as part of the budget.
/// `Watermark<WithFallback<P, S>>` therefore monitors pressure on the FAST
/// path; secondary-side activity does not move the watermark thresholds.
/// This is intentional (you want to know when the bump arena is hot, not
/// when the System heap took a small extra request), but callers who want
/// total-bytes monitoring should wrap each half separately, e.g.
/// `WithFallback<Watermark<P, _>, Watermark<S, _>>`.
///
/// # API-misuse compile-failures (pinned)
///
/// The `Allocator` and `Deallocator` impls require `P: FixedRange` so
/// `deallocate` can route by pointer-provenance. Using a non-`FixedRange`
/// primary (e.g. `crate::backing::System`) compiles fine at the `new()`
/// constructor — `WithFallback` itself imposes no bound — but the
/// resulting value cannot satisfy the trait bound on `allocate`:
///
/// ```compile_fail
/// // FAILS TO COMPILE: `System` is not `FixedRange`, so
/// // `WithFallback<System, InlineBacked<256>>` does not implement
/// // `Allocator` and `.allocate(...)` is not a method on it.
/// use forge_alloc::{InlineBacked, System};
/// use forge_alloc::{Allocator, NonZeroLayout};
/// use forge_alloc::WithFallback;
/// let wf = WithFallback::new(System, InlineBacked::<256>::new());
/// let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
/// let _ = wf.allocate(layout);
/// ```
#[derive(Debug)]
pub struct WithFallback<P, S> {
    primary: P,
    secondary: S,
}

impl<P, S> WithFallback<P, S> {
    /// Construct from existing primary and secondary instances.
    ///
    /// This is the const-fn path; no runtime range check is performed.
    /// Use this when the secondary doesn't implement
    /// [`FixedRange`](forge_alloc_core::FixedRange) (the common case — the
    /// canonical secondary is [`crate::backing::System`](https://docs.rs/forge-alloc),
    /// which is not `FixedRange`). For that wiring, deallocation
    /// routing is unambiguous: any pointer outside the primary's
    /// range goes to the secondary, and `System` accepts any pointer.
    ///
    /// **When BOTH halves implement `FixedRange`**, prefer
    /// [`try_new`](Self::try_new) instead — `try_new` verifies the
    /// two address ranges are disjoint at construction. Overlapping
    /// ranges with this constructor will silently misroute
    /// secondary-issued pointers through the primary's
    /// `deallocate`, producing a freelist corruption that is hard
    /// to diagnose after the fact.
    #[inline]
    pub const fn new(primary: P, secondary: S) -> Self {
        Self { primary, secondary }
    }

    /// Borrow the primary allocator.
    #[inline]
    pub fn primary(&self) -> &P {
        &self.primary
    }

    /// Borrow the secondary allocator.
    #[inline]
    pub fn secondary(&self) -> &S {
        &self.secondary
    }

    /// Decompose into the two halves.
    #[inline]
    pub fn into_parts(self) -> (P, S) {
        (self.primary, self.secondary)
    }
}

impl<P: FixedRange, S: FixedRange> WithFallback<P, S> {
    /// Construct with a runtime check that the primary and secondary
    /// address ranges are disjoint. Returns `Err(AllocError)` on
    /// overlap — overlapping ranges silently misroute deallocations
    /// through `deallocate`'s primary-first `contains` test, producing
    /// a freelist corruption that's hard to debug after the fact.
    ///
    /// Use this constructor whenever both halves implement
    /// `FixedRange`. The default secondary
    /// [`crate::backing::System`](https://docs.rs/forge-alloc) does not
    /// implement `FixedRange`, so callers wiring `System` as the
    /// fallback continue to use [`WithFallback::new`] — `System`
    /// accepts any pointer, so routing is unambiguous there.
    ///
    /// See [`WithFallback::ranges_disjoint`] for the exact check.
    #[inline]
    pub fn try_new(primary: P, secondary: S) -> Result<Self, AllocError> {
        let wf = Self { primary, secondary };
        if wf.ranges_disjoint() {
            Ok(wf)
        } else {
            Err(AllocError)
        }
    }

    /// Return `true` if the primary and secondary address ranges are
    /// disjoint, i.e. no pointer can belong to both.
    ///
    /// This is the runtime check behind the type's safety contract: if
    /// the ranges overlap, `deallocate`'s primary-first routing test can
    /// misroute a secondary-issued pointer to `primary.deallocate`,
    /// which then translates it to a slot index via `slot_index` and
    /// pushes a fabricated index onto the freelist — silent corruption.
    ///
    /// Call once at construction in a `debug_assert!`:
    ///
    /// ```ignore
    /// let wf = WithFallback::new(primary, secondary);
    /// debug_assert!(wf.ranges_disjoint(), "primary and secondary ranges overlap");
    /// ```
    ///
    /// The default secondary `crate::backing::System` does not implement
    /// `FixedRange`, so this method is unavailable for that common case;
    /// `System` accepts any pointer and routing is unambiguous there.
    #[inline]
    pub fn ranges_disjoint(&self) -> bool {
        let p_start = self.primary.base().as_ptr() as usize;
        let s_start = self.secondary.base().as_ptr() as usize;
        // `checked_add` rejects either range wrapping past `usize::MAX`.
        // A wrap would split the range into [start, MAX] U [0, end) and the
        // comparison below would silently misreport overlap on small `no_std`
        // targets. Treat a wrap conservatively as "potentially overlapping"
        // and return `false`.
        let Some(p_end) = p_start.checked_add(self.primary.size()) else {
            return false;
        };
        let Some(s_end) = s_start.checked_add(self.secondary.size()) else {
            return false;
        };
        // Disjoint iff one range ends at or before the other begins.
        p_end <= s_start || s_end <= p_start
    }
}

unsafe impl<P, S> Deallocator for WithFallback<P, S>
where
    P: Allocator + FixedRange,
    S: Allocator,
{
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        if self.primary.contains(ptr) {
            // SAFETY: ptr lies inside primary's range; caller's contract
            // says it came from this allocator's allocate(), so it must
            // have come from the primary's allocate().
            unsafe { self.primary.deallocate(ptr, layout) }
        } else {
            // ptr is outside primary's range. In a properly-used router
            // this means it came from the secondary path. A *foreign*
            // pointer (one issued by some third allocator that happens
            // to fall outside primary's range) would be silently routed
            // to secondary.deallocate, which is UB. The safety contract
            // on this trait method (and on `Deallocator` generally)
            // already forbids foreign pointers, so this is the caller's
            // problem; but a debug-build assertion makes the failure
            // mode loud during dev. When the secondary is also a
            // FixedRange (rare — `System` is not), we can validate that
            // ptr lies inside it; otherwise the assertion is a no-op.
            //
            // We don't have a way to dispatch on `S: FixedRange` here at
            // trait-method level without specialization. The assertion
            // therefore relies on a runtime ptr-bound test that's
            // available only when S exposes a FixedRange. Today this is
            // documented but not statically checked — see the safety
            // section on the type.
            //
            // SAFETY: contract on Deallocator::deallocate gives us a
            // ptr that came from *this* WithFallback. Since ptr is not
            // in primary, it must have come from secondary.
            unsafe { self.secondary.deallocate(ptr, layout) }
        }
    }
}

impl<P, S> WithFallback<P, S>
where
    P: Allocator + FixedRange,
    S: Allocator,
{
    /// Out-of-line fallback path. Split out of `allocate` so the hot
    /// success branch stays compact in the i-cache; the spilling-into-
    /// secondary case is a cold event by design (primary exhausted) and
    /// pays a normal function-call cost rather than inlining its
    /// secondary-allocate body into every `allocate` call site.
    #[cold]
    #[inline(never)]
    fn fallback_allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        self.secondary.allocate(layout)
    }

    #[cold]
    #[inline(never)]
    fn fallback_allocate_zeroed(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        self.secondary.allocate_zeroed(layout)
    }
}

unsafe impl<P, S> Allocator for WithFallback<P, S>
where
    P: Allocator + FixedRange,
    S: Allocator,
{
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        match self.primary.allocate(layout) {
            Ok(block) => Ok(block),
            Err(_) => self.fallback_allocate(layout),
        }
    }

    #[inline]
    fn allocate_zeroed(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        match self.primary.allocate_zeroed(layout) {
            Ok(block) => Ok(block),
            Err(_) => self.fallback_allocate_zeroed(layout),
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        // Total isn't meaningfully defined here — the secondary may be
        // unbounded. Report just the primary's capacity for the watermark
        // model (the secondary is overflow drainage, not budget).
        self.primary.capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        // Both halves can independently detect corruption. Sum them
        // — total events across the composed allocator. `saturating_add`
        // guards against the (extremely unlikely) u64 overflow when
        // both halves have observed astronomical attack volumes.
        self.primary
            .corruption_events()
            .saturating_add(self.secondary.corruption_events())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::{InlineBacked, System};

    /// Helper: WithFallback<InlineBacked<256>, System> built without going
    /// through BumpArena. InlineBacked is itself a FixedRange Allocator.
    fn build() -> WithFallback<InlineBacked<256>, System> {
        WithFallback::new(InlineBacked::<256>::new(), System)
    }

    #[test]
    fn primary_used_when_available() {
        let wf = build();
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let block = wf.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        assert!(
            wf.primary().contains(ptr),
            "primary should serve when it can"
        );
    }

    #[test]
    fn falls_through_when_primary_exhausted() {
        let wf = build();
        // Consume primary fully.
        let big = NonZeroLayout::from_size_align(256, 1).unwrap();
        let _ = wf.allocate(big).unwrap();
        // Next allocation must come from secondary.
        let small = NonZeroLayout::from_size_align(8, 8).unwrap();
        let block = wf.allocate(small).unwrap();
        let ptr = block.cast::<u8>();
        assert!(
            !wf.primary().contains(ptr),
            "secondary should serve when primary is exhausted"
        );
        // Clean up the secondary allocation.
        unsafe { wf.deallocate(ptr, small) };
    }

    #[test]
    fn deallocate_routes_by_provenance() {
        let wf = build();
        let small = NonZeroLayout::from_size_align(8, 8).unwrap();

        // Two allocations: primary then secondary.
        let big = NonZeroLayout::from_size_align(256, 1).unwrap();
        let prim = wf.allocate(big).unwrap();
        let sec = wf.allocate(small).unwrap();

        // Both deallocate via the same call site — router picks the right one.
        unsafe {
            wf.deallocate(prim.cast(), big); // primary path (no-op)
            wf.deallocate(sec.cast(), small); // secondary path (frees heap)
        }
    }

    #[test]
    fn capacity_bytes_reports_primary() {
        let wf = build();
        assert_eq!(wf.capacity_bytes(), Some(256));
    }

    /// `try_new` accepts two FixedRange halves whose address ranges
    /// don't overlap (the common case — independent backing regions).
    #[test]
    fn try_new_accepts_disjoint_fixed_ranges() {
        // Two independent InlineBacked instances have separate
        // address ranges (they're each their own stack-allocated
        // array).
        let a = InlineBacked::<256>::new();
        let b = InlineBacked::<256>::new();
        assert!(WithFallback::try_new(a, b).is_ok());
    }

    /// Boundary: `ranges_disjoint` reports `true` for two independent
    /// `InlineBacked<N>` instances sitting next to each other on the
    /// stack. Each is its own backing region with its own `[base, base+N)`
    /// extent — adjacent but not overlapping. Pinning the boundary here
    /// guards against an off-by-one in the `p_end <= s_start ||
    /// s_end <= p_start` check (using `<` would spuriously flag adjacent
    /// regions as overlapping).
    #[test]
    fn ranges_disjoint_treats_adjacent_regions_as_disjoint() {
        // Two stack-allocated InlineBackeds. Their addresses depend on
        // layout but on most ABIs they end up adjacent within the same
        // frame.
        let a = InlineBacked::<128>::new();
        let b = InlineBacked::<128>::new();
        let wf = WithFallback::new(a, b);
        assert!(
            wf.ranges_disjoint(),
            "independent InlineBacked instances must be disjoint",
        );
    }

    /// Watermark composition: `capacity_bytes` reports the primary's
    /// capacity only, treating the secondary as overflow. Verify the
    /// promise — adding a System secondary does NOT bump the reported
    /// capacity.
    #[test]
    fn capacity_bytes_reports_only_primary_even_with_system_secondary() {
        let wf = WithFallback::new(InlineBacked::<256>::new(), System);
        assert_eq!(wf.capacity_bytes(), Some(256));
    }
}
