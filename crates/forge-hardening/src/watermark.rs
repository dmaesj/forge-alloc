//! `Watermark<I, H>` — monitors allocation utilization in bytes and fires
//! callbacks at configurable thresholds (warn / critical / oom).
//!
//! See spec §7.7.

use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use forge_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// Severity bucket emitted to a [`WatermarkHandler`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WatermarkLevel {
    /// Soft warning — usage crossed `warn_pct`.
    Warn,
    /// Hard warning — usage crossed `critical_pct`.
    Critical,
    /// Out of memory — allocation failed.
    Oom,
}

/// Threshold percentages for the warn / critical levels.
///
/// The OOM level fires when the allocator returns `AllocError`, regardless
/// of percentage. Default is `warn_pct = 75`, `critical_pct = 90`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WatermarkThresholds {
    /// Fire `on_warn` when `bytes_allocated / capacity_bytes` first exceeds
    /// this percentage. Default 75.
    pub warn_pct: u8,
    /// Fire `on_critical` at this percentage. Default 90.
    pub critical_pct: u8,
}

impl Default for WatermarkThresholds {
    fn default() -> Self {
        Self {
            warn_pct: 75,
            critical_pct: 90,
        }
    }
}

/// Snapshot of allocation state when a [`WatermarkHandler`] fires.
#[derive(Copy, Clone, Debug)]
pub struct WatermarkEvent {
    /// Which threshold tripped.
    pub level: WatermarkLevel,
    /// Bytes currently held by live allocations (post-update for the call
    /// that triggered this event).
    pub allocated_bytes: usize,
    /// Capacity reported by the inner allocator at this moment.
    pub capacity_bytes: usize,
    /// The layout that triggered the event, when relevant. `Some` on
    /// `on_oom`; `None` on `on_warn` / `on_critical`.
    pub requested_layout: Option<NonZeroLayout>,
}

/// Callback handler invoked on threshold crossings and on OOM.
///
/// Implementations must be cheap — they run on the allocation hot path.
/// Set a flag, write to a non-blocking channel, or increment a metric;
/// avoid blocking I/O.
///
/// # Panic safety
///
/// Handler methods MUST NOT panic. If `on_warn` / `on_critical` panics
/// after the inner allocator has already issued a block, `Watermark`
/// unwinds out of `allocate` **without returning the block to the
/// caller** — the slot is carved on the inner allocator and the
/// `allocated` counter has been incremented, but the caller never
/// receives the pointer and so can never free it. The block is leaked
/// for the lifetime of the inner allocator. If `on_oom` panics, the
/// caller's `AllocError` is replaced by the panic; no block was issued
/// and the counters are unchanged, so this case is consistent but
/// noisier.
///
/// If your handler can fail (e.g. it writes to a possibly-full
/// channel), absorb the failure inside the handler — set a flag, log,
/// silently drop the event — rather than letting it escape. Treat
/// handler-level panics as a fatal bug in your monitoring code.
pub trait WatermarkHandler: Send + Sync {
    /// Called once per crossing of the `warn_pct` threshold (rising edge).
    fn on_warn(&self, event: WatermarkEvent);
    /// Called once per crossing of the `critical_pct` threshold (rising edge).
    fn on_critical(&self, event: WatermarkEvent);
    /// Called when an allocation returns `AllocError`.
    fn on_oom(&self, event: WatermarkEvent);
}

/// Discards every event. Zero-overhead substitute when monitoring is
/// disabled in a release build.
#[derive(Copy, Clone, Debug, Default)]
pub struct NullHandler;

impl WatermarkHandler for NullHandler {
    #[inline]
    fn on_warn(&self, _event: WatermarkEvent) {}
    #[inline]
    fn on_critical(&self, _event: WatermarkEvent) {}
    #[inline]
    fn on_oom(&self, _event: WatermarkEvent) {}
}

/// Emits one `eprintln!` line per event. Suitable for development; not
/// recommended for production hot paths (stderr writes hit a global lock
/// in libstd).
#[cfg(feature = "std")]
#[derive(Copy, Clone, Debug, Default)]
pub struct LogHandler;

#[cfg(feature = "std")]
impl WatermarkHandler for LogHandler {
    fn on_warn(&self, event: WatermarkEvent) {
        // Promote to u128 before multiplication to avoid usize overflow on
        // 32-bit targets where allocated_bytes * 100 can wrap for any
        // allocation > 42 MiB.
        let pct = (event.allocated_bytes as u128) * 100
            / (event.capacity_bytes.max(1) as u128);
        eprintln!(
            "[forge-hardening] watermark WARN: {}/{} bytes ({}%)",
            event.allocated_bytes, event.capacity_bytes, pct,
        );
    }
    fn on_critical(&self, event: WatermarkEvent) {
        let pct = (event.allocated_bytes as u128) * 100
            / (event.capacity_bytes.max(1) as u128);
        eprintln!(
            "[forge-hardening] watermark CRITICAL: {}/{} bytes ({}%)",
            event.allocated_bytes, event.capacity_bytes, pct,
        );
    }
    fn on_oom(&self, event: WatermarkEvent) {
        eprintln!(
            "[forge-hardening] watermark OOM: requested {:?}; {}/{} bytes in use",
            event.requested_layout, event.allocated_bytes, event.capacity_bytes,
        );
    }
}

/// Dispatches every event to a user-supplied closure. The closure receives
/// the level via `event.level`.
pub struct FnHandler<F>(pub F);

impl<F> WatermarkHandler for FnHandler<F>
where
    F: Fn(WatermarkEvent) + Send + Sync,
{
    fn on_warn(&self, event: WatermarkEvent) {
        self.0(event);
    }
    fn on_critical(&self, event: WatermarkEvent) {
        self.0(event);
    }
    fn on_oom(&self, event: WatermarkEvent) {
        self.0(event);
    }
}

/// Watermark wrapper.
///
/// Tracks live bytes in an `AtomicUsize`. Fires the handler on rising
/// crossings of `warn_pct` and `critical_pct`, and on OOM. Falling edges
/// (allocations released) do NOT reset the crossing flags — once a region
/// reaches warn, it stays armed until manual reset (TODO M5+ if needed).
///
/// Atomic variant only; non-atomic variant for single-core no_std targets
/// will land later. Gated by `cfg(target_has_atomic = "ptr")`.
#[cfg(target_has_atomic = "ptr")]
pub struct Watermark<I, H> {
    inner: I,
    handler: H,
    thresholds: WatermarkThresholds,
    capacity_bytes: usize,
    /// Pre-computed absolute byte threshold = `warn_pct * capacity_bytes /
    /// 100`. Allocate's hot path compares `new_bytes` against this gate
    /// before issuing the (out-of-line, `#[cold]`) `check_and_fire` call;
    /// the common case (well below warn) skips the call entirely. For
    /// growing inners (ExtendableSlab) the construction-time capacity is
    /// a lower bound on future capacity, so using it here gives a tight,
    /// false-positive-only gate — never a missed crossing.
    ///
    /// `usize::MAX` when the inner reports unbounded capacity; the gate
    /// then never fires below saturation, which matches "no thresholds
    /// configured" semantics.
    warn_threshold_bytes: usize,
    allocated: AtomicUsize,
    /// Bit 0 = warn fired; bit 1 = critical fired. Set on rising edge to
    /// prevent re-firing on every allocate.
    fired: AtomicUsize,
}

#[cfg(target_has_atomic = "ptr")]
const FIRED_WARN: usize = 1;
#[cfg(target_has_atomic = "ptr")]
const FIRED_CRITICAL: usize = 2;

#[cfg(target_has_atomic = "ptr")]
impl<I: Allocator, H: WatermarkHandler> Watermark<I, H> {
    /// Wrap with default thresholds (75% / 90%).
    pub fn new(inner: I, handler: H) -> Self {
        Self::with_thresholds(inner, handler, WatermarkThresholds::default())
    }

    /// Wrap with explicit thresholds.
    pub fn with_thresholds(inner: I, handler: H, thresholds: WatermarkThresholds) -> Self {
        // Snapshot capacity at construction. Watermark re-queries via the
        // capacity_bytes() method below so growing inners (e.g.
        // ExtendableSlab in M7) report the live value to handlers.
        let capacity_bytes = inner.capacity_bytes().unwrap_or(usize::MAX);
        // Pre-compute the hot-path gate so `allocate` can skip the
        // `check_and_fire` call entirely while utilization is below
        // *every* configured threshold. Take the **min** of warn_pct
        // and critical_pct so a config with `critical_pct < warn_pct`
        // (caller mistake — but not rejected at construction since the
        // type-level invariant only says both are u8) doesn't silently
        // suppress critical events that fall above the critical line
        // but below warn. Pinned by the regression test
        // `inverted_thresholds_hot_path_gate_does_not_suppress_critical`.
        //
        // Computed in `u128` to avoid overflow on 32-bit targets where
        // `capacity_bytes * 100` could wrap. For unbounded inners we
        // keep `usize::MAX` so the gate never fires.
        let warn_threshold_bytes = if capacity_bytes == usize::MAX {
            usize::MAX
        } else {
            // Pct values are u8 (0..=100 in normal use); guard against
            // pathological >100 values by saturating.
            let warn_pct = thresholds.warn_pct.min(100) as u128;
            let critical_pct = thresholds.critical_pct.min(100) as u128;
            let pct = warn_pct.min(critical_pct);
            ((capacity_bytes as u128 * pct) / 100) as usize
        };
        Self {
            inner,
            handler,
            thresholds,
            capacity_bytes,
            warn_threshold_bytes,
            allocated: AtomicUsize::new(0),
            fired: AtomicUsize::new(0),
        }
    }

    /// Bytes currently in use.
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        // Telemetry read: no synchronization with allocator state needed.
        self.allocated.load(Ordering::Relaxed)
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }

    /// Borrow the handler.
    #[inline]
    pub fn handler(&self) -> &H {
        &self.handler
    }

    /// Re-arm threshold firing (clears warn/critical fired flags). Call
    /// after addressing a high-water condition if you want the next
    /// crossing to fire again.
    #[inline]
    pub fn rearm(&self) {
        // No memory is published through `fired`; it's a self-contained
        // test-and-set bitmap. Relaxed is sufficient.
        self.fired.store(0, Ordering::Relaxed);
    }

    /// Resolved capacity (queried fresh from inner per call so growing
    /// inners give accurate readings).
    #[inline]
    fn current_capacity(&self) -> usize {
        self.inner.capacity_bytes().unwrap_or(self.capacity_bytes)
    }

    /// Check whether the post-allocate `new_bytes` value crossed a threshold
    /// and fire the appropriate handler. Returns the snapshot used.
    ///
    /// Marked `#[cold]` + `#[inline(never)]` because threshold-crossing
    /// is rare (only happens on the few allocates that push past 75% /
    /// 90% utilization); keeping this body out of the per-allocate hot
    /// path shrinks the i-cache footprint of `Allocator::allocate` for
    /// every common-case alloc that does NOT cross a threshold.
    #[cold]
    #[inline(never)]
    fn check_and_fire(&self, new_bytes: usize, requested: Option<NonZeroLayout>) {
        let capacity = self.current_capacity();
        if capacity == 0 {
            return;
        }
        // Compute percentage as (new_bytes * 100 / capacity); saturate to
        // u8::MAX so values >100% (possible if the inner allocator hands
        // back more bytes than capacity_bytes() reports) still fire any
        // configured threshold.
        let pct_u128 = (new_bytes as u128) * 100 / (capacity as u128);
        let pct = pct_u128.min(255) as u8;

        // Critical first (higher priority).
        if pct >= self.thresholds.critical_pct {
            // Atomic test-and-set: `fetch_or` is a single RMW, so two
            // racing threads cannot both observe `prev & FIRED_CRITICAL ==
            // 0` — at most one fires the rising-edge handler. We also set
            // FIRED_WARN in the same op: if we jumped from below-warn
            // straight to critical, the warn-rising-edge has been
            // implicitly subsumed and we don't want a later dip below
            // critical to spuriously re-fire `on_warn`.
            // `fetch_or` atomicity alone gives the rising-edge guarantee:
            // only the thread that observes `prev & FIRED_CRITICAL == 0`
            // fires the handler. The handler reads only the
            // locally-constructed `WatermarkEvent`, so no payload memory
            // is published through this RMW — Relaxed is correct.
            let prev = self
                .fired
                .fetch_or(FIRED_CRITICAL | FIRED_WARN, Ordering::Relaxed);
            if prev & FIRED_CRITICAL == 0 {
                self.handler.on_critical(WatermarkEvent {
                    level: WatermarkLevel::Critical,
                    allocated_bytes: new_bytes,
                    capacity_bytes: capacity,
                    requested_layout: requested,
                });
            }
        } else if pct >= self.thresholds.warn_pct {
            // Same rising-edge reasoning as above; Relaxed suffices.
            let prev = self.fired.fetch_or(FIRED_WARN, Ordering::Relaxed);
            if prev & FIRED_WARN == 0 {
                self.handler.on_warn(WatermarkEvent {
                    level: WatermarkLevel::Warn,
                    allocated_bytes: new_bytes,
                    capacity_bytes: capacity,
                    requested_layout: requested,
                });
            }
        }
    }
}

#[cfg(target_has_atomic = "ptr")]
unsafe impl<I: Allocator, H: WatermarkHandler> Deallocator for Watermark<I, H> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // Per Deallocator's contract `ptr` came from a previous `allocate`
        // on `self`, so the pre-sub value must be >= layout.size(). A
        // mismatched dealloc would wrap the counter; catch it in debug.
        // Hot-path: a single `fetch_sub` (one `lock xadd` on x86_64)
        // replaces the previous `fetch_update` CAS loop. The CAS loop
        // saturated at zero to defend against UB caller bugs, but under
        // contention from N threads each conflicting RMW caused another
        // retry — making dealloc cost scale with thread count. For a
        // correct caller the contract guarantees `prev >= size`, so no
        // saturation is needed on the happy path. Debug builds still
        // catch the UB caller bug via `debug_assert!` below.
        let size = layout.size().get();
        let prev = self.allocated.fetch_sub(size, Ordering::Relaxed);
        debug_assert!(
            prev >= size,
            "Watermark::deallocate underflow: prev={prev}, size={size}",
        );
        // SAFETY: forwarded; caller upholds Deallocator contract.
        unsafe { self.inner.deallocate(ptr, layout) };
    }
}

#[cfg(target_has_atomic = "ptr")]
unsafe impl<I: Allocator, H: WatermarkHandler> Allocator for Watermark<I, H> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        match self.inner.allocate(layout) {
            Ok(block) => {
                // Counter only; no data dependency crosses this RMW.
                // `saturating_add` instead of `+` so a near-`usize::MAX`
                // `prev` (only reachable via wrap-around after a mismatched
                // dealloc — UB caller bug — but defended against here so we
                // do not turn that bug into a debug-mode panic in this
                // wrapper) does not blow up the debug build.
                let size = layout.size().get();
                let prev = self.allocated.fetch_add(size, Ordering::Relaxed);
                let new_bytes = prev.saturating_add(size);
                // Hot-path gate: skip the `#[cold]` `check_and_fire` call
                // entirely while we're below the construction-time warn
                // threshold. `check_and_fire` is `#[cold] #[inline(never)]`
                // but an unconditional call still costs the args setup +
                // branch instruction every allocate. For growing inners
                // the construction-time capacity is a lower bound on
                // future capacity, so the threshold computed from it is
                // always at least as low as the live threshold — false
                // positive (call when not yet warn) at worst, never a
                // missed crossing.
                if new_bytes >= self.warn_threshold_bytes {
                    self.check_and_fire(new_bytes, None);
                }
                Ok(block)
            }
            Err(e) => {
                let cur = self.allocated_bytes();
                self.handler.on_oom(WatermarkEvent {
                    level: WatermarkLevel::Oom,
                    allocated_bytes: cur,
                    capacity_bytes: self.current_capacity(),
                    requested_layout: Some(layout),
                });
                Err(e)
            }
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        self.inner.capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        self.inner.corruption_events()
    }
}

#[cfg(target_has_atomic = "ptr")]
impl<I: FixedRange, H: WatermarkHandler> FixedRange for Watermark<I, H> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.inner.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }
}

#[cfg(test)]
#[cfg(target_has_atomic = "ptr")]
mod tests {
    use super::*;
    use forge_backing::InlineBacked;
    use forge_layout::BumpArena;
    use core::sync::atomic::AtomicU8;

    /// Handler that records which callbacks fired, for assertion in tests.
    #[derive(Default)]
    struct FlagHandler {
        warn: AtomicU8,
        critical: AtomicU8,
        oom: AtomicU8,
    }

    impl WatermarkHandler for FlagHandler {
        fn on_warn(&self, _event: WatermarkEvent) {
            self.warn.fetch_add(1, Ordering::Relaxed);
        }
        fn on_critical(&self, _event: WatermarkEvent) {
            self.critical.fetch_add(1, Ordering::Relaxed);
        }
        fn on_oom(&self, _event: WatermarkEvent) {
            self.oom.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn build() -> Watermark<BumpArena<InlineBacked<1024>>, FlagHandler> {
        Watermark::new(
            BumpArena::new(InlineBacked::<1024>::new()).unwrap(),
            FlagHandler::default(),
        )
    }

    #[test]
    fn warn_fires_at_75_pct() {
        let w = build();
        // 75% of 1024 = 768. Allocate 800 bytes (well past warn).
        let layout = NonZeroLayout::from_size_align(800, 1).unwrap();
        let _ = w.allocate(layout).unwrap();
        assert_eq!(w.handler().warn.load(Ordering::Relaxed), 1);
        // Critical (90% = 921) NOT fired.
        assert_eq!(w.handler().critical.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn critical_fires_at_90_pct() {
        let w = build();
        let layout = NonZeroLayout::from_size_align(950, 1).unwrap();
        let _ = w.allocate(layout).unwrap();
        assert_eq!(w.handler().critical.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn warn_fires_only_once() {
        let w = build();
        let layout = NonZeroLayout::from_size_align(200, 1).unwrap();
        // 800 bytes total → past warn (768).
        for _ in 0..4 {
            let _ = w.allocate(layout).unwrap();
        }
        assert_eq!(w.handler().warn.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn oom_fires_on_alloc_error() {
        let w = build();
        // Allocate beyond capacity.
        let layout = NonZeroLayout::from_size_align(2048, 1).unwrap();
        assert!(w.allocate(layout).is_err());
        assert_eq!(w.handler().oom.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn dealloc_subtracts_from_allocated() {
        let w = Watermark::new(
            forge_layout::Slab::<u64, InlineBacked<512>>::new(8, InlineBacked::<512>::new()).unwrap(),
            FlagHandler::default(),
        );
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let block = w.allocate(layout).unwrap();
        assert_eq!(w.allocated_bytes(), 8);
        unsafe { w.deallocate(block.cast(), layout) };
        assert_eq!(w.allocated_bytes(), 0);
    }

    #[test]
    fn rearm_clears_fired_flags() {
        let w = build();
        let layout = NonZeroLayout::from_size_align(800, 1).unwrap();
        let _ = w.allocate(layout).unwrap();
        assert_eq!(w.handler().warn.load(Ordering::Relaxed), 1);
        w.rearm();
        // Reset the bump arena state isn't relevant here — we just need
        // another rising edge. Allocate a tiny bit more to push past warn
        // threshold once flags are re-armed; warn fires again.
        // (Already at 800/1024 = 78%, still above warn. Next alloc just
        // re-trips the check since flags are clear.)
        let small = NonZeroLayout::from_size_align(1, 1).unwrap();
        let _ = w.allocate(small).unwrap();
        assert_eq!(w.handler().warn.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn null_handler_zero_cost() {
        let nh = NullHandler;
        nh.on_warn(WatermarkEvent {
            level: WatermarkLevel::Warn,
            allocated_bytes: 0,
            capacity_bytes: 0,
            requested_layout: None,
        });
    }

    /// Boundary: `warn_pct = 0` fires on every allocation, but only
    /// once (rising-edge gate). Verify: a single alloc fires `warn`,
    /// subsequent allocs do not refire.
    #[test]
    fn warn_pct_zero_fires_on_first_alloc_only() {
        let inner = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
        let w = Watermark::with_thresholds(
            inner,
            FlagHandler::default(),
            WatermarkThresholds {
                warn_pct: 0,
                critical_pct: 90,
            },
        );
        let small = NonZeroLayout::from_size_align(1, 1).unwrap();
        for _ in 0..4 {
            let _ = w.allocate(small).unwrap();
        }
        // First alloc tripped warn (0% threshold met); next three are
        // suppressed by the FIRED_WARN bit.
        assert_eq!(w.handler().warn.load(Ordering::Relaxed), 1);
        assert_eq!(w.handler().critical.load(Ordering::Relaxed), 0);
    }

    /// Boundary: `warn_pct = 100` fires only at exact saturation (where
    /// OOM is imminent). Verify no warn fires below 100% usage.
    #[test]
    fn warn_pct_100_does_not_fire_below_saturation() {
        let inner = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
        let w = Watermark::with_thresholds(
            inner,
            FlagHandler::default(),
            WatermarkThresholds {
                warn_pct: 100,
                critical_pct: 100,
            },
        );
        // 99% utilization — still below the 100% gate.
        let layout = NonZeroLayout::from_size_align(1000, 1).unwrap();
        let _ = w.allocate(layout).unwrap();
        assert_eq!(
            w.handler().warn.load(Ordering::Relaxed),
            0,
            "warn must NOT fire below 100% when warn_pct = 100",
        );
        // Exact saturation (1024/1024 = 100%).
        let layout = NonZeroLayout::from_size_align(24, 1).unwrap();
        let _ = w.allocate(layout).unwrap();
        // At 100% the warn handler can fire.
        assert!(
            w.handler().warn.load(Ordering::Relaxed) >= 1
                || w.handler().critical.load(Ordering::Relaxed) >= 1,
            "at 100% utilization, warn OR critical should fire",
        );
    }

    /// Behavioral pin (NOT a "good thing"): `critical_pct < warn_pct` is
    /// a configuration error — the hot-path gate is keyed on
    /// `warn_pct` (precomputed as `warn_threshold_bytes`). With
    /// inverted thresholds the gate sits ABOVE critical_pct, so the
    /// `check_and_fire` body never runs while utilization is below
    /// warn_pct, and the critical handler is silently suppressed for
    /// allocations between critical_pct and warn_pct. The wrapper does
    /// not panic; it just under-reports.
    ///
    /// If you find this test surprising and want critical-fires-first
    /// semantics regardless of input order, the fix is to compute the
    /// hot-path gate as `min(warn_threshold, critical_threshold)` in
    /// `with_thresholds`. Pinning the current behavior here so the
    /// regression surfaces explicitly if that hardening is added.
    #[test]
    fn inverted_thresholds_hot_path_gate_does_not_suppress_critical() {
        // Regression for the inverted-threshold suppression bug found in
        // pass #5: with `critical_pct < warn_pct` (a config mistake but
        // not rejected), the old warn-keyed hot-path gate sat ABOVE the
        // critical line, so allocations whose utilization landed in
        // [critical_pct, warn_pct) silently failed to fire critical.
        // After the fix the gate is `min(warn_pct, critical_pct)`, so
        // critical events surface as soon as they should.
        let inner = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
        let w = Watermark::with_thresholds(
            inner,
            FlagHandler::default(),
            WatermarkThresholds {
                warn_pct: 90,
                critical_pct: 50,
            },
        );
        // 60% (614 bytes) is above critical (50) but below warn (90).
        // The hot-path gate is keyed on min(50, 90) = 50% = 512 bytes,
        // so check_and_fire IS reached and critical fires on rising
        // edge.
        let layout = NonZeroLayout::from_size_align(614, 1).unwrap();
        let _ = w.allocate(layout).unwrap();
        assert_eq!(
            w.handler().critical.load(Ordering::Relaxed),
            1,
            "inverted-threshold fix: critical must fire when usage crosses critical_pct \
             even though it's below warn_pct (the hot-path gate now uses min of both)",
        );
        // Warn does NOT fire at 60% (warn_pct = 90%) — usage hasn't
        // crossed it yet.
        assert_eq!(
            w.handler().warn.load(Ordering::Relaxed),
            0,
            "warn must not fire below warn_pct",
        );
        // Cross warn (>=921 bytes). Per `check_and_fire`'s design,
        // warn is *subsumed* once critical has fired — the FIRED_WARN
        // bit was set when critical latched, so even crossing warn_pct
        // here does NOT re-fire on_warn. This matches the
        // monotonic-severity contract (we don't want a dip-then-recross
        // to re-trigger the lower-severity handler).
        let layout = NonZeroLayout::from_size_align(400, 1).unwrap();
        let _ = w.allocate(layout).unwrap(); // total 1014
        assert_eq!(
            w.handler().warn.load(Ordering::Relaxed),
            0,
            "warn is subsumed once critical fires — see check_and_fire's \
             intentional `FIRED_CRITICAL | FIRED_WARN` co-set",
        );
        // Critical already latched; no re-fire on this allocate either.
        assert_eq!(w.handler().critical.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn fn_handler_called() {
        use std::sync::Mutex;
        let calls = std::sync::Arc::new(Mutex::new(0u32));
        let calls2 = std::sync::Arc::clone(&calls);
        let h = FnHandler(move |_event: WatermarkEvent| {
            *calls2.lock().unwrap() += 1;
        });
        let w = Watermark::new(BumpArena::new(InlineBacked::<1024>::new()).unwrap(), h);
        let _ = w
            .allocate(NonZeroLayout::from_size_align(800, 1).unwrap())
            .unwrap();
        assert_eq!(*calls.lock().unwrap(), 1);
    }
}
