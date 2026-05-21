//! `Statistics<I>` — atomic counters for allocation observability.
//!
//! Wrap any allocator during development (or production, when the feature
//! is opted in) to see allocation patterns: total counts, current and peak
//! byte usage, and failure counts. The act of wrapping IS the opt-in; an
//! unwrapped allocator pays zero cost.
//!
//! See `docs/ARCHITECTURE.md` for design context.

use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use forge_alloc_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// Cache-line aligned wrapper. Prevents false sharing between counters
/// hammered from different code paths (alloc vs dealloc vs failure) when
/// `Statistics` wraps a multi-thread allocator like `SharedBumpArena`.
///
/// 64-byte alignment matches a typical x86_64 / AArch64 L1 line; the
/// `#[repr(C, align(64))]` rounds the total struct size up so the next
/// field also begins on a fresh line.
#[repr(C, align(64))]
#[derive(Debug, Default)]
pub struct CachePadded<T>(T);

impl<T> CachePadded<T> {
    /// Wrap a value with cache-line alignment padding.
    #[inline]
    pub const fn new(v: T) -> Self {
        Self(v)
    }
}

impl<T> core::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

/// Snapshot of allocation activity. All counters are atomic; reading is
/// `Ordering::Relaxed` because counter values are advisory (operators read
/// them for diagnostics, not for ordering guarantees).
///
/// Each counter is wrapped in [`CachePadded`] so concurrent updates from
/// different code paths (alloc / dealloc / failure) don't ping-pong as a
/// single cache line when `Statistics` wraps a multi-thread allocator
/// (e.g. `Statistics<SharedBumpArena>`). The public field type therefore
/// reads as `CachePadded<AtomicUsize>`; auto-deref keeps the call sites
/// (`stats.total_allocations.fetch_add(...)`) unchanged.
///
/// # Width: `AtomicUsize`, not `AtomicU64`
///
/// Each counter is `AtomicUsize` so this crate compiles on 32-bit
/// bare-metal targets (Cortex-M3/M4, `thumbv7em-none-eabihf`,
/// `wasm32-unknown-unknown` without the `atomics` feature) that lack
/// native 64-bit atomic ops. The convenience helpers ([`current_bytes`],
/// [`peak_bytes`], [`live_count`]) widen each load to `u64` at the
/// boundary so the public read-API is uniform across host widths.
///
/// **Practical impact on 32-bit:**
///   - `bytes_allocated` / `bytes_peak`: capped at `usize::MAX` (4 GiB
///     on 32-bit), which equals the address-space ceiling anyway —
///     a tighter cap is impossible on these targets.
///   - `total_allocations` / `total_deallocations` / `failures`: cap at
///     `u32::MAX ≈ 4.3 B`. For long-running 32-bit deployments, the
///     counter wraps after that many ops; values are advisory only.
///   - `corruption_events`: cap at `u32::MAX`. Even at one event per
///     microsecond (already an unrealistic attack rate) the counter
///     would not wrap for ~71 minutes; real workloads see ≪1 event/year.
///
/// Marked `#[non_exhaustive]` so additional observability counters can
/// be added in future releases without a breaking change.
///
/// [`current_bytes`]: AllocStats::current_bytes
/// [`peak_bytes`]: AllocStats::peak_bytes
/// [`live_count`]: AllocStats::live_count
#[non_exhaustive]
#[derive(Debug)]
pub struct AllocStats {
    /// Total successful `allocate` calls observed by this wrapper.
    pub total_allocations: CachePadded<AtomicUsize>,
    /// Total `deallocate` calls observed by this wrapper.
    pub total_deallocations: CachePadded<AtomicUsize>,
    /// Bytes currently held by live allocations.
    pub bytes_allocated: CachePadded<AtomicUsize>,
    /// High-water mark of `bytes_allocated`.
    pub bytes_peak: CachePadded<AtomicUsize>,
    /// Failed `allocate` calls (returned `AllocError`).
    pub failures: CachePadded<AtomicUsize>,
    /// Detected freelist / metadata corruption events, mirrored from
    /// the inner allocator's [`Allocator::corruption_events`] counter
    /// via `fetch_max` on each allocate/deallocate call through this
    /// wrapper.
    ///
    /// The inner allocator is the source of truth (each corruption-
    /// detection site bumps its allocator-local counter at the moment
    /// of detection); this mirror lets readers of `AllocStats` see the
    /// corruption count alongside the other counters without needing to
    /// reach through to the inner.
    ///
    /// **Eventually consistent**: between calls to this wrapper, the
    /// inner counter may have advanced. The mirror is updated on every
    /// allocate and deallocate through this wrapper.
    pub corruption_events: CachePadded<AtomicUsize>,
}

impl AllocStats {
    /// Construct fresh zeroed counters.
    pub const fn new() -> Self {
        Self {
            total_allocations: CachePadded::new(AtomicUsize::new(0)),
            total_deallocations: CachePadded::new(AtomicUsize::new(0)),
            bytes_allocated: CachePadded::new(AtomicUsize::new(0)),
            bytes_peak: CachePadded::new(AtomicUsize::new(0)),
            failures: CachePadded::new(AtomicUsize::new(0)),
            corruption_events: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    /// Bytes currently in use. Widened to `u64` at the boundary so
    /// callers get a uniform read-API across 32-bit and 64-bit hosts;
    /// the underlying counter is `AtomicUsize` (capped at `usize::MAX`).
    #[inline]
    pub fn current_bytes(&self) -> u64 {
        self.bytes_allocated.load(Ordering::Relaxed) as u64
    }

    /// Peak bytes ever in use during the wrapper's lifetime. Widened to
    /// `u64` at the boundary — see [`current_bytes`](Self::current_bytes).
    #[inline]
    pub fn peak_bytes(&self) -> u64 {
        self.bytes_peak.load(Ordering::Relaxed) as u64
    }

    /// Net live allocation count (allocations − deallocations).
    #[inline]
    pub fn live_count(&self) -> i64 {
        let a = self.total_allocations.load(Ordering::Relaxed) as i64;
        let d = self.total_deallocations.load(Ordering::Relaxed) as i64;
        a - d
    }
}

impl Default for AllocStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper that records allocation activity in [`AllocStats`].
///
/// `Send + Sync` if `I: Send + Sync`. Atomic counters are themselves `Sync`.
///
/// # Accounting invariant
///
/// `Statistics` records bytes by the layout the **outer caller** passed in.
/// Wrappers below `Statistics` that pass the layout through unchanged
/// (`PoisonOnFree`, `Quarantine`, `Watermark`) preserve this — the bytes
/// counted equal the bytes the caller requested. Wrappers below
/// `Statistics` that **inflate** the inner layout (`Canary`, `CacheJitter`,
/// `HugePageAligned`, `SplitMetadata`) consume more inner-allocator bytes
/// than the counter reports; the counter therefore reflects "bytes the
/// user asked for", not "bytes the underlying region holds". If you need
/// the latter, wrap `Statistics` INSIDE the layout-inflating wrapper:
/// `Canary<Statistics<Slab<T>>>` counts what Slab actually carved. The
/// recommended position for `Statistics` (`Statistics<PoisonOnFree<Slab>>`)
/// treats "bytes the user asked for" as the right number to
/// surface to operators; flip the nesting only if you specifically need
/// physical accounting.
///
/// # API-misuse compile-failures (pinned)
///
/// `Statistics<I>` inherits the `Sync` property of its inner allocator.
/// Wrapping a `!Sync` allocator (such as `Slab`, whose `UnsafeCell` free
/// list head makes it `!Sync` by design) does **not** silently upgrade
/// it to `Sync`. Calling `stats()` on a shared reference across threads
/// when the inner is `!Sync` is therefore rejected at compile time:
///
/// ```compile_fail
/// // FAILS TO COMPILE: `Slab` is `!Sync` (UnsafeCell on the freelist
/// // head), so `Statistics<Slab<...>>` is also `!Sync`, and
/// // `assert_sync` rejects it.
/// use forge_alloc::InlineBacked;
/// use forge_alloc::Statistics;
/// use forge_alloc::Slab;
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<Statistics<Slab<u64, InlineBacked<512>>>>();
/// ```
pub struct Statistics<I> {
    inner: I,
    stats: AllocStats,
}

impl<I> Statistics<I> {
    /// Wrap.
    #[inline]
    pub const fn new(inner: I) -> Self {
        Self {
            inner,
            stats: AllocStats::new(),
        }
    }

    /// Borrow the counters. Read-only access — counters are updated by the
    /// allocator's own methods.
    #[inline]
    pub fn stats(&self) -> &AllocStats {
        &self.stats
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }
}

unsafe impl<I: Allocator> Deallocator for Statistics<I> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // Update before forwarding so a panic inside `inner.deallocate`
        // doesn't leave the counters in a misleading state if exception
        // unwinding is enabled. (Both orderings are defensible; "before"
        // is the convention in libstd's deallocator wrappers.)
        //
        // Panic-safety note: if `inner.deallocate` panics
        // (e.g. a `Canary` corruption check below us), the
        // `total_deallocations` and `bytes_allocated` counters
        // already reflect the dealloc that didn't actually complete.
        // Counters are advisory; under a panicking inner the program
        // is already in an undefined operational state (corruption
        // detected, lower-layer assertion fired), so the counter skew
        // is acceptable. Do not key allocator correctness on the
        // counter values.
        //
        // The `corruption_events` mirror is polled below — also BEFORE
        // forwarding — for the additional reason that the inner counter
        // may have been bumped by earlier silent-disarm events that this
        // call would otherwise have surfaced post-forward; a panicking
        // forward would suppress that surfacing entirely.
        self.stats
            .total_deallocations
            .fetch_add(1, Ordering::Relaxed);
        // Hot-path: a single `fetch_sub` is a single locked-add on x86_64
        // (`lock xadd` with a negated operand). The previous implementation
        // used `fetch_update` (a CAS loop) to saturate on caller UB; under
        // contention that retries on every conflicting RMW, which makes
        // dealloc cost scale with thread count. Per the Deallocator
        // contract `ptr` was issued by a previous `allocate(layout)` on
        // `self`, so `prev >= size` always holds for a correct caller —
        // no saturation is needed on the happy path. Debug builds still
        // catch the UB caller bug via `debug_assert!` below.
        // `layout.size()` is already `NonZeroUsize`, so the counter
        // width and the increment type now match natively — no `as u64`
        // cast needed since the switch to `AtomicUsize`.
        let size = layout.size().get();
        let prev_for_assert = self
            .stats
            .bytes_allocated
            .fetch_sub(size, Ordering::Relaxed);
        debug_assert!(
            prev_for_assert >= size,
            "Statistics::deallocate underflow: prev={prev_for_assert}, size={size}",
        );
        // Mirror inner's corruption counter BEFORE forwarding so that
        // any prior silent-disarm events that have not yet been folded
        // into our mirror (a Slab freelist corruption detected during
        // an earlier `allocate` call, for example) are surfaced even
        // if `inner.deallocate` ends up panicking (e.g. a Canary check
        // below us). The mirror is monotonic via `fetch_max`, so missing
        // an update is recoverable on any subsequent call; what is NOT
        // recoverable is the operator's snapshot read happening between
        // the bump and the panic. Polling pre-forward closes that window.
        //
        // Per-call observation: under panic=abort (typical hardening
        // build), an inner panic terminates the process before any reader
        // can observe; under panic=unwind, the mirror reflects all events
        // committed up to the call boundary.
        mirror_corruption(
            &self.stats.corruption_events,
            self.inner.corruption_events(),
        );
        // SAFETY: forwarded; caller upholds Deallocator contract on `self`.
        unsafe { self.inner.deallocate(ptr, layout) };
    }
}

/// Apply `inner_val` to the `mirror` counter under a `fetch_max`-style
/// monotonic update.
///
/// Adds a fast-path that skips the locked RMW when `inner_val` does not
/// advance the mirror — on x86_64 `AtomicUsize::fetch_max` lowers to a
/// `lock cmpxchg` CAS loop (there is no native `lock max`), so the
/// steady-state no-corruption case (`inner_val == 0` against a mirror
/// at `0`) would otherwise pay one locked CAS per allocate AND per
/// deallocate. The `Relaxed` load is a plain `mov` and short-circuits
/// the cost.
///
/// Race-safe under the `fetch_max` semantics: a concurrent thread may
/// advance the mirror between our load and a (skipped) write, but our
/// skipped write would also have been a no-op against that larger value.
/// The inner counter is the source of truth; the mirror is monotonic
/// and eventually consistent.
///
/// **Width clamp.** `inner_val: u64` matches the trait return; the
/// mirror is `AtomicUsize` for 32-bit portability. We clamp the u64 to
/// `usize::MAX` before the fetch_max so a hypothetical inner counter
/// already past 4.3 B (on a 32-bit host, only reachable when the inner
/// is an aggregator like `WithFallback` that saturating-sums multiple
/// counters into u64) doesn't truncate-wrap into a smaller mirror value.
/// On 64-bit hosts the clamp is a no-op.
#[inline]
fn mirror_corruption(mirror: &AtomicUsize, inner_val: u64) {
    // Clamp first so the comparison and write are both honest on
    // 32-bit hosts. The `min` is a free comparison on all targets.
    let inner_clamped = inner_val.min(usize::MAX as u64) as usize;
    if inner_clamped > mirror.load(Ordering::Relaxed) {
        let _ = mirror.fetch_max(inner_clamped, Ordering::Relaxed);
    }
}

unsafe impl<I: Allocator> Allocator for Statistics<I> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let result = self.inner.allocate(layout);
        // Mirror inner's corruption counter. We do this even on the
        // Err path: a silent-disarm corruption may have triggered
        // before the inner ultimately ran out of capacity, and the
        // operator's first signal is the counter rising.
        //
        // Hot-path: the polled read goes through a load-and-skip-if-no-
        // advance helper (`mirror_corruption`) to avoid an unconditional
        // locked CAS on every successful allocate. See the helper's doc
        // comment for the rationale; in the steady-state no-corruption
        // case `inner_val == mirror == 0` and the locked path is skipped.
        mirror_corruption(
            &self.stats.corruption_events,
            self.inner.corruption_events(),
        );
        match result {
            Ok(block) => {
                self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
                // `saturating_add` rather than `+` so a near-`usize::MAX`
                // `prev` (only reachable via wrap-around after a mismatched
                // dealloc — UB caller bug — but defended in
                // `deallocate` below) does not turn that bug into a
                // debug-mode panic here.
                //
                // Width: counter is `AtomicUsize` (for 32-bit portability);
                // `layout.size().get()` is already `usize`, so this is
                // width-matched with no cast.
                let size = layout.size().get();
                let prev = self
                    .stats
                    .bytes_allocated
                    .fetch_add(size, Ordering::Relaxed);
                let new = prev.saturating_add(size);
                // `new` is THIS thread's local "post-add" value. Under
                // contention another thread's add may already have advanced
                // the global counter past `new` — that thread's own
                // fetch_max call will update the peak with its own (larger)
                // local `new`, so the final peak is still monotonic and
                // never below the true high-water mark. Relaxed ordering is
                // fine for advisory counters.
                //
                // Fast-path: `fetch_max` on x86_64 lowers to a CAS loop
                // (`lock cmpxchg`) which contends with every other
                // wrapper-thread's CAS. Read peak first and skip the CAS
                // entirely when our `new` doesn't actually move the
                // high-water mark — common case during steady-state
                // operation where `new` is well below the long-run peak.
                let peak_now = self.stats.bytes_peak.load(Ordering::Relaxed);
                if new > peak_now {
                    self.stats.bytes_peak.fetch_max(new, Ordering::Relaxed);
                }
                Ok(block)
            }
            Err(e) => {
                self.stats.failures.fetch_add(1, Ordering::Relaxed);
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
        // Forward to inner — the source of truth. The mirror on
        // `self.stats.corruption_events` is the observable artifact
        // for AllocStats readers; the trait method returns the live
        // count.
        self.inner.corruption_events()
    }
}

impl<I: FixedRange> FixedRange for Statistics<I> {
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
mod tests {
    use super::*;
    use crate::backing::InlineBacked;
    use crate::layout::Slab;

    fn build() -> Statistics<Slab<u64, InlineBacked<512>>> {
        Statistics::new(Slab::new(16, InlineBacked::<512>::new()).unwrap())
    }

    #[test]
    fn counts_increase_with_allocations() {
        let s = build();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        assert_eq!(s.stats().total_allocations.load(Ordering::Relaxed), 0);
        let _ = s.allocate(layout).unwrap();
        let _ = s.allocate(layout).unwrap();
        assert_eq!(s.stats().total_allocations.load(Ordering::Relaxed), 2);
        assert_eq!(s.stats().current_bytes(), 16);
    }

    #[test]
    fn peak_updates_above_current() {
        let s = build();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = s.allocate(layout).unwrap();
        let _b = s.allocate(layout).unwrap();
        assert_eq!(s.stats().peak_bytes(), 16);
        unsafe { s.deallocate(a.cast(), layout) };
        // peak does NOT decrease on dealloc.
        assert_eq!(s.stats().peak_bytes(), 16);
        assert_eq!(s.stats().current_bytes(), 8);
    }

    #[test]
    fn failures_counted_separately() {
        // Tiny capacity so the second alloc fails.
        let s: Statistics<Slab<u64, InlineBacked<32>>> =
            Statistics::new(Slab::new(1, InlineBacked::<32>::new()).unwrap());
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let _ = s.allocate(layout).unwrap();
        let _ = s.allocate(layout); // fails
        assert_eq!(s.stats().failures.load(Ordering::Relaxed), 1);
        assert_eq!(s.stats().total_allocations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn live_count_tracks_balance() {
        let s = build();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = s.allocate(layout).unwrap();
        let _ = s.allocate(layout).unwrap();
        assert_eq!(s.stats().live_count(), 2);
        unsafe { s.deallocate(a.cast(), layout) };
        assert_eq!(s.stats().live_count(), 1);
    }

    /// Fresh `Statistics<Slab>` reports zero corruption events both via
    /// the trait method and via the AllocStats mirror. Allocate +
    /// deallocate cycles update the mirror to match inner (still 0
    /// since no corruption was triggered).
    #[test]
    fn corruption_events_zero_on_uncorrupted_path() {
        let s = build();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Trait method on fresh allocator.
        assert_eq!(s.corruption_events(), 0);
        // AllocStats mirror starts at 0.
        assert_eq!(s.stats().corruption_events.load(Ordering::Relaxed), 0);
        // Round-trip a few allocations — mirror polled on each call.
        let a = s.allocate(layout).unwrap();
        let b = s.allocate(layout).unwrap();
        unsafe {
            s.deallocate(a.cast(), layout);
            s.deallocate(b.cast(), layout);
        }
        // Still 0 — no corruption was triggered.
        assert_eq!(s.corruption_events(), 0);
        assert_eq!(s.stats().corruption_events.load(Ordering::Relaxed), 0);
    }

    /// `#[non_exhaustive]` on `AllocStats` is the API forward-compat
    /// promise: callers cannot exhaustively destructure outside the
    /// crate. We can't actually test the negative compile, but a
    /// smoke test confirms the field is reachable and zero.
    #[test]
    fn alloc_stats_corruption_events_field_accessible() {
        let stats = AllocStats::new();
        assert_eq!(stats.corruption_events.load(Ordering::Relaxed), 0);
    }

    /// Positive end-to-end propagation via the Slab freelist
    /// out-of-range-next_idx path (a sibling of MAC failure that hits
    /// the same `corruption_events.fetch_add` site).
    ///
    /// **Profile requirement:** this test relies on
    /// `panic = "unwind"` (the default for `dev` and `test` profiles
    /// in cargo). Under `panic = "abort"` in a debug build the
    /// `debug_assert!(false, ...)` inside `Slab::allocate`'s
    /// corruption branch would abort the process instead of unwinding
    /// into `catch_unwind`. The workspace has no `panic = "abort"`
    /// profile today; if one is added in the future, gate this test
    /// with `#[cfg(panic = "unwind")]` or move the corruption-trigger
    /// behind a release-only guard.
    ///
    /// Setup: allocate p (slot 0) and p2 (slot 1) through
    /// `Statistics<Slab>`, deallocate p (puts a FreeLink at slot 0
    /// with `next_idx=0` and the freelist head pointing 1-based at
    /// slot 0). Then corrupt slot 0's `next_idx` to `u32::MAX` via
    /// a raw write through the backing's `base()` pointer. The next
    /// `allocate` pops slot 0, sees `next_idx > capacity` (the
    /// defense-in-depth check beside MAC verify), and bumps
    /// `corruption_events` BEFORE the `debug_assert!(false, ...)`
    /// panics. We use `catch_unwind` so the debug assert doesn't fail
    /// the test process — in release the assert is compiled out and
    /// `allocate` falls through to `next_uncarved`.
    ///
    /// Two assertions:
    /// 1. `s.corruption_events()` (forwarded to `Slab::corruption_events()`,
    ///    a plain atomic load — no mutex) reflects the bump in both
    ///    build profiles.
    /// 2. The `AllocStats.corruption_events` mirror catches up on the
    ///    next call through `Statistics`. We use a *deallocate* (of
    ///    the pre-allocated p2) because `Statistics::deallocate` polls
    ///    the inner counter and updates the mirror **before**
    ///    forwarding — guaranteeing the update lands regardless of
    ///    what the inner does. (`Statistics::allocate` polls after
    ///    forwarding, which would deadlock the debug-mode catchup
    ///    against the still-corrupted freelist.)
    ///
    #[test]
    fn corruption_events_propagates_from_slab_mac_failure() {
        use forge_alloc_core::FixedRange;
        let s = build();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        assert_eq!(s.corruption_events(), 0);

        // Allocate two slots so we have a known-valid p2 to deallocate
        // later (the mirror-catchup vehicle that survives a corrupted
        // freelist).
        let p = s.allocate(layout).unwrap();
        let p2 = s.allocate(layout).unwrap();
        // Deallocate p — slot 0 now holds a FreeLink, freelist head = 1
        // (1-based, pointing at slot 0).
        unsafe { s.deallocate(p.cast(), layout) };

        // Corrupt slot 0's FreeLink: write u32::MAX over the `next_idx`
        // field (the first 4 bytes of the slot, since `FreeLink` is
        // `next_idx: u32, mac: u32`).
        //
        // SAFETY: the backing region is owned by the still-live
        // Statistics → Slab; `base()` returns a raw pointer to it. The
        // bytes we're overwriting belong to a *deallocated* slot
        // (Slab considers them part of its freelist-link metadata,
        // not user data). The slab is `!Sync` and we hold `&s`, so no
        // concurrent reader. No `T` destructor runs (the user code
        // already deallocated `p`).
        let base = s.base().as_ptr();
        unsafe {
            core::ptr::write(base as *mut u32, u32::MAX);
        }

        // Trigger the corruption-detect branch. With the default
        // `NoProtection` MAC the `verify` step trivially passes
        // (zero-MAC matches), so the failure comes from the
        // defense-in-depth `next_idx > capacity` check beside it —
        // which bumps the same `corruption_events` counter in
        // lockstep with the MAC-failure path. catch_unwind swallows
        // the `debug_assert!(false, ...)` panic in debug builds; in
        // release the closure returns normally (allocate falls
        // through to next_uncarved after blanking the corrupted
        // head).
        let s_ref = &s;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = s_ref.allocate(layout);
        }));

        // (1) Inner counter bumped — works in both debug and release
        // because `Slab::corruption_events()` is just an atomic load
        // (no mutex / no panic-poisoning concern).
        let inner = s.corruption_events();
        assert!(
            inner >= 1,
            "Slab::corruption_events should have incremented on the freelist-OOB-next_idx path (got {inner})",
        );

        // (2) Mirror catchup via deallocate. `Statistics::deallocate`
        // mirrors the inner counter BEFORE the forward, so this works
        // even in debug where a follow-up allocate would re-trip the
        // corrupted freelist branch. p2 is a known-valid slot 1
        // pointer; Slab::deallocate just writes a FreeLink at slot 1
        // and links it ahead of the corrupted slot 0 — no panic.
        unsafe { s.deallocate(p2.cast(), layout) };
        // Mirror is `AtomicUsize` (32-bit portability); `inner` is the
        // trait method's `u64` return. Compare as u64 — the cast is a
        // widen on 32-bit (lossless) and a no-op on 64-bit.
        let mirror = s.stats().corruption_events.load(Ordering::Relaxed) as u64;
        assert!(
            mirror >= inner,
            "AllocStats.corruption_events mirror must catch up to inner ({inner}) on the next call (mirror={mirror})",
        );
    }
}
