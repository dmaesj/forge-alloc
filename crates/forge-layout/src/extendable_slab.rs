//! `ExtendableSlab<T, M>` — growable typed allocator backed by a `Vec` of
//! [`Slab`](crate::Slab) segments. On exhaustion, a new fixed-capacity
//! segment is appended; freelist offsets within each segment remain valid
//! forever (no segment is ever reallocated).
//!
//! Unlike [`Slab`](crate::Slab), `ExtendableSlab` does NOT implement
//! [`FixedRange`] — its address range grows as segments are added.
//!
//! Requires `std` because growth uses `alloc::vec::Vec` and segments are
//! backed by `MmapBacked`.
//!
//! See spec §6.8.

#![cfg(feature = "std")]

use core::ptr::NonNull;

use forge_backing::MmapBacked;
use forge_core::{AllocError, Allocator, Deallocator, FreelistProtection, NoProtection, NonZeroLayout};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::vec::Vec;

use crate::Slab;

/// Growable typed slab.
///
/// Each segment is a fixed-capacity `Slab<T, MmapBacked, M>`. Growth allocates
/// a new `MmapBacked` and appends a new segment; the previous segments'
/// pointers remain valid. Segments are never reallocated or moved.
///
/// # Thread safety
///
/// `Send + Sync`. The segment list is guarded by a `Mutex` because we may
/// need to push to a `Vec` on growth; per-segment slabs are `!Sync` but the
/// mutex serializes access to the list itself. (For high-contention
/// workloads, switch to `SlabOwner`/`SlabRemote` in M8 for per-thread
/// segments.)
///
/// # Panic safety
///
/// If a thread panics while holding the segments mutex, the mutex is
/// poisoned. `allocate`, `deallocate`, `segment_count`, and the internal
/// `build_segment` helper all call `expect("mutex poisoned")` and
/// re-panic on a poisoned lock. This is **intentional** and asymmetric
/// with [`SlabOwner::drop`](crate::SlabOwner)'s `into_inner()` recovery:
///
/// - `SlabOwner::drop` runs during unwind and MUST NOT double-panic, so
///   it recovers the inner state via `into_inner()` even on a poisoned
///   mutex.
/// - `ExtendableSlab::{allocate, deallocate, build_segment,
///   segment_count}` run on the normal call path. A poisoned mutex
///   means some prior call panicked under the lock — the allocator
///   state is suspect and the safest policy is fail-loud rather than
///   silently continue. Once poisoned, the wrapper is permanently
///   unusable; the application should treat this as a fatal allocator
///   error and recreate the `ExtendableSlab`.
///
/// Tier-3 task `a08d99` formerly tracked this asymmetry; resolved by
/// this documentation.
pub struct ExtendableSlab<T, M: FreelistProtection = NoProtection> {
    segment_capacity: usize,
    mac_factory: fn() -> M,
    /// All segments allocated so far. Each segment is independently a Slab.
    /// We hold the Vec inside a Mutex; allocate() acquires the lock only
    /// when no current segment can serve the request.
    segments: Mutex<Vec<Slab<T, MmapBacked, M>>>,
    /// Hint: index of the lowest segment that *might* have free slots.
    /// Updated by `allocate` (forward, on full-segment skip) and by
    /// `deallocate` (backward, when freeing into an earlier segment).
    /// Saves an O(N) walk through known-full segments on the alloc hot
    /// path; correctness still depends on the freelist's `Err` signal
    /// when the hint is stale.
    ///
    /// Currently `AtomicUsize`, but every read and write happens while
    /// the [`segments`](Self::segments) mutex is held, so the atomicity
    /// is vestigial — a plain `usize` (inside the `Mutex` payload) would
    /// suffice. The atomic stays here only to avoid a layout-disturbing
    /// refactor; do not migrate any access out from under the mutex
    /// without also relaxing the freelist's `Err`-based fallback path,
    /// which currently relies on the mutex's serialization across the
    /// hint read + the segment retry walk.
    ///
    /// Note: `allocate` briefly drops the mutex during the growth path
    /// to avoid holding it through the `mmap` syscall (see Phase 2 in
    /// [`Allocator::allocate`](Self::allocate)). During that window the
    /// hint is NOT read or written — only the local `len_before_drop`
    /// snapshot is used to detect concurrent growth on lock re-acquire.
    /// All hint touches remain under the mutex.
    first_open_hint: AtomicUsize,
    /// Count of `deallocate` calls where the supplied pointer was not
    /// found in any segment. In debug builds the call panics; in
    /// release it silently returns. Either way it represents either a
    /// caller-contract violation (wrong pointer, wrong allocator) or
    /// an in-progress attack probing for UAF — operators want this
    /// observable via [`Allocator::corruption_events`].
    ///
    /// `AtomicUsize` (not `AtomicU64`) for portability to 32-bit
    /// bare-metal targets that lack native 64-bit atomics. The trait
    /// method returns `u64`; cast happens at the boundary. See the
    /// `Slab::corruption_events` field for the rationale and overflow
    /// analysis.
    routing_failures: AtomicUsize,
}

impl<T> ExtendableSlab<T, NoProtection> {
    /// Construct an empty ExtendableSlab with `NoProtection`. Segments are
    /// added lazily on first allocate.
    #[inline]
    pub fn new(segment_capacity: usize) -> Self {
        Self::with_protection(segment_capacity, || NoProtection)
    }
}

impl<T, M: FreelistProtection> ExtendableSlab<T, M> {
    /// Construct an empty ExtendableSlab with an explicit freelist-protection
    /// factory.
    #[inline]
    pub fn with_protection(segment_capacity: usize, mac_factory: fn() -> M) -> Self {
        Self {
            segment_capacity,
            mac_factory,
            segments: Mutex::new(Vec::new()),
            first_open_hint: AtomicUsize::new(0),
            routing_failures: AtomicUsize::new(0),
        }
    }

    /// Construct with `initial_segments` segments pre-allocated.
    pub fn with_initial_segments(
        count: usize,
        segment_capacity: usize,
        mac_factory: fn() -> M,
    ) -> Result<Self, AllocError> {
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(Self::build_segment(segment_capacity, mac_factory)?);
        }
        Ok(Self {
            segment_capacity,
            mac_factory,
            segments: Mutex::new(v),
            first_open_hint: AtomicUsize::new(0),
            routing_failures: AtomicUsize::new(0),
        })
    }

    /// Number of segments currently allocated.
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.segments.lock().expect("ExtendableSlab mutex poisoned").len()
    }

    /// Helper: construct one segment with a freshly-built MmapBacked + MAC.
    fn build_segment(
        capacity: usize,
        mac_factory: fn() -> M,
    ) -> Result<Slab<T, MmapBacked, M>, AllocError> {
        // Mirror Slab's internal layout math so we mmap exactly what Slab
        // will request — no 50%-plus-4 KiB slop. Slab uses
        //   block_stride = align_up(max(size_of::<T>(), 8), slot_align)
        //   slot_align   = max(align_of::<T>(), 4)   // FreeLink alignment is 4
        //   total        = capacity * block_stride
        // and `MmapBacked::new(total)` rounds `total` up to a page already, so
        // we add only `slot_align - 1` worst-case alignment slack on top
        // (handles the rare case where the OS hands back a base that isn't
        // already `slot_align`-aligned — pages are typically >= 4 KiB so this
        // is defensive).
        let slot_align = core::cmp::max(core::mem::align_of::<T>(), 4);
        let raw_stride = core::cmp::max(core::mem::size_of::<T>(), 8);
        let block_stride = raw_stride
            .checked_add(slot_align - 1)
            .map(|v| v & !(slot_align - 1))
            .ok_or(AllocError)?;
        let total = block_stride.checked_mul(capacity).ok_or(AllocError)?;
        let with_slack = total.checked_add(slot_align - 1).ok_or(AllocError)?;
        let backing = MmapBacked::new(with_slack)?;
        Slab::with_protection(capacity, backing, mac_factory())
    }
}

unsafe impl<T, M: FreelistProtection + Send> Deallocator for ExtendableSlab<T, M>
where
    T: Send,
{
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // Find the segment whose range contains ptr. Segments are appended
        // and never removed, so we can safely walk the list under the mutex.
        let segs = self.segments.lock().expect("ExtendableSlab mutex poisoned");
        for (i, seg) in segs.iter().enumerate() {
            // SAFETY of forwarded deallocate: the slab issued the pointer, so
            // the layout matches. We just need to find the right slab.
            use forge_core::FixedRange;
            if seg.contains(ptr) {
                unsafe { seg.deallocate(ptr, layout) };
                // This segment now has at least one free slot. Pull the
                // hint back if it had advanced past this index.
                self.first_open_hint.fetch_min(i, Ordering::Relaxed);
                return;
            }
        }
        // Pointer not found in any segment. Bump the routing-failure
        // counter so operators reading `corruption_events` see this as
        // a security event (caller-contract violation or UAF probe).
        // `Relaxed` is correct: advisory observability counter.
        self.routing_failures.fetch_add(1, Ordering::Relaxed);
        debug_assert!(false, "ExtendableSlab::deallocate: pointer not in any segment");
    }
}

unsafe impl<T, M: FreelistProtection + Send> Allocator for ExtendableSlab<T, M>
where
    T: Send,
{
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        // Phase 1: try to satisfy the request from existing segments,
        // entirely under the lock. Common-case fast path is one
        // acquire+release with no syscalls.
        let len_before_drop = {
            let segs = self.segments.lock().expect("ExtendableSlab mutex poisoned");
            // Start from the hinted first-open segment to skip a linear
            // walk through segments known-full last time we checked.
            // Clamp to len in case the hint outran the segment count
            // (e.g. fresh empty slab where hint == 0 and segs.len() == 0).
            let start = self.first_open_hint.load(Ordering::Relaxed).min(segs.len());
            // Try from `start` forward.
            for (offset, seg) in segs.iter().enumerate().skip(start) {
                if let Ok(block) = seg.allocate(layout) {
                    // Push hint forward if every segment before us has
                    // signaled full at least once on this path. We only
                    // nudge forward, never backward — backward moves come
                    // from `deallocate`.
                    if offset > start {
                        let _ = self.first_open_hint.fetch_max(offset, Ordering::Relaxed);
                    }
                    return Ok(block);
                }
            }
            // Hint missed and we'd jump segments — re-walk from the
            // start in case a concurrent `deallocate` freed an earlier
            // slot since the last hint update.
            if start > 0 {
                for (i, seg) in segs.iter().enumerate().take(start) {
                    if let Ok(block) = seg.allocate(layout) {
                        self.first_open_hint.fetch_min(i, Ordering::Relaxed);
                        return Ok(block);
                    }
                }
            }
            // All current segments full — snapshot len so Phase 3 can
            // detect concurrent growth, then drop the lock for the mmap
            // syscall in Phase 2.
            segs.len()
        };

        // Phase 2: build a new segment WITHOUT the lock. `Self::build_segment`
        // performs an `mmap` syscall, which can take milliseconds under
        // memory pressure. Holding the mutex across that syscall would
        // block every concurrent allocate for the duration — defeating
        // the point of the shared structure (eabad2 sub-item (2)).
        let new_seg = Self::build_segment(self.segment_capacity, self.mac_factory)?;

        // Phase 3: re-acquire the lock to install the new segment.
        let mut segs = self.segments.lock().expect("ExtendableSlab mutex poisoned");

        // Race-against-self: any number of concurrent allocate calls may
        // have raced ours through Phase 2 — each one independently saw
        // all-segments-full, dropped the lock, built its own segment, and
        // is queueing up here to install. If any of THEIR segments landed
        // before ours and can serve our request, allocate from theirs
        // and let our `new_seg` drop on the return path (RAII cleanup
        // unmaps the unused mmap region). This bounds over-growth to one
        // redundant segment per racing thread, never unbounded.
        if segs.len() > len_before_drop {
            for (offset, seg) in segs.iter().enumerate().skip(len_before_drop) {
                if let Ok(block) = seg.allocate(layout) {
                    let _ = self.first_open_hint.fetch_max(offset, Ordering::Relaxed);
                    return Ok(block);
                }
            }
        }

        // No racing thread's segment could serve us. Install ours.
        let block = new_seg.allocate(layout)?;
        let new_idx = segs.len();
        segs.push(new_seg);
        // The new segment is the last one (and the only one with space
        // right after this).
        self.first_open_hint.store(new_idx, Ordering::Relaxed);
        Ok(block)
    }

    fn capacity_bytes(&self) -> Option<usize> {
        // Total bytes across all current segments. Growth means this number
        // can increase between calls — Watermark callers should call this
        // each check (per spec §7.7 rev 1.5, which we already do).
        let segs = self.segments.lock().expect("ExtendableSlab mutex poisoned");
        let total: usize = segs
            .iter()
            .filter_map(|s| s.capacity_bytes())
            .sum();
        Some(total)
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        // Two sources: per-segment Slab counters (MAC + out-of-range
        // next_idx silent disarms) plus our own `routing_failures`
        // counter (deallocate called with a pointer not in any
        // segment). `saturating_add` guards against u64 overflow.
        //
        // `routing_failures` stores `usize` for 32-bit portability; cast
        // to `u64` here so the sum matches the trait return type. The
        // per-segment `s.corruption_events()` already returns `u64`.
        let routing = self.routing_failures.load(Ordering::Relaxed) as u64;
        let segs = self.segments.lock().expect("ExtendableSlab mutex poisoned");
        segs.iter()
            .map(|s| s.corruption_events())
            .fold(routing, |acc, x| acc.saturating_add(x))
    }
}

// Note: ExtendableSlab deliberately does NOT implement FixedRange. Its
// address range grows on each new segment, so it cannot be used as the
// Primary in WithFallback. If you need that pattern, route at the
// application level.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]

    #[cfg_attr(miri, ignore = "miri-incompatible: ExtendableSlab uses MmapBacked")]
    fn fresh_extendable_has_no_segments() {
        let s: ExtendableSlab<u64> = ExtendableSlab::new(16);
        assert_eq!(s.segment_count(), 0);
    }

    #[test]

    #[cfg_attr(miri, ignore = "miri-incompatible: ExtendableSlab uses MmapBacked")]
    fn first_allocate_creates_segment() {
        let s: ExtendableSlab<u64> = ExtendableSlab::new(8);
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let _ = s.allocate(layout).unwrap();
        assert_eq!(s.segment_count(), 1);
    }

    #[test]

    #[cfg_attr(miri, ignore = "miri-incompatible: ExtendableSlab uses MmapBacked")]
    fn growth_on_segment_exhaustion() {
        let s: ExtendableSlab<u64> = ExtendableSlab::new(4); // 4 slots per segment
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Allocate 5 — forces a second segment.
        let mut ptrs = Vec::new();
        for _ in 0..5 {
            ptrs.push(s.allocate(layout).unwrap());
        }
        assert_eq!(s.segment_count(), 2);
    }

    #[test]

    #[cfg_attr(miri, ignore = "miri-incompatible: ExtendableSlab uses MmapBacked")]
    fn deallocate_routes_to_correct_segment() {
        let s: ExtendableSlab<u64> = ExtendableSlab::new(2);
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = s.allocate(layout).unwrap();
        let _b = s.allocate(layout).unwrap();
        let c = s.allocate(layout).unwrap(); // triggers growth
        let _d = s.allocate(layout).unwrap();
        // Now free a (segment 0) and c (segment 1) — both must complete
        // without debug_assert.
        unsafe {
            s.deallocate(a.cast(), layout);
            s.deallocate(c.cast(), layout);
        }
    }

    #[test]

    #[cfg_attr(miri, ignore = "miri-incompatible: ExtendableSlab uses MmapBacked")]
    fn capacity_grows_with_segments() {
        let s: ExtendableSlab<u64> = ExtendableSlab::new(4);
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let _ = s.allocate(layout).unwrap();
        let c1 = s.capacity_bytes().unwrap();
        for _ in 0..4 {
            let _ = s.allocate(layout).unwrap();
        }
        let c2 = s.capacity_bytes().unwrap();
        assert!(c2 > c1, "capacity should grow with second segment");
    }

    #[test]

    #[cfg_attr(miri, ignore = "miri-incompatible: ExtendableSlab uses MmapBacked")]
    fn initial_segments_preallocated() {
        let s: ExtendableSlab<u64, NoProtection> =
            ExtendableSlab::with_initial_segments(3, 4, || NoProtection).unwrap();
        assert_eq!(s.segment_count(), 3);
    }

    /// Boundary: `segment_capacity = 0` produces a Slab that rejects at
    /// construction (`Slab::new(0, _)` returns Err). The first allocate
    /// surfaces the failure as `AllocError` rather than panicking.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: ExtendableSlab uses MmapBacked")]
    fn segment_capacity_zero_fails_first_allocate() {
        let s: ExtendableSlab<u64> = ExtendableSlab::new(0);
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        assert!(
            s.allocate(layout).is_err(),
            "segment_capacity=0 must propagate as AllocError",
        );
    }

    /// `segment_capacity = 1, allocate 100 times` — each allocate
    /// produces a fresh segment. Verifies the segment-growth loop
    /// doesn't degenerate.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: ExtendableSlab uses MmapBacked")]
    fn one_slot_per_segment_grows_one_segment_per_alloc() {
        let s: ExtendableSlab<u64> = ExtendableSlab::new(1);
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let mut ptrs = Vec::new();
        for _ in 0..100 {
            ptrs.push(s.allocate(layout).unwrap());
        }
        assert_eq!(s.segment_count(), 100);
        // All pointers must be distinct (one segment each).
        let addrs: std::collections::HashSet<usize> = ptrs
            .iter()
            .map(|p| p.cast::<u8>().as_ptr() as usize)
            .collect();
        assert_eq!(addrs.len(), 100, "every alloc must give a distinct slot");
    }

    /// Regression for eabad2 sub-item (2): `ExtendableSlab::allocate`
    /// previously held the segments mutex through `Self::build_segment`,
    /// which performs an `mmap` syscall. The fix drops the lock for the
    /// syscall and re-acquires for install, with a race-against-self
    /// check to avoid unbounded over-growth when multiple threads grow
    /// concurrently.
    ///
    /// This test exercises the concurrent-growth path: many threads
    /// each force segment creation. Properties:
    /// 1. All allocations succeed.
    /// 2. No two allocations return the same pointer.
    /// 3. Segment count is bounded in `[total/segment_capacity,
    ///    total/segment_capacity + N_THREADS]`. The upper bound is the
    ///    worst case where every racing thread in Phase 3 fails to find
    ///    space in any newly-installed segment (because they're full)
    ///    and so commits its own. At most one extra segment can be
    ///    committed per concurrent racer, so per-episode over-growth is
    ///    bounded by the number of threads currently in Phase 2, which
    ///    is in turn bounded by N_THREADS. In practice the actual count
    ///    is far closer to the lower bound — see the race-against-self
    ///    walk at lines 270-277.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: ExtendableSlab uses MmapBacked + threads")]
    fn concurrent_growth_does_not_deadlock_or_double_allocate() {
        use std::sync::Arc;
        use std::thread;

        const N_THREADS: usize = 8;
        const ALLOCS_PER_THREAD: usize = 16;
        const SEGMENT_CAP: usize = 4;
        let total = N_THREADS * ALLOCS_PER_THREAD;

        let slab: Arc<ExtendableSlab<u64>> = Arc::new(ExtendableSlab::new(SEGMENT_CAP));
        let layout = NonZeroLayout::for_type::<u64>().unwrap();

        // Each thread returns Vec<usize> of pointer addresses since
        // NonNull<[u8]> is !Send. Main thread reconstructs NonNull for
        // deallocate (ExtendableSlab is Send+Sync, deallocate takes &self).
        let handles: Vec<_> = (0..N_THREADS)
            .map(|_| {
                let slab = Arc::clone(&slab);
                thread::spawn(move || {
                    let mut local = Vec::with_capacity(ALLOCS_PER_THREAD);
                    for _ in 0..ALLOCS_PER_THREAD {
                        let p = slab.allocate(layout).expect("concurrent allocate");
                        local.push(p.cast::<u8>().as_ptr() as usize);
                    }
                    local
                })
            })
            .collect();

        let mut all_addrs: Vec<usize> = Vec::with_capacity(total);
        for h in handles {
            all_addrs.extend(h.join().expect("thread"));
        }

        // Property 1: all served.
        assert_eq!(all_addrs.len(), total);

        // Property 2: all distinct.
        let uniq: std::collections::HashSet<usize> = all_addrs.iter().copied().collect();
        assert_eq!(uniq.len(), total, "concurrent allocs must all be distinct");

        // Property 3: segment count bounded.
        let min_segments = total / SEGMENT_CAP;
        let max_segments = min_segments + N_THREADS;
        let actual = slab.segment_count();
        assert!(
            (min_segments..=max_segments).contains(&actual),
            "segment_count {actual} not in [{min_segments}, {max_segments}] \
             — over-growth from race-against-self is bounded by N_THREADS",
        );

        // Cleanup — verify deallocate routing still works across all segments.
        for addr in all_addrs {
            let p = unsafe { NonNull::new_unchecked(addr as *mut u8) };
            unsafe { slab.deallocate(p, layout) };
        }
    }
}
