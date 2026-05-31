//! `SharedBumpArena<B>` — atomic-cursor variant of [`BumpArena`].
//!
//! `Send + Sync`. Each allocate is a `compare_exchange_weak` CAS loop that
//! handles alignment rounding atomically. No `reset()` — getting `&mut self`
//! through an `Arc<SharedBumpArena>` requires `Arc::get_mut()`, at which
//! point the arena can simply be dropped and recreated.
//!
//! Only available on targets with pointer-sized atomics
//! (`cfg(target_has_atomic = "ptr")`); single-core no_std targets without
//! atomics must use [`BumpArena`](crate::layout::BumpArena) with explicit ownership
//! discipline.

use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use forge_alloc_core::{
    AllocError, Allocator, CachePadded, Deallocator, FixedRange, NonZeroLayout, CACHE_LINE,
};

/// Atomic-cursor bump arena.
///
/// Sound under multi-thread `&self` allocation: the cursor uses a
/// `compare_exchange_weak` CAS loop to handle alignment rounding atomically.
pub struct SharedBumpArena<B: FixedRange> {
    // Held by value to extend the backing's lifetime alongside the arena
    // (RAII — backing drops when the arena drops). Not exposed by accessor:
    // see the comment on the impl block.
    //
    // We deliberately do NOT cache `base` here. Backings whose `base()` is
    // structure-relative (e.g. `InlineBacked<N>` returns `&self.storage`)
    // report a DIFFERENT address before and after the backing has been
    // moved into `Self`. An absolute pointer captured at construction
    // would point at the pre-move location for the rest of the arena's
    // life and silently corrupt every subsequent `allocate`. The fix
    // is to re-query `self.backing.base()` at each access; the happy-
    // path cost is one extra indirect load.
    backing: B,
    capacity: usize,
    /// Atomic bump cursor. CAS'd by every concurrent allocator; kept on
    /// its own cache line so contended writers don't invalidate the
    /// read-only `backing` / `capacity` fields above (which the alloc
    /// hot path reads on every retry). See [`CachePadded`].
    cursor: CachePadded<AtomicUsize>,
}

impl<B: FixedRange> SharedBumpArena<B> {
    /// Layout-pin: the cursor must occupy its own cache line. The
    /// adjacent `backing` and `capacity` fields are read on every
    /// allocate; without padding, every CAS on the cursor would
    /// invalidate the line containing them, costing each concurrent
    /// allocator a refill on every retry.
    const LAYOUT_PIN: () = {
        use core::mem::offset_of;
        let cap_off = offset_of!(SharedBumpArena<B>, capacity);
        let cur_off = offset_of!(SharedBumpArena<B>, cursor);
        assert!(
            cap_off / CACHE_LINE != cur_off / CACHE_LINE,
            "SharedBumpArena layout regression: `cursor` shares a line with `capacity`",
        );
    };
}

impl<B: FixedRange> SharedBumpArena<B> {
    /// Construct a shared bump arena.
    ///
    /// Errors if the backing reports a zero-byte range.
    pub fn new(backing: B) -> Result<Self, AllocError> {
        // Force evaluation of the layout-pin const for this `B`.
        let _: () = Self::LAYOUT_PIN;
        let capacity = backing.size();
        if capacity == 0 {
            return Err(AllocError);
        }
        Ok(Self {
            backing,
            capacity,
            cursor: CachePadded::new(AtomicUsize::new(0)),
        })
    }

    /// Bytes currently allocated.
    #[inline]
    pub fn allocated(&self) -> usize {
        // Telemetry read: no synchronization required.
        self.cursor.load(Ordering::Relaxed)
    }

    /// Total capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity().saturating_sub(self.allocated())
    }

    // No `backing()` accessor by design: SharedBumpArena is `Sync` while many
    // backings (InlineBacked, MmapBacked, …) are `!Sync` because they expose
    // their own cursor through `UnsafeCell`. Handing out `&B` from `&self`
    // would let multiple threads call B's `&self` methods concurrently and
    // race on B's interior mutability. The backing is sealed inside the
    // arena; `base`/`size` are cached at construction. If a caller needs the
    // backing back, drop the arena (`SharedBumpArena` can be unwrapped by
    // value via `Arc::into_inner` + private constructor in the future).
}

unsafe impl<B: FixedRange> Deallocator for SharedBumpArena<B> {
    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // No-op. SharedBumpArena does not support reset.
    }
}

unsafe impl<B: FixedRange> Allocator for SharedBumpArena<B> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let align = layout.align().get();
        let size = layout.size().get();
        // Re-query the backing's base at each allocate. See the
        // struct-field comment for why a cached pointer would be unsound
        // for structure-relative backings (e.g. `InlineBacked`).
        let base = self.backing.base();
        let base_addr = base.as_ptr() as usize;
        // Loop-invariant alignment constants — hoist out of the CAS retry
        // loop so an LLVM optimizer that gets confused by the `loop {…}`
        // structure can't accidentally repeat the subtraction + bitwise-not
        // each iteration. `align` is a NonZero power-of-two (NonZeroLayout
        // guarantees), so `align - 1` is the standard "low-bit mask".
        let align_minus_one = align - 1;
        let align_mask = !align_minus_one;

        // CAS loop: read current cursor, compute aligned-offset + new cursor,
        // try to publish. If another thread won the race, retry.
        let mut cur = self.cursor.load(Ordering::Relaxed);
        loop {
            let raw = base_addr.checked_add(cur).ok_or(AllocError)?;
            let aligned = raw.checked_add(align_minus_one).ok_or(AllocError)? & align_mask;
            let aligned_off = aligned - base_addr;
            let end_off = aligned_off.checked_add(size).ok_or(AllocError)?;
            if end_off > self.capacity {
                return Err(AllocError);
            }
            // NOTE: deliberately no `backing.commit()` here, unlike
            // `BumpArena`/`StackAlloc`. `SharedBumpArena` is `Sync` and
            // allocates concurrently, but `MmapBacked`'s commit watermark is
            // a plain `!Sync` `UnsafeCell` — committing from multiple threads
            // would race it. A `lazy_commit` `MmapBacked` is therefore
            // UNSUPPORTED under `SharedBumpArena` and faults on first write
            // (see `MmapFlags::lazy_commit`). Supporting it needs an atomic
            // watermark; revisit only if a lazy *shared* arena is needed.
            //
            // The CAS publishes only an integer offset. The bytes
            // [base+aligned_off, base+end_off) are owned exclusively by the
            // winning thread; another thread that subsequently reads those
            // bytes must do so through an external happens-before (e.g.
            // channel send/receive) which provides its own Acquire/Release
            // pairing. Relaxed on success and failure is correct here.
            //
            // NB: if future code adds cross-thread reads of allocated bytes
            // routed via this cursor (e.g. "scan the arena from another
            // thread"), the success ordering must be upgraded to Release
            // and the reader's load to Acquire.
            match self.cursor.compare_exchange_weak(
                cur,
                end_off,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Defense-in-depth: cursor must move forward on every
                    // successful CAS. If a future refactor accidentally
                    // recomputes `end_off` from a stale `cur` (or swaps
                    // the CAS to a different ordering that admits
                    // re-ordering), this will fire in debug. No release
                    // cost.
                    debug_assert!(
                        end_off > cur,
                        "SharedBumpArena cursor monotonicity violated: cur={cur} → end_off={end_off}",
                    );
                    // SAFETY: aligned_off + size <= capacity; the resulting
                    // ptr lies within [base, base+capacity). base is
                    // non-null per FixedRange's contract.
                    unsafe {
                        let p = base.as_ptr().add(aligned_off);
                        return Ok(NonNull::slice_from_raw_parts(
                            NonNull::new_unchecked(p),
                            size,
                        ));
                    }
                }
                Err(new_cur) => {
                    // Defense-in-depth: failed CAS means either another
                    // thread won the race (new_cur > cur) OR
                    // `compare_exchange_weak` spuriously failed and
                    // returned the same value (new_cur == cur). The
                    // cursor is still monotonic — never below cur —
                    // because only successful CAS advances it. We use
                    // `>=` rather than `>` to permit spurious failures,
                    // which Miri's strict atomic model exposes
                    // (cargo +nightly miri test).
                    debug_assert!(
                        new_cur >= cur,
                        "SharedBumpArena CAS failure returned regressed cursor: cur={cur} new_cur={new_cur}",
                    );
                    cur = new_cur;
                    core::hint::spin_loop();
                }
            }
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.capacity)
    }

    // No reset() override — Allocator's default Err(AllocError) is correct.
    // SharedBumpArena deliberately does not support reset.
}

impl<B: FixedRange> FixedRange for SharedBumpArena<B> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        // Forward to the live backing; structure-relative backings move.
        self.backing.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.capacity
    }
}

// `SharedBumpArena<B>` holds only `B`, a `usize` (capacity), and an
// `AtomicUsize` (cursor) — no raw `NonNull`/`base` field; the base is
// re-queried from `self.backing.base()` on every access (the move-safety fix).
// `AtomicUsize` is `Sync`, so the marker impls are sound given `B: Send` PLUS
// the requirement that `FixedRange::base`/`size` are callable concurrently
// through a shared `&self` without data races (which lets `B` be `!Sync`, e.g.
// `InlineBacked`). Every current backing satisfies this — `base()`/`size()`
// are pure reads of an immutable field. A future `FixedRange` impl that
// mutated through `&self` in `base()`/`size()` would make this `Sync` unsound;
// the trait documents that requirement.
unsafe impl<B: FixedRange + Send> Send for SharedBumpArena<B> {}
unsafe impl<B: FixedRange + Send> Sync for SharedBumpArena<B> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;

    #[test]
    fn single_thread_allocation_works() {
        let arena = SharedBumpArena::new(InlineBacked::<1024>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let _ = arena.allocate(layout).unwrap();
        assert_eq!(arena.allocated(), 64);
    }

    #[test]
    fn single_thread_exhaustion_returns_err() {
        let arena = SharedBumpArena::new(InlineBacked::<64>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(64, 1).unwrap();
        let _ = arena.allocate(layout).unwrap();
        let l2 = NonZeroLayout::from_size_align(1, 1).unwrap();
        assert!(arena.allocate(l2).is_err());
    }

    #[test]
    fn fixed_range_contains_allocations() {
        let arena = SharedBumpArena::new(InlineBacked::<512>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let block = arena.allocate(layout).unwrap();
        assert!(arena.contains(block.cast()));
    }

    #[test]
    fn alignment_respected_under_concurrent_layout_requests() {
        let arena = SharedBumpArena::new(InlineBacked::<256>::new()).unwrap();
        let _ = arena
            .allocate(NonZeroLayout::from_size_align(3, 1).unwrap())
            .unwrap();
        let layout = NonZeroLayout::from_size_align(8, 16).unwrap();
        let block = arena.allocate(layout).unwrap();
        assert_eq!(block.cast::<u8>().as_ptr() as usize % 16, 0);
    }

    /// Boundary: allocate up to the exact capacity boundary. Off-by-one
    /// in `end_off > self.capacity` would either spuriously reject the
    /// last byte or accept a byte past the buffer end.
    #[test]
    fn exact_capacity_alloc_succeeds_and_next_byte_fails() {
        let arena = SharedBumpArena::new(InlineBacked::<64>::new()).unwrap();
        let big = NonZeroLayout::from_size_align(64, 1).unwrap();
        let block = arena.allocate(big).unwrap();
        assert_eq!(block.len(), 64);
        // No room for even a single byte.
        let one = NonZeroLayout::from_size_align(1, 1).unwrap();
        assert!(arena.allocate(one).is_err());
        // Allocated counter must equal capacity exactly.
        assert_eq!(arena.allocated(), 64);
        assert_eq!(arena.remaining(), 0);
    }

    /// Regression: SharedBumpArena historically cached an absolute
    /// `base` pointer captured before the backing moved into Self.
    /// For structure-relative backings (`InlineBacked`) the captured
    /// pointer became stale on every move of the arena, silently
    /// corrupting all subsequent allocates. The fix re-queries
    /// `self.backing.base()` on each allocate. Verify the arena's
    /// reported base agrees with the backing's current base.
    #[test]
    fn base_pointer_matches_backing_after_move() {
        let arena = SharedBumpArena::new(InlineBacked::<256>::new()).unwrap();
        use forge_alloc_core::FixedRange;
        let arena_base = arena.base().as_ptr();
        let backing_base = arena.backing.base().as_ptr();
        assert_eq!(
            arena_base, backing_base,
            "SharedBumpArena's base must agree with the live backing — stale-pointer regression",
        );
        // Plus: a fresh allocation must land at base + 0.
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        let block = arena.allocate(layout).unwrap();
        let alloc_addr = block.cast::<u8>().as_ptr() as usize;
        assert_eq!(
            alloc_addr, backing_base as usize,
            "first alloc must be at backing.base()",
        );
    }

    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn concurrent_allocate_no_overlaps() {
        use std::collections::HashSet;
        use std::sync::Arc;
        use std::thread;

        let arena = Arc::new(SharedBumpArena::new(InlineBacked::<65536>::new()).unwrap());
        let mut handles = Vec::new();
        const PER_THREAD: usize = 64;
        for _ in 0..8 {
            let a = Arc::clone(&arena);
            handles.push(thread::spawn(move || {
                let mut ptrs = Vec::with_capacity(PER_THREAD);
                let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
                for _ in 0..PER_THREAD {
                    let block = a.allocate(layout).unwrap();
                    ptrs.push(block.cast::<u8>().as_ptr() as usize);
                }
                ptrs
            }));
        }
        let mut all: Vec<usize> = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        let unique: HashSet<usize> = all.iter().copied().collect();
        assert_eq!(
            unique.len(),
            all.len(),
            "no two allocations may share an address"
        );
        // Each allocation is 32 bytes 8-aligned. Confirm none overlap:
        let mut sorted = all.clone();
        sorted.sort();
        for w in sorted.windows(2) {
            assert!(
                w[1] - w[0] >= 32,
                "adjacent allocations must be at least 32 bytes apart"
            );
        }
    }
}
