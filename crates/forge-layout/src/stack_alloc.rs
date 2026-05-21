//! `StackAlloc<B>` — LIFO-discipline allocator. Deallocation is only valid
//! for the most recently allocated block; out-of-order frees are caught in
//! debug builds (panic) and are UB in release.
//!
//! Cheaper than [`BumpArena`](crate::BumpArena) for patterns that do
//! reclaim memory but always in LIFO order — typical of nested scope
//! allocations where teardown is the inverse of construction.
//!
//! See `docs/ARCHITECTURE.md` for the stack-allocator design.

use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::ptr::NonNull;

use forge_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// LIFO-discipline allocator over a [`FixedRange`] backing.
///
/// Maintains a frame stack so nested alloc/free with arbitrary depth works
/// correctly. Each `allocate` pushes the new frame's offset; `deallocate`
/// validates the supplied pointer matches the top of the stack and pops.
pub struct StackAlloc<B: FixedRange> {
    backing: B,
    // No cached `base` pointer: structure-relative backings (e.g.
    // `InlineBacked`) report a different `base()` after a move, so we
    // re-query through `self.backing.base()` at every access instead.
    // The happy-path cost is one extra indirect load.
    capacity: usize,
    /// Current top of stack — bytes used.
    cursor: UnsafeCell<usize>,
    /// Frame stack: each entry is `(aligned_off, prev_cursor)` for one live
    /// allocation. On deallocate we restore cursor to `prev_cursor`.
    ///
    /// `Vec` uses the global allocator via `alloc::vec::Vec`; this means
    /// `StackAlloc` allocates from the global heap for its own bookkeeping
    /// (not from the supplied backing). For pure-no-heap use cases, this is
    /// the right trade-off — the actual user allocations still flow through
    /// the backing.
    frames: UnsafeCell<Vec<(usize, usize)>>,
}

impl<B: FixedRange> StackAlloc<B> {
    /// Construct a LIFO allocator over `backing`'s entire range.
    ///
    /// The frame stack starts at capacity 0 and grows on demand via the
    /// global allocator. For workloads with a known maximum nesting
    /// depth, prefer [`Self::with_max_depth`] to pre-reserve the frame
    /// stack and avoid heap reallocations on the alloc hot path.
    pub fn new(backing: B) -> Result<Self, AllocError> {
        Self::with_max_depth(backing, 0)
    }

    /// Construct with a hint for the maximum frame depth. The frame
    /// stack is pre-allocated with `max_depth` capacity, so the alloc
    /// hot path does not trigger a `Vec` grow / reallocation for the
    /// first `max_depth` nested allocations.
    ///
    /// If `max_depth` is exceeded at runtime the `Vec` will grow
    /// normally — the constructor's value is a hint, not a hard cap.
    /// Pass `0` for the same behavior as [`Self::new`].
    pub fn with_max_depth(backing: B, max_depth: usize) -> Result<Self, AllocError> {
        let capacity = backing.size();
        if capacity == 0 {
            return Err(AllocError);
        }
        // Use `Vec::new()` + `try_reserve_exact` so an absurd `max_depth`
        // (e.g. `usize::MAX`) translates allocation failure to `AllocError`
        // instead of panicking inside `Vec::with_capacity`.
        let mut frames: Vec<(usize, usize)> = Vec::new();
        if max_depth > 0 {
            frames.try_reserve_exact(max_depth).map_err(|_| AllocError)?;
        }
        Ok(Self {
            backing,
            capacity,
            cursor: UnsafeCell::new(0),
            frames: UnsafeCell::new(frames),
        })
    }

    /// Bytes currently in use.
    #[inline]
    pub fn allocated(&self) -> usize {
        // SAFETY: !Sync.
        unsafe { *self.cursor.get() }
    }

    /// Bytes remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity - self.allocated()
    }

    /// Borrow the backing.
    #[inline]
    pub fn backing(&self) -> &B {
        &self.backing
    }
}

unsafe impl<B: FixedRange> Deallocator for StackAlloc<B> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // SAFETY: !Sync — single-threaded access to frames + cursor.
        unsafe {
            let frames = &mut *self.frames.get();
            let base_addr = self.backing.base().as_ptr() as usize;
            let p_addr = ptr.as_ptr() as usize;
            let p_off = p_addr - base_addr;
            // The top of the frame stack must be (p_off, prev_cursor). Out-of-order
            // frees fail this check.
            let Some(&(top_off, prev_cursor)) = frames.last() else {
                debug_assert!(
                    false,
                    "StackAlloc::deallocate: stack is empty (double free?)",
                );
                return;
            };
            if top_off != p_off {
                debug_assert!(
                    false,
                    "StackAlloc::deallocate: out-of-order free (ptr offset {} != top frame {})",
                    p_off, top_off,
                );
                return;
            }
            // Pop the frame and restore the cursor to where it was BEFORE
            // this allocation (recovering the alignment-pad bytes too).
            frames.pop();
            *self.cursor.get() = prev_cursor;
        }
    }
}

unsafe impl<B: FixedRange> Allocator for StackAlloc<B> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let align = layout.align().get();
        let size = layout.size().get();
        // Re-query the backing base each call; structure-relative backings
        // change address on move.
        let base = self.backing.base();
        let base_addr = base.as_ptr() as usize;

        // SAFETY: !Sync.
        unsafe {
            let cursor_ptr = self.cursor.get();
            let cur = *cursor_ptr;
            let raw = base_addr.checked_add(cur).ok_or(AllocError)?;
            let aligned = raw.checked_add(align - 1).ok_or(AllocError)? & !(align - 1);
            let aligned_off = aligned - base_addr;
            let end_off = aligned_off.checked_add(size).ok_or(AllocError)?;
            if end_off > self.capacity {
                return Err(AllocError);
            }
            // Record the frame BEFORE advancing the cursor. `cur` is the
            // value to restore on this frame's deallocate (which gives back
            // the alignment-pad bytes between `cur` and `aligned_off`).
            (&mut *self.frames.get()).push((aligned_off, cur));
            *cursor_ptr = end_off;
            let p = base.as_ptr().add(aligned_off);
            Ok(NonNull::slice_from_raw_parts(
                NonNull::new_unchecked(p),
                size,
            ))
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.capacity)
    }
}

impl<B: FixedRange> FixedRange for StackAlloc<B> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.backing.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_backing::InlineBacked;

    #[test]
    fn lifo_alloc_then_free_restores_cursor() {
        let s = StackAlloc::new(InlineBacked::<256>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let a = s.allocate(layout).unwrap();
        assert_eq!(s.allocated(), 64);
        unsafe { s.deallocate(a.cast(), layout) };
        assert_eq!(s.allocated(), 0);
    }

    #[test]
    fn nested_alloc_then_lifo_free_works() {
        let s = StackAlloc::new(InlineBacked::<256>::new()).unwrap();
        let l1 = NonZeroLayout::from_size_align(32, 8).unwrap();
        let l2 = NonZeroLayout::from_size_align(16, 8).unwrap();
        let a = s.allocate(l1).unwrap();
        let _b = s.allocate(l2).unwrap();
        assert_eq!(s.allocated(), 48);
        // Free in reverse (LIFO).
        unsafe { s.deallocate(_b.cast(), l2) };
        // After freeing the top, cursor rolls back to start of b's slot = 32.
        assert_eq!(s.allocated(), 32);
        unsafe { s.deallocate(a.cast(), l1) };
        assert_eq!(s.allocated(), 0);
    }

    #[test]
    #[should_panic(expected = "out-of-order free")]
    #[cfg(debug_assertions)]
    fn out_of_order_free_panics_in_debug() {
        let s = StackAlloc::new(InlineBacked::<256>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        let a = s.allocate(layout).unwrap();
        let _b = s.allocate(layout).unwrap();
        // Free a before b — violates LIFO.
        unsafe { s.deallocate(a.cast(), layout) };
    }

    #[test]
    fn exhaustion_returns_error() {
        let s = StackAlloc::new(InlineBacked::<64>::new()).unwrap();
        let big = NonZeroLayout::from_size_align(64, 1).unwrap();
        let _ = s.allocate(big).unwrap();
        let one = NonZeroLayout::from_size_align(1, 1).unwrap();
        assert!(s.allocate(one).is_err());
    }

    #[test]
    fn three_level_nested_lifo() {
        // The original ship's "nested LIFO" test passed only by coincidence
        // (the first allocation lived at offset 0, colliding with the empty
        // sentinel). This test pushes the first frame off zero and exercises
        // three nesting levels — only possible with a real frame stack.
        let s = StackAlloc::new(InlineBacked::<256>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
        let a = s.allocate(layout).unwrap();
        let b = s.allocate(layout).unwrap();
        let c = s.allocate(layout).unwrap();
        // Free in reverse — every pop must succeed.
        unsafe {
            s.deallocate(c.cast(), layout);
            assert_eq!(s.allocated(), 32);
            s.deallocate(b.cast(), layout);
            assert_eq!(s.allocated(), 16);
            s.deallocate(a.cast(), layout);
            assert_eq!(s.allocated(), 0);
        }
    }

    #[test]
    fn fixed_range_contains_allocations() {
        let s = StackAlloc::new(InlineBacked::<128>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
        let block = s.allocate(layout).unwrap();
        assert!(s.contains(block.cast::<u8>()));
    }
}
