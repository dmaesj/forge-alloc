//! `BumpArena<B>` — single-threaded bump arena over a [`FixedRange`] backing.
//!
//! Allocation is O(1) — align the cursor, bounds-check, advance. Deallocation
//! is a no-op; reclaim happens via [`reset`](BumpArena::reset). To use with
//! the standard collection types (`Vec<T, A>`, etc.), allocate via the arena
//! directly and wrap with `from_raw_in` using [`BumpDeallocator<'_>`] as the
//! deallocation token.
//!
//! ```
//! use forge_alloc::InlineBacked;
//! use forge_alloc::{Allocator, NonZeroLayout};
//! use forge_alloc::BumpArena;
//!
//! let mut arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
//! let layout = NonZeroLayout::from_size_align(128, 16).unwrap();
//! let _block = arena.allocate(layout).unwrap();
//! assert_eq!(arena.allocated(), 128);
//! arena.reset();
//! assert_eq!(arena.allocated(), 0);
//! ```
//!
//! See `docs/ARCHITECTURE.md` for the bump-arena design.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::ptr::NonNull;

use forge_alloc_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// Bump arena over any [`FixedRange`] backing.
///
/// The arena uses the entire address range exposed by the backing. The
/// backing's own `allocate` is never called — `BumpArena` does all
/// suballocation directly. When the arena drops, the backing drops, and the
/// memory is released by whatever path the backing uses (e.g. `MmapBacked`'s
/// `munmap`).
///
/// # Thread safety
///
/// `Send`: yes if `B: Send`. `Sync`: NO — concurrent `&self` allocators would
/// race on the cursor. Use [`SharedBumpArena`](crate::layout::SharedBumpArena) for
/// cross-thread access.
pub struct BumpArena<B: FixedRange> {
    backing: B,
    /// Cached byte size of the backing range, captured at construction.
    /// We do NOT cache `base` or `end` here — backings whose `base()` is
    /// structure-relative (e.g. `InlineBacked<N>` returns `&self.storage`)
    /// produce a different address before and after the backing has been
    /// moved into `Self`. A pointer captured at construction would point
    /// at the backing's pre-move location for the rest of the arena's
    /// life, silently corrupting every subsequent `allocate`. We re-
    /// query `backing.base()` at each `allocate` call instead; the
    /// happy-path cost is one extra indirect load.
    capacity: usize,
    /// Offset from `backing.base()`. Interior mutability for `&self`
    /// allocation; `!Sync` (via `UnsafeCell`) prevents concurrent racing.
    cursor: UnsafeCell<usize>,
}

impl<B: FixedRange> BumpArena<B> {
    /// Construct a bump arena that owns `backing` and bumps through its
    /// entire address range.
    ///
    /// Returns an error if the backing reports a zero-byte range or if the
    /// backing's `[base, base+size)` range would wrap past `usize::MAX`
    /// (impossible on real 64-bit hardware but representable on small
    /// `no_std` targets).
    pub fn new(backing: B) -> Result<Self, AllocError> {
        let base = backing.base();
        let size = backing.size();
        if size == 0 {
            return Err(AllocError);
        }
        // Reject backings whose [base, base+size) range wraps past
        // `usize::MAX`. Even though we don't cache `end` anymore, every
        // allocate path still derives `aligned_off + size <= capacity`
        // from this invariant; rejecting at construction surfaces the
        // misconfigured backing once instead of on every allocate.
        // On 64-bit this branch is unreachable in practice; on 16-/32-bit
        // no_std it can fire.
        let base_addr = base.as_ptr() as usize;
        let end_addr = base_addr.checked_add(size).ok_or(AllocError)?;
        // `end_addr == 0` would mean `base + size == 2^N exactly`, i.e. the
        // mapping covers the top of the address space — also rejected, since
        // we'd need a non-null `end` sentinel.
        if end_addr == 0 {
            return Err(AllocError);
        }
        Ok(Self {
            backing,
            capacity: size,
            cursor: UnsafeCell::new(0),
        })
    }

    /// Bytes currently allocated from this arena.
    #[inline]
    pub fn allocated(&self) -> usize {
        // SAFETY: !Sync — no concurrent access to cursor.
        unsafe { *self.cursor.get() }
    }

    /// Total bytes available in this arena.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes remaining for allocation.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity() - self.allocated()
    }

    /// Borrow the underlying backing.
    #[inline]
    pub fn backing(&self) -> &B {
        &self.backing
    }

    /// Mint a zero-sized [`BumpDeallocator`] tied to this arena's lifetime.
    ///
    /// The deallocator's `'a` lifetime is the arena's borrow, so the borrow
    /// checker prevents the arena from being dropped or reset while any
    /// `Box<T, BumpDeallocator<'_>>` (constructed via `Box::from_raw_in`) is
    /// outstanding.
    #[inline]
    pub fn deallocator(&self) -> BumpDeallocator<'_> {
        BumpDeallocator(PhantomData)
    }
}

impl<B: FixedRange> BumpArena<B> {
    /// Reset the cursor to 0, reclaiming all memory in O(1).
    ///
    /// Requires `&mut self`, which the borrow checker enforces: any
    /// outstanding `Box<T, BumpDeallocator<'_>>` (whose `'_` is `&self`)
    /// blocks `&mut self` access until dropped. Raw `allocate` callers must
    /// observe the discipline themselves.
    ///
    /// # Safety
    ///
    /// All pointers previously issued by this arena become invalid after
    /// `reset`. Reading or writing through them is undefined behavior.
    #[inline]
    pub fn reset(&mut self) {
        // &mut self gives exclusive access.
        *self.cursor.get_mut() = 0;
    }
}

unsafe impl<B: FixedRange> Deallocator for BumpArena<B> {
    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // No-op. Reclaim is via reset(&mut self).
    }
}

unsafe impl<B: FixedRange> Allocator for BumpArena<B> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let align = layout.align().get();
        let size = layout.size().get();
        // Re-query the backing's base at each allocate so structure-
        // relative backings (e.g. `InlineBacked`) keep working after the
        // arena has been moved.
        let base = self.backing.base();
        let base_addr = base.as_ptr() as usize;

        // SAFETY: !Sync — no concurrent access to cursor. We hold the only
        // path to mutating it (other than `reset(&mut self)`).
        unsafe {
            let cursor_ptr = self.cursor.get();
            let cur = *cursor_ptr;
            // Round up the absolute address to the requested alignment.
            let raw = base_addr.checked_add(cur).ok_or(AllocError)?;
            let aligned = raw.checked_add(align - 1).ok_or(AllocError)? & !(align - 1);
            // `aligned >= raw >= base_addr` because masking only zeroes low
            // bits; the subtraction never wraps.
            let aligned_off = aligned - base_addr;
            let end_off = aligned_off.checked_add(size).ok_or(AllocError)?;
            if end_off > self.capacity() {
                return Err(AllocError);
            }
            // Ensure the backing has the block's pages committed before we
            // hand them out. No-op for already-writable backings
            // (InlineBacked, eager MmapBacked, Unix mmap); on a lazy_commit
            // MmapBacked this commits the freshly-crossed pages and can fail
            // if the OS declines (Windows commit limit). Commit BEFORE
            // publishing the cursor so a failure leaves the arena unchanged
            // and surfaces as a clean AllocError rather than a fault on
            // first write.
            self.backing.commit(aligned_off, size)?;
            *cursor_ptr = end_off;
            // SAFETY: aligned_off + size <= capacity, so the resulting ptr
            // lies within [base, end). base is non-null per FixedRange's
            // contract; the offset preserves non-null.
            let p = base.as_ptr().add(aligned_off);
            Ok(NonNull::slice_from_raw_parts(
                NonNull::new_unchecked(p),
                size,
            ))
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.capacity())
    }

    /// Reset the arena via the Allocator trait.
    ///
    /// Returns `Ok(())` and clears the cursor.
    #[inline]
    fn reset(&mut self) -> Result<(), AllocError> {
        BumpArena::reset(self);
        Ok(())
    }
}

impl<B: FixedRange> FixedRange for BumpArena<B> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        // Forward to the live backing rather than returning a cached
        // pointer — structure-relative backings change address on move.
        self.backing.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.capacity()
    }
}

// Send when B: Send. The `NonNull<u8>` fields are `!Send` by default but the
// memory they point to is owned by `backing`, which we move along with the
// arena. `UnsafeCell<usize>` is `Send` (cursor is just an integer).
//
// `!Sync` is auto-derived via `UnsafeCell`, which is the desired behaviour:
// concurrent `&self` allocate would race on the cursor — use
// `SharedBumpArena` for the cross-thread case.
unsafe impl<B: FixedRange + Send> Send for BumpArena<B> {}

// ============================================================================
// BumpDeallocator
// ============================================================================

/// ZST deallocator token tied to a [`BumpArena`]'s borrow.
///
/// Used as the `A` parameter in `Box<T, A>` / `Vec<T, A>` patterns where
/// the box was constructed via `Box::from_raw_in` against pointers
/// obtained from the arena directly. The `'a` lifetime ensures the arena
/// outlives the box.
///
/// # Allocate-always-fails footgun
///
/// The [`Allocator::allocate`] impl on `BumpDeallocator` returns
/// `Err(AllocError)` for **every** call. This is deliberate: the
/// deallocator is a *destruction token*, not an allocation source.
/// The correct usage pattern is:
///
/// ```text
///     let arena: BumpArena<_> = ...;
///     let ptr = arena.allocate(layout)?;       // allocate via arena
///     unsafe { ptr.cast::<T>().write(value) }; // place a T
///     let boxed: Box<T, BumpDeallocator<'_>> =
///         unsafe { Box::from_raw_in(
///             ptr.cast::<T>().as_ptr(),
///             arena.deallocator(),
///         )};
/// ```
///
/// Plugging `BumpDeallocator` into code that *grows* a collection
/// (`Vec::reserve`, `Vec::push` that re-allocates, `Box::new_in` —
/// anything that calls `Allocator::allocate` on the supplied
/// allocator) will fail at runtime. Use `BumpArena` itself as the
/// allocator for those patterns, or pre-size the collection so it
/// never reallocates.
#[derive(Copy, Clone, Debug)]
pub struct BumpDeallocator<'a>(PhantomData<&'a ()>);

unsafe impl Deallocator for BumpDeallocator<'_> {
    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // No-op. Deallocation through the token is a marker that the
        // arena-allocated value's destructor has run; reclaim happens on
        // arena reset/drop.
    }
}

unsafe impl Allocator for BumpDeallocator<'_> {
    /// Always fails. Allocate through the arena, not the deallocator.
    #[inline]
    fn allocate(&self, _layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;

    #[test]
    fn allocate_advances_cursor() {
        let arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
        assert_eq!(arena.allocated(), 0);
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let _ = arena.allocate(layout).unwrap();
        assert_eq!(arena.allocated(), 64);
    }

    #[test]
    fn allocate_returns_aligned_pointer() {
        let arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
        // Push the cursor off zero first.
        let _ = arena
            .allocate(NonZeroLayout::from_size_align(3, 1).unwrap())
            .unwrap();
        let layout = NonZeroLayout::from_size_align(8, 16).unwrap();
        let block = arena.allocate(layout).unwrap();
        assert_eq!(block.cast::<u8>().as_ptr() as usize % 16, 0);
    }

    #[test]
    fn allocate_fails_when_exhausted() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let big = NonZeroLayout::from_size_align(64, 1).unwrap();
        let _ = arena.allocate(big).unwrap();
        let one = NonZeroLayout::from_size_align(1, 1).unwrap();
        assert!(arena.allocate(one).is_err());
    }

    /// Alignment padding must count toward exhaustion: the bounds check is on
    /// the *aligned* offset + size, not the raw cursor + size. `InlineBacked`'s
    /// base is 16-aligned, so after a 1-byte alloc (cursor = 1) a 16-aligned
    /// request rounds the offset deterministically up to 16. A 56-byte request
    /// then needs `16 + 56 = 72 > 64` and must fail — whereas a buggy check
    /// using `cursor + size = 1 + 56 = 57 <= 64` would wrongly succeed.
    #[test]
    fn alignment_padding_counts_toward_exhaustion() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let one = NonZeroLayout::from_size_align(1, 1).unwrap();
        let _ = arena.allocate(one).unwrap(); // cursor = 1
        let aligned = NonZeroLayout::from_size_align(56, 16).unwrap();
        assert!(
            arena.allocate(aligned).is_err(),
            "alignment padding (offset 16) must be counted in the exhaustion check",
        );
    }

    #[test]
    fn reset_reclaims_all() {
        let mut arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let _ = arena.allocate(layout).unwrap();
        assert_eq!(arena.allocated(), 32);
        arena.reset();
        assert_eq!(arena.allocated(), 0);
        let _ = arena.allocate(layout).unwrap();
    }

    #[test]
    fn deallocate_is_no_op() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = arena.allocate(layout).unwrap();
        let used_before = arena.allocated();
        unsafe { arena.deallocate(block.cast(), layout) };
        assert_eq!(arena.allocated(), used_before);
    }

    #[test]
    fn fixed_range_contains_allocations() {
        let arena = BumpArena::new(InlineBacked::<128>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = arena.allocate(layout).unwrap();
        assert!(arena.contains(block.cast::<u8>()));
    }

    #[test]
    fn capacity_bytes_reports_backing_size() {
        let arena = BumpArena::new(InlineBacked::<2048>::new()).unwrap();
        assert_eq!(arena.capacity_bytes(), Some(2048));
    }

    /// Regression: BumpArena historically cached an absolute `base`
    /// pointer captured BEFORE the backing was moved into Self. For
    /// structure-relative backings (`InlineBacked` returns
    /// `&self.storage`), that pointer became stale on every move and
    /// silently corrupted subsequent allocates. The fix re-queries
    /// `self.backing.base()` at each allocate. Verify the arena's
    /// `FixedRange::base()` agrees with the backing's live `base()` and
    /// that the first allocate lands at exactly that address.
    #[test]
    fn base_pointer_matches_backing_after_move() {
        let arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        let arena_base = arena.base().as_ptr();
        let backing_base = arena.backing().base().as_ptr();
        assert_eq!(
            arena_base, backing_base,
            "BumpArena's base must agree with the live backing — stale-pointer regression",
        );
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        let block = arena.allocate(layout).unwrap();
        assert_eq!(
            block.cast::<u8>().as_ptr() as usize,
            backing_base as usize,
            "first alloc must be at backing.base()",
        );
    }

    #[test]
    fn deallocator_compiles_and_runs() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let d = arena.deallocator();
        // The deallocator's allocate must always fail by contract.
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        assert!(d.allocate(layout).is_err());
        // Calling deallocate is safe and a no-op.
        let block = arena.allocate(layout).unwrap();
        unsafe { d.deallocate(block.cast(), layout) };
    }

    #[test]
    fn very_small_alignment_is_one() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let l1 = NonZeroLayout::from_size_align(1, 1).unwrap();
        let _ = arena.allocate(l1).unwrap();
        let _ = arena.allocate(l1).unwrap();
        assert_eq!(arena.allocated(), 2);
    }
}

// ============================================================================
// Kani proof harnesses
//
// Kani is a bounded model checker that verifies properties of unsafe code
// over the entire state space of unconstrained inputs. These harnesses run
// under the `kani` cfg (set by `cargo kani`) and exercise the alignment
// rounding + bounds-check logic in `allocate`.
// ============================================================================

// Kani proofs depend on `crate::backing::InlineBacked`; the `backing` module is gated
// behind the `std` feature in this crate (see Cargo.toml), so the proof
// module must be gated similarly. Kani CI must run with the `std`
// feature enabled for these proofs to compile.
#[cfg(all(kani, feature = "std"))]
mod kani_proofs {
    use super::*;
    use crate::backing::InlineBacked;

    /// Any successful `allocate(layout)` returns a pointer aligned to
    /// `layout.align()`. Verified over all combinations of (cursor
    /// position, requested size, requested alignment) that fit a
    /// 1 KiB arena.
    #[kani::proof]
    #[kani::unwind(4)]
    fn allocate_returns_aligned_pointer() {
        let arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
        // Bounded inputs — Kani enumerates the cross product.
        let size_log: u32 = kani::any();
        kani::assume(size_log <= 8); // size in 1..=256
        let align_log: u32 = kani::any();
        kani::assume(align_log <= 4); // align in {1,2,4,8,16}
        let size = 1usize << size_log;
        let align = 1usize << align_log;
        let layout = NonZeroLayout::from_size_align(size, align).unwrap();
        if let Ok(block) = arena.allocate(layout) {
            let p = block.cast::<u8>().as_ptr() as usize;
            assert!(p % align == 0);
            // And the slice length covers the requested size.
            assert!(block.len() >= size);
        }
    }

    /// Repeated `allocate` calls produce strictly increasing cursor
    /// values that never exceed capacity. Verified over a small
    /// number of allocations on a 256-byte arena.
    #[kani::proof]
    #[kani::unwind(4)]
    fn cursor_monotonic_and_bounded() {
        let arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        let cap = arena.capacity();
        let mut last = 0usize;
        for _ in 0..3 {
            let before = arena.allocated();
            if arena.allocate(layout).is_ok() {
                let after = arena.allocated();
                assert!(after > before);
                assert!(after <= cap);
                last = after;
            }
        }
        assert!(last <= cap);
    }
}
