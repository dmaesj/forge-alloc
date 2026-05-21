//! `InlineBacked<N>` — fixed-size inline storage.
//!
//! `N` bytes of memory live inside the struct itself. No heap involvement, no
//! OS calls. Compiles under `no_std`. Suitable for `BumpArena<InlineBacked<N>>`
//! patterns where the entire allocator lives on the stack, in BSS, or inside
//! a `Box`.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use forge_alloc_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// Maximum supported alignment for `InlineBacked` allocations.
///
/// Covers all standard Rust scalar types and `repr(C)` layouts produced by
/// `#[repr(align(16))]`. Higher alignments (page-aligned, cache-line-aligned)
/// require [`MmapBacked`](super::MmapBacked) instead.
pub const MAX_ALIGN: usize = 16;

/// Fixed-size inline storage backing.
///
/// The `N` bytes are stored inline in the struct, aligned to [`MAX_ALIGN`].
/// Allocations bump a cursor through this buffer; deallocation is a no-op.
/// Reset via [`reset`](Self::reset) (requires `&mut self`).
///
/// `N` MUST be a multiple of `core::mem::align_of::<usize>()`; constructing
/// with any other `N` is a compile-time error (`const _: () = assert!(…)`
/// inside `new`).
///
/// # API-misuse compile-failures (pinned)
///
/// `N` must satisfy `N % core::mem::align_of::<usize>() == 0`. On the
/// 32- and 64-bit targets this crate supports, `align_of::<usize>()` is
/// 4 or 8 respectively, so `N = 7` is never valid. The const assert
/// (`ASSERT_N_ALIGNED`) fires at the call site of `new()`:
///
/// ```compile_fail
/// // FAILS TO COMPILE: 7 is not a multiple of align_of::<usize>()
/// // on any supported target, so `ASSERT_N_ALIGNED` panics at const
/// // evaluation inside `InlineBacked::<7>::new()`.
/// use forge_alloc::InlineBacked;
/// let _ = InlineBacked::<7>::new();
/// ```
///
/// # Move invariant (structure-relative backing)
///
/// `base()` returns a pointer derived from `&self.storage`, which lives
/// inside the struct itself. Moving an `InlineBacked` (return by value,
/// `mem::swap`, `Box::new`, putting it into a container that reallocates,
/// etc.) relocates its inline storage to a new address. After such a move
/// `base()` correctly reflects the new location, but **any pointer
/// previously returned from [`allocate`](Self::allocate) is now dangling**
/// — it points at the old, now-deallocated stack slot or heap cell.
///
/// Wrappers that store an `InlineBacked` (e.g. `BumpArena<InlineBacked<N>>`)
/// must NOT cache `base()` at construction; they must re-query on every
/// allocation. The wrappers in this crate family follow that discipline.
/// Raw callers who hold their own `NonNull<u8>` between allocate sites
/// are responsible for pinning the `InlineBacked` (e.g. via `Pin<Box<…>>`
/// or by keeping it in a local that the borrow checker proves is not
/// moved) before issuing the allocation.
///
/// # Thread safety
///
/// `Send`: yes — the storage is owned (auto-derived; both `UnsafeCell` fields
/// hold `Send` payloads).
/// `Sync`: NO. The cursor uses `UnsafeCell` so that `Allocator::allocate` can
/// take `&self`; concurrent `&self` allocation would race on the cursor.
/// `UnsafeCell<T>` is `!Sync` regardless of `T`, which gives us the right
/// behavior without any extra marker field. If you need cross-thread
/// allocation use a higher-layer wrapper that adds atomicity (e.g.
/// `SharedBumpArena`).
#[repr(C, align(16))]
pub struct InlineBacked<const N: usize> {
    storage: UnsafeCell<MaybeUninit<[u8; N]>>,
    cursor: UnsafeCell<usize>,
}

impl<const N: usize> InlineBacked<N> {
    /// Compile-time check that `N` is a multiple of `align_of::<usize>()`.
    /// Referenced from [`new`](Self::new); referencing the associated const
    /// forces evaluation, which fails compilation if the invariant is violated.
    /// (We use an associated const rather than `const { ... }` because inline
    /// const blocks weren't stabilised until 1.79; MSRV is 1.70.)
    const ASSERT_N_ALIGNED: () = assert!(
        N % core::mem::align_of::<usize>() == 0,
        "InlineBacked<N>: N must be a multiple of align_of::<usize>()",
    );

    /// Construct empty inline storage.
    ///
    /// Compile-time enforces that `N` is a multiple of
    /// `core::mem::align_of::<usize>()`.
    #[inline]
    pub const fn new() -> Self {
        // Force evaluation of the compile-time check.
        let _: () = Self::ASSERT_N_ALIGNED;
        Self {
            storage: UnsafeCell::new(MaybeUninit::uninit()),
            cursor: UnsafeCell::new(0),
        }
    }

    /// Bytes already allocated from this backing.
    #[inline]
    pub fn allocated(&self) -> usize {
        // SAFETY: !Sync — no concurrent access to cursor.
        unsafe { *self.cursor.get() }
    }

    /// Capacity in bytes — always `N`.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Bytes remaining for allocation.
    #[inline]
    pub fn remaining(&self) -> usize {
        N - self.allocated()
    }

    /// Reset the cursor to 0, reclaiming all allocated memory.
    ///
    /// # Safety
    ///
    /// All previously issued pointers become invalid. The caller is
    /// responsible for ensuring no outstanding pointer is read or written
    /// after this call. The `&mut self` receiver and `BumpDeallocator<'a>`
    /// lifetime patterns enforce this at compile time for Box-style
    /// usage; raw `allocate` callers must enforce it themselves.
    #[inline]
    pub fn reset(&mut self) {
        // &mut self gives us exclusive access. No unsafe needed for the write.
        *self.cursor.get_mut() = 0;
    }

    /// Pointer to the start of the inline buffer.
    #[inline]
    fn buffer_base(&self) -> *mut u8 {
        self.storage.get() as *mut u8
    }
}

impl<const N: usize> Default for InlineBacked<N> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<const N: usize> Deallocator for InlineBacked<N> {
    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // No-op. Bump-style backing reclaims only via reset(&mut self).
    }
}

unsafe impl<const N: usize> Allocator for InlineBacked<N> {
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let align = layout.align().get();
        if align > MAX_ALIGN {
            return Err(AllocError);
        }
        let size = layout.size().get();

        // SAFETY: !Sync — no concurrent access to cursor. The unsafe block
        // brackets the cursor read/write so we can keep them in one place.
        unsafe {
            let cursor_ptr = self.cursor.get();
            let cur = *cursor_ptr;
            let base = self.buffer_base() as usize;
            // Use checked_add throughout so a high `base` address can never
            // wrap silently (debug builds would panic on the bare `+`; release
            // builds would wrap and hand out an address outside the buffer).
            let next = base
                .checked_add(cur)
                .and_then(|v| v.checked_add(align - 1))
                .ok_or(AllocError)?
                & !(align - 1);
            // Recompute the offset relative to base after alignment rounding.
            // `next >= base + cur >= base` after the mask (the mask only
            // zeroes low bits), so this subtraction never wraps.
            let aligned_off = next - base;
            // Check (aligned_off + size) <= N without overflow.
            let Some(end_off) = aligned_off.checked_add(size) else {
                return Err(AllocError);
            };
            if end_off > N {
                return Err(AllocError);
            }
            *cursor_ptr = end_off;
            let ptr = self.buffer_base().add(aligned_off);
            // ptr is non-null because buffer_base is non-null (UnsafeCell's
            // address is the struct's address, which is non-null for any
            // valid &self).
            Ok(NonNull::slice_from_raw_parts(
                NonNull::new_unchecked(ptr),
                size,
            ))
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        Some(N)
    }
}

impl<const N: usize> FixedRange for InlineBacked<N> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        // SAFETY: buffer_base derives from a valid &self, hence non-null.
        unsafe { NonNull::new_unchecked(self.buffer_base()) }
    }

    #[inline]
    fn size(&self) -> usize {
        N
    }
}

// Send is auto-derived: both `UnsafeCell<MaybeUninit<[u8; N]>>` and
// `UnsafeCell<usize>` are `Send`. We keep a `const _` check here so that any
// future field addition that would break `Send` fails the build loudly.
// (N=8 is a multiple of `align_of::<usize>()` on both 32- and 64-bit, which
// satisfies the const assert in `new()` if it were ever to fire.)
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<InlineBacked<8>>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_then_remaining_decreases() {
        let b = InlineBacked::<1024>::new();
        assert_eq!(b.remaining(), 1024);
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let _ = b.allocate(layout).unwrap();
        assert_eq!(b.remaining(), 1024 - 64);
    }

    #[test]
    fn alloc_returns_aligned_pointer() {
        let b = InlineBacked::<256>::new();
        // First, allocate one byte to push the cursor off zero.
        let _ = b
            .allocate(NonZeroLayout::from_size_align(1, 1).unwrap())
            .unwrap();
        // Then request 16-byte aligned — must round up.
        let layout = NonZeroLayout::from_size_align(8, 16).unwrap();
        let block = b.allocate(layout).unwrap();
        let addr = block.cast::<u8>().as_ptr() as usize;
        assert_eq!(addr % 16, 0, "returned ptr must be 16-aligned");
    }

    #[test]
    fn alloc_fails_when_exhausted() {
        let b = InlineBacked::<64>::new();
        let big = NonZeroLayout::from_size_align(64, 1).unwrap();
        let _ = b.allocate(big).unwrap();
        // One more byte must fail.
        let one_more = NonZeroLayout::from_size_align(1, 1).unwrap();
        assert!(b.allocate(one_more).is_err());
    }

    #[test]
    fn alloc_fails_when_align_too_high() {
        let b = InlineBacked::<256>::new();
        let layout = NonZeroLayout::from_size_align(8, 32).unwrap();
        assert!(b.allocate(layout).is_err());
    }

    #[test]
    fn reset_reclaims_everything() {
        let mut b = InlineBacked::<64>::new();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let _ = b.allocate(layout).unwrap();
        assert_eq!(b.remaining(), 32);
        b.reset();
        assert_eq!(b.remaining(), 64);
        // Subsequent allocation must succeed.
        let _ = b.allocate(layout).unwrap();
    }

    #[test]
    fn deallocate_is_no_op() {
        let b = InlineBacked::<64>::new();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = b.allocate(layout).unwrap();
        let used_before = b.allocated();
        unsafe { b.deallocate(block.cast(), layout) };
        // Allocated didn't shrink — bump-style: deallocate is no-op.
        assert_eq!(b.allocated(), used_before);
    }

    #[test]
    fn fixed_range_contains_returned_pointers() {
        let b = InlineBacked::<128>::new();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = b.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        assert!(b.contains(ptr));
        // A pointer well outside the buffer must not be reported as contained.
        // Construct via integer math (no provenance) and `NonNull::new` rather
        // than `.add()` (which has in-bounds preconditions).
        let outside_addr = b.buffer_base().wrapping_add(N_FAR_OUTSIDE);
        if let Some(outside) = NonNull::new(outside_addr) {
            assert!(!b.contains(outside));
        }
    }
    const N_FAR_OUTSIDE: usize = 1 << 20; // 1 MiB past base — well outside 128 bytes

    #[test]
    fn capacity_bytes_reports_n() {
        let b = InlineBacked::<2048>::new();
        assert_eq!(b.capacity_bytes(), Some(2048));
    }

    /// Regression: `base()` must NOT be cached at construction.
    /// Returning an `InlineBacked` by value relocates the inline storage,
    /// so a post-move `base()` query MUST reflect the new location.
    ///
    /// Constructs an `InlineBacked` inside a helper, captures `base()` after
    /// guaranteed-move arithmetic, then captures `base()` again at a moved
    /// site and asserts they agree with `&self.storage` at each point.
    /// (We cannot directly compare pre-move vs. post-move addresses because
    /// the compiler may apply NRVO and elide the move, in which case the
    /// addresses *would* coincide. The invariant we DO want to pin is
    /// "`base()` always points at the live `&self.storage`" — that's what
    /// this test verifies in two places, with a forced reborrow between.)
    #[test]
    fn base_tracks_storage_after_return_by_value() {
        fn make() -> InlineBacked<64> {
            let b = InlineBacked::<64>::new();
            // base() at construction site must equal &b.storage right now.
            let storage_here = b.storage.get() as *mut u8 as usize;
            let base_here = b.base().as_ptr() as usize;
            assert_eq!(
                storage_here, base_here,
                "base() must track &self.storage in callee"
            );
            b
        }

        let b = make();
        // base() at the receiving site must equal &b.storage *now*, regardless
        // of whether NRVO elided the move. If `base()` had cached the old
        // address at construction, a moved-without-NRVO build would fail this
        // assertion; an NRVO build would pass either way. Either way the
        // invariant "base() == &self.storage" holds, which is the property
        // every consumer (BumpArena, etc.) relies on.
        let storage_now = b.storage.get() as *mut u8 as usize;
        let base_now = b.base().as_ptr() as usize;
        assert_eq!(
            storage_now, base_now,
            "base() must track &self.storage after move/return",
        );
    }

    /// Companion to the above: construct an `InlineBacked` in a helper,
    /// return it by value, then allocate **after** the move and verify the
    /// returned region lies inside `[base(), base()+size())`. This is the
    /// "structure-relative backing keeps working after a move" invariant —
    /// the exact property that `BumpArena<InlineBacked<N>>` relies on.
    ///
    /// Note: allocating BEFORE the move and inspecting the returned ptr
    /// AFTER the move would not test this invariant — per the type's
    /// "Move invariant" doc section, the pre-move ptr is dangling after a
    /// move; the type is move-invalidates-outstanding-pointers by design.
    #[test]
    fn allocate_after_return_by_value_stays_inside_range() {
        fn make() -> InlineBacked<128> {
            InlineBacked::<128>::new()
        }

        let b = make();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = b.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        assert!(
            b.contains(ptr),
            "allocation issued AFTER move must be inside base()..base()+size()",
        );
        // Stronger: address arithmetic must be exact relative to the
        // post-move base, not a stale pre-move one.
        let base_addr = b.base().as_ptr() as usize;
        let ptr_addr = ptr.as_ptr() as usize;
        assert!(ptr_addr >= base_addr && ptr_addr < base_addr + b.size());
    }

    /// `MAX_ALIGN` is the contract advertised to callers; `#[repr(align(N))]`
    /// is the layout the compiler actually uses. They MUST agree, otherwise
    /// `allocate` will hand out under-aligned pointers (the cursor math
    /// assumes the struct's base is `MAX_ALIGN`-aligned).
    #[test]
    fn max_align_matches_struct_alignment() {
        assert_eq!(
            core::mem::align_of::<InlineBacked<64>>(),
            MAX_ALIGN,
            "MAX_ALIGN must match #[repr(align(N))] on the struct",
        );
    }

    /// `InlineBacked<0>` is legal to construct (0 satisfies the
    /// `N % align_of::<usize>() == 0` const assert) but must reject every
    /// allocation request — there is no buffer to bump through.
    #[test]
    fn zero_capacity_rejects_all_allocations() {
        let b = InlineBacked::<0>::new();
        assert_eq!(b.capacity(), 0);
        assert_eq!(b.remaining(), 0);
        let layout = NonZeroLayout::from_size_align(1, 1).unwrap();
        assert!(b.allocate(layout).is_err());
    }

    /// `InlineBacked` is `Send` (storage and cursor are both `Send`) but NOT
    /// `Sync` — `UnsafeCell` makes `&InlineBacked` non-shareable across
    /// threads, which is the invariant `allocate(&self)` relies on for cursor
    /// safety. We can't statically assert `!Sync` (no stable negative
    /// bounds), so this test uses runtime trait-object dispatch: the call
    /// compiles only because `InlineBacked<N>: Send`.
    #[test]
    fn inline_backed_is_send() {
        fn assert_send<T: Send>(_: &T) {}
        let b = InlineBacked::<64>::new();
        assert_send(&b);
    }
}
