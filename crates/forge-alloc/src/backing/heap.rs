//! `HeapBytes` — owner of a single global-allocator block, exposed as
//! [`FixedRange`].
//!
//! The heap twin of [`MmapBacked`](crate::MmapBacked)'s region-ownership
//! half. Use when you need a contiguous bounded region to compose under
//! `BumpArena` / `Slab` but mmap-level isolation (separate VM area,
//! guard-page potential) isn't worth the syscall cost.
//!
//! See the type-level documentation on [`HeapBytes`] for the factoring
//! rationale and drop-order discipline.

use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator as A2, Global};
use forge_alloc_core::{AllocError, FixedRange, NonZeroLayout};

/// Owner of a single global-allocator block, exposed as [`FixedRange`].
///
/// Pair with [`BumpArena<HeapBytes>`](crate::BumpArena) for bump
/// semantics, or [`Slab<T, BumpArena<HeapBytes>>`](crate::Slab) for
/// typed slot allocation, when you need a contiguous bounded region but
/// mmap-level isolation isn't worth the ~10-50 µs syscall cost. No
/// guard pages, no separate VM area — `HeapBytes` is essentially
/// `MmapBacked`'s heap twin for "I just need a contiguous bounded
/// region, fast."
///
/// # Drop order discipline
///
/// When composed as `Slab<T, BumpArena<HeapBytes>>` (or any wrapper
/// stack), the outer wrappers (`Slab`, `BumpArena`) drop first — their
/// drops run freelist sanity checks, no-op deallocates, etc. — and
/// then `HeapBytes::drop` returns the underlying block to the global
/// allocator. This mirrors `MmapBacked`'s drop discipline and must hold
/// for the composition to be sound; storing the exact layout (not just
/// the size) lets us hand the same `Layout` back to `Global::deallocate`
/// that we got from `Global::allocate`.
///
/// # Factoring vs `MmapBacked` / `InlineBacked`
///
/// [`MmapBacked`](crate::MmapBacked) and [`InlineBacked`](crate::InlineBacked)
/// double-duty as `FixedRange` AND `Allocator` (each carries its own
/// internal bump cursor). `HeapBytes` deliberately does not — bump
/// semantics live in [`BumpArena<B>`](crate::BumpArena), which only
/// requires `B: FixedRange`. The result is one cleanly factored heap
/// backing rather than a fourth bump-cursor implementation. Refactoring
/// `MmapBacked` / `InlineBacked` to the same shape is possible but out
/// of scope; both still work as-is.
///
/// # Thread safety
///
/// `Send` (the heap block is safely transferable between threads — the
/// global allocator is thread-safe), `!Sync` (auto-derived; `NonNull`
/// blocks the auto-impl). The `!Sync` is conservative: today every
/// method is a pure read of `base` / `size` and could in principle
/// safely cross threads with `&`, but leaving it `!Sync` matches
/// `MmapBacked` and reserves room for a future cursor field without a
/// breaking change.
pub struct HeapBytes {
    ptr: NonNull<u8>,
    /// Exact layout passed to `Global::allocate`; must be handed back
    /// verbatim to `Global::deallocate` at drop. Named `block_layout`
    /// (not `layout`) so it cannot be confused with the per-carve
    /// `layout: NonZeroLayout` argument typical of `Allocator::allocate`
    /// implementations. `HeapBytes` does not implement `Allocator`,
    /// but the distinction matters for any future reader who adds one.
    block_layout: NonZeroLayout,
}

impl HeapBytes {
    /// Allocate `size` bytes from the global allocator, aligned to
    /// [`MAX_ALIGN`](crate::MAX_ALIGN). Errors if `size == 0` or the
    /// allocator returns `AllocError`. Block contents are
    /// uninitialized; use [`new_zeroed`](Self::new_zeroed) if you
    /// need calloc semantics.
    #[inline]
    pub fn new(size: usize) -> Result<Self, AllocError> {
        Self::with_align(size, crate::MAX_ALIGN)
    }

    /// Allocate `size` bytes from the global allocator, zero-initialized,
    /// aligned to [`MAX_ALIGN`](crate::MAX_ALIGN). Same error
    /// conditions as [`new`](Self::new).
    ///
    /// Delegates to the global allocator's `allocate_zeroed`. With
    /// the default `System` allocator this typically calls `calloc`
    /// (which on glibc/musl Linux and macOS, large allocations
    /// above `MMAP_THRESHOLD` get fresh `mmap`-backed zero pages
    /// without a userspace memset; smaller heap-arena allocations
    /// still memset). With a custom `#[global_allocator]`
    /// (`jemalloc`, `mimalloc`, `snmalloc`) the actual behavior
    /// depends on the impl — most do a memset for any cached
    /// block.
    #[inline]
    pub fn new_zeroed(size: usize) -> Result<Self, AllocError> {
        Self::with_align_zeroed(size, crate::MAX_ALIGN)
    }

    /// Allocate `size` bytes from the global allocator with a custom
    /// alignment. Errors if:
    ///
    /// - `size == 0` (matches `MmapBacked`'s zero-size rejection),
    /// - `align == 0`,
    /// - `align` is not a power of two,
    /// - layout construction fails (e.g. `size` would overflow when
    ///   rounded up to `align`), or
    /// - the global allocator returns `AllocError`.
    pub fn with_align(size: usize, align: usize) -> Result<Self, AllocError> {
        if size == 0 || align == 0 || !align.is_power_of_two() {
            return Err(AllocError);
        }
        let block_layout = NonZeroLayout::from_size_align(size, align).map_err(|_| AllocError)?;
        let block = A2::allocate(&Global, block_layout.to_layout())?;
        let ptr = block.cast::<u8>();
        Ok(Self { ptr, block_layout })
    }

    /// Allocate `size` bytes from the global allocator with a custom
    /// alignment, zero-initialized. Same error conditions as
    /// [`with_align`](Self::with_align).
    pub fn with_align_zeroed(size: usize, align: usize) -> Result<Self, AllocError> {
        if size == 0 || align == 0 || !align.is_power_of_two() {
            return Err(AllocError);
        }
        let block_layout = NonZeroLayout::from_size_align(size, align).map_err(|_| AllocError)?;
        let block = A2::allocate_zeroed(&Global, block_layout.to_layout())?;
        let ptr = block.cast::<u8>();
        Ok(Self { ptr, block_layout })
    }

    /// Capacity in bytes — equal to the size requested at construction.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.block_layout.size().get()
    }
}

impl Drop for HeapBytes {
    fn drop(&mut self) {
        // SAFETY:
        // - `self.ptr` came from either `Global::allocate`
        //   (`with_align`) or `Global::allocate_zeroed`
        //   (`with_align_zeroed`); no other code path produces a
        //   `HeapBytes`. Both satisfy `Global::deallocate`'s
        //   contract identically — zero-init at allocation time
        //   does not change the deallocation requirements.
        // - We hand back the *same* layout we used to allocate, satisfying
        //   the `GlobalAlloc::dealloc` contract.
        // - No `Clone` impl, no path that copies `self.ptr` outside the
        //   struct, so this is the last live reference to the block.
        // - Drop-order discipline (see type docs): outer wrappers
        //   (`Slab`, `BumpArena`) drop their views of the region BEFORE
        //   `HeapBytes::drop` runs, so no live carved pointers reach
        //   into freed memory here.
        unsafe { A2::deallocate(&Global, self.ptr, self.block_layout.to_layout()) }
    }
}

impl FixedRange for HeapBytes {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.ptr
    }

    #[inline]
    fn size(&self) -> usize {
        self.block_layout.size().get()
    }
}

// SAFETY: `HeapBytes` owns a single heap allocation via `NonNull`.
// `NonNull<u8>` is `!Send` by default but the block it points at IS
// safely transferable between threads — the global allocator is
// thread-safe to call from any thread, and there is no thread-local
// state inside `HeapBytes`. `!Sync` is preserved by `NonNull`'s
// auto-impl block (no explicit `unsafe impl Sync` here).
unsafe impl Send for HeapBytes {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BumpArena, Slab};
    use forge_alloc_core::{Allocator, Deallocator};

    #[test]
    fn new_zero_size_errors() {
        assert!(HeapBytes::new(0).is_err());
    }

    #[test]
    fn with_align_zero_errors() {
        assert!(HeapBytes::with_align(64, 0).is_err());
    }

    #[test]
    fn with_align_non_power_of_two_errors() {
        // Spot-check a few representative non-powers-of-two.
        for bad in [3_usize, 5, 6, 12, 24] {
            assert!(
                HeapBytes::with_align(64, bad).is_err(),
                "align={bad} should be rejected",
            );
        }
    }

    #[test]
    fn with_align_power_of_two_succeeds() {
        for align in [1_usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            assert!(
                HeapBytes::with_align(1024, align).is_ok(),
                "align={align} should succeed",
            );
        }
    }

    #[test]
    fn capacity_matches_requested_size() {
        assert_eq!(HeapBytes::new(1024).unwrap().capacity(), 1024);
        assert_eq!(HeapBytes::with_align(2048, 64).unwrap().capacity(), 2048);
    }

    #[test]
    fn base_is_stable_across_observations() {
        let h = HeapBytes::new(1024).unwrap();
        let a = h.base().as_ptr();
        let b = h.base().as_ptr();
        assert_eq!(a, b, "base() must report the same address every call");
    }

    #[test]
    fn fixed_range_size_matches_capacity() {
        let h = HeapBytes::new(1024).unwrap();
        assert_eq!(h.size(), h.capacity());
    }

    /// Composition smoke: `BumpArena` over `HeapBytes` carves
    /// independent allocations within the heap region.
    #[test]
    fn bump_arena_over_heap_bytes_round_trips() {
        let bump = BumpArena::new(HeapBytes::new(1024).unwrap()).unwrap();
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let a = bump.allocate(layout).unwrap();
        let b = bump.allocate(layout).unwrap();
        assert_ne!(
            a.cast::<u8>().as_ptr(),
            b.cast::<u8>().as_ptr(),
            "two allocates must return distinct pointers",
        );
        // Both lie inside the HeapBytes region.
        let base = bump.base().as_ptr() as usize;
        let size = bump.size();
        for p in [a, b] {
            let addr = p.cast::<u8>().as_ptr() as usize;
            assert!(
                addr >= base && addr + 64 <= base + size,
                "allocation at {addr:#x} must lie in region [{base:#x}, {:#x})",
                base + size,
            );
        }
    }

    #[test]
    fn new_zeroed_returns_zeroed_block() {
        // Cap the max size under Miri so the byte-by-byte verify
        // loop doesn't blow CI's Miri-step budget. Native test
        // covers the full progression.
        #[cfg(not(miri))]
        let sizes: &[usize] = &[1, 16, 64, 1024, 64 * 1024, 1024 * 1024];
        #[cfg(miri)]
        let sizes: &[usize] = &[1, 16, 64, 1024];
        for &size in sizes {
            let h = HeapBytes::new_zeroed(size).unwrap();
            let base = h.base().as_ptr();
            assert_eq!(h.size(), size);
            // SAFETY: base..base+size is the freshly allocated block we own;
            // reading bytes is sound, the bytes are guaranteed initialized
            // to zero by `Global::allocate_zeroed`.
            unsafe {
                for i in 0..size {
                    assert_eq!(*base.add(i), 0, "size={size}, byte {i} must be zero");
                }
            }
        }
    }

    #[test]
    fn new_zeroed_zero_size_errors() {
        assert!(HeapBytes::new_zeroed(0).is_err());
    }

    #[test]
    fn with_align_zeroed_validates_args_same_as_with_align() {
        // Same validation table as with_align.
        assert!(HeapBytes::with_align_zeroed(64, 0).is_err());
        for bad in [3_usize, 5, 6, 12, 24] {
            assert!(HeapBytes::with_align_zeroed(64, bad).is_err());
        }
        for align in [1_usize, 2, 4, 8, 16, 32, 64, 128, 256] {
            let h = HeapBytes::with_align_zeroed(1024, align).unwrap();
            assert_eq!(h.capacity(), 1024);
            // Spot-check first and last byte are zero.
            let base = h.base().as_ptr();
            // SAFETY: in-range reads on a zeroed block we own.
            unsafe {
                assert_eq!(*base, 0);
                assert_eq!(*base.add(1023), 0);
            }
        }
    }

    /// Full stack: `Slab<u64, BumpArena<HeapBytes>>` — alloc and
    /// dealloc some slots and verify the round trip survives. Catches
    /// drop-order or layout-invariant bugs in the composition. Miri
    /// running over this test catches any leak in `HeapBytes::drop`.
    #[test]
    fn slab_over_bump_over_heap_round_trips() {
        let bump = BumpArena::new(HeapBytes::new(4096).unwrap()).unwrap();
        let slab: Slab<u64, _> = Slab::new(8, bump).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = slab.allocate(layout).unwrap();
        let b = slab.allocate(layout).unwrap();
        assert_ne!(
            a.cast::<u8>().as_ptr(),
            b.cast::<u8>().as_ptr(),
            "two Slab allocates must return distinct slots",
        );
        unsafe {
            slab.deallocate(a.cast(), layout);
            slab.deallocate(b.cast(), layout);
        }
        // Drop of slab + bump + HeapBytes happens here; under Miri,
        // any leak in HeapBytes::drop would surface.
    }
}
