//! `GuardPage<I>` — wraps an [`OsBacked`] allocator with leading and trailing
//! unmapped guard pages. Walking past the usable region (linear overflow,
//! out-of-bounds write) triggers a fault rather than corrupting adjacent
//! memory.
//!
//! Unlike a thin "decorator" wrapper, `GuardPage` does not forward to
//! `inner.allocate`. The inner OsBacked's own bump cursor is unaware of the
//! protected pages and would happily hand out pointers inside them. Instead,
//! `GuardPage` carves the usable subrange (`[base + page_size, base +
//! region_size - page_size)`) at construction and manages its own cursor
//! inside that subrange.
//!
//! See spec §7.2.

use core::cell::UnsafeCell;
use core::ptr::NonNull;

use forge_core::{
    AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout, OsBacked, ProtectFlags,
};

/// Guard-page wrapper.
///
/// At construction, `inner.protect(...)` marks the first and last
/// `page_size` bytes of the inner region as `PROT_NONE`. The wrapper then
/// allocates bump-style from the strictly-interior subrange.
///
/// # Thread safety
///
/// `Send` when `I: Send`. `Sync`: NO — the cursor uses `UnsafeCell`.
pub struct GuardPage<I: OsBacked> {
    inner: I,
    page_size: usize,
    /// Base of the usable subrange (inner.base + page_size).
    base: NonNull<u8>,
    /// Bytes of usable subrange (inner.region - 2*page_size).
    usable_size: usize,
    /// Bump cursor inside the usable subrange.
    cursor: UnsafeCell<usize>,
}

impl<I: OsBacked> GuardPage<I> {
    /// Wrap with leading + trailing guard pages of `page_size` bytes each.
    ///
    /// Errors if `page_size` is zero / not a power of two, if `inner.base_ptr()`
    /// is not `page_size`-aligned, or if the region is too small.
    pub fn new(inner: I, page_size: usize) -> Result<Self, AllocError> {
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(AllocError);
        }
        let region_size = inner.region_size();
        let needed = 2usize
            .checked_mul(page_size)
            .and_then(|v| v.checked_add(1))
            .ok_or(AllocError)?;
        if region_size < needed {
            return Err(AllocError);
        }
        let base_addr = inner.base_ptr().as_ptr() as usize;
        if base_addr & (page_size - 1) != 0 {
            return Err(AllocError);
        }

        // Install guards.
        // The `protect` trait is infallible-by-signature, but the underlying
        // syscall (mprotect / VirtualProtect) can fail. For a security
        // wrapper this is critical: a silent mprotect failure leaves the
        // "guard" pages writable, defeating the entire purpose. Drain the
        // per-thread last-error slot before each call and abort construction
        // if either protect raised an error.
        // SAFETY: head/tail ranges lie inside the inner region per the checks
        // above; no live allocations have been served by us yet, and inner's
        // own cursor is at 0 (fresh).
        forge_backing::mmap_clear_last_os_error();
        unsafe {
            inner.protect(inner.base_ptr(), page_size, ProtectFlags::NONE);
        }
        if forge_backing::mmap_last_os_error().is_some() {
            return Err(AllocError);
        }
        // SAFETY: same as above; tail range lies at the very end of the
        // inner region per the size check.
        unsafe {
            let tail = inner.base_ptr().as_ptr().add(region_size - page_size);
            inner.protect(NonNull::new_unchecked(tail), page_size, ProtectFlags::NONE);
        }
        if forge_backing::mmap_last_os_error().is_some() {
            return Err(AllocError);
        }

        // SAFETY: base + page_size is in-range (region_size > page_size verified).
        let base = unsafe { NonNull::new_unchecked(inner.base_ptr().as_ptr().add(page_size)) };
        let usable_size = region_size - 2 * page_size;

        Ok(Self {
            inner,
            page_size,
            base,
            usable_size,
            cursor: UnsafeCell::new(0),
        })
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }

    /// Guard page size in bytes.
    #[inline]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Bytes currently allocated from the usable subrange.
    #[inline]
    pub fn allocated(&self) -> usize {
        // SAFETY: !Sync.
        unsafe { *self.cursor.get() }
    }
}

unsafe impl<I: OsBacked> Deallocator for GuardPage<I> {
    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // Bump-style: reclaim via drop.
    }
}

unsafe impl<I: OsBacked> Allocator for GuardPage<I> {
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let align = layout.align().get();
        let size = layout.size().get();
        let base_addr = self.base.as_ptr() as usize;
        // SAFETY: !Sync.
        unsafe {
            let cursor_ptr = self.cursor.get();
            let cur = *cursor_ptr;
            let raw = base_addr.checked_add(cur).ok_or(AllocError)?;
            let aligned = raw.checked_add(align - 1).ok_or(AllocError)? & !(align - 1);
            let aligned_off = aligned - base_addr;
            let end_off = aligned_off.checked_add(size).ok_or(AllocError)?;
            if end_off > self.usable_size {
                return Err(AllocError);
            }
            *cursor_ptr = end_off;
            let p = self.base.as_ptr().add(aligned_off);
            Ok(NonNull::slice_from_raw_parts(
                NonNull::new_unchecked(p),
                size,
            ))
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.usable_size)
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        // GuardPage doesn't have a Rust-observable corruption site —
        // the guard pages trap via SIGSEGV / VirtualProtect at the OS
        // level, never returning control. The inner allocator is the
        // backing region, not a wrappable allocator; no forward needed.
        0
    }
}

impl<I: OsBacked> FixedRange for GuardPage<I> {
    /// Base of the usable subrange (not the inner region).
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.base
    }

    /// Usable bytes (inner region minus the two guard pages).
    #[inline]
    fn size(&self) -> usize {
        self.usable_size
    }
}

// Send when I: Send. !Sync via UnsafeCell.
unsafe impl<I: OsBacked + Send> Send for GuardPage<I> {}

#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use super::*;
    use forge_backing::MmapBacked;

    fn page_size() -> usize {
        4096
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn construct_succeeds_for_large_region() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        let g = GuardPage::new(inner, page_size()).unwrap();
        assert!(g.capacity_bytes().unwrap() >= 8 * 1024);
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn construct_rejects_undersized_region() {
        let inner = MmapBacked::new(4096).unwrap();
        assert!(GuardPage::new(inner, page_size()).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn construct_rejects_non_power_of_two_page() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        assert!(GuardPage::new(inner, 3000).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn allocate_within_usable_doesnt_fault() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        let g = GuardPage::new(inner, page_size()).unwrap();
        let layout = NonZeroLayout::from_size_align(256, 8).unwrap();
        let block = g.allocate(layout).unwrap();
        let p = block.cast::<u8>();
        unsafe { core::ptr::write_bytes(p.as_ptr(), 0xAA, 256) };
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn allocate_returns_ptr_past_head_guard() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        let inner_base = inner.base_ptr().as_ptr() as usize;
        let g = GuardPage::new(inner, page_size()).unwrap();
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        let block = g.allocate(layout).unwrap();
        let p_addr = block.cast::<u8>().as_ptr() as usize;
        assert!(
            p_addr >= inner_base + page_size(),
            "allocation must sit beyond the head guard"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn allocate_rejects_oversized_request() {
        let inner = MmapBacked::new(16 * 1024).unwrap();
        let g = GuardPage::new(inner, page_size()).unwrap();
        let huge = NonZeroLayout::from_size_align(64 * 1024, 1).unwrap();
        assert!(g.allocate(huge).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn fixed_range_excludes_guards() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        let inner_base = inner.base_ptr().as_ptr() as usize;
        let inner_size = inner.region_size();
        let g = GuardPage::new(inner, page_size()).unwrap();
        assert_eq!(g.base().as_ptr() as usize, inner_base + page_size());
        assert_eq!(g.size(), inner_size - 2 * page_size());
    }
}
