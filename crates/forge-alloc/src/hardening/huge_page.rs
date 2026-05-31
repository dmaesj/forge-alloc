//! `HugePageAligned<I>` — wraps an [`OsBacked`] allocator with a 2 MiB
//! (32 MiB on Apple Silicon) layout contract.
//!
//! Two effects:
//!
//! 1. **Allocation alignment** is forced up to `huge_page_size` so any
//!    allocation served by this wrapper *starts* on a huge-page boundary.
//!    Boundary alignment is a *precondition* for the OS to promote the
//!    region to a huge page, not a trigger for it.
//! 2. **Purges below the huge-page boundary become no-ops.** A partial
//!    purge inside a promoted huge page forces the kernel to demote it
//!    back to 4 KiB (or 16 KiB on Apple) — the very translation-cost
//!    we're trying to avoid. `HugePageAligned::release_pages` silently
//!    rounds down to whole huge-page units; if the resulting range is
//!    empty, the call is dropped.
//!
//! # Promotion is not automatic
//!
//! This wrapper **aligns**; it does not itself request promotion. It
//! issues **no** `madvise(MADV_HUGEPAGE)` / `MEM_LARGE_PAGES` syscall.
//! Whether the OS actually backs an aligned region with huge pages
//! depends on factors outside this layer:
//!
//! - the allocation must also *span* at least one full huge page — a
//!   1 KiB allocation at a 2 MiB boundary is aligned but never promoted,
//!   it just rounds the inner cursor up (internal fragmentation);
//! - on Linux, ambient THP policy must allow it (`THP=always`, or an
//!   explicit `madvise` the caller issues separately — e.g. via
//!   `MmapBacked` map-time flags). Under the common `THP=madvise`
//!   default an aligned-but-un-`madvise`d region is *not* promoted.
//!
//! So this type is best thought of as **huge-page *alignment*** (and
//! purge-granularity preservation), the necessary substrate for huge
//! pages — pair it with a backing/policy that actually requests them.
//! For an explicit large-page mapping see `HugePageBacked`.
//!
//! `protect` is forwarded unchanged — it doesn't break THP promotion
//! the way purges do, and a caller setting `PROT_NONE` (e.g. for a
//! guard page) explicitly wants the page split.
//!
//! **`base()` note:** unlike `HugePageBacked`, this wrapper forwards
//! `base()`/`size()` straight from the inner backing, so `base()` is
//! only inner-page-aligned (typically 4 KiB), **not** huge-page-aligned.
//! The huge-page alignment guarantee applies to pointers returned by
//! [`allocate`](Allocator::allocate), not to the region base.
//!
//! See `docs/ARCHITECTURE.md` for design context.

use core::ptr::NonNull;

use forge_alloc_core::{
    AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout, OsBacked, ProtectFlags,
};

/// HugePageAligned wrapper.
///
/// `huge_page_size` defaults to a platform-appropriate value
/// ([`default_huge_page_size`]); pass an explicit size to override
/// (useful for hugetlbfs setups with 1 GiB pages, or testing).
pub struct HugePageAligned<I: OsBacked> {
    inner: I,
    huge_page_size: usize,
}

impl<I: OsBacked> HugePageAligned<I> {
    /// Wrap with the platform-default huge page size.
    ///
    /// On Linux/Windows with x86_64 or aarch64 (non-Apple) this is
    /// 2 MiB. On macOS aarch64 (Apple Silicon, 16 KiB native granule)
    /// it is 32 MiB. Returns `None` if the platform-default would not
    /// satisfy [`Self::with_huge_page_size`]'s invariants (impossible on
    /// supported targets).
    pub fn new(inner: I) -> Option<Self> {
        Self::with_huge_page_size(inner, default_huge_page_size())
    }

    /// Wrap with an explicit huge page size. Must be a power of two and
    /// at least 4 KiB (any smaller would not be a "huge" page on any
    /// real architecture).
    pub fn with_huge_page_size(inner: I, huge_page_size: usize) -> Option<Self> {
        if !huge_page_size.is_power_of_two() || huge_page_size < 4096 {
            return None;
        }
        Some(Self {
            inner,
            huge_page_size,
        })
    }

    /// Configured huge page size in bytes.
    #[inline]
    pub fn huge_page_size(&self) -> usize {
        self.huge_page_size
    }

    /// Minimum purge granularity. Equal to [`huge_page_size`](Self::huge_page_size).
    /// Calls to [`release_pages`](OsBacked::release_pages) below this
    /// size are dropped.
    #[inline]
    pub fn min_purge_size(&self) -> usize {
        self.huge_page_size
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }

    /// Inflate a caller's layout to satisfy huge-page alignment.
    /// Returns `Err(AllocError)` if the inflated alignment is invalid
    /// (cannot happen for power-of-two huge_page_size, defensive).
    fn promote_layout(&self, layout: NonZeroLayout) -> Result<NonZeroLayout, AllocError> {
        let align = core::cmp::max(layout.align().get(), self.huge_page_size);
        NonZeroLayout::from_size_align(layout.size().get(), align).map_err(|_| AllocError)
    }
}

unsafe impl<I: OsBacked> Deallocator for HugePageAligned<I> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // `promote_layout` is a pure function of `layout`; the matching
        // `allocate(layout)` already evaluated it successfully, so it cannot
        // newly fail here. Forwarding the *un-inflated* `layout` on a
        // hypothetical error would violate the Deallocator contract (the
        // layout must match the one `allocate` passed to `inner`) for any
        // inner whose deallocate is layout-sensitive — so recompute the
        // inflated layout and treat failure as the unreachable invariant.
        let inflated = self
            .promote_layout(layout)
            .expect("HugePageAligned::deallocate: promote_layout failed for a layout that succeeded at allocate-time");
        // SAFETY: forwarded; ptr came from this wrapper's allocate which
        // used `inflated` to call inner.allocate.
        unsafe { self.inner.deallocate(ptr, inflated) }
    }
}

unsafe impl<I: OsBacked> Allocator for HugePageAligned<I> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let inflated = self.promote_layout(layout)?;
        let block = self.inner.allocate(inflated)?;
        // Debug-only contract check: inner allocators that claim to honour
        // alignment must return a pointer aligned to `huge_page_size`.
        // Cheaper than a runtime branch in release; if violated, callers
        // would see split-page translations and silent perf loss.
        debug_assert_eq!(
            block.cast::<u8>().as_ptr() as usize & (self.huge_page_size - 1),
            0,
            "inner allocator returned non-huge-page-aligned pointer despite inflated layout",
        );
        Ok(block)
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        // Inflated allocations consume the inner faster, but the
        // wrapper has no per-request bookkeeping to track precisely.
        // Best-effort: report the inner's capacity unchanged.
        self.inner.capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        self.inner.corruption_events()
    }

    #[inline]
    unsafe fn usable_size(&self, ptr: NonNull<u8>, layout: NonZeroLayout) -> Option<usize> {
        // HugePageAligned inflates ALIGNMENT only, not size: `allocate` returns
        // the inner pointer verbatim (no header/footer), so the user region is
        // exactly the inner region. This wrapper is therefore size-TRANSPARENT
        // and must FORWARD usable_size — NOT withhold it. Withholding (the
        // trait default `None`) would make an outer scrub wrapper
        // (`PoisonOnFree`/`ZeroizeOnFree`) scrub only `layout.size()` and leak
        // any rounding slack the inner reports at `ptr`. Forward with the SAME
        // inflated layout `allocate`/`deallocate` use, so the inner sees a
        // consistent layout. (Distinct from layout-INFLATING wrappers like
        // `Canary`/`GuardPage`, which hand back a sub-slice and correctly
        // withhold usable_size.)
        let inflated = self.promote_layout(layout).ok()?;
        // SAFETY: `ptr` came from this wrapper's `allocate`, which called
        // `inner.allocate(inflated)` and returned the inner pointer unchanged.
        unsafe { self.inner.usable_size(ptr, inflated) }
    }
}

unsafe impl<I: OsBacked> OsBacked for HugePageAligned<I> {
    #[inline]
    fn base_ptr(&self) -> NonNull<u8> {
        self.inner.base_ptr()
    }

    #[inline]
    fn region_size(&self) -> usize {
        self.inner.region_size()
    }

    /// Round the requested range *inward* to whole huge pages. If
    /// nothing remains after rounding, the call is dropped — this is
    /// the "refuse to fragment a huge page" guarantee. Partial purges
    /// that would force the kernel to demote a promoted huge page back
    /// to 4 KiB are silently elided.
    #[inline]
    unsafe fn release_pages(&self, ptr: NonNull<u8>, size: usize) {
        let hp = self.huge_page_size;
        let mask = hp - 1; // hp is power-of-two by construction
        let raw = ptr.as_ptr() as usize;
        // Round start up to next huge-page boundary; round end down.
        let Some(end) = raw.checked_add(size) else {
            // size would overflow — caller-contract violation; bail.
            return;
        };
        // `raw + mask` can overflow on near-`usize::MAX` addresses; if it
        // does the rounded-up start cannot lie below `end` in usable
        // address space, so the call is necessarily a no-op.
        let Some(rounded) = raw.checked_add(mask) else {
            return;
        };
        let aligned_start = rounded & !mask;
        let aligned_end = end & !mask;
        if aligned_end <= aligned_start {
            // Nothing to purge after rounding — preserve the huge page.
            return;
        }
        let aligned_size = aligned_end - aligned_start;
        // SAFETY: aligned_start lies in [raw, raw+size); aligned_size
        // <= size. The caller's range was wholly inside the inner
        // region per the OsBacked contract; ours is a subrange.
        let aligned_ptr = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(aligned_start - raw)) };
        unsafe { self.inner.release_pages(aligned_ptr, aligned_size) };
    }

    /// Forwarded unchanged. Protection changes don't fragment a huge
    /// page the way a purge does, and a caller using `PROT_NONE` for a
    /// guard page explicitly wants the page split.
    #[inline]
    unsafe fn protect(&self, ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
        // SAFETY: forwarded; caller upholds protect contract on self.
        unsafe { self.inner.protect(ptr, size, flags) }
    }
}

impl<I: OsBacked + FixedRange> FixedRange for HugePageAligned<I> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.inner.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Pass-through forward so a `commit`-aware consumer reaches the inner
    /// backing when this wrapper sits over a `lazy_commit` `MmapBacked`.
    #[inline]
    fn commit(&self, offset: usize, len: usize) -> Result<(), AllocError> {
        self.inner.commit(offset, len)
    }
}

/// Platform-default huge page size in bytes.
///
/// - x86_64 / aarch64 (non-Apple) Linux & Windows: 2 MiB.
/// - aarch64 macOS (Apple Silicon, 16 KiB native granule): 32 MiB —
///   the kernel only promotes contiguous 32 MiB-aligned regions to its
///   "superpage" tier.
/// - Other targets: 2 MiB as a reasonable default.
#[inline]
pub fn default_huge_page_size() -> usize {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        32 * 1024 * 1024
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        2 * 1024 * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::MmapBacked;
    use core::cell::Cell;

    /// A mock `OsBacked` that records the layout passed to `usable_size` and
    /// returns a sentinel — lets us prove `HugePageAligned` FORWARDS
    /// usable_size with the INFLATED layout rather than withholding it.
    /// Touches no real memory, so it runs under Miri.
    struct RecordingBacked {
        last_size: Cell<usize>,
        last_align: Cell<usize>,
    }
    impl RecordingBacked {
        fn new() -> Self {
            Self {
                last_size: Cell::new(0),
                last_align: Cell::new(0),
            }
        }
    }
    unsafe impl Deallocator for RecordingBacked {
        unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {}
    }
    unsafe impl Allocator for RecordingBacked {
        fn allocate(&self, _layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
            Err(AllocError)
        }
        unsafe fn usable_size(&self, _ptr: NonNull<u8>, layout: NonZeroLayout) -> Option<usize> {
            self.last_size.set(layout.size().get());
            self.last_align.set(layout.align().get());
            Some(0xC0DE)
        }
    }
    unsafe impl OsBacked for RecordingBacked {
        fn base_ptr(&self) -> NonNull<u8> {
            NonNull::dangling()
        }
        fn region_size(&self) -> usize {
            0
        }
        unsafe fn release_pages(&self, _ptr: NonNull<u8>, _size: usize) {}
        unsafe fn protect(&self, _ptr: NonNull<u8>, _size: usize, _flags: ProtectFlags) {}
    }

    #[test]
    fn usable_size_forwards_with_inflated_layout() {
        // HugePageAligned inflates ALIGNMENT only (size-transparent), so it must
        // forward usable_size — withholding would make an outer scrub wrapper
        // under-scrub the inner's slack. Pin the forward AND that the inner sees
        // the inflated alignment (matching allocate/deallocate), not None.
        let mock = RecordingBacked::new();
        let hp = HugePageAligned::with_huge_page_size(mock, 2 * 1024 * 1024).expect("valid params");
        let layout = NonZeroLayout::from_size_align(1024, 8).unwrap();
        // ptr is never dereferenced by the mock.
        let got = unsafe { hp.usable_size(NonNull::<u8>::dangling(), layout) };
        assert_eq!(
            got,
            Some(0xC0DE),
            "usable_size must forward to the inner, not return None"
        );
        assert_eq!(
            hp.inner().last_align.get(),
            2 * 1024 * 1024,
            "inner must see the INFLATED alignment, matching allocate/deallocate",
        );
        assert_eq!(
            hp.inner().last_size.get(),
            1024,
            "size is unchanged (alignment-only inflation)",
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn rejects_non_power_of_two() {
        let inner = MmapBacked::new(4 * 1024 * 1024).unwrap();
        assert!(HugePageAligned::with_huge_page_size(inner, 3 * 1024 * 1024).is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn rejects_too_small() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        assert!(HugePageAligned::with_huge_page_size(inner, 1024).is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn default_constructs() {
        // Use 2 MiB explicitly so this works on all hosts including
        // single-node WSL where the system may not have THP available.
        let inner = MmapBacked::new(4 * 1024 * 1024).unwrap();
        let hp =
            HugePageAligned::with_huge_page_size(inner, 2 * 1024 * 1024).expect("valid params");
        assert_eq!(hp.huge_page_size(), 2 * 1024 * 1024);
        assert_eq!(hp.min_purge_size(), 2 * 1024 * 1024);
    }

    #[test]
    fn default_huge_page_size_matches_platform() {
        // Pins the platform branch (incl. the Apple-Silicon 32 MiB arm, which
        // is exercised on aarch64-macOS CI). Runs on every host.
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        assert_eq!(default_huge_page_size(), 32 * 1024 * 1024);
        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        assert_eq!(default_huge_page_size(), 2 * 1024 * 1024);
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn allocate_returns_huge_page_aligned() {
        // 6 MiB backing GUARANTEES a 2 MiB-aligned allocation regardless of
        // the mmap base: the worst-case leading slack to reach the first
        // boundary is < 2 MiB, leaving > 4 MiB — far more than the 1 KiB
        // request. So the assertion is unconditional (previously it was
        // skipped entirely when allocate happened to return Err).
        let inner = MmapBacked::new(6 * 1024 * 1024).unwrap();
        let hp =
            HugePageAligned::with_huge_page_size(inner, 2 * 1024 * 1024).expect("valid params");
        let layout = NonZeroLayout::from_size_align(1024, 8).unwrap();
        let block = hp
            .allocate(layout)
            .expect("6 MiB backing must fit a 2 MiB-aligned 1 KiB allocation");
        let addr = block.cast::<u8>().as_ptr() as usize;
        assert_eq!(
            addr % (2 * 1024 * 1024),
            0,
            "allocation not huge-page-aligned"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn release_pages_below_huge_page_is_no_op() {
        // Small release: the wrapper rounds inward to zero and drops
        // the call. Verify by capturing the inner's last_os_error
        // before/after — if the wrapper had forwarded, madvise with a
        // 4 KiB size would have run.
        use crate::backing::{mmap_clear_last_os_error, mmap_last_os_error};
        let inner = MmapBacked::new(4 * 1024 * 1024).unwrap();
        let base = inner.base_ptr();
        let hp =
            HugePageAligned::with_huge_page_size(inner, 2 * 1024 * 1024).expect("valid params");
        mmap_clear_last_os_error();
        // Try to release 4 KiB from the very start — way below the
        // 2 MiB huge-page threshold.
        unsafe { hp.release_pages(base, 4096) };
        // We didn't trigger any failing syscall and we didn't even fire
        // a real madvise (because we dropped the call), so the slot
        // remains None.
        assert!(mmap_last_os_error().is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn release_pages_aligned_huge_page_passes_through() {
        // A 2 MiB release at a 2 MiB boundary inside the region must reach the
        // inner allocator. A 6 MiB backing GUARANTEES one full aligned huge
        // page fits regardless of mmap base (worst-case leading slack < 2 MiB,
        // leaving > 4 MiB), so this runs unconditionally — previously it
        // silently no-op'd when the huge page didn't fit.
        let hp_size = 2 * 1024 * 1024;
        let inner = MmapBacked::new(6 * 1024 * 1024).unwrap();
        let raw_base = inner.base_ptr().as_ptr() as usize;
        let aligned = (raw_base + hp_size - 1) & !(hp_size - 1);
        assert!(
            aligned + hp_size <= raw_base + 6 * 1024 * 1024,
            "6 MiB backing must contain a full aligned huge page",
        );
        let offset = aligned - raw_base;
        let aligned_ptr = unsafe { NonNull::new_unchecked(inner.base_ptr().as_ptr().add(offset)) };
        let hp = HugePageAligned::with_huge_page_size(inner, hp_size).expect("valid params");
        unsafe {
            core::ptr::write_bytes(aligned_ptr.as_ptr(), 0xCC, hp_size);
            hp.release_pages(aligned_ptr, hp_size);
            // The region is still mapped after the (forwarded) purge: a
            // post-release write must not fault. Proves release_pages did not
            // unmap, only advised the kernel.
            core::ptr::write_bytes(aligned_ptr.as_ptr(), 0x11, 4096);
            assert_eq!(*aligned_ptr.as_ptr(), 0x11);
        }
    }
}
