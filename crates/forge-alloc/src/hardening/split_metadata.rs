//! `SplitMetadata<I>` — hot/cold metadata isolation: two distinct mmap
//! regions, one for allocator bookkeeping and one for user data.
//!
//! Wraps an [`OsBacked`] allocator (the data region) and pairs it with
//! a separate [`MmapBacked`] region for metadata. Forwards
//! [`Allocator`] / [`OsBacked`] / [`FixedRange`] to the data region;
//! exposes the metadata region via [`meta_base`](SplitMetadata::meta_base)
//! and [`meta_size`](SplitMetadata::meta_size) for callers (typically a
//! hardened slab) that want to keep their free-list / block-state /
//! canary storage out of the data region's cache lines and out of
//! reach of linear overflows past user allocations.
//!
//! See `docs/ARCHITECTURE.md` for design context.

use core::ptr::NonNull;

use crate::backing::MmapBacked;
use forge_alloc_core::{
    AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout, OsBacked, ProtectFlags,
};

/// SplitMetadata wrapper.
///
/// **Note**: this primitive guards only the *data* region from
/// metadata pollution and adjacency overflows.
///
/// Adding guard pages on top requires the **inner data region** to be
/// `OsBacked` (so `GuardPage<SplitMetadata<MmapBacked>>` works, but
/// `GuardPage<SplitMetadata<Slab<...>>>` does not — `Slab` is not
/// `OsBacked` and `GuardPage` rejects it at the type level). For the
/// `HardenedSlab` recipe — the recommended security composition — the
/// guard pages wrap the OS-mapped data side and the `Slab` lives on
/// top, giving the form `Slab<T, GuardPage<SplitMetadata<MmapBacked>>>`.
#[must_use = "SplitMetadata guards the data region only. \
              For full coverage, compose with GuardPage<_> on an \
              OsBacked inner: `GuardPage<SplitMetadata<MmapBacked>>` \
              (then place a Slab on top via `Slab<T, GuardPage<SplitMetadata<MmapBacked>>>`)"]
pub struct SplitMetadata<I: Allocator> {
    /// Metadata region — held by value so its `munmap` runs when this
    /// wrapper drops.
    meta_region: MmapBacked,
    /// User-visible data region. All `Allocator` / `OsBacked` /
    /// `FixedRange` calls forward here.
    data_region: I,
}

impl<I: Allocator> SplitMetadata<I> {
    /// Wrap `data_region` and allocate a fresh metadata mmap of
    /// `meta_size` bytes.
    ///
    /// Returns `Err(AllocError)` if the metadata mmap fails.
    pub fn new(data_region: I, meta_size: usize) -> Result<Self, AllocError> {
        let meta_region = MmapBacked::new(meta_size)?;
        Ok(Self {
            meta_region,
            data_region,
        })
    }

    /// Wrap with a pre-built metadata region. Useful when the caller
    /// wants to construct the meta `MmapBacked` with specific flags
    /// (huge pages, populate, etc.).
    pub fn with_meta(data_region: I, meta_region: MmapBacked) -> Self {
        Self {
            meta_region,
            data_region,
        }
    }

    /// First byte of the metadata region.
    #[inline]
    pub fn meta_base(&self) -> NonNull<u8> {
        self.meta_region.base_ptr()
    }

    /// Length in bytes of the metadata region.
    #[inline]
    pub fn meta_size(&self) -> usize {
        self.meta_region.region_size()
    }

    /// Borrow the metadata mmap directly. Mainly for callers that
    /// want to apply `OsBacked` ops (release_pages, protect) to the
    /// meta region independently.
    #[inline]
    pub fn meta(&self) -> &MmapBacked {
        &self.meta_region
    }

    /// Borrow the data allocator.
    #[inline]
    pub fn data(&self) -> &I {
        &self.data_region
    }
}

unsafe impl<I: Allocator> Deallocator for SplitMetadata<I> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // SAFETY: forwarded; caller's contract preserved against data.
        unsafe { self.data_region.deallocate(ptr, layout) }
    }
}

unsafe impl<I: Allocator> Allocator for SplitMetadata<I> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        self.data_region.allocate(layout)
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        self.data_region.capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        self.data_region.corruption_events()
    }
}

// OsBacked is implemented only when the data region is itself OsBacked —
// i.e. SplitMetadata wraps an MmapBacked directly. When the data region
// is a higher-layer type (e.g. Slab), `release_pages` / `protect` make
// no sense at this layer and are not forwarded.
unsafe impl<I: Allocator + OsBacked> OsBacked for SplitMetadata<I> {
    #[inline]
    fn base_ptr(&self) -> NonNull<u8> {
        self.data_region.base_ptr()
    }

    #[inline]
    fn region_size(&self) -> usize {
        self.data_region.region_size()
    }

    #[inline]
    unsafe fn release_pages(&self, ptr: NonNull<u8>, size: usize) {
        // SAFETY: forwarded; affects data region only.
        unsafe { self.data_region.release_pages(ptr, size) }
    }

    #[inline]
    unsafe fn protect(&self, ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
        // SAFETY: forwarded.
        unsafe { self.data_region.protect(ptr, size, flags) }
    }
}

impl<I: Allocator + FixedRange> FixedRange for SplitMetadata<I> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.data_region.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.data_region.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MmapBacked alone is the simplest OsBacked that satisfies our
    /// bound. Most real usage wraps a slab or higher-layer type, but
    /// the wrapper semantics are testable with a bare data region.
    fn build(data_size: usize, meta_size: usize) -> SplitMetadata<MmapBacked> {
        let data = MmapBacked::new(data_size).unwrap();
        SplitMetadata::new(data, meta_size).unwrap()
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn meta_and_data_regions_are_disjoint_virtual_addresses() {
        let sm = build(64 * 1024, 64 * 1024);
        let meta = sm.meta_base().as_ptr() as usize;
        let meta_end = meta + sm.meta_size();
        let data = sm.base_ptr().as_ptr() as usize;
        let data_end = data + sm.region_size();
        // The two regions must NOT overlap — that's the entire point.
        assert!(
            meta_end <= data || data_end <= meta,
            "regions overlap: meta=[{meta:x}, {meta_end:x}) data=[{data:x}, {data_end:x})",
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn forwards_allocate_to_data_region() {
        let sm = build(64 * 1024, 16 * 1024);
        let layout = NonZeroLayout::from_size_align(128, 8).unwrap();
        let block = sm.allocate(layout).unwrap();
        let p = block.cast::<u8>().as_ptr() as usize;
        let data = sm.base_ptr().as_ptr() as usize;
        let data_end = data + sm.region_size();
        assert!(
            p >= data && p < data_end,
            "allocation must come from data region",
        );
        unsafe { sm.deallocate(block.cast(), layout) };
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn allocations_never_touch_metadata_region() {
        let sm = build(64 * 1024, 16 * 1024);
        let meta_base = sm.meta_base().as_ptr() as usize;
        let meta_end = meta_base + sm.meta_size();
        let layout = NonZeroLayout::from_size_align(256, 8).unwrap();
        for _ in 0..32 {
            let block = sm.allocate(layout).unwrap();
            let start = block.cast::<u8>().as_ptr() as usize;
            let end = start + 256;
            assert!(
                end <= meta_base || start >= meta_end,
                "allocation crosses metadata region",
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn meta_region_is_writable() {
        // The meta region exists so callers can use it for arbitrary
        // bookkeeping. Verify by writing and reading back.
        let sm = build(64 * 1024, 4 * 1024);
        let base = sm.meta_base().as_ptr();
        unsafe {
            core::ptr::write_bytes(base, 0xAB, 4 * 1024);
            for i in [0, 100, 1000, 4095] {
                assert_eq!(*base.add(i), 0xAB);
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn fixed_range_reports_data_region() {
        let sm = build(64 * 1024, 16 * 1024);
        assert_eq!(sm.base().as_ptr(), sm.data().base_ptr().as_ptr());
        assert_eq!(sm.size(), sm.data().region_size());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn protect_only_affects_data_region() {
        // PROT_NONE on the data region must not affect meta; verify
        // by re-writing meta after.
        let sm = build(64 * 1024, 4 * 1024);
        unsafe {
            // First: write to meta to confirm it's accessible.
            core::ptr::write_bytes(sm.meta_base().as_ptr(), 0x11, 4 * 1024);
            // Now PROT_NONE the data region.
            sm.protect(sm.base_ptr(), sm.region_size(), ProtectFlags::NONE);
            // Meta should STILL be writable.
            core::ptr::write_bytes(sm.meta_base().as_ptr(), 0x22, 4 * 1024);
            // Restore data so Drop can unmap it (some kernels require
            // PROT_READ|PROT_WRITE on unmap; defensive).
            sm.protect(sm.base_ptr(), sm.region_size(), ProtectFlags::RW);
        }
    }
}
