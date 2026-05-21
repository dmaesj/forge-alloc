//! `FixedRange` — allocators whose address range is determined at construction
//! and never changes.
//!
//! Required by `WithFallback<P: FixedRange, S>` so that deallocation can be
//! routed correctly: `primary.contains(ptr)` returns whether the pointer came
//! from the primary or must go to the secondary. Growing allocators
//! (`ExtendableSlab`) deliberately do *not* implement `FixedRange`.

use core::ptr::NonNull;

use super::allocator::Allocator;

/// Address range fixed at construction.
///
/// Once constructed, [`base`](Self::base) and [`size`](Self::size) never
/// change. The default [`contains`](Self::contains) check is sufficient for
/// pointer-provenance routing in `WithFallback`.
pub trait FixedRange: Allocator {
    /// First byte of the allocator's address range.
    fn base(&self) -> NonNull<u8>;

    /// Length in bytes of the allocator's address range.
    fn size(&self) -> usize;

    /// Whether `ptr` lies within `[base, base + size)`.
    ///
    /// Implemented as `(p - base) < size` using wrapping subtraction so the
    /// check remains correct when the region's end address wraps past
    /// `usize::MAX` (rare on 64-bit, but possible on 16-/32-bit no_std
    /// targets). A naive `p >= base && p < base + size` would compute an
    /// `end < base` after wrap and report every pointer as out-of-range.
    #[inline]
    fn contains(&self, ptr: NonNull<u8>) -> bool {
        let base = self.base().as_ptr() as usize;
        let p = ptr.as_ptr() as usize;
        p.wrapping_sub(base) < self.size()
    }
}
