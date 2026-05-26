//! `FixedRange` — types whose owned address range is determined at
//! construction and never changes.
//!
//! Required by `WithFallback<P: Allocator + FixedRange, S>` so deallocation
//! can be routed correctly: `primary.contains(ptr)` reports whether the
//! pointer came from the primary or must go to the secondary. Growing
//! allocators (`ExtendableSlab`) deliberately do *not* implement
//! `FixedRange`.
//!
//! # Decoupled from `Allocator`
//!
//! `FixedRange` does NOT have `Allocator` as a supertrait. The two concerns
//! are independent: a type can own a contiguous block of bytes without
//! itself being able to carve allocations out of that block. Composition
//! handles the gap: pair a `FixedRange`-only region owner (such as
//! `forge_alloc::HeapBytes`) with `forge_alloc::BumpArena` (which carries
//! the bump cursor) to get an `Allocator` over the region.
//!
//! Most existing primitives (`InlineBacked`, `MmapBacked`, `BumpArena`,
//! `Slab`) implement BOTH `FixedRange` and `Allocator` — they double-duty
//! as region owners and allocators. Downstream code that needs both
//! capabilities should write `B: Allocator + FixedRange` explicitly rather
//! than relying on a supertrait link.

use core::ptr::NonNull;

/// Address range fixed at construction.
///
/// Once constructed, [`base`](Self::base) and [`size`](Self::size) never
/// change. The default [`contains`](Self::contains) check is sufficient for
/// pointer-provenance routing in `WithFallback`.
pub trait FixedRange {
    /// First byte of the owned address range.
    fn base(&self) -> NonNull<u8>;

    /// Length in bytes of the owned address range.
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
