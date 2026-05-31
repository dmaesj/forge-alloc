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

use super::non_zero_layout::AllocError;

/// Address range fixed at construction.
///
/// Once constructed, [`base`](Self::base) and [`size`](Self::size) never
/// change. The default [`contains`](Self::contains) check is sufficient for
/// pointer-provenance routing in `WithFallback`.
pub trait FixedRange {
    /// First byte of the owned address range.
    ///
    /// **Concurrency contract:** `base` (and [`size`](Self::size)) must be
    /// callable concurrently from multiple threads through a shared `&self`
    /// without data races — i.e. they must not mutate through `&self`. A
    /// thread-safe consumer such as `SharedBumpArena` relies on this to be
    /// `Sync` while its backing is merely `Send` (not `Sync`): it only ever
    /// calls `base()`/`size()` on the shared backing, never an interior-
    /// mutating method. All in-tree backings satisfy this (these are pure
    /// reads of an immutable field).
    fn base(&self) -> NonNull<u8>;

    /// Length in bytes of the owned address range.
    ///
    /// Subject to the same concurrency contract as [`base`](Self::base): must
    /// be data-race-free when called through a shared `&self`.
    fn size(&self) -> usize;

    /// Ensure the bytes `[offset, offset + len)` (relative to
    /// [`base`](Self::base)) are backed by committed, writable memory
    /// before a consumer writes through them.
    ///
    /// The default is a no-op returning `Ok(())`: most backings hand back
    /// memory that is already writable (`InlineBacked`'s inline array, a
    /// `mmap`'d region under Unix demand-paging, an eagerly-committed
    /// `VirtualAlloc`). The hook exists for backings that *reserve* address
    /// space without committing it — currently only a `lazy_commit`
    /// [`MmapBacked`](../../forge_alloc/backing/struct.MmapBacked.html) on
    /// Windows — where a write to an uncommitted page would fault. A
    /// cursor-advancing consumer (e.g. `BumpArena`) calls this as its
    /// cursor crosses into new pages, and propagates the `Err` as an
    /// allocation failure rather than letting the OS decline a reservation
    /// turn into a hard access violation.
    ///
    /// # Contract
    ///
    /// - `[offset, offset + len)` must lie within `[0, size())`.
    /// - On `Err(AllocError)` the caller must treat the region as *not*
    ///   committed and must not write through `[offset, offset + len)`.
    /// - Idempotent and monotonic: committing a range that is already
    ///   committed succeeds without side effects.
    #[inline]
    fn commit(&self, offset: usize, len: usize) -> Result<(), AllocError> {
        let _ = (offset, len);
        Ok(())
    }

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
