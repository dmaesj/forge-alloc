//! `NonZeroLayout`, the library's internal layout contract, plus the
//! `StdCompat<A>` shim that bridges to `allocator_api2::Allocator`.
//!
//! ZSTs are absorbed at the `StdCompat` boundary so that no primitive ever
//! sees `size == 0`. This eliminates a recurring class of subtle bugs around
//! dangling-pointer handling.

use core::alloc::Layout;
use core::fmt;
use core::num::NonZeroUsize;
use core::ptr::NonNull;

pub use allocator_api2::alloc::AllocError;

/// A `Layout` with `size > 0` and a power-of-two alignment.
///
/// All primitives in this crate take `NonZeroLayout` rather than `Layout`. ZST
/// handling happens once, at the [`StdCompat`] boundary, instead of being
/// reinvented in every implementor.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct NonZeroLayout {
    size: NonZeroUsize,
    align: NonZeroUsize,
}

impl NonZeroLayout {
    /// Construct from a non-zero size and non-zero alignment.
    ///
    /// Returns [`LayoutError`] if `align` is not a power of two or if rounding
    /// `size` up to `align` would overflow `isize::MAX`.
    #[inline]
    pub const fn new(size: NonZeroUsize, align: NonZeroUsize) -> Result<Self, LayoutError> {
        if !align.get().is_power_of_two() {
            return Err(LayoutError);
        }
        // Match std::alloc::Layout's overflow rule: size, when rounded up to
        // align, must not exceed isize::MAX.
        let max = (isize::MAX as usize) - (align.get() - 1);
        if size.get() > max {
            return Err(LayoutError);
        }
        Ok(Self { size, align })
    }

    /// Construct from raw `usize` size and alignment. Convenience wrapper.
    #[inline]
    pub const fn from_size_align(size: usize, align: usize) -> Result<Self, LayoutError> {
        let Some(size) = NonZeroUsize::new(size) else {
            return Err(LayoutError);
        };
        let Some(align) = NonZeroUsize::new(align) else {
            return Err(LayoutError);
        };
        Self::new(size, align)
    }

    /// Layout for storing one `T`. Returns `None` if `T` is a ZST.
    #[inline]
    pub const fn for_type<T>() -> Option<Self> {
        let size = core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        let Some(size) = NonZeroUsize::new(size) else {
            return None;
        };
        // align_of always returns a power of two, never zero — unwrap is safe.
        let align = match NonZeroUsize::new(align) {
            Some(a) => a,
            None => return None,
        };
        match Self::new(size, align) {
            Ok(layout) => Some(layout),
            Err(_) => None,
        }
    }

    /// Layout for an array of `n` `T`s. Returns `None` if `T` is a ZST,
    /// `n` is zero, or the resulting size overflows.
    ///
    /// Mirrors `core::alloc::Layout::array::<T>(n)`'s padding rules: each
    /// element is padded out to `align_of::<T>()` (which `size_of::<T>()`
    /// already encodes via the trailing-pad rule, so for sized types the
    /// product is identical). The explicit `pad_to_align` step keeps the
    /// math correct if a future `repr` ever lets `size_of` skip the
    /// trailing pad — `#[repr(C, packed)]` types stay correct because
    /// their `align` is 1 and `pad_to_align` is a no-op.
    #[inline]
    pub const fn array<T>(n: usize) -> Option<Self> {
        let Some(elem) = Self::for_type::<T>() else {
            return None;
        };
        let padded = elem.pad_to_align();
        let Some(total) = padded.size.get().checked_mul(n) else {
            return None;
        };
        let Some(total) = NonZeroUsize::new(total) else {
            return None;
        };
        match Self::new(total, padded.align) {
            Ok(l) => Some(l),
            Err(_) => None,
        }
    }

    /// Size in bytes. Guaranteed nonzero.
    #[inline]
    pub const fn size(&self) -> NonZeroUsize {
        self.size
    }

    /// Alignment in bytes. Guaranteed nonzero and a power of two.
    #[inline]
    pub const fn align(&self) -> NonZeroUsize {
        self.align
    }

    /// Round `size` up so the layout's total footprint is a multiple of
    /// `align`. Useful for packing arrays of this layout end-to-end.
    #[inline]
    pub const fn pad_to_align(&self) -> Self {
        let align = self.align.get();
        let padded = (self.size.get() + align - 1) & !(align - 1);
        // padded >= size > 0, so unwrap is sound.
        let size = match NonZeroUsize::new(padded) {
            Some(s) => s,
            None => self.size,
        };
        Self {
            size,
            align: self.align,
        }
    }

    /// Convert to a standard `core::alloc::Layout`. Always succeeds.
    #[inline]
    pub const fn to_layout(&self) -> Layout {
        // SAFETY: NonZeroLayout's invariants are a strict superset of Layout's.
        unsafe { Layout::from_size_align_unchecked(self.size.get(), self.align.get()) }
    }
}

impl fmt::Debug for NonZeroLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NonZeroLayout")
            .field("size", &self.size.get())
            .field("align", &self.align.get())
            .finish()
    }
}

impl TryFrom<Layout> for NonZeroLayout {
    type Error = LayoutError;
    #[inline]
    fn try_from(value: Layout) -> Result<Self, Self::Error> {
        Self::from_size_align(value.size(), value.align())
    }
}

impl From<NonZeroLayout> for Layout {
    #[inline]
    fn from(value: NonZeroLayout) -> Self {
        value.to_layout()
    }
}

/// Errors returned when constructing a [`NonZeroLayout`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LayoutError;

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid NonZeroLayout (zero size, zero/non-power-of-two align, or overflow)")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LayoutError {}

/// Adapter that exposes any [`crate::Allocator`] as an
/// [`allocator_api2::alloc::Allocator`].
///
/// Handles zero-sized allocations at the boundary — the inner allocator never
/// receives a `size == 0` request.
#[derive(Copy, Clone, Debug, Default)]
pub struct StdCompat<A> {
    inner: A,
}

impl<A> StdCompat<A> {
    /// Wrap an inner allocator.
    #[inline]
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &A {
        &self.inner
    }

    /// Consume the adapter and return the inner allocator.
    #[inline]
    pub fn into_inner(self) -> A {
        self.inner
    }
}

/// Build the dangling pointer used for zero-sized allocations. Per the
/// `allocator_api2::Allocator` contract, even ZST returns must be aligned to
/// the layout's alignment; `NonNull::<u8>::dangling()` is aligned to 1 and is
/// not sufficient when the requested layout has `align > 1` (e.g.
/// `Layout::new::<[u32; 0]>()` with alignment 4).
#[inline]
fn dangling_for_align(align: usize) -> NonNull<u8> {
    // Invariants from a valid `Layout`: align is a power of two and non-zero,
    // so it is a valid pointer value.
    debug_assert!(align.is_power_of_two() && align != 0);
    // SAFETY: `align` is non-zero, so the resulting pointer is non-null.
    unsafe { NonNull::new_unchecked(align as *mut u8) }
}

unsafe impl<A: crate::Allocator> allocator_api2::alloc::Allocator for StdCompat<A> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match NonZeroLayout::try_from(layout) {
            Ok(nzl) => self.inner.allocate(nzl),
            Err(_) => Ok(NonNull::slice_from_raw_parts(
                dangling_for_align(layout.align()),
                0,
            )),
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if let Ok(nzl) = NonZeroLayout::try_from(layout) {
            // SAFETY: forwarded from the caller of `<StdCompat<_> as
            // allocator_api2::Allocator>::deallocate`, who already promises
            // that `ptr` was issued by `self` for `layout`. ZST layouts are
            // filtered out above.
            unsafe { self.inner.deallocate(ptr, nzl) }
        }
        // ZST: no-op. `ptr` is dangling per allocator-api2 contract.
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        match (NonZeroLayout::try_from(old), NonZeroLayout::try_from(new)) {
            (Ok(old_nz), Ok(new_nz)) => {
                // SAFETY: caller upholds allocator-api2's grow contract.
                unsafe { self.inner.grow(ptr, old_nz, new_nz) }
            }
            (Err(_), Ok(new_nz)) => self.inner.allocate(new_nz),
            (Ok(_), Err(_)) => Err(AllocError),
            (Err(_), Err(_)) => Ok(NonNull::slice_from_raw_parts(
                dangling_for_align(new.align()),
                0,
            )),
        }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        match (NonZeroLayout::try_from(old), NonZeroLayout::try_from(new)) {
            (Ok(old_nz), Ok(new_nz)) => {
                // SAFETY: caller upholds allocator-api2's shrink contract.
                unsafe { self.inner.shrink(ptr, old_nz, new_nz) }
            }
            (Ok(old_nz), Err(_)) => {
                // Shrinking to zero: free and return a dangling pointer
                // aligned to the new layout.
                // SAFETY: caller promises ptr/old came from self.
                unsafe { self.inner.deallocate(ptr, old_nz) };
                Ok(NonNull::slice_from_raw_parts(
                    dangling_for_align(new.align()),
                    0,
                ))
            }
            (Err(_), Err(_)) => {
                // ZST → ZST: in-spec shrink (new.size() == old.size() == 0).
                // No deallocate (ptr is dangling for old ZST); return a fresh
                // dangling pointer aligned to the new layout.
                Ok(NonNull::slice_from_raw_parts(
                    dangling_for_align(new.align()),
                    0,
                ))
            }
            // (Err, Ok): growing from zero to non-zero via shrink is a
            // contract violation in allocator-api2; the inner allocator must
            // not be called.
            (Err(_), Ok(_)) => Err(AllocError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_size() {
        assert!(NonZeroLayout::from_size_align(0, 8).is_err());
    }

    #[test]
    fn rejects_non_power_of_two_align() {
        assert!(NonZeroLayout::from_size_align(16, 3).is_err());
    }

    #[test]
    fn for_type_zst_returns_none() {
        assert!(NonZeroLayout::for_type::<()>().is_none());
    }

    #[test]
    fn for_type_struct_returns_some() {
        let l = NonZeroLayout::for_type::<u64>().expect("u64 is not a ZST");
        assert_eq!(l.size().get(), 8);
        assert_eq!(l.align().get(), 8);
    }

    #[test]
    fn pad_to_align_rounds_up() {
        let l = NonZeroLayout::from_size_align(13, 8).unwrap();
        assert_eq!(l.pad_to_align().size().get(), 16);
    }

    #[test]
    fn pad_to_align_idempotent_when_aligned() {
        let l = NonZeroLayout::from_size_align(16, 8).unwrap();
        assert_eq!(l.pad_to_align().size().get(), 16);
    }

    #[test]
    fn roundtrip_via_layout() {
        let nzl = NonZeroLayout::from_size_align(24, 8).unwrap();
        let l: Layout = nzl.into();
        let back: NonZeroLayout = l.try_into().unwrap();
        assert_eq!(nzl, back);
    }

    #[test]
    fn array_overflow_returns_none() {
        assert!(NonZeroLayout::array::<u64>(usize::MAX).is_none());
    }

    #[test]
    fn array_zero_elements_returns_none() {
        assert!(NonZeroLayout::array::<u64>(0).is_none());
    }

    /// Boundary: `size == isize::MAX - (align - 1)` is exactly the largest
    /// permitted size for the given alignment per `NonZeroLayout::new`'s
    /// overflow rule. Off-by-one here would either spuriously reject the
    /// boundary or accept a value one past it.
    #[test]
    fn new_accepts_exact_isize_max_minus_align_minus_one() {
        for align in [1, 2, 4, 8, 16, 32, 1024, 4096] {
            let max = (isize::MAX as usize) - (align - 1);
            let r = NonZeroLayout::from_size_align(max, align);
            assert!(
                r.is_ok(),
                "size = isize::MAX - (align-1) = {max} with align = {align} must be accepted",
            );
            // One past must be rejected.
            let over = max.wrapping_add(1);
            // (skip when max + 1 wraps past usize::MAX, which doesn't apply
            // for any normal align on 64-bit; on 32-bit isize::MAX < usize::MAX
            // so wrap is impossible too.)
            if over != 0 {
                assert!(
                    NonZeroLayout::from_size_align(over, align).is_err(),
                    "size = isize::MAX - (align-1) + 1 = {over} (align = {align}) must be rejected",
                );
            }
        }
    }

    /// Boundary: `pad_to_align` on a layout whose size is the maximum
    /// admissible — `size + (align - 1)` must not overflow. The constraint
    /// in `new()` guarantees `size + (align - 1) <= isize::MAX`, so the
    /// computed padded value cannot wrap.
    #[test]
    fn pad_to_align_at_boundary_does_not_overflow() {
        for align in [1, 2, 4, 8, 16, 4096] {
            let max = (isize::MAX as usize) - (align - 1);
            let l = NonZeroLayout::from_size_align(max, align).unwrap();
            // pad_to_align should produce size == isize::MAX (already aligned
            // when `size + align - 1 == isize::MAX` and align is a power of two).
            let padded = l.pad_to_align();
            assert!(padded.size().get() >= max);
            assert!(padded.size().get() <= isize::MAX as usize);
        }
    }

    /// `array::<T>(usize::MAX)` must reject via the `checked_mul` overflow
    /// guard rather than panicking or producing a bogus layout.
    #[test]
    fn array_usize_max_elements_returns_none() {
        assert!(NonZeroLayout::array::<u8>(usize::MAX).is_none());
        assert!(NonZeroLayout::array::<u32>(usize::MAX).is_none());
        assert!(NonZeroLayout::array::<u64>(usize::MAX).is_none());
        // Even for a 1-byte-aligned ZST? for_type::<()>() returns None first,
        // so array<()> is None too.
        assert!(NonZeroLayout::array::<()>(usize::MAX).is_none());
    }

    /// Regression: `StdCompat::shrink` previously returned `Err(AllocError)`
    /// for the in-spec ZST → ZST shrink case. The allocator-api2 contract
    /// requires only `new.size() <= old.size()`, so `0 <= 0` must succeed.
    #[test]
    fn stdcompat_shrink_zst_to_zst_succeeds() {
        struct NeverAlloc;
        // SAFETY: a never-called Deallocator stub — used only to satisfy the
        // bound on `StdCompat`; the ZST → ZST shrink path never reaches it.
        unsafe impl crate::Deallocator for NeverAlloc {
            unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {
                unreachable!("ZST → ZST shrink must not touch the inner allocator");
            }
        }
        // SAFETY: as above — never called for this test.
        unsafe impl crate::Allocator for NeverAlloc {
            fn allocate(&self, _layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
                unreachable!("ZST → ZST shrink must not touch the inner allocator");
            }
        }

        let s = StdCompat::new(NeverAlloc);
        let zst = Layout::from_size_align(0, 8).unwrap();
        // SAFETY: ptr is dangling for a ZST input, never dereferenced.
        let result = unsafe {
            <StdCompat<NeverAlloc> as allocator_api2::alloc::Allocator>::shrink(
                &s,
                dangling_for_align(8),
                zst,
                zst,
            )
        };
        assert!(result.is_ok(), "ZST → ZST shrink should succeed");
        let slice = result.unwrap();
        assert_eq!(slice.len(), 0);
    }
}
