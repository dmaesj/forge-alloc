//! `AlignedColBuffer<ALIGN>` — an owned, over-aligned byte buffer for
//! **zero-copy columnar interop** (Apache Arrow & friends).
//!
//! Columnar formats want their buffers aligned for SIMD: Arrow *recommends*
//! 64-byte alignment for every buffer. The global allocator only promises
//! `max_align_t` (16 on most targets), so a plain `Vec<u8>` is not guaranteed
//! to satisfy it. `AlignedColBuffer` carries the required alignment **in its
//! type** — `AlignedColBuffer<64>` (the default) always begins on a 64-byte
//! boundary and *stays* aligned across every growth, so the bytes you build up
//! can be handed to a consumer with **no copy and no re-alignment**.
//!
//! This is the alignment-parametric allocator boundary applied to the columnar
//! handoff: build the column here, then pass the raw region out.
//!
//! # Scope
//!
//! `AlignedColBuffer` is a **convenience container for interop**, not a
//! composable allocator primitive. It owns its bytes directly via the global
//! allocator and does **not** implement
//! [`Allocator`](forge_alloc_core::Allocator) /
//! [`FixedRange`](forge_alloc_core::FixedRange) /
//! [`OsBacked`](forge_alloc_core::OsBacked), so it does not compose into the
//! allocator stacks in [`forge-alloc`]. Reach for those when you need
//! allocation structure or security hardening; reach for this when you need a
//! `Send + Sync`, stably-addressed, over-aligned byte region to ship zero-copy
//! across an FFI / columnar boundary.
//!
//! # Zero-copy handoff to Arrow
//!
//! `forge-containers` deliberately takes **no Arrow dependency** (arrow-rs bumps
//! a major roughly monthly; a foundational crate should not chase it). Instead
//! the buffer satisfies the three properties Arrow's `Buffer::from_custom_-`
//! `allocation` requires of an owner — it is `Send + Sync`, its data address is
//! **stable** (the allocation never moves; only `&mut self` can reallocate, and
//! once the buffer is moved into an `Arc` no `&mut` exists), and it keeps the
//! bytes alive for as long as it is held. The consumer writes one line, using
//! whatever arrow-rs version it already pins:
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use arrow_buffer::Buffer;
//! use forge_containers::AlignedColBuffer;
//!
//! let mut col = AlignedColBuffer::<64>::new();
//! col.extend_from_typed(&[1.0_f64, 2.0, 3.0]).unwrap();
//!
//! // Zero-copy: `Buffer` views `col`'s bytes; the `Arc<AlignedColBuffer>`
//! // is the owner that keeps them alive (it satisfies arrow's blanket
//! // `Allocation` impl: RefUnwindSafe + Send + Sync).
//! let len = col.len();
//! let ptr = col.as_non_null();
//! let buffer = unsafe { Buffer::from_custom_allocation(ptr, len, Arc::new(col)) };
//! ```
//!
//! After the handoff the `Arc` is shared-only, so the buffer can no longer be
//! mutated or reallocated — the pointer Arrow holds is stable for the buffer's
//! lifetime. Build fully *before* freezing into the `Arc`.
//!
//! [`forge-alloc`]: https://docs.rs/forge-alloc

use core::mem;
use core::ptr::NonNull;
use core::slice;

use allocator_api2::alloc::{Allocator as A2, Global};
use forge_alloc_core::{AllocError, NonZeroLayout};

/// An owned byte buffer whose data is always aligned to `ALIGN` bytes,
/// preserved across every growth — built for zero-copy columnar interop.
///
/// `ALIGN` must be a non-zero power of two; a bad value is a **compile error**
/// (post-monomorphization const assertion), not a runtime panic. The default
/// is `64`, Arrow's recommended SIMD alignment.
///
/// Mutators (`push`, `extend_from_slice`, `extend_from_typed`, `reserve`) take
/// `&mut self` and may reallocate, re-aligning to `ALIGN`. Reads (`as_bytes`,
/// `as_non_null`, `len`) take `&self`. The type is `Send + Sync` with a stable
/// data address, which is exactly what a zero-copy owner (e.g. Arrow's
/// `Buffer::from_custom_allocation`) needs — see the module-level
/// documentation for the zero-copy handoff recipe.
///
/// Empty buffers do not allocate: a freshly [`new`](Self::new)'d buffer holds a
/// dangling-but-`ALIGN`-aligned pointer and zero capacity until the first byte
/// is reserved.
///
/// A non-power-of-two `ALIGN` does not compile:
///
/// ```compile_fail
/// use forge_containers::AlignedColBuffer;
/// // ALIGN = 3 is not a power of two -> const-assertion failure at compile time.
/// let _ = AlignedColBuffer::<3>::new();
/// ```
pub struct AlignedColBuffer<const ALIGN: usize = 64> {
    /// `ALIGN`-aligned pointer to the allocation. When `cap == 0` this is a
    /// dangling sentinel (`ALIGN as *mut u8`) — non-null, aligned, never read.
    ptr: NonNull<u8>,
    /// Number of initialized, valid bytes (`<= cap`).
    len: usize,
    /// Allocated capacity in bytes. `0` means no allocation is held.
    cap: usize,
}

impl<const ALIGN: usize> AlignedColBuffer<ALIGN> {
    /// Compile-time guard: `ALIGN` must be a non-zero power of two. Referenced
    /// in every constructor so an invalid `ALIGN` fails to compile rather than
    /// producing a buffer that can't form a valid `Layout`.
    const ASSERT_ALIGN: () = assert!(
        ALIGN.is_power_of_two(),
        "AlignedColBuffer ALIGN must be a non-zero power of two",
    );

    /// The dangling, `ALIGN`-aligned sentinel used while `cap == 0`.
    #[inline]
    const fn dangling() -> NonNull<u8> {
        // Strict-provenance: a no-provenance pointer at address `ALIGN`. This is
        // how `NonNull::dangling` is built internally, but for `u8` (align 1) we
        // need the `ALIGN` address rather than 1, so we construct it directly.
        let addr = core::ptr::without_provenance_mut::<u8>(ALIGN);
        // SAFETY: `ALIGN` is a non-zero power of two (enforced by `ASSERT_ALIGN`,
        // referenced in the constructors that are the only way to reach this), so
        // `addr` is non-null and `ALIGN`-aligned. It is a sentinel for a
        // zero-length buffer and is never dereferenced.
        unsafe { NonNull::new_unchecked(addr) }
    }

    /// Create an empty buffer. Does not allocate; the first
    /// [`reserve`](Self::reserve)/`push`/`extend_*` allocates.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        let () = Self::ASSERT_ALIGN;
        Self {
            ptr: Self::dangling(),
            len: 0,
            cap: 0,
        }
    }

    /// Create an empty buffer with room for at least `capacity` bytes
    /// pre-allocated (still `len == 0`). Errors only if the allocation fails or
    /// `capacity` overflows a valid `Layout`.
    pub fn with_capacity(capacity: usize) -> Result<Self, AllocError> {
        let () = Self::ASSERT_ALIGN;
        let mut buf = Self::new();
        if capacity > 0 {
            buf.grow_to(capacity)?;
        }
        Ok(buf)
    }

    /// The alignment guarantee in bytes (the `ALIGN` const parameter).
    #[inline]
    #[must_use]
    pub const fn alignment(&self) -> usize {
        ALIGN
    }

    /// Number of initialized, valid bytes.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no valid bytes.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocated capacity in bytes (`>= len`).
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// The `ALIGN`-aligned data pointer, as a `NonNull<u8>` — the argument a
    /// zero-copy consumer (e.g. Arrow) takes alongside [`len`](Self::len). Valid
    /// for `len` bytes of reads; a dangling-but-aligned sentinel when empty.
    #[inline]
    #[must_use]
    pub const fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
    }

    /// The `ALIGN`-aligned data pointer.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// The initialized bytes as a shared slice.
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8] {
        // SAFETY: `ptr` is `ALIGN`-aligned and valid for `len` initialized
        // bytes (or a dangling-aligned sentinel when `len == 0`, for which a
        // zero-length slice is always valid). Tied to `&self`.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Reinterpret the initialized bytes as a typed slice `&[T]`, or `None` if
    /// the length is not a whole multiple of `size_of::<T>()`, the data pointer
    /// is not aligned for `T`, or `T` is a ZST.
    ///
    /// # Safety
    ///
    /// The caller must guarantee the `len` initialized bytes are a valid bit
    /// pattern for `[T]` (e.g. written via [`extend_from_typed`] with the same
    /// `T`). Reinterpreting arbitrary bytes as a type with invalid bit patterns
    /// (`bool`, `char`, non-`#[repr(C)]` enums, references) is undefined
    /// behavior — this method does not and cannot check it. For plain numeric
    /// element types (`i8`..`i128`, `u8`..`u128`, `f32`, `f64`) every bit
    /// pattern is valid, so the precondition is automatically satisfied.
    ///
    /// [`extend_from_typed`]: Self::extend_from_typed
    #[must_use]
    pub unsafe fn as_typed<T>(&self) -> Option<&[T]> {
        let size = mem::size_of::<T>();
        if size == 0
            || self.len % size != 0
            || (self.ptr.as_ptr() as usize) % mem::align_of::<T>() != 0
        {
            return None;
        }
        let n = self.len / size;
        // SAFETY: alignment and `len % size == 0` are checked above, so the
        // region holds exactly `n` `T`-sized, `T`-aligned slots within the
        // initialized `len` bytes. The caller's documented precondition
        // guarantees those bytes are a valid `[T]`. Tied to `&self`.
        unsafe { Some(slice::from_raw_parts(self.ptr.as_ptr().cast::<T>(), n)) }
    }

    /// Reset the length to zero, retaining the allocation for reuse. The
    /// capacity (and the alignment guarantee) are unchanged.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Ensure capacity for at least `additional` more bytes beyond the current
    /// length, reallocating (amortized doubling) and preserving `ALIGN` if
    /// needed. Errors if the new size overflows a valid `Layout` or the
    /// allocator fails.
    pub fn reserve(&mut self, additional: usize) -> Result<(), AllocError> {
        let required = self.len.checked_add(additional).ok_or(AllocError)?;
        if required <= self.cap {
            return Ok(());
        }
        // Amortized doubling; `saturating_mul` so a huge cap can't overflow the
        // growth heuristic (the real bound is the `Layout` check in `grow_to`).
        let new_cap = core::cmp::max(required, self.cap.saturating_mul(2));
        self.grow_to(new_cap)
    }

    /// Append one byte, growing if needed.
    #[inline]
    pub fn push(&mut self, byte: u8) -> Result<(), AllocError> {
        if self.len == self.cap {
            self.reserve(1)?;
        }
        // SAFETY: after `reserve` there is room for one more byte at offset
        // `len < cap`; `ptr` is valid for `cap` bytes.
        unsafe { self.ptr.as_ptr().add(self.len).write(byte) };
        self.len += 1;
        Ok(())
    }

    /// Append a byte slice, growing if needed.
    pub fn extend_from_slice(&mut self, bytes: &[u8]) -> Result<(), AllocError> {
        self.reserve(bytes.len())?;
        // SAFETY: `reserve` guaranteed `cap - len >= bytes.len()`, so the
        // destination `[ptr+len, ptr+len+bytes.len())` lies inside the
        // allocation; source and destination are distinct allocations (the
        // input slice is borrowed, not aliasing our owned block). For
        // `bytes.len() == 0` this is a no-op on a possibly-dangling pointer,
        // which `copy_nonoverlapping` permits at count 0.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.ptr.as_ptr().add(self.len),
                bytes.len(),
            );
        }
        self.len += bytes.len();
        Ok(())
    }

    /// Append the raw bytes of a typed slice (`size_of::<T>() * slice.len()`
    /// bytes), growing if needed. Building a column of `T` this way and reading
    /// it back with [`as_typed::<T>`](Self::as_typed) round-trips when the
    /// buffer holds only `T` values and `ALIGN >= align_of::<T>()`.
    pub fn extend_from_typed<T: Copy>(&mut self, slice: &[T]) -> Result<(), AllocError> {
        let byte_len = mem::size_of_val(slice);
        self.reserve(byte_len)?;
        // SAFETY: `reserve` guaranteed room for `byte_len` bytes at offset
        // `len`. `T: Copy`, so reading its object representation as bytes is
        // sound; source (`slice`) and our owned destination do not overlap. The
        // byte copy imposes no alignment requirement on either side. No-op at
        // `byte_len == 0` (empty or ZST slice).
        unsafe {
            core::ptr::copy_nonoverlapping(
                slice.as_ptr().cast::<u8>(),
                self.ptr.as_ptr().add(self.len),
                byte_len,
            );
        }
        self.len += byte_len;
        Ok(())
    }

    /// Grow the allocation to `new_cap` bytes (`new_cap > cap`), preserving
    /// `ALIGN` and the first `len` bytes. Allocates fresh when `cap == 0`,
    /// otherwise reallocates in place where the allocator can.
    fn grow_to(&mut self, new_cap: usize) -> Result<(), AllocError> {
        debug_assert!(
            new_cap > self.cap,
            "grow_to must strictly increase capacity"
        );
        let new_layout = NonZeroLayout::from_size_align(new_cap, ALIGN).map_err(|_| AllocError)?;
        let new_block = if self.cap == 0 {
            A2::allocate(&Global, new_layout.to_layout())?
        } else {
            let old_layout =
                NonZeroLayout::from_size_align(self.cap, ALIGN).map_err(|_| AllocError)?;
            // SAFETY: `self.ptr` was returned by `Global` for `old_layout`
            // (same `ALIGN`, size == old `cap`); `new_layout` keeps `ALIGN` and
            // `new_cap > cap`, satisfying `Allocator::grow`'s contract (same
            // alignment, non-shrinking). `grow` preserves the existing bytes.
            unsafe {
                A2::grow(
                    &Global,
                    self.ptr,
                    old_layout.to_layout(),
                    new_layout.to_layout(),
                )?
            }
        };
        self.ptr = new_block.cast::<u8>();
        self.cap = new_cap;
        Ok(())
    }
}

impl<const ALIGN: usize> Default for AlignedColBuffer<ALIGN> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const ALIGN: usize> Drop for AlignedColBuffer<ALIGN> {
    fn drop(&mut self) {
        if self.cap == 0 {
            // Empty: `ptr` is the dangling sentinel, nothing to free.
            return;
        }
        let layout = NonZeroLayout::from_size_align(self.cap, ALIGN)
            .expect("AlignedColBuffer: cap/ALIGN form a valid Layout by construction");
        // SAFETY: `ptr` was produced by `Global` for exactly this `(cap, ALIGN)`
        // layout (every allocation path in `grow_to` uses `ALIGN` and sets
        // `cap`), handed back verbatim. There is no `Clone` impl and no path
        // that copies `ptr` out, so this is the last owner of the block.
        unsafe { A2::deallocate(&Global, self.ptr, layout.to_layout()) };
    }
}

// SAFETY: `AlignedColBuffer` has exclusive ownership of at most one `Global`
// allocation via `NonNull`, with no interior mutability — every mutator takes
// `&mut self`. Moving the buffer transfers that unique allocation (the global
// allocator is callable from any thread), so `Send` is sound. Sharing `&self`
// exposes only reads of the owned bytes, so `Sync` is sound. This matches the
// `Send + Sync` of `Vec<u8>`, and is the property a zero-copy owner needs.
unsafe impl<const ALIGN: usize> Send for AlignedColBuffer<ALIGN> {}
unsafe impl<const ALIGN: usize> Sync for AlignedColBuffer<ALIGN> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn is_send_and_sync() {
        _assert_send_sync::<AlignedColBuffer<64>>();
        _assert_send_sync::<AlignedColBuffer<16>>();
    }

    #[test]
    fn empty_buffer_does_not_allocate_but_is_aligned() {
        let buf = AlignedColBuffer::<64>::new();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.alignment(), 64);
        assert_eq!(buf.as_bytes(), &[] as &[u8]);
        assert_eq!(
            buf.as_non_null().as_ptr() as usize % 64,
            0,
            "sentinel is aligned"
        );
    }

    #[test]
    fn push_grows_and_stays_aligned() {
        let mut buf = AlignedColBuffer::<64>::new();
        for i in 0..1000u32 {
            buf.push((i & 0xFF) as u8).unwrap();
            assert_eq!(
                buf.as_ptr() as usize % 64,
                0,
                "data must stay 64-aligned across reallocs (len={})",
                buf.len(),
            );
        }
        assert_eq!(buf.len(), 1000);
        assert!(buf.capacity() >= 1000);
        // Contents are correct.
        for (i, &b) in buf.as_bytes().iter().enumerate() {
            assert_eq!(b, (i & 0xFF) as u8);
        }
    }

    #[test]
    fn extend_from_slice_appends() {
        let mut buf = AlignedColBuffer::<32>::new();
        buf.extend_from_slice(b"hello").unwrap();
        buf.extend_from_slice(b"").unwrap(); // empty extend is a no-op
        buf.extend_from_slice(b" world").unwrap();
        assert_eq!(buf.as_bytes(), b"hello world");
        assert_eq!(buf.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn typed_round_trip_f64_and_i64() {
        let mut buf = AlignedColBuffer::<64>::new();
        let vals = [1.0_f64, 2.5, -3.25, 4.0];
        buf.extend_from_typed(&vals).unwrap();
        // SAFETY: only f64 values were written; every bit pattern is a valid f64.
        let view = unsafe { buf.as_typed::<f64>() }.expect("aligned, whole multiple");
        assert_eq!(view, &vals);

        let mut ints = AlignedColBuffer::<64>::new();
        let xs = [10_i64, -20, 30, 1 << 40];
        ints.extend_from_typed(&xs).unwrap();
        // SAFETY: only i64 values were written.
        let iv = unsafe { ints.as_typed::<i64>() }.expect("aligned, whole multiple");
        assert_eq!(iv, &xs);
    }

    #[test]
    fn as_typed_rejects_partial_and_zst() {
        let mut buf = AlignedColBuffer::<64>::new();
        buf.extend_from_slice(&[0u8; 5]).unwrap(); // 5 bytes — not a multiple of 4
                                                   // SAFETY: bytes are valid u32 bit patterns; method still returns None on
                                                   // the size-multiple mismatch without constructing a slice.
        assert!(unsafe { buf.as_typed::<u32>() }.is_none());
        // ZST is always rejected.
        // SAFETY: () has no bytes; method returns None for ZSTs.
        assert!(unsafe { buf.as_typed::<()>() }.is_none());
    }

    #[test]
    fn clear_keeps_capacity() {
        let mut buf = AlignedColBuffer::<64>::new();
        buf.extend_from_slice(&[7u8; 128]).unwrap();
        let cap = buf.capacity();
        let ptr = buf.as_ptr();
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), cap, "clear retains the allocation");
        assert_eq!(buf.as_ptr(), ptr, "clear does not move the allocation");
        // Reuse after clear.
        buf.extend_from_slice(&[9u8; 64]).unwrap();
        assert_eq!(buf.as_bytes(), &[9u8; 64]);
    }

    #[test]
    fn with_capacity_preallocates() {
        let buf = AlignedColBuffer::<64>::with_capacity(256).unwrap();
        assert_eq!(buf.len(), 0);
        assert!(buf.capacity() >= 256);
        assert_eq!(buf.as_ptr() as usize % 64, 0);
        // with_capacity(0) behaves like new().
        let empty = AlignedColBuffer::<64>::with_capacity(0).unwrap();
        assert_eq!(empty.capacity(), 0);
    }

    #[test]
    fn larger_alignment_is_respected() {
        let mut buf = AlignedColBuffer::<128>::new();
        for _ in 0..500 {
            buf.push(0xAB).unwrap();
            assert_eq!(buf.as_ptr() as usize % 128, 0);
        }
    }

    /// Models the zero-copy handoff: freeze the buffer behind a shared owner
    /// and confirm the bytes remain readable through the captured raw pointer
    /// (what Arrow's `Buffer` does). Run under Miri this catches a
    /// use-after-free or provenance bug in the ownership transfer.
    #[test]
    fn frozen_owner_keeps_bytes_alive() {
        use alloc::sync::Arc;
        let mut buf = AlignedColBuffer::<64>::new();
        buf.extend_from_typed(&[1u32, 2, 3, 4]).unwrap();
        let ptr = buf.as_non_null();
        let len = buf.len();
        let owner = Arc::new(buf); // freeze: no more &mut, address is now stable
                                   // Read through the raw pointer while the owner Arc is alive.
                                   // SAFETY: `owner` keeps the allocation alive; `ptr`/`len` describe its
                                   // initialized, 4-aligned bytes; no &mut exists to realloc it.
        let bytes = unsafe { slice::from_raw_parts(ptr.as_ptr(), len) };
        assert_eq!(bytes.len(), 16);
        // Reinterpret as u32 (all bit patterns valid).
        // SAFETY: 16 bytes, 4-aligned start, written as u32.
        let view = unsafe { slice::from_raw_parts(ptr.as_ptr().cast::<u32>(), len / 4) };
        assert_eq!(view, &[1, 2, 3, 4]);
        drop(owner);
    }
}
