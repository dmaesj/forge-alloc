//! `ZeroizeOnFree<I>` — volatile-zeroes freed memory so the zeroing cannot be
//! optimized away, making freed secret material much harder to recover.
//!
//! This is the crypto-grade counterpart to [`PoisonOnFree`](crate::hardening::PoisonOnFree).
//! Where `PoisonOnFree` writes a configurable byte with a plain
//! `write_bytes` — which the compiler may **dead-store-eliminate** once it
//! proves the block is freed and never read again — `ZeroizeOnFree` writes
//! zeroes through [`core::ptr::write_volatile`]. Volatile accesses are
//! observable side effects on the abstract machine, so the optimizer may not
//! drop them as dead stores — *that, alone, is the non-elision guarantee*, the
//! same mechanism C's `memset_s` / `explicit_bzero` and the `zeroize` crate
//! rely on, with no external dependency. A trailing
//! [`compiler_fence`](core::sync::atomic::compiler_fence) is a defensive
//! compile-time barrier; it emits no CPU fence and does not order the volatile
//! stores against a later non-volatile read.
//!
//! For secret material (keys, plaintext-before-encryption, tokens) prefer this
//! over `PoisonOnFree<_>` with a zero pattern: it is strictly stronger because
//! the write cannot be elided. Note the same freelist-link caveat as
//! `PoisonOnFree` applies — see the type docs on composition order.
//!
//! See `docs/ARCHITECTURE.md` for the composable-wrapper design.

use core::ptr::NonNull;
use core::sync::atomic::{compiler_fence, Ordering};

use forge_alloc_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// Volatile-zero `[ptr, ptr + len)`, then a compiler fence.
///
/// Byte-wise volatile writes (the `zeroize`-crate / `explicit_bzero` approach):
/// each write is observable on the abstract machine, so the optimizer may not
/// drop it as a dead store — this is the whole non-elision guarantee.
/// Byte-wise (not word-wise) keeps it correct at any alignment; `ptr` may be
/// only byte-aligned. The trailing `compiler_fence(SeqCst)` is a defensive
/// compile-time barrier: it pins the scrub ahead of later compiler-visible
/// accesses in program order, but emits no CPU barrier and does not order the
/// volatile stores against a later non-volatile read. Non-elision does not
/// depend on it.
///
/// # Safety
///
/// `ptr` must be valid for writes of `len` bytes.
#[inline]
unsafe fn volatile_zeroize(ptr: *mut u8, len: usize) {
    for i in 0..len {
        // SAFETY: caller guarantees `[ptr, ptr+len)` is writable, so each
        // `ptr.add(i)` for `i < len` is in bounds and writable.
        unsafe { core::ptr::write_volatile(ptr.add(i), 0u8) };
    }
    compiler_fence(Ordering::SeqCst);
}

/// Wrapper that volatile-zeroes freed memory.
///
/// `Send` if `I: Send`. `Sync` if `I: Sync`. No additional synchronization
/// hazards beyond the inner allocator's.
///
/// # Coverage and composition order
///
/// Like [`PoisonOnFree`](crate::hardening::PoisonOnFree), `ZeroizeOnFree`
/// scrubs the *entire* `[ptr, ptr + size)` region **before** forwarding to
/// `self.inner.deallocate`. If the inner allocator writes back into the freed
/// region (e.g. `Slab` / `SizeClassed` stamp a freelist link over the first
/// 4–8 bytes), those bytes then hold link data rather than zeroes — exactly the
/// caveat documented on `PoisonOnFree`. For maximum persistence of the zeroing,
/// keep `ZeroizeOnFree` **outermost** and put a
/// [`Quarantine`](crate::hardening::Quarantine) between it and a freelist
/// allocator: `ZeroizeOnFree<Quarantine<Slab<..>>>`. The zero write lands on
/// the outer deallocate, and quarantine holds the slot off the freelist for
/// several cycles before any link is written.
///
/// # `grow` / `shrink`
///
/// Unlike `PoisonOnFree`, `ZeroizeOnFree` does **not** forward `grow`/`shrink`
/// to the inner allocator. It uses the [`Allocator`] trait defaults, which
/// allocate-copy-then-`self.deallocate(old)`. Routing the old allocation
/// through *this* wrapper's zeroizing `deallocate` guarantees that a moved-from
/// block (and `shrink`'s discarded tail) is erased — closing the gap where an
/// in-place-resize forward would leave the original secret bytes intact. The
/// cost is that an inner allocator's native in-place resize is not used; for
/// secret material the guaranteed erasure is the right trade.
pub struct ZeroizeOnFree<I> {
    inner: I,
}

impl<I> ZeroizeOnFree<I> {
    /// Wrap an inner allocator so freed blocks are volatile-zeroed.
    #[inline]
    pub const fn new(inner: I) -> Self {
        Self { inner }
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }
}

unsafe impl<I: Allocator> Deallocator for ZeroizeOnFree<I> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // SAFETY: per the Deallocator contract, `ptr` came from this
        // allocator's `allocate(layout)`, so `[ptr, ptr+size)` is writable for
        // `layout.size()` bytes. We scrub before handing back to the inner.
        unsafe {
            volatile_zeroize(ptr.as_ptr(), layout.size().get());
            self.inner.deallocate(ptr, layout);
        }
    }
}

unsafe impl<I: Allocator> Allocator for ZeroizeOnFree<I> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner.allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner.allocate_zeroed(layout)
    }

    // `grow` / `shrink` are deliberately NOT overridden — see the type-level
    // docs. The trait defaults allocate-copy-then-`self.deallocate(old)`, which
    // routes the moved-from block through this wrapper's zeroizing deallocate.

    /// Bulk-reclaim the inner allocator (arenas only).
    ///
    /// **This does not scrub.** `reset` forwards the inner's cursor reclaim; it
    /// does *not* zeroize the previously-issued bytes — those are erased only by
    /// the per-block `deallocate` path, or overwritten by a later `allocate`.
    /// For guaranteed erasure of secret material, free blocks individually
    /// rather than resetting.
    #[inline]
    fn reset(&mut self) -> Result<(), AllocError> {
        self.inner.reset()
    }

    #[inline]
    unsafe fn usable_size(&self, ptr: NonNull<u8>, layout: NonZeroLayout) -> Option<usize> {
        // SAFETY: forwarded; caller upholds usable_size's contract on inner.
        unsafe { self.inner.usable_size(ptr, layout) }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        self.inner.capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        self.inner.corruption_events()
    }
}

impl<I: FixedRange> FixedRange for ZeroizeOnFree<I> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.inner.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Pass-through so a `commit`-aware consumer reaches the inner backing when
    /// this wrapper sits over a `lazy_commit` `MmapBacked`.
    #[inline]
    fn commit(&self, offset: usize, len: usize) -> Result<(), AllocError> {
        self.inner.commit(offset, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;
    use crate::layout::BumpArena;

    /// `BumpArena`'s deallocate is a no-op and it never reuses bytes, so after
    /// our zeroize runs we can read the freed region directly and observe the
    /// zeroes. (A UAF read that is sound only because of bump semantics — the
    /// same technique `PoisonOnFree`'s tests use.)
    #[test]
    fn freed_bytes_are_zeroed_on_bump_arena() {
        let z: ZeroizeOnFree<BumpArena<InlineBacked<256>>> =
            ZeroizeOnFree::new(BumpArena::new(InlineBacked::<256>::new()).unwrap());
        let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
        let block = z.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0xAA, 16);
            assert_eq!(*ptr.as_ptr(), 0xAA);
            z.deallocate(ptr, layout);
            for i in 0..16 {
                assert_eq!(*ptr.as_ptr().add(i), 0x00, "byte {i} was not zeroed");
            }
        }
    }

    /// The default `grow` allocate-copy-frees through `self.deallocate`, so the
    /// moved-from block must come back zeroed — proving the design choice to
    /// forgo the inner's in-place resize actually erases the old secret.
    #[test]
    fn grow_zeroes_the_moved_from_block() {
        let z: ZeroizeOnFree<BumpArena<InlineBacked<256>>> =
            ZeroizeOnFree::new(BumpArena::new(InlineBacked::<256>::new()).unwrap());
        let old = NonZeroLayout::from_size_align(16, 8).unwrap();
        let new = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = z.allocate(old).unwrap();
        let old_ptr = block.cast::<u8>();
        unsafe {
            core::ptr::write_bytes(old_ptr.as_ptr(), 0xAA, 16);
            let grown = z.grow(old_ptr, old, new).unwrap();
            let new_ptr = grown.cast::<u8>();
            // Relies on BumpArena NOT implementing an in-place `grow`: it uses
            // the trait default (allocate-copy-free), so the cursor advances and
            // the addresses differ, leaving the old region unused and readable
            // in this test. If BumpArena ever grows in place this assertion (and
            // the moved-from read below) would no longer apply.
            assert_ne!(old_ptr.as_ptr(), new_ptr.as_ptr());
            for i in 0..16 {
                assert_eq!(
                    *old_ptr.as_ptr().add(i),
                    0x00,
                    "moved-from byte {i} was not zeroed",
                );
            }
            // The copied data survived into the new block.
            assert_eq!(*new_ptr.as_ptr(), 0xAA);
        }
    }

    #[test]
    fn inner_is_accessible() {
        let z = ZeroizeOnFree::new(InlineBacked::<64>::new());
        // FixedRange passthrough reaches the inner backing.
        assert_eq!(z.size(), 64);
    }

    /// `reset` must forward to the inner arena (not the default `Err`), so a
    /// wrapped bump arena stays resettable — regression guard for the
    /// usable_size/reset forwarding.
    #[test]
    fn reset_forwards_to_inner_arena() {
        let mut z: ZeroizeOnFree<BumpArena<InlineBacked<256>>> =
            ZeroizeOnFree::new(BumpArena::new(InlineBacked::<256>::new()).unwrap());
        let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
        let _ = z.allocate(layout).unwrap();
        assert!(z.inner().allocated() > 0);
        assert!(z.reset().is_ok(), "wrapped arena must be resettable");
        assert_eq!(z.inner().allocated(), 0);
    }
}
