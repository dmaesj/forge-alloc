//! `PoisonOnFree<I>` — overwrites freed memory with a configurable pattern
//! before returning control to the inner allocator.
//!
//! Prevents freed data (PHI, key material, session tokens) from being
//! recovered via UAF reads or from persisting in freed memory for
//! information disclosure. Complements [`crate::hardening::Watermark`] /
//! [`crate::hardening::Quarantine`] — poison destroys content, quarantine delays reuse.
//!
//! See `docs/ARCHITECTURE.md` for the composable-wrapper design.

use core::ptr::NonNull;

use forge_alloc_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// Default poison byte. `0xDE` is a recognizable hex marker that's unlikely
/// to be a valid value for most types and easy to spot in a memory dump.
pub const DEFAULT_POISON: u8 = 0xDE;

/// Wrapper that overwrites freed memory with a poison pattern.
///
/// `Send` if `I: Send`. `Sync` if `I: Sync`. No additional synchronization
/// hazards beyond the inner allocator's.
///
/// # Poison coverage caveats
///
/// `PoisonOnFree` writes the pattern across the *entire* `[ptr, ptr+size)`
/// region **before** forwarding to `self.inner.deallocate(ptr, layout)`.
/// What happens after that handoff is the inner allocator's business — and
/// some inner allocators reuse the very bytes we just poisoned for their
/// own freelist bookkeeping:
///
/// - **`Slab<T, _, M>` / `SizeClassed<_, _>`**: on `deallocate`, the slab
///   writes a `FreeLink` (`{ next_idx: u32, mac: u32 }` for `Slab`, 4 bytes
///   for `SizeClassed`'s `UntypedSlab`) at the *start* of the freed slot.
///   That overwrites the first 4–8 bytes of our poison; effective coverage
///   on a slab-backed `PoisonOnFree` is `[ptr + size_of::<FreeLink>(), ptr + size)`.
/// - **`BumpArena` / `SharedBumpArena` / `StackAlloc`**: deallocate is a
///   no-op (or pure cursor-pop), so poison persists in full.
/// - **`System` / `MmapBacked`**: the OS / global allocator may zero the
///   region for security or reuse it for its own metadata; poison may
///   survive on `MmapBacked` until the OS reclaims, never on `System`.
///
/// The security claim "freed-data disclosure prevented via UAF read" is
/// therefore *partial* whenever the inner allocator writes back into the
/// just-freed region: the bytes overlapping the inner's freelist link
/// hold link data rather than poison.
///
/// **Composition that maximizes poison persistence**:
/// `PoisonOnFree<Quarantine<Slab>>` — poison is written immediately on
/// outer-most dealloc, and `Quarantine`'s epoch delay keeps the slot off
/// `Slab`'s freelist for several deallocate calls. During that window a
/// UAF read sees fully-poisoned bytes; once `Quarantine` evicts to
/// `Slab`, the first 4–8 bytes are then overwritten with the freelist
/// link as above.
///
/// **Avoid** `Quarantine<PoisonOnFree<Slab>>` — that composition delays
/// the poison write until eviction, so a UAF read during the quarantine
/// window sees the original (un-poisoned) data. The wrapping order
/// matters for the security property, not just for the layout.
///
/// # Composition with `Canary`
///
/// [`Canary`](crate::hardening::Canary) zeros its own pre- and post-canary words
/// on deallocate, so the canary seed itself is wiped regardless of
/// composition order. Coverage of the *adjacent* bytes still depends
/// on order:
///
/// - `Canary<PoisonOnFree<Inner>>` (Canary **outer**, PoisonOnFree
///   **inner**): on deallocate, Canary verifies+zeros its canary words
///   first, then forwards to PoisonOnFree which poisons the *entire
///   inner region* — including pre-padding, the user region, and the
///   slot bytes that held the canary words (now overwritten with the
///   poison pattern). Maximum coverage.
/// - `PoisonOnFree<Canary<Inner>>` (PoisonOnFree **outer**, Canary
///   **inner**): PoisonOnFree poisons only `[user_ptr, user_ptr+size)`
///   — *not* the canary words at `user_ptr-8` / `user_ptr+size`, and
///   not the pre-padding before `user_ptr-8`. Canary then zeros the
///   canary words and forwards. Coverage of the user region is the
///   same; coverage of the padding/canary slots is empty (zeroed
///   canaries, untouched padding). Pick this only if the inner
///   allocator's freelist link sits in the user region (e.g.
///   `Slab<T, _>`) — there PoisonOnFree-first writes the poison
///   *before* the slab overwrites the first 4-8 bytes with the
///   freelist link, so post-link coverage matches the outer
///   composition.
///
/// # `grow` / `shrink`
///
/// `PoisonOnFree` does **not** forward `grow`/`shrink` to the inner
/// allocator; it uses the [`Allocator`] trait defaults, which
/// allocate-copy-then-`self.deallocate(old)`. Routing the old allocation
/// through *this* wrapper's poisoning `deallocate` guarantees the moved-from
/// block (and `shrink`'s discarded tail) is poisoned. Forwarding to the
/// inner's native `grow`/`shrink` would let a *relocating* resize free the old
/// block through the inner's deallocate, leaving the original secret bytes
/// intact and un-poisoned — the gap this choice closes, matching
/// [`ZeroizeOnFree`](crate::hardening::ZeroizeOnFree). The cost is that an
/// inner allocator's native in-place resize is not used.
pub struct PoisonOnFree<I> {
    inner: I,
    pattern: u8,
}

impl<I> PoisonOnFree<I> {
    /// Wrap with the default poison byte (`0xDE`).
    #[inline]
    pub const fn new(inner: I) -> Self {
        Self {
            inner,
            pattern: DEFAULT_POISON,
        }
    }

    /// Wrap with an explicit poison byte. Use `0x00` for zero-fill if
    /// integrating with code that reads zero-filled freed memory by
    /// convention; otherwise `0xDE`-style sentinels are easier to debug.
    #[inline]
    pub const fn with_pattern(inner: I, pattern: u8) -> Self {
        Self { inner, pattern }
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }

    /// The poison byte this wrapper writes.
    #[inline]
    pub const fn pattern(&self) -> u8 {
        self.pattern
    }
}

unsafe impl<I: Allocator> Deallocator for PoisonOnFree<I> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // SAFETY: per the Deallocator contract, ptr came from this allocator's
        // allocate(layout), so the allocation is writable for at least
        // `layout.size()` bytes. We poison the *full usable extent*: a caller
        // may query `usable_size()` and write into the slack tail
        // `[size, usable_size)`, so it must be covered too. `usable_size`'s
        // `Some(n)` is a promise the allocation is valid for `n` bytes, so the
        // write stays in bounds; fall back to `layout.size()` on `None`.
        unsafe {
            let scrub_len = self
                .inner
                .usable_size(ptr, layout)
                .unwrap_or_else(|| layout.size().get());
            core::ptr::write_bytes(ptr.as_ptr(), self.pattern, scrub_len);
            self.inner.deallocate(ptr, layout);
        }
    }
}

unsafe impl<I: Allocator> Allocator for PoisonOnFree<I> {
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
    // routes the moved-from block (and `shrink`'s discarded tail) through this
    // wrapper's poisoning `deallocate`. Forwarding to `self.inner.grow`/`shrink`
    // would let a relocating inner free the old block through the *inner's*
    // deallocate, leaving the moved-from secret bytes un-poisoned.

    /// Bulk-reclaim the inner allocator (arenas only). Forwards the inner's
    /// cursor reclaim; it does **not** poison the previously-issued bytes —
    /// those are overwritten on the per-block `deallocate` path or by a later
    /// `allocate`. Without this forward a wrapped `BumpArena` could not be
    /// reset at all (the trait default returns `Err`).
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

/// `FixedRange` passthrough so this wrapper composes over a `lazy_commit`
/// `MmapBacked` and similar backings.
///
/// **Footgun:** the poison-on-free scrub runs only in this wrapper's
/// `deallocate`. If you nest it *as a backing under* an arena —
/// `BumpArena<PoisonOnFree<..>>` — the arena carves directly from
/// `base()`/`size()` and its own `deallocate` is a no-op, so the scrub **never
/// runs**. Keep the hardening wrapper **outermost** (wrapping the allocator),
/// never as the `FixedRange` an arena consumes.
impl<I: FixedRange> FixedRange for PoisonOnFree<I> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;
    use crate::layout::Slab;

    /// Validates the documented *partial-coverage* claim against a real
    /// freelist allocator: the 32-byte slot is larger than the 8-byte
    /// `FreeLink` the slab stamps at the slot start on deallocate, so the
    /// bytes *after* the link must hold poison while the first 8 hold the
    /// link. (Previously this test asserted nothing post-deallocate.)
    #[test]
    fn freed_bytes_carry_poison_pattern() {
        let p: PoisonOnFree<Slab<[u8; 32], InlineBacked<512>>> =
            PoisonOnFree::new(Slab::new(8, InlineBacked::<512>::new()).unwrap());
        let layout = NonZeroLayout::for_type::<[u8; 32]>().unwrap();
        let block = p.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        unsafe {
            // Write a sentinel that's NOT the poison pattern.
            core::ptr::write_bytes(ptr.as_ptr(), 0xAA, 32);
            assert_eq!(*ptr.as_ptr(), 0xAA);
            p.deallocate(ptr, layout);
            // Poison ran across the whole slot BEFORE Slab::deallocate stamped
            // its 8-byte FreeLink over `[0, 8)`. The tail `[8, 32)` is never
            // touched by the slab, so it must still hold the poison pattern.
            // (UAF read sound: InlineBacked memory persists and we do not
            // reallocate before reading.)
            for i in 8..32 {
                assert_eq!(
                    *ptr.as_ptr().add(i),
                    DEFAULT_POISON,
                    "tail byte {i} (after FreeLink) not poisoned",
                );
            }
        }
    }

    /// `grow` must poison the moved-from block. Because `PoisonOnFree` does
    /// NOT override `grow`, the trait default routes the old allocation through
    /// this wrapper's poisoning `deallocate` — regression guard that a
    /// relocating grow does not leak the original secret bytes un-poisoned.
    #[test]
    fn grow_poisons_the_moved_from_block() {
        use crate::layout::BumpArena;
        let p: PoisonOnFree<BumpArena<InlineBacked<256>>> =
            PoisonOnFree::with_pattern(BumpArena::new(InlineBacked::<256>::new()).unwrap(), 0xBB);
        let old = NonZeroLayout::from_size_align(16, 8).unwrap();
        let new = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = p.allocate(old).unwrap();
        let old_ptr = block.cast::<u8>();
        unsafe {
            core::ptr::write_bytes(old_ptr.as_ptr(), 0xAA, 16);
            let grown = p.grow(old_ptr, old, new).unwrap();
            let new_ptr = grown.cast::<u8>();
            // BumpArena uses the default (allocate-copy-free) grow, so the
            // cursor advances and the moved-from region stays readable here.
            assert_ne!(old_ptr.as_ptr(), new_ptr.as_ptr());
            for i in 0..16 {
                assert_eq!(
                    *old_ptr.as_ptr().add(i),
                    0xBB,
                    "moved-from byte {i} not poisoned",
                );
            }
            // The copied data survived into the new block.
            assert_eq!(*new_ptr.as_ptr(), 0xAA);
        }
    }

    #[test]
    fn poison_pattern_observable_on_bump_arena() {
        use crate::layout::BumpArena;
        // BumpArena's deallocate is a no-op — the bytes it issued aren't
        // touched after our poison memset, so we can read them directly.
        let p: PoisonOnFree<BumpArena<InlineBacked<256>>> =
            PoisonOnFree::with_pattern(BumpArena::new(InlineBacked::<256>::new()).unwrap(), 0xBB);
        let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
        let block = p.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0xAA, 16);
            p.deallocate(ptr, layout);
            // Bytes should now be 0xBB (poison ran before BumpArena's no-op dealloc).
            for i in 0..16 {
                assert_eq!(*ptr.as_ptr().add(i), 0xBB, "byte {i} not poisoned");
            }
        }
    }

    #[test]
    fn default_pattern_is_0xde() {
        let p = PoisonOnFree::new(InlineBacked::<64>::new());
        assert_eq!(p.pattern(), 0xDE);
    }

    #[test]
    fn explicit_pattern_set() {
        let p = PoisonOnFree::with_pattern(InlineBacked::<64>::new(), 0x42);
        assert_eq!(p.pattern(), 0x42);
    }

    /// `reset` must forward to the inner arena (not the trait-default `Err`), so
    /// a wrapped `BumpArena` stays resettable — regression guard mirroring the
    /// `ZeroizeOnFree` test.
    #[test]
    fn reset_forwards_to_inner_arena() {
        use crate::layout::BumpArena;
        let mut p: PoisonOnFree<BumpArena<InlineBacked<256>>> =
            PoisonOnFree::new(BumpArena::new(InlineBacked::<256>::new()).unwrap());
        let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
        let _ = p.allocate(layout).unwrap();
        assert!(p.inner().allocated() > 0);
        assert!(p.reset().is_ok(), "wrapped arena must be resettable");
        assert_eq!(p.inner().allocated(), 0);
    }
}
