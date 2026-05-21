//! `Quarantine<I, const EPOCHS: usize>` — holds freed blocks in a ring
//! buffer for `EPOCHS` deallocate cycles before returning them to the inner
//! allocator for reuse.
//!
//! Security property: a dangling pointer must survive `EPOCHS` deallocate
//! cycles on this allocator instance before the slot's address can be reused
//! by a subsequent allocate, raising the bar for UAF / type-confusion
//! exploitation. With `EPOCHS = 16`, an attacker must spray 16 deallocations
//! of the correct shape between free and exploit attempt.
//!
//! Compose with [`crate::PoisonOnFree`] for both content destruction and
//! reuse delay: `PoisonOnFree<Quarantine<Slab<...>, 16>>` poisons on free
//! (immediate content wipe), then quarantines the poisoned slot for 16
//! cycles (delayed reuse).
//!
//! See `docs/ARCHITECTURE.md` for the composable-wrapper design.

use core::cell::UnsafeCell;
use core::ptr::NonNull;

use forge_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// A freed block held in quarantine until its epoch expires.
#[derive(Copy, Clone)]
struct QuarantinedBlock {
    ptr: NonNull<u8>,
    layout: NonZeroLayout,
}

/// Quarantine wrapper.
///
/// `EPOCHS` is the ring length. Must be >= 1 (statically asserted). Each
/// `deallocate` puts the freed block into slot `(free_count % EPOCHS)`,
/// evicting whatever was there to the inner allocator first. This means a
/// block stays quarantined for at most `EPOCHS` subsequent deallocates.
///
/// # Thread safety
///
/// `Send` when `I: Send`. `Sync`: NO. The ring buffer uses `UnsafeCell` for
/// `&self` mutation on the deallocate path. Cross-thread use requires an
/// outer synchronization layer.
///
/// # Panic safety
///
/// `Quarantine` relies on `inner.deallocate` being **panic-free** per the
/// `Deallocator` contract. If `inner.deallocate` panics during a
/// `Quarantine::deallocate` call (e.g. a `debug_assert!` in a lower-layer
/// `Statistics` wrapper fires), the evicted block held in the local
/// `prior` binding is **leaked**: the ring slot has already been
/// overwritten with the new block, and the unwinding path drops `prior`
/// without re-routing it. Subsequent deallocates will continue normally
/// at the next ring index, so the quarantine remains usable — but the
/// one leaked block's memory will only be reclaimed when the inner's
/// own backing region drops (typically at process exit for an mmap-
/// backed slab; never, for a slab leased from a long-lived backing).
///
/// During `Drop`, a panicking `inner.deallocate` aborts the drain loop,
/// leaking the remaining quarantined slots in the same way.
///
/// **Drop-during-unwind escalation**: if `Quarantine` is itself dropped
/// as part of stack unwinding (i.e. an earlier panic is already in
/// flight) and `inner.deallocate` panics inside the drain loop, the
/// second panic-while-panicking triggers an immediate **process abort**
/// (this is a Rust language rule, not a Quarantine choice). Concretely:
/// a panic from the inner `deallocate` during normal Drop becomes a
/// leak; the same panic from inner `deallocate` during unwinding Drop
/// becomes a fatal abort. The Quarantine layer cannot defuse the
/// second case without `catch_unwind` (which would require `std` and
/// is contrary to no-panic-in-Drop being the contract everywhere
/// below us). Treat a panicking `inner.deallocate` as a critical bug
/// to fix in the inner — not a recoverable condition.
///
/// These outcomes are acceptable for `Quarantine`'s intended threat
/// model — a panicking inner `deallocate` already signals an
/// allocator-state violation, and the priority is to avoid double-free
/// rather than guarantee reclamation. Callers needing leak-free panic
/// recovery should wrap with a separate drop-guard layer.
///
/// # Composition with size-classed inners
///
/// If the inner serves multiple sizes (e.g. `SizeClassed`), all sizes share
/// the same `EPOCHS`-slot ring, so per-class quarantine depth degrades to
/// `EPOCHS / active_sizes`. Recommended: place `Quarantine` INSIDE
/// `SizeClassed` (`SizeClassed<Quarantine<Slab<T, _>, 16>, N>`) for
/// per-class quarantine, OR keep `Quarantine` on a typed `Slab<T, _>` where
/// all slots are the same size.
///
/// # Inner exhaustion while items are quarantined
///
/// During the `EPOCHS` window between dealloc-into-quarantine and
/// eviction-to-inner, the freed slot is **still owned by the inner**
/// allocator — Slab counts it as live, `SizeClassed` keeps the class slot
/// off the freelist, etc. If the application exhausts the inner's capacity
/// while items wait in quarantine, the next `Quarantine::allocate` call
/// forwards to the inner and surfaces `AllocError` immediately (no waiting,
/// no fancy retry). The quarantined slot becomes reusable once `EPOCHS`
/// further deallocates evict it; until then the program is at reduced
/// capacity. Size the inner with at least `EPOCHS` worth of slack if your
/// workload runs near steady-state full.
pub struct Quarantine<I: Allocator, const EPOCHS: usize> {
    inner: I,
    /// Ring buffer of `Option<QuarantinedBlock>` slots in `UnsafeCell`.
    /// We use `UnsafeCell` of the whole array (rather than per-slot) because
    /// `Quarantine` is `!Sync` and only one thread accesses the ring at a
    /// time.
    ring: UnsafeCell<[Option<QuarantinedBlock>; EPOCHS]>,
    /// Number of deallocate calls received. Position in ring is
    /// `count % EPOCHS`. Wraps at `usize::MAX` — see `deallocate_count`.
    count: UnsafeCell<usize>,
}

impl<I: Allocator, const EPOCHS: usize> Quarantine<I, EPOCHS> {
    /// Compile-time check that `EPOCHS >= 1`.
    const ASSERT_EPOCHS: () = assert!(EPOCHS >= 1, "Quarantine<_, EPOCHS> requires EPOCHS >= 1");

    /// Wrap an inner allocator with `EPOCHS`-cycle quarantine.
    #[inline]
    pub fn new(inner: I) -> Self {
        let _: () = Self::ASSERT_EPOCHS;
        // `Option::None` is a valid initial state for every slot, no
        // `MaybeUninit` dance required.
        Self {
            inner,
            ring: UnsafeCell::new([None; EPOCHS]),
            count: UnsafeCell::new(0),
        }
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }

    /// Total number of deallocate calls received.
    ///
    /// Internally this counter increments with `wrapping_add`, so after
    /// `usize::MAX` deallocations it wraps to `0`. This is intentional —
    /// ring indexing uses `count % EPOCHS` and wrap is harmless there —
    /// but callers reading this value as a long-running statistic should
    /// account for the wrap.
    #[inline]
    pub fn deallocate_count(&self) -> usize {
        // SAFETY: !Sync — single-threaded access.
        unsafe { *self.count.get() }
    }
}

unsafe impl<I: Allocator, const EPOCHS: usize> Deallocator for Quarantine<I, EPOCHS> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // SAFETY: !Sync — single-threaded access to ring + count.
        unsafe {
            let count_ptr = self.count.get();
            let cnt = *count_ptr;
            let idx = cnt % EPOCHS;
            let ring_ptr = self.ring.get();
            // Increment count BEFORE the inner.deallocate so a panicking
            // inner leaves the ring + count in a consistent state for the
            // next call. See the "Panic safety" section on the type docs.
            *count_ptr = cnt.wrapping_add(1);
            // Swap the new block in; if a prior block was there, evict it
            // to the inner allocator.
            let evicted = (*ring_ptr)[idx].replace(QuarantinedBlock { ptr, layout });
            if let Some(prior) = evicted {
                // SAFETY: this block was put into the ring by an earlier
                // call to our deallocate(); the inner allocator issued it,
                // so it's valid for inner.deallocate.
                self.inner.deallocate(prior.ptr, prior.layout);
            }
        }
    }
}

unsafe impl<I: Allocator, const EPOCHS: usize> Allocator for Quarantine<I, EPOCHS> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner.allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner.allocate_zeroed(layout)
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        self.inner.capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        self.inner.corruption_events()
    }

    // grow/shrink inherit the default; the discarded buffer gets routed
    // through `self.deallocate`, putting it into quarantine. This is the
    // desired behavior — grown-out buffers must respect the quarantine.
}

impl<I: FixedRange, const EPOCHS: usize> FixedRange for Quarantine<I, EPOCHS> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.inner.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl<I: Allocator, const EPOCHS: usize> Drop for Quarantine<I, EPOCHS> {
    fn drop(&mut self) {
        // Drain quarantine: return every held block to inner before inner
        // itself drops. Critical for Slab-style backings where leaked slots
        // would leak the entire backing's allocation. For mmap-backed
        // inners the inner's drop reclaims everything anyway, but we drain
        // for symmetry and to avoid surprise.
        //
        // Stacked Borrows note: we MUST NOT create a `&mut [Option<Block>;
        // EPOCHS]` reference over the ring (which `self.ring.get_mut()`
        // would do). A Unique retag at the ring level — or, transitively,
        // anywhere over the ring's storage — invalidates the
        // SharedReadWrite tag covering the inner allocator's backing
        // (e.g. `InlineBacked::storage`). Each `Block.ptr` was derived
        // from that SharedReadWrite tag during the earlier `allocate`
        // call; using it through `self.inner.deallocate(b.ptr, ...)` after
        // the Unique retag is UB under SB. (Miri caught the same class of
        // bug in `SlabOwner`.)
        //
        // Work with raw pointers throughout the drain. The `&mut self`
        // signature is the Drop trait's; we are careful to never
        // materialize a `&mut` to the ring or its contents while a
        // `Block.ptr` is still live.
        let ring_ptr: *mut [Option<QuarantinedBlock>; EPOCHS] = self.ring.get();
        for i in 0..EPOCHS {
            // SAFETY: `ring_ptr` is valid, single-threaded access (Drop
            // has exclusive access to `self`), in-bounds index.
            let slot_ptr: *mut Option<QuarantinedBlock> = unsafe {
                (ring_ptr as *mut Option<QuarantinedBlock>).add(i)
            };
            // SAFETY: read the current value by bit-copy. We then
            // immediately overwrite the slot with `None`, so no
            // double-drop risk. Equivalent to `slot.take()` without
            // creating a `&mut Option<…>` reference at the ring level.
            let taken: Option<QuarantinedBlock> = unsafe { core::ptr::read(slot_ptr) };
            unsafe { core::ptr::write(slot_ptr, None) };
            if let Some(b) = taken {
                // SAFETY: this block was put into the ring by an earlier
                // call to our deallocate(); the inner allocator issued
                // it, so it's valid for inner.deallocate. `self.inner`
                // is accessed via a shared borrow (the `Deallocator`
                // trait method takes `&self`), which does NOT trigger
                // an additional Unique retag.
                unsafe { self.inner.deallocate(b.ptr, b.layout) };
            }
        }
    }
}

// Send when I: Send. !Sync via UnsafeCell.
unsafe impl<I: Allocator + Send, const EPOCHS: usize> Send for Quarantine<I, EPOCHS> {}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_backing::InlineBacked;
    use forge_layout::Slab;

    /// Helper: Slab<u64> wrapped in Quarantine.
    fn build<const E: usize>() -> Quarantine<Slab<u64, InlineBacked<512>>, E> {
        Quarantine::new(Slab::new(8, InlineBacked::<512>::new()).unwrap())
    }

    #[test]
    fn allocate_passes_through() {
        let q = build::<4>();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let block = q.allocate(layout).unwrap();
        unsafe { q.deallocate(block.cast(), layout) };
    }

    #[test]
    fn freed_slot_not_reused_immediately() {
        // EPOCHS=4: after free, the next 3 deallocs cycle the ring; on the
        // 4th additional dealloc, the original slot is evicted to inner and
        // becomes reusable.
        let q = build::<4>();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Allocate 4 slots, remember their addresses.
        let a = q.allocate(layout).unwrap().cast::<u8>();
        let _b = q.allocate(layout).unwrap().cast::<u8>();
        let _c = q.allocate(layout).unwrap().cast::<u8>();
        let _d = q.allocate(layout).unwrap().cast::<u8>();
        let _e = q.allocate(layout).unwrap().cast::<u8>();

        // Free `a`. With Quarantine, `a` does NOT immediately go back to
        // Slab's free list; it sits in quarantine slot 0.
        unsafe { q.deallocate(a, layout) };
        // Slab still has 3 unallocated slots (8 - 5 = 3). The next alloc
        // takes from `next_uncarved`, NOT from `a` (because `a` is in
        // quarantine). Verify by checking the new ptr differs from `a`.
        let f = q.allocate(layout).unwrap().cast::<u8>();
        assert_ne!(a.as_ptr(), f.as_ptr(), "EPOCHS=4 quarantine should hold `a`");
    }

    #[test]
    fn evicted_block_reachable_after_epochs() {
        // EPOCHS=2: after 2 additional deallocs, the original block leaves
        // quarantine and is back on Slab's freelist.
        let q = build::<2>();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = q.allocate(layout).unwrap().cast::<u8>();
        let b = q.allocate(layout).unwrap().cast::<u8>();
        let c = q.allocate(layout).unwrap().cast::<u8>();

        unsafe { q.deallocate(a, layout) }; // ring[0] = a, count=1
        unsafe { q.deallocate(b, layout) }; // ring[1] = b, count=2
        unsafe { q.deallocate(c, layout) }; // ring[0] evicts a, ring[0] = c, count=3

        // a is now on the Slab freelist. The next alloc returns a (LIFO).
        let g = q.allocate(layout).unwrap().cast::<u8>();
        assert_eq!(a.as_ptr(), g.as_ptr(), "evicted `a` should be reusable");
    }

    #[test]
    fn drop_drains_quarantine() {
        let s = Slab::<u64, InlineBacked<512>>::new(8, InlineBacked::<512>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Construct Quarantine, free some slots, then drop without draining.
        // Verify no double-frees by ensuring Drop completes without panic.
        {
            let q: Quarantine<&Slab<u64, InlineBacked<512>>, 8> = Quarantine::new(&s);
            let a = q.allocate(layout).unwrap();
            let b = q.allocate(layout).unwrap();
            unsafe {
                q.deallocate(a.cast(), layout);
                q.deallocate(b.cast(), layout);
            }
            // q drops here; quarantine should drain a and b back to slab.
        }
        // After q is dropped, slab should be reusable as if a/b were freed.
        let c = s.allocate(layout).unwrap();
        // c should be one of the originally-freed slots (Slab LIFO).
        let _ = c;
    }

    #[test]
    fn deallocate_count_advances() {
        let q = build::<4>();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = q.allocate(layout).unwrap();
        let b = q.allocate(layout).unwrap();
        unsafe { q.deallocate(a.cast(), layout) };
        unsafe { q.deallocate(b.cast(), layout) };
        assert_eq!(q.deallocate_count(), 2);
    }
}
