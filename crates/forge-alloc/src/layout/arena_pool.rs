//! `ArenaPool<B, F>` — recycle reset [`BumpArena`]s across a per-commit /
//! per-branch workload.
//!
//! A bump arena's natural lifecycle is "fill, then throw away". For an
//! [`MmapBacked`](crate::MmapBacked) arena, throwing it away means `munmap`,
//! and minting the next one means `mmap` plus a demand-zero re-fault storm on
//! first touch. On a hot per-commit path that fixed cost dominates. An
//! `ArenaPool` breaks the cycle: hand arenas out with [`checkout`], return them
//! with [`give_back`] (which [`reset`](BumpArena::reset)s in O(1) and retains
//! them up to a cap), and in steady state the same mappings are reused with
//! **zero** `munmap` / `mmap` / re-fault.
//!
//! When idle arenas should not hold physical RAM (a quiet period between
//! bursts), [`release_idle`](ArenaPool::release_idle) drops their pages via
//! `madvise(DONTNEED)` / `MEM_RESET` while keeping the virtual reservation warm
//! — bounding resident memory without leaving the pool. (That method requires
//! an [`OsBacked`] backing, i.e. an mmap-family one.)
//!
//! ```
//! use forge_alloc::{ArenaPool, InlineBacked};
//!
//! // Pool of up to 4 reset arenas, each over a fresh 64 KiB inline backing.
//! let mut pool = ArenaPool::new(4, || Ok(InlineBacked::<65536>::new()));
//! let arena = pool.checkout().unwrap();
//! // ... fill `arena` for one unit of work ...
//! pool.give_back(arena); // reset + retained for the next checkout
//! assert_eq!(pool.idle_count(), 1);
//! ```

use alloc::vec::Vec;

use forge_alloc_core::{AllocError, FixedRange, OsBacked};

use super::bump::BumpArena;

/// A pool of recyclable [`BumpArena`]s. See the [module docs](self).
///
/// `B` is the backing type; `F` mints a fresh backing on demand (e.g.
/// `|| MmapBacked::new(32 * 1024)`). Arenas are wrapped in [`BumpArena`]
/// internally.
pub struct ArenaPool<B: FixedRange, F> {
    factory: F,
    /// Reset, ready-to-reuse arenas. Invariant: every arena here has cursor 0
    /// (`give_back` resets before pushing).
    idle: Vec<BumpArena<B>>,
    /// Maximum number of idle arenas to retain; extras are dropped on
    /// `give_back` (the rare over-cap overflow).
    cap: usize,
}

impl<B, F> ArenaPool<B, F>
where
    B: FixedRange,
    F: FnMut() -> Result<B, AllocError>,
{
    /// Create an empty pool that retains up to `cap` idle arenas and mints new
    /// backings with `factory`.
    #[inline]
    pub fn new(cap: usize, factory: F) -> Self {
        Self {
            factory,
            idle: Vec::new(),
            cap,
        }
    }

    /// Check out an arena: reuse a reset idle one if available, otherwise mint a
    /// fresh backing via the factory and wrap it. The returned arena always
    /// starts empty (cursor 0).
    #[inline]
    pub fn checkout(&mut self) -> Result<BumpArena<B>, AllocError> {
        match self.idle.pop() {
            // Idle arenas were reset on `give_back`, so they're ready as-is.
            Some(arena) => Ok(arena),
            None => BumpArena::new((self.factory)()?),
        }
    }

    /// Return an arena to the pool. It is [`reset`](BumpArena::reset) (O(1)) and
    /// retained for reuse if the pool is below its cap; otherwise it is dropped
    /// (releasing its backing). In steady state — checkouts and give-backs
    /// balanced under the cap — this performs no allocation syscalls.
    #[inline]
    pub fn give_back(&mut self, mut arena: BumpArena<B>) {
        arena.reset();
        if self.idle.len() < self.cap {
            self.idle.push(arena);
        }
        // else: over cap — drop `arena`, releasing its backing.
    }

    /// Pre-mint up to `n` idle arenas (clamped to the remaining cap), so the
    /// first `checkout`s don't pay backing construction. Returns the number
    /// actually added; stops early and returns `Err` if the factory fails.
    pub fn prewarm(&mut self, n: usize) -> Result<usize, AllocError> {
        let target = self.cap.min(self.idle.len().saturating_add(n));
        let mut added = 0;
        while self.idle.len() < target {
            let arena = BumpArena::new((self.factory)()?)?;
            self.idle.push(arena);
            added += 1;
        }
        Ok(added)
    }

    /// Number of reset arenas currently retained and ready for checkout.
    #[inline]
    pub fn idle_count(&self) -> usize {
        self.idle.len()
    }

    /// The retention cap (maximum idle arenas).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Drop all idle arenas, releasing their backings (e.g. `munmap`). Use to
    /// reclaim address space when the pool will be idle for a long time; the
    /// next checkout mints fresh.
    #[inline]
    pub fn clear(&mut self) {
        self.idle.clear();
    }
}

impl<B, F> ArenaPool<B, F>
where
    B: FixedRange + OsBacked,
    F: FnMut() -> Result<B, AllocError>,
{
    /// Release the physical pages backing every idle arena
    /// (`madvise(DONTNEED)` / `MEM_RESET`) while keeping their virtual
    /// reservations mapped for reuse. Bounds resident memory for arenas sitting
    /// idle without dropping or re-`mmap`ing them; the next checkout re-faults
    /// the pages it touches.
    pub fn release_idle(&self) {
        for arena in &self.idle {
            // Idle arenas are reset (cursor 0): no live allocation overlaps the
            // region, so releasing the whole region is sound.
            // SAFETY: `[base_ptr, base_ptr + region_size)` is the arena's entire
            // mapping; no live allocations exist (idle + reset). `ArenaPool` is
            // `!Sync` (inherited from `BumpArena`'s `UnsafeCell` cursor), so no
            // concurrent `commit` races the `UnsafeCell<committed>` read inside
            // `release_pages` on the Windows lazy-commit path. See
            // `OsBacked::release_pages`.
            unsafe { arena.release_pages(arena.base_ptr(), arena.region_size()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;
    use crate::Allocator;
    use forge_alloc_core::NonZeroLayout;

    fn inline_factory() -> Result<InlineBacked<4096>, AllocError> {
        Ok(InlineBacked::<4096>::new())
    }

    #[test]
    fn checkout_mints_when_empty_then_give_back_recycles() {
        let mut pool = ArenaPool::new(2, inline_factory);
        assert_eq!(pool.idle_count(), 0);

        let arena = pool.checkout().unwrap();
        let cap = arena.capacity();
        // Use it, then return it.
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let _ = arena.allocate(layout).unwrap();
        pool.give_back(arena);
        assert_eq!(pool.idle_count(), 1);

        // Next checkout reuses the same backing, reset to empty.
        let arena2 = pool.checkout().unwrap();
        assert_eq!(arena2.allocated(), 0, "recycled arena must be reset");
        assert_eq!(arena2.capacity(), cap, "same backing capacity");
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn give_back_over_cap_drops_the_extra() {
        let mut pool = ArenaPool::new(1, inline_factory);
        let a = pool.checkout().unwrap();
        let b = pool.checkout().unwrap();
        pool.give_back(a);
        assert_eq!(pool.idle_count(), 1);
        // Pool is at cap; returning `b` must not exceed it.
        pool.give_back(b);
        assert_eq!(
            pool.idle_count(),
            1,
            "over-cap give_back drops, never grows past cap"
        );
    }

    #[test]
    fn prewarm_mints_up_to_cap() {
        let mut pool = ArenaPool::new(3, inline_factory);
        let added = pool.prewarm(10).unwrap();
        assert_eq!(added, 3, "prewarm clamps to remaining cap");
        assert_eq!(pool.idle_count(), 3);
        // Already full: prewarm adds nothing.
        assert_eq!(pool.prewarm(5).unwrap(), 0);
    }

    #[test]
    fn cap_zero_never_retains() {
        let mut pool = ArenaPool::new(0, inline_factory);
        let a = pool.checkout().unwrap();
        pool.give_back(a);
        assert_eq!(pool.idle_count(), 0, "cap=0 pool retains nothing");
        assert_eq!(pool.prewarm(5).unwrap(), 0, "cap=0 prewarm adds nothing");
        // Still hands out fresh arenas fine.
        let b = pool.checkout().unwrap();
        assert_eq!(b.allocated(), 0);
    }

    #[test]
    fn factory_failure_propagates_cleanly() {
        use core::cell::Cell;
        // Factory that succeeds `ok` times then fails.
        let budget = Cell::new(1usize);
        let factory = || {
            if budget.get() == 0 {
                return Err(AllocError);
            }
            budget.set(budget.get() - 1);
            Ok(InlineBacked::<4096>::new())
        };
        let mut pool = ArenaPool::new(4, factory);

        // First checkout succeeds (budget 1 -> 0).
        let a = pool.checkout().unwrap();
        // Second mint fails; error propagates, pool state unchanged.
        assert!(pool.checkout().is_err());
        assert_eq!(pool.idle_count(), 0);
        pool.give_back(a);
        assert_eq!(pool.idle_count(), 1);
        // prewarm with no budget left adds nothing and surfaces the error.
        assert!(pool.prewarm(3).is_err());
    }

    #[test]
    fn clear_drops_idle() {
        let mut pool = ArenaPool::new(4, inline_factory);
        pool.prewarm(4).unwrap();
        assert_eq!(pool.idle_count(), 4);
        pool.clear();
        assert_eq!(pool.idle_count(), 0);
    }

    // release_idle needs an OsBacked backing → MmapBacked (std + unix/windows).
    #[cfg(all(feature = "std", any(unix, windows)))]
    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap")]
    fn release_idle_then_reuse_round_trips() {
        use crate::backing::MmapBacked;
        let mut pool = ArenaPool::new(2, || MmapBacked::new(64 * 1024));
        pool.prewarm(2).unwrap();

        // Release the physical pages of all idle arenas — must not crash, and
        // the arenas must stay in the pool (still reusable).
        pool.release_idle();
        assert_eq!(
            pool.idle_count(),
            2,
            "release_idle keeps arenas, only drops pages"
        );

        // Reuse: the mapping is still valid; a fresh allocation re-faults cleanly.
        let arena = pool.checkout().unwrap();
        let layout = NonZeroLayout::from_size_align(4096, 8).unwrap();
        let block = arena.allocate(layout).unwrap();
        unsafe {
            core::ptr::write_bytes(block.cast::<u8>().as_ptr(), 0xCD, 4096);
        }
        pool.give_back(arena);
    }
}
