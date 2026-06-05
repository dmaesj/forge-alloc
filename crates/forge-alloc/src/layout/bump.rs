//! `BumpArena<B>` — single-threaded bump arena over a [`FixedRange`] backing.
//!
//! Allocation is O(1) — align the cursor, bounds-check, advance. Deallocation
//! is a no-op; reclaim happens via [`reset`](BumpArena::reset). To use with
//! the standard collection types (`Vec<T, A>`, etc.), allocate via the arena
//! directly and wrap with `from_raw_in` using [`BumpDeallocator<'_>`] as the
//! deallocation token.
//!
//! ```
//! use forge_alloc::InlineBacked;
//! use forge_alloc::{Allocator, NonZeroLayout};
//! use forge_alloc::BumpArena;
//!
//! let mut arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
//! let layout = NonZeroLayout::from_size_align(128, 16).unwrap();
//! let _block = arena.allocate(layout).unwrap();
//! assert_eq!(arena.allocated(), 128);
//! arena.reset();
//! assert_eq!(arena.allocated(), 0);
//! ```
//!
//! See `docs/ARCHITECTURE.md` for the bump-arena design.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::ptr::NonNull;

use forge_alloc_core::{
    AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout, OsBacked, ProtectFlags,
};

/// Bump arena over any [`FixedRange`] backing.
///
/// The arena uses the entire address range exposed by the backing. The
/// backing's own `allocate` is never called — `BumpArena` does all
/// suballocation directly. When the arena drops, the backing drops, and the
/// memory is released by whatever path the backing uses (e.g. `MmapBacked`'s
/// `munmap`).
///
/// # Thread safety
///
/// `Send`: yes if `B: Send`. `Sync`: NO — concurrent `&self` allocators would
/// race on the cursor. Use [`SharedBumpArena`](crate::layout::SharedBumpArena) for
/// cross-thread access.
pub struct BumpArena<B: FixedRange> {
    backing: B,
    /// Cached byte size of the backing range, captured at construction.
    /// We do NOT cache `base` or `end` here — backings whose `base()` is
    /// structure-relative (e.g. `InlineBacked<N>` returns `&self.storage`)
    /// produce a different address before and after the backing has been
    /// moved into `Self`. A pointer captured at construction would point
    /// at the backing's pre-move location for the rest of the arena's
    /// life, silently corrupting every subsequent `allocate`. We re-
    /// query `backing.base()` at each `allocate` call instead; the
    /// happy-path cost is one extra indirect load.
    capacity: usize,
    /// Offset from `backing.base()`. Interior mutability for `&self`
    /// allocation; `!Sync` (via `UnsafeCell`) prevents concurrent racing.
    cursor: UnsafeCell<usize>,
}

impl<B: FixedRange> BumpArena<B> {
    /// Construct a bump arena that owns `backing` and bumps through its
    /// entire address range.
    ///
    /// Returns an error if the backing reports a zero-byte range or if the
    /// backing's `[base, base+size)` range would wrap past `usize::MAX`
    /// (impossible on real 64-bit hardware but representable on small
    /// `no_std` targets).
    pub fn new(backing: B) -> Result<Self, AllocError> {
        let base = backing.base();
        let size = backing.size();
        if size == 0 {
            return Err(AllocError);
        }
        // Reject backings whose [base, base+size) range wraps past
        // `usize::MAX`. Even though we don't cache `end` anymore, every
        // allocate path still derives `aligned_off + size <= capacity`
        // from this invariant; rejecting at construction surfaces the
        // misconfigured backing once instead of on every allocate.
        // On 64-bit this branch is unreachable in practice; on 16-/32-bit
        // no_std it can fire.
        let base_addr = base.as_ptr() as usize;
        let end_addr = base_addr.checked_add(size).ok_or(AllocError)?;
        // `end_addr == 0` would mean `base + size == 2^N exactly`, i.e. the
        // mapping covers the top of the address space — also rejected, since
        // we'd need a non-null `end` sentinel.
        if end_addr == 0 {
            return Err(AllocError);
        }
        Ok(Self {
            backing,
            capacity: size,
            cursor: UnsafeCell::new(0),
        })
    }

    /// Bytes currently allocated from this arena.
    #[inline]
    pub fn allocated(&self) -> usize {
        // SAFETY: !Sync — no concurrent access to cursor.
        unsafe { *self.cursor.get() }
    }

    /// Total bytes available in this arena.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes remaining for allocation.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity() - self.allocated()
    }

    /// Borrow the underlying backing.
    #[inline]
    pub fn backing(&self) -> &B {
        &self.backing
    }

    /// Mint a zero-sized [`BumpDeallocator`] tied to this arena's lifetime.
    ///
    /// The deallocator's `'a` lifetime is the arena's borrow, so the borrow
    /// checker prevents the arena from being dropped or reset while any
    /// `Box<T, BumpDeallocator<'_>>` (constructed via `Box::from_raw_in`) is
    /// outstanding.
    #[inline]
    pub fn deallocator(&self) -> BumpDeallocator<'_> {
        BumpDeallocator(PhantomData)
    }
}

impl<B: FixedRange> BumpArena<B> {
    /// Reset the cursor to 0, reclaiming all memory in O(1).
    ///
    /// Requires `&mut self`, which the borrow checker enforces: any
    /// outstanding `Box<T, BumpDeallocator<'_>>` (whose `'_` is `&self`)
    /// blocks `&mut self` access until dropped. Raw `allocate` callers must
    /// observe the discipline themselves.
    ///
    /// # Safety
    ///
    /// All pointers previously issued by this arena become invalid after
    /// `reset`. Reading or writing through them is undefined behavior.
    #[inline]
    pub fn reset(&mut self) {
        // &mut self gives exclusive access.
        *self.cursor.get_mut() = 0;
    }

    /// Rewind the cursor to a previously-recorded mark (an [`allocated`]
    /// value), reclaiming everything allocated since. Like [`reset`] but to an
    /// arbitrary earlier point; the engine behind [`scope`].
    ///
    /// [`allocated`]: BumpArena::allocated
    /// [`reset`]: BumpArena::reset
    /// [`scope`]: BumpArena::scope
    ///
    /// # Safety contract (caller-upheld, mirrors [`reset`])
    ///
    /// All pointers issued by this arena *after* `mark` was recorded become
    /// invalid. The `&mut self` borrow is what makes this enforceable; the
    /// [`scope`](BumpArena::scope) guard wraps it in a safe RAII API.
    #[inline]
    pub fn rewind_to(&mut self, mark: usize) {
        debug_assert!(
            mark <= *self.cursor.get_mut(),
            "rewind_to mark is ahead of cursor"
        );
        // &mut self gives exclusive access.
        *self.cursor.get_mut() = mark;
    }

    /// Typed bump allocation: reserve aligned, **uninitialized** storage for one
    /// `T` and return a pointer to it. Because `size_of::<T>()` and
    /// `align_of::<T>()` are compile-time constants, the alignment-rounding mask
    /// and the bounds arithmetic fold to a tight branch — the typed analogue of
    /// [`allocate`](Allocator::allocate) for the common "allocate one value"
    /// case, without constructing a runtime [`NonZeroLayout`].
    ///
    /// For a zero-sized `T` this returns a well-aligned dangling pointer and
    /// consumes no space (a successful no-op, where `allocate` rejects a
    /// zero-size layout). The returned pointer is **uninitialized**; write a
    /// `T` before reading.
    #[inline]
    pub fn alloc_uninit<T>(&self) -> Result<NonNull<T>, AllocError> {
        let size = core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        // ZST: a dangling but aligned pointer is valid for a zero-sized read or
        // write, and consumes no space. Folds away entirely for non-ZST `T`.
        if size == 0 {
            return Ok(NonNull::dangling());
        }
        // Re-query the live backing base (structure-relative backings move) —
        // identical to `allocate`.
        let base = self.backing.base();
        let base_addr = base.as_ptr() as usize;
        // SAFETY: !Sync — no concurrent cursor access (same contract as
        // `allocate`; `reset`/`rewind_to`/`scope` take `&mut self`).
        unsafe {
            let cursor_ptr = self.cursor.get();
            let cur = *cursor_ptr;
            // `align` is a compile-time power of two, so `align - 1` and the
            // mask are constants the optimizer folds.
            let raw = base_addr.checked_add(cur).ok_or(AllocError)?;
            let aligned = raw.checked_add(align - 1).ok_or(AllocError)? & !(align - 1);
            let aligned_off = aligned - base_addr;
            let end_off = aligned_off.checked_add(size).ok_or(AllocError)?;
            if end_off > self.capacity() {
                return Err(AllocError);
            }
            // Commit freshly-crossed pages on lazy backings before publishing
            // the cursor (no-op for eager/inline backings) — as in `allocate`.
            self.backing.commit(aligned_off, size)?;
            *cursor_ptr = end_off;
            let p = base.as_ptr().add(aligned_off) as *mut T;
            Ok(NonNull::new_unchecked(p))
        }
    }

    /// Allocate space for one `T` and move `value` into it, returning a pointer
    /// to the initialized `T`. The typed-and-initialized companion of
    /// [`alloc_uninit`](Self::alloc_uninit).
    #[inline]
    pub fn alloc<T>(&self, value: T) -> Result<NonNull<T>, AllocError> {
        let p = self.alloc_uninit::<T>()?;
        // SAFETY: `p` is fresh, aligned, exclusive storage for one `T`.
        unsafe { p.as_ptr().write(value) };
        Ok(p)
    }

    /// Copy a slice into the arena, returning a pointer to the copy. `T: Copy`
    /// so no destructor obligations are transferred. An empty or zero-sized-`T`
    /// slice consumes no space and yields a dangling-but-aligned slice pointer
    /// of the same length.
    #[inline]
    pub fn alloc_slice_copy<T: Copy>(&self, src: &[T]) -> Result<NonNull<[T]>, AllocError> {
        let n = src.len();
        if n == 0 || core::mem::size_of::<T>() == 0 {
            // A slice of `n` ZSTs (or zero elements) needs no storage; a
            // dangling, aligned pointer with length `n` is a valid `&[T]`/`&mut`.
            return Ok(NonNull::slice_from_raw_parts(NonNull::<T>::dangling(), n));
        }
        // `array` cannot overflow here: `src` already exists in memory, so
        // `n * size_of::<T>()` fits `isize`.
        let layout = NonZeroLayout::array::<T>(n).ok_or(AllocError)?;
        let dst = self.allocate(layout)?.cast::<T>();
        // SAFETY: `dst` is fresh storage for `n` `T`s, disjoint from `src`;
        // `T: Copy` so a bytewise copy is a valid clone.
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), n) };
        Ok(NonNull::slice_from_raw_parts(dst, n))
    }

    /// Copy a string slice into the arena, returning a pointer to the copy.
    #[inline]
    pub fn alloc_str(&self, s: &str) -> Result<NonNull<str>, AllocError> {
        let bytes = self.alloc_slice_copy(s.as_bytes())?;
        // SAFETY: the bytes were copied verbatim from a valid `&str`, so they
        // remain valid UTF-8; `*mut [u8]` and `*mut str` share layout.
        let p = bytes.as_ptr() as *mut str;
        Ok(unsafe { NonNull::new_unchecked(p) })
    }

    /// Open a scratch [`Scope`]. Allocations made through the returned guard are
    /// reclaimed when it is dropped, rewinding the cursor to where it was — a
    /// **nestable, panic-safe** checkpoint.
    ///
    /// Soundness rests on ordinary borrow-checking, not an `unsafe` lifetime
    /// trick:
    /// - The `&mut self` borrow makes the arena unusable directly for the
    ///   scope's lifetime, so no *outer* allocation can land in the region the
    ///   scope will reclaim.
    /// - Every reference the scope hands out borrows the guard (`&self`), so the
    ///   borrow checker forbids it from outliving the guard — and the guard's
    ///   `Drop` is what rewinds. A scope allocation therefore cannot dangle past
    ///   the rewind.
    /// - `Drop` runs on a panic unwind too, so a panicking scope body still
    ///   rewinds (no torn cursor).
    ///
    /// ```
    /// use forge_alloc::{BumpArena, InlineBacked};
    /// let mut arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
    /// let before = arena.allocated();
    /// {
    ///     let scope = arena.scope();
    ///     let _scratch = scope.alloc_uninit::<[u8; 64]>().unwrap();
    ///     assert!(scope.arena_allocated() > before);
    /// } // scope dropped: cursor rewound
    /// assert_eq!(arena.allocated(), before);
    /// ```
    ///
    /// A scope allocation cannot escape the scope:
    /// ```compile_fail
    /// use forge_alloc::{BumpArena, InlineBacked};
    /// let mut arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
    /// let escaped;
    /// {
    ///     let scope = arena.scope();
    ///     escaped = scope.alloc_uninit::<u32>().unwrap(); // borrows `scope`
    /// } // `scope` dropped here
    /// let _use = escaped; // ERROR: `scope` does not live long enough
    /// ```
    #[inline]
    pub fn scope(&mut self) -> Scope<'_, B> {
        let mark = self.allocated();
        Scope { arena: self, mark }
    }
}

unsafe impl<B: FixedRange> Deallocator for BumpArena<B> {
    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // No-op. Reclaim is via reset(&mut self).
    }
}

unsafe impl<B: FixedRange> Allocator for BumpArena<B> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let align = layout.align().get();
        let size = layout.size().get();
        // Re-query the backing's base at each allocate so structure-
        // relative backings (e.g. `InlineBacked`) keep working after the
        // arena has been moved.
        let base = self.backing.base();
        let base_addr = base.as_ptr() as usize;

        // SAFETY: !Sync — no concurrent access to cursor. We hold the only
        // path to mutating it (other than `reset(&mut self)`).
        unsafe {
            let cursor_ptr = self.cursor.get();
            let cur = *cursor_ptr;
            // Round up the absolute address to the requested alignment.
            let raw = base_addr.checked_add(cur).ok_or(AllocError)?;
            let aligned = raw.checked_add(align - 1).ok_or(AllocError)? & !(align - 1);
            // `aligned >= raw >= base_addr` because masking only zeroes low
            // bits; the subtraction never wraps.
            let aligned_off = aligned - base_addr;
            let end_off = aligned_off.checked_add(size).ok_or(AllocError)?;
            if end_off > self.capacity() {
                return Err(AllocError);
            }
            // Ensure the backing has the block's pages committed before we
            // hand them out. No-op for already-writable backings
            // (InlineBacked, eager MmapBacked, Unix mmap); on a lazy_commit
            // MmapBacked this commits the freshly-crossed pages and can fail
            // if the OS declines (Windows commit limit). Commit BEFORE
            // publishing the cursor so a failure leaves the arena unchanged
            // and surfaces as a clean AllocError rather than a fault on
            // first write.
            self.backing.commit(aligned_off, size)?;
            *cursor_ptr = end_off;
            // SAFETY: aligned_off + size <= capacity, so the resulting ptr
            // lies within [base, end). base is non-null per FixedRange's
            // contract; the offset preserves non-null.
            let p = base.as_ptr().add(aligned_off);
            Ok(NonNull::slice_from_raw_parts(
                NonNull::new_unchecked(p),
                size,
            ))
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.capacity())
    }

    /// In-place grow when `ptr` is the most-recent allocation.
    ///
    /// If the block being grown ends exactly at the cursor (i.e. it was the last
    /// thing allocated), the grow is just a cursor advance — **no copy**, the
    /// same pointer is returned covering the larger size. This is the common
    /// case for building a `Vec`/`String` in an arena. Otherwise it falls back
    /// to allocate-new + copy (the old block is reclaimed at the next `reset`,
    /// as for any bump allocation).
    ///
    /// # Safety
    ///
    /// Same as [`Allocator::grow`]: `ptr` is a live allocation of `old` from
    /// this arena, `new.size() >= old.size()`, and `old.align() == new.align()`.
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: NonZeroLayout,
        new: NonZeroLayout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new.size() >= old.size());
        debug_assert_eq!(old.align(), new.align());
        let base = self.backing.base();
        let base_addr = base.as_ptr() as usize;
        let off = (ptr.as_ptr() as usize).wrapping_sub(base_addr);
        // SAFETY: !Sync — exclusive cursor access (same contract as `allocate`).
        unsafe {
            let cursor_ptr = self.cursor.get();
            let cur = *cursor_ptr;
            // Fast path: `ptr`'s block ends exactly at the cursor → it is the
            // most-recent allocation, so grow by advancing the cursor in place.
            if off.checked_add(old.size().get()) == Some(cur) {
                let new_end = off.checked_add(new.size().get()).ok_or(AllocError)?;
                if new_end <= self.capacity() {
                    // Commit the (possibly-)newly-crossed tail pages before
                    // publishing the cursor; idempotent over the old prefix.
                    self.backing.commit(off, new.size().get())?;
                    *cursor_ptr = new_end;
                    return Ok(NonNull::slice_from_raw_parts(ptr, new.size().get()));
                }
                // Doesn't fit in place — fall through to relocate.
            }
        }
        // Fallback: fresh allocation + copy (old block leaked until `reset`).
        let dst = self.allocate(new)?;
        // SAFETY: caller's contract gives a valid `ptr` of `old.size()` bytes;
        // `dst` is fresh, ≥ `old.size()`, and disjoint.
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                dst.cast::<u8>().as_ptr(),
                old.size().get(),
            );
        }
        Ok(dst)
    }

    /// Reset the arena via the Allocator trait.
    ///
    /// Returns `Ok(())` and clears the cursor.
    #[inline]
    fn reset(&mut self) -> Result<(), AllocError> {
        BumpArena::reset(self);
        Ok(())
    }
}

impl<B: FixedRange> FixedRange for BumpArena<B> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        // Forward to the live backing rather than returning a cached
        // pointer — structure-relative backings change address on move.
        self.backing.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.capacity()
    }
}

// When the backing is OS-managed, the arena is too: it occupies the backing's
// entire mapping, so `base_ptr` / `region_size` / `release_pages` / `protect`
// forward straight through. The motivating use case is an arena *pool* on a
// per-commit / per-branch workload: instead of dropping (and `munmap`-ing) an
// arena on pool overflow, the pool can `reset()` it and `release_pages()` the
// whole region — returning the physical pages to the OS (`madvise(DONTNEED)` /
// `MEM_RESET`) while keeping the virtual reservation warm for reuse. That
// removes both the `munmap` syscall and the demand-zero re-fault storm a fresh
// `mmap` would incur on the next commit, without changing pool semantics.
//
// SAFETY: `base_ptr` (stable, non-null) and `region_size` (accurate page-rounded
// length) are discharged by the `B: OsBacked` backing; the arena caches nothing
// and delegates every call to it. For `release_pages` / `protect`, the in-region,
// page-alignment, and no-live-overlap requirements are the caller's documented
// `unsafe` precondition (see each method's `# Safety`); like the crate's other
// OsBacked wrappers, neither the arena nor the backing re-validates them.
unsafe impl<B: FixedRange + OsBacked> OsBacked for BumpArena<B> {
    #[inline]
    fn base_ptr(&self) -> NonNull<u8> {
        self.backing.base_ptr()
    }

    #[inline]
    fn region_size(&self) -> usize {
        self.backing.region_size()
    }

    #[inline]
    unsafe fn release_pages(&self, ptr: NonNull<u8>, size: usize) {
        // SAFETY: forwarded; caller guarantees an in-region range with no live
        // allocations (after `reset()` the arena has none — the pool-overflow
        // path above). The backing clamps the reset to the committed prefix on
        // Windows, so a full-region release of a partially-committed lazy mapping
        // resets only the committed pages.
        unsafe { self.backing.release_pages(ptr, size) }
    }

    #[inline]
    unsafe fn protect(&self, ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
        // SAFETY: forwarded; caller guarantees an in-region, page-aligned range.
        unsafe { self.backing.protect(ptr, size, flags) }
    }
}

// Send when B: Send. The `NonNull<u8>` fields are `!Send` by default but the
// memory they point to is owned by `backing`, which we move along with the
// arena. `UnsafeCell<usize>` is `Send` (cursor is just an integer).
//
// `!Sync` is auto-derived via `UnsafeCell`, which is the desired behaviour:
// concurrent `&self` allocate would race on the cursor — use
// `SharedBumpArena` for the cross-thread case.
unsafe impl<B: FixedRange + Send> Send for BumpArena<B> {}

// ============================================================================
// Scope — RAII scratch checkpoint
// ============================================================================

/// A scratch scope over a [`BumpArena`], created by [`BumpArena::scope`].
///
/// Allocate through it; when it drops (normally **or on a panic unwind**) the
/// arena's cursor rewinds to where the scope began, reclaiming everything the
/// scope allocated. References handed out by the scope borrow it (`&self`), so
/// the borrow checker forbids them from outliving the rewind — no `unsafe`
/// lifetime branding is needed, and a misuse is a compile error (see the
/// `compile_fail` example on [`BumpArena::scope`]).
///
/// Scopes nest: call [`scope`](Scope::scope) on a `Scope` to checkpoint again.
///
/// While the scope is alive the underlying arena is mutably borrowed, so it
/// cannot be used directly — which is exactly what prevents an outer allocation
/// from landing in the region the scope will reclaim.
pub struct Scope<'a, B: FixedRange> {
    arena: &'a mut BumpArena<B>,
    /// Cursor offset at scope creation; the rewind target on drop.
    mark: usize,
}

impl<'a, B: FixedRange> Scope<'a, B> {
    /// Typed scratch allocation bound to this scope — see
    /// [`BumpArena::alloc_uninit`]. The returned reference borrows the scope, so
    /// it cannot outlive the rewind. Returns `&mut MaybeUninit<T>`; write a `T`
    /// before assuming it initialized.
    ///
    /// `&mut` from `&self` is the bump-allocator idiom (cf. `bumpalo::Bump::alloc`):
    /// each call returns a *disjoint* fresh region, so the `&mut`s never alias.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_uninit<T>(&self) -> Result<&mut core::mem::MaybeUninit<T>, AllocError> {
        let p = self.arena.alloc_uninit::<T>()?;
        // SAFETY: `alloc_uninit` returns fresh, properly-aligned, non-aliasing
        // storage for one `T`. Binding it to `&self` ties the reference to this
        // scope; the borrow checker then prevents it from outliving the rewind
        // in `Drop`, which reclaims exactly this memory. For a non-ZST `T` each
        // call returns a disjoint region, so the `&mut`s never alias. For a ZST,
        // every call yields the same dangling-but-aligned pointer, but a
        // `&mut MaybeUninit<ZST>` accesses zero bytes — it claims no location
        // exclusively, so the aliasing restriction is vacuously satisfied even
        // when several coexist. (Verified under Miri in the `miri_targets`
        // `alloc_uninit_and_scope_round_trip` ZST case.)
        Ok(unsafe { &mut *p.cast::<core::mem::MaybeUninit<T>>().as_ptr() })
    }

    /// Raw byte scratch allocation bound to this scope — see
    /// [`Allocator::allocate`]. The returned slice borrows the scope.
    ///
    /// `&mut` from `&self` is intentional (see [`alloc_uninit`](Self::alloc_uninit)):
    /// each call returns a disjoint fresh region.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn allocate(&self, layout: NonZeroLayout) -> Result<&mut [u8], AllocError> {
        let block = Allocator::allocate(&*self.arena, layout)?;
        // SAFETY: fresh, non-aliasing bytes from the arena, bound to `&self`
        // (this scope) so they cannot outlive the rewind. See `alloc_uninit`.
        Ok(unsafe { &mut *block.as_ptr() })
    }

    /// Allocate and initialize one `T` as scratch, bound to this scope — see
    /// [`BumpArena::alloc`]. The returned `&mut T` cannot outlive the rewind.
    #[inline]
    pub fn alloc<T>(&self, value: T) -> Result<&mut T, AllocError> {
        Ok(self.alloc_uninit::<T>()?.write(value))
    }

    /// Copy a slice into the scope, bound to it — see
    /// [`BumpArena::alloc_slice_copy`]. The returned `&mut [T]` cannot outlive
    /// the rewind.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice_copy<T: Copy>(&self, src: &[T]) -> Result<&mut [T], AllocError> {
        let p = self.arena.alloc_slice_copy(src)?;
        // SAFETY: fresh, non-aliasing storage bound to `&self`. See `alloc_uninit`.
        Ok(unsafe { &mut *p.as_ptr() })
    }

    /// Copy a string into the scope, bound to it — see [`BumpArena::alloc_str`].
    /// The returned `&mut str` cannot outlive the rewind.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_str(&self, s: &str) -> Result<&mut str, AllocError> {
        let p = self.arena.alloc_str(s)?;
        // SAFETY: fresh, non-aliasing UTF-8 storage bound to `&self`.
        Ok(unsafe { &mut *p.as_ptr() })
    }

    /// Bytes currently allocated from the underlying arena (the absolute cursor,
    /// not relative to this scope's mark). Useful in assertions/tests.
    #[inline]
    pub fn arena_allocated(&self) -> usize {
        self.arena.allocated()
    }

    /// Open a nested scratch scope inside this one. The inner scope reclaims
    /// only what *it* allocated when it drops; this outer scope is untouched.
    #[inline]
    pub fn scope(&mut self) -> Scope<'_, B> {
        self.arena.scope()
    }
}

impl<'a, B: FixedRange> Drop for Scope<'a, B> {
    #[inline]
    fn drop(&mut self) {
        // Rewind to the mark, reclaiming everything the scope allocated. Runs on
        // normal exit AND on a panic unwind, so a panicking scope body cannot
        // leave the cursor advanced. Sound because `Drop` takes `&mut self`:
        // every scope-issued reference borrows `&self`, so none can be alive
        // here (the borrow checker enforces it), and no outer allocation
        // occurred (the arena was mutably borrowed by this scope throughout).
        self.arena.rewind_to(self.mark);
    }
}

// ============================================================================
// BumpDeallocator
// ============================================================================

/// ZST deallocator token tied to a [`BumpArena`]'s borrow.
///
/// Used as the `A` parameter in `Box<T, A>` / `Vec<T, A>` patterns where
/// the box was constructed via `Box::from_raw_in` against pointers
/// obtained from the arena directly. The `'a` lifetime ensures the arena
/// outlives the box.
///
/// # Allocate-always-fails footgun
///
/// The [`Allocator::allocate`] impl on `BumpDeallocator` returns
/// `Err(AllocError)` for **every** call. This is deliberate: the
/// deallocator is a *destruction token*, not an allocation source.
/// The correct usage pattern is:
///
/// ```text
///     let arena: BumpArena<_> = ...;
///     let ptr = arena.allocate(layout)?;       // allocate via arena
///     unsafe { ptr.cast::<T>().write(value) }; // place a T
///     let boxed: Box<T, BumpDeallocator<'_>> =
///         unsafe { Box::from_raw_in(
///             ptr.cast::<T>().as_ptr(),
///             arena.deallocator(),
///         )};
/// ```
///
/// Plugging `BumpDeallocator` into code that *grows* a collection
/// (`Vec::reserve`, `Vec::push` that re-allocates, `Box::new_in` —
/// anything that calls `Allocator::allocate` on the supplied
/// allocator) will fail at runtime. Use `BumpArena` itself as the
/// allocator for those patterns, or pre-size the collection so it
/// never reallocates.
#[derive(Copy, Clone, Debug)]
pub struct BumpDeallocator<'a>(PhantomData<&'a ()>);

unsafe impl Deallocator for BumpDeallocator<'_> {
    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // No-op. Deallocation through the token is a marker that the
        // arena-allocated value's destructor has run; reclaim happens on
        // arena reset/drop.
    }
}

unsafe impl Allocator for BumpDeallocator<'_> {
    /// Always fails. Allocate through the arena, not the deallocator.
    #[inline]
    fn allocate(&self, _layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;

    #[test]
    fn allocate_advances_cursor() {
        let arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
        assert_eq!(arena.allocated(), 0);
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let _ = arena.allocate(layout).unwrap();
        assert_eq!(arena.allocated(), 64);
    }

    #[test]
    fn allocate_returns_aligned_pointer() {
        let arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
        // Push the cursor off zero first.
        let _ = arena
            .allocate(NonZeroLayout::from_size_align(3, 1).unwrap())
            .unwrap();
        let layout = NonZeroLayout::from_size_align(8, 16).unwrap();
        let block = arena.allocate(layout).unwrap();
        assert_eq!(block.cast::<u8>().as_ptr() as usize % 16, 0);
    }

    #[test]
    fn allocate_fails_when_exhausted() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let big = NonZeroLayout::from_size_align(64, 1).unwrap();
        let _ = arena.allocate(big).unwrap();
        let one = NonZeroLayout::from_size_align(1, 1).unwrap();
        assert!(arena.allocate(one).is_err());
    }

    /// Alignment padding must count toward exhaustion: the bounds check is on
    /// the *aligned* offset + size, not the raw cursor + size. `InlineBacked`'s
    /// base is 16-aligned, so after a 1-byte alloc (cursor = 1) a 16-aligned
    /// request rounds the offset deterministically up to 16. A 56-byte request
    /// then needs `16 + 56 = 72 > 64` and must fail — whereas a buggy check
    /// using `cursor + size = 1 + 56 = 57 <= 64` would wrongly succeed.
    #[test]
    fn alignment_padding_counts_toward_exhaustion() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let one = NonZeroLayout::from_size_align(1, 1).unwrap();
        let _ = arena.allocate(one).unwrap(); // cursor = 1
        let aligned = NonZeroLayout::from_size_align(56, 16).unwrap();
        assert!(
            arena.allocate(aligned).is_err(),
            "alignment padding (offset 16) must be counted in the exhaustion check",
        );
    }

    #[test]
    fn reset_reclaims_all() {
        let mut arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let _ = arena.allocate(layout).unwrap();
        assert_eq!(arena.allocated(), 32);
        arena.reset();
        assert_eq!(arena.allocated(), 0);
        let _ = arena.allocate(layout).unwrap();
    }

    #[test]
    fn deallocate_is_no_op() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = arena.allocate(layout).unwrap();
        let used_before = arena.allocated();
        unsafe { arena.deallocate(block.cast(), layout) };
        assert_eq!(arena.allocated(), used_before);
    }

    #[test]
    fn fixed_range_contains_allocations() {
        let arena = BumpArena::new(InlineBacked::<128>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = arena.allocate(layout).unwrap();
        assert!(arena.contains(block.cast::<u8>()));
    }

    #[test]
    fn capacity_bytes_reports_backing_size() {
        let arena = BumpArena::new(InlineBacked::<2048>::new()).unwrap();
        assert_eq!(arena.capacity_bytes(), Some(2048));
    }

    /// Regression: BumpArena historically cached an absolute `base`
    /// pointer captured BEFORE the backing was moved into Self. For
    /// structure-relative backings (`InlineBacked` returns
    /// `&self.storage`), that pointer became stale on every move and
    /// silently corrupted subsequent allocates. The fix re-queries
    /// `self.backing.base()` at each allocate. Verify the arena's
    /// `FixedRange::base()` agrees with the backing's live `base()` and
    /// that the first allocate lands at exactly that address.
    #[test]
    fn base_pointer_matches_backing_after_move() {
        let arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        let arena_base = arena.base().as_ptr();
        let backing_base = arena.backing().base().as_ptr();
        assert_eq!(
            arena_base, backing_base,
            "BumpArena's base must agree with the live backing — stale-pointer regression",
        );
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        let block = arena.allocate(layout).unwrap();
        assert_eq!(
            block.cast::<u8>().as_ptr() as usize,
            backing_base as usize,
            "first alloc must be at backing.base()",
        );
    }

    #[test]
    fn deallocator_compiles_and_runs() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let d = arena.deallocator();
        // The deallocator's allocate must always fail by contract.
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        assert!(d.allocate(layout).is_err());
        // Calling deallocate is safe and a no-op.
        let block = arena.allocate(layout).unwrap();
        unsafe { d.deallocate(block.cast(), layout) };
    }

    #[test]
    fn very_small_alignment_is_one() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let l1 = NonZeroLayout::from_size_align(1, 1).unwrap();
        let _ = arena.allocate(l1).unwrap();
        let _ = arena.allocate(l1).unwrap();
        assert_eq!(arena.allocated(), 2);
    }

    #[test]
    fn alloc_uninit_is_aligned_and_writable() {
        let arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        // Force a misaligned starting cursor, then allocate an 8-aligned type.
        let _pad = arena.alloc_uninit::<u8>().unwrap();
        let p = arena.alloc_uninit::<u64>().unwrap();
        assert_eq!(p.as_ptr() as usize % core::mem::align_of::<u64>(), 0);
        unsafe {
            p.as_ptr().write(0x0102_0304_0506_0708);
            assert_eq!(p.as_ptr().read(), 0x0102_0304_0506_0708);
        }
    }

    #[test]
    fn alloc_uninit_zst_consumes_nothing_and_is_aligned() {
        #[repr(align(16))]
        struct Zst;
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let before = arena.allocated();
        let p = arena.alloc_uninit::<Zst>().unwrap();
        assert_eq!(arena.allocated(), before, "ZST must consume no space");
        assert_eq!(p.as_ptr() as usize % core::mem::align_of::<Zst>(), 0);
    }

    #[test]
    fn alloc_uninit_reports_oom_like_allocate() {
        let arena = BumpArena::new(InlineBacked::<8>::new()).unwrap();
        // 8 bytes total; a u64 fits exactly once, the second must fail.
        assert!(arena.alloc_uninit::<u64>().is_ok());
        assert!(arena.alloc_uninit::<u64>().is_err());
    }

    #[test]
    fn alloc_uninit_alignment_padding_counts_toward_exhaustion() {
        // Push the cursor off an 8-aligned boundary with a 1-byte alloc, then a
        // u64 needs to skip to offset 8 — its end (16) exceeds the 8-byte
        // region, so it must OOM. This pins the alignment-rounding path (the
        // bare OOM test above starts pre-aligned and wouldn't catch a broken
        // mask).
        let arena = BumpArena::new(InlineBacked::<8>::new()).unwrap();
        let _b = arena.alloc_uninit::<u8>().unwrap(); // cursor = 1
        assert!(
            arena.alloc_uninit::<u64>().is_err(),
            "u64 must not fit once alignment padding pushes its end past capacity"
        );
    }

    #[test]
    fn alloc_writes_value_and_slice_and_str_copy() {
        let arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        let v = arena.alloc(0x1122_3344u32).unwrap();
        assert_eq!(unsafe { v.as_ptr().read() }, 0x1122_3344);

        let s = arena.alloc_slice_copy(&[1u16, 2, 3, 4]).unwrap();
        let sl = unsafe { s.as_ref() };
        assert_eq!(sl, &[1, 2, 3, 4]);

        let st = arena.alloc_str("forge").unwrap();
        assert_eq!(unsafe { st.as_ref() }, "forge");
    }

    #[test]
    fn alloc_slice_copy_empty_and_zst() {
        let arena = BumpArena::new(InlineBacked::<64>::new()).unwrap();
        let before = arena.allocated();
        // Empty slice: no space consumed, length 0.
        let e = arena.alloc_slice_copy::<u32>(&[]).unwrap();
        assert_eq!(e.len(), 0);
        // ZST slice of length 3: no space consumed, length preserved.
        let z = arena.alloc_slice_copy(&[(), (), ()]).unwrap();
        assert_eq!(z.len(), 3);
        assert_eq!(
            arena.allocated(),
            before,
            "empty/ZST slices consume no space"
        );
    }

    #[test]
    fn grow_in_place_when_last_allocation_does_not_copy() {
        let arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        let l8 = NonZeroLayout::from_size_align(8, 8).unwrap();
        let l32 = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = arena.allocate(l8).unwrap();
        let ptr = block.cast::<u8>();
        unsafe { core::ptr::write_bytes(ptr.as_ptr(), 0xAB, 8) };
        let after_first = arena.allocated();

        // It's the most-recent allocation → grow advances the cursor in place,
        // returns the SAME pointer, copies nothing.
        let grown = unsafe { arena.grow(ptr, l8, l32).unwrap() };
        assert_eq!(
            grown.cast::<u8>(),
            ptr,
            "in-place grow keeps the same pointer"
        );
        assert_eq!(grown.len(), 32);
        assert_eq!(
            arena.allocated(),
            after_first - 8 + 32,
            "cursor advanced by the grow delta, no relocation"
        );
        // The preserved prefix is intact.
        assert_eq!(unsafe { ptr.as_ptr().read() }, 0xAB);
    }

    #[test]
    fn grow_relocates_when_not_last_allocation() {
        let arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        let l8 = NonZeroLayout::from_size_align(8, 8).unwrap();
        let l32 = NonZeroLayout::from_size_align(32, 8).unwrap();
        let first = arena.allocate(l8).unwrap().cast::<u8>();
        unsafe { core::ptr::write_bytes(first.as_ptr(), 0xCD, 8) };
        // A second allocation makes `first` no longer the most-recent.
        let _second = arena.allocate(l8).unwrap();

        let grown = unsafe { arena.grow(first, l8, l32).unwrap() };
        assert_ne!(grown.cast::<u8>(), first, "non-last grow must relocate");
        assert_eq!(grown.len(), 32);
        // Relocated copy preserves the old bytes.
        unsafe {
            for i in 0..8 {
                assert_eq!(grown.cast::<u8>().as_ptr().add(i).read(), 0xCD);
            }
        }
    }

    #[test]
    fn scope_alloc_value_slice_str_are_scope_bound() {
        let mut arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        let before = arena.allocated();
        {
            let scope = arena.scope();
            let v: &mut u64 = scope.alloc(7u64).unwrap();
            assert_eq!(*v, 7);
            *v = 9;
            assert_eq!(*v, 9);
            let sl: &mut [u8] = scope.alloc_slice_copy(&[10u8, 20, 30]).unwrap();
            assert_eq!(sl, &[10, 20, 30]);
            let st: &mut str = scope.alloc_str("hi").unwrap();
            assert_eq!(&*st, "hi");
        }
        assert_eq!(arena.allocated(), before, "scope reclaims the scratch");
    }

    #[test]
    fn scope_rewinds_cursor_on_drop() {
        let mut arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        let _keep = arena.alloc_uninit::<u32>().unwrap();
        let before = arena.allocated();
        {
            let scope = arena.scope();
            let _a = scope.alloc_uninit::<[u8; 32]>().unwrap();
            let _b = scope.alloc_uninit::<[u8; 32]>().unwrap();
            assert!(scope.arena_allocated() >= before + 64);
        }
        assert_eq!(arena.allocated(), before, "scope must rewind to its mark");
        // The reclaimed region is reusable by the outer arena.
        let _reuse = arena.alloc_uninit::<[u8; 32]>().unwrap();
        assert_eq!(arena.allocated(), before + 32);
    }

    #[test]
    fn nested_scopes_rewind_independently() {
        let mut arena = BumpArena::new(InlineBacked::<512>::new()).unwrap();
        let base = arena.allocated();
        {
            let mut outer = arena.scope();
            let _o = outer.alloc_uninit::<[u8; 16]>().unwrap();
            let after_outer = outer.arena_allocated();
            {
                let inner = outer.scope();
                let _i = inner.alloc_uninit::<[u8; 64]>().unwrap();
                assert!(inner.arena_allocated() >= after_outer + 64);
            }
            assert_eq!(
                outer.arena_allocated(),
                after_outer,
                "inner scope rewinds to its own mark, leaving outer intact"
            );
        }
        assert_eq!(arena.allocated(), base, "outer scope rewinds fully");
    }

    // Panic safety: a panic inside a scope body must still rewind (Drop runs on
    // unwind). Needs std for catch_unwind.
    #[cfg(feature = "std")]
    #[test]
    fn scope_rewinds_on_panic() {
        let mut arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        let before = arena.allocated();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let scope = arena.scope();
            let _a = scope.alloc_uninit::<[u8; 64]>().unwrap();
            panic!("boom inside scope");
        }));
        assert!(r.is_err(), "the panic should propagate out of catch_unwind");
        assert_eq!(
            arena.allocated(),
            before,
            "Drop must rewind the cursor even on a panic unwind"
        );
    }

    // The `OsBacked` forward exists only for OS-managed backings, so this test
    // needs `MmapBacked` (std + unix/windows). It proves the surface forwards to
    // the backing and that the pool-overflow path — reset, release the whole
    // region's pages, reuse — round-trips without `munmap`.
    #[cfg(all(feature = "std", any(unix, windows)))]
    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn osbacked_forwards_and_release_after_reset_round_trips() {
        use crate::backing::{page_size, MmapBacked};
        use forge_alloc_core::OsBacked;

        let mut arena = BumpArena::new(MmapBacked::new(64 * 1024).unwrap()).unwrap();

        // Cross-interface consistency: the OsBacked surface must agree with the
        // independent FixedRange surface and the known reservation size — not a
        // self-referential `arena.x() == arena.backing().x()` tautology (which
        // would pass even if the forward were broken).
        assert_eq!(arena.region_size(), 64 * 1024);
        assert_eq!(arena.region_size(), arena.capacity()); // OsBacked vs cached FixedRange size
        assert_eq!(arena.base_ptr(), arena.base()); // OsBacked vs FixedRange base

        let layout = NonZeroLayout::from_size_align(page_size(), 8).unwrap();
        let block = arena.allocate(layout).unwrap();
        // SAFETY: freshly allocated page-sized block.
        unsafe { core::ptr::write_bytes(block.cast::<u8>().as_ptr(), 0xEE, page_size()) };

        // Pool-overflow path: no live allocations after reset, so releasing the
        // full region is sound; the mapping stays reserved for reuse.
        arena.reset();
        let (base, size) = (arena.base_ptr(), arena.region_size());
        // SAFETY: full region, no live allocations after reset.
        unsafe { arena.release_pages(base, size) };

        // The still-mapped arena reuses cleanly — write must not fault.
        let block2 = arena.allocate(layout).unwrap();
        // SAFETY: freshly allocated page-sized block.
        unsafe { core::ptr::write_bytes(block2.cast::<u8>().as_ptr(), 0x11, page_size()) };
    }
}

// ============================================================================
// Kani proof harnesses
//
// Kani is a bounded model checker that verifies properties of unsafe code
// over the entire state space of unconstrained inputs. These harnesses run
// under the `kani` cfg (set by `cargo kani`) and exercise the alignment
// rounding + bounds-check logic in `allocate`.
// ============================================================================

// Kani proofs depend on `crate::backing::InlineBacked`; the `backing` module is gated
// behind the `std` feature in this crate (see Cargo.toml), so the proof
// module must be gated similarly. Kani CI must run with the `std`
// feature enabled for these proofs to compile.
#[cfg(all(kani, feature = "std"))]
mod kani_proofs {
    use super::*;
    use crate::backing::InlineBacked;

    /// Any successful `allocate(layout)` returns a pointer aligned to
    /// `layout.align()`. Verified over all combinations of (cursor
    /// position, requested size, requested alignment) that fit a
    /// 1 KiB arena.
    #[kani::proof]
    #[kani::unwind(4)]
    fn allocate_returns_aligned_pointer() {
        let arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
        // Bounded inputs — Kani enumerates the cross product.
        let size_log: u32 = kani::any();
        kani::assume(size_log <= 8); // size in 1..=256
        let align_log: u32 = kani::any();
        kani::assume(align_log <= 4); // align in {1,2,4,8,16}
        let size = 1usize << size_log;
        let align = 1usize << align_log;
        let layout = NonZeroLayout::from_size_align(size, align).unwrap();
        if let Ok(block) = arena.allocate(layout) {
            let p = block.cast::<u8>().as_ptr() as usize;
            assert!(p % align == 0);
            // And the slice length covers the requested size.
            assert!(block.len() >= size);
        }
    }

    /// Repeated `allocate` calls produce strictly increasing cursor
    /// values that never exceed capacity. Verified over a small
    /// number of allocations on a 256-byte arena.
    #[kani::proof]
    #[kani::unwind(4)]
    fn cursor_monotonic_and_bounded() {
        let arena = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        let cap = arena.capacity();
        let mut last = 0usize;
        for _ in 0..3 {
            let before = arena.allocated();
            if arena.allocate(layout).is_ok() {
                let after = arena.allocated();
                assert!(after > before);
                assert!(after <= cap);
                last = after;
            }
        }
        assert!(last <= cap);
    }
}
