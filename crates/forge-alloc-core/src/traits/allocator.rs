//! The `Allocator` / `Deallocator` split.
//!
//! The standard `Allocator` trait bundles allocation and deallocation. That
//! creates a structural problem for arena-style allocators: a `Box<T, &Arena>`
//! must carry the arena reference to satisfy the deallocator bound at drop
//! time, even though `Arena::deallocate` is a no-op. This split lets arenas
//! provide a ZST `Deallocator` (`BumpDeallocator<'a>`) while still satisfying
//! standard collection types via `StdCompat`.

use core::ptr::NonNull;

use super::non_zero_layout::{AllocError, NonZeroLayout};

/// The deallocation half of an allocator. Every type that issues memory must
/// be able to take it back at drop time.
///
/// For arena allocators (`BumpArena`, parsers), `deallocate` is a no-op —
/// reclaim happens via `reset()` (see [`Allocator::reset`]).
///
/// # Safety
///
/// Trait-level invariants implementors must uphold — an implementor of this
/// trait declares the following to all callers:
///
/// 1. `deallocate` accepts any pointer previously returned by the *same
///    allocation domain*'s [`Allocator::allocate`] (or any other method
///    of this trait pair that issued memory — `allocate_zeroed`, `grow`,
///    `shrink`), paired with the same [`NonZeroLayout`] used to obtain
///    it. For stateful allocators (`Slab`, `BumpArena`, `StackAlloc`,
///    `ExtendableSlab`, etc.) the "domain" is the very `&self` that
///    issued the pointer — pointers from a sibling instance of the same
///    type are UB. For stateless allocators that delegate to a global
///    backing (`System`, `MmapBacked` — the OS manages the per-mapping
///    state — any ZST forwarder), the "domain" extends to *any
///    instance of the same type*, since the OS / global heap is the
///    actual owner. Implementors that are unclear should document
///    explicitly; conservatively, callers should pass the very
///    instance that issued the pointer.
/// 2. After `deallocate` returns the pointer is invalid; the caller may
///    not read, write, or compare its address to any *new* pointer (the
///    integer value may be reused by the allocator). Implementors may
///    additionally poison, quarantine, or return memory to the OS —
///    those are defense-in-depth choices made by hardening wrappers,
///    not requirements of the trait.
/// 3. Calling `deallocate` with a pointer that did not originate from
///    this allocator, with a `layout` that does not match the original
///    `allocate` call, or with a pointer that has already been
///    `deallocate`d, is undefined behavior.
/// 4. `deallocate` may be called from any thread that holds the
///    appropriate `&self` reference. If the implementor is `!Sync` the
///    receiver `&self` itself enforces single-thread access; if `Sync`
///    the implementor must internally serialize concurrent calls.
pub unsafe trait Deallocator {
    /// Release a previously allocated block.
    ///
    /// # Safety
    ///
    /// All four of:
    /// - `ptr` must have been returned by a previous call to the *same
    ///   allocation domain*'s `allocate` / `allocate_zeroed` / `grow` /
    ///   `shrink` (see the trait-level safety doc — for stateful
    ///   allocators that domain is the same `&self`; for stateless
    ///   forwarders to a global backing it is any instance of the
    ///   same type).
    /// - `layout` must equal the layout supplied to the call that
    ///   produced `ptr` — same `size` and same `align`. (Wrappers that
    ///   transparently inflate the layout, e.g. `Canary`, must
    ///   inverse-transform before forwarding.)
    /// - `ptr` must not have been previously `deallocate`d. Double-free
    ///   is UB.
    /// - After the call, the caller must not read, write, or *compare
    ///   against new pointers* the value of `ptr`.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout);
}

/// The allocation contract. Returns memory satisfying a [`NonZeroLayout`] or
/// [`AllocError`]. Extends [`Deallocator`] so that any allocator can also
/// reclaim what it issued.
///
/// `grow` and `shrink` have default implementations that allocate-copy-free.
/// Implementors with native in-place resize support should override.
///
/// # Safety
///
/// Trait-level invariants implementors must uphold — an implementor of this
/// trait declares the following to all callers:
///
/// 1. Every pointer returned from a successful `allocate` /
///    `allocate_zeroed` / `grow` / `shrink` call is:
///    - **non-null** (the [`NonNull<[u8]>`] type guarantees this
///      statically; the implementor must not return a pointer that is
///      dereferenced to mean "null" in any wider sense — e.g. a
///      mapping address known to be reserved-but-unwritable),
///    - **aligned** to at least `layout.align()`,
///    - **writable** for at least `layout.size()` bytes (the returned
///      slice's `len()` reports the *usable* size, which may exceed
///      `layout.size()` — callers may use the extra room, but
///      `deallocate` still requires the original `layout`).
/// 2. Two live allocations from the same allocator instance never
///    occupy overlapping memory ranges. "Live" means a pointer that has
///    been issued but not yet `deallocate`d (or invalidated by a
///    successful `grow` / `shrink` that returned a different pointer).
/// 3. Once a pointer has been returned, its bytes are not concurrently
///    written by the allocator until `deallocate` releases them.
///    Implementors that maintain internal bookkeeping (Slab freelist
///    links, Canary sentinels) must scope that bookkeeping to the
///    *non-user-visible* portion of the allocation, or to the time
///    *before* `allocate` returns / *after* `deallocate` is called.
/// 4. The implementor upholds the [`Deallocator`] safety contract for
///    every pointer it issues.
/// 5. Resizes (`grow`, `shrink`): on success, either the returned
///    pointer equals the input `ptr` (in-place resize — implementor
///    has reused the same allocation) or it does not (a fresh
///    allocation was made and the old one has been freed). On
///    `Err(AllocError)` the original allocation is untouched and the
///    input `ptr` is still live.
pub unsafe trait Allocator: Deallocator {
    /// Allocate a block satisfying `layout`. The returned slice's length is
    /// at least `layout.size()` but may be larger.
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError>;

    /// Allocate a zero-initialized block.
    #[inline]
    fn allocate_zeroed(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let block = self.allocate(layout)?;
        // SAFETY: `allocate` guarantees `block` covers at least `layout.size()`
        // bytes of writable, aligned memory.
        unsafe {
            core::ptr::write_bytes(block.cast::<u8>().as_ptr(), 0, layout.size().get());
        }
        Ok(block)
    }

    /// Grow an allocation in place if possible, otherwise allocate-copy-free.
    ///
    /// # Safety
    ///
    /// `ptr` must come from this allocator with `old` as its layout. `new` must
    /// have `new.size() >= old.size()` and the same alignment as `old`.
    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: NonZeroLayout,
        new: NonZeroLayout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new.size() >= old.size());
        debug_assert_eq!(old.align(), new.align());
        let dst = self.allocate(new)?;
        // SAFETY: caller's contract gives us a valid `ptr` of `old.size()`
        // bytes; `dst` is a fresh allocation of `new.size() >= old.size()`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                dst.cast::<u8>().as_ptr(),
                old.size().get(),
            );
            self.deallocate(ptr, old);
        }
        Ok(dst)
    }

    /// Shrink an allocation in place if possible, otherwise allocate-copy-free.
    ///
    /// # Safety
    ///
    /// `ptr` must come from this allocator with `old` as its layout. `new` must
    /// have `new.size() <= old.size()` and the same alignment as `old`.
    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: NonZeroLayout,
        new: NonZeroLayout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new.size() <= old.size());
        debug_assert_eq!(old.align(), new.align());
        let dst = self.allocate(new)?;
        // SAFETY: as in `grow`, but copying `new.size()` bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                dst.cast::<u8>().as_ptr(),
                new.size().get(),
            );
            self.deallocate(ptr, old);
        }
        Ok(dst)
    }

    /// Reclaim everything previously allocated. Default impl returns
    /// `AllocError` — only arena-style allocators implement a meaningful
    /// reset.
    ///
    /// **NOTE**: this method is provisional. A dedicated `Reset` trait is the
    /// v2.0 design (see `docs/ARCHITECTURE.md`); the method on `Allocator`
    /// exists in v1.0 to avoid a trait explosion during stabilization.
    /// Implementors of arenas should override; implementors of slabs should
    /// leave the default.
    #[inline]
    fn reset(&mut self) -> Result<(), AllocError> {
        Err(AllocError)
    }

    /// Usable size of an existing allocation, if the allocator can report
    /// it. Defaults to `None` — implementors that track usable size
    /// override.
    ///
    /// `None` means "this allocator does not track usable size at the
    /// granularity needed", not "the allocation has zero usable bytes".
    /// `Some(n)` is a contract: implementors must guarantee
    /// `n >= layout.size().get()`, otherwise a caller relying on the
    /// extra capacity could trigger a buffer overflow.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by a previous call to *this same
    ///   allocator instance*'s `allocate` / `allocate_zeroed` / `grow` /
    ///   `shrink`.
    /// - `layout` must equal the layout supplied to the call that
    ///   produced `ptr`.
    /// - `ptr` must not have been previously `deallocate`d.
    #[inline]
    unsafe fn usable_size(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) -> Option<usize> {
        None
    }

    /// Total bytes this allocator can issue, if bounded. `None` for unbounded
    /// allocators like `System`. Used by `Watermark` to compute thresholds.
    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        None
    }

    /// Detected freelist / metadata corruption events observed by this
    /// allocator since construction.
    ///
    /// Many allocators in this family include defense-in-depth checks
    /// that **silently disarm** corrupted state rather than aborting:
    /// `Slab` walks past a MAC-failed freelist link and falls back to
    /// next-uncarved; `SizeClassed` and `ExtendableSlab` route around
    /// corrupted next-idx values; etc. Silent disarm preserves safety
    /// but makes attacks invisible to operators. Each corruption-detection
    /// site increments an allocator-local counter; this method exposes it.
    ///
    /// # Contract
    ///
    /// Implementors must guarantee:
    ///
    /// 1. **Monotonic**: the value returned never decreases for the
    ///    lifetime of `self`. Implementors that store the counter in an
    ///    `AtomicUsize` must widen to `u64` before returning, or document
    ///    that the counter wraps at `usize::MAX` (which on 32-bit targets
    ///    can happen in seconds under extreme load).
    /// 2. **Thread-safe / `&self`-only**: callable from any thread that
    ///    holds `&self`. Implementors must not take a mutable lock that
    ///    could deadlock with concurrent `allocate` / `deallocate`.
    /// 3. **Eventually consistent**: a read on thread *A* may observe a
    ///    value strictly less than the count of events that have
    ///    happened-before (in the C++/Rust memory-model sense) on
    ///    thread *B*. Concretely, implementors are free to use
    ///    [`core::sync::atomic::Ordering::Relaxed`] loads; callers must
    ///    treat the result as a "rate" sample, not an exact moment.
    /// 4. **Cheap**: this method may be polled at metrics-scrape
    ///    frequency (1 Hz typical, 100 Hz worst-case for tight
    ///    dashboards). Implementors must keep it O(1) or O(n) in
    ///    `n = inner-allocator count` — never O(active allocations).
    ///
    /// # Default
    ///
    /// Returns `0`. This is the stable behavior for allocators with no
    /// corruption-detection sites (`BumpArena`, `System`, `MmapBacked`,
    /// any future allocator that does no metadata validation). Future
    /// versions of this trait will **not** change the default return
    /// value — downstream allocator impls that rely on it are
    /// forward-compatible.
    ///
    /// # Forwarding wrappers
    ///
    /// Wrapper allocators (`Statistics`, `Watermark`, `PoisonOnFree`,
    /// `Quarantine`, `WithFallback`, `NumaLocal`, `HugePageAligned`,
    /// `SplitMetadata`, `SlabOwner`) forward to their inner; the
    /// resulting count is the inner-most allocator's observed corruption.
    /// `GuardPage` returns `0` — its detection is OS-level
    /// SIGSEGV/exception, not a Rust-observable counter.
    ///
    /// Hardened wrappers that `panic!` on detected corruption
    /// (`Canary`, `CacheJitter`) do **not** maintain a local counter —
    /// detection aborts immediately, so the first event would also be the
    /// last and a per-layer delta is meaningless. Their `corruption_events`
    /// forwards to the inner allocator, surfacing any silent-disarm counts
    /// from underneath; the corruption they detect is signalled by the
    /// abort itself, not by this counter.
    #[inline]
    fn corruption_events(&self) -> u64 {
        0
    }
}

// Blanket impls let users pass `&A` or `&mut A` wherever `A: Allocator` is
// required — matches the standard library's pattern and lets arena references
// flow through Box<T, &BumpArena>.

unsafe impl<A: Deallocator + ?Sized> Deallocator for &A {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // SAFETY: forwarded; caller upholds Deallocator contract on `*self`.
        unsafe { (**self).deallocate(ptr, layout) }
    }
}

unsafe impl<A: Allocator + ?Sized> Allocator for &A {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: NonZeroLayout,
        new: NonZeroLayout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded.
        unsafe { (**self).grow(ptr, old, new) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: NonZeroLayout,
        new: NonZeroLayout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded.
        unsafe { (**self).shrink(ptr, old, new) }
    }

    #[inline]
    unsafe fn usable_size(&self, ptr: NonNull<u8>, layout: NonZeroLayout) -> Option<usize> {
        // SAFETY: forwarded; caller upholds `usable_size`'s contract on `*self`.
        unsafe { (**self).usable_size(ptr, layout) }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        (**self).capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        (**self).corruption_events()
    }

    // Note: `reset(&mut self)` is deliberately not forwarded. A `&A` is a
    // shared reference and cannot project to a `&mut A` of the underlying
    // allocator. The default `Err(AllocError)` therefore applies — matching
    // the contract that arena reset requires exclusive ownership.
}
