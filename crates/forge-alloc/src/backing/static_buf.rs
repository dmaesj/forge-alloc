//! `StaticBacked<'a>` — borrows an external byte buffer as a
//! [`FixedRange`].
//!
//! True `no_std` backing: no global allocator, no syscalls. Use when
//! you have memory from outside the allocator family (a linker-provided
//! buffer in `.bss`, a `static mut` array, a stack array, a
//! `Box::leak`ed slice) and want to compose it into `BumpArena` /
//! `Slab` for typed allocation.
//!
//! See the type-level documentation on [`StaticBacked`] for the
//! lifetime contract and composition examples.

use core::marker::PhantomData;
use core::ptr::NonNull;

use forge_alloc_core::FixedRange;

/// Borrows an external byte buffer as a [`FixedRange`].
///
/// The most no_std-friendly backing in the family: it pulls in neither
/// `alloc` (unlike [`HeapBytes`](crate::HeapBytes)) nor `std` (unlike
/// [`MmapBacked`](crate::MmapBacked)). The canonical use case is bare
/// metal — a linker-provided buffer in `.bss` or `.data` borrowed once
/// at program init and handed to a `BumpArena` for the program's
/// lifetime.
///
/// # Lifetime
///
/// Carries the lifetime of the borrowed buffer. Any allocator composed
/// over it (e.g. `BumpArena<StaticBacked<'a>>`) inherits the same
/// `'a`. For program-lifetime use, the wrapped slice must be
/// `&'static mut [u8]`.
///
/// # Composition example
///
/// ```ignore
/// // Module-level static buffer, borrowed mutably once at init.
/// // Common in embedded firmware where memory regions come from the
/// // linker script. SAFETY: this is the only place SCRATCH is
/// // borrowed; uniqueness is by program-wide convention.
/// static mut SCRATCH: [u8; 64 * 1024] = [0; 64 * 1024];
/// let buf: &'static mut [u8] = unsafe { &mut SCRATCH };
///
/// use forge_alloc::{BumpArena, StaticBacked};
/// let _arena = BumpArena::new(StaticBacked::new(buf)).unwrap();
/// ```
///
/// `Box::leak` works equally well when you do have an allocator:
///
/// ```
/// use forge_alloc::{BumpArena, StaticBacked};
/// let buf: &'static mut [u8] = Box::leak(vec![0u8; 4096].into_boxed_slice());
/// let _arena = BumpArena::new(StaticBacked::new(buf)).unwrap();
/// ```
///
/// # Factoring
///
/// `StaticBacked` implements only [`FixedRange`]. Bump / slab
/// semantics layer on top via `BumpArena<StaticBacked<'a>>` and
/// `Slab<T, BumpArena<StaticBacked<'a>>>`. This matches the factoring
/// rationale of [`HeapBytes`](crate::HeapBytes): one cleanly factored
/// borrow-backed region rather than yet another bump-cursor
/// implementation.
///
/// # Empty slices
///
/// Empty slices (`len() == 0`) are accepted: the wrapper carries a
/// non-null but possibly-dangling pointer (per Rust's slice
/// invariant) and a size of zero. Allocations against any composing
/// layer will fail at the size check; no UB results from constructing
/// the wrapper.
///
/// # Thread safety
///
/// `Send + Sync` (manually impl'd, mirroring the auto-derive that
/// `&'a mut [u8]` itself enjoys when `u8: Send + Sync`). The wrapping
/// bump cursor's `UnsafeCell` provides the non-`Sync` gate for
/// cross-thread allocation; this backing layer is happy to be shared
/// by reference.
pub struct StaticBacked<'a> {
    ptr: NonNull<u8>,
    len: usize,
    /// Carries the borrow's lifetime and Send/Sync semantics. We
    /// can't store the `&'a mut [u8]` directly because the wrapping
    /// `BumpArena` writes through pointers derived from `base()` via
    /// `&self`, which a `&mut [u8]` field would forbid (write-through-
    /// shared-ref is UB without an `UnsafeCell`). Storing the raw
    /// pointer plus a `PhantomData` marker keeps the variance,
    /// dropck, and auto-trait behavior of the original borrow.
    _marker: PhantomData<&'a mut [u8]>,
}

impl<'a> StaticBacked<'a> {
    /// Wrap a mutable byte slice as a fixed-range backing.
    ///
    /// The borrow lives as long as the returned `StaticBacked`. Empty
    /// slices are accepted (capacity = 0); allocations against any
    /// composing layer (e.g. `BumpArena`) will fail at the size check.
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> Self {
        let len = buf.len();
        // SAFETY: `as_mut_ptr` on a slice is non-null by Rust's slice
        // invariant (even an empty slice has a non-null possibly-
        // dangling pointer; that's fine for FixedRange since size==0
        // makes contains() return false for every ptr).
        let ptr = unsafe { NonNull::new_unchecked(buf.as_mut_ptr()) };
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Capacity in bytes — equal to the length of the borrowed slice.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.len
    }
}

impl<'a> FixedRange for StaticBacked<'a> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.ptr
    }

    #[inline]
    fn size(&self) -> usize {
        self.len
    }
}

// SAFETY: a `&'a mut [u8]` is `Send` when `u8: Send + Sync` (the
// std auto-trait rule for `&'a mut T`). `u8` is both, so the borrow
// itself is `Send` and we restore Send on the structurally-equivalent
// wrapper here. The `NonNull<u8>` is `!Send` by default; we override.
unsafe impl<'a> Send for StaticBacked<'a> {}
// SAFETY: `&'a mut T: Sync` when `T: Sync`. `u8: Sync` so the borrow
// is `Sync`. Sharing `&StaticBacked` across threads is sound: every
// observer sees the same (ptr, len) and neither field is mutated
// after construction.
unsafe impl<'a> Sync for StaticBacked<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BumpArena, Slab};
    use forge_alloc_core::{Allocator, Deallocator, NonZeroLayout};

    #[test]
    fn new_records_buffer_size() {
        let mut buf = [0u8; 1024];
        let s = StaticBacked::new(&mut buf);
        assert_eq!(s.capacity(), 1024);
        assert_eq!(s.size(), 1024);
    }

    #[test]
    fn new_accepts_empty_slice() {
        let mut buf = [0u8; 0];
        let s = StaticBacked::new(&mut buf);
        assert_eq!(s.capacity(), 0);
        assert_eq!(s.size(), 0);
        // Contains is false for the dangling base address — size 0
        // means every "in-range" check fails, which is correct.
        assert!(!s.contains(s.base()));
    }

    #[test]
    fn base_is_stable_across_observations() {
        let mut buf = [0u8; 1024];
        let s = StaticBacked::new(&mut buf);
        let a = s.base().as_ptr();
        let b = s.base().as_ptr();
        assert_eq!(a, b, "base() must report the same address every call");
    }

    #[test]
    fn base_points_into_borrowed_buffer() {
        let mut buf = [0u8; 1024];
        let expected = buf.as_mut_ptr();
        let s = StaticBacked::new(&mut buf);
        assert_eq!(
            s.base().as_ptr(),
            expected,
            "base() must equal the borrowed slice's data pointer",
        );
    }

    #[test]
    fn fixed_range_contains_addresses_inside_buffer() {
        let mut buf = [0u8; 256];
        let s = StaticBacked::new(&mut buf);
        let base = s.base();
        // First byte
        assert!(s.contains(base));
        // Last byte (one before the end)
        // SAFETY: 255 < size; in-bounds offset.
        let last = unsafe { NonNull::new_unchecked(base.as_ptr().add(255)) };
        assert!(s.contains(last));
    }

    /// Composition smoke: `BumpArena` over `StaticBacked` carves
    /// independent allocations within the borrowed region.
    #[test]
    fn bump_arena_over_static_backed_round_trips() {
        let mut buf = [0u8; 1024];
        let buf_base = buf.as_mut_ptr() as usize;
        let bump = BumpArena::new(StaticBacked::new(&mut buf)).unwrap();
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let a = bump.allocate(layout).unwrap();
        let b = bump.allocate(layout).unwrap();
        assert_ne!(
            a.cast::<u8>().as_ptr(),
            b.cast::<u8>().as_ptr(),
            "two allocates must return distinct pointers",
        );
        for p in [a, b] {
            let addr = p.cast::<u8>().as_ptr() as usize;
            assert!(
                addr >= buf_base && addr + 64 <= buf_base + 1024,
                "allocation at {addr:#x} must lie in buffer [{buf_base:#x}, {:#x})",
                buf_base + 1024,
            );
        }
    }

    /// Full stack: `Slab<u64, BumpArena<StaticBacked<'_>>>` — alloc
    /// and dealloc some slots and verify the round trip survives.
    #[test]
    fn slab_over_bump_over_static_round_trips() {
        let mut buf = [0u8; 4096];
        let bump = BumpArena::new(StaticBacked::new(&mut buf)).unwrap();
        let slab: Slab<u64, _> = Slab::new(8, bump).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = slab.allocate(layout).unwrap();
        let b = slab.allocate(layout).unwrap();
        assert_ne!(
            a.cast::<u8>().as_ptr(),
            b.cast::<u8>().as_ptr(),
            "two Slab allocates must return distinct slots",
        );
        unsafe {
            slab.deallocate(a.cast(), layout);
            slab.deallocate(b.cast(), layout);
        }
    }

    /// Borrow released at drop — re-acquiring with a new
    /// `StaticBacked` over the same buffer works. Confirms the
    /// wrapper doesn't pin or leak the borrow.
    #[test]
    fn buffer_can_be_reborrowed_after_drop() {
        let mut buf = [0u8; 64];
        {
            let s = StaticBacked::new(&mut buf);
            assert_eq!(s.size(), 64);
            // s dropped here; borrow released.
        }
        // Second borrow now permitted.
        let s2 = StaticBacked::new(&mut buf);
        assert_eq!(s2.size(), 64);
    }

    /// Send is auto-implemented via the explicit `unsafe impl`.
    /// This test compiles only if `StaticBacked<'_>: Send`.
    #[test]
    fn static_backed_is_send() {
        fn assert_send<T: Send>(_: &T) {}
        let mut buf = [0u8; 64];
        let s = StaticBacked::new(&mut buf);
        assert_send(&s);
    }

    /// Sync is auto-implemented via the explicit `unsafe impl`.
    /// This test compiles only if `StaticBacked<'_>: Sync`.
    #[test]
    fn static_backed_is_sync() {
        fn assert_sync<T: Sync>(_: &T) {}
        let mut buf = [0u8; 64];
        let s = StaticBacked::new(&mut buf);
        assert_sync(&s);
    }

    /// Regression: `base()` and `size()` must observe the same
    /// value across threads given a shared `&StaticBacked`. The
    /// compile-time `unsafe impl Sync` is necessary but not
    /// sufficient — this test pins the behavior at runtime.
    /// Gated on `feature = "std"` because `std::thread::scope`
    /// is std-only; the no_std lib still gets the `assert_sync`
    /// compile-time test above.
    #[test]
    #[cfg(feature = "std")]
    fn shared_ref_observes_same_address_from_two_threads() {
        let mut buf = [0u8; 4096];
        let parent_base = buf.as_mut_ptr() as usize;
        let s = StaticBacked::new(&mut buf);
        std::thread::scope(|scope| {
            let s_ref = &s;
            let h1 = scope.spawn(move || (s_ref.base().as_ptr() as usize, s_ref.size()));
            let h2 = scope.spawn(move || (s_ref.base().as_ptr() as usize, s_ref.size()));
            let (b1, n1) = h1.join().unwrap();
            let (b2, n2) = h2.join().unwrap();
            assert_eq!(b1, parent_base, "thread 1 base must equal parent base");
            assert_eq!(b2, parent_base, "thread 2 base must equal parent base");
            assert_eq!(n1, 4096);
            assert_eq!(n2, 4096);
        });
    }

    /// Regression: moving a `StaticBacked` (e.g. returning by value)
    /// must NOT change `base()` — the wrapper borrows external
    /// memory, so the address it reports is fixed by the
    /// underlying buffer, not by the wrapper's own location.
    /// Contrast with `InlineBacked` which stores storage inline and
    /// whose `base()` DOES move with the wrapper.
    #[test]
    fn base_stable_across_wrapper_move() {
        let mut buf = [0u8; 64];
        let expected = buf.as_mut_ptr() as usize;
        let s1 = StaticBacked::new(&mut buf);
        let before = s1.base().as_ptr() as usize;
        let s2 = s1;
        let after = s2.base().as_ptr() as usize;
        assert_eq!(before, expected);
        assert_eq!(after, expected, "base() must not change when the wrapper moves");
    }
}
