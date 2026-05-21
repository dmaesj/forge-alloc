//! `System` — a thin newtype that exposes the process's global allocator
//! (via [`allocator_api2::alloc::Global`]) as an [`forge_core::Allocator`].
//!
//! Used as a fallback in `WithFallback<Inner, System>` and anywhere a higher
//! layer needs to defer to the OS heap. Not intended for direct use on hot
//! paths — the whole point of this library is to avoid the system heap.
//!
//! # Implementation note
//!
//! Spec §5.3 names this primitive after [`std::alloc::System`]. The
//! implementation routes through [`allocator_api2::alloc::Global`] instead,
//! because `Global` plugs into both the stable and nightly `Allocator` ABIs
//! through a single crate. On a default `cargo` build with no
//! `#[global_allocator]` override, `Global` *is* `std::alloc::System`, so the
//! behaviour matches the spec verbatim. If a downstream user installs a
//! custom `#[global_allocator]`, `System` here will route through that
//! allocator — which is consistent with the spec's intent ("fall back to the
//! OS heap") but worth knowing.

use core::ptr::NonNull;

use forge_core::{AllocError, Allocator, Deallocator, NonZeroLayout};
use allocator_api2::alloc::Allocator as A2;

/// Adapter exposing the standard `System` allocator as an `forge_core::Allocator`.
///
/// `System` does NOT implement [`forge_core::FixedRange`] (it owns no bounded
/// region) nor [`forge_core::OsBacked`] (it does not expose page-level controls).
/// This is intentional — wrappers requiring those traits cannot accidentally
/// be applied to `System`.
#[derive(Copy, Clone, Debug, Default)]
pub struct System;

impl System {
    /// Construct. ZST — no fields.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

// We delegate to `allocator_api2::alloc::Global`. On a default cargo build,
// `Global` resolves to `std::alloc::System`; under a custom
// `#[global_allocator]`, `Global` resolves to that allocator. Either way the
// spec's "fall back to the OS heap" contract is honoured.
//
// Conversion at the boundary: forge-core takes `NonZeroLayout`, the std
// allocator API takes `core::alloc::Layout`. `NonZeroLayout::to_layout()` is
// infallible.

unsafe impl Deallocator for System {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // SAFETY: caller upholds Deallocator contract — ptr came from our
        // allocate(layout), and StdSystem (via the allocator_api2 shim) is
        // a valid receiver for the corresponding deallocate call.
        unsafe { A2::deallocate(&allocator_api2::alloc::Global, ptr, layout.to_layout()) }
    }
}

unsafe impl Allocator for System {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        A2::allocate(&allocator_api2::alloc::Global, layout.to_layout())
    }

    #[inline]
    fn allocate_zeroed(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        A2::allocate_zeroed(&allocator_api2::alloc::Global, layout.to_layout())
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: NonZeroLayout,
        new: NonZeroLayout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded; caller upholds Allocator::grow contract.
        unsafe {
            A2::grow(
                &allocator_api2::alloc::Global,
                ptr,
                old.to_layout(),
                new.to_layout(),
            )
        }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: NonZeroLayout,
        new: NonZeroLayout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded; caller upholds Allocator::shrink contract.
        unsafe {
            A2::shrink(
                &allocator_api2::alloc::Global,
                ptr,
                old.to_layout(),
                new.to_layout(),
            )
        }
    }
}

// Sanity check: ensure System is Send + Sync. The std System allocator is.
const _: () = {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    let _ = assert_send::<System>;
    let _ = assert_sync::<System>;
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_then_deallocate() {
        let s = System;
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let block = s.allocate(layout).expect("System alloc should succeed");
        let p = block.cast::<u8>();
        unsafe {
            core::ptr::write_bytes(p.as_ptr(), 0xAB, 64);
            assert_eq!(*p.as_ptr(), 0xAB);
            s.deallocate(p, layout);
        }
    }

    #[test]
    fn allocate_zeroed_returns_zeros() {
        let s = System;
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = s.allocate_zeroed(layout).unwrap();
        let p = block.cast::<u8>();
        unsafe {
            for i in 0..32 {
                assert_eq!(*p.as_ptr().add(i), 0, "byte {i} should be zeroed");
            }
            s.deallocate(p, layout);
        }
    }

    #[test]
    fn capacity_bytes_is_none() {
        // System is unbounded.
        assert_eq!(System.capacity_bytes(), None);
    }
}
