//! # forge-backing
//!
//! Layer 1 backing primitives for the forge-alloc family.
//!
//! Backings are the source of memory that higher-layer allocators
//! (`forge-layout`'s `BumpArena`, `Slab`, etc.) sub-allocate from. Each backing
//! exposes its entire region as an allocator that hands out byte-aligned
//! chunks bump-style; layout primitives layered on top impose their own
//! structure on what they receive.
//!
//! M2 ships:
//!
//! - [`InlineBacked<N>`] — fixed-size inline storage, `no_std`-friendly.
//! - [`MmapBacked`] — OS-mapped anonymous region (`mmap` / `VirtualAlloc`),
//!   `std`-only, also implements [`OsBacked`](forge_core::OsBacked).
//! - [`System`] — thin newtype over [`std::alloc::System`] for use as a
//!   fallback backing in `WithFallback<Inner, System>`.
//!
//! See `composable_allocator_spec.md` §5 for the full backing contract.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

mod inline;
pub use inline::{InlineBacked, MAX_ALIGN};

#[cfg(feature = "std")]
mod system;
#[cfg(feature = "std")]
pub use system::System;

#[cfg(feature = "std")]
mod mmap;
#[cfg(feature = "std")]
pub use mmap::{
    mmap_clear_last_os_error, mmap_last_os_error, mmap_record_os_error, MmapBacked, MmapFlags,
};
