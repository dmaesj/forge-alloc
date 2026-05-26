//! # backing
//!
//! Layer 1 backing primitives for the forge-alloc family.
//!
//! Backings are the source of memory that higher-layer allocators
//! (the `layout` module's `BumpArena`, `Slab`, etc.) sub-allocate from. Each backing
//! exposes its entire region as an allocator that hands out byte-aligned
//! chunks bump-style; layout primitives layered on top impose their own
//! structure on what they receive.
//!
//! This module ships:
//!
//! - [`InlineBacked<N>`] — fixed-size inline storage, `no_std`-friendly.
//! - [`StaticBacked<'a>`] — borrows an external `&'a mut [u8]` (e.g. a
//!   linker-provided buffer, `static mut`, `Box::leak`ed slice) as a
//!   `FixedRange`. Truly `no_std`: needs neither `alloc` nor `std`.
//! - [`MmapBacked`] — OS-mapped anonymous region (`mmap` / `VirtualAlloc`),
//!   `std`-only, also implements [`OsBacked`](forge_alloc_core::OsBacked).
//! - [`HugePageBacked`] — OS-mapped anonymous region backed by huge /
//!   large pages (Linux `MAP_HUGETLB`, macOS `VM_FLAGS_SUPERPAGE_SIZE_ANY`,
//!   Windows `MEM_LARGE_PAGES`), `std`-only. Errors when the platform
//!   can't satisfy the request; pair with [`WithFallback`](crate::WithFallback)
//!   for graceful degradation.
//! - [`HeapBytes`] — `FixedRange`-only owner of a single global-allocator
//!   block. The heap twin of `MmapBacked` for cases where mmap-level
//!   isolation isn't worth the syscall cost.
//! - [`System`] — thin newtype over [`std::alloc::System`] for use as a
//!   fallback backing in `WithFallback<Inner, System>`.

mod heap;
mod inline;
mod static_buf;
pub use heap::HeapBytes;
pub use inline::{InlineBacked, MAX_ALIGN};
pub use static_buf::StaticBacked;

#[cfg(feature = "std")]
mod system;
#[cfg(feature = "std")]
pub use system::System;

#[cfg(feature = "std")]
mod mmap;
#[cfg(feature = "std")]
pub use mmap::{
    mmap_clear_last_os_error, mmap_last_os_error, mmap_record_os_error, page_size, MmapBacked,
    MmapFlags,
};

#[cfg(feature = "std")]
mod huge_page_backed;
#[cfg(feature = "std")]
pub use huge_page_backed::HugePageBacked;
