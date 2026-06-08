//! # forge-containers
//!
//! Container and interop primitives for the [`forge-alloc`] memory family.
//!
//! Where `forge-alloc` provides **allocators** — things that *hand out* memory
//! and implement the [`Allocator`] / [`FixedRange`] / [`OsBacked`] contracts so
//! they compose into bump / slab / hardening stacks — `forge-containers`
//! provides **containers**: data structures that *hold* memory, built for the
//! same foundation.
//!
//! The first inhabitant is [`AlignedColBuffer`], an owned, over-aligned,
//! growable byte buffer for zero-copy columnar / FFI interop (hand a
//! 64-byte-aligned buffer to Apache Arrow without a copy).
//!
//! The crate is `#![no_std]` (it needs only `alloc`) and takes no heavyweight
//! interop dependency — in particular it ships **no Arrow dependency**; see
//! [`AlignedColBuffer`] for the dependency-free zero-copy handoff recipe.
//!
//! [`forge-alloc`]: https://docs.rs/forge-alloc
//! [`Allocator`]: forge_alloc_core::Allocator
//! [`FixedRange`]: forge_alloc_core::FixedRange
//! [`OsBacked`]: forge_alloc_core::OsBacked

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

mod aligned_col;

pub use aligned_col::AlignedColBuffer;
