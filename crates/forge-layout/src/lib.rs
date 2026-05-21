//! # forge-layout
//!
//! Layer 2 layout primitives for the forge-alloc family. These consume any
//! [`forge_core::Allocator`] backing (typically from `forge-backing`) and impose
//! their own structure on the memory.
//!
//! Ships:
//!
//! - [`BumpArena<B>`] + [`BumpDeallocator<'a>`] — single-threaded bump arena.
//! - [`SharedBumpArena<B>`] — atomic-cursor bump arena (`Sync` where `B: Send`).
//!   Requires `target_has_atomic = "ptr"`.
//! - [`Slab<T, B, M>`] — typed fixed-stride slab with pluggable freelist MAC.
//! - [`SizeClassed<B, CLASSES>`] — array of untyped slabs by size class;
//!   ships with [`DEFAULT_CLASS_SIZES_8`] for the spec 8/16/.../1024 set.
//! - [`StackAlloc<B>`] — LIFO frame-stack allocator over a backing buffer.
//! - [`GenerationalSlab<T, B, G>`] + [`Handle<T, G>`] — ABA-safe handles
//!   over a slab; stale handles return `None` rather than dereferencing a
//!   reused slot.
//! - [`WithFallback<P, S>`] — try-primary-then-secondary router with
//!   `FixedRange`-based deallocation routing; [`WithFallback::try_new`]
//!   verifies range disjointness when both halves are `FixedRange`.
//!
//! `std`-only:
//!
//! - [`ExtendableSlab<T, M>`] — growable typed allocator backed by a list of
//!   `Slab<T, MmapBacked, M>` segments.
//! - [`SlabOwner<T, B>`] + [`SlabRemote<T, B>`] — cross-thread typed
//!   allocator with ownership-return remote frees; configurable batch
//!   drain via [`BatchPolicy`] (Fixed / Adaptive).
//!
//! See `composable_allocator_spec.md` §6 for the full design.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

mod bump;
mod fallback;
mod generational;
mod size_classed;
mod slab;
mod stack_alloc;

pub use bump::{BumpArena, BumpDeallocator};
pub use fallback::WithFallback;
pub use generational::{GenerationInt, GenerationalSlab, Handle};
pub use size_classed::{SizeClassed, DEFAULT_CLASS_SIZES_8};
pub use slab::Slab;
pub use stack_alloc::StackAlloc;

#[cfg(feature = "std")]
mod extendable_slab;
#[cfg(feature = "std")]
pub use extendable_slab::ExtendableSlab;

#[cfg(feature = "std")]
mod slab_owner;
#[cfg(feature = "std")]
pub use slab_owner::{
    BatchPolicy, SlabOwner, SlabRemote, ADAPTIVE_COOLDOWN_TICKS, ADAPTIVE_LEVELS,
};

#[cfg(target_has_atomic = "ptr")]
mod shared_bump;
#[cfg(target_has_atomic = "ptr")]
pub use shared_bump::SharedBumpArena;
