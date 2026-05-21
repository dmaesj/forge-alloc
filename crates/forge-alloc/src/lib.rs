//! # forge-alloc
//!
//! Composable memory allocator primitives. Snap together at the type level to
//! produce application-specific allocators with zero runtime dispatch overhead,
//! compile-time-enforced guarantees, and pay-for-what-you-use security
//! hardening.
//!
//! This is the **meta-crate** of the forge-alloc family. It re-exports the
//! union of:
//!
//! - `forge-core` — trait contracts and `NonZeroLayout`
//! - `forge-backing` — Layer 1 backing primitives (added in M2)
//! - `forge-layout` — Layer 2 layout primitives (added in M3)
//! - `forge-hardening` — Layer 3 hardening wrappers (added in M5+)
//!
//! Users who want the full surface depend on `forge-alloc`. Users who only
//! need a subset can depend directly on the relevant `forge-*` crate to minimise
//! compile time and dependency footprint.
//!
//! # Quick start
//!
//! ```
//! use forge_alloc::{Allocator, BumpArena, Deallocator, InlineBacked, NonZeroLayout};
//!
//! // 1 KiB stack-local bump arena.
//! let arena = BumpArena::new(InlineBacked::<1024>::new()).unwrap();
//! let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
//!
//! let block = arena.allocate(layout).unwrap();
//! assert_eq!(block.cast::<u8>().as_ptr() as usize % 8, 0);
//! // deallocate is a no-op for BumpArena; reclaim happens via reset(&mut self).
//! unsafe { arena.deallocate(block.cast(), layout) };
//! ```
//!
//! See [`composable_allocator_spec.md`] for the full design and
//! [`COMPOSITION_RECIPES.md`] for caller-facing composition recipes.
//!
//! [`composable_allocator_spec.md`]: https://github.com/dmaesj/forge-alloc/blob/main/docs/composable_allocator_spec.md
//! [`COMPOSITION_RECIPES.md`]: https://github.com/dmaesj/forge-alloc/blob/main/docs/COMPOSITION_RECIPES.md

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub use forge_core::*;

#[doc(inline)]
pub use forge_backing::{InlineBacked, MAX_ALIGN};

#[cfg(feature = "std")]
#[doc(inline)]
pub use forge_backing::{
    mmap_clear_last_os_error, mmap_last_os_error, mmap_record_os_error, MmapBacked, MmapFlags,
    System,
};

#[doc(inline)]
pub use forge_layout::{
    BumpArena, BumpDeallocator, GenerationInt, GenerationalSlab, Handle, SizeClassed, Slab,
    StackAlloc, WithFallback, DEFAULT_CLASS_SIZES_8,
};

#[cfg(target_has_atomic = "ptr")]
#[doc(inline)]
pub use forge_layout::SharedBumpArena;

#[cfg(feature = "std")]
#[doc(inline)]
pub use forge_layout::{
    BatchPolicy, ExtendableSlab, SlabOwner, SlabRemote, ADAPTIVE_COOLDOWN_TICKS, ADAPTIVE_LEVELS,
};

#[doc(inline)]
pub use forge_hardening::{
    AllocStats, CachePadded, CacheJitter, Canary, Faulty, PoisonOnFree, Quarantine, Statistics,
    DEFAULT_POISON,
};

#[cfg(feature = "std")]
#[doc(inline)]
pub use forge_hardening::SplitMetadata;

#[cfg(feature = "std")]
#[doc(inline)]
pub use forge_hardening::{
    current_numa_node, default_huge_page_size, GuardPage, HugePageAligned, NodeSet, NumaLocal,
    NumaPolicy,
};

#[cfg(target_has_atomic = "ptr")]
#[doc(inline)]
pub use forge_hardening::{
    FnHandler, NullHandler, Watermark, WatermarkEvent, WatermarkHandler, WatermarkLevel,
    WatermarkThresholds,
};

#[cfg(all(target_has_atomic = "ptr", feature = "std"))]
#[doc(inline)]
pub use forge_hardening::LogHandler;

// ============================================================================
// Convenience type aliases for recommended compositions (spec §9.5)
// ============================================================================

/// Slab with split metadata and guard pages on both regions —
/// the recommended **maximum-hardening** composition for security-
/// critical data: PHI, key material, audit logs, allocation tokens.
///
/// Expansion: `Slab<T, GuardPage<SplitMetadata<MmapBacked>>, M>`.
///
/// Reading the type inside-out:
///
/// 1. `MmapBacked` — OS-managed anonymous mapping; provides the raw
///    bytes the slab carves from.
/// 2. `SplitMetadata<MmapBacked>` — wraps the data mmap with a
///    separate, virtual-address-disjoint metadata mmap. Forwards the
///    `OsBacked` surface from the data side and exposes the meta
///    region via `meta_base()`/`meta_size()`.
/// 3. `GuardPage<SplitMetadata<MmapBacked>>` — installs unmapped
///    guard pages at both ends of the data region so a linear
///    overflow past any slab slot traps with SIGSEGV /
///    `EXCEPTION_ACCESS_VIOLATION` rather than corrupting adjacent
///    user data.
/// 4. `Slab<T, ..., M>` — typed fixed-stride freelist allocator,
///    optionally with a freelist MAC.
///
/// **Spec note:** the v1.0 spec listed the alias as
/// `GuardPage<SplitMetadata<Slab<...>>>` with `Slab` innermost. That
/// composition doesn't compile because `SplitMetadata`/`GuardPage`
/// require an `OsBacked` inner (which `Slab` isn't); the in-tree
/// alias swaps the nesting so the `OsBacked`-requiring wrappers sit
/// on the OS-mapped side and `Slab` consumes the protected region
/// from the outside.
///
/// For freelist MAC protection (against forged links from a heap
/// disclosure), parameterize with `M`:
///
/// ```rust,ignore
/// // Requires `--features siphasher`.
/// use forge_alloc::{HardenedSlab, SipHashMAC};
/// type ClaimPool = HardenedSlab<u64, SipHashMAC>;
/// ```
///
/// On aarch64 with the `pac-stub` (and eventually `pac`) feature
/// enabled, `PacMAC` is available as the parameter.
#[cfg(feature = "std")]
pub type HardenedSlab<T, M = NoProtection> = Slab<T, GuardPage<SplitMetadata<MmapBacked>>, M>;
