//! # forge-alloc
//!
//! Composable memory allocator primitives. Snap together at the type level to
//! produce application-specific allocators with zero runtime dispatch overhead,
//! compile-time-enforced guarantees, and pay-for-what-you-use security
//! hardening.
//!
//! `forge-alloc` bundles the three implementation layers as modules:
//!
//! - **backing** — Layer 1: raw memory sources — [`InlineBacked`],
//!   [`StaticBacked`], [`HeapBytes`], [`MmapBacked`], [`HugePageBacked`],
//!   [`System`].
//! - **layout** — Layer 2: structure over a backing — [`BumpArena`], [`Slab`],
//!   [`SizeClassed`], [`StackAlloc`], [`WithFallback`], and more.
//! - **hardening** — Layer 3: security & observability wrappers — [`Canary`],
//!   [`PoisonOnFree`], [`Quarantine`], [`GuardPage`], [`Statistics`], and more.
//!
//! Conformance helpers for downstream `Allocator` / `FixedRange`
//! implementers live in [`testing`] (re-exported from
//! [`forge-alloc-core`]).
//!
//! The trait contracts ([`Allocator`], [`Deallocator`], [`OsBacked`],
//! [`FixedRange`], [`FreelistProtection`], [`NonZeroLayout`], and the rest)
//! live in the companion [`forge-alloc-core`] crate and are re-exported here,
//! so a single `forge-alloc` dependency gives you the whole surface.
//!
//! [`forge-alloc-core`]: https://docs.rs/forge-alloc-core
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
//! See [`ARCHITECTURE.md`] and [`COMPOSITION_RECIPES.md`] for the design
//! and caller-facing composition recipes.
//!
//! [`ARCHITECTURE.md`]: https://github.com/dmaesj/forge-alloc/blob/main/docs/ARCHITECTURE.md
//! [`COMPOSITION_RECIPES.md`]: https://github.com/dmaesj/forge-alloc/blob/main/docs/COMPOSITION_RECIPES.md

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

mod backing;
mod hardening;
mod layout;

pub use forge_alloc_core::*;

#[doc(inline)]
pub use forge_alloc_core::testing;

#[doc(inline)]
pub use backing::{HeapBytes, InlineBacked, StaticBacked, MAX_ALIGN};

#[cfg(feature = "std")]
#[doc(inline)]
pub use backing::System;

#[cfg(all(feature = "std", any(unix, windows)))]
#[doc(inline)]
pub use backing::{
    mmap_clear_last_os_error, mmap_last_os_error, mmap_record_os_error, page_size, HugePageBacked,
    LockedMmapBacked, MmapBacked, MmapFlags,
};

#[doc(inline)]
pub use layout::{
    BumpArena, BumpDeallocator, GenerationInt, GenerationalSlab, Handle, SizeClassed, Slab,
    StackAlloc, WithFallback, DEFAULT_CLASS_SIZES_8,
};

#[cfg(target_has_atomic = "ptr")]
#[doc(inline)]
pub use layout::SharedBumpArena;

#[cfg(feature = "std")]
#[doc(inline)]
pub use layout::{BatchPolicy, SlabOwner, SlabRemote, ADAPTIVE_COOLDOWN_TICKS, ADAPTIVE_LEVELS};

// ExtendableSlab is gated on `unix || windows` because it depends on
// `MmapBacked` for its growable segments.
#[cfg(all(feature = "std", any(unix, windows)))]
#[doc(inline)]
pub use layout::ExtendableSlab;

#[doc(inline)]
pub use hardening::{
    AllocStats, CacheJitter, Canary, Faulty, PoisonOnFree, Quarantine, Statistics, ZeroizeOnFree,
    DEFAULT_POISON,
};

// These hardening wrappers require libc / Win32 syscalls and so are
// gated on `unix || windows` in addition to `feature = "std"`.
#[cfg(all(feature = "std", any(unix, windows)))]
#[doc(inline)]
pub use hardening::SplitMetadata;

#[cfg(all(feature = "std", any(unix, windows)))]
#[doc(inline)]
pub use hardening::{
    current_numa_node, default_huge_page_size, GuardPage, HugePageAligned, NodeSet, NumaLocal,
    NumaPolicy,
};

#[cfg(target_has_atomic = "ptr")]
#[doc(inline)]
pub use hardening::{
    FnHandler, NullHandler, Watermark, WatermarkEvent, WatermarkHandler, WatermarkLevel,
    WatermarkThresholds,
};

#[cfg(all(target_has_atomic = "ptr", feature = "std"))]
#[doc(inline)]
pub use hardening::LogHandler;

// ============================================================================
// Convenience type aliases for recommended compositions
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
#[cfg(all(feature = "std", any(unix, windows)))]
pub type HardenedSlab<T, M = NoProtection> = Slab<T, GuardPage<SplitMetadata<MmapBacked>>, M>;

/// The [`HardenedSlab`] security stack over **RAM-locked, core-dump-excluded**
/// memory, wrapped in **non-elidable scrub-on-free** — the recommended
/// composition for **cryptographic key material** ("crypto allocator").
///
/// Expansion:
/// `ZeroizeOnFree<Slab<T, GuardPage<SplitMetadata<LockedMmapBacked>>, M>>`.
///
/// It adds two crypto-specific guarantees on top of [`HardenedSlab`]:
/// - **No swap:** the data region is [`LockedMmapBacked`] — `mlock` /
///   `VirtualLock` pins the secret pages in RAM so they never page to swap.
///   Construction **fails closed** if the lock cannot be taken
///   (`RLIMIT_MEMLOCK` / missing privilege), never silently leaving secrets
///   swappable.
/// - **Core-dump exclusion (Linux, best-effort):** on Linux `MADV_DONTDUMP` is
///   applied after the lock so the secret pages are excluded from core dumps.
///   It is best-effort — a `madvise` failure is ignored (the lock is the hard
///   guarantee). **There is no equivalent on Windows / other Unix** — set
///   `RLIMIT_CORE = 0` (or the platform equivalent) if you require it there.
/// - **Scrub on free:** [`ZeroizeOnFree`] volatile-zeroes a freed slot so a
///   secret does not linger in RAM after `deallocate`. It sits outermost so the
///   scrub runs before the slab writes its freelink.
///
/// Plus everything [`HardenedSlab`] already provides: guard pages trap a linear
/// overflow into/out of the secret region with a fault, [`SplitMetadata`]
/// isolates allocator metadata into a separate, cache-line-disjoint mapping, and
/// the optional freelist MAC `M` (e.g. `SipHashMAC` under `--features siphasher`)
/// detects forged links. Note the slab's freelist links live **inline** in freed
/// slots (the locked region), not in the metadata mapping; because the scrub
/// runs first, after a free those bytes hold link data, never a secret.
///
/// **Construction** (build the inner hardened slab over a locked backing, then
/// wrap in [`ZeroizeOnFree`]):
/// ```rust,ignore
/// use forge_alloc::{
///     CryptoSlab, GuardPage, LockedMmapBacked, page_size, Slab, SplitMetadata, ZeroizeOnFree,
/// };
/// let locked  = LockedMmapBacked::new(32 * 1024)?;          // fails closed if it can't lock
/// let split   = SplitMetadata::new(locked, 16 * 1024)?;
/// let guarded = GuardPage::new(split, page_size())?;
/// let pool: CryptoSlab<[u8; 64]> = ZeroizeOnFree::new(Slab::new(64, guarded)?);
/// ```
///
/// **Threat-model boundary** (see [`LockedMmapBacked`] for the full list): the
/// lock prevents *swap*, not hibernation (suspend-to-disk writes all of RAM),
/// `fork()` COW, or `ptrace`. Only the **data** region is locked — the
/// [`SplitMetadata`] metadata region and the freelist-MAC key (held in the slab
/// struct) are unlocked, which is intended (neither holds the user's secret).
#[cfg(all(feature = "std", any(unix, windows)))]
pub type CryptoSlab<T, M = NoProtection> =
    ZeroizeOnFree<Slab<T, GuardPage<SplitMetadata<LockedMmapBacked>>, M>>;
