//! # hardening
//!
//! Layer 3 hardening wrappers for the forge-alloc family. Each wrapper
//! decorates an inner [`forge_alloc_core::Allocator`] with one specific protection or
//! observability behavior. All costs are paid only when the wrapper is composed
//! — wrappers absent from the type pay zero overhead.
//!
//! Ships:
//!
//! - [`Canary<I>`] — pre/post sentinel bytes on every allocation; verifies
//!   integrity on deallocate; zeroes the canary words on free so the
//!   per-process seed doesn't linger in freed memory. Detects linear
//!   overflows.
//! - [`PoisonOnFree<I>`] — overwrites freed memory with a configurable pattern.
//!   Protects against UAF read disclosure.
//! - [`Quarantine<I, EPOCHS>`] — holds freed blocks in a ring for `EPOCHS`
//!   cycles before returning them to the inner allocator for reuse. Delays
//!   slot reuse to widen the UAF / type-confusion attack window.
//! - [`Statistics<I>`] — atomic counters (alloc count, peak usage, failures).
//!   Zero overhead for any unwrapped allocator.
//! - [`Watermark<I, H>`] — fires callbacks at configurable utilization
//!   thresholds (warn / critical / oom).
//! - [`CacheJitter<I>`] — randomized per-allocation displacement (with a
//!   per-instance keyed MAC on the displacement header) to spread metadata
//!   across cache associativity sets.
//! - [`Faulty<I, P>`] — **test/debug only** fault-injection wrapper:
//!   forces allocations to fail per a [`forge_alloc_core::AllocFaultPolicy`],
//!   making every allocator's OOM `Err` path reachable from tests,
//!   proptest, fuzz, MIRI, and Kani.
//!
//! `std`-only:
//!
//! - [`GuardPage<I>`] — unmapped pages on either side of the inner region;
//!   linear overflow / underflow traps with SIGSEGV.
//! - [`HugePageAligned<I>`] — enforces 2 MiB (32 MiB on Apple Silicon)
//!   alignment so the OS can promote the inner region to a huge page.
//! - [`NumaLocal<I>`] — applies a NUMA placement policy via `mbind()` on
//!   Linux; no-op on macOS / Windows.
//! - [`SplitMetadata<I>`] — wraps the inner region with a separate metadata
//!   `MmapBacked`; data and metadata live at unrelated virtual addresses.

mod cache_jitter;
mod canary;
mod faulty;
mod poison;
mod quarantine;
mod statistics;
mod watermark;

#[cfg(feature = "std")]
mod guard_page;
#[cfg(feature = "std")]
mod huge_page;
#[cfg(feature = "std")]
mod numa;
#[cfg(feature = "std")]
mod split_metadata;

pub use cache_jitter::CacheJitter;
pub use canary::Canary;
pub use faulty::Faulty;
pub use poison::{PoisonOnFree, DEFAULT_POISON};
pub use quarantine::Quarantine;
pub use statistics::{AllocStats, CachePadded, Statistics};

#[cfg(feature = "std")]
pub use guard_page::GuardPage;
#[cfg(feature = "std")]
pub use huge_page::{default_huge_page_size, HugePageAligned};
#[cfg(feature = "std")]
pub use numa::{current_numa_node, NodeSet, NumaLocal, NumaPolicy};
#[cfg(feature = "std")]
pub use split_metadata::SplitMetadata;
#[cfg(feature = "std")]
pub use watermark::LogHandler;
pub use watermark::{
    FnHandler, NullHandler, Watermark, WatermarkEvent, WatermarkHandler, WatermarkLevel,
    WatermarkThresholds,
};
