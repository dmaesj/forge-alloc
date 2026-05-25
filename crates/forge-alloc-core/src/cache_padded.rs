//! `CachePadded<T>` — target-aware cache-line alignment.
//!
//! Wrap a contended atomic in `CachePadded` to keep it from sharing a
//! cache line with neighboring fields. Without this, two atomics on the
//! same line cause cache-coherency ping-pong between cores even when the
//! threads writing to them touch logically independent data — "false
//! sharing." The L1-to-L1 round trip to re-fetch an invalidated line is
//! tens of nanoseconds, catastrophic in a tight allocate-deallocate loop.
//!
//! The alignment used is per-target:
//!
//! - `x86_64`, `aarch64`, `powerpc64`: **128 bytes**. x86_64's L1 line is
//!   64 bytes but the adjacent-line prefetcher pulls cache lines in
//!   pairs, so a 64-byte pad still allows false sharing across the
//!   prefetched neighbor; 128 closes that gap. Apple Silicon (M-series)
//!   AArch64 uses 128-byte coherency granularity natively.
//! - `arm`, `mips`, `mips64`, `sparc`, `hexagon`: 32 bytes.
//! - `m68k`: 16 bytes.
//! - `s390x`: 256 bytes.
//! - Anything else: 64 bytes (the historical x86 line size).
//!
//! The cfg matrix mirrors `crossbeam_utils::CachePadded`'s choices so
//! benchmarks and reasoning carry across crates. We inline the
//! definition rather than depending on `crossbeam_utils` to keep
//! `forge-alloc` dependency-free at the runtime layer.

/// Wraps a value so it occupies a whole cache line, preventing the
/// neighboring fields in a struct from being invalidated when the wrapped
/// atomic is written by another core.
///
/// `CachePadded<T>` has the same size as `T` rounded up to the target's
/// cache-line size; for an `AtomicUsize` on x86_64 / AArch64 that means
/// the wrapped value occupies 128 bytes total. Deref / DerefMut hide the
/// padding from call sites so existing code reading `padded.load(...)`
/// stays unchanged.
///
/// # Example
///
/// ```
/// use forge_alloc_core::CachePadded;
/// use core::sync::atomic::{AtomicUsize, Ordering};
///
/// struct Stats {
///     hits: CachePadded<AtomicUsize>,
///     misses: CachePadded<AtomicUsize>,
/// }
///
/// let s = Stats {
///     hits: CachePadded::new(AtomicUsize::new(0)),
///     misses: CachePadded::new(AtomicUsize::new(0)),
/// };
/// s.hits.fetch_add(1, Ordering::Relaxed);
/// ```
#[cfg_attr(
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64"
    ),
    repr(align(128))
)]
#[cfg_attr(
    any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "sparc",
        target_arch = "hexagon"
    ),
    repr(align(32))
)]
#[cfg_attr(target_arch = "m68k", repr(align(16)))]
#[cfg_attr(target_arch = "s390x", repr(align(256)))]
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "sparc",
        target_arch = "hexagon",
        target_arch = "m68k",
        target_arch = "s390x"
    )),
    repr(align(64))
)]
#[repr(C)]
#[derive(Debug, Default)]
pub struct CachePadded<T>(T);

impl<T> CachePadded<T> {
    /// Wrap a value with cache-line alignment padding.
    #[inline]
    pub const fn new(v: T) -> Self {
        Self(v)
    }

    /// Unwrap, returning the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> core::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> core::ops::DerefMut for CachePadded<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

/// The cache-line alignment used by [`CachePadded`] on this target. Surfaced
/// so dependent crates and `const _: () = assert!(...)` layout pins can
/// reference the same value the wrapper itself uses.
///
/// Per-target values:
///
/// - **`x86_64`, `aarch64`, `powerpc64`**: 128 bytes (covers Apple
///   Silicon's 128-byte coherency and x86_64's adjacent-line prefetch
///   pair).
/// - **`arm`, `mips`, `mips64`, `sparc`, `hexagon`**: 32 bytes.
/// - **`m68k`**: 16 bytes.
/// - **`s390x`**: 256 bytes.
/// - Anything else: 64 bytes (historical x86 default).
///
/// # Use in a layout pin
///
/// Combine with [`core::mem::offset_of!`] to build a compile-time
/// assertion that two contended fields in your own struct never share
/// a cache line. forge-alloc uses this pattern internally on its own
/// `AllocStats`, `Watermark`, `SharedBumpArena`, and `SlabInner` structs
/// to lock in the cache-line separation invariants.
///
/// ```
/// use forge_alloc_core::{CachePadded, CACHE_LINE};
/// use core::sync::atomic::AtomicUsize;
///
/// struct MyStats {
///     hits: CachePadded<AtomicUsize>,
///     misses: CachePadded<AtomicUsize>,
/// }
///
/// const _: () = {
///     use core::mem::offset_of;
///     assert!(
///         offset_of!(MyStats, hits) / CACHE_LINE
///             != offset_of!(MyStats, misses) / CACHE_LINE
///     );
/// };
/// ```
///
/// The assertion fires at compile time. If a future refactor reorders
/// the struct fields or drops the `CachePadded` wrappers, the build
/// fails with a pointer at the affected struct rather than silently
/// regressing throughput under multi-thread contention.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64"
))]
pub const CACHE_LINE: usize = 128;

/// See the primary [`CACHE_LINE`] doc for the per-target table; this
/// branch covers small ARM / MIPS / SPARC / Hexagon at 32 bytes.
#[cfg(any(
    target_arch = "arm",
    target_arch = "mips",
    target_arch = "mips64",
    target_arch = "sparc",
    target_arch = "hexagon"
))]
pub const CACHE_LINE: usize = 32;

/// See the primary [`CACHE_LINE`] doc for the per-target table; this
/// branch covers m68k at 16 bytes.
#[cfg(target_arch = "m68k")]
pub const CACHE_LINE: usize = 16;

/// See the primary [`CACHE_LINE`] doc for the per-target table; this
/// branch covers s390x at 256 bytes.
#[cfg(target_arch = "s390x")]
pub const CACHE_LINE: usize = 256;

/// See the primary [`CACHE_LINE`] doc for the per-target table; this
/// fallback branch covers any architecture not explicitly listed and
/// defaults to 64 bytes (historical x86 L1 line size).
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "arm",
    target_arch = "mips",
    target_arch = "mips64",
    target_arch = "sparc",
    target_arch = "hexagon",
    target_arch = "m68k",
    target_arch = "s390x"
)))]
pub const CACHE_LINE: usize = 64;

// The wrapper itself is correctly aligned: any wrapped value `T` is
// aligned to at least the cache line and the struct's stride is a
// multiple of it.
const _: () = {
    assert!(core::mem::align_of::<CachePadded<u8>>() == CACHE_LINE);
    assert!(core::mem::size_of::<CachePadded<u8>>() == CACHE_LINE);
};

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn alignment_matches_target_constant() {
        assert_eq!(
            core::mem::align_of::<CachePadded<AtomicUsize>>(),
            CACHE_LINE
        );
    }

    #[test]
    fn size_is_at_least_one_line() {
        assert!(core::mem::size_of::<CachePadded<AtomicUsize>>() >= CACHE_LINE);
    }

    #[test]
    fn deref_exposes_inner_methods() {
        let p = CachePadded::new(AtomicUsize::new(0));
        p.fetch_add(1, Ordering::Relaxed);
        assert_eq!(p.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn two_instances_land_on_different_lines() {
        // Two `CachePadded<AtomicUsize>` in a struct must have offsets
        // that differ by at least one cache line. Static-assert form
        // would be ideal but requires `offset_of!` on a generic field,
        // which is stable. Use a runtime check here as a smoke test.
        struct Pair {
            a: CachePadded<AtomicUsize>,
            b: CachePadded<AtomicUsize>,
        }
        let pair = Pair {
            a: CachePadded::new(AtomicUsize::new(0)),
            b: CachePadded::new(AtomicUsize::new(0)),
        };
        let a_addr = (&pair.a as *const _ as usize) / CACHE_LINE;
        let b_addr = (&pair.b as *const _ as usize) / CACHE_LINE;
        assert_ne!(
            a_addr, b_addr,
            "two CachePadded fields must occupy different cache lines",
        );
    }
}
