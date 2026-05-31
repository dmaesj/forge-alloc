//! `NumaLocal<I>` — bind an [`OsBacked`] allocator's memory range to one
//! or more NUMA nodes.
//!
//! The wrapper calls `mbind()` once at construction to apply a NUMA
//! placement policy to the inner allocator's entire region. Subsequent
//! page faults that touch the range allocate physical pages on the
//! chosen node(s) — this is the only point where the kernel actually
//! decides physical placement; setting policy after pages are faulted
//! has no effect.
//!
//! # Platform support
//!
//! - **Linux**: `mbind` is invoked via `libc::syscall(SYS_mbind, …)`.
//!   Failure (kernel rejects, capability missing) is captured into
//!   `crate::backing::mmap_last_os_error()` and the construction returns
//!   `AllocError` — refuse silently-degraded NUMA placement.
//! - **macOS / Apple Silicon**: UMA platform with no NUMA semantics.
//!   `NumaLocal` is a no-op; the wrapper compiles to a direct pass-
//!   through.
//! - **Windows / other**: no `mbind` equivalent that operates on an
//!   already-mapped region. `NumaLocal::new` returns the inner
//!   unchanged with the policy stored but unenforced; production
//!   Windows NUMA work belongs to `MmapBacked::with_numa_node` at
//!   MAP-time (deferred to a future release).
//!
//! `LocalAtRequest` — re-bind on every backing request — is **not**
//! implemented in v0.1. The wrap-once model doesn't fit per-allocate
//! dispatch, and most NUMA-sensitive workloads are well-served by
//! a one-shot bind at construction with thread-local slabs at the
//! application layer.
//!
//! See `docs/ARCHITECTURE.md` for design context.

use core::ptr::NonNull;

use forge_alloc_core::{
    AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout, OsBacked, ProtectFlags,
};

/// NUMA placement policy. v0.1 accepts an explicit node set rather
/// than dispatching against the calling thread's node — supply
/// `current_numa_node()` if you want the local-at-construction
/// behaviour.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumaPolicy {
    /// `MPOL_BIND` — pages must come from the listed nodes; if no
    /// node has free memory, allocation fails. Maximum strictness.
    Bind(NodeSet),
    /// `MPOL_PREFERRED` — a soft hint; falls back to other nodes
    /// under memory pressure.
    Preferred(u32),
    /// `MPOL_INTERLEAVE` — round-robin pages across the listed nodes.
    /// Bandwidth-bound workloads benefit; latency-bound ones suffer.
    Interleaved(NodeSet),
}

/// Compact set of NUMA node IDs (up to 64 nodes). Built directly into
/// a Linux nodemask word at `mbind` time. Bigger systems need a
/// dynamic representation; that's not yet shipped.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeSet {
    mask: u64,
}

impl NodeSet {
    /// Empty set.
    pub const fn empty() -> Self {
        Self { mask: 0 }
    }

    /// Single-node set.
    pub const fn single(node: u32) -> Option<Self> {
        if node >= 64 {
            return None;
        }
        Some(Self { mask: 1u64 << node })
    }

    /// Add a node. Returns `None` if `node >= 64`.
    pub const fn with(mut self, node: u32) -> Option<Self> {
        if node >= 64 {
            return None;
        }
        self.mask |= 1u64 << node;
        Some(self)
    }

    /// Bit-mask view (low 64 nodes).
    #[inline]
    pub const fn mask(&self) -> u64 {
        self.mask
    }

    /// Whether the set is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.mask == 0
    }

    /// Highest node id set, plus one. `0` if the set is empty.
    ///
    /// Informational only — the `mbind` path always passes
    /// [`mbind_maxnode`](Self::mbind_maxnode) (a constant 64), not this
    /// value. Exposed for callers that want to know the occupied range.
    #[inline]
    pub fn max_node_plus_one(&self) -> u32 {
        if self.mask == 0 {
            0
        } else {
            64 - self.mask.leading_zeros()
        }
    }

    /// `maxnode` value to pass to `mbind`. `mbind`'s `maxnode` is the
    /// *number of bits* in the nodemask; the kernel reads
    /// `ceil(maxnode / bits_per_long)` words. We hand the kernel a single
    /// `u64`, so `maxnode = 64` makes it read exactly those 8 bytes —
    /// independent of how many low bits are actually set.
    #[inline]
    pub fn mbind_maxnode(&self) -> u32 {
        64
    }
}

// Linux `mbind` mode constants (from <linux/mempolicy.h>). Defined at module
// scope — not just inside the Linux `apply_policy` — so the policy → syscall-
// args mapping below is unit-testable on every platform.
const MPOL_PREFERRED: i32 = 1;
const MPOL_BIND: i32 = 2;
const MPOL_INTERLEAVE: i32 = 3;

/// Pure mapping from a [`NumaPolicy`] to the `mbind` arguments
/// `(mode, nodemask, maxnode)`.
///
/// Returns `Err(AllocError)` for an invalid policy — an empty `Bind` /
/// `Interleaved` node set, or a `Preferred` node id `>= 64`. Because this is
/// platform-independent, the nodemask construction (and the rejection of
/// invalid policies) is exercised by tests on *every* host, not only Linux —
/// the syscall itself is no-op'd off-Linux, so without this split the bitmask
/// build would be untested on the CI host.
fn mbind_args(policy: &NumaPolicy) -> Result<(i32, u64, u32), AllocError> {
    match policy {
        NumaPolicy::Bind(s) | NumaPolicy::Interleaved(s) if s.is_empty() => Err(AllocError),
        NumaPolicy::Bind(s) => Ok((MPOL_BIND, s.mask(), s.mbind_maxnode())),
        NumaPolicy::Interleaved(s) => Ok((MPOL_INTERLEAVE, s.mask(), s.mbind_maxnode())),
        NumaPolicy::Preferred(n) => {
            let s = NodeSet::single(*n).ok_or(AllocError)?;
            Ok((MPOL_PREFERRED, s.mask(), s.mbind_maxnode()))
        }
    }
}

/// NumaLocal wrapper.
pub struct NumaLocal<I: OsBacked> {
    inner: I,
    policy: NumaPolicy,
}

impl<I: OsBacked> NumaLocal<I> {
    /// Wrap and apply `policy` to the inner allocator's region.
    ///
    /// Returns `Err(AllocError)` if the platform supports NUMA and the
    /// kernel rejects the bind (insufficient capability, invalid node
    /// id, no memory available on the bound nodes). On unsupported
    /// platforms (macOS, Windows, other) returns `Ok` without binding
    /// — caller can inspect with [`policy`](Self::policy) but the
    /// region's physical placement is the kernel's default.
    pub fn new(inner: I, policy: NumaPolicy) -> Result<Self, AllocError> {
        // Validate the policy on EVERY platform (not just Linux) so an empty
        // Bind/Interleaved set or an out-of-range `Preferred` node is rejected
        // uniformly — previously `Preferred(huge)` was accepted off-Linux
        // because the only range check lived inside the Linux syscall path.
        let args = mbind_args(&policy)?;
        apply_policy(&inner, args)?;
        Ok(Self { inner, policy })
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }

    /// Active policy.
    #[inline]
    pub fn policy(&self) -> &NumaPolicy {
        &self.policy
    }
}

unsafe impl<I: OsBacked> Deallocator for NumaLocal<I> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // SAFETY: forwarded.
        unsafe { self.inner.deallocate(ptr, layout) }
    }
}

unsafe impl<I: OsBacked> Allocator for NumaLocal<I> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner.allocate(layout)
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        self.inner.capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        self.inner.corruption_events()
    }
}

unsafe impl<I: OsBacked> OsBacked for NumaLocal<I> {
    #[inline]
    fn base_ptr(&self) -> NonNull<u8> {
        self.inner.base_ptr()
    }

    #[inline]
    fn region_size(&self) -> usize {
        self.inner.region_size()
    }

    #[inline]
    unsafe fn release_pages(&self, ptr: NonNull<u8>, size: usize) {
        // SAFETY: forwarded; caller's contract preserved.
        unsafe { self.inner.release_pages(ptr, size) }
    }

    #[inline]
    unsafe fn protect(&self, ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
        // SAFETY: forwarded.
        unsafe { self.inner.protect(ptr, size, flags) }
    }
}

impl<I: OsBacked + FixedRange> FixedRange for NumaLocal<I> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.inner.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Pass-through forward so a `commit`-aware consumer reaches the inner
    /// backing when this wrapper sits over a `lazy_commit` `MmapBacked`.
    #[inline]
    fn commit(&self, offset: usize, len: usize) -> Result<(), AllocError> {
        self.inner.commit(offset, len)
    }
}

// ============================================================================
// Platform glue: apply_policy()
// ============================================================================

#[cfg(target_os = "linux")]
fn apply_policy<I: OsBacked>(inner: &I, args: (i32, u64, u32)) -> Result<(), AllocError> {
    let (mode, mask, maxnode) = args;
    let base = inner.base_ptr().as_ptr() as *mut libc::c_void;
    let size = inner.region_size();
    // mbind's `nodemask` is an array of unsigned longs (bitmap).
    // For up to 64 nodes a single u64 suffices.
    let nodemask: u64 = mask;
    // SAFETY: the FFI signature for SYS_mbind matches the kernel's
    // ABI: (unsigned long start, unsigned long len, unsigned long mode,
    // const unsigned long *nodemask, unsigned long maxnode, unsigned flags).
    // `mode` is passed as `c_ulong` to match the kernel's `unsigned long mode`
    // (values 1–3, but the width must match the ABI, not just the value).
    let rc = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            base,
            size as libc::c_ulong,
            mode as libc::c_ulong,
            &nodemask as *const u64,
            // mbind's `maxnode` is the nodemask width in bits.
            maxnode as libc::c_ulong,
            0u32 as libc::c_uint,
        )
    };
    if rc != 0 {
        // Capture errno into the cross-crate thread-local slot so callers
        // reading `crate::backing::mmap_last_os_error()` after a failing
        // `NumaLocal::new(...)` see the actual mbind errno (EINVAL for a
        // bad node set, EPERM for missing CAP_SYS_NICE, ESRCH for an
        // off-line node, …) rather than `None` or stale state.
        crate::backing::mmap_record_os_error();
        return Err(AllocError);
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn apply_policy<I: OsBacked>(_inner: &I, _args: (i32, u64, u32)) -> Result<(), AllocError> {
    // macOS, Windows, BSD, other Unix: no equivalent operation on an
    // already-mapped region. Return Ok so the wrapper compiles and the
    // type is still useful as a marker / future-extension point. (Policy
    // validity was already checked by `mbind_args` in `new`, so an invalid
    // policy is rejected here too, not only on Linux.)
    Ok(())
}

/// Best-effort detect of the calling thread's NUMA node.
///
/// - **Linux**: uses `sched_getcpu()` and `/sys/devices/system/node/...`
///   to map CPU → node. Returns `None` on lookup failure or non-NUMA
///   systems (single-node WSL, containers without sysfs).
/// - **Other**: returns `None` — supply node IDs explicitly via the
///   `NumaPolicy` constructor instead.
#[cfg(target_os = "linux")]
#[must_use]
pub fn current_numa_node() -> Option<u32> {
    // Use the getcpu(2) syscall directly. Signature: (cpu, node,
    // tcache). We only need the node out-pointer.
    let mut node: libc::c_uint = 0;
    // SAFETY: getcpu writes through the supplied non-null out-pointer
    // and returns 0 on success / -1 on failure (errno set). We pass
    // null for cpu and tcache — both are documented as optional.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_getcpu,
            core::ptr::null_mut::<libc::c_uint>(),
            &mut node as *mut libc::c_uint,
            core::ptr::null_mut::<libc::c_void>(),
        )
    };
    if rc != 0 {
        None
    } else {
        Some(node as u32)
    }
}

/// Best-effort detect of the calling thread's NUMA node. On
/// non-Linux platforms this always returns `None` — callers should
/// supply node IDs explicitly via [`NumaPolicy`].
#[cfg(not(target_os = "linux"))]
#[must_use]
pub fn current_numa_node() -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::MmapBacked;

    #[test]
    fn nodeset_single() {
        let s = NodeSet::single(3).unwrap();
        assert_eq!(s.mask(), 0b1000);
        assert_eq!(s.max_node_plus_one(), 4);
    }

    #[test]
    fn nodeset_with() {
        let s = NodeSet::single(0)
            .unwrap()
            .with(2)
            .unwrap()
            .with(5)
            .unwrap();
        assert_eq!(s.mask(), 0b100101);
        assert_eq!(s.max_node_plus_one(), 6);
    }

    #[test]
    fn nodeset_rejects_overflow() {
        assert!(NodeSet::single(64).is_none());
        assert!(NodeSet::single(100).is_none());
        let s = NodeSet::empty();
        assert!(s.with(64).is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn empty_bind_rejected() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        let res = NumaLocal::new(inner, NumaPolicy::Bind(NodeSet::empty()));
        assert!(res.is_err());
    }

    /// On WSL / single-node systems, mbind with node 0 should succeed.
    /// On macOS/Windows it's a no-op and also succeeds.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn bind_to_node_zero_succeeds() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        let s = NodeSet::single(0).unwrap();
        // On Linux this calls mbind; on other platforms it's a no-op.
        // Either way, succeeds.
        let res = NumaLocal::new(inner, NumaPolicy::Bind(s));
        assert!(
            res.is_ok(),
            "expected mbind(MPOL_BIND, [0]) to succeed on any host"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn interleaved_succeeds() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        let s = NodeSet::single(0).unwrap();
        let res = NumaLocal::new(inner, NumaPolicy::Interleaved(s));
        assert!(res.is_ok());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn preferred_succeeds() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        let res = NumaLocal::new(inner, NumaPolicy::Preferred(0));
        assert!(res.is_ok());
    }

    // The following tests exercise the policy → mbind-args mapping (mode +
    // nodemask + maxnode) directly. They run on EVERY platform, including the
    // Windows CI host where the syscall path is a no-op — so the bitmask
    // construction is no longer untested off-Linux.

    #[test]
    fn mbind_args_bind_builds_nodemask() {
        let s = NodeSet::single(0).unwrap().with(3).unwrap();
        let (mode, mask, maxnode) = mbind_args(&NumaPolicy::Bind(s)).unwrap();
        assert_eq!(mode, MPOL_BIND);
        assert_eq!(mask, 0b1001);
        assert_eq!(maxnode, 64);
    }

    #[test]
    fn mbind_args_interleaved_builds_nodemask() {
        let s = NodeSet::single(1).unwrap();
        let (mode, mask, maxnode) = mbind_args(&NumaPolicy::Interleaved(s)).unwrap();
        assert_eq!(mode, MPOL_INTERLEAVE);
        assert_eq!(mask, 0b10);
        assert_eq!(maxnode, 64);
    }

    #[test]
    fn mbind_args_preferred_single_node() {
        let (mode, mask, maxnode) = mbind_args(&NumaPolicy::Preferred(2)).unwrap();
        assert_eq!(mode, MPOL_PREFERRED);
        assert_eq!(mask, 0b100);
        assert_eq!(maxnode, 64);
    }

    #[test]
    fn mbind_args_rejects_empty_and_out_of_range() {
        assert!(mbind_args(&NumaPolicy::Bind(NodeSet::empty())).is_err());
        assert!(mbind_args(&NumaPolicy::Interleaved(NodeSet::empty())).is_err());
        assert!(mbind_args(&NumaPolicy::Preferred(64)).is_err());
        assert!(mbind_args(&NumaPolicy::Preferred(9999)).is_err());
    }

    /// `Preferred` with an out-of-range node must be rejected by `new` on
    /// every platform — previously it was accepted off-Linux because the only
    /// range check lived inside the Linux syscall path.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn preferred_out_of_range_node_rejected_uniformly() {
        let inner = MmapBacked::new(64 * 1024).unwrap();
        let res = NumaLocal::new(inner, NumaPolicy::Preferred(9999));
        assert!(res.is_err(), "out-of-range Preferred node must be rejected");
    }

    #[test]
    fn current_numa_node_returns_some_or_none() {
        // The function must not panic on any supported platform; the
        // exact answer is host-dependent.
        let _ = current_numa_node();
    }
}
