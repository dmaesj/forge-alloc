//! `OsBacked` — allocators that manage OS-level memory mappings, and
//! `ProtectFlags` for changing region protection.
//!
//! Required by `GuardPage`, `HugePageAligned`, `NumaLocal`, `MpkBacked`. Not
//! implemented by `InlineBacked` or `System` — so `GuardPage<InlineBacked<N>>`
//! is rejected at compile time, not at runtime.

use core::ptr::NonNull;

use super::allocator::Allocator;

/// Memory protection bits passed to [`OsBacked::protect`].
///
/// `#[non_exhaustive]` so future hardware-protection primitives can
/// add bits (`PROT_MTE` tag enable, MPK key index, `PROT_GROWSDOWN`) without
/// breaking downstream callers. Construct via the provided
/// [`NONE`](Self::NONE) / [`RW`](Self::RW) / [`READ`](Self::READ) /
/// [`RX`](Self::RX) constants — never with a struct literal.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProtectFlags {
    /// Region is readable.
    pub read: bool,
    /// Region is writable.
    pub write: bool,
    /// Region is executable.
    pub exec: bool,
}

impl ProtectFlags {
    /// No access. Reads and writes fault.
    pub const NONE: Self = Self {
        read: false,
        write: false,
        exec: false,
    };
    /// Read + write, no execute. The standard data-region setting.
    pub const RW: Self = Self {
        read: true,
        write: true,
        exec: false,
    };
    /// Read only.
    pub const READ: Self = Self {
        read: true,
        write: false,
        exec: false,
    };
    /// Read + execute, no write. The standard code-region setting.
    pub const RX: Self = Self {
        read: true,
        write: false,
        exec: true,
    };
}

/// Allocators backed by OS-managed virtual memory.
///
/// Provides hooks that hardening wrappers need: knowing the base/size of the
/// region (to install guard pages around it), releasing physical pages back
/// to the OS without dropping the virtual reservation (for
/// `madvise(DONTNEED)`-style purging), and changing protection bits.
///
/// # Safety
///
/// Trait-level invariants implementors must uphold:
///
/// 1. [`base_ptr`](Self::base_ptr) returns a stable, non-null pointer to
///    the first byte of the OS-managed region and does not change for the
///    lifetime of the allocator.
/// 2. [`region_size`](Self::region_size) returns the *accurate* page-rounded
///    length of the region. The half-open range
///    `[base_ptr, base_ptr + region_size)` must be wholly inside one
///    contiguous OS mapping owned by this allocator.
/// 3. `release_pages` and `protect` must reject (no-op or panic — at the
///    implementor's discretion, but not silently truncate) calls whose
///    `[ptr, ptr + size)` is not strictly inside the region. They must
///    not affect memory outside the region.
/// 4. The region is allocated for the lifetime of `self` (i.e. only the
///    `Drop` impl unmaps it).
pub unsafe trait OsBacked: Allocator {
    /// First byte of the OS-managed region.
    fn base_ptr(&self) -> NonNull<u8>;

    /// Length in bytes of the OS-managed region.
    fn region_size(&self) -> usize;

    /// Release the physical pages backing `[ptr, ptr + size)` to the OS
    /// while keeping the virtual address range reserved.
    ///
    /// Maps to `madvise(MADV_DONTNEED)` on Linux, `madvise(MADV_FREE)` on
    /// macOS / other BSD-flavoured Unix, and `VirtualAlloc(MEM_RESET)` on
    /// Windows.
    ///
    /// Behaviour post-release: subsequent reads of the released range
    /// *do not fault*, but their contents vary by platform:
    ///
    /// - **Linux** (`MADV_DONTNEED` on a private anonymous mapping):
    ///   reads return zero-filled pages.
    /// - **macOS / BSD** (`MADV_FREE`): contents are undefined — the
    ///   kernel may reclaim the page lazily under memory pressure, so a
    ///   read may see either the previous contents (no reclaim yet) or
    ///   zeros (post-reclaim). Callers must not rely on either outcome.
    /// - **Windows** (`MEM_RESET`): contents are undefined for the same
    ///   reason.
    ///
    /// In short: portable callers must treat the released range as
    /// containing arbitrary bytes and re-initialise before re-reading.
    /// The release is *not* `munmap` — the virtual address range remains
    /// reserved.
    ///
    /// # Safety
    ///
    /// - `[ptr, ptr + size)` must lie wholly within this allocator's
    ///   `[base_ptr, base_ptr + region_size)`.
    /// - There must be no live allocations (no pointer issued by
    ///   `Allocator::allocate` and not yet `deallocate`d) overlapping the
    ///   released range — release destroys their contents.
    /// - For correctness on Windows, `ptr` and `size` should be page-
    ///   aligned; on Unix the kernel rounds internally. Misaligned calls
    ///   may release less (Windows) or more (Unix) than the caller
    ///   intended — neither is UB, but the contract is "page-aligned
    ///   ranges" if portability is required.
    unsafe fn release_pages(&self, ptr: NonNull<u8>, size: usize);

    /// Change memory protection flags on `[ptr, ptr + size)`.
    ///
    /// Maps to `mprotect` on Unix and `VirtualProtect` on Windows.
    ///
    /// # Safety
    ///
    /// - `[ptr, ptr + size)` must lie wholly within this allocator's
    ///   region.
    /// - `ptr` and `size` should be page-aligned; behaviour on
    ///   misalignment is platform-specific (Unix rounds outward; Windows
    ///   rejects with `ERROR_INVALID_PARAMETER`).
    /// - The caller must ensure no concurrent code path will read or
    ///   write through the affected range in a way that violates the new
    ///   flags. Setting `PROT_NONE` (or Windows `PAGE_NOACCESS`) on a
    ///   range that any live allocation crosses will cause SIGSEGV /
    ///   access violation on subsequent access — that is the intended
    ///   trap-on-touch behaviour for `GuardPage`, but UB-from-the-
    ///   caller's-perspective if unintentional.
    unsafe fn protect(&self, ptr: NonNull<u8>, size: usize, flags: ProtectFlags);
}
