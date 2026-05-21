//! `MmapBacked` — OS-managed anonymous memory region.
//!
//! Linux/macOS: `mmap(MAP_ANONYMOUS | MAP_PRIVATE)`.
//! Windows: `VirtualAlloc(MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)`.
//!
//! On drop, the region is returned to the OS (`munmap` / `VirtualFree`). The
//! region is laid out bump-style — each `allocate` advances a cursor through
//! the mapping; `deallocate` is a no-op. Higher layers (`BumpArena`, `Slab`)
//! impose their own structure on what they receive.

use core::cell::UnsafeCell;
use core::ptr::NonNull;

use forge_alloc_core::{
    AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout, OsBacked, ProtectFlags,
};

std::thread_local! {
    /// Last OS error captured from a failing mmap-layer syscall on this thread.
    /// Set immediately after a failure; read via [`mmap_last_os_error`].
    ///
    /// Stored as a raw error code (errno on Unix, `GetLastError` value on
    /// Windows). `None` means no failure has been recorded on this thread yet,
    /// or the slot was explicitly cleared.
    static LAST_OS_ERROR: core::cell::Cell<Option<i32>> = const { core::cell::Cell::new(None) };
}

/// Record the most recent failing-syscall error from the platform's
/// thread-local errno / GetLastError into this module's slot. Must be
/// called *immediately* after the failing syscall, before any other libc /
/// Win32 call can clobber the underlying thread-local.
#[inline]
fn capture_os_error() {
    let raw = std::io::Error::last_os_error().raw_os_error();
    LAST_OS_ERROR.with(|c| c.set(raw));
}

/// Return the most recent failing-syscall error captured into this
/// module's slot on the current thread, or `None` if none has been
/// recorded since thread start or [`mmap_clear_last_os_error`] was last
/// called.
///
/// Code is platform-specific (errno on Unix, `GetLastError` on Windows).
/// Read this *immediately* after a `MmapBacked` constructor or OS-call
/// returns an error — subsequent libc/Win32 calls in other crates may
/// overwrite the platform's underlying thread-local. The snapshot in
/// THIS module is only updated when (a) a syscall inside `MmapBacked`
/// fails, (b) a pre-syscall validation path in `MmapBacked::with_flags`
/// rejects its argument (synthetic `EINVAL`), or (c) a composing crate
/// pushes its own errno via [`mmap_record_os_error`].
#[must_use]
pub fn mmap_last_os_error() -> Option<std::io::Error> {
    LAST_OS_ERROR
        .with(|c| c.get())
        .map(std::io::Error::from_raw_os_error)
}

/// Clear the per-thread last-error slot. Mainly useful in tests; callers
/// in production typically just read [`mmap_last_os_error`] after a
/// failure.
pub fn mmap_clear_last_os_error() {
    LAST_OS_ERROR.with(|c| c.set(None));
}

/// Record the *current* platform errno / GetLastError into the per-thread
/// last-error slot from an external crate. Use this immediately after a
/// failing syscall (e.g. `mbind`, `madvise`, `pthread_*`) in a crate that
/// composes with `MmapBacked` so callers can read a single
/// [`mmap_last_os_error`] regardless of which layer's syscall failed.
///
/// # Ordering contract
///
/// Must be called **immediately after the failing syscall returns** and
/// **before any other libc / Win32 call**. The platform thread-local
/// that backs `std::io::Error::last_os_error()` (errno on Unix,
/// `GetLastError` on Windows) is volatile — any subsequent call (even a
/// no-failure one — e.g. an allocator's bookkeeping `free` or a logging
/// `write`) may clobber it before this function gets a chance to read
/// the failing code.
///
/// # Thread safety
///
/// The slot is thread-local. Concurrent calls from different threads
/// touch disjoint storage and cannot race; concurrent calls within the
/// same thread are impossible (single-threaded execution within a
/// thread). Each thread sees the most recent error captured *on that
/// thread*, regardless of which crate captured it.
#[inline]
pub fn mmap_record_os_error() {
    capture_os_error();
}

/// Record a synthetic `EINVAL` into the per-thread last-error slot. Used
/// when a [`MmapBacked`] constructor rejects its argument (size==0, page-
/// rounding overflow) without invoking the kernel — otherwise callers
/// reading [`mmap_last_os_error`] would see whatever stale value a prior
/// syscall failure left on this thread, or `None`, both of which are
/// misleading. EINVAL is the universal "validation failed" signal on
/// Unix (errno 22) and Windows (`ERROR_INVALID_PARAMETER`, 87).
#[inline]
fn capture_synthetic_einval() {
    #[cfg(unix)]
    let code: i32 = libc::EINVAL;
    #[cfg(windows)]
    // `ERROR_INVALID_PARAMETER` is defined as `u32` in `windows-sys`; the
    // crate-wide slot stores `i32` to match `std::io::Error::raw_os_error`,
    // so cast at the boundary. The value (87) is within `i32::MAX` and
    // round-trips through `Error::from_raw_os_error` without loss.
    let code: i32 = windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER as i32;
    LAST_OS_ERROR.with(|c| c.set(Some(code)));
}

/// Optional flags for [`MmapBacked::with_flags`].
///
/// Most flags route to features not yet implemented (`HugePageAligned`,
/// `NumaLocal`). Currently only `populate` is honored on platforms that support
/// it; the rest are accepted for forward compatibility but currently no-op.
///
/// `#[non_exhaustive]` so future bits (`MAP_LOCKED`, `MAP_NORESERVE`, MTE
/// enable) can be added without an API break.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MmapFlags {
    /// Request transparent / explicit huge pages. Implemented via
    /// `HugePageAligned`; ignored at this layer.
    pub huge_pages: bool,
    /// Fault all pages at allocation time so subsequent accesses don't take
    /// page-fault latency.
    ///
    /// **Platform support is asymmetric** — setting this to `true` does
    /// not guarantee eager paging on every platform:
    ///
    /// - **Linux**: maps to `MAP_POPULATE`. Kernel walks the page tables
    ///   at `mmap` time so subsequent accesses don't fault.
    /// - **macOS / BSD**: **silently ignored**. There is no portable
    ///   equivalent that operates at `mmap` time; eager paging on Darwin
    ///   requires `madvise(MADV_WILLNEED)` over the region after mapping,
    ///   which `MmapBacked` does not currently perform.
    /// - **Windows**: **silently ignored**. `VirtualAlloc` already
    ///   commits the range at allocation time, so the page-fault
    ///   latency the flag is meant to amortize is much smaller.
    ///
    /// Use [`Self::populate_supported`] to test at runtime whether
    /// setting this flag will have any effect, or branch on `cfg!` in
    /// caller code.
    pub populate: bool,
    /// Bind to a specific NUMA node. Implemented via `NumaLocal`; ignored
    /// at this layer.
    pub numa_node: Option<u32>,
    /// Append one unmapped guard page after the region. Implemented via
    /// `GuardPage`; ignored at this layer.
    pub guard_at_end: bool,
}

impl MmapFlags {
    /// Empty flag set — equivalent to [`MmapBacked::new`].
    pub const NONE: Self = Self {
        huge_pages: false,
        populate: false,
        numa_node: None,
        guard_at_end: false,
    };

    /// Returns `true` if `populate: true` will actually be honored on the
    /// current platform — `true` on Linux, `false` on macOS / BSD /
    /// Windows. Allows callers to branch on whether the eager-paging
    /// performance hint is meaningful without resorting to `cfg!` checks
    /// scattered through application code.
    #[inline]
    pub const fn populate_supported() -> bool {
        cfg!(target_os = "linux")
    }
}

impl Default for MmapFlags {
    fn default() -> Self {
        Self::NONE
    }
}

/// OS-mapped anonymous region.
///
/// `len` is rounded up to a multiple of the page size at construction. The
/// `Allocator` impl serves requests bump-style from the mapping.
///
/// # Thread safety
///
/// `Send`: yes — the mapping is identified by `(ptr, len)`, both `Send`-safe
/// values; we restore `Send` via an `unsafe impl` since `NonNull<u8>` is
/// `!Send` by default.
/// `Sync`: NO. The cursor uses `UnsafeCell` for `&self` allocation; concurrent
/// `&self` allocators would race. `UnsafeCell` is `!Sync`, which gives us the
/// right behavior without any extra marker field. Cross-thread allocation
/// belongs to higher layers (`SharedBumpArena`, `SlabRemote`).
pub struct MmapBacked {
    ptr: NonNull<u8>,
    len: usize,
    cursor: UnsafeCell<usize>,
}

impl MmapBacked {
    /// Allocate an anonymous OS-mapped region of at least `size` bytes (rounded
    /// up to the page size).
    pub fn new(size: usize) -> Result<Self, AllocError> {
        Self::with_flags(size, MmapFlags::NONE)
    }

    /// Allocate with huge-pages requested. This layer ignores the hint;
    /// `HugePageAligned` enforces 2 MiB / 32 MiB alignment.
    pub fn with_huge_pages(size: usize) -> Result<Self, AllocError> {
        Self::with_flags(
            size,
            MmapFlags {
                huge_pages: true,
                ..MmapFlags::NONE
            },
        )
    }

    /// Allocate with the supplied [`MmapFlags`].
    pub fn with_flags(size: usize, flags: MmapFlags) -> Result<Self, AllocError> {
        if size == 0 {
            // Pre-syscall rejection: record a synthetic EINVAL so
            // `mmap_last_os_error()` callers see an honest diagnostic
            // rather than whatever stale value lingers from a prior
            // failure on this thread.
            capture_synthetic_einval();
            return Err(AllocError);
        }
        let page = page_size();
        let len = match size.checked_add(page - 1).map(|s| s & !(page - 1)) {
            Some(l) => l,
            None => {
                capture_synthetic_einval();
                return Err(AllocError);
            }
        };
        // SAFETY: platform-specific os_map enforces its own invariants and
        // returns a non-null pointer to `len` writable bytes on success.
        let ptr = unsafe { os_map(len, &flags)? };
        Ok(Self {
            ptr,
            len,
            cursor: UnsafeCell::new(0),
        })
    }

    /// Bytes already allocated from this backing.
    #[inline]
    pub fn allocated(&self) -> usize {
        // SAFETY: !Sync — no concurrent access to cursor.
        unsafe { *self.cursor.get() }
    }

    /// Total size of the OS-mapped region (page-aligned).
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.len
    }

    /// Bytes remaining for allocation.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.len - self.allocated()
    }
}

impl Drop for MmapBacked {
    fn drop(&mut self) {
        // SAFETY: ptr/len pair came from os_map on construction; no copies of
        // either escape this struct (no Clone impl). Caller of MmapBacked has
        // by contract guaranteed no outstanding pointers into the region at
        // drop time (the same caller-discipline that BumpArena::reset requires).
        unsafe { os_unmap(self.ptr, self.len) };
    }
}

unsafe impl Deallocator for MmapBacked {
    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // No-op. Bump-style; reclaim via drop.
    }
}

unsafe impl Allocator for MmapBacked {
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let align = layout.align().get();
        let size = layout.size().get();
        // SAFETY: !Sync — no concurrent access to cursor.
        unsafe {
            let cursor_ptr = self.cursor.get();
            let cur = *cursor_ptr;
            let base = self.ptr.as_ptr() as usize;
            let next = base
                .checked_add(cur)
                .and_then(|v| v.checked_add(align - 1))
                .ok_or(AllocError)?
                & !(align - 1);
            let aligned_off = next - base;
            let end_off = aligned_off.checked_add(size).ok_or(AllocError)?;
            if end_off > self.len {
                return Err(AllocError);
            }
            *cursor_ptr = end_off;
            let p = self.ptr.as_ptr().add(aligned_off);
            // SAFETY: aligned_off <= len, and p derives from self.ptr which
            // is non-null; the result is non-null.
            Ok(NonNull::slice_from_raw_parts(
                NonNull::new_unchecked(p),
                size,
            ))
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.len)
    }
}

impl FixedRange for MmapBacked {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.ptr
    }

    #[inline]
    fn size(&self) -> usize {
        self.len
    }
}

unsafe impl OsBacked for MmapBacked {
    #[inline]
    fn base_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    #[inline]
    fn region_size(&self) -> usize {
        self.len
    }

    #[inline]
    unsafe fn release_pages(&self, ptr: NonNull<u8>, size: usize) {
        // SAFETY: caller has promised [ptr, ptr+size) lies wholly inside our
        // region and has no live allocations.
        unsafe { os_release_pages(ptr, size) };
    }

    #[inline]
    unsafe fn protect(&self, ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
        // SAFETY: caller has promised [ptr, ptr+size) lies inside our region.
        unsafe { os_protect(ptr, size, flags) };
    }
}

// MmapBacked owns a `NonNull<u8>` to a non-shared OS mapping. Send is fine
// (the mapping outlives the move because munmap is keyed on ptr+len, not on
// thread identity). !Sync is inherited from the `UnsafeCell<usize>` cursor —
// no extra marker field is needed.
//
// SAFETY: see the rationale above; no aliasing reference into the mapping
// escapes the struct (callers receive raw `NonNull<u8>` pointers — Rust's
// aliasing model treats those as inert, in the same way `Box<T>: Send`).
unsafe impl Send for MmapBacked {}

// ============================================================================
// Platform glue
// ============================================================================

/// The OS memory page size in bytes — 4 KiB on most x86-64, 16 KiB on
/// Apple Silicon. Pass this where a primitive needs a page-size argument
/// (such as `GuardPage`) rather than hard-coding a value that is wrong
/// on 16 KiB-page platforms.
#[cfg(unix)]
pub fn page_size() -> usize {
    // SAFETY: sysconf is async-signal-safe and always returns >= 0 for
    // _SC_PAGESIZE on conforming Unix; we still fall back defensively when
    // the call reports -1 (errno) so `with_flags` cannot hit `page - 1`
    // underflow on a pathological kernel.
    //
    // NOTE: the 4096 fallback may undersize on 16K-page systems (Apple
    // Silicon, some ARMv8) if sysconf ever fails — we'd round to 4K instead
    // of 16K, then mmap would still align internally to 16K. The behavioral
    // consequence is over-reservation at the round-up step, not unsoundness.
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if p > 0 {
        p as usize
    } else {
        4096
    }
}

/// The OS memory page size in bytes — 4 KiB on most x86-64, 16 KiB on
/// Apple Silicon. Pass this where a primitive needs a page-size argument
/// (such as `GuardPage`) rather than hard-coding a value that is wrong
/// on 16 KiB-page platforms.
#[cfg(windows)]
pub fn page_size() -> usize {
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    // SAFETY: GetSystemInfo writes a fully-initialized SYSTEM_INFO into its
    // out-pointer; we provide a stack slot of the correct size.
    let mut info: SYSTEM_INFO = unsafe { core::mem::zeroed() };
    unsafe { GetSystemInfo(&mut info) };
    // `dwPageSize` is documented to be non-zero on all supported Windows
    // editions; the explicit fallback is purely defensive so a degenerate
    // value can never trigger `page - 1` underflow in `with_flags`.
    let p = info.dwPageSize as usize;
    if p > 0 {
        p
    } else {
        4096
    }
}

#[cfg(unix)]
unsafe fn os_map(len: usize, flags: &MmapFlags) -> Result<NonNull<u8>, AllocError> {
    // `mut` is needed only on Linux, where the `MAP_POPULATE` branch below
    // reassigns it; on macOS / other Unix the binding is never mutated.
    #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
    let mut mmap_flags = libc::MAP_ANONYMOUS | libc::MAP_PRIVATE;
    if flags.populate {
        // MAP_POPULATE exists on Linux; on macOS the call still succeeds but
        // the flag is silently ignored.
        #[cfg(target_os = "linux")]
        {
            mmap_flags |= libc::MAP_POPULATE;
        }
    }
    // SAFETY: mmap with MAP_ANONYMOUS+MAP_PRIVATE, non-zero len, non-conflicting
    // flags. Returns MAP_FAILED on error which we check.
    let ptr = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            mmap_flags,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        capture_os_error();
        return Err(AllocError);
    }
    // SAFETY: mmap returned non-MAP_FAILED, so ptr is a valid non-null mapping.
    Ok(unsafe { NonNull::new_unchecked(ptr as *mut u8) })
}

#[cfg(windows)]
unsafe fn os_map(len: usize, _flags: &MmapFlags) -> Result<NonNull<u8>, AllocError> {
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
    };
    // SAFETY: VirtualAlloc with NULL base and MEM_RESERVE|MEM_COMMIT is the
    // standard anonymous-mapping pattern. Returns NULL on error.
    let p = unsafe {
        VirtualAlloc(
            core::ptr::null_mut(),
            len,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    let nn = NonNull::new(p as *mut u8);
    if nn.is_none() {
        capture_os_error();
    }
    nn.ok_or(AllocError)
}

#[cfg(unix)]
unsafe fn os_unmap(ptr: NonNull<u8>, len: usize) {
    // SAFETY: ptr/len pair came from os_map; munmap of an active mapping is
    // the only safe way to release it.
    let rc = unsafe { libc::munmap(ptr.as_ptr() as *mut libc::c_void, len) };
    if rc != 0 {
        // Drop path can't propagate Err; record errno so callers can detect
        // a previous unmap failure via `mmap_last_os_error()` if they choose.
        capture_os_error();
    }
}

#[cfg(windows)]
unsafe fn os_unmap(ptr: NonNull<u8>, _len: usize) {
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
    // SAFETY: VirtualFree with MEM_RELEASE expects the base pointer returned
    // by VirtualAlloc and size = 0; that releases both the reservation and
    // the commit. Errors are reported via thread-local; Drop can't propagate.
    let ok = unsafe { VirtualFree(ptr.as_ptr() as *mut _, 0, MEM_RELEASE) };
    if ok == 0 {
        capture_os_error();
    }
}

#[cfg(unix)]
unsafe fn os_release_pages(ptr: NonNull<u8>, size: usize) {
    // Choose advice by platform:
    //
    // Linux: MADV_DONTNEED on a private anonymous mapping immediately
    // releases the physical pages; subsequent reads see zero-filled
    // pages. This is the canonical "release-but-keep-vma" path.
    //
    // macOS: MADV_DONTNEED on a private mapping is only a hint — the
    // kernel may ignore it. MADV_FREE (added 10.12 / macOS Sierra) is
    // the documented path: the kernel may reclaim the pages under
    // memory pressure, and a subsequent read sees either old data or
    // zeros (the new contents are undefined). For "I really don't
    // need this anymore" semantics, MADV_FREE is the right choice on
    // macOS.
    //
    // Other Unix (BSD): MADV_FREE has the BSD semantics — same as
    // macOS.
    #[cfg(target_os = "linux")]
    let advice = libc::MADV_DONTNEED;
    #[cfg(not(target_os = "linux"))]
    let advice = libc::MADV_FREE;
    // SAFETY: ptr/size lie wholly inside our own mapping (per the
    // OsBacked::release_pages caller contract); advice is a valid flag.
    let rc = unsafe { libc::madvise(ptr.as_ptr() as *mut libc::c_void, size, advice) };
    if rc != 0 {
        capture_os_error();
    }
}

#[cfg(windows)]
unsafe fn os_release_pages(ptr: NonNull<u8>, size: usize) {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_RESET, PAGE_READWRITE};
    // VirtualAlloc(MEM_RESET) operates on a page-granular range; misaligned
    // `ptr` or `size` returns NULL with ERROR_INVALID_PARAMETER, which we
    // surface via capture_os_error(). Debug builds assert up front so the
    // misuse is caught in development rather than via a silent observability
    // probe in production.
    let page = page_size();
    debug_assert_eq!(
        (ptr.as_ptr() as usize) % page,
        0,
        "os_release_pages: ptr must be page-aligned on Windows MEM_RESET",
    );
    debug_assert_eq!(
        size % page,
        0,
        "os_release_pages: size must be page-aligned on Windows MEM_RESET",
    );
    // SAFETY: VirtualAlloc with MEM_RESET on an existing region tells the OS
    // the contents are discardable; the OS may reclaim the physical pages.
    // The lpProtect argument is ignored for MEM_RESET but must be valid.
    let p = unsafe { VirtualAlloc(ptr.as_ptr() as *mut _, size, MEM_RESET, PAGE_READWRITE) };
    if p.is_null() {
        capture_os_error();
    }
}

/// Map forge-alloc-core's `ProtectFlags` to a Unix `mprotect` `prot` argument.
///
/// Unlike Windows, the Unix ABI exposes each protection bit independently
/// (`PROT_READ`, `PROT_WRITE`, `PROT_EXEC`), so every one of the eight
/// `(read, write, exec)` combinations maps bit-exactly to a `mprotect`
/// argument with no over-grant or down-grade at this layer.
///
/// | `(read, write, exec)` | `prot`                              | Notes |
/// |-----------------------|-------------------------------------|-------|
/// | `(F, F, F)`           | `PROT_NONE` (== 0)                  | exact |
/// | `(T, F, F)`           | `PROT_READ`                         | exact |
/// | `(F, T, F)`           | `PROT_WRITE`                        | exact at the syscall ABI; some archs (older x86_64) implicitly grant read when write is set, but that's below this layer |
/// | `(F, F, T)`           | `PROT_EXEC`                         | exact on NX-capable HW; pre-NX implicit read |
/// | `(T, T, F)`           | `PROT_READ \| PROT_WRITE`           | exact |
/// | `(T, F, T)`           | `PROT_READ \| PROT_EXEC`            | exact |
/// | `(F, T, T)`           | `PROT_WRITE \| PROT_EXEC`           | exact (some kernels enforce W^X via seccomp/LSM; that's surfaced as `EINVAL` on the syscall, not silently masked here) |
/// | `(T, T, T)`           | `PROT_READ \| PROT_WRITE \| PROT_EXEC` | exact; some hardened kernels reject and surface `EACCES`/`EINVAL` — propagated unchanged |
///
/// Extracted so unit tests can verify the mapping table without invoking
/// `mprotect` on the host (the test runs cross-platform; only the table
/// math is platform-neutral). This is the Unix structural parallel to
/// [`win32_prot_from_flags`] — each Unix arm maps bit-exactly, unlike
/// Win32 which cannot express write-without-read combinations natively.
#[cfg(unix)]
fn unix_prot_from_flags(flags: ProtectFlags) -> i32 {
    // PROT_NONE is 0 on every Unix; the explicit assignment in the
    // "all-false" branch below documents intent without changing bits.
    let mut prot = 0i32;
    if flags.read {
        prot |= libc::PROT_READ;
    }
    if flags.write {
        prot |= libc::PROT_WRITE;
    }
    if flags.exec {
        prot |= libc::PROT_EXEC;
    }
    if !flags.read && !flags.write && !flags.exec {
        prot = libc::PROT_NONE;
    }
    prot
}

#[cfg(unix)]
unsafe fn os_protect(ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
    let prot = unix_prot_from_flags(flags);
    // SAFETY: mprotect on a region we own with valid flag bits.
    let rc = unsafe { libc::mprotect(ptr.as_ptr() as *mut libc::c_void, size, prot) };
    if rc != 0 {
        capture_os_error();
    }
}

/// Map forge-alloc-core's `ProtectFlags` to a Windows `PAGE_*` constant.
///
/// The mapping is **bit-preserving wherever the Win32 ABI can express the
/// combination**, and chooses the smallest valid superset otherwise.
/// Concretely, Win32 *does* expose a true exec-only mode (`PAGE_EXECUTE`,
/// value 16) — readers of a `PAGE_EXECUTE` page take an access violation on
/// hardware that supports NX (every supported x86_64 / aarch64 chip). On
/// the small set of legacy CPUs without NX, the kernel implicitly grants
/// read access; that downgrade is unavoidable and lives below this
/// layer. Win32 does *not* expose a write-without-read or write+exec-
/// without-read mode, so those must be upgraded.
///
/// | `(read, write, exec)`     | Win32 constant            | Notes |
/// |---------------------------|---------------------------|-------|
/// | `(F, F, F)`               | `PAGE_NOACCESS`           | exact |
/// | `(T, F, F)`               | `PAGE_READONLY`           | exact |
/// | `(T, T, F)`               | `PAGE_READWRITE`          | exact |
/// | `(T, F, T)`               | `PAGE_EXECUTE_READ`       | exact |
/// | `(T, T, T)`               | `PAGE_EXECUTE_READWRITE`  | exact |
/// | `(F, F, T)`               | `PAGE_EXECUTE`            | exact on NX-capable HW |
/// | `(F, T, F)`               | `PAGE_READWRITE`          | over-grants read |
/// | `(F, T, T)`               | `PAGE_EXECUTE_READWRITE`  | over-grants read |
///
/// Extracted so unit tests can verify the mapping table without triggering
/// the debug_assert in [`os_protect`] (which fires on write-without-read,
/// the only combination that the helper genuinely cannot express).
#[cfg(windows)]
fn win32_prot_from_flags(flags: ProtectFlags) -> u32 {
    use windows_sys::Win32::System::Memory::{
        PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS, PAGE_READONLY,
        PAGE_READWRITE,
    };
    match (flags.read, flags.write, flags.exec) {
        (false, false, false) => PAGE_NOACCESS,
        (true, false, false) => PAGE_READONLY,
        (true, true, false) => PAGE_READWRITE,
        (true, false, true) => PAGE_EXECUTE_READ,
        (true, true, true) => PAGE_EXECUTE_READWRITE,
        // Exec-only: Windows DOES support this natively via PAGE_EXECUTE.
        // On NX-capable hardware (every modern x64 / aarch64 chip) this is
        // exec-only; on legacy NX-less CPUs the kernel implicitly grants
        // read, which mirrors `mprotect(PROT_EXEC)` on the same hardware.
        // Mapping it to PAGE_EXECUTE_READ here would *unconditionally*
        // over-grant on every machine; using PAGE_EXECUTE only over-grants
        // on the legacy ones — strictly tighter.
        (false, false, true) => PAGE_EXECUTE,
        // Write-or-exec with write but without read: Windows has no
        // primitive for "write but not read", so upgrade to the smallest
        // valid superset that retains every bit the caller asked for.
        // Crucially, (false, true, true) must route to
        // PAGE_EXECUTE_READWRITE — collapsing it to PAGE_READWRITE would
        // silently drop the exec bit. The
        // debug_assert in os_protect surfaces these over-grants in dev.
        (false, true, true) => PAGE_EXECUTE_READWRITE,
        (false, true, false) => PAGE_READWRITE,
    }
}

#[cfg(windows)]
unsafe fn os_protect(ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    // Of all eight `(read, write, exec)` combinations, the only ones
    // Win32 cannot express bit-exactly are write-without-read variants:
    // `(F, T, F)` and `(F, T, T)` — Windows has no primitive for "write
    // but not read", so `win32_prot_from_flags` upgrades them to
    // `PAGE_READWRITE` / `PAGE_EXECUTE_READWRITE`. Every other
    // combination (including exec-only via `PAGE_EXECUTE`) maps exactly
    // on NX-capable hardware. A debug-build assertion flags the unavoidable
    // upgrade so misuse during development surfaces in tests:
    debug_assert!(
        !flags.write || flags.read,
        "os_protect: write-without-read upgrades to RW/RWX on Windows — \
         caller relying on no-read semantics will not get them. Set flags.read=true \
         explicitly to silence this assertion.",
    );
    let prot = win32_prot_from_flags(flags);
    let mut old: u32 = 0;
    // SAFETY: VirtualProtect on a region returned by VirtualAlloc with valid
    // PAGE_* protection constants.
    let ok = unsafe { VirtualProtect(ptr.as_ptr() as *mut _, size, prot, &mut old) };
    if ok == 0 {
        capture_os_error();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every test in this module exercises real OS mmap / VirtualAlloc paths.
    // Miri cannot model `mmap` / `VirtualAlloc` syscalls, so the entire
    // module is gated off under miri. The underlying invariants the tests
    // protect (page rounding, alignment, capacity, OS-error capture) are
    // unaffected by Miri's interpretation model — Miri's job here is to
    // detect UB in the *consumers* of MmapBacked (Slab / Bump / etc.) when
    // they're driven over InlineBacked.

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim Win32 GetSystemInfo / sysconf")]
    fn page_size_is_reasonable() {
        let p = page_size();
        assert!(p >= 4096, "page size suspiciously small: {p}");
        assert!(p.is_power_of_two());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn alloc_then_write_then_read_back() {
        let m = MmapBacked::new(16 * 1024).expect("mmap should succeed for 16 KiB");
        let layout = NonZeroLayout::from_size_align(256, 8).unwrap();
        let block = m.allocate(layout).unwrap();
        let p = block.cast::<u8>();
        unsafe {
            core::ptr::write_bytes(p.as_ptr(), 0xCD, 256);
            for i in 0..256 {
                assert_eq!(*p.as_ptr().add(i), 0xCD);
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn alloc_returns_aligned_pointer() {
        let m = MmapBacked::new(64 * 1024).unwrap();
        // First, push the cursor off zero with an odd-size allocation.
        let _ = m
            .allocate(NonZeroLayout::from_size_align(3, 1).unwrap())
            .unwrap();
        let layout = NonZeroLayout::from_size_align(64, 64).unwrap();
        let block = m.allocate(layout).unwrap();
        assert_eq!(block.cast::<u8>().as_ptr() as usize % 64, 0);
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn alloc_fails_when_exhausted() {
        let m = MmapBacked::new(8 * 1024).unwrap();
        let cap = m.capacity();
        let layout = NonZeroLayout::from_size_align(cap, 8).unwrap();
        let _ = m.allocate(layout).unwrap();
        assert!(m
            .allocate(NonZeroLayout::from_size_align(1, 1).unwrap())
            .is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn fixed_range_contains_allocations() {
        let m = MmapBacked::new(8 * 1024).unwrap();
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let block = m.allocate(layout).unwrap();
        assert!(m.contains(block.cast::<u8>()));
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn capacity_is_page_rounded() {
        let m = MmapBacked::new(1).unwrap();
        let cap = m.capacity();
        let page = page_size();
        assert_eq!(cap % page, 0);
        assert!(cap >= page);
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn zero_size_request_errors() {
        assert!(MmapBacked::new(0).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn pre_syscall_rejection_sets_synthetic_einval() {
        // Pre-syscall failure paths (size==0, page-rounding overflow) must
        // populate the thread-local last-error slot with EINVAL rather
        // than leaving stale data from prior failures. Without this,
        // mmap_last_os_error() would silently lie about what just failed.
        mmap_clear_last_os_error();
        assert!(MmapBacked::new(0).is_err());
        let e = mmap_last_os_error().expect("synthetic EINVAL captured");
        #[cfg(unix)]
        assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
        #[cfg(windows)]
        assert_eq!(
            e.raw_os_error(),
            Some(windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER as i32),
        );

        // Overflow path: size + (page-1) wraps.
        mmap_clear_last_os_error();
        assert!(MmapBacked::new(usize::MAX).is_err());
        let e = mmap_last_os_error().expect("synthetic EINVAL on overflow");
        #[cfg(unix)]
        assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
        #[cfg(windows)]
        assert_eq!(
            e.raw_os_error(),
            Some(windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER as i32),
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn last_os_error_captured_on_failure() {
        // Request an impossibly large mapping — both unix and Windows should
        // reject and set their thread-local error. We can't predict the exact
        // code (ENOMEM, EINVAL, EOVERFLOW, ERROR_NOT_ENOUGH_MEMORY, …) so we
        // only assert that *something* was captured.
        mmap_clear_last_os_error();
        assert!(mmap_last_os_error().is_none());
        // usize::MAX/2 rounds to usize::MAX-(page-1) which exceeds any
        // realistic address space, forcing a syscall failure.
        let huge = usize::MAX / 2;
        assert!(MmapBacked::new(huge).is_err());
        assert!(
            mmap_last_os_error().is_some(),
            "expected captured OS error after impossibly large mmap request",
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn os_backed_release_pages_after_alloc() {
        let m = MmapBacked::new(64 * 1024).unwrap();
        let p = m.base_ptr();
        // Write something, release, write again — must not crash.
        unsafe {
            core::ptr::write_bytes(p.as_ptr(), 0xEE, page_size());
            m.release_pages(p, page_size());
            core::ptr::write_bytes(p.as_ptr(), 0x11, page_size());
        }
    }

    /// Structural parallel to the Windows
    /// `win32_prot_from_flags_preserves_every_requested_bit` regression
    /// test: confirm that every `(read, write, exec)` combination on Unix
    /// produces the corresponding bit-exact `PROT_*` mask with no over-
    /// grant (no spurious read added to exec-only) and no down-grade
    /// (W+X must not collapse to W). Unlike Win32 — which lacks primitives
    /// for write-without-read and exec-only — Unix `mprotect` exposes each
    /// bit independently, so the table is exact across all eight rows.
    ///
    /// Running this test on a non-Unix host (Windows) verifies the table
    /// math at compile time only when this `#[cfg(unix)]` gate is active;
    /// CI on Linux/macOS exercises the assertions at runtime.
    #[cfg(unix)]
    #[test]
    fn unix_prot_from_flags_preserves_every_requested_bit() {
        // ProtectFlags is #[non_exhaustive] — build via base + field assigns.
        let mut none = ProtectFlags::NONE;
        let mut r = ProtectFlags::NONE;
        r.read = true;
        let mut w = ProtectFlags::NONE;
        w.write = true;
        let mut x = ProtectFlags::NONE;
        x.exec = true;
        let mut rw = ProtectFlags::NONE;
        rw.read = true;
        rw.write = true;
        let mut rx = ProtectFlags::NONE;
        rx.read = true;
        rx.exec = true;
        let mut wx = ProtectFlags::NONE;
        wx.write = true;
        wx.exec = true;
        let mut rwx = ProtectFlags::NONE;
        rwx.read = true;
        rwx.write = true;
        rwx.exec = true;
        // Suppress unused_mut on `none` — clippy/rustc otherwise gripe.
        let _ = &mut none;

        assert_eq!(unix_prot_from_flags(none), libc::PROT_NONE);
        assert_eq!(unix_prot_from_flags(r), libc::PROT_READ);
        assert_eq!(
            unix_prot_from_flags(w),
            libc::PROT_WRITE,
            "W must be PROT_WRITE only — Unix allows write-without-read at the syscall ABI; \
             any kernel-side implicit read-grant lives below this layer and is not our concern",
        );
        assert_eq!(
            unix_prot_from_flags(x),
            libc::PROT_EXEC,
            "X must be PROT_EXEC only — over-granting (e.g. adding PROT_READ) \
             must not appear on Unix",
        );
        assert_eq!(unix_prot_from_flags(rw), libc::PROT_READ | libc::PROT_WRITE);
        assert_eq!(unix_prot_from_flags(rx), libc::PROT_READ | libc::PROT_EXEC);
        assert_eq!(
            unix_prot_from_flags(wx),
            libc::PROT_WRITE | libc::PROT_EXEC,
            "W+X must be PROT_WRITE|PROT_EXEC — silently dropping the exec bit \
             must not appear on Unix. Hardened kernels that enforce W^X surface \
             EINVAL/EACCES at the mprotect syscall, not by silently masking bits here.",
        );
        assert_eq!(
            unix_prot_from_flags(rwx),
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        );
    }

    /// Regression: `win32_prot_from_flags` used to map `(read=false,
    /// write=true, exec=true)` to `PAGE_READWRITE`, silently dropping the
    /// caller's exec bit. Hardening wrappers that ask for W+X (uncommon but
    /// valid for JIT-like flows that don't need read) would have gotten
    /// pages that fault on instruction fetch in release builds — the
    /// debug_assert in `os_protect` only catches write-without-read in dev.
    /// The fix routes W+X through `PAGE_EXECUTE_READWRITE`.
    ///
    /// Exec-only `(F, F, T)` uses `PAGE_EXECUTE` (which Windows *does*
    /// support natively) so that callers that opt out of read on NX-capable
    /// hardware actually get exec-only semantics rather than an
    /// unconditional upgrade to RX.
    ///
    /// The mapping is tested in isolation (bypassing `os_protect`'s
    /// `debug_assert!(!(write && !read))`). Unix is unaffected — its
    /// `mprotect` path expresses each bit independently.
    #[cfg(windows)]
    #[test]
    #[cfg_attr(
        miri,
        ignore = "win32 import resolution requires actual Windows runtime"
    )]
    fn win32_prot_from_flags_preserves_every_requested_bit() {
        use windows_sys::Win32::System::Memory::{
            PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS, PAGE_READONLY,
            PAGE_READWRITE,
        };
        // ProtectFlags is #[non_exhaustive] — build via base + field assigns.
        let mut none = ProtectFlags::NONE;
        let mut r = ProtectFlags::NONE;
        r.read = true;
        let mut w = ProtectFlags::NONE;
        w.write = true;
        let mut x = ProtectFlags::NONE;
        x.exec = true;
        let mut rw = ProtectFlags::NONE;
        rw.read = true;
        rw.write = true;
        let mut rx = ProtectFlags::NONE;
        rx.read = true;
        rx.exec = true;
        let mut rwx = ProtectFlags::NONE;
        rwx.read = true;
        rwx.write = true;
        rwx.exec = true;
        let mut wx = ProtectFlags::NONE;
        wx.write = true;
        wx.exec = true;
        // Suppress unused_mut on `none` — clippy/rustc otherwise gripe.
        let _ = &mut none;

        assert_eq!(win32_prot_from_flags(none), PAGE_NOACCESS);
        assert_eq!(win32_prot_from_flags(r), PAGE_READONLY);
        assert_eq!(win32_prot_from_flags(rw), PAGE_READWRITE);
        assert_eq!(win32_prot_from_flags(rx), PAGE_EXECUTE_READ);
        assert_eq!(win32_prot_from_flags(rwx), PAGE_EXECUTE_READWRITE);
        // Exec-only is exact on NX-capable HW — Windows has PAGE_EXECUTE.
        assert_eq!(
            win32_prot_from_flags(x),
            PAGE_EXECUTE,
            "exec-only must use PAGE_EXECUTE (exec-only on NX-capable HW), \
             not PAGE_EXECUTE_READ which would unconditionally add read",
        );
        // Write-without-read upgrades — Windows cannot express write-only.
        assert_eq!(
            win32_prot_from_flags(w),
            PAGE_READWRITE,
            "W upgrades to RW (Windows has no write-only primitive)",
        );
        assert_eq!(
            win32_prot_from_flags(wx),
            PAGE_EXECUTE_READWRITE,
            "W+X must upgrade to RWX, not collapse to PAGE_READWRITE — \
             silently dropping the exec bit would fault on instruction fetch",
        );
    }
}
