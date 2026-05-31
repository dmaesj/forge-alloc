//! `HugePageBacked` — OS-mapped anonymous region backed by huge /
//! large pages.
//!
//! Drop-in shape with [`MmapBacked`](super::MmapBacked) (implements
//! `Allocator + FixedRange + OsBacked`) but the underlying mapping
//! comes from the kernel's huge-page pool. Reduces TLB pressure
//! dramatically on large arenas (one TLB entry per 2 MiB instead of
//! one per 4 KiB). Useful for long-lived working sets in the tens or
//! hundreds of MiB.
//!
//! Returns [`AllocError`] when the platform can't satisfy the
//! request (no huge pages reserved, missing privilege,
//! aarch64-macOS / iOS / Android). Compose with
//! [`WithFallback<HugePageBacked, MmapBacked>`](crate::WithFallback)
//! for transparent fallback to regular 4 KiB / 16 KiB pages.
//!
//! # Platform support
//!
//! | Target               | Mechanism                                                     | Prereqs |
//! |----------------------|---------------------------------------------------------------|---------|
//! | Linux                | `mmap(MAP_HUGETLB \| MAP_HUGE_<size> \| MAP_ANONYMOUS)`         | huge pages reserved in `/proc/sys/vm/nr_hugepages` (or NR=`hp_size`-specific path) |
//! | macOS x86_64         | `mmap(.., fd=libc::VM_FLAGS_SUPERPAGE_SIZE_ANY)`              | none (subject to kernel availability)  |
//! | macOS aarch64 / iOS  | unsupported (no userspace API)                                | returns `AllocError` (synthetic EINVAL) |
//! | Android / other Unix | unsupported in this version                                   | returns `AllocError` (synthetic EINVAL) |
//! | Windows              | `VirtualAlloc(MEM_LARGE_PAGES)`                               | `SeLockMemoryPrivilege` (admin / group policy) |

use core::cell::UnsafeCell;
use core::ptr::NonNull;

use forge_alloc_core::{
    AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout, OsBacked, ProtectFlags,
};

use super::mmap::{capture_synthetic_einval, mmap_record_os_error};
#[cfg(test)]
use super::mmap::{mmap_clear_last_os_error, mmap_last_os_error};

/// Platform-default huge / large page size in bytes.
///
/// Mirrors `hardening::default_huge_page_size` for non-Apple targets.
/// Duplicated here so the backing layer doesn't reach into hardening.
/// On aarch64 macOS this intentionally returns 2 MiB (not 32 MiB as
/// the hardening version does) because the backing path errors before
/// any syscall regardless of the size used for rounding.
///
/// - x86_64 / aarch64 (non-Apple) Linux & Windows: 2 MiB.
/// - aarch64 macOS (Apple Silicon, 16 KiB native granule): 2 MiB —
///   the file-level constant is used for size-rounding default
///   only; aarch64 macOS rejects every huge-page request anyway.
///   (Using 2 MiB instead of the hardening crate's 32 MiB keeps the
///   "minimum mapping you'd want to round up to" small enough not to
///   matter; the actual aarch64 macOS path errors before any
///   syscall.)
/// - Other targets: 2 MiB as a reasonable default.
#[inline]
fn default_huge_page_size() -> usize {
    2 * 1024 * 1024
}

/// The granularity the mapping length must be a multiple of.
///
/// Normally this is just `hp_size`. On macOS x86_64 the superpage size is
/// fixed at 2 MiB and `mmap(VM_FLAGS_SUPERPAGE_SIZE_ANY)` rejects any `len`
/// that is not a 2 MiB multiple — regardless of the caller's `hp_size` (which
/// that path otherwise ignores). A sub-2-MiB `hp_size` would otherwise round
/// the length to a non-2-MiB multiple and hard-fail even when superpages are
/// available, so we raise the granularity to the real superpage size there.
///
/// `hp_size` is always a power of two ≥ 4096 (validated by the caller), and
/// 2 MiB is a power of two, so the result is always a power of two ≥ `hp_size`
/// — keeping `& !(g - 1)` a valid round-up mask.
#[inline]
const fn rounding_granularity(hp_size: usize) -> usize {
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        let superpage = 2 * 1024 * 1024;
        if hp_size > superpage {
            hp_size
        } else {
            superpage
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
    {
        hp_size
    }
}

/// OS-mapped anonymous region backed by huge / large pages.
///
/// `min_size` is rounded up to a multiple of [`huge_page_size`](Self::huge_page_size)
/// at construction. The `Allocator` impl serves requests bump-style
/// from the mapping.
///
/// # Thread safety
///
/// `Send`: yes — the mapping is identified by `(ptr, len)`, both
/// `Send`-safe values; we restore `Send` via an `unsafe impl` since
/// `NonNull<u8>` is `!Send` by default.
/// `Sync`: NO. The cursor uses `UnsafeCell` for `&self` allocation;
/// concurrent `&self` allocators would race. `UnsafeCell` is `!Sync`,
/// which gives us the right behavior without any extra marker field.
pub struct HugePageBacked {
    ptr: NonNull<u8>,
    len: usize,
    huge_page_size: usize,
    cursor: UnsafeCell<usize>,
}

// !Sync is structural: the `cursor: UnsafeCell<usize>` field
// auto-derives `!Sync`, which is what `Allocator::allocate(&self, ..)`
// relies on for cursor safety. We deliberately do NOT add a
// compile-time `!Sync` assertion: stable Rust has no negative trait
// bounds, and the obvious `fn assert_not_sync<T>()` body is a no-op
// that pretends to enforce what it doesn't. Mirrors `MmapBacked`'s
// (also unguarded) shape.

impl HugePageBacked {
    /// Allocate an anonymous huge-page mapping of at least `min_size`
    /// bytes, rounded up to a multiple of the platform-default huge
    /// page size (2 MiB).
    ///
    /// Errors with [`AllocError`] if the kernel cannot satisfy the
    /// request (no reserved huge pages, missing privilege, or the
    /// platform doesn't expose a userspace huge-page API at all —
    /// notably aarch64 macOS, iOS, Android, and other Unix). Read
    /// [`crate::mmap_last_os_error`] on the failing thread for the
    /// kernel error code.
    pub fn new(min_size: usize) -> Result<Self, AllocError> {
        Self::with_huge_page_size(min_size, default_huge_page_size())
    }

    /// Same as [`new`](Self::new) but with an explicit huge page
    /// size. `hp_size` MUST be a power of two and at least 4096; any
    /// other value rejects with `AllocError`.
    ///
    /// On Linux, `hp_size` is encoded into the `mmap` flags via
    /// `MAP_HUGE_<size>` so the kernel draws from the matching pool
    /// (e.g. 2 MiB vs 1 GiB). On macOS x86_64 the parameter does not
    /// select a tier — `VM_FLAGS_SUPERPAGE_SIZE_ANY` (a fixed 2 MiB
    /// superpage) is the only meaningful selector — but the mapping
    /// length is still rounded up to at least 2 MiB there so the
    /// superpage `mmap` cannot be handed a non-2-MiB length it would
    /// reject. On Windows `hp_size` only
    /// affects size rounding, not the page tier (which is fixed by
    /// `GetLargePageMinimum()`).
    pub fn with_huge_page_size(min_size: usize, hp_size: usize) -> Result<Self, AllocError> {
        if min_size == 0 || hp_size < 4096 || !hp_size.is_power_of_two() {
            capture_synthetic_einval();
            return Err(AllocError);
        }
        // Round `min_size` up to the platform's *actual* mapping granularity
        // (see `rounding_granularity`), which may exceed `hp_size`.
        let round = rounding_granularity(hp_size);
        let len = match min_size.checked_add(round - 1).map(|s| s & !(round - 1)) {
            Some(l) => l,
            None => {
                capture_synthetic_einval();
                return Err(AllocError);
            }
        };
        // SAFETY: platform os_map_huge enforces its own invariants and
        // returns a non-null pointer to `len` writable bytes on
        // success; on failure errno / GetLastError has been recorded
        // via mmap_record_os_error.
        let ptr = unsafe { os_map_huge(len, hp_size)? };
        Ok(Self {
            ptr,
            len,
            huge_page_size: hp_size,
            cursor: UnsafeCell::new(0),
        })
    }

    /// Bytes already allocated from this backing.
    #[inline]
    pub fn allocated(&self) -> usize {
        // SAFETY: !Sync — no concurrent access to cursor.
        unsafe { *self.cursor.get() }
    }

    /// Total size of the mapping (rounded up to a multiple of the
    /// huge page size at construction).
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.len
    }

    /// Bytes remaining for allocation.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.len - self.allocated()
    }

    /// Huge page size in effect for this mapping. The mapping is a
    /// multiple of this value and `base()` is aligned to it.
    #[inline]
    pub const fn huge_page_size(&self) -> usize {
        self.huge_page_size
    }
}

impl Drop for HugePageBacked {
    fn drop(&mut self) {
        // SAFETY: ptr/len came from os_map_huge on construction; no
        // copies escape this struct (no Clone impl). Caller has by
        // contract guaranteed no outstanding pointers into the region
        // at drop time.
        unsafe { os_unmap_huge(self.ptr, self.len) };
    }
}

unsafe impl Deallocator for HugePageBacked {
    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // No-op. Bump-style; reclaim via drop.
    }
}

unsafe impl Allocator for HugePageBacked {
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
            // SAFETY: aligned_off <= len; p derives from self.ptr
            // which is non-null.
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

impl FixedRange for HugePageBacked {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.ptr
    }

    #[inline]
    fn size(&self) -> usize {
        self.len
    }
}

unsafe impl OsBacked for HugePageBacked {
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
        // SAFETY: caller has promised [ptr, ptr+size) lies wholly
        // inside our region.
        unsafe { os_release_pages_huge(ptr, size) };
    }

    #[inline]
    unsafe fn protect(&self, ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
        // SAFETY: caller has promised [ptr, ptr+size) lies inside our
        // region.
        unsafe { os_protect_huge(ptr, size, flags) };
    }
}

// SAFETY: HugePageBacked owns a `NonNull<u8>` to a non-shared OS
// mapping. Send is fine (the mapping outlives the move because
// munmap/VirtualFree is keyed on ptr+len, not on thread identity).
// !Sync is inherited from the `UnsafeCell<usize>` cursor.
unsafe impl Send for HugePageBacked {}

// ============================================================================
// Platform glue
// ============================================================================

/// Encode an `hp_size` (power-of-two byte count) into the
/// `MAP_HUGE_<size>` bits Linux expects in the `mmap` flag argument.
///
/// The kernel reads the size selector from bits
/// `MAP_HUGE_SHIFT..(MAP_HUGE_SHIFT + 6)` of the flags `int`, so we
/// compute the shift in `u32` (where any valid `log2(hp_size) << 26`
/// fits with room to spare) and bit-preserve-cast to `i32`. A naive
/// `(hp_log2 as i32) << 26` panics in debug for `hp_size >= 4 GiB`
/// (`hp_log2 = 32`, exceeding `i32::MAX` after the shift) and
/// silently wraps in release — see `libc::MAP_HUGE_16GB` which is
/// itself a wrapped-negative `c_int`, but the kernel only cares
/// about the bit pattern. Doing the math in `u32` avoids the
/// overflow trip entirely.
#[cfg(target_os = "linux")]
fn encode_huge_size_linux(hp_size: usize) -> i32 {
    let hp_log2: u32 = hp_size.trailing_zeros();
    // SAFETY of the math: hp_log2 in [12, 63] (constructor validates
    // hp_size >= 4096 and power-of-two; 4096 = 2^12, usize::MAX
    // covers 2^63). MAP_HUGE_SHIFT == 26 < 32 so the u32 shift
    // amount never panics. The shifted value (max 63 << 26 =
    // 0xFC000000) fits in u32; the bit-preserving `as i32` cast
    // gives the same bit pattern the kernel ABI expects.
    (hp_log2 << libc::MAP_HUGE_SHIFT as u32) as i32
}

#[cfg(target_os = "linux")]
unsafe fn os_map_huge(len: usize, hp_size: usize) -> Result<NonNull<u8>, AllocError> {
    // Encode the explicit huge-page-size into the mmap flags so the
    // kernel draws from the matching pool (2 MiB / 1 GiB / 16 GiB).
    // Bare `MAP_HUGETLB` defaults to whatever the kernel's default
    // huge page is — which may not match the `hp_size` we just used
    // to round `len`, giving the kernel a length it rejects with
    // EINVAL.
    let huge_flag = libc::MAP_HUGETLB | encode_huge_size_linux(hp_size);
    // SAFETY: mmap with MAP_ANONYMOUS|MAP_PRIVATE|MAP_HUGETLB plus
    // the page-size-encoding bits, non-zero len already rounded to a
    // multiple of `hp_size` by the caller. Returns MAP_FAILED on
    // error.
    let ptr = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | huge_flag,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        mmap_record_os_error();
        return Err(AllocError);
    }
    // SAFETY: mmap returned non-MAP_FAILED so ptr is a valid non-null
    // mapping of `len` writable bytes.
    Ok(unsafe { NonNull::new_unchecked(ptr as *mut u8) })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe fn os_map_huge(len: usize, _hp_size: usize) -> Result<NonNull<u8>, AllocError> {
    // Darwin overloads mmap's `fd` argument: passing
    // VM_FLAGS_SUPERPAGE_SIZE_ANY in `fd` requests a superpage.
    // Defensive assertion that the libc constant equals the value
    // documented by XNU (`SUPERPAGE_SIZE_ANY << VM_FLAGS_SUPERPAGE_SHIFT`,
    // i.e. 1 << 16). Catches any silent libc ABI change at compile
    // time.
    const _: () = assert!(libc::VM_FLAGS_SUPERPAGE_SIZE_ANY == 1 << 16);
    // SAFETY: standard mmap with MAP_ANON|MAP_PRIVATE, non-zero len
    // aligned to the superpage size by the caller, the special fd
    // value telling Darwin to request a superpage. Returns MAP_FAILED
    // on error.
    let ptr = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            libc::VM_FLAGS_SUPERPAGE_SIZE_ANY,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        mmap_record_os_error();
        return Err(AllocError);
    }
    // SAFETY: mmap returned non-MAP_FAILED.
    Ok(unsafe { NonNull::new_unchecked(ptr as *mut u8) })
}

// Apple Silicon (aarch64 macOS): no userspace API for requesting
// superpages; the kernel may auto-promote contiguous 32 MiB-aligned
// regions via Transparent Superpage Promotion, but that's
// opportunistic. Return a synthetic EINVAL so callers reading
// `mmap_last_os_error()` see an honest "not supported" rather than a
// stale value.
//
// iOS: same story — no userspace huge-page API regardless of
// arch.
//
// Android: bionic kernels typically have hugetlbfs disabled by
// default; even when enabled, `MAP_HUGETLB` requires CAP_SYS_ADMIN
// or `/proc/sys/vm/nr_hugepages` set, neither of which is realistic
// on consumer Android. Conservative: error out and let callers fall
// back to `MmapBacked`. A future release can add a true Android
// path if there's demand from Android server use cases.
//
// Other Unix (FreeBSD, NetBSD, OpenBSD, Solaris) have their own
// superpage / large-page APIs but the ABIs diverge enough that a
// separate implementation per OS is the honest path. Until those
// land, fall through to the same synthetic EINVAL.
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    target_os = "ios",
    target_os = "android",
    all(
        unix,
        not(target_os = "linux"),
        not(target_os = "macos"),
        not(target_os = "ios"),
        not(target_os = "android"),
    ),
))]
unsafe fn os_map_huge(_len: usize, _hp_size: usize) -> Result<NonNull<u8>, AllocError> {
    capture_synthetic_einval();
    Err(AllocError)
}

#[cfg(windows)]
unsafe fn os_map_huge(len: usize, _hp_size: usize) -> Result<NonNull<u8>, AllocError> {
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, MEM_COMMIT, MEM_LARGE_PAGES, MEM_RESERVE, PAGE_READWRITE,
    };
    // SAFETY: VirtualAlloc with NULL base, MEM_LARGE_PAGES + commit +
    // reserve, requires the calling process token to hold
    // SeLockMemoryPrivilege; returns NULL with
    // ERROR_PRIVILEGE_NOT_HELD if not. Captured via the per-thread
    // last-error slot below.
    let p = unsafe {
        VirtualAlloc(
            core::ptr::null_mut(),
            len,
            MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
            PAGE_READWRITE,
        )
    };
    let nn = NonNull::new(p as *mut u8);
    if nn.is_none() {
        mmap_record_os_error();
    }
    nn.ok_or(AllocError)
}

#[cfg(unix)]
unsafe fn os_unmap_huge(ptr: NonNull<u8>, len: usize) {
    // SAFETY: ptr/len came from os_map_huge on construction; munmap
    // is the only safe way to release the mapping. Drop path can't
    // propagate Err so we record errno for observability.
    let rc = unsafe { libc::munmap(ptr.as_ptr() as *mut libc::c_void, len) };
    if rc != 0 {
        mmap_record_os_error();
    }
}

#[cfg(windows)]
unsafe fn os_unmap_huge(ptr: NonNull<u8>, _len: usize) {
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
    // SAFETY: VirtualFree with MEM_RELEASE releases both reservation
    // and commit; `_len` is intentionally unused — Win32 requires
    // `dwSize == 0` for MEM_RELEASE. Errors recorded via thread-local.
    let ok = unsafe { VirtualFree(ptr.as_ptr() as *mut _, 0, MEM_RELEASE) };
    if ok == 0 {
        mmap_record_os_error();
    }
}

#[cfg(unix)]
unsafe fn os_release_pages_huge(ptr: NonNull<u8>, size: usize) {
    // Hugetlbfs purge semantics are kernel-version-dependent:
    //
    // - Linux <5.18: MADV_DONTNEED on a hugetlb mapping returns
    //   EINVAL — the kernel rejects per-page purge on huge pages.
    //   We'd just pollute the per-thread error slot.
    // - Linux >=5.18: MADV_DONTNEED is accepted and operates at
    //   hugepage granularity. Functionally useful, but the surface
    //   contract callers see (issue advice, ignore failure) is
    //   compatible with the older behavior.
    // - macOS / other Unix: MADV_FREE applies (huge / superpage
    //   mappings on Darwin honor it the same as regular mappings).
    //
    // We issue the advice unconditionally; older Linux's EINVAL is
    // captured into the error slot but caller policy decides
    // whether that's an actionable signal.
    #[cfg(target_os = "linux")]
    let advice = libc::MADV_DONTNEED;
    #[cfg(not(target_os = "linux"))]
    let advice = libc::MADV_FREE;
    // SAFETY: ptr/size lie wholly inside our own mapping (per
    // OsBacked::release_pages caller contract).
    let rc = unsafe { libc::madvise(ptr.as_ptr() as *mut libc::c_void, size, advice) };
    if rc != 0 {
        mmap_record_os_error();
    }
}

#[cfg(windows)]
unsafe fn os_release_pages_huge(ptr: NonNull<u8>, size: usize) {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_RESET, PAGE_READWRITE};
    // MEM_RESET on a large-page region: the documented behavior is
    // that the OS treats the contents as discardable. The operation
    // may be effectively a no-op at large-page granularity but it's
    // safe to issue.
    let hp = super::mmap::page_size();
    debug_assert_eq!((ptr.as_ptr() as usize) % hp, 0);
    debug_assert_eq!(size % hp, 0);
    // SAFETY: VirtualAlloc(MEM_RESET) tells the OS contents are
    // discardable; lpProtect is ignored for MEM_RESET but must be
    // valid.
    let p = unsafe { VirtualAlloc(ptr.as_ptr() as *mut _, size, MEM_RESET, PAGE_READWRITE) };
    if p.is_null() {
        mmap_record_os_error();
    }
}

#[cfg(unix)]
unsafe fn os_protect_huge(ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
    // Reuse mmap.rs's single source of truth for prot mapping so the
    // two backings can't silently diverge on a security-relevant
    // table.
    let prot = super::mmap::unix_prot_from_flags(flags);
    // SAFETY: mprotect on a region we own with valid flag bits.
    let rc = unsafe { libc::mprotect(ptr.as_ptr() as *mut libc::c_void, size, prot) };
    if rc != 0 {
        mmap_record_os_error();
    }
}

#[cfg(windows)]
unsafe fn os_protect_huge(ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    // Same write-without-read assertion as mmap.rs::os_protect: the
    // Win32 ABI cannot express write-without-read, so callers asking
    // for it get a silent over-grant. Surface in dev so the misuse
    // is caught in tests rather than in a security audit.
    debug_assert!(
        !flags.write || flags.read,
        "os_protect_huge: write-without-read upgrades to RW/RWX on Windows",
    );
    let prot = super::mmap::win32_prot_from_flags(flags);
    let mut old: u32 = 0;
    // SAFETY: VirtualProtect on a region we own with valid PAGE_*.
    let ok = unsafe { VirtualProtect(ptr.as_ptr() as *mut _, size, prot, &mut old) };
    if ok == 0 {
        mmap_record_os_error();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Huge-page allocation requires actual kernel support (Linux:
    // reserved huge pages; macOS x86_64: kernel still allows it;
    // Windows: SeLockMemoryPrivilege). CI runners typically have
    // NONE of these, so allocation tests are gated behind a runtime
    // probe AND opt-in via the `FORGE_ALLOC_HUGE_PAGES_AVAILABLE`
    // env var. The Apple-Silicon path always errors and is verified
    // directly.

    #[test]
    fn default_huge_page_size_is_sane() {
        let s = default_huge_page_size();
        assert!(s.is_power_of_two());
        assert_eq!(s, 2 * 1024 * 1024, "default is 2 MiB");
    }

    /// The length-rounding granularity is always a power of two and never
    /// below the requested `hp_size`, on every platform. On macOS x86_64 it
    /// is additionally raised to the fixed 2 MiB superpage size so a sub-2-MiB
    /// `hp_size` cannot produce a length the superpage `mmap` would reject.
    #[test]
    fn rounding_granularity_is_pow2_and_at_least_hp_size() {
        for hp in [4096_usize, 1 << 20, 1 << 21, 1 << 30] {
            let g = rounding_granularity(hp);
            assert!(
                g.is_power_of_two(),
                "granularity {g} for hp_size={hp} not pow2"
            );
            assert!(g >= hp, "granularity {g} below hp_size={hp}");
            #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
            assert!(
                g >= 2 * 1024 * 1024,
                "macOS x86_64 must round to >= 2 MiB superpage (hp_size={hp}, g={g})",
            );
        }
    }

    #[test]
    fn rejects_zero_min_size() {
        mmap_clear_last_os_error();
        assert!(HugePageBacked::new(0).is_err());
        assert!(mmap_last_os_error().is_some());
    }

    #[test]
    fn rejects_bad_huge_page_size() {
        for bad in [3 * 1024 * 1024_usize, 2048, 512, 0] {
            mmap_clear_last_os_error();
            assert!(
                HugePageBacked::with_huge_page_size(1 << 21, bad).is_err(),
                "hp_size={bad} should be rejected",
            );
            assert!(
                mmap_last_os_error().is_some(),
                "synthetic EINVAL must be captured for hp_size={bad}",
            );
        }
    }

    /// Apple Silicon path: every request errors with synthetic EINVAL.
    /// Verifies the synthetic-error capture path works without
    /// touching real `errno`.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn aarch64_macos_always_errors_with_einval() {
        mmap_clear_last_os_error();
        assert!(HugePageBacked::new(2 * 1024 * 1024).is_err());
        let e = mmap_last_os_error().expect("synthetic EINVAL captured");
        assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
    }

    /// Compile-time witness that the Darwin x86_64 superpage flag
    /// matches XNU's documented value. Re-evaluates if libc ever
    /// changes the constant.
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    #[test]
    fn macos_x86_64_superpage_flag_constant_matches_xnu() {
        assert_eq!(libc::VM_FLAGS_SUPERPAGE_SIZE_ANY, 1 << 16);
    }

    /// Regression: the bit-pattern produced by `encode_huge_size_linux`
    /// for `hp_size = 2 MiB` and `1 GiB` must equal `libc::MAP_HUGE_2MB`
    /// and `MAP_HUGE_1GB` respectively. Catches:
    ///   - off-by-one in the shift,
    ///   - signed-vs-unsigned shift overflow (an earlier rev did
    ///     `(hp_log2 as i32) << 26` which panics in debug for
    ///     `hp_size >= 4 GiB`),
    ///   - any future libc rev that renumbers `MAP_HUGE_SHIFT`.
    ///
    /// Runs on every Linux target; no kernel huge-page reservation
    /// needed.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_huge_size_encoding_matches_libc_constants() {
        assert_eq!(
            encode_huge_size_linux(2 * 1024 * 1024),
            libc::MAP_HUGE_2MB,
            "2 MiB encoding mismatch",
        );
        assert_eq!(
            encode_huge_size_linux(1024 * 1024 * 1024),
            libc::MAP_HUGE_1GB,
            "1 GiB encoding mismatch",
        );
        // 16 GiB: `34 << 26` overflows i32 if you do the shift in
        // signed arithmetic. The helper does it in u32 so this
        // must succeed.
        //
        // Gate on 64-bit pointer width: `16_usize * 1024 * 1024 * 1024`
        // = 2^34 overflows `usize` on 32-bit Linux (i686, armv7l) where
        // `usize` = `u32`. 16 GiB huge-page pools are 64-bit only in
        // practice, so the test restriction matches reality.
        #[cfg(target_pointer_width = "64")]
        {
            let sixteen_gib = 16_usize * 1024 * 1024 * 1024;
            let encoded = encode_huge_size_linux(sixteen_gib);
            // Compare against the libc constant when present; libc gates
            // MAP_HUGE_16GB on the same targets that expose MAP_HUGE_SHIFT,
            // so this should always be available on 64-bit Linux.
            assert_eq!(encoded, libc::MAP_HUGE_16GB, "16 GiB encoding mismatch");
        }
    }

    /// Probe whether the platform can actually allocate huge pages
    /// in this test environment. Only attempts the allocation when
    /// `FORGE_ALLOC_HUGE_PAGES_AVAILABLE=1` is set in the
    /// environment — otherwise returns `None` so dependent tests
    /// silently skip. This avoids both false-positive flakes on CI
    /// AND silent green-when-nothing-tested confusion: opting in
    /// explicitly is the only way to exercise the live path.
    fn try_huge_alloc() -> Option<HugePageBacked> {
        if std::env::var_os("FORGE_ALLOC_HUGE_PAGES_AVAILABLE").as_deref()
            != Some(std::ffi::OsStr::new("1"))
        {
            return None;
        }
        HugePageBacked::new(default_huge_page_size()).ok()
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn alloc_then_write_then_read_back_when_supported() {
        let Some(m) = try_huge_alloc() else {
            return;
        };
        let layout = NonZeroLayout::from_size_align(256, 8).unwrap();
        let block = m.allocate(layout).unwrap();
        let p = block.cast::<u8>();
        // SAFETY: 256 bytes we just allocated; in-bounds writes / reads.
        unsafe {
            core::ptr::write_bytes(p.as_ptr(), 0xAB, 256);
            for i in 0..256 {
                assert_eq!(*p.as_ptr().add(i), 0xAB);
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn capacity_is_huge_page_rounded_when_supported() {
        let Some(m) = try_huge_alloc() else {
            return;
        };
        let cap = m.capacity();
        let hp = m.huge_page_size();
        assert_eq!(cap % hp, 0);
        assert!(cap >= hp);
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn base_is_huge_page_aligned_when_supported() {
        let Some(m) = try_huge_alloc() else {
            return;
        };
        let base = m.base().as_ptr() as usize;
        let hp = m.huge_page_size();
        assert_eq!(base % hp, 0, "base must be aligned to the huge page size");
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri can't shim mmap / VirtualAlloc")]
    fn fixed_range_contains_allocations_when_supported() {
        let Some(m) = try_huge_alloc() else {
            return;
        };
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let block = m.allocate(layout).unwrap();
        assert!(m.contains(block.cast::<u8>()));
    }
}
