//! `LockedMmapBacked` — OS-mapped anonymous region with pages locked into
//! physical RAM.
//!
//! Combines `mmap`/`VirtualAlloc` (from the inner [`MmapBacked`]) with
//! `mlock`/`VirtualLock` to ensure the region's pages are **never paged out
//! to swap or disk**. The primary use case is storing cryptographic secrets:
//! pair with [`ZeroizeOnFree`](crate::ZeroizeOnFree) so secrets are scrubbed
//! on deallocation *and* never silently written to swap.
//!
//! ## Design: composition over re-implementation
//!
//! `LockedMmapBacked` wraps an eagerly-committed [`MmapBacked`] and adds the
//! lock/unlock lifecycle around it. It does NOT re-implement `mmap`/`munmap`;
//! the vetted lifecycle in `MmapBacked` is reused as-is. The only additions
//! are the `mlock`/`VirtualLock` call in the constructor and the
//! `munlock`/`VirtualUnlock` call in `Drop` (before the inner's `munmap` runs
//! as the `inner` field drops after the body).
//!
//! ## Fail-closed guarantee
//!
//! If the lock syscall fails (e.g. `RLIMIT_MEMLOCK` exceeded, missing
//! `CAP_IPC_LOCK`/`SeLockMemoryPrivilege`, or the working-set minimum is too
//! small on Windows), the constructor returns `Err(AllocError)`. The local
//! `inner` value then drops, releasing the mapping — no leak. A
//! `LockedMmapBacked` that could silently fall back to unlocked memory would
//! defeat its entire security purpose, so the fail-closed behavior is
//! non-negotiable and is not a configuration option.
//!
//! The OS error code (errno on Unix, `GetLastError` on Windows) is captured
//! into the per-thread last-error slot immediately after the failing syscall
//! so callers can read it via [`mmap_last_os_error`](super::mmap::mmap_last_os_error).
//!
//! ## `release_pages` is intentionally a no-op
//!
//! Purging pages (`MADV_DONTNEED` / `MEM_RESET`) on a locked crypto region
//! would defeat the lock and could leave secret plaintext in reclaimed pages.
//! `release_pages` is therefore a documented no-op on this type. If you need
//! page recycling for non-secret data, use [`MmapBacked`] directly.
//!
//! ## Move-safety
//!
//! Moving a `LockedMmapBacked` just moves the `inner` field (and the lock
//! bookkeeping with it). The OS lock is keyed on the virtual address range
//! `(ptr, len)`, not on the Rust value's stack address — so a move is safe.
//! `base()` returns the inner's stored absolute OS pointer, which is stable
//! across moves.
//!
//! ## Thread safety
//!
//! `Send`: the lock follows the pages (it's keyed on ptr+len). Moving the
//! struct to another thread and using it there is safe. `Send` is inherited
//! from the `inner: MmapBacked` field.
//!
//! `Sync`: NO. Inherited from `MmapBacked`'s `UnsafeCell<usize>` cursor —
//! concurrent `&self` allocation would race the cursor without an extra
//! synchronization layer.

use core::ptr::NonNull;

use forge_alloc_core::{
    AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout, OsBacked, ProtectFlags,
};

use super::mmap::{mmap_record_os_error, MmapBacked};

/// OS-mapped anonymous region whose pages are locked into physical RAM.
///
/// See the [module-level documentation](self) for the security guarantee,
/// the fail-closed constructor contract, and composition notes.
pub struct LockedMmapBacked {
    inner: MmapBacked,
}

impl LockedMmapBacked {
    /// Allocate an anonymous OS-mapped region of at least `size` bytes
    /// (rounded up to the page size), then lock all pages into physical RAM.
    ///
    /// # Errors
    ///
    /// Returns `Err(AllocError)` if:
    /// - `size` is 0 or the page-rounded size overflows `usize`.
    /// - The underlying `mmap`/`VirtualAlloc` fails.
    /// - **The lock syscall fails** (`mlock`/`VirtualLock`). This is the
    ///   fail-closed path: if the lock is not granted, the mapping is released
    ///   and `Err` is returned. Callers MUST treat `Err` here as a hard
    ///   failure; there is no unlocked fallback.
    ///
    /// On `Err`, [`mmap_last_os_error`](crate::mmap_last_os_error) holds the
    /// OS error code from the failing syscall.
    pub fn new(size: usize) -> Result<Self, AllocError> {
        // Use eager commit (`MmapBacked::new`, not `new_lazy`): `mlock`
        // requires the pages to be committed — locking uncommitted
        // (reserved-only) pages is not meaningful and would fail on Windows.
        let inner = MmapBacked::new(size)?;
        // SAFETY: inner.base()/inner.size() are the bounds of a live,
        // committed, readable+writable OS mapping. The platform lock call
        // touches no Rust state; it only asks the OS to pin the physical
        // pages. On failure we capture the OS error and let `inner` drop
        // (which calls munmap/VirtualFree) before returning Err.
        let ok = unsafe { os_lock(inner.base(), inner.size()) };
        if !ok {
            // Capture the OS error *before* `inner` drops: `inner`'s Drop
            // calls `munmap`/`VirtualFree`, which is an OS call that can
            // clobber errno/GetLastError. The error has already been captured
            // inside `os_lock` immediately after the failing syscall, so
            // `inner` can drop safely here without loss of the error code.
            return Err(AllocError);
        }
        Ok(Self { inner })
    }
}

impl Drop for LockedMmapBacked {
    fn drop(&mut self) {
        // Unlock FIRST, while the pages are still mapped, then let `inner`
        // drop (which munmaps/VirtualFrees the region). The ordering is
        // required: unlocking an already-unmapped range is undefined behaviour
        // on most platforms.
        //
        // SAFETY: self.inner.base()/size() are still live at this point in
        // Drop (the field hasn't dropped yet; Drop bodies run before fields).
        unsafe { os_unlock(self.inner.base(), self.inner.size()) };
        // `inner` drops here automatically, unmapping the region.
    }
}

// ---- Trait forwarding -------------------------------------------------------

unsafe impl Deallocator for LockedMmapBacked {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // SAFETY: forwarded verbatim; the inner's deallocate is a no-op
        // (bump-style reclaim on drop) so this is safe under the same
        // caller contract.
        unsafe { self.inner.deallocate(ptr, layout) }
    }
}

unsafe impl Allocator for LockedMmapBacked {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner.allocate(layout)
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        self.inner.capacity_bytes()
    }
}

impl FixedRange for LockedMmapBacked {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.inner.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// The region is eagerly committed and locked, so commit is a no-op
    /// (the inner's eager `MmapBacked` also returns `Ok` immediately).
    #[inline]
    fn commit(&self, offset: usize, len: usize) -> Result<(), AllocError> {
        self.inner.commit(offset, len)
    }
}

unsafe impl OsBacked for LockedMmapBacked {
    #[inline]
    fn base_ptr(&self) -> NonNull<u8> {
        self.inner.base_ptr()
    }

    #[inline]
    fn region_size(&self) -> usize {
        self.inner.region_size()
    }

    /// Intentional no-op — purging pages on a locked crypto region would
    /// defeat the memory-lock and could leak secret plaintext into reclaimed
    /// pages. Callers that need page recycling should use [`MmapBacked`]
    /// directly.
    #[inline]
    unsafe fn release_pages(&self, _ptr: NonNull<u8>, _size: usize) {
        // Deliberately empty. See module-level docs.
    }

    #[inline]
    unsafe fn protect(&self, ptr: NonNull<u8>, size: usize, flags: ProtectFlags) {
        // SAFETY: forwarded; caller has promised [ptr, ptr+size) lies inside
        // our region, satisfying OsBacked::protect's contract.
        unsafe { self.inner.protect(ptr, size, flags) }
    }
}

// `Send` is inherited from `inner: MmapBacked` (which has `unsafe impl Send`).
// The OS lock is keyed on the virtual address range, not on thread identity —
// moving the struct to another thread is safe. `!Sync` is structural: the
// `inner` field carries `UnsafeCell`, so `LockedMmapBacked` is `!Sync`
// automatically, matching the intent (no concurrent &self allocation).

// ============================================================================
// Platform glue — lock / unlock
// ============================================================================

/// Lock all pages of `[base, base + size)` into physical RAM.
///
/// Returns `true` on success, `false` on failure (OS error captured).
///
/// # Safety
///
/// `base` must point to the start of a live, committed OS mapping of at least
/// `size` bytes. `size` must be page-aligned (guaranteed by `MmapBacked::new`).
#[cfg(unix)]
unsafe fn os_lock(base: NonNull<u8>, size: usize) -> bool {
    // SAFETY: base/size come from an alive MmapBacked — page-aligned,
    // committed, within address space. mlock(2) is safe on any valid
    // anonymous mapping.
    let rc = unsafe { libc::mlock(base.as_ptr() as *const libc::c_void, size) };
    if rc != 0 {
        // Capture errno immediately, before any other call clobbers it.
        mmap_record_os_error();
        return false;
    }
    true
}

/// Unlock all pages of `[base, base + size)`.
///
/// Errors from `munlock` are captured but cannot be propagated (called from
/// `Drop`).
///
/// # Safety
///
/// `base`/`size` must describe a range previously locked with `os_lock`.
#[cfg(unix)]
unsafe fn os_unlock(base: NonNull<u8>, size: usize) {
    // SAFETY: base/size describe a range we locked in the constructor.
    // munlock on a locked mapping is safe; errors are informational only.
    let rc = unsafe { libc::munlock(base.as_ptr() as *const libc::c_void, size) };
    if rc != 0 {
        mmap_record_os_error();
    }
}

/// Lock all pages of `[base, base + size)` into the process working set.
///
/// Returns `true` on success, `false` on failure (OS error captured).
///
/// On Windows, `VirtualLock` locks pages into the working set. It may fail
/// if the process working-set minimum is too small. This is a legitimate
/// fail-closed case; we do NOT call `SetProcessWorkingSetSize` — that is a
/// privileged global side effect that could affect the rest of the process.
///
/// # Safety
///
/// `base` must point to the start of a live, committed `VirtualAlloc` region
/// of at least `size` bytes.
#[cfg(windows)]
unsafe fn os_lock(base: NonNull<u8>, size: usize) -> bool {
    use windows_sys::Win32::System::Memory::VirtualLock;
    // SAFETY: VirtualLock on a committed, valid VirtualAlloc region.
    // Returns nonzero on success, 0 on failure (GetLastError for details).
    let ok = unsafe { VirtualLock(base.as_ptr() as *mut _, size) };
    if ok == 0 {
        // Capture GetLastError immediately.
        mmap_record_os_error();
        return false;
    }
    true
}

/// Unlock all pages of `[base, base + size)` from the process working set.
///
/// Errors are captured but cannot be propagated (called from `Drop`).
///
/// # Safety
///
/// `base`/`size` must describe a range previously locked with `os_lock`.
#[cfg(windows)]
unsafe fn os_unlock(base: NonNull<u8>, size: usize) {
    use windows_sys::Win32::System::Memory::VirtualUnlock;
    // SAFETY: VirtualUnlock on a range previously locked with VirtualLock.
    // Returns nonzero on success, 0 on failure. Drop can't propagate Err.
    let ok = unsafe { VirtualUnlock(base.as_ptr() as *mut _, size) };
    if ok == 0 {
        mmap_record_os_error();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::mmap::{mmap_clear_last_os_error, mmap_last_os_error, page_size};

    // All tests in this module exercise real OS mmap / mlock / VirtualLock
    // paths. Miri cannot model these syscalls, so all tests in this module
    // are gated off under miri.

    /// The trait surface is correct: base/size are populated, base is
    /// page-aligned, size is >= the requested amount.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn trait_surface_single_page() {
        let size = page_size();
        // Construction may legitimately fail in sandboxed environments where
        // mlock is not permitted (RLIMIT_MEMLOCK=0, container policy, etc.).
        // In that case, assert the OS error was captured and skip further
        // assertions rather than failing the test.
        mmap_clear_last_os_error();
        match LockedMmapBacked::new(size) {
            Err(_) => {
                assert!(
                    mmap_last_os_error().is_some(),
                    "construction failure must capture an OS error",
                );
                // Lock not permitted in this environment — skip.
            }
            Ok(m) => {
                assert!(
                    m.base().as_ptr() as usize % page_size() == 0,
                    "base must be page-aligned"
                );
                assert!(m.size() >= size, "size must be >= requested");
                assert_eq!(m.size() % page_size(), 0, "size must be page-aligned");
                assert_eq!(
                    m.base_ptr(),
                    m.base(),
                    "OsBacked::base_ptr must equal FixedRange::base"
                );
                assert_eq!(
                    m.region_size(),
                    m.size(),
                    "OsBacked::region_size must equal FixedRange::size"
                );
            }
        }
    }

    /// Bump-allocating from a locked region yields usable memory.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn bump_allocate_returns_usable_memory() {
        let size = 4 * page_size();
        mmap_clear_last_os_error();
        let m = match LockedMmapBacked::new(size) {
            Ok(m) => m,
            Err(_) => return, // lock not permitted — skip
        };
        let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
        let block = m
            .allocate(layout)
            .expect("allocate must succeed in a fresh region");
        let p = block.cast::<u8>().as_ptr();
        // Write a pattern through the locked memory and read it back.
        unsafe {
            core::ptr::write_bytes(p, 0xAB, 64);
            for i in 0..64 {
                assert_eq!(*p.add(i), 0xAB);
            }
        }
    }

    /// FixedRange::contains is satisfied for a pointer returned by allocate.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn fixed_range_contains_allocated_pointer() {
        let m = match LockedMmapBacked::new(page_size()) {
            Ok(m) => m,
            Err(_) => return,
        };
        let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
        let block = m.allocate(layout).unwrap();
        assert!(m.contains(block.cast::<u8>()));
    }

    /// Drop must not crash — the unlock+unmap sequence runs without panic.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn drop_does_not_crash() {
        let m = match LockedMmapBacked::new(page_size()) {
            Ok(m) => m,
            Err(_) => return,
        };
        drop(m); // explicit drop to catch any panic
    }

    /// Move-safety: `base()` is stable after moving the struct to a new
    /// location. The OS lock is keyed on (ptr, len) — not on the Rust
    /// value's stack address — so moving the struct doesn't invalidate it.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn base_is_stable_after_move() {
        let m = match LockedMmapBacked::new(page_size()) {
            Ok(m) => m,
            Err(_) => return,
        };
        let base_before = m.base();
        // Move to a new binding (stack location changes).
        let m2 = m;
        assert_eq!(
            m2.base(),
            base_before,
            "base() must be identical before and after a move",
        );
    }

    /// `release_pages` is a documented no-op — calling it must not crash and
    /// must not alter the mapped data (the region remains readable/writable).
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn release_pages_is_noop() {
        let m = match LockedMmapBacked::new(page_size()) {
            Ok(m) => m,
            Err(_) => return,
        };
        let base = m.base();
        // Write a pattern, "release" (no-op), then verify the data is intact.
        unsafe {
            core::ptr::write_bytes(base.as_ptr(), 0x5A, page_size());
            m.release_pages(base, page_size());
            assert_eq!(
                *base.as_ptr(),
                0x5A,
                "data must survive the no-op release_pages"
            );
        }
    }

    /// Fail-closed: if the lock cannot be acquired, the OS error is captured
    /// and no memory is leaked (the test can only *observe* that Err is
    /// returned and an error is captured; leak-checking requires an external
    /// tool such as Valgrind or ASAN).
    ///
    /// This test exercises the error-capture path by requesting an
    /// impossibly large mapping (which will fail at the mmap step rather than
    /// the mlock step, but the same `Err + captured-error` contract holds).
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn fail_closed_captures_os_error() {
        mmap_clear_last_os_error();
        // A near-usize::MAX request will fail at mmap time on every platform.
        let huge = usize::MAX / 2;
        let result = LockedMmapBacked::new(huge);
        assert!(result.is_err(), "impossibly large request must fail");
        assert!(
            mmap_last_os_error().is_some(),
            "OS error must be captured on construction failure",
        );
    }

    /// On Unix only: setrlimit-based fail-closed test.
    ///
    /// Reduces `RLIMIT_MEMLOCK` to a tiny value, attempts to lock more than
    /// the limit, and asserts that `Err` is returned with a captured errno.
    ///
    /// CAUTION: `setrlimit` is process-global and Rust runs tests in a shared
    /// process. To limit the race window we save and restore the original
    /// limit around the test. However, concurrent tests that also attempt
    /// `mlock` could interfere if they run in parallel on the same thread.
    /// If this test proves flaky in your environment, set
    /// `FORGE_ALLOC_SKIP_SETRLIMIT_TEST=1` to skip it.
    #[cfg(unix)]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn unix_fail_closed_via_setrlimit() {
        if std::env::var_os("FORGE_ALLOC_SKIP_SETRLIMIT_TEST").is_some() {
            return;
        }

        // Save current RLIMIT_MEMLOCK.
        let mut original = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: getrlimit is safe to call; we provide a valid out-pointer.
        let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut original) };
        if rc != 0 {
            // Can't read the limit — skip rather than risk corrupting it.
            return;
        }

        // Set a tiny soft limit (0 bytes). The hard limit is left unchanged.
        // Some kernels require rlim_max >= rlim_cur, so use the saved max.
        let tiny = libc::rlimit {
            rlim_cur: 0,
            rlim_max: original.rlim_max,
        };
        // SAFETY: setrlimit with RLIMIT_MEMLOCK and a valid rlimit struct.
        let set_rc = unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &tiny) };
        if set_rc != 0 {
            // Might need CAP_SYS_RESOURCE to lower the hard limit; skip.
            return;
        }

        // Attempt to lock one page — should fail because the limit is 0.
        mmap_clear_last_os_error();
        let result = LockedMmapBacked::new(page_size());

        // Restore the original limit before asserting (so a panic here
        // doesn't leave the process with a broken limit).
        // SAFETY: restoring the previously-read limit.
        unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &original) };

        assert!(result.is_err(), "mlock must fail when RLIMIT_MEMLOCK=0");
        assert!(
            mmap_last_os_error().is_some(),
            "errno must be captured when mlock fails due to RLIMIT_MEMLOCK",
        );
    }
}
