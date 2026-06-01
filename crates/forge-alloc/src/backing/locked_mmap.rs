//! `LockedMmapBacked` — OS-mapped anonymous region with pages locked into
//! physical RAM.
//!
//! Combines `mmap`/`VirtualAlloc` (from the inner [`MmapBacked`]) with
//! `mlock`/`VirtualLock` to ensure the region's pages are **never paged out
//! to swap**. The primary use case is storing cryptographic secrets:
//! pair with [`ZeroizeOnFree`](crate::ZeroizeOnFree) so secrets are scrubbed
//! on deallocation *and* never silently written to swap.
//!
//! # Threat-model boundary
//!
//! `mlock`/`VirtualLock` guarantees these pages are **not paged to swap**.
//! It does NOT prevent:
//!
//! - **Hibernation / suspend-to-disk** — the OS writes all physical RAM,
//!   including locked pages, to the hibernate file. So "never written to disk"
//!   is too strong; the correct claim is "never paged to *swap*". Hibernation
//!   can still land the pages on disk.
//! - **Core dumps** — locked pages appear in a crash core file unless
//!   explicitly excluded (on Linux, `MADV_DONTDUMP` is applied best-effort
//!   after a successful `mlock`; on other platforms it is the caller's
//!   responsibility, e.g. `RLIMIT_CORE=0`).
//! - **`fork()` COW** — the child inherits the locked mapping AND the secret;
//!   both processes hold the data after `fork`.
//! - **`ptrace` / `/proc/<pid>/mem`** — a debugger with sufficient permission
//!   can read the pages directly regardless of the lock.
//!
//! Additionally, `LockedMmapBacked` does **NOT scrub memory on free**. Pair it
//! with [`ZeroizeOnFree`](crate::ZeroizeOnFree) to erase secrets at
//! deallocation time.
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

use super::mmap::{mmap_last_os_error, mmap_record_os_error, mmap_set_last_os_error, MmapBacked};

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
            // `os_lock` already captured errno/GetLastError into the module
            // thread-local. Snapshot it NOW before `inner` drops: the inner's
            // Drop calls `munmap`/`VirtualFree`, which can overwrite the slot
            // via `capture_os_error` if *that* syscall also fails. Restore the
            // lock error after the drop so callers see the root cause, not a
            // spurious munmap error.
            let saved = mmap_last_os_error()
                .as_ref()
                .and_then(std::io::Error::raw_os_error);
            drop(inner);
            if let Some(code) = saved {
                mmap_set_last_os_error(code);
            }
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

    /// Note: `mprotect`/`VirtualProtect` changes do NOT release the
    /// `mlock`/`VirtualLock` — the pages stay pinned into RAM regardless of
    /// the new protection bits.
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
    debug_assert_eq!(
        size % crate::backing::mmap::page_size(),
        0,
        "os_lock: size must be page-aligned",
    );
    // SAFETY: base/size come from an alive MmapBacked — page-aligned,
    // committed, within address space. mlock(2) is safe on any valid
    // anonymous mapping.
    let rc = unsafe { libc::mlock(base.as_ptr() as *const libc::c_void, size) };
    if rc != 0 {
        // Capture errno immediately, before any other call clobbers it.
        mmap_record_os_error();
        return false;
    }
    // Linux only: ask the kernel to exclude these secret pages from core dumps.
    // BEST-EFFORT: if madvise(MADV_DONTDUMP) fails, ignore the failure — the
    // mlock is the hard security guarantee; DONTDUMP is defence-in-depth only.
    // Not called on non-Linux Unix (no portable equivalent).
    #[cfg(target_os = "linux")]
    unsafe {
        libc::madvise(
            base.as_ptr() as *mut libc::c_void,
            size,
            libc::MADV_DONTDUMP,
        );
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
    debug_assert_eq!(
        size % crate::backing::mmap::page_size(),
        0,
        "os_unlock: size must be page-aligned",
    );
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
    debug_assert_eq!(
        size % crate::backing::mmap::page_size(),
        0,
        "os_lock: size must be page-aligned",
    );
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
    debug_assert_eq!(
        size % crate::backing::mmap::page_size(),
        0,
        "os_unlock: size must be page-aligned",
    );
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

    // Serializes all tests that construct a `LockedMmapBacked`. Without this,
    // the `unix_fail_closed_via_setrlimit` test — which sets a process-global
    // `RLIMIT_MEMLOCK=0` — can race with any concurrent success-path test and
    // cause it to spuriously fail. Rust runs tests multi-threaded in a single
    // process; taking this lock in *every* mlock-exercising test ensures the
    // setrlimit window is isolated.
    //
    // Note: `VirtualLock`-failure on Windows has no deterministic test
    // because forcing it to fail requires working-set manipulation.  The Unix
    // `unix_fail_closed_via_setrlimit` test covers the lock-failure branch on
    // Unix.
    static MLOCK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The trait surface is correct: base/size are populated, base is
    /// page-aligned, size is >= the requested amount.
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn trait_surface_single_page() {
        let _guard = MLOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _guard = MLOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _guard = MLOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _guard = MLOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _guard = MLOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _guard = MLOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

    /// Fail-closed: if the mapping itself cannot be allocated, the OS error is
    /// captured and `Err` is returned. This exercises the *mmap-allocation*
    /// failure path (not the lock-failure path) by requesting an impossibly
    /// large region; the same `Err + captured-error` contract applies to both
    /// failure points.
    ///
    /// (The lock-failure branch on Unix is covered by
    /// `unix_fail_closed_via_setrlimit`. The `VirtualLock`-failure branch on
    /// Windows has no deterministic test; see the `MLOCK_TEST_LOCK` comment.)
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn fail_closed_on_mapping_failure_captures_error() {
        // No MLOCK_TEST_LOCK needed here: the huge request fails at mmap time
        // before any mlock/VirtualLock is attempted, so it cannot interact
        // with the setrlimit test's RLIMIT_MEMLOCK window.
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
    /// Reduces `RLIMIT_MEMLOCK` to 0, attempts to lock one page, and asserts
    /// that `Err` is returned with a captured errno.
    ///
    /// Takes `MLOCK_TEST_LOCK` to prevent any concurrent success-path test
    /// from racing the process-global limit change. The original limit is
    /// saved and restored around the test; the env-var
    /// `FORGE_ALLOC_SKIP_SETRLIMIT_TEST=1` skips it entirely.
    #[cfg(unix)]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
    fn unix_fail_closed_via_setrlimit() {
        if std::env::var_os("FORGE_ALLOC_SKIP_SETRLIMIT_TEST").is_some() {
            return;
        }

        let _guard = MLOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

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
        let restore_rc = unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &original) };
        // Restoring a limit we just lowered should never fail. If it somehow
        // did, the rest of this test process would run with RLIMIT_MEMLOCK=0
        // and every other mlock test would fail confusingly — surface it loudly
        // rather than letting it cascade silently.
        assert_eq!(
            restore_rc, 0,
            "failed to restore RLIMIT_MEMLOCK; remaining mlock tests would spuriously fail",
        );

        assert!(result.is_err(), "mlock must fail when RLIMIT_MEMLOCK=0");
        assert!(
            mmap_last_os_error().is_some(),
            "errno must be captured when mlock fails due to RLIMIT_MEMLOCK",
        );
    }
}
