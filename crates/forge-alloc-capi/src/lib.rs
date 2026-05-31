//! # forge-alloc-capi
//!
//! A C ABI over [`forge_alloc`] for **C and C++**, aimed at **embedded /
//! bare-metal** users who want a small hardened allocator without rewriting
//! in Rust.
//!
//! It exposes a single allocator: a **bump arena** ([`forge_alloc::BumpArena`])
//! over a **caller-provided buffer** ([`forge_alloc::StaticBacked`]). That is
//! the natural embedded allocator — linear O(1) allocation out of a `.bss` /
//! linker-provided region, with bulk reclaim via [`forge_bump_reset`]. It pulls
//! in neither a global allocator nor any syscalls, so it works on
//! `#![no_std]` targets down to Cortex-M and wasm.
//!
//! ## Hardening
//!
//! [`forge_bump_free`] overwrites the freed block with `forge_alloc`'s poison
//! pattern ([`forge_alloc::DEFAULT_POISON`], `0xDE`) before reclaiming it, so a
//! stale secret cannot be recovered through a use-after-free read. This mirrors
//! [`forge_alloc::PoisonOnFree`] applied over a bump arena (whose `deallocate`
//! is otherwise a no-op, so the poison persists in full).
//!
//! ## Thread safety
//!
//! A bump arena is **not** thread-safe (`!Sync`). The caller must serialize all
//! calls that touch the same handle. This matches typical single-threaded
//! embedded use.
//!
//! ## Lifetimes / ownership
//!
//! The handle borrows the caller's buffer for as long as it lives. The buffer
//! must stay valid and unaliased from [`forge_bump_init`] until
//! [`forge_bump_destroy`]. The handle stores no heap state of its own; its
//! storage is the caller-provided `forge_bump_t` (see `forge_alloc.h`).

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use core::ffi::{c_int, c_void};
use core::ptr;
use core::slice;

use forge_alloc::{Allocator, BumpArena, NonZeroLayout, StaticBacked, DEFAULT_POISON};

/// The concrete allocator that lives inside a [`ForgeBump`] handle.
///
/// The `'static` lifetime is an FFI fiction: the caller promises (via the
/// safety contract on [`forge_bump_init`]) that the borrowed buffer outlives
/// the handle.
type Bump = BumpArena<StaticBacked<'static>>;

/// Bytes reserved for the opaque handle in the C header (`forge_bump_t`).
///
/// Kept in sync by hand with `FORGE_BUMP_STORAGE` in `include/forge_alloc.h`.
/// The static assertions below guarantee the real allocator fits; if a future
/// change grows `Bump` past this, the crate fails to compile rather than
/// silently corrupting caller storage.
const HANDLE_STORAGE: usize = 48;

const _: () = assert!(
    core::mem::size_of::<Bump>() <= HANDLE_STORAGE,
    "BumpArena<StaticBacked> grew past the C handle storage; bump FORGE_BUMP_STORAGE",
);
const _: () = assert!(
    core::mem::align_of::<Bump>() <= core::mem::align_of::<*const c_void>(),
    "BumpArena<StaticBacked> requires stronger alignment than the C handle provides",
);

/// Opaque, caller-allocated handle. Mirrors `forge_bump_t` in `forge_alloc.h`.
///
/// Treat it as a fixed-size byte blob: place it in static storage (embedded) or
/// on the stack, hand its address to the `forge_bump_*` functions, and never
/// inspect its fields. The zero-length pointer array forces the same alignment
/// the C side gets from a `void*`-containing union.
#[repr(C)]
pub struct ForgeBump {
    _align: [*const c_void; 0],
    _storage: [u8; HANDLE_STORAGE],
}

/// Initialize a bump arena over a caller-owned buffer.
///
/// Returns 1 on success, 0 if `handle`/`buf` is null, or `len` is 0 or exceeds
/// `isize::MAX` (the maximum size of a single Rust allocation/slice).
///
/// # Safety
///
/// - `handle` must point to writable storage of at least `sizeof(forge_bump_t)`
///   bytes, aligned to `_Alignof(void*)`. Re-initializing a handle overwrites
///   it without running the previous handle's `forge_bump_destroy`; since the
///   arena owns no heap this leaks nothing, but any prior borrow of the old
///   buffer is silently dropped, so you must not have outstanding allocations
///   you still intend to use.
/// - `buf` must point to `len` writable bytes that remain valid and unaliased
///   for the entire lifetime of the handle — that is, until
///   [`forge_bump_destroy`] (or until you drop the buffer, whichever is first).
#[no_mangle]
pub unsafe extern "C" fn forge_bump_init(
    handle: *mut ForgeBump,
    buf: *mut c_void,
    len: usize,
) -> c_int {
    // `len == 0` has no usable arena; `len > isize::MAX` would make the slice
    // below violate `from_raw_parts_mut`'s size ceiling (reachable on 16-/32-bit
    // no_std targets, where it would otherwise be instant UB).
    if handle.is_null() || buf.is_null() || len == 0 || len > isize::MAX as usize {
        return 0;
    }
    // SAFETY: the caller guarantees `buf..buf+len` is valid for the handle's
    // lifetime; we erase that to 'static per the documented contract. `len` is
    // bounded by `isize::MAX` above, satisfying the slice size requirement.
    let region: &'static mut [u8] = unsafe { slice::from_raw_parts_mut(buf.cast::<u8>(), len) };
    match BumpArena::new(StaticBacked::new(region)) {
        Ok(arena) => {
            // SAFETY: `handle` is writable and aligned per the contract, and
            // the static assertions above prove `Bump` fits in the storage.
            unsafe { ptr::write(handle.cast::<Bump>(), arena) };
            1
        }
        // size == 0 is already rejected above; this covers the address-space
        // wrap guard inside BumpArena::new on small no_std targets.
        Err(_) => 0,
    }
}

/// Allocate `size` bytes aligned to `align` (a power of two). Returns a pointer
/// into the caller's buffer, or null on failure (exhausted arena, `size == 0`,
/// or invalid alignment).
///
/// # Safety
///
/// `handle` must have been initialized by [`forge_bump_init`] and not since
/// destroyed. No other call may touch the same handle concurrently.
#[no_mangle]
pub unsafe extern "C" fn forge_bump_alloc(
    handle: *mut ForgeBump,
    size: usize,
    align: usize,
) -> *mut c_void {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let layout = match NonZeroLayout::from_size_align(size, align) {
        Ok(layout) => layout,
        Err(_) => return ptr::null_mut(),
    };
    // SAFETY: initialized handle per contract; `!Sync`, caller serializes.
    let arena: &Bump = unsafe { &*handle.cast::<Bump>() };
    match arena.allocate(layout) {
        Ok(block) => block.cast::<c_void>().as_ptr(),
        Err(_) => ptr::null_mut(),
    }
}

/// Like [`forge_bump_alloc`], but the returned block is zero-initialized.
///
/// # Safety
///
/// Same contract as [`forge_bump_alloc`].
#[no_mangle]
pub unsafe extern "C" fn forge_bump_alloc_zeroed(
    handle: *mut ForgeBump,
    size: usize,
    align: usize,
) -> *mut c_void {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let layout = match NonZeroLayout::from_size_align(size, align) {
        Ok(layout) => layout,
        Err(_) => return ptr::null_mut(),
    };
    // SAFETY: initialized handle per contract; `!Sync`, caller serializes.
    let arena: &Bump = unsafe { &*handle.cast::<Bump>() };
    match arena.allocate_zeroed(layout) {
        Ok(block) => block.cast::<c_void>().as_ptr(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a block. The bytes are overwritten with the poison pattern
/// ([`forge_alloc::DEFAULT_POISON`]) so freed secrets can't be recovered via a
/// use-after-free read. Space is not individually reclaimed — a bump arena's
/// `deallocate` is a no-op; reclaim happens in bulk via [`forge_bump_reset`].
/// This call exists for hygiene (scrubbing) and API symmetry. A null `ptr` (or
/// `size == 0`) is ignored. `align` is accepted for symmetry but unused, since
/// the scrub depends only on `ptr`/`size`.
///
/// The poison scrub is `O(size)`. Since reclaim is [`forge_bump_reset`]'s job,
/// a caller that does **not** need use-after-free scrubbing should simply skip
/// `free` and reclaim in bulk — `free` does no other work, so not calling it is
/// the fast path.
///
/// # Safety
///
/// `ptr`/`size` must name a live block previously returned by
/// [`forge_bump_alloc`] / [`forge_bump_alloc_zeroed`] from this same `handle`,
/// not yet freed. **A `size` larger than the original block scrubs past its end
/// — an out-of-bounds write (undefined behavior), not a graceful error.** No
/// concurrent call may touch the same handle.
#[no_mangle]
pub unsafe extern "C" fn forge_bump_free(
    handle: *mut ForgeBump,
    ptr: *mut c_void,
    size: usize,
    align: usize,
) {
    // A bump arena's `deallocate` is a no-op, so freeing reduces to the poison
    // scrub; `handle` and `align` are accepted for API symmetry but unused.
    // Scrubbing unconditionally (rather than gating on a valid layout) avoids a
    // silent no-scrub when a caller passes a non-power-of-two `align`.
    let _ = (handle, align);
    if ptr.is_null() || size == 0 {
        return;
    }
    // SAFETY: caller guarantees a live `size`-byte block at `ptr` (see # Safety).
    unsafe { ptr::write_bytes(ptr.cast::<u8>(), DEFAULT_POISON, size) };
}

/// Reclaim every allocation at once, resetting the cursor to the start of the
/// buffer. Returns 1 on success. After this, every pointer previously returned
/// by this handle is invalid.
///
/// # Safety
///
/// `handle` must be initialized. This needs exclusive access — no other call
/// may touch the same handle concurrently, and you must not use any
/// previously-issued pointer afterward.
#[no_mangle]
pub unsafe extern "C" fn forge_bump_reset(handle: *mut ForgeBump) -> c_int {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: initialized handle; exclusive access per contract.
    let arena: &mut Bump = unsafe { &mut *handle.cast::<Bump>() };
    // Inherent `BumpArena::reset` returns `()` and always succeeds.
    arena.reset();
    1
}

/// Bytes still available for allocation. Returns 0 for a null handle.
///
/// # Safety
///
/// `handle` must be initialized and not concurrently mutated.
#[no_mangle]
pub unsafe extern "C" fn forge_bump_remaining(handle: *const ForgeBump) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: initialized handle per contract.
    let arena: &Bump = unsafe { &*handle.cast::<Bump>() };
    arena.remaining()
}

/// Total capacity of the arena in bytes (the buffer length). Returns 0 for a
/// null handle.
///
/// # Safety
///
/// `handle` must be initialized and not concurrently mutated.
#[no_mangle]
pub unsafe extern "C" fn forge_bump_capacity(handle: *const ForgeBump) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: initialized handle per contract.
    let arena: &Bump = unsafe { &*handle.cast::<Bump>() };
    arena.capacity()
}

/// Bytes currently handed out. Returns 0 for a null handle.
///
/// # Safety
///
/// `handle` must be initialized and not concurrently mutated.
#[no_mangle]
pub unsafe extern "C" fn forge_bump_allocated(handle: *const ForgeBump) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: initialized handle per contract.
    let arena: &Bump = unsafe { &*handle.cast::<Bump>() };
    arena.allocated()
}

/// Tear down a handle. The arena owns no heap, so this only drops the borrow of
/// the caller's buffer; after it returns the buffer may be reused or freed. A
/// null handle is ignored.
///
/// # Safety
///
/// `handle` must have been initialized by [`forge_bump_init`] and not already
/// destroyed.
#[no_mangle]
pub unsafe extern "C" fn forge_bump_destroy(handle: *mut ForgeBump) {
    if handle.is_null() {
        return;
    }
    // SAFETY: initialized handle; drops the BumpArena in place. Storage itself
    // is owned by the caller and is not freed here.
    unsafe { ptr::drop_in_place(handle.cast::<Bump>()) };
}

// ---------------------------------------------------------------------------
// Optional runtime lang items for a SELF-CONTAINED `staticlib` linked into a
// pure C/C++ firmware image, which provides no Rust `#[panic_handler]` /
// `#[global_allocator]`. A Rust `staticlib`/`cdylib` is a final artifact and
// must define both; `forge-alloc` references `alloc`, so the allocator symbol
// is required even though the bump C surface never touches the global heap.
//
// Enable with `--no-default-features --features staticlib-rt`. Do NOT enable
// when a Rust `no_std` consumer links the rlib and supplies its own lang items
// (the duplicate definitions would collide). Embedded builds also use
// `panic = "abort"`, which bare-metal targets select by default.
// ---------------------------------------------------------------------------
#[cfg(all(not(feature = "std"), feature = "staticlib-rt"))]
mod staticlib_rt {
    use core::alloc::{GlobalAlloc, Layout};

    /// A global allocator that never succeeds. The bump C API allocates only
    /// out of the caller's buffer and never calls the global heap, so this is
    /// a link-time placeholder that is never actually invoked at runtime.
    struct AbortAlloc;

    // SAFETY: a no-op allocator that always reports failure is trivially sound
    // — it hands out no memory and frees none.
    unsafe impl GlobalAlloc for AbortAlloc {
        unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
            core::ptr::null_mut()
        }
        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
    }

    #[global_allocator]
    static GLOBAL: AbortAlloc = AbortAlloc;

    #[panic_handler]
    fn panic(_info: &core::panic::PanicInfo) -> ! {
        loop {}
    }
}
