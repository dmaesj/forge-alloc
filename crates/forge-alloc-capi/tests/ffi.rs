//! End-to-end exercise of the C ABI from Rust, so the FFI surface is covered
//! even where a C toolchain isn't available in CI. Mirrors what the C example
//! does.

use core::mem::MaybeUninit;

use forge_alloc_capi::{
    forge_bump_alloc, forge_bump_alloc_zeroed, forge_bump_allocated, forge_bump_capacity,
    forge_bump_destroy, forge_bump_free, forge_bump_init, forge_bump_remaining, forge_bump_reset,
    ForgeBump,
};

const POISON: u8 = 0xDE;

#[test]
fn init_alloc_reset_destroy_round_trip() {
    let mut buf = [0u8; 1024];
    let mut handle = MaybeUninit::<ForgeBump>::uninit();
    let h = handle.as_mut_ptr();

    unsafe {
        assert_eq!(forge_bump_init(h, buf.as_mut_ptr().cast(), buf.len()), 1);
        assert_eq!(forge_bump_capacity(h), 1024);
        assert_eq!(forge_bump_allocated(h), 0);

        let a = forge_bump_alloc(h, 64, 8);
        let b = forge_bump_alloc(h, 128, 16);
        assert!(!a.is_null() && !b.is_null());
        assert_ne!(a, b);
        assert_eq!(a as usize % 8, 0);
        assert_eq!(b as usize % 16, 0);
        assert!(forge_bump_allocated(h) >= 192);
        assert_eq!(forge_bump_remaining(h), 1024 - forge_bump_allocated(h));

        // Write then free → bytes must be poisoned (UAF read is sound here only
        // because a bump arena's deallocate is a no-op and `buf` still lives).
        core::ptr::write_bytes(a.cast::<u8>(), 0xAB, 64);
        forge_bump_free(h, a, 64, 8);
        assert_eq!(
            *a.cast::<u8>(),
            POISON,
            "free must scrub with the poison pattern"
        );
        assert_eq!(*a.cast::<u8>().add(63), POISON);

        assert_eq!(forge_bump_reset(h), 1);
        assert_eq!(forge_bump_allocated(h), 0);

        forge_bump_destroy(h);
    }
}

#[test]
fn alloc_zeroed_is_zero() {
    let mut buf = [0xFFu8; 256];
    let mut handle = MaybeUninit::<ForgeBump>::uninit();
    let h = handle.as_mut_ptr();
    unsafe {
        assert_eq!(forge_bump_init(h, buf.as_mut_ptr().cast(), buf.len()), 1);
        let p = forge_bump_alloc_zeroed(h, 32, 8);
        assert!(!p.is_null());
        let slice = core::slice::from_raw_parts(p.cast::<u8>(), 32);
        assert!(
            slice.iter().all(|&b| b == 0),
            "alloc_zeroed must return zeros"
        );
        forge_bump_destroy(h);
    }
}

#[test]
fn exhaustion_and_invalid_args_return_null_or_zero() {
    let mut buf = [0u8; 64];
    let mut handle = MaybeUninit::<ForgeBump>::uninit();
    let h = handle.as_mut_ptr();
    unsafe {
        // Null / zero-length init fails.
        assert_eq!(
            forge_bump_init(core::ptr::null_mut(), buf.as_mut_ptr().cast(), 64),
            0
        );
        assert_eq!(forge_bump_init(h, core::ptr::null_mut(), 64), 0);
        assert_eq!(forge_bump_init(h, buf.as_mut_ptr().cast(), 0), 0);

        assert_eq!(forge_bump_init(h, buf.as_mut_ptr().cast(), buf.len()), 1);
        // size 0 and non-power-of-two align are rejected.
        assert!(forge_bump_alloc(h, 0, 8).is_null());
        assert!(forge_bump_alloc(h, 16, 3).is_null());
        // Asking for more than the buffer holds fails gracefully.
        assert!(forge_bump_alloc(h, 4096, 8).is_null());
        // A fitting allocation still succeeds afterward.
        assert!(!forge_bump_alloc(h, 16, 8).is_null());
        forge_bump_destroy(h);
    }
}

/// A failed allocation must be side-effect-free: the cursor is untouched, so
/// `allocated`/`remaining` are unchanged and a fitting request still succeeds.
/// This is the bump arena's headline guarantee, exercised here at the FFI layer.
#[test]
fn failed_alloc_is_side_effect_free() {
    let mut buf = [0u8; 64];
    let mut handle = MaybeUninit::<ForgeBump>::uninit();
    let h = handle.as_mut_ptr();
    unsafe {
        assert_eq!(forge_bump_init(h, buf.as_mut_ptr().cast(), buf.len()), 1);
        assert!(!forge_bump_alloc(h, 16, 8).is_null());
        let allocated_before = forge_bump_allocated(h);
        let remaining_before = forge_bump_remaining(h);

        // Overshoot: must fail and leave the cursor exactly where it was.
        assert!(forge_bump_alloc(h, 4096, 8).is_null());
        assert_eq!(forge_bump_allocated(h), allocated_before);
        assert_eq!(forge_bump_remaining(h), remaining_before);

        // A request that still fits succeeds.
        assert!(!forge_bump_alloc(h, 16, 8).is_null());
        forge_bump_destroy(h);
    }
}

/// `reset` rewinds the cursor, so the first allocation after a reset must reuse
/// the address of the first allocation before it — proving reclaim, not just a
/// zeroed counter.
#[test]
fn reset_reuses_addresses() {
    let mut buf = [0u8; 256];
    let mut handle = MaybeUninit::<ForgeBump>::uninit();
    let h = handle.as_mut_ptr();
    unsafe {
        assert_eq!(forge_bump_init(h, buf.as_mut_ptr().cast(), buf.len()), 1);
        let first = forge_bump_alloc(h, 32, 8);
        assert!(!first.is_null());
        let _ = forge_bump_alloc(h, 32, 8);
        assert_eq!(forge_bump_reset(h), 1);
        let again = forge_bump_alloc(h, 32, 8);
        assert_eq!(
            first, again,
            "first alloc after reset must reuse the address"
        );
        forge_bump_destroy(h);
    }
}

/// A degenerate 1-byte buffer initializes, satisfies a 1-byte request once, then
/// reports exhaustion.
#[test]
fn tiny_buffer_boundary() {
    let mut buf = [0u8; 1];
    let mut handle = MaybeUninit::<ForgeBump>::uninit();
    let h = handle.as_mut_ptr();
    unsafe {
        assert_eq!(forge_bump_init(h, buf.as_mut_ptr().cast(), 1), 1);
        assert_eq!(forge_bump_capacity(h), 1);
        assert!(!forge_bump_alloc(h, 1, 1).is_null());
        assert_eq!(forge_bump_remaining(h), 0);
        assert!(forge_bump_alloc(h, 1, 1).is_null());
        forge_bump_destroy(h);
    }
}

/// A large alignment that exceeds the buffer fails cleanly (NULL); a satisfiable
/// large alignment returns a correctly-aligned pointer.
#[test]
fn over_alignment_handling() {
    let mut buf = [0u8; 256];
    let mut handle = MaybeUninit::<ForgeBump>::uninit();
    let h = handle.as_mut_ptr();
    unsafe {
        assert_eq!(forge_bump_init(h, buf.as_mut_ptr().cast(), buf.len()), 1);
        // Alignment far larger than the whole buffer: clean failure, no UB.
        assert!(forge_bump_alloc(h, 8, 4096).is_null());
        // A 64-byte alignment fits in 256 bytes and must be honored.
        let p = forge_bump_alloc(h, 8, 64);
        assert!(!p.is_null());
        assert_eq!(
            p as usize % 64,
            0,
            "returned pointer must meet the requested alignment"
        );
        forge_bump_destroy(h);
    }
}

/// The opaque C handle must be at least as large and as aligned as the real
/// allocator — the same guarantee the in-crate static assertions enforce,
/// re-checked here against the public type.
#[test]
fn handle_layout_is_sufficient() {
    assert!(core::mem::size_of::<ForgeBump>() >= core::mem::size_of::<usize>() * 4);
    assert!(core::mem::align_of::<ForgeBump>() >= core::mem::align_of::<*const ()>());
}
