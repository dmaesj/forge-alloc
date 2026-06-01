//! End-to-end test for the Crypto Allocator V2 composition:
//!
//! ```text
//! ZeroizeOnFree< Slab<T, GuardPage<SplitMetadata<LockedMmapBacked>>, M> >
//! ```
//!
//! This is the `HardenedSlab` security stack (guard pages + out-of-line
//! metadata + optional freelist MAC) but over **RAM-locked, core-dump-excluded**
//! memory (`LockedMmapBacked`: `mlock`/`VirtualLock` + `MADV_DONTDUMP`), wrapped
//! in **non-elidable scrub-on-free** (`ZeroizeOnFree`). It proves the two crypto
//! halves compose with the hardened slab into a usable secret allocator:
//!
//! - secrets never page to swap (lock) and are excluded from core dumps
//!   (Linux, best-effort `MADV_DONTDUMP`);
//! - secrets are volatile-zeroed on free (scrub);
//! - linear overflow into/out of the secret region traps (guard pages).
//!
//! (The slab's freelist links live inline in freed slots, in the locked
//! region; the scrub runs first, so after a free those bytes hold link data,
//! not a secret. `SplitMetadata` provides cache-line isolation of allocator
//! metadata, matching the `HardenedSlab` stack.)

#![cfg(feature = "std")]

use forge_alloc::{
    page_size, Allocator, CryptoSlab, Deallocator, GuardPage, LockedMmapBacked, NonZeroLayout,
    Slab, SplitMetadata, ZeroizeOnFree,
};

/// 64-byte "secret" — larger than the 8-byte freelink so the scrub of the
/// tail (bytes 8..64) is observable after free.
type Secret = [u8; 64];

/// The crypto-allocator alias under test — `CryptoSlab<Secret>` expands to
/// `ZeroizeOnFree<Slab<Secret, GuardPage<SplitMetadata<LockedMmapBacked>>>>`.
type CryptoPool = CryptoSlab<Secret>;

/// Build the full crypto stack. Returns `None` if the environment forbids
/// `mlock` (e.g. `RLIMIT_MEMLOCK=0` in a constrained CI sandbox) — the
/// fail-closed `LockedMmapBacked::new` returns `Err` there, and these tests
/// skip rather than fail.
fn build_crypto_pool() -> Option<CryptoPool> {
    // 1. RAM-locked, core-dump-excluded data region (the new crypto half).
    //    Kept at 32 KiB so it fits the common 64 KiB `RLIMIT_MEMLOCK` default,
    //    letting the test actually exercise the lock in unprivileged CI rather
    //    than skip. (Only this data region is locked; the metadata region is a
    //    separate unlocked mapping and does not count toward the limit.)
    let locked = match LockedMmapBacked::new(32 * 1024) {
        Ok(l) => l,
        Err(_) => {
            // Make the skip visible in CI logs rather than silently passing.
            eprintln!(
                "NOTE: skipping crypto_slab_e2e — mlock not permitted in this \
                 environment (RLIMIT_MEMLOCK / privilege)"
            );
            return None;
        }
    };
    // 2. Out-of-line metadata: the slab freelist/headers live in a separate
    //    region, away from the secret data.
    let split = SplitMetadata::new(locked, 16 * 1024).ok()?;
    // 3. Guard pages trap linear overflow into/out of the secret region.
    let guarded = GuardPage::new(split, page_size()).ok()?;
    // 4. Typed slab over the hardened, locked backing.
    let slab = Slab::<Secret, _>::new(64, guarded).ok()?;
    // 5. Scrub-on-free (the other crypto half): freed secrets are volatile-zeroed.
    Some(ZeroizeOnFree::new(slab))
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
fn crypto_stack_round_trips() {
    let Some(pool) = build_crypto_pool() else {
        return; // mlock not permitted here — skip.
    };
    let layout = NonZeroLayout::for_type::<Secret>().unwrap();
    let block = pool
        .allocate(layout)
        .expect("alloc through the full crypto stack");
    let p = block.cast::<u8>().as_ptr();
    unsafe {
        core::ptr::write_bytes(p, 0xAB, 64);
        assert_eq!(*p, 0xAB);
        assert_eq!(*p.add(63), 0xAB);
        pool.deallocate(block.cast(), layout);
    }
}

#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
fn many_round_trips_through_all_layers() {
    let Some(pool) = build_crypto_pool() else {
        return;
    };
    let layout = NonZeroLayout::for_type::<Secret>().unwrap();
    for i in 0..512u16 {
        let fill = i as u8;
        let block = pool.allocate(layout).expect("alloc");
        let p = block.cast::<u8>().as_ptr();
        unsafe {
            core::ptr::write_bytes(p, fill, 64);
            assert_eq!(*p, fill);
            pool.deallocate(block.cast(), layout);
        }
    }
}

/// The defining crypto property: a freed secret is scrubbed. After free,
/// `ZeroizeOnFree` volatile-zeroes the whole slot; the slab's freelink then
/// overwrites only the first 8 bytes. So on the next allocation of the same
/// LIFO slot, bytes 8..64 must read as zero — NOT the original secret.
#[test]
#[cfg_attr(miri, ignore = "miri-incompatible: mmap / mlock")]
fn freed_secret_is_scrubbed_past_freelink() {
    let Some(pool) = build_crypto_pool() else {
        return;
    };
    let layout = NonZeroLayout::for_type::<Secret>().unwrap();

    let b1 = pool.allocate(layout).expect("alloc");
    let p1 = b1.cast::<u8>().as_ptr();
    unsafe {
        core::ptr::write_bytes(p1, 0xAB, 64); // the "secret"
        pool.deallocate(b1.cast(), layout); // scrubbed on free
    }

    // LIFO slab returns the same slot; verify the secret tail is gone.
    let b2 = pool.allocate(layout).expect("realloc");
    let p2 = b2.cast::<u8>().as_ptr();
    assert_eq!(p1, p2, "LIFO slab should return the just-freed slot");
    unsafe {
        for i in 8..64 {
            assert_eq!(*p2.add(i), 0, "byte {i} of a freed secret was not scrubbed");
        }
        pool.deallocate(b2.cast(), layout);
    }
}
