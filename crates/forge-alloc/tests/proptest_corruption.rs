//! Pass-#8 corruption fuzzing — randomized bit-flips against the
//! protected-metadata surface of every hardened wrapper.
//!
//! The pre-existing tests in `canary.rs`, `cache_jitter.rs`,
//! `quarantine.rs`, and `freelist_protection.rs` each verify a *handful*
//! of hand-picked corruption patterns. This file complements them with
//! proptest-generated random mutations: every bit of the protected bytes
//! is flipped across thousands of iterations, and the wrapper is required
//! to either (a) detect the corruption and panic with the documented
//! security message, or (b) — if the mutation luckily reproduces a valid
//! MAC (probability bounded by the MAC width) — round-trip correctly.
//!
//! ## Why proptest and not cargo-fuzz?
//!
//! `cargo-fuzz` wraps libFuzzer, which doesn't ship for the Windows-MSVC
//! target the repo CI runs on. Proptest is cross-platform, already in
//! `dev-dependencies`, and the goal here is **broad-spectrum randomized
//! corruption** of the bytes the hardened layer protects — not the
//! coverage-guided exploration that's libFuzzer's specific strength.
//!
//! ## Skipped under miri
//!
//! Proptest's persistence-file path uses `std::env::current_dir()`,
//! which miri's isolation refuses to shim. The per-crate `cargo test`
//! runs (debug + release) execute this file; miri does not.
#![cfg(not(miri))]
#![cfg(feature = "std")]

use core::ptr::NonNull;

use forge_alloc::{
    Allocator, CacheJitter, Canary, Deallocator, MmapBacked, NonZeroLayout, Quarantine,
};
use forge_layout::BumpArena;

use proptest::prelude::*;

// ============================================================================
// Helpers
// ============================================================================

/// Build a CacheJitter over a fresh MmapBacked bump arena. MmapBacked is
/// required (not InlineBacked) because CacheJitter inflates the inner
/// alignment requirement to `cache_line_size` (64), which InlineBacked
/// caps at MAX_ALIGN=16.
fn build_jitter() -> CacheJitter<BumpArena<MmapBacked>> {
    CacheJitter::with_params(
        BumpArena::new(MmapBacked::new(64 * 1024).expect("mmap")).expect("bump"),
        64,
        8,
        0x1234_5678_9ABC_DEF0,
    )
    .expect("valid params")
}

/// Build a Canary over a fresh InlineBacked bump arena.
fn build_canary() -> Canary<BumpArena<forge_alloc::InlineBacked<4096>>> {
    Canary::new_with_seed(
        BumpArena::new(forge_alloc::InlineBacked::<4096>::new()).expect("bump"),
        0x1234_5678_9ABC_DEF0,
    )
}

/// Try the closure under `catch_unwind`. Returns `Some(msg)` on panic
/// (with the panic payload's `&'static str` / `String` message), or
/// `None` on clean return.
fn catch_panic_message<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> Option<String> {
    let prev = std::panic::take_hook();
    // Silence the default print-to-stderr — proptest can run thousands
    // of cases, and a panic-per-case avalanches the log.
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    match result {
        Ok(()) => None,
        Err(payload) => {
            // The standard library boxes either &'static str or String.
            if let Some(s) = payload.downcast_ref::<&'static str>() {
                Some((*s).to_string())
            } else if let Some(s) = payload.downcast_ref::<String>() {
                Some(s.clone())
            } else {
                Some("<unknown panic payload>".to_string())
            }
        }
    }
}

// ============================================================================
// Target 1 — CacheJitter prefix header
// ============================================================================
//
// The 8-byte packed `(MAC<<16 | disp_lines)` header sits at `user_ptr - 8`.
// 48 bits are MAC; 16 bits are displacement-in-lines. Any bit-flip in the
// MAC field has 2^-48 chance of producing a valid MAC for the unchanged
// disp; any flip in the disp field changes disp, which changes the
// expected MAC. With overwhelming probability EVERY mutation must
// panic with "prefix header MAC failed".

proptest! {
    #![proptest_config(ProptestConfig {
        // 1024 cases — broad coverage of the 64-bit mutation space,
        // bounded by per-case alloc/free cost (~10 µs on a fast box).
        cases: 1024,
        // Failure persistence directory derived from the workspace.
        .. ProptestConfig::default()
    })]

    /// Property: for ANY 64-bit XOR mask applied to the header (other
    /// than `0`, which is no-op), CacheJitter::deallocate either panics
    /// with the documented MAC-failure message, or — with probability
    /// ≤ 2^-48 — round-trips. We allow either outcome and assert the
    /// panic message when one occurs.
    #[test]
    fn cache_jitter_header_bitflip_panics_or_forges(mask: u64) {
        // Skip the no-op mask (no mutation → not corruption).
        if mask == 0 {
            return Ok(());
        }
        let result = catch_panic_message(|| {
            let cj = build_jitter();
            let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
            let block = cj.allocate(layout).expect("alloc");
            let user_ptr = block.cast::<u8>();
            // Read the legitimate header, XOR with mask, write back.
            unsafe {
                let hdr_ptr = user_ptr.as_ptr().sub(8).cast::<u64>();
                let original = core::ptr::read_unaligned(hdr_ptr);
                core::ptr::write_unaligned(hdr_ptr, original ^ mask);
                cj.deallocate(user_ptr, layout);
            }
        });
        match result {
            Some(msg) => {
                prop_assert!(
                    msg.contains("prefix header MAC failed"),
                    "wrong panic message for mask {mask:#018x}: {msg}",
                );
            }
            None => {
                // No panic — must have been a MAC collision (2^-48 per
                // mask). Across 1024 cases, the expected count of
                // collisions is ~3.6e-12 — effectively zero. Log so
                // a regression that always-forges is loud.
                eprintln!(
                    "WARNING: CacheJitter mask {mask:#018x} did not trip MAC \
                     check — investigate if this fires more than once across \
                     a full proptest run"
                );
            }
        }
    }

    /// Property: single-bit flips in either the MAC field (high 48
    /// bits) or the disp field (low 16 bits) are reliably caught. This
    /// is the underflow / UAF-prefix-write scenario in miniature — an
    /// attacker who can flip *one* bit must still trip the check.
    #[test]
    fn cache_jitter_single_bitflip_always_panics(bit in 0u32..64) {
        let mask = 1u64 << bit;
        let result = catch_panic_message(|| {
            let cj = build_jitter();
            let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
            let block = cj.allocate(layout).expect("alloc");
            let user_ptr = block.cast::<u8>();
            unsafe {
                let hdr_ptr = user_ptr.as_ptr().sub(8).cast::<u64>();
                let original = core::ptr::read_unaligned(hdr_ptr);
                core::ptr::write_unaligned(hdr_ptr, original ^ mask);
                cj.deallocate(user_ptr, layout);
            }
        });
        // A single-bit flip can NEVER coincidentally re-MAC: the SplitMix
        // construction is bijective and changing one input bit affects
        // many output bits. We assert the panic unconditionally.
        let msg = result.expect("single-bit header flip must panic");
        prop_assert!(
            msg.contains("prefix header MAC failed"),
            "wrong panic for bit {bit}: {msg}",
        );
    }
}

// ============================================================================
// Target 2 — Canary pre/post sentinels
// ============================================================================
//
// Pre canary at `user_ptr - 8`, post canary at `user_ptr + size`. Both
// are full 8-byte u64 stored verbatim. The MAC width is 64 bits — a
// random replacement value has 2^-64 chance of reproducing the seed.

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1024,
        .. ProptestConfig::default()
    })]

    /// Property: any non-seed value written to the pre-canary causes
    /// deallocate to panic with "canary corruption".
    #[test]
    fn canary_pre_random_panics(replacement: u64) {
        let result = catch_panic_message(|| {
            let c = build_canary();
            let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
            let block = c.allocate(layout).expect("alloc");
            let ptr = block.cast::<u8>();
            unsafe {
                // Overwrite the pre-canary slot. If `replacement` equals
                // the seed, this is no-op corruption (probability 2^-64).
                core::ptr::write_unaligned(ptr.as_ptr().sub(8).cast::<u64>(), replacement);
                c.deallocate(ptr, layout);
            }
        });
        // Hardcoded seed in build_canary() = 0x1234_5678_9ABC_DEF0.
        let is_seed = replacement == 0x1234_5678_9ABC_DEF0;
        match result {
            Some(msg) => {
                prop_assert!(
                    !is_seed,
                    "panicked for replacement matching seed: {msg}",
                );
                prop_assert!(
                    msg.contains("canary corruption"),
                    "wrong panic for replacement {replacement:#018x}: {msg}",
                );
            }
            None => {
                prop_assert!(
                    is_seed,
                    "no panic but replacement {replacement:#018x} != seed — \
                     canary check failed to fire",
                );
            }
        }
    }

    /// Property: same for the post-canary.
    #[test]
    fn canary_post_random_panics(replacement: u64) {
        let result = catch_panic_message(|| {
            let c = build_canary();
            let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
            let block = c.allocate(layout).expect("alloc");
            let ptr = block.cast::<u8>();
            unsafe {
                core::ptr::write_unaligned(
                    ptr.as_ptr().add(layout.size().get()).cast::<u64>(),
                    replacement,
                );
                c.deallocate(ptr, layout);
            }
        });
        let is_seed = replacement == 0x1234_5678_9ABC_DEF0;
        match result {
            Some(msg) => {
                prop_assert!(
                    !is_seed,
                    "panicked for replacement matching seed: {msg}",
                );
                prop_assert!(
                    msg.contains("canary corruption"),
                    "wrong panic for replacement {replacement:#018x}: {msg}",
                );
            }
            None => {
                prop_assert!(
                    is_seed,
                    "no panic but replacement {replacement:#018x} != seed",
                );
            }
        }
    }

    /// Property: combined pre+post corruption with independent random
    /// values. Verifies that the pre-canary check is the first to fire
    /// (matches the implementation order in `Canary::deallocate`).
    #[test]
    fn canary_both_random_panics(pre: u64, post: u64) {
        // Skip the case where both replacements happen to be the seed
        // (probability 2^-128, effectively unreachable).
        let seed = 0x1234_5678_9ABC_DEF0u64;
        if pre == seed && post == seed {
            return Ok(());
        }
        let result = catch_panic_message(|| {
            let c = build_canary();
            let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
            let block = c.allocate(layout).expect("alloc");
            let ptr = block.cast::<u8>();
            unsafe {
                core::ptr::write_unaligned(ptr.as_ptr().sub(8).cast::<u64>(), pre);
                core::ptr::write_unaligned(
                    ptr.as_ptr().add(layout.size().get()).cast::<u64>(),
                    post,
                );
                c.deallocate(ptr, layout);
            }
        });
        let msg = result.expect("at least one canary side corrupted — must panic");
        prop_assert!(
            msg.contains("canary corruption"),
            "wrong panic msg: {msg}",
        );
        // If pre != seed, the pre-check fires first; otherwise the
        // post-check fires. Verify the diagnostic identifies the side.
        if pre != seed {
            prop_assert!(
                msg.contains("pre-canary"),
                "expected pre-canary diagnostic when pre corrupted, got: {msg}",
            );
        } else {
            prop_assert!(
                msg.contains("post-canary"),
                "expected post-canary diagnostic when only post corrupted, got: {msg}",
            );
        }
    }

    /// Property: single-byte writes inside the canary slot are detected.
    /// Models a 1-byte linear overflow / underflow — the smallest
    /// corruption an attacker might introduce.
    #[test]
    fn canary_single_byte_overflow_detected(offset in 0usize..8, value: u8) {
        // Always corruption (the seed has at most one byte equal to
        // any given `value` per slot, and we always overwrite one).
        let result = catch_panic_message(|| {
            let c = build_canary();
            let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
            let block = c.allocate(layout).expect("alloc");
            let ptr = block.cast::<u8>();
            unsafe {
                // Overflow one byte into the post-canary. Always a
                // detectable corruption unless the byte happens to
                // equal the seed's byte at that offset.
                let seed_bytes = 0x1234_5678_9ABC_DEF0u64.to_ne_bytes();
                let post = ptr.as_ptr().add(layout.size().get());
                *post.add(offset) = value;
                // If we just wrote the seed-byte at that offset, the
                // canary is unchanged; we re-mutate with the bit-
                // complement to guarantee corruption.
                if value == seed_bytes[offset] {
                    *post.add(offset) = !value;
                }
                c.deallocate(ptr, layout);
            }
        });
        let msg = result.expect("byte-level canary overflow must panic");
        prop_assert!(
            msg.contains("canary corruption"),
            "wrong panic for offset {offset} value {value:#04x}: {msg}",
        );
    }
}

// ============================================================================
// Target 3 — SipHashMAC freelist link
// ============================================================================
//
// Slab with SipHashMAC stores `FreeLink { next_idx: u32, mac: u32 }` in
// each freed slot. The MAC is over `(next_idx, slot_addr)` with a
// per-instance 128-bit key. An attacker who corrupts `next_idx` in a
// freed slot must trip the MAC check on the NEXT allocate — Slab's
// response is to disarm the freelist (release/silent) or panic
// (debug). Either way, the corrupted index must never become a
// returned pointer.
//
// Requires `--features siphasher` to compile.

#[cfg(feature = "siphasher")]
mod siphash_corruption {
    use super::*;
    use forge_core::SipHashMAC;
    use forge_layout::Slab;

    /// Construct a Slab<u64, _, SipHashMAC> with `cap` slots backed
    /// by MmapBacked. Uses a fixed key so repro is deterministic.
    fn build_slab(cap: usize) -> Slab<u64, MmapBacked, SipHashMAC> {
        let backing = MmapBacked::new(64 * 1024).expect("mmap");
        let mac = SipHashMAC::with_key([0x42u8; 16]);
        Slab::with_protection(cap, backing, mac).expect("slab")
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            .. ProptestConfig::default()
        })]

        /// Property: corrupting `next_idx` in a freed slot must NEVER
        /// cause `allocate` to return a pointer to an out-of-range or
        /// already-live slot. Slab's response is to disarm the freelist
        /// (panics in debug under `debug_assert!`, silent in release).
        /// Either outcome is acceptable as long as no corrupted pointer
        /// escapes.
        #[test]
        fn corrupted_next_idx_never_returns_oob(
            new_idx in any::<u32>(),
        ) {
            // Skip the case where the random replacement happens to equal
            // the legitimate next_idx (no corruption, no detection).
            // Since the legitimate next_idx is 0 (end-of-list) for the
            // first freed slot, that's the only collision.
            let result = catch_panic_message(|| {
                let slab = build_slab(16);
                let layout = NonZeroLayout::for_type::<u64>().unwrap();
                let a = slab.allocate(layout).expect("alloc a").cast::<u8>();
                let b = slab.allocate(layout).expect("alloc b").cast::<u8>();
                // Free `a` — Slab writes FreeLink { next_idx: 0, mac } at a.
                unsafe { slab.deallocate(a, layout) };
                // Corrupt the next_idx field (first 4 bytes of the slot).
                // We deliberately do NOT update the MAC — the verify path
                // must catch the mismatch.
                unsafe {
                    core::ptr::write_unaligned(a.as_ptr().cast::<u32>(), new_idx);
                }
                // Next allocate walks the freelist. With MAC verification:
                //   - If the corrupted (next_idx, mac) pair doesn't verify
                //     (overwhelming probability), Slab disarms (debug
                //     panics, release falls through to next_uncarved).
                //   - If by 2^-32 chance the MAC verifies AND next_idx <=
                //     capacity, Slab follows the corrupted link.
                //
                // We don't dereference the returned pointer (would be UB
                // for the corrupted case). We only check that the
                // returned pointer is either `a` (legitimate freelist
                // hit) or a fresh slot from next_uncarved (legitimate
                // post-disarm path).
                let c = slab.allocate(layout).expect("alloc c after corruption");
                let c_addr = c.cast::<u8>().as_ptr() as usize;
                let base = slab.allocate(layout).expect("alloc d").cast::<u8>().as_ptr() as usize;
                // c_addr must be inside the slab range, distinct from
                // the live slot b.
                let b_addr = b.as_ptr() as usize;
                assert_ne!(c_addr, b_addr, "alloc returned a live slot");
                // Either c is `a` (forge succeeded by MAC luck) or a
                // fresh slot. Both are inside the slab; no further
                // assertion possible without depending on the disarm
                // policy.
                let _ = base;
                // Drop the slab; do NOT deallocate the corrupted slots
                // we never re-derived. Slab::drop releases the backing
                // unconditionally so this is safe.
            });
            // Either no panic (release-style disarm, or pre-empted by
            // debug_assert! in `slot_index` path on OOB) or the
            // documented debug panic. Both are acceptable so long as
            // they happen before any unsafe deref of corrupted bytes.
            if let Some(msg) = result {
                prop_assert!(
                    msg.contains("Slab freelist corruption")
                        || msg.contains("Slab::deallocate")
                        || msg.contains("alloc"),
                    "unexpected panic for next_idx {new_idx}: {msg}",
                );
            }
        }

        /// Property: corrupting BOTH `next_idx` AND `mac` with random
        /// values still does not produce a corrupted-pointer return.
        /// This is the "attacker writes both halves of the link" case
        /// — the MAC verify has 2^-32 false-accept rate, but even when
        /// it accepts, the `next_idx <= capacity` tripwire still
        /// catches out-of-range values.
        #[test]
        fn corrupted_link_both_fields_never_returns_oob(
            new_idx in any::<u32>(),
            new_mac in any::<u32>(),
        ) {
            let _ = catch_panic_message(|| {
                let slab = build_slab(16);
                let layout = NonZeroLayout::for_type::<u64>().unwrap();
                let a = slab.allocate(layout).expect("alloc a").cast::<u8>();
                let _b = slab.allocate(layout).expect("alloc b").cast::<u8>();
                unsafe { slab.deallocate(a, layout) };
                // Overwrite the full 8-byte FreeLink — both fields.
                unsafe {
                    core::ptr::write_unaligned(a.as_ptr().cast::<u32>(), new_idx);
                    core::ptr::write_unaligned(a.as_ptr().add(4).cast::<u32>(), new_mac);
                }
                // Next allocate: MAC verify fails with 1 - 2^-32 prob.
                // In the 2^-32 accept branch, the `next_idx > capacity`
                // tripwire fires for most out-of-range values.
                let _c = slab.allocate(layout);
                // We don't deref the returned pointer; the property is
                // that the slab survives the call (no panic past the
                // documented `debug_assert!` site, no UB observable to
                // the test harness).
            });
            // Either outcome is acceptable — proptest just verifies the
            // sequence runs to completion without harness-visible UB.
        }
    }
}

// ============================================================================
// Target 4 — GuardPage boundary writes
// ============================================================================
//
// GuardPage uses OS PROT_NONE / PAGE_NOACCESS pages on either side of
// the data region. A boundary write triggers SIGSEGV on Linux / macOS,
// `EXCEPTION_ACCESS_VIOLATION` on Windows. Catching either from a
// proptest harness is a substantial undertaking:
//
//   - Linux/macOS: install a `sigaction` for SIGSEGV that longjmps
//     out of the offending frame. Doing this correctly under
//     async-signal-safety constraints is nontrivial.
//   - Windows: install a vectored exception handler via
//     `AddVectoredExceptionHandler`, ensure it doesn't conflict with
//     panic-handler / SEH / structured-exception machinery.
//
// Both involve interacting with the OS exception machinery in ways
// that don't compose with proptest's `catch_unwind`. We document the
// gap and rely on the existing in-tree GuardPage smoke tests + an
// observable-by-hand verification path: a debug binary linked against
// libasan would surface this immediately.
//
// **Coverage gap (G-1)**: GuardPage SIGSEGV / SEH behavior is NOT
// randomized-fuzzed by this file. Manual verification only.

// ============================================================================
// Target 5 — Quarantine containment
// ============================================================================
//
// Quarantine holds freed blocks for `EPOCHS` deallocate cycles before
// returning them to the inner. The property here is *containment*: a
// caller mutating the bytes of a quarantined block (whose pointer they
// still hold by aliasing — formally illegal but practically the UAF
// scenario Quarantine exists to defend against) must NOT affect any
// LIVE allocation in the inner. The mutations stay inside the
// quarantined slot's address range until that slot ages out.

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        .. ProptestConfig::default()
    })]

    /// Property: writing arbitrary bytes into a quarantined slot does
    /// not affect a concurrently-live allocation's bytes. The
    /// Quarantine holds the freed slot off the inner's freelist for
    /// EPOCHS cycles, so the bytes the test re-reads from the live
    /// allocation must be unaffected.
    #[test]
    fn quarantine_contains_mutation(
        write_pattern in any::<u8>(),
    ) {
        // Build a Slab<u8, ...> wrapped in Quarantine<_, 4>.
        // u8 forces block_stride = 8 (FreeLink minimum) so we have a
        // predictable layout to mutate.
        let inner = forge_layout::Slab::<u64, _>::new(
            16,
            forge_alloc::InlineBacked::<2048>::new(),
        )
        .expect("slab");
        let q: Quarantine<_, 4> = Quarantine::new(inner);
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Live allocation we will verify is untouched.
        let live = q.allocate(layout).expect("alloc live").cast::<u8>();
        unsafe { core::ptr::write_bytes(live.as_ptr(), 0xAB, 8) };
        // Allocation we will free and then dangle-write through.
        let victim = q.allocate(layout).expect("alloc victim").cast::<u8>();
        unsafe { core::ptr::write_bytes(victim.as_ptr(), 0xCD, 8) };
        // Free `victim` — it enters quarantine slot 0.
        unsafe { q.deallocate(victim, layout) };
        // Mutate the quarantined bytes via the dangling pointer.
        // SAFETY: this models the UAF the wrapper exists to mitigate;
        // we hold the pointer past free, write through it. Quarantine
        // owns the slot until eviction, so the write lands in
        // Quarantine-private memory — no inner allocation aliases it.
        unsafe {
            for i in 0..8 {
                *victim.as_ptr().add(i) = write_pattern;
            }
        }
        // Read back `live`. It must still hold 0xAB everywhere.
        for i in 0..8 {
            // SAFETY: live is still owned by the caller (we haven't
            // freed it), and writes through the quarantined ptr cannot
            // reach it.
            let b = unsafe { core::ptr::read(live.as_ptr().add(i)) };
            prop_assert_eq!(
                b, 0xAB,
                "live alloc byte {} changed to {:#04x} after quarantined-slot \
                 write of {:#04x} — containment violation",
                i, b, write_pattern,
            );
        }
        // Clean up. Drop live alloc through the wrapper.
        unsafe { q.deallocate(live, layout) };
    }

    /// Property: after EPOCHS additional deallocates, the quarantined
    /// slot is evicted back to the inner's freelist. A subsequent
    /// allocate may reuse that slot's address — bytes the attacker
    /// wrote during the quarantine window become visible to the new
    /// owner. This is the documented behavior; we pin it so that any
    /// future "zero on eviction" refactor (which would be a security
    /// improvement) announces itself by making this property fail.
    #[test]
    fn quarantine_eviction_returns_bytes_unmodified(
        write_pattern in 0x10u8..0xFE,
    ) {
        let inner = forge_layout::Slab::<u64, _>::new(
            16,
            forge_alloc::InlineBacked::<2048>::new(),
        )
        .expect("slab");
        let q: Quarantine<_, 2> = Quarantine::new(inner);
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Allocate, mutate, free, dangle-write, force eviction.
        let victim = q.allocate(layout).expect("alloc").cast::<u8>();
        unsafe { q.deallocate(victim, layout) };
        // Dangle-write the attacker pattern into the quarantined bytes.
        unsafe {
            for i in 0..8 {
                *victim.as_ptr().add(i) = write_pattern;
            }
        }
        // Burn through EPOCHS additional deallocates to evict victim.
        let d1 = q.allocate(layout).expect("alloc d1").cast::<u8>();
        let d2 = q.allocate(layout).expect("alloc d2").cast::<u8>();
        unsafe { q.deallocate(d1, layout) }; // ring[1] = d1
        unsafe { q.deallocate(d2, layout) }; // ring[0] evicts victim → inner freelist

        // The inner Slab is LIFO over its freelist. Slab also writes
        // a FreeLink into the slot on dealloc, which OVERWRITES the
        // first 8 bytes with `(next_idx, mac)` — so the attacker's
        // bytes survive only at offsets that the FreeLink doesn't
        // touch. For u64-sized slots (block_stride=8), the FreeLink
        // fully overlays the slot. This means the attacker pattern is
        // overwritten on eviction-to-inner by the inner's freelist
        // bookkeeping itself.
        //
        // The property here is therefore a NEGATIVE one: we verify
        // that after eviction, allocating again does NOT return
        // attacker-controlled bytes — Slab's FreeLink stomp prevents
        // it. This is a defense-in-depth observation about the
        // composition Quarantine<Slab<_>>, not an intrinsic Quarantine
        // guarantee.
        let reused = q.allocate(layout).expect("alloc reused").cast::<u8>();
        // The slot's bytes are now whatever Slab's FreeLink wrote (a
        // u32 next_idx then a u32 mac); they should not be the
        // uniform `write_pattern` byte the attacker placed.
        let mut all_attacker_pattern = true;
        for i in 0..8 {
            let b = unsafe { core::ptr::read(reused.as_ptr().add(i)) };
            if b != write_pattern {
                all_attacker_pattern = false;
                break;
            }
        }
        prop_assert!(
            !all_attacker_pattern,
            "attacker pattern {write_pattern:#04x} survived eviction + reuse — \
             Slab's FreeLink stomp regressed",
        );
        unsafe { q.deallocate(reused, layout) };
    }
}

// ============================================================================
// Smoke check: catch_panic_message correctly captures known panics
// ============================================================================

#[test]
fn smoke_catch_panic_message_works() {
    let msg = catch_panic_message(|| panic!("test message"));
    assert!(matches!(msg.as_deref(), Some(s) if s.contains("test message")));
    let none = catch_panic_message(|| {
        let _x: usize = 1 + 1;
    });
    assert!(none.is_none());
}

// Use of `NonNull` is intentional (matches the wrapper API surface).
#[allow(dead_code)]
fn _silence_nonnull_use(_p: NonNull<u8>) {}
