//! `FreelistProtection` — pluggable integrity policy for `Slab` free lists.
//!
//! With `M = NoProtection` (the default), sign returns `0` and verify always
//! succeeds — the optimizer eliminates the calls entirely. With
//! [`SipHashMAC`] or [`PacMAC`], each free-list link's `(next_idx, slot_addr)`
//! pair is signed, so corruption of an in-band freelist pointer is detected
//! before the corrupted slot is dereferenced.
//!
//! See `docs/ARCHITECTURE.md` for the freelist layout and the push/pop algorithm.

use core::fmt;

/// Pluggable integrity policy for slab freelists.
///
/// `sign` produces a 32-bit MAC over the `(next_idx, slot_addr)` pair; `verify`
/// recomputes and compares. Implementations must be deterministic with respect
/// to their internal key — calling `sign` twice with the same inputs must
/// yield the same MAC.
pub trait FreelistProtection {
    /// Sign a freelist link. `next_idx` is the 1-based slot index being
    /// stored, or `0` for the end-of-list sentinel (so the input range
    /// is `0..=u32::MAX`). `slot_addr` is the virtual address of the slot
    /// containing the link (used as a nonce so that a copy of a freelist
    /// link to a different slot won't verify).
    fn sign(&self, next_idx: u32, slot_addr: usize) -> u32;

    /// Verify a stored MAC. Returns `Ok(())` on a valid link,
    /// `Err(FreelistCorruption)` on a mismatch.
    fn verify(
        &self,
        next_idx: u32,
        stored_mac: u32,
        slot_addr: usize,
    ) -> Result<(), FreelistCorruption>;
}

/// Returned by [`FreelistProtection::verify`] when the stored MAC does not
/// match the expected value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FreelistCorruption;

impl fmt::Display for FreelistCorruption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("freelist link MAC mismatch — heap corruption or use-after-free")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FreelistCorruption {}

/// Zero-overhead default. `sign` returns `0`; `verify` always succeeds. The
/// optimizer reliably folds calls through `NoProtection` into no-ops in
/// release builds.
#[derive(Copy, Clone, Debug, Default)]
pub struct NoProtection;

impl FreelistProtection for NoProtection {
    #[inline(always)]
    fn sign(&self, _next_idx: u32, _slot_addr: usize) -> u32 {
        0
    }

    #[inline(always)]
    fn verify(
        &self,
        _next_idx: u32,
        _stored_mac: u32,
        _slot_addr: usize,
    ) -> Result<(), FreelistCorruption> {
        Ok(())
    }
}

/// SipHash-1-3 keyed MAC over the `(next_idx, slot_addr)` pair.
///
/// The 16-byte key is initialized once at `Slab` construction. On `std`
/// builds use [`SipHashMAC::new`] to seed; in `no_std` contexts use
/// [`SipHashMAC::with_key`] and supply entropy from a hardware RNG or
/// boot-time source.
///
/// # Threat model and limitations
///
/// **MAC width — 32 bits.** The MAC type is `u32` (matches the on-disk
/// `FreeLink` layout in `Slab`'s freelist — `FreeLink` is private to
/// the slab module). 32-bit MACs give a blind-forgery probability of
/// `2^-32` per guess. Widening to 64 bits requires changing
/// `FreelistProtection::sign`/`verify` to `u64` and growing `FreeLink`
/// by 4 bytes per slot (4–25% size overhead depending on `T`). For
/// applications that need stronger guarantees, use `PacMAC` on aarch64
/// (hardware-backed pointer-authentication signature) once its
/// instruction-level implementation is complete (currently a stub).
///
/// **MAC-failure behavior in `Slab`.** Today, a MAC mismatch (or a
/// `next_idx > capacity` defense-in-depth tripwire) causes `Slab` to
/// **abandon the in-band freelist** (`*head_ptr = 0`) and fall through
/// to a fresh allocation from `next_uncarved`. The attacker-controlled
/// link chain is torn down — they cannot navigate the slab to a chosen
/// address — but the event is *not* propagated to the caller as an
/// error, and `Quarantine` (when composed) does *not* see it: the
/// allocate call returns `Ok` with a slot carved from the uncorrupted
/// uncarved region. Debug builds additionally panic via
/// `debug_assert!` so the regression surfaces in tests. For loud-fail-
/// on-corruption semantics in production, the slot the attacker
/// poisoned is leaked until slab drop — no slot is handed back to the
/// caller from the corrupted chain. MAC-failure events are counted by
/// the allocator and exposed via [`crate::Allocator::corruption_events`];
/// operators can monitor that counter to detect silent disarms at scale.
///
/// **Truncation in `SipHashMAC::mac` (private)**: we take the low 32 bits of the
/// SipHash-1-3 output. This is uniform random across forgeries provided
/// the SipHash construction is sound (it is), so collision probability
/// is `2^-32` per blind guess regardless of which 32 bits we keep.
///
/// **Entropy source for [`SipHashMAC::new`]**: `std`'s
/// `HashMap::RandomState`. This goes through the same OS RNG path that
/// hardens stdlib's `HashMap` against algorithmic-complexity DoS — on
/// Linux/macOS/Windows it draws from `getrandom`/`arc4random`/`RtlGenRandom`
/// respectively. For deployments that mandate `getrandom` directly
/// (FIPS-flow audit trail, no-`HashMap`-dep environments, or pre-stdlib
/// init paths), supply the key via [`SipHashMAC::with_key`] and source
/// the bytes from your project's existing CSPRNG. The dep is intentionally
/// not added to this crate to keep the no_std footprint minimal.
#[cfg(feature = "siphasher")]
pub struct SipHashMAC {
    key: [u8; 16],
}

/// Zeroize the key on drop so cloned instances don't leave the secret
/// in deallocated stack frames after a typed slab tears down.
///
/// Note: `Copy` is **not** derived (a typed slab takes the MAC by value
/// at construction; explicit `.clone()` is required to install the same
/// key in two slabs — that's the intended security posture, and it lets
/// us implement `Drop` to zeroize).
///
/// # Caveats — when Drop does not run
///
/// Drop-based zeroize protects only against keys that **leave scope**.
/// It does *not* fire for:
///
/// - **`static SipHashMAC`** / **`OnceCell<SipHashMAC>`** — process-
///   lifetime statics never drop; the key persists until the process
///   exits, observable via core dumps or `/proc/<pid>/mem` reads.
///   Construct on the stack inside the function that uses it (e.g. a
///   per-request slab) where the natural scope-exit fires Drop.
/// - **`panic = "abort"`** builds (and the `Quarantine`
///   abort-on-corruption path) — Drop is skipped on abort; if a
///   `SipHashMAC`-using slab is live at the time, its key remains in
///   freed-but-unmapped memory until the OS reclaims the address
///   space. The OS-level free isn't observable by other processes,
///   but a same-process attacker who triggered the abort can still
///   peek at the bytes if the abort handler reads memory before
///   teardown.
/// - **`mem::forget(slab)`** or `Box::leak` on a slab containing the
///   MAC — explicit destructor suppression. Don't combine these with
///   security-critical wrappers.
#[cfg(feature = "siphasher")]
impl Drop for SipHashMAC {
    fn drop(&mut self) {
        // Volatile writes prevent the compiler from optimizing away the
        // clear once `self` is no longer used downstream.
        for b in &mut self.key {
            // SAFETY: `b` is a valid mutable reference into our own field.
            unsafe { core::ptr::write_volatile(b, 0) };
        }
        // Compiler fence to keep the zeroization from being reordered past
        // subsequent code (which would defeat the volatile guarantee).
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Manual `Clone` with per-byte volatile copy — the derived `Clone`
/// lowered to a memcpy that the optimizer was free to coalesce, fuse
/// with surrounding writes, or vectorize through registers, leaving
/// transient byte-aligned copies of the key in caller stack frames
/// outside the original `key` slot (which `Drop` then could not
/// zeroize). The manual loop forces each byte through a volatile
/// load/store pair, defeating that smear.
///
/// The `compiler_fence` at the end prevents the optimizer from
/// reordering subsequent code into the middle of the copy and from
/// fusing the writes with later non-volatile stores.
#[cfg(feature = "siphasher")]
impl Clone for SipHashMAC {
    fn clone(&self) -> Self {
        let mut out = Self { key: [0u8; 16] };
        // Volatile per-byte copy direct from source field to destination
        // field — no intermediate stack-local `[u8; 16]` that the
        // optimizer could vectorize or spill into transient registers
        // visible after `Drop`.
        let src = self.key.as_ptr();
        let dst = out.key.as_mut_ptr();
        for i in 0..16 {
            // SAFETY: `src` and `dst` each point to the start of a
            // 16-byte array we own; `i` is in-bounds for both. The
            // pointers do not alias (distinct allocations: `self` and
            // the freshly-constructed `out`).
            unsafe {
                let b = core::ptr::read_volatile(src.add(i));
                core::ptr::write_volatile(dst.add(i), b);
            }
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
        out
    }
}

/// Wrapper around `SipHasher13` that volatile-zeros its byte
/// representation on Drop.
///
/// `SipHashMAC::mac` constructs a fresh `SipHasher13` from the stored
/// key, mixes input via `write_u32`/`write_usize`, and finalizes. The
/// hasher's internal state (`v0..v3`, `m`, `tail`, `ntail`) is derived
/// from the key on construction and remains key-equivalent for the
/// duration of the call. Without zeroing, that state stays on the
/// stack frame after `mac()` returns and may be observed by subsequent
/// frames reusing the same memory.
///
/// `SipHasher13` is a `#[derive(Debug, Clone, Default)]` struct of
/// plain `u64`/`u8` fields in the upstream siphasher crate; the
/// all-zero bit pattern is valid for every field, so writing zero
/// bytes is sound.
#[cfg(feature = "siphasher")]
struct ZeroingHasher(siphasher::sip::SipHasher13);

#[cfg(feature = "siphasher")]
impl Drop for ZeroingHasher {
    fn drop(&mut self) {
        // SAFETY: `SipHasher13` is composed of integer fields whose
        // all-zero bit patterns are valid; the pointer is into our
        // owned field. Volatile prevents the optimizer from dropping
        // the writes once the value is going out of scope.
        let p = (&mut self.0) as *mut siphasher::sip::SipHasher13 as *mut u8;
        let n = core::mem::size_of::<siphasher::sip::SipHasher13>();
        for i in 0..n {
            unsafe { core::ptr::write_volatile(p.add(i), 0) };
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(feature = "siphasher")]
impl SipHashMAC {
    /// Construct with an explicit 16-byte key. Suitable for `no_std`.
    #[inline]
    pub const fn with_key(key: [u8; 16]) -> Self {
        Self { key }
    }

    /// Seed from the OS entropy source.
    #[cfg(feature = "std")]
    pub fn new() -> Self {
        // We use the same entropy strategy std::collections::HashMap uses: a
        // process-scoped random key. We don't depend on `getrandom` directly
        // to keep the dependency set lean; HashMap's RandomState gives us
        // 16 bytes of OS entropy through a stable, audited path.
        use std::collections::hash_map::RandomState;
        use std::hash::BuildHasher;
        // Mix a unique value into each independently-seeded hasher so the two
        // halves of the key come from independent random state.
        let a = RandomState::new().hash_one(0u64).to_le_bytes();
        let b = RandomState::new().hash_one(1u64).to_le_bytes();
        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&a);
        key[8..].copy_from_slice(&b);
        Self::with_key(key)
    }

    #[inline]
    fn mac(&self, next_idx: u32, slot_addr: usize) -> u32 {
        use core::hash::Hasher;
        // Wrap the hasher so its key-derived internal state (v0..v3) is
        // volatile-zeroed when the local goes out of scope — otherwise
        // the bits live on for the lifetime of the stack frame.
        let mut hasher = ZeroingHasher(siphasher::sip::SipHasher13::new_with_key(&self.key));
        hasher.0.write_u32(next_idx);
        // Pin to `u64` rather than `usize` so the on-wire MAC input is
        // identical across 32-bit and 64-bit targets. Without this, a
        // 32-bit ARM build would feed 4 bytes of slot_addr to SipHash
        // and a 64-bit ARM build would feed 8, producing different MACs
        // for the same slot — breaking any cross-host slab persistence
        // or analysis tooling. The truncation to `u32` below is a
        // documented endianness-independent operation on the SipHash
        // u64 output, so the whole pipeline is deterministic across
        // targets.
        hasher.0.write_u64(slot_addr as u64);
        // Truncate to 32 bits — collision resistance is acceptable for
        // freelist corruption detection (the attacker must forge a specific
        // MAC for a specific slot, not find any collision).
        hasher.0.finish() as u32
    }
}

#[cfg(feature = "siphasher")]
impl FreelistProtection for SipHashMAC {
    #[inline]
    fn sign(&self, next_idx: u32, slot_addr: usize) -> u32 {
        self.mac(next_idx, slot_addr)
    }

    #[inline]
    fn verify(
        &self,
        next_idx: u32,
        stored_mac: u32,
        slot_addr: usize,
    ) -> Result<(), FreelistCorruption> {
        // Constant-time compare via `subtle`. For 32-bit scalars on modern
        // CPUs the underlying CMP is already one-cycle constant-time, so
        // the practical win is small — but the `ct_eq` wrapper documents
        // intent, prevents a future refactor from regressing into a
        // memcmp-style early-exit, and survives compiler reordering that
        // could otherwise observe the comparison's intermediate state.
        // The branch on the result is fundamental ("did the MAC match?")
        // and leaks no additional information beyond the Ok/Err return.
        use subtle::ConstantTimeEq;
        if bool::from(self.mac(next_idx, slot_addr).ct_eq(&stored_mac)) {
            Ok(())
        } else {
            Err(FreelistCorruption)
        }
    }
}

#[cfg(all(feature = "siphasher", feature = "std"))]
impl Default for SipHashMAC {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "siphasher")]
impl fmt::Debug for SipHashMAC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deliberately do not expose the key.
        f.debug_struct("SipHashMAC").finish_non_exhaustive()
    }
}

/// ARM Pointer Authentication keyed MAC. Uses the `PACIB` / `AUTIB`
/// instructions to sign and verify the `next_idx` value with `slot_addr`
/// as modifier.
///
/// **STUB — UNIMPLEMENTED.** The instruction-level body is not yet implemented
/// alongside the MTE / MPK hardware-protection track. Constructing a
/// `Slab<T, _, PacMAC>` *compiles* today (the trait is satisfied) but
/// the first `sign` / `verify` call **panics** with an explicit
/// "not yet implemented" message (see the impl below — `panic!` is
/// chosen over `unimplemented!` to read as a safety-trigger rather
/// than a maintainer TODO), so accidentally selecting `PacMAC` for
/// production today crashes the first time the slab allocates or
/// frees.
///
/// To make accidental use a build error rather than a runtime panic,
/// `PacMAC` is gated behind the `pac-stub` feature flag (separate from
/// the eventual `pac` feature that will activate the real
/// implementation). Enable `pac-stub` only in code that explicitly
/// wants to type-check against the future API:
///
/// ```toml
/// [dependencies]
/// forge-alloc-core = { version = "...", features = ["pac-stub"] }
/// ```
#[cfg(all(target_arch = "aarch64", feature = "pac-stub"))]
#[doc(hidden)]
#[deprecated(
    since = "0.1.0",
    note = "PacMAC is a stub: sign/verify panic at runtime. The PACIB/AUTIB \
            instruction-level implementation is not yet implemented. Use SipHashMAC or \
            NoProtection until then."
)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PacMAC;

#[cfg(all(target_arch = "aarch64", feature = "pac-stub"))]
#[allow(deprecated)] // `PacMAC` itself is deprecated — the impl block doesn't escalate.
impl FreelistProtection for PacMAC {
    fn sign(&self, _next_idx: u32, _slot_addr: usize) -> u32 {
        // Use `panic!` rather than `unimplemented!` so the message reads as
        // an explicit safety-trigger rather than a maintainer TODO. The
        // `pac-stub` feature gate + `#[deprecated]` make compile-time
        // detection the primary defense; this is the last line.
        panic!(
            "PacMAC::sign called: this is a STUB — not yet implemented \
             (ARM PACIB/AUTIB instruction-level impl). \
             Disable the `pac-stub` feature in production builds."
        );
    }

    fn verify(
        &self,
        _next_idx: u32,
        _stored_mac: u32,
        _slot_addr: usize,
    ) -> Result<(), FreelistCorruption> {
        panic!(
            "PacMAC::verify called: this is a STUB — not yet implemented \
             (ARM PACIB/AUTIB instruction-level impl). \
             Disable the `pac-stub` feature in production builds."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_protection_round_trips() {
        let m = NoProtection;
        let mac = m.sign(7, 0xdead_beef);
        assert_eq!(mac, 0);
        assert!(m.verify(7, mac, 0xdead_beef).is_ok());
    }

    #[test]
    fn no_protection_accepts_anything() {
        // The whole point of NoProtection: zero overhead, never errors.
        // verify() ignores the stored_mac entirely.
        let m = NoProtection;
        assert!(m.verify(0, 0xffff_ffff, 0).is_ok());
    }

    #[cfg(feature = "siphasher")]
    #[test]
    fn siphash_mac_round_trips() {
        let m = SipHashMAC::with_key([0x42; 16]);
        let mac = m.sign(123, 0xcafe_babe);
        assert!(m.verify(123, mac, 0xcafe_babe).is_ok());
    }

    #[cfg(feature = "siphasher")]
    #[test]
    fn siphash_mac_detects_corruption() {
        let m = SipHashMAC::with_key([0x42; 16]);
        let mac = m.sign(123, 0xcafe_babe);
        // Same index, different slot — must not verify (slot_addr is the nonce).
        assert!(m.verify(123, mac, 0xcafe_babf).is_err());
        // Same slot, different index — must not verify.
        assert!(m.verify(124, mac, 0xcafe_babe).is_err());
    }

    #[cfg(feature = "siphasher")]
    #[test]
    fn siphash_mac_key_isolation() {
        let a = SipHashMAC::with_key([0x00; 16]);
        let b = SipHashMAC::with_key([0xff; 16]);
        let mac_a = a.sign(1, 1);
        assert!(b.verify(1, mac_a, 1).is_err());
    }
}
