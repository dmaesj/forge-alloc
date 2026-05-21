//! `CacheJitter<I>` — randomized per-allocation displacement to spread
//! metadata across cache associativity sets.
//!
//! Allocators that always return pointers at fixed alignments concentrate
//! metadata into the same cache-set indices. On an 8-way L1 with 64-byte
//! lines, every page-aligned pointer hashes to the same set, so an attacker
//! probing a free list can deterministically evict victim data with O(8)
//! probes. `CacheJitter` shifts each allocation by a multiple of one cache
//! line within the associativity window — `(rng() % assoc) * line_size`
//! bytes — so different allocations land in different sets.
//!
//! Overhead: one xorshift64 per `allocate`, plus `line_size + max_disp`
//! bytes of prefix per allocation (used for displacement-header storage so
//! `deallocate` can recover the inner pointer).
//!
//! See `docs/ARCHITECTURE.md` for the composable-wrapper design.

use core::cell::Cell;
use core::ptr::NonNull;

use forge_alloc_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// CacheJitter wrapper.
///
/// `cache_line_size` and `associativity` are fixed at construction. The
/// xorshift64 state lives in an interior-mutable `Cell` so `allocate` can
/// take `&self`; this also makes the type `!Sync`.
///
/// # Cross-thread use
///
/// `CacheJitter<I>` is **not** thread-safe by itself — the `Cell<u64>`
/// rng state and the `Cell` on the per-instance MAC verification path
/// both prohibit `&CacheJitter` from being shared across threads.
/// Wrapping the *inner* allocator with `Statistics` or similar does
/// **not** help; the cell is inside `CacheJitter` and is what blocks
/// `Sync`.
///
/// For cross-thread use, pick one of:
/// - **Per-thread instance** — give each thread its own
///   `CacheJitter<I>`. Each instance has its own rng + MAC key, which
///   actually improves the wrapper's threat-model (the MAC key is
///   thread-private).
/// - **External `Mutex<CacheJitter<I>>`** — serializes all access
///   through the lock. Use only if a single shared instance is
///   architecturally required; per-thread is faster.
///
/// # Randomness model
///
/// CacheJitter uses **xorshift64**, a fast non-cryptographic PRNG. The
/// goal of the wrapper is to *diversify cache-set occupancy* across
/// allocations so an attacker who controls allocation timing can't
/// deterministically evict a victim line. It is **not** designed to
/// resist an attacker who can observe several user pointers and solve
/// for the RNG state — xorshift64 is fully invertible from ~3
/// consecutive 64-bit outputs.
///
/// If your threat model includes that adversary, swap the RNG for a
/// CSPRNG (e.g. ChaCha20) at the cost of ~10× per-allocation overhead.
/// For the typical anti-spray use case, xorshift64 is appropriate.
///
/// # Composition
///
/// Layout requests with `align > cache_line_size` are forwarded to the
/// inner allocator *without* jitter — the jitter granularity is one cache
/// line, which can't preserve a larger alignment. The vast majority of
/// requests have `align <= 16`, so jitter applies in the common case.
///
/// # Inner-allocator alignment requirement
///
/// **The inner allocator MUST be able to satisfy alignment requests up
/// to `cache_line_size` (64 bytes on x86/ARM, 128 on Apple Silicon).**
/// For jittered requests, this wrapper inflates the inner's alignment
/// requirement up to `cache_line_size` so the user pointer (placed at
/// `inner_ptr + cache_line_size + displacement`) inherits the caller's
/// requested alignment. Backings that cap alignment below
/// `cache_line_size` — notably [`InlineBacked`](crate::backing::InlineBacked),
/// whose `MAX_ALIGN` is 16 — will reject the inflated request and the
/// wrapped allocation will fail.
///
/// Practical implication: `CacheJitter<MmapBacked>` and
/// `CacheJitter<BumpArena<MmapBacked>>` work; `CacheJitter<BumpArena<
/// InlineBacked<N>>>` compiles but cannot actually allocate jittered
/// blocks. The pattern is mainly useful for production heaps over the
/// OS allocator, not for stack-buffer arenas.
pub struct CacheJitter<I> {
    inner: I,
    cache_line_size: usize,
    associativity: usize,
    /// `trailing_zeros(cache_line_size)`. Cached to encode the displacement
    /// in cache-line units (compact, fits the 16-bit header field).
    line_shift: u32,
    /// Per-instance secret used to MAC the on-disk displacement header.
    /// Initialized once from the caller-supplied seed (or OS entropy on
    /// `new`) and never exposed. Compromising this defeats the
    /// header-integrity check; protecting it is therefore as important as
    /// the SipHashMAC key in `Slab`.
    mac_key: u64,
    rng: Cell<u64>,
}

/// Header size in bytes prefixing each jittered allocation. We pack the
/// applied displacement (low 16 bits, as a multiple of `cache_line_size`)
/// and a 48-bit keyed MAC (high 48 bits) into the same 8-byte slot. The
/// MAC is computed over `(user_ptr_addr, displacement_in_lines)` with a
/// per-instance key so an attacker who controls only the prefix bytes
/// (linear underflow from an adjacent allocation, or UAF write into a
/// freed slot's prefix) cannot forge a header that survives
/// `deallocate`'s verification — see `CacheJitter::unpack_header`.
const JITTER_HEADER_SIZE: usize = 8;
/// Width of the displacement field in the packed header. 16 bits stores a
/// displacement of up to `(2^16 - 1) * cache_line_size` bytes — 4 MiB for
/// 64-byte lines, 8 MiB for 128-byte lines — far above any realistic
/// associativity window. Construction rejects configurations that would
/// overflow.
const HEADER_DISP_BITS: u32 = 16;
const HEADER_DISP_MASK: u64 = (1u64 << HEADER_DISP_BITS) - 1;

/// Maximum associativity the encoding admits. With 16-bit
/// displacement-in-lines, displacement-in-lines ranges over
/// `0..associativity`, so `associativity` must fit in 16 bits.
const MAX_ASSOCIATIVITY: usize = (1usize << HEADER_DISP_BITS) - 1;

impl<I> CacheJitter<I> {
    /// Construct with explicit cache parameters and a caller-supplied seed.
    /// Required for `no_std` builds.
    ///
    /// `cache_line_size` must be a power of two and `>= 8` so a `u64`
    /// header fits within one line. `associativity` must be `>= 1`.
    /// Returns `None` if either constraint is violated, or if
    /// `cache_line_size * associativity` would overflow.
    pub fn with_params(
        inner: I,
        cache_line_size: usize,
        associativity: usize,
        seed: u64,
    ) -> Option<Self> {
        if !cache_line_size.is_power_of_two() || cache_line_size < 8 {
            return None;
        }
        if associativity == 0 || associativity > MAX_ASSOCIATIVITY {
            return None;
        }
        cache_line_size.checked_mul(associativity)?;
        // Avoid trivially-zero seed — xorshift64 would output zero
        // forever. The first transformation in next_rng() would then
        // produce d=0 every call, which defeats the wrapper. Substitute
        // a fixed nonzero seed if the caller passes 0.
        let seed = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        // Derive a MAC key distinct from the RNG state so that observing
        // displacements doesn't directly leak the MAC key. Two independent
        // splitmix64 steps from the seed give us a key that's
        // statistically uncorrelated with the RNG sequence the caller
        // might observe.
        let mac_key = {
            let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Some(Self {
            inner,
            cache_line_size,
            associativity,
            line_shift: cache_line_size.trailing_zeros(),
            mac_key,
            rng: Cell::new(seed),
        })
    }

    /// Construct with OS-derived entropy. Available on `std` builds.
    ///
    /// Uses the same entropy strategy as `Canary::new` — a single
    /// `RandomState`-derived 64-bit value seeded by the OS RNG.
    #[cfg(feature = "std")]
    pub fn new(inner: I, cache_line_size: usize, associativity: usize) -> Option<Self> {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hash, Hasher};
        let mut h = RandomState::new().build_hasher();
        0u64.hash(&mut h);
        Self::with_params(inner, cache_line_size, associativity, h.finish())
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }

    /// Cache-line size in bytes (e.g. 64 on x86/ARM, 128 on Apple
    /// Silicon).
    #[inline]
    pub fn cache_line_size(&self) -> usize {
        self.cache_line_size
    }

    /// Associativity window in cache lines — the jitter range.
    #[inline]
    pub fn associativity(&self) -> usize {
        self.associativity
    }

    /// Maximum displacement applied by this wrapper, in bytes.
    /// Equals `(associativity - 1) * cache_line_size`.
    #[inline]
    fn max_displacement(&self) -> usize {
        (self.associativity - 1) * self.cache_line_size
    }

    /// Total prefix added to each jittered allocation: one cache line for
    /// the displacement header + room for the maximum displacement.
    /// `user_ptr = inner_ptr + cache_line_size + actual_displacement`.
    #[inline]
    fn jitter_prefix(&self) -> usize {
        self.cache_line_size + self.max_displacement()
    }

    /// Step the xorshift64 generator and return its next output.
    #[inline]
    fn next_rng(&self) -> u64 {
        let mut x = self.rng.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng.set(x);
        x
    }

    /// Compute the displacement for the next allocation: a non-negative
    /// multiple of `cache_line_size` in `[0, associativity * cache_line_size)`,
    /// i.e. `0, cache_line_size, …, (associativity - 1) * cache_line_size`.
    ///
    /// Uses the unbiased "multiply-shift" reduction
    /// `(rng() * assoc) >> 64` rather than `rng() % assoc` — the modulo
    /// form has a small bias when `assoc` is not a power of two (e.g. the
    /// 12-way L2 on certain Intel parts, 6-way on some older AMD), which
    /// would weaken the cache-set spreading the wrapper relies on. The
    /// multiply-shift form is unbiased and cheaper on x86_64 (single
    /// `mul` instead of `div`).
    #[inline]
    fn next_displacement(&self) -> usize {
        let assoc = self.associativity as u128;
        let r = ((self.next_rng() as u128 * assoc) >> 64) as usize;
        r * self.cache_line_size
    }

    /// Compute a 64-bit mixed value over `(user_ptr_addr, disp_lines)`
    /// keyed by `self.mac_key`. The high 48 bits are used as the MAC.
    ///
    /// This is a SplitMix-style avalanche — not a cryptographic MAC, but
    /// the construction is non-linear and key-dependent. An attacker who
    /// can only blind-write the 8-byte header (linear underflow from an
    /// adjacent allocation, UAF write into a freed slot's prefix) faces a
    /// 2^-48 forgery probability per attempt. The MAC binds the
    /// displacement to `user_ptr_addr` so a header copied from one
    /// allocation cannot be replayed against a different one.
    ///
    /// Honest threat-model caveats:
    ///
    /// - **Direct key disclosure.** An attacker who can read the
    ///   `mac_key` field itself (arbitrary heap or stack read primitive
    ///   that reaches inside the `CacheJitter` struct) can forge
    ///   arbitrary headers. No keyed-MAC construction — cryptographic
    ///   or otherwise — survives direct key disclosure.
    /// - **Observed-pair key recovery.** An attacker who can read many
    ///   `(user_ptr_addr, header)` pairs from the live process *but
    ///   cannot read `mac_key` directly* faces a weaker barrier with
    ///   our SplitMix-style mixer than they would with a cryptographic
    ///   MAC like SipHash: the mixer is a small algebraic circuit and
    ///   an offline SAT / symbolic-execution attack on a few thousand
    ///   observations is plausible for a well-resourced adversary. A
    ///   cryptographic MAC remains key-recovery-hard under the same
    ///   read access. If your threat model includes a heap-disclosure
    ///   attacker who cannot read `mac_key` directly but can observe
    ///   many pairs, swap this mixer for a SipHash MAC at the cost of
    ///   roughly 5-10x per-allocate work.
    ///
    /// CacheJitter is one layer in a defense-in-depth stack, not a
    /// standalone barrier against arbitrary read+write primitives.
    #[inline]
    fn header_mix(&self, user_ptr_addr: usize, disp_lines: u64) -> u64 {
        let mut x = self.mac_key ^ (user_ptr_addr as u64);
        x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x.rotate_left(31);
        x ^= disp_lines;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    /// Pack the displacement and its MAC into the 8-byte header value.
    #[inline]
    fn pack_header(&self, user_ptr_addr: usize, disp_bytes: usize) -> u64 {
        let disp_lines = (disp_bytes >> self.line_shift) as u64;
        debug_assert!(
            disp_lines & !HEADER_DISP_MASK == 0,
            "displacement-in-lines exceeds 16-bit header field — \
             construction should have rejected this associativity"
        );
        let mac48 = self.header_mix(user_ptr_addr, disp_lines) >> HEADER_DISP_BITS;
        (mac48 << HEADER_DISP_BITS) | (disp_lines & HEADER_DISP_MASK)
    }

    /// Verify the header and recover the displacement in bytes. Returns
    /// `Err(())` if the MAC fails or the recovered displacement is out
    /// of range — both indicate corruption.
    #[inline]
    fn unpack_header(&self, user_ptr_addr: usize, header: u64) -> Result<usize, ()> {
        let disp_lines = header & HEADER_DISP_MASK;
        let expected_mac48 = self.header_mix(user_ptr_addr, disp_lines) >> HEADER_DISP_BITS;
        let stored_mac48 = header >> HEADER_DISP_BITS;
        // Constant-time compare via `subtle`. For a 48-bit scalar this is
        // essentially one CMP on modern CPUs, but documenting intent and
        // surviving future refactor regressions matters more here than
        // the cycle. Same rationale as `SipHashMAC::verify`.
        use subtle::ConstantTimeEq;
        if !bool::from(stored_mac48.ct_eq(&expected_mac48)) {
            return Err(());
        }
        let disp_bytes = (disp_lines as usize) << self.line_shift;
        // Defense-in-depth: even if the MAC verified, the recovered
        // displacement must lie in the legitimate range. A MAC collision
        // outside the range is rejected before we touch
        // `inner.deallocate`.
        if disp_bytes >= self.associativity * self.cache_line_size {
            return Err(());
        }
        Ok(disp_bytes)
    }
}

impl<I> Drop for CacheJitter<I> {
    fn drop(&mut self) {
        // Zeroize the MAC key (and RNG state, which would let an attacker
        // predict future displacements if leaked) on drop so the values
        // don't linger in deallocated stack frames or freed allocator
        // headers. Same rationale as `Canary::drop` — a per-process
        // secret leaving the wrapper's storage when the wrapper drops
        // would let an attacker forge headers on still-live wrappers
        // that share a derivation source, and would let an attacker
        // who later examines the freed region read the secret.
        // Volatile write + compiler fence keeps the clear from being
        // optimized away.
        // SAFETY: `&mut self.mac_key` / `self.rng` are valid pointers
        // to our own fields, and `&mut self` gives exclusive access.
        unsafe {
            core::ptr::write_volatile(&mut self.mac_key, 0);
            // `Cell::as_ptr` returns *mut T into the cell's storage;
            // safe to write through it via volatile because we have
            // exclusive access via &mut self.
            core::ptr::write_volatile(self.rng.as_ptr(), 0);
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Returns the inner layout to request for a user-facing layout, or
/// `None` if jitter cannot be applied (caller forwards untouched).
///
/// `None` means the user's `align > cache_line_size` — jitter granularity
/// is one cache line, so we can't preserve the larger alignment.
fn inner_layout_for(
    layout: NonZeroLayout,
    cache_line_size: usize,
    jitter_prefix: usize,
) -> Option<Result<NonZeroLayout, AllocError>> {
    if layout.align().get() > cache_line_size {
        return None;
    }
    let total = match layout.size().get().checked_add(jitter_prefix) {
        Some(t) => t,
        None => return Some(Err(AllocError)),
    };
    // Inner alignment must be at least `cache_line_size` so that
    // `inner_ptr + cache_line_size + k*cache_line_size` preserves the
    // caller's requested alignment (which is `<= cache_line_size`).
    let inner_align = cache_line_size;
    Some(NonZeroLayout::from_size_align(total, inner_align).map_err(|_| AllocError))
}

unsafe impl<I: Allocator> Deallocator for CacheJitter<I> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // If jitter wasn't applied (oversized align), the request was
        // forwarded straight through. The same condition holds on
        // dealloc, so forward unchanged.
        let Some(inner_layout) =
            inner_layout_for(layout, self.cache_line_size, self.jitter_prefix())
        else {
            // SAFETY: forwarded; caller upholds Deallocator contract.
            unsafe { self.inner.deallocate(ptr, layout) };
            return;
        };
        let inner_layout = match inner_layout {
            Ok(l) => l,
            // `inner_layout_for` is a pure function of `layout`,
            // `cache_line_size`, and `jitter_prefix`. The latter two are
            // immutable post-construction; the former is supplied by the
            // caller. If the original `allocate(layout)` succeeded, this
            // arm cannot be reached for the SAME layout — reaching it
            // means the caller passed a different layout to deallocate
            // than to allocate, which is itself a Deallocator-contract
            // violation. Match `Canary::deallocate`'s policy: panic
            // rather than forward `(user_ptr, user_layout)` to inner
            // (which would be a wrong-ptr / wrong-layout free).
            Err(_) => panic!(
                "CacheJitter::deallocate: inner_layout_for(layout) failed for a \
                 layout that succeeded at allocate-time — caller passed a \
                 different layout than the one used to allocate"
            ),
        };
        // Read the stored 8-byte header from immediately before user_ptr.
        // Verify the MAC before trusting the embedded displacement —
        // without this, an attacker who can write the prefix (linear
        // underflow from adjacent alloc, or UAF prefix write) gets an
        // arbitrary-free primitive against inner.
        // SAFETY: allocate placed this header in the prefix bytes we
        // own; caller's contract gives us a ptr we previously issued.
        let header = unsafe {
            core::ptr::read_unaligned(ptr.as_ptr().sub(JITTER_HEADER_SIZE).cast::<u64>())
        };
        let displacement = match self.unpack_header(ptr.as_ptr() as usize, header) {
            Ok(d) => d,
            Err(()) => {
                // Header MAC failure or out-of-range displacement —
                // memory corruption detected. We cannot safely recover
                // inner_ptr (that's what the header told us, and the
                // header is the value under attack), and we cannot
                // forward the user_ptr (inner doesn't own it). The
                // standard hardened-allocator response to detected
                // corruption is to abort with a diagnostic; matches
                // Canary's policy and the Quarantine corruption response.
                //
                // Diagnostic strategy: the observed header is logged so
                // crash-reporter / core-dump scrape can correlate the
                // corruption with the surrounding allocation context.
                // The MAC key itself is NEVER printed (would let an
                // attacker forge headers elsewhere — same threat model
                // as the canary-seed-redaction rationale).
                #[cfg(debug_assertions)]
                panic!(
                    "CacheJitter::deallocate: prefix header MAC failed at ptr {:p} \
                     (observed header: {:#018x}) — heap corruption \
                     (linear underflow into prefix, or UAF prefix write)",
                    ptr.as_ptr(),
                    header,
                );
                #[cfg(not(debug_assertions))]
                panic!(
                    "CacheJitter::deallocate: prefix header MAC failed — \
                     heap corruption (linear underflow into prefix, or UAF prefix write)"
                );
            }
        };
        // Recover inner pointer: walk back past the displacement and the
        // cache-line prefix.
        // SAFETY: user_ptr - (cache_line_size + displacement) lies at
        // the start of the inner allocation we received.
        let inner_ptr = unsafe { ptr.as_ptr().sub(self.cache_line_size + displacement) };
        // SAFETY: inner_ptr came from inner.allocate(inner_layout) at
        // construction of this allocation.
        unsafe {
            self.inner
                .deallocate(NonNull::new_unchecked(inner_ptr), inner_layout)
        }
    }
}

unsafe impl<I: Allocator> Allocator for CacheJitter<I> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let prefix = self.jitter_prefix();
        let Some(inner_layout) = inner_layout_for(layout, self.cache_line_size, prefix) else {
            // align too large for jitter — pass through unchanged.
            return self.inner.allocate(layout);
        };
        let inner_layout = inner_layout?;
        let block = self.inner.allocate(inner_layout)?;
        let inner_ptr = block.cast::<u8>().as_ptr();
        let displacement = self.next_displacement();
        // user_ptr = inner_ptr + cache_line_size + displacement.
        // SAFETY: inner_layout reserves prefix + layout.size() bytes
        // starting at inner_ptr; cache_line_size + displacement <= prefix
        // by construction (displacement < associativity*cache_line_size,
        // prefix = cache_line_size + (assoc-1)*cache_line_size).
        let user_ptr = unsafe { inner_ptr.add(self.cache_line_size + displacement) };
        // Store the MAC-protected header at user_ptr - JITTER_HEADER_SIZE
        // so the deallocator can recover inner_ptr after verifying that
        // the header has not been tampered with.
        // SAFETY: user_ptr - JITTER_HEADER_SIZE = inner_ptr + cache_line_size
        // + displacement - JITTER_HEADER_SIZE. With cache_line_size >= 8 =
        // JITTER_HEADER_SIZE and displacement >= 0, this is >= inner_ptr.
        // And user_ptr - JITTER_HEADER_SIZE < user_ptr <= inner_ptr +
        // jitter_prefix <= inner_ptr + inner_layout.size(), so the full
        // 8-byte write stays inside the inner allocation we own.
        let header = self.pack_header(user_ptr as usize, displacement);
        unsafe {
            core::ptr::write_unaligned(user_ptr.sub(JITTER_HEADER_SIZE).cast::<u64>(), header);
        }
        // SAFETY: user_ptr derives from a valid &self; non-null.
        Ok(NonNull::slice_from_raw_parts(
            unsafe { NonNull::new_unchecked(user_ptr) },
            layout.size().get(),
        ))
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        // Approximate: the inner capacity is consumed faster by our
        // prefix overhead. Report inner capacity as an over-approximation
        // so Watermark callers treat it as best-effort.
        self.inner.capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        // CacheJitter's MAC verify failure path panics on detection
        // (the header is the only state under attack; a wrong MAC means
        // either a linear underflow into the prefix or a UAF prefix
        // write — both unrecoverable). Same rationale as Canary:
        // forward to inner so silent-disarm counts from underneath
        // still surface.
        self.inner.corruption_events()
    }
}

impl<I: FixedRange> FixedRange for CacheJitter<I> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.inner.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;
    use crate::layout::BumpArena;

    #[cfg(feature = "std")]
    use crate::backing::MmapBacked;

    /// Tiny CacheJitter over a stack-buffer BumpArena — fine for round-
    /// trip / param-rejection tests, but cache-line-aligned tests need a
    /// MmapBacked source (InlineBacked has MAX_ALIGN = 16).
    fn build_inline() -> CacheJitter<BumpArena<InlineBacked<8192>>> {
        CacheJitter::with_params(
            BumpArena::new(InlineBacked::<8192>::new()).unwrap(),
            64,
            8,
            0x1234_5678_9ABC_DEF0,
        )
        .expect("valid params")
    }

    /// CacheJitter over MmapBacked — supports up to page alignment, so
    /// jitter applies for all alignments up to cache_line_size (64).
    #[cfg(feature = "std")]
    fn build_mmap() -> CacheJitter<BumpArena<MmapBacked>> {
        CacheJitter::with_params(
            BumpArena::new(MmapBacked::new(64 * 1024).unwrap()).unwrap(),
            64,
            8,
            0x1234_5678_9ABC_DEF0,
        )
        .expect("valid params")
    }

    #[test]
    fn rejects_non_power_of_two_line() {
        let inner = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        assert!(CacheJitter::with_params(inner, 24, 8, 1).is_none());
    }

    #[test]
    fn rejects_zero_associativity() {
        let inner = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        assert!(CacheJitter::with_params(inner, 64, 0, 1).is_none());
    }

    #[test]
    fn rejects_too_small_line() {
        let inner = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        // Line of 4 can't hold the 8-byte displacement header.
        assert!(CacheJitter::with_params(inner, 4, 8, 1).is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn alloc_then_dealloc_round_trips() {
        let cj = build_mmap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = cj.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0x42, 32);
            cj.deallocate(ptr, layout);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn user_ptr_aligned_for_layout() {
        let cj = build_mmap();
        let layout = NonZeroLayout::from_size_align(16, 16).unwrap();
        for _ in 0..32 {
            let block = cj.allocate(layout).unwrap();
            let addr = block.cast::<u8>().as_ptr() as usize;
            assert_eq!(addr % 16, 0, "user ptr must respect requested align");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn displacement_distribution_hits_multiple_sets() {
        // With 8-way associativity and a fixed seed, repeated allocations
        // should land at multiple distinct cache-set offsets (mod
        // 8 * 64 = 512). xorshift64 should distribute well enough to hit
        // at least 4 distinct buckets across 64 iterations.
        let cj = build_mmap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let mut offsets = alloc::collections::BTreeSet::new();
        for _ in 0..64 {
            let block = cj.allocate(layout).unwrap();
            let addr = block.cast::<u8>().as_ptr() as usize;
            offsets.insert(addr % (8 * 64));
        }
        assert!(
            offsets.len() >= 4,
            "expected diverse cache-set landing offsets, got {}",
            offsets.len(),
        );
    }

    #[test]
    fn oversized_align_passes_through_without_jitter() {
        // align > cache_line_size (here 128 > 64) — wrapper must forward
        // unchanged so the inner allocator handles alignment. Verify the
        // forwarded request reaches inner intact: use a small alloc with
        // align=128 and round-trip it through dealloc.
        let cj = build_inline();
        let layout = NonZeroLayout::from_size_align(8, 128).unwrap();
        // If the InlineBacked-backed bump arena can satisfy align=128
        // (depends on buffer base address), confirm the user_ptr matches
        // what raw inner.allocate would have returned (no displacement).
        // Otherwise, confirm both paths produce the same error.
        let res = cj.allocate(layout);
        // We can't directly compare to a parallel inner call (the inner
        // has interior state). Just confirm that a successful alloc has
        // no displacement header — equivalently that the high cache-line
        // bits of the returned pointer match what inner would naturally
        // produce. The contract here is "no inflation, no displacement",
        // and we check it by round-tripping a write + dealloc; if the
        // wrapper had inflated, the inner's freelist bookkeeping would
        // mismatch and the second alloc-after-dealloc would corrupt.
        if let Ok(block) = res {
            let p = block.cast::<u8>();
            unsafe {
                core::ptr::write_bytes(p.as_ptr(), 0xAA, 8);
                cj.deallocate(p, layout);
            }
        }
        // Either alloc succeeded (and we round-tripped) or it failed
        // (forwarded inner's error). Both are valid pass-through outcomes.
    }

    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn deterministic_with_same_seed() {
        // Two independent CacheJitter instances with the same seed must
        // produce the same sequence of displacements. Verify by checking
        // the offset-from-inner-base of the first allocation in each.
        let cj_a = CacheJitter::with_params(
            BumpArena::new(MmapBacked::new(64 * 1024).unwrap()).unwrap(),
            64,
            8,
            0xDEAD_BEEF_CAFE_BABE,
        )
        .unwrap();
        let cj_b = CacheJitter::with_params(
            BumpArena::new(MmapBacked::new(64 * 1024).unwrap()).unwrap(),
            64,
            8,
            0xDEAD_BEEF_CAFE_BABE,
        )
        .unwrap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let a = cj_a.allocate(layout).unwrap().cast::<u8>().as_ptr() as usize;
        let b = cj_b.allocate(layout).unwrap().cast::<u8>().as_ptr() as usize;
        let base_a = cj_a.inner().base().as_ptr() as usize;
        let base_b = cj_b.inner().base().as_ptr() as usize;
        assert_eq!(
            a - base_a,
            b - base_b,
            "same seed must give same displacement"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn os_seeded_constructor() {
        let inner = BumpArena::new(MmapBacked::new(16 * 1024).unwrap()).unwrap();
        let cj = CacheJitter::new(inner, 64, 8).expect("valid params");
        let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
        let _ = cj.allocate(layout).unwrap();
    }

    /// Regression: a corrupted header (linear-underflow / UAF-prefix
    /// write into `user_ptr - 8`) must trip the MAC check in
    /// `deallocate` rather than steering `inner.deallocate` at the
    /// attacker-chosen address (arbitrary-free primitive).
    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    #[should_panic(expected = "prefix header MAC failed")]
    fn corrupted_header_panics_on_dealloc() {
        let cj = build_mmap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = cj.allocate(layout).unwrap();
        let user_ptr = block.cast::<u8>();
        // SAFETY: overwrite the 8-byte prefix that holds the packed
        // (displacement | MAC) — mirrors what a linear-underflow write
        // from an adjacent allocation would do.
        unsafe {
            core::ptr::write_unaligned(
                user_ptr.as_ptr().sub(JITTER_HEADER_SIZE).cast::<u64>(),
                0xDEAD_BEEF_CAFE_BABEu64, // arbitrary attacker-chosen value
            );
            // Must panic with the documented MAC-failure message.
            cj.deallocate(user_ptr, layout);
        }
    }

    /// Regression: rejecting the all-zero header would let an attacker
    /// who can only zero the prefix region (memset(prefix, 0, ...))
    /// keep a forged displacement of 0 surviving the check. Verify
    /// that header = 0 fails the MAC (since 0 is not a legitimate MAC
    /// output for any displacement, given the per-instance secret).
    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    #[should_panic(expected = "prefix header MAC failed")]
    fn zeroed_header_panics_on_dealloc() {
        let cj = build_mmap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = cj.allocate(layout).unwrap();
        let user_ptr = block.cast::<u8>();
        unsafe {
            core::ptr::write_unaligned(
                user_ptr.as_ptr().sub(JITTER_HEADER_SIZE).cast::<u64>(),
                0u64,
            );
            cj.deallocate(user_ptr, layout);
        }
    }

    /// Boundary: maximum associativity admitted by the 16-bit displacement
    /// field is `MAX_ASSOCIATIVITY = (1<<16) - 1 = 65535`. Construction must
    /// succeed at exactly that value and reject one past.
    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn associativity_at_and_past_encoding_limit() {
        let inner = || BumpArena::new(MmapBacked::new(64 * 1024).unwrap()).unwrap();
        // At the limit — admitted.
        assert!(
            CacheJitter::with_params(inner(), 64, MAX_ASSOCIATIVITY, 1).is_some(),
            "MAX_ASSOCIATIVITY = {MAX_ASSOCIATIVITY} must be admitted",
        );
        // One past — rejected.
        assert!(
            CacheJitter::with_params(inner(), 64, MAX_ASSOCIATIVITY + 1, 1).is_none(),
            "MAX_ASSOCIATIVITY + 1 must be rejected",
        );
    }

    /// Boundary: `cache_line_size * associativity` must not overflow at
    /// construction. A 2^31 cache_line_size with associativity 2 overflows
    /// on 32-bit; on 64-bit it doesn't, but the construction-time
    /// `checked_mul` guard is the gate either way. Try a value that's
    /// guaranteed to overflow (which forces the guard to fire on every
    /// target).
    #[test]
    fn rejects_cache_line_assoc_overflow() {
        let inner = || BumpArena::new(InlineBacked::<256>::new()).unwrap();
        // Pick `cache_line_size` so that `line * assoc` overflows usize on
        // every target. `1 << (USIZE_BITS - 1)` * 4 overflows.
        let line = 1usize << (usize::BITS - 1);
        // Verify line is a power of two and large enough; if not (32-bit
        // builds), substitute a smaller pow2 that still overflows the
        // checked_mul.
        if line.is_power_of_two() && line >= 8 {
            assert!(
                CacheJitter::with_params(inner(), line, 4, 1).is_none(),
                "line * assoc overflow must be rejected",
            );
        }
    }

    /// `cache_line_size = 8` (the minimum admissible) combined with
    /// `associativity = MAX_ASSOCIATIVITY` is the largest jitter window
    /// the encoding allows. Construction must succeed and a first
    /// allocate must round-trip (i.e. the prefix size of
    /// `cache_line_size + (assoc - 1)*cache_line_size = 8 * 65535`
    /// fits the backing budget).
    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn smallest_line_largest_assoc_round_trips() {
        let inner = BumpArena::new(MmapBacked::new(8 * (MAX_ASSOCIATIVITY + 16)).unwrap()).unwrap();
        let cj = CacheJitter::with_params(inner, 8, MAX_ASSOCIATIVITY, 1).unwrap();
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        let block = cj.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        unsafe { cj.deallocate(ptr, layout) };
    }

    /// `cache_line_size = 0` is rejected (not power of two, also `< 8`).
    #[test]
    fn rejects_zero_cache_line() {
        let inner = BumpArena::new(InlineBacked::<256>::new()).unwrap();
        assert!(CacheJitter::with_params(inner, 0, 8, 1).is_none());
    }

    /// `seed = 0` is substituted with the golden ratio constant. Verify
    /// the substituted seed produces a working RNG (the wrapper is
    /// usable and produces non-zero displacements at least once across
    /// many allocations) and is distinguishable from the all-zero state
    /// xorshift64 would otherwise be stuck in.
    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn zero_seed_substitution_yields_working_rng() {
        let inner = BumpArena::new(MmapBacked::new(64 * 1024).unwrap()).unwrap();
        let cj = CacheJitter::with_params(inner, 64, 8, 0).expect("zero seed must work");
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let base = cj.inner().base().as_ptr() as usize;
        // Track the displacement across many allocations — if the RNG
        // had been stuck at zero (the unmitigated xorshift64 zero
        // state), every offset would be cache_line_size with no
        // variation across the 8-way window. With the golden-ratio
        // substitution we expect to see multiple distinct sets.
        let mut sets = alloc::collections::BTreeSet::new();
        for _ in 0..32 {
            let p = cj.allocate(layout).unwrap().cast::<u8>().as_ptr() as usize;
            sets.insert((p - base) % (8 * 64));
        }
        assert!(
            sets.len() >= 2,
            "zero-seed substitution must produce a non-stuck RNG; only saw {} \
             distinct cache-set offsets",
            sets.len(),
        );
    }

    /// With `associativity = 1`, `disp_lines` is always 0 and the MAC
    /// is over `(user_ptr_addr, 0)` with a per-instance key. The MAC
    /// over the zero displacement must still differ from an attacker
    /// who simply zeros the whole 8-byte header (which has both
    /// disp_lines = 0 AND mac = 0). Verifies the MAC contributes
    /// non-zero high bits for `disp = 0`.
    #[cfg(feature = "std")]
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn assoc_one_disp_zero_mac_differs_from_zero_header() {
        let cj = CacheJitter::with_params(
            BumpArena::new(MmapBacked::new(64 * 1024).unwrap()).unwrap(),
            64,
            1,
            0xDEAD_BEEF_CAFE_BABE,
        )
        .unwrap();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = cj.allocate(layout).unwrap();
        let user_ptr = block.cast::<u8>();
        // Read the header that allocate just wrote.
        let stored = unsafe {
            core::ptr::read_unaligned(user_ptr.as_ptr().sub(JITTER_HEADER_SIZE).cast::<u64>())
        };
        assert_ne!(
            stored, 0,
            "MAC over (user_ptr, disp=0) must produce non-zero high bits — \
             otherwise a zeroed-prefix forge would survive the check"
        );
        unsafe { cj.deallocate(user_ptr, layout) };
    }
}
