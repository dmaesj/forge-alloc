//! `Canary<I>` — wraps an inner allocator and writes 8-byte sentinel values
//! immediately before and after every allocation. Verifies canary integrity
//! on deallocate; corruption indicates a linear buffer overflow or adjacent
//! heap corruption.
//!
//! Overhead: per allocation, max(8, align)+8 extra bytes plus one verify
//! per free. When wrapped in a `#[cfg(debug_assertions)]` type alias the
//! release build pays nothing.
//!
//! # Composition limits
//!
//! `Canary<Slab<T, _>>` is restricted: `Slab` issues fixed-stride slots
//! sized for `T`, but `Canary::inner_layout` inflates each request by
//! `max(8, align) + 8` bytes. The inflated request exceeds `Slab`'s
//! `block_stride` for any non-trivially-sized `T` whose stride is the
//! minimum possible (the slab rejects allocations whose layout exceeds
//! the stride). Workarounds:
//!
//! - Use `Canary<BumpArena<_>>` or `Canary<MmapBacked>` where the
//!   inner allocator has no per-allocation stride cap.
//! - Use a `Slab<U, _>` where `U` is a struct with enough padding to
//!   absorb the canary overhead (e.g. `#[repr(C)] struct Padded { t: T,
//!   _pad: [u8; 16] }`) and inflate `align_of::<U>` to 8 to keep the
//!   pre-pad term predictable.
//! - For the same observability without the layout penalty, pair `Slab`
//!   with [`crate::hardening::Quarantine`] (catches use-after-free) and rely on the
//!   SipHash MAC on the freelist link (catches link forgery).
//!
//! See `docs/ARCHITECTURE.md` for the composable-wrapper design.

use core::ptr::NonNull;

use forge_alloc_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// The Canary wrapper.
///
/// The `value: u64` is the canary word stored both before and after the user
/// region; corruption is detected by a `read_unaligned + compare` on free.
///
/// # Do not auto-derive `Debug` on wrappers
///
/// `Canary` deliberately does NOT derive `Debug` — its `value` field is
/// a per-process secret whose leakage to a log scraper, crash reporter,
/// or core dump would let an attacker forge valid canaries elsewhere in
/// the process. If you wrap `Canary` inside an outer type and `#[derive(Debug)]`
/// on that outer type, the derived `Debug` impl will print the canary
/// seed by default. Either hand-write `Debug` on the outer type with
/// the canary field redacted, or skip `#[derive(Debug)]` entirely.
/// (`Canary::canary_value()` exists as the *explicit* accessor for
/// callers that knowingly need the value — the type-level absence of
/// `Debug` is what keeps accidental exposure out of the call graph.)
///
/// # Memory layout per allocation
///
/// ```text
///   inner_ptr ── pre-pad ─┬─ pre canary ─┬─ user data ─┬─ post canary ─┐
///   <───── max(8, align) bytes ──────>   <── size ──>   <── 8 bytes ──>
/// ```
///
/// The pre canary sits at `user_ptr - 8` regardless of alignment; the
/// pre-pad absorbs the alignment slack when `align > 8`. Inner allocation
/// size is `max(8, align) + size + 8`.
pub struct Canary<I> {
    inner: I,
    value: u64,
}

impl<I> Canary<I> {
    /// Wrap with a caller-supplied 64-bit canary seed. Required for
    /// `no_std` builds. The seed should be unpredictable (hardware RNG,
    /// boot-time entropy) to prevent canary forgery.
    #[inline]
    pub const fn new_with_seed(inner: I, seed: u64) -> Self {
        // Ensure the canary is never trivially zero, which is a common
        // value in freshly-zeroed memory and would weaken the check.
        let value = if seed == 0 {
            0xCA_FE_BA_BE_DE_AD_BE_EF
        } else {
            seed
        };
        Self { inner, value }
    }

    /// Wrap with OS-derived entropy. Equivalent to
    /// `new_with_seed(inner, <random>)`. Available on `std` builds.
    #[cfg(feature = "std")]
    pub fn new(inner: I) -> Self {
        // Reuse the same entropy strategy as SipHashMAC: a HashMap
        // RandomState run once gives us 64 bits of OS-derived randomness.
        use std::collections::hash_map::RandomState;
        use std::hash::BuildHasher;
        Self::new_with_seed(inner, RandomState::new().hash_one(0u64))
    }

    /// Borrow the inner allocator.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.inner
    }

    /// Current canary value. Exposed for debugging; do not log in production
    /// (leaking the canary defeats the integrity check).
    #[inline]
    pub fn canary_value(&self) -> u64 {
        self.value
    }

    /// Pre-pad bytes for a layout with `align`. Equals `max(8, align)`.
    #[inline]
    const fn pre_pad(align: usize) -> usize {
        if align < 8 {
            8
        } else {
            align
        }
    }

    /// Inner-allocator layout for a user-facing layout.
    ///
    /// Returns `Err(AllocError)` if the augmented size overflows.
    fn inner_layout(layout: NonZeroLayout) -> Result<NonZeroLayout, AllocError> {
        let pre = Self::pre_pad(layout.align().get());
        let size = layout.size().get();
        let total = pre
            .checked_add(size)
            .and_then(|v| v.checked_add(8))
            .ok_or(AllocError)?;
        // The inner allocation alignment must accommodate the pre-pad +
        // size + post structure. max(8, align) covers all canary access.
        let align = core::cmp::max(8, layout.align().get());
        NonZeroLayout::from_size_align(total, align).map_err(|_| AllocError)
    }

    /// Verify a stored canary at `ptr` matches `self.value`. Panics if
    /// corruption is detected.
    ///
    /// The panic message NEVER includes the expected canary value — that
    /// would leak the per-process canary seed into log scrapers, crash
    /// reporters, and core-dump archives, letting an attacker forge valid
    /// canaries elsewhere. In `debug_assertions` builds we additionally
    /// emit the observed value to aid debugging; release builds emit only
    /// the corruption site label.
    ///
    /// # Safety
    ///
    /// `ptr` must point to 8 readable bytes.
    #[inline]
    unsafe fn verify(&self, ptr: *const u8, where_: &'static str) {
        // SAFETY: caller guarantees 8 readable bytes.
        let actual = unsafe { core::ptr::read_unaligned(ptr.cast::<u64>()) };
        // Constant-time compare via `subtle`. For a single u64 the
        // underlying x86_64 / AArch64 CMP is already one-cycle constant-
        // time at the hardware level — the `ct_eq` wrapper exists to
        // document intent, survive a future refactor that might introduce
        // multi-word secret material, and prevent the compiler from
        // reordering a partial test that leaks intermediate state. The
        // branch on the result is fundamental (panic vs continue) and
        // leaks no information beyond the panic itself.
        use subtle::ConstantTimeEq;
        if !bool::from(actual.ct_eq(&self.value)) {
            // Panicking from inside deallocate is the standard "shout
            // loudly on corruption" pattern; alternatives (silent return,
            // log + continue) are strictly less safe.
            #[cfg(debug_assertions)]
            panic!("canary corruption detected at {where_}: observed {actual:#018x}");
            #[cfg(not(debug_assertions))]
            panic!("canary corruption detected at {where_}");
        }
    }
}

impl<I> Drop for Canary<I> {
    fn drop(&mut self) {
        // Zeroize the canary seed on drop so the value doesn't linger in
        // deallocated stack frames or freed allocator headers. Volatile
        // write + compiler fence keeps the clear from being optimized away.
        // SAFETY: `&mut self.value` is a valid pointer to our own field.
        unsafe { core::ptr::write_volatile(&mut self.value, 0) };
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

unsafe impl<I: Allocator> Deallocator for Canary<I> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        let pre = Self::pre_pad(layout.align().get());
        // SAFETY: ptr came from our allocate(), so user_ptr - pre = inner_ptr.
        let inner_ptr = unsafe { ptr.as_ptr().sub(pre) };
        // Verify pre canary (at user_ptr - 8).
        // SAFETY: pre >= 8 so user_ptr - 8 lies inside the inner allocation.
        unsafe {
            self.verify(ptr.as_ptr().sub(8), "pre-canary");
            // Verify post canary at user_ptr + size.
            self.verify(ptr.as_ptr().add(layout.size().get()), "post-canary");
            // Zero the canary words on free. The seed is a per-process
            // secret; if it persists in deallocated memory until OS
            // reclaim it's UAF-readable by any code that later borrows
            // the freed region (Slab freelist reuse, BumpArena reset,
            // mmap remap). `write_volatile` defeats the dead-store
            // elimination the optimizer would otherwise apply to a
            // write into about-to-be-freed memory.
            core::ptr::write_volatile(ptr.as_ptr().sub(8).cast::<u64>(), 0);
            core::ptr::write_volatile(ptr.as_ptr().add(layout.size().get()).cast::<u64>(), 0);
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
        }
        // SAFETY: inner_ptr / inner layout reproduce the pair returned by
        // self.inner.allocate(inner_layout) during this allocation.
        // The original `allocate(layout)` already evaluated `inner_layout(layout)?`
        // successfully — `Self::inner_layout` is a pure function of `layout`, so
        // the second call here cannot newly fail. Documented as an invariant
        // panic (not a runtime-recoverable error) for completeness.
        let inner_layout = Self::inner_layout(layout)
            .expect("Canary::deallocate: inner_layout(layout) failed for a layout that succeeded at allocate-time — caller passed a different layout than the one used to allocate");
        unsafe {
            self.inner
                .deallocate(NonNull::new_unchecked(inner_ptr), inner_layout);
        }
    }
}

unsafe impl<I: Allocator> Allocator for Canary<I> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let pre = Self::pre_pad(layout.align().get());
        let inner_layout = Self::inner_layout(layout)?;
        let block = self.inner.allocate(inner_layout)?;
        let inner_ptr = block.cast::<u8>().as_ptr();
        // SAFETY: inner_layout reserves pre + size + 8 bytes aligned to
        // max(8, align). user_ptr = inner_ptr + pre is aligned to
        // max(8, align), which is >= layout.align(). The pre canary at
        // user_ptr - 8 lies within [inner_ptr, inner_ptr + pre).
        unsafe {
            let user_ptr = inner_ptr.add(pre);
            // Write pre canary.
            core::ptr::write_unaligned(user_ptr.sub(8).cast::<u64>(), self.value);
            // Write post canary.
            core::ptr::write_unaligned(user_ptr.add(layout.size().get()).cast::<u64>(), self.value);
            Ok(NonNull::slice_from_raw_parts(
                NonNull::new_unchecked(user_ptr),
                layout.size().get(),
            ))
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        // The inner can hold the inner_layout-augmented allocations; our
        // surface capacity is approximately inner_capacity / (1 + overhead).
        // We don't track overhead at runtime, so report inner capacity as
        // an over-approximation; Watermark callers should treat this as
        // best-effort.
        self.inner.capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        // Canary's own detection always panics (the corruption-detected
        // panic terminates the process), so there's no meaningful
        // "events since last check" delta at this layer — the first
        // event would be the last. Forward to inner so silent-disarm
        // counts from underneath still surface through Canary.
        self.inner.corruption_events()
    }
}

impl<I: FixedRange> FixedRange for Canary<I> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.inner.base()
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Pass-through forward so a `commit`-aware consumer reaches the inner
    /// backing when this wrapper sits over a `lazy_commit` `MmapBacked`.
    #[inline]
    fn commit(&self, offset: usize, len: usize) -> Result<(), AllocError> {
        self.inner.commit(offset, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;
    use crate::layout::BumpArena;

    fn build() -> Canary<BumpArena<InlineBacked<2048>>> {
        Canary::new_with_seed(
            BumpArena::new(InlineBacked::<2048>::new()).unwrap(),
            0x1234_5678_9ABC_DEF0,
        )
    }

    #[test]
    fn alloc_then_dealloc_passes_verification() {
        let c = build();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let block = c.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        unsafe {
            // Write something in the user region — must not affect canaries.
            core::ptr::write_bytes(ptr.as_ptr(), 0x42, 32);
            c.deallocate(ptr, layout);
        }
    }

    #[test]
    fn alignment_above_eight_works() {
        let c = build();
        let layout = NonZeroLayout::from_size_align(16, 16).unwrap();
        let block = c.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        assert_eq!(ptr.as_ptr() as usize % 16, 0, "user ptr must be 16-aligned");
        unsafe { c.deallocate(ptr, layout) };
    }

    #[test]
    #[should_panic(expected = "canary corruption")]
    fn pre_canary_overflow_detected() {
        let c = build();
        let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
        let block = c.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        unsafe {
            // Corrupt the pre canary by writing into the byte just before.
            *ptr.as_ptr().sub(1) = 0xFF;
            c.deallocate(ptr, layout); // must panic
        }
    }

    #[test]
    #[should_panic(expected = "canary corruption")]
    fn post_canary_overflow_detected() {
        let c = build();
        let layout = NonZeroLayout::from_size_align(16, 8).unwrap();
        let block = c.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        unsafe {
            // Linear overflow past end of user region.
            *ptr.as_ptr().add(16) = 0xFF;
            c.deallocate(ptr, layout); // must panic
        }
    }

    #[test]
    fn canary_zero_seed_replaced() {
        let c = Canary::new_with_seed(InlineBacked::<64>::new(), 0);
        assert_ne!(c.canary_value(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn os_seeded_constructor_nonzero() {
        let c = Canary::new(InlineBacked::<64>::new());
        // 64-bit OS entropy hitting exactly zero is 2^-64; treat as bug if so.
        assert_ne!(c.canary_value(), 0);
    }

    /// Threat-model pin: the Canary wrapper holds a single per-INSTANCE
    /// secret (`value: u64`); the canary is shared across all
    /// allocations issued through that wrapper. There is NO per-
    /// allocation canary diversification. This is intentional — the
    /// design protects against linear over/underflow corruption, not
    /// against an attacker who has already disclosed one canary value
    /// and uses it to forge canaries on adjacent allocations.
    ///
    /// Pin the contract here so a future "diversify per allocation"
    /// refactor announces itself by breaking this test.
    #[test]
    fn canary_value_is_per_instance_not_per_allocation() {
        let c = build();
        let layout = NonZeroLayout::from_size_align(32, 8).unwrap();
        let v_before_first = c.canary_value();
        let block_a = c.allocate(layout).unwrap();
        let v_after_first = c.canary_value();
        unsafe { c.deallocate(block_a.cast(), layout) };
        let v_after_dealloc = c.canary_value();
        let _block_b = c.allocate(layout).unwrap();
        let v_after_second = c.canary_value();
        assert_eq!(v_before_first, v_after_first);
        assert_eq!(v_after_first, v_after_dealloc);
        assert_eq!(v_after_dealloc, v_after_second);
    }
}
