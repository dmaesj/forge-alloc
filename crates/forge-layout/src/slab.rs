//! `Slab<T, B, M>` — typed fixed-stride block allocator with optional
//! freelist MAC.
//!
//! Each slot holds either a live `T` (size & alignment per `T`) or a
//! a `FreeLink` (when on the free list). The block stride is the max of the
//! two, so slot reuse is always size- and alignment-compatible.
//!
//! Freelist uses 1-based indices (`0` = empty), separating the "list empty"
//! sentinel from any valid slot — see spec §4.5 / §6.2 (revision 1.5) for the
//! design.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use forge_core::{
    AllocError, Allocator, Deallocator, FixedRange, FreelistProtection, NoProtection,
    NonZeroLayout,
};

// Cross-check: `B: FixedRange` is required so that `Slab` can re-query
// `self.backing.base()` after the struct has moved. Backings whose
// `base()` is structure-relative (e.g. `InlineBacked`) return DIFFERENT
// addresses before and after a move; storing an absolute `NonNull<u8>`
// captured at construction time would leave the slab pointing at the
// OLD location, corrupting every subsequent `allocate` / `deallocate`.
// See the "Slab move-safety" pin test for the exact failure mode.

/// A free-list link stored inside a free slot.
///
/// `next_idx` is the 1-based index of the next free slot; `0` means end of
/// list. `mac` is the integrity tag computed by [`FreelistProtection::sign`];
/// [`NoProtection`] always writes `0`. The field exists unconditionally so
/// that block_stride is stable across `M` choices.
#[repr(C)]
#[derive(Copy, Clone)]
struct FreeLink {
    next_idx: u32,
    mac: u32,
}

/// Fixed-stride typed slab.
///
/// `T` is the value type; `B` is the underlying backing (any `Allocator`);
/// `M` is the freelist integrity policy (default [`NoProtection`], with
/// zero overhead). The slab takes one large allocation from `B` at
/// construction; individual `allocate` / `deallocate` calls return slots
/// from within that allocation in O(1).
///
/// # Usage discipline (read before unsafe-calling either trait method)
///
/// `Slab` issues raw `NonNull<u8>` pointers into typed-but-uninitialised
/// slots — it does **not** track value lifecycle. The caller is responsible
/// for the four following invariants:
///
/// 1. **Allocate-then-write**: the bytes inside the returned slice are
///    uninitialised. The caller must write a valid `T` (e.g. via
///    `core::ptr::write`) before reading.
/// 2. **Drop-before-deallocate**: `deallocate` overwrites the slot with a
///    a `FreeLink`. If `T: Drop`, the caller must run `T`'s destructor
///    (typically `core::ptr::drop_in_place::<T>(ptr.as_ptr().cast())`)
///    *before* calling `deallocate`. Failure to do so leaks resources owned
///    by `T` (file handles, heap allocations, locks) and may cause UB if
///    `T`'s drop is required for soundness.
/// 3. **Layout-must-fit-stride**: callers should request layouts whose
///    `size <= block_stride()` and `align <= max(align_of::<T>(),
///    align_of::<FreeLink>())`. Mis-sized requests fail at allocate;
///    mis-aligned ones may not match a slot index and trip a debug
///    assertion in `deallocate`.
/// 4. **Slab-drop-does-not-drop-Ts**: when the `Slab` itself drops, it
///    returns the underlying backing region to `B` but does *not* iterate
///    live slots to drop `T`. Callers responsible for any still-live `T`
///    must drain them before dropping the slab — e.g. via a higher-level
///    typed wrapper such as `GenerationalSlab`, which tracks per-slot
///    generations and runs `T::drop` for outstanding handles.
///
/// `Slab::allocate` returns a `NonNull<[u8]>` whose slice length is
/// [`block_stride()`](Self::block_stride), **not** the requested
/// `layout.size()`. Callers who care about the exact requested size must
/// remember it themselves; callers who want to use the extra padding bytes
/// (e.g. for footers / metadata) may write through the full stride window.
///
/// # Thread safety
///
/// `Send` if `T`, `B`, `M` are `Send`. `Sync`: NO. The free list head and
/// next-uncarved cursor live in `UnsafeCell`s so that `Allocator::allocate`
/// can take `&self`. Cross-thread deallocation uses the M8 `SlabRemote` —
/// not this type directly.
///
/// # API-misuse compile-failures (pinned)
///
/// `T` must not be a zero-sized type. The `ASSERT_T_NON_ZST` associated
/// const turns the previously runtime-only rejection
/// (`size_of::<T>() == 0` → `AllocError`) into a build error, so the
/// failure surfaces at the call site instead of after a successful build.
///
/// ```compile_fail
/// // FAILS TO COMPILE: ZST T is rejected by `Slab::ASSERT_T_NON_ZST`.
/// // The const_assert fires when `with_protection` is monomorphised, so
/// // the build halts before any test runs.
/// use forge_backing::InlineBacked;
/// use forge_layout::Slab;
/// let _: Slab<(), InlineBacked<128>> =
///     Slab::new(8, InlineBacked::<128>::new()).unwrap();
/// ```
pub struct Slab<T, B: Allocator + FixedRange, M: FreelistProtection = NoProtection> {
    backing: B,
    mac: M,
    /// Byte offset from `backing.base()` to the start of the slab's
    /// `capacity * block_stride` slot region.
    ///
    /// We deliberately do NOT store an absolute pointer. Backings whose
    /// `base()` is structure-relative (e.g. `InlineBacked<N>` returns
    /// `&self.storage`) report a DIFFERENT address before and after the
    /// backing has been moved. An absolute pointer captured at
    /// construction would then point at the backing's OLD location after
    /// the slab was returned from its constructor by value. Storing the
    /// offset and computing `self.backing.base().as_ptr().add(offset)`
    /// at each access keeps the slab move-safe.
    base_offset: usize,
    /// 1-based slot index; 0 = list empty.
    free_head: UnsafeCell<u32>,
    /// Index of the first slot never yet allocated (always carved from front).
    next_uncarved: UnsafeCell<u32>,
    block_stride: usize,
    /// `block_stride.trailing_zeros()` when `block_stride.is_power_of_two()`,
    /// `0` otherwise. `slot_index` uses this to replace runtime `/ stride`
    /// and `% stride` with `>> shift` and `& (stride - 1)` for the common
    /// pow2-stride case (every `T` whose size and align are powers of two
    /// with size ≥ size_of::<FreeLink>(), which is most real types). The
    /// sentinel `0` is safe because real strides are always ≥ size_of::<FreeLink>() = 8,
    /// so a true pow2 stride has shift ≥ 3.
    stride_shift: u32,
    capacity: u32,
    backing_layout: NonZeroLayout,
    /// Count of detected freelist corruption events (MAC verify
    /// failures + out-of-range `next_idx` defense-in-depth tripwires).
    /// Each event bumps this counter before the freelist is abandoned;
    /// the slab keeps serving allocations from `next_uncarved`. Exposed
    /// via [`Allocator::corruption_events`] for operator observability.
    ///
    /// **Width:** `AtomicUsize` (not `AtomicU64`) so this compiles on
    /// 32-bit bare-metal targets (Cortex-M3/M4, `thumbv7em-none-eabihf`)
    /// that lack native 64-bit atomic ops. The trait method
    /// [`Allocator::corruption_events`] still returns `u64`; the cast
    /// happens at the trait boundary. Practical impact: on 32-bit
    /// hosts the counter saturates at `u32::MAX ≈ 4.3 B` corruption
    /// events — overflow is irrelevant in any realistic timeframe
    /// (one event/ns ≈ 4.3 s, but real workloads see ≪1 event/year).
    corruption_events: AtomicUsize,
    _phantom: PhantomData<T>,
}

impl<T, B: Allocator + FixedRange> Slab<T, B, NoProtection> {
    /// Construct a slab with the default `NoProtection` policy.
    ///
    /// `capacity` is the number of `T` slots. Errors if the backing cannot
    /// supply the required region or if the total size overflows.
    pub fn new(capacity: usize, backing: B) -> Result<Self, AllocError> {
        Self::with_protection(capacity, backing, NoProtection)
    }
}

impl<T, B: Allocator + FixedRange, M: FreelistProtection> Slab<T, B, M> {
    /// Compile-time assertion that `T` is not a ZST.
    ///
    /// Forcing this associated const inside `with_protection` triggers a
    /// compile error when the slab is instantiated with a ZST `T`,
    /// promoting the previously runtime-only ZST rejection (`size_of::<T>()
    /// == 0` → `AllocError`) to a build-time error. This is purely
    /// additive: every `T` that was accepted before still compiles, and
    /// the runtime check below remains for backwards-compatibility and
    /// defense-in-depth against any future generic path that might bypass
    /// the const.
    const ASSERT_T_NON_ZST: () = assert!(
        size_of::<T>() > 0,
        "Slab<T, B, M>: T must not be a zero-sized type — a freelist over \
         zero-byte slots has no meaningful pointer arithmetic. See the \
         `compile_fail` doctest on `Slab` for the rejection example.",
    );

    /// Construct a slab with an explicit freelist-protection policy.
    ///
    /// `capacity` must be `> 0` and ≤ `u32::MAX` — the slab uses 32-bit slot
    /// indices internally. `T` must not be a ZST (a freelist over zero-sized
    /// slots has no meaning); this is now enforced at **compile time** via
    /// `ASSERT_T_NON_ZST` — instantiating `Slab<(), _, _>` is a build error
    /// rather than a runtime `AllocError`.
    pub fn with_protection(capacity: usize, backing: B, mac: M) -> Result<Self, AllocError> {
        // Force compile-time evaluation of the ZST check. If `T` is a ZST
        // the build fails here; otherwise the const is `()` and emits no
        // code.
        let _: () = Self::ASSERT_T_NON_ZST;
        // capacity == 0 makes the slab unusable.
        if capacity == 0 {
            return Err(AllocError);
        }
        // ZST T: belt-and-braces — `ASSERT_T_NON_ZST` already rejected
        // this at compile time, but a future generic path that somehow
        // bypasses the const should still produce an honest error rather
        // than dividing by zero downstream.
        if size_of::<T>() == 0 {
            return Err(AllocError);
        }
        // Slot indices fit in `u32`; reject overly large slabs up front.
        let cap_u32 = u32::try_from(capacity).map_err(|_| AllocError)?;

        // block_stride = max(size_of::<T>(), size_of::<FreeLink>()), then
        // round up to max(align_of::<T>(), align_of::<FreeLink>()) so each
        // slot is properly aligned for both views.
        let slot_align = core::cmp::max(align_of::<T>(), align_of::<FreeLink>());
        let raw_stride = core::cmp::max(size_of::<T>(), size_of::<FreeLink>());
        // Round `raw_stride` up to `slot_align` without risking overflow on
        // pathologically large `T`. `slot_align` is always a power of two so
        // the mask is correct.
        let block_stride = raw_stride
            .checked_add(slot_align - 1)
            .map(|v| v & !(slot_align - 1))
            .ok_or(AllocError)?;

        // Total bytes = capacity * block_stride.
        let total = block_stride.checked_mul(capacity).ok_or(AllocError)?;
        let backing_layout = NonZeroLayout::from_size_align(total, slot_align)
            .map_err(|_| AllocError)?;

        let block = backing.allocate(backing_layout)?;
        // Capture the OFFSET of the allocated region from `backing.base()`.
        // We need a stable identifier that survives the imminent move of
        // `backing` into `Self`; the offset is invariant under struct
        // moves (the relative layout inside the backing is fixed), while
        // an absolute `NonNull<u8>` captured here would point at the
        // backing's pre-move address and silently corrupt every later
        // access. See the struct-field comment on `base_offset`.
        let block_addr = block.cast::<u8>().as_ptr() as usize;
        let backing_base_addr = backing.base().as_ptr() as usize;
        // `block_addr >= backing_base_addr` always for a fresh
        // `backing.allocate(...)` whose backing implements `FixedRange`
        // honestly. If the backing returns a pointer outside its own
        // range, that's a backing bug, not ours — defend with a
        // checked subtraction so we surface it as `AllocError` rather
        // than producing a wrap-bounded offset that explodes later.
        let base_offset = block_addr
            .checked_sub(backing_base_addr)
            .ok_or(AllocError)?;

        // Pre-compute the pow2-stride shift; 0 is the "not pow2" sentinel
        // (real strides are always ≥ 8, so pow2 strides have shift ≥ 3).
        let stride_shift = if block_stride.is_power_of_two() {
            block_stride.trailing_zeros()
        } else {
            0
        };

        Ok(Self {
            backing,
            mac,
            base_offset,
            free_head: UnsafeCell::new(0),
            next_uncarved: UnsafeCell::new(0),
            block_stride,
            stride_shift,
            capacity: cap_u32,
            backing_layout,
            corruption_events: AtomicUsize::new(0),
            _phantom: PhantomData,
        })
    }

    /// Resolve the slab's base pointer from the (current) backing
    /// location plus the captured offset. Recomputing every call keeps
    /// us safe against moves of the slab between construction and use.
    #[inline]
    fn base_ptr(&self) -> NonNull<u8> {
        // SAFETY: `backing.base()` is the start of the backing's region;
        // `base_offset` is `<= backing.size() - capacity*block_stride`
        // (the backing.allocate at construction reserved that range).
        // The resulting pointer is non-null because backing.base() is non-null.
        unsafe {
            NonNull::new_unchecked(
                self.backing.base().as_ptr().add(self.base_offset),
            )
        }
    }

    /// Number of slots in this slab.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    /// Bytes per slot (≥ `size_of::<T>()`).
    #[inline]
    pub fn block_stride(&self) -> usize {
        self.block_stride
    }

    /// Borrow the underlying backing.
    #[inline]
    pub fn backing(&self) -> &B {
        &self.backing
    }

    /// Pointer to slot `idx` (0-based). No bounds check — internal helper.
    #[inline]
    fn slot_ptr(&self, idx: u32) -> *mut u8 {
        // SAFETY: base + idx*stride is in-range when idx < capacity. Callers
        // verify the bound.
        unsafe { self.base_ptr().as_ptr().add(idx as usize * self.block_stride) }
    }

    /// 0-based slot index for `ptr`, or `None` if it's not aligned to a slot
    /// boundary or out of range.
    #[inline]
    fn slot_index(&self, ptr: NonNull<u8>) -> Option<u32> {
        let p = ptr.as_ptr() as usize;
        let base = self.base_ptr().as_ptr() as usize;
        if p < base {
            return None;
        }
        let offset = p - base;
        // Pow2-stride fast path: replace `/ stride` and `% stride` with
        // `>> shift` and `& (stride - 1)`. On x86-64 this removes a 20-40
        // cycle integer divide from every deallocate when T's stride is a
        // power of two (the common case: any T whose `size` and `align`
        // are both powers of two and `size >= 8`).
        let (idx, rem) = if self.stride_shift != 0 {
            let mask = self.block_stride - 1;
            (offset >> self.stride_shift, offset & mask)
        } else {
            (offset / self.block_stride, offset % self.block_stride)
        };
        if rem != 0 {
            return None;
        }
        if idx >= self.capacity as usize {
            return None;
        }
        // `idx < self.capacity` and `self.capacity: u32`, so `idx` always
        // fits in `u32`. Use `try_from` to make that explicit and defend
        // against future capacity-type changes.
        u32::try_from(idx).ok()
    }
}

unsafe impl<T, B: Allocator + FixedRange, M: FreelistProtection> Deallocator for Slab<T, B, M> {
    /// Push the slot identified by `ptr` onto the freelist.
    ///
    /// # Safety
    ///
    /// Per the [`Deallocator`] contract, `ptr` must have been returned by a
    /// previous call to `self.allocate(layout)`. Specifically:
    ///
    /// - `ptr` must lie at the base of a slot in this slab (not an offset
    ///   within a slot, not a pointer from another slab or allocator).
    /// - The caller is responsible for running `T`'s destructor (e.g. via
    ///   `core::ptr::drop_in_place`) before calling `deallocate`. This method
    ///   overwrites the slot's bytes with a a `FreeLink`.
    /// - Passing the same `ptr` twice without an intervening `allocate` is a
    ///   double-free and is UB.
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: NonZeroLayout) {
        // Layout sanity: an honest caller's layout fits within block_stride.
        // Release builds skip the check (per contract this is UB anyway).
        debug_assert!(
            _layout.size().get() <= self.block_stride
                && _layout.align().get() <= align_of::<T>().max(align_of::<FreeLink>()),
            "Slab::deallocate: layout exceeds slot stride or alignment",
        );

        // Resolve the slot index. In a properly used Slab the index is valid;
        // an out-of-range pointer is UB (per the trait contract). Debug
        // builds catch it; release builds skip.
        let idx = match self.slot_index(ptr) {
            Some(i) => i,
            None => {
                debug_assert!(false, "Slab::deallocate: pointer outside slab range");
                return;
            }
        };

        // Push slot onto the free list.
        // SAFETY: !Sync — no concurrent access to free_head.
        unsafe {
            let head_ptr = self.free_head.get();
            let old_head = *head_ptr; // 1-based; 0 means empty
            let mac = self.mac.sign(old_head, ptr.as_ptr() as usize);
            let link = FreeLink {
                next_idx: old_head,
                mac,
            };
            // Write the FreeLink into the slot's memory.
            //
            // Stacked Borrows: we MUST NOT write through the user-supplied
            // `ptr` directly. `ptr`'s provenance was derived from the
            // backing's `SharedReadWrite` tag at allocate time, and an
            // outer wrapper (e.g. `Quarantine::drop`, `SlabOwner::drop`,
            // `PoisonOnFree::drop`) may have taken a `&mut self` covering
            // the whole composition — that Unique retag invalidates the
            // older SharedReadWrite tag in the borrow stack. Writing
            // through the stale tag is then UB.
            //
            // The fix is to re-derive the slot pointer through `&self`:
            // `self.slot_ptr(idx)` calls `self.base_ptr()` which calls
            // `self.backing.base()`, each of which traverses fresh shared
            // reborrows. The resulting pointer sits at the top of the
            // borrow stack and is valid even after the outer Unique
            // retag. (Miri pass #7 caught the original bug across
            // SlabOwner / Quarantine / PoisonOnFree / etc.)
            let slot_ptr = self.slot_ptr(idx);
            slot_ptr.cast::<FreeLink>().write(link);
            *head_ptr = idx + 1; // store 1-based
        }
    }
}

unsafe impl<T, B: Allocator + FixedRange, M: FreelistProtection> Allocator for Slab<T, B, M> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        // Reject layouts the slab cannot satisfy.
        let req_align = layout.align().get();
        let req_size = layout.size().get();
        if req_align > align_of::<T>().max(align_of::<FreeLink>()) {
            return Err(AllocError);
        }
        if req_size > self.block_stride {
            return Err(AllocError);
        }

        // SAFETY: !Sync — no concurrent access to free_head or next_uncarved.
        unsafe {
            // Try to pop from the free list first.
            let head_ptr = self.free_head.get();
            let head = *head_ptr; // 1-based; 0 = empty
            if head != 0 {
                // head - 1 is the slot index.
                let slot_idx = head - 1;
                let slot = self.slot_ptr(slot_idx);
                let link = slot.cast::<FreeLink>().read();
                // Verify MAC. On corruption, drop the link (don't propagate)
                // and fall through to next_uncarved — defense-in-depth, the
                // attacker's poisoned link is now disarmed.
                let mac_ok = self
                    .mac
                    .verify(link.next_idx, link.mac, slot as usize)
                    .is_ok();
                // Defense-in-depth: even with `NoProtection` (or a future MAC
                // impl with a bug), reject a next_idx that would cause OOB
                // slot_ptr arithmetic. A valid free-list entry can only point
                // to a slot index in `0..capacity` (we store `idx+1` as 1-based
                // and the slab is single-threaded). `next_idx > capacity` ⇒
                // either corruption or an out-of-spec foreign write — treat
                // the same as MAC failure.
                if mac_ok && link.next_idx <= self.capacity {
                    *head_ptr = link.next_idx;
                    return Ok(NonNull::slice_from_raw_parts(
                        NonNull::new_unchecked(slot),
                        self.block_stride,
                    ));
                } else {
                    // Record the corruption event BEFORE the debug_assert
                    // so the counter reflects the detection regardless of
                    // build profile: in release the assert is compiled out
                    // and only the counter remains; in debug the counter
                    // is updated and then the assert panics — but the
                    // counter increment is already visible to any panic
                    // handler (or `catch_unwind`-wrapped test) that
                    // inspects `corruption_events` post-detection. The
                    // event is the FIRST observable sign of an in-progress
                    // attack — silent disarm without the counter leaves
                    // operators blind. Ordering of the two statements
                    // matches `ExtendableSlab::deallocate` and
                    // `UntypedSlab::{allocate_slot, free_slot}` so the
                    // debug/release semantics agree across all
                    // corruption-detect sites.
                    //
                    // `Relaxed` is correct: the counter is advisory,
                    // monotonically increasing, and read eventually-
                    // consistently — no other state synchronizes against
                    // it.
                    self.corruption_events.fetch_add(1, Ordering::Relaxed);
                    debug_assert!(
                        false,
                        "Slab freelist corruption: mac_ok={mac_ok}, next_idx={}, capacity={}",
                        link.next_idx, self.capacity,
                    );
                    // Abandon the free list to prevent following corrupted
                    // links; force fresh allocation from next_uncarved.
                    *head_ptr = 0;
                }
            }
            // Carve from next_uncarved.
            let nxt_ptr = self.next_uncarved.get();
            let nxt = *nxt_ptr;
            if nxt >= self.capacity {
                return Err(AllocError);
            }
            let slot = self.slot_ptr(nxt);
            *nxt_ptr = nxt + 1;
            Ok(NonNull::slice_from_raw_parts(
                NonNull::new_unchecked(slot),
                self.block_stride,
            ))
        }
    }

    #[inline]
    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.capacity as usize * self.block_stride)
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        // Cast `usize → u64` at the trait boundary so the public API
        // stays uniform across 32-bit and 64-bit targets. Lossless: on
        // 32-bit hosts the inner counter is u32 (≤ u32::MAX ≈ 4.3 B);
        // on 64-bit hosts it is u64.
        self.corruption_events.load(Ordering::Relaxed) as u64
    }
}

impl<T, B: Allocator + FixedRange, M: FreelistProtection> FixedRange for Slab<T, B, M> {
    #[inline]
    fn base(&self) -> NonNull<u8> {
        self.base_ptr()
    }

    #[inline]
    fn size(&self) -> usize {
        self.capacity as usize * self.block_stride
    }
}

impl<T, B: Allocator + FixedRange, M: FreelistProtection> Drop for Slab<T, B, M> {
    fn drop(&mut self) {
        // Debug-only sanity check: walk the freelist and verify every
        // carved slot has been returned. A mismatch means the caller
        // dropped the slab with live allocations outstanding — for
        // `T: Drop` the destructor never runs, which is a real leak of
        // resources owned by `T` (heap allocations inside `T`, file
        // handles, locks, etc.). For `T: Copy` (or any `!Drop` type),
        // the only loss is the un-reclaimed slot index, which is fine
        // because the backing region drops on the next line anyway.
        // We therefore skip the check when `T: !Drop` so existing test
        // patterns (allocate-then-drop-slab on `u64`-style payloads)
        // continue to compile cleanly.
        //
        // Walks freelist links by `next_idx` only (no MAC verification);
        // a corrupted chain would either loop or land on an out-of-range
        // index, both of which we detect explicitly.
        //
        // We compute the imbalance BEFORE returning the backing region so
        // that — even if the eventual `debug_assert!` panics — the backing
        // chunk is still released. Without this ordering an assertion-on-
        // leak would *itself* leak the backing region (the asserting drop
        // unwinds past the deallocate call), upgrading the bug we wanted
        // to catch into a strictly worse leak.
        //
        // **Drop-during-unwind escalation**: the `debug_assert!` below
        // only fires in debug builds, and only when the caller failed to
        // free all live slots before drop. If the slab is being dropped
        // as part of an in-flight panic-unwind AND a slot is leaked AND
        // we are in a debug build, the assertion's panic-while-panicking
        // triggers an **immediate process abort** (Rust language rule).
        // Release builds never assert and never abort here. Treat the
        // debug-only abort as a louder version of the leak-detection
        // signal it already is — not as a regression. The condition is
        // a caller bug (live slots at slab drop); the abort makes the
        // bug impossible to ignore.
        #[cfg(debug_assertions)]
        let imbalance: Option<(u32, u32)> = if core::mem::needs_drop::<T>() {
            // SAFETY: &mut self — exclusive access; the cells are owned.
            let next_uncarved = unsafe { *self.next_uncarved.get() };
            let mut head = unsafe { *self.free_head.get() };
            let mut freelist_len: u32 = 0;
            // Bound the walk to capacity to defend against a corrupted
            // cycle (would otherwise loop forever).
            while head != 0 && freelist_len <= self.capacity {
                let slot_idx = head - 1;
                if slot_idx >= self.capacity {
                    // Corrupted index — abandon the count; surface as a
                    // softer assertion below.
                    break;
                }
                let slot_ptr = self.slot_ptr(slot_idx);
                // SAFETY: slot holds a FreeLink (we put it there in deallocate).
                let link = unsafe { slot_ptr.cast::<FreeLink>().read() };
                head = link.next_idx;
                freelist_len += 1;
            }
            if freelist_len == next_uncarved {
                None
            } else {
                Some((next_uncarved, freelist_len))
            }
        } else {
            None
        };
        // SAFETY: base and backing_layout came from a single backing.allocate
        // call in `with_protection`. We hold the only path to either field
        // (no Clone impl, no exposed mutator). Run the deallocate BEFORE the
        // assertion so a leaked-T panic does not also leak the backing.
        // `base_ptr()` recomputes from the (post-move-safe) backing.base()
        // and the captured offset — same address the construction site
        // recorded into `base_offset`.
        unsafe { self.backing.deallocate(self.base_ptr(), self.backing_layout) };
        #[cfg(debug_assertions)]
        if let Some((next_uncarved, freelist_len)) = imbalance {
            debug_assert!(
                false,
                "Slab dropped with {} live slot(s) (carved={}, freelist={}). \
                 Caller failed to deallocate all outstanding `T`s before drop — \
                 any T: Drop on those slots was leaked.",
                next_uncarved - freelist_len,
                next_uncarved,
                freelist_len,
            );
        }
    }
}

// Send if all components are Send. !Sync via UnsafeCell.
unsafe impl<T, B, M> Send for Slab<T, B, M>
where
    T: Send,
    B: Allocator + FixedRange + Send,
    M: FreelistProtection + Send,
{
}

// ============================================================================
// Kani proof harnesses (spec M13)
//
// These prove correctness properties of the freelist push/pop and slot-index
// recovery logic on a tiny slab. Kani enumerates all input combinations
// symbolically; the proofs run under `cargo kani` only and are invisible to
// stable builds.
// ============================================================================

// Kani proofs depend on `forge_backing::InlineBacked`; forge-backing is gated
// behind the `std` feature in this crate (see Cargo.toml), so the proof
// module must be gated similarly. Kani CI must run with the `std`
// feature enabled for these proofs to compile.
#[cfg(all(kani, feature = "std"))]
mod kani_proofs {
    use super::*;
    use forge_backing::InlineBacked;

    /// Allocate-then-deallocate-then-allocate returns the SAME slot
    /// pointer. This is the LIFO property of the freelist push/pop.
    #[kani::proof]
    #[kani::unwind(3)]
    fn alloc_dealloc_alloc_returns_same_slot() {
        let s: Slab<u64, InlineBacked<512>, NoProtection> =
            Slab::new(8, InlineBacked::<512>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = s.allocate(layout).unwrap().cast::<u8>();
        let a_ptr = a.as_ptr();
        unsafe { s.deallocate(a, layout) };
        let b = s.allocate(layout).unwrap().cast::<u8>();
        assert!(b.as_ptr() == a_ptr);
    }

    /// Two distinct live allocations never overlap. (Single-step
    /// version — full coverage of N allocations would need a loop
    /// Kani can unwind.)
    #[kani::proof]
    #[kani::unwind(3)]
    fn two_allocs_never_overlap() {
        let s: Slab<u64, InlineBacked<512>, NoProtection> =
            Slab::new(8, InlineBacked::<512>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = s.allocate(layout).unwrap().cast::<u8>();
        let b = s.allocate(layout).unwrap().cast::<u8>();
        assert!(a.as_ptr() != b.as_ptr());
    }

    /// Slot-index recovery from a returned pointer round-trips
    /// correctly: the index computed from the pointer matches the
    /// slot the allocator just carved out.
    #[kani::proof]
    #[kani::unwind(3)]
    fn slot_index_round_trips() {
        let s: Slab<u64, InlineBacked<512>, NoProtection> =
            Slab::new(8, InlineBacked::<512>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = s.allocate(layout).unwrap().cast::<u8>();
        let idx = s.slot_index(a).expect("slot index must resolve");
        // idx is 0-based; first carved slot is index 0.
        assert!(idx == 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_backing::InlineBacked;

    /// Test struct to exercise the slab.
    #[derive(Debug, PartialEq)]
    struct Foo(u64);

    /// A backing big enough to hold many Foo slots.
    /// Foo is 8 bytes, FreeLink is 8 bytes, so block_stride = 8.
    /// 128 slots × 8 bytes = 1024 bytes — fits in InlineBacked<1024>.
    fn make_slab() -> Slab<Foo, InlineBacked<1024>, NoProtection> {
        Slab::new(128, InlineBacked::<1024>::new()).unwrap()
    }

    #[test]
    fn capacity_matches() {
        let s = make_slab();
        assert_eq!(s.capacity(), 128);
        assert_eq!(s.block_stride(), 8);
        assert_eq!(s.capacity_bytes(), Some(1024));
    }

    #[test]
    fn allocate_returns_distinct_slots() {
        let s = make_slab();
        let layout = NonZeroLayout::for_type::<Foo>().unwrap();
        let a = s.allocate(layout).unwrap();
        let b = s.allocate(layout).unwrap();
        assert_ne!(a.cast::<u8>().as_ptr(), b.cast::<u8>().as_ptr());
        // Slots are stride-apart.
        assert_eq!(
            (b.cast::<u8>().as_ptr() as usize) - (a.cast::<u8>().as_ptr() as usize),
            8
        );
    }

    #[test]
    fn allocate_then_deallocate_reuses_slot() {
        let s = make_slab();
        let layout = NonZeroLayout::for_type::<Foo>().unwrap();
        let a = s.allocate(layout).unwrap();
        let a_addr = a.cast::<u8>().as_ptr();
        unsafe { s.deallocate(a.cast(), layout) };
        let b = s.allocate(layout).unwrap();
        // LIFO — the just-freed slot should come back.
        assert_eq!(a_addr, b.cast::<u8>().as_ptr());
    }

    #[test]
    fn allocate_exhausts_capacity() {
        let s: Slab<u64, InlineBacked<64>, NoProtection> =
            Slab::new(8, InlineBacked::<64>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        for _ in 0..8 {
            assert!(s.allocate(layout).is_ok());
        }
        assert!(s.allocate(layout).is_err());
    }

    #[test]
    fn allocate_rejects_oversized_layout() {
        let s = make_slab();
        let too_big = NonZeroLayout::from_size_align(64, 8).unwrap();
        assert!(s.allocate(too_big).is_err());
    }

    #[test]
    fn allocate_rejects_overaligned_layout() {
        let s = make_slab();
        let over_aligned = NonZeroLayout::from_size_align(8, 64).unwrap();
        assert!(s.allocate(over_aligned).is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn alloc_dealloc_alloc_round_trip_many() {
        let s: Slab<u64, InlineBacked<1024>, NoProtection> =
            Slab::new(128, InlineBacked::<1024>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Allocate a bunch, free a bunch, re-allocate — all the freed slots
        // come back from the freelist in LIFO order.
        let mut ptrs = Vec::new();
        for _ in 0..64 {
            ptrs.push(s.allocate(layout).unwrap());
        }
        // Free in reverse order.
        for p in ptrs.iter().rev() {
            unsafe { s.deallocate(p.cast(), layout) };
        }
        // Re-allocate — should get them back in the original order (LIFO).
        for p in ptrs.iter() {
            let b = s.allocate(layout).unwrap();
            assert_eq!(p.cast::<u8>().as_ptr(), b.cast::<u8>().as_ptr());
        }
    }

    #[test]
    fn pow2_stride_shift_is_set() {
        // u64 stride is 8 (pow2) — shift must be 3.
        let s: Slab<u64, InlineBacked<64>, NoProtection> =
            Slab::new(8, InlineBacked::<64>::new()).unwrap();
        assert_eq!(s.block_stride, 8);
        assert_eq!(s.stride_shift, 3);
    }

    #[cfg(feature = "std")]
    #[test]
    fn non_pow2_stride_shift_is_zero_sentinel() {
        // String is 24 bytes on 64-bit — not a power of two. Shift sentinel
        // is 0, forcing the slow div/mod path in slot_index.
        let s: Slab<String, InlineBacked<256>, NoProtection> =
            Slab::new(8, InlineBacked::<256>::new()).unwrap();
        assert_eq!(s.block_stride, 24);
        assert_eq!(s.stride_shift, 0);
        // Round-trip verifies both alloc and dealloc handle the non-pow2 path.
        let layout = NonZeroLayout::for_type::<String>().unwrap();
        let a = s.allocate(layout).unwrap();
        unsafe { s.deallocate(a.cast(), layout) };
        // After dealloc, the slot is back on the freelist — the next alloc
        // must return the same pointer (LIFO).
        let b = s.allocate(layout).unwrap();
        assert_eq!(a.cast::<u8>().as_ptr(), b.cast::<u8>().as_ptr());
        // Balance the alloc/dealloc so Slab's debug-only leak check passes
        // on drop (we never wrote a String into the slot, so it's safe to
        // free without drop_in_place).
        unsafe { s.deallocate(b.cast(), layout) };
    }

    #[test]
    fn fixed_range_contains_slots() {
        let s = make_slab();
        let layout = NonZeroLayout::for_type::<Foo>().unwrap();
        let a = s.allocate(layout).unwrap();
        assert!(s.contains(a.cast::<u8>()));
    }

    #[cfg(feature = "std")]
    #[test]
    fn slab_with_string_payload() {
        // A larger T to verify block_stride > FreeLink size.
        // String is ~24 bytes on 64-bit.
        let s: Slab<String, InlineBacked<256>, NoProtection> =
            Slab::new(8, InlineBacked::<256>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<String>().unwrap();
        let a = s.allocate(layout).unwrap();
        unsafe {
            a.cast::<String>().as_ptr().write("hello".to_string());
            // Then drop and free; we never re-read so this is safe.
            core::ptr::drop_in_place(a.cast::<String>().as_ptr());
            s.deallocate(a.cast(), layout);
        }
    }

    /// Boundary: `Slab::new(0, _)` must fail — a zero-capacity slab is
    /// useless and would underflow the index math at every `next_uncarved`
    /// check (we use `>=` so 0 >= 0 correctly fails alloc, but rejecting at
    /// construction is the documented contract).
    #[test]
    fn rejects_zero_capacity() {
        let r = Slab::<u64, InlineBacked<64>, NoProtection>::new(0, InlineBacked::<64>::new());
        assert!(r.is_err());
    }

    // Note: the previous runtime test `rejects_zst_payload` constructed
    // `Slab::<(), InlineBacked<64>, NoProtection>::new(...)` and expected
    // `Err(AllocError)`. Pass #8 promoted that rejection to a
    // compile-time const_assert (`Slab::ASSERT_T_NON_ZST`), so the
    // misuse can no longer be expressed as a runtime test — it would
    // fail to compile at every call site. The equivalent pin lives as a
    // `compile_fail` doctest on the `Slab` type's docs (line 98).

    /// `capacity = usize::MAX` triggers either the u32 conversion guard or
    /// the `block_stride * capacity` overflow guard — never panics.
    #[test]
    fn rejects_usize_max_capacity() {
        let r = Slab::<u64, InlineBacked<64>, NoProtection>::new(usize::MAX, InlineBacked::<64>::new());
        assert!(r.is_err());
    }

    /// `T` whose `align_of` is 1 and `size_of` is 1 (e.g. `u8`) — stride
    /// must round up to `size_of::<FreeLink>() = 8` so a freelist link
    /// fits in the slot.
    #[test]
    fn stride_for_u8_payload_rounds_up_to_freelink_size() {
        let s: Slab<u8, InlineBacked<128>, NoProtection> =
            Slab::new(8, InlineBacked::<128>::new()).unwrap();
        assert_eq!(s.block_stride(), 8, "u8 stride must round up to FreeLink size");
        // And round-trip an allocation to confirm the freelist can store
        // a FreeLink in the u8-sized slot.
        let layout = NonZeroLayout::for_type::<u8>().unwrap();
        let a = s.allocate(layout).unwrap();
        unsafe { s.deallocate(a.cast(), layout) };
        let b = s.allocate(layout).unwrap();
        assert_eq!(a.cast::<u8>().as_ptr(), b.cast::<u8>().as_ptr());
        unsafe { s.deallocate(b.cast(), layout) };
    }

    /// Allocate exactly `capacity` slots; the `capacity+1`-th allocate
    /// must return `AllocError` (next_uncarved exhaustion).
    #[test]
    fn allocate_capacity_plus_one_returns_err() {
        const CAP: usize = 4;
        let s: Slab<u64, InlineBacked<64>, NoProtection> =
            Slab::new(CAP, InlineBacked::<64>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // capacity allocs all succeed.
        for i in 0..CAP {
            assert!(s.allocate(layout).is_ok(), "slot {i} should succeed");
        }
        // The (CAP+1)-th must fail.
        assert!(
            s.allocate(layout).is_err(),
            "alloc past capacity must return AllocError",
        );
    }

    /// Boundary: a release-build double-free is documented UB, but the
    /// debug build's `debug_assert!` on stride alignment catches a
    /// pointer pulled from a freelist-link slot's `next_idx` byte (which
    /// is NOT on a stride boundary).
    ///
    /// Direct verification: deallocate the same slot twice without
    /// `debug_assertions` would corrupt; in debug, the slot's `next_idx`
    /// loops onto itself in the freelist and the next allocate either
    /// returns the same slot OR loops via the defense-in-depth
    /// `next_idx > capacity` rejection. We can't assert UB safely, but
    /// we can check that a single allocate after a (legitimate) free
    /// returns the LIFO-correct slot.
    #[test]
    fn lifo_property_holds_after_alloc_dealloc_realloc() {
        let s: Slab<u64, InlineBacked<128>, NoProtection> =
            Slab::new(16, InlineBacked::<128>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Regression: slab.base() must agree with the slab's owned backing's
        // base. A prior bug stored an absolute pointer captured BEFORE the
        // backing moved into Self, leaving `slab.base` pointing at the old
        // location of the InlineBacked's storage. That pointer was stale
        // for the rest of the slab's life and writes through it landed
        // in someone else's stack frame.
        use forge_core::FixedRange;
        let backing_storage = s.backing().base();
        assert_eq!(
            s.base().as_ptr(),
            backing_storage.as_ptr(),
            "Slab base pointer must agree with current backing.base() — \
             stale-pointer bug if not",
        );
        // Alloc 3, free in reverse, re-alloc in order. Each re-alloc
        // returns the most-recently-freed slot.
        let a = s.allocate(layout).unwrap();
        let b = s.allocate(layout).unwrap();
        let c = s.allocate(layout).unwrap();
        let a_addr = a.cast::<u8>().as_ptr();
        let b_addr = b.cast::<u8>().as_ptr();
        let c_addr = c.cast::<u8>().as_ptr();
        unsafe {
            s.deallocate(a.cast(), layout);
            s.deallocate(b.cast(), layout);
            s.deallocate(c.cast(), layout);
        }
        // Free order: a, b, c — so head is c. LIFO: alloc returns c, b, a.
        let r1 = s.allocate(layout).unwrap().cast::<u8>().as_ptr();
        let r2 = s.allocate(layout).unwrap().cast::<u8>().as_ptr();
        let r3 = s.allocate(layout).unwrap().cast::<u8>().as_ptr();
        assert_eq!(r1, c_addr);
        assert_eq!(r2, b_addr);
        assert_eq!(r3, a_addr);
    }

    #[cfg(feature = "siphasher")]
    #[test]
    fn siphash_protected_slab_round_trips() {
        use forge_core::SipHashMAC;
        let s: Slab<u64, InlineBacked<1024>, SipHashMAC> = Slab::with_protection(
            128,
            InlineBacked::<1024>::new(),
            SipHashMAC::with_key([0x42; 16]),
        )
        .unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = s.allocate(layout).unwrap();
        unsafe { s.deallocate(a.cast(), layout) };
        let b = s.allocate(layout).unwrap();
        assert_eq!(a.cast::<u8>().as_ptr(), b.cast::<u8>().as_ptr());
    }
}
