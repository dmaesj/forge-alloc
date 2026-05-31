//! `SizeClassed<B, CLASSES>` — array of `CLASSES` untyped slabs with
//! geometrically increasing block sizes. Routes each allocate request
//! to the smallest slab whose stride satisfies the request; oversized
//! requests fall through to the backing.
//!
//! See `docs/ARCHITECTURE.md` for the composable-allocator design.

use core::cell::UnsafeCell;
use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use forge_alloc_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

/// Default class sizes (bytes): 8, 16, 32, 64, 128, 256, 512, 1024.
pub const DEFAULT_CLASS_SIZES_8: [usize; 8] = [8, 16, 32, 64, 128, 256, 512, 1024];

/// Free-list link stored inside a free slot. Same layout as `Slab`'s
/// internal `FreeLink` (1-based indices + zero sentinel) so the
/// algorithm is identical; we duplicate it here rather than expose it
/// from the typed slab to keep crate-internal types separate.
#[repr(C)]
#[derive(Copy, Clone)]
struct FreeLink {
    next_idx: u32,
}

/// Erased `Slab<u8>` with a runtime block stride. Internal — wrap with
/// `SizeClassed` for the public API.
///
/// Stores a **byte offset** from the backing's `base()` rather than an
/// absolute pointer. This mirrors the stale-base-pointer fix applied to
/// `Slab`, `BumpArena`, `SharedBumpArena`, and `StackAlloc`: an
/// absolute pointer cached at construction time would point at the
/// backing's *pre-move* location forever once `SizeClassed` is returned
/// by value, silently corrupting any subsequent `allocate`/`deallocate`
/// on a structure-relative backing such as `InlineBacked<N>`. By
/// storing the offset and resolving `backing_base + offset` at each
/// access, the slab follows the backing wherever it lives.
struct UntypedSlab {
    base_offset: usize,
    block_stride: usize,
    capacity: u32,
    /// 1-based slot index; `0` = freelist empty.
    free_head: UnsafeCell<u32>,
    /// 0-based index of the first slot never yet carved.
    next_uncarved: UnsafeCell<u32>,
    /// Detected corruption events at this class: `next_idx > capacity`
    /// defense-in-depth tripwire on the freelist, plus `slot_index`-
    /// returns-None caller-contract violations on `free_slot`. Exposed
    /// via `SizeClassed::corruption_events`.
    ///
    /// `AtomicUsize` (not `AtomicU64`) for 32-bit bare-metal portability;
    /// see `Slab::corruption_events` for the rationale and overflow
    /// analysis.
    corruption_events: AtomicUsize,
}

impl UntypedSlab {
    /// Construct over a pre-allocated region of `capacity * block_stride`
    /// bytes starting at `backing_base + base_offset`. The caller is
    /// responsible for ensuring the region is at least that large and
    /// stays valid for the slab's lifetime.
    fn new(base_offset: usize, block_stride: usize, capacity: u32) -> Self {
        Self {
            base_offset,
            block_stride,
            capacity,
            free_head: UnsafeCell::new(0),
            next_uncarved: UnsafeCell::new(0),
            corruption_events: AtomicUsize::new(0),
        }
    }

    /// Resolve the slab's current absolute base from the live backing
    /// base pointer. Per-call recompute defeats the pre-move-cached-
    /// pointer UAF described in the stale-base-pointer design note.
    #[inline]
    fn base_ptr(&self, backing_base: NonNull<u8>) -> NonNull<u8> {
        // SAFETY: at construction we placed the region inside
        // [backing_base, backing_base + backing.size()) and segments are
        // never reallocated; `add(base_offset)` therefore lands inside
        // the same allocation.
        unsafe { NonNull::new_unchecked(backing_base.as_ptr().add(self.base_offset)) }
    }

    #[inline]
    fn slot_ptr(&self, backing_base: NonNull<u8>, idx: u32) -> *mut u8 {
        // SAFETY: caller verifies idx < capacity.
        unsafe {
            self.base_ptr(backing_base)
                .as_ptr()
                .add(idx as usize * self.block_stride)
        }
    }

    /// Compute the 0-based slot index for `ptr`, or `None` if `ptr` is
    /// not aligned to a slot boundary or out of range.
    fn slot_index(&self, backing_base: NonNull<u8>, ptr: NonNull<u8>) -> Option<u32> {
        let p = ptr.as_ptr() as usize;
        let base = self.base_ptr(backing_base).as_ptr() as usize;
        if p < base {
            return None;
        }
        let offset = p - base;
        if offset % self.block_stride != 0 {
            return None;
        }
        let idx = offset / self.block_stride;
        if idx >= self.capacity as usize {
            return None;
        }
        u32::try_from(idx).ok()
    }

    /// `Some(ptr)` if a slot was popped; `None` if both the freelist and
    /// the uncarved region are exhausted.
    ///
    /// # Safety
    ///
    /// Caller must be the unique accessor of this slab (the wrapping
    /// `SizeClassed` is `!Sync`).
    unsafe fn allocate_slot(&self, backing_base: NonNull<u8>) -> Option<NonNull<u8>> {
        // SAFETY: !Sync — single-threaded access.
        let head = unsafe { *self.free_head.get() };
        if head != 0 {
            // Freelist non-empty — pop.
            let slot_idx = head - 1;
            // SAFETY: head was a valid 1-based slot index by construction.
            let slot_ptr = self.slot_ptr(backing_base, slot_idx);
            // SAFETY: slot holds a FreeLink because allocate_slot/free_slot
            // are the only writers and they only put FreeLink there when
            // pushing onto the free list. Aligned read is sound because
            // `block_stride` is a power of two >= size_of::<FreeLink>() = 4
            // and base came from `backing.allocate(layout)` with align ==
            // stride, so every slot_ptr satisfies align_of::<FreeLink>().
            let link = unsafe { slot_ptr.cast::<FreeLink>().read() };
            // Defense-in-depth: UntypedSlab has no MAC, so a corrupted
            // next_idx would route the next pop to an arbitrary slot index.
            // Reject `next_idx > capacity` and abandon the freelist — the
            // wider class-region bound limits blast radius, and carving
            // fresh from `next_uncarved` lets the slab continue safely.
            if link.next_idx > self.capacity {
                // Record BEFORE the debug_assert so the counter reflects
                // the detection in both debug and release builds. See
                // `Slab::allocate` for the full rationale. `Relaxed` is
                // correct: advisory observability counter, no other
                // state synchronizes against it.
                self.corruption_events.fetch_add(1, Ordering::Relaxed);
                debug_assert!(
                    false,
                    "UntypedSlab freelist corruption: next_idx={} > capacity={}",
                    link.next_idx, self.capacity,
                );
                unsafe { *self.free_head.get() = 0 };
                // Fall through to next_uncarved carve below.
            } else {
                unsafe { *self.free_head.get() = link.next_idx };
                return Some(unsafe { NonNull::new_unchecked(slot_ptr) });
            }
        }
        // Freelist empty — try to carve from the uncarved region.
        let nuc = unsafe { *self.next_uncarved.get() };
        if nuc >= self.capacity {
            return None;
        }
        let p = self.slot_ptr(backing_base, nuc);
        unsafe { *self.next_uncarved.get() = nuc + 1 };
        Some(unsafe { NonNull::new_unchecked(p) })
    }

    /// # Safety
    ///
    /// `ptr` must have been returned by a previous call to
    /// [`allocate_slot`](Self::allocate_slot) on *this* slab. Caller is
    /// the unique accessor (wrapped by `!Sync` `SizeClassed`).
    unsafe fn free_slot(&self, backing_base: NonNull<u8>, ptr: NonNull<u8>) {
        let Some(idx) = self.slot_index(backing_base, ptr) else {
            // Record BEFORE the debug_assert so the counter reflects the
            // detection in both debug and release builds. See
            // `Slab::allocate` for the full rationale. `Relaxed` is
            // correct: advisory observability counter.
            self.corruption_events.fetch_add(1, Ordering::Relaxed);
            debug_assert!(false, "UntypedSlab::free_slot: pointer outside range");
            return;
        };
        // SAFETY: !Sync.
        let old_head = unsafe { *self.free_head.get() };
        // SAFETY: slot is at least `block_stride` bytes (>= size_of::<FreeLink>()
        // = 4) and slot_ptr is `block_stride`-aligned, which is also a
        // multiple of align_of::<FreeLink>() = 4. Aligned write is sound.
        unsafe {
            ptr.as_ptr()
                .cast::<FreeLink>()
                .write(FreeLink { next_idx: old_head });
            *self.free_head.get() = idx + 1;
        };
    }
}

/// SizeClassed allocator.
///
/// `CLASSES` size classes are configured at construction. An allocate
/// request is routed to the smallest class whose stride is `>= size`
/// AND whose stride is a multiple of the requested alignment (any
/// class with `stride < align` is skipped). Oversized or
/// over-aligned requests fall through to `backing.allocate(layout)`.
///
/// Deallocation routes by provenance: if `ptr` falls inside one of
/// the class regions, it's a class allocation and gets pushed onto
/// that class's freelist. Otherwise it came from the fallback path
/// and is forwarded to `backing.deallocate`.
///
/// `!Sync` — concurrent allocate would race on the per-class
/// `free_head` / `next_uncarved` cells.
pub struct SizeClassed<B: Allocator + FixedRange, const CLASSES: usize> {
    backing: B,
    class_sizes: [usize; CLASSES],
    slabs: [UntypedSlab; CLASSES],
    /// We allocated this many backing layouts for the class regions;
    /// remember them so Drop can return them to `backing`.
    class_layouts: [NonZeroLayout; CLASSES],
    /// O(1) routing table for `pick_class`. Indexed by
    /// `req.next_power_of_two().trailing_zeros()` (i.e. log2 of the
    /// smallest pow2 ≥ req). Stores the class index whose stride is the
    /// smallest pow2 ≥ 2^k, or `CLASS_LOOKUP_NONE` for "no class fits".
    /// 32 entries covers all addressable sizes on 32-bit; on 64-bit a
    /// request that overflows `next_power_of_two` falls through to backing.
    class_lookup: [u8; 32],
    /// Cached `region_size` per class — `slabs[i].capacity * slabs[i].block_stride`.
    /// Lives here (not on UntypedSlab) because `route_dealloc` needs the
    /// pair `(base_offset, region_size)` in cache-hot form for fast bounds
    /// rejection. The base address is computed *live* via
    /// `self.backing.base() + slabs[i].base_offset` — caching it would
    /// reintroduce the pre-move-stale-pointer UAF previously
    /// surfaced for `Slab`/`BumpArena`/`SharedBumpArena`/`StackAlloc`.
    region_sizes: [usize; CLASSES],
}

/// Sentinel in `class_lookup` meaning "no class fits this size; use backing fallback".
const CLASS_LOOKUP_NONE: u8 = u8::MAX;

impl<B: Allocator + FixedRange, const CLASSES: usize> SizeClassed<B, CLASSES> {
    /// Construct with explicit class sizes and a uniform per-class
    /// slot count.
    ///
    /// Each class allocates a region of `class_size * slots_per_class`
    /// bytes from `backing`. Class sizes must be strictly increasing
    /// and each must be a power of two (the inferred per-class
    /// alignment requirement). Returns `Err(AllocError)` if any
    /// constraint is violated, if the backing rejects a class region,
    /// or if `CLASSES == 0`.
    ///
    /// `FreeLink` is 4 bytes (u32), so the minimum class size is 4.
    pub fn with_class_sizes(
        backing: B,
        class_sizes: [usize; CLASSES],
        slots_per_class: u32,
    ) -> Result<Self, AllocError> {
        if CLASSES == 0 || slots_per_class == 0 {
            return Err(AllocError);
        }
        // Validate strictly increasing, power-of-two, >= sizeof(FreeLink).
        let min_size = core::cmp::max(size_of::<FreeLink>(), align_of::<FreeLink>());
        let mut last = 0usize;
        for &s in &class_sizes {
            if s < min_size || !s.is_power_of_two() || s <= last {
                return Err(AllocError);
            }
            last = s;
        }
        // Allocate one region per class.
        // We use MaybeUninit to defer initialising the slab array until
        // we've successfully built every entry; on partial failure the
        // already-allocated regions get returned to the backing in
        // reverse order. The class_layouts array is also init-tracked.
        // `MaybeUninit::uninit().assume_init()` on an array OF MaybeUninits
        // is sound: each element is uninit, but `MaybeUninit<T>` itself is
        // always "init" regardless of T's state (this is the documented
        // MSRV-friendly idiom; `MaybeUninit::uninit_array` is unstable, and
        // `[const { ... }; N]` requires Rust 1.79+ vs. our 1.70 MSRV).
        let mut slabs: [core::mem::MaybeUninit<UntypedSlab>; CLASSES] =
            unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        let mut class_layouts: [core::mem::MaybeUninit<NonZeroLayout>; CLASSES] =
            unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        let mut built = 0usize;
        for i in 0..CLASSES {
            let stride = class_sizes[i];
            let total = match stride.checked_mul(slots_per_class as usize) {
                Some(t) => t,
                None => break,
            };
            let layout = match NonZeroLayout::from_size_align(total, stride) {
                Ok(l) => l,
                Err(_) => break,
            };
            let region = match backing.allocate(layout) {
                Ok(b) => b,
                Err(_) => break,
            };
            // Convert the absolute region pointer to an offset *now*,
            // BEFORE `backing` is moved into `Self` below. Storing the
            // offset and resolving `backing.base() + offset` at each
            // access keeps the slab pointing at the live backing region
            // even when the constructor's return-by-value moves the
            // backing's storage (cf. `InlineBacked<N>::base()` returning
            // `&self.storage`). This is the same anti-UAF pattern
            // applied to `Slab`/`BumpArena`/`SharedBumpArena`/
            // `StackAlloc`.
            let region_addr = region.cast::<u8>().as_ptr() as usize;
            let backing_addr = backing.base().as_ptr() as usize;
            debug_assert!(
                region_addr >= backing_addr,
                "backing.allocate produced a pointer below backing.base() — \
                 backing impl bug, not a SizeClassed bug",
            );
            let base_offset = region_addr - backing_addr;
            slabs[i].write(UntypedSlab::new(base_offset, stride, slots_per_class));
            class_layouts[i].write(layout);
            built += 1;
        }
        if built != CLASSES {
            // Undo partial construction in reverse order.
            for j in (0..built).rev() {
                // SAFETY: slabs[j] is initialised; same for class_layouts[j].
                let slab = unsafe { slabs[j].assume_init_read() };
                let layout = unsafe { class_layouts[j].assume_init_read() };
                // Compute the absolute base from the still-construction-
                // local `backing`. We have not moved `backing` into Self
                // yet, so this is the same address we returned from
                // `backing.allocate(layout)` above.
                let base = unsafe {
                    NonNull::new_unchecked(backing.base().as_ptr().add(slab.base_offset))
                };
                unsafe { backing.deallocate(base, layout) };
            }
            return Err(AllocError);
        }
        // All slabs built; convert MaybeUninit arrays to T arrays.
        // SAFETY: every element is initialised.
        let slabs = unsafe {
            core::mem::transmute_copy::<
                [core::mem::MaybeUninit<UntypedSlab>; CLASSES],
                [UntypedSlab; CLASSES],
            >(&slabs)
        };
        let class_layouts = unsafe {
            core::mem::transmute_copy::<
                [core::mem::MaybeUninit<NonZeroLayout>; CLASSES],
                [NonZeroLayout; CLASSES],
            >(&class_layouts)
        };
        // Build the O(1) routing tables now that every slab base is known.
        // class_lookup[k] = index of smallest class whose stride >= 2^k.
        // CLASSES <= 255 is enforced by `u8` storage.
        if CLASSES > CLASS_LOOKUP_NONE as usize {
            // Undo construction; this is an API misuse caught at construction.
            for j in (0..CLASSES).rev() {
                let base = unsafe {
                    NonNull::new_unchecked(backing.base().as_ptr().add(slabs[j].base_offset))
                };
                unsafe { backing.deallocate(base, class_layouts[j]) };
            }
            return Err(AllocError);
        }
        let mut class_lookup = [CLASS_LOOKUP_NONE; 32];
        // First pass: mark each class size's bit position with that class index.
        for (i, &s) in class_sizes.iter().enumerate() {
            let bit = s.trailing_zeros() as usize;
            if bit < class_lookup.len() && class_lookup[bit] == CLASS_LOOKUP_NONE {
                class_lookup[bit] = i as u8;
            }
        }
        // Second pass (right-to-left): fill gaps so requests landing on a
        // bit position without an exact class round UP to the next-larger
        // class. After this pass, class_lookup[k] is the smallest class
        // whose stride is >= 2^k (or NONE if none exists, including all
        // bit positions above the largest class).
        let mut next = CLASS_LOOKUP_NONE;
        for k in (0..class_lookup.len()).rev() {
            if class_lookup[k] != CLASS_LOOKUP_NONE {
                next = class_lookup[k];
            } else {
                class_lookup[k] = next;
            }
        }

        // Cache per-class region sizes for O(1) dealloc routing. The
        // base address is recomputed *live* on each route_dealloc call
        // via `self.backing.base() + slabs[i].base_offset`; caching the
        // absolute base here would reintroduce the move-stale-pointer
        // UAF previously fixed for `Slab`/`BumpArena`/etc.
        let mut region_sizes = [0usize; CLASSES];
        for i in 0..CLASSES {
            region_sizes[i] = slabs[i].capacity as usize * slabs[i].block_stride;
        }

        Ok(Self {
            backing,
            class_sizes,
            slabs,
            class_layouts,
            class_lookup,
            region_sizes,
        })
    }

    /// Borrow the inner backing.
    #[inline]
    pub fn backing(&self) -> &B {
        &self.backing
    }

    /// Class sizes in bytes.
    #[inline]
    pub fn class_sizes(&self) -> &[usize; CLASSES] {
        &self.class_sizes
    }

    /// Find the smallest class whose stride `>= size` AND `>= align`,
    /// or `None` if no class fits.
    ///
    /// O(1): one `next_power_of_two` (lzcnt-class instruction on modern x86)
    /// + one array lookup against the precomputed `class_lookup` table.
    #[inline]
    fn pick_class(&self, size: usize, align: usize) -> Option<usize> {
        let req = if size > align { size } else { align };
        // `req` is always >= 1 (NonZeroLayout guarantees size>0 and align>=1).
        // For `req > 2^(usize::BITS - 1)`, `next_power_of_two` would overflow;
        // such requests can't fit any class anyway, so route to backing.
        let max_bit = usize::BITS as usize - 1;
        if req > (1usize << max_bit) {
            return None;
        }
        let pow2 = req.next_power_of_two();
        let bit = pow2.trailing_zeros() as usize;
        // class_lookup has 32 entries; on 64-bit a request > 2^31 lands here
        // too. The table fill leaves NONE for any uncovered bit position.
        if bit >= self.class_lookup.len() {
            return None;
        }
        let idx = self.class_lookup[bit];
        if idx == CLASS_LOOKUP_NONE {
            None
        } else {
            Some(idx as usize)
        }
    }

    /// Find the class whose region contains `ptr`, or `None` for
    /// fallback-path pointers.
    ///
    /// Walks `(backing_base + base_offset, region_size)` per class.
    /// `backing_base` is read once from `self.backing.base()` and is
    /// the *live* base — this is what defeats the move-stale-pointer
    /// class of bug. `region_sizes[i]` is the const cached size.
    /// With `CLASSES <= 16` and const-known length the loop unrolls.
    #[inline]
    fn route_dealloc(&self, ptr: NonNull<u8>) -> Option<usize> {
        let p = ptr.as_ptr() as usize;
        let backing_base = self.backing.base().as_ptr() as usize;
        (0..CLASSES).find(|&i| {
            // p - (backing_base + base_offset) < region_size, via wrapping
            // sub so a region whose mathematical end exceeds usize::MAX
            // (impossible on 64-bit but possible in pathological 32-bit
            // configurations) doesn't misroute.
            p.wrapping_sub(backing_base.wrapping_add(self.slabs[i].base_offset))
                < self.region_sizes[i]
        })
    }
}

/// Convenience constructor for the spec's default 8-class set.
impl<B: Allocator + FixedRange> SizeClassed<B, 8> {
    /// Construct with the spec-default 8 classes (8, 16, 32, 64, 128,
    /// 256, 512, 1024 bytes) and `slots_per_class` per class.
    pub fn with_default_classes(backing: B, slots_per_class: u32) -> Result<Self, AllocError> {
        Self::with_class_sizes(backing, DEFAULT_CLASS_SIZES_8, slots_per_class)
    }
}

unsafe impl<B: Allocator + FixedRange, const CLASSES: usize> Deallocator
    for SizeClassed<B, CLASSES>
{
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        match self.route_dealloc(ptr) {
            Some(i) => {
                // Resolve the live backing base once; the slab's
                // free_slot uses it to recompute its current region.
                let backing_base = self.backing.base();
                // SAFETY: ptr came from this class's allocate; layout
                // bounded by class stride (caller-contract).
                unsafe { self.slabs[i].free_slot(backing_base, ptr) };
            }
            None => {
                // Fallback path. SAFETY: ptr came from backing.allocate
                // with the same layout.
                unsafe { self.backing.deallocate(ptr, layout) };
            }
        }
    }
}

unsafe impl<B: Allocator + FixedRange, const CLASSES: usize> Allocator for SizeClassed<B, CLASSES> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size().get();
        let align = layout.align().get();
        if let Some(i) = self.pick_class(size, align) {
            // Resolve the live backing base once per call; allocate_slot
            // uses it to recompute the class region's current base.
            let backing_base = self.backing.base();
            // SAFETY: !Sync — single-threaded access.
            if let Some(ptr) = unsafe { self.slabs[i].allocate_slot(backing_base) } {
                return Ok(NonNull::slice_from_raw_parts(ptr, self.class_sizes[i]));
            }
            // Class exhausted — fall through to backing.
        }
        self.backing.allocate(layout)
    }

    #[inline]
    unsafe fn usable_size(&self, ptr: NonNull<u8>, layout: NonZeroLayout) -> Option<usize> {
        // A class-routed allocation occupies a full class slot, which can far
        // exceed `layout.size()` (a 5-byte request → 8-byte slot). Report the
        // class slot size so an outer scrub wrapper (`PoisonOnFree`/
        // `ZeroizeOnFree`) wipes the whole slot on free, not just the requested
        // prefix. Fallback allocations forward to the backing. Routed by
        // provenance (same as `deallocate`), so `n >= layout.size()` holds:
        // `pick_class` always chose a class with stride >= the request.
        match self.route_dealloc(ptr) {
            Some(i) => Some(self.class_sizes[i]),
            // SAFETY: a fallback ptr came from `backing.allocate(layout)`.
            None => unsafe { self.backing.usable_size(ptr, layout) },
        }
    }

    fn capacity_bytes(&self) -> Option<usize> {
        // Sum across classes; fallback's capacity is not included
        // because it may be unbounded.
        let mut total = 0usize;
        for slab in &self.slabs {
            total = total.checked_add(slab.capacity as usize * slab.block_stride)?;
        }
        Some(total)
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        // Sum per-class UntypedSlab counters plus the backing fallback's
        // counter (in case it's a hardened type like Slab<_, _, SipHashMAC>).
        // `saturating_add` guards against u64 overflow.
        //
        // Per-class counters are `AtomicUsize` (for 32-bit portability);
        // cast each load to `u64` before the fold so the saturating sum
        // operates in u64 width — preventing premature saturation when a
        // 32-bit host has multiple classes each near u32::MAX.
        let class_total = self
            .slabs
            .iter()
            .map(|s| s.corruption_events.load(Ordering::Relaxed) as u64)
            .fold(0u64, |acc, x| acc.saturating_add(x));
        class_total.saturating_add(self.backing.corruption_events())
    }
}

impl<B: Allocator + FixedRange, const CLASSES: usize> Drop for SizeClassed<B, CLASSES> {
    fn drop(&mut self) {
        // Return each class region to the backing. We recompute each
        // region's current absolute base from `backing.base() +
        // base_offset` rather than reading a cached pointer — the
        // backing's `base()` is the live address after any moves the
        // wrapper underwent. (See the stale-base-pointer UAF
        // previously fixed for Slab/BumpArena/etc.)
        for i in 0..CLASSES {
            // SAFETY: we issued backing.allocate(class_layouts[i]) at
            // construction, recorded the offset from backing.base(),
            // and the backing region is still live (we're inside its
            // Drop chain).
            let base = unsafe {
                NonNull::new_unchecked(self.backing.base().as_ptr().add(self.slabs[i].base_offset))
            };
            unsafe { self.backing.deallocate(base, self.class_layouts[i]) };
        }
    }
}

// Test module uses `MmapBacked`, which since 0.3.1 is gated on
// `all(feature = "std", any(unix, windows))`. The previous gate of
// just `feature = "std"` would fail to compile on
// `wasm32-wasip1+std` (and is masked today only by the proptest
// dev-dep also failing on wasm32).
#[cfg(all(test, feature = "std", any(unix, windows)))]
mod tests {
    use super::*;
    use crate::backing::{InlineBacked, MmapBacked};
    use crate::layout::BumpArena;

    /// Build a SizeClassed using an MmapBacked-backed BumpArena —
    /// supports up to page alignment, so class strides up to a page
    /// (typically 4 KiB) work.
    fn build_mmap() -> SizeClassed<BumpArena<MmapBacked>, 4> {
        SizeClassed::with_class_sizes(
            BumpArena::new(MmapBacked::new(64 * 1024).unwrap()).unwrap(),
            [8, 16, 32, 64],
            4,
        )
        .expect("valid")
    }

    #[test]
    fn rejects_zero_slots_per_class() {
        let r =
            SizeClassed::<_, 4>::with_class_sizes(InlineBacked::<8192>::new(), [8, 16, 32, 64], 0);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_non_power_of_two() {
        let r =
            SizeClassed::<_, 4>::with_class_sizes(InlineBacked::<8192>::new(), [8, 24, 32, 64], 4);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_non_increasing() {
        let r =
            SizeClassed::<_, 4>::with_class_sizes(InlineBacked::<8192>::new(), [8, 8, 16, 32], 4);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_classes_below_freelink_size() {
        // size_of::<FreeLink>() = 4; class size of 2 must be rejected.
        let r =
            SizeClassed::<_, 4>::with_class_sizes(InlineBacked::<8192>::new(), [2, 4, 8, 16], 4);
        assert!(r.is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn pick_class_picks_smallest_fit() {
        let sc = build_mmap();
        // size=5, align=1 fits in 8-byte class.
        let layout = NonZeroLayout::from_size_align(5, 1).unwrap();
        let block = sc.allocate(layout).unwrap();
        assert_eq!(block.len(), 8, "slice length == class stride");
        unsafe { sc.deallocate(block.cast(), layout) };
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn alignment_drives_class_pick() {
        let sc = build_mmap();
        // size=4 fits 8-class, but align=16 forces class 16 minimum.
        let layout = NonZeroLayout::from_size_align(4, 16).unwrap();
        let block = sc.allocate(layout).unwrap();
        let addr = block.cast::<u8>().as_ptr() as usize;
        assert_eq!(addr % 16, 0, "alignment must be respected");
        assert!(block.len() >= 16, "class stride should accommodate align");
        unsafe { sc.deallocate(block.cast(), layout) };
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn oversized_falls_back_to_backing() {
        let sc = build_mmap();
        // Max class is 64; ask for 128 → falls through.
        let layout = NonZeroLayout::from_size_align(128, 8).unwrap();
        let block = sc.allocate(layout).unwrap();
        // The slice length should be the requested size (backing path),
        // not a class stride.
        assert_eq!(block.len(), 128);
        unsafe { sc.deallocate(block.cast(), layout) };
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn freelist_recycles_within_class() {
        let sc = build_mmap();
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        let a = sc.allocate(layout).unwrap();
        let addr_a = a.cast::<u8>().as_ptr() as usize;
        unsafe { sc.deallocate(a.cast(), layout) };
        let b = sc.allocate(layout).unwrap();
        let addr_b = b.cast::<u8>().as_ptr() as usize;
        assert_eq!(addr_a, addr_b, "freed slot must be reused");
        unsafe { sc.deallocate(b.cast(), layout) };
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn class_exhaustion_falls_through_to_backing() {
        let sc = build_mmap();
        // 4 slots in the 8-class. Allocate 5 to exhaust + fall through.
        let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
        let mut ptrs = Vec::new();
        for _ in 0..4 {
            ptrs.push(sc.allocate(layout).unwrap());
        }
        // 5th must come from the backing path (size 8 fits in InlineBacked).
        let extra = sc.allocate(layout).unwrap();
        ptrs.push(extra);
        for p in &ptrs {
            unsafe { sc.deallocate(p.cast(), layout) };
        }
    }

    /// `SizeClassed::with_class_sizes` derives the class-region
    /// alignment from `stride` (line 274 — `NonZeroLayout::from_size_align(
    /// total, stride)`), so a stride that exceeds the backing's
    /// `MAX_ALIGN` is rejected by `backing.allocate(...)` at
    /// construction. `InlineBacked<N>` caps alignment at
    /// `MAX_ALIGN = 16`, so any class ≥ 32 is unbuildable on this
    /// backing — `with_default_classes` (which uses up to 1024) MUST
    /// return `Err(AllocError)` rather than constructing a slab whose
    /// `allocate` would later hand out under-aligned pointers.
    ///
    /// This is a RUNTIME check (the backing's `Allocator::allocate`
    /// surface returns `AllocError` for over-alignment); the type
    /// system does not currently enforce
    /// `backing::MAX_ALIGN >= max(class_sizes)` at compile time — the
    /// safety contract on the type does not yet enforce this statically.
    #[test]
    fn inline_backed_rejects_oversize_class_at_construction() {
        // Stride 32 already exceeds InlineBacked's MAX_ALIGN = 16, so
        // even this minimal "two-class" misuse must fail.
        let r = SizeClassed::<_, 2>::with_class_sizes(InlineBacked::<8192>::new(), [8, 32], 4);
        assert!(
            r.is_err(),
            "InlineBacked (MAX_ALIGN=16) must reject class stride 32"
        );
        // And the default 8-class set (up to 1024) is unbuildable on
        // InlineBacked for the same reason — pinning so a future
        // backing-alignment upgrade doesn't silently start succeeding.
        let r2 = SizeClassed::<_, 8>::with_default_classes(InlineBacked::<{ 8 * 1024 }>::new(), 4);
        assert!(
            r2.is_err(),
            "InlineBacked cannot satisfy 1024-byte class alignment"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn default_8_class_constructor_builds() {
        // 8 classes × 4 slots × largest 1024 = needs ≥8 KiB of backing
        // plus alignment slack up to the 1024 stride. InlineBacked tops
        // out at MAX_ALIGN=16, so we must use MmapBacked here.
        let inner = BumpArena::new(MmapBacked::new(64 * 1024).unwrap()).unwrap();
        let sc = SizeClassed::<_, 8>::with_default_classes(inner, 4);
        assert!(sc.is_ok());
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn pick_class_handles_non_sequential_powers_of_two() {
        // Sparse class set: 8, 32, 128 (skipping 16, 64). The lookup
        // table's gap-fill must route size=16 to class 32 (smallest pow2
        // ≥ 16 that's a configured class) and size=64 to class 128.
        let sc: SizeClassed<BumpArena<MmapBacked>, 3> = SizeClassed::with_class_sizes(
            BumpArena::new(MmapBacked::new(16 * 1024).unwrap()).unwrap(),
            [8, 32, 128],
            4,
        )
        .expect("valid");
        // size=16 → class 32 (gap fill: bit 4 routes to class index 1)
        let l16 = NonZeroLayout::from_size_align(16, 1).unwrap();
        let b16 = sc.allocate(l16).unwrap();
        assert_eq!(b16.len(), 32);
        // size=64 → class 128 (gap fill: bit 6 routes to class index 2)
        let l64 = NonZeroLayout::from_size_align(64, 1).unwrap();
        let b64 = sc.allocate(l64).unwrap();
        assert_eq!(b64.len(), 128);
        unsafe {
            sc.deallocate(b16.cast(), l16);
            sc.deallocate(b64.cast(), l64);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn capacity_bytes_sums_classes() {
        let sc = build_mmap();
        // 4 + 16 + 32 + 64 striped, times 4 slots: 8*4 + 16*4 + 32*4 + 64*4
        // = 32 + 64 + 128 + 256 = 480
        let expected = (8 + 16 + 32 + 64) * 4usize;
        assert_eq!(sc.capacity_bytes(), Some(expected));
    }

    /// `usable_size` reports the class slot size, not the requested size, so an
    /// outer scrub wrapper wipes the whole class-rounded slot on free. A 5-byte
    /// request routes to the 8-byte class. (Uses the mmap-backed helper because
    /// the 64-byte class needs 64-byte alignment, beyond `InlineBacked`'s
    /// `MAX_ALIGN`.)
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn usable_size_reports_class_size() {
        let sc = build_mmap();
        let layout = NonZeroLayout::from_size_align(5, 1).unwrap();
        let block = sc.allocate(layout).unwrap();
        let ptr = block.cast::<u8>();
        let us = unsafe { sc.usable_size(ptr, layout) };
        assert_eq!(us, Some(8), "usable_size must report the class slot size");
        unsafe { sc.deallocate(ptr, layout) };
    }
}
