//! `GenerationalSlab<T, B, G>` — typed allocator that returns stable
//! [`Handle<T, G>`] values instead of raw pointers. Each slot carries a
//! generation counter; a handle is only valid while its generation matches
//! the slot's. This prevents ABA / use-after-free at the API level: a stale
//! handle returns `None` rather than producing undefined behavior.
//!
//! Interleaved layout: generation + state live together in a single
//! `GenerationalSlot<T>` struct, backed by one contiguous allocation.
//! One cache-line load covers both the generation check and the value access.
//!
//! See `docs/ARCHITECTURE.md` for the generational-slab design.

use core::marker::PhantomData;
use core::ptr::NonNull;

use forge_alloc_core::{AllocError, Allocator, FixedRange, NonZeroLayout};

/// Width of the generation counter. Public so callers can pick `u32`
/// (default — 2^32 reuses per slot before wraparound) or `u64` (effectively
/// unbounded). Internal trait sealed via the empty trait pattern.
pub trait GenerationInt: Copy + Eq + sealed::Sealed {
    /// Initial generation value.
    const ZERO: Self;
    /// Increment, wrapping at the type's max value.
    fn wrapping_inc(self) -> Self;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

impl GenerationInt for u32 {
    const ZERO: Self = 0;
    #[inline]
    fn wrapping_inc(self) -> Self {
        self.wrapping_add(1)
    }
}

impl GenerationInt for u64 {
    const ZERO: Self = 0;
    #[inline]
    fn wrapping_inc(self) -> Self {
        self.wrapping_add(1)
    }
}

/// Stable, non-pointer handle to a slot in a [`GenerationalSlab`].
///
/// `Copy` (cheap to pass), `Eq` + `Hash` (works in maps), and notably does
/// NOT carry a reference to the slab — handles do not extend the slab's
/// lifetime. A handle outliving its slab is allowed and gives `None` on
/// access.
///
/// # Generation wraparound — ABA-after-2^G-reuses
///
/// The generation counter increments on every `remove` of a slot and is
/// `wrapping_add(1)` (per `GenerationInt`). After `2^G` reuses of the
/// *same* slot, the counter returns to its original value — a stale
/// `Handle` whose generation matches that original value will then be
/// accepted as valid by `get`/`remove`, even though the slot now holds
/// a different `T`. This is the classic ABA problem at a long horizon.
///
/// - `G = u32` (default): wrap after **2^32 ≈ 4.3 billion** reuses of
///   the same slot. Realistic for long-running servers with high churn
///   on a small slab (e.g. a 1024-slot connection pool processing
///   millions of connections per second).
/// - `G = u64`: wrap after 2^64 reuses — effectively unreachable in
///   any realistic deployment (>500 years at 1 GHz of pure slot churn).
///   Recommended for long-lived servers; the per-handle cost is 8
///   extra bytes (8 bytes for `u32`, 16 for `u64` — the wider counter
///   forces 8-byte struct alignment and 4 bytes of tail padding).
///
/// `Copy` means a handle can outlive the slot's original lifetime
/// arbitrarily — including past 2^G recycles. If your handles can
/// realistically be stashed for that long (audit logs, persisted
/// session records, snapshot indices), use `Handle<T, u64>`. For
/// per-request handles that never escape the request scope, `u32` is
/// fine.
///
/// # Known limitation — cross-pool handle confusion
///
/// `Handle<T, G>` is typed by `T` (and `G`) but **not** by which
/// `GenerationalSlab<T, B, G>` issued it. Passing a handle returned
/// by `pool_a.insert(...)` to `pool_b.get(...)` is currently a
/// runtime concern, not a compile error: the slot index will be
/// interpreted against `pool_b`'s slot array, and ABA-safety
/// degenerates to "match-by-coincidence" on the generation counter.
///
/// Closing this gap requires runtime-unique branding — either an
/// invariant-lifetime tag (`generativity` crate style) or a
/// monotonic pool-id passed through `PhantomData` — and is
/// **API-breaking** because every `Handle<T, G>` user signature
/// would gain an extra type parameter. The v0.1 API ships without
/// it; v2.0+ may revisit. See the "Generational-handle slab" recipe
/// in `docs/COMPOSITION_RECIPES.md` for the documented pitfall.
///
/// Until branded, callers who keep multiple pools of the same `T`
/// must NOT mix their handles. The naming convention recommended in
/// the recipes doc is to type-alias each pool's handle:
/// `type SessionHandle = Handle<Session, u32>` per pool, in
/// different modules.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Handle<T, G: GenerationInt = u32> {
    index: u32,
    generation: G,
    _phantom: PhantomData<*const T>,
}

/// One slot in the generational slab — generation + value state colocated.
enum SlotState<T> {
    Occupied(T),
    Free { next_free: Option<u32> },
}

struct GenerationalSlot<T, G: GenerationInt> {
    generation: G,
    state: SlotState<T>,
}

/// Generational slab.
///
/// `capacity` slots, each holding either a live `T` or a free-list link.
/// Handles encode `(slot_index, generation_at_allocate)`; access via
/// [`get`](Self::get) / [`get_mut`](Self::get_mut) / [`remove`](Self::remove)
/// only succeeds when the slot's current generation equals the handle's.
///
/// # Thread safety
///
/// All mutation goes through `&mut self`. The slab is `Send + Sync` if `T`
/// and `B` are, but concurrent mutation requires external synchronization
/// (`Mutex<GenerationalSlab<T, B>>`).
pub struct GenerationalSlab<T, B: Allocator + FixedRange, G: GenerationInt = u32> {
    backing: B,
    /// Byte offset from `backing.base()` to the start of the slot array.
    ///
    /// We deliberately do NOT store an absolute `NonNull<GenerationalSlot<T, G>>`
    /// captured from `backing.allocate(...)` at construction. Backings whose
    /// `base()` is structure-relative (e.g. `InlineBacked<N>` returns
    /// `&self.storage`) report a DIFFERENT address before and after the
    /// backing has been moved. An absolute pointer captured at construction
    /// time would then point at the backing's OLD location after `Self {
    /// backing, ... }` moves the backing into Self by-value — silently
    /// corrupting every subsequent `insert` / `get` / `remove`. Matches the
    /// Slab move-safety pattern (see `UntypedSlab::base_offset` in size_classed.rs).
    slots_offset: usize,
    capacity: u32,
    /// Free-list head: `None` = empty (must carve from `len`).
    free_head: Option<u32>,
    /// First slot never yet allocated.
    len: u32,
    backing_layout: NonZeroLayout,
    /// Ties `T` and `G` to the struct so the field-less slot region (we
    /// store only an offset, not a typed pointer) still carries the right
    /// type identity. `*const` keeps `Send`/`Sync` opt-in via the explicit
    /// `unsafe impl`s below.
    _phantom: PhantomData<(*const T, G)>,
}

impl<T, B: Allocator + FixedRange> GenerationalSlab<T, B, u32> {
    /// Construct with default `u32` generation counter.
    pub fn new(capacity: usize, backing: B) -> Result<Self, AllocError> {
        Self::with_generation(capacity, backing)
    }
}

impl<T, B: Allocator + FixedRange, G: GenerationInt> GenerationalSlab<T, B, G> {
    /// Construct with an explicit generation-width parameter.
    ///
    /// Use `u64` for high-churn long-running servers where 2^32 reuses of
    /// the same slot could realistically occur.
    pub fn with_generation(capacity: usize, backing: B) -> Result<Self, AllocError> {
        if capacity == 0 {
            return Err(AllocError);
        }
        let cap_u32 = u32::try_from(capacity).map_err(|_| AllocError)?;
        let layout = NonZeroLayout::array::<GenerationalSlot<T, G>>(capacity).ok_or(AllocError)?;
        let block = backing.allocate(layout)?;
        let slots_ptr = block.cast::<GenerationalSlot<T, G>>();
        // Compute offset from backing.base() — see field docs on
        // `slots_offset`. Absolute `slots_ptr` is only valid in the local
        // frame; after `Self { backing, ... }` moves `backing` by-value,
        // any structure-relative backing's base address changes.
        let base_addr = backing.base().as_ptr() as usize;
        let slots_addr = slots_ptr.as_ptr() as usize;
        debug_assert!(
            slots_addr >= base_addr,
            "GenerationalSlab: allocate() returned a pointer below backing.base() — \
             impossible if `B: FixedRange` honours the OsBacked / FixedRange contract"
        );
        let slots_offset = slots_addr - base_addr;
        // Initialize every slot to Free with generation ZERO.
        // SAFETY: `slots_ptr..slots_ptr+capacity` is uninitialized memory of
        // the right size and alignment per the layout request. We write
        // through the LOCAL (pre-move) pointer; the post-move recompute
        // (`backing.base() + slots_offset`) hits the SAME bytes inside the
        // moved backing's storage, since the bytes are a sub-region of the
        // backing and travel with it on the move.
        unsafe {
            for i in 0..capacity {
                let p = slots_ptr.as_ptr().add(i);
                p.write(GenerationalSlot {
                    generation: G::ZERO,
                    state: SlotState::Free { next_free: None },
                });
            }
        }
        Ok(Self {
            backing,
            slots_offset,
            capacity: cap_u32,
            free_head: None,
            len: 0,
            backing_layout: layout,
            _phantom: PhantomData,
        })
    }

    /// Recompute the live slot-array pointer from the backing's *current*
    /// base. Must be called every time the slab needs to dereference into
    /// the slot region — never cached across calls.
    ///
    /// For backings with a structure-relative `base()` (e.g.
    /// `InlineBacked<N>`), the backing's storage moves with the
    /// `GenerationalSlab` struct itself. The offset captured at
    /// construction stays valid (the relative position of the slot array
    /// inside the backing's storage is invariant); recomputing through
    /// the live `backing.base()` plus that offset always lands on the
    /// correct bytes.
    #[inline]
    fn slots(&self) -> NonNull<GenerationalSlot<T, G>> {
        let base = self.backing.base().as_ptr();
        // SAFETY: `slots_offset` was the difference between
        // `backing.allocate(layout)`'s result and `backing.base()` at
        // construction; the slot region is a sub-range of the backing's
        // storage and `slots_offset + capacity * size_of` <= region.
        let p = unsafe { base.add(self.slots_offset) };
        // SAFETY: derived from a non-null backing.base() plus an in-range
        // offset; the result is non-null.
        unsafe { NonNull::new_unchecked(p.cast::<GenerationalSlot<T, G>>()) }
    }

    /// Number of slots.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    /// Insert a value and return its handle. Errors if the slab is full.
    pub fn insert(&mut self, value: T) -> Result<Handle<T, G>, AllocError> {
        // Try free list first. Resolve `slot_idx` AND `slot` pointer in a
        // single branch so the freelist-pop path does ONE
        // `slots.as_ptr().add(idx)` computation rather than two — the
        // optimizer should CSE, but folding the access ourselves avoids
        // relying on noalias deduction across the `self.free_head =
        // *next_free` write that sits between the two derefs.
        let (slot_idx, slot) = match self.free_head {
            Some(idx) => {
                // SAFETY: idx came from our free_head, set by remove().
                let slot = unsafe { &mut *self.slots().as_ptr().add(idx as usize) };
                match &slot.state {
                    SlotState::Free { next_free } => {
                        self.free_head = *next_free;
                    }
                    SlotState::Occupied(_) => {
                        // Inconsistent state — should never happen.
                        return Err(AllocError);
                    }
                }
                (idx, slot)
            }
            None => {
                if self.len >= self.capacity {
                    return Err(AllocError);
                }
                let idx = self.len;
                self.len += 1;
                // SAFETY: idx < capacity (we just checked); slot is
                // initialized by `with_generation`.
                let slot = unsafe { &mut *self.slots().as_ptr().add(idx as usize) };
                (idx, slot)
            }
        };
        // Place the value and capture the current generation for the handle.
        slot.state = SlotState::Occupied(value);
        let generation = slot.generation;
        // Defense-in-depth: the slot we just filled must lie within
        // `[0, len)` — freelist-pop returns previously-allocated slots
        // and the fresh-allocation branch above bumped `self.len`. A
        // mismatch indicates either freelist corruption or a stale
        // `len` (refactor regression). Compiled out in release.
        debug_assert!(
            (slot_idx as usize) < self.len as usize,
            "GenerationalSlab::insert: filled slot {slot_idx} but len={} — \
             freelist corruption or stale len",
            self.len,
        );
        Ok(Handle {
            index: slot_idx,
            generation,
            _phantom: PhantomData,
        })
    }

    /// Borrow the value behind a handle, or `None` if stale / vacant.
    pub fn get(&self, handle: Handle<T, G>) -> Option<&T> {
        if handle.index >= self.capacity {
            return None;
        }
        // SAFETY: index is in-range; slot is initialized.
        let slot = unsafe { &*self.slots().as_ptr().add(handle.index as usize) };
        if slot.generation != handle.generation {
            return None;
        }
        match &slot.state {
            SlotState::Occupied(v) => Some(v),
            SlotState::Free { .. } => None,
        }
    }

    /// Borrow mutably behind a handle, or `None` if stale / vacant.
    pub fn get_mut(&mut self, handle: Handle<T, G>) -> Option<&mut T> {
        if handle.index >= self.capacity {
            return None;
        }
        // SAFETY: as above; &mut self gives exclusive access.
        let slot = unsafe { &mut *self.slots().as_ptr().add(handle.index as usize) };
        if slot.generation != handle.generation {
            return None;
        }
        match &mut slot.state {
            SlotState::Occupied(v) => Some(v),
            SlotState::Free { .. } => None,
        }
    }

    /// Remove the value behind a handle and return it, or `None` if stale.
    /// Bumps the slot's generation so the handle (and any copies) become
    /// invalid.
    pub fn remove(&mut self, handle: Handle<T, G>) -> Option<T> {
        if handle.index >= self.capacity {
            return None;
        }
        // SAFETY: as above.
        let slot = unsafe { &mut *self.slots().as_ptr().add(handle.index as usize) };
        if slot.generation != handle.generation {
            return None;
        }
        // Swap in a Free state and extract the prior value.
        let prior = core::mem::replace(
            &mut slot.state,
            SlotState::Free {
                next_free: self.free_head,
            },
        );
        self.free_head = Some(handle.index);
        // Bump generation — old handle is now stale.
        slot.generation = slot.generation.wrapping_inc();
        match prior {
            SlotState::Occupied(v) => Some(v),
            SlotState::Free { .. } => None, // can't happen; we checked above
        }
    }

    /// True if the handle would currently access a live value.
    #[inline]
    pub fn contains(&self, handle: Handle<T, G>) -> bool {
        self.get(handle).is_some()
    }
}

impl<T, B: Allocator + FixedRange, G: GenerationInt> Drop for GenerationalSlab<T, B, G> {
    fn drop(&mut self) {
        // Drop every Occupied value, then return the backing chunk.
        //
        // **Panic-safety contract**: `T::drop` is allowed to panic. If it
        // does:
        //
        // - During **normal** drop (no in-flight unwind): the panic
        //   escapes this body. Slots at indices > i are left undropped
        //   (Rust's loop unwind semantics — `mem::replace`'s temporary
        //   is the unwinding value), and the `backing.deallocate` call
        //   below this loop **does not run**, leaking the entire
        //   backing region. This is strictly worse than Slab's pattern
        //   (Slab deallocates the backing FIRST), but unlike Slab we
        //   cannot reorder — we need the backing memory live to read
        //   each `slot.state` pointer.
        // - During **drop while unwinding** (a panic was already in
        //   flight when `GenerationalSlab` started dropping): a second
        //   panic in `T::drop` triggers **immediate process abort**
        //   per Rust language rules (`drop-while-panicking → abort`).
        //   This is unavoidable without `catch_unwind` (std-only) and
        //   matches the standard library's contract everywhere.
        //
        // Production guidance: ensure `T::drop` is panic-free for any
        // `T` stored in a `GenerationalSlab`. Wrap fallible cleanup in
        // a separate `try_close()`-style API and run it before the slab
        // drops.
        //
        // SAFETY: every slot 0..capacity was initialized in construction.
        // Use the live `self.slots()` accessor so the post-move backing
        // base is honoured — caching the construction-time pointer would
        // double-bug the structure-relative-backing case at drop time too.
        unsafe {
            let slots = self.slots();
            for i in 0..self.capacity as usize {
                let slot = &mut *slots.as_ptr().add(i);
                if let SlotState::Occupied(_) = &slot.state {
                    // Replace with Free to run T's Drop.
                    let _ =
                        core::mem::replace(&mut slot.state, SlotState::Free { next_free: None });
                }
            }
            self.backing
                .deallocate(slots.cast::<u8>(), self.backing_layout);
        }
    }
}

// Send when T, B, G are Send. Sync when T, B are Sync; G is Copy and !Sync
// is impossible for Copy types in practice. The slab itself permits &mut
// access through methods, so cross-thread use via Arc<Mutex<...>> is the
// standard pattern.
unsafe impl<T: Send, B: Allocator + FixedRange + Send, G: GenerationInt + Send> Send
    for GenerationalSlab<T, B, G>
{
}
unsafe impl<T: Sync, B: Allocator + FixedRange + Sync, G: GenerationInt + Sync> Sync
    for GenerationalSlab<T, B, G>
{
}

// ============================================================================
// Kani proof harnesses
// ============================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::backing::InlineBacked;

    /// `insert` then `get` returns the inserted value via the issued handle.
    #[kani::proof]
    #[kani::unwind(3)]
    fn insert_then_get_round_trips() {
        let mut s: GenerationalSlab<u64, InlineBacked<512>> =
            GenerationalSlab::new(4, InlineBacked::<512>::new()).unwrap();
        let v: u64 = kani::any();
        if let Ok(h) = s.insert(v) {
            assert!(s.get(h) == Some(&v));
        }
    }

    /// After `remove`, the old handle no longer resolves (`get` returns
    /// `None`) — the generation counter on the slot mismatches the
    /// handle's frozen generation.
    #[kani::proof]
    #[kani::unwind(3)]
    fn remove_invalidates_old_handle() {
        let mut s: GenerationalSlab<u64, InlineBacked<512>> =
            GenerationalSlab::new(4, InlineBacked::<512>::new()).unwrap();
        if let Ok(h) = s.insert(0xAA) {
            let _ = s.remove(h);
            // After remove, the handle's generation is stale.
            assert!(s.get(h).is_none());
        }
    }

    /// Reusing a slot via subsequent insert hands out a NEW handle whose
    /// generation differs from any prior handle to the same slot.
    #[kani::proof]
    #[kani::unwind(3)]
    fn reused_slot_gets_new_handle() {
        let mut s: GenerationalSlab<u64, InlineBacked<512>> =
            GenerationalSlab::new(4, InlineBacked::<512>::new()).unwrap();
        if let Ok(h_old) = s.insert(0xAA) {
            let _ = s.remove(h_old);
            if let Ok(h_new) = s.insert(0xBB) {
                // Same slot index could be reused, but generation must
                // have moved — so the two handles are not interchangeable.
                assert!(s.get(h_old).is_none());
                assert!(s.get(h_new) == Some(&0xBB));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;

    #[test]
    fn insert_then_get_round_trip() {
        let mut s: GenerationalSlab<u64, InlineBacked<1024>> =
            GenerationalSlab::new(16, InlineBacked::<1024>::new()).unwrap();
        let h = s.insert(42).unwrap();
        assert_eq!(s.get(h), Some(&42));
    }

    #[test]
    fn remove_invalidates_handle() {
        let mut s: GenerationalSlab<u64, InlineBacked<1024>> =
            GenerationalSlab::new(16, InlineBacked::<1024>::new()).unwrap();
        let h = s.insert(42).unwrap();
        assert_eq!(s.remove(h), Some(42));
        // Second access through the same handle must return None.
        assert_eq!(s.get(h), None);
        assert_eq!(s.remove(h), None);
    }

    #[test]
    fn reused_slot_returns_new_value_only_to_new_handle() {
        let mut s: GenerationalSlab<u64, InlineBacked<512>> =
            GenerationalSlab::new(4, InlineBacked::<512>::new()).unwrap();
        let h1 = s.insert(1).unwrap();
        s.remove(h1);
        let h2 = s.insert(2).unwrap();
        // ABA test: stale h1 must NOT see h2's value.
        assert_eq!(s.get(h1), None, "stale handle must not access reused slot");
        assert_eq!(s.get(h2), Some(&2));
    }

    #[test]
    fn capacity_exhaustion_returns_err() {
        let mut s: GenerationalSlab<u64, InlineBacked<256>> =
            GenerationalSlab::new(4, InlineBacked::<256>::new()).unwrap();
        for i in 0..4 {
            s.insert(i).unwrap();
        }
        assert!(s.insert(5).is_err());
    }

    #[test]
    fn get_mut_mutates() {
        let mut s: GenerationalSlab<u64, InlineBacked<512>> =
            GenerationalSlab::new(4, InlineBacked::<512>::new()).unwrap();
        let h = s.insert(0).unwrap();
        *s.get_mut(h).unwrap() = 99;
        assert_eq!(s.get(h), Some(&99));
    }

    #[test]
    fn handles_are_copy() {
        let mut s: GenerationalSlab<u64, InlineBacked<256>> =
            GenerationalSlab::new(2, InlineBacked::<256>::new()).unwrap();
        let h = s.insert(7).unwrap();
        let h2 = h; // Copy
        assert_eq!(s.get(h), Some(&7));
        assert_eq!(s.get(h2), Some(&7));
    }

    #[test]
    #[cfg(feature = "std")]
    fn drops_occupied_values() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        struct Counter(Arc<AtomicUsize>);
        impl Drop for Counter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let mut s: GenerationalSlab<Counter, InlineBacked<512>> =
                GenerationalSlab::new(4, InlineBacked::<512>::new()).unwrap();
            s.insert(Counter(drops.clone())).unwrap();
            s.insert(Counter(drops.clone())).unwrap();
            // slab dropped here; both Counters should run Drop.
        }
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    /// Regression: `GenerationalSlab` used to
    /// cache `slots: NonNull<GenerationalSlot<T, G>>` captured from
    /// `backing.allocate(...)` BEFORE the backing was moved into Self. For
    /// structure-relative backings like `InlineBacked<N>` (whose `base()`
    /// returns `&self.storage`), the cached pointer aimed at the *pre-move*
    /// location forever, silently corrupting every subsequent
    /// `insert`/`get`/`remove`.
    ///
    /// This test constructs a slab inside a helper, returns it by value to
    /// a caller frame, defeats NRVO with a large stack local, then exercises
    /// the slab. Each access must land on the LIVE backing's slot region,
    /// not the pre-move location.
    #[test]
    fn slab_survives_move_with_inline_backed_nrvo_defeating() {
        #[inline(never)]
        fn make() -> GenerationalSlab<u64, InlineBacked<512>> {
            let mut s: GenerationalSlab<u64, InlineBacked<512>> =
                GenerationalSlab::new(8, InlineBacked::<512>::new()).unwrap();
            // Pre-populate from the constructor frame so the bug's window
            // is exercised: writes here go through the construction-time
            // backing location. The returned struct must still read the
            // SAME values from the moved backing's storage.
            for i in 0..4u64 {
                let _ = s.insert(0xAA00 + i).unwrap();
            }
            s
        }
        // Force enough locals to defeat NRVO. The 4096-byte array sits
        // on the caller's stack, pushing the returned `slab` to a
        // different address than `make`'s local.
        let _arr = [0u8; 4096];
        let mut slab = make();
        // The construction-time pointer (if cached) would now be stale.
        let h_new = slab.insert(0xDEADBEEF).unwrap();
        // Live read through the new handle — must round-trip.
        assert_eq!(slab.get(h_new), Some(&0xDEADBEEF));
        // Insert/get round trips for more values to stress the post-move
        // slot region.
        let mut handles = alloc::vec::Vec::new();
        for v in 0..3u64 {
            handles.push(slab.insert(0xCAFE_0000 + v).unwrap());
        }
        for (i, h) in handles.iter().enumerate() {
            assert_eq!(slab.get(*h), Some(&(0xCAFE_0000 + i as u64)));
        }
        // Remove + insert: exercise the freelist path on the post-move
        // storage. If the slots pointer were stale, the bumped generation
        // would be written into the WRONG slot and the new handle's read
        // would either return None or the stale value.
        assert_eq!(slab.remove(h_new), Some(0xDEADBEEF));
        let h_reuse = slab.insert(0x1234).unwrap();
        assert_eq!(slab.get(h_reuse), Some(&0x1234));
    }

    /// Companion: assert that the slot region the slab dereferences
    /// actually lies inside the LIVE backing's range after the move.
    /// A stale-pointer bug would have the slab reading from a different
    /// address range than `slab.backing.base()` reports.
    #[test]
    fn slab_slots_lie_inside_live_backing_after_move() {
        #[inline(never)]
        fn make() -> GenerationalSlab<u64, InlineBacked<512>> {
            GenerationalSlab::new(4, InlineBacked::<512>::new()).unwrap()
        }
        let _arr = [0u8; 4096];
        let mut slab = make();
        // Insert and capture the address of the slot.
        let h = slab.insert(0x55AA).unwrap();
        let val_ref: &u64 = slab.get(h).unwrap();
        let val_addr = val_ref as *const u64 as usize;
        // Resolve the live backing base + size via FixedRange.
        let base = FixedRange::base(&slab.backing).as_ptr() as usize;
        let size = FixedRange::size(&slab.backing);
        assert!(
            val_addr >= base && val_addr < base + size,
            "slot address {val_addr:#x} outside live backing range \
             [{base:#x}, {:#x}) — stale-pointer bug",
            base + size,
        );
    }
}
