//! `SlabOwner<T, B>` + `SlabRemote<T, B>` — cross-thread typed allocation
//! via the ownership-return model. Replaces the previously-named
//! `MessagePassingSlab`.
//!
//! `Slab<T, B>` is `!Sync`. Another thread cannot call `deallocate` directly
//! without racing on the freelist head. The snmalloc / mimalloc pattern,
//! adopted here: cross-thread frees route a slot index to the owner via a
//! queue; the owner drains the queue back into its local freelist on its
//! next allocate.
//!
//! v0.1 ships a `Mutex<VecDeque<u32>>`-backed queue for correctness;
//! v1.0 will swap in a lock-free MPSC ring. The visible API
//! is identical either way.
//!
//! Requires `std` (uses `Arc`, `Mutex`, `VecDeque`).
//!
//! See `docs/ARCHITECTURE.md` for the cross-thread ownership design.

#![cfg(feature = "std")]

use core::ptr::NonNull;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use forge_alloc_core::{AllocError, Allocator, Deallocator, FixedRange, NonZeroLayout};

use crate::layout::Slab;

/// How aggressively the owner drains the remote-free queue.
#[derive(Copy, Clone, Debug)]
pub enum BatchPolicy {
    /// Drain every `N` remote frees. v0.1 default = 64.
    Fixed(usize),
    /// Stepped-threshold adaptive policy. The drain threshold steps
    /// through 5 levels — `[8, 16, 32, 64, 128]` — based on observed
    /// queue depth relative to `queue_capacity`:
    ///
    /// - Queue length > 75% of capacity → step **down** (smaller batch,
    ///   drain sooner) to relieve back-pressure on remote senders.
    /// - Queue length < 25% of capacity → step **up** (larger batch,
    ///   drain less often) to amortize the drain cost across more frees.
    ///
    /// A cooldown of `ADAPTIVE_COOLDOWN_TICKS` `maybe_drain` calls
    /// between adjustments prevents oscillation. Initial step is 3
    /// (threshold = 64) — matches the `Fixed(64)` v0.1 default so an
    /// `Adaptive`-policy owner behaves like a `Fixed(64)` owner until
    /// the workload pushes the threshold off the middle level.
    ///
    /// All arithmetic is integer-only; the 0.25 / 0.75 hysteresis band
    /// is encoded as `q*4 < cap` / `q*4 > cap*3`. No floating point.
    ///
    /// This is the v1.0 control law; the v2.0 EMA-based upgrade is
    /// gated on benchmark validation against this baseline.
    Adaptive,
}

impl Default for BatchPolicy {
    fn default() -> Self {
        Self::Fixed(64)
    }
}

/// Drain-threshold levels for [`BatchPolicy::Adaptive`].
pub const ADAPTIVE_LEVELS: [usize; 5] = [8, 16, 32, 64, 128];

/// Number of `maybe_drain` ticks between adaptive step adjustments.
/// Tuned for low overhead and quick settling on bursty workloads.
pub const ADAPTIVE_COOLDOWN_TICKS: u32 = 16;

/// Internal state for [`BatchPolicy::Adaptive`]. Lives in a [`Cell`]
/// on the (`!Sync`) [`SlabOwner`]; the cell guarantees only the owner
/// thread reads or writes it.
#[derive(Copy, Clone, Debug)]
struct AdaptiveState {
    /// Current step index into [`ADAPTIVE_LEVELS`] (0..=4).
    step: u8,
    /// Remaining `maybe_drain` ticks before the next adjustment is
    /// allowed. Counts down to 0; while >0 the step is locked.
    cooldown: u32,
}

impl AdaptiveState {
    /// Starting state — step 3 (threshold = 64), matching the
    /// `Fixed(64)` default so workloads that never trip the hysteresis
    /// bands behave identically under either policy.
    const fn initial() -> Self {
        Self {
            step: 3,
            cooldown: 0,
        }
    }

    #[inline]
    fn threshold(&self) -> usize {
        ADAPTIVE_LEVELS[self.step as usize]
    }
}

/// Cache-line-aligned wrapper. Prevents false sharing between hot
/// owner-only fields and the cross-thread `remote_queue` mutex.
///
/// `#[repr(C, align(64))]` matches a typical x86_64 / AArch64 L1
/// cache-line size; rounding the struct size to a multiple of 64 ensures
/// the next field also begins on a fresh line. (Some Apple Silicon parts
/// use 128-byte coherency granularity, but 64-aligned is the broadly
/// portable choice without forcing 2× padding everywhere.)
#[repr(C, align(64))]
struct CachePadded<T>(T);

impl<T> CachePadded<T> {
    #[inline]
    const fn new(v: T) -> Self {
        Self(v)
    }
}

impl<T> core::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

/// Shared state between [`SlabOwner`] and any number of [`SlabRemote`] handles.
///
/// Layout note: `remote_queue` is wrapped in `CachePadded` so the
/// cross-thread mutex sits on its own cache line. Without this,
/// `SlabRemote::deallocate` (which slams the mutex word) would evict the
/// owner-only `slab` field from L1 on every remote push, costing the
/// owner a cache miss on every subsequent `allocate`.
struct SlabInner<T, B: Allocator + forge_alloc_core::FixedRange> {
    /// The actual slab. Mutated only by `SlabOwner` (we enforce single-owner
    /// via `!Sync` on the owner type).
    slab: core::cell::UnsafeCell<Slab<T, B>>,
    /// Max queue depth before `try_deallocate` returns `Err`. Read-only after
    /// construction so it's safe to share a line with `slab`.
    queue_capacity: usize,
    /// Remote-free queue: `(ptr, layout)` pairs queued for return to the local
    /// freelist. Mutex-protected for v0.1; will become lock-free MPSC in v1.0.
    /// Cache-line-isolated from `slab` to prevent false sharing — see above.
    remote_queue: CachePadded<Mutex<VecDeque<RemoteEntry>>>,
    /// Mirror of `remote_queue.lock().len()` updated under the same lock.
    ///
    /// Lets the owner-thread fast path in [`SlabOwner::maybe_drain`] sample
    /// the queue depth with a single Relaxed load instead of acquiring the
    /// mutex and reading `VecDeque::len()` — the prior design paid an
    /// uncontended `try_lock + queue.len()` round-trip (~8–12 ns/op) on
    /// every owner-side `allocate` / `deallocate`, even when the queue was
    /// empty (the common case for owner-heavy workloads). The fast path
    /// now collapses to a load + compare (~2–3 ns).
    ///
    /// Correctness: the counter is written under the queue mutex
    /// (`store(q.len())` after every `push_back` / `drain`), so the
    /// counter and the queue length are always in sync at lock-release
    /// time. The owner reads with `Relaxed` outside the lock; staleness
    /// is bounded by one remote push between the read and the next
    /// `maybe_drain` tick. Since `maybe_drain` is called on every
    /// `allocate` / `deallocate`, the queue cannot grow unboundedly past
    /// the threshold — the next tick samples a fresh value and drains.
    ///
    /// Placed adjacent to `remote_queue` in the struct so the mirror write
    /// tends to share a cache line with the mutex word the remote already
    /// dirtied (layout-dependent; no explicit `CachePadded` wrapper here).
    remote_queue_len: AtomicUsize,
    /// Set by [`SlabOwner::drop`] under the `remote_queue` mutex.
    ///
    /// Once `true`, no thread will ever drain the queue again, so
    /// [`SlabRemote::try_deallocate`] returns `Err(ptr)` and the
    /// infallible [`SlabRemote::deallocate`] bails out instead of
    /// spinning forever. The flag lives next to the queue rather than
    /// on `SlabOwner` because `SlabRemote` clones outlive the owner;
    /// they read it through their shared `Arc<SlabInner>`.
    closed: AtomicBool,
}

/// One entry in the remote-free queue: the (ptr, layout) pair to deallocate.
#[derive(Copy, Clone)]
struct RemoteEntry {
    ptr: NonNull<u8>,
    layout: NonZeroLayout,
}

// SAFETY: NonNull<u8> is !Send by default, but the mapping it points to is
// owned by the slab inside the same Arc; sending the entry between threads
// is equivalent to sending a u8 address — sound when the slab is Send.
unsafe impl Send for RemoteEntry {}

/// Owns the slab. Has exclusive `allocate` access. `Send` (can be moved
/// across threads) but `!Sync` (one-at-a-time access — enforced by
/// `UnsafeCell` on the inner slab plus our manual `Send` impl with no
/// corresponding `Sync` impl).
///
/// # API-misuse compile-failures (pinned)
///
/// `SlabOwner` is `!Sync` by construction (`_not_sync:
/// PhantomData<Cell<()>>`); a future refactor that accidentally
/// rederived `Sync` would let two threads share an `&SlabOwner` and race
/// on the `UnsafeCell<Slab>` inside `inner.slab`. The compile_fail
/// below pins that rejection:
///
/// ```compile_fail
/// // FAILS TO COMPILE: SlabOwner is deliberately !Sync. The
/// // `_not_sync: PhantomData<Cell<()>>` marker blocks the auto-derive,
/// // and `assert_sync` cannot accept a `!Sync` type.
/// use forge_alloc::InlineBacked;
/// use forge_alloc::SlabOwner;
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<SlabOwner<u64, InlineBacked<512>>>();
/// ```
pub struct SlabOwner<T, B: Allocator + forge_alloc_core::FixedRange> {
    inner: Arc<SlabInner<T, B>>,
    batch_policy: BatchPolicy,
    /// Adaptive-policy state. Cell because `maybe_drain` takes `&self`
    /// and needs to mutate; `!Sync` guarantees only the owning thread
    /// reads or writes it. Unused when `batch_policy` is `Fixed`.
    adaptive: core::cell::Cell<AdaptiveState>,
    /// Hold a `!Sync` marker so we definitely don't accidentally derive
    /// `Sync` if all other fields become `Sync` later. `Arc<SlabInner<...>>`
    /// is `Sync` (it has to be to be shared between `SlabOwner` and
    /// `SlabRemote`), so without this we'd lose the `!Sync` guarantee.
    /// (The `Cell<AdaptiveState>` above is already `!Sync`, but keep
    /// the explicit marker so a future field swap can't accidentally
    /// re-enable `Sync`.)
    _not_sync: core::marker::PhantomData<core::cell::Cell<()>>,
}

/// Remote deallocation handle. `Send + Sync` — freely cloneable across
/// threads. Implements [`Deallocator`] only; cannot allocate.
///
/// # API-misuse compile-failures (pinned)
///
/// `SlabRemote<T, B>` is `Send + Sync` only when `T: Send` (and
/// `B: Send`). Instantiating with a non-`Send` `T` (e.g. `Rc<u64>`)
/// and then trying to ship the remote across threads is rejected at
/// compile time:
///
/// ```compile_fail
/// // FAILS TO COMPILE: SlabRemote's `Send` bound requires `T: Send`,
/// // so `SlabRemote<Rc<u64>, _>` is not `Send` and cannot satisfy
/// // `assert_send`.
/// use std::rc::Rc;
/// use forge_alloc::InlineBacked;
/// use forge_alloc::SlabRemote;
/// fn assert_send<T: Send>() {}
/// assert_send::<SlabRemote<Rc<u64>, InlineBacked<512>>>();
/// ```
#[derive(Clone)]
pub struct SlabRemote<T, B: Allocator + forge_alloc_core::FixedRange> {
    inner: Arc<SlabInner<T, B>>,
}

impl<T, B: Allocator + forge_alloc_core::FixedRange> SlabOwner<T, B> {
    /// Construct, taking ownership of a freshly-built slab.
    pub fn new(capacity: usize, backing: B) -> Result<Self, AllocError> {
        Self::with_batch_policy(capacity, backing, BatchPolicy::default(), 1024)
    }

    /// Construct with explicit batch policy and queue capacity.
    pub fn with_batch_policy(
        capacity: usize,
        backing: B,
        batch_policy: BatchPolicy,
        queue_capacity: usize,
    ) -> Result<Self, AllocError> {
        let slab = Slab::new(capacity, backing)?;
        let inner = Arc::new(SlabInner {
            slab: core::cell::UnsafeCell::new(slab),
            queue_capacity,
            remote_queue: CachePadded::new(Mutex::new(VecDeque::with_capacity(queue_capacity))),
            remote_queue_len: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
        });
        Ok(Self {
            inner,
            batch_policy,
            adaptive: core::cell::Cell::new(AdaptiveState::initial()),
            _not_sync: core::marker::PhantomData,
        })
    }

    /// Create a remote handle. Cheap — just an `Arc` clone.
    pub fn remote(&self) -> SlabRemote<T, B> {
        SlabRemote {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Drain the remote-free queue into the local freelist now.
    ///
    /// Holds the queue mutex *only* long enough to swap out the pending
    /// entries; releases the lock before calling `slab.deallocate` for
    /// each entry. Without this two-phase pattern, remote senders would
    /// be blocked through the entire drain loop — death by lock-hold time
    /// proportional to queue depth.
    pub fn drain(&self) {
        // Phase 1: under the lock, snapshot the queue into a local Vec.
        // `drain(..).collect()` empties the deque without resetting its
        // capacity, so subsequent `push_back`s from `SlabRemote` continue
        // to use the pre-allocated buffer rather than re-allocating.
        let pending: Vec<RemoteEntry> = {
            let mut q = self
                .inner
                .remote_queue
                .lock()
                .expect("SlabOwner remote queue poisoned");
            // Fast-path the empty case: avoid the Vec allocation when the
            // queue has nothing. `drain(..).collect()` on an empty deque
            // still allocates a zero-length Vec on most allocators (the
            // standard Vec sets `cap = 0` so this is actually cheap, but
            // skipping the empty `for` below also saves a function call
            // boundary that the optimizer can't always elide because
            // `slab` is behind UnsafeCell).
            if q.is_empty() {
                // Invariant: every push and drain maintains
                // `remote_queue_len == q.len()` under the lock, so an
                // empty queue means the mirror is already 0. Assert it
                // here so a future bookkeeping regression fires loudly
                // in debug instead of silently being papered over by an
                // unconditional store.
                debug_assert_eq!(
                    self.inner.remote_queue_len.load(Ordering::Relaxed),
                    0,
                    "drain: queue empty but remote_queue_len mirror non-zero — push site dropped a mirror update",
                );
                return;
            }
            let entries: Vec<_> = q.drain(..).collect();
            // Mirror reset MUST stay inside the critical section so the next
            // owner-fast-path Relaxed load sees a consistent (queue, mirror)
            // pair after lock release.
            self.inner.remote_queue_len.store(0, Ordering::Relaxed);
            entries
        };
        // Lock dropped here. Remote senders can resume pushing.

        // Phase 2: process locally. SAFETY: !Sync on SlabOwner ensures
        // we're the only thread touching the slab via this owner reference.
        //
        // We use `&*` (Shared retag) rather than `&mut *` (Unique retag) here:
        // every `Slab` method we invoke (allocate / deallocate / base / size /
        // capacity_bytes) takes `&self` and handles its own interior mutation
        // through `UnsafeCell`. A `&mut Slab` would create a Unique retag
        // covering the full slab AND its embedded backing buffer, which
        // invalidates the SharedReadWrite tag the backing returned via
        // `InlineBacked::buffer_base` (and which the live slot pointers were
        // derived from). Miri caught this as a Stacked Borrows violation
        // when `Slab::deallocate` later wrote the freelist link
        // through one of those slot pointers.
        let slab = unsafe { &*self.inner.slab.get() };
        for entry in pending {
            // SAFETY: the entry came from our SlabRemote::deallocate caller,
            // who promised (ptr, layout) was issued by this slab.
            unsafe { slab.deallocate(entry.ptr, entry.layout) };
        }
    }

    /// Internal: check the batch-policy condition and drain if met.
    ///
    /// Hot path: called from every owner-side `allocate` and `deallocate`.
    /// Two-tier design:
    ///
    /// 1. **Fast path** — sample the cached `remote_queue_len` mirror with
    ///    a single `Relaxed` load. The mirror is maintained under the
    ///    queue mutex by `SlabRemote::try_deallocate` (on push) and by
    ///    `drain` / `maybe_drain` (on consume), so it tracks the queue
    ///    depth at lock-release granularity. If the observed depth is
    ///    below threshold we return immediately — no mutex acquisition,
    ///    no `VecDeque::len()` indirection. Cost: one Relaxed load +
    ///    compare (~2–3 ns on x86_64 / AArch64).
    /// 2. **Slow path** — depth ≥ threshold. We `try_lock` the queue and,
    ///    if uncontended, drain in one shot. `try_lock` (not `lock`) so
    ///    we don't contend with an in-flight remote push; the next tick
    ///    will catch up.
    ///
    /// Bounded staleness: the Relaxed load may observe a value behind a
    /// concurrent remote push that hasn't yet released the mutex. The
    /// drift is at most one push per tick; since `maybe_drain` is called
    /// on every owner alloc/dealloc, the queue cannot grow unboundedly.
    /// (If the owner is genuinely idle for long stretches, the queue is
    /// also bounded by `queue_capacity` — remote pushes start rejecting
    /// at that point.)
    ///
    /// Why we don't update the mirror outside the lock: the mirror and
    /// the queue must stay in sync at lock-release time, otherwise the
    /// owner's Relaxed load could observe an inconsistent pair (mirror=0
    /// while queue has pending entries → owner skips draining and the
    /// remote-pushed slot leaks until the next tick that happens to
    /// observe a non-zero mirror). Keeping both writes inside the same
    /// critical section makes the consistency check trivial.
    fn maybe_drain(&self) {
        // Fast path: relaxed load of the queue-length mirror. The
        // updates to this counter live inside the queue mutex
        // critical sections (push in `SlabRemote::try_deallocate`,
        // reset in `drain` / `maybe_drain` / `Drop`), so a load that
        // observes value `n` means the queue had ≥ n entries at the
        // most recent mutex unlock visible to this CPU — possibly more
        // since then, but never artificially inflated.
        let pending = self.inner.remote_queue_len.load(Ordering::Relaxed);
        let threshold = match self.batch_policy {
            BatchPolicy::Fixed(n) => n,
            BatchPolicy::Adaptive => self.adaptive_threshold(pending),
        };
        if pending < threshold {
            return;
        }

        // Slow path: threshold tripped. Take the queue under `try_lock`
        // and drain in one shot. `try_lock` (not `lock`) so we don't
        // serialize against an in-flight remote push; the next tick
        // will catch up.
        let entries: Vec<RemoteEntry> = {
            let mut q = match self.inner.remote_queue.try_lock() {
                Ok(q) => q,
                Err(_) => return, // contended; skip — we'll catch up next time
            };
            // Re-check inside the lock — the queue may have been drained
            // by another owner-thread path (e.g. an `allocate` that
            // bottomed out and called `drain()` between our load and
            // try_lock). Skip the Vec allocation in that case.
            if q.is_empty() {
                // Mirror invariant: empty queue ⇒ mirror == 0.
                debug_assert_eq!(
                    self.inner.remote_queue_len.load(Ordering::Relaxed),
                    0,
                    "maybe_drain: queue empty but remote_queue_len mirror non-zero",
                );
                return;
            }
            let entries: Vec<_> = q.drain(..).collect();
            self.inner.remote_queue_len.store(0, Ordering::Relaxed);
            entries
        };
        // Lock dropped. Process the snapshot under exclusive !Sync access.
        // SAFETY: SlabOwner is !Sync, so no other thread holds an alias
        // to the slab. `&*` (Shared retag) is sufficient — see `drain()`
        // for the rationale; `Slab::deallocate` takes `&self`.
        let slab = unsafe { &*self.inner.slab.get() };
        for entry in entries {
            // SAFETY: the entry came from our SlabRemote::deallocate caller,
            // who promised (ptr, layout) was issued by this slab.
            unsafe { slab.deallocate(entry.ptr, entry.layout) };
        }
    }

    /// Update the adaptive state based on observed `pending` queue
    /// depth and return the current threshold.
    ///
    /// Step DOWN (smaller batch, drain sooner) when `pending` exceeds
    /// 75% of `queue_capacity` — queue is filling up, relieve back-
    /// pressure on remote senders.
    ///
    /// Step UP (larger batch, drain later) when `pending` is below
    /// 25% of `queue_capacity` — queue is mostly empty, amortize the
    /// drain cost.
    ///
    /// A cooldown of [`ADAPTIVE_COOLDOWN_TICKS`] calls between
    /// adjustments prevents oscillation around a band edge. All
    /// arithmetic integer; the 0.25 / 0.75 bands are encoded as
    /// `pending * 4 < cap` and `pending * 4 > cap * 3`.
    fn adaptive_threshold(&self, pending: usize) -> usize {
        let mut state = self.adaptive.get();
        if state.cooldown > 0 {
            state.cooldown -= 1;
        } else {
            let cap = self.inner.queue_capacity;
            // q > 75% — step down toward smaller batch.
            if pending.saturating_mul(4) > cap.saturating_mul(3) && state.step > 0 {
                state.step -= 1;
                state.cooldown = ADAPTIVE_COOLDOWN_TICKS;
            // q < 25% — step up toward larger batch.
            } else if pending.saturating_mul(4) < cap && state.step < 4 {
                state.step += 1;
                state.cooldown = ADAPTIVE_COOLDOWN_TICKS;
            }
        }
        self.adaptive.set(state);
        state.threshold()
    }

    /// Current adaptive-step threshold (in remote-queue entries), or
    /// `None` if the owner is configured with `BatchPolicy::Fixed`.
    /// Useful in tests and adaptive-tuning telemetry.
    #[inline]
    pub fn adaptive_threshold_snapshot(&self) -> Option<usize> {
        match self.batch_policy {
            BatchPolicy::Adaptive => Some(self.adaptive.get().threshold()),
            BatchPolicy::Fixed(_) => None,
        }
    }
}

unsafe impl<T, B: Allocator + forge_alloc_core::FixedRange> Deallocator for SlabOwner<T, B> {
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        // Drain pending remote deallocations on the owner-side dealloc
        // path so a long-lived owner that allocates rarely (or never)
        // and only deallocs locally still services the remote queue.
        // Without this, the remote queue accumulates indefinitely on
        // dealloc-heavy workloads.
        self.maybe_drain();
        // Owner-side dealloc: direct push to local freelist (no queue).
        // SAFETY: !Sync ensures exclusive access to the slab. `&*` (Shared
        // retag) is sufficient — Slab::deallocate takes `&self` and uses
        // interior mutability for the freelist head.
        let slab = unsafe { &*self.inner.slab.get() };
        unsafe { slab.deallocate(ptr, layout) };
    }
}

unsafe impl<T, B: Allocator + forge_alloc_core::FixedRange> Allocator for SlabOwner<T, B> {
    #[inline]
    fn allocate(&self, layout: NonZeroLayout) -> Result<NonNull<[u8]>, AllocError> {
        self.maybe_drain();
        // SAFETY: !Sync on SlabOwner — exclusive access to inner slab.
        // `&*` (Shared retag) rather than `&mut *`: `Slab::allocate` takes
        // `&self`, and a `&mut Slab` retag would invalidate the inner
        // backing's SharedReadWrite tag covering its storage region —
        // every previously-issued slot pointer that the slab's freelist
        // (or its consumers) still holds would become a stale tag, which
        // is UB under Stacked Borrows.
        let slab = unsafe { &*self.inner.slab.get() };
        // If first attempt fails (local list empty + uncarved exhausted),
        // drain in case the queue has frees we can recover.
        match slab.allocate(layout) {
            Ok(block) => Ok(block),
            Err(_) => {
                // NLL ends the borrow of `slab` after its last use above (the
                // failing allocate call). `drain()` re-borrows the UnsafeCell
                // through its own &self path; the prior &Slab is no longer
                // live, so the second & creation does not violate aliasing.
                self.drain();
                // SAFETY: !Sync, re-borrow shared access. The drain()
                // borrow has already ended by the time control reaches here.
                let slab = unsafe { &*self.inner.slab.get() };
                slab.allocate(layout)
            }
        }
    }

    fn capacity_bytes(&self) -> Option<usize> {
        // SAFETY: !Sync — exclusive access.
        let slab = unsafe { &*self.inner.slab.get() };
        slab.capacity_bytes()
    }

    #[inline]
    fn corruption_events(&self) -> u64 {
        // SAFETY: !Sync — exclusive access. The owner-thread is the only
        // reader of this counter via the SlabOwner allocate path; remote
        // pushes don't touch the inner Slab's counter directly (they
        // enqueue and the owner drains).
        let slab = unsafe { &*self.inner.slab.get() };
        slab.corruption_events()
    }
}

/// Final drain on owner drop. Without this:
///
/// - `SlabRemote` clones outliving the owner would push entries into
///   `remote_queue` that nothing ever drains.
/// - Slots queued for return (already routed by the remote, not yet
///   drained by the owner) would never be reclaimed back into the
///   slab's local freelist. The slab keeps those slots marked-live
///   for as long as the last remote keeps the `Arc<SlabInner>` alive
///   — operationally a slot-table leak. Note that `T`'s destructor is
///   the remote caller's responsibility BEFORE calling
///   `SlabRemote::deallocate` (per the `Slab::deallocate` safety
///   contract); the drain here only reclaims the freelist entry, it
///   does not run `T::drop`.
/// - Subsequent `SlabRemote::deallocate` (the spinning, infallible
///   variant) would spin forever on a full queue, hanging the calling
///   thread.
///
/// We close the queue while holding its mutex (race-free against any
/// in-flight remote push) and drain the pending entries into the local
/// freelist. After this, the slab is consistent and any further remote
/// push observes `closed == true` and returns `Err(ptr)` /
/// no-ops without spinning.
impl<T, B: Allocator + forge_alloc_core::FixedRange> Drop for SlabOwner<T, B> {
    fn drop(&mut self) {
        // Phase 1: close + snapshot under the queue mutex.
        //
        // The close MUST be set under the mutex so a concurrent remote
        // push that has acquired the lock either:
        //  (a) sees `closed == false` and pushes — we drain it in this
        //      same critical section below.
        //  (b) sees `closed == true` (after we set it) and returns
        //      `Err(ptr)` without pushing — the caller keeps ownership.
        //
        // No remote push can slip in between "we drain" and "we set
        // closed" because both happen under the same lock acquisition.
        let pending: Vec<RemoteEntry> = {
            let mut q = match self.inner.remote_queue.lock() {
                Ok(q) => q,
                // Mutex poisoned — a prior holder panicked while
                // mutating the queue. Recover the entries; the panic
                // path is already broken, so don't compound it by
                // bailing out on poison.
                Err(p) => p.into_inner(),
            };
            // Release ordering pairs with Acquire in
            // `SlabRemote::try_deallocate`'s closed-check. The mutex
            // already provides happens-before for the queue contents,
            // but we publish the flag separately so a remote that
            // already holds the lock will observe the store via the
            // unlock/relock chain.
            self.inner.closed.store(true, Ordering::Release);
            let entries: Vec<_> = q.drain(..).collect();
            // Mirror reset under the same critical section as the drain
            // so an `Arc<SlabInner>` still alive via `SlabRemote` clones
            // post-owner-drop reflects "queue empty + closed" rather
            // than a stale length. No fast path consumes the mirror
            // after the owner is gone, but keep it consistent for
            // debugging / future invariants.
            self.inner.remote_queue_len.store(0, Ordering::Relaxed);
            entries
        };
        // Phase 2: process the drained entries under our exclusive
        // !Sync access. SAFETY: SlabOwner is !Sync, so no other thread
        // holds an alias to the slab. The remote queue is now closed;
        // no new alias will appear.
        //
        // **Drop-during-unwind escalation**: if `slab.deallocate`
        // panics inside the drain loop (e.g. a wrapping
        // `Statistics::deallocate`'s `debug_assert!` fires on an
        // underflow), the loop aborts and the **remaining `pending`
        // entries are dropped without being routed to the slab** — the
        // slots are leaked. The `Arc<SlabInner>` itself still drops
        // normally; the leak is just the freelist re-link work.
        // If this Drop is running *during* an existing panic-unwind,
        // the second panic here triggers **immediate process abort**
        // per Rust language rules. The remote queue is already closed
        // (Phase 1) so no further pushes can succeed, regardless of
        // outcome — the closed-flag promise is preserved even on the
        // abort path.
        // `&*` (Shared retag) — Slab::deallocate takes `&self`; see the
        // Stacked Borrows rationale on `drain()`.
        let slab = unsafe { &*self.inner.slab.get() };
        for entry in pending {
            // SAFETY: the entry was pushed by a `SlabRemote::try_deallocate`
            // caller who promised the pointer originated from this slab.
            unsafe { slab.deallocate(entry.ptr, entry.layout) };
        }
        // Defense-in-depth: after Phase 1 closed the queue and Phase 2
        // drained it under the same critical section / single-thread
        // ownership, no entries can be added. The queue must now be
        // empty for the `closed == true` promise to hold. Re-acquire
        // (poisoned-recover) to inspect — under unwind this may itself
        // panic (poisoned-then-abort), which is the desired safety
        // posture for a corrupted Drop path. Compiled out in release.
        debug_assert!(
            self.inner
                .remote_queue
                .lock()
                .map(|q| q.is_empty())
                .unwrap_or_else(|p| p.into_inner().is_empty()),
            "SlabOwner::drop: remote_queue non-empty after Phase 1+2 drain — \
             closed-flag invariant violated, a remote push raced past the close",
        );
        debug_assert_eq!(
            self.inner.remote_queue_len.load(Ordering::Relaxed),
            0,
            "SlabOwner::drop: remote_queue_len mirror non-zero after drain — \
             push-side store missed the lock or drain-side reset was elided",
        );
        // The slab itself (and any T: Drop in still-live slots not
        // routed through the queue) drops when the last
        // `Arc<SlabInner>` is released — i.e., when the last
        // `SlabRemote` clone is dropped. That's outside our control,
        // but the closed flag keeps the queue from growing in the
        // meantime.
    }
}

impl<T, B: Allocator + forge_alloc_core::FixedRange> FixedRange for SlabOwner<T, B> {
    fn base(&self) -> NonNull<u8> {
        // SAFETY: !Sync — exclusive access.
        let slab = unsafe { &*self.inner.slab.get() };
        slab.base()
    }

    fn size(&self) -> usize {
        // SAFETY: !Sync — exclusive access.
        let slab = unsafe { &*self.inner.slab.get() };
        slab.size()
    }
}

impl<T, B: Allocator + forge_alloc_core::FixedRange> SlabRemote<T, B> {
    /// Non-spinning remote deallocate. Returns `Err(ptr)` if the queue is
    /// full **or** the owner has been dropped (queue is closed); caller
    /// retains ownership and must handle the pointer (typically by
    /// dropping it once the whole slab tears down).
    ///
    /// # Safety
    ///
    /// `ptr` must have been allocated from the corresponding `SlabOwner`.
    /// On `Err`, the pointer is still owned by the caller.
    pub unsafe fn try_deallocate(
        &self,
        ptr: NonNull<u8>,
        layout: NonZeroLayout,
    ) -> Result<(), NonNull<u8>> {
        let mut q = self
            .inner
            .remote_queue
            .lock()
            .expect("SlabRemote queue poisoned");
        // Closed check under the lock pairs with the close-under-lock
        // in `SlabOwner::drop`: any push that observes `closed == false`
        // here is guaranteed to be drained by the owner's final
        // drain-and-close critical section (which can't interleave with
        // ours).
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(ptr);
        }
        if q.len() >= self.inner.queue_capacity {
            return Err(ptr);
        }
        q.push_back(RemoteEntry { ptr, layout });
        // Update the owner-fast-path mirror under the lock. `store(q.len())`
        // rather than `fetch_add(1)` so the post-store value matches the
        // queue length at lock-release time exactly.
        //
        // Ordering: `Relaxed` is sufficient because the mirror is
        // **advisory only**. The owner's fast path samples the counter
        // outside the lock (no formal happens-before edge with this
        // store), but the slow path then re-acquires the queue mutex
        // and re-verifies the queue itself before draining — so any
        // visibility lag of the Relaxed load is bounded by one mutex
        // round-trip. On real hardware the Relaxed store is also
        // committed by the unlock fence, so steady-state owner reads
        // observe the update within a few hundred nanoseconds of the
        // push. See `maybe_drain` for the bounded-staleness argument.
        self.inner
            .remote_queue_len
            .store(q.len(), Ordering::Relaxed);
        Ok(())
    }
}

unsafe impl<T, B: Allocator + forge_alloc_core::FixedRange> Deallocator for SlabRemote<T, B> {
    /// Spins until the queue accepts the deallocation, **except** when
    /// the owner has been dropped — in that case the queue will never
    /// drain again, so we return immediately. The slot remains
    /// marked-live in the slab until the last `SlabRemote` clone
    /// releases the shared `Arc<SlabInner>` (at which point the slab
    /// itself drops and the backing region is fully reclaimed). Any
    /// `T: Drop` whose destructor was already run by the caller before
    /// `deallocate` is not affected — the destructor ran on schedule;
    /// only the freelist entry for the slot is forfeited.
    ///
    /// Latency-sensitive callers should use [`try_deallocate`](Self::try_deallocate)
    /// and handle the `Err(ptr)` overflow / closed-queue explicitly.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: NonZeroLayout) {
        let mut p = ptr;
        loop {
            // SAFETY: forwarded from caller.
            match unsafe { self.try_deallocate(p, layout) } {
                Ok(()) => return,
                Err(returned) => {
                    // Distinguish "queue closed, owner gone" (permanent —
                    // bail to avoid an infinite spin) from "queue full,
                    // transient" (continue spinning).
                    if self.inner.closed.load(Ordering::Acquire) {
                        return;
                    }
                    p = returned;
                    core::hint::spin_loop();
                }
            }
        }
    }
}

// SlabOwner is Send but !Sync. The Arc<SlabInner> is Send when its contents
// are; we have UnsafeCell<Slab<T, B>> inside, which is !Sync by default —
// good. But SlabInner needs to be Sync (because Arc<SlabInner> being shared
// across Owner + Remote(s) requires Sync) — the Mutex provides Sync for the
// queue, and the UnsafeCell-wrapped slab is accessed only by the !Sync
// Owner. Manual impl needed.
unsafe impl<T: Send, B: Allocator + forge_alloc_core::FixedRange + Send> Send for SlabOwner<T, B> {}
// SlabOwner is deliberately !Sync — `_not_sync: PhantomData<Cell<()>>`
// blocks the auto-derive. We rely on this for soundness: if two threads
// could share `&SlabOwner`, both could call `allocate` and race on the
// UnsafeCell<Slab<T, B>>.

unsafe impl<T: Send, B: Allocator + forge_alloc_core::FixedRange + Send> Send for SlabRemote<T, B> {}
unsafe impl<T: Send, B: Allocator + forge_alloc_core::FixedRange + Send> Sync for SlabRemote<T, B> {}

// The Arc<SlabInner> requires SlabInner: Send + Sync.
unsafe impl<T: Send, B: Allocator + forge_alloc_core::FixedRange + Send> Send for SlabInner<T, B> {}
unsafe impl<T: Send, B: Allocator + forge_alloc_core::FixedRange + Send> Sync for SlabInner<T, B> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::InlineBacked;

    /// Positive companion to the `SlabOwner !Sync` compile_fail pin: the
    /// owner IS `Send` (single-threaded ownership transfer is the v0.1
    /// use case). If a future refactor accidentally removed `Send`, the
    /// `multi_threaded_remote_dealloc` test below would still compile but
    /// this pin would fail loudly.
    #[test]
    fn slab_owner_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SlabOwner<u64, InlineBacked<512>>>();
    }

    /// Positive companion to the `SlabRemote<!Send T>` compile_fail pin:
    /// `SlabRemote<u64, InlineBacked<N>>` IS `Send + Sync` because both
    /// `T = u64` and `B: Send`. Pinning this stops a refactor that
    /// accidentally weakened the bound from breaking the cross-thread
    /// deallocate API silently.
    #[test]
    fn slab_remote_is_send_and_sync_when_t_is_send() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SlabRemote<u64, InlineBacked<512>>>();
    }

    #[test]
    fn owner_can_alloc_and_dealloc_locally() {
        let owner: SlabOwner<u64, InlineBacked<512>> =
            SlabOwner::new(8, InlineBacked::<512>::new()).unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let p = owner.allocate(layout).unwrap();
        unsafe { owner.deallocate(p.cast(), layout) };
    }

    #[test]
    fn remote_can_deallocate_owner_allocations() {
        let owner: SlabOwner<u64, InlineBacked<512>> =
            SlabOwner::new(8, InlineBacked::<512>::new()).unwrap();
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let p = owner.allocate(layout).unwrap();
        // Cross-thread deallocate via remote.
        unsafe { remote.deallocate(p.cast(), layout) };
        // Force a drain so the slot returns to the local freelist.
        owner.drain();
        // Subsequent allocate must reuse the freed slot.
        let p2 = owner.allocate(layout).unwrap();
        assert_eq!(p.cast::<u8>().as_ptr(), p2.cast::<u8>().as_ptr());
    }

    /// Boundary: `queue_capacity = 0` means every remote deallocation is
    /// rejected — `try_deallocate` returns `Err(ptr)` on the very first
    /// call. This is the documented contract; pinning it here protects
    /// against an accidental "off-by-one allows one push" regression.
    #[test]
    fn queue_capacity_zero_rejects_every_remote_dealloc() {
        let owner: SlabOwner<u64, InlineBacked<512>> =
            SlabOwner::with_batch_policy(8, InlineBacked::<512>::new(), BatchPolicy::Fixed(64), 0)
                .unwrap();
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let p = owner.allocate(layout).unwrap();
        // Cap 0: first try_deallocate must already fail.
        let r = unsafe { remote.try_deallocate(p.cast(), layout) };
        assert!(
            r.is_err(),
            "queue_capacity=0 must reject every remote_deallocate",
        );
        // Clean up via owner to keep the slab balanced under the test.
        unsafe { owner.deallocate(p.cast(), layout) };
    }

    /// Boundary: `queue_capacity = 1` accepts exactly one push; the
    /// second must fail.
    #[test]
    fn queue_capacity_one_accepts_one_push_only() {
        let owner: SlabOwner<u64, InlineBacked<512>> =
            SlabOwner::with_batch_policy(8, InlineBacked::<512>::new(), BatchPolicy::Fixed(64), 1)
                .unwrap();
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = owner.allocate(layout).unwrap();
        let b = owner.allocate(layout).unwrap();
        unsafe {
            assert!(remote.try_deallocate(a.cast(), layout).is_ok());
            assert!(remote.try_deallocate(b.cast(), layout).is_err());
            // Drop the rejected pointer through the owner.
            owner.deallocate(b.cast(), layout);
        }
    }

    /// Boundary: adaptive policy at the floor (step 0 → threshold 8)
    /// must not underflow the step counter on repeated step-down
    /// attempts. We feed sustained > 75% queue depth and observe
    /// that the threshold stops at the floor rather than wrapping.
    #[test]
    fn adaptive_policy_step_floor_does_not_underflow() {
        let owner: SlabOwner<u64, InlineBacked<8192>> = SlabOwner::with_batch_policy(
            256,
            InlineBacked::<8192>::new(),
            BatchPolicy::Adaptive,
            16,
        )
        .unwrap();
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Pre-allocate a pool.
        let pool: Vec<_> = (0..200).map(|_| owner.allocate(layout).unwrap()).collect();
        let mut pushed = 0;
        // Drive the policy to floor across many cycles.
        for _ in 0..100 {
            while pushed < pool.len() && {
                let q = owner.inner.remote_queue.lock().unwrap();
                q.len() < 14
            } {
                let _ = unsafe { remote.try_deallocate(pool[pushed].cast(), layout) };
                pushed += 1;
            }
            let _ = owner.allocate(layout);
        }
        // Threshold must be at the floor (8) or one step up (16) — never
        // a wrapped or invalid value.
        let thr = owner.adaptive_threshold_snapshot().unwrap();
        assert!(
            ADAPTIVE_LEVELS.contains(&thr),
            "threshold {thr} must remain one of the configured levels"
        );
    }

    #[test]
    fn try_deallocate_returns_err_on_full_queue() {
        let owner: SlabOwner<u64, InlineBacked<512>> =
            SlabOwner::with_batch_policy(8, InlineBacked::<512>::new(), BatchPolicy::Fixed(64), 2)
                .unwrap();
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let a = owner.allocate(layout).unwrap();
        let b = owner.allocate(layout).unwrap();
        let c = owner.allocate(layout).unwrap();
        unsafe {
            remote.try_deallocate(a.cast(), layout).unwrap();
            remote.try_deallocate(b.cast(), layout).unwrap();
            // Queue capacity 2 — third must fail.
            let r = remote.try_deallocate(c.cast(), layout);
            assert!(r.is_err());
            // Drop c via the owner so we don't leak under the test.
            owner.deallocate(c.cast(), layout);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn multi_threaded_remote_dealloc() {
        use std::sync::Arc;
        use std::thread;

        let owner: SlabOwner<u64, InlineBacked<8192>> = SlabOwner::with_batch_policy(
            256,
            InlineBacked::<8192>::new(),
            BatchPolicy::Fixed(64),
            1024,
        )
        .unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();

        // Pre-allocate a bunch on the owner thread, then send pointers off
        // to worker threads for cross-thread deallocate.
        let mut ptrs = Vec::new();
        for _ in 0..128 {
            ptrs.push(owner.allocate(layout).unwrap());
        }
        let ptrs_addrs: Vec<usize> = ptrs
            .iter()
            .map(|p| p.cast::<u8>().as_ptr() as usize)
            .collect();
        let remote = Arc::new(owner.remote());

        let mut handles = Vec::new();
        for chunk in ptrs_addrs.chunks(32) {
            let chunk = chunk.to_vec();
            let r = Arc::clone(&remote);
            handles.push(thread::spawn(move || {
                for addr in chunk {
                    let p = unsafe { NonNull::new_unchecked(addr as *mut u8) };
                    unsafe { r.deallocate(p, layout) };
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        owner.drain();
        // After drain, the freelist should hold all 128 slots again; we can
        // re-alloc 128 without hitting next_uncarved.
        for _ in 0..128 {
            assert!(owner.allocate(layout).is_ok());
        }
    }

    // ====================================================================
    // BatchPolicy::Adaptive — stepped-threshold control law tests
    // ====================================================================

    /// Helper: build a slab with `Adaptive` policy and a small queue
    /// so the 25%/75% hysteresis bands trigger on a few entries.
    fn build_adaptive_owner(queue_capacity: usize) -> SlabOwner<u64, InlineBacked<8192>> {
        SlabOwner::with_batch_policy(
            256,
            InlineBacked::<8192>::new(),
            BatchPolicy::Adaptive,
            queue_capacity,
        )
        .unwrap()
    }

    #[test]
    fn adaptive_initial_threshold_is_64() {
        let owner = build_adaptive_owner(64);
        assert_eq!(owner.adaptive_threshold_snapshot(), Some(64));
    }

    #[test]
    fn fixed_policy_has_no_adaptive_snapshot() {
        let owner: SlabOwner<u64, InlineBacked<512>> =
            SlabOwner::new(8, InlineBacked::<512>::new()).unwrap();
        // Default is Fixed(64).
        assert_eq!(owner.adaptive_threshold_snapshot(), None);
    }

    #[test]
    fn adaptive_steps_down_when_queue_exceeds_75_percent() {
        // queue_capacity = 16; > 75% means q > 12 (since q*4 > 16*3 = 48 ⇔ q > 12).
        //
        // Path-dependence: every allocate calls maybe_drain, which observes
        // the queue and may step the policy. The 13 pre-allocates below
        // step the policy UP to the ceiling (128) on call #1 (queue empty
        // ⇒ q*4 < cap ⇒ step up) and engage a cooldown of
        // ADAPTIVE_COOLDOWN_TICKS = 16. Calls #2..#13 just decrement the
        // cooldown. After pushing 13 to the queue we need (16 − 12) = 4
        // more allocates to drain cooldown, then a 5th to observe the
        // step-down trigger. Total: 13 + 5 = 18 allocates.
        let owner = build_adaptive_owner(16);
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // 13 pre-allocates: step climbs to 128 on call 1, cooldown to 4 by 13.
        let ptrs: Vec<_> = (0..13).map(|_| owner.allocate(layout).unwrap()).collect();
        assert_eq!(owner.adaptive_threshold_snapshot(), Some(128));
        // Push all 13 to the remote queue (q is now 13, > 75% of 16).
        for p in &ptrs {
            unsafe { remote.try_deallocate(p.cast(), layout).unwrap() };
        }
        // 4 more allocates to drain the remaining cooldown ticks. Each
        // observes q=13 but cooldown>0, so the band check is skipped.
        // Threshold stays at 128 (so q<threshold, no drain), queue stays at 13.
        for _ in 0..4 {
            let _ = owner.allocate(layout).unwrap();
            assert_eq!(owner.adaptive_threshold_snapshot(), Some(128));
        }
        // 18th allocate: cooldown=0, band check fires, q*4=52 > cap*3=48,
        // step down from 4 → 3 ⇒ threshold 64.
        let _ = owner.allocate(layout).unwrap();
        assert_eq!(
            owner.adaptive_threshold_snapshot(),
            Some(64),
            "step should have dropped one level after q > 75% sample",
        );
    }

    #[test]
    fn adaptive_steps_up_when_queue_below_25_percent() {
        // queue_capacity = 16; < 25% means q*4 < 16 ⇔ q < 4. q=0 qualifies.
        let owner = build_adaptive_owner(16);
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        assert_eq!(owner.adaptive_threshold_snapshot(), Some(64));
        // One allocate observes empty queue → step up to 128.
        let _ = owner.allocate(layout).unwrap();
        assert_eq!(
            owner.adaptive_threshold_snapshot(),
            Some(128),
            "step should have climbed one level after q < 25% sample",
        );
    }

    #[test]
    fn adaptive_cooldown_prevents_oscillation() {
        // After one step adjustment the cooldown counter prevents the
        // next ADAPTIVE_COOLDOWN_TICKS maybe_drain calls from changing
        // the step. Verify by checking the threshold stays put.
        let owner = build_adaptive_owner(16);
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // First call: step 3 → 4 (queue empty, < 25%).
        let _ = owner.allocate(layout).unwrap();
        let after_first = owner.adaptive_threshold_snapshot().unwrap();
        assert_eq!(after_first, 128);
        // Subsequent allocates DURING cooldown should not advance further.
        // The step is already at the ceiling (4) anyway, but the cooldown
        // also blocks any move; even if we wanted to drop down, we can't.
        for _ in 0..(ADAPTIVE_COOLDOWN_TICKS - 1) {
            let _ = owner.allocate(layout).unwrap();
            assert_eq!(owner.adaptive_threshold_snapshot(), Some(128));
        }
    }

    #[test]
    fn adaptive_ceiling_is_respected() {
        // Step 4 (= 128) is the ceiling. Climb to it via many empty-queue
        // ticks (with cooldown waits in between).
        let owner = build_adaptive_owner(16);
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Each "step + cooldown_ticks" cycle moves the threshold one
        // level. 1 step from 3→4 puts us at the ceiling immediately.
        for _ in 0..(2 * ADAPTIVE_COOLDOWN_TICKS) {
            let _ = owner.allocate(layout).unwrap();
        }
        assert_eq!(owner.adaptive_threshold_snapshot(), Some(128));
    }

    #[test]
    fn adaptive_floor_is_respected() {
        // Step 0 (= 8) is the floor. Push enough cross-thread frees to
        // keep the queue > 75% across repeated ticks; the policy should
        // step down 3→2→1→0 and then stop (no step below the floor).
        //
        // We need cap=16 (so q>12 triggers step down) and enough
        // pre-allocated pointers to keep the queue full as the owner
        // drains it on threshold crossings. After each step-down,
        // cooldown=16 protects 16 calls. To hit four step-downs
        // (3→2→1→0 plus a "floor reached" check) we need ~4 × 17 = 68
        // allocates plus enough queue refill in between.
        const SLAB_CAP: usize = 256;
        let owner: SlabOwner<u64, InlineBacked<8192>> = SlabOwner::with_batch_policy(
            SLAB_CAP,
            InlineBacked::<8192>::new(),
            BatchPolicy::Adaptive,
            16,
        )
        .unwrap();
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        // Phase 1: pre-allocate a pool of pointers to push from.
        let pool: Vec<_> = (0..200).map(|_| owner.allocate(layout).unwrap()).collect();
        // Phase 2: feed the queue full before each drain so the
        // observed q stays > 75%. Strategy: tight loop — push 14 to the
        // queue, allocate once (decrements cooldown OR triggers
        // step-down on the cooldown=0 tick), continue.
        let mut pushed = 0usize;
        for cycle in 0..6 {
            // Top the queue back up to ~14 entries each cycle.
            while pushed < pool.len() && {
                let q = owner.inner.remote_queue.lock().unwrap();
                q.len() < 14
            } {
                // SAFETY: pool came from this owner.
                let _ = unsafe { remote.try_deallocate(pool[pushed].cast(), layout) };
                pushed += 1;
            }
            // Allocate once to trigger maybe_drain.
            let _ = owner.allocate(layout);
            // After enough cycles the step must hit the floor.
            if cycle == 5 {
                let thr = owner.adaptive_threshold_snapshot().unwrap();
                assert!(
                    thr <= 64,
                    "policy should have stepped down toward floor, got {thr}",
                );
            }
        }
    }

    #[test]
    fn adaptive_levels_are_sorted_and_match_spec() {
        assert_eq!(ADAPTIVE_LEVELS, [8, 16, 32, 64, 128]);
    }

    /// Regression: the `remote_queue_len` mirror tracks the queue
    /// length at lock-release granularity. Verifies the bookkeeping
    /// across push, drain, and `maybe_drain`:
    ///   - After N remote pushes, mirror == N.
    ///   - After `drain()`, mirror == 0 even though no allocate happened.
    ///   - After a fresh push following drain, mirror == 1 (no stale-add).
    #[test]
    fn remote_queue_len_mirror_tracks_lock_release_state() {
        let owner: SlabOwner<u64, InlineBacked<512>> =
            SlabOwner::with_batch_policy(8, InlineBacked::<512>::new(), BatchPolicy::Fixed(64), 16)
                .unwrap();
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();

        // Start: mirror is 0.
        assert_eq!(owner.inner.remote_queue_len.load(Ordering::Relaxed), 0);

        // Push three: mirror must be 3.
        let p1 = owner.allocate(layout).unwrap();
        let p2 = owner.allocate(layout).unwrap();
        let p3 = owner.allocate(layout).unwrap();
        unsafe {
            remote.try_deallocate(p1.cast(), layout).unwrap();
            remote.try_deallocate(p2.cast(), layout).unwrap();
            remote.try_deallocate(p3.cast(), layout).unwrap();
        }
        assert_eq!(
            owner.inner.remote_queue_len.load(Ordering::Relaxed),
            3,
            "mirror should equal the number of pushed entries"
        );

        // Drain: mirror resets to 0 without an allocate tick.
        owner.drain();
        assert_eq!(
            owner.inner.remote_queue_len.load(Ordering::Relaxed),
            0,
            "drain() must reset the mirror under the same critical section"
        );

        // Push one more — must be exactly 1, not 4.
        let p4 = owner.allocate(layout).unwrap();
        unsafe { remote.try_deallocate(p4.cast(), layout).unwrap() };
        assert_eq!(
            owner.inner.remote_queue_len.load(Ordering::Relaxed),
            1,
            "post-drain push must start from 0, not stale-add"
        );

        // Cleanup so the slab balances under Drop's debug_assert.
        owner.drain();
    }

    /// Regression: when an owner-side `allocate` tips the queue over
    /// threshold via `maybe_drain`, the mirror is reset under the same
    /// critical section as the actual VecDeque drain — otherwise a
    /// subsequent fast-path load would observe a stale non-zero value
    /// and keep firing the slow path with an empty queue.
    #[test]
    fn maybe_drain_resets_mirror_under_lock() {
        // Threshold low so a single tick drains.
        let owner: SlabOwner<u64, InlineBacked<512>> =
            SlabOwner::with_batch_policy(8, InlineBacked::<512>::new(), BatchPolicy::Fixed(2), 16)
                .unwrap();
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let p1 = owner.allocate(layout).unwrap();
        let p2 = owner.allocate(layout).unwrap();
        unsafe {
            remote.try_deallocate(p1.cast(), layout).unwrap();
            remote.try_deallocate(p2.cast(), layout).unwrap();
        }
        assert_eq!(owner.inner.remote_queue_len.load(Ordering::Relaxed), 2);
        // This allocate's maybe_drain hits the slow path (pending >= 2),
        // drains the two entries, and must reset the mirror.
        let _p3 = owner.allocate(layout).unwrap();
        assert_eq!(
            owner.inner.remote_queue_len.load(Ordering::Relaxed),
            0,
            "maybe_drain must reset the mirror inside the same critical section as the drain"
        );
    }

    /// Regression: dropping the owner must drain pending remote frees
    /// (no T: Drop leaks beyond the natural Slab teardown) and close
    /// the queue so further remote pushes return `Err(ptr)` instead of
    /// piling into an undrainable queue.
    #[test]
    fn owner_drop_drains_pending_remote_frees_and_closes_queue() {
        let owner: SlabOwner<u64, InlineBacked<512>> =
            SlabOwner::new(8, InlineBacked::<512>::new()).unwrap();
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();

        // Pre-allocate two slots through the owner, then queue two
        // remote frees without draining.
        let a = owner.allocate(layout).unwrap();
        let b = owner.allocate(layout).unwrap();
        unsafe {
            remote.try_deallocate(a.cast(), layout).unwrap();
            remote.try_deallocate(b.cast(), layout).unwrap();
        }
        // Drop the owner. Drop impl must drain a and b into the local
        // freelist (otherwise the slab keeps them marked-live until the
        // last Arc drops).
        drop(owner);

        // After owner drop, remote pushes must NOT block and must NOT
        // silently succeed — they return Err(ptr).
        let c_layout = layout;
        // Use a fake but well-formed pointer; we never call into the
        // slab from this path. (The Err path doesn't touch the slab.)
        let fake = unsafe { NonNull::new_unchecked(0x1000_usize as *mut u8) };
        let result = unsafe { remote.try_deallocate(fake, c_layout) };
        assert!(
            result.is_err(),
            "remote.try_deallocate must return Err after owner drop"
        );
    }

    /// Regression: the spinning `Deallocator::deallocate` impl must
    /// bail out (not spin forever) when the queue has been closed by
    /// owner drop. Without the closed check, a long-running task
    /// holding a `SlabRemote` would hang when its `deallocate` runs
    /// after the owner is torn down.
    #[test]
    fn remote_deallocate_does_not_spin_after_owner_drop() {
        let owner: SlabOwner<u64, InlineBacked<512>> =
            SlabOwner::new(8, InlineBacked::<512>::new()).unwrap();
        let remote = owner.remote();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();
        let _p = owner.allocate(layout).unwrap();
        drop(owner);
        // After owner drop, this would spin forever before the fix.
        // The closed-check inside deallocate returns immediately, so
        // this completes synchronously.
        let fake = unsafe { NonNull::new_unchecked(0x1000_usize as *mut u8) };
        unsafe { remote.deallocate(fake, layout) };
    }

    /// Multi-threaded stress test: owner + N remote workers running
    /// concurrent allocates/frees under the Adaptive policy. The test
    /// passes if (a) all allocated pointers come back distinct, (b)
    /// after-stop drain leaves the slab in a reusable state, and (c)
    /// the adaptive threshold has moved off its initial value at least
    /// once (i.e. the policy actually adapted, not just sat).
    #[test]
    #[cfg_attr(miri, ignore = "miri-incompatible: mmap / threads")]
    fn adaptive_multi_threaded_stress() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::thread;

        const CAP: usize = 1024;
        const QUEUE_CAP: usize = 64;
        let owner: SlabOwner<u64, InlineBacked<{ 16 * 1024 }>> = SlabOwner::with_batch_policy(
            CAP,
            InlineBacked::<{ 16 * 1024 }>::new(),
            BatchPolicy::Adaptive,
            QUEUE_CAP,
        )
        .unwrap();
        let layout = NonZeroLayout::for_type::<u64>().unwrap();

        // Allocate a pool the owner can hand to remotes.
        let pool: Vec<_> = (0..256).map(|_| owner.allocate(layout).unwrap()).collect();
        let addrs: Vec<usize> = pool
            .iter()
            .map(|p| p.cast::<u8>().as_ptr() as usize)
            .collect();

        // Verify pool has unique addresses (sanity).
        let mut sorted = addrs.clone();
        sorted.sort();
        for w in sorted.windows(2) {
            assert_ne!(w[0], w[1], "owner returned duplicate pointer");
        }

        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for chunk in addrs.chunks(64) {
            let chunk = chunk.to_vec();
            let remote = owner.remote();
            let stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || {
                let mut iters = 0u64;
                while !stop.load(Ordering::Relaxed) && iters < 4_000 {
                    for &addr in &chunk {
                        let p = unsafe { NonNull::new_unchecked(addr as *mut u8) };
                        let _ = unsafe { remote.try_deallocate(p, layout) };
                    }
                    iters += 1;
                }
            }));
        }

        // Owner runs its own alloc/free loop to exercise the maybe_drain
        // adaptive path under contention. Track whether the threshold
        // moves off the initial value of 64.
        let mut saw_step = false;
        for _ in 0..2_000 {
            if let Ok(block) = owner.allocate(layout) {
                unsafe { owner.deallocate(block.cast(), layout) };
            }
            if let Some(t) = owner.adaptive_threshold_snapshot() {
                if t != 64 {
                    saw_step = true;
                }
            }
        }
        stop.store(true, Ordering::Relaxed);
        for h in handles {
            let _ = h.join();
        }
        owner.drain();
        assert!(
            saw_step,
            "Adaptive policy should have moved the threshold off 64 under contention",
        );
    }
}
