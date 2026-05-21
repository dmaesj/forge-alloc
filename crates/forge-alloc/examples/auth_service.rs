//! `auth_service` — observable + UAF-resistant token allocator.
//!
//! Security-sensitive services need both *observability* (so SREs
//! can see when something's wrong) and *hardening* (so when
//! something IS wrong, the consequences are bounded). This example
//! stacks both into a single composed allocator for session tokens.
//!
//! ```text
//!   token allocation request
//!         │
//!         ▼
//!   ┌──────────────────────────────────────────────────────┐
//!   │  Statistics<Watermark<Quarantine<Slab<Token, _>>>>    │
//!   │                                                       │
//!   │  ┌─────────────────────────────────────────────┐     │
//!   │  │  Statistics: 6 atomic counters              │     │
//!   │  │   ─ total_allocations, total_deallocations   │     │
//!   │  │   ─ bytes_allocated, bytes_peak              │     │
//!   │  │   ─ failures, corruption_events (mirror)     │     │
//!   │  └─────────────────────────────────────────────┘     │
//!   │  ┌─────────────────────────────────────────────┐     │
//!   │  │  Watermark: capacity-utilisation thresholds  │     │
//!   │  │   ─ fires handler at 75% / 90% / OOM         │     │
//!   │  └─────────────────────────────────────────────┘     │
//!   │  ┌─────────────────────────────────────────────┐     │
//!   │  │  Quarantine<_, 4>: delays slot reuse         │     │
//!   │  │   ─ a freed token slot is held for 4 frees   │     │
//!   │  │     before returning to the inner slab        │     │
//!   │  │   ─ attacker race "free + immediate alloc" is │     │
//!   │  │     foiled across the quarantine window       │     │
//!   │  └─────────────────────────────────────────────┘     │
//!   │  ┌─────────────────────────────────────────────┐     │
//!   │  │  Slab<Token, MmapBacked>: fixed-capacity     │     │
//!   │  │     pool, O(1) alloc/dealloc                 │     │
//!   │  └─────────────────────────────────────────────┘     │
//!   └──────────────────────────────────────────────────────┘
//! ```
//!
//! Why this stack:
//! - **Statistics outermost**: SRE dashboards read the snapshot.
//!   Putting Statistics *outside* the security wrappers means its
//!   counters reflect user-visible request size, not the inflated
//!   inner layout. (Inverting this order makes the metric report
//!   sentinel overhead — see COMPATIBILITY_MATRIX item 17.)
//! - **Watermark below Statistics**: capacity thresholds are
//!   inner-layer-relative; placing Watermark right above the
//!   security wrappers lets it observe true slab utilisation.
//! - **Quarantine for UAF resistance**: a freed session token whose
//!   slot is immediately reallocated would let an attacker who held
//!   a stale pointer access a NEW session. Delaying reuse by 4
//!   frees forces the attacker to win a race they can't measure.
//! - **Plain Slab innermost**: actual allocation. The hardening
//!   layers are *defense in depth*, not replacements for sound
//!   freelist semantics.

use forge_alloc::{
    AllocStats, Allocator, Deallocator, MmapBacked, NonZeroLayout, NullHandler, Quarantine, Slab,
    Statistics, Watermark,
};
use std::sync::atomic::Ordering;

// ============================================================================
// Composition type alias
// ============================================================================

type TokenPool = Statistics<Watermark<Quarantine<Slab<Token, MmapBacked>, 4>, NullHandler>>;

// ============================================================================
// Domain types
// ============================================================================

#[repr(C)]
struct Token {
    session_id: u64,
    issued_at_unix_ms: u64,
    expires_at_unix_ms: u64,
    user_id: u32,
    flags: u32,
    // ... in a real auth service: cryptographic material redacted
    // by a custom Debug impl, or stored in a separately-allocated
    // region under HardenedSlab.
    _reserved: [u8; 32],
}

fn build_pool() -> TokenPool {
    Statistics::new(Watermark::new(
        Quarantine::<_, 4>::new(Slab::new(1024, MmapBacked::new(1 << 20).unwrap()).unwrap()),
        NullHandler,
    ))
}

fn report(name: &str, stats: &AllocStats) {
    println!(
        "  {name}: live={}, total_allocs={}, peak_bytes={}, corruption_events={}",
        stats.live_count(),
        stats.total_allocations.load(Ordering::Relaxed),
        stats.peak_bytes(),
        stats.corruption_events.load(Ordering::Relaxed),
    );
}

fn main() {
    println!("auth_service — observable + UAF-resistant token pool");
    println!("----------------------------------------------------");

    let pool = build_pool();
    let layout = NonZeroLayout::for_type::<Token>().unwrap();
    println!("pool: Statistics<Watermark<Quarantine<Slab<Token, Mmap>, 4>>>");

    // Issue 10 tokens.
    let tokens: Vec<_> = (0..10)
        .map(|i| {
            let p = pool.allocate(layout).unwrap();
            unsafe {
                let t = p.cast::<Token>().as_ptr();
                (*t).session_id = 0x1000 + i;
                (*t).user_id = i as u32;
                (*t).issued_at_unix_ms = 1_700_000_000_000 + i;
            }
            p
        })
        .collect();
    println!("\nissued 10 tokens:");
    report("post-issue", pool.stats());

    // Revoke 5 (typical "user logged out" pattern).
    for p in tokens.iter().take(5) {
        unsafe { pool.deallocate(p.cast(), layout) };
    }
    println!("\nrevoked 5 tokens (sent to Quarantine — slots NOT yet reclaimable):");
    report("post-revoke", pool.stats());

    // Issue 5 more. Because Quarantine has EPOCHS=4, only the FIRST
    // few revoked slots return to the inner slab; the rest are still
    // in the quarantine ring. New allocs come from the inner's
    // next_uncarved region.
    let _more: Vec<_> = (0..5).map(|_| pool.allocate(layout).unwrap()).collect();
    println!("\nissued 5 more after revoke (some come from new slots, not reused):");
    report("post-reissue", pool.stats());

    println!("\nin a production auth service:");
    println!("  - SRE dashboard scrapes pool.stats() every 10s");
    println!("  - alerts fire when corruption_events > 0 or live_count");
    println!("    approaches the slab capacity");
    println!("  - Quarantine's EPOCHS-deep ring forces attackers to win");
    println!("    a race they can't observe to recycle a freed token");
}
