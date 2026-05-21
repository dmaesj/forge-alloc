//! `web_service` — three-tier allocator architecture.
//!
//! The canonical "hot / warm / cold" composition for a service that
//! processes inbound requests against persistent per-connection state,
//! with occasional large allocations that fall outside the budgeted
//! region.
//!
//! ```text
//!   incoming request
//!         │
//!         ▼
//!   ┌─────────────────────────────────────────────┐
//!   │  HOT: per-request scratch                    │
//!   │  BumpArena<InlineBacked<64 KiB>>             │
//!   │  reset between requests                      │
//!   │  ~2 ns/alloc; everything freed in one swoop  │
//!   └─────────────────────────────────────────────┘
//!         │ (looks up connection)
//!         ▼
//!   ┌─────────────────────────────────────────────┐
//!   │  WARM: persistent connection state            │
//!   │  Slab<Connection, MmapBacked>                │
//!   │  fixed-capacity, LIFO freelist reuse          │
//!   │  ~53 ns/alloc; lifetime spans many requests   │
//!   └─────────────────────────────────────────────┘
//!         │ (occasional huge response body)
//!         ▼
//!   ┌─────────────────────────────────────────────┐
//!   │  COLD: oversized fallback                    │
//!   │  System (libc malloc)                        │
//!   │  unbounded; serves anything that doesn't fit │
//!   │  ~54 ns/alloc; only on the rare path         │
//!   └─────────────────────────────────────────────┘
//! ```
//!
//! Why this composition:
//! - **Per-request scratch on a bump arena**: parsing input, building
//!   intermediate structures, formatting output — all have a clear
//!   "lifetime ends at the response" boundary. Bump-and-reset is the
//!   optimal allocator shape for that.
//! - **Per-connection state on a slab**: connections come and go, but
//!   each lives for many requests. A typed slab with LIFO reuse keeps
//!   alloc latency bounded and zero-fragmentation.
//! - **System as the cold fallback**: occasional unusually large
//!   responses (a multi-MiB JSON dump) shouldn't blow the bump arena
//!   or push out other connections from the slab. Route them to
//!   System and pay the kernel malloc cost only on the cold path.
//! - **WithFallback at the connection layer**: routes by pointer
//!   provenance — `primary.contains(ptr)` decides who deallocates.

use forge_alloc::{
    Allocator, BumpArena, Deallocator, InlineBacked, MmapBacked, NonZeroLayout, Slab, System,
    WithFallback,
};

// ============================================================================
// Composition type aliases — read these first; they're the architecture.
// ============================================================================

/// Per-request scratch: 64 KiB on the stack, reset between requests.
type RequestScratch = BumpArena<InlineBacked<{ 64 * 1024 }>>;

/// Per-connection state: 1024 slots of `Connection`, MmapBacked.
type ConnectionPool = Slab<Connection, MmapBacked>;

/// Connection allocator with System fallback for outsized requests.
/// `WithFallback<P, S>` requires the primary to implement `FixedRange`
/// so it can route deallocations by pointer provenance.
type ConnAllocator = WithFallback<ConnectionPool, System>;

// ============================================================================
// Domain types
// ============================================================================

#[repr(C)]
struct Connection {
    id: u64,
    state: u32,
    last_request_ms: u64,
    _pad: [u8; 12],
}

fn main() {
    println!("web_service — three-tier allocator architecture");
    println!("-----------------------------------------------");

    // Build the warm path: a connection pool with System fallback.
    let conn_pool: ConnectionPool =
        Slab::new(1024, MmapBacked::new(1 << 20).unwrap()).unwrap();
    let conns: ConnAllocator = WithFallback::new(conn_pool, System);

    // Open three connections (typical client churn).
    let conn_layout = NonZeroLayout::for_type::<Connection>().unwrap();
    let conn_ptrs: Vec<_> = (0..3)
        .map(|i| {
            let p = conns.allocate(conn_layout).unwrap();
            unsafe {
                let c = p.cast::<Connection>().as_ptr();
                (*c).id = 1000 + i;
                (*c).state = 1;
                (*c).last_request_ms = 0;
            }
            println!("  warm: opened connection id={}", 1000 + i);
            p
        })
        .collect();

    // Per-request loop. Each request gets a fresh bump arena scratch;
    // reset between requests reclaims it in one cycle.
    let mut scratch = RequestScratch::new(InlineBacked::<{ 64 * 1024 }>::new()).unwrap();

    for req in 0..5 {
        println!("\nrequest #{req}:");

        // HOT: parse, format, intermediate work — all on scratch.
        let parse_buf_layout = NonZeroLayout::from_size_align(512, 8).unwrap();
        let _parse_buf = scratch.allocate(parse_buf_layout).unwrap();
        let _format_buf = scratch.allocate(parse_buf_layout).unwrap();
        println!("  hot: 2 scratch allocs ({} bytes total)", 2 * 512);

        // WARM: bump a per-connection counter (already allocated above).
        unsafe {
            let c = conn_ptrs[req % conn_ptrs.len()]
                .cast::<Connection>()
                .as_ptr();
            (*c).last_request_ms += 1;
            println!("  warm: connection id={} processed", (*c).id);
        }

        // COLD: every 3rd request happens to need a big response body.
        if req % 3 == 0 {
            let big_layout = NonZeroLayout::from_size_align(4 * 1024 * 1024, 8).unwrap();
            match conns.allocate(big_layout) {
                Ok(p) => {
                    println!("  cold: 4 MiB allocation routed to System (over slab budget)");
                    // SAFETY: just-issued pointer with matching layout.
                    unsafe { conns.deallocate(p.cast(), big_layout) };
                }
                Err(_) => println!("  cold: 4 MiB allocation rejected"),
            }
        }

        // End-of-request: reset the bump arena. Everything we
        // allocated for this request is now reclaimed.
        scratch.reset();
        println!("  scratch reset; ready for next request");
    }

    // Tear down connections at the warm tier.
    for (i, p) in conn_ptrs.into_iter().enumerate() {
        unsafe { conns.deallocate(p.cast(), conn_layout) };
        println!("\nwarm: closed connection #{i}");
    }
}
