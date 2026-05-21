//! `db_executor` — per-query arena + buffer pool + hardened audit log.
//!
//! Database query executor pattern: each query gets its own arena
//! (cleared on completion), shared page-pool resources, and a
//! security-hardened audit log because regulators want tamper-evident
//! query records.
//!
//! ```text
//!   query arrives
//!         │
//!         ▼
//!   ┌─────────────────────────────────────────────────────┐
//!   │  PER-QUERY ARENA: BumpArena<MmapBacked>             │
//!   │  cleared between queries via reset()                 │
//!   │  ~2 ns/alloc; everything (plan, joins, sort scratch) │
//!   │   in one address range, freed in one cycle           │
//!   └─────────────────────────────────────────────────────┘
//!         │ (touches buffer pool for shared data)
//!         ▼
//!   ┌─────────────────────────────────────────────────────┐
//!   │  BUFFER POOL: Slab<Page, MmapBacked>                │
//!   │  fixed slot count; page-sized blocks                │
//!   │  LIFO reuse maximises cache locality                │
//!   └─────────────────────────────────────────────────────┘
//!         │ (writes audit record on commit)
//!         ▼
//!   ┌─────────────────────────────────────────────────────┐
//!   │  AUDIT LOG: HardenedSlab<AuditEntry>                │
//!   │  Slab<AuditEntry, GuardPage<SplitMetadata<Mmap>>>   │
//!   │  guard pages around data; metadata at separate      │
//!   │   virtual address; freelist links MAC'd             │
//!   │  any linear overflow that touches audit state traps │
//!   │   with SIGSEGV before bookkeeping can be corrupted  │
//!   └─────────────────────────────────────────────────────┘
//! ```
//!
//! Why this composition:
//! - **Per-query arena**: a query's working state is bounded by the
//!   query's lifetime. Allocating from a per-query arena means the
//!   query plan, hash tables, sort scratch, intermediate join
//!   results are all reclaimed in one operation when the query
//!   completes. No tracking individual heap allocations.
//! - **Buffer pool slab**: pages are shared across queries (one
//!   query may read a page another query brought into memory).
//!   A typed slab with O(1) alloc/dealloc keeps the buffer manager
//!   simple.
//! - **HardenedSlab for audit**: audit records are compliance-
//!   relevant; their integrity must survive even an attacker who
//!   has corrupted other parts of the process. Guard pages trap
//!   linear overflows; split metadata moves the bookkeeping out
//!   of the same virtual neighborhood as the data; the freelist
//!   MAC catches forged links. Per-op cost is ~115 ns over a bare
//!   Slab — fine for audit (rare path), unacceptable for buffer
//!   pool (hot path).

use forge_alloc::{
    Allocator, BumpArena, Deallocator, GuardPage, HardenedSlab, MmapBacked, NonZeroLayout, Slab,
    SplitMetadata,
};

// ============================================================================
// Composition type aliases
// ============================================================================

type QueryArena = BumpArena<MmapBacked>;
type BufferPool = Slab<Page, MmapBacked>;
type AuditLog = HardenedSlab<AuditEntry>;

// ============================================================================
// Domain types
// ============================================================================

const PAGE_SIZE: usize = 8 * 1024;

#[repr(C, align(8))]
struct Page {
    page_id: u64,
    table_id: u32,
    dirty: u32,
    _payload: [u8; PAGE_SIZE - 16],
}

#[repr(C)]
struct AuditEntry {
    timestamp_unix_ms: u64,
    query_hash: u64,
    user_id: u32,
    rows_affected: u32,
    _result_code: u32,
    _reserved: [u8; 12],
}

fn main() {
    println!("db_executor — per-query arena + buffer pool + hardened audit log");
    println!("----------------------------------------------------------------");

    // Buffer pool (warm-tier, shared across queries).
    let buffer_pool: BufferPool =
        Slab::new(64, MmapBacked::new(1 << 22).unwrap()).unwrap();
    let page_layout = NonZeroLayout::for_type::<Page>().unwrap();
    println!("buffer_pool: 64 pages × 8 KiB each = {} KiB", 64 * 8);

    // Audit log (cold-tier, security-hardened).
    let audit_backing = GuardPage::new(
        SplitMetadata::new(MmapBacked::new(4 * 1024 * 1024).unwrap(), 64 * 1024).unwrap(),
        4096,
    )
    .unwrap();
    let audit: AuditLog = Slab::new(1024, audit_backing).unwrap();
    let audit_layout = NonZeroLayout::for_type::<AuditEntry>().unwrap();
    println!("audit_log: HardenedSlab (guard pages + split metadata + 1024 slots)");

    // Process several queries.
    let queries = [
        ("SELECT", 42),
        ("UPDATE", 17),
        ("DELETE", 3),
        ("SELECT", 1024),
    ];

    for (i, (kind, rows)) in queries.iter().enumerate() {
        println!("\nquery #{i}: {kind} ({rows} rows affected)");

        // Per-query arena. Lives only for this query; reset at end.
        let mut arena: QueryArena = BumpArena::new(MmapBacked::new(1 << 20).unwrap()).unwrap();

        // Hot: query plan + intermediate scratch.
        let plan_layout = NonZeroLayout::from_size_align(2048, 16).unwrap();
        let _plan = arena.allocate(plan_layout).unwrap();
        let _scratch = arena.allocate(plan_layout).unwrap();
        println!("  arena: plan + scratch ({} bytes)", 2 * 2048);

        // Warm: borrow some pages from the buffer pool.
        let pages: Vec<_> = (0..2).map(|_| buffer_pool.allocate(page_layout).unwrap()).collect();
        unsafe {
            for (j, p) in pages.iter().enumerate() {
                let pg = p.cast::<Page>().as_ptr();
                (*pg).page_id = (i * 100 + j) as u64;
                (*pg).table_id = 7;
                (*pg).dirty = 0;
            }
        }
        println!("  buffer: borrowed {} pages from pool", pages.len());

        // Cold: write audit record on commit.
        let audit_p = audit.allocate(audit_layout).unwrap();
        unsafe {
            let e = audit_p.cast::<AuditEntry>().as_ptr();
            (*e).timestamp_unix_ms = 1_700_000_000_000 + i as u64 * 100;
            (*e).query_hash = 0xCAFE_BABE_DEAD_BEEFu64.wrapping_add(i as u64);
            (*e).user_id = 0x1000;
            (*e).rows_affected = *rows;
        }
        println!("  audit: wrote tamper-evident entry to HardenedSlab");

        // Return pages to pool. (Audit entries stay live for the
        // retention period; we don't deallocate them here.)
        for p in pages {
            unsafe { buffer_pool.deallocate(p.cast(), page_layout) };
        }

        // Reset the per-query arena. Plan + scratch reclaimed in one op.
        arena.reset();
        println!("  query arena reset");
    }
    println!("\nall queries processed; audit log retained for compliance");
}
