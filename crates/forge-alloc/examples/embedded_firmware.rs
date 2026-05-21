//! `embedded_firmware` — sole `Slab` over inline storage.
//!
//! Demonstrates the `no_std`-compatible composition for bare-metal
//! firmware: a single fixed-capacity typed pool over a stack-backed
//! storage region. No mmap, no system allocator, no growth — the
//! memory budget is set at compile time.
//!
//! ```text
//!                  main() / firmware entry
//!                          │
//!                          ▼
//!     ┌────────────────────────────────────────┐
//!     │  Slab<SensorReading, InlineBacked<8K>> │
//!     │                                        │
//!     │   capacity = 256 readings × 32 B       │
//!     │   freelist LIFO; reuse on dealloc      │
//!     │   no allocator-internal heap touches   │
//!     └────────────────────────────────────────┘
//! ```
//!
//! Why this composition:
//! - **`no_std`-compatible**: `Slab` and `InlineBacked` both live in
//!   `forge-core` / `forge-backing` without the `std` feature. This
//!   example uses `println!` for clarity; in real firmware, replace
//!   with your platform's logging primitive (`defmt::info!`, RTT,
//!   serial, etc.).
//! - **InlineBacked**: storage is part of the struct. No mmap (no
//!   MMU required), no malloc, no growth. The capacity is the
//!   compile-time-fixed memory budget.
//! - **Slab**: typed pool with O(1) alloc + O(1) dealloc via LIFO
//!   freelist. Bounded, predictable, real-time safe (no syscalls).
//! - **No hardening**: in a controlled firmware environment with
//!   no remote code execution surface, security overhead is
//!   inappropriate. Catch corruption via hardware watchdog + ECC RAM
//!   instead.

use forge_alloc::{Allocator, Deallocator, InlineBacked, NonZeroLayout, Slab};

#[repr(C)]
struct SensorReading {
    timestamp_us: u64,
    sensor_id: u32,
    value_q16: i32,        // fixed-point Q16.16
    flags: u32,
    _reserved: [u8; 12],   // pad to 32 B total
}

const ARENA_BYTES: usize = 8 * 1024; // 8 KiB — fits 256 × 32 B
const POOL_CAPACITY: usize = 256;

type SensorPool = Slab<SensorReading, InlineBacked<ARENA_BYTES>>;

fn main() {
    println!("embedded_firmware — single Slab over 8 KiB inline storage");
    println!("---------------------------------------------------------");

    // Pool lives on the stack — in a real firmware, this would be a
    // `static` or in a fixed RAM region.
    let pool: SensorPool = Slab::new(POOL_CAPACITY, InlineBacked::<ARENA_BYTES>::new()).unwrap();
    println!("pool: {} slots × {} B = {} B used", POOL_CAPACITY, 32, POOL_CAPACITY * 32);

    let layout = NonZeroLayout::for_type::<SensorReading>().unwrap();

    // Simulate the sensor-acquisition loop: allocate, fill, hand off
    // (typically to a DMA queue or a ring buffer), then later free
    // when the consumer is done.
    let mut handles = [core::ptr::null_mut::<u8>(); 4];
    for (i, slot) in handles.iter_mut().enumerate() {
        let p = pool.allocate(layout).unwrap();
        // SAFETY: just-allocated, uninitialised slot of the right
        // size+alignment for SensorReading.
        unsafe {
            let r = p.cast::<SensorReading>().as_ptr();
            (*r).timestamp_us = 1_000_000 + i as u64;
            (*r).sensor_id = i as u32;
            (*r).value_q16 = (i as i32) << 16;
            (*r).flags = 0;
        }
        *slot = p.cast::<u8>().as_ptr();
        println!("  acquired reading #{i} at {:p}", *slot);
    }

    // Consumer side: process each reading and return the slot.
    for (i, &ptr) in handles.iter().enumerate() {
        let p = core::ptr::NonNull::new(ptr).unwrap();
        // SAFETY: ptr was issued by pool.allocate above with this layout;
        // we don't deref the SensorReading after this — its destructor
        // (POD type, no Drop) runs implicitly.
        unsafe { pool.deallocate(p, layout) };
        println!("  released reading #{i}");
    }
    println!("all slots returned to freelist; pool ready for next batch");
}
