//! `cli_batch` — single bump arena, drop at end.
//!
//! The simplest possible composition: one allocator, no deallocation,
//! everything freed when `main` returns. Appropriate for short-lived
//! CLI tools, build scripts, one-shot batch processors — anything
//! that runs to completion and exits.
//!
//! ```text
//!                        main()
//!                          │
//!                          ▼
//!   ┌────────────────────────────────────────────────┐
//!   │     BumpArena<InlineBacked<256 KiB>>           │
//!   │                                                │
//!   │   all allocs ──► bump cursor (~2 ns/op)        │
//!   │   no deallocs ──► all reclaimed at process exit│
//!   └────────────────────────────────────────────────┘
//! ```
//!
//! Why this composition:
//! - **InlineBacked**: the 64 KiB lives on the stack of `main`, no
//!   mmap syscall, no heap allocation. The OS reclaims it when the
//!   process exits. (Larger arenas — 256 KiB+ — should use
//!   `MmapBacked` instead; Windows defaults to a 1 MiB thread stack
//!   and large inline backings can overflow it in debug builds where
//!   copy elision is unreliable.)
//! - **BumpArena**: no per-allocation bookkeeping. The cursor is a
//!   single atomic-free integer. Every alloc is one add + one bounds
//!   check.
//! - **No hardening**: a one-shot tool has no attack surface worth
//!   defending — the process dies in seconds. Don't pay for what you
//!   don't need.

use forge_alloc::{Allocator, BumpArena, InlineBacked, NonZeroLayout};

const ARENA_SIZE: usize = 64 * 1024; // 64 KiB — fits Windows 1 MiB default stack with headroom

type Arena = BumpArena<InlineBacked<ARENA_SIZE>>;

fn main() {
    println!("cli_batch — single bump arena, drop at end");
    println!("------------------------------------------");

    // The arena lives on the stack of main.
    let arena = Arena::new(InlineBacked::<ARENA_SIZE>::new()).unwrap();
    println!("arena: 64 KiB inline-backed, capacity {} bytes", arena.capacity());

    // Allocate a series of small buffers — typical CLI workload
    // (parsing a config file, building a string of output, etc.).
    let layout = NonZeroLayout::from_size_align(64, 8).unwrap();
    let mut total = 0usize;
    for i in 0..512 {
        match arena.allocate(layout) {
            Ok(_p) => {
                total += 64;
                if i % 100 == 0 {
                    println!("  alloc #{i}: total used = {total} bytes");
                }
            }
            Err(_) => {
                println!("  arena exhausted at alloc #{i}");
                break;
            }
        }
    }
    println!("done. arena drops with main(); OS reclaims stack.");
    // No explicit deallocs anywhere. The arena's Drop runs on scope
    // exit; the stack-backed storage is reclaimed by `main`'s return.
}
