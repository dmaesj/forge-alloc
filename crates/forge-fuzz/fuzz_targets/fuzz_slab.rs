#![no_main]
//! Fuzz target: Slab<u64, InlineBacked<N>, NoProtection> under arbitrary
//! alloc/free interleavings. Verifies invariant 3 (no overlap) and exercises
//! the freelist push/pop machinery looking for double-free / use-after-free
//! detection via the MAC verify (with NoProtection, MAC is constant 0).

use forge_alloc::InlineBacked;
use forge_alloc::{Allocator, Deallocator, NoProtection, NonZeroLayout};
use forge_alloc::Slab;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Op {
    Alloc,
    Free(u8), // index into "live" Vec
}

#[derive(Arbitrary, Debug)]
struct Input {
    ops: Vec<Op>,
}

fuzz_target!(|input: Input| {
    let s: Slab<u64, InlineBacked<8192>, NoProtection> =
        match Slab::new(64, InlineBacked::<8192>::new()) {
            Ok(s) => s,
            Err(_) => return,
        };
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let mut live: Vec<core::ptr::NonNull<u8>> = Vec::new();
    for op in input.ops.iter().take(512) {
        match op {
            Op::Alloc => {
                if let Ok(block) = s.allocate(layout) {
                    let p = block.cast::<u8>();
                    for q in &live {
                        assert_ne!(p.as_ptr(), q.as_ptr(), "duplicate live ptr");
                    }
                    live.push(p);
                }
            }
            Op::Free(idx) => {
                if !live.is_empty() {
                    let idx = (*idx as usize) % live.len();
                    let p = live.swap_remove(idx);
                    unsafe { s.deallocate(p, layout) };
                }
            }
        }
    }
});
