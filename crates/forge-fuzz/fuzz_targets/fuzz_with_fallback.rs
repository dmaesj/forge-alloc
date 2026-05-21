#![no_main]
//! Fuzz target: WithFallback<InlineBacked<N>, System>. Verifies provenance
//! routing: every pointer issued must come back via either primary's no-op
//! deallocate or secondary's real deallocate, with no leak / no
//! double-free.

use forge_backing::{InlineBacked, System};
use forge_core::{Allocator, Deallocator, FixedRange, NonZeroLayout};
use forge_layout::WithFallback;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Op {
    Alloc { size: u16 },
    Free(u8),
}

#[derive(Arbitrary, Debug)]
struct Input {
    ops: Vec<Op>,
}

fuzz_target!(|input: Input| {
    let wf = WithFallback::new(InlineBacked::<8192>::new(), System);
    let mut live: Vec<(core::ptr::NonNull<u8>, NonZeroLayout)> = Vec::new();
    for op in input.ops.iter().take(512) {
        match op {
            Op::Alloc { size } => {
                let size = (*size as usize).max(1).min(512);
                let layout = NonZeroLayout::from_size_align(size, 1).unwrap();
                if let Ok(block) = wf.allocate(layout) {
                    let p = block.cast::<u8>();
                    live.push((p, layout));
                }
            }
            Op::Free(idx) => {
                if !live.is_empty() {
                    let idx = (*idx as usize) % live.len();
                    let (p, layout) = live.swap_remove(idx);
                    unsafe { wf.deallocate(p, layout) };
                }
            }
        }
    }
    // Drain any remaining live secondary allocations to satisfy the OS heap.
    // Primary allocations are no-op cleaned at arena drop.
    for (p, layout) in live.drain(..) {
        if !wf.primary().contains(p) {
            unsafe { wf.deallocate(p, layout) };
        }
    }
});
