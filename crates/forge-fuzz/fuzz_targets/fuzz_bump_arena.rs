#![no_main]
//! Fuzz target: BumpArena<InlineBacked<N>> under an arbitrary sequence of
//! allocation requests interleaved with resets. Verifies the no-overlap
//! invariant: no two live allocations overlap.

use forge_alloc::InlineBacked;
use forge_alloc::{Allocator, NonZeroLayout};
use forge_alloc::BumpArena;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Op {
    Alloc { size: u16, align_log: u8 },
    Reset,
}

#[derive(Arbitrary, Debug)]
struct Input {
    ops: Vec<Op>,
}

fuzz_target!(|input: Input| {
    let mut arena = match BumpArena::new(InlineBacked::<8192>::new()) {
        Ok(a) => a,
        Err(_) => return,
    };
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for op in input.ops.iter().take(256) {
        match op {
            Op::Alloc { size, align_log } => {
                let size = (*size as usize).max(1).min(256);
                // Mask to 3 bits: align_log in 0..=7, so align in {1, 2, 4, 8, 16, 32, 64, 128}.
                let align = 1usize << ((*align_log as usize) & 7);
                let layout = match NonZeroLayout::from_size_align(size, align) {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                if let Ok(block) = arena.allocate(layout) {
                    let start = block.cast::<u8>().as_ptr() as usize;
                    let end = start + block.len();
                    assert_eq!(start % align, 0, "alignment violated");
                    assert!(end - start >= size, "size too small");
                    for (s, e) in &ranges {
                        assert!(end <= *s || start >= *e, "overlap with live range");
                    }
                    ranges.push((start, end));
                }
            }
            Op::Reset => {
                arena.reset();
                ranges.clear();
            }
        }
    }
});
