//! Compile-only smoke test for `forge-layout` under `--no-default-features`.
//! Under no_std, `BumpArena`, `Slab`, `WithFallback`, and `SharedBumpArena`
//! (when atomics are available) must compile and be nameable.

#![no_std]

use forge_alloc::InlineBacked;
use forge_alloc::{AllocError, Allocator, NoProtection, NonZeroLayout};
use forge_alloc::{BumpArena, BumpDeallocator, Slab, WithFallback};

#[cfg(target_has_atomic = "ptr")]
use forge_alloc::SharedBumpArena;

fn _bump_compose() -> Result<(), AllocError> {
    let arena = BumpArena::new(InlineBacked::<256>::new())?;
    let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
    let _ = arena.allocate(layout)?;
    Ok(())
}

fn _slab_compose() -> Result<(), AllocError> {
    let s: Slab<u64, InlineBacked<512>, NoProtection> = Slab::new(8, InlineBacked::<512>::new())?;
    let layout = NonZeroLayout::for_type::<u64>().unwrap();
    let _ = s.allocate(layout)?;
    Ok(())
}

fn _fallback_compose() -> Result<(), AllocError> {
    let wf = WithFallback::new(
        InlineBacked::<128>::new(),
        InlineBacked::<128>::new(), // secondary that has FixedRange too — uncommon but valid
    );
    let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
    let _ = wf.allocate(layout)?;
    Ok(())
}

#[cfg(target_has_atomic = "ptr")]
fn _shared_bump_compose() -> Result<(), AllocError> {
    let arena = SharedBumpArena::new(InlineBacked::<256>::new())?;
    let layout = NonZeroLayout::from_size_align(8, 8).unwrap();
    let _ = arena.allocate(layout)?;
    Ok(())
}

// Surface checks: BumpDeallocator is nameable and ZST.
const _BD_SIZE_ZERO: () = {
    assert!(core::mem::size_of::<BumpDeallocator<'static>>() == 0);
};

#[test]
fn smoke() {}
