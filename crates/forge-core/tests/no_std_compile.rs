//! Compile-only smoke test: every type re-exported from `forge_core` under
//! `--no-default-features` is reachable and the trait surface composes.
//!
//! # CI gate — REQUIRED
//!
//! Running `cargo test --all-features` does **not** exercise the no_std
//! surface: it builds with `std` enabled, so the `#![no_std]` attribute
//! here is overridden by the host crate's std feature, and any
//! accidental `use std::…` introduced into `forge_core`'s public surface
//! will go unnoticed.
//!
//! CI **must** add the following step, separate from the default test
//! run, to actually validate the no_std surface:
//!
//! ```sh
//! cargo check -p forge-core --no-default-features --tests
//! cargo check -p forge-layout --no-default-features --tests
//! ```
//!
//! These commands verify that the surface compiles when `std` is not
//! linked. A run is not needed — `cargo check` is sufficient because the
//! file's `#[test] fn smoke` body is empty.
//!
//! On no_std targets without `std`, this file is the canonical surface
//! check.

#![no_std]

use forge_core::{
    AllocError, Deallocator, FixedRange, FreelistProtection, NoProtection, NonZeroLayout,
    OsBacked, ProtectFlags,
};

// Sanity: NoProtection signs/verifies under no_std.
fn _no_protection_works() {
    let p = NoProtection;
    let mac = p.sign(0, 0);
    let _ = p.verify(0, mac, 0);
}

// Sanity: NonZeroLayout::for_type compiles under no_std.
fn _nzl_for_type() -> Option<NonZeroLayout> {
    NonZeroLayout::for_type::<u64>()
}

// Type erasure check: ensure trait object dyn Allocator is at least nameable
// (we use dyn Deallocator since Allocator requires Sized in some paths).
fn _accepts_deallocator(_d: &dyn Deallocator) {}

// Surface check: ProtectFlags constants compile.
const _: ProtectFlags = ProtectFlags::RW;
const _: ProtectFlags = ProtectFlags::READ;
const _: ProtectFlags = ProtectFlags::RX;
const _: ProtectFlags = ProtectFlags::NONE;

// Surface check: FixedRange / OsBacked are nameable.
fn _accepts_fixed_range<T: FixedRange>() {}
fn _accepts_os_backed<T: OsBacked>() {}

// AllocError nameable.
const _AE: fn() -> AllocError = || AllocError;

#[test]
fn smoke() {
    // Run-time smoke covered by the rest of the test suite. This test exists
    // so `cargo check --no-default-features` compiles every surface item.
}
