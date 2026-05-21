//! # forge-core
//!
//! Core trait contracts and primitive layout type for the `forge-alloc`
//! family of composable allocator crates.
//!
//! Defines the foundation that the higher layers depend on:
//!
//! - [`Allocator`] / [`Deallocator`] — the split allocation trait
//! - [`NonZeroLayout`] — non-zero-size, power-of-two-align layout contract
//! - [`StdCompat`] — bridge to [`allocator_api2::alloc::Allocator`]
//! - [`OsBacked`] / [`FixedRange`] — structural traits for backings and ranges
//! - [`FreelistProtection`] (+ [`NoProtection`], optional `SipHashMAC` / `PacMAC`)
//! - [`AllocFaultPolicy`] — the OOM fault-injection seam for the
//!   `forge-hardening` `Faulty` wrapper (+ built-in policies)
//!
//! Higher layers (`forge-backing`, `forge-layout`, `forge-hardening`) consume these
//! traits to produce primitive types; the `forge-alloc` meta-crate re-exports
//! everything for convenience.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

pub mod traits;

pub use traits::{
    AllocError, AllocFaultPolicy, Allocator, AlwaysFail, Deallocator, FailAfter, FailEveryNth,
    FailOnSize, FixedRange, FreelistCorruption, FreelistProtection, NeverFail, NoProtection,
    NonZeroLayout, OsBacked, ProtectFlags, StdCompat,
};

#[cfg(feature = "siphasher")]
pub use traits::SipHashMAC;

#[cfg(all(target_arch = "aarch64", feature = "pac-stub"))]
#[allow(deprecated)] // re-exporting the deprecated PacMAC stub is intentional
pub use traits::PacMAC;
