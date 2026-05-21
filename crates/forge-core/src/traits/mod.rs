//! Foundation traits (M1).
//!
//! These traits define the contracts that all backing primitives, layout
//! primitives, and hardening wrappers compose against. They are intentionally
//! small — each captures one structural property.

mod alloc_fault_policy;
mod allocator;
mod fixed_range;
mod freelist_protection;
mod non_zero_layout;
mod os_backed;

pub use alloc_fault_policy::{
    AllocFaultPolicy, AlwaysFail, FailAfter, FailEveryNth, FailOnSize, NeverFail,
};
pub use allocator::{Allocator, Deallocator};
pub use fixed_range::FixedRange;
pub use freelist_protection::{FreelistCorruption, FreelistProtection, NoProtection};
pub use non_zero_layout::{AllocError, LayoutError, NonZeroLayout, StdCompat};
pub use os_backed::{OsBacked, ProtectFlags};

#[cfg(feature = "siphasher")]
pub use freelist_protection::SipHashMAC;

#[cfg(all(target_arch = "aarch64", feature = "pac-stub"))]
pub use freelist_protection::PacMAC;
