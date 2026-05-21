# forge-alloc

Composable memory allocator primitives. Snap together at the type level to
produce application-specific allocators with zero runtime dispatch overhead,
compile-time-enforced guarantees, and pay-for-what-you-use security
hardening.

```rust
use forge_alloc::*;

// A bump arena over a 64 KiB OS-mapped region with poison-on-free + stats.
type Scratch = Statistics<PoisonOnFree<BumpArena<MmapBacked>>>;

// A typed slab with separated metadata and guard pages on the data region
// — the recommended "hardened slab" composition for security-critical data.
// (Alias: `HardenedSlab<T>` in `forge-alloc`.)
type Hardened<T> = Slab<T, GuardPage<SplitMetadata<MmapBacked>>>;
```

The design is laid out in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md);
a per-release summary lives in [`CHANGELOG.md`](CHANGELOG.md).

## Workspace layout

| crate | role |
|---|---|
| `forge-core` | trait contracts (`Allocator`, `Deallocator`, `OsBacked`, `FixedRange`, `FreelistProtection`, `AllocFaultPolicy`), `NonZeroLayout`, `StdCompat<A>` |
| `forge-backing` | Layer 1 backings — `InlineBacked<N>`, `MmapBacked`, `System` |
| `forge-layout` | Layer 2 layout — `BumpArena`, `SharedBumpArena`, `Slab`, `SizeClassed`, `StackAlloc`, `ExtendableSlab`, `GenerationalSlab`, `SlabOwner` / `SlabRemote`, `WithFallback` |
| `forge-hardening` | Layer 3 hardening — `Canary`, `PoisonOnFree`, `Quarantine`, `Statistics`, `Watermark`, `GuardPage`, `CacheJitter`, `HugePageAligned`, `NumaLocal`, `SplitMetadata`; plus `Faulty` (test-only fault injection) |
| `forge-alloc` | meta-crate; re-exports the union of the above. Most users depend on this; users who only need a subset depend directly on the relevant `forge-*` crate to minimise compile time and dependency footprint |
| `forge-bench` | Criterion benchmarks (workspace-internal, `publish = false`) |
| `forge-fuzz` | cargo-fuzz targets (workspace-excluded; nightly only) |

## Status

All Layer 0–3 primitives are implemented and tested — the backings,
layout primitives, and hardening wrappers listed above, plus
`BatchPolicy::Adaptive` cross-thread batching. Verification CI is live:
Kani proof harnesses on the unsafe arithmetic, a full MIRI run across
the `no_std` subsets, a cross-compile matrix down to 32-bit bare-metal,
loom concurrency models, and CodSpeed continuous benchmarking. Broader
Kani proof coverage is ongoing.

Two hardware-gated features are not yet implemented — ARM MTE (memory
tagging) and x86 MPK (protection keys) — both need the target silicon
to develop against.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the three-layer
  mental model. Start here.
- [`docs/APPLICATION_RECIPES.md`](docs/APPLICATION_RECIPES.md) —
  application-architecture walkthroughs (CLI batch, embedded firmware,
  web service, game engine, auth service, database executor), each
  backed by a runnable example in
  [`crates/forge-alloc/examples/`](crates/forge-alloc/examples/).
- [`docs/COMPOSITION_RECIPES.md`](docs/COMPOSITION_RECIPES.md) — the
  type-level cookbook: how to wire one primitive.
- [`docs/COMPATIBILITY_MATRIX.md`](docs/COMPATIBILITY_MATRIX.md) — the
  catalogue of combinations that don't work or shouldn't be tried
  (compile-time rejections, construction errors, footguns, and
  v2.0-deferred constraints).
- [`docs/PERFORMANCE_TRADEOFFS.md`](docs/PERFORMANCE_TRADEOFFS.md) — the
  pay-for-what-you-use cost curve across the hardening tiers.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to add a primitive and
  submit a change.

## MSRV

Rust 1.70 — `NonNull::slice_from_raw_parts` is stable since 1.70.

## License

Dual-licensed at your option under either:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual-licensed as above without any
additional terms or conditions.
