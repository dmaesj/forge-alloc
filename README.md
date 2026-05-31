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

The design is laid out in [`docs/ARCHITECTURE.md`](https://github.com/dmaesj/forge-alloc/blob/main/docs/ARCHITECTURE.md);
a per-release summary lives in [`CHANGELOG.md`](https://github.com/dmaesj/forge-alloc/blob/main/CHANGELOG.md).

## Crates

| crate | role |
|---|---|
| `forge-alloc` | The library — Layer 1 backings, Layer 2 layout primitives, and Layer 3 hardening wrappers, plus the trait re-exports. **This is the crate to depend on.** |
| `forge-alloc-core` | The trait contracts (`Allocator`, `Deallocator`, `OsBacked`, `FixedRange`, `FreelistProtection`, `AllocFaultPolicy`), `NonZeroLayout`, and `StdCompat<A>`. Re-exported by `forge-alloc`; depend on it directly only to *implement* the traits without pulling in the implementations. |
| `forge-alloc-capi` | C ABI for C/C++ — a hardened bump arena over a caller-provided buffer, for embedded users. `staticlib` for firmware, `cdylib` for desktop. See [`docs/C_API.md`](https://github.com/dmaesj/forge-alloc/blob/main/docs/C_API.md). (`publish = false`, provisional ABI) |
| `forge-bench` | Criterion / CodSpeed benchmarks (workspace-internal, `publish = false`) |
| `forge-fuzz` | cargo-fuzz targets (workspace-excluded; nightly only) |

Inside `forge-alloc` the three implementation layers are modules —
**backing** (`InlineBacked`, `MmapBacked`, `System`), **layout**
(`BumpArena`, `SharedBumpArena`, `Slab`, `SizeClassed`, `StackAlloc`,
`ExtendableSlab`, `GenerationalSlab`, `SlabOwner`/`SlabRemote`,
`WithFallback`), and **hardening** (`Canary`, `PoisonOnFree`,
`Quarantine`, `Statistics`, `Watermark`, `GuardPage`, `CacheJitter`,
`HugePageAligned`, `NumaLocal`, `SplitMetadata`, plus `Faulty` for
test-only fault injection).

One wrapper is for testing rather than production: `Faulty` injects allocation failures on a policy you pick, so the out-of-memory and fallback paths your code rarely exercises can be driven deterministically in tests. See the [composition recipes](https://github.com/dmaesj/forge-alloc/blob/main/docs/COMPOSITION_RECIPES.md) for wiring.

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

- [`docs/ARCHITECTURE.md`](https://github.com/dmaesj/forge-alloc/blob/main/docs/ARCHITECTURE.md) — the three-layer
  mental model. Start here.
- [`docs/APPLICATION_RECIPES.md`](https://github.com/dmaesj/forge-alloc/blob/main/docs/APPLICATION_RECIPES.md) —
  application-architecture walkthroughs (CLI batch, embedded firmware,
  web service, game engine, auth service, database executor), each
  backed by a runnable example in
  [`crates/forge-alloc/examples/`](https://github.com/dmaesj/forge-alloc/tree/main/crates/forge-alloc/examples).
- [`docs/COMPOSITION_RECIPES.md`](https://github.com/dmaesj/forge-alloc/blob/main/docs/COMPOSITION_RECIPES.md) — the
  type-level cookbook: how to wire one primitive.
- [`docs/COMPATIBILITY_MATRIX.md`](https://github.com/dmaesj/forge-alloc/blob/main/docs/COMPATIBILITY_MATRIX.md) — the
  catalogue of combinations that don't work or shouldn't be tried
  (compile-time rejections, construction errors, footguns, and
  v2.0-deferred constraints).
- [`docs/PERFORMANCE_TRADEOFFS.md`](https://github.com/dmaesj/forge-alloc/blob/main/docs/PERFORMANCE_TRADEOFFS.md) — the
  pay-for-what-you-use cost curve across the hardening tiers.
- [`CONTRIBUTING.md`](https://github.com/dmaesj/forge-alloc/blob/main/CONTRIBUTING.md) — how to add a primitive and
  submit a change.

## MSRV

Rust 1.84 — `core::ptr::without_provenance_mut` is stable since 1.84.

## License

Dual-licensed at your option under either:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/dmaesj/forge-alloc/blob/main/LICENSE-APACHE))
- MIT License ([LICENSE-MIT](https://github.com/dmaesj/forge-alloc/blob/main/LICENSE-MIT))

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual-licensed as above without any
additional terms or conditions.
