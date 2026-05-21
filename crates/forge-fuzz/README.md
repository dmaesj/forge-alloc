# forge-fuzz

`cargo-fuzz` targets for the forge-alloc family. Nightly Rust only — the
libfuzzer-sys dependency requires nightly-specific build configuration.

This crate is **excluded from the default workspace** (see root `Cargo.toml`)
so that `cargo build --workspace` on stable does not fail. To run any target:

```bash
rustup toolchain install nightly
cargo +nightly install cargo-fuzz
cd crates/forge-fuzz
cargo +nightly fuzz run fuzz_bump_arena
cargo +nightly fuzz run fuzz_slab
cargo +nightly fuzz run fuzz_with_fallback
```

Each target consumes arbitrary bytes as a sequence of allocation operations
and checks the core allocator invariants (no overlap, alignment, size).

A failing case is minimized into `crates/forge-fuzz/artifacts/<target>/`
and can be replayed for debugging:

```bash
cargo +nightly fuzz run fuzz_bump_arena artifacts/fuzz_bump_arena/crash-<hash>
```

CI should run each target for a fixed wall-clock budget (e.g. 5 minutes per
target on every PR) and block merges on new findings.
