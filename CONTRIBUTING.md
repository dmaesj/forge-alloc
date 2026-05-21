# Contributing

Thanks for considering a contribution. This doc covers the mechanics of
adding a new primitive, wrapper, or trait extension, and how to get a
change reviewed and merged.

## Where things live

```
crates/
├── forge-alloc-core/   # Layer 0 — traits, NonZeroLayout, StdCompat
├── forge-alloc/        # Layers 1-3, as src/ modules:
│   ├── src/backing/    #   Layer 1 — InlineBacked, MmapBacked, System
│   ├── src/layout/     #   Layer 2 — BumpArena, Slab, SizeClassed, etc.
│   └── src/hardening/  #   Layer 3 — Canary, CacheJitter, Quarantine, etc.
├── forge-bench/        # Criterion benchmarks (workspace-internal)
└── forge-fuzz/         # cargo-fuzz targets (nightly only, workspace-excluded)
```

The layer boundary tells you where a new primitive belongs:

- **Layer 0 / `forge-alloc-core`** — anything that's a trait, the
  `NonZeroLayout` type, or a feature flag that fans out to the layers.
- **Layer 1 / `forge-alloc` `src/backing/`** — primitives that
  *produce* memory. Each Layer-1 primitive answers "where do these
  bytes come from?": OS mapping (`MmapBacked`), inline buffer
  (`InlineBacked`), global heap (`System`).
- **Layer 2 / `forge-alloc` `src/layout/`** — primitives that
  *organize* memory served by a backing. Bump cursor, freelist,
  fallback router, etc. They consume a `B: Allocator` (typically a
  backing) and impose structure.
- **Layer 3 / `forge-alloc` `src/hardening/`** — wrappers that
  decorate any allocator with one specific behaviour: canaries,
  poison-on-free, statistics, guard pages, NUMA binding, cache jitter,
  etc. (`Faulty`, the test-only fault-injection wrapper, also lives
  here.)

Wrappers are *transparent* — they implement `Allocator`/`Deallocator`
(and ideally `FixedRange` / `OsBacked` when applicable) by forwarding
to the wrapped allocator with their own side-effect interposed. The
type-level composition makes the cost visible and pay-for-what-you-use.

## Adding a new primitive

1. **Pick the layer.** A wrapper belongs in `src/hardening/`. A new
   allocation strategy that consumes a backing belongs in `src/layout/`.
   A new source of memory belongs in `src/backing/`. (All three are
   modules of the `forge-alloc` crate.)
2. **Decide the public surface.** Implement `Allocator` and
   `Deallocator`. If the primitive's address range is fixed at
   construction, implement `FixedRange` too — this enables routing via
   `WithFallback` and observability via `Watermark`.
3. **Write a module-level doc block** that covers (in this order):
   what the primitive does, what backing it requires, thread-safety
   (`Send` and `Sync` impl rationale), and `# Safety` invariants for
   any `unsafe trait` impls.
4. **Add unit tests** that exercise the happy path, the exhaustion
   path, edge alignments, and any size/alignment composition limits.
5. **Add proptest cases** in `crates/forge-alloc/tests/proptest_correctness.rs`
   (or the appropriate per-area test file) for the invariants the
   primitive promises.
6. **If the primitive has `unsafe` blocks**, add `#[cfg(kani)]` proof
   harnesses inside the source file under a `mod kani_proofs` block.
   See `crates/forge-alloc/src/layout/bump.rs` for the pattern.
7. **Re-export it.** Add to `crates/forge-alloc/src/lib.rs` so
   `forge_alloc::*` users see it.
8. **Update `CHANGELOG.md`** under `[Unreleased]`.

## Running the test suite

```sh
# Stable build + clippy + format check + doc gate (matches CI):
cargo fmt --all
cargo check --workspace --all-features
cargo test --workspace --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo doc --workspace --all-features --no-deps  # RUSTDOCFLAGS=-D warnings in CI

# no_std surface (library crates, --lib only):
cargo check -p forge-alloc-core --no-default-features --lib
cargo check -p forge-alloc --no-default-features --lib

# MIRI (nightly only — validates unsafe blocks):
cargo +nightly miri test -p forge-alloc-core --all-features
cargo +nightly miri test -p forge-alloc --test miri_targets --test nrvo_defeat

# Kani (nightly only — proves bounded properties on unsafe code):
cargo kani -p forge-alloc
```

The GitHub Actions workflow at `.github/workflows/ci.yml` runs the
stable / no_std / MIRI / cross-compile / loom matrix on Linux + macOS +
Windows; `.github/workflows/codspeed.yml` runs the benchmark
regression gate. Locally you only need the stable + no_std checks to
land green before opening a pull request — CI covers the rest.

## Submitting a change

1. **Fork** the repository and create a topic branch off `main`.
2. Make your change; keep commits focused and the working tree clean.
3. Run the stable + no_std checks above — they must pass.
4. **Open a pull request** against `main`. Describe *why* the change
   is needed, not just *what* it does. Link any related issue.
5. CI must be green before review. A maintainer will review; expect
   extra scrutiny on anything touching `unsafe` code, atomic
   orderings, or a public trait contract — see below.

## Review expectations

Every change touching `unsafe` code, soundness-relevant invariants, or
a public trait contract gets an adversarial review pass: a reviewer
actively looks for correctness bugs, soundness regressions, doc/code
mismatches, contract changes that accidentally widen UB, and
mechanical hygiene issues (wrong asserts, broken intra-doc links,
clippy lints that crept past). The bar is high because every allocator
in this family makes memory-safety promises that downstream `unsafe`
code relies on. If you're submitting `unsafe`-heavy work, self-review
against that checklist first — it speeds up the merge.

## Commit message convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new functionality
- `fix:` — bug fix
- `docs:` — documentation only
- `refactor:` — code restructure without behaviour change
- `chore:` — build / CI / tooling
- `test:` — test additions
- `perf:` — performance improvement

Multi-paragraph bodies are fine. The first line should be ≤ 72 chars.

## License

By contributing, you agree that your contribution will be dual-licensed
under MIT and Apache-2.0 (see `LICENSE-MIT` and `LICENSE-APACHE`).
