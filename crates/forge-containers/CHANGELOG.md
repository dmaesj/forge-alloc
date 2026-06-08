# Changelog

All notable changes to `forge-containers` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

### Added

- `AlignedColBuffer<const ALIGN: usize = 64>` — an owned, over-aligned, growable
  byte buffer for zero-copy columnar / FFI interop. Const-generic alignment
  (default 64, Arrow's recommended SIMD alignment) preserved across every
  growth; `Send + Sync` with a stable data address so it can serve as a
  zero-copy owner (e.g. `arrow_buffer::Buffer::from_custom_allocation`). Typed
  `extend_from_typed` / `as_typed`, byte `push` / `extend_from_slice`,
  `reserve` / `with_capacity` / `clear`. Dependency-free — no Arrow dependency;
  the consumer performs the zero-copy handoff with its own arrow-rs version.

[0.1.0]: https://github.com/dmaesj/forge-alloc/releases/tag/forge-containers-v0.1.0
