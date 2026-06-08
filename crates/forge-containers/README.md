# forge-containers

Container and interop primitives for the [forge-alloc](https://docs.rs/forge-alloc)
memory family.

`forge-alloc` gives you **allocators** — things that *hand out* memory and
implement the `Allocator` / `FixedRange` / `OsBacked` contracts so they compose
into bump / slab / hardening stacks. `forge-containers` gives you **containers**
— data structures that *hold* memory, built for the same foundation.

## Contents

- **`AlignedColBuffer<const ALIGN = 64>`** — an owned, over-aligned, growable
  byte buffer for zero-copy columnar / FFI interop. The const-generic alignment
  (default 64, Arrow's recommended SIMD alignment) is preserved across every
  growth, and the buffer is `Send + Sync` with a stable data address, so it can
  serve directly as a zero-copy owner (e.g.
  `arrow_buffer::Buffer::from_custom_allocation`). **Dependency-free** — it
  ships no Arrow dependency; the consumer performs the one-line zero-copy
  handoff with its own arrow-rs version.

`#![no_std]` (requires `alloc`). Licensed under MIT OR Apache-2.0.
