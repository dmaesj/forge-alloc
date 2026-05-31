# C API (`forge-alloc-capi`)

A C ABI over forge-alloc for **C and C++**, scoped to the use case where
forge-alloc has something the ecosystem doesn't: a small, hardened allocator
for **embedded / bare-metal** programs that can't or won't link a Rust runtime
or a full general-purpose allocator.

The crate lives at [`crates/forge-alloc-capi`](../crates/forge-alloc-capi);
its [README](../crates/forge-alloc-capi/README.md) has the build-and-link
recipes. This document records *why the surface is shaped the way it is*.

## What it exposes

One allocator: a **bump arena over a caller-provided buffer**
(`BumpArena<StaticBacked<'static>>`). The C caller hands in a region
(`forge_bump_init(handle, buf, len)`); allocation aligns the cursor,
bounds-checks, and advances; reclamation is bulk via `forge_bump_reset`.
`forge_bump_free` overwrites the block with forge-alloc's poison pattern
(`0xDE`) before the (no-op) reclaim, so freed secrets can't be read back via a
use-after-free.

The handle is an opaque, caller-allocated `forge_bump_t`. The crate owns no
heap and makes no syscalls, so the whole thing is `#![no_std]`. Because a Rust
`staticlib` is a final artifact, the `no_std` build must define a
`#[panic_handler]` and a `#[global_allocator]`: the optional `staticlib-rt`
feature supplies them for a pure C/C++ firmware image (the allocator aborts if
called, which the bump-only surface never does), or a Rust `no_std` consumer
supplies its own and links the rlib. See the
[README](../crates/forge-alloc-capi/README.md#building) for the exact commands.

## Why this scope, and not more

This is the narrow slice of "forge-alloc, callable from C" that is both
**aligned with forge-alloc's strengths** and **underserved by existing tools**:

- **Bump-over-static-buffer is the embedded allocator.** It maps directly onto
  how firmware gets memory (a linker section, a `static` array), needs no MMU,
  no `mmap`, no global allocator, and reclaims in bulk. It also handles
  arbitrary runtime sizes and alignments through one simple C surface.
- **Hardened embedded allocation has essentially no incumbent.** The serious
  hardened allocators (GrapheneOS `hardened_malloc`, LLVM `scudo`) assume a
  real OS/MMU and are Linux/Android-centric. forge-alloc already compiles for
  `thumbv7em-none-eabihf` and `wasm32`, so a poison-scrubbing bump arena for
  embedded C is a place it can be first-class rather than an also-ran.

### Deliberately excluded

- **A general-purpose `malloc`/`free` replacement.** That means
  `calloc`/`realloc`/`aligned_alloc`/`malloc_usable_size`, C++ `operator
  new`/`delete` in all their forms, and a thread-safe general-sized heap with
  thread caching to be competitive. forge-alloc's primitives are arena / slab /
  bump — *structured*, not a general concurrent heap — so a competitive drop-in
  `malloc` is a different project, and the field (jemalloc, mimalloc,
  hardened_malloc, scudo) is mature. Not worth being the youngest entrant.
- **A runtime-stride hardened pool.** A fixed-size object pool with guard pages
  and a freelist MAC is squarely in forge-alloc's wheelhouse — but the
  implementation, `Slab<T>`, is typed by the Rust element type `T`, which does
  not map cleanly to a runtime block size chosen from C. Exposing it would mean
  either a fixed menu of block sizes or a typed shim per element; neither is a
  clean C ABI. Left for the Rust API.
- **Guard pages / split metadata over the C ABI.** Those wrappers
  (`GuardPage`, `SplitMetadata`) require `mmap`/Win32 and `std`, i.e. an OS —
  the opposite of the embedded target this crate serves. They remain available
  in the Rust API for hosted use.
- **`grow` / `realloc`.** There is no in-place resize. A bump arena can't
  generally grow a non-top allocation, so a C caller that needs realloc-like
  behavior must size the request up front, or `reset` and rebuild. Documented
  here so the absence is intentional, not an oversight.

## ABI stability

`publish = false` and the ABI is provisional: the handle size
(`FORGE_BUMP_STORAGE`) and the function set may change before a 1.0. A
compile-time assertion in the crate guarantees the real allocator fits the
handle storage, so a layout change can't silently corrupt caller memory — it
fails the build instead. The intent is to stabilize and publish once the
surface has had real external use.

## Verification

The FFI surface is exercised two ways:

- **CI-enforced:** an in-crate Rust integration test
  ([`tests/ffi.rs`](../crates/forge-alloc-capi/tests/ffi.rs)) drives every
  entry point end-to-end (init → alloc/zeroed → poison-on-free → reset →
  destroy, plus exhaustion and invalid-argument paths), so the behavior is
  covered even where a C toolchain isn't. This is the guarantee that travels
  with the repo.
- **One-time manual check** (not yet wired into CI): the C and C++ examples were
  compiled against the generated header and linked the built **desktop `std`
  cdylib** (MSVC), the exported symbol names were checked against the header, and
  both examples ran successfully. Separately, the **`no_std` `staticlib-rt`
  staticlib was build-verified for `thumbv7em-none-eabihf`** (it cannot be built
  on a host toolchain — see the README caveat). A CI job that compiles the
  examples and the embedded staticlib would make all of this reproducible.

## Roadmap (not commitments)

If a concrete consumer materializes, the natural extensions are: a
poison/quarantine-composed variant once `forge-alloc` grows the accessor needed
to reset through a wrapper; a fixed-size pool surface for a small set of common
block sizes; and a generated header via `cbindgen` to keep `forge_alloc.h` in
lockstep with the Rust signatures.
