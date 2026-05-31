# forge-alloc-capi

A **C ABI** over [`forge-alloc`](../forge-alloc) for **C and C++**, aimed at
**embedded / bare-metal** users who want a small hardened allocator without
rewriting in Rust.

It exposes one allocator: a **bump arena over a caller-provided buffer**. That
is the natural embedded allocator — linear O(1) allocation out of a
`.bss`/linker-provided region, with bulk reclaim via `forge_bump_reset()`. It
uses **no global allocator and no syscalls**, so it works on `#![no_std]`
targets down to Cortex-M and wasm.

`forge_bump_free()` scrubs the freed block with forge-alloc's poison pattern
(`0xDE`) so a stale secret can't be recovered through a use-after-free read.

> **Status:** provisional ABI, `publish = false`. Pinned, stable, and ready to
> use locally; not yet on crates.io.

## API

The full surface is in [`include/forge_alloc.h`](include/forge_alloc.h):

| Function | Purpose |
|---|---|
| `forge_bump_init(handle, buf, len)` | Bind a handle to a caller buffer |
| `forge_bump_alloc(handle, size, align)` | Allocate (returns `NULL` on failure) |
| `forge_bump_alloc_zeroed(handle, size, align)` | Allocate, zero-initialized |
| `forge_bump_free(handle, ptr, size, align)` | Poison-scrub a block |
| `forge_bump_reset(handle)` | Reclaim everything, rewind cursor |
| `forge_bump_remaining` / `_capacity` / `_allocated` | Introspection |
| `forge_bump_destroy(handle)` | Release the buffer borrow |

The handle (`forge_bump_t`) is an **opaque, caller-allocated** struct — put it
in static storage or on the stack; never inspect its fields. A compile-time
assertion in the Rust crate guarantees the real allocator fits the handle.

**Thread safety:** a handle is **not** thread-safe. Serialize all calls on a
handle. (This matches typical single-threaded embedded use.)

## Minimal usage

```c
#include "forge_alloc.h"
#include <stdint.h>

static uint8_t POOL[4096];           /* memory from .bss — no heap */

int main(void) {
    forge_bump_t arena;
    if (!forge_bump_init(&arena, POOL, sizeof POOL)) return 1;

    void *p = forge_bump_alloc(&arena, 64, 8);   /* 64 bytes, 8-aligned */
    /* ... use p ... */
    forge_bump_free(&arena, p, 64, 8);           /* scrubs the bytes   */

    forge_bump_reset(&arena);                    /* bulk reclaim       */
    forge_bump_destroy(&arena);
    return 0;
}
```

Full examples: [`examples/bump_example.c`](examples/bump_example.c) and
[`examples/bump_example.cpp`](examples/bump_example.cpp) (with a small RAII
wrapper).

## Building

### Desktop (shared library, for trying it out)

```sh
cargo build -p forge-alloc-capi --release
# -> target/release/{libforge_alloc_capi.so | .dylib | forge_alloc_capi.dll(+.lib)}
```

Compile and link a consumer (Linux/macOS):

```sh
cc app.c -I crates/forge-alloc-capi/include \
   -L target/release -lforge_alloc_capi -lpthread -ldl -lm -o app
```

On Windows (MSVC), link the import library `forge_alloc_capi.dll.lib` and keep
the DLL on the path:

```bat
cl /I crates\forge-alloc-capi\include app.c /link target\release\forge_alloc_capi.dll.lib
```

### Embedded / bare-metal (static library)

A Rust `staticlib` is a *final* artifact, so in `no_std` mode it must define a
`#[panic_handler]` and a `#[global_allocator]` itself. Which side provides them
depends on your firmware:

**Pure C/C++ firmware** — let the crate provide them via `staticlib-rt`:

```sh
cargo build -p forge-alloc-capi --release \
    --no-default-features --features staticlib-rt \
    --target thumbv7em-none-eabihf
# -> target/thumbv7em-none-eabihf/release/libforge_alloc_capi.a
```

Then link the `.a` into your image. The provided global allocator aborts if
called, which never happens: the bump API allocates only out of *your* buffer,
not the global heap.

> **`staticlib-rt` requires an abort-panic target.** Bare-metal targets
> (`thumbv7em-none-eabihf`, `wasm32-unknown-unknown`) default to
> `panic = "abort"` and build cleanly. It will **not** build on a normal host
> toolchain (Linux/macOS/Windows default to `panic = "unwind"`, which a no_std
> staticlib can't support) — so you can't smoke-test this build locally; build
> it for the embedded target directly.

**Rust `no_std` firmware** — your own crate already defines `#[panic_handler]`
and `#[global_allocator]`, so **add `forge-alloc-capi` as a dependency** with
`default-features = false` (do *not* enable `staticlib-rt` — it would collide
with your lang items, and do *not* `cargo build -p` it standalone — that builds
the `staticlib`/`cdylib` artifacts, which demand their own lang items and fail
to link). As a Cargo dependency, Cargo emits only the `lib` (rlib), which
defers the lang items to your binary:

```toml
# your firmware crate's Cargo.toml
[dependencies]
forge-alloc-capi = { version = "0.1", default-features = false }
```

You'd then call the same `forge_bump_*` functions from Rust (they're exported on
the rlib too), or just use `forge-alloc` directly.

The `std` feature (on by default) only exists so the desktop *shared* library
can pull a panic handler and allocator from std; embedded users turn it off.

## What's deliberately not here

A general-purpose `malloc`/`free` replacement, and a runtime-stride hardened
*pool* (which would need a typed `Slab<T>` that doesn't map cleanly to a C ABI).
The bump arena is the piece that fits embedded C/C++ cleanly today; richer
compositions live in the Rust API. See the repo
[`docs/C_API.md`](../../docs/C_API.md) for the rationale.
