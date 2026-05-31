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
not the global heap. Build your image with `panic = "abort"` (bare-metal
targets select this by default).

**Rust `no_std` firmware** — your own crate already defines these lang items,
so omit `staticlib-rt` (enabling it would collide) and depend on the `lib`
(rlib) target instead:

```sh
cargo build -p forge-alloc-capi --release \
    --no-default-features --target thumbv7em-none-eabihf
```

The `std` feature (on by default) only exists so the desktop *shared* library
can pull a panic handler and allocator from std; embedded users turn it off.

## What's deliberately not here

A general-purpose `malloc`/`free` replacement, and a runtime-stride hardened
*pool* (which would need a typed `Slab<T>` that doesn't map cleanly to a C ABI).
The bump arena is the piece that fits embedded C/C++ cleanly today; richer
compositions live in the Rust API. See the repo
[`docs/C_API.md`](../../docs/C_API.md) for the rationale.
