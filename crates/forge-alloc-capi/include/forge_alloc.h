/*
 * forge_alloc.h — C ABI for forge-alloc.
 *
 * A hardened bump allocator over a caller-provided buffer, for embedded C/C++.
 * Linear O(1) allocation out of a static/.bss region; bulk reclaim via
 * forge_bump_reset(). No global allocator, no syscalls — works on bare metal.
 *
 * free() overwrites the freed block with forge-alloc's poison pattern (0xDE)
 * so stale secrets can't be read back via use-after-free.
 *
 * THREAD SAFETY: a handle is NOT thread-safe. Serialize all calls on a handle.
 *
 * Link against libforge_alloc_capi (staticlib for firmware, cdylib for
 * desktop). See the crate README for build instructions.
 */
#ifndef FORGE_ALLOC_H
#define FORGE_ALLOC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Bytes of caller storage for a handle. Kept in sync with HANDLE_STORAGE in
 * src/lib.rs, where a compile-time assertion guarantees the real allocator
 * fits. Do not rely on this value directly; use sizeof(forge_bump_t).
 */
#define FORGE_BUMP_STORAGE 48

/*
 * Opaque, caller-allocated handle. Place it in static storage (embedded) or on
 * the stack; pass its address to the functions below. Never inspect the fields.
 * The void* in the union forces pointer alignment without needing C11
 * _Alignas, matching the Rust handle's alignment.
 */
typedef struct forge_bump {
  union {
    void *_align;
    unsigned char _storage[FORGE_BUMP_STORAGE];
  } _opaque;
} forge_bump_t;

/*
 * Initialize a bump arena over [buf, buf+len). The buffer must outlive the
 * handle and must not be aliased elsewhere. Returns nonzero on success, 0 if
 * any argument is null/zero. `len == 0` fails.
 */
int forge_bump_init(forge_bump_t *handle, void *buf, size_t len);

/*
 * Allocate `size` bytes aligned to `align` (a power of two). Returns a pointer
 * into the buffer, or NULL on failure (arena exhausted, size == 0, or align not
 * a power of two).
 */
void *forge_bump_alloc(forge_bump_t *handle, size_t size, size_t align);

/* As forge_bump_alloc(), but the block is zero-initialized. */
void *forge_bump_alloc_zeroed(forge_bump_t *handle, size_t size, size_t align);

/*
 * Free a block: its bytes are scrubbed with the poison pattern. Space is not
 * individually reclaimed (a bump arena reclaims in bulk via reset); this is for
 * hygiene and symmetry. `ptr` may be NULL (ignored). `size`/`align` must match
 * the original allocation.
 */
void forge_bump_free(forge_bump_t *handle, void *ptr, size_t size, size_t align);

/*
 * Reclaim everything at once and rewind to the start of the buffer. Returns
 * nonzero on success. All previously returned pointers become invalid.
 */
int forge_bump_reset(forge_bump_t *handle);

/* Bytes still available for allocation. 0 for a null handle. */
size_t forge_bump_remaining(const forge_bump_t *handle);

/* Total capacity (buffer length) in bytes. 0 for a null handle. */
size_t forge_bump_capacity(const forge_bump_t *handle);

/* Bytes currently handed out. 0 for a null handle. */
size_t forge_bump_allocated(const forge_bump_t *handle);

/*
 * Tear down a handle. The arena owns no heap, so this just releases the borrow
 * of the buffer; afterward the buffer may be reused or freed. NULL is ignored.
 */
void forge_bump_destroy(forge_bump_t *handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FORGE_ALLOC_H */
