/*
 * bump_example.c — using forge-alloc's bump arena from C.
 *
 * Embedded-style: the arena lives over a static .bss buffer, no heap involved.
 *
 * Desktop build (from the repo root):
 *   cargo build -p forge-alloc-capi --release
 *   cc examples/.../bump_example.c \
 *      -I crates/forge-alloc-capi/include \
 *      -L target/release -lforge_alloc_capi -lpthread -ldl -lm \
 *      -o bump_example
 *   ./bump_example
 *
 * (On Windows link the cdylib import lib forge_alloc_capi.dll.lib and keep the
 * DLL on PATH; the -lpthread/-ldl/-lm are the std runtime deps on Linux. For
 * bare metal, link the staticlib built with --no-default-features --features
 * staticlib-rt into your firmware image.)
 */
#include "forge_alloc.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* A linker-provided / static region — the canonical embedded source of memory. */
static uint8_t POOL[4096];

int main(void) {
  forge_bump_t arena;
  if (!forge_bump_init(&arena, POOL, sizeof POOL)) {
    fprintf(stderr, "init failed\n");
    return 1;
  }

  void *a = forge_bump_alloc(&arena, 64, 8);
  void *b = forge_bump_alloc(&arena, 128, 16);
  if (!a || !b) {
    fprintf(stderr, "alloc failed\n");
    return 2;
  }
  memset(a, 0xAB, 64);

  printf("allocated=%zu remaining=%zu\n", forge_bump_allocated(&arena),
         forge_bump_remaining(&arena));

  /* free() scrubs the block with the poison pattern (0xDE). */
  forge_bump_free(&arena, a, 64, 8);

  /* Bulk reclaim — the bump arena's way of "freeing everything". */
  forge_bump_reset(&arena);
  printf("after reset allocated=%zu\n", forge_bump_allocated(&arena));

  forge_bump_destroy(&arena);
  return 0;
}
