// bump_example.cpp — a small RAII wrapper around the forge-alloc C ABI.
//
// Build (from the repo root):
//   cargo build -p forge-alloc-capi --release
//   c++ examples/.../bump_example.cpp \
//       -I crates/forge-alloc-capi/include \
//       -L target/release -lforge_alloc_capi -lpthread -ldl -lm \
//       -o bump_example_cpp
//   ./bump_example_cpp

#include "forge_alloc.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

// Owns a forge_bump_t handle and ties its lifetime to the C++ object. The
// underlying buffer is borrowed and must outlive the wrapper.
class BumpArena {
public:
  BumpArena(void *buf, std::size_t len) { ok_ = forge_bump_init(&h_, buf, len) != 0; }
  ~BumpArena() {
    if (ok_) {
      forge_bump_destroy(&h_);
    }
  }

  BumpArena(const BumpArena &) = delete;
  BumpArena &operator=(const BumpArena &) = delete;

  explicit operator bool() const { return ok_; }

  void *alloc(std::size_t size, std::size_t align) { return forge_bump_alloc(&h_, size, align); }
  void free(void *p, std::size_t size, std::size_t align) { forge_bump_free(&h_, p, size, align); }
  void reset() { forge_bump_reset(&h_); }
  std::size_t remaining() const { return forge_bump_remaining(&h_); }

private:
  forge_bump_t h_{};
  bool ok_{false};
};

static std::uint8_t POOL[8192];

int main() {
  BumpArena arena(POOL, sizeof POOL);
  if (!arena) {
    std::fprintf(stderr, "init failed\n");
    return 1;
  }

  void *p = arena.alloc(256, alignof(std::max_align_t));
  if (!p) {
    std::fprintf(stderr, "alloc failed\n");
    return 2;
  }
  std::printf("remaining=%zu\n", arena.remaining());

  arena.free(p, 256, alignof(std::max_align_t));
  arena.reset();
  // forge_bump_destroy runs in ~BumpArena().
  return 0;
}
