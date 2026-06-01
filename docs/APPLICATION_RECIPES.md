# Application Recipes

How real application architectures compose `forge-alloc` primitives
into hot/warm/cold-path allocator stacks. Each recipe is a runnable
example in [`crates/forge-alloc/examples/`](../crates/forge-alloc/examples/);
this doc explains the architectural reasoning behind each.

If you're new to the crate, read these BEFORE
[`COMPOSITION_RECIPES.md`](COMPOSITION_RECIPES.md) — that doc is the
type-level cookbook (how to wire one primitive); this one is the
application-level cookbook (how to mix multiple primitives across the
lifetime classes of your code).

## Index

- [Recipe 1: `cli_batch` — single arena, drop at end](#recipe-1-cli_batch--single-arena-drop-at-end)
- [Recipe 2: `embedded_firmware` — sole slab on inline storage](#recipe-2-embedded_firmware--sole-slab-on-inline-storage)
- [Recipe 3: `web_service` — three-tier hot/warm/cold](#recipe-3-web_service--three-tier-hotwarmcold)
- [Recipe 4: `game_frame` — per-frame bump + entity slab + cross-thread audio](#recipe-4-game_frame--per-frame-bump--entity-slab--cross-thread-audio)
- [Recipe 5: `auth_service` — observable + UAF-resistant token allocator](#recipe-5-auth_service--observable--uaf-resistant-token-allocator)
- [Recipe 6: `db_executor` — per-query arena + buffer pool + hardened audit log](#recipe-6-db_executor--per-query-arena--buffer-pool--hardened-audit-log)

Run any of them with:

```bash
cargo run -p forge-alloc --example <name>
```

---

## Recipe 1: `cli_batch` — single arena, drop at end

**Workload**: short-lived CLI tools, build scripts, batch processors.
Everything runs to completion and exits.

**Composition**:
```rust
type Arena = BumpArena<InlineBacked<{ 64 * 1024 }>>;
```

**Why**:
- One lifetime class. Nothing outlives `main`.
- A bump arena is the cheapest possible allocator (~2 ns/op).
- Inline-backed storage avoids the mmap syscall entirely; the
  arena's 64 KiB lives on the stack.
- No deallocs because nothing needs to come back — the OS reclaims
  the stack when the process exits.

**When to escalate beyond this recipe**: when your tool runs long
enough that an attacker could attempt to exploit it (rare for CLIs),
or when you allocate so much that the per-frame budget is unclear
(use `MmapBacked` instead of `InlineBacked` — larger arenas don't
fit a 1 MiB stack on Windows).

**Example**: [`crates/forge-alloc/examples/cli_batch.rs`](../crates/forge-alloc/examples/cli_batch.rs)

---

## Recipe 2: `embedded_firmware` — sole slab on inline storage

**Workload**: bare-metal firmware. Bounded memory budget; no MMU;
no syscalls; real-time deadlines.

**Composition**:
```rust
type SensorPool = Slab<SensorReading, InlineBacked<{ 8 * 1024 }>>;
```

**Why**:
- `no_std`-compatible at the trait level (both `Slab` and
  `InlineBacked` work without the `std` feature).
- Fixed capacity at compile time = bounded memory at compile time.
- Slab's LIFO freelist gives O(1) alloc + O(1) dealloc — real-time
  safe.
- Zero allocator-internal heap touches.

**Variations**:
- For long-lived references to slots that may be recycled, use
  `GenerationalSlab` instead — handles encode `(slot_index,
  generation_at_alloc)` and detect ABA reuse at access time.
- For a global allocator (`#[global_allocator]`), wrap with a
  `GlobalAllocAdapter` (not yet shipped — track in v0.2).

**When to escalate**: if you need cross-thread allocations (rare in
firmware), switch to `SharedBumpArena` or `SlabOwner`/`SlabRemote`.

**Example**: [`crates/forge-alloc/examples/embedded_firmware.rs`](../crates/forge-alloc/examples/embedded_firmware.rs)

---

## Recipe 3: `web_service` — three-tier hot/warm/cold

**Workload**: HTTP/RPC service. Each request has a short lifetime
inside a longer-lived connection; occasional requests need
oversized buffers.

**Composition**:
```rust
type RequestScratch = BumpArena<InlineBacked<{ 64 * 1024 }>>;     // per-request
type ConnectionPool = Slab<Connection, MmapBacked>;                // per-connection
type ConnAllocator  = WithFallback<ConnectionPool, System>;        // + cold path
```

**Why**:
- **Per-request scratch**: parsing, formatting, intermediate
  structures all have lifetime = "this request." Bump-allocate and
  reset between requests; no per-allocation overhead.
- **Per-connection slab**: connections persist across requests but
  have bounded count. A typed slab with LIFO reuse keeps alloc
  latency predictable.
- **System fallback**: 99% of requests don't need oversized buffers,
  but the 1% that DO shouldn't blow the bump arena or push out
  other connections. `WithFallback` routes by pointer provenance —
  the primary's `FixedRange::contains(ptr)` decides who deallocates.

**Failure modes to watch**:
- If your request scratch overflows often, EITHER make it bigger or
  shed load. Don't silently fall back to System per allocation; the
  observability and reuse properties of the bump arena disappear.
- If your connection count exceeds slab capacity, the next accept()
  must reject. Pair with `Watermark` to alert SREs before this hits.

**When to escalate**: add `Statistics<Watermark<...>>` outside the
connection allocator for production observability. Add
`Quarantine<Slab<Connection, _>>` if revoked connections might be
UAF'd by attackers holding stale connection IDs.

**Example**: [`crates/forge-alloc/examples/web_service.rs`](../crates/forge-alloc/examples/web_service.rs)

---

## Recipe 4: `game_frame` — per-frame bump + entity slab + cross-thread audio

**Workload**: real-time game engine. Frame-scoped work, persistent
entity state, low-latency audio thread.

**Composition**:
```rust
type FrameScratch = BumpArena<MmapBacked>;            // per-frame, 1 MiB
type EntityPool   = Slab<Entity, MmapBacked>;         // persistent
type AudioOwner   = SlabOwner<AudioEvent, MmapBacked>;// cross-thread
```

The audio thread holds the `SlabOwner`; the gameplay (main) thread
holds a `SlabRemote` clone and ships event pointers to the audio
thread via an `mpsc::Sender`.

**Why**:
- **Frame scratch**: visibility queries, skinning matrices, particle
  systems, debug strings — all live for one frame. Bump and reset.
- **Entity pool**: bounded count, lifetime spans many frames. A
  typed slab gives stable handles. Use `GenerationalSlab` if you
  want ABA-safe entity IDs that survive entity recycle.
- **Cross-thread audio**: the mixer thread runs at audio block rate
  (~3 ms cadence at 48 kHz/256 samples) and must not stall waiting
  for a lock. A `SlabOwner` owned by the mixer thread + a
  `SlabRemote` clone in the gameplay thread gives lock-free
  message passing.

**Why `MmapBacked` and not `InlineBacked` for the scratch?** Windows
defaults to a 1 MiB thread stack; a 1 MiB `InlineBacked` overflows
it. For per-frame scratch above ~64 KiB, use `MmapBacked`.

**When to escalate**: if entity destruction must trigger T::drop
(common for entities holding resources), use `GenerationalSlab`
which runs T::drop on remove. If audio-thread message rate gets
high, the adaptive `BatchPolicy` already amortises drain cost; if
not adaptive enough, switch to `BatchPolicy::Fixed(N)`.

**Example**: [`crates/forge-alloc/examples/game_frame.rs`](../crates/forge-alloc/examples/game_frame.rs)

---

## Recipe 5: `auth_service` — observable + UAF-resistant token allocator

**Workload**: session tokens for an auth service. Operators need to
see allocation patterns; attackers shouldn't be able to recycle a
freed token slot.

**Composition**:
```rust
type TokenPool =
    Statistics<
        Watermark<
            Quarantine<
                Slab<Token, MmapBacked>,
                4 /* EPOCHS */,
            >,
            NullHandler,
        >,
    >;
```

Stack ordering (outermost to innermost):

1. **`Statistics` (outermost)**: SRE dashboards read `pool.stats()`.
   Putting Statistics outside the security wrappers means its
   counters reflect user-visible request size, not inflated inner
   layouts. (See [`COMPATIBILITY_MATRIX.md`](COMPATIBILITY_MATRIX.md)
   item 17 for the alternate ordering and its trade-off.)
2. **`Watermark`**: fires a handler when capacity utilisation crosses
   warn / critical thresholds. `NullHandler` is a no-op for the
   demo; production uses `LogHandler` (writes to `log::warn!` /
   `log::error!`) or a custom `FnHandler`.
3. **`Quarantine<_, 4>`**: a freed slot is held for 4 subsequent
   frees before returning to the inner slab. An attacker who races
   "free + immediate alloc" hoping to reclaim the same slot is
   foiled across the quarantine window.
4. **`Slab<Token, MmapBacked>` (innermost)**: the actual pool.
   Hardening wraps it; it does the work.

**Why this stack**:
- Observability without security is a false-confidence trap (you
  see what attackers do, but they still do it).
- Security without observability is operationally blind (you'd
  know corruption was happening only post-mortem).
- Together: dashboard alerts on `corruption_events > 0`,
  `Watermark` fires before capacity exhausted, `Quarantine` foils
  the most common UAF reuse pattern.

**When to escalate**: for cryptographic key material rather than
session tokens, replace `Slab<Token, _>` with
`HardenedSlab<Key, SipHashMAC>` (guard pages + split metadata +
freelist MAC). See Recipe 6 for the hardened-allocator usage.

For **secrets that must never reach disk** (private keys, symmetric
keys, passwords), escalate one further to `CryptoSlab<Key, SipHashMAC>`
— the same hardened stack but over `LockedMmapBacked` (`mlock` /
`VirtualLock` so the pages never swap, plus `MADV_DONTDUMP` to exclude
them from core dumps on Linux) and wrapped in `ZeroizeOnFree` (freed
secrets are volatile-zeroed). It **fails closed**: if the memory cannot
be locked (`RLIMIT_MEMLOCK` / missing privilege) construction returns
`Err` rather than silently leaving secrets swappable. Note the
threat-model boundary documented on the `CryptoSlab` / `LockedMmapBacked`
types: the lock does not defend against hibernation, `fork()` COW, or
`ptrace`. See the end-to-end usage in
[`crates/forge-alloc/tests/crypto_slab_e2e.rs`](../crates/forge-alloc/tests/crypto_slab_e2e.rs).

**Example**: [`crates/forge-alloc/examples/auth_service.rs`](../crates/forge-alloc/examples/auth_service.rs)

---

## Recipe 6: `db_executor` — per-query arena + buffer pool + hardened audit log

**Workload**: database query executor. Per-query intermediate state,
shared page pool, compliance-relevant audit log.

**Composition**:
```rust
type QueryArena = BumpArena<MmapBacked>;                          // per-query
type BufferPool = Slab<Page, MmapBacked>;                          // shared
type AuditLog   = HardenedSlab<AuditEntry>;                        // hardened
//                = Slab<AuditEntry, GuardPage<SplitMetadata<MmapBacked>>>
```

**Why**:
- **Per-query arena**: a query's working set (plan, hash tables,
  sort scratch, intermediate join results) is bounded by the
  query lifetime. Bump-allocate everything; reset at query
  completion. No per-allocation tracking.
- **Buffer pool slab**: pages are shared across queries. LIFO reuse
  maximises cache locality (the page just freed by query A is the
  most likely candidate for query B's first read).
- **HardenedSlab for audit**: audit records are compliance-relevant
  (SOX, HIPAA, GDPR). Their integrity must survive an attacker
  who has corrupted other parts of the process. Guard pages trap
  linear overflows from adjacent allocations; split metadata
  moves bookkeeping out of the data region's virtual neighborhood;
  the freelist MAC (with `siphasher` feature) catches forged
  links.

**Cost-aware layering**: the audit log is the only hardened
allocator in this stack. The hot path (per-query arena) and the
warm path (buffer pool) pay no security overhead, because they
don't need it — query intermediates are ephemeral and the buffer
pool's contents are public (they're already on disk). Save the
~115 ns/op cost of `HardenedSlab` for where it matters.

**When to escalate**:
- For multi-tenant databases where one tenant must not be able to
  read another's pages, wrap the buffer pool with
  `PoisonOnFree<Slab<Page, _>>` so freed pages are scrubbed
  before reuse.
- For very large buffer pools, consider
  `HugePageAligned<MmapBacked>` to reduce TLB pressure.
- For high-concurrency executors, swap `BumpArena` for
  `SharedBumpArena` so multiple worker threads can carve from the
  same per-query region.

**Example**: [`crates/forge-alloc/examples/db_executor.rs`](../crates/forge-alloc/examples/db_executor.rs)

---

## How to use these recipes

Match your workload's lifetime classes to the recipe that fits:

| Your workload | Closest recipe |
|---|---|
| Build script / one-shot CLI | `cli_batch` |
| Bare-metal firmware / `no_std` | `embedded_firmware` |
| HTTP/RPC service | `web_service` |
| Real-time game / simulation | `game_frame` |
| Auth / session management | `auth_service` |
| Database / query engine | `db_executor` |

If your workload mixes patterns, pick the closest recipe and graft
on the layer you need (e.g. a service with audit-log requirements
combines `web_service` + the `HardenedSlab` audit from
`db_executor`).

## See also

- [`COMPOSITION_RECIPES.md`](COMPOSITION_RECIPES.md) — type-level cookbook (how to wire one primitive).
- [`PERFORMANCE_TRADEOFFS.md`](PERFORMANCE_TRADEOFFS.md) — what each layer costs (the Criterion bench backs this doc).
- [`COMPATIBILITY_MATRIX.md`](COMPATIBILITY_MATRIX.md) — which combinations don't work or shouldn't be tried.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the three-layer mental model.
