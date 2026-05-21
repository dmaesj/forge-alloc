//! `game_frame` — per-frame bump + persistent slab + cross-thread audio.
//!
//! Real-time game architecture: one allocator per "lifetime class"
//! the game engine recognises. Frame-scoped work goes to a per-frame
//! bump arena; entity state lives in a slab; the audio thread shares
//! a typed pool with the main thread via SlabOwner / SlabRemote.
//!
//! ```text
//!   main thread (60 Hz frame loop)
//!         │
//!         ▼
//!   ┌────────────────────────────────────────────────────────┐
//!   │  FRAME SCRATCH: BumpArena<MmapBacked>                   │
//!   │  1 MiB OS-mapped region, reset at end of every frame    │
//!   │  ~2 ns/alloc; deterministic per-frame budget            │
//!   └────────────────────────────────────────────────────────┘
//!         │
//!         ▼
//!   ┌────────────────────────────────────────────────────────┐
//!   │  ENTITIES: Slab<Entity, MmapBacked>                     │
//!   │  fixed capacity; LIFO reuse on entity destruction       │
//!   │  ~53 ns/alloc; lifetime spans many frames               │
//!   └────────────────────────────────────────────────────────┘
//!         │ (spawns audio events via SlabRemote handle)
//!         ▼
//!   ┌────────────────────────────────────────────────────────┐
//!   │  AUDIO MESSAGES: SlabOwner<AudioEvent, MmapBacked>     │
//!   │                  ─ owner thread: audio mixer            │
//!   │                  ─ remote handle: main thread sends    │
//!   │  Send+Sync remote; main thread enqueues via remote      │
//!   │  Audio thread owns the local slab; drains on its loop  │
//!   └────────────────────────────────────────────────────────┘
//! ```
//!
//! Why this composition:
//! - **Per-frame bump**: a frame's working set (visibility queries,
//!   skinning matrices, particle systems, debug strings) is
//!   ephemeral. Bump-allocate everything; reset at frame end. No
//!   per-allocation cost, no fragmentation, deterministic.
//! - **Entity slab**: entities outlive frames but have bounded count.
//!   A typed slab gives stable handles (via raw pointer or via
//!   `GenerationalSlab` if you want ABA-safe IDs).
//! - **SlabOwner / SlabRemote for cross-thread audio**: the audio
//!   mixer owns its own pool (low-latency, no lock contention with
//!   gameplay). The main thread (which can spawn sound effects)
//!   holds a `SlabRemote` clone. Sends are lock-free push via the
//!   remote handle; the audio thread drains on its own schedule.

use forge_alloc::{
    Allocator, BatchPolicy, BumpArena, Deallocator, MmapBacked, NonZeroLayout, Slab, SlabOwner,
};
use std::sync::mpsc;
use std::thread;

// ============================================================================
// Composition type aliases
// ============================================================================

type FrameScratch = BumpArena<MmapBacked>;
type EntityPool = Slab<Entity, MmapBacked>;
type AudioOwner = SlabOwner<AudioEvent, MmapBacked>;

// ============================================================================
// Domain types
// ============================================================================

#[repr(C)]
struct Entity {
    id: u32,
    x_q16: i32,
    y_q16: i32,
    z_q16: i32,
    health: u16,
    flags: u16,
}

#[repr(C)]
struct AudioEvent {
    sample_id: u32,
    volume_q8: u16,
    pan_q8: u16,
    timestamp_us: u64,
}

fn main() {
    println!("game_frame — per-frame bump + entity slab + cross-thread audio");
    println!("--------------------------------------------------------------");

    // === Entity slab (warm tier) ===
    let entities: EntityPool = Slab::new(4096, MmapBacked::new(1 << 20).unwrap()).unwrap();
    let entity_layout = NonZeroLayout::for_type::<Entity>().unwrap();

    // === Audio owner + remote handle (cross-thread tier) ===
    //
    // Main thread is the OWNER of the audio-event pool. It allocates
    // events when entities trigger sound effects, then ships them to
    // the audio thread via a channel. The audio thread holds a
    // SlabRemote handle and deallocates events after processing —
    // the dealloc is enqueued for the owner to drain on its next
    // alloc/dealloc call.
    let audio_backing = MmapBacked::new(1 << 16).unwrap();
    let audio_owner: AudioOwner = SlabOwner::with_batch_policy(
        1024,
        audio_backing,
        BatchPolicy::Adaptive,
        1024, // remote queue capacity
    )
    .unwrap();
    let audio_remote = audio_owner.remote(); // Send + Sync — moves to audio thread
    let audio_layout = NonZeroLayout::for_type::<AudioEvent>().unwrap();

    // Channel for main → audio. We send the raw pointer + layout pair;
    // SAFETY: pointer is owned by the audio_owner pool, valid until
    // the remote.deallocate call below releases it.
    let (audio_tx, audio_rx) = mpsc::channel::<(usize, NonZeroLayout)>();

    // Spawn the audio "thread" — in a real engine this is the mixer.
    let mixer_handle = thread::spawn(move || {
        let mut events_processed = 0;
        while let Ok((ptr_addr, layout)) = audio_rx.recv() {
            // SAFETY: ptr_addr came from audio_owner.allocate, which we
            // received via the channel. We're the only consumer of this
            // event; deallocating returns its slot to the owner's queue.
            let ptr = unsafe {
                core::ptr::NonNull::new_unchecked(ptr_addr as *mut u8)
            };
            unsafe { audio_remote.deallocate(ptr, layout) };
            events_processed += 1;
        }
        events_processed
    });

    // === Spawn some persistent entities ===
    let entity_ptrs: Vec<_> = (0..3)
        .map(|i| {
            let p = entities.allocate(entity_layout).unwrap();
            unsafe {
                let e = p.cast::<Entity>().as_ptr();
                (*e).id = i as u32;
                (*e).x_q16 = i << 16;
                (*e).health = 100;
            }
            println!("  warm: spawned entity id={i}");
            p
        })
        .collect();

    // === Frame loop ===
    let mut scratch = FrameScratch::new(MmapBacked::new(1 << 20).unwrap()).unwrap();
    let scratch_layout = NonZeroLayout::from_size_align(256, 16).unwrap();

    for frame in 0..3 {
        println!("\nframe {frame}:");

        // Frame scratch: simulate visibility + skinning matrices.
        let _viz = scratch.allocate(scratch_layout).unwrap();
        let _skin = scratch.allocate(scratch_layout).unwrap();
        println!("  hot:   2 scratch allocs ({} B total)", 2 * 256);

        // Entity update: just touch each.
        for (i, p) in entity_ptrs.iter().enumerate() {
            unsafe {
                let e = p.cast::<Entity>().as_ptr();
                (*e).x_q16 += 1 << 14; // move slightly
                if frame == 0 && i == 0 {
                    // Trigger a sound effect — allocate on the owner
                    // (this thread) and ship the pointer to the
                    // audio thread via the channel.
                    let evt_p = audio_owner.allocate(audio_layout).unwrap();
                    let evt = evt_p.cast::<AudioEvent>().as_ptr();
                    (*evt).sample_id = 42;
                    (*evt).volume_q8 = 200;
                    (*evt).pan_q8 = 128;
                    (*evt).timestamp_us = frame as u64 * 16_667; // 60 Hz
                    let addr = evt_p.cast::<u8>().as_ptr() as usize;
                    audio_tx.send((addr, audio_layout)).unwrap();
                    println!("  audio: queued sample event from entity #{i}");
                }
            }
        }

        // End-of-frame: reset scratch. Everything for this frame is gone.
        scratch.reset();
        println!("  scratch reset");
    }

    // Tear down entities.
    for (i, p) in entity_ptrs.into_iter().enumerate() {
        unsafe { entities.deallocate(p.cast(), entity_layout) };
        println!("\nwarm: despawned entity #{i}");
    }

    // Close the channel — signals the audio thread's recv() to error.
    drop(audio_tx);
    let processed = mixer_handle.join().unwrap();
    println!("audio mixer joined after processing {processed} events");
    // audio_owner drops here; SlabOwner::drop sets closed=true under
    // the queue mutex and drains any remaining remote-pushed entries.
}
