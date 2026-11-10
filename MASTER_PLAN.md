# MASTER_PLAN.md — execution plan for the `gesture_rust` rearchitecture

Derived from `ARCHITECTURE.md`. This is the **sequenced work plan**: what to do, in what
order, how to prove each step worked, and what deliberately stays untouched.

Companion documents: `ARCHITECTURE.md` (diagnosis), `ROADMAP.md` (older feature list),
`AGENTS.md` (project context).

---

## 1. Definition of done

The rearchitecture is finished when all of these are true and measurable:

| Target | Current | Done when |
| :-- | :-- | :-- |
| One frame path | `run` + `tick` + mock branch | one loop; `InputSource::Camera \| Video \| Synthetic` |
| One source of truth for "what is sounding" | 4 places | an `InstrumentState` owned by the simulation layer; sinks translate it |
| `unsafe impl` count | 2 | 0 |
| File I/O / GPU upload on the inference thread | yes (`--ascii-dump`, RGBA conversion) | none; presenter thread owns it |
| Locks in the audio callback | 1 per block | 0 (atomics / double buffer) |
| Longest function | `Pipeline::advance`, 212 lines, CC 29 | no function > ~60 lines, CC ≤ 12 |
| `pipeline.rs` | 1707 lines | < 700 lines |
| Untested presentation code | ~1.65k lines | stage view testable from a snapshot |
| CI | none | fmt + clippy `-D warnings` + check + test green |
| Latency | unmeasured; ~60–90 ms worst case | measured, documented, ≤ 50 ms target |

---

## 2. Guiding rules (apply to every task)

1. **Measure before you refactor.** Every phase 0 number becomes the comparison for
   phase 1–4. A "simplification" that makes the frame loop slower is not one.
2. **One owner per piece of state.** If two threads mutate the same struct, one of them
   is wrong.
3. **The frame loop is a real-time thread.** No locks, no allocation, no syscalls, no
   blocking send. Everything else goes to a queue or a dedicated thread.
4. **Rate tiers are explicit.** Audio (44.1 k) → control (callback block) → gesture
   (30 Hz) → presentation (15 Hz). Each boundary is a named buffer with a stated cadence.
5. **Each task is one commit, independently verifiable.** Prefer additive steps
   (add the estimator, then use it) over big-bang swaps.
6. **Never do two structural changes in one commit.** Fix correctness first, structure
   second — otherwise you cannot tell which change moved the numbers.

---

## 3. Phase 0 — Baseline & guardrails *(before touching anything)*

**Goal:** make the current behaviour measurable and protected, so every later change
can be compared against it.

### T0.1 — Freeze a baseline measurement — **done**, see `bench/baseline.md`
Headline numbers (release build, live camera, MediaPipe): capture p50 **33.2 ms**,
inference p50 **8.2 ms**, camera→publish 0.01 ms, dispatch→sink < 1 ms, 29.4 fps,
0 frames dropped. The camera read is the loop; inference is a quarter of it.

### T0.2 — Local gate (`Makefile`, already written)
No GitHub/GitLab here, so the gate is `make verify` = `check` + `test` + `fmt --check` +
`clippy --all-targets -D warnings`, run before every commit. First OpenCV build is slow;
everything after is warm. **Effort: done.**
*Unblocks:* every later phase gets a regression gate.

### T0.3 — Clear the live lints (the gate is red today)
`make lint-count` reports 8, of which 5 are auto-fixable (`make clippy-fix`):

| Lint | Where |
| :-- | :-- |
| `chunks_exact` with a constant chunk size | `pipeline.rs:859` |
| `push` immediately after creation | `core/geometry.rs:72` |
| manual `RangeInclusive::contains` ×3 | `core/grid.rs:133`, `:178`, `core/scale.rs:116` |
| very complex type | `pipeline.rs:1068` (`test_pipeline` helper) |
| `contains()` instead of `iter().any()` | `pipeline.rs:1493` |
| unnecessary `usize -> usize` cast | `vision/tracker.rs:494` |
| too many arguments (9/7) | `vision/tracker.rs:638` |

All are one-liners except the last two, which are fixed by the Phase 2 refactor anyway
(`bench_sampling` moving to `benches/`, `PipelineConfig` grouping). **Effort S.**
Until they are cleared, `make verify` stops at clippy — the intent is that it stays red
until the tree is clean.

### T0.4 — End-to-end latency instrumentation — **done**
`src/core/telemetry.rs`: a 128-sample ring per stage with nearest-rank percentiles,
published into `FrameMetrics.perf` once a second and printed by `--debug` as
`p50/p95/p99/max` for capture, inference, camera→publish and dispatch→sink. The
audio stream's rate is logged at startup; cpal 0.17 exposes no buffer size, so the
block length still has to be read from the callback (Phase 1).

**Phase 0 exit:** done. Baseline file in the repo, gate green, percentiles printed.

---

## 4. Phase 1 — Hot-path and correctness fixes *(small, isolated, no restructuring)*

Each of these is a standalone commit. Do them in this order.

| Task | Change | Files | Verify | Effort |
| :-- | :-- | :-- | :-- | :-- |
| **1.1** | Use the stream's real sample rate; delete `Synthesizer.sample_rate` | `audio/synthesizer.rs`, `main.rs` | add a test asserting `start()` uses `config.sample_rate`; print it | S |
| **1.2** | Remove `unsafe impl Sync for StreamWrapper`; relax `Sink: Send + Sync` → `Sink: Send` | `audio/synthesizer.rs`, `core/mod.rs` | compile + existing tests | S |
| **1.3** | Sink thread stops polling: `rx.recv()` + a `Shutdown` sentinel (or rely on the sender dropping) | `pipeline.rs`, `core/event_bus.rs` | measured dispatch→sink is already < 1 ms, so this is a CPU/wakeup win, not a latency win | S |
| **1.4** | Band-limited waveforms + frequency ramp. Keep phase in radians; ramp `frequency` toward the target over ~15 ms; synthesise the square/saw with a band limit; soft-clip (`tanh`) instead of `clamp(-1,1)` | `audio/synthesizer.rs`, `core/instrument.rs` | unit test: phase continuity across a frequency step; test: no sample exceeds ±1 | M |
| **1.5** | Mock-mode deadline clock (`next = start + 33 ms`, sleep the remainder) | `pipeline.rs` | assert measured mock fps ≈ 30 | S |
| **1.6** | ORT `intra_op_thread_count` (CLI `--ort-threads`, default 1–2) and, if available, provider selection | `vision/tracker.rs`, `vision/yolo_pose.rs` | `--debug` stage timings before/after | S |
| **1.7** | Detection hysteresis: require N consecutive above-threshold frames before a gesture starts, and a grace window before emitting release | `vision/tracker.rs` or `perception/estimator.rs` | new test: a confidence sequence 0.4/0.6/0.4/0.6 does not flicker | S |
| **1.8** | Capture-resolution experiment: run `--camera-width 320 --camera-height 240` and compare capture_ms; keep whichever wins | none (measurement) | baseline diff | S |
| **1.9** | Remove the duplicated bench scaffolding from `tracker.rs` (three ~20-line copies) into `benches/` | `vision/tracker.rs` | benches still run | S |

**Phase 1 exit:** no locks/allocations/syscalls in the audio callback, no polling in
the event path, honest clocks, measurable latency. Roughly 1 day of work, zero
architectural risk.

### Phase 1 result (branch `phase1-hot-path`)

| Task | Outcome | Measured |
| :-- | :-- | :-- |
| 1.1 | `Synthesizer::new(gain)`; the callback integrates against `config.sample_rate` and prints it | `[Audio] device stream: 44100 Hz, 2 channels` |
| 1.2 | `Sink: Send + Sync` → `Sink: Send`; the only remaining `unsafe` is the documented `StreamWrapper` (`cpal::Stream` is `!Send`, held by the thread that owns it) | compiles, tests green |
| 1.3 | `BusMessage::{Event, Shutdown}`; the sink thread blocks on `rx.recv()` instead of waking every 10 ms; `run` closes the bus to unblock it on exit | dispatch→sink p99 0.1 ms |
| 1.4 | `Voice` with a 15 ms one-pole glide on frequency *and* amplitude, band-limited square/saw (harmonics capped by Nyquist, Gibbs overshoot divided out), averaged voices, `tanh` soft clip | new tests: shape values, range, glide |
| 1.5 | Mock mode runs on a deadline clock (`next += 33 ms`, sleep the remainder) | mock fps 26.4 → 30.3 |
| 1.6 | `--ort-threads` (0 = runtime default). Measured detector p50: 13.7 ms @1 thread, 11.7 @2, 10.8 @4, 9.7 @8 — pinning is a knob, not a win; the frame is paced by the 33 ms camera read | `[Stages]` line |
| 1.7 | `DetectionGate` (2 confirm frames, 3 grace frames) applied in `Pipeline::advance`; a miss inside the grace window keeps the previous keypoints instead of releasing | new pipeline test: flicker never fires, release only after the window |
| 1.8 | skipped — needs a live comparison to eyeball | — |
| 1.9 | Bench scaffolding in `tracker.rs` reduced from four ~20-line copies to `time_stage`/`time_legacy` | `cargo test --release -- --include-ignored bench_sampling` still runs |

Deferred to Phase 3 by design: the audio callback still takes a `Mutex` (held only at
the ~30 Hz event rate, and it allocates nothing), and note attack/release envelopes
belong to the voice layer.

Tests: 112 → 124 passing, gate green (`make verify`).

---

## 5. Phase 2 — State ownership & the frame loop *(the heart of the plan)*

**Goal:** one loop, one config, one snapshot. This is where the 1707-line `pipeline.rs`
becomes readable.

**Done (branch `phase2-config-and-commands`, merged to master):** T2.1 and T2.2.
`RuntimeConfig { active, feed, ascii, feed_fps, ascii_fps, ramp }` replaced the four
`AtomicBool`s and the `AtomicU8` ramp the dashboard used to write directly; the UI now
only sends `UiCommand::{Start, Stop, SetFeed, SetAscii, SetRamp, SetInstrument}` and the
pipeline drains that channel at the top of every frame. `InstrumentConfig.version`,
`Pipeline.applied_version`, `Ramp::index`/`from_index` and the cached
`feed_interval`/`ascii_interval` fields are gone. `Pipeline` has 32 fields (was 38),
127 tests pass, and the gate is green. T2.3–T2.5 are still open.

### T2.1 — `RuntimeConfig` replaces the atomics
```rust
pub struct RuntimeConfig {
    pub active: bool, pub feed: bool, pub ascii: bool,
    pub ramp: Ramp, pub labels: bool, pub show_box: bool,
}
```
One `Arc<Mutex<RuntimeConfig>>`; the pipeline snapshots it once per tick. Delete
`active`, `feed_wanted`, `ascii_wanted`, `ascii_ramp` atomics and their `next_feed` /
`next_ascii` / interval fields (keep the cadence logic, move it into the presenter).
**Files:** `main.rs`, `pipeline.rs`, `dashboard/mod.rs`. **Effort S.**
*Verification:* the toggle tests still pass; the struct field count of `Pipeline` drops.

### T2.2 — Command channel replaces the version counter
`crossbeam::channel<UiCommand>` with `SetInstrument(InstrumentConfig)`,
`SetCadence { feed_fps, ascii_fps }`, `SetRamp(Ramp)`, `Pause`, `Resume`, `Shutdown`.
Delete `InstrumentConfig.version` and `Pipeline.applied_version`. The UI sends
commands; the pipeline receives them in the same loop that reads frames (or a small
select). **Effort S.**
*Verification:* the picker-propagation test
(`a_picker_change_reaches_the_sinks_while_the_loop_is_running`) still passes, now
asserting on a received command.

### T2.3 — `InputSource` enum; delete `Pipeline::tick`
```rust
enum InputSource {
    Camera { cam: WebcamSource },
    Video { decoder: ..., fps: f32 },
    Synthetic { cadence: Duration, phase: f64 },
}
```
`run` becomes: receive → `advance` → loop. Mock and video are sources, not
special cases. `Option<HandTracker>` stays (mock genuinely has no tracker) but the
`Option<WebcamSource>` and the mock branch in `run` disappear. Move the inline `tick`
path into tests as a helper that constructs a synthetic source and calls `run` with
`max_frames`. **Effort M.**
*Verification:* all pipeline tests still pass through one code path.
*Unblocks:* `--video` reproducible runs, fixture tests, the benchmark harness.

### T2.4 — Split `advance()` into four named stages
```rust
fn advance(&mut self, frame: CapturedFrame) -> Result<()> {
    self.sync_instrument();
    let detection = self.infer(&frame)?;
    self.update_tracking(&detection);
    self.update_visuals(&frame, &detection);   // presenter thread, not here
    self.dispatch_gesture(&detection);
    self.publish();
}
```
Each stage ~40 lines, each unit-testable in isolation. **Effort M.**
*Verification:* CC drops from 29 to ≤ 12 per function; new tests per stage.

### T2.5 — Group `Pipeline`'s 32 fields
`Pipeline { source, tracker, strategy, instrument, presenter, control, metrics, clocks }`.
**Effort S.** Do it as the tail of T2.4.

**Phase 2 exit:** `run` is the only loop; `advance` is four functions; the UI talks to
the pipeline through commands and a snapshot, not four atomics and a version number.
Roughly 2–3 days. **Risk:** highest of the plan — land each task as a separate commit
with the existing 112 tests as the guard.

---

## 6. Phase 3 — The missing instrument layer

**Goal:** one object that knows what is sounding, one estimator that knows where the
hand is.

### T3.1 — Canonical `HandState`
Add `perception/estimator.rs`: takes raw landmark detections, applies hysteresis +
smoothing, outputs
```rust
struct HandState { landmarks: [Landmark; 21], palm: (f32,f32), bbox: Rect,
                   spread: f32, extension: [f32;5], tilt: f32, confidence: f32,
                   present: bool }
```
Delete `strategies::spread_volume` (duplicates `geometry::spread` with a different
normalisation) and fold `TipSmoother` into the estimator. Document the smoothing
constant in one place instead of three. **Effort M.**
*Verification:* one test suite for the estimator (jitter suppression vs sweep
following — the existing test `the_smoothing_kills_jitter_but_follows_a_sweep` moves
there).

### T3.2 — `Strategy::process(&mut self, &HandState)`
Change `Strategy` to `process(&mut self, state: &HandState) -> Option<Performance>`.
Delete the `Mutex` in `TipSmoother`. Re-evaluate whether `Sink: Sync` is needed at all
(Phase 1 already relaxed it). **Effort S.**

### T3.3 — Voice/envelope module
`instrument/voice.rs`: `Voice { freq_target, freq_current, amp, env: Adsr, timbre }`
with `attack()`, `release()`, `set_target()`. Both sinks consume it: audio renders it,
MIDI maps it onto note-on/off + CC. `NoteGate` (already excellent) moves into
`output/midi_sink.rs` unchanged. **Effort M.**

### T3.4 — A real `Release`, not `note: 0`
Replace the `GestureEvent::no_gesture()` dummy payload with
`enum Performance { NoteOn { note, vel }, NoteOff, SetVolume(f32), Signal { freq, amp } }`
or keep the two-variant payload but add an explicit `Release` variant. The `note == 0`
magic value and the "dummy payload" comment disappear. **Effort S.**

### T3.5 — Presentation out of the event type
`GestureEvent.text` and `.color` are UI fields riding through every sink. Move them to
the snapshot; derive colour from the pitch class in the presenter. **Effort S.**

**Phase 3 exit:** adding a new output (OSC) or a new gesture touches exactly one file.
Roughly 2 days.

---

## 7. Phase 4 — Presentation decoupling

**Goal:** drawing is a pure function of a snapshot; nothing that touches the GPU or the
disk runs on the inference thread.

1. **T4.1 — `presentation/` module.** Move `vision/ascii.rs` → `presentation/ascii.rs`;
   move `visualizer/mod.rs` into it (or delete it once the dashboard can dump a frame).
   **Effort S.**
2. **T4.2 — Presenter thread.** A `Presenter` receives `(frame_pixels, ascii_grid)` at
   its own cadence on its own thread. `--ascii-dump`, stdout printing and RGBA
   conversion all move there. Delete six throttle fields from `Pipeline`. **Effort M.**
3. **T4.3 — `StageView` snapshot.** The UI receives
   `StageView { hand: HandStateView, instrument: InstrumentView, perf: PerfView, trail: [f32;49] }`
   instead of the 39-field `FrameMetrics`. Split `FrameMetrics` into
   `Tracking` / `Performance` / `Perf` / `Context` sub-structs first, then compose.
   **Effort M.**
4. **T4.4 — Move `note_trail` into the published snapshot** (ROADMAP finding 10): the
   pipeline owns the decay, the UI only draws it. **Effort S.**
5. **T4.5 — Headless stage tests.** `Skeleton::draw` and `keyboard()` become pure
   functions returning primitives; snapshot-test the primitive lists (cell count for a
   given layout, letterbox mapping, trail brightness). This is the cheapest way to
   cover 1.65k currently untested lines. **Effort M.**
6. **T4.6 — Cache the static backdrop.** The 49 base cells + separators + octave labels
   only change when the panel size or layout changes. **Effort S.**

**Phase 4 exit:** no I/O or GPU work on the frame thread; the UI is testable. ~2 days.

---

## 8. Phase 5 — Observability & test infrastructure

| Task | Detail | Effort |
| :-- | :-- | :-- |
| T5.1 | `tracing` replacing `println!`/`eprintln!`; `--debug` = `tracing_subscriber` env filter | S |
| T5.2 | `--video clip.mp4` source (from T2.3) so runs are reproducible | M |
| T5.3 | Tracker fixture test: a 640×480 PNG with a hand → assert gold bbox ± tolerance and 21 landmarks within bounds | M |
| T5.4 | Golden trace: `--max-frames 120 --expect trace.json` asserting the note sequence | M |
| T5.5 | Percentiles (p50/p95/p99) per stage in the Performance card | S |
| T5.6 | Optional subsystem budget: if inference > budget, auto-skip ASCII/feed and say so in the log | S |

---

## 9. Phase 6 — Features unlocked by the new boundaries

Only after phases 2–4. Each is now a single-file change instead of a five-file one:

- **Multi-hand** — `Vec<HandState>` from the estimator, roles in the instrument layer.
  (Currently would require simultaneous edits to `Pipeline`, `FrameMetrics`,
  `Skeleton::draw` and `Dashboard` — that is the clearest symptom of the missing layers.)
- **OSC sink** — implement `Sink`, push into the sink list. S.
- **Named instrument presets** (`--preset kalimba`) — one struct in `instrument/`. S.
- **`--calibrate`** — per-user band replacing `NOTE_MARGIN`. M.
- **Keyboard shortcuts** for Start/Stop and toggles. S.
- **MIDI polyphony** — voice pool, now in one place. M.
- **Use `Landmark.z`** — depth for proximity/volume. M.

---

## 10. Dependency graph & ordering

```
Phase 0 (baseline, CI, latency)
   │
   ├─► Phase 1 (hot-path fixes: 1.1→1.9, independent of each other)
   │        │
   │        ▼
   ├─► Phase 2 (2.1 → 2.2 → 2.3 → 2.4 → 2.5)   ← strict order
   │        │
   │        ▼
   ├─► Phase 3 (3.1 → 3.2 → 3.3 → 3.4, 3.5 anytime after 2.4)
   │        │
   │        ▼
   ├─► Phase 4 (4.1, 4.2 need 2.3/2.4; 4.3 needs 4.2; 4.5 needs 4.3)
   │        │
   │        ▼
   └─► Phase 5, then Phase 6
```

**Estimated total:** ~10 working days for phases 0–4, ~3 days for phase 5.
Phases 0 and 1 are ~2 days and carry almost no risk; they are worth doing even if the
rest never happens.

---

## 11. Suggested first three commits (start here)

1. `chore: CI + baseline measurements` — T0.1, T0.2, T0.3.
2. `fix(audio): use the device sample rate; remove unsafe Sync impl` — T1.1, T1.2.
3. `feat(telemetry): p50/p95 stage latency + dispatch→sink delta` — T0.4, T1.6 partially.

After three commits you have the numbers, the gate, and a measurable latency figure —
everything later is judged against them.

---

## 12. Explicit non-goals

- **Do not rewrite the UI framework.** iced is working; swapping to egui is a rewrite,
  not a simplification. Only reconsider if the UI keeps growing after it is testable.
- **Do not add multi-hand, OSC or GPU providers before phases 2–4.** They currently
  require touching four subsystems at once; after the boundaries they require one.
- **Do not micro-optimise clones.** The ROADMAP already measured snapshot cloning at
  0.44 µs/frame — 0.0015 % of a 33 ms frame. The camera read (31 ms) and inference
  (8–69 ms) are the only real costs.
- **Do not merge structural and behavioural changes in one commit.** If a refactor and
  a tuning change land together, the before/after numbers are meaningless.
- **Do not delete `ReuseCache`, `NoteGate`, or the affine sampler.** These are the
  parts of the codebase that are already at industry standard.

---

## 13. Progress tracker

| Phase | Tasks | Effort | Status |
| :-- | :-- | :-- | :-- |
| 0 Baseline & CI | T0.1–T0.4 | 1 d | ☑ done |
| 1 Hot path & correctness | T1.1–T1.9 | 1–2 d | ☑ done (T1.8 skipped) |
| 2 Frame loop & state | T2.1–T2.5 | 2–3 d | ◑ T2.1, T2.2 done |
| 3 Instrument layer | T3.1–T3.5 | 2 d | ☐ |
| 4 Presentation | T4.1–T4.6 | 2 d | ☐ |
| 5 Observability & test infra | T5.1–T5.6 | 3 d | ☐ |
| 6 Unlocked features | — | ongoing | ☐ |

Fill in per task: branch name, before/after frame time, before/after test count, and
whether the `Definition of done` table row is satisfied.
