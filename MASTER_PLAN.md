# MASTER_PLAN.md — execution plan for the `gesture_rust` rearchitecture

Derived from `ARCHITECTURE.md`. This is the **sequenced work plan**: what to do, in what
order, how to prove each step worked, and what deliberately stays untouched.

Companion documents: `ARCHITECTURE.md` (diagnosis), `ROADMAP.md` (older feature list),
`AGENTS.md` (project context).

---

## 1. Definition of done

The rearchitecture is finished when all of these are true and measurable:

| Target | Started at | Done when | Where we are |
| :-- | :-- | :-- | :-- |
| One frame path | `run` + `tick` + mock branch | one loop; `FrameSource::Camera \| Video` | ☑ one loop (`next_frame` → `advance`); ☑ `FrameSource` enum now has its second member (`--video`) |
| One source of truth for "what is sounding" | 4 places | an `InstrumentState` owned by the simulation layer; sinks translate it | ☑ `HandState` owns the hand, `Performance` owns what is sounding, `Voice` owns what is audible |
| `unsafe impl` count | 2 | 0 | ◑ 1 left: `StreamWrapper` (`cpal::Stream` is `!Send`), documented |
| File I/O / GPU upload on the inference thread | yes (`--ascii-dump`, RGBA conversion) | none; presenter thread owns it | ☑ moved to the presenter thread (Phase 4, T4.2) |
| Locks in the audio callback | 1 per block | 0 (atomics / double buffer) | ☑ 0: the callback reads four atomics, the voice is owned by the closure |
| Longest function | `Pipeline::advance`, 212 lines, CC 29 | no function > ~60 lines, CC ≤ 12 | ☑ `advance` is 27 lines calling six named stages |
| `pipeline.rs` | 1707 lines | < 700 lines | ◑ 1905 lines total, **1099 excluding the test module**; the next cut is moving telemetry out |
| Untested presentation code | ~1.65k lines | stage view testable from a snapshot | ◑ the presenter and the keyboard geometry are testable; the iced rail is not |
| Reproducible runs | impossible (a webcam is not a fixture) | `--video clip.mp4` + `--expect trace.json` | ☑ both exist; `make trace-check` is a gate |
| CI | none | fmt + clippy `-D warnings` + check + test green | ☑ `make verify`, green |
| Latency | unmeasured; ~60–90 ms worst case | measured, documented, ≤ 50 ms target | ☑ measured, in `bench/baseline.md`; the loop is paced by the 33 ms camera read |

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

Deferred to Phase 3 by design: the audio callback still took a `Mutex`, and note
attack/release envelopes belonged to the voice layer. Both were cleared in Phase 3 —
see §6.

Tests: 112 → 124 passing, gate green (`make verify`).

---

## 5. Phase 2 — State ownership & the frame loop *(the heart of the plan)*

**Goal:** one loop, one config, one snapshot. This is where the 1707-line `pipeline.rs`
becomes readable.

**Done (branches `phase2-config-and-commands` and `phase2-loop-rewrite`): T2.1–T2.5.**
`RuntimeConfig { active, feed, ascii, feed_fps, ascii_fps, ramp }` replaced the four
`AtomicBool`s and the `AtomicU8` ramp the dashboard used to write directly; the UI now
only sends `UiCommand::{Start, Stop, SetFeed, SetAscii, SetRamp, SetInstrument}` and the
pipeline drains that channel at the top of every frame. `InstrumentConfig.version`,
`Pipeline.applied_version`, `Ramp::index`/`from_index` and the cached
`feed_interval`/`ascii_interval` fields are gone. `run` is the only loop, `advance` is six
small stages, and `Pipeline` has 17 fields holding four subsystem structs instead of 27
loose ones. 127 tests pass, the gate is green, and the measured loop is unchanged.

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

### T2.3 — one frame path; `Pipeline::tick` deleted — **done**
`run` is now receive → `advance` → loop, and mock mode and the camera are produced by the
same function (`next_frame`) feeding the same `advance`.

*Deviation from the sketch:* there is no `InputSource` enum yet. `Option<WebcamSource>`
is still the discriminator because a video source does not exist, and inventing an enum
before its second member would be abstraction for its own sake. When `--video` lands, the
enum goes into `next_frame` and nothing else changes. The inline single-threaded path the
tests used to rely on is replaced by `advance(synthetic_frame(t))`.
*Verification:* all pipeline tests run through one code path; measured loop unchanged.
*Unblocks:* `--video` reproducible runs, fixture tests, the benchmark harness.

### T2.4 — Split `advance()` into named stages — **done**
Six functions of 20–60 lines each — `debounce`, `record_timing`, `update_preview`,
`update_tracking`, `update_ascii`, `dispatch_gesture` — instead of one 210-line body,
followed by `publish` and `report_frame`. `update_preview` and `update_ascii` are now
separate from the tracking, which is what makes moving them to a presenter thread
(Phase 4) a cut-and-paste job rather than a surgery. **Effort M.**
*Verification:* the cadence tests (`the_feed_is_only_converted_when_the_stage_shows_it`,
`the_preview_upload_is_throttled_to_its_cadence`,
`the_ascii_layer_only_runs_when_something_wants_to_see_it`) now exercise one stage each.

### T2.5 — Group `Pipeline`'s fields — **done**
27 loose fields → 17, of which four are named subsystems: `RunFlags { max_frames,
emit_frames, debug }`, `Telemetry { started, last_publish, last_detection, avg_gap_ms,
next_mock, the capture/inference/latency rings, the dispatch ring }`, `Presentation { ascii
layer, cols, stdout, dump, next_feed, next_ascii }` and `CaptureCounters`. **Effort S.**
*Verification:* gate green; measured loop unchanged.

**Phase 2 exit:** reached. `run` is the only loop, `advance` is six functions, and the UI
talks to the pipeline through commands and a snapshot rather than four atomics and a
version number.

### Phase 2 result — before/after (release build, 300 frames, live camera)

| | before (Phase 1) | after (Phase 2) |
| :-- | :-- | :-- |
| fps | 29.4 | **29.9** |
| ms/frame | 33.30 | 33.31 |
| capture p50 / p95 | 33.2 / 37.3 | 33.5 / 38.4 |
| inference p50 / p95 | 8.2 / 14.7 | 11.3 / 14.2 |
| dropped frames | 0 | 0 |
| mock fps | 30.3 | 30.4 |

Inference p50 moved from 8.2 to 11.3 ms between the two runs — the same code, so this is
machine drift (the earlier run had no hand in view, so the landmarker mostly did not run;
the Phase 2 run had a hand and ran it). The refactor is a wash on speed, which is the
point: the win is that the loop is one path with named stages, not a micro-optimisation.
With the UI open (250 frames, `--ascii`): 29.7 fps, capture 33.2 p50, inference 11.5 p50,
2 frames dropped.

---

## 6. Phase 3 — The missing instrument layer *(done, branch `phase3-instrument`)*

**Goal:** one object that knows what is sounding, one estimator that knows where the
hand is.

### Phase 3 result

| Task | Outcome | Verified |
| :-- | :-- | :-- |
| **T3.1** `HandState` | `src/gesture/estimator.rs`: `HandEstimator::update(&HandDetectionResult) -> HandState { present, landmarks, hand_box, frame_size, confidence, tip, volume, geometry, jitter_px }`. It owns the hysteresis gate, the fingertip EMA, the geometry and the jitter measurement; the strategies read it and mutate nothing. `strategies/` → `gesture/`. | estimator test suite: jitter suppression vs sweep following, gate confirm/grace behaviour |
| **T3.2** `Strategy::process(&mut self, &HandState)` | Done. `TipSmoother` and its `Mutex` are gone from both strategies; `hand_geometry` now takes `&[Landmark]` instead of the whole `HandDetectionResult` (the tracker's box no longer travels through the gesture layer). | keyboard/theremin tests pass unchanged in meaning |
| **T3.3** Voice/envelope | Partly, and deliberately. The voice is one monophonic `Voice` in `audio/synthesizer.rs` with a one-pole envelope: attack 12 ms, release 60 ms, pitch glide 15 ms. A released voice fades to `SILENT` and the callback then skips the oscillator. **Deviation:** no `instrument/voice.rs`. A `Voice` is an audio-rendering object; MIDI's "voice" is a note gate with a completely different lifecycle, so sharing the struct would couple two sinks that have nothing in common. The shared abstraction that *does* exist is `Performance`, which both sinks consume. | `the_envelope_ramps_up_and_down_instead_of_stepping`, `a_frequency_change_glides_instead_of_jumping` |
| **T3.4** A real `Release` | `enum Performance { Note { note, velocity }, Signal { frequency, amplitude }, Release }`. `GestureEvent` and `GesturePayload` are deleted; `note == 0` and the "dummy payload" comment are gone. `NoteGate::attack` no longer special-cases note 0 — `Performance::Release` calls `gate.release()` directly. | MIDI test `the_release_event_ends_the_note_exactly_once` |
| **T3.5** Presentation out of the event type | Done. `Performance` carries no `text`/`color`; `EventLogEntry` is `{ uptime_ms, detail }` and the log line is derived from the performance. The backdrop colour is derived from the pitch class in the presenter. | — |

The same commit made the audio callback lock-free (the Phase 1 deferral): the control
side writes four atomics (`Target { sounding, frequency, amplitude, waveform }`) and the
callback owns the voice outright. No `Mutex`, no allocation, no syscall inside the block.

**Phase 3 exit:** adding a gesture is one file in `src/gesture/`; adding an output is one
`Sink` impl in `src/output/` plus one line in `main.rs`. `Sink::handle_event` became
`Sink::handle(&Performance)`. Tests 127 → 131.

---

## 7. Phase 4 — Presentation decoupling *(done: T4.1–T4.6)*

**Goal:** drawing is a pure function of a snapshot; nothing that touches the GPU or the
disk runs on the inference thread.

1. **T4.1 — `presentation/` module — done.** `vision/ascii.rs` → `presentation/ascii.rs`,
   `visualizer/mod.rs` → `presentation/annotator.rs` (the `--image` skeleton dump).
   `presentation/` is now `ascii`, `presenter`, `annotator`.
2. **T4.2 — Presenter thread — done.** `Presenter` owns the ASCII layer, the two cadence
   deadlines, the stdout mode and the dump path, and runs on its own thread draining a
   capacity-1 `PresentJob` channel. `Pipeline::advance` now calls one function,
   `hand_over`, which moves the frame (no copy) into that channel and does no drawing.
   `update_preview` and `update_ascii` are deleted from the pipeline; so are
   `frame_to_rgba`, `bgr_to_rgba`, `print_ascii`, `feed_interval`, `feed_due`,
   `should_upload_feed` and `ascii_wanted` — five of `Presentation`'s six fields went with
   the presenter, and the two that remain (`stdout`, `dump`) only decide whether a frame
   is worth handing over at all.
   `publish` moves the presenter's pixels and grid out of the shared snapshot, replaces the
   rest, and moves them back, so the two threads never overwrite each other.
   **Deviation:** the job is `(Option<Mat>, Vec<Landmark>, frame_size)` rather than
   `(frame_pixels, ascii_grid)` — the presenter does the conversion itself, which keeps the
   frame thread from touching the RGBA buffer at all.
3. **T4.3 — `StageView` snapshot — done.** `FrameMetrics` is now five named views instead
   of ~35 loose fields: `hand` (where the hand is, plus `HandGeometry`), `sound` (note,
   volume, frequency, trail, fade), `perf` (fps, stage profile, per-stage times, counters),
   `scene` (RGBA pixels, ASCII grid, keyboard layout) and `status` (mode, sink, scale, root,
   waveform, paused, log). Each view has one writer: the frame loop fills
   `hand`/`sound`/`perf`/`status`, the presenter thread fills `scene.pixels`/`scene.ascii`.
   `publish` moves the scene out, replaces everything else, and moves the picture back, so
   the two threads cannot overwrite each other.
4. **T4.4 — `note_trail` into the snapshot — done.** `KeyboardTrail` lives in
   `dashboard/metrics.rs` and is owned by the `Pipeline`; `advance()` advances it once per
   frame and publishes `trail` + `fade_ms`. `Dashboard`'s `note_trail`, `fade_tau` and
   `last_tick` are gone - the stage draws a fact about the run, not a history of the window.
   Pausing puts the keyboard dark too.
5. **T4.5 — Headless stage tests — done.** The keyboard geometry is computed by a `Keyboard`
   struct (`band`, `cell_w`, `cell_h`, `cell(index)`) that knows nothing about iced. Two
   tests exercise it without a window: the drawn band is inset by exactly `NOTE_MARGIN`, and
   for 400 fingertip positions the cell the canvas lights is the cell `band_position` +
   `GridLayout::cell_at` picks. That is the invariant the old comment only claimed.
6. **T4.6 — Cache the static backdrop — done.** The gradient/glow layer and the keyboard
   grid (cell tints, separators, octave labels, note names) are iced `canvas::Cache`s held
   in the program's `State`, invalidated only when the capture resolution or the instrument
   changes. Roughly 75 path constructions per frame became two per window resize.

**Measurements after T4.2 (release build, 300 frames):** mock 30.2 fps / 32.91 ms per
frame, 0 dropped; live camera `--no-ui` 29.9 fps, capture p50 33.4 ms, inference p50 10.2
ms, 0 dropped; live with `--ascii` (headless) 29.4 fps, capture p50 33.3 ms, inference p50
11.7 ms, **0 dropped** — Phase 2's UI+ASCII run dropped 2 frames. Tests 131 → 139.

**Phase 4 exit (met):** no I/O or GPU work on the frame thread (the `image::Handle` upload
is iced's job on the UI thread, which is where it belongs), and the stage is testable from
a snapshot.


---

## 8. Phase 5 — Observability & test infrastructure *(complete: T5.1–T5.6)*

| Task | Detail | Effort | Status |
| :-- | :-- | :-- | :-- |
| T5.1 | `tracing` replacing `println!`/`eprintln!`; `--debug` = `tracing_subscriber` env filter | S | ☑ |
| T5.2 | `--video clip.mp4` source (from T2.3) so runs are reproducible | M | ☑ |
| T5.3 | Tracker fixture test: real ONNX graphs over a recorded clip | M | ☑ |
| T5.4 | Golden trace: `--max-frames 120 --expect trace.json` asserting the note sequence | M | ☑ |
| T5.5 | Percentiles (p50/p95/p99) per stage in the Performance card | S | ☑ |
| T5.6 | Optional subsystem budget: if inference > budget, auto-skip ASCII/feed and say so in the log | S | ☑ |

### Phase 5 result

**T5.2 — `VideoSource` (`src/vision/video.rs`).** `next_frame` now dispatches over a
`FrameSource::{Camera, Video}` enum — the second member finally exists, so the abstraction
T2.3 deferred is now justified rather than speculative. The clip is paced by a **deadline
clock at its own reported frame rate**, and that pacing lives in `wait_for_next()`, *outside*
the timed capture region: `capture_ms` measures the 0.2 ms OpenCV decode, not the 33 ms wait.
Fold the two together and a fixture run reports a capture cost it never paid.

**T5.4 — `src/core/trace.rs`.** A `Trace` is a `Vec<TraceFrame>` of
`{ frame, note, frequency, volume, detected }`, volumes rounded to two decimals because the
last digits of an oscillator's geometry are not reproducible. `--trace <path>` writes one,
`--expect <path>` diffs one and exits non-zero on a difference. `make trace-check` is the gate.
The mock hand is driven by the **frame index** rather than the wall clock, which is what makes
the mock trace a golden reference at all: `the_mock_hand_walks_up_the_keyboard_in_a_stable_order`
asserts `[(64, 13), (67, 18), (69, 28)]` — three held notes, not sixty identical events.

**T5.3 — tracker fixture test.** `tracker_finds_the_hand_in_the_clip` runs the real
MediaPipe graphs over 120 frames of `fixtures/hand_clip.mp4`: 106/120 detections, every
detection carrying 21 landmarks, every box square and plausibly placed. It is `#[ignore]`d
because the ONNX assets are gitignored; `cargo test --release -- --include-ignored
tracker_finds_the_hand_in_the_clip` is the way to run it.

**T5.1 — `tracing`.** Every production `println!`/`eprintln!` is gone; the six that remain
are in tests and benches. One `init_logging(debug: bool)` in `main.rs` installs an
`EnvFilter` — `INFO` by default, `DEBUG` under `--debug`, and `RUST_LOG` wins over both. The
subscriber also pins `ort::logging` to `warn`, which is what the `debug: bool` threaded
through four structs was really trying to do.

The point is not the log lines, it is the **parameters that stopped existing**. `debug: bool`
was a field on `MediaPipeTracker`, `YoloPoseTracker`, `Presenter`, `PipelineConfig` and
`RunFlags`: five copies of one opinion about verbosity, passed down because the leaf stages
could not otherwise decide for themselves. A level filter is set once, at the top, and every
module emits at a level without knowing who is listening. The expensive diagnostics — the
anchor table, the 21 landmark coordinates — are still only computed when they can be used,
guarded by `tracing::enabled!(Level::DEBUG)` rather than by a struct field.

**T5.5 — percentiles on the Performance card.** `PerfView` already held a `StageProfile` per
stage; the card now prints `p50 / p95 / p99 / max` for capture, inference, camera→UI, dispatch
and the tail window, with a marker when a stage's p95 exceeds 33 ms. No pipeline change — the
data was computed once a second and thrown away.

**T5.6 — subsystem budget.** `--budget-ms <f64>` defaults to `FRAME_BUDGET_MS` (33.0, the
interval a 30 fps capture implies). When inference's **p95** exceeds it, the pipeline clears
`ascii` and `feed` in the `RuntimeConfig` it owns and logs once. The decision is on the tail
rather than the median: a stage whose median fits but whose p95 does not is the one that makes
the instrument stutter, and shedding the picture is the cheapest way to buy back the time.

Three things make it honest rather than surprising. The guard fires **once**, not every frame,
so it does not fight the user; a manual toggle beats it, because `shed` only suppresses the
automatic decision, not the human one; and `--budget-ms 0` turns the whole thing off. The
picture is decided *after* the timing is recorded, so a shed that fires on a frame's tail
takes effect on that frame's picture rather than the next one.

**Gate:** `make verify` green, **151 tests passing** (6 ignored), clippy clean. `make
trace-check` reproduces the fixture trace exactly across three consecutive runs.

### A caveat on reproducibility

The golden trace matched three times in a row on an idle machine and differed by 36 frames
when a `cargo build --release` was competing for cores in the same window. The trace records
what the *model* saw, and the detector's ROI anchor is seeded from the previous frame, so load
changes the timings, the timings change the anchor, and the anchor changes the detections. The
fixture clip is reproducible; the *tracker's* response to it is only reproducible when the
machine is otherwise idle. `make trace-check` is a regression gate for a quiet machine, not a
CI-grade assertion — which is consistent with there being no CI.

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
   ├─► Phase 1 (hot-path fixes: 1.1→1.9, independent of each other)   ☑
   │        │
   │        ▼
   ├─► Phase 2 (2.1 → 2.2 → 2.3 → 2.4 → 2.5)   ← strict order          ☑
   │        │
   │        ▼
   ├─► Phase 3 (3.1 → 3.2 → 3.3 → 3.4, 3.5 anytime after 2.4)          ☑
   │        │
   │        ▼
   ├─► Phase 4 (4.1, 4.2 need 2.3/2.4; 4.3 needs 4.2; 4.5 needs 4.3)   ☑
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
| 2 Frame loop & state | T2.1–T2.5 | 2–3 d | ☑ done |
| 3 Instrument layer | T3.1–T3.5 | 2 d | ☑ done (T3.3 kept the voice in `audio/` on purpose) |
| 4 Presentation | T4.1–T4.6 | 2 d | ☑ done |
| 5 Observability & test infra | T5.1–T5.6 | 3 d | ☑ done |
| 6 Unlocked features | — | ongoing | ☐ |

Branch `phase5-observability` carries T5.1, T5.5 and T5.6; `phase4-snapshot` is already merged
into `master`. Current gate: `make verify` green, **151 tests passing**, 6 ignored benches,
clippy clean.

### Where to pick up

1. **Shrink `pipeline.rs`** — 2050 lines, 1150 of them tests. `Telemetry` plus the trace is the
   obvious next extraction into `src/core/`; the frame loop itself is now four stages and a
   handful of helpers.
2. **T5.7 — a headless UI test.** `dashboard/` is still 1955 lines with no tests. The keyboard
   geometry (T4.5) proved the pattern: split the drawing from the widget and the arithmetic
   becomes testable. `gauges.rs` and `metrics.rs` are next.
3. **Phase 6** — multi-hand, OSC sink, `--calibrate`, `Landmark.z`. All are now single-file
   changes; see §9.

Current shape: `src/` is **10,953 lines of Rust across 35 files**. `pipeline.rs` is 2,050 of
them (1,150 excluding its test module) and owns capture, tracking, gesture dispatch, sinks,
trace and telemetry; the presenter is 450 lines and owns everything a human sees; the
estimator is 304. The iced layer (`dashboard/`) is 1,955 lines and is still the least-tested
part of the app — T4.5 closed the gap for the keyboard geometry, not for the rail.
