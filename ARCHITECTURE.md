# ARCHITECTURE.md — architecture review of `gesture_rust`

**Read-only analysis, no code changed.** Written as a reference for future sessions
(branch: `master`, commit `bbb7c40`). Focus: **simplification**, measured against
how real-time game engines and real-time audio applications are normally structured.

---

## 0. Executive summary

The project is a **perception → gesture → sound** pipeline that has grown organically
into a single monolithic loop. The data flow is conceptually right (that is the
game-engine pattern), but the *ownership* of state is wrong in three places, and the
hot path is polluted with work that a real-time app keeps off the critical path.

**Top 10 simplifications, in order of payoff:**

| # | Issue | Standard practice | Effort |
| :-- | :-- | :-- | :-- |
| 1 | Two frame paths (`run` + `tick`) and `Option<WebcamSource>`/`Option<HandTracker>` special-casing mock mode | One tick, an `InputSource` enum; mock is just another source | M |
| 2 | "What is playing now" lives in 4 places (`Pipeline.gesture_active`, `FrameMetrics.note/volume/frequency`, `NoteGate.held`, `SynthesizerState.voices`) | One voice/note state owned by an instrument layer; sinks become transports | M |
| 3 | Control state is 4 loose `AtomicBool`s + an `AtomicU8` + a `version: u64` counter | A single `RuntimeConfig` snapshot or a UI→pipeline command channel | S |
| 4 | `std::fs::write` of the ASCII frame **on the inference thread, every ASCII tick** | Never do file I/O on a frame-critical thread; queue to an IO thread or a writer with backpressure | S |
| 5 | cpal callback locks a `Mutex<SynthesizerState>` per audio block; `unsafe impl Send + Sync` on the stream | Lock-free, allocation-free, syscall-free audio callback; double-buffered control block | M |
| 6 | Oscillator sample rate is hard-coded 44100 while the stream config is the device's (`device.default_output_config()`), and frequency is snapped mid-note | Use the stream's actual rate; ramp frequency (portamento), band-limit the waveforms, soft-clip instead of hard clamp | M |
| 7 | Sink thread polls `recv_timeout(20 ms)` | Event-driven receive with a shutdown sentinel — measured dispatch→sink is sub-millisecond, so the real cost is 50 idle wakeups per second, not attack latency | S |
| 8 | `FrameMetrics` is a 39-field god-struct cloned into the shared snapshot every frame and cloned again by the UI every tick | Small snapshots per subsystem, published at the consumer's cadence, double-buffered | M |
| 9 | Spread/volume is computed twice with two different normalisations (`spread_volume` vs `hand_geometry().spread`), and smoothing happens in 3 places (ROI EMA, tip EMA, reuse cache) | One estimator layer producing one canonical hand state | M |
| 10 | 36 CLI flags, ~1.5k lines of UI with zero tests, no CI | Layered config + presets; a pure-function presentation layer that can be tested headlessly | M |

**What is already good — do not rewrite it:** the capture→inference→sink thread split,
the capacity-1 frame channel with drop-and-count, the `ReuseCache`, the palm-anchored
ROI window, the `NoteGate` (a genuinely correct MIDI voice model), the affine sampler
with reused buffers, `anyhow` context everywhere, `rustfmt`/`clippy` cleanliness, and
112 passing tests. The problem is not the components; it is that they are welded
together in one 1707-line file with 32 fields.

---

## 1. Facts measured this sweep

| Item | Value |
| :-- | :-- |
| Source lines | 8,719 across 26 `.rs` files |
| Largest units | `pipeline.rs` 1707, `vision/tracker.rs` 790, `dashboard/mod.rs` 767, `dashboard/skeleton.rs` 741 |
| Tests | 112 pass, 4 ignored benches, 0.75 s run |
| Untested lines | `dashboard/mod.rs` 767, `dashboard/skeleton.rs` 741, `dashboard/gauges.rs` 139, `main.rs` 449, `visualizer/mod.rs` 100, `core/event_bus.rs` 20 ≈ 2.2 k |
| Complexity hotspots (AFT) | `Pipeline::advance` 29, `Skeleton::draw` 20, `Pipeline::run` 17, `Dashboard::update` 16, `keyboard()` 16, `sample_affine_to_tensor` 16, `run_detector` 12 |
| `Pipeline` struct | 32 fields; `PipelineConfig` 21 fields |
| `FrameMetrics` | 39 fields |
| `Dashboard` | 20 fields |
| CLI | 36 flags |
| Diagnostics | 0 errors, 0 warnings; 1 dead-code hit (`Visualizer::draw_skeleton`, a call-graph false positive — it *is* used by `--image`) |
| Runtime budget | ~33 ms/frame; capture ~31 ms, inference ~8 ms (MediaPipe) / ~69 ms (YOLO pose) |

---

## 2. As-is architecture

```
main.rs (449 lines, 36 flags)
  ├─ builds tracker / camera / sinks / strategy
  ├─ wires 5 Arc<Atomic*> + Arc<Mutex<InstrumentConfig>> + Arc<Mutex<FrameMetrics>>
  ├─ thread A: pipeline::run()      (capture → inference → gesture → publish)
  ├─ thread B: capture_frames()     (webcam → bounded(1) channel)
  ├─ thread C: sink drain loop      (recv_timeout(20 ms) → sinks)
  └─ main thread: iced Dashboard    (33 ms tick → clone FrameMetrics → draw)
```

**Thread table (as-is):**

| Thread | Owns | Blocking work on the hot path? |
| :-- | :-- | :-- |
| Capture | `WebcamSource` | camera read only — good |
| Pipeline | tracker, strategy, `FrameMetrics`, ASCII layer, event bus, **file writes** (`--ascii-dump`) | yes: RGBA conversion, ASCII resize, `std::fs::write` |
| Sinks | `Vec<Box<dyn Sink>>`, event receiver | MIDI write (fine), cpal state lock (fine) |
| UI | `FrameMetrics` clone, image handle, trail | GPU texture upload (documented, throttled) |
| Audio callback (cpal-owned) | `Voice` array | **mutex lock per block** |

---

## 3. Target architecture (the blueprint)

The canonical shape for a real-time interactive app (game engine or instrument) is
five strictly separated layers, each with a single owner and a bounded budget:

```
┌────────────────────────────────────────────────────────────────────────────┐
│  1. INPUT          2. PERCEPTION        3. SIMULATION        4. OUTPUT     │
│  InputSource  →   HandState      →    InstrumentState  →   Audio / MIDI    │
│  (camera,        (tracker,            (voices, grid,       (cpal, midir)  │
│   video, mock)    estimator)          note gate, envelope)                 │
│                                    │                                        │
│                                    └──→ 5. PRESENTATION (snapshot consumer) │
└────────────────────────────────────────────────────────────────────────────┘
           channel            channel            channel
      (bounded 1, newest) (bounded 1, newest) (bounded, typed)
```

Rules that make this shape work in practice:

1. **One owner per state.** Nothing reads another layer's internals; it reads a
   published snapshot or receives a typed message. No `Arc<Mutex<T>>` shared across
   three layers.
2. **Rate tiers.** Audio rate (44.1 kHz) → control rate (~1 kHz, the audio callback's
   block boundary) → gesture rate (30 Hz) → UI rate (15–30 Hz). Every tier boundary
   is an explicit buffer, not an implicit "whatever thread got there first".
3. **Hot path is allocation-free, lock-free, syscall-free.** Anything that can block
   (files, MIDI, GPU uploads, JSON) goes to a dedicated thread or a queue.
4. **Fixed timestep for simulation.** Games run simulation at a fixed tick and render
   at whatever rate. Here: gesture/simulation at a fixed tick, presentation at its own
   cadence, audio at sample rate. Mock mode, `--max-frames` and tests all become
   trivial because the tick is deterministic.
5. **Determinism/replay.** A `--video file` source plus a tick counter makes every run
   reproducible — this is the single biggest missing testing capability (the ROADMAP
   lists "tracker fixture tests" as M; a recorded clip source solves it and the
   benchmark harness at the same time).
6. **Graceful degradation.** Optional subsystems (ASCII, feed, labels) are explicitly
   budgeted and switchable, ideally feature-gated at compile time, so a slow machine
   can drop the cheapest layer rather than miss frames.

**Concrete module layout that expresses this:**

```
src/
  app/            main.rs, cli.rs, config.rs, runtime.rs (RuntimeConfig, command channel)
  perception/     camera.rs, sampling.rs, tracker.rs, yolo_pose.rs, reuse.rs, estimator.rs
  gesture/        grid.rs, scale.rs, geometry.rs, strategies/, note_state.rs
  instrument/     voice.rs, envelope.rs, waveform.rs, midi_gate.rs
  output/         audio_sink.rs, midi_sink.rs, (osc_sink.rs)
  presentation/   dashboard/, ascii.rs, visualizer/
  telemetry/      snapshots.rs, metrics.rs, log.rs, tracing_init.rs
  core/           traits + shared domain types (only what ≥2 layers need)
```

Note what moved: `vision/ascii.rs` → `presentation/` (it is a rendering concern, not
perception); `core/instrument.rs` + `midi/mod.rs`'s gate + `audio/synthesizer.rs`'s
voice → `instrument/` (one voice model, two transports); the atomics in `main.rs` →
`app/runtime.rs`.

---

## 4. Layer-by-layer audit

### 4.1 Entry point & configuration — `src/main.rs`

**Current.** 449 lines: 36 flags, tracker construction, sink construction, strategy
selection, five `Arc<Atomic*>`s, one `Arc<Mutex<InstrumentConfig>>`, one
`Arc<Mutex<FrameMetrics>>`, then thread spawn and UI hand-off.

**Industry standard.** Config is *layered*: defaults → config file → env → CLI, and
user-facing knobs are grouped into presets. Engines never expose 36 flat flags; they
expose a profile plus a handful of overrides. Settings are a single immutable
`Config` value resolved once at startup, not scattered atomics that mutation is
propagated through by hand.

**Simplify.**
- Extract `cli.rs` (Args + enums) and `config.rs` (validated `Config` struct). `main`
  becomes ~60 lines: resolve config → build app → run.
- Replace `feed`/`ascii`/`active`/`ramp` atomics with **one** `Arc<Mutex<RuntimeConfig>>`
  (`active: bool, feed: bool, ascii: bool, ramp: Ramp, labels: bool, box: bool`) that
  the pipeline snapshots once per tick. Four atomics + four `Instant`/`Duration`
  fields in `Pipeline` collapse into one loop over one config.
- The `InstrumentConfig.version: u64` counter (instrument.rs:90) plus
  `Pipeline.applied_version` is a manual dirty-flag. A command channel
  (`crossbeam::channel<UiCommand>`) makes "the UI changed something" a *message*
  instead of a polled counter, and removes the version field entirely.
- Asset handling: model paths default to strings and fail at startup. Standard
  practice is an asset manifest checked once before the loop starts (existence, size,
  checksum) with a clear error telling the user to run the fetch script. Cheap, and
  it turns a stack trace into a message.

### 4.2 Orchestration — `src/pipeline.rs` (1707 lines)

**Current.** `run()` (305-432) spawns capture and sink threads and loops; `tick()`
(469-515) is a *second* full frame path used by mock mode and tests; `advance()`
(524-735) does instrument sync, metrics, feed conversion, ASCII rendering, file
writes, geometry, gesture dispatch, event draining, publishing and debug printing —
cyclomatic complexity 29. The struct carries 32 fields, 6 of which exist only to
throttle the feed/ASCII cadence.

**Industry standard.** A game loop is one function: `while running { dt = tick();
input(); simulate(dt); present(); }`. Every stage is a named unit with its own budget
and its own metrics. Frame pacing uses a fixed simulation step with accumulator
catch-up, and dropped work is *explicit* ("skipped render this frame"), not implicit.

**Simplify.**
- **Unify `run` and `tick`.** The mock-mode branch (`let mock = self.tracker.is_none()`
  at :360, `if self.camera.is_none()` at :470) exists only because there is no source
  abstraction — the one abstraction the project deliberately deleted. Put it back in a
  minimal form:
  ```rust
  enum Source { Camera(WebcamSource), Synthetic { cadence: Duration } }
  ```
  `run` then has exactly one path: receive a frame → `advance`. Mock, video-file and
  camera become three constructors, and the "tracker is Option" dance disappears
  (mock still needs no tracker, but that belongs to `Source`, not to the pipeline).
- **Split `advance` into four named stages**: `update_instrument()`,
  `update_tracking(detection)`, `update_visuals(frame)`, `dispatch_gesture(detection)`.
  Each is independently testable; each owns its own throttle state. `advance` becomes
  10 lines. This is the single highest-value refactor in the file: it turns a 210-line
  function with five unrelated concerns into four functions of ~40 lines, and it makes
  "what does the pipeline do per frame" answerable without scrolling.
- **Move presentation work out of the pipeline.** `frame_to_rgba`, `AsciiLayer::render`,
  `print_ascii` and `std::fs::write` are presentation. They belong behind a
  `Presenter` trait owned by a dedicated thread that receives frames at its own
  cadence. As written, an ASCII dump write (pipeline.rs:644) can stall inference —
  a disk hiccup becomes an audio dropout.
- **Kill the polling constants.** `SINK_POLL_MS = 20` (pipeline.rs:41) is *not* an
  attack-latency problem — measured dispatch→sink is under a millisecond, because
  `recv_timeout` returns as soon as an event arrives. It is a wakeup cost: the sink
  thread wakes 50 times a second to find nothing. Use `rx.recv()` plus a
  `UiCommand::Shutdown` sentinel and the idle wakeups disappear.
- **Group state.** `Pipeline` → `{ source, tracker, strategy, instrument, presenter,
  metrics, control }`. Fewer fields, each field meaningful.

### 4.3 Perception — `src/vision/`

**Current.** `HandTracker` enum (mediapipe | yolo) both producing
`HandDetectionResult { landmarks, box, frame_size, score, timings }`. Good pieces:
palm-anchored ROI, EMA anchor, stale-miss drop, `ReuseCache` capped at 3 frames,
reused `StageInput` buffers, layout-agnostic pose decoder, zero-copy `TensorRef`.
Issues: ORT sessions are built with defaults (tracker.rs:158-163); the confidence gate
is a raw per-frame argmax compare with no hysteresis (:265); three ~20-line duplicate
blocks in `tracker.rs` (bench scaffolding, :606-627, :705-731, :748-774);
`HandDetectionResult` mixes model output with per-stage timings; `r#box` is an awkward
name; nothing here is tested on real model output.

**Industry standard.** Vision pipelines in real-time apps separate (a) the *model
adapter* (raw graph → detections), (b) the *state estimator* (track, smooth, gate),
and (c) the *consumer-facing state*. Threshold gates always have hysteresis/deadband
and a grace period. Execution backends are explicitly configured (thread count,
provider), never defaulted.

**Simplify.**
- Add an `estimator` layer (`perception/estimator.rs`) that owns smoothing, hysteresis
  and the canonical `HandState` the strategies consume. Today smoothing is scattered
  across ROI EMA (`smooth_palm`), tip EMA (`TipSmoother`), and the reuse cache — three
  ad-hoc filters with three constants. One estimator is easier to tune, test and reason
  about.
- Detection hysteresis (ROADMAP finding 8) is not a feature, it is a correctness
  requirement for a gate: with flickering detection, the release event fires and the
  note re-attacks. Debounce at the perception boundary, not in the sinks.
  *(done in Phase 1: `core/detection_gate.rs`, applied in `Pipeline::advance`)*
- ORT thread pinning (`--ort-threads`) is available, but measurement killed the
  assumption: the BlazeHand detector is fastest with the runtime default of one
  thread per core (9.7 ms p50) and 40 % slower pinned to one (13.7 ms). Keep the knob
  for A/B testing; do not default to a small number.
- Move `timings` out of the domain result into a `StageReport` the telemetry layer
  collects. Strategies and presenters should not see µs fields.
- `r#box` → `bbox`; `frame_size` → keep, but consider making the whole result
  normalised-with-a-scale so consumers cannot mix pixel and normalised space (the
  project's AGENTS.md already calls coordinate discipline a rule — a type makes the
  rule unbreakable).

### 4.4 Gesture mapping — `src/strategies/`, `src/core/{grid,scale,geometry}`

**Current.** `Strategy::process(&self, …) -> Option<GestureEvent>` (core/mod.rs:66).
`FingerPositionStrategy` (EMA tip → grid cell) and `ThereminStrategy` (log x → Hz)
share `TipSmoother`, `band_position`, `spread_volume`. `hand_geometry()` is computed
every frame in `advance` purely for the dashboard gauges.

**Industry standard.** In games, the input layer produces *semantic intent*
("player is over key C4, with force 0.7"), and gameplay owns state transitions. In
instruments, note selection has hysteresis, velocity curves and explicit retrigger
semantics.

**Simplify.**
- **`Strategy::process` should take `&mut self`.** The trait is declared `Send + Sync`
  and `process(&self)`, which forces `TipSmoother` to carry a `Mutex` (strategies/mod.rs:23)
  even though the pipeline owns the strategy on exactly one thread. Interior
  mutability + a lock in a 30 Hz loop is pure ceremony: change the signature, delete the
  Mutex. Same for `Sink: Sync` (see 4.6).
- **One geometry, one normalisation.** `spread_volume` normalises thumb-index distance
  by the *hand box height*; `hand_geometry().spread` normalises the same pair by
  *palm length*. Two numbers, two units, one physical quantity. Pick one canonical
  `HandState` and let both the strategy and the gauges read it.
- **`GestureEvent` carries presentation data.** `text: String` and `color: (u8,u8,u8)`
  (types.rs:18-23) are UI fields riding through every sink. Sinks do not need colours.
  Split: `PerformanceEvent { payload }` for outputs, and let the UI derive its own
  presentation from the instrument state. This removes string allocations from the
  event path and makes the event type honest.

### 4.5 Instrument / voice state — the missing layer

**Current.** There is no object that answers "what is playing right now, and what is
its envelope?". The pipeline knows (`gesture_active`, `metrics.note/volume/frequency`),
MIDI knows (`NoteGate.held/cc/attacked`), the synthesizer knows (`voices: Vec<Voice>`).
Each sink implements its own attack/hold/release semantics independently, and the
pipeline has to send an explicit release event (`GestureEvent::no_gesture()` — a
"dummy payload", types.rs:49) to keep sinks from latching.

**Industry standard.** Instruments have a **voice pool** and an **envelope** in one
place; output backends (MIDI, OSC, internal DSP) are thin adapters over that state.
MIDI voice allocation, note-on/off pairing, CC vs velocity semantics, ADSR, portamento
and legato rules belong to the instrument layer, not to each transport.

**Simplify.**
- Introduce `InstrumentState { held: Option<Note>, target: f32, envelope: Adsr, timbre:
  Waveform }` owned by the pipeline/simulation layer, with `attack()`, `release()`,
  `set_target()`. The audio sink then only renders the state; the MIDI sink only
  translates it. The `no_gesture()` sentinel becomes a real `Release` message and the
  `note == 0` magic value disappears — magic sentinels in a payload enum are a
  classic source of bugs (already documented in the code comments).
- `GesturePayload::Signal` vs `Note` is a *sink capability* distinction, not a gesture
  distinction: the theremin's frequency *is* a note (it is quantised to nothing, but
  MIDI could still send the nearest semitone with a pitch-bend). If both strategies
  emitted the same "target pitch + amplitude" state, the MIDI sink's "Signal is
  dropped" special case disappears.

### 4.6 Output sinks — `src/audio/synthesizer.rs`, `src/midi/mod.rs`

**Current.** `Sink: Send + Sync { start, stop, handle_event, set_waveform }`.
Audio: cpal stream wrapped in `StreamWrapper` with `unsafe impl Send` **and** `unsafe
impl Sync` (synthesizer.rs:49-52); the callback locks `Arc<Mutex<SynthesizerState)>`
per block (:135); phase is a `0..1` scalar and frequency is written directly onto the
voice (:29-34); mixed sample is hard-clamped (:141); sample rate is the
`44100.0` passed to `Synthesizer::new` (main.rs:306) while the stream is built from
`device.default_output_config()` (:126). MIDI: a well-designed `NoteGate` (:29-108)
with tests.

**Industry standard (real-time audio).**
- The audio callback never locks, allocates or syscalls. Control data is delivered via
  atomics or a double-buffered block swapped outside the callback.
- Oscillators are phase-continuous and band-limited (BLIT/polyBLEP/synthesised
  square); frequency changes are ramped, not snapped.
- Output has headroom and a soft limiter, not `clamp(-1,1)`.
- The device's actual sample rate is used, and the stream buffer size is chosen
  deliberately because it *is* the latency.
- A stream that is `!Send` is never smuggled across threads; it is created on and
  confined to the audio thread.

**Simplify (concrete, ordered).**
1. **Use the stream's sample rate.** `Synthesizer::new(44100.0, …)` +
   `device.default_output_config()` is a latent detuning bug on 48 kHz devices. Drop
   the `sample_rate` field entirely and read `config.sample_rate` in `start`.
2. **Drop `unsafe impl Sync`.** `Arc<Mutex<Vec<Box<dyn Sink>>>>` only needs `Sink:
   Send`; the `Sync` bound on the trait is unused, and with it the second `unsafe impl`
   disappears. One fewer line of unsound-by-assertion code.
3. **Make the audio hot path lock-free.** Two `AtomicU32`s (`freq_bits`, `amp_bits`)
   plus an `AtomicBool` `sounding` is enough for a monophonic theremin; the `voices`
   Vec and the Mutex are only needed once there is real polyphony. Even then, the
   standard pattern is a small fixed-size voice array + atomic control words.
4. **Envelope + portamento in one place.** A raised-cosine ramp on frequency and
   amplitude changes, ADSR on attack/release. This is the difference between "a tone"
   and "an instrument", and it belongs in `instrument/voice.rs`, not in the cpal
   callback.
5. **Reuse the RGBA/voice buffers.** `frame_to_rgba` allocates a 1.2 MB `Vec` every
   feed tick (:858). Keep a scratch buffer in the presenter and clear it. Trivial, but
   it is exactly the kind of allocation a real-time app avoids.

### 4.7 Presentation — `src/dashboard/`, `src/visualizer/`, `src/vision/ascii.rs`

**Current.** ~1.65k lines with **zero tests**. `Dashboard` holds 20 fields, of which
8 are mirrors of pipeline state (`active`, `feed`, `ascii`, `ramp`, `ramp_index`,
`instrument`, `instrument_view`, `metrics`) and 3 are UI-only (trail, histories,
tab). `Skeleton::draw` is 164 lines and 20 branches. `keyboard()` redraws all 49
cells every tick. The ASCII renderer lives in `vision/`.

**Industry standard (games).** Presentation is a **pure function of a snapshot**, so
it can be rendered headlessly, snapshotted and diff-tested. State that only affects
drawing never crosses a thread boundary. Immediate-mode UIs are the norm precisely
because "draw(world_state)" has no hidden state.

**Simplify.**
- Make `Skeleton::draw` / `keyboard()` take an immutable `StageView` struct and return
  a list of primitives. Then the UI can be tested with a golden primitive list — the
  ROADMAP's "UI unit tests" item becomes cheap.
- Move `note_trail` into the published snapshot (ROADMAP finding 10): the trail is
  instrument state, not UI state, and it already has a decay function (`fade_trail`)
  that is tested. One source of truth, one fewer 49-element array cloned per tick.
- Split the ASCII layer into `presentation/ascii.rs` and run it on the presenter
  thread, not the inference thread. The 5 ASCII flags (`--ascii`, `--ascii-cols`,
  `--ascii-fps`, `--ascii-ramp`, `--ascii-dump`) and the `AsciiLayer` +
  `AsciiGrid` in `FrameMetrics` are a whole feature currently welded into the frame
  loop; behind a `Presenter` trait they are a plug-in.
- `visualizer/mod.rs` (100 lines) exists only for `--image`. It is a presentation
  concern living at the top level; fold it into `presentation/` or delete it once the
  dashboard can dump a frame.
- Consider whether iced is the right tool for a HUD: it is 767 lines of card code plus
  a synchronous GPU upload per frame. An immediate-mode canvas (egui) is lighter and
  composes better with a snapshot-based renderer. **Trade-off:** iced is already
  working, and swapping is a rewrite, not a simplification — only worth it if the UI
  keeps growing.

### 4.8 Telemetry & observability

**Current.** `println!`/`eprintln!` gated on `--debug`; one EMA fps; last-frame values
for every stage; a 12-entry event log.

**Industry standard.** Structured logging with per-subsystem spans, and *percentiles*
(p50/p95/p99) rather than instantaneous values, because real-time regressions are
tail events. A dropped frame or a 100 ms inference spike is invisible in an average.

**Simplify.** Adopt `tracing` (the ROADMAP already lists it). Add a rolling ring of
stage times per stage, publish `p50/p95` to the Performance card. Keep the debug print
but generate it from the same ring so the terminal and the UI agree.

### 4.9 Build, tests, tooling

**Current.** 112 tests, all unit-level, 4 benches `#[ignore]`d, no CI, clippy has one
live lint, no lint/CI gate, no fixture images for the ONNX stages, no integration test
that runs a recorded clip end-to-end.

**Industry standard.** Games ship a headless harness: fixed input, fixed tick count,
asserted outputs, golden snapshots. Audio apps have deterministic test vectors. CI is
table stakes.

**Simplify.**
- CI: `cargo fmt --check` + `cargo clippy -D warnings` + `cargo test`. One file.
- A `--video clip.mp4 --ticks 300 --expect note_trace.json` mode: deterministic input
  makes the whole pipeline testable, including tracker behaviour and the release path.
- Move the bench duplication out of `tracker.rs` (three ~20-line copies in
  `bench_sampling`) into a separate `benches/` crate; it is 60 lines of duplicated
  scaffolding in the production file.

---

## 5. Cross-cutting themes

**1. State ownership is the core problem.** Four subsystems each track "the current
note", three atomics track UI toggles, the instrument config needs a version counter
to be usable. Standard fix: single-owner state + typed messages + read-only snapshots.
Every `Arc<Mutex<T>>` in the app is a smell worth auditing:
`FrameMetrics` (pipeline writes, UI reads — acceptable as a snapshot),
`InstrumentConfig` (UI writes, pipeline reads — replace with a command channel),
`Vec<Box<dyn Sink>>` (one thread should own the sinks — it does; the pipeline's
`set_waveform` push is the leak), `SynthesizerState` (audio callback + control thread
— acceptable *if* lock-free), `TipSmoother` (unnecessary — see 4.4).

**2. Latency budget is now measured, not guessed.** For a gesture instrument,
hand→sound should be < 50 ms, and users perceive > 80 ms as "laggy". The live
baseline (`bench/baseline.md`) gives camera 33.2 ms (p50) + inference 8.2 ms = **41 ms**
to the event bus, dispatch→sink < 1 ms. The missing piece is the cpal block size,
which cpal 0.17 does not expose and which has to be read from the callback's block
length. With the YOLO backend the same figure is 33.7 + 64.7 (p95) with frames going
stale by up to 52 ms.

**3. Rate tiers are implicit.** Inference runs at camera rate; UI at 30 Hz; ASCII at
15 Hz; feed at 15 Hz; audio at 44.1 kHz. Making the tiers explicit (each boundary is a
queue with a stated cadence) is what allows the ASCII and feed gating to be a clean
feature rather than six fields in `Pipeline`.

**4. Three smoothing layers with three constants.** ROI EMA (0.3), tip EMA (0.35),
reuse cache (3 px / 3 % / 3 frames). Fine individually, collectively undocumented and
untunable. A single estimator makes the instrument's responsiveness a single knob.

**5. Determinism.** Mock mode sleeps (`thread::sleep(33 ms)` → actually 40.8 ms on
macOS per the ROADMAP), which makes every mock telemetry number wrong and makes UI
tests impossible. A tick counter + injectable clock fixes both, and it is the standard
game-engine pattern (`fixed timestep`, `sim_clock`).

**6. No graceful degradation policy.** Frames are dropped when inference is behind —
good. But there is no notion of "if we are over budget, skip the optional layer".
Engines implement quality tiers; the toggle machinery here is 60 % of the way there.

---

## 6. Prioritised simplification backlog

| Priority | Action | Why it is a *simplification*, not just a refactor | Effort |
| :-- | :-- | :-- | :-- |
| 1 | One frame path: `InputSource` enum; delete the mock branch, `Option<HandTracker>`, `Option<WebcamSource>`, and `Pipeline::tick` | Removes an entire duplicated code path and makes mock/video/test sources symmetric | M |
| 2 | Split `advance()` into 4 named stages | A 29-branch function becomes 4 readable ones; each becomes unit-testable | S |
| 3 | `RuntimeConfig` snapshot + `UiCommand` channel | Deletes 4 atomics, the `version`/`applied_version` pair, and the polling loops | S |
| 4 | Move ASCII + frame→RGBA + file dump to a presenter thread | Removes presentation I/O from the real-time loop; the pipeline loses 6 fields | M |
| 5 | Lock-free audio control (atomics), use the device sample rate, drop `unsafe impl Sync` | Fewer unsafe impls, no lock in the callback, one latent bug gone | S |
| 6 | Voice/envelope layer shared by both sinks; real `Release` message instead of `note: 0` | Two sinks stop reimplementing attack/release; the sentinel payload disappears | M |
| 7 | `Strategy::process(&mut self)`; delete `TipSmoother`'s Mutex; drop `Sink: Sync` | Removes interior mutability that has no reason to exist | S |
| 8 | Canonical `HandState` from one estimator (hysteresis + smoothing) | Replaces 3 filters and 2 spread normalisations with one tested unit | M |
| 9 | Presentation as a pure function of a snapshot; move `note_trail` into the snapshot | Makes 1.65k untested UI lines testable | M |
| 10 | Percentile telemetry + `tracing` | Replaces ad-hoc `println!` with a consistent observability layer | S |
| 11 | ~~ORT thread pinning~~ (done, a knob not a win); capture at 320×240 experiment | Cost reduction with no structural change | S |
| 12 | CI + fixture/clip harness | Turns "does it work?" into an automated check | M |

**Explicitly not worth doing now:** rewriting iced, multi-hand support, OSC sink,
GPU execution providers. These are features, and the architecture is the bottleneck
before any of them are added — multi-hand in particular would currently require
touching `Pipeline`, `FrameMetrics`, `Skeleton::draw` and `Dashboard` simultaneously,
which is the clearest symptom of the missing layer boundaries.

---

## 7. Quick reference: where to look

| Concern | File | Lines |
| :-- | :-- | :-- |
| Frame loop / duplication | `src/pipeline.rs` | 305-432 (`run`), 469-515 (`tick`), 524-735 (`advance`) |
| Shared state | `src/main.rs` 384-416; `src/dashboard/metrics.rs` 64-133 | |
| Audio hot path | `src/audio/synthesizer.rs` | 28-36, 49-52, 120-155 |
| MIDI voice semantics | `src/midi/mod.rs` | 29-108 |
| Gesture mapping | `src/strategies/mod.rs` | 22-73, 119-141 |
| Perception gating | `src/vision/tracker.rs` | 191-299 |
| UI ↔ pipeline coupling | `src/dashboard/mod.rs` | 76-219 |
| Presentation internals | `src/dashboard/skeleton.rs` | 292-642 |

---

## 8. Open questions for the next session

1. Is mock mode a source or a test fixture? If it is only a fixture, the `Source`
   enum shrinks to `Camera` + `Video`, and mock moves into the test harness.
2. Do we want polyphony? If yes, the voice pool must be designed before the MIDI sink
   grows a voice pool of its own.
3. What is the accepted latency target? Without a number (e.g. ≤ 50 ms), "optimise"
   has no stop condition.
4. Should the ASCII renderer be a feature flag (`--features ascii`) so it is compiled
   out of production runs, or a runtime toggle only?
