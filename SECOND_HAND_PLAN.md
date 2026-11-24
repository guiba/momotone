# Two-Hand Design Analysis — momotone

*A plan for adding a second hand, and for what that second hand could do to the sound.*
Options, trade-offs, measured costs, and a recommended sequence.

**Status:** all three PRs in §11 are built. PR 1 (`feat/effect-chain`) shipped the effect
chain; PR 2 (`feat/detector-window-probe`) measured the two detector windows (§16); PR 3
(`feat/two-hand-detection`) implements two-hand detection (§17). Where the build differed
from this plan, §15 and §17 say so and why.

Grounded in the current tree: `src/pipeline.rs` (1962 lines), `src/vision/tracker.rs` (697),
`src/vision/yolo_pose.rs` (486), `src/gesture/estimator.rs` (480), `src/audio/synthesizer.rs` (486),
`src/dashboard/skeleton.rs` (950). Numbers quoted are the measured ones in `ROADMAP.md`.

---

## TL;DR

**Three things are actually being asked, and they have very different costs:**

| Idea | Real cost | Verdict |
| :--- | :--- | :--- |
| **Effects/modulation driven by a second hand** | ~0 ms extra inference — the modulator needs no landmarks, and the detector already scores every anchor | Worth doing; the interesting one |
| **A second melodic voice (chords, drones, duet)** | same detection cost **plus** a rewrite of the monophonic synth and MIDI gate | Worth doing, but it is a second project |
| **Cheaper version of the same expressiveness with one hand** | ~0 ms | Do this first — it is 80 % of the musical value at 20 % of the effort |

The key insight: **a second hand does not need landmarks.** Most modulation gestures only need the *palm box* (position + apparent size), which the detector already produces. The expensive part — the 224×224 landmark graph — can stay on one hand and the modulator hand can run at 10 Hz or not at all.

---

## Decisions log

### Step 1 — Detection *(locked)*

| Question | Decision | Why |
| :--- | :--- | :--- |
| Detection route | **One full-frame MediaPipe pass + top-k anchors + spatial NMS** | The graph already scores all 2016 anchors; only the argmax is kept. Sees the whole frame for the same 9.5 ms. No new model, no second crop. |
| Second-hand fidelity | **Palm box only — no landmarker** | Position, apparent size, presence and approach rate come free from `PalmBox`. Marginal inference cost ≈ 0. |
| Hand identity | **Sticky slot IDs + a UI role swap** | No anatomical guessing, survives crossing hands, and the roles stay semantic (*pitch* / *modulator*) rather than left/right. |
| Capture resolution | **Stay at 640×480; measure before optimising** | Inference is 11.1 ms of a 33 ms budget and the second hand adds essentially nothing, so there is no headroom to buy yet. |
| Confidence gate | **One `--score-threshold` (0.6) for both hands** | Fewer knobs; the modulator's instability is handled by decay rather than a second threshold. |
| Single-hand behaviour | **The detected hand plays pitch** | Two-hand tracking is additive: a single-hand run behaves exactly like today. |
| Role changes | **Manual only** (`SwapHands`) | Automatic promotion could hijack the melody mid-sweep. Predictable beats clever. |
| Opt-in | **`--hands two` by default**; `--hands single` is the escape hatch | The second hand is free, so it should be on. `single` is for A/B testing and for the old instrument. |

#### The design this picks

**1. Detector input: full-frame letterbox replaces the palm-anchored crop.**

`detector_crop()` (`vision/tracker.rs:97-107`) samples a `min(W,H) = 480` px square, slid to contain the last palm. Replace it with `letterbox_map(fw, fh, DETECTOR_INPUT)` — already written, already used by the pose backend (`vision/sampling.rs:241-252`).

At 640×480 into 192×192: scale 0.3, drawn 192×144, 24 px of black padding top and bottom. The whole frame is now in the input.

**Consequences, stated honestly:**

- **Effective detector resolution per hand drops ~25 %** (a 100 px palm now occupies 30 model pixels instead of 40). The landmarker is *not* affected — it crops from the full-resolution frame at 224×224.
- **Reach improves.** The centred square only ever sampled x = 80..560; a full frame samples x = 0..640. The outer keyboard cells become physically visible to the model.
- **`detector_crop`, `smooth_palm`, `ROI_ANCHOR_WEIGHT`, `ROI_STALE_FRAMES` and `note_miss()` become dead code.** Delete them in the same commit — clippy runs with `-D warnings` and `aft_inspect` tracks dead code.
- The sampler cost is unchanged: still 192×192 output pixels (~146 µs).
- Anchors whose priors fall in the 24 px padding bands sample pure black and can still fire. The threshold and NMS reject them, but `--debug` should print `"{n} palms above threshold"` so this is measurable rather than assumed.

**2. Anchor selection: argmax → threshold + greedy NMS.**

```rust
const MAX_HANDS: usize = 2;
const HAND_SEPARATION: f32 = 1.0;   // palm centres must be > K * max(half_a, half_b) apart
// tunable later as --hand-separation

// one inference pass, exactly as today
for (i, &s) in scores.iter().enumerate() {
    let conf = sigmoid(s);
    if conf < threshold { continue; }              // cheap early exit, no full sort
    let base = i * 18;
    let cx = anchors[i].0 * DETECTOR_INPUT as f64 + boxes[base];
    let cy = anchors[i].1 * DETECTOR_INPUT as f64 + boxes[base + 1];
    let half = boxes[base + 2].max(boxes[base + 3]) * 2.0 * 0.5 * applied.m00;
    candidates.push((conf, applied.frame_x(cx, cy), applied.frame_y(cx, cy), half));
}
candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
// greedy: accept a candidate only if it is far enough from every accepted palm centre
```

Note the `×2` expansion already baked into `half` (`tracker.rs:293`): anchors belonging to the same hand predict palm centres inside one palm box, so the separation rule merges them naturally. Two palms that genuinely overlap will merge into one hand — acceptable for a modulator.

**3. Stable slots, semantic roles.**

The tracker owns slot identity (it already owns the frame-to-frame history); the pipeline owns roles.

```rust
struct PalmBox { cx: f32, cy: f32, half: f32, score: f32 }

struct HandDetection {
    palm: PalmBox,
    landmarks: Option<Vec<Landmark>>,   // Some only for the pitch hand
    box: (f32, f32, f32, f32),
}

// tracker state
slots: [Option<(f64, f64)>; 2],         // last palm centre per slot
// assignment: nearest slot within SLOT_MAX_JUMP (e.g. 0.35 * min(frame side));
// an unmatched palm takes an empty slot; no slot free -> it is dropped or
// replaces the weakest-scoring slot.
```

`UiCommand::SwapHands` flips which slot drives pitch. Roles stay *pitch*/*modulator* — never left/right, which would be wrong for crossed hands.

**4. Only the pitch hand gets landmarks — and there is always exactly one pitch hand.** `run_landmarks()` runs once, on the hand in the pitch role. `ReuseCache` stays a single instance, no duplication.

The invariant this settles: **when only one hand is detected, that hand is the pitch hand**, regardless of which slot it landed in. So the landmarker always runs on exactly one hand, and a single-hand session behaves exactly as it does today. The corollary matters: a hand in the modulator slot *cannot* produce pitch, because it has no fingertips — so the fallback rule is required, not cosmetic.

**5. One gate, one threshold — but the modulator fades, it does not click.** Two `DetectionGate` instances (2 confirm / 3 grace), both keyed on `--score-threshold`. One gate per hand, because a hand flickering across the threshold must not flicker in the sound. The pitch gate decides note on/off; the modulator produces an **EMA'd level that decays to neutral** across the grace window rather than a hard on/off, so a brief dropout slides the filter back instead of snapping it.

**6. Inter-hand metrics live in the estimator**, because they need both hands:

```rust
gap_px      = |palm_pitch - palm_mod|
gap_norm    = gap_px / min(frame_w, frame_h)   -> mapped through a band like proximity
overlapping = gap_px < half_pitch + half_mod   -> "hands touching"
```

Every effect that reads these needs a defined value for the one-hand and no-hand cases. That is a spec requirement, not an edge case.

#### What this costs

| | Detector | Landmarker(s) | Inference p50 | Headroom |
| :--- | ---: | ---: | ---: | ---: |
| Today | 9.5 ms | 1 × 4.7 | 11.1 ms | 22 ms |
| **Decided design** | 9.5 ms | 1 × 4.7 | **~11.2 ms** | **22 ms** |

The second hand is nearly free *because* it has no landmarks. The decode loop adds a threshold test and a sort over a handful of survivors — sub-millisecond.

#### Verification for this step

- `cargo test` — pure-function tests on the NMS: synthetic anchor scores with two clusters → exactly two palms, no duplicates within a hand; one cluster → one hand.
- `make fixture` → compare against the recorded 106/120 detection rate for the full-frame pass vs. the anchored crop. **This is the measurement that decides whether the resolution loss is acceptable.**
- `cargo run -- --debug` already prints candidate counts for the pose path; add the equivalent line for the palm detector (`"{n} palms above threshold"`).
- The `#[ignore]`d `tracker_finds_the_hand_in_the_clip` test is the natural place to assert two-hand behaviour on the real clip — record a clip with two hands and extend it.
- `make mock-trace-check` must keep passing: mock mode needs a second synthetic palm before this can be gated.

#### Still open after this step

1. **Modulator fade constant** — how long does a lost modulator hand take to slide its effects back to neutral? Candidates: reuse `DEFAULT_FADE_MS = 450`, or the 3-frame grace window.
2. **The NMS separation constant** — `HAND_SEPARATION = 1.0` is a guess. Two palms that overlap merge into one detection; what separation actually reads as "two hands" on a real clip?
3. **Fallback if the fixture run shows a detection-rate regression** — keep an anchored crop for the pitch hand plus a full-frame pass for the modulator (2 detector runs, ~19 ms), or raise the landmarker crop instead?
4. **Two-hand fixture** — `make record` captures one hand. A regression gate for two hands needs a two-hand clip and an extended `mock_trace.json`.
5. **What "neutral" means per effect** — filter wide open, vibrato zero, delay off. Decided per effect in Step 2.

### Step 2 — Effects *(locked)*

| Question | Decision |
| :--- | :--- |
| Fork | **Modulation only.** No voice pool, no MIDI polyphony — the instrument stays monophonic. |
| First effects | **Lowpass filter, vibrato, delay/echo.** Inter-hand distance is a *source*, not an effect. Detuned-pair thickness parked. |
| Architecture | **Modular chain**: every effect is a declared stage with a neutral value, an enable bit, and a named source. Adding one is a checklist, not a redesign. |
| Hand leaves frame | **Fade to neutral** — the same idea as the keyboard trail (`--fade-ms`), not a hard cut. |
| Transport | `BusMessage::Modulation(Modulation)` + `Sink::set_modulation(&mut self, _m: &Modulation) {}`, defaulting to a no-op exactly like `set_waveform` (`core/mod.rs:68-69`). |

#### The effect chain, concretely

The boundary is the audio callback: it takes no lock, allocates nothing, makes no syscall (`audio/synthesizer.rs:38-52` documents why atomics were chosen). So "modular" cannot mean `Vec<Box<dyn Effect>>` in the hot path. It means a **declared, fixed-size chain whose stages are individually enabled**.

```rust
// core/effects.rs
pub enum Effect { Lowpass, Vibrato, Delay }     // declaring a new effect starts here

// one bit per effect; the chain is fixed, the enabled set is not
pub struct EffectState {
    enabled: u8,
    cutoff_hz: f32,       // neutral: wide open
    vibrato_hz: f32,      // neutral: 0
    vibrato_depth: f32,   // neutral: 0
    delay_ms: u32,        // neutral: 0
    feedback: f32,        // neutral: 0
}
```

**Chain order is a documented constant**, so a new effect has an obvious place to sit:

```text
frequency -> vibrato (LFO on pitch) -> oscillator -> lowpass -> delay -> soft clip
```

Two traps worth naming:

- **Vibrato must be applied after the existing glide.** The glide is already a one-pole ramp on `frequency` (`synthesizer.rs:129-130`); if the LFO is summed before the glide, the glide smooths the vibrato away.
- **The delay buffer cannot be allocated per sample.** Size it once in `Synthesizer::start()` from the device sample rate and a `MAX_DELAY_MS` constant.

#### Checklist for adding an effect

1. Add a variant to `Effect` and a bit in the enable mask.
2. Add its parameters to `EffectState` and to `Target` as atomics (`f32` stored as `to_bits()`, same as `frequency`/`amplitude` today).
3. Write the stage as a free function `fn stage(state, sample) -> sample`. Allocation-free.
4. Define its **neutral value** and where in the chain it sits.
5. Map a source to it in `gesture/modulation.rs`.
6. Add a meter for it on the MODULATOR card.

Every effect is then a triple — **source → parameter → curve** — which is what makes the design extensible: a new effect needs no new detection, only a new mapping.

**Available sources** (all from a palm box, so none require landmarks): modulator proximity, modulator x, modulator y, modulator approach rate, inter-hand gap, hands-touching, pitch-hand approach, pitch-hand pinch.

#### Proposed first mapping

| Effect | Driven by | Neutral | Range |
| :--- | :--- | :--- | :--- |
| Lowpass cutoff | modulator proximity (near = bright) | wide open | 200 - 8000 Hz |
| Vibrato depth | inter-hand gap (hands wide = more) | 0 | 0 - 50 cents |
| Vibrato rate | modulator height (high = faster) | 0.5 Hz | 0.5 - 8 Hz |
| Delay time | modulator x position | 0 ms | 0 - 600 ms |
| Delay feedback | inter-hand gap | 0 | 0 - 0.6 |

Each source has exactly one job in this table, so moving the modulator hand does not move three unrelated parameters at once. Delay is gated on **both hands being present** - there is nothing to modulate from a missing hand.

Enable/disable per effect from the CLI and the dashboard (`--effects lowpass,vibrato,delay`, `--no-vibrato`), so the chain is opt-in rather than assumed.

#### Still open in Step 2

1. **Mapping as data or as code?** Hardcoded in Step 2; a `Mapping { source, param, band }` table configured from the CLI/UI is the real payoff, and should come once there are three effects to configure.
2. **What `approach` should do.** It is already computed and smoothed (`estimator.rs:183-202`) and used by nothing. Envelope/attack is its natural home, parked behind filter/vibrato/delay.
3. **The exact bands.** 200-8000 Hz cutoff, 0-50 cents vibrato, 0-600 ms delay are starting values, not decisions. They need a listening session.

### Step 3 - Sequencing *(locked)*

| Question | Decision |
| :--- | :--- |
| Build order | **Effect chain before detection.** The chain is testable without a camera and delivers expressiveness with one hand; the second hand then only adds *sources*. |
| Resolution trade-off | **Measure before committing.** Record a two-hand clip and run `make fixture`; only then decide on the full-frame pass. |
| Fade | **Share `--fade-ms` (450)** with the keyboard trail. One constant, one feel. |
| MIDI | **Audio-only at first.** `Sink::set_modulation` is a defaulted no-op, so `MidiOutputEngine` is untouched until the audio path is proven and the CC mapping is tested against the OP-1. |

#### PR 1 - the effect chain (no second hand involved)

Files: `core/effects.rs` (new), `audio/synthesizer.rs`, `core/event_bus.rs`, `core/mod.rs`, `pipeline.rs`, `main.rs`.

1. `Effect` enum, enable bitmask, neutral values.
2. `EffectState` + one atomic per parameter in `Target`, same `to_bits()` trick as `frequency`/`amplitude`.
3. Stages as free functions, applied in the documented order. Lowpass, vibrato, delay.
4. Delay ring buffer sized once in `Synthesizer::start()` from the device sample rate.
5. `BusMessage::Modulation` + `Sink::set_modulation` default no-op. MIDI deliberately does nothing yet.
6. Drive it from the pitch hand's existing signals (`approach`, `spread`, `proximity`) so the chain is audible before any tracker change.
7. Tests: each stage is a pure function of `(state, sample)`; assert attenuation, sidebands, delay frames, and pass-through when disabled. Assert decay to neutral when the source hand disappears.
8. `--effects lowpass,vibrato,delay` and a MODULATOR-style card with a toggle per effect.

Gate: `make verify` plus `make fixture`, with the numbers recorded in `bench/baseline.md` - the audio callback gains branches, and a perf claim needs a measurement.

#### PR 2 - measure the detection change

No behaviour change yet: run the palm detector over the letterboxed full frame *and* the anchored crop, print both detection counts under `--debug`, and compare against the recorded 106/120 on the fixture clip. This is the experiment that either approves the full-frame pass or forces the fallback (anchored crop for pitch + full frame for the modulator).

#### PR 3 - two hands

Only once PR 2 says the resolution loss is acceptable: argmax -> top-k + NMS, sticky slots, per-hand gates, two palm boxes, mock mode gets a second palm, the trace gains a modulator column.

## 1. Where the one-hand assumption lives today

Every stage of the pipeline is written for exactly one hand. This is the inventory of what has to change.

| Layer | Single-hand coupling | Where |
| :--- | :--- | :--- |
| Detector | Argmax over 2016 anchor scores → **one** `PalmBox` | `vision/tracker.rs:261-279` |
| Detector ROI | A single 192×192 square window slid to contain *the* last palm | `detector_crop()` `vision/tracker.rs:97-107` |
| Result type | `HandDetectionResult { landmarks: Option<Vec<Landmark>>, box: Option<...>, score }` — one hand per frame | `core/mod.rs:43-55` |
| Reuse cache | One `ReuseCache` keyed on one palm anchor | `vision/reuse.rs:42-94` |
| Gate | One `DetectionGate` (2 confirm / 3 grace frames) | `gesture/estimator.rs:31-32` |
| Estimator | One `HandEstimator` → one `HandState` (tip, volume, proximity, approach, geometry, jitter) | `gesture/estimator.rs:34-168` |
| Strategy | `Strategy::process(&HandState) -> Option<Performance>` — one hand in, one performance out | `core/mod.rs` trait, `gesture/keyboard.rs:52` |
| Sinks | `Synthesizer` = one `Voice`; `MidiOutputEngine` = one `NoteGate` with `held: Option<u8>` | `audio/synthesizer.rs:79-143`, `midi/mod.rs:29-38` |
| Snapshot | `HandView` has one landmark Vec, one tip, one box | `dashboard/metrics.rs:128-151` |
| Stage | One skeleton, one detector box, one play-head line, one trail | `dashboard/skeleton.rs:647-753` |
| Presenter | `PresentJob { landmarks: Vec<Landmark> }` | `presentation/presenter.rs:29-33` |
| Mock mode | `synthetic_detection()` builds one hand | `vision/mock.rs:21-70` |
| Annotator | `--image` overlay draws one skeleton | `presentation/annotator.rs` |
| Trace | A golden trace of one note/volume per frame | `core/trace.rs`, `fixtures/mock_trace.json` |

**Already multi-hand-capable:** the Ultralytics pose decoder. `decode_pose()` returns a `Vec<RawDetection>` for every candidate and `run()` then throws away everything but the argmax (`yolo_pose.rs:288-299`). The multi-hand plumbing is *already there* in the pose path; it is the MediaPipe path that hard-codes one winner.

---

## 2. Three distinct designs for "a second hand"

### Design A — **Roles** (primary + modulator) ★ recommended
Hand 1 plays pitch. Hand 2 never plays a note; it drives parameters — timbre, filter, vibrato, delay, envelope. This keeps the instrument monophonic (no synth rewrite), keeps the keyboard unambiguous, and gives the player a "pedal board" made of a hand.

- Musical model: guitar (left hand frets, right hand picks), or a theremin + a filtering hand.
- Cost: one extra detection, no extra voice.
- Failure mode: a player who moves the modulator hand carelessly makes the sound wander. Needs deadbands and smoothing — the estimator already has the pattern (`PROXIMITY_EMA_ALPHA`, `APPROACH_EMA_ALPHA`).

### Design B — **Two voices** (duet / chord / drone)
Both hands pick notes. Requires polyphony: a voice pool in the synthesizer and a voice pool in the MIDI gate.

- Musical model: two melody lines, or a melody plus a drone root chosen by the left hand.
- Cost: same detection, plus `Voice` → `[Voice; N]` and `NoteGate` → `Vec<HeldNote>`, plus a release policy (what happens when a hand disappears?).
- Failure mode: two hands sweeping the same pentatonic grid at 30 Hz can produce clusters. Quantisation per voice is fine (the grid is already consonant), but the note floor (`--min-note-ms`) has to be per-voice.

### Design C — **Inter-hand geometry** (the relationship *is* the instrument)
Not two hands as two controllers, but one control defined by both: distance between fingertips, angle of the line joining them, how far apart the palms are, hands touching vs. hands wide.

- Musical model: the "hands apart = louder/brighter" metaphor is instantly legible and needs no training.
- Cost: A's detection cost; the mapping is a handful of floats.
- Failure mode: requires both hands visible. When only one is detected, every derived value must have a defined fallback.

These are not mutually exclusive. A realistic instrument is **A + C**: the left hand as a modulator, and the distance between the two hands as one global parameter.

---

## 3. How to actually detect two hands

*Decision made — see the Decisions log above. The four options below are the reasoning trail; Option 1 is the one being built.*

### Option 1 — **Full-frame detector pass + top-k anchors (MediaPipe)** ★ cheapest real path
The BlazeHand graph already scores **all 2016 anchors**; the code just keeps the argmax (`tracker.rs:261-279`). Take the top-k scores, apply spatial NMS on the anchor centres, and you have two (or more) palm boxes from **one inference pass**.

The catch is the ROI window: at 640×480 the crop side is `min(W,H) = 480`, so a single 192×192 detector window only ever samples a 480 px square of the frame — a second hand at the far edge is literally not in the input. Two ways out:

1. **Letterbox the whole frame into 192×192** (the sampler already supports this — `letterbox_map` exists, and `Affine` has independent `m00`/`m11`, so anisotropic scaling is expressible). One pass sees everything. Cost unchanged: ~9.5 ms. Slight resolution loss for a hand that was previously sampled from a 480 px region; acceptable for palm detection.
2. **Two anchored crops** (one per tracked palm): two detector runs, 2 × 9.5 ms. Accurate but expensive.

**Cost (option 1.1):** 9.5 ms detector + 4–5.4 ms landmarker for hand 1 ≈ **14.2 ms**, versus 11.1 ms today. Fits comfortably in the 33 ms budget.

### Option 2 — **Pose model, keep N detections (YOLO)** ★ already half-built
`decode_pose` returns every detection; `run()` filters by threshold and takes the best. Change "best" to "all above threshold, NMS'd, capped at 2" and multi-hand works.

The problem is not multi-hand — it is that 640×640 costs 64–69 ms, already over budget (measured 21 fps, 13 of 73 frames dropped). Multi-hand here is free *conditional on* the ROADMAP item "make the pose model cheap" (`--imgsz 320 --dynamic`, ~4× fewer FLOPs). At 320 the pose path becomes ~16 ms and is the best long-term answer: one pass, N hands, all 21 keypoints per hand, plus per-keypoint visibility scores.

### Option 3 — **Box-only modulator** ★ the smart cost cut
The modulator hand needs *presence, position, apparent size* — all of which come from the palm box. It does **not** need 21 landmarks. So:

- Run the detector once (full-frame) → two palm boxes.
- Run the landmarker only on the pitch hand.
- Marginal cost of the second hand: **~0 ms** beyond a wider detector window.

This unlocks most of the effect menu (§5) — proximity, position, approach rate, presence gating, inter-hand distance — without a second model run. Finger-level gestures on the modulator hand would then need landmarks, and could be enabled on demand with a cadence (see below).

### Option 4 — **Cadence the second hand**
Modulation does not need 30 Hz. Run the second hand's landmarker every *n*th frame (n = 3 → 10 Hz) and hold the last value. Vibrato and filter sweeps are generated in the audio domain anyway; a 10 Hz control signal is musically transparent. Marginal cost ≈ 4.7 ms / 3 ≈ **1.6 ms/frame**.

### Cost summary

| Path | Detector | Landmarker(s) | Inference p50 | Headroom vs 33 ms |
| :--- | ---: | ---: | ---: | ---: |
| Today (MediaPipe, one hand) | 9.5 ms | 4–5.4 ms | **11.1 ms** | 22 ms |
| 1 — full-frame pass + 2 landmarkers | 9.5 | 2 × 4.7 | **~19.4 ms** | 13 ms |
| 3 — full-frame pass + 1 landmarker (box-only modulator) | 9.5 | 1 × 4.7 | **~14.2 ms** | 19 ms |
| 4 — 2 landmarkers, second at 10 Hz | 9.5 | 4.7 + 1.6 | **~15.8 ms** | 17 ms |
| 2 — YOLO pose @640, N hands | 64–69 | — | **64 ms** | over budget |
| 2 — YOLO pose @320 dynamic, N hands | ~16 | — | **~16 ms** | 17 ms |

Note the budget guard (`Pipeline::shed_if_over_budget`, `pipeline.rs:628-660`) currently sheds **ascii and feed** on a p95 breach. With two hands it should also be able to shed the **second hand** first — picture and a spare hand are both non-sound-path work, and a shed hand is less noticeable than a dark stage.

---

## 4. Which hand is which

Four mechanisms, in increasing complexity:

1. **Frame side** (`left` = modulation, `right` = pitch). With mirroring on (the default), the user's right hand appears on the right — the same side the high notes are on. Zero cost, intuitive, breaks if the player crosses their hands.
2. **Sticky IDs by centroid tracking.** Greedy assignment of this frame's detections to last frame's hand slots, with a max-jump threshold. Handles occlusion and crossing hands; needs a "which slot is pitch" state that survives a dropout. ~30 lines.
3. **Geometric handedness.** Chirality from the landmarks: the sign of the cross product of (wrist→index MCP) and (wrist→pinky MCP), disambiguated by which side the thumb tip is on. **Caveat:** mirroring flips x, which inverts the sign, so the sign must be computed on unmirrored coordinates or flipped by the `mirror` flag. Reasonable, but not robust when the hand is edge-on.
4. **Landmark `z` chirality.** The signed volume of (wrist, index MCP, pinky MCP, thumb MCP). This is rotation-invariant in 3D, so it does not suffer the palm-vs-back ambiguity. But `z` from a crop-normalised landmarker is exactly the thing the codebase deliberately avoids (`gesture/mod.rs:94-99` documents why). Treat as experimental.
5. **A real handedness model** — the shipped `hand_detector.onnx` has only two outputs (scores, boxes); there is no handedness tensor. Adding BlazeHand's handedness variant is a third tiny model and a second crop. Not worth it for role assignment.

**Recommendation:** sticky IDs (#2) plus a `--hands pitch=right` config, and show the role assignment in the UI so the player can see which hand is doing what. Do not try to name hands "left"/"right" anatomically — name them **pitch hand** and **modulator hand**, and let the player swap them with a button.

---

## 5. What the second hand could do to the sound

A menu, ordered by musical payoff per unit of work. Each entry states the gesture source, the parameter it drives, and what it costs.

### 5.1 Filter / tone hand — ★ the best first effect
- **Gesture:** modulator hand's proximity (already computed: `state.proximity`), or its vertical position in the band.
- **Drives:** one-pole (or two-pole) lowpass cutoff, 200 Hz → 8 kHz.
- **Feel:** raising the far hand brightens the note; lowering it darkens it. Reads like a theremin player cupping their hand.
- **Cost:** a few multiplies per audio sample in `Voice::next_sample`. No allocation, no lock.

### 5.2 Vibrato hand
- **Gesture:** modulator proximity → depth; modulator height → rate. Or the *inter-hand distance* → depth.
- **Drives:** an LFO added to the oscillator phase.
- **Cost:** one phase accumulator. Trivial. This is the single cheapest expressive win in the whole menu.

### 5.3 Hands-apart / hands-together — a natural global control
- **Gesture:** distance between the two index tips (or the two palm centres), normalised by frame size; hands touching → 0, hands at opposite edges → 1.
- **Drives:** vibrato depth, delay amount, "spread" of a detuned voice pair, or simply volume.
- **Why it works:** it is the one two-hand gesture that needs no calibration and no role assignment — it is symmetric, so it works even if the tracker cannot tell which hand is which.

### 5.4 Second hand as a switch bank
- **Gesture:** `HandGeometry::extension[0..5]` already exists (`core/geometry.rs:23`): five per-finger extension values. Turn each into a boolean with a threshold + hysteresis.
- **Drives:** thumb = octave shift, index = sustain/hold, middle = vibrato on, ring = delay on, pinky = mute.
- **Cost:** no new model, no new geometry. Discrete, readable in the UI (draw the second skeleton's fingers lit/dim), easy to test.
- **Risk:** five boolean gates from noisy landmarks is a flicker machine unless the extensions are smoothed (the estimator currently smooths only the tip — ROADMAP §Gesture layer notes this).

### 5.5 Envelope / attack hand
- **Gesture:** the modulator hand's `approach` rate (already computed and EMA-smoothed, `estimator.rs:183-202`), or a pinch of the modulator hand.
- **Drives:** attack time (slow approach → swelling attack), release time, or a "charge and release" — bring the modulator hand close to arm the note, pinch to fire it.
- **Why it's appealing:** the estimator already produces this signal and nothing uses it except the dashboard readout.

### 5.6 Drone / chord hand (needs polyphony)
- **Gesture:** modulator hand's index tip on the same grid, but its cell becomes a sustained root or fifth under the melody.
- **Drives:** a second voice, `Voice` → a small pool.
- **Cost:** the synth and MIDI gate rewrites (see §6). The pentatonic grid guarantees the two hands can never clash, which makes this unusually safe to build.

### 5.7 Delay / echo hand
- **Gesture:** modulator hand's y position → delay time; proximity → feedback; presence → on/off.
- **Cost:** a fixed-size ring buffer allocated once at `start()`. The audio callback must not allocate, so size it at `max_delay_ms × sample_rate` up front.

### 5.8 Detuned pair / "thickness"
- **Gesture:** inter-hand distance → detune in cents between two oscillators.
- **Feel:** hands apart = thick/chorus, hands together = pure unison. Very cheap (two oscillators, one sum).

### 5.9 Tempo / rhythm from the modulator
- **Gesture:** periodicity of the modulator hand's motion (already measurable from the jitter/approach machinery).
- **Drives:** LFO rate or delay time as a beat multiple. Interesting but hard to make reliable; park it.

### 5.10 Panning / spatial (needs stereo)
- **Gesture:** modulator x position → left/right balance.
- **Cost:** `cpal` is currently mono. Requires a stereo stream config and a two-channel buffer. Do this only if spatial effects matter enough to justify it.

### 5.11 The do-it-with-one-hand alternative
Worth stating plainly: §5.4 (finger switches), §5.5 (approach-based envelope), and a hand-shape filter mapping are all achievable **with the hand you already have**. `geometry.extension`, `spread`, `tilt_deg`, and `approach` are computed every frame and used by nothing except the dashboard. A single hand with five finger gates already gives "effects" without touching the tracker at all. This is the right first step if the goal is expressive sound rather than two-hand performance per se.

---

## 6. What the audio engine would need

`Voice` today (`audio/synthesizer.rs:79-142`) is: target/current frequency, target/current amplitude, phase, waveform - with one-pole attack/release ramps, band-limited harmonics, and soft clipping. `Target` is four atomics; the callback takes no lock, allocates nothing, makes no syscall. That constraint is the design boundary; anything added must respect it - see the Step 2 chain above.

**What the decided design does *not* need:** no voice pool, no `[Voice; N]`, no per-voice `NoteGate`. Polyphony is Phase 4, and when it happens it is the only genuinely structural change to the audio path.

**Sink API.** The cleanest extension consistent with the existing design is a second method on `Sink`, defaulted, exactly like `set_waveform` (`core/mod.rs:68-69`):

```rust
fn set_modulation(&mut self, _m: &Modulation) {}
```

and a second bus variant, `BusMessage::Modulation(Modulation)`, so pitch events stay untouched and sinks that cannot modulate ignore them. For MIDI this is the free win: **CC 1 (modulation wheel) and CC 11 (expression)** already exist on every synth, so a second hand driving CC 1 works with no audio-code change at all.

---

## 7. Data model — the minimum viable change

Rather than a wholesale rewrite, the type changes can be additive:

1. `HandDetectionResult` → `HandsDetection { hands: Vec<HandDetection>, frame_size, timings }`, where `HandDetection { palm: PalmBox, box, score, landmarks: Option<Vec<Landmark>> }`. Slots are ordered, pitch first. A fixed `[HandSlot; 2]` is preferable to a growing Vec: two hands is a fixed-size problem, and it keeps the role semantics explicit.
2. `HandState` → `HandsState { pitch: HandState, modulator: Option<ModulatorState>, gap_norm: f32, touching: bool }`, where `ModulatorState` is palm-only (centre, half, score, smoothed proximity, smoothed approach). `HandEstimator` holds two gates plus the inter-hand computation.
3. `Strategy::process(&HandsState) -> Option<Performance>` — existing strategies only read `.primary`, so `keyboard.rs` and `theremin.rs` need no change.
4. `Modulation` struct in `core/`, produced by a new `gesture/modulation.rs` (a "modulation strategy" parallel to the pitch strategy).
5. `HandView` gains `secondary: HandView` or becomes `[HandView; 2]`; the stage draws two skeletons in two colours and only the pitch hand lights the keyboard trail.
6. Mock mode gets a second synthetic hand driven by the same frame index — necessary for `make mock-trace-check` to remain a real gate.

**Anti-pattern to avoid:** don't make everything `Vec<Hand>` "just in case". Two hands is a fixed-size problem; `[HandSlot; 2]` is cheaper, allocation-free, and makes the role semantics explicit.

---

## 8. UI / dashboard consequences

- Two skeletons, two colours (e.g. the existing `GROUP_COLORS` palette for the pitch hand, a muted grey/teal for the modulator). Draw the modulator hand *before* the pitch hand so the playing hand is on top.
- Which cell lights the trail? Only the pitch hand. A second trail would make the keyboard unreadable.
- The rail needs a **MODULATOR** card: which hand is which, a swap button, an enable toggle per effect, and a small meter per driven parameter (cutoff, vibrato depth/rate, delay) so the player can see what their other hand is doing. The modulator meter is what makes an analog effect learnable - without it, the second hand is invisible.
- The `lost_ms` readout becomes per-hand.
- ASCII art and the feed are unaffected (they work on pixels, not landmarks).

---

## 9. Testing plan

The codebase's strength is that `advance()` is a pure function of the frame and pipeline state, so two-hand behaviour is testable without a camera:

- **Detector NMS** — synthetic anchor scores with two clusters → exactly two palm boxes, no duplicates within a hand's radius. Pure function, unit test.
- **Assignment stability** — two hands crossing; sticky IDs do not swap when the centroids are closer than the jump threshold.
- **Gate independence** — hand A drops for 2 frames while B is solid; A is still `present` inside its grace window, B unaffected.
- **Degenerate cases** — one hand, zero hands, both hands overlapping, a hand clipped by the frame edge. Every derived inter-hand value needs a defined fallback.
- **Effect stages** — pure functions of `(state, sample)`, so each stage is testable without a device: the lowpass attenuates above its cutoff, vibrato adds the expected sidebands, delay returns a sample after the configured number of frames, and a disabled stage is a pass-through. Add a stage to a `Voice::next_sample` test rather than listening for it.
- **Neutral on loss** — feed a hand, remove it, assert every parameter decays to its neutral value rather than holding or snapping.
- **Mock mode** — a second synthetic palm with a deterministic phase offset; golden trace extended to a two-hand trace (`fixtures/mock_trace.json` becomes per-hand).
- **Budget** — measure `make fixture` with two hands and record the numbers in `bench/baseline.md`. Perf claims need numbers, per the working conventions.

---

## 10. Risks and open problems

1. **Occlusion.** Two hands overlapping is the single most common failure, and the NMS merges them into one palm. That is *fine* for a modulator — two hands touching is itself a gesture — but it means inter-hand distance reads 0 whenever the hands are close, so effects driven by `gap_norm` must be monotonic and predictable in that region.
2. **False positives.** A full-frame detector pass sees faces, arms and the player's torso, where the anchored crop never looked. `DEFAULT_SCORE_THRESHOLD = 0.6` already rejects most of it (a real hand scores ~0.9, an empty frame ~0.15), but the second hand will surface cases the single-argmax path never did. Measure the false-positive rate before lowering any threshold.
3. **Detector resolution.** The palm-anchored crop sampled a hand at 192 px of 480; a full-frame pass samples it at 192 px of 640. Roughly 25 % fewer effective pixels per hand. Whether that costs detections is an empirical question — `make fixture` answers it.
4. **Trace reproducibility.** The golden trace is already machine-sensitive (ROADMAP, "Open gaps"). Adding a second hand increases the sensitivity; keep the mock trace as the CI gate.
5. **Monophony is load-bearing.** `--min-note-ms`, the trail, and `NoteGate` all assume one sounding note. Two voices makes all three need per-voice state.
6. **Budget.** With the decided design inference stays near 11 ms, so there is no immediate pressure. The headroom is what pays for the effects engine later — if the modulator ever gets landmarks, the number goes to ~19 ms of the 33 ms budget, and the budget guard needs a new shed order (second hand before ascii/feed).

---

## 11. Recommended sequence

| Phase | What | Effort | Why first |
| :--- | :--- | :---: | :--- |
| **0** | **Effect chain.** `core/effects.rs` (enum + enable mask + neutrals), stages in `Voice`, `Target` atomics, `BusMessage::Modulation`, `Sink::set_modulation` (no-op by default). Drive it from the hand you already have (`approach`, `spread`, `proximity`). | **S-M** | Testable without a camera, zero tracker risk, and the instrument is more expressive before a second hand exists. |
| **1** | **Measure the detection change.** Run the palm detector over both the letterboxed full frame and the anchored crop, print both counts under `--debug`, compare with the recorded 106/120. Needs a two-hand clip. | **S** | Cheapest possible experiment, and it decides whether Phase 2 is allowed to happen. |
| **2** | **Two hands.** Full-frame letterbox pass; argmax -> top-k + spatial NMS in `run_detector()`. Sticky slots, per-hand gates, two palm boxes, mock mode + trace extended. | **M** | Unlocks proximity, position, approach, presence and inter-hand distance for essentially 0 ms of extra inference. |
| **3** | **Mapping + UI.** `gesture/modulation.rs` wires sources to filter, vibrato and delay. Two skeletons, a MODULATOR card with a toggle per effect, a swap button. | **M** | Makes the second hand audible and visible. |
| **4** | **Mapping as data.** A `Mapping { source, param, band }` table configured from the CLI/UI instead of hardcoded wiring. | **M** | This is where the modularity requirement pays off - effects become additions, not edits. |
| **5** | **MIDI modulation.** CC mapping for the parameters the audio path already carries. | **S** | Deferred deliberately: prove the audio chain first, then test the CCs against the OP-1. |
| **6** | **Landmarks for the modulator hand** (finger switches, pinch), cadenced at 10 Hz. | **M** | Only worth it once Phase 3 proves players use the second hand. |
| **7** | **Polyphony** - parked by decision. Voice pool in the synth, `Vec<HeldNote>` in the MIDI gate, drone/chord mapping. | **L** | A different instrument. Revisit only after the modulation story is solid. |
| **8** | **Cheaper pose model** (`--imgsz 320 --dynamic`), which makes multi-hand essentially free and removes the two-stage complexity entirely. | **M** | Long-term architecture win, independent of hands. |

Build order follows the decisions: **chain (0), measurement (1), hands (2), mapping (3)**. The chain can be built and tested before the second hand exists, which de-risks the whole thing - you get a working effect engine that a single hand can drive, and the second hand becomes a source swap rather than a new subsystem.

---

## 13. PR 1 spec - the effect chain, file by file

Goal: a working, individually enabled effect chain (lowpass, vibrato, delay) in the audio path, driven by the hand that already exists. No tracker changes, no second hand. Everything here is testable without a camera.

### `src/core/effects.rs` (new)

The declaration point for every effect. Keeps the "adding an effect is a checklist" rule in one file.

```rust
pub enum Effect { Lowpass, Vibrato, Delay }          // name(), parse(), all(), bit()

pub struct EffectMask(u8);                            // NONE, ALL, parse("lowpass,delay"),
                                                      // is_on(effect), set(effect, on), names()

pub struct Modulation {                               // the payload that travels to sinks
    pub enabled: EffectMask,
    pub cutoff_hz: f32,        // neutral: NEUTRAL_CUTOFF_HZ (12 kHz, transparent)
    pub vibrato_hz: f32,       // neutral: 4.0
    pub vibrato_cents: f32,    // neutral: 0.0
    pub delay_ms: f32,         // neutral: 0.0
    pub feedback: f32,         // neutral: 0.0
}

impl Modulation { pub fn neutral() -> Self; }        // Default delegates to neutral()
```

Constants (starting values, tuned by listening):

```text
CUTOFF_MIN_HZ 200    CUTOFF_MAX_HZ 8000   NEUTRAL_CUTOFF_HZ 12000
VIBRATO_RATE_MIN 0.5 VIBRATO_RATE_MAX 8   VIBRATO_MAX_CENTS 50
DELAY_MAX_MS 600     FEEDBACK_MAX 0.6
```

Three pure functions, all unit-testable:

```rust
/// Sources in, parameters out. In this iteration the pitch hand stands in for the
/// modulator; when a second hand arrives, the gap and the modulator's palm replace them.
pub fn modulation_from_hand(
    enabled: EffectMask,
    proximity: f32,   // 0..1 - how close the hand looks
    height: f32,      // 0..1 - index tip height in the band, 1 = top of frame
    spread: f32,      // 0..1 - thumb-to-index openness
    x_position: f32,  // 0..1 - index tip across the band
) -> Modulation;

/// One EMA step toward `target`; alpha = frame_dt / fade_seconds.
pub fn blend(current: Modulation, target: Modulation, alpha: f32) -> Modulation;

/// Cheap change test so the bus does not carry 30 identical messages a second.
pub fn is_different(a: &Modulation, b: &Modulation) -> bool;
```

Mapping table implemented here: cutoff <- proximity (near = bright), vibrato depth <- spread, vibrato rate <- height, delay time <- x position, feedback <- proximity.

Tests: `parse` accepts `lowpass,vibrato,delay` and rejects junk; empty string is NONE; `blend` reaches neutral after ~10 steps at alpha 0.07; `is_different` ignores sub-threshold jitter but catches a real cutoff change; the mapping saturates at both ends of every band.

### `src/core/mod.rs`

- `pub mod effects;`
- Add to `Sink`, next to `set_waveform`, with the same defaulted no-op contract:

```rust
/// Timbral change from the effect chain. Sinks that cannot express it (MIDI in this
/// iteration) ignore it, so every sink stays trivially conformant.
fn set_modulation(&mut self, _modulation: &Modulation) {}
```

### `src/core/event_bus.rs`

- `BusMessage::Modulation(Modulation)` as a third variant alongside `Event` and `Shutdown`.
- `EventBus::dispatch_modulation(&self, modulation: Modulation)`.
- Doc comment: modulation is *not* a performance, so it does not feed the dispatch-latency sample and the sink loop must not treat it as a `QueuedEvent`.

Tests: dispatching a modulation then a shutdown drains in order, and the shutdown still breaks the loop.

### `src/audio/synthesizer.rs`

**`Target`** gains five atomics beside the existing four (`enabled: AtomicU8`, four `AtomicU32` holding `f32` bit patterns) plus `Target::set_modulation(&Modulation)` and `Target::chain()`, which reads them into a plain `Chain` struct **once per callback block**, not per sample.

**`Voice`** gains callback-local state:

```rust
filter_y: f32,          // one-pole state
filter_cutoff: f32,     // the cutoff the cached coefficient was computed for
filter_k: f32,
lfo_phase: f32,
delay: Vec<f32>,        // ring buffer, sized once in start()
delay_write: usize,
```

`Voice::idle(sample_rate)` allocates the ring buffer (`(DELAY_MAX_MS / 1000.0 * sample_rate) as usize + 1`, ~106 KB at 44.1 kHz). That is the only allocation, and it happens before the stream plays.

**`Voice::next_sample(sample_rate, chain)`**, in the documented order:

```text
1. glide frequency toward target          (existing, unchanged)
2. vibrato: vibrated = frequency + cents_to_hz(freq, cents) * sin(2*pi*lfo_phase)
   LFO phase advances by vibrato_hz / sample_rate
   `vibrated` is local - self.frequency is NOT polluted by it
3. oscillator sample at `vibrated`
4. lowpass: y += (x - y) * k, k recomputed only when the cutoff changed
5. delay: out = x + feedback * ring[write - taps]; no-op when delay_ms < 1
6. soft_clip(out)                          (existing)
```

Two stage functions extracted as free functions so they are testable without a device:

```rust
fn one_pole(state: &mut f32, sample: f32, k: f32) -> f32;
fn comb(buffer: &mut [f32], write: &mut usize, sample: f32, taps: usize, feedback: f32) -> f32;
```

`Sink::set_modulation` for `Synthesizer` writes the atomics. No lock, no allocation - the same contract as `set_waveform`.

Tests:
- `one_pole` attenuates a 4 kHz sine far more at a 500 Hz cutoff than at 8 kHz.
- `comb` with taps = 10 returns the impulse at sample 10 and nothing before it.
- `comb` with feedback 0 is a pass-through; with 0.5 it rings.
- An all-off chain has the same RMS as the current voice.
- Vibrato does not shift the centre pitch: mean frequency over a cycle equals the target.

### `src/midi/mod.rs`

**No change.** It inherits the default no-op `set_modulation`, deliberately: CC mapping waits until the audio chain is proven and the CCs are tested against the OP-1.

### `src/pipeline.rs`

- `Pipeline` gains `modulation: Modulation` and `last_modulation: Instant`.
- New stage in `advance()`, **before** `dispatch_gesture` so the chain is set when a note attacks:

```rust
fn update_modulation(&mut self, state: &HandState, settings: &RuntimeConfig) {
    let target = if state.present {
        modulation_from_hand(settings.effects, state.proximity, height, spread, x)
    } else {
        Modulation::neutral_with(settings.effects)
    };
    let dt = self.last_modulation.elapsed();
    let alpha = (dt.as_secs_f32() / (self.trail.fade_ms() / 1000.0)).clamp(0.0, 1.0);
    let next = blend(self.modulation, target, alpha);
    if is_different(&self.modulation, &next) {
        self.event_bus.dispatch_modulation(next);
    }
    self.modulation = next;
    self.metrics.sound.modulation = next;
    self.last_modulation = Instant::now();
}
```

- `pause()` sets `modulation` to neutral and dispatches it once, so a stopped run is silent *and* unfiltered.
- `apply_commands()` handles `UiCommand::SetEffect(effect, on)` by writing into the shared `RuntimeConfig`.
- `drain_events` and the sink thread each need a `BusMessage::Modulation(m) => for sink in sinks { sink.set_modulation(m) }` arm. The dispatch-age sample stays on `Event` only.

Tests (all through `advance(synthetic_frame(..))`, no camera):
- A hand that moves closer raises the cutoff; a hand that leaves decays it back to neutral.
- `--effects none` keeps every parameter at neutral whatever the hand does.
- Repeated identical frames dispatch no modulation messages at all.

### `src/dashboard/metrics.rs`

- `SoundView` gains `pub modulation: Modulation`; the manual `Default` impl returns `Modulation::neutral()`.

### `src/dashboard/mod.rs`

- `Message::ToggleEffect(String)` -> `UiCommand::SetEffect(effect, on)`.
- New `effects_card(&FrameMetrics)` in the **basic** rail above `loudness_card`, mirroring its structure: one `small_button` per effect plus a `Meter` per active parameter, reusing the existing `meter_card` helper. An analog effect the player cannot see is an effect they cannot learn.

### `src/main.rs`

- `#[arg(long, default_value = "lowpass")] effects: String`, doc comment naming the three names and `none`.
- Parse through `EffectMask::parse`, error out with the available names exactly like `--volume` and `--ascii-ramp` do.
- Seed `RuntimeConfig` with `effects`.

### `README.md` + `CHANGELOG.md`

- Flags table: `--effects lowpass,vibrato,delay`.
- CHANGELOG under **Unreleased**: the chain, its default (`lowpass` on, the others off), and the fact that modulation travels on the bus as its own message type.

### Acceptance

```bash
make verify                        # check + test + fmt --check + clippy -D warnings
make fixture                       # headless run; pitch/volume trace unchanged
make mock-trace-check              # the deterministic gate still passes
cargo run -- --mode mock --no-ui --debug   # watch the chain respond to the mock hand
```

The perf claim that matters: the callback gains a branch per stage and a few multiplies per sample. Measure inference per frame before and after - the loop is camera-paced at ~30 fps, so the number to watch is CPU headroom, not fps.

---

## 14. What is settled, and what is next

**Settled.** Detection: full-frame MediaPipe pass with top-k NMS, palm-box-only modulator, sticky slots with manual role swap, `--hands two` by default, one shared confidence threshold, capture resolution unchanged. Sound: modulation only (no polyphony), first chain = lowpass + vibrato + delay, inter-hand gap as a source, effects fade to neutral on hand loss using the existing `--fade-ms`, parameters travel on `BusMessage::Modulation` to a defaulted `Sink::set_modulation`. Sequencing: chain first, then measure, then hands.

**Open, in order of importance:**

1. **The `HAND_SEPARATION` constant** - it decides what counts as two hands, and it is measurable once a two-hand clip exists.
2. **Detection-rate regression** - does the full-frame pass find hands as reliably as the anchored crop? If it does not, the fallback is an anchored crop for the pitch hand plus a full-frame pass for the modulator.
3. **The effect bands themselves** - cutoff range, vibrato depth ceiling, delay max. Starting values, tuned by listening.
4. **MIDI CC mapping** - deferred until the audio chain is proven; CC 1 means pitch modulation on many synths, so it needs a test against the OP-1.
5. **A two-hand fixture and trace** - `make record` captures one hand; the regression gate needs a two-hand recording and a per-hand `mock_trace.json`.

---

## 15. PR 1 status - what was built, and where it differs from the spec

Built on `feat/effect-chain`, verified with `make verify` (208 tests) and
`make mock-trace-check`.

| Spec | Built | Difference |
| :--- | :--- | :--- |
| `src/core/effects.rs` with `Effect`, `EffectMask`, `Modulation`, `modulation_from_hand`, `blend`, `is_different` | as specced | added `all`/`on` as parse aliases for `EffectMask::ALL`, and named deadband constants (`CUTOFF_DEADBAND_HZ`, `VIBRATO_DEPTH_DEADBAND_CENTS`, `DELAY_DEADBAND_MS`, `FEEDBACK_DEADBAND`) so the "is it worth a message?" thresholds are documented in one place |
| `Sink::set_modulation` default no-op | as specced | - |
| `BusMessage::Modulation` | as specced | modulation is deliberately *not* timed, so it does not pollute the dispatch-latency sample that note events feed |
| Synthesizer chain `glide -> vibrato -> oscillator -> lowpass -> delay -> soft clip` | as specced | the delay ring keeps the voice awake while an echo is still in the ring (`is_silent` also checks `echo_tail`), so a released note rings out instead of being cut off |
| One-pole coefficient recomputed only when the cutoff moves | as specced | - |
| Delay ring sized once in `Synthesizer::start` | sized in `Voice::idle(sample_rate)`, which the stream calls before it plays | same guarantee: no allocation in the callback |
| Pipeline `update_modulation()` stage | as specced, run before `update_tracking` so a note attacks with the chain already in place | `pause()` also resets the chain to neutral, so a resumed run does not start from whatever the last hand left behind |
| Dashboard EFFECTS card | as specced: three toggles + one meter per stage, reading the published `Modulation` rather than the button states | - |
| `--effects` CLI flag | as specced, default `lowpass` | an unparseable list fails at startup rather than silently disabling the chain, same as `--ascii-ramp` |

Two things the spec left open and the implementation had to decide:

- **A disabled stage still tracks its parameter.** `modulation_from_hand` computes every
  parameter regardless of the mask; the mask decides which stages the voice runs. That
  means switching a stage back on resumes from the value the hand is currently making
  rather than from zero, and it is what the test
  `a_toggle_flips_the_chain_without_disturbing_the_levels` asserts.
- **The comb stores outputs, not inputs.** `y[n] = x[n] + feedback * y[n - taps]` is what
  makes feedback repeat; storing the input would give exactly one echo. Verified by
  `comb_echoes_an_impulse_after_the_configured_delay` and `without_feedback_the_echo_does_not_repeat`.

**Not done, by design:** MIDI modulation (section 12), and any change to the tracker.

---

## 16. PR 2 status - the detection measurement

`--detector-window both` runs the anchored crop as the behaviour and scores the whole
frame alongside it, then prints the comparison. `make probe` and `make probe-single`
run it on the two fixtures. Nothing in the instrument changes: `crop` is still the
default, and `both` is a measuring mode, not a playing mode.

### The numbers (200 frames each, release build)

| clip | window | frames with a hand | mean best score | hands/frame @ 1.0 | frames with 2+ hands @ 1.0 | detector µs/frame |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `hand_clip.mp4` (640×480, one hand) | crop | 173 | 0.838 | 0.87 | 1 | 8126 |
| | frame | **162** | 0.785 | 0.81 | **0** | 7902 |
| `2_hands_640.mp4` (640×480, two hands) | crop | 200 | 0.953 | 1.46 | 92 | 10486 |
| | frame | 200 | 0.957 | **1.90** | **179** | 10181 |
| `2_hands_clip.mp4` (1620×1080, two hands) | crop | 199 | 0.951 | 1.01 | 4 | 8900 |
| | frame | 198 | 0.949 | **1.84** | **171** | 8436 |

### What that says

**1. The full-frame window costs the same as the crop.** 7.9 ms vs 8.1 ms detector time.
Both sample the same 192×192 buffer; only the affine differs. The earlier "the frame
window is 2.5× slower" reading was contention from a second app instance running in
another terminal, not the sampler.

**2. Only the full frame sees a second hand.** On the 640×480 two-hand clip the anchored
crop finds a second palm in 92 of 200 frames; the letterboxed frame finds it in 179. At
native 1620×1080 the crop finds two hands in 4 frames out of 200 and the frame in 171 -
the crop's `min(W,H)` window simply cannot span two hands that far apart. This is the
reach problem the plan predicted, and it is decisive: **option 1 (full-frame pass) is the
route**.

**3. The full frame does lose detections on a single, distant hand - 173 → 162.** It is
real, but it is not a gate problem. Sweeping `--score-threshold` 0.6 → 0.5 → 0.4 moves
the frame window 162 → 165 → 169 while the crop goes 173 → 178 → 179 *and its false
positives grow from 1 frame to 11*. The gap is the 25% resolution loss the letterbox
 imposes on a hand that the crop could have magnified, not a threshold that is set too
high. Lowering the threshold buys crop noise, not frame recall.

**4. No false positives from the whole frame.** Across 753 detections on the two-hand
clips, not one palm centre landed in the top 22% of the frame where the face and torso
are. The padding bands do not manufacture hands.

### `HAND_SEPARATION`, measured

Hands per frame on `2_hands_640.mp4` as the separation gate widens:

| gate | 0.25 | 0.5 | 1.0 | 2.0 |
| :--- | ---: | ---: | ---: | ---: |
| crop | 1.46 | 1.46 | 1.46 | 1.18 |
| frame | 1.90 | 1.90 | 1.90 | 1.34 |

One hand's duplicate anchors are merged at every gate up to 1.0, and genuinely separate
hands only start merging at 2.0. The plateau is 0.25-1.0, so **`HAND_SEPARATION = 1.0`**
is the value - it is the widest gate that still separates the hands while leaving margin
before they start to merge. Measured palm-centre separation in the clip is 1.3-2.6 palm
radii, which is consistent with that.

### Decision

Option 1 stands. For PR 3 the pitch hand keeps the anchored crop - it is the more
sensitive one and pitch reliability is not worth trading for reach the pitch hand does not
need - and the **modulator** comes from a full-frame pass. That is two detector runs,
~16 ms of the 33 ms budget, which the probe already proved is affordable: `both` runs
exactly those two passes side by side in one process and still finishes inside the frame
interval, and the landmarker adds ~4-5 ms on top.

The alternative - one full-frame pass for everything - is cheaper but loses 11 detections
in 200 frames on the hand that plays the melody. Losing a note is worse than losing a
modifier, so the pitch hand keeps the crop.

---

## 17. PR 3 status - two hands, implemented

Built on `feat/two-hand-detection`, verified with `make verify` (220 tests) and the three
golden traces: `make mock-trace-check`, `make trace-check` and `make hands-check`.

### What the code does

| Piece | Where | What it does |
| :--- | :--- | :--- |
| Hand count | `src/core/mod.rs` | `MAX_HANDS = 2`; `HandDetection { landmarks, r#box, score }`; `HandDetectionResult.hands` in **role order**, with `pitch()` and `modulator()` |
| Which window | `src/vision/tracker.rs` | `DetectorWindow { Anchored, TwoHands, Probe }`, chosen by `--hands one\|two\|probe`; default `two` |
| Finding hands | `run_detector_window` + `non_overlapping` | every anchor above `--score-threshold` is collected best-first and merged by spatial NMS at the measured `HAND_SEPARATION = 1.0` |
| Keeping identity | `match_to_slots` | a hand stays in the slot whose remembered palm is within `SLOT_MAX_JUMP = 0.35` of the frame's short side, so crossing hands does not swap roles |
| Landmarking | `track_hand(frame, palm, slot)` | one function, run once per hand, each with its own `ReuseCache` - the second hand is processed exactly like the first |
| Estimating | `src/gesture/estimator.rs` | `update(Option<&HandDetection>, frame_size)`; the pipeline holds `[HandEstimator; MAX_HANDS]` |
| Roles | `src/pipeline.rs` | `pitch_slot: usize`, flipped by `UiCommand::SwapHands`; the anchored crop re-seeds from the pitch slot's palm |
| Modulation | `src/core/effects.rs` | `modulation_from_hands(enabled, proximity, height, gap, x)`; `gap_level(distance, touching)` normalises palm distance by the sum of the palm radii so it is scale-free |
| One-hand fallback | `update_modulation` | no second hand -> the pitch hand drives the chain and `gap` is that hand's openness - the pre-PR-3 behaviour, unchanged |
| Drawing | `src/dashboard/skeleton.rs` | `draw_hand` with a `HandStyle`; the modulator uses `MODULATOR_COLORS` at 0.55 strength and carries a "mod" label on the palm |

### Measurements

Two-hand fixture, 120 frames, release build:

| | one hand (`--hands one`) | two hands (`--hands two`) |
| :--- | ---: | ---: |
| inference p50 | 11.8 ms | 16.9 ms |
| inference p95 | 14.8 ms | 19.2 ms |
| inference max | 15.9 ms | 23.6 ms |

The second hand costs about 5 ms per frame - one extra detector pass plus a landmarker run,
both mostly hidden by the reuse cache - and the run still sits inside the 33 ms frame
budget, so the budget guard never sheds the feed or the ASCII art.

### The inter-hand band, measured

`gap_level` divides the palm-centre distance by the sum of the two palm radii. Across both
two-hand clips (108 sampled frames each) that ratio runs **0.8 - 1.4**, mean 1.1, p95 1.36 -
and it is the *same* numbers at 640x480 and at the native 1620x1080, which is the point of
normalising by the hands rather than the frame. `GAP_TOUCHING = 1.0` and `GAP_WIDE = 1.5`
therefore put the whole observed sweep inside the band with headroom, instead of a 2.0 top
that a real pair of hands at a desk never reaches.

### What is deliberately not done yet

- **Performance.** Two detector passes are the expensive part. A single full-frame pass
  feeding both hands, or a palm-box-only modulator that skips the landmarker, are both
  available next - the plan's measurements say the crop is worth keeping for the pitch hand,
  so the cheap win is the modulator's landmark stage, not the detector.
- **Polyphony.** Deferred, as in PR 1.
- **MIDI modulation.** The audio chain has to be proven first.
- **Automatic role assignment.** Deliberately manual: promotion on the fly would hijack the
  melody mid-sweep.
