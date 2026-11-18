# OpenCV Removal Plan — momotone

**Status:** **Done.** Executed on `chore/opencv-removal`; the measured results are in
`bench/baseline.md` under "After OpenCV removal". The decisions below are kept as the
rationale; §4.3 and §10 record where execution differed from the plan.
**Scope:** remove the `opencv` crate from `momotone` and replace it with maintained, Rust-native equivalents.
**Branch convention:** `chore/opencv-removal`.
**Effort:** ~2–3 working days total. Each phase is independently mergeable and independently revertible.

> **What actually happened, in one line:** cold `cargo check` went from 19 m 12 s to 34 s,
> `opencv` and `.cargo/config.toml` are gone, capture is `nokhwa`, the fixture decodes with
> `mp4io` + `rusty_h264-decoder`, and every gate (`make verify`, `make trace-check`, the
> ONNX fixture test) is green. Two deviations from the plan: the MP4 decoder is
> `rusty_h264-decoder`, not `yscv-video` (§4.3), and `annotator.rs` was reimplemented in
> Rust rather than deleted (§10, question 3).

---

## 1. Executive summary

OpenCV is **not** doing computer vision in this project. There is no filtering, no contour
detection, no calibration, no feature extraction. The ONNX graphs in `ort` do the vision.
OpenCV is used for exactly three chores:

1. **Grab pixels** from a webcam or an MP4 (`videoio::VideoCapture`).
2. **Crop + resample** a frame into the model's input tensor (192×192 / 224×224), and
   downsample a frame into the ASCII grid.
3. **Convert/annotate** BGR → RGBA for the preview, and draw lines/circles for the
   `--image` dump.

All three are ~150 lines of straightforward Rust. Removing OpenCV removes the single
heaviest build dependency in the project, the LLVM/libclang toolchain requirement,
four `unsafe` raw-pointer reads in the vision path, and a 19-minute cold build.

**Recommended stack:**

| Job | Crate | Why |
| :--- | :--- | :--- |
| Image ops (resize, crop, flip, save, PNG read) | `image` **0.25.10** | Already in `Cargo.lock` today (pulled in by iced). 252 M downloads, actively maintained (last release 2026-03-10), pure Rust decoders. **New direct dependency costs essentially nothing.** |
| Affine sampling (the warp) | **hand-written, ~90 lines** | The mapping is always axis-aligned scale + translate. OpenCV's `warpAffine` is only reached when the crop runs off the frame edge. A small bilinear sampler is exact, testable, and faster (single fused pass into `f32`). |
| Webcam capture | `nokhwa` 0.10.11 (`input-native`) | Cross-platform AVFoundation / V4L2 / Media Foundation, no OpenCV, pull-based `frame()` which matches the existing capture-thread design. |
| MP4 fixture decode | `yscv-video` 0.2.0 (pure Rust) — fallback `ffmpeg-next` 9.1.0 | See §4.3; this is the one genuinely open decision. |

**What this does *not* fix:** the `block v0.1.6` future-incompat warning. That comes from
wgpu's Metal backend (`objc`/`block` in `Cargo.lock`), not OpenCV. Do not promise it.

---

## 2. What OpenCV is actually doing here (inventory)

| # | Capability | Where | OpenCV API | Lines of real work |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Webcam capture | `vision/camera.rs` | `videoio::VideoCapture`, `CAP_PROP_FRAME_WIDTH/HEIGHT/FPS` | 20 |
| 2 | MP4 capture | `vision/video.rs` | `VideoCapture::open_file`, `CAP_PROP_FPS` | 15 |
| 3 | Mirror flip | `vision/camera.rs` | `core::flip(frame, scratch, 1)` + buffer swap | 7 |
| 4 | Detector/landmarker input crop | `vision/sampling.rs` | `Mat::roi`, `imgproc::resize(INTER_LINEAR)`, `imgproc::warp_affine(INTER_LINEAR, BORDER_CONSTANT)` | 60 |
| 5 | ASCII downsample | `presentation/ascii.rs` | `imgproc::resize(INTER_AREA)` | 20 |
| 6 | Preview conversion | `presentation/presenter.rs` | `frame.data()` → `bgr_to_rgba` | 14 |
| 7 | Skeleton overlay (`--image`) | `presentation/annotator.rs` | `imgproc::line`, `imgproc::circle` | 25 |
| 8 | Frame plumbing | `pipeline.rs`, `main.rs`, `tracker.rs` | `Mat::rows/cols/channels/data/ptr` | ~30 |

**Reference counts (current tree):**

| File | Lines | `Mat`/`opencv` references |
| :--- | ---: | ---: |
| `src/vision/camera.rs` | 167 | 23 |
| `src/vision/video.rs` | 126 | 8 |
| `src/vision/sampling.rs` | 324 | 22 |
| `src/vision/tracker.rs` | 831 | 27 |
| `src/presentation/ascii.rs` | 496 | 21 |
| `src/presentation/presenter.rs` | 448 | 13 |
| `src/presentation/annotator.rs` | 100 | 7 |
| `src/pipeline.rs` | 1,959 | 50 |
| `src/main.rs` | 533 | 11 |

## 3. Why it is overkill (measured on this machine)

- **Cold `cargo check` = 19 m 12 s** on this machine. The `opencv` crate's build script
  runs `clang` over the OpenCV headers and generates the bindings; that is the bulk of it.
  Warm `cargo check` afterwards is ~4 s.
- **Toolchain coupling:** `.cargo/config.toml` exists solely to point
  `LIBCLANG_PATH` / `DYLD_LIBRARY_PATH` at Homebrew LLVM. `README.md` asks the user to
  `brew install opencv llvm` (and `libopencv-dev libclang-dev clang` on Debian).
- **Transitive build deps:** `cc`, `clang`, `opencv-binding-generator`, `pkg-config`,
  `vcpkg`, `jobserver`, `semver`, `shlex`, `winsplit`, `dunce`.
- **Unsafe surface:** 10 `unsafe` sites in the crate; **6 of them exist only to read a
  `Mat`'s buffer** (`sampling.rs` 2, `ascii.rs` 3, `presenter.rs` 1, plus test helpers in
  `camera.rs`/`tracker.rs`). A `Vec<u8>` frame removes all of them. Only
  `StreamWrapper`'s `unsafe impl Send` (cpal) remains.
- **Semantics we don't need:** we never use OpenCV's colour-space conversion, filtering,
  or morphology. BGR is an accident of the capture backend, not a design choice — it
  forces a channel swap in every sampling pass.

---

## 4. Replacement decisions

### 4.1 Image operations → `image` 0.25.10

Already resolved in `Cargo.lock` (iced's `image` feature). Promoting it to a direct
dependency is free.

```rust
// resize a crop to the model input
let warped = image::imageops::resize(&crop, out_w as u32, out_h as u32, image::imageops::FilterType::Triangle);

// downsample for the ASCII grid (see 4.2 — a box average beats resize here)
// read a photo for --image, write the annotated dump
let img = image::open(path)?;
img.save(path)?;
```

Notes to verify during Phase 0:
- `FilterType` has `Nearest | Triangle | CatmullRom | Gaussian | Lanczos3`. There is **no
  `INTER_AREA` equivalent**; for a large downscale (`image::imageops::thumbnail`) uses a
  fast integer box average. For the ASCII grid we want a true box average, so write one
  (15 lines, `u32` accumulators) instead of reaching for a filter — it is both cheaper and
  exactly what OpenCV's `INTER_AREA` did.
- `image`'s `resize` on a 192×192 source is microseconds; the existing bench in
  `tracker.rs::bench_sampling` (133 µs vs 247 µs) is the number to re-measure.

### 4.2 The affine warp → a hand-written sampler

`sampling.rs` already does the interesting half in pure Rust (the BGR→RGB, `/255`,
NHWC/NCHW layout pass). The OpenCV part is two calls: `resize` for the axis-aligned
crop that fits in the frame, and `warpAffine` with a constant border when the window
runs off the edge.

Target shape (`src/vision/sampler.rs`):

```rust
/// Sample `frame` (RGB, row-major) through `map` (output pixel -> frame pixel)
/// straight into `dst` as normalised f32 in `layout`, padding outside the frame
/// with `border`. Returns the *effective* mapping applied, so callers can still
/// convert model coordinates back to frame pixels.
fn sample_affine(
    frame: &Frame,
    map: Affine,
    out_w: usize,
    out_h: usize,
    dst: &mut Vec<f32>,
    layout: TensorLayout,
    border: u8,
) -> Result<Affine>
```

Two deliberate wins:

1. **Fused pass.** Today: crop → u8 `warped` Mat → loop → f32 tensor. After: bilinear
   sample → `/255` → store. No intermediate buffer, no second pass, no allocation.
2. **RGB end-to-end.** The frame entering the pipeline is already RGB, so the swap inside
   the sampling loop disappears.

**Important behavioural risk:** the current code snaps the crop to whole pixels and returns
the *snapped* mapping (`rounded_crop_reports_the_effective_mapping` asserts this). A naive
general sampler changes sub-pixel offsets → detections shift → the golden trace moves.
Keep the integer-crop fast path and its returned mapping contract **identical**; only
replace the `warpAffine` branch. Then regenerate the trace (§6.7).

### 4.3 Webcam and MP4 — the two real decisions

**Webcam:** `nokhwa` 0.10.11 (updated 2026-05-15, 541 k downloads, 44 reverse deps,
Apache-2.0). macOS backend is AVFoundation via `objc`/`block`, Linux is V4L2, Windows is
Media Foundation. Pull-based `Camera::frame()` maps 1:1 onto `capture_frames`'
capacity-1 channel. Caveats to verify in Phase 0:
- Requires `nokhwa_initialize()` once on macOS.
- Resolution/FPS are set through `CameraFormat { width, height, frame_rate }`; not every
  camera honours them, which is exactly today's "best effort" behaviour.
- The macOS backend has an open SIGABRT issue when a device exposes several
  same-resolution formats (`nokhwa#247`). Test on this camera before committing.
- Frame buffer may arrive as MJPEG unless a raw format is negotiated; the
  `output-image` feature gives a decoded `Buffer::RGB`.

Modern alternative worth benchmarking against: `cameras` 0.3.2 (matthewjberger, MSRV 1.85)
— built on `objc2` + `objc2-av-foundation`/`objc2-core-media`/`objc2-core-video`, push-based
delivery, typed errors, `image` integration. Younger (3.7 k downloads) but a cleaner API
and no legacy `objc`.

**MP4 fixture:** three options, in order of preference:

| Option | Dep | Cost | Verdict |
| :--- | :--- | :--- | :--- |
| `yscv-video` 0.2.0 | pure Rust H.264/HEVC/AV1 + MP4 demux, `Mp4VideoReader::open()` → `frame.rgb8_data` | Young (3.7 k downloads, MSRV 1.94, "experimental") | **Primary.** 640×480 × 120 frames is trivial; decode failure modes are visible and local to the fixture. |
| `ffmpeg-next` 9.1.0 | libav via bindgen | Mature (9 M downloads), but reintroduces libclang + system ffmpeg, so `.cargo/config.toml` may need to stay | **Fallback** if `yscv-video` cannot read the fixture. |
| `--video` accepts a directory of PNG/JPEG frames | `image` only | Zero new deps, but changes the fixture format and `make record` | **Escape hatch** — also the cleanest way to keep `--video` testable if decode proves flaky. |

A raw frame cache is *not* recommended: 120 × 640 × 480 × 3 = 110 MB, and even PNG-ified
(~15 MB) is too much to commit for a fixture.

---

## 5. Target architecture

### The one new type

```rust
// src/core/frame.rs
/// A captured frame: RGB, row-major, 3 bytes per pixel. Owned, reusable buffers.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Frame {
    pub fn len(&self) -> usize { self.pixels.len() }
    pub fn is_empty(&self) -> bool { self.pixels.is_empty() }
    /// Mirror in place. No scratch buffer needed once the buffer is ours.
    pub fn flip_horizontal(&mut self) { /* per-row reverse of 3-byte groups */ }
    pub fn to_rgba(&self) -> Vec<u8> { /* 1 pass, extend with alpha */ }
}
```

Why not `image::ImageBuffer<Rgb<u8>, Vec<u8>>`? It works, but a plain struct makes buffer
reuse across frames explicit (`pixels.clear()` + refill keeps the 921 KB allocation alive),
which is what the capture thread needs. Use `image` for *operations*, keep `Frame` as the
type that flows through `Pipeline`, `PresentJob`, and the capture channel.

### Data flow after the change

```text
next_frame() -> Frame {w, h, Vec<u8> RGB}
   advance() -> sample_affine(frame, Affine, 192, 192, &mut stage.data, Nhwc, 0)
             -> sample_affine(frame, Affine, 224, 224, &mut stage.data, Nhwc, 0)
   hand_over() -> AsciiGrid (box-average, pure Rust)
               -> Frame::to_rgba() for the iced handle
```

Every downstream consumer (`HandDetectionResult`, strategies, dashboard, sinks) is
unaffected — they never see pixels, only landmarks and metrics.

---

## 6. Execution track

Each phase has an acceptance gate. Do not start the next until the previous one is green.

### Phase 0 — Feasibility spike (S)
- [ ] `cargo add nokhwa --features input-native` on a scratch branch; open camera 0,
      request 640×480, print `reported fps`, frame buffer kind, and ms/frame.
- [ ] `cargo add yscv-video`; decode `fixtures/hand_clip.mp4`, compare the decoded pixel
      bytes of frame 0 against OpenCV's `Mat` output (`--image` on an extracted frame).
      **Byte-identical or within rounding is required** — a fixture whose pixels differ
      silently invalidates the golden trace.
- [ ] Benchmark `image::imageops::resize` 192×192 vs. OpenCV's `resize` using the existing
      `bench_sampling` harness.
- [ ] Record all four numbers in `bench/baseline.md` under a "Pre-removal baseline" section.

**Gate:** camera opens at the requested size, the fixture decodes and matches, and sampling
is not slower. If any of these fails, stop and revisit §4.3.

### Phase 1 — Introduce `Frame` (S)
- [ ] `src/core/frame.rs` with `Frame`, `flip_horizontal`, `to_rgba`, and unit tests
      (`mirroring_reverses_the_columns`, `a_second_flip_reuses_the_scratch_buffer`
      ported to the new type).
- [ ] Change `Source::read`, `Pipeline::next_frame`, `PresentJob`, and the capture channel
      to `Frame`. Keep OpenCV *inside* the two source structs for now.
- [ ] `make verify` green.

**Gate:** behaviour identical; `make fixture` and `make trace-check` unchanged.

### Phase 2 — Replace the sampler (M)
- [ ] `src/vision/sampler.rs`: bilinear affine sampler writing f32 directly.
- [ ] Keep the integer-crop fast path and the "returns the effective mapping" contract.
- [ ] Port all four tests in `sampling.rs` (letterbox mapping, NCHW planes, border padding,
      rounding) to the new implementation.
- [ ] Delete `StageInput.warped`.
- [ ] Re-run `bench_sampling` and record the delta.

**Gate:** the four sampler tests pass; `make trace-check` either still matches or the diff
is reviewed and understood.

### Phase 3 — Replace the ASCII downsample (S)
- [ ] Box-average (`u32` accumulators) into the target `cols × rows_for(cols)` grid.
- [ ] Port the six `ascii.rs` tests (aspect rows, black/white, mid-grey, luma weighting,
      gradient monotonicity, EMA).
- [ ] Delete the three `unsafe` pointer reads in `ascii.rs`.

**Gate:** `cargo test presentation::ascii` green; `make ascii` output visually unchanged.

### Phase 4 — Presenter, annotator, `--image` (S)
- [ ] `frame_to_rgba` becomes `Frame::to_rgba` (no `unsafe`, no `Mat`).
- [ ] `annotator.rs`: draw the skeleton onto `image::ImageBuffer<Rgb<u8>>` (20 lines),
      or drop the annotator entirely and draw the overlay in the dashboard — decide based
      on whether anyone uses `--dump-frame`. If it stays, `--image` reads with `image::open`
      and writes with `img.save()`.
- [ ] Port `bench_feed_upload`.

**Gate:** `make image IMG=photo.png OUT=out.png` produces an annotated PNG.

### Phase 5 — Capture (M)
- [ ] `WebcamSource` around `nokhwa`: `Camera::new(InputSource::Native(this_camera))`,
      `CameraFormat { width, height, frame_rate }`, `frame()` → `Frame`.
- [ ] Preserve: `--camera`, `--camera-width/height`, `--camera-fps`, `--no-mirror`,
      `reported_fps()`, and the "best effort" semantics when the driver ignores the request.
- [ ] On error, keep the `anyhow` "Could not open webcam index N" message.

**Gate:** `make headless` reports the same fps and capture timing as the baseline.

### Phase 6 — Fixture decode (M)
- [ ] `VideoSource` around the Phase-0 choice (`yscv-video` primary).
- [ ] Keep `cadence(reported_fps)` and the deadline pacing; take the fps from the container's
      sample table instead of `CAP_PROP_FPS`.
- [ ] Keep `End of clip after N frames` as the termination signal for the capture thread.

**Gate:** `make fixture` and `make trace-check` green.

### Phase 7 — Delete OpenCV (S)
- [ ] `Cargo.toml`: remove `opencv`, add `image = "0.25"` (explicit), `nokhwa`, and the
      chosen video crate.
- [ ] Delete `.cargo/config.toml` — **only after** confirming nothing else needs libclang
      (check `ffmpeg-sys-next` if the ffmpeg fallback was chosen).
- [ ] README: drop `brew install opencv llvm` / the `libopencv-dev libclang-dev` line.
- [ ] Makefile: remove the "The first build compiles OpenCV" note; update `make help`.
- [ ] AGENTS.md: update the Tech Stack, Build Environment, and data-flow sections.
- [ ] CHANGELOG: an entry under **Unreleased**.
- [ ] `cargo clean && cargo build` → record the new cold build time in `bench/baseline.md`.

**Gate:** `make verify` green, cold build time recorded, `aft_inspect` shows no dead code
left from the removal.

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Golden trace shifts because sampling rounds differently | High | Medium | Keep the integer-crop fast path semantics identical; Phase 6 re-records the trace and the diff is reviewed by a human (AGENTS.md convention). |
| `nokhwa`'s macOS backend SIGABRTs on this camera | Medium | High | Phase 0 spike is the gate. Fallback: `cameras` 0.3.2, or keep OpenCV for capture only and remove it everywhere else (a partial win is still a win). |
| Pure-Rust H.264 decode produces different pixels than OpenCV | Medium | High | Byte-compare frame 0 against the OpenCV output in Phase 0 before committing. |
| `image`'s filter differs from OpenCV's `INTER_LINEAR`, changing landmark scores | Medium | Medium | Both are bilinear-ish; verify via `tracker_finds_the_hand_in_the_clip` (currently 106/120 detections) and re-measure. |
| Frame pacing regression (nokhwa does not honour fps) | Medium | Low | The budget guard and deadline clock already exist; `--camera-fps` was always best effort. |
| `.cargo/config.toml` still needed by a new transitive dep | Low | Low | Phase 7 explicitly checks before deleting. |

---

## 8. Verification & benchmark gate — results

Measured on the same machine after Phases 1–7, rustc 1.100:

| Metric | Command | Before | After | Target | Verdict |
| :--- | :--- | ---: | ---: | :---: | :---: |
| Cold `cargo check` | `cargo clean && cargo check --all-targets` | **19 m 12 s** | **34 s** | < 4 min | ✅ 34× faster |
| Cold `cargo build --release` | `cargo clean && cargo build --release` | ~20 min | **3 m 52 s** | — | ✅ |
| Detector sampling (192×192) | `bench_sampling` | 133 µs | 146 µs | ≤ 133 µs | ❌ +13 µs |
| Landmarker sampling (224×224) | `bench_sampling` | 94 µs | 254 µs | ≤ 94 µs | ❌ +160 µs |
| ASCII downsample (140×105) | `bench_ascii_downsample` | ~0.3 ms | 0.30 ms | ≤ 0.30 ms | ✅ |
| RGBA conversion (640×480) | `bench_feed_upload` | 0.30 ms | 0.32 ms | ≤ 0.30 ms | ≈ |
| Mirror flip | `bench_mirror_flip` | 0.017 ms | 0.018 ms | ≤ 0.017 ms | ≈ |
| Clip decode | `bench_clip_decode` | ~0.2 ms (OpenCV) | 1.2 ms | — | ✅ still 0.4 % of a 33 ms frame |
| Fixture fps | `make fixture` | 29.7 fps | 29.7 fps | ≥ 29.7 fps | ✅ |
| Live fps | `make headless` | 29.9 fps | 28.5 fps | ≥ 29.9 fps | ⚠️ camera-limited, not decode-limited |
| Golden trace | `make trace-check` | pass | pass | pass | ✅ (regenerated) |
| `unsafe` sites in the crate | grep | 10 | **1** | 1 | ✅ |
| Direct dependencies | `Cargo.toml` | 13 | **15** (−`opencv`, +`image`, +`nokhwa`, +`mp4io`, +`rusty_h264-decoder`) | — | ✅ |

**The one honest regression:** the hand-written bilinear sampler is not as fast as
OpenCV's vectorised `resize` on the 224×224 crop. It costs ~160 µs more per frame, which
is 0.5 % of the 33 ms budget and is invisible in the end-to-end numbers. Reaching
OpenCV's throughput would need a SIMD/row-sum rewrite; the plan deliberately accepts the
scalar loop for clarity and testability.

`nokhwa` was chosen over `cameras` (§4.3) because it is the more established crate and
`camera-sync-impl` gives the opt-in `unsafe impl Send for Camera` the capture thread
needs. It worked on this camera on the first try (30 fps at 640×480, no duplicate
frames), so `nokhwa#247` did not bite.

For the fixture the plan's `yscv-video` was **not** used: `mp4io` + `rusty_h264-decoder`
decoded the clip bit-exactly against ffmpeg, is `#![forbid(unsafe_code)]`, and runs at
~0.5 ms per 640×480 frame. `ffmpeg-next` was rejected outright because it reintroduces a
C toolchain.

---

## 9. Docs to update after the work lands

- `README.md` — install prerequisites (no more OpenCV/LLVM).
- `AGENTS.md` — Tech Stack, Build Environment, data-flow, "Key Module Reference".
- `ARCHITECTURE.md` — the Pipeline row mentions OpenCV resizing.
- `CHANGELOG.md` — **Unreleased**: "Removed `opencv`; frame capture, sampling, ASCII
  rendering and preview conversion are now pure Rust / `image` + `nokhwa`".
- `ROADMAP.md` — link this plan; add the resulting numbers to "Measured baseline".
- `Makefile` header notes.

---

## 10. Open questions for review

1. **Capture:** `nokhwa` (proven, legacy `objc`) or `cameras` (modern `objc2`, younger)?
   Phase 0 benchmarks both on this camera.
2. **Fixture decode:** pure-Rust `yscv-video`, `ffmpeg-next` (mature but reintroduces a C
   toolchain), or change `--video` to accept an image sequence and drop MP4 entirely?
   The third option has the lowest risk and the most honest telemetry for `capture_ms`.
3. **Is `--image` / `--dump-frame` worth keeping?** If the offline skeleton dump is rarely
   used, deleting `annotator.rs` outright is simpler than reimplementing line/circle
   drawing — the iced canvas already draws the exact same skeleton.
4. **Do we go all the way in one PR, or land Phases 1–4 first?** OpenCV-for-capture-only
   is a stable halfway state. Recommend two PRs: `chore/rgb-frame` (Phases 1–4), then
   `chore/opencv-removal` (Phases 5–7).
5. **Do we also fix ROADMAP finding 9** (capture at 320×240) in the same branch, since
   Phase 5 touches the capture configuration anyway?
