//! Hand tracking, in two backends.
//!
//! `HandTracker` is what the pipeline holds; it dispatches to either
//! `MediaPipeTracker` - the two-stage BlazeHand palm detector (192x192) plus
//! landmarker (224x224) - or `YoloPoseTracker`, one Ultralytics pose graph that
//! returns the hand box and the 21 keypoints in a single pass.
//!
//! Both produce the same `HandDetectionResult`, so the strategies, the dashboard and
//! the sinks do not care which one is running. Trackers only compute geometry; all
//! drawing lives in `crate::visualizer`.

use anyhow::{anyhow, Context, Result};
use opencv::prelude::*;
use std::time::Instant;

use ort::session::Session;
use ort::value::TensorRef;

use crate::core::HandDetectionResult;
use crate::core::Landmark;
use crate::core::StageTimings;
use crate::vision::reuse::ReuseCache;
use crate::vision::sampling::{sample_affine_to_tensor, Affine, StageInput, TensorLayout};
use crate::vision::yolo_pose::YoloPoseTracker;

const NUM_KEYPOINTS: usize = 21;
const DETECTOR_INPUT: usize = 192;
const LANDMARKER_INPUT: usize = 224;
/// Default detector confidence gate. A real hand scores ~0.9; an empty frame tops
/// out around 0.15, so 0.7 is a strict gate that rejects near-misses and
/// false positives. Overridable with `--score-threshold`.
pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.6;

/// How many consecutive frames without a detection before the detector window
/// stops following the last palm position and returns to the centred square.
const ROI_STALE_FRAMES: u32 = 10;
/// Smoothing factor of the palm anchor: 1.0 would track the palm exactly, 0.3
/// keeps the window steady against landmark jitter.
const ROI_ANCHOR_WEIGHT: f64 = 0.3;

/// The tracker the pipeline talks to, whichever backend the run asked for.
pub enum HandTracker {
    MediaPipe(Box<MediaPipeTracker>),
    YoloPose(Box<YoloPoseTracker>),
}

impl HandTracker {
    pub fn mediapipe(
        detector_path: &str,
        landmarker_path: &str,
        anchors_path: &str,
        score_threshold: f32,
        debug: bool,
        reuse_landmarks: bool,
        ort_threads: usize,
    ) -> Result<Self> {
        Ok(Self::MediaPipe(Box::new(MediaPipeTracker::new(
            detector_path,
            landmarker_path,
            anchors_path,
            score_threshold,
            debug,
            reuse_landmarks,
            ort_threads,
        )?)))
    }

    pub fn yolo_pose(
        pose_path: &str,
        input_size: usize,
        score_threshold: f32,
        debug: bool,
        reuse_landmarks: bool,
        ort_threads: usize,
    ) -> Result<Self> {
        Ok(Self::YoloPose(Box::new(YoloPoseTracker::new(
            pose_path,
            input_size,
            score_threshold,
            debug,
            reuse_landmarks,
            ort_threads,
        )?)))
    }

    pub fn process_frame(&mut self, frame: &Mat) -> Result<HandDetectionResult> {
        match self {
            Self::MediaPipe(tracker) => tracker.process_frame(frame),
            Self::YoloPose(tracker) => tracker.process_frame(frame),
        }
    }
}

/// The largest axis-aligned square the detector can take from the frame, slid so
/// that it contains `palm` (where the hand was last frame). Without history this
/// is exactly the centred square the pipeline used before.
///
/// This is what makes the outer keys playable: at 640x480 a centred square only
/// ever samples x = 80..560, so a hand pushed to the right edge was never seen by
/// the model at all.
fn detector_crop(palm: Option<(f64, f64)>, fw: f64, fh: f64) -> (f64, f64, f64) {
    let side = fw.min(fh);
    match palm {
        Some((px, py)) => (
            (px - side * 0.5).clamp(0.0, fw - side),
            (py - side * 0.5).clamp(0.0, fh - side),
            side,
        ),
        None => ((fw - side) * 0.5, (fh - side) * 0.5, side),
    }
}

/// Low-pass the palm anchor so a jittery detection does not shake the window.
fn smooth_palm(previous: Option<(f64, f64)>, next: (f64, f64)) -> (f64, f64) {
    match previous {
        Some((px, py)) => (
            px + ROI_ANCHOR_WEIGHT * (next.0 - px),
            py + ROI_ANCHOR_WEIGHT * (next.1 - py),
        ),
        None => next,
    }
}

/// Palm centre and the radius of the x2-expanded hand box, in frame pixels.
struct PalmBox {
    cx: f64,
    cy: f64,
    half: f64,
    score: f32,
}

pub struct MediaPipeTracker {
    pub detector_session: Session,
    pub landmarker_session: Session,
    /// Anchor centres in normalised detector space (0..1).
    pub anchors: Vec<(f32, f32)>,
    pub score_threshold: f32,
    pub debug: bool,
    /// Per-stage input buffers, allocated once here and refilled in place every
    /// frame (see `StageInput`).
    detector_input: StageInput,
    landmarker_input: StageInput,
    /// Palm centre the detector reported recently, smoothed. The detector's
    /// square window is slid to keep this point inside, so a hand pushed against
    /// the frame edge is still sampled instead of being cropped out of the model
    /// input (see `detector_crop`).
    roi_palm: Option<(f64, f64)>,
    roi_stale: u32,
    /// Keypoints reused while the palm is stationary (see `ReuseCache`).
    cache: ReuseCache,
}

impl MediaPipeTracker {
    pub fn new(
        detector_path: &str,
        landmarker_path: &str,
        anchors_path: &str,
        score_threshold: f32,
        debug: bool,
        reuse_landmarks: bool,
        ort_threads: usize,
    ) -> Result<Self> {
        // `--ort-threads 0` is ONNX Runtime's own default: one thread per core.
        // Measured live with the webcam open, the BlazeHand detector is ~40 % faster
        // using every core (9.7 ms p50) than pinned to one (13.7 ms), and the frame is
        // paced by the 33 ms camera read anyway - so the default is left alone and the
        // flag is an A/B knob for the jitter a full pool causes the UI and capture.
        let threads = ort_threads;
        let pin_error = |e| anyhow!("Failed to set ORT intra-op threads to {threads}: {e}");
        let detector_session = Session::builder()
            .with_context(|| format!("Failed to load detector {detector_path}"))?
            .with_intra_threads(threads)
            .map_err(pin_error)?
            .commit_from_file(detector_path)?;
        let landmarker_session = Session::builder()
            .with_context(|| format!("Failed to load landmarker {landmarker_path}"))?
            .with_intra_threads(threads)
            .map_err(pin_error)?
            .commit_from_file(landmarker_path)?;

        let anchors = load_anchors(anchors_path)?;
        if anchors.is_empty() {
            return Err(anyhow!("anchors file {anchors_path} is empty"));
        }

        if debug {
            println!(
                "[Tracker] Loaded detector: {detector_path}, landmarker: {landmarker_path}, {} anchors, ORT intra-op threads {}.",
                anchors.len(),
                if threads == 0 {
                    "default (all cores)"
                } else {
                    "pinned"
                }
            );
        }

        Ok(Self {
            detector_session,
            landmarker_session,
            anchors,
            score_threshold: score_threshold.clamp(0.0, 1.0),
            debug,
            detector_input: StageInput::new(DETECTOR_INPUT, DETECTOR_INPUT),
            landmarker_input: StageInput::new(LANDMARKER_INPUT, LANDMARKER_INPUT),
            roi_palm: None,
            roi_stale: 0,
            cache: ReuseCache::new(reuse_landmarks),
        })
    }

    /// Detector stage: returns the palm centre and the x2-expanded hand radius,
    /// in frame pixels.
    fn run_detector(&mut self, frame: &Mat) -> Result<Option<PalmBox>> {
        let fw = frame.cols() as f64;
        let fh = frame.rows() as f64;

        // The detector wants a square input. The window follows the palm so a
        // hand at the frame edge is still fully sampled instead of being cut out
        // of the model input.
        let (tx, ty, side) = detector_crop(self.roi_palm, fw, fh);
        let k = side / DETECTOR_INPUT as f64; // detector pixel -> frame pixel

        if self.debug {
            let palm = self.roi_palm;
            eprintln!("[Tracker] detector window x={tx:.0} y={ty:.0} side={side:.0} palm={palm:?}");
        }

        let map = sample_affine_to_tensor(
            frame,
            Affine::scale_translate(k, tx, ty),
            DETECTOR_INPUT,
            DETECTOR_INPUT,
            &mut self.detector_input,
            TensorLayout::Nhwc,
            0.0,
        )?;

        // `TensorRef` wraps the reused buffer in place - ort does not copy it.
        let input = TensorRef::from_array_view((
            [1usize, DETECTOR_INPUT, DETECTOR_INPUT, 3],
            self.detector_input.data.as_slice(),
        ))?;
        // `collect()` materialises `SessionOutputs`, which borrows the session, so
        // the extracted slices stay valid after the loop.
        let outputs: Vec<_> = self
            .detector_session
            .run(vec![("input_1", input)])?
            .into_iter()
            .collect();

        let mut boxes: Option<&[f32]> = None;
        let mut scores: Option<&[f32]> = None;

        for entry in &outputs {
            let (_, data) = entry.1.try_extract_tensor::<f32>()?;
            if data.len() == self.anchors.len() {
                scores = Some(data);
            } else if data.len() >= self.anchors.len() * 4 {
                boxes = Some(data);
            }
        }

        let scores = scores.ok_or_else(|| anyhow!("detector did not return a score tensor"))?;
        let boxes = boxes.ok_or_else(|| anyhow!("detector did not return a box tensor"))?;

        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

        let mut best_idx = 0usize;
        let mut best_conf = f32::MIN;
        for (i, &s) in scores.iter().enumerate() {
            let conf = sigmoid(s);
            if conf > best_conf {
                best_conf = conf;
                best_idx = i;
            }
        }

        if self.debug {
            eprintln!(
                "[Tracker] best anchor {best_idx} conf={best_conf:.3} (threshold {})",
                self.score_threshold
            );
        }

        if best_conf < self.score_threshold {
            self.note_miss();
            return Ok(None);
        }

        // Box is [cx, cy, w, h] in detector pixels, offsets relative to the anchor,
        // followed by 7 palm keypoint offsets (18 values per anchor).
        let base = best_idx * 18;
        let (ax, ay) = self.anchors[best_idx];
        let cx = ax as f64 * DETECTOR_INPUT as f64 + boxes[base] as f64;
        let cy = ay as f64 * DETECTOR_INPUT as f64 + boxes[base + 1] as f64;
        let bw = boxes[base + 2] as f64;
        let bh = boxes[base + 3] as f64;

        // Back to frame pixels using the mapping that was actually applied.
        let frame_cx = map.frame_x(cx, cy);
        let frame_cy = map.frame_y(cx, cy);
        let half = bw.max(bh) * 2.0 * 0.5 * map.m00; // expand to cover the whole hand

        if half <= 0.0 || frame_cx.is_nan() || frame_cy.is_nan() {
            self.note_miss();
            return Ok(None);
        }

        // Remember where the hand is so the next frame's window contains it.
        self.roi_palm = Some(smooth_palm(self.roi_palm, (frame_cx, frame_cy)));
        self.roi_stale = 0;

        Ok(Some(PalmBox {
            cx: frame_cx,
            cy: frame_cy,
            half,
            score: best_conf,
        }))
    }

    /// A missed frame keeps the window where it was; after enough misses the
    /// anchor is dropped so the search starts from the centred square again.
    fn note_miss(&mut self) {
        self.roi_stale += 1;
        if self.roi_stale >= ROI_STALE_FRAMES {
            self.roi_palm = None;
            self.roi_stale = 0;
        }
    }

    /// Landmarker stage: crop a square around the palm into 224x224 and run it.
    fn run_landmarks(&mut self, frame: &Mat, cx: f64, cy: f64, half: f64) -> Result<Vec<Landmark>> {
        // The box the detector produced is the palm expanded x2. Cropping a square
        // slightly larger than it, centred on the *true* palm centre, keeps the
        // fingers inside the crop even when the hand is clipped by the frame edge:
        // warpAffine pads the off-frame half with black instead of sliding the crop
        // left, which is what used to happen with a clamped box.
        let side = half * 2.0 * 1.2;
        let s = side / LANDMARKER_INPUT as f64; // crop pixel -> frame pixel

        let map = sample_affine_to_tensor(
            frame,
            Affine::scale_translate(s, cx - side * 0.5, cy - side * 0.5),
            LANDMARKER_INPUT,
            LANDMARKER_INPUT,
            &mut self.landmarker_input,
            TensorLayout::Nhwc,
            0.0,
        )?;

        let input = TensorRef::from_array_view((
            [1usize, LANDMARKER_INPUT, LANDMARKER_INPUT, 3],
            self.landmarker_input.data.as_slice(),
        ))?;
        let outputs: Vec<_> = self
            .landmarker_session
            .run(vec![("input_1", input)])?
            .into_iter()
            .collect();

        let mut raw: Option<&[f32]> = None;
        for entry in &outputs {
            let (_, data) = entry.1.try_extract_tensor::<f32>()?;
            if data.len() == NUM_KEYPOINTS * 3 {
                raw = Some(data);
                break;
            }
        }

        let raw = raw.ok_or_else(|| anyhow!("landmarker returned no keypoints"))?;

        let mut landmarks = Vec::with_capacity(NUM_KEYPOINTS);
        for i in 0..NUM_KEYPOINTS {
            let u = raw[3 * i] as f64; // pixel inside the 224x224 crop
            let v = raw[3 * i + 1] as f64;
            let z = raw[3 * i + 2];
            landmarks.push(Landmark {
                x: map.frame_x(u, v) as f32,
                y: map.frame_y(u, v) as f32,
                z,
            });
        }

        Ok(landmarks)
    }

    pub fn process_frame(&mut self, frame: &Mat) -> Result<HandDetectionResult> {
        let mut result = HandDetectionResult {
            hand_landmarks: None,
            r#box: None,
            frame_size: (frame.cols() as usize, frame.rows() as usize),
            score: 0.0,
            timings: StageTimings::default(),
        };

        let detector_start = Instant::now();
        let detection = self.run_detector(frame)?;
        result.timings.detector_us = detector_start.elapsed().as_secs_f64() * 1e6;

        if let Some(detection) = detection {
            // The palm box is reported unclamped: a constant 2 x half square keeps
            // the strategy's spread normalisation stable even when the hand is
            // clipped by the edge of the view (the canvas/ OpenCV clip the overlay
            // when it is drawn).
            result.r#box = Some((
                (detection.cx - detection.half) as f32,
                (detection.cy - detection.half) as f32,
                (2.0 * detection.half) as f32,
                (2.0 * detection.half) as f32,
            ));
            result.score = detection.score;

            let palm = (detection.cx, detection.cy, detection.half);
            if let Some(landmarks) = self.cache.reuse(palm) {
                // Same palm, same crop: the keypoints from the previous run are
                // still valid, so the 224x224 model is not executed this frame.
                result.hand_landmarks = Some(landmarks);
                result.timings.landmarker_us = 0.0;
                if self.debug {
                    eprintln!(
                        "[Tracker] palm steady, reusing landmarks ({} frame{} in a row)",
                        self.cache.streak(),
                        if self.cache.streak() > 1 { "s" } else { "" }
                    );
                }
            } else {
                let landmarker_start = Instant::now();
                match self.run_landmarks(frame, detection.cx, detection.cy, detection.half) {
                    Ok(landmarks) => {
                        result.timings.landmarker_us =
                            landmarker_start.elapsed().as_secs_f64() * 1e6;
                        if self.debug {
                            let (mut minx, mut miny, mut maxx, mut maxy) =
                                (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                            for lm in &landmarks {
                                minx = minx.min(lm.x);
                                maxx = maxx.max(lm.x);
                                miny = miny.min(lm.y);
                                maxy = maxy.max(lm.y);
                            }
                            eprintln!(
                                "[Tracker] landmarks bbox x={minx:.1}..{maxx:.1} y={miny:.1}..{maxy:.1}"
                            );
                            for (i, lm) in landmarks.iter().enumerate() {
                                eprintln!(
                                    "[Tracker] lm[{i}] = ({:.1}, {:.1}, {:.3})",
                                    lm.x, lm.y, lm.z
                                );
                            }
                        }
                        self.cache.refresh(palm, &landmarks);
                        result.hand_landmarks = Some(landmarks);
                    }
                    Err(e) => {
                        if self.debug {
                            eprintln!("[Tracker] landmarker stage skipped: {e}");
                        }
                        self.cache.clear();
                    }
                }
            }
        } else {
            // No hand: the cache is worthless and must not be reused on the next
            // sighting.
            self.cache.clear();
        }

        Ok(result)
    }
}

fn load_anchors(path: &str) -> Result<Vec<(f32, f32)>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read anchors file {path}"))?;

    let mut anchors = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        let x = parts[0]
            .parse::<f32>()
            .with_context(|| format!("bad anchor x in {line}"))?;
        let y = parts[1]
            .parse::<f32>()
            .with_context(|| format!("bad anchor y in {line}"))?;
        anchors.push((x, y));
    }

    Ok(anchors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core;

    /// 2x2 BGR frame: (10,20,30) (40,50,60) / (70,80,90) (100,110,120)
    const TEST_PIXELS: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];

    fn test_frame() -> Mat {
        let mut frame =
            core::Mat::new_rows_cols_with_default(2, 2, core::CV_8UC3, core::Scalar::all(0.0))
                .unwrap();
        for y in 0..2 {
            let row = frame.ptr_mut(y).unwrap();
            unsafe {
                for x in 0..2 {
                    let src = &TEST_PIXELS[(y as usize * 2 + x) * 3..];
                    *row.add(x * 3) = src[0];
                    *row.add(x * 3 + 1) = src[1];
                    *row.add(x * 3 + 2) = src[2];
                }
            }
        }
        frame
    }

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    fn resample(frame: &Mat, map: Affine, out_w: usize, out_h: usize) -> (Affine, Vec<f32>) {
        let mut stage = StageInput::new(out_w, out_h);
        let map = sample_affine_to_tensor(
            frame,
            map,
            out_w,
            out_h,
            &mut stage,
            TensorLayout::Nhwc,
            0.0,
        )
        .unwrap();
        (map, stage.data)
    }

    /// Channel `c` of output pixel (oy, ox) in a (out_h, out_w, 3) buffer.
    fn px(data: &[f32], out_w: usize, oy: usize, ox: usize, c: usize) -> f32 {
        data[(oy * out_w + ox) * 3 + c]
    }

    #[test]
    fn identity_crop_is_normalised_rgb() {
        let frame = test_frame();
        let (map, data) = resample(&frame, Affine::scale_translate(1.0, 0.0, 0.0), 2, 2);

        assert_eq!(data.len(), 2 * 2 * 3);
        assert!(near(px(&data, 2, 0, 0, 0), 30.0 / 255.0)); // R of BGR(10,20,30)
        assert!(near(px(&data, 2, 0, 0, 1), 20.0 / 255.0));
        assert!(near(px(&data, 2, 0, 0, 2), 10.0 / 255.0));
        assert!(near(px(&data, 2, 1, 1, 1), 110.0 / 255.0)); // G of the bottom-right pixel
        assert!((map.m00 - 1.0).abs() < 1e-9 && map.tx == 0.0 && map.ty == 0.0);
    }

    #[test]
    fn rounded_crop_reports_the_effective_mapping() {
        let frame =
            core::Mat::new_rows_cols_with_default(4, 4, core::CV_8UC3, core::Scalar::all(0.0))
                .unwrap();
        let (map, _) = resample(&frame, Affine::scale_translate(4.0 / 3.0, 0.0, 0.0), 3, 3);

        assert!((map.m00 - 4.0 / 3.0).abs() < 1e-9);
        assert!((map.m11 - 4.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn crop_running_off_the_frame_is_padded_with_black() {
        let frame = test_frame();
        // 3x2 output starting one pixel left of the frame: only the warp path can do this.
        let (map, data) = resample(&frame, Affine::scale_translate(1.0, -1.0, 0.0), 3, 2);

        assert_eq!(data.len(), 2 * 3 * 3);
        assert!(near(px(&data, 3, 0, 0, 0), 0.0)); // column -1 is outside the frame
        assert!(near(px(&data, 3, 1, 1, 0), 90.0 / 255.0)); // frame pixel (1,0) -> R
        assert!(near(px(&data, 3, 1, 2, 0), 120.0 / 255.0)); // frame pixel (1,1) -> R
        assert!((map.tx + 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_detector_window_follows_the_palm() {
        // 640x480: the square window is 480 wide, so a centred crop only ever
        // sampled x = 80..560 - the outer 80 px on each side were invisible.
        let centred = detector_crop(None, 640.0, 480.0);
        assert_eq!((centred.0, centred.1, centred.2), (80.0, 0.0, 480.0));

        // Hand hard against the right edge: the window slides to contain it.
        let right = detector_crop(Some((620.0, 240.0)), 640.0, 480.0);
        assert_eq!((right.0, right.1), (160.0, 0.0));
        assert!(right.0 + right.2 >= 620.0);

        // ...and the left edge.
        let left = detector_crop(Some((20.0, 240.0)), 640.0, 480.0);
        assert_eq!((left.0, left.1), (0.0, 0.0));
        assert!(left.0 <= 20.0 && left.0 + left.2 >= 20.0);

        // A palm in the middle still gets the centred square.
        let mid = detector_crop(Some((320.0, 240.0)), 640.0, 480.0);
        assert_eq!((mid.0, mid.1), (80.0, 0.0));
    }

    #[test]
    fn the_anchor_is_smoothed_and_never_overshoots() {
        assert_eq!(smooth_palm(None, (100.0, 50.0)), (100.0, 50.0));

        let moved = smooth_palm(Some((0.0, 0.0)), (100.0, 0.0));
        assert!((moved.0 - 30.0).abs() < 1e-9);

        // Converges on the palm without ever passing it.
        let mut anchor = (0.0, 0.0);
        for _ in 0..50 {
            anchor = smooth_palm(Some(anchor), (100.0, 0.0));
        }
        assert!(anchor.0 < 100.0 && anchor.0 > 99.0);
    }

    #[test]
    fn stage_buffers_are_reused_across_frames() {
        let frame = test_frame();
        let mut stage = StageInput::new(2, 2);
        sample_affine_to_tensor(
            &frame,
            Affine::scale_translate(1.0, 0.0, 0.0),
            2,
            2,
            &mut stage,
            TensorLayout::Nhwc,
            0.0,
        )
        .unwrap();
        let ptr = stage.data.as_ptr();
        let warped_ptr = stage.warped.data();
        sample_affine_to_tensor(
            &frame,
            Affine::scale_translate(1.0, 0.0, 0.0),
            2,
            2,
            &mut stage,
            TensorLayout::Nhwc,
            0.0,
        )
        .unwrap();

        // Neither the f32 buffer nor the OpenCV scratch matrix is reallocated.
        assert_eq!(ptr, stage.data.as_ptr());
        assert_eq!(warped_ptr, stage.warped.data());
    }

    use std::time::{Duration, Instant};

    /// The old per-pixel sampler, kept only so the benchmark has something to compare
    /// against. Run with `cargo test --release -- --include-ignored bench_sampling`.
    fn legacy_sample(frame: &Mat, map: Affine, out_w: usize, out_h: usize) -> Vec<f32> {
        let fw = frame.cols() as f64;
        let fh = frame.rows() as f64;
        let mut data = vec![0.0f32; out_h * out_w * 3];
        unsafe {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let sx = map.m00 * ox as f64 + map.m01 * oy as f64 + map.tx;
                    let sy = map.m10 * ox as f64 + map.m11 * oy as f64 + map.ty;
                    if sx < 0.0 || sy < 0.0 || sx >= fw || sy >= fh {
                        continue;
                    }
                    let x0 = sx.floor() as usize;
                    let y0 = sy.floor() as usize;
                    let x1 = (x0 + 1).min(frame.cols() as usize - 1);
                    let y1 = (y0 + 1).min(frame.rows() as usize - 1);
                    let fx = sx - x0 as f64;
                    let fy = sy - y0 as f64;
                    let row0 = frame.ptr(y0 as i32).unwrap();
                    let row1 = frame.ptr(y1 as i32).unwrap();
                    let px = |p: *const u8, c: usize| -> f64 { *p.add(c) as f64 };
                    let c00 = row0.add(x0 * 3);
                    let c01 = row0.add(x1 * 3);
                    let c10 = row1.add(x0 * 3);
                    let c11 = row1.add(x1 * 3);
                    let b = (px(c00, 0) * (1.0 - fx) + px(c01, 0) * fx) * (1.0 - fy)
                        + (px(c10, 0) * (1.0 - fx) + px(c11, 0) * fx) * fy;
                    let g = (px(c00, 1) * (1.0 - fx) + px(c01, 1) * fx) * (1.0 - fy)
                        + (px(c10, 1) * (1.0 - fx) + px(c11, 1) * fx) * fy;
                    let r = (px(c00, 2) * (1.0 - fx) + px(c01, 2) * fx) * (1.0 - fy)
                        + (px(c10, 2) * (1.0 - fx) + px(c11, 2) * fx) * fy;
                    let i = (oy * out_w + ox) * 3;
                    data[i] = (r / 255.0) as f32;
                    data[i + 1] = (g / 255.0) as f32;
                    data[i + 2] = (b / 255.0) as f32;
                }
            }
        }
        data
    }

    /// One stage, sampled N times: either with a buffer reused across calls (what
    /// the tracker does) or allocated per call (what it used to do).
    fn time_stage(frame: &Mat, affine: Affine, size: usize, fresh_buffer: bool) -> Duration {
        const N: u32 = 300;
        let mut reused = StageInput::new(size, size);
        let start = Instant::now();
        for _ in 0..N {
            let map = if fresh_buffer {
                let mut stage = StageInput::new(size, size);
                sample_affine_to_tensor(
                    frame,
                    affine,
                    size,
                    size,
                    &mut stage,
                    TensorLayout::Nhwc,
                    0.0,
                )
                .unwrap()
            } else {
                sample_affine_to_tensor(
                    frame,
                    affine,
                    size,
                    size,
                    &mut reused,
                    TensorLayout::Nhwc,
                    0.0,
                )
                .unwrap()
            };
            std::hint::black_box(map);
        }
        start.elapsed() / N
    }

    fn time_legacy(frame: &Mat, affine: Affine, size: usize) -> Duration {
        const N: u32 = 300;
        let start = Instant::now();
        for _ in 0..N {
            std::hint::black_box(legacy_sample(frame, affine, size, size));
        }
        start.elapsed() / N
    }

    #[test]
    #[ignore]
    fn bench_sampling() {
        let frame = core::Mat::new_rows_cols_with_default(
            720,
            1280,
            core::CV_8UC3,
            core::Scalar::all(127.0),
        )
        .unwrap();
        // The affine each stage really applies on a 640x480 frame.
        let stages = [
            ("detector", 192, Affine::scale_translate(3.75, 280.0, 0.0)),
            (
                "landmarker",
                224,
                Affine::scale_translate(2.7, 200.0, 100.0),
            ),
        ];
        for (name, size, affine) in stages {
            println!(
                "{name} {size}x{size}: reused {:?}  fresh {:?}  legacy {:?}",
                time_stage(&frame, affine, size, false),
                time_stage(&frame, affine, size, true),
                time_legacy(&frame, affine, size),
            );
        }
    }

    /// The only test that runs the real ONNX graphs. It needs the model assets and
    /// the recorded clip, so it is `#[ignore]`d rather than part of `make verify`:
    /// `cargo test --release -- --include-ignored tracker_finds_the_hand_in_the_clip`.
    #[test]
    #[ignore]
    fn tracker_finds_the_hand_in_the_clip() {
        use crate::vision::video::VideoSource;

        const CLIP: &str = "fixtures/hand_clip.mp4";
        let mut source =
            VideoSource::new(CLIP, false).expect("record a clip with `make record` first");
        let mut tracker = HandTracker::mediapipe(
            "assets/hand_detector.onnx",
            "assets/hand_landmarks_detector.onnx",
            "assets/anchors.csv",
            0.6,
            false,
            true,
            0,
        )
        .expect("the model assets must be present");

        let mut seen = 0;
        let mut frames = 0;
        let mut landmark_counts = Vec::new();
        let mut boxes = Vec::new();
        for _ in 0..120 {
            let frame = match source.read() {
                Ok(frame) => frame,
                Err(_) => break,
            };
            frames += 1;
            let result = tracker.process_frame(&frame).unwrap();
            if let Some(landmarks) = &result.hand_landmarks {
                seen += 1;
                landmark_counts.push(landmarks.len());
                if let Some(b) = result.r#box {
                    boxes.push((b, result.frame_size));
                }
            }
        }

        assert!(frames > 0, "the clip should yield frames");
        // A hand is in front of the camera for most of the clip; a detector that
        // finds it in a minority of frames is not working, however it scores.
        assert!(seen * 2 > frames, "detected {seen} of {frames} frames");
        for count in &landmark_counts {
            assert_eq!(*count, 21, "a detection must carry all 21 landmarks");
        }
        // A box is (x, y, w, h) around the palm, reported unclamped so it can leave
        // the frame when the hand is clipped by the edge. It must still be a real
        // square whose centre is a plausible palm.
        for ((x, y, w, h), (fw, fh)) in &boxes {
            let (x, y, w, h) = (*x, *y, *w, *h);
            let (fw, fh) = (*fw as f32, *fh as f32);
            assert!(
                w > 0.0 && h > 0.0,
                "a box must have positive extent, got {w}x{h}"
            );
            assert!(near(w, h), "the hand box is square, got {w}x{h}");
            let cx = x + w / 2.0;
            let cy = y + h / 2.0;
            assert!(cx > -w && cx < fw + w, "palm centre {cx} is off the frame");
            assert!(cy > -h && cy < fh + h, "palm centre {cy} is off the frame");
        }
        println!(
            "clip: {frames} frames, {seen} detections, first box {:?}",
            boxes.first()
        );
    }
}
