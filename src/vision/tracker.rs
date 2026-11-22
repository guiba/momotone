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
use std::time::Instant;

use ort::session::Session;
use ort::value::TensorRef;

use crate::core::frame::Frame;
use crate::core::HandDetectionResult;
use crate::core::Landmark;
use crate::core::StageTimings;
use crate::vision::reuse::ReuseCache;
use crate::vision::sampling::{
    letterbox_map, sample_affine_to_tensor, Affine, StageInput, TensorLayout,
};
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

/// Which square the palm detector sees.
///
/// `Anchored` is today's behaviour: a `min(W, H)` square slid to contain the palm
/// seen last frame. `FullFrame` letterboxes the whole image into the same 192x192
/// buffer, so every anchor in the frame is scored - the only way to see a second
/// hand without a second detector run. `Both` keeps the anchored detection as the
/// behaviour and scores the full frame alongside it, which is how the resolution
/// trade-off between the two is measured before the choice is made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DetectorWindow {
    Anchored,
    FullFrame,
    Both,
}

/// Two palm centres this far apart, in units of the larger palm radius, count as
/// two different hands.
///
/// Measured, not guessed. `--detector-window both` sweeps `SEPARATIONS` over the
/// two-hand clip. On `fixtures/2_hands_640.mp4` the whole-frame window reports 1.90
/// hands per frame at 0.25, 0.5 and 1.0 and 1.34 at 2.0; the anchored crop reports 1.46,
/// 1.46, 1.46 and 1.18. So one hand's duplicate anchors are merged at every gate up to
/// 1.0, and genuinely separate hands only start merging at 2.0. Measured palm-centre
/// separation in that clip is 1.3-2.6 palm radii. 1.0 is the widest gate that still
/// separates the hands while leaving margin before they merge.
const HAND_SEPARATION: f64 = 1.0;
/// The separation values the probe sweeps, from "clearly one hand" to "clearly two".
const SEPARATIONS: [f64; 4] = [0.25, 0.5, 1.0, 2.0];
/// How many hands the detector is allowed to report. The pipeline uses one pitch
/// hand plus one modulator.
const MAX_HANDS: usize = 2;

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
        reuse_landmarks: bool,
        ort_threads: usize,
        window: DetectorWindow,
    ) -> Result<Self> {
        Ok(Self::MediaPipe(Box::new(MediaPipeTracker::new(
            detector_path,
            landmarker_path,
            anchors_path,
            score_threshold,
            reuse_landmarks,
            ort_threads,
            window,
        )?)))
    }

    pub fn yolo_pose(
        pose_path: &str,
        input_size: usize,
        score_threshold: f32,
        reuse_landmarks: bool,
        ort_threads: usize,
    ) -> Result<Self> {
        Ok(Self::YoloPose(Box::new(YoloPoseTracker::new(
            pose_path,
            input_size,
            score_threshold,
            reuse_landmarks,
            ort_threads,
        )?)))
    }

    pub fn process_frame(&mut self, frame: &Frame) -> Result<HandDetectionResult> {
        match self {
            Self::MediaPipe(tracker) => tracker.process_frame(frame),
            Self::YoloPose(tracker) => tracker.process_frame(frame),
        }
    }

    /// What the `DetectorWindow::Both` run measured, or `None` if this run was not a
    /// probe. Printed at the end of a `--detector-window both` run so the choice
    /// between the two windows is a number rather than an impression.
    ///
    /// `hands` is the mean number of distinct hands per frame at each separation in
    /// `SEPARATIONS`; `two+` is how many frames found more than one. On a one-hand
    /// clip a non-zero `two+` is NMS splitting one hand into two - the floor under
    /// `HAND_SEPARATION`. On the two-hand clip it is the ceiling: the widest gate that
    /// still sees both hands.
    pub fn probe_report(&self) -> Option<String> {
        let Self::MediaPipe(tracker) = self else {
            return None;
        };
        if tracker.probe_crop.frames == 0 {
            return None;
        }
        let gates = SEPARATIONS
            .iter()
            .map(|gate| format!("{gate}"))
            .collect::<Vec<_>>()
            .join(" ");
        let row = |name: &str, probe: &WindowProbe| {
            let over = probe.frames.max(1) as f64;
            let hands = probe
                .hands
                .iter()
                .map(|total| format!("{:.2}", *total as f64 / over))
                .collect::<Vec<_>>()
                .join(" ");
            let two = probe
                .multi
                .iter()
                .map(|total| total.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            let score = format!("{:.3}", probe.conf as f64 / (1000.0 * over));
            let us = probe.us as f64 / over;
            format!(
                "  {name:<6} {}/{} frames with a hand  score {score}  {us:.0} us/frame  hands {hands}  two+ {two}",
                probe.hits,
                probe.frames,
            )
        };
        Some(format!(
            "detector windows over {} frames, hands per frame at separations [{}]:\n{}\n{}",
            tracker.probe_crop.frames,
            gates,
            row("crop", &tracker.probe_crop),
            row("frame", &tracker.probe_full),
        ))
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
#[derive(Debug, Clone, Copy)]
struct PalmBox {
    cx: f64,
    cy: f64,
    half: f64,
    score: f32,
}

/// What one detector pass over one window reports.
#[derive(Debug)]
struct WindowReport {
    /// The highest-scoring anchor's palm box, whether or not it clears the gate.
    best: PalmBox,
    /// Anchors that cleared the gate before NMS.
    above: usize,
    /// Distinct hands after NMS at `HAND_SEPARATION`, best first.
    hands: Vec<PalmBox>,
    /// How many hands the same candidates produce at each swept separation.
    hands_at: [usize; SEPARATIONS.len()],
}

/// Convert one anchor's detector output into a palm box in frame pixels.
///
/// `boxes` is `[cx, cy, w, h]` in detector pixels, offsets relative to the anchor,
/// followed by 7 palm keypoint offsets (18 values per anchor). The `x2` expansion
/// turns the palm box into a box that covers the whole hand.
fn anchor_palm_box(anchor: (f32, f32), boxes: &[f32], base: usize, map: &Affine) -> PalmBox {
    let cx = anchor.0 as f64 * DETECTOR_INPUT as f64 + boxes[base] as f64;
    let cy = anchor.1 as f64 * DETECTOR_INPUT as f64 + boxes[base + 1] as f64;
    let bw = boxes[base + 2] as f64;
    let bh = boxes[base + 3] as f64;
    PalmBox {
        cx: map.frame_x(cx, cy),
        cy: map.frame_y(cx, cy),
        half: bw.max(bh) * 2.0 * 0.5 * map.m00,
        score: 0.0,
    }
}

/// Anchors whose palm boxes are far enough apart to be different hands.
///
/// Greedy, best-first: accept a candidate only when its centre is more than
/// `separation` x the larger radius away from every accepted palm. Anchors that
/// belong to the same hand predict centres inside one palm box, so they merge
/// without any special casing. Two palms that genuinely overlap merge too - for a
/// modulator that is the right answer, since there is no way to tell them apart
/// from the box alone.
fn non_overlapping(candidates: &[PalmBox], separation: f64, max: usize) -> Vec<PalmBox> {
    let mut hands: Vec<PalmBox> = Vec::with_capacity(max);
    for candidate in candidates {
        if hands.len() == max {
            break;
        }
        let clear = hands.iter().all(|hand| {
            let dx = candidate.cx - hand.cx;
            let dy = candidate.cy - hand.cy;
            (dx * dx + dy * dy).sqrt() > separation * candidate.half.max(hand.half)
        });
        if clear {
            hands.push(*candidate);
        }
    }
    hands
}

pub struct MediaPipeTracker {
    pub detector_session: Session,
    pub landmarker_session: Session,
    /// Anchor centres in normalised detector space (0..1).
    pub anchors: Vec<(f32, f32)>,
    pub score_threshold: f32,
    /// Per-stage input buffers, allocated once here and refilled in place every
    /// frame (see `StageInput`).
    detector_input: StageInput,
    landmarker_input: StageInput,
    /// Which square the palm detector sees. See `DetectorWindow`.
    window: DetectorWindow,
    /// Palm centre the detector reported recently, smoothed. The detector's
    /// square window is slid to keep this point inside, so a hand pushed against
    /// the frame edge is still sampled instead of being cropped out of the model
    /// input (see `detector_crop`).
    roi_palm: Option<(f64, f64)>,
    roi_stale: u32,
    /// Keypoints reused while the palm is stationary (see `ReuseCache`).
    cache: ReuseCache,
    /// Palm boxes that cleared the gate this frame, reused across frames so the
    /// detector stage does not allocate.
    candidates: Vec<PalmBox>,
    /// One `WindowProbe` per window the `Both` run scored. These are the numbers that
    /// decide whether the whole-frame window is good enough to build two-hand
    /// tracking on, and which `HAND_SEPARATION` to pick.
    probe_crop: WindowProbe,
    probe_full: WindowProbe,
}

/// Tally of one detector window over a probe run.
#[derive(Debug, Default)]
struct WindowProbe {
    frames: u64,
    /// Frames in which at least one palm cleared the gate.
    hits: u64,
    /// Sum of the best palm score, so the report can show the mean.
    conf: u64,
    /// Total anchors above the gate across all frames.
    anchors: u64,
    /// For each separation in `SEPARATIONS`: the total number of hands found, and the
    /// number of frames that found more than one. The first says how many hands the
    /// window sees; the second is the false-positive count - a frame with two "hands"
    /// where only one exists is NMS failing to merge one hand's duplicate anchors.
    hands: [u64; SEPARATIONS.len()],
    multi: [u64; SEPARATIONS.len()],
    us: u64,
}

impl WindowProbe {
    fn observe(&mut self, report: &WindowReport, us: u64) {
        self.frames += 1;
        self.hits += report.hands.first().map_or(0, |_| 1);
        self.conf += (report.best.score * 1000.0) as u64;
        self.anchors += report.above as u64;
        self.us += us;
        for (index, count) in report.hands_at.iter().enumerate() {
            self.hands[index] += *count as u64;
            self.multi[index] += (*count > 1) as u64;
        }
    }
}

impl MediaPipeTracker {
    pub fn new(
        detector_path: &str,
        landmarker_path: &str,
        anchors_path: &str,
        score_threshold: f32,
        reuse_landmarks: bool,
        ort_threads: usize,
        window: DetectorWindow,
    ) -> Result<Self> {
        // `--ort-threads 0` is ONNX Runtime's own default: one thread per core.
        // Measured live with the webcam open, the BlazeHand detector is ~40 % faster
        // using every core (9.7 ms p50) than pinned to one (13.7 ms), and the frame is
        // paced by the 33 ms camera read anyway - so the default is left alone and the
        // flag is an A/B knob for the jitter a full pool causes the UI and capture.
        let threads = ort_threads;
        let pin_error = |e| anyhow!("Failed to set ORT intra-op threads to {threads}: {e}");
        let env = ort::init().build()?;
        let detector_session = Session::builder(&env)
            .with_context(|| format!("Failed to load detector {detector_path}"))?
            .with_intra_threads(threads)
            .map_err(pin_error)?
            .commit_from_file(detector_path)?;
        let landmarker_session = Session::builder(&env)
            .with_context(|| format!("Failed to load landmarker {landmarker_path}"))?
            .with_intra_threads(threads)
            .map_err(pin_error)?
            .commit_from_file(landmarker_path)?;

        let anchors = load_anchors(anchors_path)?;
        if anchors.is_empty() {
            return Err(anyhow!("anchors file {anchors_path} is empty"));
        }

        tracing::info!(
            "loaded detector {detector_path}, landmarker {landmarker_path}, {} anchors, ORT intra-op threads {}",
            anchors.len(),
            if threads == 0 {
                "default (all cores)"
            } else {
                "pinned"
            }
        );

        Ok(Self {
            detector_session,
            landmarker_session,
            anchors,
            score_threshold: score_threshold.clamp(0.0, 1.0),
            detector_input: StageInput::new(DETECTOR_INPUT, DETECTOR_INPUT),
            landmarker_input: StageInput::new(LANDMARKER_INPUT, LANDMARKER_INPUT),
            window,
            roi_palm: None,
            roi_stale: 0,
            cache: ReuseCache::new(reuse_landmarks),
            candidates: Vec::new(),
            probe_crop: WindowProbe::default(),
            probe_full: WindowProbe::default(),
        })
    }

    /// Detector stage: runs the palm graph over one window and reports every hand
    /// it saw in that window.
    ///
    /// `map` is the mapping from the 192x192 model input back to frame pixels; it is
    /// either the palm-anchored square (`detector_crop`) or the letterboxed whole
    /// frame (`letterbox_map`). Everything else is the same for both: the graph
    /// scores all 2016 anchors either way, so seeing the whole frame costs nothing
    /// beyond the resolution the square crop gives a hand.
    fn run_detector_window(&mut self, frame: &Frame, map: Affine) -> Result<WindowReport> {
        let map = sample_affine_to_tensor(
            frame,
            map,
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
        // Anchors that clear the gate, best first. The buffer is a tracker field so
        // this loop does not allocate; usually only a handful of the 2016 anchors
        // survive, since the detector puts a long tail of near-duplicate anchors on
        // one hand and those are exactly what the NMS has to merge.
        self.candidates.clear();

        for (i, &s) in scores.iter().enumerate() {
            let conf = sigmoid(s);
            if conf > best_conf {
                best_conf = conf;
                best_idx = i;
            }
            if conf < self.score_threshold {
                continue;
            }
            let palm = anchor_palm_box(self.anchors[i], boxes, i * 18, &map);
            if palm.half > 0.0 && palm.cx.is_finite() && palm.cy.is_finite() {
                self.candidates.push(PalmBox {
                    score: conf,
                    ..palm
                });
            }
        }

        self.candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let hands = non_overlapping(&self.candidates, HAND_SEPARATION, MAX_HANDS);
        // The separation sweep: how many hands the same candidates produce under each
        // gate. Cheap - the candidate list is a handful of anchors - and it is what
        // turns `HAND_SEPARATION` from a guess into a measured number.
        let mut hands_at = [0usize; SEPARATIONS.len()];
        for (index, separation) in SEPARATIONS.iter().enumerate() {
            hands_at[index] = non_overlapping(&self.candidates, *separation, MAX_HANDS).len();
        }

        tracing::debug!(
            "best anchor {best_idx} conf={best_conf:.3} (threshold {}, {} anchors above, hands at {})",
            self.score_threshold,
            self.candidates.len(),
            SEPARATIONS
                .iter()
                .zip(hands_at.iter())
                .map(|(gate, count)| format!("{gate}={count}"))
                .collect::<Vec<_>>()
                .join(" "),
        );

        Ok(WindowReport {
            best: PalmBox {
                score: best_conf,
                ..anchor_palm_box(self.anchors[best_idx], boxes, best_idx * 18, &map)
            },
            above: self.candidates.len(),
            hands,
            hands_at,
        })
    }

    /// Choose the window this frame is seen through, and run the detector through it.
    ///
    /// `Both` is the measurement mode the two-hand decision needs: it keeps the
    /// anchored crop as the behaviour and scores the whole frame alongside it, so a
    /// run can report how many hands and how many anchors the full-frame pass would
    /// have seen without changing what the instrument does.
    fn run_detector(&mut self, frame: &Frame) -> Result<Option<PalmBox>> {
        let fw = frame.width() as f64;
        let fh = frame.height() as f64;

        // The detector wants a square input. The window follows the palm so a
        // hand at the frame edge is still fully sampled instead of being cut out
        // of the model input.
        let (tx, ty, side) = detector_crop(self.roi_palm, fw, fh);
        let crop = Affine::scale_translate(side / DETECTOR_INPUT as f64, tx, ty);

        tracing::debug!(
            "detector window {:?} x={tx:.0} y={ty:.0} side={side:.0} palm={:?}",
            self.window,
            self.roi_palm
        );

        if self.window == DetectorWindow::Both {
            let boxes = |hands: &[PalmBox]| {
                hands
                    .iter()
                    .map(|p| format!("({:.0},{:.0}) r={:.0} {:.3}", p.cx, p.cy, p.half, p.score))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let started = Instant::now();
            let anchored = self.run_detector_window(frame, crop)?;
            let crop_us = started.elapsed().as_micros() as u64;
            let started = Instant::now();
            let full = self.run_detector_window(frame, letterbox_map(fw, fh, DETECTOR_INPUT))?;
            let full_us = started.elapsed().as_micros() as u64;
            self.probe_crop.observe(&anchored, crop_us);
            self.probe_full.observe(&full, full_us);
            if tracing::enabled!(tracing::Level::DEBUG) {
                tracing::debug!(
                    "crop hands: {}; frame hands: {}",
                    boxes(&anchored.hands),
                    boxes(&full.hands),
                );
            }
            // The instrument still runs on the crop; only the numbers are collected.
            return Ok(self.commit_palm(&anchored));
        }

        let map = if self.window == DetectorWindow::FullFrame {
            letterbox_map(fw, fh, DETECTOR_INPUT)
        } else {
            crop
        };
        let report = self.run_detector_window(frame, map)?;
        Ok(self.commit_palm(&report))
    }

    /// Take the best hand from a window pass, remember it as the anchor for the next
    /// frame's crop, and count a miss when there is none.
    fn commit_palm(&mut self, report: &WindowReport) -> Option<PalmBox> {
        let palm = report.hands.first().copied();
        match palm {
            Some(palm) => {
                self.roi_palm = Some(smooth_palm(self.roi_palm, (palm.cx, palm.cy)));
                self.roi_stale = 0;
            }
            None => self.note_miss(),
        }
        palm
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
    fn run_landmarks(
        &mut self,
        frame: &Frame,
        cx: f64,
        cy: f64,
        half: f64,
    ) -> Result<Vec<Landmark>> {
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

    pub fn process_frame(&mut self, frame: &Frame) -> Result<HandDetectionResult> {
        let mut result = HandDetectionResult {
            hand_landmarks: None,
            r#box: None,
            frame_size: (frame.width(), frame.height()),
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
                tracing::debug!(
                    "palm steady, reusing landmarks ({} frame{} in a row)",
                    self.cache.streak(),
                    if self.cache.streak() > 1 { "s" } else { "" }
                );
            } else {
                let landmarker_start = Instant::now();
                match self.run_landmarks(frame, detection.cx, detection.cy, detection.half) {
                    Ok(landmarks) => {
                        result.timings.landmarker_us =
                            landmarker_start.elapsed().as_secs_f64() * 1e6;
                        if tracing::enabled!(tracing::Level::DEBUG) {
                            let (mut minx, mut miny, mut maxx, mut maxy) =
                                (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                            for lm in &landmarks {
                                minx = minx.min(lm.x);
                                maxx = maxx.max(lm.x);
                                miny = miny.min(lm.y);
                                maxy = maxy.max(lm.y);
                            }
                            tracing::debug!(
                                "landmarks bbox x={minx:.1}..{maxx:.1} y={miny:.1}..{maxy:.1}"
                            );
                            for (i, lm) in landmarks.iter().enumerate() {
                                tracing::debug!(
                                    "lm[{i}] = ({:.1}, {:.1}, {:.3})",
                                    lm.x,
                                    lm.y,
                                    lm.z
                                );
                            }
                        }
                        self.cache.refresh(palm, &landmarks);
                        result.hand_landmarks = Some(landmarks);
                    }
                    Err(e) => {
                        tracing::warn!("landmarker stage skipped: {e}");
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

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    fn palm(cx: f64, cy: f64, half: f64, score: f32) -> PalmBox {
        PalmBox {
            cx,
            cy,
            half,
            score,
        }
    }

    #[test]
    fn palms_inside_one_another_merge_and_palms_apart_do_not() {
        // Centres 40 px apart with 60 px radii: the second centre sits inside the
        // first palm box, and the gate asks for more than 1.0 * 60 = 60 px, so this
        // is one hand written by three anchors, not three hands.
        let nested = [
            palm(200.0, 240.0, 60.0, 0.90),
            palm(240.0, 240.0, 60.0, 0.85),
            palm(205.0, 236.0, 58.0, 0.80),
        ];
        let hands = non_overlapping(&nested, HAND_SEPARATION, MAX_HANDS);
        assert_eq!(hands.len(), 1);
        assert!(
            near(hands[0].score, 0.90),
            "the strongest anchor takes the slot"
        );

        // 70 px apart is more than one palm radius, so the gate reads two hands. Where
        // exactly that line is drawn is what the two-hand clip has to measure.
        let apart = [
            palm(200.0, 240.0, 60.0, 0.90),
            palm(270.0, 240.0, 60.0, 0.70),
        ];
        assert_eq!(non_overlapping(&apart, HAND_SEPARATION, MAX_HANDS).len(), 2);

        // The cap is a cap, and an empty candidate list is not a hand.
        assert_eq!(non_overlapping(&apart, HAND_SEPARATION, 1).len(), 1);
        assert!(non_overlapping(&[], HAND_SEPARATION, MAX_HANDS).is_empty());
    }

    #[test]
    fn a_wider_gate_merges_hands_that_a_narrow_one_keeps_apart() {
        // The two-hand clip measures palm centres 1.3-2.6 palm radii apart. This pair is
        // at 1.4 radii, the tightest real case in the recording: 70 px between centres,
        // 50 px radii.
        let hands = [
            palm(200.0, 240.0, 50.0, 0.90),
            palm(270.0, 240.0, 50.0, 0.80),
        ];
        assert_eq!(non_overlapping(&hands, 1.0, MAX_HANDS).len(), 2);
        assert_eq!(non_overlapping(&hands, 0.25, MAX_HANDS).len(), 2);
        assert_eq!(non_overlapping(&hands, 2.0, MAX_HANDS).len(), 1);
    }

    #[test]
    fn an_anchor_offset_is_added_to_its_prior_in_model_pixels() {
        // Identity mapping, so model pixels are frame pixels. The prior (0.5, 0.5) of
        // a 192x192 input sits at (96, 96); the box is [dx, dy, w, h] relative to it.
        let boxes = [4.0, -2.0, 20.0, 10.0];
        let palm = anchor_palm_box(
            (0.5, 0.5),
            &boxes,
            0,
            &Affine::scale_translate(1.0, 0.0, 0.0),
        );
        assert!((palm.cx - 100.0).abs() < 1e-9);
        assert!((palm.cy - 94.0).abs() < 1e-9);
        // max(w, h) expanded x2, halved: the radius of the box covering the whole hand.
        assert!((palm.half - 20.0).abs() < 1e-9);
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
        let frame = Frame::solid(2, 2, 7);
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

        // The f32 buffer is not reallocated between frames.
        assert_eq!(ptr, stage.data.as_ptr());
    }

    use std::time::{Duration, Instant};

    /// One stage, sampled N times: either with a buffer reused across calls (what
    /// the tracker does) or allocated per call (what it used to do).
    fn time_stage(frame: &Frame, affine: Affine, size: usize, fresh_buffer: bool) -> Duration {
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

    #[test]
    #[ignore]
    fn bench_sampling() {
        let frame = Frame::solid(720, 1280, 127);
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
                "{name} {size}x{size}: reused {:?}  fresh {:?}",
                time_stage(&frame, affine, size, false),
                time_stage(&frame, affine, size, true),
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
            true,
            0,
            DetectorWindow::Anchored,
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
