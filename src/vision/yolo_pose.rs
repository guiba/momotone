//! One-stage hand pose: an Ultralytics YOLO pose model returns the hand box *and*
//! the 21 landmarks in a single inference pass over the whole frame.
//!
//! Compared with the two-stage MediaPipe path (`tracker.rs`) this trades two model
//! runs (192x192 palm detector + 224x224 landmarker) for one letterboxed pass. The
//! upside is robustness: the hand is not found twice, the landmark crop can never
//! cut the fingers off, every keypoint carries a visibility score, and the graph
//! natively sees more than one hand.
//!
//! The keypoint order is the MediaPipe order (wrist, thumb x4, index x4, middle x4,
//! ring x4, pinky x4), so the strategies, the skeleton and the keyboard overlay need
//! no changes.

use anyhow::{anyhow, Context, Result};
use opencv::prelude::*;
use std::time::Instant;

use ort::session::Session;
use ort::value::TensorRef;

use crate::core::{HandDetectionResult, Landmark, StageTimings};
use crate::vision::reuse::{square_box, ReuseCache};
use crate::vision::sampling::{letterbox_map, sample_affine_to_tensor, StageInput, TensorLayout};

const NUM_KEYPOINTS: usize = 21;
/// Ultralytics letterboxes the input into a square and fills the border with this
/// mid-grey; the MediaPipe crops pad with black instead.
const LETTERBOX_PAD: f64 = 114.0;

#[derive(Debug, Clone, Copy)]
struct Keypoint {
    x: f64,
    y: f64,
    visibility: f32,
}

/// One raw detection, still in letterboxed model pixels.
#[derive(Debug, Clone)]
struct RawDetection {
    score: f32,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    keypoints: [Keypoint; NUM_KEYPOINTS],
}

impl RawDetection {
    fn mean_visibility(&self) -> f32 {
        self.keypoints.iter().map(|k| k.visibility).sum::<f32>() / NUM_KEYPOINTS as f32
    }
}

/// Centre and half-side of the square box a palm anchor describes.
fn box_anchor(rect: (f32, f32, f32, f32)) -> (f64, f64, f64) {
    (
        (rect.0 + rect.2 * 0.5) as f64,
        (rect.1 + rect.3 * 0.5) as f64,
        (rect.2.max(rect.3) as f64) * 0.5,
    )
}

/// A detection mapped back into frame pixels.
struct Pose {
    score: f32,
    /// Square hand box (x, y, w, h), same convention as the MediaPipe tracker.
    r#box: (f32, f32, f32, f32),
    landmarks: Vec<Landmark>,
}

/// Pose exports carry `4` box values, then optionally an objectness and a class id,
/// then three values (x, y, visibility) per keypoint.
fn attribute_width(keypoints: usize) -> [usize; 3] {
    let kpts = keypoints * 3;
    [4 + kpts, 5 + kpts, 6 + kpts]
}

/// Turn the raw output tensor into detections, whichever of the two Ultralytics
/// layouts the export used: `(1, attributes, detections)` (YOLOv8/v11) or
/// `(1, detections, attributes)` (YOLO26's NMS-free head).
fn decode_pose(dims: &[usize], data: &[f32]) -> Result<Vec<RawDetection>> {
    if dims.len() < 2 {
        return Err(anyhow!("pose output has an unexpected shape {dims:?}"));
    }
    let (a, b) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    if a * b != data.len() {
        return Err(anyhow!(
            "pose output shape {dims:?} does not match {} floats",
            data.len()
        ));
    }

    let widths = attribute_width(NUM_KEYPOINTS);
    // Which axis carries the per-detection attributes decides the stride: YOLOv8/v11
    // exports (attributes, detections), YOLO26's NMS-free head (detections, attributes).
    let (attrs, detections, attribute_major) = if widths.contains(&a) {
        (a, b, true)
    } else if widths.contains(&b) {
        (b, a, false)
    } else {
        return Err(anyhow!(
            "pose output {dims:?} has no keypoint axis (expected {}/{}/{} values per detection)",
            widths[0],
            widths[1],
            widths[2]
        ));
    };

    let at = |d: usize, attr: usize| -> f32 {
        if attribute_major {
            data[attr * detections + d]
        } else {
            data[d * attrs + attr]
        }
    };

    // The keypoints are the tail of a detection's attributes; the leading values are
    // the box plus whatever confidence fields that export has.
    let keypoint_offset = attrs - NUM_KEYPOINTS * 3;
    let confidence_index = if keypoint_offset >= 5 { Some(4) } else { None };

    let mut out = Vec::with_capacity(detections);
    for d in 0..detections {
        let score = match confidence_index {
            Some(i) => at(d, i),
            None => 1.0,
        };
        let mut keypoints = [Keypoint {
            x: 0.0,
            y: 0.0,
            visibility: 0.0,
        }; NUM_KEYPOINTS];
        for (i, kp) in keypoints.iter_mut().enumerate() {
            let base = keypoint_offset + i * 3;
            kp.x = at(d, base) as f64;
            kp.y = at(d, base + 1) as f64;
            kp.visibility = at(d, base + 2);
        }
        out.push(RawDetection {
            score,
            x1: at(d, 0) as f64,
            y1: at(d, 1) as f64,
            x2: at(d, 2) as f64,
            y2: at(d, 3) as f64,
            keypoints,
        });
    }
    Ok(out)
}

pub struct YoloPoseTracker {
    session: Session,
    input_name: String,
    input_size: usize,
    stage: StageInput,
    score_threshold: f32,
    debug: bool,
    cache: ReuseCache,
    last_score: f32,
}

impl YoloPoseTracker {
    pub fn new(
        path: &str,
        input_size: usize,
        score_threshold: f32,
        debug: bool,
        reuse_landmarks: bool,
    ) -> Result<Self> {
        let mut input_size = input_size.max(32);
        let session = Session::builder()
            .with_context(|| format!("Failed to load pose model {path}"))?
            .commit_from_file(path)
            .with_context(|| format!("Failed to compile pose model {path}"))?;

        // Ultralytics exports are usually fixed at their training size. Honour what
        // the graph declares rather than trusting `--pose-input`, and say so when the
        // two disagree: a static 640x640 graph will reject any other buffer.
        let declared = session
            .inputs()
            .first()
            .and_then(|input| match input.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => shape
                    .iter()
                    .copied()
                    .filter(|&d| d > 0)
                    .map(|d| d as usize)
                    .next_back(),
                _ => None,
            });
        if let Some(side) = declared {
            if side != input_size && debug {
                println!(
                    "[Pose] {path} is exported at a fixed {side}x{side} input; --pose-input {input_size} ignored. Re-export with `--dynamic` to change it."
                );
            }
            input_size = side;
        }

        // Ultralytics exports name the input `images`; read it from the graph rather
        // than assume it, so a differently exported model still loads.
        let input_name = session
            .inputs()
            .first()
            .map(|input| input.name().to_string())
            .unwrap_or_else(|| "images".to_string());

        if debug {
            let outputs: Vec<String> = session
                .outputs()
                .iter()
                .map(|out| out.name().to_string())
                .collect();
            println!(
                "[Pose] Loaded {path} (input '{input_name}', {input_size}x{input_size}, outputs {outputs:?})"
            );
        }

        Ok(Self {
            session,
            input_name,
            input_size,
            stage: StageInput::new(input_size, input_size),
            score_threshold: score_threshold.clamp(0.0, 1.0),
            debug,
            cache: ReuseCache::new(reuse_landmarks),
            last_score: 0.0,
        })
    }

    /// One letterboxed inference pass, returning the best hand in frame pixels.
    fn run(&mut self, frame: &Mat) -> Result<Option<Pose>> {
        let map = letterbox_map(frame.cols() as f64, frame.rows() as f64, self.input_size);
        let applied = sample_affine_to_tensor(
            frame,
            map,
            self.input_size,
            self.input_size,
            &mut self.stage,
            TensorLayout::Nchw,
            LETTERBOX_PAD,
        )?;

        let input = TensorRef::from_array_view((
            [1usize, 3, self.input_size, self.input_size],
            self.stage.data.as_slice(),
        ))?;
        let outputs: Vec<_> = self
            .session
            .run(vec![(self.input_name.as_str(), input)])?
            .into_iter()
            .collect();

        let mut detections = Vec::new();
        for entry in &outputs {
            let (shape, data) = match entry.1.try_extract_tensor::<f32>() {
                Ok(pair) => pair,
                // Not the pose head (a side branch, or an FP16 output we cannot read
                // as f32) - leave it to the error below.
                Err(_) => continue,
            };
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            match decode_pose(&dims, data) {
                Ok(found) => detections.extend(found),
                Err(e) => {
                    if self.debug {
                        eprintln!("[Pose] skipped output {dims:?}: {e}");
                    }
                }
            }
        }

        if detections.is_empty() {
            return Err(anyhow!(
                "no pose head in the model output; expected a (1, {}, N) or (1, N, {}) Ultralytics pose export in FP32",
                attribute_width(NUM_KEYPOINTS)[1],
                attribute_width(NUM_KEYPOINTS)[1],
            ));
        }

        if self.debug {
            let mut scores: Vec<f32> = detections.iter().map(|d| d.score).collect();
            scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(5);
            eprintln!(
                "[Pose] {} candidate detections, top scores {:?} (threshold {})",
                detections.len(),
                scores,
                self.score_threshold
            );
        }

        let best = detections
            .into_iter()
            .filter(|d| d.score >= self.score_threshold)
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let Some(best) = best else {
            return Ok(None);
        };

        if self.debug {
            eprintln!(
                "[Pose] conf {:.3} box {:.0},{:.0} .. {:.0},{:.0} mean keypoint visibility {:.2}",
                best.score,
                best.x1,
                best.y1,
                best.x2,
                best.y2,
                best.mean_visibility()
            );
        }

        // Map model pixels back to frame pixels through the mapping that was actually
        // applied to the input.
        let landmarks: Vec<Landmark> = best
            .keypoints
            .iter()
            .map(|kp| Landmark {
                x: applied.frame_x(kp.x, kp.y) as f32,
                y: applied.frame_y(kp.x, kp.y) as f32,
                // `z` is unused by the strategies, so a YOLO keypoint carries its
                // visibility here instead of a depth.
                z: kp.visibility,
            })
            .collect();

        let x1 = applied.frame_x(best.x1, best.y1);
        let y1 = applied.frame_y(best.x1, best.y1);
        let x2 = applied.frame_x(best.x2, best.y2);
        let y2 = applied.frame_y(best.x2, best.y2);
        let half = ((x2 - x1) * 0.5).max((y2 - y1) * 0.5);
        let cx = (x1 + x2) * 0.5;
        let cy = (y1 + y2) * 0.5;

        Ok(Some(Pose {
            score: best.score,
            r#box: square_box((cx, cy, half)),
            landmarks,
        }))
    }

    pub fn process_frame(&mut self, frame: &Mat) -> Result<HandDetectionResult> {
        let mut result = HandDetectionResult {
            hand_landmarks: None,
            r#box: None,
            frame_size: (frame.cols() as usize, frame.rows() as usize),
            score: 0.0,
            timings: StageTimings::default(),
        };

        // A hand that has not moved or resized since the last pose pass does not need
        // a new one - here that saves the *whole* model, not just the landmarker.
        if let Some(anchor) = self.cache.anchor() {
            if let Some(landmarks) = self.cache.reuse(anchor) {
                result.hand_landmarks = Some(landmarks);
                result.r#box = Some(square_box(anchor));
                result.score = self.last_score;
                if self.debug {
                    eprintln!(
                        "[Pose] hand steady, reusing the previous pose ({} frame{} in a row)",
                        self.cache.streak(),
                        if self.cache.streak() > 1 { "s" } else { "" }
                    );
                }
                return Ok(result);
            }
        }

        let start = Instant::now();
        let pose = self.run(frame)?;
        // The single stage is reported as the detector so the dashboard's performance
        // card keeps meaning "the inference we actually ran".
        result.timings.detector_us = start.elapsed().as_secs_f64() * 1e6;

        match pose {
            Some(pose) => {
                result.score = pose.score;
                result.r#box = Some(pose.r#box);
                let anchor = box_anchor(pose.r#box);
                self.cache.refresh(anchor, &pose.landmarks);
                self.last_score = pose.score;
                result.hand_landmarks = Some(pose.landmarks);
            }
            None => {
                // No hand: the cache is worthless and must not be reused on the next
                // sighting.
                self.cache.clear();
                self.last_score = 0.0;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_row(rect: (f64, f64, f64, f64), conf: f32, has_class: bool) -> Vec<f32> {
        let mut row = vec![
            rect.0 as f32,
            rect.1 as f32,
            rect.2 as f32,
            rect.3 as f32,
            conf,
        ];
        if has_class {
            row.push(0.0);
        }
        for i in 0..NUM_KEYPOINTS {
            row.push(rect.0 as f32 + i as f32);
            row.push(rect.1 as f32 + i as f32);
            row.push(0.9);
        }
        row
    }

    /// Flatten one row per detection into the (attributes, detections) layout.
    fn attribute_major(rows: &[Vec<f32>]) -> Vec<f32> {
        let dets = rows.len();
        let attrs = rows[0].len();
        let mut data = vec![0.0f32; attrs * dets];
        for (d, row) in rows.iter().enumerate() {
            for (a, &value) in row.iter().enumerate() {
                data[a * dets + d] = value;
            }
        }
        data
    }

    #[test]
    fn the_attribute_major_pose_head_decodes() {
        // (1, attributes, detections): the YOLOv8/v11 pose head, no class column.
        let rows = vec![
            synthetic_row((10.0, 20.0, 110.0, 120.0), 0.87, false),
            synthetic_row((50.0, 60.0, 150.0, 160.0), 0.31, false),
        ];
        let attrs = rows[0].len();
        let detections = decode_pose(&[1, attrs, rows.len()], &attribute_major(&rows)).unwrap();
        assert_eq!(detections.len(), 2);
        assert!((detections[0].score - 0.87).abs() < 1e-6);
        assert_eq!((detections[0].x1, detections[0].y2), (10.0, 120.0));
        assert_eq!((detections[1].x1, detections[1].score), (50.0, 0.31));
        assert_eq!(
            (detections[0].keypoints[8].x, detections[0].keypoints[8].y),
            (18.0, 28.0)
        );
    }

    #[test]
    fn the_nms_free_head_decodes() {
        // (1, detections, attributes): YOLO26's head, with a class column.
        let mut data = Vec::new();
        for _ in 0..300 {
            data.extend(synthetic_row((10.0, 20.0, 110.0, 120.0), 0.87, true));
        }
        let attrs = 4 + 1 + 1 + NUM_KEYPOINTS * 3;
        let detections = decode_pose(&[1, 300, attrs], &data).unwrap();
        assert_eq!(detections.len(), 300);
        assert!((detections[0].score - 0.87).abs() < 1e-6);
        assert_eq!(detections[0].keypoints[20].x, 30.0);
    }

    #[test]
    fn the_index_tip_is_keypoint_eight() {
        let data = synthetic_row((0.0, 0.0, 100.0, 100.0), 0.9, true);
        let attrs = data.len();
        let detections = decode_pose(&[1, 1, attrs], &data).unwrap();
        // Wrist, thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20).
        assert_eq!(detections[0].keypoints[4].x, 4.0);
        assert_eq!(detections[0].keypoints[8].x, 8.0);
        assert_eq!(detections[0].keypoints[20].x, 20.0);
    }

    #[test]
    fn an_unrecognisable_output_is_rejected_not_silently_misread() {
        let err = decode_pose(&[1, 64, 100], &[0.0f32; 6400]).unwrap_err();
        assert!(err.to_string().contains("no keypoint axis"));
    }

    #[test]
    fn the_square_box_is_centred_on_the_detected_hand() {
        // A tall, narrow hand box becomes a square sized by its longer side, exactly
        // like the MediaPipe path reports.
        let rect = square_box((320.0, 240.0, 100.0));
        assert_eq!(rect, (220.0, 140.0, 200.0, 200.0));
        assert_eq!(box_anchor(rect), (320.0, 240.0, 100.0));
    }
}
