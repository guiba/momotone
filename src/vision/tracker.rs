//! Two-stage hand tracking: BlazeHand detector (192x192) + landmarker (224x224).
//!
//! The detector returns an axis-aligned hand box which is turned into a
//! rectangle in frame pixel coordinates; the landmarker stage is wired up but
//! only runs once a confident box is found.

use anyhow::{anyhow, Context, Result};
use ndarray::Array4;
use opencv::core::{Point, Rect, Scalar};
use opencv::imgproc;
use opencv::prelude::*;

use ort::session::Session;
use ort::value::Value;

use crate::core::{HandDetectionResult, Landmark};

const NUM_KEYPOINTS: usize = 21;
const DETECTOR_INPUT: usize = 192;
const LANDMARKER_INPUT: usize = 224;
const SCORE_THRESHOLD: f32 = 0.3;

/// Sample `frame` through an affine mapping (output pixel -> frame pixel) into
/// a (1, out_h, out_w, 3) RGB float tensor in [0, 1].
///
/// Sampling is done by hand so the mapping direction is explicit; OpenCV's
/// `warpAffine` expects the *forward* (frame -> output) matrix, which is easy
/// to get backwards here.
fn sample_affine_to_tensor(
    frame: &Mat,
    m00: f64,
    m01: f64,
    tx: f64,
    m10: f64,
    m11: f64,
    ty: f64,
    out_w: usize,
    out_h: usize,
) -> Result<Array4<f32>> {
    if frame.channels() != 3 {
        return Err(anyhow!("expected a 3-channel frame, got {}", frame.channels()));
    }

    let fw = frame.cols() as f64;
    let fh = frame.rows() as f64;

    let mut data = vec![0.0f32; out_h * out_w * 3];

    unsafe {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let sx = m00 * ox as f64 + m01 * oy as f64 + tx;
                let sy = m10 * ox as f64 + m11 * oy as f64 + ty;

                if sx < 0.0 || sy < 0.0 || sx >= fw || sy >= fh {
                    continue; // stays black
                }

                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(frame.cols() as usize - 1);
                let y1 = (y0 + 1).min(frame.rows() as usize - 1);
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                let row0 = frame.ptr(y0 as i32)?;
                let row1 = frame.ptr(y1 as i32)?;
                let c00 = row0.add(x0 * 3);
                let c01 = row0.add(x1 * 3);
                let c10 = row1.add(x0 * 3);
                let c11 = row1.add(x1 * 3);

                // BGR -> RGB
                let px = |p: *const u8, c: usize| -> f64 { *p.add(c) as f64 };
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

    let tensor = Array4::from_shape_vec((1, out_h, out_w, 3), data)
        .with_context(|| format!("failed to build {out_w}x{out_h} input tensor"))?;
    Ok(tensor)
}

pub struct HandTrackerImpl {
    pub detector_session: Session,
    pub landmarker_session: Session,
    /// Anchor centres in normalised detector space (0..1).
    pub anchors: Vec<(f32, f32)>,
    pub debug: bool,
}

impl HandTrackerImpl {
    pub fn new(
        detector_path: &str,
        landmarker_path: &str,
        anchors_path: &str,
        debug: bool,
    ) -> Result<Self> {
        let detector_session = Session::builder()
            .with_context(|| format!("Failed to load detector {detector_path}"))?
            .commit_from_file(detector_path)?;
        let landmarker_session = Session::builder()
            .with_context(|| format!("Failed to load landmarker {landmarker_path}"))?
            .commit_from_file(landmarker_path)?;

        let anchors = load_anchors(anchors_path)?;
        if anchors.is_empty() {
            return Err(anyhow!("anchors file {anchors_path} is empty"));
        }

        if debug {
            println!(
                "[Tracker] Loaded detector: {detector_path}, landmarker: {landmarker_path}, {} anchors.",
                anchors.len()
            );
        }

        Ok(Self {
            detector_session,
            landmarker_session,
            anchors,
            debug,
        })
    }

    /// Detector stage: returns the hand box in frame pixel coordinates.
    fn run_detector(&mut self, frame: &Mat) -> Result<Option<(f32, f32, f32, f32, f32)>> {
        let fw = frame.cols() as f64;
        let fh = frame.rows() as f64;

        // The detector wants a square input: take the largest centred square.
        let side = fw.min(fh);
        let k = side / DETECTOR_INPUT as f64; // detector pixel -> frame pixel
        let tx = 0.5 * (fw - side);
        let ty = 0.5 * (fh - side);

        let tensor = sample_affine_to_tensor(frame, k, 0.0, tx, 0.0, k, ty, DETECTOR_INPUT, DETECTOR_INPUT)?;

        let outputs: Vec<_> = self
            .detector_session
            .run(vec![("input_1", Value::from_array(tensor)?)])?
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
                "[Tracker] best anchor {best_idx} conf={best_conf:.3} (threshold {SCORE_THRESHOLD})"
            );
        }

        if best_conf < SCORE_THRESHOLD {
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

        // Back to frame pixels. The detector box is the palm region, so expand it
        // to cover the whole hand.
        let frame_cx = tx + cx * k;
        let frame_cy = ty + cy * k;
        let half = bw.max(bh) * 2.0 * k * 0.5; // expand to cover the whole hand

        let x0 = (frame_cx - half).max(0.0);
        let y0 = (frame_cy - half).max(0.0);
        let x1 = (frame_cx + half).min(fw);
        let y1 = (frame_cy + half).min(fh);

        if x1 <= x0 || y1 <= y0 {
            return Ok(None);
        }

        Ok(Some((x0 as f32, y0 as f32, (x1 - x0) as f32, (y1 - y0) as f32, best_conf)))
    }

    /// Landmarker stage: crop the detected box into a 224x224 square and run it.
    fn run_landmarks(
        &mut self,
        frame: &Mat,
        box_x0: f32,
        box_y0: f32,
        box_w: f32,
        box_h: f32,
    ) -> Result<Vec<Landmark>> {
        let cx = box_x0 as f64 + box_w as f64 * 0.5;
        let cy = box_y0 as f64 + box_h as f64 * 0.5;
        let side = (box_w.max(box_h) as f64) * 1.2;
        let s = side / LANDMARKER_INPUT as f64; // crop pixel -> frame pixel

        let tensor = sample_affine_to_tensor(
            frame,
            s,
            0.0,
            cx - side * 0.5,
            0.0,
            s,
            cy - side * 0.5,
            LANDMARKER_INPUT,
            LANDMARKER_INPUT,
        )?;

        let outputs: Vec<_> = self
            .landmarker_session
            .run(vec![("input_1", Value::from_array(tensor)?)])?
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

        let ox = cx - side * 0.5;
        let oy = cy - side * 0.5;

        let mut landmarks = Vec::with_capacity(NUM_KEYPOINTS);
        for i in 0..NUM_KEYPOINTS {
            let u = raw[3 * i] as f64; // pixel inside the 224x224 crop
            let v = raw[3 * i + 1] as f64;
            let z = raw[3 * i + 2];
            landmarks.push(Landmark {
                x: (ox + u * s) as f32,
                y: (oy + v * s) as f32,
                z,
            });
        }

        Ok(landmarks)
    }

    pub fn process_frame(&mut self, frame: &Mat) -> Result<HandDetectionResult> {
        let mut result = HandDetectionResult {
            hand_landmarks: None,
            r#box: None,
            score: 0.0,
        };

        if let Some((x0, y0, w, h, conf)) = self.run_detector(frame)? {
            result.r#box = Some((x0, y0, w, h));
            result.score = conf;

            match self.run_landmarks(frame, x0, y0, w, h) {
                Ok(landmarks) => {
                    result.hand_landmarks = Some(landmarks.clone());
                    if self.debug {
                        let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                        for lm in &landmarks {
                            minx = minx.min(lm.x);
                            maxx = maxx.max(lm.x);
                            miny = miny.min(lm.y);
                            maxy = maxy.max(lm.y);
                        }
                        eprintln!(
                            "[Tracker] landmarks bbox x={minx:.1}..{maxx:.1} y={miny:.1}..{maxy:.1}"
                        );
                    }
                }
                Err(e) => {
                    if self.debug {
                        eprintln!("[Tracker] landmarker stage skipped: {e}");
                    }
                }
            }
        }

        Ok(result)
    }

    /// Draw the axis-aligned bounding box of the detected hand.
    pub fn draw_box_on_frame(&self, frame: &mut Mat, detection: &HandDetectionResult) -> Result<()> {
        let Some((x, y, w, h)) = detection.r#box else {
            return Ok(());
        };

        let fw = frame.cols();
        let fh = frame.rows();
        let x0 = x.max(0.0).min(fw as f32) as i32;
        let y0 = y.max(0.0).min(fh as f32) as i32;
        let x1 = (x + w).max(0.0).min(fw as f32) as i32;
        let y1 = (y + h).max(0.0).min(fh as f32) as i32;
        if x1 <= x0 || y1 <= y0 {
            return Ok(());
        }

        imgproc::rectangle(
            frame,
            Rect::new(x0, y0, x1 - x0, y1 - y0),
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        if let Some(landmarks) = &detection.hand_landmarks {
            for lm in landmarks {
                imgproc::circle(
                    frame,
                    Point::new(lm.x as i32, lm.y as i32),
                    3,
                    Scalar::new(255.0, 0.0, 0.0, 0.0),
                    -1,
                    imgproc::LINE_8,
                    0,
                )?;
            }
        }

        Ok(())
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
