//! Two-stage hand tracking: BlazeHand detector (192x192) + landmarker (224x224).
//!
//! The detector returns an axis-aligned hand box which is turned into a
//! rectangle in frame pixel coordinates; the landmarker stage only runs once a
//! confident box is found. The tracker only computes geometry - all drawing is
//! handled by `crate::visualizer`.

use anyhow::{anyhow, Context, Result};
use opencv::prelude::*;
use opencv::{core, imgproc};
use std::time::Instant;

use ort::session::Session;
use ort::value::TensorRef;

use crate::core::{HandDetectionResult, Landmark, StageTimings};

const NUM_KEYPOINTS: usize = 21;
const DETECTOR_INPUT: usize = 192;
const LANDMARKER_INPUT: usize = 224;
const SCORE_THRESHOLD: f32 = 0.3;

const INV_255: f32 = 1.0 / 255.0;

/// Affine mapping from output (model) pixels to frame pixels:
/// `frame_x = m00 * out_x + m01 * out_y + tx`, and the same for y.
#[derive(Clone, Copy, Debug)]
struct Affine {
    m00: f64,
    m01: f64,
    tx: f64,
    m10: f64,
    m11: f64,
    ty: f64,
}

impl Affine {
    /// Axis-aligned mapping: one output pixel covers `scale` frame pixels, starting
    /// at (tx, ty). Both stages use this.
    fn scale_translate(scale: f64, tx: f64, ty: f64) -> Self {
        Affine {
            m00: scale,
            m01: 0.0,
            tx,
            m10: 0.0,
            m11: scale,
            ty,
        }
    }

    fn frame_x(&self, out_x: f64, out_y: f64) -> f64 {
        self.tx + self.m00 * out_x + self.m01 * out_y
    }

    fn frame_y(&self, out_x: f64, out_y: f64) -> f64 {
        self.ty + self.m10 * out_x + self.m11 * out_y
    }
}

/// Scratch buffers for one inference stage, allocated once when the tracker is
/// built and reused every frame: `warped` is the OpenCV crop output (8UC3), `data`
/// the model input, (1, out_h, out_w, 3) RGB f32 in [0, 1].
struct StageInput {
    warped: Mat,
    data: Vec<f32>,
}

impl StageInput {
    fn new(out_w: usize, out_h: usize) -> Self {
        StageInput {
            warped: Mat::default(),
            data: vec![0.0f32; out_w * out_h * 3],
        }
    }
}

/// Resample `frame` through `map` (output pixel -> frame pixel) into `stage.data`
/// as a (1, out_h, out_w, 3) RGB float tensor in [0, 1].
///
/// Returns the *effective* mapping that was applied: the `resize` fast path snaps
/// the crop to whole pixels, so callers must use the returned map (not the one they
/// passed in) when converting model coordinates back to frame pixels.
///
/// OpenCV does all of the sampling: `resize` for the axis-aligned crop both stages
/// actually need (the common case), `warpAffine` otherwise - which expects the
/// *forward* frame -> output matrix, i.e. the inverse of the mapping handed in here,
/// and pads with black outside the frame exactly like the old sampler did. The only
/// Rust-side work is one bulk BGR -> RGB / u8 -> f32 pass over the reused buffers,
/// instead of the previous loop that made two `frame.ptr(y)` FFI calls and four
/// corner loads for every one of the 192x192 / 224x224 pixels.
fn sample_affine_to_tensor(
    frame: &Mat,
    map: Affine,
    out_w: usize,
    out_h: usize,
    stage: &mut StageInput,
) -> Result<Affine> {
    if frame.channels() != 3 {
        return Err(anyhow!(
            "expected a 3-channel frame, got {}",
            frame.channels()
        ));
    }

    let fw = frame.cols();
    let fh = frame.rows();
    let out_size = core::Size::new(out_w as i32, out_h as i32);
    let applied;

    if map.m01.abs() < 1e-9 && map.m10.abs() < 1e-9 && map.m00 > 0.0 && map.m11 > 0.0 {
        // Pure scale + translation: crop to whole pixels, then let resize interpolate.
        // OpenCV reuses `stage.warped`'s buffer as long as the size/type match.
        let x0 = map.tx.floor() as i32;
        let y0 = map.ty.floor() as i32;
        let w = (map.m00 * out_w as f64).round() as i32;
        let h = (map.m11 * out_h as f64).round() as i32;

        if x0 >= 0 && y0 >= 0 && w > 0 && h > 0 && x0 + w <= fw && y0 + h <= fh {
            let roi = frame.roi(core::Rect::new(x0, y0, w, h))?;
            imgproc::resize(
                &roi,
                &mut stage.warped,
                out_size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            applied = Affine {
                m00: w as f64 / out_w as f64,
                m01: 0.0,
                tx: x0 as f64,
                m10: 0.0,
                m11: h as f64 / out_h as f64,
                ty: y0 as f64,
            };
        } else {
            // Crop runs off the edge of the frame; warpAffine clamps and pads with black.
            applied = warp_affine(frame, map, out_size, &mut stage.warped)?;
        }
    } else {
        applied = warp_affine(frame, map, out_size, &mut stage.warped)?;
    }

    let n = out_h * out_w * 3;
    if stage.data.len() != n {
        stage.data.resize(n, 0.0);
    }
    // Safety: `stage.warped` was just written by resize/warpAffine as a contiguous
    // out_h x out_w 3-channel u8 matrix (no ROI), and `stage.data` has exactly `n`
    // f32 elements.
    unsafe {
        let src = std::slice::from_raw_parts(stage.warped.data(), n);
        let dst = stage.data.as_mut_ptr();
        for i in (0..n).step_by(3) {
            // BGR -> RGB while normalising, in the same pass.
            *dst.add(i) = src[i + 2] as f32 * INV_255;
            *dst.add(i + 1) = src[i + 1] as f32 * INV_255;
            *dst.add(i + 2) = src[i] as f32 * INV_255;
        }
    }

    Ok(applied)
}

/// Sample `frame` through the inverse of `map` using `warpAffine`, which takes the
/// forward frame -> output matrix. Returns the mapping that was applied.
fn warp_affine(frame: &Mat, map: Affine, out_size: core::Size, dst: &mut Mat) -> Result<Affine> {
    let Affine {
        m00,
        m01,
        tx,
        m10,
        m11,
        ty,
    } = map;
    let det = m00 * m11 - m01 * m10;
    if det.abs() < 1e-12 {
        return Err(anyhow!("degenerate affine mapping (det = {det})"));
    }
    let a11 = m11 / det;
    let a12 = -m01 / det;
    let a21 = -m10 / det;
    let a22 = m00 / det;
    let fwd = [
        a11,
        a12,
        -(a11 * tx + a12 * ty),
        a21,
        a22,
        -(a21 * tx + a22 * ty),
    ];
    // warpAffine insists on a 2x3 matrix.
    let fwd_mat = Mat::from_slice(&fwd)?;
    let forward = fwd_mat.reshape(1, 2)?;

    imgproc::warp_affine(
        frame,
        dst,
        &forward,
        out_size,
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        core::Scalar::all(0.0),
    )?;

    Ok(map)
}

pub struct HandTrackerImpl {
    pub detector_session: Session,
    pub landmarker_session: Session,
    /// Anchor centres in normalised detector space (0..1).
    pub anchors: Vec<(f32, f32)>,
    pub debug: bool,
    /// Per-stage input buffers, allocated once here and refilled in place every
    /// frame (see `StageInput`).
    detector_input: StageInput,
    landmarker_input: StageInput,
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
            detector_input: StageInput::new(DETECTOR_INPUT, DETECTOR_INPUT),
            landmarker_input: StageInput::new(LANDMARKER_INPUT, LANDMARKER_INPUT),
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

        let map = sample_affine_to_tensor(
            frame,
            Affine::scale_translate(k, tx, ty),
            DETECTOR_INPUT,
            DETECTOR_INPUT,
            &mut self.detector_input,
        )?;

        // `TensorRef` wraps the reused buffer in place - ort does not copy it.
        let input = TensorRef::from_array_view((
            [1usize, DETECTOR_INPUT, DETECTOR_INPUT, 3],
            self.detector_input.data.as_slice(),
        ))?;
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

        // Back to frame pixels using the mapping that was actually applied.
        let frame_cx = map.frame_x(cx, cy);
        let frame_cy = map.frame_y(cx, cy);
        let half = bw.max(bh) * 2.0 * 0.5 * map.m00; // expand to cover the whole hand

        let x0 = (frame_cx - half).max(0.0);
        let y0 = (frame_cy - half).max(0.0);
        let x1 = (frame_cx + half).min(fw);
        let y1 = (frame_cy + half).min(fh);

        if x1 <= x0 || y1 <= y0 {
            return Ok(None);
        }

        Ok(Some((
            x0 as f32,
            y0 as f32,
            (x1 - x0) as f32,
            (y1 - y0) as f32,
            best_conf,
        )))
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

        let map = sample_affine_to_tensor(
            frame,
            Affine::scale_translate(s, cx - side * 0.5, cy - side * 0.5),
            LANDMARKER_INPUT,
            LANDMARKER_INPUT,
            &mut self.landmarker_input,
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
            score: 0.0,
            timings: StageTimings::default(),
        };

        let detector_start = Instant::now();
        let detection = self.run_detector(frame)?;
        result.timings.detector_us = detector_start.elapsed().as_secs_f64() * 1e6;

        if let Some((x0, y0, w, h, conf)) = detection {
            result.r#box = Some((x0, y0, w, h));
            result.score = conf;

            let landmarker_start = Instant::now();
            match self.run_landmarks(frame, x0, y0, w, h) {
                Ok(landmarks) => {
                    result.timings.landmarker_us = landmarker_start.elapsed().as_secs_f64() * 1e6;
                    result.hand_landmarks = Some(landmarks.clone());
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
                    let src = &TEST_PIXELS[(y as usize * 2 + x as usize) * 3..];
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
        let map = sample_affine_to_tensor(frame, map, out_w, out_h, &mut stage).unwrap();
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
    fn stage_buffers_are_reused_across_frames() {
        let frame = test_frame();
        let mut stage = StageInput::new(2, 2);
        sample_affine_to_tensor(
            &frame,
            Affine::scale_translate(1.0, 0.0, 0.0),
            2,
            2,
            &mut stage,
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
        )
        .unwrap();

        // Neither the f32 buffer nor the OpenCV scratch matrix is reallocated.
        assert_eq!(ptr, stage.data.as_ptr());
        assert_eq!(warped_ptr, stage.warped.data());
    }

    use std::time::Instant;

    /// The old per-pixel sampler, kept only so the benchmark has something to compare
    /// against. Run with `cargo test --release -- --include-ignored bench_sampling`.
    fn legacy_sample(
        frame: &Mat,
        m00: f64,
        m01: f64,
        tx: f64,
        m10: f64,
        m11: f64,
        ty: f64,
        out_w: usize,
        out_h: usize,
    ) -> Vec<f32> {
        let fw = frame.cols() as f64;
        let fh = frame.rows() as f64;
        let mut data = vec![0.0f32; out_h * out_w * 3];
        unsafe {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let sx = m00 * ox as f64 + m01 * oy as f64 + tx;
                    let sy = m10 * ox as f64 + m11 * oy as f64 + ty;
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
        const N: u32 = 300;

        // Detector stage (192x192), buffers reused vs allocated per call.
        let mut stage = StageInput::new(192, 192);
        let t = Instant::now();
        for _ in 0..N {
            let map = sample_affine_to_tensor(
                &frame,
                Affine::scale_translate(3.75, 280.0, 0.0),
                192,
                192,
                &mut stage,
            )
            .unwrap();
            std::hint::black_box(map);
        }
        let reused_192 = t.elapsed() / N;

        let t = Instant::now();
        for _ in 0..N {
            let mut fresh = StageInput::new(192, 192);
            let map = sample_affine_to_tensor(
                &frame,
                Affine::scale_translate(3.75, 280.0, 0.0),
                192,
                192,
                &mut fresh,
            )
            .unwrap();
            std::hint::black_box(map);
        }
        let fresh_192 = t.elapsed() / N;

        let t = Instant::now();
        for _ in 0..N {
            std::hint::black_box(legacy_sample(
                &frame, 3.75, 0.0, 280.0, 0.0, 3.75, 0.0, 192, 192,
            ));
        }
        let old_192 = t.elapsed() / N;

        // Landmarker stage (224x224).
        let mut stage = StageInput::new(224, 224);
        let t = Instant::now();
        for _ in 0..N {
            let map = sample_affine_to_tensor(
                &frame,
                Affine::scale_translate(2.7, 200.0, 100.0),
                224,
                224,
                &mut stage,
            )
            .unwrap();
            std::hint::black_box(map);
        }
        let reused_224 = t.elapsed() / N;

        let t = Instant::now();
        for _ in 0..N {
            let mut fresh = StageInput::new(224, 224);
            let map = sample_affine_to_tensor(
                &frame,
                Affine::scale_translate(2.7, 200.0, 100.0),
                224,
                224,
                &mut fresh,
            )
            .unwrap();
            std::hint::black_box(map);
        }
        let fresh_224 = t.elapsed() / N;

        let t = Instant::now();
        for _ in 0..N {
            std::hint::black_box(legacy_sample(
                &frame, 2.7, 0.0, 200.0, 0.0, 2.7, 100.0, 224, 224,
            ));
        }
        let old_224 = t.elapsed() / N;

        println!("192x192: reused {reused_192:?}  fresh {fresh_192:?}  old {old_192:?}");
        println!("224x224: reused {reused_224:?}  fresh {fresh_224:?}  old {old_224:?}");
    }
}
