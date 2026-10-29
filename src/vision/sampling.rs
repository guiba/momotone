//! Input sampling shared by every ONNX vision stage.
//!
//! A stage takes an affine crop of the frame (scale + translate, optionally with a
//! fill for the part of the window that falls outside the image) and needs it as a
//! float tensor in either NHWC (MediaPipe) or NCHW (Ultralytics YOLO) layout. The
//! buffers are allocated once per tracker and refilled in place every frame.

use anyhow::{anyhow, Result};
use opencv::prelude::*;
use opencv::{core, imgproc};

const INV_255: f32 = 1.0 / 255.0;

/// Affine mapping from output (model) pixels to frame pixels:
/// `frame_x = m00 * out_x + m01 * out_y + tx`, and the same for y.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Affine {
    pub(crate) m00: f64,
    pub(crate) m01: f64,
    pub(crate) tx: f64,
    pub(crate) m10: f64,
    pub(crate) m11: f64,
    pub(crate) ty: f64,
}

impl Affine {
    /// Axis-aligned mapping: one output pixel covers `scale` frame pixels, starting
    /// at (tx, ty). Both MediaPipe stages use this.
    pub(crate) fn scale_translate(scale: f64, tx: f64, ty: f64) -> Self {
        Affine {
            m00: scale,
            m01: 0.0,
            tx,
            m10: 0.0,
            m11: scale,
            ty,
        }
    }

    pub(crate) fn frame_x(&self, out_x: f64, out_y: f64) -> f64 {
        self.tx + self.m00 * out_x + self.m01 * out_y
    }

    pub(crate) fn frame_y(&self, out_x: f64, out_y: f64) -> f64 {
        self.ty + self.m10 * out_x + self.m11 * out_y
    }
}

/// Scratch buffers for one inference stage, allocated once when the tracker is
/// built and reused every frame: `warped` is the OpenCV crop output (8UC3), `data`
/// the model input in the layout the stage wants.
pub(crate) struct StageInput {
    pub(crate) warped: Mat,
    pub(crate) data: Vec<f32>,
}

impl StageInput {
    pub(crate) fn new(out_w: usize, out_h: usize) -> Self {
        StageInput {
            warped: Mat::default(),
            data: vec![0.0f32; out_w * out_h * 3],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TensorLayout {
    /// (1, h, w, 3) RGB float in [0, 1] - what the MediaPipe stages expect.
    Nhwc,
    /// (1, 3, h, w) RGB float in [0, 1] - what Ultralytics exports expect.
    Nchw,
}

/// Sample `frame` through `map` (output pixel -> frame pixel) into `stage.data`.
/// Anything outside the image is filled with `border` (MediaPipe crops pad with
/// black, the letterboxed YOLO input pads with Ultralytics' mid-grey 114).
///
/// Returns the *effective* mapping that was applied: the `resize` fast path snaps
/// the crop to whole pixels, so callers must use the returned map (not the one they
/// passed in) when converting model coordinates back to frame pixels.
///
/// OpenCV does all of the sampling: `resize` for the axis-aligned crop both stages
/// actually need (the common case), `warpAffine` otherwise - which expects the
/// *forward* frame -> output matrix, i.e. the inverse of the mapping handed in here.
pub(crate) fn sample_affine_to_tensor(
    frame: &Mat,
    map: Affine,
    out_w: usize,
    out_h: usize,
    stage: &mut StageInput,
    layout: TensorLayout,
    border: f64,
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
            let crop = frame.roi(core::Rect::new(x0, y0, w, h))?;
            imgproc::resize(
                &crop,
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
            // The window runs off the edge of the frame; warpAffine clamps and pads.
            applied = warp_affine(frame, map, out_size, &mut stage.warped, border)?;
        }
    } else {
        applied = warp_affine(frame, map, out_size, &mut stage.warped, border)?;
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
        match layout {
            TensorLayout::Nhwc => {
                for i in (0..n).step_by(3) {
                    // BGR -> RGB while normalising, in the same pass.
                    *dst.add(i) = src[i + 2] as f32 * INV_255;
                    *dst.add(i + 1) = src[i + 1] as f32 * INV_255;
                    *dst.add(i + 2) = src[i] as f32 * INV_255;
                }
            }
            TensorLayout::Nchw => {
                let plane = out_w * out_h;
                for i in (0..n).step_by(3) {
                    let p = i / 3;
                    *dst.add(p) = src[i + 2] as f32 * INV_255;
                    *dst.add(plane + p) = src[i + 1] as f32 * INV_255;
                    *dst.add(2 * plane + p) = src[i] as f32 * INV_255;
                }
            }
        }
    }

    Ok(applied)
}

/// Build the letterbox mapping for a square model input: the frame is scaled to
/// fit inside `size x size` and centred, with the leftover border filled by the
/// padding colour.
pub(crate) fn letterbox_map(fw: f64, fh: f64, size: usize) -> Affine {
    let size = size as f64;
    let scale = (size / fw).min(size / fh);
    let drawn_w = fw * scale;
    let drawn_h = fh * scale;
    // out_x = scale * frame_x + pad  =>  frame_x = (out_x - pad) / scale
    Affine::scale_translate(1.0 / scale, -((size - drawn_w) * 0.5) / scale, -((size - drawn_h) * 0.5) / scale)
}

/// Sample `frame` through the inverse of `map` using `warpAffine`, which takes the
/// forward frame -> output matrix. Returns the mapping that was applied.
fn warp_affine(
    frame: &Mat,
    map: Affine,
    out_size: core::Size,
    dst: &mut Mat,
    border: f64,
) -> Result<Affine> {
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
        core::Scalar::all(border),
    )?;

    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(rows: i32, cols: i32) -> Mat {
        // Gradient frame: R = 50 everywhere, B grows left to right, G top to bottom.
        let mut mat = Mat::new_rows_cols_with_default(
            rows,
            cols,
            core::CV_8UC3,
            core::Scalar::all(0.0),
        )
        .unwrap();
        for y in 0..rows {
            let row = mat.ptr_mut(y).unwrap();
            unsafe {
                for x in 0..cols {
                    *row.add(x as usize * 3) = (x * 10) as u8; // B
                    *row.add(x as usize * 3 + 1) = (y * 10) as u8; // G
                    *row.add(x as usize * 3 + 2) = 50; // R
                }
            }
        }
        mat
    }

    #[test]
    fn a_letterbox_maps_model_pixels_back_to_the_frame() {
        // 640x480 into 640x640: the frame is drawn at full width, centred, with 80 px
        // of padding above and below.
        let map = letterbox_map(640.0, 480.0, 640);
        assert!((map.m00 - 1.0).abs() < 1e-9);
        assert!((map.frame_x(0.0, 80.0) - 0.0).abs() < 1e-9);
        assert!((map.frame_x(640.0, 80.0) - 640.0).abs() < 1e-9);
        assert!((map.frame_y(0.0, 80.0) - 0.0).abs() < 1e-9);
        assert!((map.frame_y(0.0, 560.0) - 480.0).abs() < 1e-9);
        assert!((map.frame_y(320.0, 320.0) - 240.0).abs() < 1e-9); // dead centre
    }

    #[test]
    fn an_nchw_stage_has_three_colour_planes() {
        let mut stage = StageInput::new(4, 4);
        // A square frame needs no padding, so this takes the resize fast path.
        let applied = sample_affine_to_tensor(
            &frame(4, 4),
            letterbox_map(4.0, 4.0, 4),
            4,
            4,
            &mut stage,
            TensorLayout::Nchw,
            114.0,
        )
        .unwrap();
        assert!((applied.m00 - 1.0).abs() < 1e-9);

        let plane = 4 * 4;
        // Output pixel (0,0) is frame pixel (0,0) = BGR (50, 0, 0) -> R = 50/255.
        assert!((stage.data[0] - 50.0 / 255.0).abs() < 1e-6);
        assert!((stage.data[plane] - 0.0).abs() < 1e-6);
        assert!((stage.data[2 * plane] - 0.0).abs() < 1e-6);
        // Output pixel (ox=3, oy=2) is frame pixel (3, 2): R = 50, G = 20, B = 30.
        let p = 2 * 4 + 3;
        assert!((stage.data[p] - 50.0 / 255.0).abs() < 1e-6);
        assert!((stage.data[plane + p] - 20.0 / 255.0).abs() < 1e-6);
        assert!((stage.data[2 * plane + p] - 30.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn a_frame_smaller_than_the_input_is_padded_with_the_border_colour() {
        let mut stage = StageInput::new(8, 8);
        // A 6x3 frame into 8x8 leaves 2 px of padding top and bottom, so the first and
        // last output rows sample entirely outside the image.
        sample_affine_to_tensor(
            &frame(3, 6),
            letterbox_map(6.0, 3.0, 8),
            8,
            8,
            &mut stage,
            TensorLayout::Nhwc,
            114.0,
        )
        .unwrap();
        assert!((stage.data[0] - 114.0 / 255.0).abs() < 1e-6);
        assert!((stage.data[1] - 114.0 / 255.0).abs() < 1e-6);
        assert!((stage.data[2] - 114.0 / 255.0).abs() < 1e-6);
        let last = (8 * 8 - 1) * 3;
        assert!((stage.data[last] - 114.0 / 255.0).abs() < 1e-6);
    }
}
