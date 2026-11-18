//! Input sampling shared by every ONNX vision stage.
//!
//! A stage takes an axis-aligned crop of the frame (scale + translate, with a fill
//! for the part of the window that falls outside the image) and needs it as a float
//! tensor in either NHWC (MediaPipe) or NCHW (Ultralytics YOLO) layout. The buffers
//! are allocated once per tracker and refilled in place every frame.
//!
//! This used to be `imgproc::resize` plus `imgproc::warp_affine`. Every mapping the
//! pipeline builds is axis-aligned - the `warp_affine` branch was reached only
//! because the crop window ran off the edge of the frame - so the sampling is a
//! bilinear sweep over one buffer. Doing it in a single pass writes the f32 tensor
//! directly: no intermediate 8-bit image, no channel swap, no allocation per frame.

use anyhow::{anyhow, Result};

use crate::core::frame::Frame;

const INV_255: f32 = 1.0 / 255.0;

/// Affine mapping from output (model) pixels to frame pixels:
/// `frame_x = m00 * out_x + m01 * out_y + tx`, and the same for y.
///
/// Every mapping built here is a pure scale and translation; the off-diagonal
/// fields are kept so a rotation or a shear can still be expressed.
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
    /// at (tx, ty). Both MediaPipe stages and the letterbox use this.
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

    /// The frame coordinate of the centre of output pixel (ox, oy).
    ///
    /// A source pixel `j` is taken to cover `[j, j + 1)`, so the centre of output
    /// pixel `i` maps to `tx + scale * (i + 0.5) - 0.5`. With `scale == 1` that is
    /// exactly `tx + i`, which is what makes an identity crop sample whole pixels.
    fn centre(&self, ox: usize, oy: usize) -> (f64, f64) {
        let ox = ox as f64;
        let oy = oy as f64;
        (
            self.tx + self.m00 * (ox + 0.5) + self.m01 * (oy + 0.5) - 0.5,
            self.ty + self.m10 * (ox + 0.5) + self.m11 * (oy + 0.5) - 0.5,
        )
    }

    fn is_axis_aligned(&self) -> bool {
        self.m01.abs() < 1e-9 && self.m10.abs() < 1e-9
    }
}

/// Scratch buffer for one inference stage, allocated once when the tracker is built
/// and refilled in place every frame.
pub(crate) struct StageInput {
    pub(crate) data: Vec<f32>,
}

impl StageInput {
    pub(crate) fn new(out_w: usize, out_h: usize) -> Self {
        StageInput {
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
/// Returns the *effective* mapping that was applied: the integer-crop fast path
/// snaps the window to whole pixels, so callers must use the returned map (not the
/// one they passed in) when converting model coordinates back to frame pixels.
pub(crate) fn sample_affine_to_tensor(
    frame: &Frame,
    map: Affine,
    out_w: usize,
    out_h: usize,
    stage: &mut StageInput,
    layout: TensorLayout,
    border: f64,
) -> Result<Affine> {
    let fw = frame.width();
    let fh = frame.height();
    if fw == 0 || fh == 0 {
        return Err(anyhow!("cannot sample an empty frame"));
    }

    let applied = if map.is_axis_aligned() && map.m00 > 0.0 && map.m11 > 0.0 {
        // Snap the window to whole pixels, then sample. Snapping is what `resize`
        // did, and the snapped mapping is what the caller needs back so its
        // coordinate conversion agrees with the pixels the model actually saw.
        let x0 = map.tx.floor() as i64;
        let y0 = map.ty.floor() as i64;
        let w = (map.m00 * out_w as f64).round() as i64;
        let h = (map.m11 * out_h as f64).round() as i64;

        if x0 >= 0 && y0 >= 0 && w > 0 && h > 0 && (x0 + w) <= fw as i64 && (y0 + h) <= fh as i64 {
            Affine {
                m00: w as f64 / out_w as f64,
                m01: 0.0,
                tx: x0 as f64,
                m10: 0.0,
                m11: h as f64 / out_h as f64,
                ty: y0 as f64,
            }
        } else {
            // The window runs off the edge of the frame; the sampler fills the part
            // that falls outside with `border`, exactly as `warpAffine`'s
            // BORDER_CONSTANT did.
            map
        }
    } else {
        if map.m00 * map.m11 - map.m01 * map.m10 < 1e-12 {
            return Err(anyhow!("degenerate affine mapping"));
        }
        map
    };

    let n = out_h * out_w * 3;
    if stage.data.len() != n {
        stage.data.resize(n, 0.0);
    }

    let border = [(border as f32) * INV_255; 3];
    let pixels = frame.pixels();
    let plane = out_w * out_h;

    for oy in 0..out_h {
        for ox in 0..out_w {
            let rgb = sample_bilinear(pixels, fw, fh, applied.centre(ox, oy), border);
            match layout {
                TensorLayout::Nhwc => {
                    let dst = (oy * out_w + ox) * 3;
                    stage.data[dst] = rgb[0];
                    stage.data[dst + 1] = rgb[1];
                    stage.data[dst + 2] = rgb[2];
                }
                TensorLayout::Nchw => {
                    let p = oy * out_w + ox;
                    stage.data[p] = rgb[0];
                    stage.data[plane + p] = rgb[1];
                    stage.data[2 * plane + p] = rgb[2];
                }
            }
        }
    }

    Ok(applied)
}

/// Bilinear sample of one output pixel centre.
///
/// A centre that falls outside the image is answered with `border`, matching
/// OpenCV's `BORDER_CONSTANT`. Inside the image the two taps on each axis are
/// clamped to the nearest real pixel, so an edge sample replicates the border
/// pixel rather than inventing one.
fn sample_bilinear(
    pixels: &[u8],
    fw: usize,
    fh: usize,
    centre: (f64, f64),
    border: [f32; 3],
) -> [f32; 3] {
    let (fx, fy) = centre;
    if fx < 0.0 || fy < 0.0 || fx >= fw as f64 || fy >= fh as f64 {
        return border;
    }

    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let wx = (fx - x0 as f64) as f32;
    let wy = (fy - y0 as f64) as f32;

    let x1 = (x0 + 1).min(fw - 1);
    let y1 = (y0 + 1).min(fh - 1);

    let w00 = (1.0 - wx) * (1.0 - wy);
    let w10 = wx * (1.0 - wy);
    let w01 = (1.0 - wx) * wy;
    let w11 = wx * wy;

    let a = (y0 * fw + x0) * 3;
    let b = (y0 * fw + x1) * 3;
    let c = (y1 * fw + x0) * 3;
    let d = (y1 * fw + x1) * 3;

    [
        (w00 * pixels[a] as f32
            + w10 * pixels[b] as f32
            + w01 * pixels[c] as f32
            + w11 * pixels[d] as f32)
            * INV_255,
        (w00 * pixels[a + 1] as f32
            + w10 * pixels[b + 1] as f32
            + w01 * pixels[c + 1] as f32
            + w11 * pixels[d + 1] as f32)
            * INV_255,
        (w00 * pixels[a + 2] as f32
            + w10 * pixels[b + 2] as f32
            + w01 * pixels[c + 2] as f32
            + w11 * pixels[d + 2] as f32)
            * INV_255,
    ]
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
    Affine::scale_translate(
        1.0 / scale,
        -((size - drawn_w) * 0.5) / scale,
        -((size - drawn_h) * 0.5) / scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(rows: usize, cols: usize) -> Frame {
        // Gradient frame: R = 50 everywhere, B grows left to right, G top to bottom.
        let mut pixels = vec![0u8; rows * cols * 3];
        for y in 0..rows {
            for x in 0..cols {
                let i = (y * cols + x) * 3;
                pixels[i] = 50; // R
                pixels[i + 1] = (y * 10) as u8; // G
                pixels[i + 2] = (x * 10) as u8; // B
            }
        }
        Frame::new(cols, rows, pixels)
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
    fn an_identity_crop_samples_whole_pixels() {
        let mut stage = StageInput::new(4, 4);
        let applied = sample_affine_to_tensor(
            &frame(4, 4),
            letterbox_map(4.0, 4.0, 4),
            4,
            4,
            &mut stage,
            TensorLayout::Nhwc,
            114.0,
        )
        .unwrap();
        assert!((applied.m00 - 1.0).abs() < 1e-9);
        // Output pixel (3, 2) is frame pixel (3, 2): R = 50, G = 20, B = 30.
        let dst = (2 * 4 + 3) * 3;
        assert!((stage.data[dst] - 50.0 / 255.0).abs() < 1e-6);
        assert!((stage.data[dst + 1] - 20.0 / 255.0).abs() < 1e-6);
        assert!((stage.data[dst + 2] - 30.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn an_nchw_stage_has_three_colour_planes() {
        let mut stage = StageInput::new(4, 4);
        sample_affine_to_tensor(
            &frame(4, 4),
            letterbox_map(4.0, 4.0, 4),
            4,
            4,
            &mut stage,
            TensorLayout::Nchw,
            114.0,
        )
        .unwrap();
        let plane = 4 * 4;
        let p = 2 * 4 + 3; // output pixel (3, 2) = frame pixel (3, 2)
        assert!((stage.data[p] - 50.0 / 255.0).abs() < 1e-6);
        assert!((stage.data[plane + p] - 20.0 / 255.0).abs() < 1e-6);
        assert!((stage.data[2 * plane + p] - 30.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn a_downsample_averages_the_pixels_a_cell_covers() {
        // 4x4 -> 2x2: output (0,0) covers frame pixels (0,0), (1,0), (0,1), (1,1).
        let mut stage = StageInput::new(2, 2);
        let applied = sample_affine_to_tensor(
            &frame(4, 4),
            Affine::scale_translate(2.0, 0.0, 0.0),
            2,
            2,
            &mut stage,
            TensorLayout::Nhwc,
            0.0,
        )
        .unwrap();
        assert!((applied.m00 - 2.0).abs() < 1e-9);
        assert!((stage.data[0] - 50.0 / 255.0).abs() < 1e-6);
        assert!((stage.data[1] - 5.0 / 255.0).abs() < 1e-6); // G: (0 + 0 + 10 + 10) / 4
        assert!((stage.data[2] - 5.0 / 255.0).abs() < 1e-6); // B: (0 + 10 + 0 + 10) / 4
    }

    #[test]
    fn a_rounded_crop_reports_the_effective_mapping() {
        // A window that is not an integer number of pixels wide is snapped to whole
        // pixels, and the snapped mapping is what the caller gets back so its
        // coordinate conversion agrees with the pixels the model actually saw.
        let mut stage = StageInput::new(3, 3);
        let applied = sample_affine_to_tensor(
            &frame(12, 12),
            Affine::scale_translate(3.33, 1.4, 2.6),
            3,
            3,
            &mut stage,
            TensorLayout::Nhwc,
            0.0,
        )
        .unwrap();
        assert_eq!(applied.tx, 1.0);
        assert_eq!(applied.ty, 2.0);
        assert!((applied.m00 - 10.0 / 3.0).abs() < 1e-9); // round(3.33 * 3) = 10
        assert!((applied.m11 - 10.0 / 3.0).abs() < 1e-9);
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

    #[test]
    fn a_crop_running_off_the_frame_is_padded_with_black() {
        // The detector window can slide so its square hangs off the edge.
        let mut stage = StageInput::new(4, 4);
        sample_affine_to_tensor(
            &frame(4, 4),
            Affine::scale_translate(2.0, 4.0, 0.0), // window starts at the right edge
            4,
            4,
            &mut stage,
            TensorLayout::Nhwc,
            0.0,
        )
        .unwrap();
        // Every output pixel is outside the 4x4 image, so all of it is padding.
        assert!(stage.data.iter().all(|&v| v == 0.0));
    }
}
