//! The webcam, via AVFoundation.
//!
//! `nokhwa` is the only Rust library that talks to macOS' camera API directly, and
//! it has one sharp edge this module papers over: the Mac webcam streams **YUYV
//! (4:2:2)**, and nokhwa's bundled decoder turns that into RGBA through a
//! `flat_map`/`flatten` iterator that cannot vectorise - ~53 ms for a 640x480
//! frame on this machine, which is larger than the whole 33 ms frame budget. The
//! conversion here is a plain indexed pass over the scanline (1.4 ms) and writes
//! RGB straight into the `Frame`, so there is no BGR->RGB swap and no RGBA
//! intermediate.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use nokhwa::{
    utils::{
        CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
    },
    Camera,
};

use crate::core::frame::Frame;

/// Ask the OS for camera access and wait for the answer. The callback is
/// asynchronous, so opening the camera without waiting races the permission
/// prompt on the first run.
fn request_camera_permission() -> bool {
    let (tx, rx) = std::sync::mpsc::channel();
    nokhwa::nokhwa_initialize(move |ok| {
        let _ = tx.send(ok);
    });
    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(ok) => ok,
        Err(_) => {
            tracing::warn!("camera permission prompt did not answer within 10 s");
            false
        }
    }
}

/// YUYV (4:2:2) -> RGB, one indexed pass, no intermediate buffer.
///
/// Each four-byte group carries two pixels that share their chroma:
/// `Y0 U Y1 V`. Rec. 601: R = Y + 1.140 V, G = Y - 0.396 U - 0.581 V,
/// B = Y + 2.032 U. Clamping happens in f32 before the truncation to u8.
fn yuyv_to_rgb(data: &[u8], out: &mut [u8]) {
    for (i, group) in data.as_chunks::<4>().0.iter().enumerate() {
        let y0 = group[0] as f32;
        let u = group[1] as f32 - 128.0;
        let y1 = group[2] as f32;
        let v = group[3] as f32 - 128.0;

        let base = i * 6;
        out[base] = (y0 + 1.140 * v).clamp(0.0, 255.0) as u8;
        out[base + 1] = (y0 - 0.396 * u - 0.581 * v).clamp(0.0, 255.0) as u8;
        out[base + 2] = (y0 + 2.032 * u).clamp(0.0, 255.0) as u8;
        out[base + 3] = (y1 + 1.140 * v).clamp(0.0, 255.0) as u8;
        out[base + 4] = (y1 - 0.396 * u - 0.581 * v).clamp(0.0, 255.0) as u8;
        out[base + 5] = (y1 + 2.032 * u).clamp(0.0, 255.0) as u8;
    }
}

pub struct WebcamSource {
    camera: Camera,
    /// Mirror the frame horizontally. A raw webcam is seen from the camera's point
    /// of view, so the user's right hand lands on the *left* of the image; flipping
    /// it makes the stage behave like a mirror, which is also how the keyboard reads
    /// (low notes left, high notes right).
    mirror: bool,
    requested_fps: u32,
    /// The fourcc the device actually agreed to, resolved once at open time.
    format: FrameFormat,
}

impl WebcamSource {
    /// Opens the camera and asks for a capture resolution and frame rate.
    ///
    /// The request is attempted exactly; if the device will not honour it, the
    /// highest-rate format it offers is used instead and everything downstream
    /// scales to whatever arrives. Pinning the rate matters for CPU: a camera
    /// defaulting to 60 fps would run the ONNX stages twice as often for no gain.
    pub fn new(camera_id: i32, width: u32, height: u32, fps: u32, mirror: bool) -> Result<Self> {
        if !request_camera_permission() {
            return Err(anyhow!(
                "camera access was not granted; allow it under System Settings > Privacy & Security > Camera"
            ));
        }

        let index = CameraIndex::Index(camera_id as u32);
        let requested = CameraFormat::new(Resolution::new(width, height), FrameFormat::YUYV, fps);

        let mut camera = Camera::new(
            index,
            RequestedFormat::new::<nokhwa::pixel_format::RgbAFormat>(RequestedFormatType::Exact(
                requested,
            )),
        )
        .with_context(|| format!("Could not open webcam index {camera_id}"))?;

        if camera.open_stream().is_err() {
            // The exact window is not offered by this device; take the best it has.
            tracing::debug!(
                "webcam {camera_id} does not offer {width}x{height} at {fps} fps, using its highest-rate format"
            );
            camera = Camera::new(
                CameraIndex::Index(camera_id as u32),
                RequestedFormat::new::<nokhwa::pixel_format::RgbAFormat>(
                    RequestedFormatType::AbsoluteHighestFrameRate,
                ),
            )
            .with_context(|| format!("Could not open webcam index {camera_id}"))?;
            camera
                .open_stream()
                .with_context(|| format!("Could not start webcam stream {camera_id}"))?;
        }

        let format = camera.camera_format();
        tracing::info!("webcam {camera_id} opened at {format}");

        Ok(Self {
            camera,
            mirror,
            requested_fps: fps,
            format: format.format(),
        })
    }

    /// The rate the device actually reports (0 when it will not say).
    pub fn reported_fps(&self) -> f64 {
        self.camera.frame_rate() as f64
    }

    pub fn requested_fps(&self) -> u32 {
        self.requested_fps
    }

    /// One captured frame, in the coordinate space the rest of the pipeline works
    /// in (mirrored by default).
    pub fn get_frame(&mut self) -> Result<Frame> {
        let buffer = self
            .camera
            .frame()
            .with_context(|| "Failed to read frame from webcam")?;
        let width = buffer.resolution().width() as usize;
        let height = buffer.resolution().height() as usize;
        let raw = buffer.buffer();

        let mut frame = Frame::new(width, height, vec![0u8; width * height * 3]);
        match self.format {
            FrameFormat::YUYV => yuyv_to_rgb(raw, frame.pixels_mut()),
            FrameFormat::RAWRGB => frame.pixels_mut().copy_from_slice(raw),
            FrameFormat::RAWBGR => {
                let pixels = frame.pixels_mut();
                for (dst, src) in pixels
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(raw.as_chunks::<3>().0)
                {
                    dst[0] = src[2];
                    dst[1] = src[1];
                    dst[2] = src[0];
                }
            }
            FrameFormat::GRAY => {
                let pixels = frame.pixels_mut();
                for (dst, &value) in pixels.as_chunks_mut::<3>().0.iter_mut().zip(raw.iter()) {
                    dst[0] = value;
                    dst[1] = value;
                    dst[2] = value;
                }
            }
            other => {
                return Err(anyhow!("webcam returned an unsupported format: {other}"));
            }
        }

        if self.mirror {
            frame.flip_horizontal();
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuyv_decodes_to_rgb() {
        // Mid-grey with neutral chroma: Y = 128, U = V = 128 -> R = G = B = 128.
        let mut out = [0u8; 6];
        yuyv_to_rgb(&[128, 128, 128, 128], &mut out);
        assert_eq!(out, [128, 128, 128, 128, 128, 128]);
    }

    #[test]
    fn yuyv_clamps_out_of_range_components() {
        let mut out = [0u8; 6];
        // Y = 255, V = 255 pushes R above 255; it must saturate, not wrap.
        yuyv_to_rgb(&[255, 128, 0, 255], &mut out);
        assert_eq!(out[0], 255);
        assert!(out[1] < 255);
    }

    #[test]
    fn mirroring_reverses_the_columns() {
        let mut frame = Frame::new(2, 1, vec![1, 2, 3, 4, 5, 6]);
        frame.flip_horizontal();
        assert_eq!(frame.pixels(), [4, 5, 6, 1, 2, 3]);
    }
}
