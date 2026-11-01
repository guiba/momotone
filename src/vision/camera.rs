use anyhow::{anyhow, Result};
use opencv::core;
use opencv::prelude::*;
use opencv::videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst};

/// Reverse the columns of `frame`, writing into `scratch`.
///
/// OpenCV's `flip` needs a destination Mat, and the bindings take two separate
/// borrows, so `flip(frame, frame, 1)` cannot be expressed. Handing it a buffer
/// that already exists does the same job for less: OpenCV keeps the scratch
/// allocation when the size matches, so after the first frame the mirror is one
/// row swap and no allocator traffic.
fn flip_horizontal(frame: &mut Mat, scratch: &mut Mat) -> Result<()> {
    // flip_code 1 is cv::FLIP_HORIZONTAL; the bindings expose no constant.
    core::flip(frame, scratch, 1)?;
    // The caller keeps the flipped pixels, and `scratch` is left holding the raw
    // frame — which is exactly the buffer the next capture reads into.
    std::mem::swap(frame, scratch);
    Ok(())
}

pub struct WebcamSource {
    cap: videoio::VideoCapture,
    /// Mirror the frame horizontally. A raw webcam is seen from the camera's
    /// point of view, so the user's right hand lands on the *left* of the
    /// image; flipping it makes the stage behave like a mirror, which is also
    /// how the keyboard reads (low notes left, high notes right).
    mirror: bool,
    /// Destination for the mirror flip: allocated once, swapped through every frame.
    flip_scratch: Mat,
    /// Capture rate asked for at open time (`--camera-fps`).
    requested_fps: u32,
}

impl WebcamSource {
    /// Opens the camera and asks for a capture resolution and frame rate. Most
    /// webcams honour these; if they do not, the native settings are used and
    /// everything still works (the dashboard scales to whatever arrives).
    ///
    /// Pinning the rate matters for CPU: a camera defaulting to 60 fps would run
    /// the ONNX stages twice as often for no gain, since the whole loop is paced
    /// by whatever the device delivers.
    pub fn new(camera_id: i32, width: u32, height: u32, fps: u32, mirror: bool) -> Result<Self> {
        let mut cap = videoio::VideoCapture::new(camera_id, videoio::CAP_ANY)?;
        if !cap.is_opened()? {
            return Err(anyhow!("Could not open webcam index {}", camera_id));
        }
        // Best effort: not every backend honours these.
        let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, width as f64);
        let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, height as f64);
        let _ = cap.set(videoio::CAP_PROP_FPS, fps as f64);
        Ok(Self {
            cap,
            mirror,
            flip_scratch: Mat::default(),
            requested_fps: fps,
        })
    }

    /// The rate the driver actually reports (0 when it will not say).
    pub fn reported_fps(&self) -> f64 {
        self.cap.get(videoio::CAP_PROP_FPS).unwrap_or(0.0)
    }

    pub fn requested_fps(&self) -> u32 {
        self.requested_fps
    }

    /// One captured frame, in the coordinate space the rest of the pipeline
    /// works in (mirrored by default).
    pub fn get_frame(&mut self) -> Result<Mat> {
        let mut frame = Mat::default();
        if !self.cap.read(&mut frame)? {
            return Err(anyhow!("Failed to read frame from webcam"));
        }
        if self.mirror {
            flip_horizontal(&mut frame, &mut self.flip_scratch)?;
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    /// 2x3 single-channel frame, one distinct value per column, so a reversal
    /// is unambiguous.
    fn stripe_frame() -> Mat {
        Mat::from_slice_2d(&[[1u8, 2, 3], [4, 5, 6]]).unwrap()
    }

    fn pixels(frame: &Mat) -> Vec<u8> {
        (0..frame.rows())
            .flat_map(|r| {
                let row = frame.ptr(r).unwrap();
                (0..frame.cols()).map(move |c| unsafe { *row.add(c as usize) })
            })
            .collect()
    }

    #[test]
    fn mirroring_reverses_the_columns() {
        let mut frame = stripe_frame();
        let mut scratch = Mat::default();
        flip_horizontal(&mut frame, &mut scratch).unwrap();
        assert_eq!(pixels(&frame), [3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn a_second_flip_reuses_the_scratch_buffer_instead_of_allocating() {
        let mut frame = stripe_frame();
        let original = frame.data();
        let mut scratch = Mat::default();

        flip_horizontal(&mut frame, &mut scratch).unwrap();
        flip_horizontal(&mut frame, &mut scratch).unwrap();

        assert_eq!(
            frame.data(),
            original,
            "the scratch Mat keeps its allocation across frames"
        );
        assert_eq!(
            pixels(&frame),
            [1, 2, 3, 4, 5, 6],
            "flipping twice is the identity"
        );
    }

    #[test]
    #[ignore]
    /// Quantifies the mirror flip. Run with
    /// `cargo test --release -- --include-ignored bench_mirror_flip --nocapture`.
    fn bench_mirror_flip() {
        const N: u32 = 2000;
        let frame =
            Mat::new_rows_cols_with_default(480, 640, core::CV_8UC3, core::Scalar::all(127.0))
                .unwrap();

        // Old path: a brand-new destination Mat every frame.
        let start = Instant::now();
        for _ in 0..N {
            let mut flipped = Mat::default();
            core::flip(&frame, &mut flipped, 1).unwrap();
            black_box(flipped.data());
        }
        let fresh = start.elapsed().as_secs_f64() * 1000.0 / N as f64;

        // New path: one scratch buffer, swapped in and out.
        let mut working = frame.clone();
        let mut scratch = Mat::default();
        let start = Instant::now();
        for _ in 0..N {
            flip_horizontal(&mut working, &mut scratch).unwrap();
            black_box(working.data());
        }
        let reused = start.elapsed().as_secs_f64() * 1000.0 / N as f64;

        println!(
            "mirror flip 640x480: fresh Mat {:.4} ms/frame, reused scratch {:.4} ms/frame",
            fresh, reused
        );
    }
}
