use anyhow::{anyhow, Result};
use opencv::core;
use opencv::prelude::*;
use opencv::videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst};

pub struct WebcamSource {
    cap: videoio::VideoCapture,
    /// Mirror the frame horizontally. A raw webcam is seen from the camera's
    /// point of view, so the user's right hand lands on the *left* of the
    /// image; flipping it makes the stage behave like a mirror, which is also
    /// how the keyboard reads (low notes left, high notes right).
    mirror: bool,
}

impl WebcamSource {
    /// Opens the camera and asks for a capture resolution. Most webcams honour
    /// this; if they do not, the native resolution is used and everything still
    /// works (the dashboard scales to whatever arrives).
    pub fn new(camera_id: i32, width: u32, height: u32, mirror: bool) -> Result<Self> {
        let mut cap = videoio::VideoCapture::new(camera_id, videoio::CAP_ANY)?;
        if !cap.is_opened()? {
            return Err(anyhow!("Could not open webcam index {}", camera_id));
        }
        // Best effort: not every backend honours these.
        let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, width as f64);
        let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, height as f64);
        Ok(Self { cap, mirror })
    }

    /// One captured frame, in the coordinate space the rest of the pipeline
    /// works in (mirrored by default).
    pub fn get_frame(&mut self) -> Result<Mat> {
        let mut frame = Mat::default();
        if !self.cap.read(&mut frame)? {
            return Err(anyhow!("Failed to read frame from webcam"));
        }
        if self.mirror {
            // flip_code 1 is cv::FLIP_HORIZONTAL; the bindings expose no constant.
            let mut flipped = Mat::default();
            core::flip(&frame, &mut flipped, 1)?;
            return Ok(flipped);
        }
        Ok(frame)
    }
}
