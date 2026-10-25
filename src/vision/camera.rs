use anyhow::{anyhow, Result};
use opencv::prelude::*;
use opencv::videoio;


pub struct WebcamSource {
    cap: videoio::VideoCapture,
}

impl WebcamSource {
    pub fn new(camera_id: i32) -> Result<Self> {
        let cap = videoio::VideoCapture::new(camera_id, videoio::CAP_ANY)?;
        if !cap.is_opened()? {
            return Err(anyhow!("Could not open webcam index {}", camera_id));
        }
        Ok(Self { cap })
    }

    pub fn get_frame(&mut self) -> Result<Mat> {
        let mut frame = Mat::default();
        if !self.cap.read(&mut frame)? {
            return Err(anyhow!("Failed to read frame from webcam"));
        }
        Ok(frame)
    }
}
