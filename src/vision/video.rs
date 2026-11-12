//! A file-backed frame source.
//!
//! A recorded clip is the only way to make a run reproducible: the same frames,
//! the same detector output, the same note sequence, every time. That is what
//! makes a golden trace test possible, and it is why this sits next to the webcam
//! rather than replacing it.
//!
//! Frames are paced by a deadline clock at the clip's own frame rate, so the
//! telemetry a fixture produces is comparable with a live camera instead of a
//! "read as fast as OpenCV can" burst.

use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use opencv::prelude::*;
use opencv::videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst};

use crate::vision::camera::flip_horizontal;

/// Frames of a clip with no usable rate metadata are paced like a camera.
const FALLBACK_FPS: f64 = 30.0;

/// The deadline interval a clip should be read at. Kept separate from the source
/// so the pacing rule itself can be tested.
fn cadence(reported_fps: f64) -> Duration {
    let fps = if reported_fps.is_finite() && reported_fps > 0.0 {
        reported_fps
    } else {
        FALLBACK_FPS
    };
    Duration::from_secs_f64(1.0 / fps)
}

pub struct VideoSource {
    cap: videoio::VideoCapture,
    /// Same mirroring rule as the webcam: a fixture recorded from a webcam is
    /// still seen from the camera's point of view.
    mirror: bool,
    flip_scratch: Mat,
    /// Rate the loop is paced at, from the clip's metadata.
    cadence: Duration,
    /// Deadline of the next frame. One clock per source keeps the pacing honest.
    next: Instant,
    frames_read: u64,
}

impl VideoSource {
    pub fn new(path: &str, mirror: bool) -> Result<Self> {
        // The default constructor makes an unopened capture; `open_file` then points
        // it at the clip. `new(-1, CAP_ANY)` would try to attach a camera first.
        let mut cap = videoio::VideoCapture::default()?;
        cap.open_file(path, videoio::CAP_ANY)?;
        if !cap.is_opened()? {
            return Err(anyhow!("Could not open video {path}"));
        }
        // Clips recorded at 30 fps are read at 30 fps; anything faster is
        // throttled, anything slower runs at its own rate.
        let reported = cap.get(videoio::CAP_PROP_FPS).unwrap_or(0.0);
        Ok(Self {
            cap,
            mirror,
            flip_scratch: Mat::default(),
            cadence: cadence(reported),
            next: Instant::now(),
            frames_read: 0,
        })
    }

    pub fn fps(&self) -> f64 {
        1.0 / self.cadence.as_secs_f64()
    }

    /// Sleep until the clip's frame deadline. Kept separate from `read` so the
    /// capture stage measures decoding, not pacing.
    pub fn wait_for_next(&mut self) {
        let now = Instant::now();
        if now < self.next {
            std::thread::sleep(self.next - now);
        }
        self.next += self.cadence;
    }

    /// The next frame, or `Err` when the clip is exhausted - which is how the
    /// capture thread knows the run is over.
    pub fn read(&mut self) -> Result<Mat> {
        let mut frame = Mat::default();
        if !self.cap.read(&mut frame)? {
            return Err(anyhow!("End of clip after {} frames", self.frames_read));
        }
        if self.mirror {
            flip_horizontal(&mut frame, &mut self.flip_scratch)?;
        }
        self.frames_read += 1;
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clip_with_a_rate_is_read_at_that_rate() {
        assert_eq!(cadence(60.0), Duration::from_secs_f64(1.0 / 60.0));
    }

    #[test]
    fn a_clip_without_a_rate_is_paced_like_a_camera() {
        for reported in [0.0, -1.0, f64::NAN] {
            assert_eq!(cadence(reported), Duration::from_secs_f64(1.0 / 30.0));
        }
    }

    #[test]
    fn a_missing_clip_names_the_file_it_could_not_open() {
        match VideoSource::new("definitely-not-a-clip.mp4", false) {
            // Some backends open lazily, so an Ok here is not a failure - the read
            // is the real gate.
            Ok(_) => {}
            Err(error) => assert!(
                error.to_string().contains("definitely-not-a-clip.mp4"),
                "the error should name the file, got {error}"
            ),
        }
    }
}
