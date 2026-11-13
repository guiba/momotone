//! Presentation: everything that turns a frame into pixels or glyphs for a human.
//!
//! This layer has its own thread for two reasons. First, drawing the preview,
//! quantising the frame into ASCII art, printing to a terminal and writing a dump
//! file are all things the instrument does not need in order to make sound, and
//! each of them can block - a GPU texture upload, a resize, a disk write, a
//! terminal flush. Second, `present` is a pure function of a frame, a settings
//! snapshot and a clock, which is what makes the whole layer testable without a
//! webcam.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::Receiver;
use opencv::prelude::*;

use crate::core::runtime::RuntimeConfig;
use crate::core::Landmark;
use crate::dashboard::metrics::FrameMetrics;
use crate::presentation::ascii::{AsciiGrid, AsciiLayer, Ramp};

/// Default camera preview rate: 15 fps is plenty for a picture, and every upload is
/// a blocking round trip to the renderer.
pub const DEFAULT_FEED_FPS: f32 = 15.0;

/// One frame's worth of material for the presenter: pixels if the source has any,
/// and the skeleton in mock mode, where there are none.
pub struct PresentJob {
    pub frame: Option<Mat>,
    pub landmarks: Vec<Landmark>,
    pub frame_size: (usize, usize),
}

pub struct Presenter {
    /// Per-cell EMA state so the art settles instead of flickering.
    ascii: AsciiLayer,
    /// Columns of the ASCII grid; rows follow from the frame aspect.
    cols: usize,
    /// Terminal mode: print the art to stdout so the renderer can be checked
    /// without a window (`--no-ui --ascii`).
    stdout: bool,
    /// Optional path to write each rendered ASCII frame to.
    dump: Option<String>,
    /// Wall clock at which the next camera frame is allowed to go to the UI. Seeded
    /// with "now" so the very first frame is uploaded.
    next_feed: Instant,
    next_ascii: Instant,
}

impl Presenter {
    pub fn new(cols: usize, stdout: bool, dump: Option<String>) -> Self {
        let now = Instant::now();
        Self {
            ascii: AsciiLayer::new(),
            cols,
            stdout,
            dump,
            next_feed: now,
            next_ascii: now,
        }
    }

    /// Is anyone looking at anything? The frame thread uses this to avoid handing
    /// over a megabyte of pixels nobody asked for. `feed` is the toggle *and* the
    /// existence of a stage: a headless run has no window to draw into.
    pub fn wanted(feed: bool, ascii: bool, stdout: bool, dump: &Option<String>) -> bool {
        feed || ascii || stdout || dump.is_some()
    }

    /// Drain the job channel until the pipeline closes it. Runs on its own thread.
    pub fn run(
        mut self,
        shared: Arc<Mutex<FrameMetrics>>,
        control: Arc<Mutex<RuntimeConfig>>,
        jobs: Receiver<PresentJob>,
    ) {
        while let Ok(job) = jobs.recv() {
            let settings = control
                .lock()
                .map(|config| config.clone())
                .unwrap_or_else(|_| {
                    RuntimeConfig::new(DEFAULT_FEED_FPS, 15.0, false, Ramp::Blocks)
                });

            let mut shared = match shared.lock() {
                Ok(shared) => shared,
                Err(_) => continue,
            };
            if let Err(error) = self.present(&job, &settings, Instant::now(), &mut shared) {
                tracing::warn!("presentation failed: {error}");
            }
        }
    }

    /// One frame, two outputs: the preview pixels and the ASCII grid. Both have
    /// their own cadence, so a frame that is not due changes nothing. `now` is a
    /// parameter rather than `Instant::now()` so the cadence can be driven from a
    /// test without sleeping.
    pub fn present(
        &mut self,
        job: &PresentJob,
        settings: &RuntimeConfig,
        now: Instant,
        metrics: &mut FrameMetrics,
    ) -> Result<()> {
        // The art has its own cadence and its own toggle: it is a deliberately coarse
        // view of the frame, so building it costs a resize plus one pass per cell
        // instead of a 1.2 MB texture upload, and it can update slower than the
        // skeleton without looking stale.
        let art_wanted = settings.ascii || self.stdout || self.dump.is_some();
        if art_wanted && feed_due(self.next_ascii, now) {
            self.next_ascii = now + feed_interval(settings.ascii_fps);
            metrics.scene.ascii = match &job.frame {
                Some(frame) => self.ascii.render(frame, self.cols).unwrap_or_else(|error| {
                    tracing::warn!("ascii render failed: {error}");
                    AsciiGrid::default()
                }),
                // Mock mode has no camera pixels, so the art comes from the
                // synthetic skeleton - the toggle still demos without a webcam.
                None if !job.landmarks.is_empty() => {
                    self.ascii
                        .render_landmarks(&job.landmarks, job.frame_size, self.cols)
                }
                None => AsciiGrid::default(),
            };

            if !metrics.scene.ascii.is_empty() {
                let ramp = settings.ramp;
                if self.stdout {
                    // Home the cursor rather than clearing the screen, so
                    // consecutive frames overwrite in place and the art animates
                    // instead of scrolling.
                    print!("\x1B[H{}", metrics.scene.ascii.text(ramp));
                }
                if let Some(path) = &self.dump {
                    std::fs::write(path, metrics.scene.ascii.text(ramp))?;
                }
            }
        }

        if settings.feed && feed_due(self.next_feed, now) {
            if let Some(frame) = &job.frame {
                self.next_feed = now + feed_interval(settings.feed_fps);
                metrics.scene.pixels = frame_to_rgba(frame);
            }
        } else {
            // Keep the snapshot free of pixels the UI would discard.
            metrics.scene.pixels.clear();
        }
        Ok(())
    }
}

/// `fps <= 0` means "every frame". The interval is clamped to a minute so an absurd
/// rate cannot overflow `Instant + Duration`.
fn feed_interval(fps: f32) -> Duration {
    if fps > 0.0 {
        Duration::from_secs_f64((1.0 / fps as f64).min(60.0))
    } else {
        Duration::ZERO
    }
}

/// Is a frame due? `next` is the wall clock at which the output is allowed; an
/// interval of zero makes every frame due.
fn feed_due(next: Instant, now: Instant) -> bool {
    now >= next
}

/// BGR OpenCV frame -> RGBA bytes for the iced `image` handle.
fn frame_to_rgba(frame: &Mat) -> Vec<u8> {
    let count = (frame.cols() as usize) * (frame.rows() as usize) * (frame.channels() as usize);
    // Safe: `frame` is a captured, contiguous Mat, so its buffer holds exactly
    // `count` bytes starting at `data()`.
    let src = unsafe { std::slice::from_raw_parts(frame.data(), count) };
    bgr_to_rgba(src)
}

fn bgr_to_rgba(src: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(src.len() / 3 * 4);
    for pixel in src.as_chunks::<3>().0 {
        rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], 255]);
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::runtime::RuntimeConfig;
    use crate::presentation::ascii::DEFAULT_COLS;
    use opencv::core::{Mat, Scalar};

    fn settings(feed: bool, ascii: bool, feed_fps: f32, ascii_fps: f32) -> RuntimeConfig {
        RuntimeConfig {
            active: true,
            feed,
            ascii,
            feed_fps,
            ascii_fps,
            ramp: Ramp::Blocks,
        }
    }

    fn frame_of(value: u8) -> Mat {
        Mat::new_rows_cols_with_default(
            480,
            640,
            opencv::core::CV_8UC3,
            Scalar::new(value as f64, value as f64, value as f64, 0.0),
        )
        .unwrap()
    }

    fn job(value: u8) -> PresentJob {
        PresentJob {
            frame: Some(frame_of(value)),
            landmarks: Vec::new(),
            frame_size: (640, 480),
        }
    }

    #[test]
    fn bgr_pixels_become_rgba() {
        // Two BGR pixels.
        assert_eq!(
            bgr_to_rgba(&[1, 2, 3, 4, 5, 6]),
            vec![3, 2, 1, 255, 6, 5, 4, 255]
        );
    }

    #[test]
    fn the_preview_is_only_converted_when_the_stage_shows_it() {
        let mut presenter = Presenter::new(DEFAULT_COLS, false, None);
        let mut metrics = FrameMetrics::default();
        let now = Instant::now();

        presenter
            .present(
                &job(10),
                &settings(false, false, 15.0, 15.0),
                now,
                &mut metrics,
            )
            .unwrap();
        assert!(metrics.scene.pixels.is_empty(), "feed off, no pixels");

        presenter
            .present(
                &job(10),
                &settings(true, false, 15.0, 15.0),
                now,
                &mut metrics,
            )
            .unwrap();
        assert_eq!(metrics.scene.pixels.len(), 640 * 480 * 4);
        assert_eq!(
            &metrics.scene.pixels[..4],
            &[10, 10, 10, 255],
            "BGR became RGBA"
        );
    }

    #[test]
    fn the_upload_is_throttled_to_its_cadence() {
        let mut presenter = Presenter::new(DEFAULT_COLS, false, None);
        let mut metrics = FrameMetrics::default();
        let settings = settings(true, false, 1.0, 15.0);
        let now = Instant::now();

        presenter
            .present(&job(10), &settings, now, &mut metrics)
            .unwrap();
        assert_eq!(
            metrics.scene.pixels.len(),
            640 * 480 * 4,
            "the first frame uploads"
        );

        // One frame later (33 ms) the 1 fps window has not elapsed, so the
        // conversion is skipped entirely - that is the 1.2 MB per frame the
        // throttle exists to avoid.
        presenter
            .present(
                &job(255),
                &settings,
                now + Duration::from_millis(33),
                &mut metrics,
            )
            .unwrap();
        assert!(
            metrics.scene.pixels.is_empty(),
            "a frame inside the feed interval carries no pixels"
        );

        presenter
            .present(
                &job(255),
                &settings,
                now + Duration::from_millis(1100),
                &mut metrics,
            )
            .unwrap();
        assert_eq!(
            &metrics.scene.pixels[..4],
            &[255, 255, 255, 255],
            "the second is due"
        );
    }

    #[test]
    fn the_art_only_runs_when_something_wants_to_see_it() {
        assert!(
            !Presenter::wanted(false, false, false, &None),
            "a cleared toggle and no sink means no work"
        );
        assert!(
            Presenter::wanted(true, false, false, &None),
            "the stage is drawing it"
        );
        assert!(
            Presenter::wanted(false, false, false, &Some("art.txt".to_string())),
            "a dump file is a reason to build the grid even headless"
        );
        assert!(
            Presenter::wanted(false, false, true, &None),
            "stdout counts too"
        );
    }

    #[test]
    fn a_flat_frame_produces_a_flat_grid_at_the_requested_resolution() {
        let mut presenter = Presenter::new(32, false, None);
        let mut metrics = FrameMetrics::default();

        presenter
            .present(
                &job(128),
                &settings(false, true, 15.0, 15.0),
                Instant::now(),
                &mut metrics,
            )
            .unwrap();

        // 640x480 at 32 columns is 24 rows, and the frame is one flat grey, so every
        // cell has to land in the same bucket.
        assert_eq!(
            (metrics.scene.ascii.cols, metrics.scene.ascii.rows),
            (32, 24)
        );
        assert!(
            metrics
                .scene
                .ascii
                .cells
                .iter()
                .all(|&level| level == metrics.scene.ascii.cells[0]),
            "a flat frame must produce a flat grid"
        );
    }

    #[test]
    fn mock_mode_draws_ascii_from_the_synthetic_hand() {
        let mut presenter = Presenter::new(40, false, None);
        let mut metrics = FrameMetrics::default();
        let landmarks = crate::pipeline::synthetic_detection(0.0)
            .hand_landmarks
            .unwrap();
        let job = PresentJob {
            frame: None,
            landmarks,
            frame_size: (640, 480),
        };

        presenter
            .present(
                &job,
                &settings(false, true, 0.0, 0.0),
                Instant::now(),
                &mut metrics,
            )
            .unwrap();

        assert_eq!(
            (metrics.scene.ascii.cols, metrics.scene.ascii.rows),
            (40, 30)
        );
        assert!(
            metrics.scene.ascii.cells.iter().any(|&level| level > 0),
            "the mock hand lights up cells"
        );
        assert!(
            metrics.scene.ascii.cells.contains(&0),
            "and the background stays dark"
        );
    }

    #[test]
    fn the_grid_keeps_its_render_until_its_own_cadence_is_due() {
        let mut presenter = Presenter::new(16, false, None);
        let mut metrics = FrameMetrics::default();
        let settings = settings(false, true, 15.0, 1.0);
        let now = Instant::now();

        presenter
            .present(&job(255), &settings, now, &mut metrics)
            .unwrap();
        let lit = metrics.scene.ascii.cells.iter().all(|&level| level > 0);

        // A second frame inside the interval must not rebuild the art.
        presenter
            .present(
                &job(0),
                &settings,
                now + Duration::from_millis(33),
                &mut metrics,
            )
            .unwrap();
        assert!(lit, "the first render lit the whole grid");
        assert!(
            metrics.scene.ascii.cells.iter().all(|&level| level > 0),
            "the grid keeps the previous render until its cadence is due"
        );
    }

    #[test]
    #[ignore]
    /// What converting a 640x480 frame costs: the number the feed throttle saves.
    /// Run with
    /// `cargo test --release -- --include-ignored bench_feed_upload --nocapture`.
    fn bench_feed_upload() {
        let frame = frame_of(128);
        let start = Instant::now();
        for _ in 0..200 {
            let pixels = frame_to_rgba(&frame);
            assert_eq!(pixels.len(), 640 * 480 * 4);
        }
        println!(
            "frame_to_rgba 640x480: {:.3} ms/frame",
            start.elapsed().as_secs_f64() * 1000.0 / 200.0
        );
    }
}
