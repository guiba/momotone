//! The capture -> inference -> gesture -> sink loop, decoupled from the UI.
//!
//! `Pipeline::tick` advances exactly one frame and publishes a `FrameMetrics`
//! snapshot. `main` either runs it inline (`--no-ui`) or on a dedicated thread
//! so the iced dashboard never blocks the camera.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::Receiver;
use opencv::prelude::*;

use crate::core::event_bus::EventBus;
use crate::core::geometry::hand_geometry;
use crate::core::types::{GestureEvent, GesturePayload};
use crate::core::{HandDetectionResult, Landmark, Sink, Strategy};
use crate::dashboard::metrics::{EventLogEntry, FrameMetrics, LOG_CAPACITY};
use crate::vision::camera::WebcamSource;
use crate::vision::tracker::HandTrackerImpl;

/// Mock mode emulates a 30 fps camera so the telemetry looks real.
const MOCK_CADENCE_MS: u64 = 33;
/// How often the paused loop checks the Start/Stop flag and refreshes the UI.
const PAUSE_POLL_MS: u64 = 50;

pub struct Pipeline {
    camera: Option<WebcamSource>,
    tracker: Option<HandTrackerImpl>,
    strategy: Box<dyn Strategy>,
    sinks: Vec<Box<dyn Sink>>,
    event_bus: EventBus,
    event_rx: Receiver<GestureEvent>,
    shared: Arc<Mutex<FrameMetrics>>,
    metrics: FrameMetrics,
    previous_landmarks: Vec<Landmark>,
    /// True while a gesture is actively driving the sinks. Used to send exactly
    /// one release event when it stops, so the audio/MIDI sinks do not latch the
    /// last note and keep sounding after the hand leaves the frame.
    gesture_active: bool,
    last_detection: Instant,
    started: Instant,
    max_frames: Option<u32>,
    debug: bool,
    emit_frames: bool,
    /// Set by the dashboard's Start/Stop button. While it is false the loop
    /// captures nothing and runs no inference.
    active: Arc<AtomicBool>,
    /// Set by the dashboard's `feed` toggle. The camera frame is only converted
    /// to RGBA (1.2 MB per frame at 640x480) when the stage actually shows it.
    feed_wanted: Arc<AtomicBool>,
}

impl Pipeline {
    pub fn new(
        camera: Option<WebcamSource>,
        tracker: Option<HandTrackerImpl>,
        strategy: Box<dyn Strategy>,
        sinks: Vec<Box<dyn Sink>>,
        event_bus: EventBus,
        event_rx: Receiver<GestureEvent>,
        shared: Arc<Mutex<FrameMetrics>>,
        max_frames: Option<u32>,
        debug: bool,
        emit_frames: bool,
        active: Arc<AtomicBool>,
        feed_wanted: Arc<AtomicBool>,
    ) -> Self {
        Pipeline {
            camera,
            tracker,
            strategy,
            sinks,
            event_bus,
            event_rx,
            shared,
            metrics: FrameMetrics::default(),
            previous_landmarks: Vec::new(),
            gesture_active: false,
            last_detection: Instant::now(),
            started: Instant::now(),
            max_frames,
            debug,
            emit_frames,
            active,
            feed_wanted,
        }
    }

    /// Runs until the UI asks for shutdown or `--max-frames` is reached. The
    /// `active` flag (the dashboard's Stop button) pauses the loop without
    /// tearing down the camera or the sinks.
    pub fn run(&mut self, running: &AtomicBool) -> Result<()> {
        while running.load(Ordering::Relaxed) {
            if !self.active.load(Ordering::Relaxed) {
                self.pause();
                std::thread::sleep(Duration::from_millis(PAUSE_POLL_MS));
                continue;
            }

            if !self.tick()? {
                break;
            }
            if self
                .max_frames
                .is_some_and(|n| self.metrics.frames >= n as u64)
            {
                break;
            }
        }

        for sink in &mut self.sinks {
            let _ = sink.stop();
        }
        Ok(())
    }

    /// While stopped, release whatever the sinks are holding and publish a
    /// "paused" snapshot once, so the note does not latch and the keyboard trail
    /// fades to nothing.
    fn pause(&mut self) {
        if self.gesture_active {
            self.gesture_active = false;
            self.event_bus.dispatch(GestureEvent::no_gesture());
        }
        while let Ok(event) = self.event_rx.try_recv() {
            for sink in &mut self.sinks {
                let _ = sink.handle_event(&event);
            }
        }

        if self.metrics.paused {
            return;
        }
        self.metrics.paused = true;
        self.metrics.detected = false;
        self.metrics.note = None;
        self.metrics.volume = 0.0;
        self.metrics.frequency = None;
        self.metrics.landmarks.clear();
        self.metrics.landmark_count = 0;
        self.metrics.hand_box = None;
        self.metrics.pitch_x = 0.0;
        if let Ok(mut shared) = self.shared.lock() {
            *shared = self.metrics.clone();
        }
    }

    /// One camera frame end to end. Returns `false` when the camera is gone.
    pub fn tick(&mut self) -> Result<bool> {
        // Mock mode has no camera, so pace it like one to keep the telemetry
        // (fps, jitter, history) realistic.
        if self.camera.is_none() {
            std::thread::sleep(Duration::from_millis(MOCK_CADENCE_MS));
        }

        self.metrics.frames += 1;
        self.metrics.uptime_ms = self.started.elapsed().as_millis() as u64;
        self.metrics.frame_id = self.metrics.frames;
        self.metrics.paused = false;

        let frame_start = Instant::now();

        let detection = match (&mut self.camera, &mut self.tracker) {
            (Some(camera), Some(tracker)) => {
                let frame = match camera.get_frame() {
                    Ok(frame) => frame,
                    Err(e) => {
                        if self.debug {
                            eprintln!("[Pipeline] camera frame failed: {e}");
                        }
                        return Ok(false);
                    }
                };
                self.metrics.frame_size = (frame.cols() as usize, frame.rows() as usize);
                if should_upload_feed(self.emit_frames, &self.feed_wanted) {
                    self.metrics.frame_pixels = frame_to_rgba(&frame);
                } else {
                    // Keep the shared snapshot free of pixels the UI will discard.
                    self.metrics.frame_pixels.clear();
                }
                tracker.process_frame(&frame)?
            }
            _ => {
                self.metrics.frame_size = (640, 480);
                synthetic_detection(self.started.elapsed().as_secs_f64())
            }
        };

        let frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        let instantaneous_fps = (1000.0 / frame_ms.max(1e-6)) as f32;
        self.metrics.frame_ms = frame_ms;
        self.metrics.fps = if self.metrics.fps == 0.0 {
            instantaneous_fps
        } else {
            0.9 * self.metrics.fps + 0.1 * instantaneous_fps
        };
        self.metrics.detector_us = detection.timings.detector_us;
        self.metrics.landmarker_us = detection.timings.landmarker_us;

        // Tracking state. `clone_from` keeps the Vec's capacity, so the landmark
        // list stops reallocating every frame.
        self.metrics.detected = detection.r#box.is_some();
        self.metrics.confidence = detection.score;
        self.metrics.hand_box = detection.r#box;
        match &detection.hand_landmarks {
            Some(landmarks) => self.metrics.landmarks.clone_from(landmarks),
            None => self.metrics.landmarks.clear(),
        }
        self.metrics.landmark_count = self.metrics.landmarks.len();
        // The note anchor: landmark 8 is the index fingertip.
        self.metrics.pitch_x = self.metrics.landmarks.get(8).map_or(0.0, |l| l.x);

        if self.metrics.detected {
            self.last_detection = Instant::now();
        }
        self.metrics.lost_ms = self.last_detection.elapsed().as_millis() as u64;
        self.metrics.jitter_px = landmark_jitter(&self.previous_landmarks, &self.metrics.landmarks);
        self.previous_landmarks.clone_from(&self.metrics.landmarks);

        if let Some(geometry) = hand_geometry(&detection) {
            self.metrics.spread = geometry.spread;
            self.metrics.extension = geometry.extension;
            self.metrics.tilt_deg = geometry.tilt_deg;
            self.metrics.palm_length = geometry.palm_length;
        } else {
            self.metrics.spread = 0.0;
            self.metrics.extension = [0.0; 5];
            self.metrics.tilt_deg = 0.0;
            self.metrics.palm_length = 0.0;
        }

        // Gesture -> sinks. A sink that never receives anything keeps whatever it
        // was last told, so the *absence* of a gesture has to be announced too:
        // without the release event the synthesizer keeps oscillating forever.
        if let Some(event) = self.strategy.process(&detection) {
            match &event.payload {
                GesturePayload::Note { note, velocity } => {
                    self.metrics.note = Some(*note);
                    self.metrics.volume = *velocity;
                    self.metrics.frequency = None;
                }
                GesturePayload::Signal {
                    frequency,
                    amplitude,
                } => {
                    self.metrics.frequency = Some(*frequency);
                    self.metrics.volume = *amplitude;
                    self.metrics.note = None;
                }
            }
            self.gesture_active = true;
            self.event_bus.dispatch(event);
        } else {
            self.metrics.note = None;
            self.metrics.volume = 0.0;
            self.metrics.frequency = None;

            if self.gesture_active {
                let release = GestureEvent::no_gesture();
                self.gesture_active = false;
                self.event_bus.dispatch(release);
            }
        }

        while let Ok(event) = self.event_rx.try_recv() {
            for sink in &mut self.sinks {
                if let Err(e) = sink.handle_event(&event) {
                    if self.debug {
                        eprintln!("[Pipeline] sink error: {e}");
                    }
                }
            }
            self.push_log(&event);
        }

        if let Ok(mut shared) = self.shared.lock() {
            *shared = self.metrics.clone();
        }

        if self.debug && self.metrics.frames % 10 == 0 {
            println!(
                "[Pipeline] fps {:.1} conf {:.2} note {:?} vol {:.2} spread {:.2} jitter {:.1} px tip x {:.0}",
                self.metrics.fps,
                self.metrics.confidence,
                self.metrics.note,
                self.metrics.volume,
                self.metrics.spread,
                self.metrics.jitter_px,
                self.metrics.pitch_x,
            );
        }

        Ok(true)
    }

    fn push_log(&mut self, event: &GestureEvent) {
        let detail = match &event.payload {
            GesturePayload::Note { note: 0, .. } => "release".to_string(),
            GesturePayload::Note { note, velocity } => {
                format!(
                    "note {} ({}) vel {:.2}",
                    note,
                    crate::dashboard::metrics::note_name(*note),
                    velocity
                )
            }
            GesturePayload::Signal {
                frequency,
                amplitude,
            } => {
                format!("{:.0} Hz amp {:.2}", frequency, amplitude)
            }
        };

        if self.metrics.log.len() >= LOG_CAPACITY {
            self.metrics.log.pop_front();
        }
        self.metrics.log.push_back(EventLogEntry {
            uptime_ms: self.metrics.uptime_ms,
            text: event.text.clone(),
            detail,
        });
    }
}

/// The stage only needs camera pixels while the feed toggle is on; headless runs
/// never need them.
fn should_upload_feed(emit_frames: bool, want: &AtomicBool) -> bool {
    emit_frames && want.load(Ordering::Relaxed)
}

/// BGR OpenCV frame -> RGBA bytes for the iced `image` handle.
fn frame_to_rgba(frame: &Mat) -> Vec<u8> {
    let count = (frame.cols() as usize) * (frame.rows() as usize) * (frame.channels() as usize);
    // Safe: `frame` is a freshly captured, contiguous Mat, so its buffer holds
    // exactly `count` bytes starting at `data()`.
    let src = unsafe { std::slice::from_raw_parts(frame.data(), count) };
    bgr_to_rgba(src)
}

fn bgr_to_rgba(src: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(src.len() / 3 * 4);
    for pixel in src.chunks_exact(3) {
        rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], 255]);
    }
    rgba
}

fn landmark_jitter(previous: &[Landmark], current: &[Landmark]) -> f32 {
    if previous.len() != current.len() || current.is_empty() {
        return 0.0;
    }
    let total: f32 = previous
        .iter()
        .zip(current)
        .map(|(a, b)| ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt())
        .sum();
    total / current.len() as f32
}

/// A synthetic, moving hand so the dashboard can be exercised without a
/// webcam or the ONNX models (`--mode mock`).
fn synthetic_detection(seconds: f64) -> HandDetectionResult {
    let t = seconds as f32;
    // The centre sweeps almost the full 640 px view so the mock hand walks the
    // whole keyboard, low notes on the left and high notes on the right.
    let cx = 320.0 + 240.0 * (t * 0.7).sin();
    let cy = 250.0 + 50.0 * (t * 0.45).cos();
    let curl = 0.45 + 0.45 * (t * 1.1).sin();
    let thumb_angle = 55.0 + 25.0 * (t * 0.8).cos();

    let wrist = Landmark {
        x: cx,
        y: cy + 140.0,
        z: 0.0,
    };
    let thumb_cmc = Landmark {
        x: cx - 45.0,
        y: cy + 95.0,
        z: 0.0,
    };

    let mut landmarks = vec![wrist, thumb_cmc.clone()];
    landmarks.extend(finger_chain(thumb_cmc, thumb_angle, 95.0, curl, 3));

    let knuckles = [
        (cx - 25.0, cy + 90.0, -12.0, 130.0), // index
        (cx, cy + 90.0, 0.0, 140.0),          // middle
        (cx + 25.0, cy + 90.0, 12.0, 130.0),  // ring
        (cx + 48.0, cy + 90.0, 24.0, 110.0),  // pinky
    ];
    for (x, y, angle, length) in knuckles {
        let base = Landmark { x, y, z: 0.0 };
        landmarks.extend(finger_chain(base, angle, length, curl, 4));
    }

    let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for landmark in &landmarks {
        minx = minx.min(landmark.x);
        maxx = maxx.max(landmark.x);
        miny = miny.min(landmark.y);
        maxy = maxy.max(landmark.y);
    }

    HandDetectionResult {
        hand_landmarks: Some(landmarks),
        r#box: Some((minx, miny, maxx - minx, maxy - miny)),
        frame_size: (640, 480),
        score: 0.85 + 0.12 * (t * 0.6).sin(),
        timings: Default::default(),
    }
}

/// Builds `joints` landmarks walking away from `base` in the given direction,
/// bending progressively according to `curl`. 0 degrees points straight up.
fn finger_chain(
    base: Landmark,
    angle_deg: f32,
    length: f32,
    curl: f32,
    joints: usize,
) -> Vec<Landmark> {
    let angle = angle_deg * std::f32::consts::PI / 180.0;
    let segment = length / joints as f32;

    let mut x = base.x;
    let mut y = base.y;
    let mut chain = Vec::with_capacity(joints);

    for i in 0..joints {
        let a = angle + curl * i as f32 * 0.35;
        x += a.sin() * segment;
        y -= a.cos() * segment;
        chain.push(Landmark { x, y, z: 0.0 });
    }

    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_hand_has_21_landmarks() {
        let detection = synthetic_detection(1.0);
        let landmarks = detection.hand_landmarks.unwrap();
        assert_eq!(landmarks.len(), 21);
        assert!(detection.r#box.is_some());
    }

    #[test]
    fn synthetic_hand_geometry_is_sane() {
        let detection = synthetic_detection(2.0);
        let geometry = hand_geometry(&detection).unwrap();
        assert!(geometry.palm_length > 0.0);
        assert!(geometry.spread > 0.0);
        assert!(geometry.extension[1] > geometry.extension[0]);
    }

    #[test]
    fn jitter_is_zero_for_identical_landmarks() {
        let landmarks = synthetic_detection(0.0).hand_landmarks.unwrap();
        assert!((landmark_jitter(&landmarks, &landmarks) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn bgr_pixels_become_rgba() {
        // Two BGR pixels.
        assert_eq!(
            bgr_to_rgba(&[1, 2, 3, 4, 5, 6]),
            vec![3, 2, 1, 255, 6, 5, 4, 255]
        );
    }

    /// A sink that records every event it is handed, so the tests can inspect
    /// what the pipeline actually dispatched.
    #[derive(Clone)]
    struct RecordingSink {
        events: Arc<Mutex<Vec<GestureEvent>>>,
    }

    impl Sink for RecordingSink {
        fn start(&mut self) -> Result<()> {
            Ok(())
        }
        fn stop(&mut self) -> Result<()> {
            Ok(())
        }
        fn handle_event(&mut self, event: &GestureEvent) -> Result<()> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    /// Fires on every frame, i.e. a hand that never leaves the view.
    struct AlwaysOn;

    impl Strategy for AlwaysOn {
        fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent> {
            detection
                .hand_landmarks
                .as_ref()
                .map(|_| GestureEvent::new_note(60, 0.8, "Finger Position"))
        }
    }

    /// Fires on the first frame only, then goes silent - i.e. the hand leaves.
    struct FirstFrameOnly {
        armed: AtomicBool,
    }

    impl Strategy for FirstFrameOnly {
        fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent> {
            if detection.hand_landmarks.is_none() || !self.armed.load(Ordering::Relaxed) {
                return None;
            }
            self.armed.store(false, Ordering::Relaxed);
            Some(GestureEvent::new_note(60, 0.8, "Finger Position"))
        }
    }

    fn test_pipeline(strategy: Box<dyn Strategy>) -> (Pipeline, Arc<Mutex<Vec<GestureEvent>>>) {
        let (bus, rx) = EventBus::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let pipeline = Pipeline::new(
            None,
            None,
            strategy,
            vec![Box::new(RecordingSink {
                events: events.clone(),
            })],
            bus,
            rx,
            Arc::new(Mutex::new(FrameMetrics::default())),
            None,
            false,
            false,
            Arc::new(AtomicBool::new(true)),
            Arc::new(AtomicBool::new(false)),
        );
        (pipeline, events)
    }

    #[test]
    #[ignore]
    /// Quantifies what the feed gate saves. Run with
    /// `cargo test --release -- --include-ignored bench_feed_upload`.
    fn bench_feed_upload() {
        let frame = opencv::core::Mat::new_rows_cols_with_default(
            480,
            640,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::new(128.0, 128.0, 128.0, 0.0),
        )
        .unwrap();
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

    #[test]
    fn the_feed_is_only_converted_when_the_stage_shows_it() {
        let want = AtomicBool::new(false);
        assert!(!should_upload_feed(true, &want), "feed off means no upload");
        want.store(true, Ordering::Relaxed);
        assert!(should_upload_feed(true, &want));
        assert!(!should_upload_feed(false, &want), "headless never uploads");
    }

    #[test]
    fn pausing_releases_the_note_and_marks_the_snapshot() {
        let (mut pipeline, events) = test_pipeline(Box::new(AlwaysOn));

        // With no camera the pipeline synthesises a hand, so the strategy fires.
        pipeline.tick().unwrap();

        pipeline.active.store(false, Ordering::Relaxed);
        pipeline.pause();

        let recorded = events.lock().unwrap().clone();
        assert!(recorded
            .iter()
            .any(|e| matches!(e.payload, GesturePayload::Note { note: 0, .. })),
            "stopping must release the held note");
        assert!(pipeline.metrics.paused);
        assert!(pipeline.metrics.note.is_none());
        assert_eq!(pipeline.metrics.volume, 0.0);
    }

    #[test]
    fn a_release_event_is_sent_when_the_gesture_ends() {
        let (mut pipeline, events) = test_pipeline(Box::new(FirstFrameOnly {
            armed: AtomicBool::new(true),
        }));

        pipeline.tick().unwrap(); // hand present -> note
        pipeline.tick().unwrap(); // hand gone   -> release
        pipeline.tick().unwrap(); // still gone  -> nothing more

        let sent = events.lock().unwrap();
        assert_eq!(sent.len(), 2, "exactly one note and one release");
        match sent[0].payload {
            GesturePayload::Note { note, velocity } => {
                assert_eq!(note, 60);
                assert!((velocity - 0.8).abs() < 1e-6);
            }
            _ => panic!("expected the gesture note first"),
        }
        match sent[1].payload {
            GesturePayload::Note { note, velocity } => {
                assert_eq!(note, 0, "release must silence the sink");
                assert_eq!(velocity, 0.0);
            }
            _ => panic!("expected the release event second"),
        }
    }

    #[test]
    fn nothing_is_emitted_while_no_gesture_has_started() {
        let (mut pipeline, events) = test_pipeline(Box::new(FirstFrameOnly {
            armed: AtomicBool::new(false),
        }));
        pipeline.tick().unwrap();
        pipeline.tick().unwrap();
        assert!(events.lock().unwrap().is_empty());
        assert_eq!(pipeline.metrics.volume, 0.0);
        assert!(!pipeline.gesture_active);
    }
}
