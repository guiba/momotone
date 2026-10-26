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
    last_detection: Instant,
    started: Instant,
    max_frames: Option<u32>,
    debug: bool,
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
            last_detection: Instant::now(),
            started: Instant::now(),
            max_frames,
            debug,
        }
    }

    /// Runs until the UI asks for shutdown or `--max-frames` is reached.
    pub fn run(&mut self, running: &AtomicBool) -> Result<()> {
        while running.load(Ordering::Relaxed) {
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

    /// One camera frame end to end. Returns `false` when the camera is gone.
    pub fn tick(&mut self) -> Result<bool> {
        // Mock mode has no camera, so pace it like one to keep the telemetry
        // (fps, jitter, history) realistic.
        if self.camera.is_none() {
            std::thread::sleep(Duration::from_millis(MOCK_CADENCE_MS));
        }

        self.metrics.frames += 1;
        self.metrics.uptime_ms = self.started.elapsed().as_millis() as u64;

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

        // Tracking state.
        self.metrics.detected = detection.r#box.is_some();
        self.metrics.confidence = detection.score;
        self.metrics.hand_box = detection.r#box;
        self.metrics.landmarks = detection.hand_landmarks.clone().unwrap_or_default();
        self.metrics.landmark_count = self.metrics.landmarks.len();

        if self.metrics.detected {
            self.last_detection = Instant::now();
        }
        self.metrics.lost_ms = self.last_detection.elapsed().as_millis() as u64;
        self.metrics.jitter_px = landmark_jitter(&self.previous_landmarks, &self.metrics.landmarks);
        self.previous_landmarks = self.metrics.landmarks.clone();

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

        // Gesture -> sinks.
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
            self.event_bus.dispatch(event);
        } else {
            self.metrics.note = None;
            self.metrics.volume = 0.0;
            self.metrics.frequency = None;
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
                "[Pipeline] fps {:.1} conf {:.2} note {:?} vol {:.2} spread {:.2} jitter {:.1} px",
                self.metrics.fps,
                self.metrics.confidence,
                self.metrics.note,
                self.metrics.volume,
                self.metrics.spread,
                self.metrics.jitter_px,
            );
        }

        Ok(true)
    }

    fn push_log(&mut self, event: &GestureEvent) {
        let detail = match &event.payload {
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
    let cx = 320.0 + 90.0 * (t * 0.7).sin();
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
}
