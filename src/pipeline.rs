//! The capture -> inference -> gesture -> sink loop, decoupled from the UI.
//!
//! `Pipeline::run` splits the frame loop into three stages on three threads:
//!
//! * **capture** reads the webcam and pushes the frame into a capacity-1 channel.
//!   When inference is behind, the queued frame is dropped rather than making the
//!   pipeline work on stale pixels.
//! * **inference** is the pipeline thread: tracker -> strategy -> event bus, plus
//!   the `FrameMetrics` publish the dashboard reads.
//! * **sinks** drain the event bus and talk to the audio/MIDI outputs, so a
//!   blocking MIDI write cannot stall a frame.
//!
//! `Pipeline::tick` is the same frame driven single-threaded, which is what the
//! tests and any headless one-shot use.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use opencv::prelude::*;

use crate::core::event_bus::EventBus;
use crate::core::geometry::hand_geometry;
use crate::core::instrument::InstrumentConfig;
use crate::core::types::{GestureEvent, GesturePayload};
use crate::core::{HandDetectionResult, Landmark, Sink, Strategy};
use crate::dashboard::metrics::{cell_index, EventLogEntry, FrameMetrics, LOG_CAPACITY};
use crate::vision::camera::WebcamSource;
use crate::vision::tracker::HandTracker;

/// Mock mode emulates a 30 fps camera so the telemetry looks real.
const MOCK_CADENCE_MS: u64 = 33;
/// How often the paused loop checks the Start/Stop flag and refreshes the UI.
const PAUSE_POLL_MS: u64 = 50;
/// How long inference waits for the capture stage before re-checking the flags.
const FRAME_POLL_MS: u64 = 20;
/// How often the sink thread wakes up to look for an event or a shutdown.
const SINK_POLL_MS: u64 = 20;

/// A frame handed from the capture stage to inference.
struct CapturedFrame {
    frame: Mat,
    /// Wall clock the pixels came off the camera, so end-to-end latency is known.
    captured_at: Instant,
    /// Time the capture stage spent reading them.
    capture_ms: f64,
}

/// One frame's worth of work, split by stage so the dashboard can show camera
/// time and inference time separately instead of one conflated number.
struct FrameOutcome {
    detection: HandDetectionResult,
    frame: Option<Mat>,
    capture_ms: f64,
    inference_ms: f64,
    latency_ms: f64,
}

#[derive(Default)]
struct CaptureCounters {
    captured: AtomicU64,
    dropped: AtomicU64,
}

/// The output sinks, shared between the pipeline thread (which only pushes an
/// instrument change) and the sink thread (which drives them).
type SinkList = Arc<Mutex<Vec<Box<dyn Sink>>>>;

pub struct Pipeline {
    camera: Option<WebcamSource>,
    tracker: Option<HandTracker>,
    strategy: Box<dyn Strategy>,
    /// Shared with the sink thread, which is the only thing that writes to the
    /// outputs; the pipeline thread touches them only when the instrument changes.
    sinks: SinkList,
    event_bus: EventBus,
    /// Drained by whoever owns it: the pipeline thread while running inline, the
    /// sink thread once `run` has spawned it. Exactly one consumer, so events reach
    /// the sinks in the order they were dispatched.
    event_rx: Option<Receiver<GestureEvent>>,
    shared: Arc<Mutex<FrameMetrics>>,
    /// The instrument the player is currently on. The dashboard pickers write
    /// to it; the pipeline notices the version change and re-keys the strategy
    /// and the sinks on the next frame.
    instrument: Arc<Mutex<InstrumentConfig>>,
    /// Version of `instrument` last applied to the strategy and the sinks. Starts
    /// above any real version so the first frame always re-keys, which also keeps
    /// the published layout in sync with the config even when the snapshot was
    /// never seeded.
    applied_version: u64,
    metrics: FrameMetrics,
    previous_landmarks: Vec<Landmark>,
    /// Frames the capture stage read and dropped, so a bottleneck is visible
    /// instead of invisible.
    capture: Arc<CaptureCounters>,
    /// True while a gesture is actively driving the sinks. Used to send exactly
    /// one release event when it stops, so the audio/MIDI sinks do not latch the
    /// last note and keep sounding after the hand leaves the frame.
    gesture_active: bool,
    last_detection: Instant,
    /// Wall clock of the last published frame; the fps is derived from the gap
    /// between publishes, which stays honest once the stages overlap.
    last_publish: Instant,
    /// Exponentially smoothed publish interval, in milliseconds.
    avg_gap_ms: f64,
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

/// Everything the loop needs to run, grouped so `Pipeline::new` is not a
/// thirteen-argument call.
pub struct PipelineConfig {
    pub camera: Option<WebcamSource>,
    pub tracker: Option<HandTracker>,
    pub strategy: Box<dyn Strategy>,
    pub sinks: Arc<Mutex<Vec<Box<dyn Sink>>>>,
    pub event_bus: EventBus,
    pub event_rx: Option<Receiver<GestureEvent>>,
    pub shared: Arc<Mutex<FrameMetrics>>,
    pub instrument: Arc<Mutex<InstrumentConfig>>,
    pub active: Arc<AtomicBool>,
    pub feed_wanted: Arc<AtomicBool>,
    pub max_frames: Option<u32>,
    pub debug: bool,
    pub emit_frames: bool,
}

impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let PipelineConfig {
            camera,
            tracker,
            strategy,
            sinks,
            event_bus,
            event_rx,
            shared,
            instrument,
            active,
            feed_wanted,
            max_frames,
            debug,
            emit_frames,
        } = config;

        // `main` seeds the snapshot with the run's context (mode, sink, scale).
        // The pipeline keeps its own copy so publishing a frame does not wipe
        // those fields back to their defaults.
        let context = shared
            .lock()
            .map(|snapshot| FrameMetrics {
                mode: snapshot.mode.clone(),
                sink: snapshot.sink.clone(),
                scale: snapshot.scale.clone(),
                root: snapshot.root.clone(),
                layout: snapshot.layout.clone(),
                ..FrameMetrics::default()
            })
            .unwrap_or_default();

        Pipeline {
            camera,
            tracker,
            strategy,
            sinks,
            event_bus,
            event_rx,
            shared,
            instrument,
            applied_version: u64::MAX,
            metrics: context,
            previous_landmarks: Vec::new(),
            capture: Arc::new(CaptureCounters::default()),
            gesture_active: false,
            last_detection: Instant::now(),
            last_publish: Instant::now(),
            avg_gap_ms: 0.0,
            started: Instant::now(),
            max_frames,
            debug,
            emit_frames,
            active,
            feed_wanted,
        }
    }

    /// Pick up any change the dashboard made to the instrument. One mutex lock
    /// and an integer compare per frame; the expensive part (rebuilding the grid
    /// and re-keying the strategy) only runs when the version actually moved.
    fn sync_instrument(&mut self) {
        let Ok(config) = self.instrument.lock() else {
            return;
        };
        if config.version == self.applied_version {
            return;
        }
        self.applied_version = config.version;

        self.strategy.apply_layout(config.layout.clone());
        if let Ok(mut sinks) = self.sinks.lock() {
            for sink in sinks.iter_mut() {
                sink.set_waveform(config.waveform);
            }
        }
        self.metrics.scale = config.scale.name.to_string();
        self.metrics.root = config.root_name.clone();
        self.metrics.waveform = config.waveform.name().to_string();
        self.metrics.layout = config.layout.clone();

        if self.debug {
            println!(
                "[Instrument] {} {} - {}x{} grid, {} waveform",
                config.scale.name,
                config.root_name,
                config.layout.rows,
                config.layout.cols,
                config.waveform,
            );
        }
    }

    /// Runs until the UI asks for shutdown or `--max-frames` is reached. The
    /// `active` flag (the dashboard's Stop button) pauses the loop without
    /// tearing down the camera or the sinks.
    pub fn run(&mut self, running: &Arc<AtomicBool>) -> Result<()> {
        // Stage 1: the camera, on its own thread. The channel holds a single frame,
        // so inference always takes the newest one instead of queueing up behind
        // its own latency and drifting away from the live view.
        let (frame_tx, frame_rx) = crossbeam_channel::bounded(1);
        let capture_handle = self.camera.take().map(|mut camera| {
            let tx = frame_tx.clone();
            let running = running.clone();
            let active = self.active.clone();
            let counters = self.capture.clone();
            std::thread::spawn(move || {
                if let Err(e) = capture_frames(
                    &mut || camera.get_frame(),
                    &tx,
                    &active,
                    &running,
                    &counters,
                ) {
                    eprintln!("[Capture] {e}");
                }
            })
        });
        drop(frame_tx); // closing the sender ends inference when capture exits

        // Stage 3: the sinks. A blocking MIDI write, or any sink that hiccups, no
        // longer holds the frame loop hostage.
        let sink_handle = {
            let sinks = self.sinks.clone();
            let running = running.clone();
            let debug = self.debug;
            let rx = self.event_rx.take();
            std::thread::spawn(move || {
                let Some(rx) = rx else {
                    return;
                };
                while running.load(Ordering::Relaxed) {
                    match rx.recv_timeout(Duration::from_millis(SINK_POLL_MS)) {
                        Ok(event) => apply_to_sinks(&sinks, &event, debug),
                        Err(RecvTimeoutError::Timeout) => continue,
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
                // Flush whatever was dispatched on the way out, then close the outputs.
                while let Ok(event) = rx.try_recv() {
                    apply_to_sinks(&sinks, &event, debug);
                }
                if let Ok(mut sinks) = sinks.lock() {
                    for sink in sinks.iter_mut() {
                        let _ = sink.stop();
                    }
                }
            })
        };

        // Stage 2: this thread.
        let mock = self.tracker.is_none();
        while running.load(Ordering::Relaxed) {
            if !self.active.load(Ordering::Relaxed) {
                // Still honour picker changes while stopped, so the header and the
                // sinks match what the dashboard shows before the player starts again.
                self.sync_instrument();
                self.pause();
                std::thread::sleep(Duration::from_millis(PAUSE_POLL_MS));
                continue;
            }

            if mock {
                // Mock mode has no camera, so `tick` drives the frame itself and
                // paces it like a 30 fps one to keep the telemetry realistic.
                if !self.tick()? {
                    break;
                }
            } else {
                let outcome = match frame_rx.recv_timeout(Duration::from_millis(FRAME_POLL_MS)) {
                    Ok(captured) => {
                        let inference_start = Instant::now();
                        let latency_ms =
                            inference_start.duration_since(captured.captured_at).as_secs_f64() * 1000.0;
                        let Some(tracker) = self.tracker.as_mut() else {
                            break;
                        };
                        let detection = tracker.process_frame(&captured.frame)?;
                        FrameOutcome {
                            detection,
                            frame: Some(captured.frame),
                            capture_ms: captured.capture_ms,
                            inference_ms: inference_start.elapsed().as_secs_f64() * 1000.0,
                            latency_ms,
                        }
                    }
                    // Capture is behind; re-check the Start/Stop flag and try again.
                    Err(RecvTimeoutError::Timeout) => continue,
                    // The capture stage stopped: the camera is gone.
                    Err(RecvTimeoutError::Disconnected) => break,
                };
                self.advance(outcome)?;
            }
            if self
                .max_frames
                .is_some_and(|n| self.metrics.frames >= n as u64)
            {
                break;
            }
        }

        if self.debug {
            let frames = self.metrics.frames.max(1);
            println!(
                "[Pipeline] {} frames in {:.2} s: {:.1} fps, {:.2} ms/frame end to end (capture {:.2} ms, inference {:.2} ms, {} frames dropped)",
                self.metrics.frames,
                self.metrics.uptime_ms as f64 / 1000.0,
                self.metrics.fps,
                self.metrics.uptime_ms as f64 / frames as f64,
                self.metrics.capture_ms,
                self.metrics.frame_ms,
                self.metrics.dropped,
            );
        }

        running.store(false, Ordering::Relaxed);
        if let Some(handle) = capture_handle {
            let _ = handle.join();
        }
        let _ = sink_handle.join();
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
        if let Some(rx) = &self.event_rx {
            drain_events(rx, &self.sinks, self.debug);
        }
        // Otherwise the stopped interval shows up as a huge fps dip on resume.
        self.last_publish = Instant::now();

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
        self.metrics.tip_x = 0.0;
        self.metrics.tip_y = 0.0;
        if let Ok(mut shared) = self.shared.lock() {
            *shared = self.metrics.clone();
        }
    }

    /// One camera frame end to end. Returns `false` when the camera is gone.
    ///
    /// This is the single-threaded path: it captures, infers and publishes. The
    /// staged `run` loop does the same thing with the camera and the sinks on their
    /// own threads.
    pub fn tick(&mut self) -> Result<bool> {
        if self.camera.is_none() {
            // Mock mode has no camera, so pace it like one to keep the telemetry
            // (fps, jitter, history) realistic.
            std::thread::sleep(Duration::from_millis(MOCK_CADENCE_MS));
            self.advance(FrameOutcome {
                detection: synthetic_detection(self.started.elapsed().as_secs_f64()),
                frame: None,
                capture_ms: 0.0,
                inference_ms: 0.0,
                latency_ms: 0.0,
            })?;
            return Ok(true);
        }

        let outcome = {
            let Some(camera) = self.camera.as_mut() else {
                return Ok(false);
            };
            let capture_start = Instant::now();
            let frame = match camera.get_frame() {
                Ok(frame) => frame,
                Err(e) => {
                    if self.debug {
                        eprintln!("[Pipeline] camera frame failed: {e}");
                    }
                    return Ok(false);
                }
            };
            let capture_ms = capture_start.elapsed().as_secs_f64() * 1000.0;
            let Some(tracker) = self.tracker.as_mut() else {
                return Ok(false);
            };
            let inference_start = Instant::now();
            let detection = tracker.process_frame(&frame)?;
            FrameOutcome {
                detection,
                frame: Some(frame),
                capture_ms,
                inference_ms: inference_start.elapsed().as_secs_f64() * 1000.0,
                latency_ms: 0.0,
            }
        };

        self.advance(outcome)?;
        Ok(true)
    }

    /// Everything that happens to a frame once the tracker has returned: the
    /// instrument sync, metrics, the gesture dispatch, and the publish.
    ///
    /// The instrument check lives here rather than in `tick` so it happens on
    /// every frame whichever stage produced it - the staged `run` loop never goes
    /// through `tick`, and a picker change that never reaches the strategy or the
    /// sinks is invisible except by ear.
    fn advance(&mut self, outcome: FrameOutcome) -> Result<()> {
        self.sync_instrument();
        let FrameOutcome {
            detection,
            frame,
            capture_ms,
            inference_ms,
            latency_ms,
        } = outcome;

        let now = Instant::now();
        let gap_ms = now.duration_since(self.last_publish).as_secs_f64() * 1000.0;
        self.last_publish = now;

        // Smooth the *interval*, not the reciprocal rate. With the stages
        // overlapping, individual gaps alternate between a queued frame (a
        // millisecond) and a full inference, and an EMA of fps turns that jitter
        // into a number that bears no relation to the frame rate.
        self.avg_gap_ms = if self.avg_gap_ms == 0.0 {
            gap_ms
        } else {
            0.9 * self.avg_gap_ms + 0.1 * gap_ms
        };
        self.metrics.fps = (1000.0 / self.avg_gap_ms.max(1.0)) as f32;

        self.metrics.frames += 1;
        self.metrics.uptime_ms = self.started.elapsed().as_millis() as u64;
        self.metrics.frame_id = self.metrics.frames;
        self.metrics.paused = false;
        self.metrics.frame_size = detection.frame_size;
        self.metrics.frame_ms = inference_ms;
        self.metrics.capture_ms = capture_ms;
        self.metrics.latency_ms = latency_ms;
        self.metrics.captured = self.capture.captured.load(Ordering::Relaxed);
        self.metrics.dropped = self.capture.dropped.load(Ordering::Relaxed);

        if let Some(frame) = frame.as_ref() {
            if should_upload_feed(self.emit_frames, &self.feed_wanted) {
                self.metrics.frame_pixels = frame_to_rgba(frame);
            } else {
                // Keep the shared snapshot free of pixels the UI will discard.
                self.metrics.frame_pixels.clear();
            }
        }

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
        if let Some(tip) = self.metrics.landmarks.get(8) {
            self.metrics.tip_x = tip.x;
            self.metrics.tip_y = tip.y;
        }
        // Which cell the sounding note belongs to, for the crosshair and the
        // cell highlight. Notes are unique in the grid, so this is exact.
        if let Some(note) = self.metrics.note {
            if let Some(cell) = cell_index(&self.metrics.layout, note) {
                self.metrics.grid_row = cell / self.metrics.layout.cols;
                self.metrics.grid_col = cell % self.metrics.layout.cols;
            }
        }

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
            self.push_log(&event);
            self.event_bus.dispatch(event);
        } else {
            self.metrics.note = None;
            self.metrics.volume = 0.0;
            self.metrics.frequency = None;

            if self.gesture_active {
                let release = GestureEvent::no_gesture();
                self.gesture_active = false;
                self.push_log(&release);
                self.event_bus.dispatch(release);
            }
        }

        // Inline mode: this thread still owns the receiver. Once `run` has spawned
        // the sink thread it is gone, and that stage is the only consumer.
        if let Some(rx) = &self.event_rx {
            drain_events(rx, &self.sinks, self.debug);
        }

        if let Ok(mut shared) = self.shared.lock() {
            *shared = self.metrics.clone();
        }

        if self.debug && self.metrics.frames.is_multiple_of(10) {
            println!(
                "[Pipeline] fps {:.1} inference {:.2} ms (capture {:.2} ms, latency {:.2} ms, dropped {}) detector {:.0} us, landmarker {:.0} us conf {:.2} note {:?} vol {:.2} spread {:.2} jitter {:.1} px tip {:.0},{:.0}",
                self.metrics.fps,
                self.metrics.frame_ms,
                self.metrics.capture_ms,
                self.metrics.latency_ms,
                self.metrics.dropped,
                self.metrics.detector_us,
                self.metrics.landmarker_us,
                self.metrics.confidence,
                self.metrics.note,
                self.metrics.volume,
                self.metrics.spread,
                self.metrics.jitter_px,
                self.metrics.tip_x,
                self.metrics.tip_y,
            );
        }

        Ok(())
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

/// Read frames from the camera and queue them for inference.
///
/// The channel holds one frame. If inference is still busy with the previous one,
/// the new frame is dropped and counted rather than letting the pipeline drift
/// behind the camera, working on pixels that are already stale.
fn capture_frames(
    mut read: impl FnMut() -> Result<Mat>,
    tx: &Sender<CapturedFrame>,
    active: &AtomicBool,
    running: &AtomicBool,
    counters: &CaptureCounters,
) -> Result<()> {
    while running.load(Ordering::Relaxed) {
        if !active.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(PAUSE_POLL_MS));
            continue;
        }
        let start = Instant::now();
        let frame = read()?;
        counters.captured.fetch_add(1, Ordering::Relaxed);
        let captured = CapturedFrame {
            frame,
            captured_at: Instant::now(),
            capture_ms: start.elapsed().as_secs_f64() * 1000.0,
        };
        if tx.try_send(captured).is_err() {
            counters.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(())
}

fn apply_to_sinks(sinks: &SinkList, event: &GestureEvent, debug: bool) {
    let Ok(mut sinks) = sinks.lock() else {
        return;
    };
    for sink in sinks.iter_mut() {
        if let Err(e) = sink.handle_event(event) {
            if debug {
                eprintln!("[Sink] {e}");
            }
        }
    }
}

fn drain_events(rx: &Receiver<GestureEvent>, sinks: &SinkList, debug: bool) {
    while let Ok(event) = rx.try_recv() {
        apply_to_sinks(sinks, &event, debug);
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
    use crate::core::grid::GridLayout;
    use crate::core::instrument::Waveform;
    use crate::strategies::FingerPositionStrategy;

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
        /// Timbres the pipeline pushed into the sinks, in order. Without this the
        /// picker wiring is only audible, not testable.
        waveforms: Arc<Mutex<Vec<String>>>,
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
        fn set_waveform(&mut self, waveform: Waveform) {
            self.waveforms.lock().unwrap().push(waveform.name().to_string());
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

    fn test_pipeline(
        strategy: Box<dyn Strategy>,
        max_frames: Option<u32>,
    ) -> (
        Pipeline,
        Arc<Mutex<Vec<GestureEvent>>>,
        Arc<Mutex<InstrumentConfig>>,
        Arc<Mutex<Vec<String>>>,
    ) {
        let (bus, rx) = EventBus::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let waveforms = Arc::new(Mutex::new(Vec::new()));
        let instrument = Arc::new(Mutex::new(InstrumentConfig::default()));
        let pipeline = Pipeline::new(PipelineConfig {
            camera: None,
            tracker: None,
            strategy,
            sinks: Arc::new(Mutex::new(vec![Box::new(RecordingSink {
                events: events.clone(),
                waveforms: waveforms.clone(),
            })])),
            event_bus: bus,
            event_rx: Some(rx),
            shared: Arc::new(Mutex::new(FrameMetrics::default())),
            instrument: instrument.clone(),
            active: Arc::new(AtomicBool::new(true)),
            feed_wanted: Arc::new(AtomicBool::new(false)),
            max_frames,
            debug: false,
            emit_frames: false,
        });
        (pipeline, events, instrument, waveforms)
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
    fn a_picker_change_rekeys_the_keyboard_on_the_next_frame() {
        let (mut pipeline, events, instrument, _waveforms) = test_pipeline(Box::new(FingerPositionStrategy::default()), None);
        pipeline.tick().unwrap();
        assert_eq!(pipeline.metrics.layout.len(), 15, "the default grid is 3 x 5");
        assert_eq!(pipeline.metrics.scale, "major pentatonic");

        {
            let mut config = instrument.lock().unwrap();
            config.set_scale("chromatic");
            config.set_octaves(4);
            config.set_waveform(Waveform::Saw);
        }

        pipeline.tick().unwrap();
        assert_eq!(pipeline.metrics.layout.rows, 4);
        assert_eq!(pipeline.metrics.layout.cols, 12);
        assert_eq!(pipeline.metrics.scale, "chromatic");

        // The note the strategy just produced must belong to the new grid, not
        // the old one: the layout was pushed into the strategy, not merely into
        // the dashboard's copy of it.
        let recorded = events.lock().unwrap();
        let note = match recorded.last().unwrap().payload {
            GesturePayload::Note { note, .. } => note,
            _ => panic!("expected a note event"),
        };
        assert!(pipeline.metrics.layout.notes.contains(&note), "{note} is not in the new grid");
    }

    #[test]
    fn an_unchanged_config_does_not_rebuild_every_frame() {
        let (mut pipeline, _events, _instrument, _waveforms) = test_pipeline(Box::new(FingerPositionStrategy::default()), None);
        pipeline.sync_instrument();
        let version = pipeline.applied_version;
        pipeline.sync_instrument();
        pipeline.sync_instrument();
        assert_eq!(pipeline.applied_version, version, "a lock alone must not re-key the keyboard");
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
    fn publishing_preserves_the_run_context() {
        let (bus, rx) = EventBus::new();
        let shared = Arc::new(Mutex::new(FrameMetrics {
            mode: "Signal".to_string(),
            sink: "audio".to_string(),
            scale: "major pentatonic".to_string(),
            root: "C".to_string(),
            layout: GridLayout::new(0, &crate::core::scale::preset("major pentatonic").unwrap(), 3),
            ..FrameMetrics::default()
        }));
        let mut pipeline = Pipeline::new(PipelineConfig {
            camera: None,
            tracker: None,
            strategy: Box::new(AlwaysOn),
            sinks: Arc::new(Mutex::new(Vec::new())),
            event_bus: bus,
            event_rx: Some(rx),
            shared: shared.clone(),
            instrument: Arc::new(Mutex::new(InstrumentConfig::default())),
            active: Arc::new(AtomicBool::new(true)),
            feed_wanted: Arc::new(AtomicBool::new(false)),
            max_frames: None,
            debug: false,
            emit_frames: false,
        });
        pipeline.tick().unwrap();

        let snapshot = shared.lock().unwrap();
        assert_eq!(snapshot.scale, "major pentatonic");
        assert_eq!(snapshot.root, "C");
        assert_eq!(snapshot.layout.len(), 15);
        assert_eq!(snapshot.mode, "Signal");
        assert!(snapshot.detected, "the mock hand should register");
    }

    #[test]
    fn pausing_releases_the_note_and_marks_the_snapshot() {
        let (mut pipeline, events, _instrument, _waveforms) = test_pipeline(Box::new(AlwaysOn), None);

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
        let (mut pipeline, events, _instrument, _waveforms) = test_pipeline(Box::new(FirstFrameOnly {
            armed: AtomicBool::new(true),
        }), None);

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
        let (mut pipeline, events, _instrument, _waveforms) = test_pipeline(Box::new(FirstFrameOnly {
            armed: AtomicBool::new(false),
        }), None);
        pipeline.tick().unwrap();
        pipeline.tick().unwrap();
        assert!(events.lock().unwrap().is_empty());
        assert_eq!(pipeline.metrics.volume, 0.0);
        assert!(!pipeline.gesture_active);
    }

    #[test]
    fn the_capture_stage_keeps_only_the_newest_frame() {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let counters = Arc::new(CaptureCounters::default());
        let running = Arc::new(AtomicBool::new(true));
        let active = Arc::new(AtomicBool::new(true));

        let mut produced = 0;
        let mut read = || -> Result<Mat> {
            produced += 1;
            if produced >= 5 {
                running.store(false, Ordering::Relaxed);
            }
            Ok(Mat::new_rows_cols_with_default(
                4,
                4,
                opencv::core::CV_8UC3,
                opencv::core::Scalar::all(7.0),
            )
            .unwrap())
        };

        capture_frames(&mut read, &tx, &active, &running, &counters).unwrap();

        assert_eq!(counters.captured.load(Ordering::Relaxed), 5);
        assert_eq!(
            counters.dropped.load(Ordering::Relaxed),
            4,
            "a frame inference has not reached yet is dropped, not queued"
        );
        assert!(rx.try_recv().is_ok(), "inference always gets the newest frame");
    }

    #[test]
    fn a_paused_capture_stage_reads_no_frames() {
        let (tx, _rx) = crossbeam_channel::bounded(1);
        let counters = Arc::new(CaptureCounters::default());
        let running = Arc::new(AtomicBool::new(true));
        let active = Arc::new(AtomicBool::new(false));

        // The loop would otherwise sleep on the Stop flag forever.
        let stop = running.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(120));
            stop.store(false, Ordering::Relaxed);
        });

        let mut produced = 0;
        let mut read = || -> Result<Mat> {
            produced += 1;
            Ok(Mat::new_rows_cols_with_default(
                2,
                2,
                opencv::core::CV_8UC3,
                opencv::core::Scalar::all(0.0),
            )
            .unwrap())
        };

        capture_frames(&mut read, &tx, &active, &running, &counters).unwrap();
        assert_eq!(produced, 0, "Stop means the camera is not read at all");
        assert_eq!(counters.captured.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_picker_change_reaches_the_sinks_while_the_loop_is_running() {
        let (mut pipeline, _events, instrument, waveforms) = test_pipeline(Box::new(AlwaysOn), Some(8));
        let running = Arc::new(AtomicBool::new(true));

        // Change the timbre mid-run, exactly as clicking the picker does.
        let late = instrument.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(80));
            late.lock().unwrap().set_waveform(Waveform::Square);
        });

        pipeline.run(&running).unwrap();

        let applied = waveforms.lock().unwrap().clone();
        assert!(
            applied.iter().any(|w| w == "square"),
            "the pipeline must push a picker change into the sinks, got {applied:?}"
        );
        assert_eq!(pipeline.metrics.waveform, "square", "the header shows the timbre in use");
    }

    #[test]
    fn fps_is_the_publish_cadence_not_the_stage_times() {
        // Mock mode paces itself at ~30 fps and does no inference at all, so a
        // frame-rate read off the stage durations would report infinity.
        let (mut pipeline, _events, _instrument, _waveforms) = test_pipeline(Box::new(AlwaysOn), Some(20));
        let running = Arc::new(AtomicBool::new(true));
        pipeline.run(&running).unwrap();

        assert!(pipeline.metrics.frames >= 15, "the loop ran to the frame limit");
        assert!(
            pipeline.metrics.fps > 20.0 && pipeline.metrics.fps < 45.0,
            "fps tracks the frames per second, got {}",
            pipeline.metrics.fps
        );
    }

    #[test]
    fn the_staged_run_terminates_and_the_sink_thread_flushes_every_event() {
        let (mut pipeline, events, _instrument, _waveforms) = test_pipeline(Box::new(AlwaysOn), Some(3));
        let running = Arc::new(AtomicBool::new(true));

        pipeline.run(&running).unwrap();

        assert_eq!(pipeline.metrics.frames, 3, "--max-frames stops the frame loop");
        assert!(pipeline.metrics.detector_us == 0.0, "mock mode runs no model");
        let recorded = events.lock().unwrap();
        assert_eq!(
            recorded.len(),
            3,
            "events dispatched by inference reach the sinks on their own thread"
        );
        assert!(!running.load(Ordering::Relaxed), "run clears the flag on exit");
    }
}
