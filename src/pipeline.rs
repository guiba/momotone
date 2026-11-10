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

use crate::core::detection_gate::DetectionGate;
use crate::core::event_bus::{BusMessage, EventBus};
use crate::core::geometry::hand_geometry;
use crate::core::instrument::InstrumentConfig;
use crate::core::runtime::{RuntimeConfig, UiCommand};
use crate::core::telemetry::{profile, Samples, StageProfile, PROFILE_REFRESH_FRAMES};
use crate::core::types::{GestureEvent, GesturePayload};
use crate::core::{HandDetectionResult, Landmark, Sink, Strategy};
use crate::dashboard::metrics::{cell_index, note_name, EventLogEntry, FrameMetrics, LOG_CAPACITY};
use crate::vision::ascii::{AsciiGrid, AsciiLayer, Ramp, DEFAULT_ASCII_FPS};
use crate::vision::camera::WebcamSource;
use crate::vision::tracker::HandTracker;

/// Mock mode emulates a 30 fps camera so the telemetry looks real. The loop uses a
/// deadline rather than a flat sleep: macOS `thread::sleep(33 ms)` actually returns
/// after ~41 ms, so a naive sleep makes every mock number 25 % wrong.
const MOCK_CADENCE: Duration = Duration::from_millis(33);
/// How long inference waits for the capture stage before re-checking the flags.
const FRAME_POLL_MS: u64 = 20;
/// Frames of confirmed detection before a gesture may start, and frames of no
/// detection tolerated before the hand is declared lost.
const DETECT_CONFIRM_FRAMES: u32 = 2;
const DETECT_GRACE_FRAMES: u32 = 3;
/// How often the paused loop checks the Start/Stop flag and refreshes the UI.
const PAUSE_POLL_MS: u64 = 50;
/// Frames per second the dashboard is handed a new camera image. The iced
/// renderer uploads each `image::Handle` to the GPU synchronously (a 640x480
/// RGBA frame is under its 2 MB sync threshold), so every upload costs a
/// blocking `wgpu` round trip inside the render loop. The keyboard and skeleton
/// still update every frame; only the preview is throttled. `--feed-fps 0`
/// sends every frame.
pub const DEFAULT_FEED_FPS: f32 = 15.0;

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
    event_rx: Option<Receiver<BusMessage>>,
    shared: Arc<Mutex<FrameMetrics>>,
    /// What the dashboard has asked for: start/stop, feed, ASCII, cadences and the
    /// glyph ramp. The pipeline applies it from commands; the capture thread is the
    /// only other reader, because it must not read the camera while the player is
    /// stopped.
    control: Arc<Mutex<RuntimeConfig>>,
    /// Commands from the UI, drained at the top of every frame.
    control_rx: Option<Receiver<UiCommand>>,
    metrics: FrameMetrics,
    previous_landmarks: Vec<Landmark>,
    /// Frames the capture stage read and dropped, so a bottleneck is visible
    /// instead of invisible.
    capture: Arc<CaptureCounters>,
    /// Debounces the detection gate so a flickering anchor score does not click the
    /// note on and off.
    gate: DetectionGate,
    /// Deadline of the next synthetic frame, for mock mode's paced clock.
    next_mock_frame: Instant,
    /// Rolling stage timings. The frame loop is paced by the slowest stage, so a
    /// regression shows up as a tail here long before the average moves.
    stage_capture: Samples,
    stage_inference: Samples,
    stage_latency: Samples,
    /// Read by the sink thread, which is a different thread, hence the shared ring.
    stage_dispatch: Arc<Mutex<Samples>>,
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
    /// Wall clock at which the next camera frame is allowed to go to the UI.
    /// Seeded with "now" so the very first frame is uploaded.
    next_feed: Instant,
    next_ascii: Instant,
    /// Columns of the ASCII grid; rows follow from the frame aspect.
    ascii_cols: usize,
    /// Terminal mode: print the art to stdout so the renderer can be checked
    /// without a window (`--no-ui --ascii`).
    ascii_stdout: bool,
    /// Optional path to write each rendered ASCII frame to.
    ascii_dump: Option<String>,
    /// Per-cell EMA state so the art settles instead of flickering.
    ascii_layer: AsciiLayer,
}

/// Everything the loop needs to run, grouped so `Pipeline::new` is not a
/// thirteen-argument call.
pub struct PipelineConfig {
    pub camera: Option<WebcamSource>,
    pub tracker: Option<HandTracker>,
    pub strategy: Box<dyn Strategy>,
    pub sinks: Arc<Mutex<Vec<Box<dyn Sink>>>>,
    pub event_bus: EventBus,
    pub event_rx: Option<Receiver<BusMessage>>,
    pub shared: Arc<Mutex<FrameMetrics>>,
    pub control: Arc<Mutex<RuntimeConfig>>,
    pub control_rx: Receiver<UiCommand>,
    /// The instrument the run starts on; later changes arrive as commands.
    pub instrument: InstrumentConfig,
    pub ascii_cols: usize,
    pub ascii_dump: Option<String>,
    pub ascii_stdout: bool,
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
            control,
            control_rx,
            instrument,
            ascii_cols,
            ascii_dump,
            ascii_stdout,
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

        let mut pipeline = Pipeline {
            camera,
            tracker,
            strategy,
            sinks,
            event_bus,
            event_rx,
            shared,
            control,
            control_rx: Some(control_rx),
            metrics: context,
            previous_landmarks: Vec::new(),
            capture: Arc::new(CaptureCounters::default()),
            gate: DetectionGate::new(DETECT_CONFIRM_FRAMES, DETECT_GRACE_FRAMES),
            next_mock_frame: Instant::now(),
            stage_capture: Samples::default(),
            stage_inference: Samples::default(),
            stage_latency: Samples::default(),
            stage_dispatch: Arc::new(Mutex::new(Samples::default())),
            gesture_active: false,
            last_detection: Instant::now(),
            last_publish: Instant::now(),
            avg_gap_ms: 0.0,
            started: Instant::now(),
            max_frames,
            debug,
            emit_frames,
            next_feed: Instant::now(),
            next_ascii: Instant::now(),
            ascii_cols,
            ascii_stdout,
            ascii_dump,
            ascii_layer: AsciiLayer::new(),
        };
        // Seed the instrument so the strategy, the sinks and the published keyboard
        // agree from the very first frame.
        pipeline.apply_instrument(&instrument);
        pipeline
    }

    /// Hand the frame snapshot to the dashboard. The pixel buffer is *moved*
    /// rather than cloned: the UI takes ownership of it on its next tick, so
    /// copying 1.2 MB of RGBA into a Vec that is handed straight back is a
    /// wasted copy every frame.
    fn publish(&mut self) {
        if let Ok(mut shared) = self.shared.lock() {
            let pixels = std::mem::take(&mut self.metrics.frame_pixels);
            *shared = self.metrics.clone();
            shared.frame_pixels = pixels;
        }
    }

    /// A snapshot of what the UI has asked for, taken once per frame so the rest of
    /// the tick reads plain bools instead of four atomics.
    fn settings(&self) -> RuntimeConfig {
        match self.control.lock() {
            Ok(config) => config.clone(),
            Err(_) => RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks),
        }
    }

    /// Is the player running? `Stop` pauses the loop without tearing down the
    /// camera or the sinks.
    fn is_active(&self) -> bool {
        self.control
            .lock()
            .map(|config| config.active)
            .unwrap_or(true)
    }

    /// Apply everything the dashboard has sent since the last frame. Commands are
    /// drained in order, so a Stop cannot be overtaken by a toggle sent after it.
    fn apply_commands(&mut self) {
        let Some(rx) = &self.control_rx else {
            return;
        };
        // Collect first: `rx` borrows `self` while applying a command needs it
        // mutably. The queue is a handful of clicks, so the Vec is noise.
        let commands: Vec<UiCommand> = rx.try_iter().collect();
        for command in commands {
            if let UiCommand::SetInstrument(instrument) = &command {
                self.apply_instrument(instrument);
                continue;
            }
            let Some(mut config) = self.control.lock().ok() else {
                continue;
            };
            match command {
                UiCommand::Start => config.active = true,
                UiCommand::Stop => config.active = false,
                UiCommand::SetFeed(on) => config.feed = on,
                UiCommand::SetAscii(on) => config.ascii = on,
                UiCommand::SetRamp(ramp) => config.ramp = ramp,
                UiCommand::SetInstrument(_) => unreachable!("handled above"),
            }
        }
    }

    /// Re-key the strategy, retimbre the sinks and repaint the header. Runs only
    /// when a picker actually changed something.
    fn apply_instrument(&mut self, config: &InstrumentConfig) {
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
    /// `active` flag in the runtime config (the dashboard's Stop button) pauses the
    /// loop without tearing down the camera or the sinks.
    pub fn run(&mut self, running: &Arc<AtomicBool>) -> Result<()> {
        // Stage 1: the camera, on its own thread. The channel holds a single frame,
        // so inference always takes the newest one instead of queueing up behind
        // its own latency and drifting away from the live view.
        let (frame_tx, frame_rx) = crossbeam_channel::bounded(1);
        let capture_handle = self.camera.take().map(|mut camera| {
            let tx = frame_tx.clone();
            let running = running.clone();
            let control = self.control.clone();
            let counters = self.capture.clone();
            std::thread::spawn(move || {
                if let Err(e) = capture_frames(
                    &mut || camera.get_frame(),
                    &tx,
                    &control,
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
            let debug = self.debug;
            let dispatch = self.stage_dispatch.clone();
            let rx = self.event_rx.take();
            std::thread::spawn(move || {
                let Some(rx) = rx else {
                    return;
                };
                while let Ok(message) = rx.recv() {
                    let queued = match message {
                        BusMessage::Event(queued) => queued,
                        BusMessage::Shutdown => break,
                    };
                    // How long the gesture waited in the queue before an output saw
                    // it: part of the hand-to-sound budget that neither capture nor
                    // inference accounts for.
                    let age_ms = queued.dispatched_at.elapsed().as_secs_f64() * 1000.0;
                    if let Ok(mut samples) = dispatch.lock() {
                        samples.push(age_ms);
                    }
                    apply_to_sinks(&sinks, &queued.event, debug);
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
            self.apply_commands();
            if !self.is_active() {
                // Still honour commands while stopped, so the header and the sinks
                // match what the dashboard shows before the player starts again.
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
                        let latency_ms = inference_start
                            .duration_since(captured.captured_at)
                            .as_secs_f64()
                            * 1000.0;
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
            let perf = self.stage_profile();
            println!(
                "[Stages] p50/p95/p99/max ms - capture {:.1}/{:.1}/{:.1}/{:.1}  inference {:.1}/{:.1}/{:.1}/{:.1}  camera->publish {:.1}/{:.1}/{:.1}/{:.1}  dispatch->sink {:.1}/{:.1}/{:.1}/{:.1}  ({} samples)",
                perf.capture.p50, perf.capture.p95, perf.capture.p99, perf.capture.max,
                perf.inference.p50, perf.inference.p95, perf.inference.p99, perf.inference.max,
                perf.end_to_end.p50, perf.end_to_end.p95, perf.end_to_end.p99, perf.end_to_end.max,
                perf.dispatch.p50, perf.dispatch.p95, perf.dispatch.p99, perf.dispatch.max,
                perf.inference.count,
            );
        }

        running.store(false, Ordering::Relaxed);
        // Unblock the sink thread: everything dispatched before this arrives first,
        // because the channel is FIFO.
        self.event_bus.close();
        if let Some(handle) = capture_handle {
            let _ = handle.join();
        }
        let _ = sink_handle.join();
        Ok(())
    }

    /// Percentiles of the four stages. Recomputed once a second rather than every
    /// frame: the sort is cheap but repeating it 30 times a second is not needed.
    fn stage_profile(&self) -> StageProfile {
        let dispatch = self
            .stage_dispatch
            .lock()
            .map(|samples| profile(&samples))
            .unwrap_or_default();
        StageProfile {
            inference: profile(&self.stage_inference),
            capture: profile(&self.stage_capture),
            end_to_end: profile(&self.stage_latency),
            dispatch,
        }
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
        self.publish();
    }

    /// One camera frame end to end. Returns `false` when the camera is gone.
    ///
    /// This is the single-threaded path: it captures, infers and publishes. The
    /// staged `run` loop does the same thing with the camera and the sinks on their
    /// own threads.
    pub fn tick(&mut self) -> Result<bool> {
        if self.camera.is_none() {
            // Mock mode has no camera, so pace it with a deadline clock rather than a
            // flat sleep: sleeping 33 ms on macOS returns after ~41 ms, which makes
            // every mock telemetry number 25 % wrong.
            let now = Instant::now();
            if now < self.next_mock_frame {
                std::thread::sleep(self.next_mock_frame - now);
            }
            self.next_mock_frame += MOCK_CADENCE;
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
        self.apply_commands();
        let settings = self.settings();
        let FrameOutcome {
            mut detection,
            frame,
            capture_ms,
            inference_ms,
            latency_ms,
        } = outcome;

        // Debounce the detector. The anchor score flickers around the threshold, so
        // a raw compare makes the note click on and off while the hand sits still:
        // a gesture needs a couple of confirmed sightings to start, and a couple of
        // missed frames do not end it.
        let present = self.gate.observe(detection.hand_landmarks.is_some());
        if !present {
            detection.hand_landmarks = None;
            detection.r#box = None;
            detection.score = 0.0;
        } else if detection.hand_landmarks.is_none() {
            // Inside the grace window: keep the last known keypoints instead of
            // releasing, so a one-frame dropout is inaudible.
            detection.hand_landmarks = Some(self.previous_landmarks.clone());
            detection.r#box = self.metrics.hand_box;
            detection.score = self.metrics.confidence;
        }

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

        // Stage timings, in milliseconds, into the rolling rings the summary and
        // the dashboard read as percentiles.
        self.stage_capture.push(capture_ms);
        self.stage_inference.push(inference_ms);
        self.stage_latency.push(latency_ms);

        self.metrics.frames += 1;
        if self.metrics.frames.is_multiple_of(PROFILE_REFRESH_FRAMES) {
            self.metrics.perf = self.stage_profile();
        }
        self.metrics.uptime_ms = self.started.elapsed().as_millis() as u64;
        self.metrics.paused = false;
        self.metrics.frame_size = detection.frame_size;
        self.metrics.frame_ms = inference_ms;
        self.metrics.capture_ms = capture_ms;
        self.metrics.latency_ms = latency_ms;
        self.metrics.captured = self.capture.captured.load(Ordering::Relaxed);
        self.metrics.dropped = self.capture.dropped.load(Ordering::Relaxed);

        if let Some(frame) = frame.as_ref() {
            if should_upload_feed(self.emit_frames, settings.feed) && feed_due(self.next_feed, now)
            {
                self.next_feed = now + feed_interval(settings.feed_fps);
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

        // ASCII backdrop. It has its own cadence and its own toggle: the art is
        // a deliberately coarse view of the frame, so building it costs a resize
        // plus one pass per cell instead of a 1.2 MB texture upload, and it can
        // update slower than the skeleton without looking stale.
        if ascii_wanted(
            self.emit_frames,
            settings.ascii,
            &self.ascii_dump,
            self.ascii_stdout,
        ) && feed_due(self.next_ascii, now)
        {
            self.next_ascii = now + feed_interval(settings.ascii_fps);
            let grid = match &frame {
                Some(frame) => self
                    .ascii_layer
                    .render(frame, self.ascii_cols)
                    .unwrap_or_else(|error| {
                        if self.debug {
                            eprintln!("[Ascii] {error}");
                        }
                        AsciiGrid::default()
                    }),
                // Mock mode has no camera pixels, so the art comes from the
                // synthetic skeleton - the toggle still demos without a webcam.
                None if !self.metrics.landmarks.is_empty() => self.ascii_layer.render_landmarks(
                    &self.metrics.landmarks,
                    detection.frame_size,
                    self.ascii_cols,
                ),
                None => AsciiGrid::default(),
            };
            self.metrics.ascii = grid;

            if !self.metrics.ascii.is_empty() {
                let ramp = settings.ramp;
                if self.ascii_stdout {
                    print_ascii(&self.metrics.ascii, ramp);
                }
                if let Some(path) = &self.ascii_dump {
                    if let Err(error) = std::fs::write(path, self.metrics.ascii.text(ramp)) {
                        if self.debug {
                            eprintln!("[Ascii] failed to write {path}: {error}");
                        }
                    }
                }
            }
        }

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

        self.publish();

        if self.debug && self.metrics.frames.is_multiple_of(10) {
            // What the player actually hears this frame: a keyboard note, a
            // theremin frequency, or nothing.
            let sounding = match (self.metrics.note, self.metrics.frequency) {
                (Some(note), _) => format!("note {} ({})", note, note_name(note)),
                (None, Some(freq)) => format!("{freq:.0} Hz"),
                _ => "silent".to_string(),
            };
            println!(
                "[Pipeline] fps {:.1} inference {:.2} ms (capture {:.2} ms, latency {:.2} ms, dropped {}) detector {:.0} us, landmarker {:.0} us conf {:.2} {sounding} vol {:.2} spread {:.2} jitter {:.1} px tip {:.0},{:.0}",
                self.metrics.fps,
                self.metrics.frame_ms,
                self.metrics.capture_ms,
                self.metrics.latency_ms,
                self.metrics.dropped,
                self.metrics.detector_us,
                self.metrics.landmarker_us,
                self.metrics.confidence,
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
                format!("note {} ({}) vel {:.2}", note, note_name(*note), velocity)
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
    control: &Arc<Mutex<RuntimeConfig>>,
    running: &AtomicBool,
    counters: &CaptureCounters,
) -> Result<()> {
    while running.load(Ordering::Relaxed) {
        if !control.lock().map(|config| config.active).unwrap_or(false) {
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

fn drain_events(rx: &Receiver<BusMessage>, sinks: &SinkList, debug: bool) {
    while let Ok(message) = rx.try_recv() {
        match message {
            BusMessage::Event(queued) => apply_to_sinks(sinks, &queued.event, debug),
            BusMessage::Shutdown => break,
        }
    }
}

/// The stage only needs camera pixels while the feed toggle is on; headless runs
/// never need them.
fn should_upload_feed(emit_frames: bool, want: bool) -> bool {
    emit_frames && want
}

/// The ASCII grid is only worth building if something will look at it: the stage
/// toggle, a terminal run, or a dump file. A dump or terminal run works even
/// with `--no-ui`, where there is no stage to toggle.
fn ascii_wanted(emit_frames: bool, want: bool, dump: &Option<String>, stdout: bool) -> bool {
    stdout || dump.is_some() || should_upload_feed(emit_frames, want)
}

/// Home the cursor rather than clearing the screen, so consecutive frames
/// overwrite in place and the art animates instead of scrolling.
fn print_ascii(grid: &AsciiGrid, ramp: Ramp) {
    print!("\x1B[H{}", grid.text(ramp));
}

/// `fps <= 0` means "every frame". The interval is clamped to a minute so an
/// absurd rate cannot overflow `Instant + Duration`.
fn feed_interval(fps: f32) -> Duration {
    if fps > 0.0 {
        Duration::from_secs_f64((1.0 / fps as f64).min(60.0))
    } else {
        Duration::ZERO
    }
}

/// Is a camera frame due? `next` is the wall clock at which the upload is
/// allowed; an interval of zero makes every frame due.
fn feed_due(next: Instant, now: Instant) -> bool {
    now >= next
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
    for pixel in src.as_chunks::<3>().0 {
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
    use crate::core::runtime::control_channel;
    use crate::strategies::FingerPositionStrategy;
    use crate::vision::ascii::{DEFAULT_ASCII_FPS, DEFAULT_COLS};

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
            self.waveforms
                .lock()
                .unwrap()
                .push(waveform.name().to_string());
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

    /// A continuous gesture, like the theremin: no note, just frequency and
    /// loudness on every frame.
    struct AlwaysSignal;

    impl Strategy for AlwaysSignal {
        fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent> {
            detection
                .hand_landmarks
                .as_ref()
                .map(|_| GestureEvent::new_signal(220.0, 0.5, "Theremin"))
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

    /// A mock pipeline plus the handles a test wants to inspect. A struct rather
    /// than a four-tuple: the call sites read as names instead of positional slots.
    struct TestHarness {
        pipeline: Pipeline,
        events: Arc<Mutex<Vec<GestureEvent>>>,
        /// The dashboard's half of the control channel: a test "clicks" a button by
        /// sending the same command the UI sends.
        control: Sender<UiCommand>,
        waveforms: Arc<Mutex<Vec<String>>>,
    }

    fn test_pipeline(strategy: Box<dyn Strategy>, max_frames: Option<u32>) -> TestHarness {
        test_pipeline_with(
            strategy,
            max_frames,
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks),
            false,
            DEFAULT_COLS,
            Arc::new(Mutex::new(FrameMetrics::default())),
        )
    }

    /// A pipeline with no camera and no model: the synthetic hand drives every
    /// frame, so the loop can be tested without a webcam or the ONNX assets.
    fn test_pipeline_with(
        strategy: Box<dyn Strategy>,
        max_frames: Option<u32>,
        config: RuntimeConfig,
        emit_frames: bool,
        ascii_cols: usize,
        shared: Arc<Mutex<FrameMetrics>>,
    ) -> TestHarness {
        let (bus, rx) = EventBus::new();
        let (control_tx, control_rx) = control_channel();
        let events = Arc::new(Mutex::new(Vec::new()));
        let waveforms = Arc::new(Mutex::new(Vec::new()));
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
            shared,
            control: Arc::new(Mutex::new(config)),
            control_rx,
            instrument: InstrumentConfig::default(),
            ascii_cols,
            ascii_dump: None,
            ascii_stdout: false,
            max_frames,
            debug: false,
            emit_frames,
        });
        TestHarness {
            pipeline,
            events,
            control: control_tx,
            waveforms,
        }
    }

    #[test]
    #[ignore]
    /// Quantifies what the feed gate and the cadence save. Run with
    /// `cargo test --release -- --include-ignored bench_feed_upload --nocapture`.
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
        let converted = start.elapsed().as_secs_f64() * 1000.0 / 200.0;

        println!("frame_to_rgba 640x480: {:.3} ms/frame", converted);
    }

    #[test]
    #[ignore]
    /// What publishing the snapshot costs with the camera pixels inside it, which
    /// is the copy `Pipeline::publish` now avoids by moving the buffer. Run with
    /// `cargo test --release -- --include-ignored bench_publish_snapshot --nocapture`.
    fn bench_publish_snapshot() {
        const FRAME: usize = 640 * 480 * 4;
        let full = FrameMetrics {
            frame_pixels: vec![7u8; FRAME],
            ..FrameMetrics::default()
        };
        let empty = FrameMetrics::default();

        let start = Instant::now();
        let mut carried = 0usize;
        for _ in 0..200 {
            let snapshot = full.clone();
            carried += snapshot.frame_pixels.len();
        }
        let with_pixels = start.elapsed().as_secs_f64() * 1000.0 / 200.0;

        let start = Instant::now();
        let mut carried_empty = 0usize;
        for _ in 0..200 {
            let snapshot = empty.clone();
            carried_empty += snapshot.frame_pixels.len();
        }
        let without_pixels = start.elapsed().as_secs_f64() * 1000.0 / 200.0;

        println!(
            "FrameMetrics publish: {:.3} ms with a 1.2 MB frame, {:.3} ms without ({:.3} ms saved)",
            with_pixels,
            without_pixels,
            with_pixels - without_pixels
        );
        assert_eq!(carried, 200 * FRAME, "the clone really carried the pixels");
        assert_eq!(carried_empty, 0);
    }

    #[test]
    fn a_continuous_signal_replaces_the_note_in_the_snapshot_and_the_log() {
        let TestHarness {
            mut pipeline,
            events,
            ..
        } = test_pipeline(Box::new(AlwaysSignal), None);
        pipeline.tick().unwrap(); // first sighting: the gate is still confirming
        pipeline.tick().unwrap(); // second sighting: the gesture is live

        // The OUTPUT card reads `frequency` when there is no note, so a theremin
        // run must publish the hertz and not leave a stale note behind.
        assert_eq!(pipeline.metrics.frequency, Some(220.0));
        assert_eq!(pipeline.metrics.note, None);
        assert!((pipeline.metrics.volume - 0.5).abs() < 1e-6);

        let logged = pipeline.metrics.log.back().unwrap().detail.clone();
        assert!(logged.contains("Hz"), "log said {logged:?}");

        let recorded = events.lock().unwrap();
        assert!(matches!(
            recorded.last().unwrap().payload,
            GesturePayload::Signal { .. }
        ));
    }

    #[test]
    fn a_picker_change_rekeys_the_keyboard_on_the_next_frame() {
        let TestHarness {
            mut pipeline,
            events,
            control,
            ..
        } = test_pipeline(Box::new(FingerPositionStrategy::default()), None);
        pipeline.tick().unwrap();
        assert_eq!(
            pipeline.metrics.layout.len(),
            15,
            "the default grid is 3 x 5"
        );
        assert_eq!(pipeline.metrics.scale, "major pentatonic");

        let instrument = InstrumentConfig::new("chromatic", "C", 4, Waveform::Saw).unwrap();
        control.send(UiCommand::SetInstrument(instrument)).unwrap();

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
        assert!(
            pipeline.metrics.layout.notes.contains(&note),
            "{note} is not in the new grid"
        );
    }

    #[test]
    fn commands_land_in_the_config_the_pipeline_owns() {
        let TestHarness {
            mut pipeline,
            control,
            ..
        } = test_pipeline(Box::new(AlwaysOn), None);
        assert!(pipeline.is_active(), "a run starts running");

        control.send(UiCommand::Stop).unwrap();
        control.send(UiCommand::SetAscii(true)).unwrap();
        control.send(UiCommand::SetRamp(Ramp::Shades)).unwrap();
        pipeline.apply_commands();

        let settings = pipeline.settings();
        assert!(!settings.active, "Stop cleared the flag");
        assert!(settings.ascii, "the ASCII toggle reached the config");
        assert_eq!(settings.ramp, Ramp::Shades, "the ramp reached the config");

        control.send(UiCommand::Start).unwrap();
        pipeline.apply_commands();
        assert!(pipeline.is_active(), "Start runs the loop again");
    }

    #[test]
    fn an_unchanged_config_does_not_rebuild_every_frame() {
        let TestHarness { mut pipeline, .. } =
            test_pipeline(Box::new(FingerPositionStrategy::default()), None);
        let before = pipeline.metrics.layout.clone();
        pipeline.apply_commands();
        pipeline.apply_commands();
        assert_eq!(
            pipeline.metrics.layout, before,
            "an idle control channel must not re-key the keyboard"
        );
    }

    #[test]
    fn the_feed_is_only_converted_when_the_stage_shows_it() {
        assert!(!should_upload_feed(true, false), "feed off means no upload");
        assert!(should_upload_feed(true, true));
        assert!(!should_upload_feed(false, true), "headless never uploads");
    }

    #[test]
    fn the_preview_upload_is_throttled_to_its_cadence() {
        let deadline = Instant::now();
        let later = deadline + Duration::from_millis(66);
        assert!(
            feed_due(deadline, deadline),
            "a frame at the deadline is due"
        );
        assert!(feed_due(deadline, later), "an overdue deadline is due");
        assert!(
            !feed_due(later, deadline),
            "nothing is due before its deadline"
        );
        assert_eq!(
            feed_interval(0.0),
            Duration::ZERO,
            "a zero rate means every frame"
        );
        assert_eq!(
            feed_interval(-1.0),
            Duration::ZERO,
            "so does a nonsense rate"
        );
    }

    #[test]
    fn pixels_are_moved_into_the_snapshot_and_dropped_between_uploads() {
        let frame = opencv::core::Mat::new_rows_cols_with_default(
            480,
            640,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::new(10.0, 20.0, 30.0, 0.0),
        )
        .unwrap();
        let shared = Arc::new(Mutex::new(FrameMetrics::default()));
        let TestHarness { mut pipeline, .. } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            RuntimeConfig {
                active: true,
                feed: true,
                // The art has its own cadence: the frame is rendered twice, but only
                // the first is inside the interval.
                ascii: true,
                feed_fps: 1.0,
                ascii_fps: 1.0,
                ramp: Ramp::Blocks,
            },
            true,
            32,
            shared.clone(),
        );

        let outcome = || {
            let mut detection = synthetic_detection(0.0);
            detection.frame_size = (640, 480);
            FrameOutcome {
                detection,
                frame: Some(frame.clone()),
                capture_ms: 1.0,
                inference_ms: 1.0,
                latency_ms: 1.0,
            }
        };

        pipeline.advance(outcome()).unwrap();
        {
            let snapshot = shared.lock().unwrap();
            assert_eq!(
                snapshot.frame_pixels.len(),
                640 * 480 * 4,
                "the first frame is uploaded"
            );
            assert_eq!(
                &snapshot.frame_pixels[..4],
                &[30, 20, 10, 255],
                "BGR became RGBA"
            );
            // 640x480 at 32 columns is 24 rows, and the frame is one flat grey,
            // so every cell has to land in the same bucket.
            assert_eq!((snapshot.ascii.cols, snapshot.ascii.rows), (32, 24));
            assert!(
                snapshot
                    .ascii
                    .cells
                    .iter()
                    .all(|&level| level == snapshot.ascii.cells[0]),
                "a flat frame must produce a flat grid"
            );
        }
        // The buffer was moved into the snapshot rather than cloned, so the
        // pipeline is left holding nothing instead of a second 1.2 MB copy.
        assert!(
            pipeline.metrics.frame_pixels.is_empty(),
            "publishing must not keep a second copy of the frame"
        );

        // A second frame inside the interval must not rebuild the art: the grid
        // is still the render from the first frame, even though this one is white.
        let bright = opencv::core::Mat::new_rows_cols_with_default(
            480,
            640,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::all(255.0),
        )
        .unwrap();
        let mut detection = synthetic_detection(0.0);
        detection.frame_size = (640, 480);
        pipeline
            .advance(FrameOutcome {
                detection,
                frame: Some(bright),
                capture_ms: 1.0,
                inference_ms: 1.0,
                latency_ms: 1.0,
            })
            .unwrap();

        let snapshot = shared.lock().unwrap();
        assert!(
            snapshot.frame_pixels.is_empty(),
            "a frame inside the feed interval carries no pixels"
        );
        assert!(
            snapshot.ascii.cells.iter().all(|&level| level == 0),
            "the ASCII grid keeps the previous render until its own cadence is due"
        );
    }

    #[test]
    fn publishing_preserves_the_run_context() {
        let shared = Arc::new(Mutex::new(FrameMetrics {
            mode: "Signal".to_string(),
            sink: "audio".to_string(),
            scale: "major pentatonic".to_string(),
            root: "C".to_string(),
            layout: GridLayout::new(
                0,
                &crate::core::scale::preset("major pentatonic").unwrap(),
                3,
            ),
            ..FrameMetrics::default()
        }));
        let TestHarness { mut pipeline, .. } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            RuntimeConfig::new(0.0, DEFAULT_ASCII_FPS, false, Ramp::Blocks),
            false,
            DEFAULT_COLS,
            shared.clone(),
        );
        pipeline.tick().unwrap();
        pipeline.tick().unwrap();

        let snapshot = shared.lock().unwrap();
        assert_eq!(snapshot.scale, "major pentatonic");
        assert_eq!(snapshot.root, "C");
        assert_eq!(snapshot.layout.len(), 15);
        assert_eq!(snapshot.mode, "Signal");
        assert!(snapshot.detected, "the mock hand should register");
    }

    #[test]
    fn the_ascii_layer_only_runs_when_something_wants_to_see_it() {
        assert!(
            !ascii_wanted(true, false, &None, false),
            "a cleared toggle and no sink means no work"
        );
        assert!(
            ascii_wanted(true, true, &None, false),
            "the stage is drawing it"
        );
        assert!(
            ascii_wanted(false, false, &Some("art.txt".to_string()), false),
            "a dump file is a reason to build the grid even headless"
        );
        assert!(ascii_wanted(false, false, &None, true), "stdout counts too");
    }

    #[test]
    fn mock_mode_draws_ascii_from_the_synthetic_hand() {
        let shared = Arc::new(Mutex::new(FrameMetrics::default()));
        let TestHarness { mut pipeline, .. } = test_pipeline_with(
            Box::new(FingerPositionStrategy::new(
                0,
                &crate::core::scale::preset("major pentatonic").unwrap(),
                3,
            )),
            None,
            RuntimeConfig {
                active: true,
                feed: false,
                ascii: true,
                feed_fps: 0.0,
                ascii_fps: 0.0,
                ramp: Ramp::Blocks,
            },
            true,
            40,
            shared.clone(),
        );
        // Two frames: the detection gate needs a second sighting before the hand
        // counts, and the mock art is drawn from the landmarks.
        pipeline.tick().unwrap();
        pipeline.tick().unwrap();

        let snapshot = shared.lock().unwrap();
        assert_eq!((snapshot.ascii.cols, snapshot.ascii.rows), (40, 30));
        assert!(
            snapshot.ascii.cells.iter().any(|&level| level > 0),
            "the mock hand lights up cells"
        );
        assert!(
            snapshot.ascii.cells.contains(&0),
            "the empty background stays empty"
        );
    }

    #[test]
    fn pausing_releases_the_note_and_marks_the_snapshot() {
        let TestHarness {
            mut pipeline,
            events,
            control,
            ..
        } = test_pipeline(Box::new(AlwaysOn), None);

        // With no camera the pipeline synthesises a hand, so the strategy fires once
        // the gate has confirmed the gesture.
        pipeline.tick().unwrap();
        pipeline.tick().unwrap();

        control.send(UiCommand::Stop).unwrap();
        pipeline.apply_commands();
        pipeline.pause();

        let recorded = events.lock().unwrap().clone();
        assert!(
            recorded
                .iter()
                .any(|e| matches!(e.payload, GesturePayload::Note { note: 0, .. })),
            "stopping must release the held note"
        );
        assert!(pipeline.metrics.paused);
        assert!(pipeline.metrics.note.is_none());
        assert_eq!(pipeline.metrics.volume, 0.0);
    }

    #[test]
    fn a_release_event_is_sent_when_the_gesture_ends() {
        let TestHarness {
            mut pipeline,
            events,
            ..
        } = test_pipeline(
            Box::new(FirstFrameOnly {
                armed: AtomicBool::new(true),
            }),
            None,
        );

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
        let TestHarness {
            mut pipeline,
            events,
            ..
        } = test_pipeline(
            Box::new(FirstFrameOnly {
                armed: AtomicBool::new(false),
            }),
            None,
        );
        pipeline.tick().unwrap();
        pipeline.tick().unwrap();
        assert!(events.lock().unwrap().is_empty());
        assert_eq!(pipeline.metrics.volume, 0.0);
        assert!(!pipeline.gesture_active);
    }

    #[test]
    fn the_gate_needs_two_sightings_and_only_releases_after_the_grace_window() {
        let TestHarness {
            mut pipeline,
            events,
            ..
        } = test_pipeline(Box::new(AlwaysOn), None);

        let frame = |detection: HandDetectionResult| FrameOutcome {
            detection,
            frame: None,
            capture_ms: 0.0,
            inference_ms: 0.0,
            latency_ms: 0.0,
        };
        let found = || {
            let mut detection = synthetic_detection(0.0);
            detection.frame_size = (640, 480);
            frame(detection)
        };
        let absent = || frame(HandDetectionResult::default());

        // The confidence flicker the tracker produces (0.97 -> 0.00 with a hand in
        // view) must not click the instrument on and off.
        pipeline.advance(found()).unwrap();
        pipeline.advance(absent()).unwrap();
        pipeline.advance(found()).unwrap();
        pipeline.advance(absent()).unwrap();
        assert!(events.lock().unwrap().is_empty(), "a flicker is not a hand");
        assert!(!pipeline.metrics.detected);

        // Two sightings in a row start the gesture; the grace window keeps it alive
        // while the anchor score dips, and only the fourth miss releases it.
        pipeline.advance(found()).unwrap();
        pipeline.advance(found()).unwrap();
        assert!(pipeline.metrics.detected, "the second sighting confirms it");
        pipeline.advance(absent()).unwrap();
        pipeline.advance(absent()).unwrap();
        pipeline.advance(absent()).unwrap();
        assert!(
            pipeline.metrics.detected,
            "three misses are still inside the grace window"
        );
        pipeline.advance(absent()).unwrap();
        assert!(!pipeline.metrics.detected, "the fourth miss loses the hand");

        let sent = events.lock().unwrap();
        assert!(
            sent.iter()
                .any(|e| matches!(e.payload, GesturePayload::Note { note: 0, .. })),
            "the release is emitted once the grace window expires"
        );
    }

    #[test]
    fn the_capture_stage_keeps_only_the_newest_frame() {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let counters = Arc::new(CaptureCounters::default());
        let running = Arc::new(AtomicBool::new(true));
        let control = Arc::new(Mutex::new(RuntimeConfig::new(
            DEFAULT_FEED_FPS,
            DEFAULT_ASCII_FPS,
            false,
            Ramp::Blocks,
        )));

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

        capture_frames(&mut read, &tx, &control, &running, &counters).unwrap();

        assert_eq!(counters.captured.load(Ordering::Relaxed), 5);
        assert_eq!(
            counters.dropped.load(Ordering::Relaxed),
            4,
            "a frame inference has not reached yet is dropped, not queued"
        );
        assert!(
            rx.try_recv().is_ok(),
            "inference always gets the newest frame"
        );
    }

    #[test]
    fn a_paused_capture_stage_reads_no_frames() {
        let (tx, _rx) = crossbeam_channel::bounded(1);
        let counters = Arc::new(CaptureCounters::default());
        let running = Arc::new(AtomicBool::new(true));
        let paused = RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks);
        let paused = Arc::new(Mutex::new(RuntimeConfig {
            active: false,
            ..paused
        }));

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

        capture_frames(&mut read, &tx, &paused, &running, &counters).unwrap();
        assert_eq!(produced, 0, "Stop means the camera is not read at all");
        assert_eq!(counters.captured.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_picker_change_reaches_the_sinks_while_the_loop_is_running() {
        let TestHarness {
            mut pipeline,
            control,
            waveforms,
            ..
        } = test_pipeline(Box::new(AlwaysOn), Some(8));
        let running = Arc::new(AtomicBool::new(true));

        // Change the timbre mid-run, exactly as clicking the picker does.
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(80));
            let mut instrument = InstrumentConfig::default();
            instrument.set_waveform(Waveform::Square);
            let _ = control.send(UiCommand::SetInstrument(instrument));
        });

        pipeline.run(&running).unwrap();

        let applied = waveforms.lock().unwrap().clone();
        assert!(
            applied.iter().any(|w| w == "square"),
            "the pipeline must push a picker change into the sinks, got {applied:?}"
        );
        assert_eq!(
            pipeline.metrics.waveform, "square",
            "the header shows the timbre in use"
        );
    }

    #[test]
    fn fps_is_the_publish_cadence_not_the_stage_times() {
        // Mock mode paces itself at ~30 fps and does no inference at all, so a
        // frame-rate read off the stage durations would report infinity.
        let TestHarness { mut pipeline, .. } = test_pipeline(Box::new(AlwaysOn), Some(20));
        let running = Arc::new(AtomicBool::new(true));
        pipeline.run(&running).unwrap();

        assert!(
            pipeline.metrics.frames >= 15,
            "the loop ran to the frame limit"
        );
        assert!(
            pipeline.metrics.fps > 20.0 && pipeline.metrics.fps < 45.0,
            "fps tracks the frames per second, got {}",
            pipeline.metrics.fps
        );
    }

    #[test]
    fn the_staged_run_terminates_and_the_sink_thread_flushes_every_event() {
        let TestHarness {
            mut pipeline,
            events,
            ..
        } = test_pipeline(Box::new(AlwaysOn), Some(3));
        let running = Arc::new(AtomicBool::new(true));

        pipeline.run(&running).unwrap();

        assert_eq!(
            pipeline.metrics.frames, 3,
            "--max-frames stops the frame loop"
        );
        assert!(
            pipeline.metrics.detector_us == 0.0,
            "mock mode runs no model"
        );
        let recorded = events.lock().unwrap();
        assert_eq!(
            recorded.len(),
            2,
            "events dispatched by inference reach the sinks on their own thread: of the \
             three frames, the first is spent confirming the detection"
        );
        assert!(
            !running.load(Ordering::Relaxed),
            "run clears the flag on exit"
        );
    }
}
