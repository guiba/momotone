//! The capture -> inference -> gesture -> sink loop, decoupled from the UI.
//!
//! `Pipeline::run` is the frame loop, split into three stages on three threads:
//!
//! * **capture** reads the webcam and pushes the frame into a capacity-1 channel.
//!   When inference is behind, the queued frame is dropped rather than making the
//!   pipeline work on stale pixels.
//! * **inference** is the pipeline thread: tracker -> strategy -> event bus, plus
//!   the `FrameMetrics` publish the dashboard reads.
//! * **sinks** drain the event bus and talk to the audio/MIDI outputs, so a
//!   blocking MIDI write cannot stall a frame.
//!
//! A frame is produced by `next_frame` - the webcam, or the synthetic hand in mock
//! mode - and everything else (`advance`) is the same whichever source it came from.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use opencv::prelude::*;

use crate::core::event_bus::{BusMessage, EventBus};
use crate::core::instrument::InstrumentConfig;
use crate::core::runtime::{RuntimeConfig, UiCommand};
use crate::core::telemetry::{profile, Samples, StageProfile, PROFILE_REFRESH_FRAMES};
use crate::core::types::Performance;
use crate::core::{HandDetectionResult, Landmark, Sink};
use crate::dashboard::metrics::{
    cell_index, note_name, EventLogEntry, FrameMetrics, KeyboardTrail, LOG_CAPACITY,
};
use crate::gesture::{HandEstimator, HandState, Strategy};
use crate::presentation::ascii::{Ramp, DEFAULT_ASCII_FPS};
use crate::presentation::presenter::{PresentJob, Presenter, DEFAULT_FEED_FPS};
use crate::vision::camera::WebcamSource;
use crate::vision::tracker::HandTracker;

/// Mock mode emulates a 30 fps camera so the telemetry looks real. The loop uses a
/// deadline rather than a flat sleep: macOS `thread::sleep(33 ms)` actually returns
/// after ~41 ms, so a naive sleep makes every mock number 25 % wrong.
const MOCK_CADENCE: Duration = Duration::from_millis(33);
/// How long inference waits for the capture stage before re-checking the flags.
const FRAME_POLL_MS: u64 = 20;
/// How often the paused loop checks the Start/Stop flag and refreshes the UI.
const PAUSE_POLL_MS: u64 = 50;

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

/// What the frame stage produced this iteration.
enum FrameRead {
    /// A frame to process.
    Frame(FrameOutcome),
    /// The capture stage has not handed one over yet.
    Waiting,
    /// The capture stage is gone: the camera closed or the run is winding down.
    Ended,
}

/// A mock frame: synthetic landmarks, no pixels, no model.
fn synthetic_frame(seconds: f64) -> FrameOutcome {
    FrameOutcome {
        detection: synthetic_detection(seconds),
        frame: None,
        capture_ms: 0.0,
        inference_ms: 0.0,
        latency_ms: 0.0,
    }
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
    // --- the stages -----------------------------------------------------------
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

    // --- what the hand is doing ---------------------------------------------
    /// The single source of truth about the hand: hysteresis, the grace window, the
    /// smoothed fingertip, the loudness and the geometry all live here.
    estimator: HandEstimator,
    /// True while a gesture is actively driving the sinks. Used to send exactly
    /// one release event when it stops, so the audio/MIDI sinks do not latch the
    /// last note and keep sounding after the hand leaves the frame.
    gesture_active: bool,
    /// The keyboard glow. Derived from the gesture, not from the window, so the
    /// stage draws a fact about the run instead of maintaining its own history.
    trail: KeyboardTrail,
    metrics: FrameMetrics,

    // --- run state -------------------------------------------------------------
    run: RunFlags,
    telemetry: Telemetry,
    presentation: Presentation,
    /// Taken by `run` and handed to the presenter thread.
    jobs_rx: Option<Receiver<PresentJob>>,
    /// Frames the capture stage read and dropped, so a bottleneck is visible
    /// instead of invisible.
    counters: Arc<CaptureCounters>,
}

/// How this run is bounded: when to stop, whether to publish frames at all, and
/// whether to print the per-frame line.
struct RunFlags {
    max_frames: Option<u32>,
    emit_frames: bool,
    debug: bool,
}

/// Clocks and the rolling samples of each stage: everything the dashboard and the
/// summary line measure.
struct Telemetry {
    started: Instant,
    /// Wall clock of the last published frame; the fps is derived from the gap
    /// between publishes, which stays honest once the stages overlap.
    last_publish: Instant,
    last_detection: Instant,
    /// Exponentially smoothed publish interval, in milliseconds.
    avg_gap_ms: f64,
    /// Deadline of the next synthetic frame, for mock mode's paced clock.
    next_mock: Instant,
    /// Rolling stage timings. The frame loop is paced by the slowest stage, so a
    /// regression shows up as a tail here long before the average moves.
    capture: Samples,
    inference: Samples,
    latency: Samples,
    /// Read by the sink thread, which is a different thread, hence the shared ring.
    dispatch: Arc<Mutex<Samples>>,
}

/// Presentation wiring: who the frame is handed to, and what that thread does with
/// it. All drawing, terminal printing and file I/O happens on the presenter's own
/// thread, so nothing the instrument does not need for sound can add latency.
struct Presentation {
    /// `None` once the run has ended: dropping the last sender closes the channel
    /// and the presenter's loop exits.
    jobs: Option<Sender<PresentJob>>,
    /// Taken by `run` and moved onto its own thread.
    presenter: Option<Presenter>,
    /// Terminal mode and the dump path: either makes a frame worth sending even with
    /// no window open.
    stdout: bool,
    dump: Option<String>,
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
    /// How long a released key keeps glowing on the stage.
    pub fade_ms: f32,
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
            fade_ms,
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
                fade_ms: snapshot.fade_ms,
                ..FrameMetrics::default()
            })
            .unwrap_or_default();

        // One clock reading for every deadline, so the first frame is due immediately.
        let now = Instant::now();
        // Presentation runs on its own thread: the frame loop only ever hands it a
        // frame, and never converts a pixel itself.
        let (jobs_tx, jobs_rx) = crossbeam_channel::bounded(1);
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
            estimator: HandEstimator::new(),
            gesture_active: false,
            trail: KeyboardTrail::new(fade_ms),
            counters: Arc::new(CaptureCounters::default()),
            run: RunFlags {
                max_frames,
                emit_frames,
                debug,
            },
            telemetry: Telemetry {
                started: now,
                last_publish: now,
                last_detection: now,
                avg_gap_ms: 0.0,
                next_mock: now,
                capture: Samples::default(),
                inference: Samples::default(),
                latency: Samples::default(),
                dispatch: Arc::new(Mutex::new(Samples::default())),
            },
            presentation: Presentation {
                jobs: Some(jobs_tx),
                presenter: Some(Presenter::new(
                    ascii_cols,
                    ascii_stdout,
                    ascii_dump.clone(),
                    debug,
                )),
                stdout: ascii_stdout,
                dump: ascii_dump,
            },
            jobs_rx: Some(jobs_rx),
        };
        // Seed the instrument so the strategy, the sinks and the published keyboard
        // agree from the very first frame.
        pipeline.apply_instrument(&instrument);
        pipeline
    }

    /// Hand the frame to the presenter. The channel holds one frame: a presenter
    /// still drawing the previous one drops this frame rather than drifting behind
    /// the camera, which is exactly right for a picture nobody is measuring.
    fn hand_over(&mut self, frame: Option<Mat>, state: &HandState, settings: &RuntimeConfig) {
        let Some(jobs) = &self.presentation.jobs else {
            return;
        };
        if !Presenter::wanted(
            self.run.emit_frames && settings.feed,
            settings.ascii,
            self.presentation.stdout,
            &self.presentation.dump,
        ) {
            return;
        }
        let _ = jobs.try_send(PresentJob {
            frame,
            landmarks: state.landmarks.clone(),
            frame_size: state.frame_size,
        });
    }

    /// Hand the frame snapshot to the dashboard. The pixel buffer is *moved*
    /// rather than cloned: the UI takes ownership of it on its next tick, so
    /// copying 1.2 MB of RGBA into a Vec that is handed straight back is a
    /// wasted copy every frame.
    fn publish(&mut self) {
        if let Ok(mut shared) = self.shared.lock() {
            // The presenter thread owns the picture; refreshing the telemetry must
            // not wipe it.
            let pixels = std::mem::take(&mut shared.frame_pixels);
            let grid = std::mem::take(&mut shared.ascii);
            *shared = self.metrics.clone();
            shared.frame_pixels = pixels;
            shared.ascii = grid;
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

        if self.run.debug {
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
            let counters = self.counters.clone();
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
            let debug = self.run.debug;
            let dispatch = self.telemetry.dispatch.clone();
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
                    apply_to_sinks(&sinks, &queued.performance, debug);
                }
                if let Ok(mut sinks) = sinks.lock() {
                    for sink in sinks.iter_mut() {
                        let _ = sink.stop();
                    }
                }
            })
        };

        // Stage 4: presentation. Converting a frame to RGBA, quantising it into the
        // ASCII grid, printing to a terminal and writing a dump file are all things
        // the instrument does not need in order to make sound; on their own thread
        // they cannot show up as frame latency.
        let presenter_handle = std::thread::spawn({
            let shared = self.shared.clone();
            let control = self.control.clone();
            let jobs = self.jobs_rx.take().unwrap();
            let presenter = self.presentation.presenter.take().unwrap();
            move || presenter.run(shared, control, jobs)
        });

        // Stage 2: this thread.
        while running.load(Ordering::Relaxed) {
            self.apply_commands();
            if !self.is_active() {
                // Still honour commands while stopped, so the header and the sinks
                // match what the dashboard shows before the player starts again.
                self.pause();
                std::thread::sleep(Duration::from_millis(PAUSE_POLL_MS));
                continue;
            }

            match self.next_frame(&frame_rx)? {
                FrameRead::Frame(outcome) => self.advance(outcome)?,
                FrameRead::Waiting => continue,
                FrameRead::Ended => break,
            }

            if self
                .run
                .max_frames
                .is_some_and(|n| self.metrics.frames >= n as u64)
            {
                break;
            }
        }

        if self.run.debug {
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
        // Dropping the last job sender ends the presenter after it has drawn what is
        // already queued.
        drop(self.presentation.jobs.take());
        if let Some(handle) = capture_handle {
            let _ = handle.join();
        }
        let _ = sink_handle.join();
        let _ = presenter_handle.join();
        Ok(())
    }

    /// Percentiles of the four stages. Recomputed once a second rather than every
    /// frame: the sort is cheap but repeating it 30 times a second is not needed.
    fn stage_profile(&self) -> StageProfile {
        let dispatch = self
            .telemetry
            .dispatch
            .lock()
            .map(|samples| profile(&samples))
            .unwrap_or_default();
        StageProfile {
            inference: profile(&self.telemetry.inference),
            capture: profile(&self.telemetry.capture),
            end_to_end: profile(&self.telemetry.latency),
            dispatch,
        }
    }

    /// While stopped, release whatever the sinks are holding and publish a
    /// "paused" snapshot once, so the note does not latch and the keyboard trail
    /// fades to nothing.
    fn pause(&mut self) {
        if self.gesture_active {
            self.gesture_active = false;
            self.event_bus.dispatch(Performance::Release);
        }
        if let Some(rx) = &self.event_rx {
            drain_events(rx, &self.sinks, self.run.debug);
        }
        // Otherwise the stopped interval shows up as a huge fps dip on resume.
        self.telemetry.last_publish = Instant::now();

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
        self.trail.clear();
        self.metrics.trail.clear();
        self.publish();
    }

    /// The next frame, from whichever source this run has: the webcam through the
    /// capacity-1 channel the capture thread fills, or the synthetic hand in mock
    /// mode (no camera, no model). Mock mode is paced by a deadline clock rather
    /// than a flat sleep, so its telemetry is honest.
    fn next_frame(&mut self, frames: &Receiver<CapturedFrame>) -> Result<FrameRead> {
        let Some(tracker) = self.tracker.as_mut() else {
            // Sleeping 33 ms on macOS returns after ~41 ms, which would make every
            // mock number 25 % wrong; aim at the deadline instead.
            let now = Instant::now();
            if now < self.telemetry.next_mock {
                std::thread::sleep(self.telemetry.next_mock - now);
            }
            self.telemetry.next_mock += MOCK_CADENCE;
            return Ok(FrameRead::Frame(synthetic_frame(
                self.telemetry.started.elapsed().as_secs_f64(),
            )));
        };

        match frames.recv_timeout(Duration::from_millis(FRAME_POLL_MS)) {
            Ok(captured) => {
                let inference_start = Instant::now();
                let detection = tracker.process_frame(&captured.frame)?;
                Ok(FrameRead::Frame(FrameOutcome {
                    detection,
                    frame: Some(captured.frame),
                    capture_ms: captured.capture_ms,
                    inference_ms: inference_start.elapsed().as_secs_f64() * 1000.0,
                    latency_ms: inference_start
                        .duration_since(captured.captured_at)
                        .as_secs_f64()
                        * 1000.0,
                }))
            }
            Err(RecvTimeoutError::Timeout) => Ok(FrameRead::Waiting),
            Err(RecvTimeoutError::Disconnected) => Ok(FrameRead::Ended),
        }
    }

    /// A frame, stage by stage: resolve the hand, record the timings, publish the
    /// tracking, run the gesture, hand the picture to the presenter.
    ///
    /// Each stage is a function of the frame and the pipeline's state, so the
    /// gesture code can be exercised without a camera or an ONNX model.
    fn advance(&mut self, outcome: FrameOutcome) -> Result<()> {
        self.apply_commands();
        let settings = self.settings();
        let FrameOutcome {
            detection,
            frame,
            capture_ms,
            inference_ms,
            latency_ms,
        } = outcome;

        // One resolution of the raw detection, shared by every stage below.
        let state = self.estimator.update(&detection);

        self.record_timing(&detection, capture_ms, inference_ms, latency_ms);
        self.update_tracking(&state);
        self.dispatch_gesture(&state);
        // The keyboard glow follows the gesture, on the frame loop's clock.
        self.update_trail();
        // Last, and off this thread: nothing a human looks at can delay a frame.
        self.hand_over(frame, &state, &settings);

        // Inline mode: this thread still owns the receiver. Once `run` has spawned
        // the sink thread it is gone, and that stage is the only consumer.
        if let Some(rx) = &self.event_rx {
            drain_events(rx, &self.sinks, self.run.debug);
        }
        self.publish();
        self.report_frame();
        Ok(())
    }

    /// Frame counter, publish-cadence fps and the rolling percentiles of each stage.
    fn record_timing(
        &mut self,
        detection: &HandDetectionResult,
        capture_ms: f64,
        inference_ms: f64,
        latency_ms: f64,
    ) {
        let now = Instant::now();
        let gap_ms = now
            .duration_since(self.telemetry.last_publish)
            .as_secs_f64()
            * 1000.0;
        self.telemetry.last_publish = now;

        // Smooth the *interval*, not the reciprocal rate. With the stages
        // overlapping, individual gaps alternate between a queued frame (a
        // millisecond) and a full inference, and an EMA of fps turns that jitter
        // into a number that bears no relation to the frame rate.
        self.telemetry.avg_gap_ms = if self.telemetry.avg_gap_ms == 0.0 {
            gap_ms
        } else {
            0.9 * self.telemetry.avg_gap_ms + 0.1 * gap_ms
        };
        self.metrics.fps = (1000.0 / self.telemetry.avg_gap_ms.max(1.0)) as f32;

        // Stage timings, in milliseconds, into the rolling rings the summary and
        // the dashboard read as percentiles.
        self.telemetry.capture.push(capture_ms);
        self.telemetry.inference.push(inference_ms);
        self.telemetry.latency.push(latency_ms);

        self.metrics.frames += 1;
        if self.metrics.frames.is_multiple_of(PROFILE_REFRESH_FRAMES) {
            self.metrics.perf = self.stage_profile();
        }
        self.metrics.uptime_ms = self.telemetry.started.elapsed().as_millis() as u64;
        self.metrics.paused = false;
        self.metrics.frame_size = detection.frame_size;
        self.metrics.frame_ms = inference_ms;
        self.metrics.capture_ms = capture_ms;
        self.metrics.latency_ms = latency_ms;
        self.metrics.captured = self.counters.captured.load(Ordering::Relaxed);
        self.metrics.dropped = self.counters.dropped.load(Ordering::Relaxed);
        self.metrics.detector_us = detection.timings.detector_us;
        self.metrics.landmarker_us = detection.timings.landmarker_us;
    }

    /// Where the hand is, copied out of the estimator into the snapshot the
    /// dashboard reads.
    fn update_tracking(&mut self, state: &HandState) {
        self.metrics.detected = state.present;
        self.metrics.confidence = state.confidence;
        self.metrics.hand_box = state.hand_box;
        // `clone_from` keeps the Vec's capacity, so the landmark list stops
        // reallocating every frame.
        self.metrics.landmarks.clone_from(&state.landmarks);
        self.metrics.landmark_count = state.landmarks.len();
        // The note anchor: the smoothed index fingertip the pitch is read from.
        self.metrics.tip_x = state.tip.0;
        self.metrics.tip_y = state.tip.1;
        // Which cell the sounding note belongs to, for the crosshair and the
        // cell highlight. Notes are unique in the grid, so this is exact.
        if let Some(note) = self.metrics.note {
            if let Some(cell) = cell_index(&self.metrics.layout, note) {
                self.metrics.grid_row = cell / self.metrics.layout.cols;
                self.metrics.grid_col = cell % self.metrics.layout.cols;
            }
        }

        if state.present {
            self.telemetry.last_detection = Instant::now();
        }
        self.metrics.lost_ms = self.telemetry.last_detection.elapsed().as_millis() as u64;
        self.metrics.jitter_px = state.jitter_px;

        self.metrics.spread = state.geometry.spread;
        self.metrics.extension = state.geometry.extension;
        self.metrics.tilt_deg = state.geometry.tilt_deg;
        self.metrics.palm_length = state.geometry.palm_length;
    }

    /// Fade the keyboard trail and light the key that is sounding. The trail is
    /// part of the snapshot rather than the window, so pausing the run also puts
    /// the keyboard dark and a resumed run starts from a clean grid.
    fn update_trail(&mut self) {
        self.trail.advance(Instant::now(), self.metrics.note);
        // Reuse the snapshot's buffer instead of allocating 49 floats a frame.
        self.metrics.trail.clear();
        self.metrics.trail.extend(self.trail.levels());
        self.metrics.fade_ms = self.trail.fade_ms();
    }

    /// Gesture -> metrics -> event bus.
    ///
    /// A sink that never receives anything keeps whatever it was last told, so the
    /// *absence* of a gesture has to be announced too: without the release the
    /// synthesizer keeps oscillating forever. It is announced exactly once, not on
    /// every silent frame.
    fn dispatch_gesture(&mut self, state: &HandState) {
        let Some(performance) = self.strategy.process(state) else {
            return self.release();
        };
        if matches!(performance, Performance::Release) {
            return self.release();
        }

        match &performance {
            Performance::Note { note, velocity } => {
                self.metrics.note = Some(*note);
                self.metrics.volume = *velocity;
                self.metrics.frequency = None;
            }
            Performance::Signal {
                frequency,
                amplitude,
            } => {
                self.metrics.frequency = Some(*frequency);
                self.metrics.volume = *amplitude;
                self.metrics.note = None;
            }
            Performance::Release => unreachable!("handled above"),
        }
        self.gesture_active = true;
        self.push_log(&performance);
        self.event_bus.dispatch(performance);
    }

    /// Silence the outputs, but only if something is actually sounding - a sink that
    /// is already silent should not be told to be silent 30 times a second.
    fn release(&mut self) {
        self.metrics.note = None;
        self.metrics.volume = 0.0;
        self.metrics.frequency = None;
        if self.gesture_active {
            self.gesture_active = false;
            self.push_log(&Performance::Release);
            self.event_bus.dispatch(Performance::Release);
        }
    }

    /// The line `--debug` prints every ten frames: what the player is hearing and
    /// how long the frame took.
    fn report_frame(&self) {
        if self.run.debug && self.metrics.frames.is_multiple_of(10) {
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
    }

    /// One line in the dashboard's event log. The strategy name used to come along
    /// on the event; the performance describes itself well enough.
    fn push_log(&mut self, performance: &Performance) {
        let detail = match performance {
            Performance::Note { note, velocity } => {
                format!("note {} ({}) vel {:.2}", note, note_name(*note), velocity)
            }
            Performance::Signal {
                frequency,
                amplitude,
            } => {
                format!("{:.0} Hz amp {:.2}", frequency, amplitude)
            }
            Performance::Release => "release".to_string(),
        };

        if self.metrics.log.len() >= LOG_CAPACITY {
            self.metrics.log.pop_front();
        }
        self.metrics.log.push_back(EventLogEntry {
            uptime_ms: self.metrics.uptime_ms,
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

fn apply_to_sinks(sinks: &SinkList, performance: &Performance, debug: bool) {
    let Ok(mut sinks) = sinks.lock() else {
        return;
    };
    for sink in sinks.iter_mut() {
        if let Err(e) = sink.handle(performance) {
            if debug {
                eprintln!("[Sink] {e}");
            }
        }
    }
}

fn drain_events(rx: &Receiver<BusMessage>, sinks: &SinkList, debug: bool) {
    while let Ok(message) = rx.try_recv() {
        match message {
            BusMessage::Event(queued) => apply_to_sinks(sinks, &queued.performance, debug),
            BusMessage::Shutdown => break,
        }
    }
}

/// A synthetic, moving hand so the dashboard can be exercised without a
/// webcam or the ONNX models (`--mode mock`).
pub(crate) fn synthetic_detection(seconds: f64) -> HandDetectionResult {
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
    use crate::core::geometry::hand_geometry;
    use crate::core::grid::GridLayout;
    use crate::core::instrument::Waveform;
    use crate::core::runtime::control_channel;
    use crate::dashboard::metrics::{note_index, DEFAULT_FADE_MS};
    use crate::gesture::keyboard::FingerPositionStrategy;
    use crate::presentation::ascii::{DEFAULT_ASCII_FPS, DEFAULT_COLS};

    #[test]
    fn synthetic_hand_has_21_landmarks() {
        let detection = synthetic_detection(1.0);
        let landmarks = detection.hand_landmarks.unwrap();
        assert_eq!(landmarks.len(), 21);
        assert!(detection.r#box.is_some());
    }

    #[test]
    fn synthetic_hand_geometry_is_sane() {
        let landmarks = synthetic_detection(2.0).hand_landmarks.unwrap();
        let geometry = hand_geometry(&landmarks).unwrap();
        assert!(geometry.palm_length > 0.0);
        assert!(geometry.spread > 0.0);
        assert!(geometry.extension[1] > geometry.extension[0]);
    }

    /// A sink that records every event it is handed, so the tests can inspect
    /// what the pipeline actually dispatched.
    #[derive(Clone)]
    struct RecordingSink {
        events: Arc<Mutex<Vec<Performance>>>,
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
        fn handle(&mut self, performance: &Performance) -> Result<()> {
            self.events.lock().unwrap().push(performance.clone());
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
        fn process(&mut self, state: &HandState) -> Option<Performance> {
            state.present.then_some(Performance::Note {
                note: 60,
                velocity: 0.8,
            })
        }
    }

    /// A continuous gesture, like the theremin: no note, just frequency and
    /// loudness on every frame.
    struct AlwaysSignal;

    impl Strategy for AlwaysSignal {
        fn process(&mut self, state: &HandState) -> Option<Performance> {
            state.present.then_some(Performance::Signal {
                frequency: 220.0,
                amplitude: 0.5,
            })
        }
    }

    /// Fires on the first frame only, then goes silent - i.e. the hand leaves.
    struct FirstFrameOnly {
        armed: AtomicBool,
    }

    impl Strategy for FirstFrameOnly {
        fn process(&mut self, state: &HandState) -> Option<Performance> {
            if !state.present || !self.armed.load(Ordering::Relaxed) {
                return None;
            }
            self.armed.store(false, Ordering::Relaxed);
            Some(Performance::Note {
                note: 60,
                velocity: 0.8,
            })
        }
    }

    /// A mock pipeline plus the handles a test wants to inspect. A struct rather
    /// than a four-tuple: the call sites read as names instead of positional slots.
    struct TestHarness {
        pipeline: Pipeline,
        events: Arc<Mutex<Vec<Performance>>>,
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
            fade_ms: DEFAULT_FADE_MS,
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
        pipeline.advance(synthetic_frame(0.0)).unwrap(); // first sighting: the gate is still confirming
        pipeline.advance(synthetic_frame(0.0)).unwrap(); // second sighting: the gesture is live

        // The OUTPUT card reads `frequency` when there is no note, so a theremin
        // run must publish the hertz and not leave a stale note behind.
        assert_eq!(pipeline.metrics.frequency, Some(220.0));
        assert_eq!(pipeline.metrics.note, None);
        assert!((pipeline.metrics.volume - 0.5).abs() < 1e-6);

        let logged = pipeline.metrics.log.back().unwrap().detail.clone();
        assert!(logged.contains("Hz"), "log said {logged:?}");

        let recorded = events.lock().unwrap();
        assert!(matches!(
            recorded.last().unwrap(),
            Performance::Signal { .. }
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
        pipeline.advance(synthetic_frame(0.0)).unwrap();
        assert_eq!(
            pipeline.metrics.layout.len(),
            15,
            "the default grid is 3 x 5"
        );
        assert_eq!(pipeline.metrics.scale, "major pentatonic");

        let instrument = InstrumentConfig::new("chromatic", "C", 4, Waveform::Saw).unwrap();
        control.send(UiCommand::SetInstrument(instrument)).unwrap();

        pipeline.advance(synthetic_frame(0.0)).unwrap();
        assert_eq!(pipeline.metrics.layout.rows, 4);
        assert_eq!(pipeline.metrics.layout.cols, 12);
        assert_eq!(pipeline.metrics.scale, "chromatic");

        // The note the strategy just produced must belong to the new grid, not
        // the old one: the layout was pushed into the strategy, not merely into
        // the dashboard's copy of it.
        let recorded = events.lock().unwrap();
        let note = match recorded.last().unwrap() {
            Performance::Note { note, .. } => note,
            _ => panic!("expected a note performance"),
        };
        assert!(
            pipeline.metrics.layout.notes.contains(note),
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
    fn the_picture_is_handed_over_only_when_someone_is_watching() {
        let TestHarness {
            mut pipeline,
            control,
            ..
        } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks),
            true,
            DEFAULT_COLS,
            Arc::new(Mutex::new(FrameMetrics::default())),
        );
        let jobs = pipeline.jobs_rx.take().unwrap();

        // Nothing is on, so the frame is dropped rather than copied to a thread that
        // has nowhere to draw it.
        pipeline.advance(synthetic_frame(0.0)).unwrap();
        assert!(jobs.try_recv().is_err(), "no audience, no job");

        // The dashboard turns the feed on; now the frame goes over.
        control.send(UiCommand::SetFeed(true)).unwrap();
        pipeline.advance(synthetic_frame(0.0)).unwrap();
        let job = jobs.try_recv().unwrap();
        assert!(job.frame.is_none(), "a mock frame carries no pixels");
        assert_eq!(job.landmarks.len(), 21, "but it does carry the hand");
        assert_eq!(job.frame_size, (640, 480));
    }

    #[test]
    fn publishing_keeps_whatever_the_presenter_drew() {
        // The presenter writes the picture into the snapshot on its own thread; a
        // telemetry refresh from the frame thread must not wipe it.
        let shared = Arc::new(Mutex::new(FrameMetrics {
            frame_pixels: vec![7u8; 8],
            ascii: crate::presentation::ascii::AsciiGrid {
                cols: 4,
                rows: 2,
                cells: vec![3u8; 8],
            },
            ..FrameMetrics::default()
        }));
        let TestHarness { mut pipeline, .. } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks),
            false,
            DEFAULT_COLS,
            shared.clone(),
        );
        pipeline.advance(synthetic_frame(0.0)).unwrap();

        let snapshot = shared.lock().unwrap();
        assert_eq!(snapshot.frames, 1, "the telemetry moved");
        assert_eq!(snapshot.frame_pixels, vec![7u8; 8], "the picture survived");
        assert_eq!(snapshot.ascii.cols, 4, "and so did the art");
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
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks),
            false,
            DEFAULT_COLS,
            shared.clone(),
        );
        pipeline.advance(synthetic_frame(0.0)).unwrap();
        pipeline.advance(synthetic_frame(0.0)).unwrap();

        let snapshot = shared.lock().unwrap();
        assert_eq!(snapshot.scale, "major pentatonic");
        assert_eq!(snapshot.root, "C");
        assert_eq!(snapshot.layout.len(), 15);
        assert_eq!(snapshot.mode, "Signal");
        assert!(snapshot.detected, "the mock hand should register");
    }

    #[test]
    fn the_keyboard_trail_is_part_of_the_snapshot_not_the_window() {
        let harness = test_pipeline(Box::new(AlwaysOn), Some(5));
        let mut pipeline = harness.pipeline;
        pipeline.run(&Arc::new(AtomicBool::new(true))).unwrap();

        // The held key is lit, and the glow is published so the stage needs no
        // state of its own.
        assert_eq!(pipeline.metrics.trail[note_index(60)], 1.0);
        assert_eq!(pipeline.metrics.fade_ms, DEFAULT_FADE_MS);

        pipeline.pause();
        assert!(
            pipeline.metrics.trail.iter().all(|&level| level == 0.0),
            "a stopped player has no glowing key"
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
        pipeline.advance(synthetic_frame(0.0)).unwrap();
        pipeline.advance(synthetic_frame(0.0)).unwrap();

        control.send(UiCommand::Stop).unwrap();
        pipeline.apply_commands();
        pipeline.pause();

        let recorded = events.lock().unwrap().clone();
        assert!(
            recorded
                .iter()
                .any(|performance| matches!(performance, Performance::Release)),
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

        pipeline.advance(synthetic_frame(0.0)).unwrap(); // hand present -> note
        pipeline.advance(synthetic_frame(0.0)).unwrap(); // hand gone   -> release
        pipeline.advance(synthetic_frame(0.0)).unwrap(); // still gone  -> nothing more

        let sent = events.lock().unwrap();
        assert_eq!(sent.len(), 2, "exactly one note and one release");
        match sent[0] {
            Performance::Note { note, velocity } => {
                assert_eq!(note, 60);
                assert!((velocity - 0.8).abs() < 1e-6);
            }
            _ => panic!("expected the gesture note first"),
        }
        assert!(
            matches!(sent[1], Performance::Release),
            "the second message must be the release"
        );
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
        pipeline.advance(synthetic_frame(0.0)).unwrap();
        pipeline.advance(synthetic_frame(0.0)).unwrap();
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
                .any(|performance| matches!(performance, Performance::Release)),
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
