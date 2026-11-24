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

use crate::core::effects::{
    blend, gap_level, is_different, modulation_from_hands, openness, Modulation,
};
use crate::core::event_bus::{BusMessage, EventBus};
use crate::core::frame::Frame;
use crate::core::instrument::InstrumentConfig;
use crate::core::runtime::{RuntimeConfig, UiCommand};
use crate::core::telemetry::{shed_subsystems, RunStats, PROFILE_REFRESH_FRAMES};
use crate::core::trace::Trace;
use crate::core::types::Performance;
use crate::core::{box_centre, HandDetectionResult, Sink, MAX_HANDS};
use crate::dashboard::metrics::{
    cell_index, note_frequency, note_name, EventLogEntry, FrameMetrics, KeyboardTrail, SceneView,
    SoundView, LOG_CAPACITY,
};
use crate::gesture::{band_position, HandEstimator, HandState, Strategy};
use crate::presentation::ascii::{Ramp, DEFAULT_ASCII_FPS};
use crate::presentation::presenter::{PresentJob, Presenter, DEFAULT_FEED_FPS};
use crate::vision::camera::WebcamSource;
use crate::vision::mock::{synthetic_detection, MOCK_CADENCE};
use crate::vision::tracker::HandTracker;
use crate::vision::video::VideoSource;
use anyhow::Result;
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};

/// Where the pixels come from. Mock mode has neither: `next_frame` synthesises the
/// hand instead of reading a frame.
///
/// A recorded clip is the reproducible option - the same frames every run, which is
/// what makes a golden trace test possible.
pub enum FrameSource {
    Camera(WebcamSource),
    Video(VideoSource),
}

/// Anything the capture stage can pull frames from.
///
/// `wait_for_next` is separated from `read` on purpose: pacing a source is a
/// property of the source, not a cost of decoding a frame, and folding the two
/// together is what makes a fixture run look slower than it is.
trait FrameReader {
    fn wait_for_next(&mut self) {}
    fn read(&mut self) -> Result<Frame>;
}

impl FrameReader for FrameSource {
    /// Pace the source. A webcam arrives at its own rate, so there is nothing to
    /// do; a clip can be decoded far faster than it was recorded, and un-paced
    /// fixture runs would report a frame rate that bears no relation to reality.
    fn wait_for_next(&mut self) {
        match self {
            FrameSource::Camera(_) => {}
            FrameSource::Video(video) => video.wait_for_next(),
        }
    }

    fn read(&mut self) -> Result<Frame> {
        match self {
            FrameSource::Camera(camera) => camera.get_frame(),
            FrameSource::Video(video) => video.read(),
        }
    }
}

/// How long inference waits for the capture stage before re-checking the flags.
const FRAME_POLL_MS: u64 = 20;
/// How often the paused loop checks the Start/Stop flag and refreshes the UI.
const PAUSE_POLL_MS: u64 = 50;

/// A frame handed from the capture stage to inference.
struct CapturedFrame {
    frame: Frame,
    /// Wall clock the pixels came off the camera, so end-to-end latency is known.
    captured_at: Instant,
    /// Time the capture stage spent reading them.
    capture_ms: f64,
}

/// One frame's worth of work, split by stage so the dashboard can show camera
/// time and inference time separately instead of one conflated number.
struct FrameOutcome {
    detection: HandDetectionResult,
    frame: Option<Frame>,
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
fn synthetic_frame(seconds: f64, pitch_slot: usize) -> FrameOutcome {
    FrameOutcome {
        detection: synthetic_detection(seconds, pitch_slot),
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
    source: Option<FrameSource>,
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

    // --- what the hands are doing ---------------------------------------------
    /// The single source of truth about each hand slot: hysteresis, the grace
    /// window, the smoothed fingertip, the loudness and the geometry all live here.
    /// Slot 0 and slot 1 run the identical estimator; which of the two plays pitch is
    /// the role the player picks, and `UiCommand::SwapHands` trades it.
    estimators: [HandEstimator; MAX_HANDS],
    /// Which hand slot plays notes. The other one modulates. Both the tracker and the
    /// mock generator report hands in role order, so index 0 is always this slot.
    pitch_slot: usize,
    /// True while a gesture is actively driving the sinks. Used to send exactly
    /// one release event when it stops, so the audio/MIDI sinks do not latch the
    /// last note and keep sounding after the hand leaves the frame.
    gesture_active: bool,
    /// The keyboard glow. Derived from the gesture, not from the window, so the
    /// stage draws a fact about the run instead of maintaining its own history.
    trail: KeyboardTrail,
    /// The effect chain's parameters, smoothed frame by frame. One writer (this
    /// loop), one published value, so a sink never has to reconstruct the chain.
    modulation: Modulation,
    /// When the chain was last recomputed: the denominator of its fade.
    last_modulation: Instant,
    metrics: FrameMetrics,
    /// The run recorded frame by frame, when the run was asked to record one or
    /// check itself against one.
    trace: Trace,
    trace_out: Option<String>,
    trace_expect: Option<String>,
    budget: Budget,

    // --- run state -------------------------------------------------------------
    run: RunFlags,
    stats: RunStats,
    /// When a hand was last seen, for the "lost for N ms" readout.
    last_detection: Instant,
    /// The same, for the modulator hand.
    last_modulator: Instant,
    /// Deadline of the next synthetic frame, for mock mode's paced clock.
    next_mock: Instant,
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
    /// Whether the presenter is handed frames at all: with no window and no ASCII
    /// output there is nothing to draw, so the frame is not copied.
    emit_frames: bool,
}

/// Load shedding. The ASCII art and the camera preview are the only work in the
/// loop that is not on the sound path, so when inference is running out of time
/// they are the first things to go.
///
/// `limit_ms <= 0` disables the guard entirely: a run that quietly turns off
/// features the user asked for is worse than one that is a little slow.
#[derive(Debug, Clone, Copy)]
struct Budget {
    limit_ms: f64,
    /// Set once the guard has fired, so the log says it once rather than every
    /// frame after the tail blows.
    shed: bool,
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
    pub source: Option<FrameSource>,
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
    /// Write the frame-by-frame sounding trace to this file.
    pub trace_out: Option<String>,
    /// Compare the run against a recorded trace and fail on a difference.
    pub trace_expect: Option<String>,
    /// Shed the ASCII backdrop and the camera preview when inference's p95 exceeds
    /// this many milliseconds. 0 (the default) leaves the guard off.
    pub budget_ms: f64,
    pub max_frames: Option<u32>,
    pub emit_frames: bool,
}

impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let PipelineConfig {
            source,
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
            trace_out,
            trace_expect,
            budget_ms,
            max_frames,
            emit_frames,
        } = config;

        // `main` seeds the snapshot with the run's context (mode, sink, scale).
        // The pipeline keeps its own copy so publishing a frame does not wipe
        // those fields back to their defaults.
        let context = shared
            .lock()
            .map(|snapshot| FrameMetrics {
                status: snapshot.status.clone(),
                scene: SceneView {
                    layout: snapshot.scene.layout.clone(),
                    ..SceneView::default()
                },
                sound: SoundView {
                    fade_ms: snapshot.sound.fade_ms,
                    ..SoundView::default()
                },
                ..FrameMetrics::default()
            })
            .unwrap_or_default();

        // One clock reading for every deadline, so the first frame is due immediately.
        let now = Instant::now();
        // Presentation runs on its own thread: the frame loop only ever hands it a
        // frame, and never converts a pixel itself.
        let (jobs_tx, jobs_rx) = crossbeam_channel::bounded(1);
        let mut pipeline = Pipeline {
            source,
            tracker,
            strategy,
            sinks,
            event_bus,
            event_rx,
            shared,
            control,
            control_rx: Some(control_rx),
            metrics: context,
            estimators: [HandEstimator::new(), HandEstimator::new()],
            pitch_slot: 0,
            gesture_active: false,
            trail: KeyboardTrail::new(fade_ms),
            modulation: Modulation::neutral(),
            last_modulation: now,
            trace: Trace::default(),
            trace_out,
            trace_expect,
            budget: Budget {
                limit_ms: budget_ms,
                shed: false,
            },
            counters: Arc::new(CaptureCounters::default()),
            run: RunFlags {
                max_frames,
                emit_frames,
            },
            stats: RunStats::new(),
            last_detection: now,
            last_modulator: now,
            next_mock: now,
            presentation: Presentation {
                jobs: Some(jobs_tx),
                presenter: Some(Presenter::new(ascii_cols, ascii_stdout, ascii_dump.clone())),
                stdout: ascii_stdout,
                dump: ascii_dump,
            },
            jobs_rx: Some(jobs_rx),
        };
        // The loudness model the run opens with lives in the same config the
        // dashboard will change later, so the estimator is armed from it rather
        // than from a second default.
        if let Ok(config) = pipeline.control.lock() {
            let (model, near, far) = config.volume_settings();
            pipeline.estimators[0].set_volume(model, near, far);
            pipeline.estimators[1].set_volume(model, near, far);
            pipeline.metrics.status.volume_model = model.name().to_string();
            // Which stages are in the chain is a runtime knob too, so the neutral
            // levels start with the enabled mask the run was launched with.
            pipeline.modulation = Modulation::neutral_with(config.effects);
            pipeline.metrics.sound.modulation = pipeline.modulation;
        }
        // Seed the instrument so the strategy, the sinks and the published keyboard
        // agree from the very first frame.
        pipeline.apply_instrument(&instrument);
        pipeline
    }

    /// Hand the frame to the presenter. The channel holds one frame: a presenter
    /// still drawing the previous one drops this frame rather than drifting behind
    /// the camera, which is exactly right for a picture nobody is measuring.
    fn hand_over(&mut self, frame: Option<Frame>, pitch: &HandState, settings: &RuntimeConfig) {
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
            landmarks: pitch.landmarks.clone(),
            frame_size: pitch.frame_size,
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
            let pixels = std::mem::take(&mut shared.scene.pixels);
            let grid = std::mem::take(&mut shared.scene.ascii);
            *shared = self.metrics.clone();
            shared.scene.pixels = pixels;
            shared.scene.ascii = grid;
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
                UiCommand::SetVolume(model) => {
                    config.volume = model;
                    for estimator in &mut self.estimators {
                        estimator.set_volume(model, config.proximity_near, config.proximity_far);
                    }
                    self.metrics.status.volume_model = model.name().to_string();
                }
                UiCommand::SetEffect(effect, on) => config.effects.set(effect, on),
                UiCommand::SwapHands => {
                    // The roles live here; the tracker is told which slot plays pitch so
                    // its anchored crop - the sensitive window - follows the right hand.
                    self.pitch_slot = 1 - self.pitch_slot;
                    tracing::debug!("hand roles swapped; pitch is now slot {}", self.pitch_slot);
                }
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
        self.metrics.status.scale = config.scale.name.to_string();
        self.metrics.status.root = config.root_name.clone();
        self.metrics.status.waveform = config.waveform.name().to_string();
        self.metrics.scene.layout = config.layout.clone();

        tracing::info!(
            "instrument {} {} - {}x{} grid, {} waveform",
            config.scale.name,
            config.root_name,
            config.layout.rows,
            config.layout.cols,
            config.waveform,
        );
    }

    /// Runs until the UI asks for shutdown or `--max-frames` is reached. The
    /// `active` flag in the runtime config (the dashboard's Stop button) pauses the
    /// loop without tearing down the camera or the sinks.
    pub fn run(&mut self, running: &Arc<AtomicBool>) -> Result<()> {
        // Stage 1: the frame source, on its own thread. The channel holds a single
        // frame, so inference always takes the newest one instead of queueing up
        // behind its own latency and drifting away from the live view.
        let (frame_tx, frame_rx) = crossbeam_channel::bounded(1);
        let capture_handle = self.source.take().map(|mut source| {
            let tx = frame_tx.clone();
            let running = running.clone();
            let control = self.control.clone();
            let counters = self.counters.clone();
            std::thread::spawn(move || {
                if let Err(e) = capture_frames(&mut source, &tx, &control, &running, &counters) {
                    tracing::warn!("capture stage stopped: {e}");
                }
            })
        });
        drop(frame_tx); // closing the sender ends inference when capture exits

        // Stage 3: the sinks. A blocking MIDI write, or any sink that hiccups, no
        // longer holds the frame loop hostage.
        let sink_handle = {
            let sinks = self.sinks.clone();
            let dispatch = self.stats.dispatch_ring();
            let rx = self.event_rx.take();
            std::thread::spawn(move || {
                let Some(rx) = rx else {
                    return;
                };
                while let Ok(message) = rx.recv() {
                    match message {
                        BusMessage::Event(queued) => {
                            // How long the gesture waited in the queue before an output saw
                            // it: part of the hand-to-sound budget that neither capture nor
                            // inference accounts for.
                            let age_ms = queued.dispatched_at.elapsed().as_secs_f64() * 1000.0;
                            if let Ok(mut samples) = dispatch.lock() {
                                samples.push(age_ms);
                            }
                            apply_to_sinks(&sinks, &queued.performance);
                        }
                        // Chain parameters are continuous state, not attacks, so they are
                        // not timed and do not feed the latency sample.
                        BusMessage::Modulation(modulation) => apply_modulation(&sinks, &modulation),
                        BusMessage::Shutdown => break,
                    }
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
                .is_some_and(|n| self.metrics.perf.frames >= n as u64)
            {
                break;
            }
        }

        let frames = self.metrics.perf.frames.max(1);
        tracing::info!(
            "{} frames in {:.2} s: {:.1} fps, {:.2} ms/frame end to end (capture {:.2} ms, inference {:.2} ms, {} frames dropped)",
            self.metrics.perf.frames,
            self.metrics.perf.uptime_ms as f64 / 1000.0,
            self.metrics.perf.fps,
            self.metrics.perf.uptime_ms as f64 / frames as f64,
            self.metrics.perf.capture_ms,
            self.metrics.perf.frame_ms,
            self.metrics.perf.dropped,
        );
        let perf = self.stats.profile();
        tracing::info!(
            "stage tails p50/p95/p99/max ms - capture {:.1}/{:.1}/{:.1}/{:.1}  inference {:.1}/{:.1}/{:.1}/{:.1}  camera->publish {:.1}/{:.1}/{:.1}/{:.1}  dispatch->sink {:.1}/{:.1}/{:.1}/{:.1}  ({} samples)",
            perf.capture.p50, perf.capture.p95, perf.capture.p99, perf.capture.max,
            perf.inference.p50, perf.inference.p95, perf.inference.p99, perf.inference.max,
            perf.end_to_end.p50, perf.end_to_end.p95, perf.end_to_end.p99, perf.end_to_end.max,
            perf.dispatch.p50, perf.dispatch.p95, perf.dispatch.p99, perf.dispatch.max,
            perf.inference.count,
        );

        // The crop-vs-frame comparison this run measured, if it was a measuring run.
        if let Some(tracker) = &self.tracker {
            if let Some(report) = tracker.probe_report() {
                tracing::info!("detector probe: {report}");
            }
        }

        if let Some(path) = &self.trace_out {
            if let Err(error) = crate::core::trace::write(path, &self.trace) {
                tracing::warn!("could not write trace {path}: {error}");
            } else {
                tracing::info!("{} frames written to {path}", self.trace.len());
            }
        }
        if let Some(path) = &self.trace_expect {
            match crate::core::trace::read(path) {
                Ok(expected) => {
                    let differences = crate::core::trace::diff(&expected, &self.trace);
                    if differences.is_empty() {
                        tracing::info!("trace matches {path} ({} frames)", self.trace.len());
                    } else {
                        for line in differences.iter().take(20) {
                            tracing::warn!("{line}");
                        }
                        if differences.len() > 20 {
                            tracing::warn!("... and {} more", differences.len() - 20);
                        }
                        // Non-zero, so `make trace-check` is a gate and not a note.
                        anyhow::bail!("[Trace] {path}: {} frames differ", differences.len());
                    }
                }
                Err(error) => tracing::warn!("{error}"),
            }
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

    /// Load shedding. If inference's tail is blowing the frame budget, the picture
    /// work - ASCII and the preview upload - is turned off, because it is the only
    /// thing in the loop that is not on the sound path.
    ///
    /// The guard fires once and says so. A run that silently stops drawing what the
    /// user asked for is worse than one that is a little slow.
    fn shed_if_over_budget(&mut self) {
        if self.budget.limit_ms <= 0.0 || self.budget.shed {
            return;
        }
        // Only worth checking once the ring has enough frames for a tail.
        if !self
            .metrics
            .perf
            .frames
            .is_multiple_of(PROFILE_REFRESH_FRAMES)
        {
            return;
        }
        if !shed_subsystems(&self.metrics.perf.profile.inference, self.budget.limit_ms) {
            return;
        }
        self.budget.shed = true;
        if let Ok(mut config) = self.control.lock() {
            config.ascii = false;
            config.feed = false;
        }
        tracing::warn!(
            "inference p95 {:.1} ms exceeds the {:.0} ms budget - ascii and camera preview off",
            self.metrics.perf.profile.inference.p95,
            self.budget.limit_ms
        );
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
            drain_events(rx, &self.sinks);
        }
        // Otherwise the stopped interval shows up as a huge fps dip on resume.
        self.stats.reset_publish_clock();

        if self.metrics.status.paused {
            return;
        }
        self.metrics.status.paused = true;
        self.metrics.hand.detected = false;
        self.metrics.sound.note = None;
        self.metrics.sound.volume = 0.0;
        self.metrics.sound.frequency = None;
        // A stopped run is not just silent - the chain goes back to neutral so a
        // resumed run starts from an unfiltered note rather than wherever the last
        // hand left it.
        self.modulation = Modulation::neutral();
        self.metrics.sound.modulation = self.modulation;
        self.event_bus.dispatch_modulation(self.modulation);
        self.metrics.hand.landmarks.clear();
        self.metrics.hand.landmark_count = 0;
        self.metrics.hand.hand_box = None;
        self.metrics.hand.tip_x = 0.0;
        self.metrics.hand.tip_y = 0.0;
        self.metrics.modulator = crate::dashboard::metrics::ModulatorView::default();
        self.trail.clear();
        self.sync_trail();
        // Keep the published picture state truthful while stopped too, so the header
        // buttons and the guard's decisions read the same way paused as running.
        let settings = self.settings();
        self.metrics.status.feed = settings.feed;
        self.metrics.status.ascii = settings.ascii;
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
            if now < self.next_mock {
                std::thread::sleep(self.next_mock - now);
            }
            self.next_mock += MOCK_CADENCE;
            // The hand is driven by the *frame index*, not the wall clock: a mock
            // run of N frames always shows the same hand at the same place, which
            // is what makes a golden trace test possible. The deadline clock above
            // still paces the loop, so the telemetry stays honest.
            let nominal_seconds = self.metrics.perf.frames as f64 * MOCK_CADENCE.as_secs_f64();
            return Ok(FrameRead::Frame(synthetic_frame(
                nominal_seconds,
                self.pitch_slot,
            )));
        };

        match frames.recv_timeout(Duration::from_millis(FRAME_POLL_MS)) {
            Ok(captured) => {
                let inference_start = Instant::now();
                let detection = tracker.process_frame(&captured.frame, self.pitch_slot)?;
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
        let FrameOutcome {
            detection,
            frame,
            capture_ms,
            inference_ms,
            latency_ms,
        } = outcome;

        // One resolution of the raw detection, shared by every stage below. Each slot
        // runs the same estimator, so the second hand is tracked exactly like the first.
        let states = self.resolve_hands(&detection);
        let pitch = states[0].clone();
        let modulator = states[1].clone();

        self.record_timing(&detection, capture_ms, inference_ms, latency_ms);
        // Read the UI's wishes *after* the timing, so a shed that fires on this
        // frame's tail takes effect on this frame's picture rather than the next one.
        let settings = self.settings();
        // Publish what the picture subsystems *are*, not what the UI asked for: the
        // budget guard can switch them off without a click, and a mirror in the
        // dashboard would then be lying.
        self.metrics.status.feed = settings.feed;
        self.metrics.status.ascii = settings.ascii;
        // The chain is smoothed before the gesture is dispatched, so a note attacks
        // with the filter and wobble already in place rather than catching up to them.
        self.update_modulation(&pitch, &modulator, &settings);
        self.update_tracking(&pitch, &modulator);
        self.dispatch_gesture(&pitch);
        // The keyboard glow follows the gesture, on the frame loop's clock.
        self.update_trail();
        // Last, and off this thread: nothing a human looks at can delay a frame.
        self.hand_over(frame, &pitch, &settings);

        // Inline mode: this thread still owns the receiver. Once `run` has spawned
        // the sink thread it is gone, and that stage is the only consumer.
        if let Some(rx) = &self.event_rx {
            drain_events(rx, &self.sinks);
        }
        self.publish();
        // The trace is the smallest thing a run produces, so recording it costs
        // nothing worth naming and makes a regression visible without a screenshot.
        self.trace.push(&self.metrics);
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
        self.metrics.perf.fps = self.stats.record(capture_ms, inference_ms, latency_ms);

        self.metrics.perf.frames += 1;
        if self
            .metrics
            .perf
            .frames
            .is_multiple_of(PROFILE_REFRESH_FRAMES)
        {
            self.metrics.perf.profile = self.stats.profile();
            // Load shedding: the picture is the only work in the loop that is not on
            // the sound path, so it is what goes when inference runs out of time.
            self.shed_if_over_budget();
        }
        self.metrics.perf.uptime_ms = self.stats.started().elapsed().as_millis() as u64;
        self.metrics.status.paused = false;
        self.metrics.hand.frame_size = detection.frame_size;
        self.metrics.perf.frame_ms = inference_ms;
        self.metrics.perf.capture_ms = capture_ms;
        self.metrics.perf.latency_ms = latency_ms;
        self.metrics.perf.captured = self.counters.captured.load(Ordering::Relaxed);
        self.metrics.perf.dropped = self.counters.dropped.load(Ordering::Relaxed);
        self.metrics.perf.detector_us = detection.timings.detector_us;
        self.metrics.perf.landmarker_us = detection.timings.landmarker_us;
    }

    /// Resolve every hand slot this frame into a `HandState`.
    ///
    /// Slot 0 is the pitch hand and slot 1 the modulator; the tracker keeps them that
    /// way, so this stage is a plain per-slot pass through identical estimators.
    fn resolve_hands(&mut self, detection: &HandDetectionResult) -> [HandState; MAX_HANDS] {
        [
            self.estimators[0].update(detection.pitch(), detection.frame_size),
            self.estimators[1].update(detection.modulator(), detection.frame_size),
        ]
    }

    /// Distance between the two palm centres, in pixels, together with the centre-to-centre
    /// distance at which the two hand boxes would touch. `None` when either hand has no
    /// box, which is what makes the inter-hand sources fall back to their single-hand
    /// values rather than read a meaningless zero.
    ///
    /// The touching distance comes from the hands' own sizes, so dividing one by the other
    /// makes the gap reading independent of resolution and of how far the player sits.
    fn palm_distance(&self, pitch: &HandState, modulator: &HandState) -> Option<(f32, f32)> {
        let a = pitch.hand_box?;
        let b = modulator.hand_box?;
        let (ax, ay) = box_centre(a);
        let (bx, by) = box_centre(b);
        let distance = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
        Some((distance, (a.2 + b.2) / 2.0))
    }

    /// Smooth the effect chain towards what the hands are asking for, and put it on
    /// the bus only when it actually moved.
    ///
    /// The smoothing is the point. A hand held still produces numbers that wobble by
    /// a percent per frame; without a time constant the filter would shimmer. When no
    /// hand is present the target is neutral, so the chain fades out over the same
    /// `--fade-ms` the keyboard trail uses instead of clicking off.
    fn update_modulation(
        &mut self,
        pitch: &HandState,
        modulator: &HandState,
        settings: &RuntimeConfig,
    ) {
        // The hand that drives the chain is the modulator when there is one, and the
        // pitch hand when there is not - so a one-hand run behaves exactly as it did
        // before the second hand existed.
        let driver = if modulator.present { modulator } else { pitch };
        // What feeds vibrato depth and echo feedback. With two hands it is the distance
        // between them; with one it is that hand's openness.
        let gap = match self.palm_distance(pitch, modulator) {
            Some((distance, touching)) => gap_level(distance, touching),
            None => openness(driver.geometry.spread),
        };

        let target = if driver.present {
            // `band_position` is the same inset the keyboard is drawn with, so the
            // hand's height and width mean the same thing here as they do on stage.
            let (x, y) = band_position(driver.tip.0, driver.tip.1, driver.frame_size);
            // The filter's knee is a multiple of the pitch that is actually sounding,
            // so the same hand position darkens C2 and C6 equally. Before a note has
            // ever sounded, `cutoff_for` uses the middle of the keyboard.
            let sounding = self
                .metrics
                .sound
                .frequency
                .or_else(|| self.metrics.sound.note.map(note_frequency))
                .unwrap_or(0.0);
            modulation_from_hands(
                settings.effects,
                driver.proximity,
                1.0 - y, // higher in the frame = faster wobble
                gap,
                x,
                sounding,
            )
        } else {
            Modulation::neutral_with(settings.effects)
        };

        let fade_seconds = self.trail.fade_ms().max(1.0) / 1000.0;
        let alpha = self.last_modulation.elapsed().as_secs_f32() / fade_seconds;
        let next = blend(self.modulation, target, alpha);

        if is_different(&self.modulation, &next) {
            self.event_bus.dispatch_modulation(next);
        }
        self.modulation = next;
        self.metrics.sound.modulation = next;
        self.last_modulation = Instant::now();
    }

    /// Where the hands are, copied out of the estimators into the snapshot the
    /// dashboard reads.
    fn update_tracking(&mut self, pitch: &HandState, modulator: &HandState) {
        self.metrics.hand.detected = pitch.present;
        self.metrics.hand.confidence = pitch.confidence;
        self.metrics.hand.hand_box = pitch.hand_box;
        // `clone_from` keeps the Vec's capacity, so the landmark list stops
        // reallocating every frame.
        self.metrics.hand.landmarks.clone_from(&pitch.landmarks);
        self.metrics.hand.landmark_count = pitch.landmarks.len();
        // The note anchor: the smoothed index fingertip the pitch is read from.
        self.metrics.hand.tip_x = pitch.tip.0;
        self.metrics.hand.tip_y = pitch.tip.1;
        // Which cell the sounding note belongs to, for the crosshair and the
        // cell highlight. Notes are unique in the grid, so this is exact.
        if let Some(note) = self.metrics.sound.note {
            if let Some(cell) = cell_index(&self.metrics.scene.layout, note) {
                self.metrics.hand.grid_row = cell / self.metrics.scene.layout.cols;
                self.metrics.hand.grid_col = cell % self.metrics.scene.layout.cols;
            }
        }

        if pitch.present {
            self.last_detection = Instant::now();
        }

        // The modulator gets its own view so the stage can draw it and the rail can
        // show the inter-hand numbers without recomputing them in the UI.
        if modulator.present {
            self.last_modulator = Instant::now();
        }
        let gap = self.palm_distance(pitch, modulator);
        let view = &mut self.metrics.modulator;
        view.detected = modulator.present;
        view.confidence = modulator.confidence;
        view.landmarks.clone_from(&modulator.landmarks);
        view.hand_box = modulator.hand_box;
        view.proximity = modulator.proximity;
        view.centre = modulator.hand_box.map(box_centre).unwrap_or((0.0, 0.0));
        match gap {
            Some((distance, touching)) => {
                view.gap_px = distance;
                view.gap_ratio = distance / touching.max(1.0);
                view.gap_level = gap_level(distance, touching);
            }
            None => {
                view.gap_px = 0.0;
                view.gap_ratio = 0.0;
                view.gap_level = 0.0;
            }
        }
        view.lost_ms = self.last_modulator.elapsed().as_millis() as u64;
        self.metrics.hand.lost_ms = self.last_detection.elapsed().as_millis() as u64;
        self.metrics.hand.jitter_px = pitch.jitter_px;

        self.metrics.hand.geometry.spread = pitch.geometry.spread;
        self.metrics.hand.geometry.extension = pitch.geometry.extension;
        self.metrics.hand.geometry.tilt_deg = pitch.geometry.tilt_deg;
        self.metrics.hand.geometry.palm_length = pitch.geometry.palm_length;
        // How close the hand looks, independent of which model is currently driving
        // the volume: the HUD needs it even when loudness comes from the pinch.
        self.metrics.sound.proximity = pitch.proximity;
        self.metrics.sound.proximity_ratio = pitch.proximity_ratio;
        self.metrics.sound.approach = pitch.approach;
    }

    /// Fade the keyboard trail and light the key that is sounding. The trail is
    /// part of the snapshot rather than the window, so pausing the run also puts
    /// the keyboard dark and a resumed run starts from a clean grid.
    fn update_trail(&mut self) {
        self.trail.advance(Instant::now(), self.metrics.sound.note);
        self.sync_trail();
    }

    /// Mirror the trail into the snapshot, copying into the existing buffer so the
    /// frame loop does not allocate 49 floats a frame. The length is always one level
    /// per playable note: the stage draws a cell for every note of the layout, so an
    /// empty or short buffer makes a draw index past the end.
    fn sync_trail(&mut self) {
        let levels = self.trail.levels();
        if self.metrics.sound.trail.len() != levels.len() {
            self.metrics.sound.trail.resize(levels.len(), 0.0);
        }
        self.metrics.sound.trail.copy_from_slice(levels);
        self.metrics.sound.fade_ms = self.trail.fade_ms();
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
                self.metrics.sound.note = Some(*note);
                self.metrics.sound.volume = *velocity;
                self.metrics.sound.frequency = None;
            }
            Performance::Signal {
                frequency,
                amplitude,
            } => {
                self.metrics.sound.frequency = Some(*frequency);
                self.metrics.sound.volume = *amplitude;
                self.metrics.sound.note = None;
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
        self.metrics.sound.note = None;
        self.metrics.sound.volume = 0.0;
        self.metrics.sound.frequency = None;
        if self.gesture_active {
            self.gesture_active = false;
            self.push_log(&Performance::Release);
            self.event_bus.dispatch(Performance::Release);
        }
    }

    /// The line `--debug` prints every ten frames: what the player is hearing and
    /// how long the frame took.
    fn report_frame(&self) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        // Thirty lines a second is noise; every tenth frame is a heartbeat.
        if self.metrics.perf.frames.is_multiple_of(10) {
            // What the player actually hears this frame: a keyboard note, a
            // theremin frequency, or nothing.
            let sounding = match (self.metrics.sound.note, self.metrics.sound.frequency) {
                (Some(note), _) => format!("note {} ({})", note, note_name(note)),
                (None, Some(freq)) => format!("{freq:.0} Hz"),
                _ => "silent".to_string(),
            };
            tracing::debug!(
                "fps {:.1} inference {:.2} ms (capture {:.2} ms, latency {:.2} ms, dropped {}) hands {} detector {:.0} us, landmarker {:.0} us conf {:.2} {sounding} vol {:.2} proximity {:.2} ({:.3} palm/frame) approach {:.2} spread {:.2} jitter {:.1} px tip {:.0},{:.0} gap {:.0} px ({:.2}) chain {} tone {:.0} Hz vibrato {:.0}c@{:.1} delay {:.0} ms fb {:.2}",
                self.metrics.perf.fps,
                self.metrics.perf.frame_ms,
                self.metrics.perf.capture_ms,
                self.metrics.perf.latency_ms,
                self.metrics.perf.dropped,
                self.metrics.modulator.detected as usize + self.metrics.hand.detected as usize,
                self.metrics.perf.detector_us,
                self.metrics.perf.landmarker_us,
                self.metrics.hand.confidence,
                self.metrics.sound.volume,
                self.metrics.sound.proximity,
                self.metrics.sound.proximity_ratio,
                self.metrics.sound.approach,
                self.metrics.hand.geometry.spread,
                self.metrics.hand.jitter_px,
                self.metrics.hand.tip_x,
                self.metrics.hand.tip_y,
                self.metrics.modulator.gap_px,
                self.metrics.modulator.gap_level,
                self.metrics.sound.modulation.enabled.names(),
                self.metrics.sound.modulation.cutoff_hz,
                self.metrics.sound.modulation.vibrato_cents,
                self.metrics.sound.modulation.vibrato_hz,
                self.metrics.sound.modulation.delay_ms,
                self.metrics.sound.modulation.feedback,
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

        if self.metrics.status.log.len() >= LOG_CAPACITY {
            self.metrics.status.log.pop_front();
        }
        self.metrics.status.log.push_back(EventLogEntry {
            uptime_ms: self.metrics.perf.uptime_ms,
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
    source: &mut dyn FrameReader,
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
        // Deliberately outside the timed region: waiting for the next frame is a
        // property of the source, not a cost of reading it.
        source.wait_for_next();
        let start = Instant::now();
        let frame = source.read()?;
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

fn apply_to_sinks(sinks: &SinkList, performance: &Performance) {
    let Ok(mut sinks) = sinks.lock() else {
        return;
    };
    for sink in sinks.iter_mut() {
        if let Err(e) = sink.handle(performance) {
            tracing::warn!("sink rejected a performance: {e}");
        }
    }
}

fn drain_events(rx: &Receiver<BusMessage>, sinks: &SinkList) {
    while let Ok(message) = rx.try_recv() {
        match message {
            BusMessage::Event(queued) => apply_to_sinks(sinks, &queued.performance),
            BusMessage::Modulation(modulation) => apply_modulation(sinks, &modulation),
            BusMessage::Shutdown => break,
        }
    }
}

fn apply_modulation(sinks: &SinkList, modulation: &Modulation) {
    let Ok(mut sinks) = sinks.lock() else {
        return;
    };
    for sink in sinks.iter_mut() {
        sink.set_modulation(modulation);
    }
}

/// A synthetic, moving hand so the dashboard can be exercised without a webcam or
/// the ONNX models (`--mode mock`). The generator itself lives in `vision::mock`;
/// it is re-exported here so the pipeline tests read it the same way the loop does.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::effects::{Effect, EffectMask, NEUTRAL_CUTOFF_HZ};
    use crate::core::grid::GridLayout;
    use crate::core::instrument::Waveform;
    use crate::core::runtime::control_channel;
    use crate::core::telemetry::{Latency, StageProfile, FRAME_BUDGET_MS};
    use crate::core::NOTE_COUNT;
    use crate::dashboard::metrics::{note_index, StatusView, DEFAULT_FADE_MS};
    use crate::gesture::keyboard::FingerPositionStrategy;
    use crate::presentation::ascii::{DEFAULT_ASCII_FPS, DEFAULT_COLS};

    /// A sink that records every event it is handed, so the tests can inspect
    /// what the pipeline actually dispatched.
    #[derive(Clone)]
    struct RecordingSink {
        events: Arc<Mutex<Vec<Performance>>>,
        /// Timbres the pipeline pushed into the sinks, in order. Without this the
        /// picker wiring is only audible, not testable.
        waveforms: Arc<Mutex<Vec<String>>>,
        /// Chain parameters the pipeline pushed, in order. Same reason: the effect
        /// wiring has to be observable, not just audible.
        modulations: Arc<Mutex<Vec<Modulation>>>,
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
        fn set_modulation(&mut self, modulation: &Modulation) {
            self.modulations.lock().unwrap().push(*modulation);
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
        modulations: Arc<Mutex<Vec<Modulation>>>,
    }

    fn test_pipeline(strategy: Box<dyn Strategy>, max_frames: Option<u32>) -> TestHarness {
        test_pipeline_with(
            strategy,
            max_frames,
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks),
            false,
            DEFAULT_COLS,
            0.0,
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
        budget_ms: f64,
        shared: Arc<Mutex<FrameMetrics>>,
    ) -> TestHarness {
        let (bus, rx) = EventBus::new();
        let (control_tx, control_rx) = control_channel();
        let events = Arc::new(Mutex::new(Vec::new()));
        let waveforms = Arc::new(Mutex::new(Vec::new()));
        let modulations = Arc::new(Mutex::new(Vec::new()));
        let pipeline = Pipeline::new(PipelineConfig {
            source: None,
            tracker: None,
            strategy,
            sinks: Arc::new(Mutex::new(vec![Box::new(RecordingSink {
                events: events.clone(),
                waveforms: waveforms.clone(),
                modulations: modulations.clone(),
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
            trace_out: None,
            trace_expect: None,
            budget_ms,
            max_frames,
            emit_frames,
        });
        TestHarness {
            pipeline,
            events,
            control: control_tx,
            waveforms,
            modulations,
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
            scene: SceneView {
                pixels: vec![7u8; FRAME],
                ..SceneView::default()
            },
            ..FrameMetrics::default()
        };
        let empty = FrameMetrics::default();

        let start = Instant::now();
        let mut carried = 0usize;
        for _ in 0..200 {
            let snapshot = full.clone();
            carried += snapshot.scene.pixels.len();
        }
        let with_pixels = start.elapsed().as_secs_f64() * 1000.0 / 200.0;

        let start = Instant::now();
        let mut carried_empty = 0usize;
        for _ in 0..200 {
            let snapshot = empty.clone();
            carried_empty += snapshot.scene.pixels.len();
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
    #[ignore]
    /// Print the first 60 notes the mock hand plays, so the golden trace in
    /// `the_mock_hand_walks_up_the_keyboard_in_a_stable_order` can be regenerated
    /// after a change to the grid or the synthetic hand. Run with
    /// `cargo test -- --include-ignored print_mock_trace --nocapture`.
    fn print_mock_trace() {
        let harness = test_pipeline(Box::new(FingerPositionStrategy::default()), Some(60));
        let mut pipeline = harness.pipeline;
        pipeline.run(&Arc::new(AtomicBool::new(true))).unwrap();
        let events = harness.events.lock().unwrap();
        println!("[trace] {:?}", runs(&events));
    }

    /// Collapse a trace into the notes held and how long each one lasted. Sixty
    /// frames of the same note is one fact, not sixty.
    fn runs(events: &[Performance]) -> Vec<(u8, usize)> {
        let mut out: Vec<(u8, usize)> = Vec::new();
        for event in events {
            let Performance::Note { note, .. } = *event else {
                continue;
            };
            match out.last_mut() {
                Some((last, length)) if *last == note => *length += 1,
                _ => out.push((note, 1)),
            }
        }
        out
    }

    #[test]
    fn the_mock_hand_walks_up_the_keyboard_in_a_stable_order() {
        // The mock hand is driven by the frame index, so this trace is exactly the
        // same run to run: it is a golden reference, not a flaky sample. A change
        // to the grid, the fingertip smoothing or the synthetic hand that moves a
        // note or a transition shows up here as a diff.
        let harness = test_pipeline(Box::new(FingerPositionStrategy::default()), Some(60));
        let mut pipeline = harness.pipeline;
        pipeline.run(&Arc::new(AtomicBool::new(true))).unwrap();
        let events = harness.events.lock().unwrap();

        // E4 (64) -> G4 (67) -> A4 (69): the sweep crosses two cell boundaries on
        // its way right, and each is held long enough to be a real note rather than
        // a flicker.
        assert_eq!(runs(&events), vec![(64, 13), (67, 18), (69, 28)]);
    }

    #[test]
    fn a_continuous_signal_replaces_the_note_in_the_snapshot_and_the_log() {
        let TestHarness {
            mut pipeline,
            events,
            ..
        } = test_pipeline(Box::new(AlwaysSignal), None);
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap(); // first sighting: the gate is still confirming
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap(); // second sighting: the gesture is live

        // The OUTPUT card reads `frequency` when there is no note, so a theremin
        // run must publish the hertz and not leave a stale note behind.
        assert_eq!(pipeline.metrics.sound.frequency, Some(220.0));
        assert_eq!(pipeline.metrics.sound.note, None);
        assert!((pipeline.metrics.sound.volume - 0.5).abs() < 1e-6);

        let logged = pipeline.metrics.status.log.back().unwrap().detail.clone();
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
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        assert_eq!(
            pipeline.metrics.scene.layout.len(),
            15,
            "the default grid is 3 x 5"
        );
        assert_eq!(pipeline.metrics.status.scale, "major pentatonic");

        let instrument = InstrumentConfig::new("chromatic", "C", 4, Waveform::Saw).unwrap();
        control.send(UiCommand::SetInstrument(instrument)).unwrap();

        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        assert_eq!(pipeline.metrics.scene.layout.rows, 4);
        assert_eq!(pipeline.metrics.scene.layout.cols, 12);
        assert_eq!(pipeline.metrics.status.scale, "chromatic");

        // The note the strategy just produced must belong to the new grid, not
        // the old one: the layout was pushed into the strategy, not merely into
        // the dashboard's copy of it.
        let recorded = events.lock().unwrap();
        let note = match recorded.last().unwrap() {
            Performance::Note { note, .. } => note,
            _ => panic!("expected a note performance"),
        };
        assert!(
            pipeline.metrics.scene.layout.notes.contains(note),
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
        let before = pipeline.metrics.scene.layout.clone();
        pipeline.apply_commands();
        pipeline.apply_commands();
        assert_eq!(
            pipeline.metrics.scene.layout, before,
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
            0.0,
            Arc::new(Mutex::new(FrameMetrics::default())),
        );
        let jobs = pipeline.jobs_rx.take().unwrap();

        // Nothing is on, so the frame is dropped rather than copied to a thread that
        // has nowhere to draw it.
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        assert!(jobs.try_recv().is_err(), "no audience, no job");

        // The dashboard turns the feed on; now the frame goes over.
        control.send(UiCommand::SetFeed(true)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
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
            scene: SceneView {
                pixels: vec![7u8; 8],
                ascii: crate::presentation::ascii::AsciiGrid {
                    cols: 4,
                    rows: 2,
                    cells: vec![3u8; 8],
                },
                ..SceneView::default()
            },
            ..FrameMetrics::default()
        }));
        let TestHarness { mut pipeline, .. } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks),
            false,
            DEFAULT_COLS,
            0.0,
            shared.clone(),
        );
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();

        let snapshot = shared.lock().unwrap();
        assert_eq!(snapshot.perf.frames, 1, "the telemetry moved");
        assert_eq!(snapshot.scene.pixels, vec![7u8; 8], "the picture survived");
        assert_eq!(snapshot.scene.ascii.cols, 4, "and so did the art");
    }

    #[test]
    fn publishing_preserves_the_run_context() {
        let shared = Arc::new(Mutex::new(FrameMetrics {
            status: StatusView {
                mode: "Signal".to_string(),
                sink: "audio".to_string(),
                scale: "major pentatonic".to_string(),
                root: "C".to_string(),
                ..StatusView::default()
            },
            scene: SceneView {
                layout: GridLayout::new(
                    0,
                    &crate::core::scale::preset("major pentatonic").unwrap(),
                    3,
                ),
                ..SceneView::default()
            },
            ..FrameMetrics::default()
        }));
        let TestHarness { mut pipeline, .. } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks),
            false,
            DEFAULT_COLS,
            0.0,
            shared.clone(),
        );
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();

        let snapshot = shared.lock().unwrap();
        assert_eq!(snapshot.status.scale, "major pentatonic");
        assert_eq!(snapshot.status.root, "C");
        assert_eq!(snapshot.scene.layout.len(), 15);
        assert_eq!(snapshot.status.mode, "Signal");
        assert!(snapshot.hand.detected, "the mock hand should register");
    }

    #[test]
    fn the_keyboard_trail_is_part_of_the_snapshot_not_the_window() {
        let harness = test_pipeline(Box::new(AlwaysOn), Some(5));
        let mut pipeline = harness.pipeline;
        pipeline.run(&Arc::new(AtomicBool::new(true))).unwrap();

        // The held key is lit, and the glow is published so the stage needs no
        // state of its own.
        assert_eq!(pipeline.metrics.sound.trail[note_index(60)], 1.0);
        assert_eq!(pipeline.metrics.sound.fade_ms, DEFAULT_FADE_MS);

        pipeline.pause();
        assert_eq!(
            pipeline.metrics.sound.trail.len(),
            NOTE_COUNT,
            "a stopped player publishes a level per note, not an empty buffer"
        );
        assert!(
            pipeline
                .metrics
                .sound
                .trail
                .iter()
                .all(|&level| level == 0.0),
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
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();

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
        assert!(pipeline.metrics.status.paused);
        assert!(pipeline.metrics.sound.note.is_none());
        assert_eq!(pipeline.metrics.sound.volume, 0.0);
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

        pipeline.advance(synthetic_frame(0.0, 0)).unwrap(); // hand present -> note
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap(); // hand gone   -> release
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap(); // still gone  -> nothing more

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
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        assert!(events.lock().unwrap().is_empty());
        assert_eq!(pipeline.metrics.sound.volume, 0.0);
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
            let mut detection = synthetic_detection(0.0, 0);
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
        assert!(!pipeline.metrics.hand.detected);

        // Two sightings in a row start the gesture; the grace window keeps it alive
        // while the anchor score dips, and only the fourth miss releases it.
        pipeline.advance(found()).unwrap();
        pipeline.advance(found()).unwrap();
        assert!(
            pipeline.metrics.hand.detected,
            "the second sighting confirms it"
        );
        pipeline.advance(absent()).unwrap();
        pipeline.advance(absent()).unwrap();
        pipeline.advance(absent()).unwrap();
        assert!(
            pipeline.metrics.hand.detected,
            "three misses are still inside the grace window"
        );
        pipeline.advance(absent()).unwrap();
        assert!(
            !pipeline.metrics.hand.detected,
            "the fourth miss loses the hand"
        );

        let sent = events.lock().unwrap();
        assert!(
            sent.iter()
                .any(|performance| matches!(performance, Performance::Release)),
            "the release is emitted once the grace window expires"
        );
    }

    /// A source that produces a fixed number of frames, then switches the run off
    /// the way the Stop button does.
    struct ScriptedSource {
        produced: usize,
        stop_at: usize,
        running: Arc<AtomicBool>,
    }

    impl FrameReader for ScriptedSource {
        fn read(&mut self) -> Result<Frame> {
            self.produced += 1;
            if self.produced >= self.stop_at {
                self.running.store(false, Ordering::Relaxed);
            }
            Ok(Frame::solid(4, 4, 7))
        }
    }

    struct CountingSource {
        produced: usize,
    }

    impl FrameReader for CountingSource {
        fn read(&mut self) -> Result<Frame> {
            self.produced += 1;
            Ok(Frame::solid(2, 2, 0))
        }
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

        let mut source = ScriptedSource {
            produced: 0,
            stop_at: 5,
            running: running.clone(),
        };

        capture_frames(&mut source, &tx, &control, &running, &counters).unwrap();

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

        let mut source = CountingSource { produced: 0 };

        capture_frames(&mut source, &tx, &paused, &running, &counters).unwrap();
        assert_eq!(
            source.produced, 0,
            "Stop means the camera is not read at all"
        );
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
            pipeline.metrics.status.waveform, "square",
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
            pipeline.metrics.perf.frames >= 15,
            "the loop ran to the frame limit"
        );
        assert!(
            pipeline.metrics.perf.fps > 20.0 && pipeline.metrics.perf.fps < 45.0,
            "fps tracks the frames per second, got {}",
            pipeline.metrics.perf.fps
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
            pipeline.metrics.perf.frames, 3,
            "--max-frames stops the frame loop"
        );
        assert!(
            pipeline.metrics.perf.detector_us == 0.0,
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

    /// A mock frame that pretends inference took `inference_ms`, so the budget guard
    /// can be exercised without a model that is actually slow.
    fn slow_frame(seconds: f64, inference_ms: f64) -> FrameOutcome {
        FrameOutcome {
            detection: synthetic_detection(seconds, 0),
            frame: None,
            capture_ms: 0.0,
            inference_ms,
            latency_ms: 0.0,
        }
    }

    #[test]
    fn an_over_budget_run_sheds_the_picture_and_says_so() {
        let config = RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, true, Ramp::Blocks);
        let shared = Arc::new(Mutex::new(FrameMetrics::default()));
        let TestHarness {
            mut pipeline,
            control,
            ..
        } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            config,
            true,
            DEFAULT_COLS,
            FRAME_BUDGET_MS,
            shared.clone(),
        );
        let jobs = pipeline.jobs_rx.take().unwrap();

        // The guard needs a tail, and a tail needs a second of frames. Drive 30
        // frames whose inference takes 41 ms each - a pose model, roughly - and the
        // profile that refreshes on the 30th frame trips the guard.
        for frame in 0..PROFILE_REFRESH_FRAMES {
            pipeline
                .advance(slow_frame(frame as f64 * 0.033, 41.0))
                .unwrap();
        }

        let applied = pipeline.control.lock().unwrap().clone();
        assert!(!applied.ascii, "the backdrop is the first thing to go");
        assert!(!applied.feed, "and so is the preview upload");

        // Drain whatever the 29 pre-shed frames queued, then check that the shed
        // stops handing frames over rather than merely flipping a flag.
        while jobs.try_recv().is_ok() {}
        pipeline.advance(slow_frame(1.0, 41.0)).unwrap();
        assert!(
            jobs.try_recv().is_err(),
            "shed work is not handed to the presenter"
        );
        // The user can still turn it back on: the guard fires once, not forever.
        control.send(UiCommand::SetAscii(true)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        assert!(
            pipeline.control.lock().unwrap().ascii,
            "a manual toggle beats the guard"
        );
    }

    #[test]
    fn the_snapshot_carries_the_picture_state_the_run_actually_has() {
        // The dashboard's buttons used to keep their own mirror of feed/ascii. The
        // budget guard can clear those without a click, so the mirror would drift
        // from reality; the snapshot is the single source of truth.
        let shared = Arc::new(Mutex::new(FrameMetrics::default()));
        let TestHarness {
            mut pipeline,
            control,
            ..
        } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, true, Ramp::Blocks),
            false,
            DEFAULT_COLS,
            0.0,
            shared.clone(),
        );
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        assert!(shared.lock().unwrap().status.ascii, "asked for, so on");

        control.send(UiCommand::SetAscii(false)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        assert!(
            !shared.lock().unwrap().status.ascii,
            "the published state follows the pipeline, not the initial request"
        );
    }

    /// A frame with no hand in it at all.
    fn empty_frame() -> FrameOutcome {
        FrameOutcome {
            detection: HandDetectionResult::default(),
            frame: None,
            capture_ms: 0.0,
            inference_ms: 0.0,
            latency_ms: 0.0,
        }
    }

    #[test]
    /// The chain is driven by the hand, reaches the sinks, and decays when the hand
    /// leaves rather than switching off. The sleeps are the frame interval: the fade
    /// is a wall-clock time constant, so a test that advances frames as fast as the
    /// CPU allows would never see it.
    fn the_chain_follows_the_hand_and_fades_when_it_leaves() {
        let mut config =
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks);
        config.effects = EffectMask::ALL;
        let TestHarness {
            mut pipeline,
            modulations,
            ..
        } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            config,
            false,
            DEFAULT_COLS,
            0.0,
            Arc::new(Mutex::new(FrameMetrics::default())),
        );

        for frame in 0..20 {
            pipeline
                .advance(synthetic_frame(frame as f64 * 0.033, 0))
                .unwrap();
            std::thread::sleep(Duration::from_millis(25));
        }

        let driven = pipeline.modulation;
        assert!(
            driven.enabled.is_on(Effect::Lowpass),
            "the run asked for the chain"
        );
        assert!(
            driven.cutoff_hz < NEUTRAL_CUTOFF_HZ,
            "a hand in view should have closed the filter from neutral, got {}",
            driven.cutoff_hz
        );
        assert!(
            driven.delay_ms > 0.0,
            "the hand swept the delay away from zero"
        );
        assert!(
            !modulations.lock().unwrap().is_empty(),
            "the chain reached the sinks"
        );

        let before = pipeline.modulation;
        for _ in 0..20 {
            pipeline.advance(empty_frame()).unwrap();
            std::thread::sleep(Duration::from_millis(25));
        }
        let after = pipeline.modulation;
        assert!(
            (after.cutoff_hz - NEUTRAL_CUTOFF_HZ).abs()
                < (before.cutoff_hz - NEUTRAL_CUTOFF_HZ).abs() * 0.5,
            "half a second without a hand should have faded most of the way to neutral: {} -> {}",
            before.cutoff_hz,
            after.cutoff_hz
        );
        assert!(
            after.enabled == EffectMask::ALL,
            "the hand left, but the chain the run asked for is still in"
        );
    }

    #[test]
    /// A chain that was never asked for is never sent: with every stage off there is
    /// nothing continuous to deliver, and the sinks stay idle.
    fn a_run_with_no_effects_sends_no_modulation() {
        let TestHarness {
            pipeline,
            modulations,
            ..
        } = test_pipeline(Box::new(AlwaysOn), Some(5));
        assert_eq!(pipeline.modulation.enabled, EffectMask::NONE);
        assert!(
            modulations.lock().unwrap().is_empty(),
            "no chain, no traffic"
        );
    }

    #[test]
    /// Switching a stage off from the dashboard takes effect on the next frame, and
    /// the parameter behind it is left alone rather than zeroed.
    fn a_toggle_flips_the_chain_without_disturbing_the_levels() {
        let mut config =
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks);
        config.effects = EffectMask::ALL;
        let TestHarness {
            mut pipeline,
            control,
            modulations,
            ..
        } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            config,
            false,
            DEFAULT_COLS,
            0.0,
            Arc::new(Mutex::new(FrameMetrics::default())),
        );
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();

        control
            .send(UiCommand::SetEffect(Effect::Delay, false))
            .unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        let after = pipeline.modulation;

        assert!(
            !after.enabled.is_on(Effect::Delay),
            "the stage was switched off"
        );
        assert!(
            after.enabled.is_on(Effect::Lowpass),
            "the others are untouched"
        );
        assert!(
            after.delay_ms > 0.0,
            "a disabled stage keeps tracking the hand, so switching it back on does not start from zero"
        );
        assert!(
            !modulations.lock().unwrap().is_empty(),
            "the change was dispatched"
        );
    }

    #[test]
    /// The mock run has two hands, so the second slot is filled and the inter-hand
    /// gap is published. Without a second hand the modulator view reads as lost.
    fn both_hand_slots_are_published() {
        let TestHarness {
            mut pipeline,
            modulations,
            ..
        } = test_pipeline(Box::new(AlwaysOn), Some(3));
        let _ = modulations;
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();

        assert!(
            pipeline.metrics.modulator.detected,
            "the mock run drives two hands"
        );
        assert!(
            pipeline.metrics.modulator.gap_px > 0.0,
            "the hands are apart"
        );
        assert!(pipeline.metrics.modulator.lost_ms == 0);

        pipeline.advance(empty_frame()).unwrap();
        pipeline.advance(empty_frame()).unwrap();
        pipeline.advance(empty_frame()).unwrap();
        pipeline.advance(empty_frame()).unwrap();
        assert!(
            !pipeline.metrics.modulator.detected,
            "with nothing in view the modulator is lost"
        );
        assert!(pipeline.metrics.modulator.gap_px == 0.0, "no hands, no gap");
    }

    #[test]
    /// Swapping hands changes which slot plays pitch, and the tracker's identity
    /// survives the swap: the same physical hand keeps its slot across frames.
    fn swapping_hands_moves_the_pitch_to_the_other_slot() {
        let TestHarness {
            mut pipeline,
            control,
            ..
        } = test_pipeline(Box::new(AlwaysOn), None);
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        let before_tip = pipeline.metrics.hand.tip_x;
        let before_mod = pipeline.metrics.modulator.centre.0;

        control.send(UiCommand::SwapHands).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();

        assert!(
            (pipeline.metrics.hand.tip_x - before_mod).abs() > 1.0,
            "the pitch hand is now the one that was modulating: {before_tip} -> {}",
            pipeline.metrics.hand.tip_x
        );
        assert!(
            pipeline.metrics.modulator.detected,
            "both hands are still tracked"
        );
    }

    #[test]
    /// A single hand is a complete instrument: with the modulator slot empty the pitch
    /// hand drives the whole chain, and the inter-hand sources fall back to that hand's
    /// openness rather than reading a gap of zero.
    fn one_hand_drives_the_chain_by_itself() {
        let mut config =
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks);
        config.effects = EffectMask::ALL;
        let TestHarness { mut pipeline, .. } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            config,
            false,
            DEFAULT_COLS,
            0.0,
            Arc::new(Mutex::new(FrameMetrics::default())),
        );

        let only_pitch = {
            let mut detection = synthetic_detection(0.0, 0);
            detection.hands.truncate(1);
            detection
        };
        for _ in 0..8 {
            pipeline
                .advance(FrameOutcome {
                    detection: only_pitch.clone(),
                    frame: None,
                    capture_ms: 0.0,
                    inference_ms: 0.0,
                    latency_ms: 0.0,
                })
                .unwrap();
        }

        assert!(pipeline.metrics.hand.detected, "the pitch hand is live");
        assert!(
            !pipeline.metrics.modulator.detected,
            "the second slot stayed empty"
        );
        assert!(
            pipeline.modulation.vibrato_cents > 0.0,
            "with no second hand the wobble comes from the pitch hand's own openness"
        );
    }

    #[test]
    /// Stopping the run silences the chain as well as the note, so a resumed run does
    /// not start from whatever the last hand left behind.
    fn pausing_neutralises_the_chain() {
        let mut config =
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks);
        config.effects = EffectMask::ALL;
        let TestHarness { mut pipeline, .. } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            config,
            false,
            DEFAULT_COLS,
            0.0,
            Arc::new(Mutex::new(FrameMetrics::default())),
        );
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        pipeline.pause();

        assert_eq!(pipeline.modulation.enabled, EffectMask::NONE);
        assert_eq!(pipeline.modulation.cutoff_hz, NEUTRAL_CUTOFF_HZ);
        assert_eq!(
            pipeline.metrics.sound.modulation.enabled,
            EffectMask::NONE,
            "the rail shows a silent chain"
        );
    }

    #[test]
    /// The dashboard shows the chain the frame loop computed, not the buttons someone
    /// clicked, so the rail cannot drift from the sound.
    fn the_published_chain_is_the_one_the_loop_computed() {
        let mut config =
            RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, false, Ramp::Blocks);
        config.effects = EffectMask::parse("lowpass,vibrato").expect("valid list");
        let shared = Arc::new(Mutex::new(FrameMetrics::default()));
        let TestHarness { mut pipeline, .. } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            config,
            false,
            DEFAULT_COLS,
            0.0,
            shared.clone(),
        );
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();
        pipeline.advance(synthetic_frame(0.0, 0)).unwrap();

        let published = shared.lock().unwrap().sound.modulation;
        assert_eq!(
            published, pipeline.modulation,
            "the snapshot mirrors the loop"
        );
        assert!(published.enabled.is_on(Effect::Vibrato));
        assert!(
            !published.enabled.is_on(Effect::Delay),
            "delay was never asked for"
        );
    }

    #[test]
    fn a_run_with_no_budget_keeps_its_picture() {
        let config = RuntimeConfig::new(DEFAULT_FEED_FPS, DEFAULT_ASCII_FPS, true, Ramp::Blocks);
        let TestHarness { mut pipeline, .. } = test_pipeline_with(
            Box::new(AlwaysOn),
            None,
            config,
            false,
            DEFAULT_COLS,
            0.0,
            Arc::new(Mutex::new(FrameMetrics::default())),
        );
        pipeline.metrics.perf.frames = PROFILE_REFRESH_FRAMES;
        pipeline.metrics.perf.profile = StageProfile {
            inference: Latency {
                p50: 8.0,
                p95: 41.0,
                p99: 60.0,
                max: 90.0,
                count: PROFILE_REFRESH_FRAMES as usize,
            },
            ..StageProfile::default()
        };
        for frame in 0..PROFILE_REFRESH_FRAMES {
            pipeline
                .advance(slow_frame(frame as f64 * 0.033, 41.0))
                .unwrap();
        }
        assert!(
            pipeline.control.lock().unwrap().ascii,
            "no budget means no shedding"
        );
    }
}
