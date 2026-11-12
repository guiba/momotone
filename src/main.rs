mod audio;
mod core;
mod dashboard;
mod gesture;
mod midi;
mod pipeline;
mod presentation;
mod vision;

use anyhow::{Context, Result};
use clap::Parser;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::audio::synthesizer::Synthesizer;
use crate::core::event_bus::EventBus;
use crate::core::runtime::{control_channel, RuntimeConfig};
use crate::core::Sink;
use crate::dashboard::metrics::{FrameMetrics, SceneView, SoundView, StatusView, DEFAULT_FADE_MS};
use crate::gesture::keyboard::FingerPositionStrategy;
use crate::gesture::theremin::ThereminStrategy;
use crate::gesture::Strategy;
use crate::midi::MidiOutputEngine;
use crate::pipeline::Pipeline;
use crate::presentation::annotator::Visualizer;
use crate::presentation::ascii::{Ramp, DEFAULT_ASCII_FPS, DEFAULT_COLS};
use crate::presentation::presenter::DEFAULT_FEED_FPS;
use crate::vision::camera::WebcamSource;
use crate::vision::tracker::{HandTracker, DEFAULT_SCORE_THRESHOLD};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = false)]
    debug: bool,

    #[arg(short, long, value_enum, default_value_t = Mode::Signal)]
    mode: Mode,

    #[arg(long, default_value = "assets/hand_landmarks_detector.onnx")]
    landmarker_path: String,

    #[arg(long, default_value = "assets/hand_detector.onnx")]
    detector_path: String,

    #[arg(long, default_value = "assets/anchors.csv")]
    anchors_path: String,

    /// Which hand model to run. `mediapipe` is the two-stage BlazeHand pair (palm
    /// detector then landmarker); `yolo` runs a single Ultralytics pose graph that
    /// returns the hand box and all 21 keypoints in one pass.
    #[arg(long, value_enum, default_value = "mediapipe")]
    model: Model,

    /// ONNX Runtime intra-op threads. 0 leaves the runtime default (one thread per
    /// core); a positive number pins the pool, which is worth trying if the model's
    /// thread churn makes the UI or the capture jitter.
    #[arg(long, default_value_t = 0)]
    ort_threads: usize,

    /// ONNX file for `--model yolo`. Get one with `scripts/get_pose_model.sh`.
    #[arg(long, default_value = "assets/yolo_hand_pose.onnx")]
    pose_path: String,

    /// Square model input for `--model yolo`. 640 matches the Ultralytics training
    /// size; 416 or 320 is several times cheaper on a CPU and usually still good
    /// enough for a single hand in front of the camera.
    #[arg(long, default_value_t = 640)]
    pose_input: usize,

    #[arg(short, long, default_value_t = 0.5)]
    amplitude: f32,

    /// Detector confidence gate (0-1). Raise it if the tracker fires on things
    /// that are not hands; lower it if hands at an angle are missed.
    #[arg(long, default_value_t = DEFAULT_SCORE_THRESHOLD)]
    score_threshold: f32,

    /// Write every annotated frame to this PNG path (image mode only).
    #[arg(long)]
    dump_frame: Option<String>,

    /// Run the detector on a single still image instead of the webcam.
    #[arg(long)]
    image: Option<String>,

    /// Stop after this many frames.
    #[arg(long)]
    max_frames: Option<u32>,

    #[arg(long, default_value = "OP-1")]
    port: String,
    /// Minimum time a MIDI note sounds before the voice may move to the next
    /// cell. The strategy fires every frame, so without a floor a hand sliding
    /// across the keyboard re-attacks on every cell it crosses. 0 attacks on
    /// every change.
    #[arg(long, default_value_t = midi::DEFAULT_MIN_NOTE_MS)]
    min_note_ms: u64,

    /// Webcam index.
    #[arg(long, default_value_t = 0)]
    camera: i32,

    /// Requested capture resolution (the inference stages crop to a square
    /// anyway, so 640x480 keeps both the GPU and the dashboard cheap).
    #[arg(long, default_value_t = 640)]
    camera_width: u32,

    #[arg(long, default_value_t = 480)]
    camera_height: u32,

    /// Capture frame rate. Defaults to 30: the loop is paced by the camera, so a
    /// device that defaults to 60 fps would run inference twice as often for no
    /// gain.
    #[arg(long, default_value_t = 30)]
    camera_fps: u32,

    /// How often the dashboard is handed a new camera image. iced uploads every
    /// `image::Handle` to the GPU synchronously, so a 640x480 preview costs a
    /// blocking wgpu round trip per upload while the skeleton and keyboard keep
    /// updating at full frame rate. 0 sends every frame.
    #[arg(long, default_value_t = DEFAULT_FEED_FPS)]
    feed_fps: f32,

    /// Draw the camera as ASCII art behind the skeleton. The dashboard's `ascii`
    /// button flips the same flag at runtime; this starts the layer switched on.
    #[arg(long, default_value_t = false)]
    ascii: bool,

    /// Columns of the ASCII grid. Rows are derived from the frame aspect, and the
    /// stage resamples the grid to the cells that actually fit the panel.
    #[arg(long, default_value_t = DEFAULT_COLS)]
    ascii_cols: usize,

    /// How often the ASCII grid is rebuilt. It is a backdrop, not a video, so a
    /// lower rate than the camera is fine. 0 rebuilds every frame.
    #[arg(long, default_value_t = DEFAULT_ASCII_FPS)]
    ascii_fps: f32,

    /// Glyph ramp: `blocks` (▏▎▍▌▋▊▉█, safest in the window), `shades` (the
    /// bottom-up brightness scale ▁▂▃▄▅▆▇█), `ascii` (the classic `.:-=+*#@`),
    /// `braille` (dot matrix - good in a terminal, the iced canvas falls back to a
    /// non-monospaced font for those codepoints) or `dots` (sparse halftone).
    /// The dashboard's RENDER card switches ramp at runtime.
    #[arg(long, default_value = "blocks")]
    ascii_ramp: String,

    /// Write each rendered ASCII frame to this file. Handy for checking the art
    /// headlessly, e.g. with `--mode mock --no-ui --ascii-dump art.txt`.
    #[arg(long)]
    ascii_dump: Option<String>,

    /// Stop mirroring the webcam. By default the frame is flipped horizontally
    /// so the stage behaves like a mirror: your right hand shows up on the
    /// right, which is also where the high notes are.
    #[arg(long, default_value_t = false)]
    no_mirror: bool,

    /// How long the keyboard trail lingers after a key is released, in
    /// milliseconds. This is the decay time constant: after this many ms a lit
    /// key is at ~37 % of its brightness, so 450 ms reads as a ~1.5 s tail.
    #[arg(long, default_value_t = DEFAULT_FADE_MS)]
    fade_ms: f32,

    /// Run the landmark model on every frame even when the palm is stationary,
    /// disabling the reuse fast path (useful for A/B testing the stage timings).
    #[arg(long, default_value_t = false)]
    landmarker_always: bool,

    /// Scale the pitch is quantised to. Quantising means every reachable key is
    /// consonant with every other one, so sweeping the hand cannot clash. Use
    /// `chromatic` for the old every-semitone behaviour.
    #[arg(long, default_value = "major pentatonic")]
    scale: String,

    /// Root of the scale: C, D, Eb, F# ...
    #[arg(long, default_value = "C")]
    root: String,

    /// Number of octaves the keyboard spans. With a five-degree scale this is the
    /// number of rows in the grid.
    #[arg(long, default_value_t = 3)]
    octaves: usize,

    /// Oscillator timbre: sine, triangle, square or saw.
    #[arg(long, default_value = "sine")]
    waveform: String,

    /// Which landmark -> output mapping drives the sinks. `position` plays the
    /// degree x octave keyboard; `theremin` glides continuously through a
    /// frequency band instead, so nothing is quantised to a scale.
    #[arg(long, value_enum, default_value = "position")]
    gesture: Gesture,

    /// Frequency band the theremin sweeps when `--gesture theremin`.
    #[arg(long, default_value_t = gesture::theremin::DEFAULT_LOW_HZ)]
    theremin_low_hz: f32,
    #[arg(long, default_value_t = gesture::theremin::DEFAULT_HIGH_HZ)]
    theremin_high_hz: f32,

    /// Run the pipeline without opening the iced dashboard.
    #[arg(long, default_value_t = false)]
    no_ui: bool,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
enum Model {
    #[value(name = "mediapipe")]
    MediaPipe,
    Yolo,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
enum Gesture {
    Position,
    Theremin,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
enum Mode {
    Signal,
    Midi,
    Mock,
}

/// Build the tracker for the model this run asked for.
fn build_tracker(args: &Args) -> Result<HandTracker> {
    match args.model {
        Model::MediaPipe => HandTracker::mediapipe(
            &args.detector_path,
            &args.landmarker_path,
            &args.anchors_path,
            args.score_threshold,
            args.debug,
            !args.landmarker_always,
            args.ort_threads,
        ),
        Model::Yolo => HandTracker::yolo_pose(
            &args.pose_path,
            args.pose_input,
            args.score_threshold,
            args.debug,
            !args.landmarker_always,
            args.ort_threads,
        ),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!(
        "Starting Gesture-to-Audio Engine in {:?} mode...",
        args.mode
    );

    let waveform = core::instrument::Waveform::parse(&args.waveform).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown waveform '{}'; available: {}",
            args.waveform,
            core::instrument::Waveform::all()
                .iter()
                .map(|w| w.name())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
    let instrument =
        core::instrument::InstrumentConfig::new(&args.scale, &args.root, args.octaves, waveform)?;

    let ascii_ramp = Ramp::parse(&args.ascii_ramp).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown ascii ramp '{}'; available: {}",
            args.ascii_ramp,
            Ramp::names().join(", ")
        )
    })?;

    // 1. Headless single-image mode: verify the detector offline.
    if let Some(path) = &args.image {
        let mut tracker = build_tracker(&args).context("Failed to initialize hand tracker")?;

        let mut frame = opencv::imgcodecs::imread(path, opencv::imgcodecs::IMREAD_COLOR)
            .with_context(|| format!("Failed to read image {path}"))?;
        let detection = tracker.process_frame(&frame)?;
        Visualizer::new().draw_skeleton(&mut frame, &detection)?;

        let out = args
            .dump_frame
            .clone()
            .unwrap_or_else(|| "output_frame.png".to_string());
        opencv::imgcodecs::imwrite(&out, &frame, &opencv::core::Vector::new())?;

        println!(
            "Detection: box={:?} score={:.3} landmarks={} detector={:.0}µs landmarker={:.0}µs",
            detection.r#box,
            detection.score,
            detection.hand_landmarks.as_ref().map_or(0, |l| l.len()),
            detection.timings.detector_us,
            detection.timings.landmarker_us,
        );
        println!("Annotated frame written to {out}");
        return Ok(());
    }

    // 2. Sinks.
    let (event_bus, event_rx) = EventBus::new();
    let mut sinks: Vec<Box<dyn Sink>> = Vec::new();
    let sink_label = match args.mode {
        Mode::Signal => {
            println!("Initializing Audio Engine... (gain {:.2})", args.amplitude);
            let mut synth = Synthesizer::new(args.amplitude);
            synth.start()?;
            sinks.push(Box::new(synth));
            "audio"
        }
        Mode::Midi => {
            println!("Initializing MIDI Engine (Port: {})...", args.port);
            let mut midi_engine = MidiOutputEngine::new(&args.port, args.min_note_ms);
            midi_engine.start()?;
            sinks.push(Box::new(midi_engine));
            "midi"
        }
        Mode::Mock => {
            println!("Running in Mock mode (synthetic hand, no camera or models)");
            "none"
        }
    };

    // 3. Vision. Mock mode needs neither a webcam nor the ONNX assets.
    let tracker = if args.mode == Mode::Mock {
        None
    } else {
        Some(build_tracker(&args).context("Failed to initialize hand tracker")?)
    };
    let camera = if args.mode == Mode::Mock {
        None
    } else {
        Some(
            WebcamSource::new(
                args.camera,
                args.camera_width,
                args.camera_height,
                args.camera_fps,
                !args.no_mirror,
            )
            .context("Failed to open webcam")?,
        )
    };

    if args.debug {
        if let Some(camera) = &camera {
            println!(
                "[Camera] requested {}x{} @ {} fps, driver reports {} fps",
                args.camera_width,
                args.camera_height,
                camera.requested_fps(),
                camera.reported_fps()
            );
        }
    }

    let strategy: Box<dyn Strategy> = match args.gesture {
        Gesture::Position => Box::new(FingerPositionStrategy::new(
            instrument.root,
            &instrument.scale,
            instrument.octaves,
        )),
        Gesture::Theremin => Box::new(ThereminStrategy::new(
            args.theremin_low_hz,
            args.theremin_high_hz,
        )),
    };
    // The keyboard the dashboard draws is the one the instrument config built;
    // the strategy only needs it when it quantises to a grid.
    let layout = instrument.layout.clone();
    let scale_name = instrument.scale.name.to_string();
    let root_name = instrument.root_name.clone();

    let shared = Arc::new(Mutex::new(FrameMetrics {
        status: StatusView {
            mode: format!("{:?} / {:?}", args.mode, args.gesture),
            sink: sink_label.to_string(),
            scale: scale_name,
            root: root_name,
            ..StatusView::default()
        },
        scene: SceneView {
            layout,
            ..SceneView::default()
        },
        sound: SoundView {
            fade_ms: args.fade_ms,
            ..SoundView::default()
        },
        ..FrameMetrics::default()
    }));

    let running = Arc::new(AtomicBool::new(true));
    // Start/Stop, feed, ASCII and the glyph ramp all travel down one channel
    // instead of living in four atomics the UI writes and the loop reads.
    let (control_tx, control_rx) = control_channel();
    let control = Arc::new(Mutex::new(RuntimeConfig::new(
        args.feed_fps,
        args.ascii_fps,
        args.ascii,
        ascii_ramp,
    )));

    let mut pipeline = Pipeline::new(pipeline::PipelineConfig {
        camera,
        tracker,
        strategy,
        sinks: Arc::new(Mutex::new(sinks)),
        event_bus,
        event_rx: Some(event_rx),
        shared: shared.clone(),
        control: control.clone(),
        control_rx,
        instrument: instrument.clone(),
        ascii_cols: args.ascii_cols,
        ascii_dump: args.ascii_dump.clone(),
        fade_ms: args.fade_ms,
        // With no window there is no stage to look at, so `--ascii` means the
        // art goes to stdout instead.
        ascii_stdout: args.no_ui && args.ascii,
        max_frames: args.max_frames,
        debug: args.debug,
        emit_frames: !args.no_ui,
    });

    // 4. Either run the loop in the foreground, or hand it to a thread and
    //    give the main thread to the iced dashboard.
    if args.no_ui {
        pipeline.run(&running)?;
        println!("Shutting down...");
        return Ok(());
    }

    let pipeline_flag = running.clone();
    let handle = std::thread::spawn(move || {
        if let Err(e) = pipeline.run(&pipeline_flag) {
            eprintln!("[Pipeline] {e}");
        }
    });

    let ui_result = dashboard::run(shared, control_tx, instrument, ascii_ramp);

    running.store(false, Ordering::Relaxed);
    let _ = handle.join();

    println!("Shutting down...");
    ui_result?;
    Ok(())
}
