mod audio;
mod core;
mod dashboard;
mod midi;
mod pipeline;
mod sources;
mod strategies;
mod vision;
mod visualizer;

use anyhow::{Context, Result};
use clap::Parser;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::audio::synthesizer::Synthesizer;
use crate::core::event_bus::EventBus;
use crate::core::Sink;
use crate::dashboard::metrics::{FrameMetrics, DEFAULT_FADE_MS};
use crate::midi::MidiOutputEngine;
use crate::pipeline::Pipeline;
use crate::strategies::FingerPositionStrategy;
use crate::vision::camera::WebcamSource;
use crate::vision::tracker::HandTrackerImpl;
use crate::visualizer::Visualizer;

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

    #[arg(short, long, default_value_t = 0.5)]
    amplitude: f32,

    /// Detector confidence gate (0-1). Raise it if the tracker fires on things
    /// that are not hands; lower it if hands at an angle are missed.
    #[arg(long, default_value_t = crate::vision::tracker::DEFAULT_SCORE_THRESHOLD)]
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

    /// Webcam index.
    #[arg(long, default_value_t = 0)]
    camera: i32,

    /// Requested capture resolution (the inference stages crop to a square
    /// anyway, so 640x480 keeps both the GPU and the dashboard cheap).
    #[arg(long, default_value_t = 640)]
    camera_width: u32,

    #[arg(long, default_value_t = 480)]
    camera_height: u32,

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

    /// Run the pipeline without opening the iced dashboard.
    #[arg(long, default_value_t = false)]
    no_ui: bool,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
enum Mode {
    Signal,
    Midi,
    Mock,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!(
        "Starting Gesture-to-Audio Engine in {:?} mode...",
        args.mode
    );

    // 1. Headless single-image mode: verify the detector offline.
    if let Some(path) = &args.image {
        let mut tracker = HandTrackerImpl::new(
            &args.detector_path,
            &args.landmarker_path,
            &args.anchors_path,
            args.score_threshold,
            args.debug,
        )
        .context("Failed to initialize hand tracker")?;

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
            let mut synth = Synthesizer::new(44100.0, 1, args.amplitude);
            synth.start()?;
            sinks.push(Box::new(synth));
            "audio"
        }
        Mode::Midi => {
            println!("Initializing MIDI Engine (Port: {})...", args.port);
            let mut midi_engine = MidiOutputEngine::new(&args.port);
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
        Some(
            HandTrackerImpl::new(
                &args.detector_path,
                &args.landmarker_path,
                &args.anchors_path,
                args.score_threshold,
                args.debug,
            )
            .context("Failed to initialize hand tracker")?,
        )
    };
    let camera = if args.mode == Mode::Mock {
        None
    } else {
        Some(
            WebcamSource::new(args.camera, args.camera_width, args.camera_height, !args.no_mirror)
                .context("Failed to open webcam")?,
        )
    };

    let shared = Arc::new(Mutex::new(FrameMetrics {
        mode: format!("{:?}", args.mode),
        sink: sink_label.to_string(),
        ..FrameMetrics::default()
    }));

    let running = Arc::new(AtomicBool::new(true));
    // Start/Stop button: cleared by the UI, checked by the pipeline every frame.
    let active = Arc::new(AtomicBool::new(true));
    let mut pipeline = Pipeline::new(
        camera,
        tracker,
        Box::new(FingerPositionStrategy),
        sinks,
        event_bus,
        event_rx,
        shared.clone(),
        args.max_frames,
        args.debug,
        !args.no_ui,
        active.clone(),
    );

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

    let ui_result = dashboard::run(shared, active, args.fade_ms);

    running.store(false, Ordering::Relaxed);
    let _ = handle.join();

    println!("Shutting down...");
    ui_result?;
    Ok(())
}
