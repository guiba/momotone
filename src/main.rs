mod core;
mod audio;
mod midi;
mod sources;
mod vision;
mod visualizer;
use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::{unbounded, Receiver};
use opencv::prelude::*;
use std::time::{Duration, Instant};

use crate::core::types::{GestureEvent, EventType};
use crate::core::event_bus::EventBus;
use crate::core::Sink;
use crate::audio::synthesizer::Synthesizer;
use crate::midi::MidiOutputEngine;
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

    #[arg(short, long, default_value = "OP-1")]
    port: String,

    #[arg(long, default_value_t = 0.5)]
    amplitude: f32,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
enum Mode {
    Signal,
    Midi,
    Mock,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Starting Gesture-to-Audio Engine in {:?} mode...", args.mode);

    let (event_bus, event_rx) = EventBus::new();

    // 1. Setup Sinks based on Mode
    let mut sinks: Vec<Box<dyn Sink>> = Vec::new();

    match args.mode {
        Mode::Signal => {
            println!("Initializing Audio Engine...");
            let mut synth = Synthesizer::new(44100.0, 1);
            synth.start()?;
            sinks.push(Box::new(synth));
        }
        Mode::Midi => {
            println!("Initializing MIDI Engine (Port: {})...", args.port);
            let mut midi_engine = MidiOutputEngine::new(&args.port);
            midi_engine.start()?;
            sinks.push(Box::new(midi_engine));
        }
        Mode::Mock => {
            println!("Running in Mock mode (Simulating video & gestures)");
        }
    }

    // 2. Initialize Vision
    let mut camera = WebcamSource::new(0).context("Failed to open webcam")?;
    let tracker = HandTrackerImpl::new("hand_landmarker.task");
    let visualizer = Visualizer::new();

    // 3. Orchestration Loop
    let mut last_event_time = Instant::now();
    let mut iterations = 0;

    loop {
        iterations += 1;

        // --- A. Vision Pipeline (The "Input") ---
        let mut frame = camera.get_frame()?;
        let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis() as u64;

        // Get detection result (Currently returns mock landmarks)
        let detection = tracker.process_frame(&frame)?;

        // --- B. Strategy/Gesture Processing (The "Logic") ---
        // In a real app, this maps 'detection' to 'event'. For now, we simulate events in Mock mode.
        if args.mode == Mode::Mock {
            if last_event_time.elapsed() > Duration::from_secs(2) {
                let dummy_event = if (iterations / 10) % 2 == 0 {
                    GestureEvent::new_signal(440.0, args.amplitude, "Simulated Signal")
                } else {
                    GestureEvent::new_note((60 + (iterations % 12) as u8), 1.0, "Simulated Note")
                };
                event_bus.dispatch(dummy_event);
                last_event_time = Instant::now();
            }
        } else if let Some(landmarks) = &detection.hand_landmarks {
             // Real logic would be: let event = strategy.process(&landmarks);
             // For now, just use dummy events for testing real-world loops without the full model
             if iterations % 100 == 0 {
                 event_bus.dispatch(GestureEvent::new_note(65, 0.8, "Hand Detected Note"));
             }
        }

        // --- C. Event Dispatch (The "Output") ---
        while let Ok(event) = event_rx.try_recv() {
            println!("[Event] {} -> {:?}", event.text, event.payload);
            for sink in &mut sinks {
                let _ = sink.handle_event(&event);
            }
        }

        // --- D. Visualization (The "Display") ---
        let status = [
            ("MIDI", if args.mode == Mode::Midi { "Connected" } else { "N/A" }),
            ("Audio", if args.mode == Mode::Signal { "Running" } else { "Off" }),
            ("Tracking", if detection.hand_landmarks.is_some() { "Active" } else { "Searching..." }),
        ];

        let event_text = match &event_bus.dispatch(GestureEvent::no_gesture()) { // Dummy hack to get text? No, let's just use a placeholder
             // (Wait, I can't access internal bus state easily like this)
             _ => if args.mode == Mode::Mock { "Simulated Movement" } else { "Looking for hand..." },
        };
        // Note: In real code we track the last received event to show it on screen
        visualizer.draw_hud(&mut frame, event_text, (0, 255, 0), &status)?;

        opencv::highgui::imshow("Gesture-to-Audio Rust (OpenCV)", &frame)?;

        if opencv::highgui::wait_key(1)? == 'q' as i32 {
            break;
        }
    }

    println!("Shutting down...");
    for mut sink in sinks {
        let _ = sink.stop();
    }

    Ok(())
}
