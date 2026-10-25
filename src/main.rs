  mod core;
  mod audio;
  mod midi;
  mod sources;
  mod vision;
  mod visualizer;
  mod strategies;

  use anyhow::{Context, Result};
  use clap::Parser;


  use std::time::{Duration, Instant};

  use crate::core::types::GestureEvent;
  use crate::core::event_bus::EventBus;
  use crate::core::Sink;
  use crate::audio::synthesizer::Synthesizer;
  use crate::midi::MidiOutputEngine;
  use crate::vision::camera::WebcamSource;
  use crate::vision::tracker::HandTrackerImpl;
  use crate::visualizer::Visualizer;
  use crate::strategies::FingerPositionStrategy;
  use crate::core::Strategy;

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

    /// Write every annotated frame to this PNG path (for debugging the box).
    #[arg(long)]
    dump_frame: Option<String>,

    /// Run the detector on a single still image instead of the webcam.
    #[arg(long)]
    image: Option<String>,

    /// Stop after this many webcam frames (useful with --dump-frame).
    #[arg(long)]
    max_frames: Option<u32>,

      #[arg(long, default_value = "OP-1")]
      port: String,
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

       // 2. Initialize Vision, Strategy and the Visualizer
        let mut tracker = HandTrackerImpl::new(&args.detector_path, &args.landmarker_path, &args.anchors_path, args.debug)
            .context("Failed to initialize hand tracker")?;
        let visualizer = Visualizer::new();
        let strategy = FingerPositionStrategy;

        // Headless single-image mode: useful for verifying detection offline.
        if let Some(path) = &args.image {
             let mut frame = opencv::imgcodecs::imread(path, opencv::imgcodecs::IMREAD_COLOR)
                 .with_context(|| format!("Failed to read image {path}"))?;
            let detection = tracker.process_frame(&frame)?;
            visualizer.draw_skeleton(&mut frame, &detection)?;
           let out = args.dump_frame.clone().unwrap_or_else(|| "output_frame.png".to_string());
           opencv::imgcodecs::imwrite(&out, &frame, &opencv::core::Vector::new())?;
           println!("Detection: box={:?} score={:.3} landmarks={}",
               detection.r#box, detection.score, detection.hand_landmarks.as_ref().map_or(0, |l| l.len()));
           println!("Annotated frame written to {out}");
           return Ok(());
       }

        let mut camera = WebcamSource::new(0).context("Failed to open webcam")?;
       // 3. Orchestration Loop
       let mut last_event_time = Instant::now();
       let mut iterations = 0;

       loop {
           iterations += 1;

           // --- A. Vision Pipeline (The "Input") ---
           let mut frame = camera.get_frame()?;

           // Get detection result
           let detection = tracker.process_frame(&frame)?;

          // --- B. Strategy/Gesture Processing (The "Logic") ---
          if args.mode == Mode::Mock {
              if last_event_time.elapsed() > Duration::from_secs(2) {
                  let dummy_event = if (iterations / 10) % 2 == 0 {
                      GestureEvent::new_signal(440.0, args.amplitude, "Simulated Signal")
                  } else {
                      GestureEvent::new_note(60 + (iterations % 12) as u8, 1.0, "Simulated Note")
                  };
                  event_bus.dispatch(dummy_event);
                  last_event_time = Instant::now();
              }
            } else if detection.r#box.is_some() {
              if let Some(gesture_event) = strategy.process(&detection) {
                  event_bus.dispatch(gesture_event);
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
            visualizer.draw_skeleton(&mut frame, &detection)?;
            let status = [
                ("MIDI", if args.mode == Mode::Midi { "Connected" } else { "N/A" }),
                ("Audio", if args.mode == Mode::Signal { "Running" } else { "Off" }),
                ("Tracking", if detection.r#box.is_some() { "Active" } else { "Searching..." }),
            ];

            let event_text = if args.mode == Mode::Mock { 
                "Simulated Movement" 
            } else if detection.r#box.is_some() {
                "Hand detected"
            } else {
                "Looking for hand..."
            };

            visualizer.draw_hud(&mut frame, event_text, (0, 255, 0), &status)?;

            opencv::highgui::imshow("Gesture-to-Audio Rust (OpenCV)", &frame)?;

            if let Some(path) = &args.dump_frame {
                opencv::imgcodecs::imwrite(path, &frame, &opencv::core::Vector::new())?;
            }

            if args.max_frames.map_or(false, |n| iterations >= n) {
                break;
            }

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