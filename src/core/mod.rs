pub mod event_bus;
pub mod geometry;
pub mod scale;
pub mod types;

use crate::core::types::GestureEvent;
use anyhow::Result;

// Traits defining the interface for all components.
// Using Send + Sync to ensure they can be safely passed between threads.

#[derive(Debug, Clone)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Wall-clock time each ONNX stage took for the frame that produced a result.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimings {
    pub detector_us: f64,
    pub landmarker_us: f64,
}

/// The playable MIDI range: C2 (36) to C6 (84), 49 semitones.
pub const NOTE_LOW: u8 = 36;
pub const NOTE_HIGH: u8 = 84;
pub const NOTE_COUNT: usize = (NOTE_HIGH - NOTE_LOW + 1) as usize;
/// Fraction of the frame width left unused at each edge. The outermost notes
/// would otherwise need the hand pressed against the frame border, where the
/// detector loses it.
pub const NOTE_MARGIN: f32 = 0.10;

#[derive(Debug, Clone, Default)]
pub struct HandDetectionResult {
    pub hand_landmarks: Option<Vec<Landmark>>,
    /// (x, y, w, h) bounding box of the detected hand, in frame pixels.
    pub r#box: Option<(f32, f32, f32, f32)>,
    /// Size of the frame those pixels live in, so strategies can normalise
    /// against the whole view rather than the hand box.
    pub frame_size: (usize, usize),
    /// Detection confidence of the winning anchor (0..1).
    pub score: f32,
    /// Per-stage inference cost of this frame, in microseconds.
    pub timings: StageTimings,
}

/// Interface for an output sink (Audio or MIDI).
pub trait Sink: Send + Sync {
    fn start(&mut self) -> Result<()>;
    fn stop(&mut self) -> Result<()>;
    fn handle_event(&mut self, event: &GestureEvent) -> Result<()>;
}

/// Interface for mapping hand landmarks to gesture events.
pub trait Strategy: Send + Sync {
    fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent>;
}
