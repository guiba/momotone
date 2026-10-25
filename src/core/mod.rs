pub mod types;
pub mod event_bus;

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

#[derive(Debug, Clone, Default)]
pub struct HandDetectionResult {
    pub hand_landmarks: Option<Vec<Landmark>>,
    /// (x, y, w, h) bounding box of the detected hand, in frame pixels.
    pub r#box: Option<(f32, f32, f32, f32)>,
    /// Detection confidence of the winning anchor (0..1).
    pub score: f32,
}

/// Interface for an output sink (Audio or MIDI).
pub trait Sink: Send + Sync {
    fn start(&mut self) -> Result<()>;
    fn stop(&mut self) -> Result<()>;
    fn handle_event(&mut self, event: &GestureEvent) -> Result<()>;
}

/// Interface for the source of movement (Webcam/Mock).
pub trait Source: Send + Sync {
    fn update(&mut self) -> Result<Option<Vec<Landmark>>>; 
}

/// Interface for hand tracking.
pub trait Tracker: Send + Sync {
    fn update(&mut self, timestamp_ms: u64) -> Result<HandDetectionResult>;
    fn stop(&mut self);
}

/// Interface for mapping hand landmarks to gesture events.
pub trait Strategy: Send + Sync {
    fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent>;
}
