//! The single struct the vision pipeline publishes for the dashboard.
//!
//! The pipeline thread overwrites it under a mutex; the iced app clones it on
//! every UI tick. Keeping it one flat struct means the UI never has to reach
//! into the tracker or the sinks.

use std::collections::VecDeque;

use crate::core::Landmark;

/// Maximum number of events kept in the on-screen log.
pub const LOG_CAPACITY: usize = 12;

#[derive(Debug, Clone)]
pub struct EventLogEntry {
    pub uptime_ms: u64,
    pub text: String,
    pub detail: String,
}

#[derive(Debug, Clone, Default)]
pub struct FrameMetrics {
    // --- tracking ---
    pub detected: bool,
    pub confidence: f32,
    pub landmark_count: usize,
    pub landmarks: Vec<Landmark>,
    pub hand_box: Option<(f32, f32, f32, f32)>,
    pub frame_size: (usize, usize),
    /// Milliseconds since the last frame in which a hand was found.
    pub lost_ms: u64,
    /// Mean landmark displacement between consecutive frames, in pixels.
    pub jitter_px: f32,

    // --- gesture output ---
    pub note: Option<u8>,
    pub volume: f32,
    pub frequency: Option<f32>,
    pub spread: f32,
    pub extension: [f32; 5],
    pub tilt_deg: f32,
    pub palm_length: f32,

    // --- performance ---
    pub fps: f32,
    pub frame_ms: f64,
    pub detector_us: f64,
    pub landmarker_us: f64,
    pub frames: u64,
    pub uptime_ms: u64,

    // --- context ---
    pub mode: String,
    pub sink: String,
    pub log: VecDeque<EventLogEntry>,
}

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// MIDI note number -> scientific pitch name, e.g. 60 -> "C4".
pub fn note_name(midi: u8) -> String {
    format!("{}{}", NOTE_NAMES[(midi as usize) % 12], midi / 12 - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn middle_c_is_c4() {
        assert_eq!(note_name(60), "C4");
        assert_eq!(note_name(36), "C2");
        assert_eq!(note_name(84), "C6");
        assert_eq!(note_name(61), "C#4");
    }
}
