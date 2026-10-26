//! The single struct the vision pipeline publishes for the dashboard.
//!
//! The pipeline thread overwrites it under a mutex; the iced app clones it on
//! every UI tick. Keeping it one flat struct means the UI never has to reach
//! into the tracker or the sinks.

use std::collections::VecDeque;

use crate::core::Landmark;

/// Maximum number of events kept in the on-screen log.
pub const LOG_CAPACITY: usize = 12;

/// The note range `FingerPositionStrategy` can produce: MIDI 36 (C2) to 84 (C6).
pub const NOTE_LOW: u8 = 36;
pub const NOTE_HIGH: u8 = 84;
pub const NOTE_COUNT: usize = (NOTE_HIGH - NOTE_LOW + 1) as usize;

/// E-folding time of the on-screen note trail. A note lit up now is at ~37 %
/// of its brightness after this many seconds, and effectively gone after ~3 s.
const FADE_TAU: f32 = 0.45;

/// Index of `note` in the trail/keyboard, clamped into the playable range.
pub fn note_index(note: u8) -> usize {
    (note.clamp(NOTE_LOW, NOTE_HIGH) - NOTE_LOW) as usize
}

/// Equal-tempered frequency of a MIDI note, A4 (69) = 440 Hz.
pub fn note_frequency(note: u8) -> f32 {
    440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0)
}

/// Exponentially fade the whole keyboard trail by `dt_secs`.
pub fn fade_trail(trail: &mut [f32; NOTE_COUNT], dt_secs: f32) {
    let decay = (-dt_secs / FADE_TAU).exp();
    for level in trail.iter_mut() {
        *level *= decay;
    }
}

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

    /// The camera frame as RGBA pixels, handed to the dashboard so it can draw
    /// the video and the skeleton in the same canvas. Only produced when the UI
    /// is running.
    pub frame_id: u64,
    pub frame_pixels: Vec<u8>,

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

    #[test]
    fn keyboard_covers_the_strategy_range() {
        assert_eq!(NOTE_COUNT, 49);
        assert_eq!(note_index(36), 0);
        assert_eq!(note_index(84), 48);
        // Out-of-range notes are clamped rather than panicking.
        assert_eq!(note_index(0), 0);
        assert_eq!(note_index(200), 48);
    }

    #[test]
    fn a4_is_concert_pitch() {
        assert!((note_frequency(69) - 440.0).abs() < 1e-3);
        assert!((note_frequency(60) - 261.626).abs() < 1e-3);
    }

    #[test]
    fn the_trail_fades_exponentially_and_drops_old_notes() {
        let mut trail = [0.0; NOTE_COUNT];
        trail[note_index(60)] = 1.0;
        // After one time constant the note is at 1/e of its brightness...
        fade_trail(&mut trail, FADE_TAU);
        assert!((trail[note_index(60)] - 0.367879).abs() < 1e-3);
        // ...and after a few constants it is below the drawing threshold.
        fade_trail(&mut trail, 3.0 * FADE_TAU);
        assert!(trail[note_index(60)] < 0.02);
        // Untouched cells stay at zero.
        assert_eq!(trail[note_index(36)], 0.0);
    }
}
