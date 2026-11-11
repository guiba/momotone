//! The single struct the vision pipeline publishes for the dashboard.
//!
//! The pipeline thread overwrites it under a mutex; the iced app clones it on
//! every UI tick. Keeping it one flat struct means the UI never has to reach
//! into the tracker or the sinks.

use std::collections::VecDeque;

use crate::core::grid::GridLayout;
use crate::core::telemetry::StageProfile;
use crate::core::{Landmark, NOTE_HIGH, NOTE_LOW};
use crate::vision::ascii::AsciiGrid;

/// Maximum number of events kept in the on-screen log.
pub const LOG_CAPACITY: usize = 12;

/// Default e-folding time of the on-screen note trail, in milliseconds. A key
/// lit up now is at ~37 % of its brightness after one time constant and
/// effectively gone after three. `--fade-ms` overrides it.
pub const DEFAULT_FADE_MS: f32 = 450.0;

/// Index of `note` in the trail/keyboard, clamped into the playable range.
pub fn note_index(note: u8) -> usize {
    (note.clamp(NOTE_LOW, NOTE_HIGH) - NOTE_LOW) as usize
}

/// Position of `note` in the grid layout, if the layout contains it.
pub fn cell_index(layout: &GridLayout, note: u8) -> Option<usize> {
    layout.notes.iter().position(|&n| n == note)
}

/// Equal-tempered frequency of a MIDI note, A4 (69) = 440 Hz.
pub fn note_frequency(note: u8) -> f32 {
    440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0)
}

/// MIDI note closest to a frequency, clamped to the playable keyboard range.
/// The theremin has no discrete note, so the ASCII backdrop uses the nearest
/// key to pick its colour.
pub fn note_for_frequency(freq: f32) -> u8 {
    if !freq.is_finite() || freq <= 0.0 {
        return NOTE_LOW;
    }
    let semitones = 69.0 + 12.0 * (freq / 440.0).log2();
    semitones.round().clamp(NOTE_LOW as f32, NOTE_HIGH as f32) as u8
}

/// Exponentially fade the whole keyboard trail by `dt_secs`, using `tau_secs`
/// as the time constant.
pub fn fade_trail(trail: &mut [f32], dt_secs: f32, tau_secs: f32) {
    let tau = tau_secs.max(1e-3);
    let decay = (-dt_secs / tau).exp();
    for level in trail.iter_mut() {
        *level *= decay;
    }
}

#[derive(Debug, Clone)]
pub struct EventLogEntry {
    pub uptime_ms: u64,
    pub detail: String,
}

#[derive(Debug, Clone, Default)]
pub struct FrameMetrics {
    // --- tracking ---
    /// True when the dashboard's Stop button has paused the pipeline.
    pub paused: bool,
    pub detected: bool,
    pub confidence: f32,
    pub landmark_count: usize,
    pub landmarks: Vec<Landmark>,
    pub hand_box: Option<(f32, f32, f32, f32)>,
    pub frame_size: (usize, usize),
    /// Index fingertip position in frame pixels, published so the dashboard can
    /// show where on the keyboard the hand is sitting.
    pub tip_x: f32,
    pub tip_y: f32,
    /// Grid cell the hand is over: column (degree) and row (octave), row 0 = top.
    pub grid_col: usize,
    pub grid_row: usize,
    /// Milliseconds since the last frame in which a hand was found.
    pub lost_ms: u64,
    /// Mean landmark displacement between consecutive frames, in pixels.
    pub jitter_px: f32,

    /// The camera frame as RGBA pixels, handed to the dashboard so it can draw
    /// the video and the skeleton in the same canvas. Only produced when the UI
    /// is running.
    pub frame_pixels: Vec<u8>,
    /// Quantised luminance grid behind the skeleton: the ASCII art backdrop.
    /// Published only while the `ascii` toggle is on (or a dump/terminal output
    /// is requested), at `--ascii-fps`.
    pub ascii: AsciiGrid,

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
    /// Tail behaviour of each stage, refreshed once a second. The instantaneous
    /// fields below are the last frame; this is what a regression actually looks
    /// like.
    pub perf: StageProfile,
    /// Time inference spent on the frame that produced this snapshot.
    pub frame_ms: f64,
    /// Time the capture stage spent reading that frame from the camera.
    pub capture_ms: f64,
    /// Wall-clock gap between the frame leaving the camera and being published.
    pub latency_ms: f64,
    pub detector_us: f64,
    pub landmarker_us: f64,
    pub frames: u64,
    /// Frames the camera produced, and the ones inference could not keep up with.
    pub captured: u64,
    pub dropped: u64,
    pub uptime_ms: u64,

    // --- context ---
    pub mode: String,
    pub sink: String,
    /// The scale the pitch is quantised to, e.g. "major pentatonic".
    pub scale: String,
    /// Root letter of that scale, e.g. "C".
    pub root: String,
    /// Timbre of the oscillator, e.g. "saw".
    pub waveform: String,
    /// The degree x octave keyboard the hand plays on.
    pub layout: GridLayout,
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
    use crate::core::NOTE_COUNT;

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
        let tau = DEFAULT_FADE_MS / 1000.0;
        let mut trail = [0.0; NOTE_COUNT];
        trail[note_index(60)] = 1.0;
        // After one time constant the note is at 1/e of its brightness...
        fade_trail(&mut trail, tau, tau);
        assert!((trail[note_index(60)] - 0.367879).abs() < 1e-3);
        // ...and after a few constants it is below the drawing threshold.
        fade_trail(&mut trail, 3.0 * tau, tau);
        assert!(trail[note_index(60)] < 0.02);
        // Untouched cells stay at zero.
        assert_eq!(trail[note_index(36)], 0.0);
    }

    #[test]
    fn a_longer_tau_keeps_the_tail_longer() {
        let mut fast = [0.0; NOTE_COUNT];
        let mut slow = [0.0; NOTE_COUNT];
        fast[note_index(48)] = 1.0;
        slow[note_index(48)] = 1.0;
        fade_trail(&mut fast, 0.2, 0.45);
        fade_trail(&mut slow, 0.2, 2.0);
        assert!(slow[note_index(48)] > fast[note_index(48)]);
        // A zero/negative tau is clamped rather than freezing the trail.
        fade_trail(&mut fast, 1.0, 0.0);
        assert!(fast[note_index(48)] < 0.02);
    }

    #[test]
    fn a_frequency_maps_to_the_nearest_key_clamped_to_the_keyboard() {
        assert_eq!(note_for_frequency(440.0), 69);
        // Half a semitone either side of A4 rounds to the closer key.
        assert_eq!(note_for_frequency(440.0 * 2.0f32.powf(0.04)), 69);
        assert_eq!(note_for_frequency(440.0 * 2.0f32.powf(0.06)), 70);
        // Out of range or nonsense input lands on the nearest playable key.
        assert_eq!(note_for_frequency(10.0), NOTE_LOW);
        assert_eq!(note_for_frequency(20000.0), NOTE_HIGH);
        assert_eq!(note_for_frequency(0.0), NOTE_LOW);
        assert_eq!(note_for_frequency(f32::NAN), NOTE_LOW);
    }
}
