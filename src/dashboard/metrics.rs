//! The single struct the vision pipeline publishes for the dashboard.
//!
//! The pipeline thread overwrites it under a mutex; the iced app clones it on
//! every UI tick. Keeping it one flat struct means the UI never has to reach
//! into the tracker or the sinks.

use std::collections::VecDeque;
use std::time::Instant;

use crate::core::geometry::HandGeometry;
use crate::core::grid::GridLayout;
use crate::core::telemetry::StageProfile;
use crate::core::{Landmark, NOTE_COUNT, NOTE_HIGH, NOTE_LOW};
use crate::presentation::ascii::AsciiGrid;

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

/// The keyboard's fading glow. Owned by the frame loop, published in the
/// snapshot, and drawn by the stage without any UI-side state of its own.
#[derive(Debug, Clone)]
pub struct KeyboardTrail {
    /// Decay time constant in seconds (`--fade-ms`).
    tau_secs: f32,
    /// Wall clock of the last decay step.
    last: Instant,
    /// Per-semitone brightness, index = note - NOTE_LOW.
    levels: Vec<f32>,
}

impl Default for KeyboardTrail {
    fn default() -> Self {
        KeyboardTrail {
            tau_secs: DEFAULT_FADE_MS / 1000.0,
            last: Instant::now(),
            levels: vec![0.0; NOTE_COUNT],
        }
    }
}

impl KeyboardTrail {
    pub fn new(fade_ms: f32) -> Self {
        KeyboardTrail {
            tau_secs: fade_ms.max(1.0) / 1000.0,
            ..KeyboardTrail::default()
        }
    }

    /// Fade everything by the time since the last frame, then light the key that
    /// is being played. Called once per frame by the pipeline, so the trail is a
    /// fact about the run rather than a fact about the window.
    pub fn advance(&mut self, now: Instant, playing: Option<u8>) {
        // `Instant` cannot go backwards: a clock jump or a re-used trail would
        // otherwise hand `exp()` a negative age and brighten every key.
        let dt = if now > self.last {
            now.duration_since(self.last).as_secs_f32()
        } else {
            0.0
        };
        self.last = now;
        fade_trail(&mut self.levels, dt, self.tau_secs);
        if let Some(note) = playing {
            self.levels[note_index(note)] = 1.0;
        }
    }

    /// Go silent: a stopped player has no key glowing.
    pub fn clear(&mut self) {
        self.levels.fill(0.0);
    }

    pub fn levels(&self) -> &[f32] {
        &self.levels
    }

    pub fn fade_ms(&self) -> f32 {
        self.tau_secs * 1000.0
    }
}

#[derive(Debug, Clone)]
pub struct EventLogEntry {
    pub uptime_ms: u64,
    pub detail: String,
}

/// Where the hand is and how steady it looks. Written by the frame loop out of the
/// estimator's `HandState`.
#[derive(Debug, Clone, Default)]
pub struct HandView {
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
    /// Resolution-independent hand shape: spread, per-finger extension, tilt, palm.
    pub geometry: HandGeometry,
}

/// What is sounding, including the keyboard glow. Owned by the frame loop.
#[derive(Debug, Clone, Default)]
pub struct SoundView {
    pub note: Option<u8>,
    pub volume: f32,
    pub frequency: Option<f32>,
    /// How close the hand looks, 0.0 (far) .. 1.0 (at the lens), regardless of which
    /// model is currently driving `volume`.
    pub proximity: f32,
    /// The raw apparent size behind `proximity`: palm lengths per frame side.
    pub proximity_ratio: f32,
    /// -1.0 (backing away) .. 1.0 (charging in), smoothed.
    pub approach: f32,
    /// Per-semitone keyboard brightness, index = note - NOTE_LOW.
    pub trail: Vec<f32>,
    /// The trail's decay time constant, in milliseconds, for the header readout.
    pub fade_ms: f32,
}

/// How fast the run is going: instantaneous numbers plus the tail of each stage,
/// refreshed once a second.
#[derive(Debug, Clone, Default)]
pub struct PerfView {
    pub fps: f32,
    pub profile: StageProfile,
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
}

/// The picture the stage draws. Only the presenter thread writes the pixel and
/// ASCII buffers, so refreshing the telemetry never wipes them.
#[derive(Debug, Clone, Default)]
pub struct SceneView {
    /// The camera frame as RGBA pixels, handed to the dashboard so it can draw the
    /// video and the skeleton in the same canvas. Only produced when the UI is
    /// running.
    pub pixels: Vec<u8>,
    /// Quantised luminance grid behind the skeleton: the ASCII art backdrop.
    /// Published only while the `ascii` toggle is on (or a dump/terminal output is
    /// requested), at `--ascii-fps`.
    pub ascii: AsciiGrid,
    /// The degree x octave keyboard the hand plays on.
    pub layout: GridLayout,
}

/// What the run *is*: mode, output, instrument, and the event log.
#[derive(Debug, Clone, Default)]
pub struct StatusView {
    /// True when the dashboard's Stop button has paused the pipeline.
    pub paused: bool,
    pub mode: String,
    pub sink: String,
    /// The scale the pitch is quantised to, e.g. "major pentatonic".
    pub scale: String,
    /// Root letter of that scale, e.g. "C".
    pub root: String,
    /// Timbre of the oscillator, e.g. "saw".
    pub waveform: String,
    /// Which signal is driving loudness this run: "proximity", "pinch" or "blend".
    pub volume_model: String,
    /// The picture subsystems as the pipeline actually ran them. The dashboard used
    /// to mirror these in its own fields; the budget guard can turn them off without
    /// a click, so a mirror would lie. One writer (the frame loop), one reader (the UI).
    pub feed: bool,
    pub ascii: bool,
    pub log: VecDeque<EventLogEntry>,
}

/// The whole snapshot. Each field is a view with one owner, so the UI reads a
/// named group instead of digging through a flat bag of thirty-odd fields.
#[derive(Debug, Clone, Default)]
pub struct FrameMetrics {
    pub hand: HandView,
    pub sound: SoundView,
    pub perf: PerfView,
    pub scene: SceneView,
    pub status: StatusView,
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
    use std::time::Duration;

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
    fn a_held_key_stays_lit_while_the_trail_behind_it_fades() {
        let mut trail = KeyboardTrail::new(450.0);
        let start = Instant::now();
        trail.advance(start, Some(60));
        // A key that is still being played is always at full brightness...
        assert_eq!(trail.levels()[note_index(60)], 1.0);
        // ...but the note it replaced decays away over its own time constant.
        trail.advance(start + Duration::from_millis(450), None);
        let held_again = trail.levels()[note_index(60)];
        assert!((held_again - 0.367879).abs() < 1e-3);
        trail.clear();
        assert!(trail.levels().iter().all(|&level| level == 0.0));
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
