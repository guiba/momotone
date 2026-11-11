//! One place that decides *where the hand is*.
//!
//! The tracker hands over raw keypoints for a frame. The estimator is the only
//! thing that turns them into the state a strategy is allowed to read:
//!
//! * **presence** - debounced, because the detector's anchor score flickers
//!   around the threshold and a raw compare clicks the note on and off;
//! * **continuity** - inside the grace window the last known keypoints are reused,
//!   so a one-frame dropout is inaudible;
//! * **the tip** - the index fingertip, EMA-smoothed, which is what the pitch is
//!   read from;
//! * **the loudness** - thumb-to-index spread, saturating at `MAX_SPREAD`;
//! * **the geometry** - spread, per-finger extension, tilt, palm length.
//!
//! Before this, the pipeline, each strategy and the dashboard each carried their
//! own copy of "the current hand": the gate in the pipeline, the tip smoother in
//! every strategy, the geometry recomputed in two places. Now there is one.

use crate::core::detection_gate::DetectionGate;
use crate::core::geometry::{hand_geometry, HandGeometry};
use crate::core::{HandDetectionResult, Landmark};
use crate::gesture::{spread_volume, PITCH_EMA_ALPHA};

/// Frames of solid sighting before a gesture starts, and frames of missed
/// detections it may survive before it is released.
const CONFIRM_FRAMES: u32 = 2;
const GRACE_FRAMES: u32 = 3;

/// A hand, resolved. Everything a strategy needs, in frame pixels.
#[derive(Debug, Clone, Default)]
pub struct HandState {
    /// True when a gesture is live - either sighted this frame, or still inside
    /// the grace window.
    pub present: bool,
    /// The 21 MediaPipe landmarks, or the last ones seen while `present`.
    pub landmarks: Vec<Landmark>,
    /// `(x, y, w, h)` of the hand box, in frame pixels.
    pub hand_box: Option<(f32, f32, f32, f32)>,
    /// Size of the frame those pixels live in.
    pub frame_size: (usize, usize),
    /// Detector confidence of the frame that produced these keypoints.
    pub confidence: f32,
    /// EMA-smoothed index fingertip - the pitch anchor.
    pub tip: (f32, f32),
    /// Thumb-to-index spread, 0.0 (pinched) to 1.0 (full volume).
    pub volume: f32,
    /// Resolution-independent hand descriptors for the dashboard.
    pub geometry: HandGeometry,
    /// Mean per-landmark displacement from the previous frame, in pixels: how
    /// noisy the model is right now.
    pub jitter_px: f32,
}

/// The pipeline's single source of truth about the hand. One instance per run; it
/// is only ever touched by the frame thread, so it needs no interior locking.
pub struct HandEstimator {
    gate: DetectionGate,
    tip: Option<(f32, f32)>,
    previous_landmarks: Vec<Landmark>,
    previous_box: Option<(f32, f32, f32, f32)>,
    previous_confidence: f32,
}

impl Default for HandEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl HandEstimator {
    pub fn new() -> Self {
        Self {
            gate: DetectionGate::new(CONFIRM_FRAMES, GRACE_FRAMES),
            tip: None,
            previous_landmarks: Vec::new(),
            previous_box: None,
            previous_confidence: 0.0,
        }
    }

    /// Resolve one frame's raw detection into the state the strategies read.
    pub fn update(&mut self, detection: &HandDetectionResult) -> HandState {
        let present = self.gate.observe(detection.hand_landmarks.is_some());
        if !present {
            self.reset();
            return HandState {
                present: false,
                frame_size: detection.frame_size,
                ..HandState::default()
            };
        }

        // Either sighted this frame, or inside the grace window and carried over
        // from the last one.
        let landmarks = match &detection.hand_landmarks {
            Some(landmarks) => landmarks.clone(),
            None => self.previous_landmarks.clone(),
        };
        let hand_box = detection.r#box.or(self.previous_box);
        let confidence = if detection.hand_landmarks.is_some() {
            detection.score
        } else {
            self.previous_confidence
        };

        let jitter_px = landmark_jitter(&self.previous_landmarks, &landmarks);
        let tip = self.smooth_tip(landmarks.get(8));
        let geometry = hand_geometry(&landmarks).unwrap_or_default();
        let volume = spread_volume(&landmarks, hand_box);

        self.previous_landmarks.clone_from(&landmarks);
        self.previous_box = hand_box;
        self.previous_confidence = confidence;

        HandState {
            present: true,
            landmarks,
            hand_box,
            frame_size: detection.frame_size,
            confidence,
            tip,
            volume,
            geometry,
            jitter_px,
        }
    }

    /// A gesture that ends (or never starts) must not leave the smoother holding
    /// a stale position, or the next sighting is dragged back to where the hand
    /// was before it left.
    fn reset(&mut self) {
        self.tip = None;
        self.previous_landmarks.clear();
        self.previous_box = None;
        self.previous_confidence = 0.0;
    }

    /// One EMA step on the index fingertip. `alpha = 0.35` is a ~3 frame time
    /// constant: fast enough to follow a deliberate sweep, slow enough to swallow
    /// per-frame landmark jitter.
    fn smooth_tip(&mut self, tip: Option<&Landmark>) -> (f32, f32) {
        let Some(tip) = tip else {
            self.tip = None;
            return (0.0, 0.0);
        };
        let smoothed = match self.tip {
            Some((previous_x, previous_y)) => (
                previous_x + PITCH_EMA_ALPHA * (tip.x - previous_x),
                previous_y + PITCH_EMA_ALPHA * (tip.y - previous_y),
            ),
            None => (tip.x, tip.y),
        };
        self.tip = Some(smoothed);
        smoothed
    }
}

/// Mean per-landmark displacement between two frames: a direct read of how noisy
/// the model is, independent of the smoothing that hides it from the pitch.
fn landmark_jitter(previous: &[Landmark], current: &[Landmark]) -> f32 {
    if previous.len() != current.len() || current.is_empty() {
        return 0.0;
    }
    let total: f32 = previous
        .iter()
        .zip(current)
        .map(|(a, b)| ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt())
        .sum();
    total / current.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn landmark(x: f32, y: f32) -> Landmark {
        Landmark { x, y, z: 0.0 }
    }

    /// A 21-point hand whose index tip (8) is at `(tip_x, tip_y)`, thumb tip (4)
    /// at `thumb_x`, and whose wrist sits at `(300, 400)` so the palm length is
    /// well defined.
    fn hand(tip_x: f32, tip_y: f32, thumb_x: f32) -> HandDetectionResult {
        let landmarks = (0..21)
            .map(|i| match i {
                0 => landmark(300.0, 420.0),                  // wrist
                4 => landmark(thumb_x, tip_y),                // thumb tip
                8 => landmark(tip_x, tip_y),                  // index tip
                9 => landmark(300.0, 340.0), // middle MCP - the palm length reference
                _ => landmark(300.0 + i as f32 * 4.0, 340.0), // the other knuckles
            })
            .collect();
        HandDetectionResult {
            hand_landmarks: Some(landmarks),
            r#box: Some((200.0, 140.0, 200.0, 200.0)),
            frame_size: (640, 480),
            score: 0.9,
            timings: Default::default(),
        }
    }

    fn empty() -> HandDetectionResult {
        HandDetectionResult::default()
    }

    #[test]
    /// A single blip is not a gesture; two sightings in a row are.
    fn a_blip_is_not_a_hand_but_two_sightings_are() {
        let mut estimator = HandEstimator::new();
        assert!(!estimator.update(&hand(300.0, 240.0, 440.0)).present);
        assert!(estimator.update(&hand(300.0, 240.0, 440.0)).present);
    }

    #[test]
    /// Inside the grace window the keypoints survive, so a dropout does not make
    /// the synthesizer stutter; after it the hand is gone.
    fn the_grace_window_carries_the_last_keypoints() {
        let mut estimator = HandEstimator::new();
        estimator.update(&hand(300.0, 240.0, 440.0));
        let live = estimator.update(&hand(300.0, 240.0, 440.0));
        assert!(live.present && live.landmarks.len() == 21);

        let carried = estimator.update(&empty());
        assert!(carried.present, "one missed frame is still a gesture");
        assert_eq!(carried.landmarks.len(), 21, "carried over, not cleared");

        let gone = estimator.update(&empty());
        assert!(gone.present, "still inside the three frame grace window");

        estimator.update(&empty());
        estimator.update(&empty());
        let released = estimator.update(&empty());
        assert!(!released.present, "the grace window ran out");
        assert!(released.landmarks.is_empty());
    }

    #[test]
    /// The tip is smoothed: a still hand that jitters gives a far steadier
    /// average than the raw samples, and a deliberate sweep still arrives.
    fn smoothing_steadies_the_tip_but_still_follows_a_sweep() {
        let mut estimator = HandEstimator::new();
        estimator.update(&hand(300.0, 240.0, 440.0));
        estimator.update(&hand(300.0, 240.0, 440.0));

        let jitter = [280.0, 320.0, 280.0, 320.0, 280.0, 320.0, 280.0, 320.0];
        let smoothed: Vec<f32> = jitter
            .iter()
            .map(|&x| estimator.update(&hand(x, 240.0, 440.0)).tip.0)
            .collect();
        let spread_of = |values: &[f32]| {
            values.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
                - values.iter().min_by(|a, b| a.total_cmp(b)).unwrap()
        };
        assert!(
            spread_of(&smoothed) < spread_of(&jitter),
            "raw {jitter:?} swings {}, smoothed {smoothed:?} swings {}",
            spread_of(&jitter),
            spread_of(&smoothed)
        );

        // A real sweep across the frame gets there, lag and all.
        let mut x = 320.0;
        while x <= 640.0 {
            estimator.update(&hand(x, 240.0, 440.0));
            x += 20.0;
        }
        let tip = estimator.update(&hand(640.0, 240.0, 440.0)).tip.0;
        assert!(
            tip > 500.0,
            "the smoother never caught up with the sweep: {tip}"
        );
    }

    #[test]
    /// Losing the hand resets the smoother, so the next sighting is not dragged
    /// back towards where the hand was before it left.
    fn losing_the_hand_resets_the_smoother() {
        let mut estimator = HandEstimator::new();
        estimator.update(&hand(100.0, 240.0, 440.0));
        estimator.update(&hand(100.0, 240.0, 440.0));
        estimator.update(&empty());
        estimator.update(&empty());
        estimator.update(&empty());
        estimator.update(&empty());
        estimator.update(&hand(600.0, 60.0, 440.0));
        let fresh = estimator.update(&hand(600.0, 60.0, 440.0));
        let tip_x = fresh.tip.0;
        assert!((tip_x - 600.0).abs() < 1e-6, "tip was {tip_x}");
    }

    #[test]
    fn volume_and_geometry_are_reported_once() {
        let mut estimator = HandEstimator::new();
        estimator.update(&hand(300.0, 240.0, 440.0));
        let pinched = estimator.update(&hand(300.0, 240.0, 302.0)).volume;
        let wide = estimator.update(&hand(300.0, 240.0, 420.0)).volume;
        assert!(pinched < wide, "spread must raise the volume");

        let state = estimator.update(&hand(300.0, 240.0, 420.0));
        assert!(state.geometry.palm_length > 0.0);
        assert!(state.geometry.extension[1] > 0.0);
        assert!(state.jitter_px >= 0.0);
    }
}
