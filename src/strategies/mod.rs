use crate::core::types::GestureEvent;
use crate::core::{HandDetectionResult, Strategy, NOTE_HIGH, NOTE_LOW, NOTE_MARGIN};
use std::sync::Mutex;

pub struct SimplePinchStrategy {
    pub threshold: f32,
}

impl Default for SimplePinchStrategy {
    fn default() -> Self {
        Self { threshold: 0.05 }
    }
}

impl Strategy for SimplePinchStrategy {
    fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent> {
        let landmarks = detection.hand_landmarks.as_ref()?;
        if landmarks.len() < 8 {
            return None;
        }

        let thumb = &landmarks[4];
        let index = &landmarks[8];

        let dx = thumb.x - index.x;
        let dy = thumb.y - index.y;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < self.threshold {
            return Some(GestureEvent::new_note(60, 1.0, "Pinch Detected"));
        }

        None
    }
}

pub struct FrequencyStrategy;

impl Strategy for FrequencyStrategy {
    fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent> {
        let landmarks = detection.hand_landmarks.as_ref()?;
        if landmarks.is_empty() {
            return None;
        }

        // Map the Y position of the first landmark to frequency (approx 200Hz - 1000Hz)
        // Note: In OpenCV, y=0 is top, y=1 is bottom.
        let y = landmarks[0].y;
        let freq = 200.0 + (1.0 - y as f32) * 800.0;

        Some(GestureEvent::new_signal(freq, 0.5, "Y-Position Signal"))
    }
}

/// Thumb-index spread (as a fraction of the hand box height) that drives the
/// volume to its maximum. Anything at or beyond this is full volume.
const MAX_SPREAD: f32 = 0.6;

/// Weight of the current frame in the exponential moving average of the index
/// tip position. 0.35 is a ~3 frame time constant: fast enough to follow a
/// deliberate sweep, slow enough to swallow per-frame landmark jitter.
const PITCH_EMA_ALPHA: f32 = 0.35;

/// Pitch comes from the *horizontal* position of the index fingertip, smoothed
/// with an EMA so a shaky hand does not hop between semitones. The state lives
/// in the strategy, so a hand that leaves the view resets it.
pub struct FingerPositionStrategy {
    smoothed_index_x: Mutex<Option<f32>>,
}

impl Default for FingerPositionStrategy {
    fn default() -> Self {
        Self {
            smoothed_index_x: Mutex::new(None),
        }
    }
}

impl FingerPositionStrategy {
    /// Feed one sample of the index tip x into the smoother and return the
    /// averaged position. A gap in tracking restarts the average, so the next
    /// sighting is not dragged back towards where the hand was before it left.
    fn smooth_index_x(&self, x: f32) -> f32 {
        let mut state = self.smoothed_index_x.lock().unwrap();
        match state.as_mut() {
            Some(previous) => {
                *previous += PITCH_EMA_ALPHA * (x - *previous);
                *previous
            }
            None => {
                *state = Some(x);
                x
            }
        }
    }

    fn reset(&self) {
        *self.smoothed_index_x.lock().unwrap() = None;
    }
}

impl Strategy for FingerPositionStrategy {
    fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent> {
        let landmarks = detection.hand_landmarks.as_ref()?;
        if landmarks.len() < 9 {
            self.reset();
            return None;
        }

        // MediaPipe indices: Thumb tip (4), Index finger tip (8)
        let thumb = &landmarks[4];
        let index = &landmarks[8];

        // 1. Pitch from the *horizontal* position of the index tip: sweeping the
        // hand across the view walks the keyboard left (low) to right (high).
        // The outer NOTE_MARGIN of the frame is not playable, otherwise the
        // extreme notes would need the hand at the very edge of the image.
        let width = detection.frame_size.0.max(1) as f32;
        let margin = NOTE_MARGIN * width;
        let usable = (width - 2.0 * margin).max(1.0);
        let x_pos = (self.smooth_index_x(index.x) - margin) / usable;
        let note_f = NOTE_LOW as f32 + x_pos.clamp(0.0, 1.0) * (NOTE_HIGH - NOTE_LOW) as f32;
        let note = note_f.clamp(NOTE_LOW as f32, NOTE_HIGH as f32) as u8;

        // 2. Map Thumb <-> Index distance to Velocity / volume (0.0 to 1.0):
        // fingers far apart -> loud, pinched together -> silent. Normalised
        // against the hand box so the mapping is resolution independent.
        let (_, _, _, bh) = detection.r#box.unwrap_or((0.0, 0.0, 1.0, 1.0));
        let dx = thumb.x - index.x;
        let dy = thumb.y - index.y;
        let distance = (dx * dx + dy * dy).sqrt() / bh.max(1.0);

        let velocity = (distance / MAX_SPREAD).clamp(0.0, 1.0);

        Some(GestureEvent::new_note(note, velocity, "Finger Position"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::GesturePayload;
    use crate::core::{HandDetectionResult, Landmark};

    /// A detection with a 200x200 hand box and a fixed landmark layout; only the
    /// thumb tip x coordinate changes, i.e. the thumb/index spread.
    fn detection_with_spread(thumb_x: f32) -> HandDetectionResult {
        let landmarks = (0..21)
            .map(|i| Landmark {
                x: if i == 4 { thumb_x } else { 1100.0 },
                y: 600.0,
                z: 0.0,
            })
            .collect();
        HandDetectionResult {
            hand_landmarks: Some(landmarks),
            r#box: Some((1000.0, 500.0, 200.0, 200.0)),
            frame_size: (1280, 1080),
            score: 0.9,
            timings: Default::default(),
        }
    }

    fn velocity(thumb_x: f32) -> f32 {
        let event = FingerPositionStrategy::default()
            .process(&detection_with_spread(thumb_x))
            .unwrap();
        match event.payload {
            GesturePayload::Note { velocity, .. } => velocity,
            _ => panic!("expected a note payload"),
        }
    }

    #[test]
    fn volume_grows_with_thumb_index_distance() {
        // Hand box height is 200 px, so spread/200 is the normalised distance and
        // MAX_SPREAD (0.6 -> 120 px) is where the volume saturates.
        let pinched = velocity(1102.0); //   2 px apart
        let mid = velocity(1160.0); //    60 px apart
        let wide = velocity(1220.0); //  120 px apart
        assert!(
            pinched < mid && mid < wide,
            "volume must rise with distance: {pinched} < {mid} < {wide}"
        );
        assert!((pinched - 2.0 / 200.0 / 0.6).abs() < 1e-6);
        assert!((mid - 0.5).abs() < 1e-6);
        assert!(
            (wide - 1.0).abs() < 1e-6,
            "beyond MAX_SPREAD saturates at 1.0"
        );
    }

    #[test]
    fn no_gesture_without_landmarks() {
        let empty = HandDetectionResult {
            hand_landmarks: None,
            r#box: Some((0.0, 0.0, 10.0, 10.0)),
            frame_size: (640, 480),
            score: 0.1,
            timings: Default::default(),
        };
        assert!(FingerPositionStrategy::default().process(&empty).is_none());
    }

    /// The note produced when the index tip (landmark 8) sits at `index_x` in a
    /// 640 px wide frame, with `hand_box` as whatever the detector reported.
    /// The playable band is inset by NOTE_MARGIN on both sides, so 64 px .. 576
    /// px maps onto C2 .. C6.
    fn note_at(index_x: f32, hand_box: (f32, f32, f32, f32)) -> u8 {
        let landmarks = (0..21)
            .map(|i| Landmark {
                x: if i == 8 { index_x } else { 300.0 },
                y: 300.0,
                z: 0.0,
            })
            .collect();
        let event = FingerPositionStrategy::default()
            .process(&HandDetectionResult {
                hand_landmarks: Some(landmarks),
                r#box: Some(hand_box),
                frame_size: (640, 480),
                score: 0.9,
                timings: Default::default(),
            })
            .unwrap();
        match event.payload {
            GesturePayload::Note { note, .. } => note,
            _ => panic!("expected a note payload"),
        }
    }

    /// Same as `note_at`, but reusing one strategy instance so the EMA accumulates
    /// across calls.
    fn note_at_with(strategy: &FingerPositionStrategy, index_x: f32) -> u8 {
        let landmarks = (0..21)
            .map(|i| Landmark {
                x: if i == 8 { index_x } else { 300.0 },
                y: 300.0,
                z: 0.0,
            })
            .collect();
        let event = strategy
            .process(&HandDetectionResult {
                hand_landmarks: Some(landmarks),
                r#box: Some((200.0, 200.0, 200.0, 200.0)),
                frame_size: (640, 480),
                score: 0.9,
                timings: Default::default(),
            })
            .unwrap();
        match event.payload {
            GesturePayload::Note { note, .. } => note,
            _ => panic!("expected a note payload"),
        }
    }

    #[test]
    fn pitch_follows_the_horizontal_position() {
        let palm = (200.0, 200.0, 200.0, 200.0);
        // The edges of the inset band are exactly C2 and C6, the middle is C4.
        assert_eq!(note_at(64.0, palm), 36);
        assert_eq!(note_at(320.0, palm), 60);
        assert_eq!(note_at(576.0, palm), 84);
        // Off the frame the note clamps instead of running past the range.
        assert_eq!(note_at(0.0, palm), 36);
        assert_eq!(note_at(640.0, palm), 84);
        // Sweeping right is strictly rising.
        assert!(note_at(100.0, palm) < note_at(250.0, palm) && note_at(250.0, palm) < note_at(500.0, palm));
    }

    #[test]
    fn the_smoothing_kills_jitter_but_follows_a_sweep() {
        // A hand that is genuinely still jitters a few pixels around 314 px.
        // Quantising each frame on its own crosses the 58/59/60 boundary; the
        // average of the same samples sits firmly inside one key. (A key is
        // 640 * 0.8 / 48 ~= 10.7 px wide.)
        let jitter = [308.0, 320.0, 308.0, 320.0, 308.0, 320.0, 308.0, 320.0];
        let palm = (200.0, 200.0, 200.0, 200.0);
        let raw: Vec<u8> = jitter.iter().map(|&x| note_at(x, palm)).collect();

        let strategy = FingerPositionStrategy::default();
        let smoothed: Vec<u8> = jitter
            .iter()
            .map(|&x| note_at_with(&strategy, x))
            .collect();

        let flaps = |notes: &[u8]| {
            notes
                .windows(2)
                .filter(|pair| pair[0] != pair[1])
                .count()
        };
        assert!(
            flaps(&smoothed) < flaps(&raw),
            "raw {raw:?} flips {} times, smoothed {smoothed:?} flips {}",
            flaps(&raw),
            flaps(&smoothed)
        );
        assert!(flaps(&smoothed) <= 1, "smoothed trail {smoothed:?}");

        // A deliberate sweep still reaches the far end of the keyboard.
        let mut x = 320.0;
        while x <= 600.0 {
            note_at_with(&strategy, x);
            x += 20.0;
        }
        assert_eq!(note_at_with(&strategy, 640.0), NOTE_HIGH);
    }

    #[test]
    fn losing_the_hand_resets_the_smoother() {
        let strategy = FingerPositionStrategy::default();
        note_at_with(&strategy, 100.0);
        strategy.reset();
        // Without the reset the old position would still be pulling the average.
        assert_eq!(note_at_with(&strategy, 640.0), NOTE_HIGH);
    }

    #[test]
    fn the_detector_box_does_not_select_the_note() {
        // Identical fingertip, completely different bounding boxes. If the box
        // were driving the pitch these two would disagree.
        assert_eq!(note_at(320.0, (200.0, 200.0, 200.0, 200.0)), note_at(320.0, (0.0, 0.0, 640.0, 480.0)));
        assert_eq!(note_at(320.0, (600.0, 100.0, 40.0, 40.0)), 60);
    }
}
