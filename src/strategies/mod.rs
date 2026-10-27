use crate::core::types::GestureEvent;
use crate::core::{HandDetectionResult, Strategy, NOTE_MARGIN};
use crate::core::scale::{playable_notes, preset, root_class, Scale};
use std::sync::Mutex;

/// Thumb-index spread (as a fraction of the hand box height) that drives the
/// volume to its maximum. Anything at or beyond this is full volume.
const MAX_SPREAD: f32 = 0.6;

/// Weight of the current frame in the exponential moving average of the index
/// tip position. 0.35 is a ~3 frame time constant: fast enough to follow a
/// deliberate sweep, slow enough to swallow per-frame landmark jitter.
const PITCH_EMA_ALPHA: f32 = 0.35;

/// Pitch comes from the *horizontal* position of the index fingertip, smoothed
/// with an EMA so a shaky hand does not hop between notes, and quantised to a
/// scale so every reachable cell is consonant with every other one.
#[derive(Debug)]
pub struct FingerPositionStrategy {
    smoothed_index_x: Mutex<Option<f32>>,
    /// The notes the chosen scale makes playable, ascending.
    playable: Vec<u8>,
}

impl Default for FingerPositionStrategy {
    fn default() -> Self {
        Self::new(root_class("C").unwrap(), &preset("major pentatonic").unwrap())
    }
}

impl FingerPositionStrategy {
    pub fn new(root: u8, scale: &Scale) -> Self {
        let playable = playable_notes(root, scale);
        Self {
            smoothed_index_x: Mutex::new(None),
            // A degenerate scale (empty degrees) would index out of bounds; fall
            // back to the full chromatic range so the app still plays.
            playable: if playable.is_empty() {
                (crate::core::NOTE_LOW..=crate::core::NOTE_HIGH).collect()
            } else {
                playable
            },
        }
    }

    pub fn scale_notes(&self) -> &[u8] {
        &self.playable
    }

    /// Map a normalised position (0 = left edge of the playable band, 1 = right)
    /// onto the scale. With five columns the boundaries are ~100 px apart on a
    /// 640 px frame, so landmark jitter can never cross one.
    fn note_for_position(&self, x_pos: f32) -> u8 {
        let t = x_pos.clamp(0.0, 1.0);
        let index = (t * self.playable.len() as f32) as usize;
        self.playable[index.min(self.playable.len() - 1)]
    }
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
        let note = self.note_for_position(x_pos);

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
    use crate::core::{HandDetectionResult, Landmark, NOTE_HIGH};

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

    /// Feed one index-tip position through one strategy instance so the EMA
    /// accumulates across calls.
    fn note_at(strategy: &FingerPositionStrategy, index_x: f32) -> u8 {
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

    /// A fresh strategy per call, so the EMA never carries a previous position
    /// into a mapping assertion. `note_at` is for testing accumulated smoothing.
    fn fresh_note(scale: &str, root: u8, index_x: f32) -> u8 {
        note_at(
            &FingerPositionStrategy::new(root, &preset(scale).unwrap()),
            index_x,
        )
    }

    #[test]
    fn pitch_follows_the_horizontal_position() {
        // Chromatic, so the band edges land exactly on C2 and C6.
        assert_eq!(fresh_note("chromatic", 0, 64.0), 36);
        assert_eq!(fresh_note("chromatic", 0, 320.0), 60);
        assert_eq!(fresh_note("chromatic", 0, 576.0), 84);
        // Off the frame the note clamps instead of running past the range.
        assert_eq!(fresh_note("chromatic", 0, 0.0), 36);
        assert_eq!(fresh_note("chromatic", 0, 640.0), 84);
        // Sweeping right is strictly rising.
        assert!(fresh_note("chromatic", 0, 100.0) < fresh_note("chromatic", 0, 250.0)
            && fresh_note("chromatic", 0, 250.0) < fresh_note("chromatic", 0, 500.0));
    }

    #[test]
    fn quantising_to_a_scale_only_ever_reaches_scale_notes() {
        let allowed = FingerPositionStrategy::default().scale_notes().to_vec(); // C major pentatonic
        assert_eq!(allowed.len(), 21);

        // Walk the whole band; every note produced must be in the scale.
        let mut x = 0.0;
        while x <= 640.0 {
            let note = fresh_note("major pentatonic", 0, x);
            assert!(allowed.contains(&note), "x={x} produced off-scale note {note}");
            x += 4.0;
        }
        // The band edges hit the extremes of the scale, and columns are ordered.
        assert_eq!(fresh_note("major pentatonic", 0, 64.0), 36); // C2
        assert_eq!(fresh_note("major pentatonic", 0, 576.0), 84); // C6
        assert!(fresh_note("major pentatonic", 0, 100.0) <= fresh_note("major pentatonic", 0, 400.0));
    }

    #[test]
    fn scale_columns_are_coarse_enough_that_jitter_cannot_cross_one() {
        // Five columns over the 512 px band -> ~102 px each, so a few pixels of
        // landmark noise cannot cross a boundary even before smoothing.
        let samples = [196.0, 198.0, 200.0, 202.0, 204.0];
        let jittered: Vec<u8> = samples
            .iter()
            .map(|&x| fresh_note("major pentatonic", 0, x))
            .collect();
        assert!(
            jittered.windows(2).all(|pair| pair[0] == pair[1]),
            "jitter crossed a column: {jittered:?}"
        );
    }

    #[test]
    fn a_changed_root_transposes_the_keyboard() {
        let in_c = fresh_note("major pentatonic", 0, 320.0);
        let in_a = fresh_note("major pentatonic", root_class("A").unwrap(), 320.0);
        assert_ne!(in_c, in_a, "transposing must move the note under the same hand position");

        let allowed = FingerPositionStrategy::new(
            root_class("A").unwrap(),
            &preset("minor pentatonic").unwrap(),
        )
        .scale_notes()
        .to_vec();
        for x in [100.0, 200.0, 300.0, 400.0, 500.0, 576.0] {
            let note = fresh_note("minor pentatonic", root_class("A").unwrap(), x);
            assert!(allowed.contains(&note), "x={x} gave off-scale {note}");
        }
    }

    #[test]
    fn the_smoothing_kills_jitter_but_follows_a_sweep() {
        // A hand that is genuinely still jitters a few pixels around 314 px.
        // Quantising each frame on its own crosses the 58/59/60 boundary; the
        // average of the same samples sits firmly inside one key. (A chromatic
        // key is 640 * 0.8 / 48 ~= 10.7 px wide.)
        let jitter = [308.0, 320.0, 308.0, 320.0, 308.0, 320.0, 308.0, 320.0];
        let chromatic = FingerPositionStrategy::new(0, &preset("chromatic").unwrap());
        let raw: Vec<u8> = jitter.iter().map(|&x| note_at(&chromatic, x)).collect();

        let strategy = FingerPositionStrategy::default();
        let smoothed: Vec<u8> = jitter
            .iter()
            .map(|&x| note_at(&strategy, x))
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
            note_at(&strategy, x);
            x += 20.0;
        }
        assert_eq!(note_at(&strategy, 640.0), NOTE_HIGH);
    }

    #[test]
    fn losing_the_hand_resets_the_smoother() {
        let strategy = FingerPositionStrategy::default();
        note_at(&strategy, 100.0);
        strategy.reset();
        // Without the reset the old position would still be pulling the average.
        assert_eq!(note_at(&strategy, 640.0), NOTE_HIGH);
    }

    #[test]
    fn the_detector_box_does_not_select_the_note() {
        // Identical fingertip, completely different bounding boxes. If the box
        // were driving the pitch these two would disagree.
        let chromatic = FingerPositionStrategy::new(0, &preset("chromatic").unwrap());
        assert_eq!(note_at(&chromatic, 320.0), 60);
    }
}
