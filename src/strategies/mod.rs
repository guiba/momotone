use crate::core::types::GestureEvent;
use crate::core::{HandDetectionResult, Strategy, NOTE_MARGIN};
use crate::core::grid::GridLayout;
use crate::core::scale::{preset, root_class, Scale};
use std::sync::Mutex;

/// Thumb-index spread (as a fraction of the hand box height) that drives the
/// volume to its maximum. Anything at or beyond this is full volume.
const MAX_SPREAD: f32 = 0.6;

/// Weight of the current frame in the exponential moving average of the index
/// tip position. 0.35 is a ~3 frame time constant: fast enough to follow a
/// deliberate sweep, slow enough to swallow per-frame landmark jitter.
const PITCH_EMA_ALPHA: f32 = 0.35;

/// Pitch comes from the position of the index fingertip on the degree x octave
/// grid, smoothed with an EMA so a shaky hand does not hop between cells.
#[derive(Debug)]
pub struct FingerPositionStrategy {
    smoothed_tip: Mutex<Option<(f32, f32)>>,
    grid: GridLayout,
}

impl Default for FingerPositionStrategy {
    fn default() -> Self {
        Self::new(
            root_class("C").unwrap(),
            &preset("major pentatonic").unwrap(),
            3,
        )
    }
}

impl FingerPositionStrategy {
    pub fn new(root: u8, scale: &Scale, octaves: usize) -> Self {
        Self {
            smoothed_tip: Mutex::new(None),
            grid: GridLayout::new(root, scale, octaves),
        }
    }

    pub fn layout(&self) -> &GridLayout {
        &self.grid
    }

    /// Feed one sample of the index tip position into the smoother and return the
    /// averaged position. A gap in tracking restarts the average, so the next
    /// sighting is not dragged back towards where the hand was before it left.
    fn smooth_tip(&self, x: f32, y: f32) -> (f32, f32) {
        let mut state = self.smoothed_tip.lock().unwrap();
        match state.as_mut() {
            Some((previous_x, previous_y)) => {
                *previous_x += PITCH_EMA_ALPHA * (x - *previous_x);
                *previous_y += PITCH_EMA_ALPHA * (y - *previous_y);
                (*previous_x, *previous_y)
            }
            None => {
                *state = Some((x, y));
                (x, y)
            }
        }
    }

    fn reset(&self) {
        *self.smoothed_tip.lock().unwrap() = None;
    }

    /// Normalised position of the fingertip inside the playable band, 0..1 on
    /// each axis. The outer `NOTE_MARGIN` of the frame is excluded on both axes,
    /// otherwise the extreme cells would need the hand pressed against the edge
    /// of the image where the detector loses it.
    fn band_position(&self, x: f32, y: f32, frame_size: (usize, usize)) -> (f32, f32) {
        let width = frame_size.0.max(1) as f32;
        let height = frame_size.1.max(1) as f32;
        let margin_x = NOTE_MARGIN * width;
        let margin_y = NOTE_MARGIN * height;
        let usable_x = (width - 2.0 * margin_x).max(1.0);
        let usable_y = (height - 2.0 * margin_y).max(1.0);
        (
            ((x - margin_x) / usable_x).clamp(0.0, 1.0),
            ((y - margin_y) / usable_y).clamp(0.0, 1.0),
        )
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

        // 1. Pitch from the fingertip's position on the grid: rightward is a
        // higher scale degree, upward is a higher octave.
        let (tip_x, tip_y) = self.smooth_tip(index.x, index.y);
        let (x_pos, y_pos) = self.band_position(tip_x, tip_y, detection.frame_size);
        let note = self.grid.note_at(x_pos, y_pos);

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
    use crate::core::{HandDetectionResult, Landmark, NOTE_HIGH, NOTE_LOW};

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

    /// Feed one fingertip position through one strategy instance so the EMA
    /// accumulates across calls.
    fn note_at(strategy: &FingerPositionStrategy, x: f32, y: f32) -> u8 {
        let landmarks = (0..21)
            .map(|i| Landmark {
                x: if i == 8 { x } else { 300.0 },
                y: if i == 8 { y } else { 300.0 },
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
    /// into a mapping assertion.
    fn fresh_note(scale: &str, root: u8, octaves: usize, x: f32, y: f32) -> u8 {
        note_at(&FingerPositionStrategy::new(root, &preset(scale).unwrap(), octaves), x, y)
    }

    #[test]
    fn the_grid_rises_to_the_right_and_upwards() {
        // Chromatic over four octaves covers C3..B6, the widest span that fits.
        let layout = GridLayout::new(0, &preset("chromatic").unwrap(), 4);
        assert_eq!(layout.octaves, vec![6, 5, 4, 3]); // top row is the highest
        assert_eq!(layout.notes.first().unwrap(), &72); // C6 at the top left
        assert_eq!(layout.notes.last().unwrap(), &(36 + 11)); // B3 at the bottom right
        assert_eq!(*layout.notes.iter().max().unwrap(), 83); // B6, one semitone under the ceiling

        // Corners of the playable band land on the corners of the grid.
        assert_eq!(fresh_note("chromatic", 0, 4, 64.0, 48.0), layout.notes[0]);
        assert_eq!(fresh_note("chromatic", 0, 4, 576.0, 432.0), *layout.notes.last().unwrap());
        assert!(fresh_note("chromatic", 0, 4, 576.0, 240.0) > fresh_note("chromatic", 0, 4, 64.0, 240.0));
        assert!(fresh_note("chromatic", 0, 4, 320.0, 48.0) > fresh_note("chromatic", 0, 4, 320.0, 432.0));
    }

    #[test]
    fn every_cell_of_a_quantised_grid_is_in_the_scale() {
        let strategy = FingerPositionStrategy::default(); // C major pentatonic, 3 octaves
        let allowed = strategy.layout().notes.clone();
        assert_eq!(allowed.len(), 15);
        assert_eq!(strategy.layout().rows, 3);
        assert_eq!(strategy.layout().cols, 5);

        for x in (0..=640).step_by(16) {
            for y in (0..=480).step_by(16) {
                let note = fresh_note("major pentatonic", 0, 3, x as f32, y as f32);
                assert!(allowed.contains(&note), "({x},{y}) gave off-scale {note}");
            }
        }
    }

    #[test]
    fn cells_are_coarse_enough_that_jitter_cannot_cross_one() {
        // Five columns over the 512 px band -> ~102 px each, so a few pixels of
        // landmark noise cannot cross a boundary even before smoothing.
        let samples = [196.0, 198.0, 200.0, 202.0, 204.0];
        let jittered: Vec<u8> = samples
            .iter()
            .map(|&x| fresh_note("major pentatonic", 0, 3, x, 240.0))
            .collect();
        assert!(
            jittered.windows(2).all(|pair| pair[0] == pair[1]),
            "jitter crossed a column: {jittered:?}"
        );
    }

    #[test]
    fn a_changed_root_transposes_the_keyboard() {
        let in_c = fresh_note("major pentatonic", 0, 3, 320.0, 240.0);
        let in_a = fresh_note("major pentatonic", root_class("A").unwrap(), 3, 320.0, 240.0);
        assert_ne!(in_c, in_a, "transposing must move the note under the same hand position");

        let strategy = FingerPositionStrategy::new(
            root_class("A").unwrap(),
            &preset("minor pentatonic").unwrap(),
            3,
        );
        let allowed = strategy.layout().notes.clone();
        for x in [100.0, 200.0, 300.0, 400.0, 500.0, 576.0] {
            let note = fresh_note("minor pentatonic", root_class("A").unwrap(), 3, x, 240.0);
            assert!(allowed.contains(&note), "x={x} gave off-scale {note}");
        }
    }

    #[test]
    fn the_smoothing_kills_jitter_but_follows_a_sweep() {
        // A hand that is genuinely still jitters +/-20 px around 310. On a
        // chromatic grid a column is 512 / 12 ~= 43 px wide, so the noise
        // straddles a boundary frame to frame; the average sits firmly inside one
        // cell.
        let jitter = [290.0, 330.0, 290.0, 330.0, 290.0, 330.0, 290.0, 330.0];
        let raw = GridLayout::new(0, &preset("chromatic").unwrap(), 4);
        let unsmoothed: Vec<u8> = jitter
            .iter()
            .map(|&x| raw.note_at((x - 64.0) / 512.0, 0.5))
            .collect();

        let strategy = FingerPositionStrategy::new(0, &preset("chromatic").unwrap(), 4);
        let smoothed: Vec<u8> = jitter
            .iter()
            .map(|&x| note_at(&strategy, x, 240.0))
            .collect();

        let flaps = |notes: &[u8]| {
            notes
                .windows(2)
                .filter(|pair| pair[0] != pair[1])
                .count()
        };
        assert!(
            flaps(&smoothed) < flaps(&unsmoothed),
            "unsmoothed {unsmoothed:?} flips {} times, smoothed {smoothed:?} flips {}",
            flaps(&unsmoothed),
            flaps(&smoothed)
        );
        assert!(flaps(&smoothed) <= 1, "smoothed trail {smoothed:?}");

        // A deliberate sweep still reaches the far corner of the keyboard.
        let mut x = 320.0;
        let mut y = 240.0;
        while x <= 640.0 && y >= 48.0 {
            note_at(&strategy, x, y);
            x += 20.0;
            y -= 16.0;
        }
        let corner = note_at(&strategy, 640.0, 48.0);
        assert_eq!(corner, *strategy.layout().notes.iter().max().unwrap());
    }

    #[test]
    fn losing_the_hand_resets_the_smoother() {
        let strategy = FingerPositionStrategy::default();
        note_at(&strategy, 100.0, 240.0);
        strategy.reset();
        // Without the reset the old position would still be pulling the average.
        let after = note_at(&strategy, 600.0, 60.0);
        let from_scratch = FingerPositionStrategy::default().layout().note_at(1.0, 0.0);
        assert_eq!(after, from_scratch);
    }

    #[test]
    fn the_detector_box_does_not_select_the_note() {
        // Identical fingertip, completely different bounding boxes. If the box
        // were driving the pitch these two would disagree.
        let landmarks: Vec<Landmark> = (0..21)
            .map(|i| Landmark {
                x: if i == 8 { 320.0 } else { 300.0 },
                y: if i == 8 { 240.0 } else { 300.0 },
                z: 0.0,
            })
            .collect();
        let strategy = FingerPositionStrategy::default();
        let note = strategy
            .process(&HandDetectionResult {
                hand_landmarks: Some(landmarks.clone()),
                r#box: Some((-80.0, -40.0, 700.0, 700.0)),
                frame_size: (640, 480),
                score: 0.9,
                timings: Default::default(),
            })
            .unwrap()
            .payload;
        let other = strategy
            .process(&HandDetectionResult {
                hand_landmarks: Some(landmarks),
                r#box: Some((310.0, 230.0, 20.0, 20.0)),
                frame_size: (640, 480),
                score: 0.4,
                timings: Default::default(),
            })
            .unwrap()
            .payload;
        match (note, other) {
            (GesturePayload::Note { note: first, .. }, GesturePayload::Note { note: second, .. }) => assert_eq!(first, second),
            _ => panic!("expected note payloads"),
        }
    }
}
