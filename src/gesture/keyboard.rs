//! The keyboard: the index fingertip's position on a degree x octave grid selects
//! the note, the thumb-index spread selects the loudness.
//!
//! Pitch and volume come from the `HandState` the estimator produced, so this file
//! is purely a mapping - no smoothing, no geometry, no thresholds. That is what
//! makes the two strategies comparable: they read the same state and disagree only
//! about how to read it.

use crate::core::grid::GridLayout;
use crate::core::scale::{preset, root_class, Scale};
use crate::core::types::Performance;
use crate::gesture::band_position;
use crate::gesture::estimator::HandState;
use crate::gesture::Strategy;

#[derive(Debug)]
pub struct FingerPositionStrategy {
    grid: GridLayout,
}

impl FingerPositionStrategy {
    pub fn new(root: u8, scale: &Scale, octaves: usize) -> Self {
        Self {
            grid: GridLayout::new(root, scale, octaves),
        }
    }

    /// Live re-keying: the dashboard pickers change the scale/root/octave span and
    /// the pipeline pushes the new grid in here. Nothing else has to change - the
    /// estimator tracks a fingertip, not a note, so re-keying mid-sweep does not
    /// disturb the smoothing.
    fn set_grid(&mut self, layout: GridLayout) {
        self.grid = layout;
    }
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

impl Strategy for FingerPositionStrategy {
    fn apply_layout(&mut self, layout: GridLayout) {
        self.set_grid(layout);
    }

    fn process(&mut self, state: &HandState) -> Option<Performance> {
        if !state.present {
            return None;
        }

        // 1. Pitch from the fingertip's position on the grid: rightward is a
        // higher scale degree, upward is a higher octave.
        let (x_pos, y_pos) = band_position(state.tip.0, state.tip.1, state.frame_size);
        let note = self.grid.note_at(x_pos, y_pos);

        // 2. Thumb <-> index distance to velocity / volume (0.0 to 1.0): fingers
        // far apart -> loud, pinched together -> silent.
        Some(Performance::Note {
            note,
            velocity: state.volume,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Landmark;

    /// A `HandState` with the index tip at `(x, y)` in a 640x480 frame. The
    /// estimator owns the smoothing and the volume, so tests that are only about
    /// the grid hand the state over directly.
    fn state(x: f32, y: f32) -> HandState {
        HandState {
            present: true,
            landmarks: (0..21)
                .map(|i| Landmark {
                    x: if i == 8 { x } else { 300.0 },
                    y: if i == 8 { y } else { 300.0 },
                    z: 0.0,
                })
                .collect(),
            hand_box: Some((200.0, 200.0, 200.0, 200.0)),
            frame_size: (640, 480),
            confidence: 0.9,
            tip: (x, y),
            volume: 0.5,
            proximity: 0.5,
            proximity_ratio: 0.12,
            approach: 0.0,
            geometry: Default::default(),
            jitter_px: 0.0,
        }
    }

    fn note_at(strategy: &mut FingerPositionStrategy, x: f32, y: f32) -> u8 {
        match strategy.process(&state(x, y)).unwrap() {
            Performance::Note { note, .. } => note,
            _ => panic!("expected a note performance"),
        }
    }

    /// A fresh strategy per point check, so nothing carries between assertions.
    fn fresh_note(scale: &str, root: u8, octaves: usize, x: f32, y: f32) -> u8 {
        let mut strategy = FingerPositionStrategy::new(root, &preset(scale).unwrap(), octaves);
        note_at(&mut strategy, x, y)
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
        assert_eq!(
            fresh_note("chromatic", 0, 4, 576.0, 432.0),
            *layout.notes.last().unwrap()
        );
        assert!(
            fresh_note("chromatic", 0, 4, 576.0, 240.0)
                > fresh_note("chromatic", 0, 4, 64.0, 240.0)
        );
        assert!(
            fresh_note("chromatic", 0, 4, 320.0, 48.0)
                > fresh_note("chromatic", 0, 4, 320.0, 432.0)
        );
    }

    #[test]
    fn every_cell_of_a_quantised_grid_is_in_the_scale() {
        let layout = GridLayout::new(0, &preset("major pentatonic").unwrap(), 3);
        assert_eq!(layout.notes.len(), 15);
        assert_eq!(layout.rows, 3);
        assert_eq!(layout.cols, 5);

        for x in (0..=640).step_by(16) {
            for y in (0..=480).step_by(16) {
                let note = fresh_note("major pentatonic", 0, 3, x as f32, y as f32);
                assert!(
                    layout.notes.contains(&note),
                    "({x},{y}) gave off-scale {note}"
                );
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
        let in_a = fresh_note(
            "major pentatonic",
            root_class("A").unwrap(),
            3,
            320.0,
            240.0,
        );
        assert_ne!(
            in_c, in_a,
            "transposing must move the note under the same hand position"
        );

        // Re-keying the same strategy instance mid-sweep keeps the state usable.
        let layout = GridLayout::new(
            root_class("A").unwrap(),
            &preset("minor pentatonic").unwrap(),
            3,
        );
        let mut strategy = FingerPositionStrategy::default();
        strategy.apply_layout(layout.clone());
        for x in [100.0, 200.0, 300.0, 400.0, 500.0, 576.0] {
            let note = note_at(&mut strategy, x, 240.0);
            assert!(layout.notes.contains(&note), "x={x} gave off-scale {note}");
        }
    }

    #[test]
    fn no_gesture_when_no_hand_is_present() {
        let mut strategy = FingerPositionStrategy::default();
        assert!(strategy.process(&HandState::default()).is_none());
    }
}
