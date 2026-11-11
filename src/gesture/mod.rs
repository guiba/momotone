//! Turning a hand into a performance.
//!
//! `estimator` resolves the raw detection into a `HandState`; `keyboard` and
//! `theremin` are the two ways of reading that state - a degree x octave
//! keyboard, or a continuous theremin band. Both are `Strategy` implementations
//! from `crate::core`, so adding a gesture means adding one file here.

pub mod estimator;
pub mod keyboard;
pub mod theremin;

use crate::core::grid::GridLayout;
use crate::core::types::Performance;
use crate::core::Landmark;
use crate::core::NOTE_MARGIN;

pub use estimator::{HandEstimator, HandState};

/// Interface for mapping a hand to a performance.
///
/// `&mut self` because a strategy may hold state (the keyboard's grid), and the
/// frame thread is the only one that ever calls it. The *gesture* state - presence,
/// smoothing, geometry - lives in `HandEstimator`, not in the strategy.
pub trait Strategy: Send {
    fn process(&mut self, state: &HandState) -> Option<Performance>;
    /// Swap the keyboard out from under the strategy without rebuilding it.
    fn apply_layout(&mut self, _layout: GridLayout) {}
}

/// Thumb-index spread (as a fraction of the hand box height) that drives the
/// volume to its maximum. Anything at or beyond this is full volume.
const MAX_SPREAD: f32 = 0.6;

/// Weight of the current frame in the exponential moving average of the index
/// tip position. 0.35 is a ~3 frame time constant: fast enough to follow a
/// deliberate sweep, slow enough to swallow per-frame landmark jitter.
pub const PITCH_EMA_ALPHA: f32 = 0.35;

/// Normalised position of a point inside the playable band, 0..1 on each axis.
/// The outer `NOTE_MARGIN` of the frame is excluded on both axes, otherwise the
/// extreme cells would need the hand pressed against the edge of the image where
/// the detector loses it.
pub fn band_position(x: f32, y: f32, frame_size: (usize, usize)) -> (f32, f32) {
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

/// Thumb-tip <-> index-tip distance as a fraction of the hand box height,
/// saturating at `MAX_SPREAD`. Normalised against the hand box so the mapping is
/// resolution independent; note this is a *different* normalisation from
/// `HandGeometry::spread`, which uses palm length - the box is a constant square
/// the detector keeps, so it is the steadier denominator for loudness. Callers
/// must have at least 9 landmarks.
pub fn spread_volume(landmarks: &[Landmark], hand_box: Option<(f32, f32, f32, f32)>) -> f32 {
    let (_, _, _, box_height) = hand_box.unwrap_or((0.0, 0.0, 1.0, 1.0));
    let dx = landmarks[4].x - landmarks[8].x;
    let dy = landmarks[4].y - landmarks[8].y;
    let distance = (dx * dx + dy * dy).sqrt() / box_height.max(1.0);
    (distance / MAX_SPREAD).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_band_excludes_the_margin() {
        let (x, y) = band_position(64.0, 48.0, (640, 480));
        assert!((x - 0.0).abs() < 1e-6 && (y - 0.0).abs() < 1e-6);
        let (x, y) = band_position(576.0, 432.0, (640, 480));
        assert!((x - 1.0).abs() < 1e-6 && (y - 1.0).abs() < 1e-6);
        // Outside the band saturates rather than going negative.
        let (x, _) = band_position(0.0, 240.0, (640, 480));
        assert_eq!(x, 0.0);
    }

    #[test]
    fn spread_saturates_at_full_volume() {
        let landmarks: Vec<Landmark> = (0..21)
            .map(|i| Landmark {
                x: if i == 4 { 440.0 } else { 320.0 },
                y: 240.0,
                z: 0.0,
            })
            .collect();
        // 120 px gap over a 200 px box is exactly MAX_SPREAD.
        let full = spread_volume(&landmarks, Some((200.0, 140.0, 200.0, 200.0)));
        assert!((full - 1.0).abs() < 1e-6);
        assert_eq!(spread_volume(&landmarks, None), 1.0);
    }

    #[test]
    fn finger_names_cover_every_finger() {
        use crate::core::geometry::FINGER_NAMES;
        assert_eq!(FINGER_NAMES.len(), 5);
    }
}
