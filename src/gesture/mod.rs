//! Turning a hand into a performance.
//!
//! `estimator` resolves the raw detection into a `HandState`; `keyboard` and
//! `theremin` are the two ways of reading that state - a degree x octave
//! keyboard, or a continuous theremin band. Both are `Strategy` implementations
//! from `crate::core`, so adding a gesture means adding one file here.

pub mod estimator;
pub mod keyboard;
pub mod theremin;

use crate::core::geometry::HandGeometry;
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

/// Proximity calibration, in palm lengths per frame. `palm_length / min(W, H)` is
/// resolution independent: the same hand at the same distance reads the same value
/// at 640x480 and at 1280x720. On the fixture clip - a person at a desk, hand
/// sweeping from arm's length to within reach of the lens - the ratio runs 0.069 to
/// 0.135, so the band is 0.05..0.15 by default: at 0.05 (palm ~24 px) the hand is
/// out of reach and silent, at 0.15 (palm ~72 px) it is right at the lens and full
/// volume. Move the band with `--proximity-far` / `--proximity-near` if your camera
/// sits closer or further than a desk webcam.
pub const PROXIMITY_FAR: f32 = 0.05;
pub const PROXIMITY_NEAR: f32 = 0.15;
/// Weight of the current frame in the moving average of the proximity level. Lower
/// than the pitch smoother's because size is noisier than position: the palm
/// length jitters by ~2.4 % per frame even when the hand is still.
pub const PROXIMITY_EMA_ALPHA: f32 = 0.25;
/// How many frames a full sweep of the band (silent to full) takes to count as a
/// maximum-strength approach. 15 frames is half a second at 30 fps.
pub const APPROACH_FULL_BAND_FRAMES: f32 = 15.0;
/// Smoothing of the approach rate, so a single noisy frame cannot punch a note.
pub const APPROACH_EMA_ALPHA: f32 = 0.35;

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

/// How big the hand looks: palm length (wrist to middle knuckle) as a fraction of
/// the shorter frame side. Perspective makes apparent size a direct proxy for
/// distance - twice as close is twice as big - so this is the proximity signal.
///
/// The landmark `z` is deliberately *not* used. The landmarker crops a square
/// anchored to the detected palm and resamples it to a fixed 224x224, so the hand
/// always fills the same crop however far away it is; depth arrives divided by the
/// hand's own size and describes how the fingers curl, not how far the hand is.
pub fn proximity_ratio(geometry: &HandGeometry, frame_size: (usize, usize)) -> f32 {
    let reference = frame_size.0.min(frame_size.1).max(1) as f32;
    geometry.palm_length / reference
}

/// Map a raw ratio onto 0.0 (at or beyond `far`) .. 1.0 (at or beyond `near`).
/// A degenerate band (near <= far) is treated as "always full volume" rather than
/// dividing by zero.
pub fn proximity_level(ratio: f32, near: f32, far: f32) -> f32 {
    if near <= far {
        return 1.0;
    }
    ((ratio - far) / (near - far)).clamp(0.0, 1.0)
}

/// Pinch as a fraction of the loud ceiling proximity sets: a pinched hand is
/// 40 % as loud as an open one at the same distance.
const BLEND_PINCH_WEIGHT: f32 = 0.4;

/// Combine the two loudness sources according to the selected model.
pub fn volume_for(model: crate::core::instrument::VolumeModel, proximity: f32, pinch: f32) -> f32 {
    use crate::core::instrument::VolumeModel;
    match model {
        VolumeModel::Proximity => proximity,
        VolumeModel::Pinch => pinch,
        // Proximity is the ceiling, the pinch decides how much of it is used.
        VolumeModel::Blend => {
            (proximity * (BLEND_PINCH_WEIGHT + (1.0 - BLEND_PINCH_WEIGHT) * pinch)).clamp(0.0, 1.0)
        }
    }
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

    /// A geometry whose palm is `palm` pixels long.
    fn palm(palm: f32) -> HandGeometry {
        HandGeometry {
            palm_length: palm,
            ..HandGeometry::default()
        }
    }

    #[test]
    fn proximity_is_resolution_independent() {
        // The same hand at the same distance, captured at two resolutions: the
        // palm scales with the frame, so the ratio does not.
        let at_640 = proximity_ratio(&palm(60.0), (640, 480));
        let at_1280 = proximity_ratio(&palm(90.0), (1280, 720));
        assert!((at_640 - at_1280).abs() < 1e-6, "{at_640} vs {at_1280}");
        assert!((at_640 - 0.125).abs() < 1e-6);
    }

    #[test]
    fn the_proximity_band_saturates_at_both_ends() {
        assert_eq!(
            proximity_level(PROXIMITY_NEAR, PROXIMITY_NEAR, PROXIMITY_FAR),
            1.0
        );
        assert_eq!(
            proximity_level(PROXIMITY_FAR, PROXIMITY_NEAR, PROXIMITY_FAR),
            0.0
        );
        // Beyond either end it clamps instead of running past 0 or 1.
        assert_eq!(proximity_level(0.5, PROXIMITY_NEAR, PROXIMITY_FAR), 1.0);
        assert_eq!(proximity_level(0.01, PROXIMITY_NEAR, PROXIMITY_FAR), 0.0);
        // Half way between far and near is half volume.
        let middle = (PROXIMITY_NEAR + PROXIMITY_FAR) / 2.0;
        assert!((proximity_level(middle, PROXIMITY_NEAR, PROXIMITY_FAR) - 0.5).abs() < 1e-6);
        // A nonsensical band must not divide by zero.
        assert_eq!(proximity_level(0.1, 0.05, 0.2), 1.0);
    }

    #[test]
    fn each_volume_model_reads_its_own_source() {
        use crate::core::instrument::VolumeModel;
        assert_eq!(volume_for(VolumeModel::Proximity, 0.8, 0.2), 0.8);
        assert_eq!(volume_for(VolumeModel::Pinch, 0.8, 0.2), 0.2);
        // Blend: proximity is the ceiling, the pinch sets how much of it is used.
        let open = volume_for(VolumeModel::Blend, 1.0, 1.0);
        let pinched = volume_for(VolumeModel::Blend, 1.0, 0.0);
        assert!((open - 1.0).abs() < 1e-6);
        assert!((pinched - BLEND_PINCH_WEIGHT).abs() < 1e-6);
        assert!(pinched < open, "closing the hand must quiet the blend");
        assert_eq!(
            volume_for(VolumeModel::Blend, 0.0, 1.0),
            0.0,
            "far away is silent whatever the hand does"
        );
    }
}
