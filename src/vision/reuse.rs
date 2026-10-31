//! Keypoint reuse policy, shared by both vision backends.
//!
//! A hand that has not moved and not changed shape produces the same model input,
//! so the previous keypoints can be served without running the graph again. The
//! reference is deliberately *not* updated while reuse is active, which bounds the
//! drift to `REUSE_MAX_FRAMES` frames.

use crate::core::Landmark;

/// A palm may drift by this many pixels (and this much of its size) and still
/// count as stationary.
const REUSE_MOVE_PX: f64 = 3.0;
const REUSE_SIZE_TOL: f64 = 0.03;
/// Never reuse for more than this many consecutive frames: hand *shape* can change
/// while the palm stays put, and volume is read from the fingertips.
const REUSE_MAX_FRAMES: u32 = 3;

/// `palm` is (centre x, centre y, radius) in frame pixels.
pub(crate) fn is_steady(palm: (f64, f64, f64), previous: Option<(f64, f64, f64)>) -> bool {
    let Some((previous_x, previous_y, previous_half)) = previous else {
        return false;
    };
    if previous_half <= 0.0 {
        return false;
    }
    (palm.0 - previous_x).abs() < REUSE_MOVE_PX
        && (palm.1 - previous_y).abs() < REUSE_MOVE_PX
        && (palm.2 - previous_half).abs() / previous_half < REUSE_SIZE_TOL
}

/// Centre and half-side of the square box a palm anchor describes.
pub(crate) fn square_box(palm: (f64, f64, f64)) -> (f32, f32, f32, f32) {
    let side = 2.0 * palm.2;
    (
        (palm.0 - palm.2) as f32,
        (palm.1 - palm.2) as f32,
        side as f32,
        side as f32,
    )
}

pub(crate) struct ReuseCache {
    landmarks: Vec<Landmark>,
    palm: Option<(f64, f64, f64)>,
    streak: u32,
    enabled: bool,
}

impl ReuseCache {
    pub(crate) fn new(enabled: bool) -> Self {
        ReuseCache {
            landmarks: Vec::new(),
            palm: None,
            streak: 0,
            enabled,
        }
    }

    /// Keypoints to use for this frame, or `None` when the model has to run.
    pub(crate) fn reuse(&mut self, palm: (f64, f64, f64)) -> Option<Vec<Landmark>> {
        if !self.enabled
            || self.landmarks.is_empty()
            || self.streak >= REUSE_MAX_FRAMES
            || !is_steady(palm, self.palm)
        {
            return None;
        }
        self.streak += 1;
        Some(self.landmarks.clone())
    }

    pub(crate) fn refresh(&mut self, palm: (f64, f64, f64), landmarks: &[Landmark]) {
        // clear + extend_from_slice keeps the existing allocation.
        self.landmarks.clear();
        self.landmarks.extend_from_slice(landmarks);
        self.palm = Some(palm);
        self.streak = 0;
    }

    /// The palm the cached keypoints belong to, if any.
    pub(crate) fn anchor(&self) -> Option<(f64, f64, f64)> {
        self.palm
    }

    pub(crate) fn clear(&mut self) {
        self.landmarks.clear();
        self.palm = None;
        self.streak = 0;
    }

    pub(crate) fn streak(&self) -> u32 {
        self.streak
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Landmark;

    fn landmark(x: f32, y: f32) -> Landmark {
        Landmark { x, y, z: 0.0 }
    }

    #[test]
    fn a_still_hand_reuses_its_landmarks() {
        let mut cache = ReuseCache::new(true);
        cache.refresh((100.0, 100.0, 50.0), &[landmark(1.0, 2.0)]);
        let reused = cache.reuse((101.0, 100.5, 50.0)).expect("steady palm reuses");
        assert_eq!(reused.len(), 1);
        assert_eq!(cache.streak(), 1);
    }

    #[test]
    fn reuse_is_capped_and_disabled_by_flag() {
        let mut cache = ReuseCache::new(true);
        cache.refresh((100.0, 100.0, 50.0), &[landmark(1.0, 2.0)]);
        for frame in 1..=REUSE_MAX_FRAMES {
            assert!(cache.reuse((100.0, 100.0, 50.0)).is_some(), "frame {frame} reuses");
        }
        assert!(cache.reuse((100.0, 100.0, 50.0)).is_none(), "streak is capped");

        let mut off = ReuseCache::new(false);
        let mut fresh = ReuseCache::new(true);
        fresh.refresh((100.0, 100.0, 50.0), &[landmark(1.0, 2.0)]);
        assert!(off.reuse((100.0, 100.0, 50.0)).is_none());
    }

    #[test]
    fn a_moved_or_resized_palm_is_not_steady() {
        let previous = Some((100.0, 100.0, 50.0));
        assert!(is_steady((102.0, 100.0, 51.0), previous));
        assert!(!is_steady((104.0, 100.0, 51.0), previous), "moved past 3 px");
        assert!(!is_steady((100.0, 100.0, 60.0), previous), "grew past 3 %");
        assert!(!is_steady((100.0, 100.0, 50.0), None), "nothing to compare against");
    }
}
