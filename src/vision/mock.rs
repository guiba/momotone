//! A synthetic hand, so the instrument can be run without a webcam or the ONNX
//! models (`--mode mock`).
//!
//! The hand is driven by the **frame index**, not the wall clock. That is what makes
//! a mock run reproducible: N frames always show the same hand at the same place, so
//! a golden trace recorded from a mock run is a real reference rather than a
//! screenshot. The deadline clock in the pipeline still paces the loop, so the
//! telemetry stays honest even though the geometry is not time-based.
//!
//! Everything here is geometry only - no pixels, no model, no I/O.

use crate::core::{HandDetectionResult, Landmark};

/// Mock mode emulates a 30 fps camera so the telemetry looks real. The loop uses a
/// deadline rather than a flat sleep: macOS `thread::sleep(33 ms)` actually returns
/// after ~41 ms, so a naive sleep makes every mock number 25 % wrong.
pub const MOCK_CADENCE: std::time::Duration = std::time::Duration::from_millis(33);

/// A synthetic, moving hand shaped like a real one: a wrist, a thumb and four
/// fingers, each a chain of joints that curls together over time.
pub fn synthetic_detection(seconds: f64) -> HandDetectionResult {
    let t = seconds as f32;
    // The centre sweeps almost the full 640 px view so the mock hand walks the
    // whole keyboard, low notes on the left and high notes on the right.
    let cx = 320.0 + 240.0 * (t * 0.7).sin();
    let cy = 250.0 + 50.0 * (t * 0.45).cos();
    let curl = 0.45 + 0.45 * (t * 1.1).sin();
    let thumb_angle = 55.0 + 25.0 * (t * 0.8).cos();

    let wrist = Landmark {
        x: cx,
        y: cy + 140.0,
        z: 0.0,
    };
    let thumb_cmc = Landmark {
        x: cx - 45.0,
        y: cy + 95.0,
        z: 0.0,
    };

    let mut landmarks = vec![wrist, thumb_cmc.clone()];
    landmarks.extend(finger_chain(thumb_cmc, thumb_angle, 95.0, curl, 3));

    let knuckles = [
        (cx - 25.0, cy + 90.0, -12.0, 130.0), // index
        (cx, cy + 90.0, 0.0, 140.0),          // middle
        (cx + 25.0, cy + 90.0, 12.0, 130.0),  // ring
        (cx + 48.0, cy + 90.0, 24.0, 110.0),  // pinky
    ];
    for (x, y, angle, length) in knuckles {
        let base = Landmark { x, y, z: 0.0 };
        landmarks.extend(finger_chain(base, angle, length, curl, 4));
    }

    let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for landmark in &landmarks {
        minx = minx.min(landmark.x);
        maxx = maxx.max(landmark.x);
        miny = miny.min(landmark.y);
        maxy = maxy.max(landmark.y);
    }

    HandDetectionResult {
        hand_landmarks: Some(landmarks),
        r#box: Some((minx, miny, maxx - minx, maxy - miny)),
        frame_size: (640, 480),
        score: 0.85 + 0.12 * (t * 0.6).sin(),
        timings: Default::default(),
    }
}

/// Builds `joints` landmarks walking away from `base` in the given direction,
/// bending progressively according to `curl`. 0 degrees points straight up.
fn finger_chain(
    base: Landmark,
    angle_deg: f32,
    length: f32,
    curl: f32,
    joints: usize,
) -> Vec<Landmark> {
    let angle = angle_deg * std::f32::consts::PI / 180.0;
    let segment = length / joints as f32;

    let mut x = base.x;
    let mut y = base.y;
    let mut chain = Vec::with_capacity(joints);

    for i in 0..joints {
        let a = angle + curl * i as f32 * 0.35;
        x += a.sin() * segment;
        y -= a.cos() * segment;
        chain.push(Landmark { x, y, z: 0.0 });
    }

    chain
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::geometry::hand_geometry;

    #[test]
    fn synthetic_hand_has_21_landmarks() {
        let detection = synthetic_detection(1.0);
        let landmarks = detection.hand_landmarks.as_ref().unwrap();
        assert_eq!(
            landmarks.len(),
            21,
            "MediaPipe order: wrist, thumb x5, then 4 x 4"
        );
        assert!(detection.r#box.is_some(), "the mock hand carries a box");
    }

    #[test]
    fn synthetic_hand_geometry_is_sane() {
        let landmarks = synthetic_detection(2.0).hand_landmarks.unwrap();
        let geometry = hand_geometry(&landmarks).unwrap();
        assert!(geometry.palm_length > 0.0);
        assert!(geometry.spread > 0.0);
        // A synthetic hand with fingers longer than the thumb, as built above.
        assert!(geometry.extension[1] > geometry.extension[0]);
    }

    #[test]
    fn the_synthetic_hand_curls_over_time() {
        // The whole point of the mock is that it moves: a curling finger changes the
        // thumb-index spread, which is what the volume is derived from.
        let open =
            hand_geometry(synthetic_detection(0.4).hand_landmarks.as_ref().unwrap()).unwrap();
        let closed =
            hand_geometry(synthetic_detection(1.9).hand_landmarks.as_ref().unwrap()).unwrap();
        assert!(
            (open.spread - closed.spread).abs() > 0.05,
            "the mock hand has to actually move, or the tests prove nothing"
        );
    }

    #[test]
    fn the_same_frame_number_gives_the_same_hand() {
        // `Landmark` is not `PartialEq` - the coordinates are floats - so the
        // comparison is on the numbers themselves.
        let a = synthetic_detection(2.0).hand_landmarks.unwrap();
        let b = synthetic_detection(2.0).hand_landmarks.unwrap();
        for (left, right) in a.iter().zip(b.iter()) {
            assert_eq!((left.x, left.y), (right.x, right.y));
        }
    }
}
