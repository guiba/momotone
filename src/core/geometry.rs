//! Resolution-independent geometry derived from the 21 MediaPipe landmarks.
//!
//! Everything here is computed in *frame pixels* and normalised by the hand's
//! own size (palm length), so the numbers stay comparable across camera
//! resolutions and distances. The dashboard reads these values; strategies can
//! reuse them instead of recomputing.

use crate::core::HandDetectionResult;
use crate::core::Landmark;

pub const FINGER_NAMES: [&str; 5] = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

/// `(tip, knuckle)` landmark index for each finger, in `FINGER_NAMES` order.
/// MediaPipe: wrist 0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20.
const FINGER_CHAINS: [(usize, usize); 5] = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)];

#[derive(Debug, Clone, Copy, Default)]
pub struct HandGeometry {
    /// Thumb-tip <-> index-tip distance, as a fraction of the palm length.
    /// 0 is a pinch, ~1 is a fully spread hand.
    pub spread: f32,
    /// Tip-to-knuckle distance per finger, as a fraction of the palm length.
    /// A folded finger is ~0.3, a straight one ~1.1.
    pub extension: [f32; 5],
    /// Angle (degrees) of the knuckle line (index MCP -> pinky MCP) measured
    /// from the horizontal. 0 means an upright hand, +/-90 a sideways one.
    pub tilt_deg: f32,
    /// Wrist -> middle-finger MCP distance, in frame pixels.
    pub palm_length: f32,
}

fn distance(a: &Landmark, b: &Landmark) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

/// Returns `None` when the landmark set is too small to be a hand.
pub fn hand_geometry(detection: &HandDetectionResult) -> Option<HandGeometry> {
    let landmarks = detection.hand_landmarks.as_ref()?;
    if landmarks.len() < 21 {
        return None;
    }

    let palm_length = distance(&landmarks[0], &landmarks[9]).max(1.0);

    let mut extension = [0.0; 5];
    for (i, (tip, knuckle)) in FINGER_CHAINS.iter().enumerate() {
        extension[i] = distance(&landmarks[*tip], &landmarks[*knuckle]) / palm_length;
    }

    let dx = landmarks[17].x - landmarks[5].x;
    let dy = landmarks[17].y - landmarks[5].y;
    let tilt_deg = dy.atan2(dx).to_degrees();

    Some(HandGeometry {
        spread: distance(&landmarks[4], &landmarks[8]) / palm_length,
        extension,
        tilt_deg,
        palm_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Landmark;

    /// 21 landmarks of a straight, upright hand: wrist at (100, 300), five
    /// fingers 120 px tall, thumb sticking out by `thumb_x`.
    fn upright_hand(thumb_x: f32) -> HandDetectionResult {
        let mut landmarks = Vec::new();
        landmarks.push(Landmark {
            x: 100.0,
            y: 300.0,
            z: 0.0,
        }); // wrist
        landmarks.push(Landmark {
            x: 80.0,
            y: 280.0,
            z: 0.0,
        }); // thumb CMC
        landmarks.push(Landmark {
            x: thumb_x,
            y: 240.0,
            z: 0.0,
        }); // thumb MCP
        landmarks.push(Landmark {
            x: thumb_x,
            y: 220.0,
            z: 0.0,
        }); // thumb IP
        landmarks.push(Landmark {
            x: thumb_x,
            y: 200.0,
            z: 0.0,
        }); // thumb tip
        for finger in 0..4 {
            let base_x = 100.0 + finger as f32 * 30.0;
            for joint in 0..4 {
                landmarks.push(Landmark {
                    x: base_x,
                    y: 220.0 - joint as f32 * 40.0,
                    z: 0.0,
                });
            }
        }
        HandDetectionResult {
            hand_landmarks: Some(landmarks),
            r#box: Some((60.0, 100.0, 200.0, 250.0)),
            score: 0.9,
            timings: Default::default(),
        }
    }

    #[test]
    fn straight_fingers_are_extended() {
        let geometry = hand_geometry(&upright_hand(40.0)).unwrap();
        // A 120 px finger over an ~85 px palm; the thumb is much shorter.
        assert!(geometry.extension[1] > 1.0);
        assert!(geometry.extension[0] < geometry.extension[1]);
    }

    #[test]
    fn spread_grows_with_thumb_distance() {
        let narrow = hand_geometry(&upright_hand(80.0)).unwrap();
        let wide = hand_geometry(&upright_hand(200.0)).unwrap();
        assert!(narrow.spread < wide.spread);
    }

    #[test]
    fn upright_hand_has_no_tilt() {
        let geometry = hand_geometry(&upright_hand(40.0)).unwrap();
        assert!((geometry.tilt_deg - 0.0).abs() < 1e-3);
    }

    #[test]
    fn no_geometry_without_landmarks() {
        let empty = HandDetectionResult::default();
        assert!(hand_geometry(&empty).is_none());
    }
}
