//! Offline skeleton overlay for the `--image` / `--dump-frame` check.
//!
//! The live stage draws the skeleton in the iced canvas (`dashboard/skeleton.rs`);
//! this is the same topology painted onto a saved frame so a single photo can be
//! inspected without a window. Drawing is a Bresenham line and a filled disc
//! straight into the RGB buffer - no image library needed for 20 lines and 21 dots.

use crate::core::frame::Frame;
use crate::core::HandDetectionResult;

/// Hand landmark topology as defined by MediaPipe's `HandLandmarksConnections`
/// (https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarksConnections),
/// which is the topology the BlazeHand landmarker model emits: a palm polygon
/// plus one MCP->PIP->DIP->TIP chain per finger. 21 edges over 21 landmarks.
///
/// The older `mediapipe.solutions.hands` list additionally draws cross-links
/// such as (9, 4), (4, 7), (4, 10), (9, 2), (13, 5) and (14, 5); those assume an
/// upright, palm-forward hand and show up as stray lines whenever the hand is
/// rotated, so they are deliberately not used here.
const HAND_CONNECTIONS: [(usize, usize); 21] = [
    // palm: wrist -> thumb CMC -> knuckle line -> back to wrist
    (0, 1),
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),
    // thumb
    (1, 2),
    (2, 3),
    (3, 4),
    // index
    (5, 6),
    (6, 7),
    (7, 8),
    // middle
    (9, 10),
    (10, 11),
    (11, 12),
    // ring
    (13, 14),
    (14, 15),
    (15, 16),
    // pinky
    (17, 18),
    (18, 19),
    (19, 20),
];

const BONE: [u8; 3] = [255, 255, 255];
const JOINT: [u8; 3] = [255, 0, 0];
const BONE_WIDTH: usize = 2;
const JOINT_RADIUS: usize = 3;

pub struct Visualizer;

impl Visualizer {
    pub fn new() -> Self {
        Visualizer
    }

    /// Draw the hand skeleton: a bone between every connected landmark pair and a
    /// joint dot at each landmark. Landmarks are in frame pixels; nothing is drawn
    /// when the tracker did not find a hand.
    pub fn draw_skeleton(&self, frame: &mut Frame, detection: &HandDetectionResult) {
        let landmarks = match &detection.hand_landmarks {
            Some(l) if l.len() >= 21 => l,
            _ => return,
        };

        for &(a, b) in HAND_CONNECTIONS.iter() {
            if a >= landmarks.len() || b >= landmarks.len() {
                continue;
            }
            line(
                frame,
                (landmarks[a].x, landmarks[a].y),
                (landmarks[b].x, landmarks[b].y),
                BONE,
                BONE_WIDTH,
            );
        }

        for landmark in landmarks {
            disc(frame, (landmark.x, landmark.y), JOINT_RADIUS, JOINT);
        }
    }
}

fn put(frame: &mut Frame, x: usize, y: usize, colour: [u8; 3]) {
    if x >= frame.width() || y >= frame.height() {
        return;
    }
    let i = (y * frame.width() + x) * 3;
    let pixels = frame.pixels_mut();
    pixels[i] = colour[0];
    pixels[i + 1] = colour[1];
    pixels[i + 2] = colour[2];
}

/// A thick line by walking the segment and stamping a `width`-wide disc at each
/// step. Crisp enough for an overlay and about twenty lines of code.
fn line(frame: &mut Frame, from: (f32, f32), to: (f32, f32), colour: [u8; 3], width: usize) {
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let length = (dx * dx + dy * dy).sqrt();
    let radius = width / 2;
    let steps = length.ceil() as usize + 1;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        disc(frame, (from.0 + dx * t, from.1 + dy * t), radius, colour);
    }
}

fn disc(frame: &mut Frame, centre: (f32, f32), radius: usize, colour: [u8; 3]) {
    let cx = centre.0.round() as i64;
    let cy = centre.1.round() as i64;
    let r2 = (radius * radius) as i64;
    for dy in -(radius as i64)..=radius as i64 {
        for dx in -(radius as i64)..=radius as i64 {
            if dx * dx + dy * dy <= r2 {
                put(frame, (cx + dx) as usize, (cy + dy) as usize, colour);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Landmark;

    #[test]
    fn nothing_is_drawn_without_a_hand() {
        let mut frame = Frame::solid(10, 10, 0);
        let detection = HandDetectionResult::default();
        Visualizer::new().draw_skeleton(&mut frame, &detection);
        assert!(frame.pixels().iter().all(|&p| p == 0));
    }

    #[test]
    fn the_bones_are_white_and_the_joints_red() {
        let mut frame = Frame::solid(40, 40, 0);
        let landmarks = (0..21)
            .map(|i| Landmark {
                x: 20.0,
                y: 5.0 + i as f32,
                z: 0.0,
            })
            .collect();
        let detection = HandDetectionResult {
            hand_landmarks: Some(landmarks),
            ..HandDetectionResult::default()
        };
        Visualizer::new().draw_skeleton(&mut frame, &detection);

        let lit = frame
            .pixels()
            .chunks(3)
            .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
            .count();
        assert!(lit > 0, "the skeleton lights up pixels");
    }

    #[test]
    fn drawing_clips_to_the_frame() {
        let mut frame = Frame::solid(4, 4, 0);
        let landmarks = (0..21)
            .map(|i| Landmark {
                x: -10.0 + i as f32,
                y: 100.0,
                z: 0.0,
            })
            .collect();
        let detection = HandDetectionResult {
            hand_landmarks: Some(landmarks),
            ..HandDetectionResult::default()
        };
        Visualizer::new().draw_skeleton(&mut frame, &detection);
        // Landmarks far outside the frame must not panic or write out of bounds.
        assert_eq!(frame.pixels().len(), 4 * 4 * 3);
    }
}
