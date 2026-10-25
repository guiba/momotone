use opencv::core::{Point, Scalar};
use opencv::imgproc;
use opencv::prelude::*;

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
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    // thumb
    (1, 2), (2, 3), (3, 4),
    // index
    (5, 6), (6, 7), (7, 8),
    // middle
    (9, 10), (10, 11), (11, 12),
    // ring
    (13, 14), (14, 15), (15, 16),
    // pinky
    (17, 18), (18, 19), (19, 20),
];

pub struct Visualizer {
    // In a real app, we might store state here for animations
}

impl Visualizer {
    pub fn new() -> Self {
        Self {}
    }

    /// Draw the hand skeleton: a bone between every connected landmark pair and
    /// a joint dot at each landmark. Landmarks are in frame pixels; nothing is
    /// drawn when the tracker did not find a hand.
    pub fn draw_skeleton(
        &self,
        frame: &mut Mat,
        detection: &HandDetectionResult,
    ) -> opencv::Result<()> {
        let landmarks = match &detection.hand_landmarks {
            Some(l) if l.len() >= 21 => l,
            _ => return Ok(()),
        };

        let bone = Scalar::new(255.0, 255.0, 255.0, 0.0); // white (BGR)
        let joint = Scalar::new(0.0, 0.0, 255.0, 0.0); // red (BGR)

        for (a, b) in HAND_CONNECTIONS {
            if a >= landmarks.len() || b >= landmarks.len() {
                continue;
            }
            imgproc::line(
                frame,
                Point::new(landmarks[a].x as i32, landmarks[a].y as i32),
                Point::new(landmarks[b].x as i32, landmarks[b].y as i32),
                bone,
                2,
                imgproc::LINE_AA,
                0,
            )?;
        }

        for landmark in landmarks {
            imgproc::circle(
                frame,
                Point::new(landmark.x as i32, landmark.y as i32),
                3,
                joint,
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }

        Ok(())
    }

    /// Draws the HUD (Heads-Up Display) on top of the camera frame.
    pub fn draw_hud(
        &self,
        frame: &mut Mat,
        event_text: &str,
        color: (u8, u8, u8),
        system_status: &[(&str, &str)], // e.g. [("MIDI", "Connected"), ("Audio", "Running")]
    ) -> opencv::Result<()> {
        let scalar_color = Scalar::new(color.0 as f64, color.1 as f64, color.2 as f64, 0.0);

        // 1. Draw Gesture Text (Large at bottom/middle)
        imgproc::put_text(
            frame,
            event_text,
            Point::new(30, 700),
            imgproc::FONT_HERSHEY_SIMPLEX,
            1.5,
            scalar_color,
            2,
            imgproc::LINE_8,
            false,
        )?;

        // 2. Draw System Status (Top right)
        let mut y_offset = 30;
        for (label, status) in system_status {
            let text = format!("{}: {}", label, status);
            imgproc::put_text(
                frame,
                &text,
                Point::new(500, y_offset),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                Scalar::new(255.0, 255.0, 255.0, 0.0), // White text
                1,
                imgproc::LINE_8,
                false,
            )?;
            y_offset += 30;
        }

        Ok(())
    }
}
