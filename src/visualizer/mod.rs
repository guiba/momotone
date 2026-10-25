use opencv::prelude::*;
use opencv::imgproc;
use opencv::core::{Point, Scalar};


pub struct Visualizer {
    // In a real app, we might store state here for animations
}

impl Visualizer {
    pub fn new() -> Self {
        Self {}
    }

    /// Draws the HUD (Heads-Up Display) on top of the camera frame.
    pub fn draw_hud(
        &self, 
        frame: &mut Mat, 
        event_text: &str, 
        color: (u8, u8, u8),
        system_status: &[(&str, &str)], // e.g., [("MIDI", "Connected"), ("Audio", "Running")]
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

    /// Draw the hand skeleton (landmarks are in frame pixels).
    pub fn draw_skeleton(&self, frame: &mut Mat, landmarks: &[crate::core::Landmark]) -> opencv::Result<()> {
        for landmark in landmarks {
            imgproc::circle(
                frame,
                Point::new(landmark.x as i32, landmark.y as i32),
                4,
                Scalar::new(255.0, 0.0, 0.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }
        Ok(())
    }
}
