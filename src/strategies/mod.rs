use crate::core::{HandDetectionResult, Landmark, Strategy};
use crate::core::types::GestureEvent;

pub struct SimplePinchStrategy {
    pub threshold: f32,
}

impl Default for SimplePinchStrategy {
    fn default() -> Self {
        Self { threshold: 0.05 }
    }
}

impl Strategy for SimplePinchStrategy {
    fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent> {
        let landmarks = detection.hand_landmarks.as_ref()?;
        if landmarks.len() < 8 {
            return None;
        }
        
        let thumb = &landmarks[4];
        let index = &landmarks[8];

        let dx = thumb.x - index.x;
        let dy = thumb.y - index.y;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < self.threshold {
            return Some(GestureEvent::new_note(60, 1.0, "Pinch Detected"));
        }

        None
    }
}

pub struct FrequencyStrategy;

impl Strategy for FrequencyStrategy {
    fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent> {
        let landmarks = detection.hand_landmarks.as_ref()?;
        if landmarks.is_empty() {
            return None;
        }

        // Map the Y position of the first landmark to frequency (approx 200Hz - 1000Hz)
        // Note: In OpenCV, y=0 is top, y=1 is bottom.
        let y = landmarks[0].y;
        let freq = 200.0 + (1.0 - y as f32) * 800.0;

        Some(GestureEvent::new_signal(freq, 0.5, "Y-Position Signal"))
    }
}

pub struct FingerPositionStrategy;

impl Strategy for FingerPositionStrategy {
    fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent> {
        let landmarks = detection.hand_landmarks.as_ref()?;
        if landmarks.len() < 9 {
            return None;
        }

        // MediaPipe indices: Thumb tip (4), Index finger tip (8)
        let thumb = &landmarks[4];
        let index = &landmarks[8];

        // 1. Map Index Y position to MIDI Note (36 to 84 range)
        // y=0 (top) -> note 84; y=1 (bottom) -> note 36
        let y_pos = index.y as f32;
        let note_f = (1.0 - y_pos) * (84.0 - 36.0) + 36.0;
        let note = note_f.clamp(36.0, 84.0) as u8;

        // 2. Map Distance between Thumb and Index to Velocity (0.0 to 1.0)
        let dx = thumb.x - index.x;
        let dy = thumb.y - index.y;
        let distance = (dx * dx + dy * dy).sqrt();

        // Pinching (distance close to 0) -> Velocity 1.0
        // Large distance -> Velocity 0.0
        let velocity = (1.0 - (distance * 5.0)).clamp(0.0, 1.0);

        Some(GestureEvent::new_note(note, velocity, "Finger Position"))
    }
}
