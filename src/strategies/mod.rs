use crate::core::types::GestureEvent;
use crate::core::{HandDetectionResult, Strategy};

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

/// Thumb-index spread (as a fraction of the hand box height) that drives the
/// volume to its maximum. Anything at or beyond this is full volume.
const MAX_SPREAD: f32 = 0.6;

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

        // Landmarks are in frame pixels; normalise them against the hand box so
        // the mapping does not depend on the camera resolution.
        let (_, by, _, bh) = detection.r#box.unwrap_or((0.0, 0.0, 1.0, 1.0));
        let y_pos = (index.y - by) / bh.max(1.0);
        let note_f = (1.0 - y_pos) * (84.0 - 36.0) + 36.0;
        let note = note_f.clamp(36.0, 84.0) as u8;

        // 2. Map Thumb <-> Index distance to Velocity / volume (0.0 to 1.0):
        // fingers far apart -> loud, pinched together -> silent.
        let dx = thumb.x - index.x;
        let dy = thumb.y - index.y;
        let distance = (dx * dx + dy * dy).sqrt() / bh.max(1.0);

        let velocity = (distance / MAX_SPREAD).clamp(0.0, 1.0);

        Some(GestureEvent::new_note(note, velocity, "Finger Position"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::GesturePayload;
    use crate::core::{HandDetectionResult, Landmark};

    /// A detection with a 200x200 hand box and a fixed landmark layout; only the
    /// thumb tip x coordinate changes, i.e. the thumb/index spread.
    fn detection_with_spread(thumb_x: f32) -> HandDetectionResult {
        let landmarks = (0..21)
            .map(|i| Landmark {
                x: if i == 4 { thumb_x } else { 1100.0 },
                y: 600.0,
                z: 0.0,
            })
            .collect();
        HandDetectionResult {
            hand_landmarks: Some(landmarks),
            r#box: Some((1000.0, 500.0, 200.0, 200.0)),
            score: 0.9,
            timings: Default::default(),
        }
    }

    fn velocity(thumb_x: f32) -> f32 {
        let event = FingerPositionStrategy
            .process(&detection_with_spread(thumb_x))
            .unwrap();
        match event.payload {
            GesturePayload::Note { velocity, .. } => velocity,
            _ => panic!("expected a note payload"),
        }
    }

    #[test]
    fn volume_grows_with_thumb_index_distance() {
        // Hand box height is 200 px, so spread/200 is the normalised distance and
        // MAX_SPREAD (0.6 -> 120 px) is where the volume saturates.
        let pinched = velocity(1102.0); //   2 px apart
        let mid = velocity(1160.0); //    60 px apart
        let wide = velocity(1220.0); //  120 px apart
        assert!(
            pinched < mid && mid < wide,
            "volume must rise with distance: {pinched} < {mid} < {wide}"
        );
        assert!((pinched - 2.0 / 200.0 / 0.6).abs() < 1e-6);
        assert!((mid - 0.5).abs() < 1e-6);
        assert!(
            (wide - 1.0).abs() < 1e-6,
            "beyond MAX_SPREAD saturates at 1.0"
        );
    }

    #[test]
    fn no_gesture_without_landmarks() {
        let empty = HandDetectionResult {
            hand_landmarks: None,
            r#box: Some((0.0, 0.0, 10.0, 10.0)),
            score: 0.1,
            timings: Default::default(),
        };
        assert!(FingerPositionStrategy.process(&empty).is_none());
    }
}
