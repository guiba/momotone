use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    Continuous,
    Discrete,
}

/// Represents the data associated with a gesture.
/// Instead of Python's dict payload, we use an Enum for type safety and performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GesturePayload {
    // For Continuous gestures (e.g., frequency modulation)
    Signal { frequency: f32, amplitude: f32 },
    // For Discrete gestures (e.g., note on/off)
    Note { note: u8, velocity: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureEvent {
    pub event_type: EventType,
    pub payload: GesturePayload,
    pub text: String,
    // Color as RGB components for visualizer
    pub color: (u8, u8, u8),
}

impl GestureEvent {
    /// Helper to create a continuous signal event
    pub fn new_signal(frequency: f32, amplitude: f32, text: &str) -> Self {
        Self {
            event_type: EventType::Continuous,
            payload: GesturePayload::Signal {
                frequency,
                amplitude,
            },
            text: text.to_string(),
            color: (0, 255, 0), // Green for signal
        }
    }

    /// Helper to create a discrete note event
    pub fn new_note(note: u8, velocity: f32, text: &str) -> Self {
        Self {
            event_type: EventType::Discrete,
            payload: GesturePayload::Note { note, velocity },
            text: text.to_string(),
            color: (0, 255, 255), // Cyan for notes
        }
    }

    /// Helper for "No Hand" state
    pub fn no_gesture() -> Self {
        Self {
            event_type: EventType::Discrete,
            payload: GesturePayload::Note {
                note: 0,
                velocity: 0.0,
            }, // Dummy payload
            text: "No Hand Detected".to_string(),
            color: (255, 0, 0), // Red for empty/error
        }
    }
}
