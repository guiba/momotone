use crate::core::types::GestureEvent;
use std::time::{Duration, Instant};

pub struct MockSource {
    start_time: Instant,
}

impl MockSource {
    pub fn new() -> Self {
        Self { start_time: Instant::now() }
    }
}

/// Simulates hand tracking data by periodically generating events.
pub trait SourceTrait {
    fn next_event(&mut self) -> Option<GestureEvent>;
}

impl MockSource {
    // In a real source (like Webcam), this would return landmarks/frame info.
    // For the mock, we just yield random gesture events every few seconds.
}

pub struct MockSourceImpl {
    last_event_time: Instant,
}

impl MockSourceImpl {
    pub fn new() -> Self {
        Self { last_event_time: Instant::now() }
    }

    pub fn get_next(&mut self) -> Option<GestureEvent> {
        if self.last_event_time.elapsed() < Duration::from_secs(2) {
            return None;
        }
        self.last_event_time = Instant::now();
        
        // Randomly alternate between a note and continuous signal for simulation
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
        
        if now % 2 == 0 {
            Some(GestureEvent::new_note((now % 13) as u8, 0.8, "Mock Note"))
        } else {
            Some(GestureEvent::new_signal(220.0 + (now % 440) as f32, 0.5, "Mock Signal"))
        }
    }
}

impl MockSourceImpl {
    // Wait, I'll use a simpler approach for the mock source in the controller directly or here.
}
