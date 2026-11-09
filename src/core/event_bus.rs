use crate::core::types::GestureEvent;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::time::Instant;

/// An event plus the wall clock it was dispatched on.
///
/// The sink thread measures `dispatched_at.elapsed()` when it picks the event up,
/// which is the slice of the latency budget nothing else sees: how long a gesture
/// sat in the queue before reaching an output.
pub struct QueuedEvent {
    pub event: GestureEvent,
    pub dispatched_at: Instant,
}

/// A simple high-performance event bus using crossbeam channels.
#[derive(Clone)]
pub struct EventBus {
    sender: Sender<QueuedEvent>,
}

impl EventBus {
    pub fn new() -> (Self, Receiver<QueuedEvent>) {
        let (tx, rx) = unbounded();
        (Self { sender: tx }, rx)
    }

    /// Send an event to all subscribers.
    pub fn dispatch(&self, event: GestureEvent) {
        let _ = self.sender.send(QueuedEvent {
            event,
            dispatched_at: Instant::now(),
        });
    }
}
