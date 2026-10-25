use crate::core::types::GestureEvent;
use crossbeam_channel::{unbounded, Receiver, Sender};

/// A simple high-performance event bus using crossbeam channels.
#[derive(Clone)]
pub struct EventBus {
    sender: Sender<GestureEvent>,
}

impl EventBus {
    pub fn new() -> (Self, Receiver<GestureEvent>) {
        let (tx, rx) = unbounded();
        (Self { sender: tx }, rx)
    }

    /// Send an event to all subscribers.
    pub fn dispatch(&self, event: GestureEvent) {
        let _ = self.sender.send(event);
    }
}
