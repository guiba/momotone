use crate::core::types::GestureEvent;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::time::Instant;

/// An event plus the wall clock it was dispatched on.
///
/// The sink thread measures `dispatched_at.elapsed()` when it picks the event up,
/// which is the slice of the latency budget nothing else sees: how long a gesture
/// sat in the queue before reaching an output.
#[derive(Debug, Clone)]
pub struct QueuedEvent {
    pub event: GestureEvent,
    pub dispatched_at: Instant,
}

/// What travels on the bus.
///
/// `Shutdown` is the end-of-run sentinel: it lets the sink thread block on the
/// channel instead of waking on a timeout to check a flag, which is how a real-time
/// app drains an event queue without polling it.
#[derive(Debug, Clone)]
pub enum BusMessage {
    Event(QueuedEvent),
    Shutdown,
}

/// A simple high-performance event bus using crossbeam channels.
#[derive(Clone)]
pub struct EventBus {
    sender: Sender<BusMessage>,
}

impl EventBus {
    pub fn new() -> (Self, Receiver<BusMessage>) {
        let (tx, rx) = unbounded();
        (Self { sender: tx }, rx)
    }

    /// Send an event to all subscribers.
    pub fn dispatch(&self, event: GestureEvent) {
        let _ = self.sender.send(BusMessage::Event(QueuedEvent {
            event,
            dispatched_at: Instant::now(),
        }));
    }

    /// Tell the sinks the run is over, so their blocking `recv` returns.
    pub fn close(&self) {
        let _ = self.sender.send(BusMessage::Shutdown);
    }
}
