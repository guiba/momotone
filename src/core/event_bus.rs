use crate::core::effects::Modulation;
use crate::core::types::Performance;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::time::Instant;

/// A performance plus the wall clock it was dispatched on.
///
/// The sink thread measures `dispatched_at.elapsed()` when it picks the message up,
/// which is the slice of the latency budget nothing else sees: how long a gesture sat
/// in the queue before reaching an output.
#[derive(Debug, Clone)]
pub struct QueuedEvent {
    pub performance: Performance,
    pub dispatched_at: Instant,
}

/// What travels on the bus.
///
/// `Shutdown` is the end-of-run sentinel: it lets the sink thread block on the
/// channel instead of waking on a timeout to check a flag, which is how a real-time
/// app drains an event queue without polling it.
///
/// `Modulation` is deliberately *not* a `QueuedEvent`: the chain parameters are
/// continuous state, not attacks, so they are not timed and must not be counted in
/// the dispatch-latency sample. A sink that cannot modulate ignores them.
#[derive(Debug, Clone)]
pub enum BusMessage {
    Event(QueuedEvent),
    Modulation(Modulation),
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

    /// Send a performance to the sinks.
    pub fn dispatch(&self, performance: Performance) {
        let _ = self.sender.send(BusMessage::Event(QueuedEvent {
            performance,
            dispatched_at: Instant::now(),
        }));
    }

    /// Send the effect chain's parameters. The channel is FIFO, so a modulation sent
    /// before a note reaches the sinks while that note is attacking.
    pub fn dispatch_modulation(&self, modulation: Modulation) {
        let _ = self.sender.send(BusMessage::Modulation(modulation));
    }

    /// Tell the sinks the run is over, so their blocking `recv` returns.
    pub fn close(&self) {
        let _ = self.sender.send(BusMessage::Shutdown);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::effects::Modulation;
    use crate::core::types::Performance;

    #[test]
    /// Modulation is state, not an attack: it shares the channel with the notes and
    /// still has to arrive in order, and the shutdown still ends the drain.
    fn modulation_and_shutdown_drain_in_order() {
        let (bus, rx) = EventBus::new();
        bus.dispatch_modulation(Modulation::neutral());
        bus.dispatch(Performance::Release);
        bus.close();

        let messages: Vec<BusMessage> = rx.try_iter().collect();
        assert_eq!(messages.len(), 3, "nothing was dropped");
        assert!(matches!(messages[0], BusMessage::Modulation(_)));
        assert!(matches!(messages[1], BusMessage::Event(_)));
        assert!(matches!(messages[2], BusMessage::Shutdown));
    }
}
