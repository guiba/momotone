use crate::core::types::Performance;
use crate::core::Sink;
use anyhow::{anyhow, Result};
use midir::{MidiOutput, MidiOutputConnection};
use std::sync::Mutex;
use std::time::Instant;

/// How long a note must be sounding before the voice is allowed to move to a
/// different cell. The strategy emits a gesture event every frame, so at 30 fps
/// a hand sweeping the keyboard crosses a cell every few frames; without a floor
/// the voice re-attacks on every crossing and the sweep reads as a blur instead
/// of an arpeggio. 0 disables the floor (attack on every cell change).
pub const DEFAULT_MIN_NOTE_MS: u64 = 120;

const NOTE_ON: u8 = 0x90;
const NOTE_OFF: u8 = 0x80;
const CONTROL_CHANGE: u8 = 0xB0;
/// MIDI CC 7 is volume. It is the only way to change the loudness of a note that
/// is already sounding: velocity only means anything at the attack.
const VOLUME_CC: u8 = 7;

/// Attack discipline for the monophonic MIDI voice.
///
/// It owns the decision of *what to send* and returns the raw messages, so the
/// engine only has to write them to the midir connection. Keeping the state out
/// of the connection means the note-on/off behaviour is testable without a synth
/// attached.
#[derive(Debug)]
pub struct NoteGate {
    /// The note that is currently sounding, if any.
    held: Option<u8>,
    /// Volume the held note was last told to have, so the CC only moves when it
    /// actually changed.
    cc: u8,
    /// When `held` was attacked.
    attacked: Instant,
    min_note_ms: u64,
}

impl NoteGate {
    pub fn new(min_note_ms: u64) -> Self {
        Self {
            held: None,
            cc: 0,
            attacked: Instant::now(),
            min_note_ms,
        }
    }

    /// Messages to send for one frame's worth of performance. A release is passed
    /// in explicitly, not encoded as note 0.
    pub fn attack(&mut self, note: u8, velocity: f32, now: Instant) -> Vec<[u8; 3]> {
        if velocity <= 0.0 {
            return self.release();
        }
        let volume = volume_cc(velocity);

        let Some(previous) = self.held else {
            return self.strike(note, volume, now);
        };

        if previous == note {
            // Already sounding: loudness is a CC, never a second attack. Sending
            // note-on again would make the receiving synth re-pluck the note 30
            // times a second.
            return self.set_volume(volume);
        }

        if now.duration_since(self.attacked).as_millis() < self.min_note_ms as u128 {
            // The floor: let the note we are already attacking ring for its
            // minimum length instead of chopping it into 33 ms slivers. The
            // volume still follows the hand while we wait.
            return self.set_volume(volume);
        }

        let mut messages = vec![[NOTE_OFF, previous, 0]];
        messages.extend(self.strike(note, volume, now));
        messages
    }

    /// Release whatever is held. Safe to call repeatedly: a second release sends
    /// nothing, so there is never an orphan note-off for a note nobody played.
    pub fn release(&mut self) -> Vec<[u8; 3]> {
        match self.held {
            Some(note) => {
                self.held = None;
                self.cc = 0;
                vec![[NOTE_OFF, note, 0]]
            }
            None => Vec::new(),
        }
    }

    fn strike(&mut self, note: u8, volume: u8, now: Instant) -> Vec<[u8; 3]> {
        self.held = Some(note);
        self.cc = volume;
        self.attacked = now;
        vec![[NOTE_ON, note, volume]]
    }

    fn set_volume(&mut self, volume: u8) -> Vec<[u8; 3]> {
        if self.held.is_none() || volume == self.cc {
            return Vec::new();
        }
        self.cc = volume;
        vec![[CONTROL_CHANGE, VOLUME_CC, volume]]
    }
}

/// Velocity 0 on a note-on is MIDI for "note off", so a note the gate believes is
/// held is never attacked with a zero volume; a whisper still counts as an attack.
fn volume_cc(velocity: f32) -> u8 {
    ((velocity.clamp(0.0, 1.0) * 127.0).round() as u8).max(1)
}

pub struct MidiOutputEngine {
    port_name: String,
    connection: Option<Mutex<MidiOutputConnection>>,
    gate: NoteGate,
}

impl MidiOutputEngine {
    pub fn new(port_name: &str, min_note_ms: u64) -> Self {
        Self {
            port_name: port_name.to_string(),
            connection: None,
            gate: NoteGate::new(min_note_ms),
        }
    }

    fn send(&self, message: &[u8]) -> Result<()> {
        let conn = self
            .connection
            .as_ref()
            .ok_or_else(|| anyhow!("MIDI not connected"))?;
        conn.lock()
            .map_err(|_| anyhow!("MIDI connection lock poisoned"))?
            .send(message)
            .map_err(|e| anyhow!("MIDI send failed: {e}"))
    }
}

impl Sink for MidiOutputEngine {
    fn start(&mut self) -> Result<()> {
        // `midir::ConnectError` is not `Sync` on Linux - it carries a raw ALSA
        // pointer - so it cannot go through `?` into an `anyhow::Error`. Turning it
        // into a message here is both the fix and a better error for the user.
        let midi_out = MidiOutput::new("Gesture MIDI")
            .map_err(|e| anyhow!("could not open a MIDI output: {e}"))?;
        let available_ports = midi_out.ports();

        // Find the port with a matching name
        let port = available_ports
            .iter()
            .find(|p| {
                midi_out
                    .port_name(p)
                    .unwrap_or_default()
                    .contains(&self.port_name)
            })
            .ok_or_else(|| anyhow!("MIDI port '{}' not found", self.port_name))?;

        let conn = midi_out
            .connect(port, "Gesture MIDI Rust")
            .map_err(|e| anyhow!("could not connect to the MIDI port: {e}"))?;
        self.connection = Some(Mutex::new(conn));
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        // Release any held note before dropping the connection.
        for message in self.gate.release() {
            let _ = self.send(&message);
        }
        // Connection is closed when dropped
        self.connection = None;
        Ok(())
    }

    fn handle(&mut self, performance: &Performance) -> Result<()> {
        if self.connection.is_none() {
            return Ok(()); // not started, ignore performances
        }

        let messages = match performance {
            Performance::Note { note, velocity } => {
                // `FingerPositionStrategy` already emits MIDI numbers (36-84);
                // the old +60 offset pushed values past 127 and wrapped them.
                self.gate.attack(*note, *velocity, Instant::now())
            }
            Performance::Release => self.gate.release(),
            // A theremin has no MIDI equivalent: there is no note to attack and
            // no continuous pitch to send, so it is simply dropped.
            Performance::Signal { .. } => Vec::new(),
        };

        for message in messages {
            self.send(&message)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A gate whose clock advances one frame at a time, so the floor is testable
    /// without sleeping.
    struct TestClock {
        base: Instant,
        frame: Duration,
    }

    impl TestClock {
        fn new(frame_ms: u64) -> Self {
            Self {
                base: Instant::now(),
                frame: Duration::from_millis(frame_ms),
            }
        }
        fn at(&self, frame: u32) -> Instant {
            self.base + self.frame * frame
        }
    }

    #[test]
    fn a_held_note_attacks_once_and_then_only_moves_the_volume() {
        let clock = TestClock::new(33);
        let mut gate = NoteGate::new(DEFAULT_MIN_NOTE_MS);

        // Thirty frames of the same note: exactly one attack, no re-plucking.
        let mut messages = Vec::new();
        for frame in 0..30 {
            messages.extend(gate.attack(60, 0.8, clock.at(frame)));
        }

        assert_eq!(messages, vec![[NOTE_ON, 60, 102]]);
        assert_eq!(gate.held, Some(60));
    }

    #[test]
    fn velocity_of_a_held_note_is_sent_as_cc_7() {
        let clock = TestClock::new(33);
        let mut gate = NoteGate::new(DEFAULT_MIN_NOTE_MS);

        gate.attack(60, 0.5, clock.at(0));
        assert_eq!(
            gate.attack(60, 0.9, clock.at(1)),
            vec![[CONTROL_CHANGE, VOLUME_CC, 114]]
        );
        // Repeating the same volume sends nothing.
        assert!(gate.attack(60, 0.9, clock.at(2)).is_empty());
        assert_eq!(gate.held, Some(60));
    }

    #[test]
    fn a_cell_change_releases_the_old_note_before_attacking_the_new_one() {
        let clock = TestClock::new(33);
        let mut gate = NoteGate::new(DEFAULT_MIN_NOTE_MS);

        gate.attack(60, 1.0, clock.at(0));
        // Four frames later (132 ms) the floor has passed.
        let messages = gate.attack(65, 0.7, clock.at(4));
        assert_eq!(messages, vec![[NOTE_OFF, 60, 0], [NOTE_ON, 65, 89]]);
        assert_eq!(gate.held, Some(65));
    }

    #[test]
    fn a_fast_sweep_is_quantised_to_one_attack_per_floor() {
        let clock = TestClock::new(33);
        let mut gate = NoteGate::new(120);

        let mut attacks = Vec::new();
        // Sixteen frames is half a second, and every frame lands on a different
        // cell: a hand sliding across the keyboard at ~10 keys a second.
        for frame in 0..16 {
            for message in gate.attack(36 + frame as u8, 1.0, clock.at(frame)) {
                if message[0] == NOTE_ON {
                    attacks.push(message[1]);
                }
            }
        }

        // 16 frames is 528 ms; at a 120 ms floor that is four attacks, one every
        // four frames, so the sweep reads as an arpeggio instead of a blur.
        assert_eq!(attacks, vec![36, 40, 44, 48]);
    }

    #[test]
    fn a_zero_floor_retriggers_on_every_cell_change() {
        let clock = TestClock::new(33);
        let mut gate = NoteGate::new(0);

        assert_eq!(gate.attack(60, 1.0, clock.at(0)), vec![[NOTE_ON, 60, 127]]);
        assert_eq!(
            gate.attack(61, 1.0, clock.at(1)),
            vec![[NOTE_OFF, 60, 0], [NOTE_ON, 61, 127]]
        );
    }

    #[test]
    fn the_release_event_ends_the_note_exactly_once() {
        let clock = TestClock::new(33);
        let mut gate = NoteGate::new(DEFAULT_MIN_NOTE_MS);

        gate.attack(60, 0.8, clock.at(0));
        assert_eq!(gate.release(), vec![[NOTE_OFF, 60, 0]]);
        // The pipeline keeps running after the hand leaves; a second release must
        // not send an orphan note-off.
        assert!(gate.release().is_empty());
        assert_eq!(gate.held, None);
    }

    #[test]
    fn pinching_the_hand_silences_the_voice() {
        let clock = TestClock::new(33);
        let mut gate = NoteGate::new(DEFAULT_MIN_NOTE_MS);

        gate.attack(60, 1.0, clock.at(0));
        assert_eq!(gate.attack(60, 0.0, clock.at(1)), vec![[NOTE_OFF, 60, 0]]);
        assert_eq!(gate.held, None);
    }

    #[test]
    fn a_note_is_never_attacked_with_zero_velocity() {
        assert_eq!(volume_cc(0.001), 1);
        assert_eq!(volume_cc(1.0), 127);
    }

    #[test]
    fn events_are_ignored_until_the_engine_is_connected() {
        // No port attached: `handle` must be a no-op, not an error, so a
        // missing synth never spams the sink thread.
        let mut engine = MidiOutputEngine::new("OP-1", DEFAULT_MIN_NOTE_MS);
        assert!(engine
            .handle(&Performance::Note {
                note: 60,
                velocity: 1.0,
            })
            .is_ok());
        assert_eq!(engine.gate.held, None);
    }
}
