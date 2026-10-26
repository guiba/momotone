use crate::core::types::{GestureEvent, GesturePayload};
use crate::core::Sink;
use anyhow::{anyhow, Result};
use midir::{MidiOutput, MidiOutputConnection};
use std::sync::Mutex;

pub struct MidiOutputEngine {
    port_name: String,
    connection: Option<Mutex<MidiOutputConnection>>,
    /// The note currently held down, so the engine can send a proper note-off
    /// when the gesture ends instead of leaving it ringing forever.
    held_note: Option<u8>,
}

impl MidiOutputEngine {
    pub fn new(port_name: &str) -> Self {
        Self {
            port_name: port_name.to_string(),
            connection: None,
            held_note: None,
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
        let midi_out = MidiOutput::new("Gesture MIDI")?;
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

        let conn = midi_out.connect(port, "Gesture MIDI Rust")?;
        self.connection = Some(Mutex::new(conn));
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        // Release any held note before dropping the connection.
        let _ = self.note_off(0);
        // Connection is closed when dropped
        self.connection = None;
        Ok(())
    }

    fn handle_event(&mut self, event: &GestureEvent) -> Result<()> {
        if self.connection.is_none() {
            return Ok(()); // not started, ignore events
        }

        match &event.payload {
            GesturePayload::Note { note, velocity } => {
                // note 0 is the pipeline's "no gesture" release event.
                if *note == 0 || *velocity <= 0.0 {
                    self.note_off(0)?;
                } else {
                    // `FingerPositionStrategy` already emits MIDI numbers (36-84);
                    // the old +60 offset pushed values past 127 and wrapped them.
                    self.note_on(*note, *velocity)?;
                }
            }
            GesturePayload::Signal { .. } => {}
        }
        Ok(())
    }
}

// MIDI-specific helpers (replaces the Python logic)
impl MidiOutputEngine {
    pub fn note_on(&mut self, note: u8, velocity: f32) -> Result<()> {
        let vel = (velocity.clamp(0.0, 1.0) * 127.0) as u8;
        if vel == 0 {
            return self.note_off(note);
        }

        // Release a different held note before striking the next one, otherwise
        // every frame leaves a new note ringing.
        if let Some(previous) = self.held_note {
            if previous != note {
                self.send(&[0x80, previous, 0])?;
            }
        }
        self.send(&[0x90, note, vel])?;
        self.held_note = Some(note);
        Ok(())
    }

    /// `note == 0` means "release whatever is held" (the no-gesture event).
    pub fn note_off(&mut self, note: u8) -> Result<()> {
        let target = if note == 0 { self.held_note } else { Some(note) };
        if let Some(note) = target {
            self.send(&[0x80, note, 0])?;
            self.held_note = None;
        }
        Ok(())
    }

    pub fn control_change(&mut self, cc: u8, value: f32) -> Result<()> {
        let vel = (value.clamp(0.0, 1.0) * 127.0) as u8;
        self.send(&[0xB0, cc, vel])
    }
}

// We'll need to implement Sink for it fully properly if we want polymorphic use in Orchestrator
