use crate::core::types::{GestureEvent, GesturePayload};
use crate::core::Sink;
use anyhow::{anyhow, Result};
use midir::{MidiOutput, MidiOutputConnection};
use std::sync::Mutex;

pub struct MidiOutputEngine {
    port_name: String,
    connection: Option<Mutex<MidiOutputConnection>>,
}

impl MidiOutputEngine {
    pub fn new(port_name: &str) -> Self {
        Self {
            port_name: port_name.to_string(),
            connection: None,
        }
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
        // Connection is closed when dropped
        self.connection = None;
        Ok(())
    }

    fn handle_event(&mut self, event: &GestureEvent) -> Result<()> {
        let mut conn = match self.connection.as_mut() {
            Some(c) => c.lock().unwrap(),
            None => return Ok(()), // If not started, ignore events
        };

        match &event.payload {
            GesturePayload::Note { note, velocity } => {
                if *note == 0 {
                    // Note off for all (simplification)
                    conn.send(&[0xB0, *note as u8 + 60, 0])?; // Using 60 as base offset if needed
                } else {
                    // Note on
                    let note_val = *note;
                    let vel_val = (*velocity * 127.0) as u8;
                    conn.send(&[0x90, note_val + 60, vel_val])?; // Offset by 60 to be in range
                }
            }
            _ => {}
        }
        Ok(())
    }
}

// Helper implementation for MIDI specifically (replaces the Python logic)
impl MidiOutputEngine {
    pub fn note_on(&mut self, note: u8, velocity: f32) -> Result<()> {
        if let Some(conn_mutex) = &self.connection {
            let vel = (velocity.clamp(0.0, 1.0) * 127.0) as u8;
            // Offset to ensure MIDI range is usable (e.g., starting at C3/60)
            conn_mutex.lock().unwrap().send(&[0x90, note + 60, vel])?;
            Ok(())
        } else {
            Err(anyhow!("MIDI not connected"))
        }
    }

    pub fn note_off(&mut self, note: u8) -> Result<()> {
        if let Some(conn_mutex) = &self.connection {
            conn_mutex.lock().unwrap().send(&[0x80, note + 60, 0])?;
            Ok(())
        } else {
            Err(anyhow!("MIDI not connected"))
        }
    }

    pub fn control_change(&mut self, cc: u8, value: f32) -> Result<()> {
        if let Some(conn_mutex) = &self.connection {
            let vel = (value.clamp(0.0, 1.0) * 127.0) as u8;
            conn_mutex.lock().unwrap().send(&[0xB0, cc, vel])?;
            Ok(())
        } else {
            Err(anyhow!("MIDI not connected"))
        }
    }
}

// We'll need to implement Sink for it fully properly if we want polymorphic use in Orchestrator
