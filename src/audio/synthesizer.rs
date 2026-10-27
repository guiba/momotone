use crate::core::types::{GestureEvent, GesturePayload};
use crate::core::Sink;
use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

/// A single sine wave oscillator.
#[derive(Clone)]
struct Voice {
    frequency: f32,
    amplitude: f32,
    phase: f32,
}

impl Voice {
    fn new(frequency: f32, amplitude: f32) -> Self {
        Self {
            frequency,
            amplitude,
            phase: 0.0,
        }
    }

    fn next_sample(&mut self, sample_rate: f32) -> f32 {
        let val = self.amplitude * (self.phase * 2.0 * std::f32::consts::PI).sin();
        self.phase += self.frequency / sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        val
    }
}

/// Shared state between the main thread and the audio callback.
struct SynthesizerState {
    voices: Vec<Voice>,
}

/// Wrapper to allow cpal::Stream (which is !Sync) to be used inside a Sync struct.
/// This is safe because we only access it through &mut self in Sink methods,
/// ensuring exclusive access if the caller respects Rust's borrowing rules.
/// The stream is never read - holding it is the point, dropping it stops the audio.
struct StreamWrapper(#[allow(dead_code)] cpal::Stream);

unsafe impl Send for StreamWrapper {}
unsafe impl Sync for StreamWrapper {}

pub struct Synthesizer {
    sample_rate: f32,
    channels: u16,
    /// Master gain applied to every event amplitude (`--amplitude`).
    gain: f32,
    state: Arc<Mutex<SynthesizerState>>,
    stream: Option<StreamWrapper>,
}

impl Synthesizer {
    pub fn new(sample_rate: f32, channels: u16, gain: f32) -> Self {
        Self {
            sample_rate,
            channels,
            gain: gain.clamp(0.0, 1.0),
            state: Arc::new(Mutex::new(SynthesizerState { voices: Vec::new() })),
            stream: None,
        }
    }

    pub fn update_from_event(&self, event: &GestureEvent) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| anyhow!("Mutex lock error"))?;
        match &event.payload {
            GesturePayload::Signal {
                frequency,
                amplitude,
            } => {
                let amplitude = (amplitude * self.gain).clamp(0.0, 1.0);
                if state.voices.is_empty() {
                    state.voices.push(Voice::new(*frequency, amplitude));
                } else {
                    for voice in state.voices.iter_mut() {
                        voice.frequency = *frequency;
                        voice.amplitude = amplitude;
                    }
                }
            }
            GesturePayload::Note { note, velocity } => {
                // note 0 is the pipeline's "no gesture" release: it must silence
                // the oscillator, otherwise the last note latches and keeps
                // sounding after the hand leaves the frame.
                if *note == 0 || *velocity <= 0.0 {
                    state.voices.clear();
                } else {
                    let amplitude = (velocity * self.gain).clamp(0.0, 1.0);
                    let freq = 440.0 * 2.0f32.powf((*note as i32 - 69) as f32 / 12.0);
                    if state.voices.is_empty() {
                        state.voices.push(Voice::new(freq, amplitude));
                    } else {
                        state.voices[0].frequency = freq;
                        state.voices[0].amplitude = amplitude;
                    }
                }
            }
        }
        Ok(())
    }
}

impl Sink for Synthesizer {
    fn start(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("No audio output device found"))?;
        let config: cpal::StreamConfig = device.default_output_config()?.into();

        let state_ref = self.state.clone();
        let sample_rate = self.sample_rate;
        let channels = self.channels as usize;

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if let Ok(mut state) = state_ref.lock() {
                    for frame in data.chunks_mut(channels) {
                        let mut mixed_sample = 0.0;
                        for voice in state.voices.iter_mut() {
                            mixed_sample += voice.next_sample(sample_rate);
                        }
                        mixed_sample = mixed_sample.clamp(-1.0, 1.0);
                        for sample in frame.iter_mut() {
                            *sample = mixed_sample;
                        }
                    }
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?;

        stream.play()?;
        self.stream = Some(StreamWrapper(stream));
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        self.stream.take();
        Ok(())
    }

    fn handle_event(&mut self, event: &GestureEvent) -> Result<()> {
        self.update_from_event(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_note_starts_a_voice_and_the_release_kills_it() {
        let synth = Synthesizer::new(44100.0, 1, 1.0);

        synth
            .update_from_event(&GestureEvent::new_note(60, 0.8, "Finger Position"))
            .unwrap();
        assert_eq!(synth.state.lock().unwrap().voices.len(), 1);

        // The pipeline sends this the frame after the hand leaves: without it the
        // oscillator would latch and keep sounding forever.
        synth.update_from_event(&GestureEvent::no_gesture()).unwrap();
        assert!(synth.state.lock().unwrap().voices.is_empty());
    }

    #[test]
    fn zero_velocity_is_silent() {
        let synth = Synthesizer::new(44100.0, 1, 1.0);
        synth
            .update_from_event(&GestureEvent::new_note(60, 0.0, "pinched"))
            .unwrap();
        assert!(synth.state.lock().unwrap().voices.is_empty());
    }

    #[test]
    fn gain_scales_the_amplitude() {
        let synth = Synthesizer::new(44100.0, 1, 0.25);
        synth
            .update_from_event(&GestureEvent::new_note(60, 1.0, "Finger Position"))
            .unwrap();
        let voice = &synth.state.lock().unwrap().voices[0];
        assert!((voice.amplitude - 0.25).abs() < 1e-6);
        assert!((voice.frequency - 261.6).abs() < 0.1);
    }
}
