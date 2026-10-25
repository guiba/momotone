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
struct StreamWrapper(cpal::Stream);

unsafe impl Send for StreamWrapper {}
unsafe impl Sync for StreamWrapper {}

pub struct Synthesizer {
    sample_rate: f32,
    channels: u16,
    state: Arc<Mutex<SynthesizerState>>,
    stream: Option<StreamWrapper>,
}

impl Synthesizer {
    pub fn new(sample_rate: f32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
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
                if state.voices.is_empty() {
                    state.voices.push(Voice::new(*frequency, *amplitude));
                } else {
                    for voice in state.voices.iter_mut() {
                        voice.frequency = *frequency;
                        voice.amplitude = *amplitude;
                    }
                }
            }
            GesturePayload::Note { note, velocity } => {
                if *note == 0 {
                    state.voices.clear();
                } else {
                    let freq = 440.0 * 2.0f32.powf((*note as i32 - 69) as f32 / 12.0);
                    if state.voices.is_empty() {
                        state.voices.push(Voice::new(freq, *velocity));
                    } else {
                        state.voices[0].frequency = freq;
                        state.voices[0].amplitude = *velocity;
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
