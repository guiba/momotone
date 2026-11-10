use crate::core::instrument::Waveform;
use crate::core::types::{GestureEvent, GesturePayload};
use crate::core::Sink;
use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

/// Time constant of the frequency and amplitude glide, in seconds. The strategy
/// retargets the voice every frame (33 ms); without a ramp the oscillator jumps
/// straight to the new pitch and a keyboard sweep clicks.
const GLIDE_SECONDS: f32 = 0.015;
/// Harmonics kept for the discontinuous waveforms. A step function contains every
/// harmonic; a 44.1 kHz sample can only play up to 22.05 kHz, so anything above
/// that would alias into the audible band.
const MAX_HARMONICS: usize = 24;

/// A single oscillator. The waveform is the timbre: sine is the pure theremin,
/// the others add harmonics and read more like a real instrument.
#[derive(Clone)]
struct Voice {
    /// The pitch the hand is asking for.
    target_frequency: f32,
    /// The pitch actually being generated, glided towards the target.
    frequency: f32,
    target_amplitude: f32,
    amplitude: f32,
    /// Normalised position in the cycle, 0..1.
    phase: f32,
    waveform: Waveform,
}

impl Voice {
    fn new(frequency: f32, amplitude: f32, waveform: Waveform) -> Self {
        Self {
            target_frequency: frequency,
            frequency,
            target_amplitude: amplitude,
            amplitude,
            phase: 0.0,
            waveform,
        }
    }

    fn set_target(&mut self, frequency: f32, amplitude: f32) {
        self.target_frequency = frequency;
        self.target_amplitude = amplitude;
    }

    fn next_sample(&mut self, sample_rate: f32) -> f32 {
        // One-pole glide: never overshoots, costs one multiply per sample, and
        // keeps the phase continuous when the hand moves to another key.
        let glide = 1.0 / (sample_rate * GLIDE_SECONDS);
        self.frequency += (self.target_frequency - self.frequency) * glide;
        self.amplitude += (self.target_amplitude - self.amplitude) * glide;

        let value = self
            .waveform
            .sample_band_limited(self.phase, harmonic_count(self.frequency, sample_rate));
        self.phase += self.frequency / sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        soft_clip(self.amplitude * value)
    }
}

/// How many harmonics fit below Nyquist at this pitch. A 200 Hz square can carry
/// 110 harmonics; the cap keeps the additive sum cheap on a CPU.
fn harmonic_count(frequency: f32, sample_rate: f32) -> usize {
    if frequency <= 0.0 {
        return 1;
    }
    ((sample_rate * 0.5 / frequency).floor() as usize).clamp(1, MAX_HARMONICS)
}

/// Saturating limiter instead of `clamp(-1, 1)`. Hard clamping a summed wave turns
/// an overload into a distorted, harmonic-rich one; this flattens it gently and
/// leaves everything under 0.9 untouched.
fn soft_clip(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude <= 0.9 {
        return value;
    }
    value.signum() * (0.9 + 0.1 * ((magnitude - 0.9) / 0.1).tanh())
}

/// Shared state between the control thread and the audio callback.
struct SynthesizerState {
    voices: Vec<Voice>,
    /// Timbre applied to the next voice that starts.
    waveform: Waveform,
}

/// Wrapper to allow cpal::Stream (which is !Send) to be used inside a Send struct.
/// This is safe because the stream is only ever touched from the thread that owns
/// the sink: `handle_event` writes the shared state, the stream itself is never
/// read - holding it is the point, dropping it stops the audio.
struct StreamWrapper(#[allow(dead_code)] cpal::Stream);

unsafe impl Send for StreamWrapper {}

pub struct Synthesizer {
    /// Master gain applied to every event amplitude (`--amplitude`).
    gain: f32,
    state: Arc<Mutex<SynthesizerState>>,
    stream: Option<StreamWrapper>,
}

impl Synthesizer {
    /// The sample rate is not a parameter: it is whatever the output device
    /// reports, and the oscillator must integrate against exactly that.
    pub fn new(gain: f32) -> Self {
        Self {
            gain: gain.clamp(0.0, 1.0),
            state: Arc::new(Mutex::new(SynthesizerState {
                voices: Vec::new(),
                waveform: Waveform::Sine,
            })),
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
                let waveform = state.waveform;
                if state.voices.is_empty() {
                    state
                        .voices
                        .push(Voice::new(*frequency, amplitude, waveform));
                } else {
                    for voice in state.voices.iter_mut() {
                        voice.set_target(*frequency, amplitude);
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
                    let waveform = state.waveform;
                    if state.voices.is_empty() {
                        state.voices.push(Voice::new(freq, amplitude, waveform));
                    } else {
                        // A held note keeps its envelope; only the loudness moves.
                        state.voices[0].set_target(freq, amplitude);
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
        println!(
            "[Audio] device stream: {} Hz, {} channels",
            config.sample_rate, config.channels
        );

        let state_ref = self.state.clone();
        let sample_rate = config.sample_rate as f32;
        let channels = config.channels as usize;

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if let Ok(mut state) = state_ref.lock() {
                    // Voices are averaged rather than summed, so N voices can never
                    // exceed the amplitude the gain asked for.
                    let scale = 1.0 / state.voices.len().max(1) as f32;
                    for frame in data.chunks_mut(channels) {
                        let mut mixed = 0.0;
                        for voice in state.voices.iter_mut() {
                            mixed += voice.next_sample(sample_rate);
                        }
                        let mixed = soft_clip(mixed * scale);
                        for sample in frame.iter_mut() {
                            *sample = mixed;
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

    /// Change the timbre. Applied to every voice already sounding as well, so a
    /// picker click is audible on the note being held, not only on the next one.
    fn set_waveform(&mut self, waveform: Waveform) {
        if let Ok(mut state) = self.state.lock() {
            state.waveform = waveform;
            for voice in state.voices.iter_mut() {
                voice.waveform = waveform;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_note_starts_a_voice_and_the_release_kills_it() {
        let synth = Synthesizer::new(1.0);

        synth
            .update_from_event(&GestureEvent::new_note(60, 0.8, "Finger Position"))
            .unwrap();
        assert_eq!(synth.state.lock().unwrap().voices.len(), 1);

        // The pipeline sends this the frame after the hand leaves: without it the
        // oscillator would latch and keep sounding forever.
        synth
            .update_from_event(&GestureEvent::no_gesture())
            .unwrap();
        assert!(synth.state.lock().unwrap().voices.is_empty());
    }

    #[test]
    fn a_signal_glides_one_voice_instead_of_restarting_it() {
        let synth = Synthesizer::new(1.0);
        synth
            .update_from_event(&GestureEvent::new_signal(220.0, 0.5, "Theremin"))
            .unwrap();
        synth
            .update_from_event(&GestureEvent::new_signal(440.0, 0.8, "Theremin"))
            .unwrap();

        let state = synth.state.lock().unwrap();
        assert_eq!(state.voices.len(), 1, "a continuous signal rides one voice");
        assert!((state.voices[0].target_frequency - 440.0).abs() < 1e-6);
        assert!((state.voices[0].target_amplitude - 0.8).abs() < 1e-6);
    }

    #[test]
    fn zero_velocity_is_silent() {
        let synth = Synthesizer::new(1.0);
        synth
            .update_from_event(&GestureEvent::new_note(60, 0.0, "pinched"))
            .unwrap();
        assert!(synth.state.lock().unwrap().voices.is_empty());
    }

    #[test]
    fn gain_scales_the_amplitude() {
        let synth = Synthesizer::new(0.25);
        synth
            .update_from_event(&GestureEvent::new_note(60, 1.0, "Finger Position"))
            .unwrap();
        let voice = &synth.state.lock().unwrap().voices[0];
        assert!((voice.amplitude - 0.25).abs() < 1e-6);
        assert!((voice.frequency - 261.6).abs() < 0.1);
    }

    #[test]
    fn a_waveform_change_reaches_the_voice_already_sounding() {
        let mut synth = Synthesizer::new(1.0);
        synth
            .update_from_event(&GestureEvent::new_note(60, 0.5, "Finger Position"))
            .unwrap();
        assert_eq!(
            synth.state.lock().unwrap().voices[0].waveform,
            Waveform::Sine
        );

        synth.set_waveform(Waveform::Saw);
        {
            let state = synth.state.lock().unwrap();
            assert_eq!(state.waveform, Waveform::Saw);
            assert_eq!(state.voices[0].waveform, Waveform::Saw);
        }

        // A voice started after the change inherits it.
        synth
            .update_from_event(&GestureEvent::no_gesture())
            .unwrap();
        synth
            .update_from_event(&GestureEvent::new_note(69, 1.0, "Finger Position"))
            .unwrap();
        assert_eq!(
            synth.state.lock().unwrap().voices[0].waveform,
            Waveform::Saw
        );
    }

    #[test]
    fn a_frequency_change_glides_instead_of_jumping() {
        let mut voice = Voice::new(261.6, 1.0, Waveform::Sine);
        voice.set_target(1046.5, 1.0);
        // One sample at 44.1 kHz is a 1/44100 s step against a 15 ms time
        // constant, so the pitch has barely moved - it certainly has not arrived.
        voice.next_sample(44100.0);
        assert!(
            voice.frequency > 261.6,
            "the pitch moves towards the target"
        );
        assert!(
            voice.frequency < 300.0,
            "the pitch must not snap: it moved to {}",
            voice.frequency
        );
    }

    #[test]
    fn the_oscillator_stays_inside_the_output_range() {
        for waveform in Waveform::all() {
            let mut voice = Voice::new(440.0, 1.0, waveform);
            for _ in 0..2000 {
                let sample = voice.next_sample(44100.0);
                assert!(
                    sample.abs() <= 1.0,
                    "{} produced {} at sample {}",
                    waveform.name(),
                    sample,
                    voice.phase
                );
            }
        }
    }

    #[test]
    fn band_limiting_keeps_the_square_close_to_a_square() {
        let mut voice = Voice::new(100.0, 1.0, Waveform::Square);
        let mut extremes: f32 = 0.0;
        let mut total: f32 = 0.0;
        for _ in 0..4410 {
            // 100 samples per cycle at 44.1 kHz: 44 cycles.
            let sample = voice.next_sample(44100.0).abs();
            extremes = extremes.max(sample);
            total += sample;
        }
        // A square spends most of its cycle near full amplitude, unlike a sine.
        assert!(total / 4410.0 > 0.7, "square averaged {}", total / 4410.0);
        assert!(extremes <= 1.0);
    }

    #[test]
    fn the_soft_limiter_flattens_without_chopping() {
        assert!(
            (soft_clip(0.5) - 0.5).abs() < 1e-6,
            "small signals pass through"
        );
        assert!(soft_clip(5.0) <= 1.0);
        assert!(soft_clip(-5.0) >= -1.0);
        assert!(soft_clip(0.95) > soft_clip(0.9), "the knee is monotonic");
    }
}
