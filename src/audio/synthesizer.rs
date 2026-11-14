//! The audio output: one monophonic voice, driven by the event bus.
//!
//! The split is deliberate, and it is the one real-time audio apps get right:
//!
//! * the **control side** (`apply`) only writes targets into a handful of atomics;
//! * the **callback** reads those atomics and does every float operation.
//!
//! Nothing inside the callback locks, allocates, or touches the filesystem, so a
//! 30 Hz stream of gesture events can never delay an audio block. The voice state
//! (phase, the glided pitch, the envelope) lives in the callback, where it can be
//! mutated without any synchronisation at all - the control side never needs it.

use crate::core::instrument::Waveform;
use crate::core::types::Performance;
use crate::core::Sink;
use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

/// Time constant of the amplitude ramp upwards. A note never starts on a step;
/// the 12 ms attack is what stops a keyboard sweep from clicking.
const ATTACK_SECONDS: f32 = 0.012;
/// Time constant of the amplitude ramp downwards. Longer than the attack, because
/// a note that cuts off instantly sounds like a gate closing, and the ear reads
/// that as a click.
const RELEASE_SECONDS: f32 = 0.06;
/// Time constant of the pitch glide. The strategy retargets the voice every frame
/// (33 ms), so without a ramp a keyboard sweep jumps between notes in one sample.
const GLIDE_SECONDS: f32 = 0.015;
/// Below this the voice is inaudible, and the callback skips the oscillator.
const SILENT: f32 = 1e-4;
/// Harmonics kept for the discontinuous waveforms. A step function contains every
/// harmonic; a 44.1 kHz sample can only play up to 22.05 kHz, so anything above
/// that would alias into the audible band.
const MAX_HARMONICS: usize = 24;

/// What the control side wants the voice to be doing.
///
/// Four atomics instead of a mutex: one writer (the sink thread) and one reader
/// (the audio callback). A read is not atomic as a whole, so a block can catch a
/// half-updated target - pitch and loudness a frame apart. That is inaudible and
/// it is the standard trade-off for this pattern; a lock in the audio callback
/// would be worse.
#[derive(Debug, Default)]
struct Target {
    sounding: AtomicBool,
    /// f32 stored as its bit pattern.
    frequency: AtomicU32,
    amplitude: AtomicU32,
    waveform: AtomicU8,
}

impl Target {
    fn set(&self, frequency: f32, amplitude: f32, waveform: Waveform) {
        self.frequency.store(frequency.to_bits(), Ordering::Relaxed);
        self.amplitude.store(amplitude.to_bits(), Ordering::Relaxed);
        self.waveform.store(waveform.index(), Ordering::Relaxed);
        self.sounding.store(true, Ordering::Relaxed);
    }

    fn stop(&self) {
        self.sounding.store(false, Ordering::Relaxed);
    }

    fn is_sounding(&self) -> bool {
        self.sounding.load(Ordering::Relaxed)
    }

    fn read(&self) -> (f32, f32, Waveform) {
        (
            f32::from_bits(self.frequency.load(Ordering::Relaxed)),
            f32::from_bits(self.amplitude.load(Ordering::Relaxed)),
            Waveform::from_index(self.waveform.load(Ordering::Relaxed)),
        )
    }
}

/// A single oscillator with an attack/release envelope. The waveform is the
/// timbre: sine is the pure theremin, the others add harmonics and read more like
/// a real instrument.
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
    fn idle() -> Self {
        Self {
            target_frequency: 0.0,
            frequency: 0.0,
            target_amplitude: 0.0,
            amplitude: 0.0,
            phase: 0.0,
            waveform: Waveform::Sine,
        }
    }

    fn set_target(&mut self, frequency: f32, amplitude: f32, waveform: Waveform) {
        self.target_frequency = frequency;
        self.target_amplitude = amplitude;
        self.waveform = waveform;
    }

    fn release(&mut self) {
        self.target_amplitude = 0.0;
    }

    fn is_silent(&self) -> bool {
        self.target_amplitude <= SILENT && self.amplitude <= SILENT
    }

    fn next_sample(&mut self, sample_rate: f32) -> f32 {
        // One-pole ramps: never overshoot, one multiply per sample, and the phase
        // stays continuous when the hand moves to another key. The envelope
        // attacks faster than it releases, which is how a struck note sounds.
        let envelope = if self.target_amplitude > self.amplitude {
            ATTACK_SECONDS
        } else {
            RELEASE_SECONDS
        };
        self.frequency +=
            (self.target_frequency - self.frequency) * (1.0 / (sample_rate * GLIDE_SECONDS));
        self.amplitude +=
            (self.target_amplitude - self.amplitude) * (1.0 / (sample_rate * envelope));

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

/// MIDI number -> hertz. A4 = 69 is 440 Hz by convention.
fn note_frequency(note: u8) -> f32 {
    440.0 * 2.0f32.powf((note as i32 - 69) as f32 / 12.0)
}

/// `cpal::Stream` is `!Send`. Holding it is the point - dropping it stops the
/// audio - and the stream is only ever touched by the thread that owns this sink.
struct StreamWrapper(#[allow(dead_code)] cpal::Stream);

unsafe impl Send for StreamWrapper {}

pub struct Synthesizer {
    /// Master gain applied to every performance amplitude (`--amplitude`).
    gain: f32,
    target: Arc<Target>,
    stream: Option<StreamWrapper>,
}

impl Synthesizer {
    /// The sample rate is not a parameter: it is whatever the output device
    /// reports, and the oscillator must integrate against exactly that.
    pub fn new(gain: f32) -> Self {
        Self {
            gain: gain.clamp(0.0, 1.0),
            target: Arc::new(Target::default()),
            stream: None,
        }
    }

    /// Apply one frame's performance. This only ever sets targets - it never
    /// touches the stream, so it is safe to call at the frame rate.
    pub fn apply(&self, performance: &Performance) -> Result<()> {
        let waveform = Waveform::from_index(self.target.waveform.load(Ordering::Relaxed));
        match performance {
            Performance::Signal {
                frequency,
                amplitude,
            } => self.target.set(
                *frequency,
                (amplitude * self.gain).clamp(0.0, 1.0),
                waveform,
            ),
            Performance::Note { note, velocity } => {
                if *velocity <= 0.0 {
                    // A pinched hand is silent: release instead of attacking a
                    // note nobody is asking for.
                    self.target.stop();
                } else {
                    let amplitude = (velocity * self.gain).clamp(0.0, 1.0);
                    self.target.set(note_frequency(*note), amplitude, waveform);
                }
            }
            Performance::Release => self.target.stop(),
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
        tracing::info!(
            "device stream: {} Hz, {} channels",
            config.sample_rate,
            config.channels
        );

        let target = self.target.clone();
        let sample_rate = config.sample_rate as f32;
        let channels = config.channels as usize;

        // The voice belongs to the callback: nothing else can reach it, so it
        // needs no lock and no allocation per block.
        let mut voice = Voice::idle();
        let stream = device.build_output_stream(
            config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if target.is_sounding() {
                    let (frequency, amplitude, waveform) = target.read();
                    voice.set_target(frequency, amplitude, waveform);
                } else {
                    voice.release();
                }

                if voice.is_silent() {
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }
                    return;
                }

                for frame in data.chunks_mut(channels) {
                    let value = voice.next_sample(sample_rate);
                    for sample in frame.iter_mut() {
                        *sample = value;
                    }
                }
            },
            |err| tracing::error!("audio stream error: {err}"),
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

    fn handle(&mut self, performance: &Performance) -> Result<()> {
        self.apply(performance)
    }

    /// Change the timbre. The callback reads the timbre from the target on every
    /// block, so a picker click is audible on the note being held, not only on the
    /// next one.
    fn set_waveform(&mut self, waveform: Waveform) {
        self.target
            .waveform
            .store(waveform.index(), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_note_sets_the_target_and_a_release_silences_it() {
        let synth = Synthesizer::new(1.0);

        synth
            .apply(&Performance::Note {
                note: 60,
                velocity: 0.8,
            })
            .unwrap();
        let (frequency, amplitude, waveform) = synth.target.read();
        assert!(
            (frequency - 261.6).abs() < 0.1,
            "MIDI 60 is middle C, got {frequency}"
        );
        assert!((amplitude - 0.8).abs() < 1e-6);
        assert_eq!(waveform, Waveform::Sine);

        // The pipeline sends this the frame after the hand leaves: without it the
        // oscillator would latch and keep sounding forever.
        synth.apply(&Performance::Release).unwrap();
        assert!(!synth.target.is_sounding());
    }

    #[test]
    fn a_continuous_signal_retargets_the_same_voice() {
        let synth = Synthesizer::new(1.0);
        synth
            .apply(&Performance::Signal {
                frequency: 220.0,
                amplitude: 0.5,
            })
            .unwrap();
        synth
            .apply(&Performance::Signal {
                frequency: 440.0,
                amplitude: 0.8,
            })
            .unwrap();

        let (frequency, amplitude, _) = synth.target.read();
        assert!((frequency - 440.0).abs() < 1e-6);
        assert!((amplitude - 0.8).abs() < 1e-6);
    }

    #[test]
    fn zero_velocity_is_silent() {
        let synth = Synthesizer::new(1.0);
        synth
            .apply(&Performance::Note {
                note: 60,
                velocity: 0.0,
            })
            .unwrap();
        assert!(!synth.target.is_sounding());
    }

    #[test]
    fn gain_scales_the_amplitude() {
        let synth = Synthesizer::new(0.25);
        synth
            .apply(&Performance::Note {
                note: 60,
                velocity: 1.0,
            })
            .unwrap();
        let (_, amplitude, _) = synth.target.read();
        assert!((amplitude - 0.25).abs() < 1e-6);
    }

    #[test]
    fn a_waveform_change_reaches_the_note_being_held() {
        let mut synth = Synthesizer::new(1.0);
        synth
            .apply(&Performance::Note {
                note: 60,
                velocity: 0.5,
            })
            .unwrap();
        assert_eq!(synth.target.read().2, Waveform::Sine);

        synth.set_waveform(Waveform::Saw);
        assert_eq!(synth.target.read().2, Waveform::Saw);
    }

    #[test]
    fn the_envelope_ramps_up_and_down_instead_of_stepping() {
        let mut voice = Voice::idle();
        voice.set_target(440.0, 1.0, Waveform::Sine);

        // After one attack time constant the amplitude is near 63 % of the
        // target: it started from zero, so it has not reached it yet.
        for _ in 0..(44100.0 * ATTACK_SECONDS) as usize {
            voice.next_sample(44100.0);
        }
        assert!(
            voice.amplitude > 0.5,
            "attack only reached {}",
            voice.amplitude
        );
        assert!(voice.amplitude < 1.0, "the attack is a ramp, not a step");

        voice.release();
        for _ in 0..(44100.0 * RELEASE_SECONDS) as usize {
            voice.next_sample(44100.0);
        }
        assert!(
            voice.amplitude < 0.4,
            "release only reached {}",
            voice.amplitude
        );

        // A released voice fades to nothing, and then the callback skips it.
        for _ in 0..(44100.0 * RELEASE_SECONDS * 10.0) as usize {
            voice.next_sample(44100.0);
        }
        assert!(voice.is_silent(), "the voice went silent");
    }

    #[test]
    fn a_frequency_change_glides_instead_of_jumping() {
        let mut voice = Voice::idle();
        voice.set_target(261.6, 1.0, Waveform::Sine);
        // Five time constants: long enough for the voice to settle on the note it
        // is already playing, which is the state a key change starts from.
        for _ in 0..(44100.0 * GLIDE_SECONDS * 5.0) as usize {
            voice.next_sample(44100.0);
        }
        let held = voice.frequency;
        voice.set_target(1046.5, 1.0, Waveform::Sine);

        // One sample at 44.1 kHz is a 1/44100 s step against a 15 ms time
        // constant, so the pitch has barely moved - it certainly has not arrived.
        voice.next_sample(44100.0);
        assert!(voice.frequency > held, "the pitch moves towards the target");
        assert!(
            voice.frequency < held + 40.0,
            "the pitch must not snap: it moved to {}",
            voice.frequency
        );
    }

    #[test]
    fn the_oscillator_stays_inside_the_output_range() {
        for waveform in Waveform::all() {
            let mut voice = Voice::idle();
            voice.set_target(440.0, 1.0, waveform);
            for _ in 0..2000 {
                let sample = voice.next_sample(44100.0);
                assert!(
                    sample.abs() <= 1.0,
                    "{} produced {} at phase {}",
                    waveform.name(),
                    sample,
                    voice.phase
                );
            }
        }
    }

    #[test]
    fn band_limiting_keeps_the_square_close_to_a_square() {
        let mut voice = Voice::idle();
        voice.set_target(100.0, 1.0, Waveform::Square);
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
