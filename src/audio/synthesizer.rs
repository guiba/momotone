//! The audio output: one monophonic voice through a small effect chain, driven by the
//! event bus.
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
//!
//! The chain - `glide -> vibrato -> oscillator -> lowpass -> delay -> soft clip` - is
//! read once per audio block, not per sample, so the effect parameters are as cheap to
//! change mid-note as the timbre is.

use crate::core::effects::{
    cents_to_hz, Effect, EffectMask, Modulation, DELAY_MAX_MS, NEUTRAL_CUTOFF_HZ,
};
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
/// Nine atomics instead of a mutex: one writer (the sink thread) and one reader
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
    /// One bit per enabled stage in the chain.
    enabled: AtomicU8,
    cutoff_hz: AtomicU32,
    vibrato_hz: AtomicU32,
    vibrato_cents: AtomicU32,
    delay_ms: AtomicU32,
    feedback: AtomicU32,
}

impl Target {
    fn set(&self, frequency: f32, amplitude: f32, waveform: Waveform) {
        self.frequency.store(frequency.to_bits(), Ordering::Relaxed);
        self.amplitude.store(amplitude.to_bits(), Ordering::Relaxed);
        self.waveform.store(waveform.index(), Ordering::Relaxed);
        self.sounding.store(true, Ordering::Relaxed);
    }

    /// The chain parameters live here too, so the callback reads them in one pass
    /// per block rather than locking on every toggle.
    fn set_modulation(&self, modulation: &Modulation) {
        self.enabled
            .store(modulation.enabled.bits(), Ordering::Relaxed);
        self.cutoff_hz
            .store(modulation.cutoff_hz.to_bits(), Ordering::Relaxed);
        self.vibrato_hz
            .store(modulation.vibrato_hz.to_bits(), Ordering::Relaxed);
        self.vibrato_cents
            .store(modulation.vibrato_cents.to_bits(), Ordering::Relaxed);
        self.delay_ms
            .store(modulation.delay_ms.to_bits(), Ordering::Relaxed);
        self.feedback
            .store(modulation.feedback.to_bits(), Ordering::Relaxed);
    }

    /// Snapshot of the chain for one audio block. Read once per block, not per
    /// sample: 44100 atomic loads a second would cost more than the filter.
    fn chain(&self) -> Chain {
        let mask = EffectMask::from_bits(self.enabled.load(Ordering::Relaxed));
        Chain {
            lowpass: mask.is_on(Effect::Lowpass),
            vibrato: mask.is_on(Effect::Vibrato),
            delay: mask.is_on(Effect::Delay),
            // A fresh target defaults to zero, which would be a closed filter. Only
            // enabled stages read these, but the clamp keeps the arithmetic sane if
            // a stage is switched on before the pipeline has ever computed a level.
            cutoff_hz: f32::from_bits(self.cutoff_hz.load(Ordering::Relaxed)).max(1.0),
            vibrato_hz: f32::from_bits(self.vibrato_hz.load(Ordering::Relaxed)),
            vibrato_cents: f32::from_bits(self.vibrato_cents.load(Ordering::Relaxed)),
            delay_ms: f32::from_bits(self.delay_ms.load(Ordering::Relaxed)),
            feedback: f32::from_bits(self.feedback.load(Ordering::Relaxed)),
        }
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

/// The chain as one block reads it. Cheap to copy, no allocation, and the
/// `bool`s mean a disabled stage is a branch rather than a multiply by one.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Chain {
    lowpass: bool,
    vibrato: bool,
    delay: bool,
    cutoff_hz: f32,
    vibrato_hz: f32,
    vibrato_cents: f32,
    delay_ms: f32,
    feedback: f32,
}

impl Chain {
    /// Nothing enabled: the voice behaves exactly as it did before the chain existed.
    fn idle() -> Self {
        Chain {
            lowpass: false,
            vibrato: false,
            delay: false,
            cutoff_hz: NEUTRAL_CUTOFF_HZ,
            vibrato_hz: 0.0,
            vibrato_cents: 0.0,
            delay_ms: 0.0,
            feedback: 0.0,
        }
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
    /// The chain this voice is running, refreshed once per audio block.
    chain: Chain,
    /// One-pole filter state, and the cutoff its coefficient was derived from.
    filter_y: f32,
    filter_cutoff: f32,
    filter_k: f32,
    /// LFO position for the vibrato, 0..1.
    lfo_phase: f32,
    /// Delay line, sized when the stream opens so the callback never allocates.
    delay: Vec<f32>,
    delay_write: usize,
    /// Magnitude of the last sample the delay stage produced. An echo outlives the
    /// note that made it, so "silent" has to mean the ring is empty too, not just
    /// that the envelope has closed.
    echo_tail: f32,
}

impl Voice {
    fn idle(sample_rate: f32) -> Self {
        Self {
            target_frequency: 0.0,
            frequency: 0.0,
            target_amplitude: 0.0,
            amplitude: 0.0,
            phase: 0.0,
            waveform: Waveform::Sine,
            chain: Chain::idle(),
            filter_y: 0.0,
            filter_cutoff: NEUTRAL_CUTOFF_HZ,
            filter_k: 1.0,
            lfo_phase: 0.0,
            // Roughly 106 KB at 44.1 kHz: allocated once, before the stream plays,
            // and never reallocated by the callback.
            delay: vec![0.0; (DELAY_MAX_MS * sample_rate / 1000.0) as usize + 1],
            delay_write: 0,
            echo_tail: 0.0,
        }
    }

    fn set_chain(&mut self, chain: Chain) {
        self.chain = chain;
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
        self.target_amplitude <= SILENT && self.amplitude <= SILENT && self.echo_tail <= SILENT
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

        // Vibrato rides on top of the glided pitch rather than inside the glide: the
        // glide is a one-pole ramp towards the target, and an LFO summed before it
        // would simply be smoothed away. `pitch` is local, so the wobble cannot leak
        // into `self.frequency` and drift the note.
        let pitch = if self.chain.vibrato && self.chain.vibrato_cents > 0.0 {
            self.lfo_phase += self.chain.vibrato_hz / sample_rate;
            if self.lfo_phase >= 1.0 {
                self.lfo_phase -= 1.0;
            }
            self.frequency
                + cents_to_hz(self.chain.vibrato_cents, self.frequency)
                    * (self.lfo_phase * std::f32::consts::TAU).sin()
        } else {
            self.frequency
        };

        let value = self
            .waveform
            .sample_band_limited(self.phase, harmonic_count(pitch, sample_rate));
        self.phase += pitch / sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        let value = self.amplitude * value;

        let value = if self.chain.lowpass {
            let k = self.filter_coefficient(sample_rate);
            one_pole(&mut self.filter_y, value, k)
        } else {
            // Track the signal while the stage is off, so switching it on mid-note
            // starts from the current value rather than a stale one and clicks.
            self.filter_y = value;
            value
        };

        let taps = (self.chain.delay_ms * sample_rate / 1000.0) as usize;
        let value = if self.chain.delay && taps >= 1 {
            comb(
                &mut self.delay,
                &mut self.delay_write,
                value,
                taps,
                self.chain.feedback,
            )
        } else {
            // Keep the ring warm with the dry signal. A stage that was off stores
            // nothing, so switching it back on after a long silence would otherwise
            // replay whatever happened to be sitting at the tap position.
            self.delay[self.delay_write] = value;
            self.delay_write = (self.delay_write + 1) % self.delay.len();
            value
        };
        // The echo is the only part of the chain that can still be moving after the
        // envelope has closed, so it is what decides whether the voice is done.
        self.echo_tail = value.abs();

        soft_clip(value)
    }

    /// `k` for the one-pole at the current cutoff. Recomputed only when the cutoff
    /// moved, because an `exp` per sample would cost more than the filter.
    fn filter_coefficient(&mut self, sample_rate: f32) -> f32 {
        let cutoff = self.chain.cutoff_hz;
        if (cutoff - self.filter_cutoff).abs() > 1.0 {
            self.filter_cutoff = cutoff;
            self.filter_k = 1.0 - (-(std::f32::consts::TAU * cutoff / sample_rate)).exp();
        }
        self.filter_k
    }
}

/// One-pole lowpass: `y += (x - y) * k`. Two multiplies and a subtract, no state
/// beyond the previous output.
fn one_pole(state: &mut f32, sample: f32, k: f32) -> f32 {
    *state += (sample - *state) * k;
    *state
}

/// A comb delay: `y[n] = x[n] + feedback * y[n - taps]`. The ring holds the stage's
/// *output*, not its input, which is what makes feedback repeat: each tap is the
/// previous echo, so the tail decays geometrically instead of echoing once.
/// The ring is written after the tap is read, so `taps = 0` cannot feed the current
/// sample into itself.
fn comb(buffer: &mut [f32], write: &mut usize, sample: f32, taps: usize, feedback: f32) -> f32 {
    let length = buffer.len();
    let taps = taps.min(length - 1);
    let read = (*write + length - taps) % length;
    let out = sample + feedback * buffer[read];
    buffer[*write] = out;
    *write = (*write + 1) % length;
    out
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
        let mut voice = Voice::idle(sample_rate);
        let stream = device.build_output_stream(
            config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // One chain read per block, not per sample: the whole effect chain is
                // a handful of atomic loads and a copy of a small struct.
                voice.set_chain(target.chain());
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

    /// Change the effect chain. Like the timbre, the callback reads it from the
    /// target every block, so a toggle is audible on the note being held rather than
    /// on the next one.
    fn set_modulation(&mut self, modulation: &Modulation) {
        self.target.set_modulation(modulation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

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
        let mut voice = Voice::idle(44100.0);
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
        let mut voice = Voice::idle(44100.0);
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
            let mut voice = Voice::idle(44100.0);
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
        let mut voice = Voice::idle(44100.0);
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

    #[test]
    /// The one-pole keeps what is below its cutoff and throws away what is above it.
    fn a_closed_filter_darkens_a_note_more_than_an_open_one() {
        let rms_at = |cutoff: f32| {
            let mut voice = Voice::idle(44100.0);
            voice.set_chain(Chain {
                lowpass: true,
                cutoff_hz: cutoff,
                ..Chain::idle()
            });
            voice.set_target(4000.0, 1.0, Waveform::Sine);
            let mut sum = 0.0;
            // Half a second: long enough for the attack and the filter to settle.
            for _ in 11025..22050 {
                let sample = voice.next_sample(44100.0);
                sum += sample * sample;
            }
            (sum / 11025.0).sqrt()
        };

        let dark = rms_at(500.0);
        let bright = rms_at(8000.0);
        assert!(
            dark < bright * 0.5,
            "4 kHz through a 500 Hz pole measured {dark}, through 8 kHz {bright}"
        );
    }

    #[test]
    /// Parked at its neutral value the stage is a pass-through, so switching it on at
    /// the top of the band is not a discontinuity in the sound.
    fn a_neutral_cutoff_is_transparent() {
        let level_with = |chain: Chain| {
            let mut voice = Voice::idle(44100.0);
            voice.set_chain(chain);
            voice.set_target(440.0, 0.8, Waveform::Sine);
            let mut total = 0.0;
            for _ in 4410..8820 {
                total += voice.next_sample(44100.0).abs();
            }
            total
        };
        let open = level_with(Chain::idle());
        let neutral = level_with(Chain {
            lowpass: true,
            cutoff_hz: NEUTRAL_CUTOFF_HZ,
            ..Chain::idle()
        });
        assert!(
            (neutral - open).abs() / open < 0.01,
            "a 12 kHz pole over a 440 Hz note moved the level by {}",
            (neutral - open) / open
        );
    }

    #[test]
    /// The comb taps the sample `taps` frames back. Nothing exists before the tap,
    /// so an impulse does not appear until the delay has elapsed - and the tap is
    /// scaled by the feedback, which is what makes the tail decay.
    fn comb_echoes_an_impulse_after_the_configured_delay() {
        let mut buffer = vec![0.0; 2000];
        let mut write = 0usize;
        let mut out = Vec::new();
        for frame in 0..30 {
            let impulse = if frame == 0 { 1.0 } else { 0.0 };
            out.push(comb(&mut buffer, &mut write, impulse, 10, 0.5));
        }
        assert!((out[0] - 1.0).abs() < 1e-6, "the dry signal always passes");
        assert!(
            out[1..10].iter().all(|&value| value.abs() < 1e-6),
            "nothing arrives before the tap"
        );
        assert!(
            (out[10] - 0.5).abs() < 1e-6,
            "the echo arrived at the wrong level: {}",
            out[10]
        );
        assert!(
            (out[20] - 0.25).abs() < 1e-6,
            "the second echo decayed to {}",
            out[20]
        );
    }

    #[test]
    /// Feedback is what turns one echo into a tail; with none of it the stage is just
    /// a delayed copy that never repeats.
    fn without_feedback_the_echo_does_not_repeat() {
        let mut buffer = vec![0.0; 2000];
        let mut write = 0usize;
        let mut out = Vec::new();
        for frame in 0..30 {
            let impulse = if frame == 0 { 1.0 } else { 0.0 };
            out.push(comb(&mut buffer, &mut write, impulse, 5, 0.0));
        }
        assert!(out.iter().skip(1).all(|&value| value.abs() < 1e-6));
    }

    #[test]
    /// Vibrato wiggles the pitch without dragging the note off it: counted over a
    /// second, the average frequency is still the note the hand asked for.
    fn vibrato_does_not_shift_the_centre_pitch() {
        let crossings = |chain: Chain| {
            let mut voice = Voice::idle(44100.0);
            voice.set_chain(chain);
            voice.set_target(440.0, 1.0, Waveform::Sine);
            // Let the attack and the pitch glide finish before measuring, otherwise
            // the ramp from silence counts as part of the rate.
            for _ in 0..4410 {
                voice.next_sample(44100.0);
            }
            let mut previous = 0.0f32;
            let mut count = 0u64;
            for _ in 0..44100 {
                let sample = voice.next_sample(44100.0);
                if (previous < 0.0) != (sample < 0.0) {
                    count += 1;
                }
                previous = sample;
            }
            count as f32 / 2.0
        };
        let plain = crossings(Chain::idle());
        let wobbling = crossings(Chain {
            vibrato: true,
            vibrato_hz: 5.0,
            vibrato_cents: 50.0,
            ..Chain::idle()
        });
        assert!(
            (wobbling - plain).abs() < plain * 0.02,
            "plain {plain} Hz, vibrato {wobbling} Hz"
        );
        assert!((plain - 440.0).abs() < 5.0, "the oscillator itself drifted");
    }

    #[test]
    /// Modulation reaches the callback through the target, exactly as the timbre
    /// does: a write on the control side, one read per block on the audio side.
    fn modulation_travels_to_the_chain() {
        let synth = Synthesizer::new(1.0);
        let modulation = Modulation {
            enabled: EffectMask::parse("lowpass,delay").expect("valid list"),
            cutoff_hz: 1200.0,
            vibrato_hz: 4.0,
            vibrato_cents: 0.0,
            delay_ms: 250.0,
            feedback: 0.3,
        };
        synth.target.set_modulation(&modulation);

        let chain = synth.target.chain();
        assert!(chain.lowpass && chain.delay, "both stages enabled");
        assert!(!chain.vibrato, "vibrato was not asked for");
        assert!((chain.cutoff_hz - 1200.0).abs() < 1e-6);
        assert!((chain.delay_ms - 250.0).abs() < 1e-6);
    }

    #[test]
    /// A target nobody has ever written to holds zeros. That would be a closed filter,
    /// so the chain clamps it: nothing is enabled, and the cutoff cannot be a divide
    /// by zero waiting to happen.
    fn an_untouched_target_is_not_a_closed_filter() {
        let chain = Target::default().chain();
        assert!(!chain.lowpass, "nothing is enabled by default");
        assert!(chain.cutoff_hz >= 1.0, "cutoff was {}", chain.cutoff_hz);
    }

    #[test]
    /// A released note is silent, but an echo already in the delay line has to ring
    /// out. Without the tail in `is_silent` the callback would cut the echo off the
    /// moment the hand left the frame.
    fn a_released_note_rings_out_through_the_echo() {
        let mut voice = Voice::idle(44100.0);
        voice.set_chain(Chain {
            delay: true,
            delay_ms: 200.0,
            feedback: 0.5,
            ..Chain::idle()
        });
        voice.set_target(440.0, 1.0, Waveform::Sine);
        for _ in 0..4410 {
            voice.next_sample(44100.0);
        }
        voice.release();
        assert!(
            !voice.is_silent(),
            "the envelope has closed but the echo is still in the ring"
        );

        let tail = (0..4410).fold(0.0f32, |loudest, _| {
            loudest.max(voice.next_sample(44100.0).abs())
        });
        assert!(
            tail > 0.01,
            "the echo died instantly: loudest tail sample {tail}"
        );

        for _ in 0..44100 * 3 {
            voice.next_sample(44100.0);
        }
        assert!(
            voice.is_silent(),
            "the tail is finite: it still had {} in it",
            voice.echo_tail
        );
    }

    #[test]
    #[ignore]
    /// What the chain costs per audio sample, measured in the release build:
    /// `cargo test --release -- --include-ignored bench_chain_cost --nocapture`.
    /// The point is that it is nanoseconds, not milliseconds: the frame loop has a
    /// 33 ms budget, and this runs in the audio callback at 44.1 kHz.
    fn bench_chain_cost() {
        const SAMPLES: usize = 1_000_000;
        let run = |chain: Chain, waveform: Waveform| {
            let mut voice = Voice::idle(44100.0);
            voice.set_chain(chain);
            voice.set_target(440.0, 0.8, waveform);
            let started = Instant::now();
            for _ in 0..SAMPLES {
                voice.next_sample(44100.0);
            }
            started.elapsed().as_secs_f64() * 1e9 / SAMPLES as f64
        };

        let all = Chain {
            lowpass: true,
            vibrato: true,
            delay: true,
            cutoff_hz: 2000.0,
            vibrato_hz: 5.0,
            vibrato_cents: 40.0,
            delay_ms: 300.0,
            feedback: 0.4,
        };

        println!(
            "ns/sample - sine bare {:.1}, sine + chain {:.1}, square bare {:.1}, square + chain {:.1}",
            run(Chain::idle(), Waveform::Sine),
            run(all, Waveform::Sine),
            run(Chain::idle(), Waveform::Square),
            run(all, Waveform::Square)
        );
    }
}
