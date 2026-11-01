//! A continuous theremin: the index fingertip's horizontal position sweeps a
//! frequency band and the thumb-index spread is the loudness.
//!
//! Unlike `FingerPositionStrategy` there is no keyboard underneath the hand -
//! nothing is quantised to a scale or a cell, so the pitch glides continuously
//! as the hand moves. That is what the `GesturePayload::Signal` branch of every
//! sink is for: the synthesizer can oscillate at any frequency, while the MIDI
//! sink has nothing to do with it (MIDI cannot express a continuous pitch).

use crate::core::types::GestureEvent;
use crate::core::{HandDetectionResult, Strategy};
use crate::strategies::{band_position, spread_volume, TipSmoother};

/// The band the theremin spans. These are C2 and C6, the same extreme octaves
/// the keyboard grid is allowed to reach, so switching strategies does not
/// change how far the hand has to travel - only how the travel is read.
pub const DEFAULT_LOW_HZ: f32 = 65.4;
pub const DEFAULT_HIGH_HZ: f32 = 1046.5;

#[derive(Debug)]
pub struct ThereminStrategy {
    tip: TipSmoother,
    low_hz: f32,
    high_hz: f32,
}

impl ThereminStrategy {
    pub fn new(low_hz: f32, high_hz: f32) -> Self {
        // Guard against a nonsensical band: `high / low` inside a powf would
        // otherwise produce NaN and silence the output.
        let low = low_hz.max(1.0);
        Self {
            tip: TipSmoother::default(),
            low_hz: low,
            high_hz: high_hz.max(low + 1.0),
        }
    }

    /// The band is mapped logarithmically, because that is how the ear hears
    /// pitch: half way across the frame is one octave up, not half way up in
    /// hertz. With the defaults the centre of the band lands on middle C.
    fn frequency(&self, x_position: f32) -> f32 {
        self.low_hz * (self.high_hz / self.low_hz).powf(x_position)
    }
}

impl Default for ThereminStrategy {
    fn default() -> Self {
        Self::new(DEFAULT_LOW_HZ, DEFAULT_HIGH_HZ)
    }
}

impl Strategy for ThereminStrategy {
    fn process(&self, detection: &HandDetectionResult) -> Option<GestureEvent> {
        let landmarks = detection.hand_landmarks.as_ref()?;
        if landmarks.len() < 9 {
            self.tip.reset();
            return None;
        }

        // Only the horizontal position drives the pitch: a theremin is a
        // one-dimensional control, and letting the hand's height move the pitch
        // too would make every gesture two axes the player has to learn. The
        // smoother still tracks both coordinates so the state is identical to
        // the keyboard strategy's.
        let (tip_x, tip_y) = self.tip.sample(landmarks[8].x, landmarks[8].y);
        let (x_position, _) = band_position(tip_x, tip_y, detection.frame_size);

        Some(GestureEvent::new_signal(
            self.frequency(x_position),
            spread_volume(landmarks, detection.r#box),
            "Theremin",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::GesturePayload;
    use crate::core::Landmark;

    /// A hand whose index tip is at (`tip_x`, 240) and whose thumb tip is at
    /// (`thumb_x`, 240); everything else sits still. Frame is 640x480, hand box
    /// 200 px tall, so a 120 px thumb-index gap is full volume.
    fn hand(tip_x: f32, thumb_x: f32) -> HandDetectionResult {
        let landmarks = (0..21)
            .map(|i| Landmark {
                x: match i {
                    4 => thumb_x,
                    8 => tip_x,
                    _ => 300.0,
                },
                y: 240.0,
                z: 0.0,
            })
            .collect();
        HandDetectionResult {
            hand_landmarks: Some(landmarks),
            r#box: Some((200.0, 140.0, 200.0, 200.0)),
            frame_size: (640, 480),
            score: 0.9,
            timings: Default::default(),
        }
    }

    fn signal(strategy: &ThereminStrategy, tip_x: f32, thumb_x: f32) -> (f32, f32) {
        match strategy.process(&hand(tip_x, thumb_x)).unwrap().payload {
            GesturePayload::Signal {
                frequency,
                amplitude,
            } => (frequency, amplitude),
            _ => panic!("expected a signal payload"),
        }
    }

    /// A fresh strategy per point check, so the tip smoother never carries one
    /// sample into the next.
    fn frequency_at(tip_x: f32) -> f32 {
        signal(&ThereminStrategy::default(), tip_x, 440.0).0
    }

    #[test]
    fn the_band_spans_four_octaves_of_the_keyboard_range() {
        // 640 px frame with a 10 % margin each side leaves a 512 px band, so
        // x = 64 is the left edge of the playable band and x = 576 the right.
        let left = frequency_at(64.0);
        let middle = frequency_at(320.0);
        let right = frequency_at(576.0);

        assert!(
            (left - DEFAULT_LOW_HZ).abs() < 1e-3,
            "left edge is C2, got {left}"
        );
        assert!(
            (right - DEFAULT_HIGH_HZ).abs() < 1e-3,
            "right edge is C6, got {right}"
        );
        // Logarithmic mapping: the middle of the band is the geometric mean of
        // the two ends, which for C2..C6 is exactly one octave up - middle C.
        assert!((middle - 261.63).abs() < 0.5, "middle is C4, got {middle}");
        assert!(middle > left && right > middle);
    }

    #[test]
    fn the_tip_is_clamped_to_the_band_so_the_edges_saturate() {
        // Off the frame entirely: still the ends of the band, never NaN or 0 Hz.
        assert!((frequency_at(-500.0) - DEFAULT_LOW_HZ).abs() < 1e-3);
        assert!((frequency_at(1200.0) - DEFAULT_HIGH_HZ).abs() < 1e-3);
    }

    #[test]
    fn amplitude_follows_the_spread_and_saturates() {
        // The hand box is 200 px tall, so MAX_SPREAD (0.6) is a 120 px gap.
        let pinched = signal(&ThereminStrategy::default(), 320.0, 322.0).1; // 2 px apart
        let open = signal(&ThereminStrategy::default(), 320.0, 440.0).1; // 120 px apart
        assert!(pinched < open);
        assert!((pinched - 2.0 / 200.0 / 0.6).abs() < 1e-6);
        assert!((open - 1.0).abs() < 1e-6);
        // Wider than the saturation point is still full volume.
        assert!((signal(&ThereminStrategy::default(), 320.0, 600.0).1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn smoothing_steadies_the_pitch_but_still_follows_a_sweep() {
        let strategy = ThereminStrategy::default();
        // A hand that is genuinely still jitters +/-20 px around the middle of
        // the band; smoothed, the pitch swings far less than the raw positions.
        let jitter = [
            300.0, 340.0, 300.0, 340.0, 300.0, 340.0, 300.0, 340.0, 300.0, 340.0,
        ];
        let smoothed: Vec<f32> = jitter
            .iter()
            .map(|&x| signal(&strategy, x, 440.0).0)
            .collect();
        let raw: Vec<f32> = jitter.iter().map(|&x| frequency_at(x)).collect();

        let spread = |values: &[f32]| {
            values.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
                - values.iter().min_by(|a, b| a.total_cmp(b)).unwrap()
        };
        assert!(
            spread(&smoothed) < spread(&raw),
            "raw {raw:?} swings {:.1} Hz, smoothed {smoothed:?} swings {:.1} Hz",
            spread(&raw),
            spread(&smoothed)
        );

        // A deliberate sweep across the keyboard still reaches the top of the band.
        let mut x = 320.0;
        while x <= 640.0 {
            signal(&strategy, x, 440.0);
            x += 20.0;
        }
        let (top, _) = signal(&strategy, 640.0, 440.0);
        assert!(
            (top - DEFAULT_HIGH_HZ).abs() < 1e-3,
            "sweep never reached the top: {top}"
        );
    }

    #[test]
    fn no_landmarks_means_no_signal() {
        let strategy = ThereminStrategy::default();
        assert!(strategy.process(&HandDetectionResult::default()).is_none());
    }
}
