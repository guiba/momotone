//! A continuous theremin: the index fingertip's horizontal position sweeps a
//! frequency band and the thumb-index spread is the loudness.
//!
//! Unlike `FingerPositionStrategy` there is no keyboard underneath the hand -
//! nothing is quantised to a scale or a cell, so the pitch glides continuously as
//! the hand moves. That is what the `Signal` branch of every sink is for: the
//! synthesizer can oscillate at an arbitrary frequency, while the MIDI sink has
//! nothing to do with it (MIDI cannot express a continuous pitch).

use crate::core::types::Performance;
use crate::gesture::band_position;
use crate::gesture::estimator::HandState;
use crate::gesture::Strategy;

/// The band the theremin spans. These are C2 and C6, the same extreme octaves the
/// keyboard grid is allowed to reach, so switching strategies does not change how
/// far the hand has to travel - only how the travel is read.
pub const DEFAULT_LOW_HZ: f32 = 65.4;
pub const DEFAULT_HIGH_HZ: f32 = 1046.5;

#[derive(Debug)]
pub struct ThereminStrategy {
    low_hz: f32,
    high_hz: f32,
}

impl ThereminStrategy {
    pub fn new(low_hz: f32, high_hz: f32) -> Self {
        // Guard against a nonsensical band: `high / low` inside a powf would
        // otherwise produce NaN and silence the output.
        let low = low_hz.max(1.0);
        Self {
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
    fn process(&mut self, state: &HandState) -> Option<Performance> {
        if !state.present {
            return None;
        }

        // Only the horizontal position drives the pitch: a theremin is a
        // one-dimensional control, and letting the hand's height move the pitch
        // too would make every gesture two axes the player has to learn.
        let (x_position, _) = band_position(state.tip.0, state.tip.1, state.frame_size);

        Some(Performance::Signal {
            frequency: self.frequency(x_position),
            amplitude: state.volume,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Landmark;
    use crate::gesture::spread_volume;

    /// A state with the index tip at `(tip_x, 240)` and a thumb-index gap that
    /// gives the requested volume. Hand box is 200 px tall, so 120 px is full
    /// volume.
    fn state(tip_x: f32, volume: f32) -> HandState {
        let landmarks: Vec<Landmark> = (0..21)
            .map(|i| Landmark {
                x: if i == 4 {
                    300.0 + volume * 120.0
                } else {
                    300.0
                },
                y: 240.0,
                z: 0.0,
            })
            .collect();
        HandState {
            present: true,
            landmarks,
            hand_box: Some((200.0, 140.0, 200.0, 200.0)),
            frame_size: (640, 480),
            confidence: 0.9,
            tip: (tip_x, 240.0),
            volume,
            proximity: volume,
            proximity_ratio: 0.12,
            approach: 0.0,
            geometry: Default::default(),
            jitter_px: 0.0,
        }
    }

    fn signal(strategy: &mut ThereminStrategy, tip_x: f32, volume: f32) -> (f32, f32) {
        match strategy.process(&state(tip_x, volume)).unwrap() {
            Performance::Signal {
                frequency,
                amplitude,
            } => (frequency, amplitude),
            _ => panic!("expected a signal performance"),
        }
    }

    #[test]
    fn the_band_spans_four_octaves_of_the_keyboard_range() {
        // 640 px frame with a 10 % margin each side leaves a 512 px band, so
        // x = 64 is the left edge of the playable band and x = 576 the right.
        let mut strategy = ThereminStrategy::default();
        let left = signal(&mut strategy, 64.0, 0.5).0;
        let middle = signal(&mut strategy, 320.0, 0.5).0;
        let right = signal(&mut strategy, 576.0, 0.5).0;

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
        let mut strategy = ThereminStrategy::default();
        assert!((signal(&mut strategy, -500.0, 0.5).0 - DEFAULT_LOW_HZ).abs() < 1e-3);
        assert!((signal(&mut strategy, 1200.0, 0.5).0 - DEFAULT_HIGH_HZ).abs() < 1e-3);
    }

    #[test]
    fn amplitude_is_the_volume_the_estimator_gave() {
        let mut strategy = ThereminStrategy::default();
        assert!((signal(&mut strategy, 320.0, 0.25).1 - 0.25).abs() < 1e-6);
        assert!((signal(&mut strategy, 320.0, 1.0).1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn an_absurd_band_is_clamped_instead_of_going_nan() {
        let mut strategy = ThereminStrategy::new(0.0, 0.0);
        let (frequency, _) = signal(&mut strategy, 320.0, 1.0);
        assert!(frequency.is_finite() && frequency > 0.0);
    }

    #[test]
    fn no_gesture_without_a_present_hand() {
        let mut strategy = ThereminStrategy::default();
        assert!(strategy.process(&HandState::default()).is_none());
    }

    #[test]
    fn spread_volume_still_saturates_at_the_box_height() {
        // Kept as a separate check so the loudness curve is not silently tied to
        // the palm-length normalisation the dashboard uses.
        let landmarks: Vec<Landmark> = (0..21)
            .map(|i| Landmark {
                x: if i == 4 { 440.0 } else { 320.0 },
                y: 240.0,
                z: 0.0,
            })
            .collect();
        let volume = spread_volume(&landmarks, Some((200.0, 140.0, 200.0, 200.0)));
        assert!((volume - 1.0).abs() < 1e-6);
    }
}
