//! The instrument the player is actually playing.
//!
//! `InstrumentConfig` is the single source of truth for scale, root, octave span
//! and waveform. `main` builds it from the command line, the dashboard mutates it
//! through its pickers and sends the result to the pipeline as a `UiCommand`; the
//! pipeline applies it to the strategy and the sinks.
//!
//! The setters return whether anything actually changed, so a click on the option
//! that is already selected sends no command at all.

use anyhow::{anyhow, Result};

use crate::core::grid::GridLayout;
use crate::core::scale::{preset, root_class, Scale};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waveform {
    Sine,
    Triangle,
    Square,
    Saw,
}

impl Waveform {
    pub fn all() -> [Waveform; 4] {
        [
            Waveform::Sine,
            Waveform::Triangle,
            Waveform::Square,
            Waveform::Saw,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            Waveform::Sine => "sine",
            Waveform::Triangle => "triangle",
            Waveform::Square => "square",
            Waveform::Saw => "saw",
        }
    }

    pub fn parse(name: &str) -> Option<Waveform> {
        let key = name.trim().to_lowercase();
        match key.as_str() {
            "s" | "sin" | "sine" => Some(Waveform::Sine),
            "t" | "tri" | "triangle" => Some(Waveform::Triangle),
            "sq" | "square" | "pulse" => Some(Waveform::Square),
            "saw" | "sawtooth" => Some(Waveform::Saw),
            _ => None,
        }
    }

    /// Sample value of one cycle at normalised phase `phase` in 0..1, truncated
    /// to the `harmonics` that actually fit below Nyquist.
    ///
    /// A step function contains every harmonic, so the naive `if phase < 0.5`
    /// square aliases badly above a few hundred Hz. The sum is capped, and the
    /// ~18 % Gibbs overshoot at the discontinuity is divided back out.
    pub fn sample_band_limited(self, phase: f32, harmonics: usize) -> f32 {
        let tau = 2.0 * std::f32::consts::PI;
        let phase = phase - phase.floor();
        const GIBBS: f32 = 1.18;
        match self {
            Waveform::Sine => (phase * tau).sin(),
            // asin(sin(x)) is a triangle whose harmonics already fall off as 1/k^2,
            // so it needs no truncation to stay in range.
            Waveform::Triangle => (2.0 / std::f32::consts::PI) * (phase * tau).sin().asin(),
            Waveform::Square => {
                let mut sum = 0.0;
                for k in (1..=harmonics).step_by(2) {
                    sum += (phase * tau * k as f32).sin() / k as f32;
                }
                (4.0 / std::f32::consts::PI) * sum / GIBBS
            }
            Waveform::Saw => {
                // 2p - 1 = -(2/pi) * sum sin(2*pi*k*p)/k over all k.
                let mut sum = 0.0;
                for k in 1..=harmonics {
                    sum += (phase * tau * k as f32).sin() / k as f32;
                }
                -(2.0 / std::f32::consts::PI) * sum / GIBBS
            }
        }
    }
}

impl std::fmt::Display for Waveform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InstrumentConfig {
    pub scale: Scale,
    /// Root pitch class, 0..11.
    pub root: u8,
    pub root_name: String,
    pub octaves: usize,
    pub waveform: Waveform,
    /// Derived from the fields above; recomputed on every change.
    pub layout: GridLayout,
}

impl InstrumentConfig {
    /// The instrument every session starts from: C major pentatonic, three
    /// octaves, sine. Used both as the CLI default and as the fallback when the
    /// shared config cannot be read.
    pub fn default_preset() -> InstrumentConfig {
        Self::new("major pentatonic", "C", 3, Waveform::Sine)
            .expect("the default preset is always available")
    }

    pub fn new(
        scale_name: &str,
        root_name: &str,
        octaves: usize,
        waveform: Waveform,
    ) -> Result<Self> {
        let scale = preset(scale_name).ok_or_else(|| {
            anyhow!(
                "unknown scale '{}'; available: {}",
                scale_name,
                core_list_scales()
            )
        })?;
        let root =
            root_class(root_name).ok_or_else(|| anyhow!("unknown root note '{}'", root_name))?;

        let mut config = InstrumentConfig {
            scale,
            root,
            root_name: root_name.trim().to_string(),
            octaves: octaves.max(1),
            waveform,
            layout: GridLayout::default(),
        };
        config.layout = GridLayout::new(config.root, &config.scale, config.octaves);
        Ok(config)
    }

    pub fn set_scale(&mut self, name: &str) -> bool {
        match preset(name) {
            Some(scale) if scale != self.scale => {
                self.scale = scale;
                self.bump();
                true
            }
            _ => false,
        }
    }

    pub fn set_root(&mut self, name: &str) -> bool {
        match root_class(name) {
            Some(root) if root != self.root => {
                self.root = root;
                self.root_name = name.trim().to_string();
                self.bump();
                true
            }
            _ => false,
        }
    }

    pub fn set_octaves(&mut self, octaves: usize) -> bool {
        let octaves = octaves.max(1);
        if octaves != self.octaves {
            self.octaves = octaves;
            self.bump();
            true
        } else {
            false
        }
    }

    pub fn set_waveform(&mut self, waveform: Waveform) -> bool {
        if waveform != self.waveform {
            self.waveform = waveform;
            self.bump();
            true
        } else {
            false
        }
    }

    fn bump(&mut self) {
        self.layout = GridLayout::new(self.root, &self.scale, self.octaves);
    }
}

impl Default for InstrumentConfig {
    fn default() -> Self {
        Self::default_preset()
    }
}

fn core_list_scales() -> String {
    crate::core::scale::PRESETS
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_change_rebuilds_the_layout() {
        let mut config = InstrumentConfig::new("major pentatonic", "C", 3, Waveform::Sine).unwrap();
        let before = config.layout.clone();

        assert!(config.set_scale("chromatic"));
        assert_ne!(config.layout, before);
        assert_eq!(config.layout.cols, 12);

        // Re-picking the current value is a no-op: the UI must not send a command
        // that changes nothing.
        assert!(!config.set_scale("chromatic"));
        assert!(!config.set_octaves(3));
    }

    #[test]
    fn unknown_names_are_rejected() {
        assert!(InstrumentConfig::new("lydian", "C", 3, Waveform::Sine).is_err());
        assert!(InstrumentConfig::new("major", "H", 3, Waveform::Sine).is_err());
    }

    #[test]
    fn transposing_and_widening_the_grid_change_the_notes() {
        let mut config = InstrumentConfig::new("major pentatonic", "C", 3, Waveform::Sine).unwrap();
        let in_c = config.layout.notes.clone();
        assert!(config.set_root("A"));
        assert!(
            config.layout.notes.iter().any(|&n| n % 12 == 9),
            "A must appear"
        );
        assert_ne!(config.layout.notes, in_c);

        // Four octaves of A pentatonic cannot fit inside C2..C6, so the grid
        // shrinks to the largest span that does rather than running off the top.
        assert!(config.set_octaves(4));
        assert!(config.layout.rows < 4);
        assert!(config
            .layout
            .notes
            .iter()
            .all(|&n| n <= crate::core::NOTE_HIGH));
    }

    #[test]
    fn waveform_shapes_match_their_textbook_values() {
        // 24 harmonics is what the playable band actually gets at a 44.1k sample
        // rate, so the band-limited values are close to the ideal shapes.
        let sine = [
            Waveform::Sine.sample_band_limited(0.0, 24),
            Waveform::Sine.sample_band_limited(0.25, 24),
            Waveform::Sine.sample_band_limited(0.5, 24),
            Waveform::Sine.sample_band_limited(0.75, 24),
        ];
        assert!(sine[0].abs() < 1e-6);
        assert!((sine[1] - 1.0).abs() < 1e-6);
        assert!((sine[2] - 0.0).abs() < 1e-6);
        assert!((sine[3] + 1.0).abs() < 1e-6);

        let square = [
            Waveform::Square.sample_band_limited(0.1, 24),
            Waveform::Square.sample_band_limited(0.4, 24),
            Waveform::Square.sample_band_limited(0.6, 24),
            Waveform::Square.sample_band_limited(0.9, 24),
        ];
        assert!(
            square[0] > 0.8 && square[1] > 0.8,
            "first half of the cycle is high"
        );
        assert!(square[2] < -0.8 && square[3] < -0.8, "second half is low");

        // Triangle peaks a quarter of the way in; saw rises across the cycle.
        assert!((Waveform::Triangle.sample_band_limited(0.25, 24) - 1.0).abs() < 1e-3);
        assert!((Waveform::Triangle.sample_band_limited(0.75, 24) + 1.0).abs() < 1e-3);
        assert!(
            Waveform::Saw.sample_band_limited(0.9, 24) > Waveform::Saw.sample_band_limited(0.1, 24)
        );

        // All four stay inside [-1, 1] across a cycle.
        for waveform in Waveform::all() {
            for i in 0..=200 {
                let value = waveform.sample_band_limited(i as f32 / 200.0, 24);
                assert!(
                    (-1.0..=1.0).contains(&value),
                    "{waveform} out of range at {i}: {value}"
                );
            }
        }
    }

    #[test]
    fn waveform_names_round_trip() {
        for waveform in Waveform::all() {
            assert_eq!(Waveform::parse(waveform.name()), Some(waveform));
        }
        assert_eq!(Waveform::parse("Sawtooth"), Some(Waveform::Saw));
        assert_eq!(Waveform::parse("wobble"), None);
    }
}
