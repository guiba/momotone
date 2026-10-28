//! The instrument the player is actually playing.
//!
//! `InstrumentConfig` is the single source of truth for scale, root, octave
//! span and waveform. `main` builds it from the command line, the dashboard
//! mutates it through its pickers, and the pipeline applies it to the strategy
//! and the sinks whenever its `version` changes.
//!
//! Everything is behind one `Arc<Mutex<_>>`: the UI writes, the pipeline reads.
//! A tick is a mutex lock and a version compare, so a picker change takes effect
//! on the very next frame.

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
        [Waveform::Sine, Waveform::Triangle, Waveform::Square, Waveform::Saw]
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

    /// Sample value of one cycle at normalised phase `phase` in 0..1.
    pub fn sample(self, phase: f32) -> f32 {
        let phase = phase - phase.floor();
        match self {
            Waveform::Sine => (phase * 2.0 * std::f32::consts::PI).sin(),
            Waveform::Triangle => 1.0 - 4.0 * (phase - 0.5).abs(),
            Waveform::Square => if phase < 0.5 { 1.0 } else { -1.0 },
            Waveform::Saw => 2.0 * phase - 1.0,
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
    /// Bumped on every change so the pipeline knows when to re-apply.
    pub version: u64,
}

impl InstrumentConfig {
    /// The instrument every session starts from: C major pentatonic, three
    /// octaves, sine. Used both as the CLI default and as the fallback when the
    /// shared config cannot be read.
    pub fn default_preset() -> InstrumentConfig {
        Self::new("major pentatonic", "C", 3, Waveform::Sine)
            .expect("the default preset is always available")
    }

    pub fn new(scale_name: &str, root_name: &str, octaves: usize, waveform: Waveform) -> Result<Self> {
        let scale = preset(scale_name).ok_or_else(|| {
            anyhow!(
                "unknown scale '{}'; available: {}",
                scale_name,
                core_list_scales()
            )
        })?;
        let root = root_class(root_name).ok_or_else(|| anyhow!("unknown root note '{}'", root_name))?;

        let mut config = InstrumentConfig {
            scale,
            root,
            root_name: root_name.trim().to_string(),
            octaves: octaves.max(1),
            waveform,
            layout: GridLayout::default(),
            version: 0,
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
        self.version += 1;
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
    fn a_change_rebuilds_the_layout_and_bumps_the_version() {
        let mut config = InstrumentConfig::new("major pentatonic", "C", 3, Waveform::Sine).unwrap();
        let before = config.layout.clone();
        let version = config.version;

        assert!(config.set_scale("chromatic"));
        assert_eq!(config.version, version + 1);
        assert_ne!(config.layout, before);
        assert_eq!(config.layout.cols, 12);

        // Setting the same value is a no-op: the version must not move, or the
        // pipeline would rebuild the keyboard every tick.
        assert!(!config.set_scale("chromatic"));
        assert_eq!(config.version, version + 1);
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
        assert!(config.layout.notes.iter().any(|&n| n % 12 == 9), "A must appear");
        assert_ne!(config.layout.notes, in_c);

        // Four octaves of A pentatonic cannot fit inside C2..C6, so the grid
        // shrinks to the largest span that does rather than running off the top.
        assert!(config.set_octaves(4));
        assert!(config.layout.rows < 4);
        assert!(config.layout.notes.iter().all(|&n| n <= crate::core::NOTE_HIGH));
    }

    #[test]
    fn waveform_shapes_match_their_textbook_values() {
        // Each waveform is sampled at the same four points of the cycle.
        let sine = [
            Waveform::Sine.sample(0.0),
            Waveform::Sine.sample(0.25),
            Waveform::Sine.sample(0.5),
            Waveform::Sine.sample(0.75),
        ];
        assert!(sine[0].abs() < 1e-6);
        assert!((sine[1] - 1.0).abs() < 1e-6);
        assert!((sine[2] - 0.0).abs() < 1e-6);
        assert!((sine[3] + 1.0).abs() < 1e-6);

        let square = [
            Waveform::Square.sample(0.1),
            Waveform::Square.sample(0.4),
            Waveform::Square.sample(0.6),
            Waveform::Square.sample(0.9),
        ];
        assert_eq!(square, [1.0, 1.0, -1.0, -1.0]);

        // Triangle peaks in the middle, saw ramps and jumps back.
        assert!((Waveform::Triangle.sample(0.5) - 1.0).abs() < 1e-6);
        assert!((Waveform::Triangle.sample(0.0) + 1.0).abs() < 1e-6);
        assert!((Waveform::Saw.sample(0.0) + 1.0).abs() < 1e-6);
        assert!((Waveform::Saw.sample(0.999) - 1.0).abs() < 1e-2);

        // All four stay inside [-1, 1] across a cycle.
        for waveform in Waveform::all() {
            for i in 0..=100 {
                let value = waveform.sample(i as f32 / 100.0);
                assert!((-1.0..=1.0).contains(&value), "{waveform} out of range at {i}");
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
