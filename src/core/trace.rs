//! A recorded run, small enough to keep in a repository.
//!
//! A trace is the frame-by-frame sounding state: what was playing, how loud, and
//! whether a hand was seen. Written to a file it becomes a golden reference - run
//! the same fixture clip twice and a diff is a regression, not a guess.
//!
//! The trace records *behaviour*, not internals. A change to the grid, the
//! smoothing, the detection gate or the model all show up here, and none of them
//! need a screenshot to be noticed.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::dashboard::metrics::FrameMetrics;

/// One frame of a run.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TraceFrame {
    pub frame: u64,
    pub note: Option<u8>,
    pub frequency: Option<f32>,
    pub volume: f32,
    pub detected: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Trace {
    pub frames: Vec<TraceFrame>,
}

impl Trace {
    pub fn push(&mut self, snapshot: &FrameMetrics) {
        self.frames.push(TraceFrame {
            frame: snapshot.perf.frames,
            note: snapshot.sound.note,
            frequency: snapshot.sound.frequency,
            volume: round2(snapshot.sound.volume),
            detected: snapshot.hand.detected,
        });
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }
}

/// Volumes come out of a fingertip distance and an envelope, so the last digits
/// are noise. Two decimal places is the resolution a run can be expected to
/// reproduce frame to frame.
fn round2(value: f32) -> f32 {
    (value * 100.0).round() / 100.0
}

/// Frames that differ, as readable lines. A trace is short enough that printing
/// the whole diff beats printing a count.
pub fn diff(expected: &Trace, actual: &Trace) -> Vec<String> {
    let mut lines = Vec::new();
    if expected.frames.len() != actual.frames.len() {
        lines.push(format!(
            "length: expected {} frames, got {}",
            expected.frames.len(),
            actual.frames.len()
        ));
    }
    for (index, (want, got)) in expected.frames.iter().zip(actual.frames.iter()).enumerate() {
        if want != got {
            lines.push(format!("frame {index}: expected {want:?}, got {got:?}"));
        }
    }
    lines
}

pub fn read(path: &str) -> Result<Trace> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| anyhow!("bad trace file {path}: {e}"))
}

pub fn write(path: &str, trace: &Trace) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(trace)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(note: Option<u8>, volume: f32) -> TraceFrame {
        TraceFrame {
            frame: 0,
            note,
            frequency: None,
            volume,
            detected: true,
        }
    }

    #[test]
    fn a_volume_rounded_to_two_places_compares_exactly() {
        assert_eq!(round2(0.845_678), 0.85);
        assert_eq!(round2(1.0 / 3.0), 0.33);
    }

    #[test]
    fn a_changed_note_is_a_diff_and_an_unchanged_one_is_not() {
        let expected = Trace {
            frames: vec![frame(Some(64), 0.8), frame(Some(67), 0.6)],
        };
        assert!(diff(&expected, &expected.clone()).is_empty());

        let mut drifted = expected.clone();
        drifted.frames[1].note = Some(69);
        let lines = diff(&expected, &drifted);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("frame 1"));
        assert!(lines[0].contains("67") && lines[0].contains("69"));
    }

    #[test]
    fn a_missing_frame_is_reported_as_a_length_difference() {
        let expected = Trace {
            frames: vec![frame(Some(64), 0.5); 3],
        };
        let actual = Trace {
            frames: vec![frame(Some(64), 0.5); 2],
        };
        let lines = diff(&expected, &actual);
        assert!(lines[0].contains("expected 3 frames, got 2"));
    }
}
