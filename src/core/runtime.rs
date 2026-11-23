//! The knobs the dashboard can change while a run is in progress.
//!
//! Everything the UI is allowed to flip lives in one struct owned by the pipeline,
//! so there is no second copy of "is the feed on" to keep in sync. The UI never
//! writes it: it sends a `UiCommand`, and the pipeline applies it at the top of a
//! frame. The capture thread is the only other reader - it has to know whether the
//! player is running before it reads the camera.
//!
//! This replaces the four `AtomicBool`s, the `AtomicU8` ramp and the
//! `InstrumentConfig.version` counter that used to spread the same state across
//! `main.rs`, `pipeline.rs` and `dashboard/mod.rs`.

use crate::core::effects::EffectMask;
use crate::core::instrument::{InstrumentConfig, VolumeModel};
use crate::gesture::{PROXIMITY_FAR, PROXIMITY_NEAR};
use crate::presentation::ascii::Ramp;
use crossbeam_channel::{unbounded, Receiver, Sender};

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeConfig {
    /// The dashboard's Start/Stop button. Cleared means: read no frames, run no
    /// inference, and make the sinks silent.
    pub active: bool,
    /// Whether the stage is showing the webcam, so the pipeline should bother
    /// converting a frame to RGBA at all.
    pub feed: bool,
    /// Whether the ASCII backdrop is drawn.
    pub ascii: bool,
    /// Preview cadence in frames per second; 0 means every frame.
    pub feed_fps: f32,
    /// ASCII cadence in frames per second; 0 means every frame.
    pub ascii_fps: f32,
    /// Glyph ramp the ASCII grid is rendered with.
    pub ramp: Ramp,
    /// What decides loudness this run: how close the hand is, how open it is, or a
    /// mix of the two.
    pub volume: VolumeModel,
    /// Which stages are in the effect chain. The parameters are continuous, so a
    /// stage can join a note that is already sounding.
    pub effects: EffectMask,
    /// Palm-length-per-frame ratios that bound the proximity band: `far` is silence,
    /// `near` is full volume.
    pub proximity_near: f32,
    pub proximity_far: f32,
}

impl RuntimeConfig {
    pub fn new(feed_fps: f32, ascii_fps: f32, ascii: bool, ramp: Ramp) -> Self {
        Self {
            active: true,
            feed: false,
            ascii,
            feed_fps,
            ascii_fps,
            ramp,
            volume: VolumeModel::Proximity,
            // A config built without a chain runs the plain note; `--effects` and the
            // dashboard decide what is added.
            effects: EffectMask::NONE,
            proximity_near: PROXIMITY_NEAR,
            proximity_far: PROXIMITY_FAR,
        }
    }

    /// The loudness settings, straight from the config, for the estimator.
    pub fn volume_settings(&self) -> (VolumeModel, f32, f32) {
        (self.volume, self.proximity_near, self.proximity_far)
    }
}

/// One message from the dashboard to the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum UiCommand {
    Start,
    Stop,
    SetFeed(bool),
    SetAscii(bool),
    SetRamp(Ramp),
    /// Swap the loudness source: proximity, pinch or a blend.
    SetVolume(VolumeModel),
    /// Switch one stage of the effect chain on or off.
    SetEffect(crate::core::effects::Effect, bool),
    /// Trade the pitch and modulator roles between the two hand slots. Manual only:
    /// automatic promotion could hijack the melody in the middle of a sweep.
    SwapHands,
    /// The pickers changed the instrument: re-key the strategy, retimbre the
    /// sinks and repaint the header.
    SetInstrument(InstrumentConfig),
}

/// The control channel between the UI thread and the frame loop. Unbounded: a
/// click is one small message, and a bounded channel would have to drop UI input.
pub fn control_channel() -> (Sender<UiCommand>, Receiver<UiCommand>) {
    unbounded()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_reach_the_config_that_owns_them() {
        let (tx, rx) = control_channel();
        tx.send(UiCommand::Stop).unwrap();
        tx.send(UiCommand::SetFeed(true)).unwrap();
        tx.send(UiCommand::SetRamp(Ramp::Shades)).unwrap();

        let received: Vec<UiCommand> = rx.try_iter().collect();
        assert_eq!(
            received,
            vec![
                UiCommand::Stop,
                UiCommand::SetFeed(true),
                UiCommand::SetRamp(Ramp::Shades)
            ],
            "the channel is FIFO, so a stop cannot overtake an earlier toggle"
        );
    }

    #[test]
    fn nothing_is_applied_when_the_ui_never_speaks() {
        let (_tx, rx) = control_channel();
        let commands: Vec<UiCommand> = rx.try_iter().collect();
        assert!(commands.is_empty(), "an idle channel costs no work");
    }
}
