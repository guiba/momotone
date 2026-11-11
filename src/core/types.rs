use serde::{Deserialize, Serialize};

/// What the instrument should do this frame.
///
/// `Note` is the quantised keyboard mapping; `Signal` is the continuous theremin
/// mapping, which the synthesizer can render and the MIDI sink cannot (MIDI has no
/// continuous pitch to send).
///
/// `Release` is explicit rather than implied: a sink that is never told the gesture
/// stopped keeps playing whatever it was last told. The old code encoded "no
/// gesture" as `Note { note: 0 }`, a magic value every sink had to recognise by
/// convention.
///
/// UI-only fields - a label and a colour - used to ride through every sink on this
/// type. They belong in the published snapshot, not in the performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Performance {
    Note { note: u8, velocity: f32 },
    Signal { frequency: f32, amplitude: f32 },
    Release,
}
