//! The sound chain: which effects exist, what each one does, and how a hand drives it.
//!
//! This is the declaration point. An effect is a triple - **source -> parameter ->
//! curve** - so adding one means adding a variant here, a stage in the synthesizer,
//! and a mapping. It does not mean touching the tracker: every source listed below
//! comes from a palm box or from the geometry the estimator already computes.
//!
//! The chain order is fixed and documented in `audio/synthesizer.rs`:
//!
//! ```text
//! glide -> vibrato -> oscillator -> lowpass -> delay -> soft clip
//! ```
//!
//! Individual stages are switched on by the enable mask, and the levels behind them
//! are continuous, so a stage can be added to a running performance without the
//! player hearing a step.

/// Lowest cutoff the filter can reach. Below this the note stops being dark and
/// starts being gone, which is a different gesture.
pub const CUTOFF_MIN_HZ: f32 = 200.0;
/// Brightest the filter opens. Deliberately below Nyquist so the one-pole never
/// gets close to transparent at both ends of the band.
pub const CUTOFF_MAX_HZ: f32 = 8000.0;
/// What "no filter" means. A one-pole at 12 kHz passes everything the instrument
/// produces, so handing the voice this value is equivalent to the stage not being
/// there - without needing a branch in the audio callback to say so.
pub const NEUTRAL_CUTOFF_HZ: f32 = 12000.0;

pub const VIBRATO_RATE_MIN_HZ: f32 = 0.5;
pub const VIBRATO_RATE_MAX_HZ: f32 = 8.0;
pub const VIBRATO_MAX_CENTS: f32 = 50.0;

pub const DELAY_MAX_MS: f32 = 600.0;
pub const FEEDBACK_MAX: f32 = 0.6;

/// Thumb-tip to index-tip distance, normalised by palm length, that counts as a
/// fully open hand. A pinched hand sits near `SPREAD_CLOSED`; a relaxed hand at
/// desk distance lands around 0.7.
pub const SPREAD_CLOSED: f32 = 0.3;
pub const SPREAD_OPEN: f32 = 1.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    Lowpass,
    Vibrato,
    Delay,
}

impl Effect {
    pub fn all() -> [Effect; 3] {
        [Effect::Lowpass, Effect::Vibrato, Effect::Delay]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Effect::Lowpass => "lowpass",
            Effect::Vibrato => "vibrato",
            Effect::Delay => "delay",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "lowpass" | "filter" | "tone" => Some(Effect::Lowpass),
            "vibrato" | "wobble" => Some(Effect::Vibrato),
            "delay" | "echo" => Some(Effect::Delay),
            _ => None,
        }
    }

    fn bit(&self) -> u8 {
        match self {
            Effect::Lowpass => 1,
            Effect::Vibrato => 2,
            Effect::Delay => 4,
        }
    }
}

/// Which stages are in the chain this run. One bit per effect, so adding a stage
/// costs a variant in `Effect` and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectMask(u8);

impl EffectMask {
    pub const NONE: EffectMask = EffectMask(0);
    pub const ALL: EffectMask = EffectMask(1 | 2 | 4);

    pub fn from_bits(bits: u8) -> Self {
        EffectMask(bits)
    }

    pub fn bits(&self) -> u8 {
        self.0
    }

    /// `lowpass,vibrato` or `none`. Returns `None` on an unknown name rather than
    /// silently dropping it, so a typo fails loudly at startup instead of quietly
    /// turning an effect off.
    pub fn parse(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("none")
            || trimmed.eq_ignore_ascii_case("off")
        {
            return Some(Self::NONE);
        }
        if trimmed.eq_ignore_ascii_case("all") || trimmed.eq_ignore_ascii_case("on") {
            return Some(Self::ALL);
        }
        let mut mask = 0u8;
        for name in trimmed.split(',') {
            let effect = Effect::parse(name)?;
            mask |= effect.bit();
        }
        Some(EffectMask(mask))
    }

    pub fn is_on(&self, effect: Effect) -> bool {
        self.0 & effect.bit() != 0
    }

    pub fn set(&mut self, effect: Effect, on: bool) {
        if on {
            self.0 |= effect.bit();
        } else {
            self.0 &= !effect.bit();
        }
    }

    /// Comma-separated names of the stages that are on, for logging and the UI.
    pub fn names(&self) -> String {
        let on: Vec<&str> = Effect::all()
            .into_iter()
            .filter(|&effect| self.is_on(effect))
            .map(|effect| effect.name())
            .collect();
        if on.is_empty() {
            "none".to_string()
        } else {
            on.join(", ")
        }
    }
}

impl Default for EffectMask {
    fn default() -> Self {
        Self::NONE
    }
}

/// The parameters the chain runs with, as produced by the gesture layer and sent to
/// the sinks. Every field has a neutral value, so "the hand left the frame" is a
/// value rather than a special case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Modulation {
    pub enabled: EffectMask,
    pub cutoff_hz: f32,
    pub vibrato_hz: f32,
    pub vibrato_cents: f32,
    pub delay_ms: f32,
    pub feedback: f32,
}

impl Modulation {
    /// What the chain does when no hand is driving it: nothing, at the levels the
    /// enabled stages would settle to.
    pub fn neutral_with(enabled: EffectMask) -> Self {
        Modulation {
            enabled,
            cutoff_hz: NEUTRAL_CUTOFF_HZ,
            vibrato_hz: VIBRATO_RATE_MIN_HZ,
            vibrato_cents: 0.0,
            delay_ms: 0.0,
            feedback: 0.0,
        }
    }

    pub fn neutral() -> Self {
        Self::neutral_with(EffectMask::NONE)
    }
}

impl Default for Modulation {
    fn default() -> Self {
        Self::neutral()
    }
}

/// One cent as a ratio of the pitch. A cent is 2^(1/1200) - 1, i.e. 0.000577 of the
/// frequency. For the depths this instrument uses (at most half a semitone) the
/// linear term is within about 1.5 % of the exact value, and it is a multiply rather
/// than a `powf` per sample.
const CENTS_TO_RATIO: f32 = 0.000_577_5;

pub fn cents_to_hz(cents: f32, frequency: f32) -> f32 {
    frequency * cents * CENTS_TO_RATIO
}

/// Normalise the raw thumb-to-index distance into 0..1 openness.
pub fn openness(spread: f32) -> f32 {
    ((spread - SPREAD_CLOSED) / (SPREAD_OPEN - SPREAD_CLOSED)).clamp(0.0, 1.0)
}

/// Sources in, parameters out.
///
/// In this iteration the pitch hand stands in for the modulator: proximity, height,
/// openness and horizontal position all come from the one hand the tracker reports.
/// When a second hand arrives, these sources move to it and the inter-hand gap
/// replaces `spread` as the vibrato depth - the mapping stays the same shape.
pub fn modulation_from_hand(
    enabled: EffectMask,
    proximity: f32,
    height: f32,
    spread: f32,
    x_position: f32,
) -> Modulation {
    let proximity = proximity.clamp(0.0, 1.0);
    let height = height.clamp(0.0, 1.0);
    let x_position = x_position.clamp(0.0, 1.0);
    let open = openness(spread);

    Modulation {
        enabled,
        // A hand pushed towards the lens opens the filter; one backing off darkens it.
        cutoff_hz: CUTOFF_MIN_HZ + (CUTOFF_MAX_HZ - CUTOFF_MIN_HZ) * proximity,
        // Higher in the frame = faster wobble; a wider-open hand = deeper wobble.
        vibrato_hz: VIBRATO_RATE_MIN_HZ + (VIBRATO_RATE_MAX_HZ - VIBRATO_RATE_MIN_HZ) * height,
        vibrato_cents: VIBRATO_MAX_CENTS * open,
        // Sweeping the hand across the frame walks the delay time.
        delay_ms: DELAY_MAX_MS * x_position,
        feedback: FEEDBACK_MAX * proximity,
    }
}

/// One smoothing step towards `target`. The enable mask switches immediately -
/// clicking a toggle should be audible on the note being held - while the levels
/// slide, so turning a stage on ramps it in instead of stepping it.
pub fn blend(current: Modulation, target: Modulation, alpha: f32) -> Modulation {
    let alpha = alpha.clamp(0.0, 1.0);
    let mix = |previous: f32, next: f32| previous + (next - previous) * alpha;
    Modulation {
        enabled: target.enabled,
        cutoff_hz: mix(current.cutoff_hz, target.cutoff_hz),
        vibrato_hz: mix(current.vibrato_hz, target.vibrato_hz),
        vibrato_cents: mix(current.vibrato_cents, target.vibrato_cents),
        delay_ms: mix(current.delay_ms, target.delay_ms),
        feedback: mix(current.feedback, target.feedback),
    }
}

/// Is this worth a message on the bus? A held hand produces almost-identical
/// numbers frame after frame; sending them would put thirty no-ops a second into a
/// channel that only needs to carry change.
///
/// The deadbands are deliberately coarse: they are all well under the smallest
/// movement a player can hear (a quarter of a semitone, a millisecond of echo), so a
/// change that passes them is a change someone can notice.
pub const CUTOFF_DEADBAND_HZ: f32 = 2.0;
pub const VIBRATO_RATE_DEADBAND_HZ: f32 = 0.1;
pub const VIBRATO_DEPTH_DEADBAND_CENTS: f32 = 0.5;
pub const DELAY_DEADBAND_MS: f32 = 2.0;
pub const FEEDBACK_DEADBAND: f32 = 0.01;

pub fn is_different(a: &Modulation, b: &Modulation) -> bool {
    if a.enabled != b.enabled {
        return true;
    }
    (a.cutoff_hz - b.cutoff_hz).abs() > CUTOFF_DEADBAND_HZ
        || (a.vibrato_hz - b.vibrato_hz).abs() > VIBRATO_RATE_DEADBAND_HZ
        || (a.vibrato_cents - b.vibrato_cents).abs() > VIBRATO_DEPTH_DEADBAND_CENTS
        || (a.delay_ms - b.delay_ms).abs() > DELAY_DEADBAND_MS
        || (a.feedback - b.feedback).abs() > FEEDBACK_DEADBAND
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_chain_is_off_and_neutral() {
        let modulation = Modulation::default();
        assert_eq!(modulation.enabled, EffectMask::NONE);
        assert_eq!(modulation.cutoff_hz, NEUTRAL_CUTOFF_HZ);
        assert_eq!(modulation.vibrato_cents, 0.0);
        assert_eq!(modulation.delay_ms, 0.0);
    }

    #[test]
    fn effect_names_round_trip() {
        for effect in Effect::all() {
            assert_eq!(
                Effect::parse(effect.name()),
                Some(effect),
                "{}",
                effect.name()
            );
        }
        assert_eq!(Effect::parse("echo"), Some(Effect::Delay), "aliases work");
        assert_eq!(Effect::parse("elephant"), None);
    }

    #[test]
    fn a_mask_parses_names_and_rejects_typos() {
        let mask = EffectMask::parse("lowpass, delay").expect("valid list");
        assert!(mask.is_on(Effect::Lowpass));
        assert!(mask.is_on(Effect::Delay));
        assert!(!mask.is_on(Effect::Vibrato));
        assert_eq!(mask.names(), "lowpass, delay");

        assert_eq!(EffectMask::parse(""), Some(EffectMask::NONE));
        assert_eq!(EffectMask::parse("none"), Some(EffectMask::NONE));
        assert_eq!(EffectMask::parse("all"), Some(EffectMask::ALL));
        assert_eq!(
            EffectMask::parse("lowpass,xyz"),
            None,
            "a typo must not be ignored"
        );
    }

    #[test]
    fn toggling_a_stage_flips_only_its_bit() {
        let mut mask = EffectMask::NONE;
        mask.set(Effect::Vibrato, true);
        mask.set(Effect::Vibrato, true);
        assert_eq!(
            mask,
            EffectMask::from_bits(2),
            "setting twice is not double-counting"
        );
        mask.set(Effect::Lowpass, true);
        mask.set(Effect::Vibrato, false);
        assert_eq!(mask.names(), "lowpass");
    }

    #[test]
    fn the_band_saturates_at_both_ends() {
        let far = modulation_from_hand(EffectMask::ALL, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(far.cutoff_hz, CUTOFF_MIN_HZ);
        assert_eq!(far.vibrato_cents, 0.0);
        assert_eq!(far.delay_ms, 0.0);

        // Out-of-range inputs clamp instead of running past the band.
        let beyond = modulation_from_hand(EffectMask::ALL, 4.0, 2.0, 9.0, 2.0);
        assert_eq!(beyond.cutoff_hz, CUTOFF_MAX_HZ);
        assert_eq!(beyond.vibrato_hz, VIBRATO_RATE_MAX_HZ);
        assert_eq!(beyond.vibrato_cents, VIBRATO_MAX_CENTS);
        assert_eq!(beyond.delay_ms, DELAY_MAX_MS);

        let behind = modulation_from_hand(EffectMask::ALL, -1.0, -1.0, -1.0, -1.0);
        assert_eq!(
            behind.cutoff_hz, CUTOFF_MIN_HZ,
            "a hand below the band sits at the bottom of it"
        );
        assert_eq!(behind.vibrato_hz, VIBRATO_RATE_MIN_HZ);
    }

    #[test]
    fn each_source_moves_exactly_one_parameter() {
        let baseline = modulation_from_hand(EffectMask::ALL, 0.5, 0.5, 0.7, 0.5);

        let closer = modulation_from_hand(EffectMask::ALL, 0.9, 0.5, 0.7, 0.5);
        assert!(closer.cutoff_hz > baseline.cutoff_hz);
        assert_eq!(
            closer.delay_ms, baseline.delay_ms,
            "proximity must not move delay time"
        );

        let wider = modulation_from_hand(EffectMask::ALL, 0.5, 0.5, 0.7, 0.9);
        assert!(wider.delay_ms > baseline.delay_ms);
        assert_eq!(wider.cutoff_hz, baseline.cutoff_hz);
    }

    #[test]
    fn blending_slides_to_neutral_rather_than_snapping() {
        // One second of frames at the default 450 ms time constant: alpha is
        // frame_interval / fade, so 30 frames is about two time constants.
        let mut current = modulation_from_hand(EffectMask::ALL, 1.0, 1.0, 1.0, 1.0);
        let target = Modulation::neutral_with(EffectMask::ALL);
        for _ in 0..30 {
            current = blend(current, target, 0.073);
        }
        assert!(
            (current.cutoff_hz - NEUTRAL_CUTOFF_HZ).abs() < 600.0,
            "after two time constants the filter should be most of the way open, got {}",
            current.cutoff_hz
        );
        assert!(current.vibrato_cents < 6.0, "the wobble faded out");
        assert!(current.delay_ms < 75.0, "the delay faded out");
        assert!(current.delay_ms > 0.0, "it faded, it did not snap");
    }

    #[test]
    fn an_enabled_stage_is_adopted_at_once_while_its_level_fades() {
        let current = Modulation::neutral();
        let target = modulation_from_hand(EffectMask::from_bits(2), 0.5, 0.5, 0.5, 0.5);
        let blended = blend(current, target, 0.1);
        assert!(
            blended.enabled.is_on(Effect::Vibrato),
            "a toggle is not smoothed"
        );
        assert!(
            blended.vibrato_cents < target.vibrato_cents,
            "the level ramps in"
        );
    }

    #[test]
    fn noise_does_not_ping_the_bus() {
        let a = modulation_from_hand(EffectMask::ALL, 0.5, 0.5, 0.7, 0.5);
        let b = modulation_from_hand(EffectMask::ALL, 0.5002, 0.5002, 0.7001, 0.5001);
        assert!(
            !is_different(&a, &b),
            "sub-threshold jitter is not a change"
        );

        let moved = modulation_from_hand(EffectMask::ALL, 0.6, 0.5, 0.7, 0.5);
        assert!(is_different(&a, &moved), "a real move is a change");
    }

    #[test]
    /// The thresholds are in units a listener can reason about, not arbitrary numbers:
    /// a wobble that moves the echo by less than a rounding error, or the vibrato by
    /// less than a cent, is not a change worth sending.
    fn the_deadbands_ignore_jitter_smaller_than_a_cent_or_a_millisecond() {
        let steady = modulation_from_hand(EffectMask::ALL, 0.5, 0.5, 0.7, 0.5);
        let wobbling = modulation_from_hand(
            EffectMask::ALL,
            0.5 + 0.0002,
            0.5 + 0.0002,
            0.7 + 0.001,
            0.5 + 0.002,
        );
        assert!(
            !is_different(&steady, &wobbling),
            "noise below the deadbands"
        );

        // A cent of vibrato depth at a mid-register pitch is under a hertz of pitch
        // move; the deadband must be finer than that or real expression would be lost.
        let cent_at_440 = cents_to_hz(1.0, 440.0);
        assert!(cent_at_440 < 1.0, "a cent at 440 Hz is {} Hz", cent_at_440);
        assert!(
            VIBRATO_DEPTH_DEADBAND_CENTS * cent_at_440 < 1.0,
            "the vibrato deadband drops more than a hertz of pitch"
        );
    }

    #[test]
    fn a_cent_is_a_cent() {
        // 100 cents is a semitone: 440 Hz -> 466.16 Hz, a rise of 26.16 Hz. The linear
        // approximation lands a little under the exact value, which is the price of
        // not calling `powf` once per audio sample.
        let semitone = cents_to_hz(100.0, 440.0);
        assert!(
            (semitone - 26.16).abs() < 1.3,
            "100 cents gave {semitone} Hz, exact is 26.16"
        );
        // Half a semitone is half the rise, to first order.
        assert!((cents_to_hz(50.0, 440.0) - semitone / 2.0).abs() < 0.5);
    }
}
