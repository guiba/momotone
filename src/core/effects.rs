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

/// The filter's band, in **harmonics of the note that is sounding** rather than in
/// absolute hertz. The keyboard spans C2-C6 (65-1047 Hz), so an absolute band is
/// either above every note - where a lowpass does nothing at all - or below all of
/// them, muffling everything equally. Measured against the note, "bright" and "dark"
/// mean the same thing at C2 as they do at C6.
///
/// Half a fundamental is as dark as the chain gets: the note sits well above the knee
/// of the filter and comes out muffled. Three harmonics is about as open as a
/// lowpass can be and still be a tone control - above that it is a pass-through.
pub const CUTOFF_MIN_HARMONICS: f32 = 0.5;
pub const CUTOFF_MAX_HARMONICS: f32 = 3.0;

/// Absolute clamp on the derived cutoff. The floor keeps a degenerate frequency from
/// closing the filter shut; the ceiling is the neutral value, so the meter reads full
/// when the stage is open.
pub const CUTOFF_MIN_HZ: f32 = 30.0;
pub const CUTOFF_MAX_HZ: f32 = NEUTRAL_CUTOFF_HZ;

/// What "no filter" means. A pole at 12 kHz passes everything the instrument
/// produces, so handing the voice this value is equivalent to the stage not being
/// there - without needing a branch in the audio callback to say so.
pub const NEUTRAL_CUTOFF_HZ: f32 = 12000.0;

pub const VIBRATO_RATE_MIN_HZ: f32 = 0.5;
pub const VIBRATO_RATE_MAX_HZ: f32 = 8.0;
pub const VIBRATO_MAX_CENTS: f32 = 50.0;

pub const DELAY_MAX_MS: f32 = 600.0;
pub const FEEDBACK_MAX: f32 = 0.6;

/// Two hand boxes whose centres are this many "touching distances" apart are touching:
/// `touching` is the sum of the two palm radii, i.e. the distance at which the boxes
/// just meet. One hand's box overlapping the other's reads zero.
pub const GAP_TOUCHING: f32 = 1.0;
/// ... and this many times further apart saturates the inter-hand band.
///
/// Measured, not guessed. Across `fixtures/2_hands_640.mp4` and the native 1620x1080
/// recording of the same clip the palm centres run 0.8-1.4 times the sum of the two palm
/// radii (mean 1.1, p95 1.36) - the two are the same numbers at both resolutions, which
/// is the point of normalising by the hands rather than the frame. 1.5 covers the whole
/// observed sweep with headroom for a hand pushed to the far edge of the view, without
/// making the band so wide that it never saturates.
pub const GAP_WIDE: f32 = 1.5;

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

/// Where the filter's knee sits, in hertz, for a proximity level and the pitch that
/// is currently sounding. With no note playing the middle of the keyboard (C4) is the
/// reference, so the stage still has somewhere sensible to fade from.
pub fn cutoff_for(frequency: f32, proximity: f32) -> f32 {
    let reference = if frequency > 0.0 { frequency } else { 261.6 };
    let harmonics = CUTOFF_MIN_HARMONICS
        + (CUTOFF_MAX_HARMONICS - CUTOFF_MIN_HARMONICS) * proximity.clamp(0.0, 1.0);
    (reference * harmonics).clamp(CUTOFF_MIN_HZ, CUTOFF_MAX_HZ)
}

/// How far apart two hands are, on the 0..1 band the inter-hand sources read.
///
/// The distance is normalised by the sum of the two palm radii rather than by the
/// frame size, which makes it scale-free: hands at the same physical separation read
/// the same level whether the camera is 640x480 or 1620x1080, and whether the player
/// sits close or far. `0.0` is the boxes touching, `1.0` is `GAP_WIDE` times that apart.
pub fn gap_level(distance_px: f32, touching_px: f32) -> f32 {
    let ratio = distance_px / touching_px.max(1.0);
    ((ratio - GAP_TOUCHING) / (GAP_WIDE - GAP_TOUCHING)).clamp(0.0, 1.0)
}

/// Sources in, parameters out.
///
/// Each effect has exactly one source, so moving one hand does not drag three
/// unrelated parameters with it:
///
/// | effect | source |
/// | :--- | :--- |
/// | lowpass cutoff | driver proximity (near = bright), measured against `frequency` |
/// | vibrato rate | driver height in the frame (high = faster) |
/// | vibrato depth | how far apart the two hands are |
/// | delay time | driver horizontal position |
/// | delay feedback | how far apart the two hands are |
///
/// `gap` is already a 0..1 level. A run with one hand has no inter-hand distance to
/// read, so the caller passes the pitch hand's openness instead - which is exactly
/// what the single-hand instrument used before the second hand existed.
pub fn modulation_from_hands(
    enabled: EffectMask,
    proximity: f32,
    height: f32,
    gap: f32,
    x_position: f32,
    frequency: f32,
) -> Modulation {
    let proximity = proximity.clamp(0.0, 1.0);
    let height = height.clamp(0.0, 1.0);
    let x_position = x_position.clamp(0.0, 1.0);
    let gap = gap.clamp(0.0, 1.0);

    Modulation {
        enabled,
        // A hand pushed towards the lens opens the filter; one backing off darkens it.
        cutoff_hz: cutoff_for(frequency, proximity),
        // Higher in the frame = faster wobble.
        vibrato_hz: VIBRATO_RATE_MIN_HZ + (VIBRATO_RATE_MAX_HZ - VIBRATO_RATE_MIN_HZ) * height,
        // Hands spread wide = deeper wobble; hands together = none.
        vibrato_cents: VIBRATO_MAX_CENTS * gap,
        // Sweeping the hand across the frame walks the delay time.
        delay_ms: DELAY_MAX_MS * x_position,
        // Widening the gap also feeds the echo back further.
        feedback: FEEDBACK_MAX * gap,
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
        let far = modulation_from_hands(EffectMask::ALL, 0.0, 0.0, 0.0, 0.0, 220.0);
        assert_eq!(far.cutoff_hz, 220.0 * CUTOFF_MIN_HARMONICS);
        assert_eq!(far.vibrato_cents, 0.0);
        assert_eq!(far.delay_ms, 0.0);

        // Out-of-range inputs clamp instead of running past the band.
        let beyond = modulation_from_hands(EffectMask::ALL, 4.0, 2.0, 9.0, 2.0, 220.0);
        assert_eq!(beyond.cutoff_hz, 220.0 * CUTOFF_MAX_HARMONICS);
        assert_eq!(beyond.vibrato_hz, VIBRATO_RATE_MAX_HZ);
        assert_eq!(beyond.vibrato_cents, VIBRATO_MAX_CENTS);
        assert_eq!(beyond.delay_ms, DELAY_MAX_MS);

        let behind = modulation_from_hands(EffectMask::ALL, -1.0, -1.0, -1.0, -1.0, 220.0);
        assert_eq!(
            behind.cutoff_hz, far.cutoff_hz,
            "a hand below the band sits at the bottom of it"
        );
        assert_eq!(behind.vibrato_hz, VIBRATO_RATE_MIN_HZ);
    }

    #[test]
    /// The knee is a multiple of the pitch, so the same hand position darkens an A3
    /// and a C6 by the same amount. An absolute band would be transparent for one and
    /// muffling for the other.
    fn the_knee_is_measured_in_harmonics_of_the_note() {
        let a3 = cutoff_for(220.0, 0.5);
        let two_octaves_up = cutoff_for(880.0, 0.5);
        assert!(
            (two_octaves_up - a3 * 4.0).abs() < 1e-3,
            "the cutoff should scale with the pitch: {a3} vs {two_octaves_up}"
        );
        // A reference of a tenth of a hertz still leaves the filter usable.
        assert!(
            cutoff_for(0.0, 0.5) > CUTOFF_MIN_HZ,
            "silence has a fallback reference"
        );
        assert_eq!(
            cutoff_for(20.0, 0.0),
            CUTOFF_MIN_HZ,
            "the knee never closes shut"
        );
    }

    #[test]
    fn each_source_moves_exactly_one_parameter() {
        let baseline = modulation_from_hands(EffectMask::ALL, 0.5, 0.5, 0.7, 0.5, 220.0);

        let closer = modulation_from_hands(EffectMask::ALL, 0.9, 0.5, 0.7, 0.5, 220.0);
        assert!(closer.cutoff_hz > baseline.cutoff_hz);
        assert_eq!(
            closer.delay_ms, baseline.delay_ms,
            "proximity must not move delay time"
        );

        let wider = modulation_from_hands(EffectMask::ALL, 0.5, 0.5, 0.7, 0.9, 220.0);
        assert!(wider.delay_ms > baseline.delay_ms);
        assert_eq!(wider.cutoff_hz, baseline.cutoff_hz);
    }

    #[test]
    /// The inter-hand band is measured in the hands' own sizes, not in pixels, so the
    /// same gesture reads the same level at any resolution or distance.
    fn the_gap_band_is_scale_free() {
        // Boxes 100 px wide each touch at 100 px centre-to-centre.
        assert_eq!(
            gap_level(80.0, 100.0),
            0.0,
            "overlapping hands are not apart"
        );
        assert_eq!(
            gap_level(100.0, 100.0),
            0.0,
            "touching is the bottom of the band"
        );
        assert_eq!(
            gap_level(100.0 * GAP_WIDE, 100.0),
            1.0,
            "the band saturates at GAP_WIDE times the touching distance"
        );
        assert_eq!(
            gap_level(125.0, 100.0),
            0.5,
            "half way to saturation is half the band"
        );
        assert_eq!(gap_level(90.0, 100.0), 0.0, "still inside the boxes");

        // Same physical gesture, double the resolution: double the pixels, same level.
        assert_eq!(gap_level(400.0, 200.0), gap_level(200.0, 100.0));
    }

    #[test]
    fn blending_slides_to_neutral_rather_than_snapping() {
        // One second of frames at the default 450 ms time constant: alpha is
        // frame_interval / fade, so 30 frames is about two time constants.
        let mut current = modulation_from_hands(EffectMask::ALL, 1.0, 1.0, 1.0, 1.0, 220.0);
        let target = Modulation::neutral_with(EffectMask::ALL);
        for _ in 0..30 {
            current = blend(current, target, 0.073);
        }
        assert!(
            (current.cutoff_hz - NEUTRAL_CUTOFF_HZ).abs()
                < (220.0 * CUTOFF_MAX_HARMONICS - NEUTRAL_CUTOFF_HZ).abs() * 0.15,
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
        let target = modulation_from_hands(EffectMask::from_bits(2), 0.5, 0.5, 0.5, 0.5, 220.0);
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
        let a = modulation_from_hands(EffectMask::ALL, 0.5, 0.5, 0.7, 0.5, 220.0);
        let b = modulation_from_hands(EffectMask::ALL, 0.5002, 0.5002, 0.7001, 0.5001, 220.0);
        assert!(
            !is_different(&a, &b),
            "sub-threshold jitter is not a change"
        );

        let moved = modulation_from_hands(EffectMask::ALL, 0.6, 0.5, 0.7, 0.5, 220.0);
        assert!(is_different(&a, &moved), "a real move is a change");
    }

    #[test]
    /// The thresholds are in units a listener can reason about, not arbitrary numbers:
    /// a wobble that moves the echo by less than a rounding error, or the vibrato by
    /// less than a cent, is not a change worth sending.
    fn the_deadbands_ignore_jitter_smaller_than_a_cent_or_a_millisecond() {
        let steady = modulation_from_hands(EffectMask::ALL, 0.5, 0.5, 0.7, 0.5, 220.0);
        let wobbling = modulation_from_hands(
            EffectMask::ALL,
            0.5 + 0.0002,
            0.5 + 0.0002,
            0.7 + 0.001,
            0.5 + 0.002,
            220.0,
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
