//! Confirmation and grace windows for hand detection.
//!
//! A raw "is the best anchor above the threshold" compare flickers: the winning
//! anchor's score crosses the gate from frame to frame (measured live: 0.97 ->
//! 0.67 -> 0.00 with a hand plainly in view), so the instrument clicks on and off
//! while the hand is sitting still. Requiring a few consecutive sightings before a
//! gesture starts, and tolerating a few misses before declaring the hand gone,
//! turns that into a stable present/absent decision.
//!
//! This deliberately does not know about thresholds: the tracker has already
//! applied its own. The gate only sees the boolean "a hand was found".

#[derive(Debug, Clone, Copy)]
pub struct DetectionGate {
    /// Frames above the threshold needed before a gesture is allowed to start.
    confirm_frames: u32,
    /// Frames of no detection tolerated before the hand is declared lost.
    grace_frames: u32,
    hits: u32,
    misses: u32,
    present: bool,
}

impl DetectionGate {
    pub fn new(confirm_frames: u32, grace_frames: u32) -> Self {
        Self {
            confirm_frames,
            grace_frames,
            hits: 0,
            misses: 0,
            present: false,
        }
    }

    /// Returns whether a gesture may run this frame.
    pub fn observe(&mut self, found: bool) -> bool {
        if found {
            self.misses = 0;
            if self.present {
                return true;
            }
            self.hits += 1;
            self.present = self.hits >= self.confirm_frames.max(1);
            return self.present;
        }

        if self.present {
            if self.misses < self.grace_frames {
                self.misses += 1;
                return true;
            }
            self.present = false;
        }
        self.hits = 0;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hand_needs_two_sightings_to_start() {
        let mut gate = DetectionGate::new(2, 3);
        assert!(!gate.observe(true), "first sighting is not enough");
        assert!(gate.observe(true), "the second confirms it");
        assert!(gate.observe(true));
    }

    #[test]
    fn a_single_blip_does_not_start_a_gesture() {
        let mut gate = DetectionGate::new(2, 3);
        assert!(!gate.observe(true));
        assert!(!gate.observe(false), "the blip resets the count");
        assert!(!gate.observe(true), "a lone sighting is still not a hand");
        assert!(gate.observe(true), "two in a row is");
    }

    #[test]
    fn the_grace_window_survives_a_flicker() {
        let mut gate = DetectionGate::new(2, 3);
        gate.observe(true);
        gate.observe(true);
        // The confidence dips for three frames, then comes back: no release in
        // between, so the note does not click.
        assert!(gate.observe(false), "grace frame 1");
        assert!(gate.observe(false), "grace frame 2");
        assert!(gate.observe(false), "grace frame 3");
        assert!(!gate.observe(false), "four misses is a lost hand");
        assert!(
            !gate.observe(true),
            "a new hand has to earn the confirmation window again"
        );
    }

    #[test]
    fn a_found_hand_cancels_the_grace_clock() {
        let mut gate = DetectionGate::new(1, 2);
        gate.observe(true);
        assert!(gate.observe(false), "miss 1");
        assert!(gate.observe(true), "found again, the grace clock resets");
        assert!(gate.observe(false), "miss 1 again");
        assert!(gate.observe(false), "miss 2");
        assert!(
            !gate.observe(false),
            "the third consecutive miss loses the hand"
        );
    }
}
