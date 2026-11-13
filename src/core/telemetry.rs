//! Rolling stage timings.
//!
//! Real-time regressions are tail events: a single 120 ms inference spike is
//! completely invisible in an average. These rings let the `--debug` summary and
//! the dashboard report p50/p95/p99 instead of "the last frame took X ms".

/// Samples kept per stage. At 30 fps that is ~4 s of history - long enough that a
/// p99 is not dominated by one outlier, short enough that recomputing a percentile
/// costs a sort of 128 floats.
const WINDOW: usize = 128;

#[derive(Debug, Clone)]
pub struct Samples {
    data: [f64; WINDOW],
    len: usize,
    /// Write position; wraps so the ring overwrites the oldest sample.
    head: usize,
}

impl Default for Samples {
    fn default() -> Self {
        Self {
            data: [0.0; WINDOW],
            len: 0,
            head: 0,
        }
    }
}

impl Samples {
    pub fn push(&mut self, value: f64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % WINDOW;
        if self.len < WINDOW {
            self.len += 1;
        }
    }

    pub fn count(&self) -> usize {
        self.len
    }

    /// The kept samples, sorted ascending.
    fn sorted(&self) -> Vec<f64> {
        let mut values = self.data[..self.len].to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values
    }

    /// Nearest-rank percentile, `p` in 0..=1. 0.0 for an empty ring: p95 of 100
    /// samples is the 95th value, not the 96th.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        let sorted = self.sorted();
        let rank = (p.clamp(0.0, 1.0) * self.len as f64).ceil() as usize;
        sorted[rank.max(1) - 1]
    }

    pub fn max(&self) -> f64 {
        self.sorted().last().copied().unwrap_or(0.0)
    }
}

/// One stage's tail behaviour, in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Latency {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub count: usize,
}

pub fn profile(samples: &Samples) -> Latency {
    Latency {
        p50: samples.percentile(0.50),
        p95: samples.percentile(0.95),
        p99: samples.percentile(0.99),
        max: samples.max(),
        count: samples.count(),
    }
}

/// The four stages of the frame loop, side by side: how long the camera read
/// took, how long inference took, how stale the pixels were when they were
/// published, and how long a gesture waited between dispatch and the sink.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StageProfile {
    pub inference: Latency,
    pub capture: Latency,
    pub end_to_end: Latency,
    pub dispatch: Latency,
}

/// Refresh cadence of the published percentiles: a second of frames is plenty for
/// a tail readout, and recomputing four sorts every frame would be wasted work.
pub const PROFILE_REFRESH_FRAMES: u64 = 30;

/// The frame interval a 30 fps capture implies, in milliseconds.
pub const FRAME_BUDGET_MS: f64 = 33.0;

/// Should the optional subsystems (ASCII art, camera preview) be shed?
///
/// The rule is on the *tail*, not the average: a stage whose median fits but whose
/// p95 does not is the one that makes a instrument stutter. `budget_ms <= 0` turns
/// the guard off entirely.
///
/// `count` matters: one slow frame in a three-frame run is not a trend, so the ring
/// needs a few samples before the guard acts.
pub fn shed_subsystems(inference: &Latency, budget_ms: f64) -> bool {
    budget_ms > 0.0 && inference.count >= 5 && inference.p95 > budget_ms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_of_a_known_sequence() {
        let mut ring = Samples::default();
        for i in 1..=100 {
            ring.push(i as f64);
        }
        assert_eq!(ring.count(), 100);
        assert!((ring.percentile(0.50) - 50.0).abs() < 1e-9);
        assert!((ring.percentile(0.95) - 95.0).abs() < 1e-9);
        assert!((ring.percentile(0.99) - 99.0).abs() < 1e-9);
        assert!((ring.max() - 100.0).abs() < 1e-9);
        assert!(
            (ring.percentile(0.0) - 1.0).abs() < 1e-9,
            "p0 is the minimum"
        );
    }

    #[test]
    fn the_ring_keeps_only_the_newest_window() {
        let mut ring = Samples::default();
        // A long stretch of quiet frames, then a slow one.
        for _ in 0..1000 {
            ring.push(1.0);
        }
        ring.push(200.0);
        assert_eq!(ring.count(), WINDOW);
        // The old 1 ms samples still fill the window; the spike survives at the top
        // of it, which is exactly the point - the tail is visible even though the
        // median never moves.
        assert!((ring.percentile(1.0) - 200.0).abs() < 1e-9);
        assert!((ring.percentile(0.50) - 1.0).abs() < 1e-9);
        assert!(ring.percentile(0.99) < 2.0, "one spike is not a p99");
    }

    #[test]
    fn an_empty_ring_reports_zero_rather_than_garbage() {
        let ring = Samples::default();
        assert_eq!(ring.count(), 0);
        assert_eq!(ring.percentile(0.95), 0.0);
        assert_eq!(profile(&ring), Latency::default());
    }

    #[test]
    fn an_outlier_moves_the_tail_but_not_the_median() {
        let mut ring = Samples::default();
        for _ in 0..100 {
            ring.push(8.0);
        }
        // Five slow frames in a hundred: 5 % of the window.
        for _ in 0..5 {
            ring.push(120.0);
        }
        let p = profile(&ring);
        assert!((p.p50 - 8.0).abs() < 1e-9, "the median must not be dragged");
        assert!((p.p99 - 120.0).abs() < 1e-9, "the tail must be visible");
        assert!(p.max == 120.0);
    }
}

#[test]
fn a_tail_over_the_budget_sheds_the_picture_work() {
    let steady = Latency {
        p50: 8.0,
        p95: 12.0,
        p99: 14.0,
        max: 20.0,
        count: 100,
    };
    assert!(
        !shed_subsystems(&steady, FRAME_BUDGET_MS),
        "a healthy stage keeps the art"
    );

    let spiky = Latency {
        p50: 8.0,
        p95: 41.0,
        p99: 60.0,
        max: 90.0,
        count: 100,
    };
    assert!(
        shed_subsystems(&spiky, FRAME_BUDGET_MS),
        "a tail over budget is the signal"
    );
}

#[test]
fn the_guard_is_off_when_no_budget_is_given() {
    let spiky = Latency {
        p50: 8.0,
        p95: 41.0,
        p99: 60.0,
        max: 90.0,
        count: 100,
    };
    for limit in [0.0, -1.0] {
        assert!(
            !shed_subsystems(&spiky, limit),
            "limit {limit} means no guard"
        );
    }
}

#[test]
fn a_stage_is_not_judged_on_three_frames() {
    // A p95 over budget during a three-frame warm-up is noise, not a trend.
    let warmup = Latency {
        p50: 30.0,
        p95: 40.0,
        p99: 40.0,
        max: 40.0,
        count: 3,
    };
    assert!(!shed_subsystems(&warmup, FRAME_BUDGET_MS));
}
