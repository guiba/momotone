//! Small canvas-drawn instruments: a level bar and a rolling sparkline.
//!
//! These are hand-drawn rather than `progress_bar` widgets because the HUD
//! wants exact colours, segment ticks, and an overlay history trace.

use iced::border::Radius;
use iced::widget::canvas::{Frame, Geometry, LineCap, Path, Program, Stroke};
use iced::{mouse, Color, Point, Rectangle, Renderer, Size, Theme};

const BG: Color = Color::from_rgb8(26, 30, 36);
const GRID: Color = Color::from_rgb8(56, 63, 72);

/// Fraction of a track a value fills, clamped so an out-of-range level can never
/// draw outside the widget. A non-finite level reads as silence rather than drawing
/// a path the renderer cannot rasterise. Kept separate from the drawing so the
/// mapping is testable without a renderer.
fn filled_width(width: f32, value: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    width * value.clamp(0.0, 1.0)
}

/// The quarter marks on a level bar: the three interior boundaries.
fn quarter_positions(width: f32) -> Vec<f32> {
    (1..4).map(|i| width * i as f32 / 4.0).collect()
}

/// Sample values mapped to canvas points, oldest sample at x = 0 and the newest at
/// the right edge. The y axis is inverted (0 is the top of the widget), so a rising
/// trace rises on screen; values outside `range` are clamped rather than clipped, and
/// a non-finite sample is treated as the bottom of the range rather than poisoning the
/// path handed to the renderer.
fn trace_points(samples: &[f32], range: (f32, f32), size: Size) -> Vec<Point> {
    if samples.len() < 2 {
        return Vec::new();
    }
    let (min, max) = range;
    let span = (max - min).max(1e-6);
    let step = size.width / (samples.len() - 1) as f32;
    samples
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let value = if value.is_finite() {
                value.clamp(min, max)
            } else {
                min
            };
            Point::new(
                i as f32 * step,
                size.height - (value - min) / span * size.height,
            )
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct Meter {
    /// 0.0..1.0
    pub value: f32,
    pub color: Color,
}

impl<Message> Program<Message> for Meter {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let mut frame = Frame::new(renderer, bounds.size());
        let height = bounds.height;
        let radius = Radius::new(height * 0.5);

        // Track.
        frame.fill(
            &Path::rounded_rectangle(
                Point::new(0.0, 0.0),
                Size::new(bounds.width, height),
                radius,
            ),
            BG,
        );

        // Level.
        let filled = filled_width(bounds.width, self.value);
        if filled > 0.0 {
            frame.fill(
                &Path::rounded_rectangle(Point::new(0.0, 0.0), Size::new(filled, height), radius),
                self.color,
            );
        }

        // Quarter ticks.
        for x in quarter_positions(bounds.width) {
            frame.stroke(
                &Path::line(Point::new(x, 0.0), Point::new(x, height)),
                Stroke::default().with_color(GRID).with_width(1.0),
            );
        }

        vec![frame.into_geometry()]
    }
}

#[derive(Debug, Clone)]
pub struct Sparkline {
    /// Most recent samples, oldest first.
    pub samples: Vec<f32>,
    pub range: (f32, f32),
    pub color: Color,
}

impl<Message> Program<Message> for Sparkline {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let mut frame = Frame::new(renderer, bounds.size());

        frame.fill(&Path::rectangle(Point::new(0.0, 0.0), bounds.size()), BG);

        // Baseline grid, drawn even when there is nothing to trace yet.
        for y in quarter_positions(bounds.height) {
            frame.stroke(
                &Path::line(Point::new(0.0, y), Point::new(bounds.width, y)),
                Stroke::default().with_color(GRID).with_width(1.0),
            );
        }

        let points = trace_points(&self.samples, self.range, bounds.size());
        // A sparkline with fewer than two samples has nothing to connect; the grid
        // above is still worth showing.
        let Some(head) = points.last() else {
            return vec![frame.into_geometry()];
        };
        let trace = Path::new(|builder| {
            for (i, point) in points.iter().enumerate() {
                if i == 0 {
                    builder.move_to(*point);
                } else {
                    builder.line_to(*point);
                }
            }
        });

        frame.stroke(
            &trace,
            Stroke::default()
                .with_color(self.color)
                .with_width(2.0)
                .with_line_cap(LineCap::Round),
        );

        // Head marker.
        frame.fill(&Path::circle(*head, 3.0), self.color);

        vec![frame.into_geometry()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_level_outside_the_track_is_clamped_instead_of_overdrawing() {
        assert_eq!(filled_width(100.0, 0.5), 50.0);
        assert_eq!(filled_width(100.0, 1.0), 100.0);
        assert_eq!(filled_width(100.0, 1.5), 100.0);
        assert_eq!(filled_width(100.0, -0.5), 0.0);
        assert_eq!(filled_width(100.0, f32::NAN), 0.0);
    }

    #[test]
    fn the_marks_on_a_bar_are_at_its_quarters() {
        assert_eq!(quarter_positions(100.0), vec![25.0, 50.0, 75.0]);
    }

    #[test]
    fn a_rising_trace_rises_on_screen() {
        // Canvas y grows downwards, so the oldest (lowest) sample must sit at the
        // bottom and the newest (highest) at the top.
        let points = trace_points(&[0.0, 0.5, 1.0], (0.0, 1.0), Size::new(100.0, 50.0));
        assert_eq!(points.len(), 3);
        assert_eq!(points[0], Point::new(0.0, 50.0));
        assert_eq!(points[1], Point::new(50.0, 25.0));
        assert_eq!(points[2], Point::new(100.0, 0.0));
    }

    #[test]
    fn samples_beyond_the_displayed_range_are_clamped_not_wrapped() {
        let points = trace_points(&[2.0, -1.0], (0.0, 1.0), Size::new(100.0, 40.0));
        assert_eq!(points[0], Point::new(0.0, 0.0));
        assert_eq!(points[1], Point::new(100.0, 40.0));
    }

    #[test]
    fn a_flat_range_does_not_divide_by_zero() {
        let points = trace_points(&[1.0, 1.0, 1.0], (1.0, 1.0), Size::new(60.0, 30.0));
        assert!(points.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }

    #[test]
    fn a_lost_sample_sits_on_the_baseline_instead_of_breaking_the_path() {
        let points = trace_points(&[0.5, f32::NAN, 1.0], (0.0, 1.0), Size::new(100.0, 50.0));
        assert_eq!(points[1], Point::new(50.0, 50.0));
    }

    #[test]
    fn a_single_sample_is_not_a_trace() {
        assert!(trace_points(&[0.5], (0.0, 1.0), Size::new(100.0, 50.0)).is_empty());
        assert!(trace_points(&[], (0.0, 1.0), Size::new(100.0, 50.0)).is_empty());
    }
}
