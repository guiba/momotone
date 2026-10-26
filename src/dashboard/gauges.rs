//! Small canvas-drawn instruments: a level bar and a rolling sparkline.
//!
//! These are hand-drawn rather than `progress_bar` widgets because the HUD
//! wants exact colours, segment ticks, and an overlay history trace.

use iced::border::Radius;
use iced::widget::canvas::{Frame, Geometry, LineCap, Path, Program, Stroke};
use iced::{mouse, Color, Point, Rectangle, Renderer, Size, Theme};

const BG: Color = Color::from_rgb8(26, 30, 36);
const GRID: Color = Color::from_rgb8(56, 63, 72);

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
        let filled = bounds.width * self.value.clamp(0.0, 1.0);
        if filled > 0.0 {
            frame.fill(
                &Path::rounded_rectangle(Point::new(0.0, 0.0), Size::new(filled, height), radius),
                self.color,
            );
        }

        // Quarter ticks.
        for i in 1..4 {
            let x = bounds.width * i as f32 / 4.0;
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

        // Baseline grid.
        for i in 1..4 {
            let y = bounds.height * i as f32 / 4.0;
            frame.stroke(
                &Path::line(Point::new(0.0, y), Point::new(bounds.width, y)),
                Stroke::default().with_color(GRID).with_width(1.0),
            );
        }

        if self.samples.len() < 2 {
            return vec![frame.into_geometry()];
        }

        let (min, max) = self.range;
        let span = (max - min).max(1e-6);
        let n = self.samples.len();
        let step = bounds.width / (n - 1) as f32;

        let point = |i: usize| {
            let value = self.samples[i].clamp(min, max);
            Point::new(
                i as f32 * step,
                bounds.height - (value - min) / span * bounds.height,
            )
        };

        let trace = Path::new(|builder| {
            for i in 0..n {
                if i == 0 {
                    builder.move_to(point(i));
                } else {
                    builder.line_to(point(i));
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
        frame.fill(&Path::circle(point(n - 1), 3.0), self.color);

        vec![frame.into_geometry()]
    }
}
