//! The stage: a stylised backdrop with the hand skeleton drawn on top.
//!
//! The webcam feed is off by default — the panel is a dark vertical gradient,
//! a soft centre glow, and a ruler grid locked to the camera frame's pixel
//! grid. Landmarks and the detector box arrive in *frame pixels*; `Mapping`
//! scales them into the widget bounds while preserving the frame's aspect
//! ratio, and the optional feed image is drawn into exactly the same
//! letterboxed rectangle, so the overlay always lines up with the image.

use iced::widget::canvas::{Frame, Geometry, Image, Path, Program, Stroke, Text};
use iced::widget::image::Handle;
use iced::{mouse, Color, Point, Rectangle, Renderer, Size, Theme};

use crate::core::Landmark;

/// MediaPipe `HandLandmarksConnections`: (a, b, finger group) where group 0 is
/// the palm, 1..5 are thumb..pinky. See `visualizer/mod.rs` for the same list
/// in OpenCV form.
const CONNECTIONS: [(usize, usize, usize); 21] = [
    (0, 1, 0),
    (0, 5, 0),
    (5, 9, 0),
    (9, 13, 0),
    (13, 17, 0),
    (0, 17, 0),
    (1, 2, 1),
    (2, 3, 1),
    (3, 4, 1),
    (5, 6, 2),
    (6, 7, 2),
    (7, 8, 2),
    (9, 10, 3),
    (10, 11, 3),
    (11, 12, 3),
    (13, 14, 4),
    (14, 15, 4),
    (15, 16, 4),
    (17, 18, 5),
    (18, 19, 5),
    (19, 20, 5),
];

const GROUP_COLORS: [Color; 6] = [
    Color::from_rgb8(150, 164, 178), // palm
    Color::from_rgb8(255, 107, 107), // thumb
    Color::from_rgb8(78, 161, 255),  // index
    Color::from_rgb8(255, 209, 102), // middle
    Color::from_rgb8(6, 214, 160),   // ring
    Color::from_rgb8(199, 125, 255), // pinky
];

const FINGER_TIPS: [usize; 5] = [4, 8, 12, 16, 20];
/// The outer palm loop, used as a faint hull fill behind the bones.
const PALM_HULL: [usize; 5] = [0, 5, 9, 13, 17];

// --- stage backdrop -----------------------------------------------------------
const PANEL_TOP: Color = Color::from_rgb8(23, 28, 35);
const PANEL_BOTTOM: Color = Color::from_rgb8(11, 13, 17);
const GLOW: Color = Color::from_rgb8(48, 61, 78);
const GRID: Color = Color::from_rgb8(26, 31, 39);
const GRID_MAJOR: Color = Color::from_rgb8(40, 47, 57);
const STAGE_EDGE: Color = Color::from_rgb8(52, 60, 70);

const GRADIENT_STEPS: usize = 24;
const GLOW_RINGS: usize = 12;
/// Camera pixels between grid lines: the grid doubles as a ruler of the frame.
const GRID_STEP_PX: f32 = 32.0;
/// Every Nth grid line is drawn brighter.
const GRID_MAJOR_EVERY: usize = 4;

#[derive(Debug, Clone)]
pub struct Skeleton {
    /// The camera frame as RGBA pixels. Only drawn when `show_feed` is true.
    pub image: Option<Handle>,
    /// Header toggle: put the webcam image back behind the skeleton.
    pub show_feed: bool,
    pub landmarks: Vec<Landmark>,
    pub frame_size: (usize, usize),
    pub hand_box: Option<(f32, f32, f32, f32)>,
    pub show_box: bool,
    pub show_labels: bool,
    pub detected: bool,
    /// Milliseconds since the tracker last saw a hand.
    pub lost_ms: u64,
}

fn tint(color: Color, alpha: f32) -> Color {
    Color {
        r: color.r,
        g: color.g,
        b: color.b,
        a: alpha,
    }
}

fn mix(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: 1.0,
    }
}

struct Mapping {
    scale: f32,
    offset_x: f32,
    offset_y: f32,
}

impl Mapping {
    fn new(frame_size: (usize, usize), bounds: Rectangle) -> Mapping {
        let fw = frame_size.0.max(1) as f32;
        let fh = frame_size.1.max(1) as f32;
        let scale = (bounds.width / fw).min(bounds.height / fh);
        Mapping {
            scale,
            offset_x: (bounds.width - fw * scale) * 0.5,
            offset_y: (bounds.height - fh * scale) * 0.5,
        }
    }

    fn point(&self, landmark: &Landmark) -> Point {
        Point::new(
            self.offset_x + landmark.x * self.scale,
            self.offset_y + landmark.y * self.scale,
        )
    }

    fn rect(&self, x: f32, y: f32, w: f32, h: f32) -> Rectangle {
        Rectangle {
            x: self.offset_x + x * self.scale,
            y: self.offset_y + y * self.scale,
            width: w * self.scale,
            height: h * self.scale,
        }
    }
}

/// Gradient + glow + ruler grid. Drawn every tick; it is a handful of flat
/// primitives, so it costs far less than the RGBA video upload it replaced.
fn backdrop(frame: &mut Frame, bounds: Rectangle, stage: Rectangle, scale: f32) {
    // Vertical gradient over the whole panel.
    let strip = bounds.height / GRADIENT_STEPS as f32;
    for i in 0..GRADIENT_STEPS {
        let t = i as f32 / (GRADIENT_STEPS - 1) as f32;
        frame.fill(
            &Path::rectangle(
                Point::new(0.0, strip * i as f32),
                Size::new(bounds.width, strip + 1.0),
            ),
            mix(PANEL_TOP, PANEL_BOTTOM, t),
        );
    }

    // Soft centre glow, filled ring by ring from the outside in.
    let centre = Point::new(stage.x + stage.width * 0.5, stage.y + stage.height * 0.5);
    let radius = stage.width.min(stage.height) * 0.8;
    for ring in (0..GLOW_RINGS).rev() {
        let t = (ring + 1) as f32 / GLOW_RINGS as f32;
        frame.fill(&Path::circle(centre, radius * t), mix(GLOW, PANEL_BOTTOM, t));
    }

    // Ruler grid, one line every GRID_STEP_PX camera pixels.
    let step = GRID_STEP_PX * scale;
    if step > 3.0 {
        let cols = (stage.width / step).floor() as usize;
        let rows = (stage.height / step).floor() as usize;
        for i in 1..=cols {
            let x = stage.x + step * i as f32;
            frame.stroke(
                &Path::line(Point::new(x, stage.y), Point::new(x, stage.y + stage.height)),
                Stroke::default()
                    .with_color(if i % GRID_MAJOR_EVERY == 0 {
                        GRID_MAJOR
                    } else {
                        GRID
                    })
                    .with_width(1.0),
            );
        }
        for i in 1..=rows {
            let y = stage.y + step * i as f32;
            frame.stroke(
                &Path::line(Point::new(stage.x, y), Point::new(stage.x + stage.width, y)),
                Stroke::default()
                    .with_color(if i % GRID_MAJOR_EVERY == 0 {
                        GRID_MAJOR
                    } else {
                        GRID
                    })
                    .with_width(1.0),
            );
        }
    }

    // The letterboxed frame region, so the coordinate space is visible.
    frame.stroke(
        &Path::rectangle(Point::new(stage.x, stage.y), Size::new(stage.width, stage.height)),
        Stroke::default().with_color(STAGE_EDGE).with_width(1.0),
    );
}

impl<Message> Program<Message> for Skeleton {
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
        let map = Mapping::new(self.frame_size, bounds);

        // The panel itself, so the letterbox bars are part of the stage.
        let stage = Rectangle {
            x: map.offset_x,
            y: map.offset_y,
            width: bounds.width - 2.0 * map.offset_x,
            height: bounds.height - 2.0 * map.offset_y,
        };
        backdrop(&mut frame, bounds, stage, map.scale);

        // Camera feed, only when the `feed` toggle is on.
        if self.show_feed {
            if let Some(handle) = &self.image {
                frame.draw_image(stage, Image::new(handle.clone()));
            }
        }

        if self.landmarks.len() < 21 {
            frame.fill_text(Text {
                content: if self.detected {
                    "hand found, landmark stage failed".to_string()
                } else {
                    "no hand detected - searching...".to_string()
                },
                position: frame.center(),
                max_width: bounds.width,
                color: Color::from_rgb8(140, 150, 160),
                size: 18.0.into(),
                ..Default::default()
            });
            frame.fill_text(Text {
                content: format!("lost for {} ms", self.lost_ms),
                position: Point::new(stage.x + 12.0, stage.y + stage.height - 14.0),
                max_width: stage.width,
                color: Color::from_rgb8(90, 100, 110),
                size: 11.0.into(),
                ..Default::default()
            });
            return vec![frame.into_geometry()];
        }

        // Detector box, when asked for.
        if self.show_box {
            if let Some((x, y, w, h)) = self.hand_box {
                let rect = map.rect(x, y, w, h);
                frame.stroke(
                    &Path::rectangle(
                        Point::new(rect.x, rect.y),
                        Size::new(rect.width, rect.height),
                    ),
                    Stroke::default()
                        .with_color(Color::from_rgb8(255, 190, 90))
                        .with_width(1.5),
                );
            }
        }

        // Palm hull: a translucent fill so the bones read as a hand, not a wire.
        let hull = Path::new(|path| {
            for (i, joint) in PALM_HULL.iter().enumerate() {
                let position = map.point(&self.landmarks[*joint]);
                if i == 0 {
                    path.move_to(position);
                } else {
                    path.line_to(position);
                }
            }
            path.close();
        });
        frame.fill(&hull, tint(GROUP_COLORS[0], 0.12));

        // Bones: a wide translucent pass for the glow, then the crisp bone.
        for (a, b, group) in CONNECTIONS {
            let line = Path::line(map.point(&self.landmarks[a]), map.point(&self.landmarks[b]));
            let color = GROUP_COLORS[group];
            frame.stroke(
                &line,
                Stroke::default().with_color(tint(color, 0.25)).with_width(8.0),
            );
            frame.stroke(
                &line,
                Stroke::default().with_color(color).with_width(2.5),
            );
        }

        // Joints; fingertips get a halo so the finger chains stand out.
        for (index, landmark) in self.landmarks.iter().enumerate() {
            let position = map.point(landmark);
            let is_tip = FINGER_TIPS.contains(&index);
            let radius = if is_tip { 5.5 } else { 3.5 };
            // Landmark 0 is the wrist, 1..4 the thumb, then four per finger.
            let group = if index == 0 {
                0
            } else if index <= 4 {
                1
            } else {
                2 + (index - 5) / 4
            };
            let color = if index == 0 {
                Color::from_rgb8(240, 244, 248)
            } else {
                GROUP_COLORS[group]
            };

            if is_tip {
                frame.fill(&Path::circle(position, radius + 6.0), tint(color, 0.18));
            }
            frame.fill(&Path::circle(position, radius + 2.0), tint(color, 0.35));
            frame.fill(&Path::circle(position, radius), color);
        }

        if self.show_labels {
            for (finger, tip) in FINGER_TIPS.iter().enumerate() {
                let position = map.point(&self.landmarks[*tip]);
                frame.fill_text(Text {
                    content: crate::core::geometry::FINGER_NAMES[finger].to_string(),
                    position: Point::new(position.x + 8.0, position.y - 8.0),
                    max_width: bounds.width,
                    color: GROUP_COLORS[finger + 1],
                    size: 12.0.into(),
                    ..Default::default()
                });
            }
        }

        vec![frame.into_geometry()]
    }
}
