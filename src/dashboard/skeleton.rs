//! The stage: the camera feed plus the hand skeleton, drawn on one iced canvas.
//!
//! Landmarks and the detector box arrive in *frame pixels*; `Mapping` scales
//! them into the widget bounds while preserving the frame's aspect ratio, and
//! the video is drawn into exactly the same letterboxed rectangle, so the
//! overlay always lines up with the image.

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

/// Panel background and the dark under-stroke that keeps the bones readable on
/// a bright camera frame.
const PANEL: Color = Color::from_rgb8(16, 19, 24);
const OUTLINE: Color = Color::from_rgb8(10, 12, 16);

#[derive(Debug, Clone)]
pub struct Skeleton {
    /// The camera frame as RGBA pixels, drawn under the skeleton.
    pub image: Option<Handle>,
    pub landmarks: Vec<Landmark>,
    pub frame_size: (usize, usize),
    pub hand_box: Option<(f32, f32, f32, f32)>,
    pub show_box: bool,
    pub show_labels: bool,
    pub detected: bool,
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
        frame.fill(
            &Path::rectangle(Point::new(0.0, 0.0), bounds.size()),
            PANEL,
        );

        // Camera feed, letterboxed to the frame aspect ratio.
        let stage = Rectangle {
            x: map.offset_x,
            y: map.offset_y,
            width: bounds.width - 2.0 * map.offset_x,
            height: bounds.height - 2.0 * map.offset_y,
        };

        if let Some(handle) = &self.image {
            frame.draw_image(stage, Image::new(handle.clone()));
        } else {
            // Mock mode / no camera: a faint grid so the stage is not blank.
            for i in 1..8 {
                let x = stage.x + stage.width * i as f32 / 8.0;
                frame.stroke(
                    &Path::line(Point::new(x, stage.y), Point::new(x, stage.y + stage.height)),
                    Stroke::default()
                        .with_color(Color::from_rgb8(30, 35, 42))
                        .with_width(1.0),
                );
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

        // Bones: a dark under-stroke first, then the coloured bone on top, so
        // the skeleton reads against both dark and bright frames.
        for (a, b, group) in CONNECTIONS {
            let line = Path::line(map.point(&self.landmarks[a]), map.point(&self.landmarks[b]));
            frame.stroke(
                &line,
                Stroke::default().with_color(OUTLINE).with_width(5.0),
            );
            frame.stroke(
                &line,
                Stroke::default()
                    .with_color(GROUP_COLORS[group])
                    .with_width(2.5),
            );
        }

        // Joints; fingertips are drawn larger so the finger chain reads clearly.
        for (index, landmark) in self.landmarks.iter().enumerate() {
            let position = map.point(landmark);
            let is_tip = FINGER_TIPS.contains(&index);
            let radius = if is_tip { 5.5 } else { 3.0 };
            // Landmark 0 is the wrist, 1..4 the thumb, then four per finger.
            let group = if index == 0 {
                0
            } else if index <= 4 {
                1
            } else {
                2 + (index - 5) / 4
            };
            let joint = Path::circle(position, radius);
            frame.fill(&joint, OUTLINE);
            frame.fill(
                &Path::circle(position, radius - 1.2),
                if index == 0 {
                    Color::from_rgb8(240, 244, 248)
                } else {
                    GROUP_COLORS[group]
                },
            );
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
