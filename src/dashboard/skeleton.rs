//! The stage: a degree x octave note grid with the hand skeleton drawn over it.
//!
//! The grid *is* the instrument: columns are the scale's degrees, rows are
//! octaves, and every cell is coloured by its pitch class. Each note keeps a
//! brightness that decays exponentially, so the cell you just released fades out
//! while the one you are holding glows.
//!
//! Landmarks and the detector box arrive in *frame pixels*; `Mapping` scales
//! them into the widget bounds while preserving the frame's aspect ratio, and
//! the optional feed image is drawn into exactly the same letterboxed
//! rectangle, so the overlay always lines up with the image.

use std::cell::RefCell;

use iced::widget::canvas::{Cache, Frame, Geometry, Image, Path, Program, Stroke, Text};
use iced::widget::image::Handle;
use iced::{mouse, Color, Point, Rectangle, Renderer, Size, Theme};

use crate::core::grid::GridLayout;
use crate::core::Landmark;
use crate::core::{NOTE_LOW, NOTE_MARGIN};
use crate::dashboard::metrics::{note_for_frequency, note_frequency, note_index, note_name};
use crate::presentation::ascii::{AsciiGrid, Ramp};

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

// --- keyboard ---------------------------------------------------------------
/// One hue per pitch class, stepping 30° around the colour wheel: C red, D
/// yellow-green, E green, F cyan, G blue, A violet, B magenta...
const PITCH_COLORS: [Color; 12] = [
    Color::from_rgb8(228, 88, 88),  // C
    Color::from_rgb8(228, 158, 88), // C#
    Color::from_rgb8(228, 228, 88), // D
    Color::from_rgb8(158, 228, 88), // D#
    Color::from_rgb8(88, 228, 88),  // E
    Color::from_rgb8(88, 228, 158), // F
    Color::from_rgb8(88, 158, 228), // F#
    Color::from_rgb8(88, 228, 228), // G
    Color::from_rgb8(88, 88, 228),  // G#
    Color::from_rgb8(158, 88, 228), // A
    Color::from_rgb8(228, 88, 158), // A#
    Color::from_rgb8(228, 88, 228), // B
];

const PANEL_TOP: Color = Color::from_rgb8(23, 28, 35);
const PANEL_BOTTOM: Color = Color::from_rgb8(11, 13, 17);
const GLOW: Color = Color::from_rgb8(48, 61, 78);
const GRID: Color = Color::from_rgb8(26, 31, 39);
const GRID_MAJOR: Color = Color::from_rgb8(44, 52, 62);
const STAGE_EDGE: Color = Color::from_rgb8(52, 60, 70);
const MUTED: Color = Color::from_rgb8(122, 132, 142);

const GRADIENT_STEPS: usize = 24;
const GLOW_RINGS: usize = 12;
/// Below this the trail is too dim to be worth a fill.
const TRAIL_EPS: f32 = 0.02;

// --- ascii art -------------------------------------------------------------
/// A monospaced glyph is roughly 0.6 em wide and 1 em tall. Deriving the cell
/// grid from that ratio is what makes the art fill the panel instead of either
/// stretching or leaving a band of empty panel at the bottom.
const GLYPH_ASPECT: f32 = 0.6;
/// Smallest glyph that still reads as art rather than noise; the grid is
/// coarsened until every cell is at least this tall.
const MIN_GLYPH_SIZE: f32 = 6.0;
/// A row darker than this is skipped entirely - an empty row costs a text run.
const ROW_SILENT: f32 = 0.02;

/// The keyboard's playable band and the size of one cell, derived from the stage
/// and the layout only. `NOTE_MARGIN` is inset on both axes so a cell lines up
/// with the position the strategy reads - the lit cell is the one the hand is
/// actually over. Because this is computed separately from the drawing, the
/// tests can check that alignment without a window.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Keyboard {
    stage: Rectangle,
    band: Rectangle,
    cell_w: f32,
    cell_h: f32,
}

impl Keyboard {
    fn new(stage: Rectangle, layout: &GridLayout) -> Option<Keyboard> {
        if layout.is_empty() {
            return None;
        }
        let inset_x = NOTE_MARGIN * stage.width;
        let inset_y = NOTE_MARGIN * stage.height;
        let band = Rectangle {
            x: stage.x + inset_x,
            y: stage.y + inset_y,
            width: stage.width - 2.0 * inset_x,
            height: stage.height - 2.0 * inset_y,
        };
        Some(Keyboard {
            stage,
            band,
            cell_w: band.width / layout.cols as f32,
            cell_h: band.height / layout.rows as f32,
        })
    }

    /// The rectangle of cell `index` (row-major, top row first).
    fn cell(&self, index: usize, layout: &GridLayout) -> Rectangle {
        let row = index / layout.cols;
        let col = index % layout.cols;
        Rectangle {
            x: self.band.x + self.cell_w * col as f32,
            y: self.band.y + self.cell_h * row as f32,
            width: self.cell_w,
            height: self.cell_h,
        }
    }
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

/// The colour of a pitch class (0 = C, 11 = B), shared by the keyboard and the
/// ASCII backdrop so the art always matches the key being played.
fn pitch_color(note: u8) -> Color {
    PITCH_COLORS[(note % 12) as usize]
}

/// The cell grid that tiles the panel: glyph size, then how many columns and
/// rows fit at that size. A glyph is `GLYPH_ASPECT` em wide and 1 em tall, so
/// deriving the counts from the size is what stops the art from stretching.
/// `None` when the panel is too small for the art to read as art.
fn cell_grid(grid: &AsciiGrid, bounds: Rectangle) -> Option<(usize, usize, f32)> {
    if grid.is_empty() || bounds.width < 40.0 || bounds.height < 20.0 {
        return None;
    }
    // Use the grid's own column count when the panel can hold it at a legible
    // size; only a panel too narrow for the glyph forces a coarser grid. Doing
    // it this way avoids a divide/multiply round trip whose float error would
    // silently drop the last column.
    let ideal = bounds.width / (grid.cols as f32 * GLYPH_ASPECT);
    let (size, cols) = if ideal < MIN_GLYPH_SIZE {
        (
            MIN_GLYPH_SIZE,
            ((bounds.width / (MIN_GLYPH_SIZE * GLYPH_ASPECT)).floor() as usize).max(1),
        )
    } else {
        (ideal, grid.cols)
    };
    let rows = ((bounds.height / size + 1e-6).floor() as usize).max(1);
    Some((cols, rows, size))
}

/// The colour the backdrop should take: the pitch colours of the keys that are
/// still lit, blended by how bright each trail cell is. Squaring the level lets
/// the held key dominate its fading neighbours, so moving from one note to
/// another cross-fades over the trail's time constant instead of snapping.
/// With nothing sounding (or in theremin mode, where there is no note) it falls
/// back to the nearest key the frequency implies, and finally to neutral grey.
fn backdrop_colour(trail: &[f32], current: Option<u8>, frequency: Option<f32>) -> Color {
    let (mut r, mut g, mut b, mut weight) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for (index, &level) in trail.iter().enumerate() {
        if level < TRAIL_EPS {
            continue;
        }
        let w = level * level;
        let color = pitch_color(NOTE_LOW + index as u8);
        r += color.r * w;
        g += color.g * w;
        b += color.b * w;
        weight += w;
    }
    if weight >= TRAIL_EPS {
        return Color {
            r: r / weight,
            g: g / weight,
            b: b / weight,
            a: 1.0,
        };
    }
    match current.or_else(|| frequency.map(note_for_frequency)) {
        Some(note) => pitch_color(note),
        None => MUTED,
    }
}

/// The ASCII backdrop: one monospaced text run per row, tinted with the colour
/// of the note being played. Drawing whole rows rather than individual cells
/// keeps the geometry count at `rows` text runs instead of rows * cols glyph
/// runs.
fn ascii_art(frame: &mut Frame, bounds: Rectangle, grid: &AsciiGrid, ramp: Ramp, colour: Color) {
    let Some((cols, rows, size)) = cell_grid(grid, bounds) else {
        return;
    };

    for row in 0..rows {
        // Nearest-neighbour resample: the grid was built at the frame's aspect,
        // the panel has whatever aspect the window happens to have.
        let source_row = row * grid.rows / rows;
        let mean = grid.row_mean(source_row);
        if mean < ROW_SILENT {
            continue;
        }
        let mut content = String::with_capacity(cols + 1);
        for col in 0..cols {
            content.push(ramp.glyph(grid.level(source_row, col * grid.cols / cols)));
        }
        frame.fill_text(Text {
            content,
            position: Point::new(0.0, row as f32 * size),
            max_width: f32::INFINITY,
            color: tint(colour, 0.12 + 0.68 * mean),
            size: size.into(),
            ..Default::default()
        });
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

/// Gradient + centre glow. Cheap flat primitives, far less than the RGBA video
/// upload it replaced.
fn backdrop(frame: &mut Frame, bounds: Rectangle, stage: Rectangle) {
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

    let centre = Point::new(stage.x + stage.width * 0.5, stage.y + stage.height * 0.5);
    let radius = stage.width.min(stage.height) * 0.8;
    for ring in (0..GLOW_RINGS).rev() {
        let t = (ring + 1) as f32 / GLOW_RINGS as f32;
        frame.fill(
            &Path::circle(centre, radius * t),
            mix(GLOW, PANEL_BOTTOM, t),
        );
    }

    frame.stroke(
        &Path::rectangle(
            Point::new(stage.x, stage.y),
            Size::new(stage.width, stage.height),
        ),
        Stroke::default().with_color(STAGE_EDGE).with_width(1.0),
    );
}

/// The part of the keyboard that never changes between frames: the pitch-class
/// tint of every cell, the separators, the octave labels and - when the cells
/// are big enough - the note names. Cached, so it is built once per window size
/// and instrument rather than once per frame.
fn keyboard_grid(frame: &mut Frame, keyboard: &Keyboard, layout: &GridLayout) {
    let band = keyboard.band;

    // 1. Every cell, faintly tinted with its pitch class. Row 0 is the highest
    // octave, so the keyboard reads upward like a staff.
    for (index, &note) in layout.notes.iter().enumerate() {
        let cell = keyboard.cell(index, layout);
        frame.fill(
            &Path::rectangle(
                Point::new(cell.x, cell.y),
                Size::new(cell.width, cell.height),
            ),
            tint(pitch_color(note), 0.10),
        );
    }

    // 2. Cell separators. Octave rows get the heavier line so the ladder of
    // octaves is visible at a glance.
    for col in 0..=layout.cols {
        let x = band.x + keyboard.cell_w * col as f32;
        frame.stroke(
            &Path::line(Point::new(x, band.y), Point::new(x, band.y + band.height)),
            Stroke::default().with_color(GRID).with_width(1.0),
        );
    }
    for row in 0..=layout.rows {
        let y = band.y + keyboard.cell_h * row as f32;
        frame.stroke(
            &Path::line(Point::new(band.x, y), Point::new(band.x + band.width, y)),
            Stroke::default().with_color(GRID_MAJOR).with_width(1.5),
        );
    }

    // 3. Octave labels down the left of the grid.
    for row in 0..layout.rows {
        frame.fill_text(Text {
            content: note_name(layout.root + layout.octaves[row] * 12),
            position: Point::new(
                keyboard.stage.x + 8.0,
                band.y + keyboard.cell_h * row as f32 + keyboard.cell_h / 2.0 - 6.0,
            ),
            max_width: keyboard.stage.width,
            color: MUTED,
            size: 11.0.into(),
            ..Default::default()
        });
    }

    // 4. Note names inside the cells, when the cells are big enough to hold them.
    if keyboard.cell_w > 34.0 && keyboard.cell_h > 20.0 {
        for (index, &note) in layout.notes.iter().enumerate() {
            let cell = keyboard.cell(index, layout);
            frame.fill_text(Text {
                content: note_name(note),
                position: Point::new(cell.x + 6.0, cell.y + 6.0),
                max_width: keyboard.stage.width,
                color: MUTED,
                size: 11.0.into(),
                ..Default::default()
            });
        }
    }
}

/// The part that changes every frame: the fading trail and the cell being played.
/// `trail[note - NOTE_LOW]` is that note's current brightness (already faded).
fn keyboard_trail(
    frame: &mut Frame,
    keyboard: &Keyboard,
    layout: &GridLayout,
    trail: &[f32],
    current: Option<u8>,
    volume: f32,
) {
    // 1. The trail: cells you have just played, glowing and fading out.
    for (index, &note) in layout.notes.iter().enumerate() {
        let level = trail[note_index(note)];
        if level < TRAIL_EPS {
            continue;
        }
        let cell = keyboard.cell(index, layout);
        let color = pitch_color(note);

        frame.fill(
            &Path::rectangle(
                Point::new(cell.x, cell.y),
                Size::new(cell.width, cell.height),
            ),
            tint(color, 0.42 * level),
        );
        // Bright cap at the top of every lit cell, so the fade reads even when the
        // cell itself is nearly transparent.
        frame.fill(
            &Path::rectangle(Point::new(cell.x, cell.y), Size::new(cell.width, 5.0)),
            tint(color, 0.85 * level),
        );

        // The held cell additionally carries a loudness bar rising from its floor.
        if current == Some(note) {
            let bar = cell.height * volume.clamp(0.0, 1.0);
            frame.fill(
                &Path::rectangle(
                    Point::new(cell.x, cell.y + cell.height - bar),
                    Size::new(cell.width, bar),
                ),
                tint(color, 0.55),
            );
        }
    }

    // 2. Outline the cell being played and label it.
    if let Some(note) = current {
        let index = layout.notes.iter().position(|&n| n == note).unwrap_or(0);
        let cell = keyboard.cell(index, layout);
        let color = pitch_color(note);
        frame.stroke(
            &Path::rectangle(
                Point::new(cell.x, cell.y),
                Size::new(cell.width, cell.height),
            ),
            Stroke::default().with_color(color).with_width(2.0),
        );
        frame.fill_text(Text {
            content: format!("{}  {:.0} Hz", note_name(note), note_frequency(note)),
            position: Point::new(cell.x, cell.y - 16.0),
            max_width: keyboard.stage.width,
            color,
            size: 13.0.into(),
            ..Default::default()
        });
    }
}
/// Geometry that is identical from frame to frame: the gradient backdrop and the
/// keyboard's static grid. iced's canvas `Cache` only re-runs the closure when the
/// layer's size changes, so the two keys below are what tell us when an *input*
/// changed - the camera resolution (which moves the letterboxed stage) or the
/// instrument (which changes the grid).
pub struct StageCache {
    backdrop: Cache,
    grid: Cache,
    // `draw` takes the state immutably, so the invalidation keys live behind a
    // RefCell - the caches themselves already have interior mutability.
    backdrop_key: RefCell<(usize, usize)>,
    grid_key: RefCell<(f32, f32, Vec<u8>)>,
}

impl Default for StageCache {
    fn default() -> Self {
        StageCache {
            backdrop: Cache::new(),
            grid: Cache::new(),
            backdrop_key: RefCell::new((0, 0)),
            grid_key: RefCell::new((0.0, 0.0, Vec::new())),
        }
    }
}

pub struct Skeleton {
    /// The camera frame as RGBA pixels. Only drawn when `show_feed` is true.
    pub image: Option<Handle>,
    /// Header toggle: put the webcam image back behind the skeleton.
    pub show_feed: bool,
    /// Quantised luminance grid for the ASCII backdrop, and the toggle that
    /// decides whether it is drawn.
    pub ascii: AsciiGrid,
    pub show_ascii: bool,
    /// Glyph ramp used to render the grid.
    pub ramp: Ramp,
    /// The continuous theremin frequency, used to pick a colour when the
    /// gesture does not quantise to a note.
    pub frequency: Option<f32>,
    pub landmarks: Vec<Landmark>,
    pub frame_size: (usize, usize),
    pub hand_box: Option<(f32, f32, f32, f32)>,
    pub show_box: bool,
    pub show_labels: bool,
    pub detected: bool,
    /// Milliseconds since the tracker last saw a hand.
    pub lost_ms: u64,
    /// Per-semitone brightness of the keyboard, index = note - NOTE_LOW, as
    /// published by the frame loop.
    pub trail: Vec<f32>,
    /// The degree x octave keyboard the hand plays on.
    pub layout: GridLayout,
    pub current_note: Option<u8>,
    pub volume: f32,
}

impl<Message> Program<Message> for Skeleton {
    type State = StageCache;

    fn draw(
        &self,
        state: &StageCache,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let map = Mapping::new(self.frame_size, bounds);

        // The panel itself, so the letterbox bars are part of the stage.
        let stage = Rectangle {
            x: map.offset_x,
            y: map.offset_y,
            width: bounds.width - 2.0 * map.offset_x,
            height: bounds.height - 2.0 * map.offset_y,
        };

        // Layer 1: the backdrop, rebuilt only when the window or the capture
        // resolution changes the letterbox.
        let backdrop_key = (self.frame_size.0, self.frame_size.1);
        if *state.backdrop_key.borrow() != backdrop_key {
            state.backdrop.clear();
            *state.backdrop_key.borrow_mut() = backdrop_key;
        }
        let mut geometry = vec![state.backdrop.draw(renderer, bounds.size(), |frame| {
            backdrop(frame, bounds, stage)
        })];

        // Layer 2: the camera rendered as block glyphs, tinted with the note that
        // is sounding. The grid itself is produced by the presenter thread; this
        // is only the text runs.
        if self.show_ascii {
            let colour = backdrop_colour(&self.trail, self.current_note, self.frequency);
            let mut layer = Frame::new(renderer, bounds.size());
            ascii_art(&mut layer, bounds, &self.ascii, self.ramp, colour);
            geometry.push(layer.into_geometry());
        }

        let keyboard = Keyboard::new(stage, &self.layout);

        // Layer 3: the instrument - cells, separators, octave and note labels.
        if let Some(keyboard) = keyboard {
            let grid_key = (stage.width, stage.height, self.layout.notes.clone());
            if *state.grid_key.borrow() != grid_key {
                state.grid.clear();
                *state.grid_key.borrow_mut() = grid_key;
            }
            geometry.push(state.grid.draw_with_bounds(renderer, stage, |frame| {
                keyboard_grid(frame, &keyboard, &self.layout)
            }));
        }

        // Layer 4: everything that moves - the trail, the play head, the feed and
        // the hand.
        let mut frame = Frame::new(renderer, bounds.size());
        if let Some(keyboard) = keyboard {
            keyboard_trail(
                &mut frame,
                &keyboard,
                &self.layout,
                &self.trail,
                self.current_note,
                self.volume,
            );
        }

        // Play head: the note comes from the x position of the index tip, so a
        // vertical line through that fingertip ties the gesture to the key.
        if let Some(note) = self.current_note {
            if self.landmarks.len() > 8 {
                let tip = map.point(&self.landmarks[8]);
                frame.stroke(
                    &Path::line(
                        Point::new(tip.x, stage.y),
                        Point::new(tip.x, stage.y + stage.height),
                    ),
                    Stroke::default()
                        .with_color(tint(pitch_color(note), 0.35))
                        .with_width(1.5),
                );
            }
        }

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
                color: MUTED,
                size: 18.0.into(),
                ..Default::default()
            });
            frame.fill_text(Text {
                content: format!("lost for {} ms", self.lost_ms),
                position: Point::new(stage.x + 12.0, stage.y + 6.0),
                max_width: stage.width,
                color: Color::from_rgb8(90, 100, 110),
                size: 11.0.into(),
                ..Default::default()
            });
            geometry.push(frame.into_geometry());
            return geometry;
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
                Stroke::default()
                    .with_color(tint(color, 0.25))
                    .with_width(8.0),
            );
            frame.stroke(&line, Stroke::default().with_color(color).with_width(2.5));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::NOTE_COUNT;
    use crate::gesture::band_position;

    fn grid(cols: usize, rows: usize) -> AsciiGrid {
        AsciiGrid {
            cols,
            rows,
            cells: vec![0u8; cols * rows],
        }
    }

    #[test]
    fn the_band_the_canvas_draws_is_the_band_the_strategy_reads() {
        let stage = Rectangle {
            x: 12.0,
            y: 30.0,
            width: 640.0,
            height: 480.0,
        };
        let layout = GridLayout::new(
            0,
            &crate::core::scale::preset("major pentatonic").unwrap(),
            3,
        );
        let keyboard = Keyboard::new(stage, &layout).unwrap();
        let band = keyboard.band;

        assert!((band.x - (stage.x + NOTE_MARGIN * stage.width)).abs() < 1e-6);
        assert!((band.y - (stage.y + NOTE_MARGIN * stage.height)).abs() < 1e-6);
        assert!((band.width - stage.width * (1.0 - 2.0 * NOTE_MARGIN)).abs() < 1e-6);
        assert!((band.height - stage.height * (1.0 - 2.0 * NOTE_MARGIN)).abs() < 1e-6);
        // An empty layout draws nothing rather than dividing by zero.
        assert!(Keyboard::new(stage, &GridLayout::default()).is_none());
    }

    #[test]
    fn the_cell_that_lights_is_the_cell_the_fingertip_is_over() {
        // A 640x480 frame filling the panel, so frame pixels and stage pixels match.
        let bounds = Rectangle {
            x: 0.0,
            y: 0.0,
            width: 640.0,
            height: 480.0,
        };
        let map = Mapping::new((640, 480), bounds);
        let stage = Rectangle {
            x: map.offset_x,
            y: map.offset_y,
            width: bounds.width - 2.0 * map.offset_x,
            height: bounds.height - 2.0 * map.offset_y,
        };
        let layout = GridLayout::new(
            0,
            &crate::core::scale::preset("major pentatonic").unwrap(),
            3,
        );
        let keyboard = Keyboard::new(stage, &layout).unwrap();

        // Sample inside the playable band; outside it the strategy clamps to the
        // nearest cell, so the tip is no longer inside the cell it plays.
        for x in (64..=576).step_by(32) {
            for y in (48..=432).step_by(24) {
                // What the strategy plays at this fingertip...
                let (nx, ny) = band_position(x as f32, y as f32, (640, 480));
                let index = layout.cell_at(nx, ny);
                // ...must be the cell the stage glows.
                let cell = keyboard.cell(index, &layout);
                let tip = map.point(&Landmark {
                    x: x as f32,
                    y: y as f32,
                    z: 0.0,
                });
                assert!(
                    cell.x - 1e-3 <= tip.x && tip.x <= cell.x + cell.width + 1e-3,
                    "({x},{y}) played cell {index} but the tip drew at {}",
                    tip.x
                );
                assert!(
                    cell.y - 1e-3 <= tip.y && tip.y <= cell.y + cell.height + 1e-3,
                    "({x},{y}) played cell {index} but the tip drew at {}",
                    tip.y
                );
            }
        }
    }

    #[test]
    fn the_cell_grid_tiles_the_panel_without_overflowing_it() {
        let panel = Rectangle {
            x: 0.0,
            y: 0.0,
            width: 1000.0,
            height: 700.0,
        };
        let (cols, rows, size) = cell_grid(&grid(140, 105), panel).unwrap();
        assert_eq!((cols, rows), (140, 58));
        assert!(cols as f32 * size * GLYPH_ASPECT <= panel.width + 1e-3);
        assert!(rows as f32 * size <= panel.height + 1e-3);
    }

    #[test]
    fn a_coarse_grid_gets_bigger_glyphs_and_fewer_rows() {
        let panel = Rectangle {
            x: 0.0,
            y: 0.0,
            width: 1000.0,
            height: 700.0,
        };
        let (cols, rows, size) = cell_grid(&grid(32, 24), panel).unwrap();
        assert_eq!((cols, rows), (32, 13));
        assert!(size > 30.0);
    }

    #[test]
    fn a_panel_too_small_or_an_empty_grid_draws_nothing() {
        assert!(cell_grid(
            &grid(140, 105),
            Rectangle {
                x: 0.0,
                y: 0.0,
                width: 30.0,
                height: 12.0,
            }
        )
        .is_none());
        assert!(cell_grid(
            &AsciiGrid::default(),
            Rectangle {
                x: 0.0,
                y: 0.0,
                width: 900.0,
                height: 700.0,
            }
        )
        .is_none());
    }

    #[test]
    fn the_backdrop_takes_the_colour_of_the_held_key() {
        use crate::core::NOTE_COUNT;
        let mut trail = vec![0.0; NOTE_COUNT];
        trail[note_index(60)] = 1.0; // C4 -> pitch class C
        assert_eq!(backdrop_colour(&trail, Some(60), None), PITCH_COLORS[0]);
    }

    #[test]
    fn a_fading_trail_cross_fades_towards_the_held_key() {
        let mut trail = vec![0.0; NOTE_COUNT];
        trail[note_index(64)] = 1.0; // E, still held
        trail[note_index(60)] = 0.1; // C, nearly gone
        let colour = backdrop_colour(&trail, Some(64), None);
        let (c, e) = (PITCH_COLORS[0], PITCH_COLORS[4]);
        // Squared levels: 1.0 and 0.01, so the blend is 99 % E and 1 % C.
        let weight = 1.0 + 0.01;
        for (channel, expected) in [
            (colour.r, (e.r + c.r * 0.01) / weight),
            (colour.g, (e.g + c.g * 0.01) / weight),
            (colour.b, (e.b + c.b * 0.01) / weight),
        ] {
            assert!((channel - expected).abs() < 1e-3);
        }
        assert!(colour.g < e.g); // the fading C drags the green down a little
    }

    #[test]
    fn a_silent_backdrop_uses_the_theremin_key_or_neutral_grey() {
        let trail = vec![0.0; NOTE_COUNT];
        // 440 Hz is A4, pitch class A.
        assert_eq!(backdrop_colour(&trail, None, Some(440.0)), PITCH_COLORS[9]);
        assert_eq!(backdrop_colour(&trail, None, None), MUTED);
    }
}
