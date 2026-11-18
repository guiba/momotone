//! Camera frame -> quantised luminance grid, the source of the dashboard's
//! ASCII art backdrop.
//!
//! The pipeline only publishes *levels* (0..LEVELS-1), never characters: the
//! glyph set is a rendering choice (`--ascii-ramp`, or the dashboard's RENDER
//! card), and the stage resamples the grid to whatever cell grid actually fits
//! the panel. Keeping the grid resolution-independent also means the terminal
//! dump and the window never disagree about what the frame looked like.
//!
//! Rows are derived from the frame's aspect ratio, so the art has the same
//! proportions as the camera view before the UI stretches it into the panel.

use anyhow::Result;

use crate::core::frame::Frame;
use crate::core::Landmark;

/// Number of brightness buckets. Nine is enough for a legible ramp: the glyph
/// families below fill a cell monotonically from empty to solid.
pub const LEVELS: usize = 9;
/// Columns the grid is built at; rows follow from the frame aspect.
pub const DEFAULT_COLS: usize = 140;
/// Frames per second the art is refreshed. It is a backdrop, not a video, so
/// a slower cadence than the feed costs the eyes nothing.
pub const DEFAULT_ASCII_FPS: f32 = 15.0;
/// How much of each cell's new brightness is taken per render. A raw grid
/// flickers badly at 15 fps because each cell averages a 4x8 pixel block of a
/// moving image; the EMA turns that into a stable wash.
const EMA_ALPHA: f32 = 0.6;

/// One glyph per brightness bucket. Every ramp is monotone in *coverage*, so
/// the same grid reads as the same image however it is characterised - only the
/// texture changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ramp {
    /// Left-to-right fraction blocks: guaranteed to exist in SF Mono and Menlo,
    /// and every glyph has the monospace advance, so the grid never shears.
    Blocks,
    /// Bottom-up shades: the classic brightness scale. A lit region grows from
    /// its baseline, which reads more like a solid body than the left-aligned
    /// fraction blocks do.
    Shades,
    /// The classic typewriter ramp.
    Ascii,
    /// Dot-matrix detail, filled column by column. Renders in a terminal; in the
    /// iced canvas these codepoints fall back to a font that is not monospaced.
    Braille,
    /// Sparse halftone: every step adds exactly one braille dot, so a lit region
    /// reads as a scattering of dots instead of a solid block.
    Dots,
}

impl Ramp {
    /// Every ramp, in the order the dashboard lists them.
    pub fn all() -> &'static [Ramp] {
        &[
            Ramp::Blocks,
            Ramp::Shades,
            Ramp::Ascii,
            Ramp::Braille,
            Ramp::Dots,
        ]
    }

    pub fn names() -> Vec<String> {
        Self::all()
            .iter()
            .map(|ramp| ramp.name().to_string())
            .collect()
    }

    pub fn parse(name: &str) -> Option<Ramp> {
        match name.trim().to_lowercase().as_str() {
            "blocks" | "block" => Some(Ramp::Blocks),
            "shades" | "shade" | "half" => Some(Ramp::Shades),
            "ascii" | "text" | "classic" => Some(Ramp::Ascii),
            "braille" => Some(Ramp::Braille),
            "dots" | "dot" | "halftone" => Some(Ramp::Dots),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Ramp::Blocks => "blocks",
            Ramp::Shades => "shades",
            Ramp::Ascii => "ascii",
            Ramp::Braille => "braille",
            Ramp::Dots => "dots",
        }
    }

    pub fn glyphs(self) -> &'static [char; LEVELS] {
        match self {
            Ramp::Blocks => &BLOCKS,
            Ramp::Shades => &SHADES,
            Ramp::Ascii => &ASCII,
            Ramp::Braille => &BRAILLE,
            Ramp::Dots => &DOTS,
        }
    }

    pub fn glyph(self, level: u8) -> char {
        let level = (level as usize).min(LEVELS - 1);
        self.glyphs()[level]
    }

    /// The whole ramp as a string, for the dashboard's preview line.
    pub fn preview(self) -> String {
        self.glyphs().iter().collect()
    }
}

const BLOCKS: [char; LEVELS] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];
const SHADES: [char; LEVELS] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const ASCII: [char; LEVELS] = [' ', '.', ':', '-', '=', '+', '*', '#', '@'];
const BRAILLE: [char; LEVELS] = ['⠀', '⡀', '⣀', '⣄', '⣤', '⣦', '⣶', '⣷', '⣿'];
const DOTS: [char; LEVELS] = ['⠀', '⠄', '⠆', '⠧', '⠯', '⠟', '⠿', '⡿', '⣿'];

#[derive(Debug, Clone, Default)]
pub struct AsciiGrid {
    pub cols: usize,
    pub rows: usize,
    /// Row-major brightness levels, `cells[row * cols + col]`.
    pub cells: Vec<u8>,
}

impl AsciiGrid {
    pub fn is_empty(&self) -> bool {
        self.cols == 0 || self.rows == 0 || self.cells.len() != self.cols * self.rows
    }

    pub fn level(&self, row: usize, col: usize) -> u8 {
        if row >= self.rows || col >= self.cols {
            return 0;
        }
        self.cells[row * self.cols + col]
    }

    /// Mean brightness of a row, in 0..1. Used to decide whether a row is worth
    /// a text run at all, and to scale its colour.
    pub fn row_mean(&self, row: usize) -> f32 {
        if self.is_empty() || row >= self.rows {
            return 0.0;
        }
        let start = row * self.cols;
        let total: u32 = self.cells[start..start + self.cols]
            .iter()
            .map(|&level| level as u32)
            .sum();
        total as f32 / (self.cols * (LEVELS - 1)) as f32
    }

    pub fn text(&self, ramp: Ramp) -> String {
        let mut out = String::with_capacity(self.cols * self.rows + self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                out.push(ramp.glyph(self.level(row, col)));
            }
            out.push('\n');
        }
        out
    }
}

/// Brightness buckets are linear in luminance.
fn level_of(luminance: f32) -> u8 {
    let bucket = (luminance * LEVELS as f32 / 256.0) as usize;
    bucket.min(LEVELS - 1) as u8
}

/// Rows that keep the grid's aspect equal to the frame's.
pub fn rows_for(cols: usize, frame_size: (usize, usize)) -> usize {
    let fw = frame_size.0.max(1) as f64;
    let fh = frame_size.1.max(1) as f64;
    ((cols as f64) * fh / fw).round().max(1.0) as usize
}

/// Downsample `frame` to `cols x rows_for(cols)` and quantise each cell.
///
/// Each cell is the mean luminance of the block of pixels it covers - the same
/// area-average the OpenCV `INTER_AREA` resize produced - accumulated in `u32`
/// so a 640x480 frame costs one pass over the pixels and no allocation.
fn frame_levels(frame: &Frame, cols: usize) -> Result<Vec<u8>> {
    let fw = frame.width();
    let fh = frame.height();
    let rows = rows_for(cols, (fw, fh));
    let pixels = frame.pixels();
    let mut cells = vec![0u8; cols * rows];

    for cy in 0..rows {
        let y0 = cy * fh / rows;
        let y1 = ((cy + 1) * fh / rows).max(y0 + 1);
        for cx in 0..cols {
            let x0 = cx * fw / cols;
            let x1 = ((cx + 1) * fw / cols).max(x0 + 1);
            let (mut red, mut green, mut blue) = (0u32, 0u32, 0u32);
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * fw + x) * 3;
                    red += pixels[i] as u32;
                    green += pixels[i + 1] as u32;
                    blue += pixels[i + 2] as u32;
                }
            }
            let count = ((y1 - y0) * (x1 - x0)) as u32;
            // The weights are the same Rec. 601 luma the frame helper uses.
            let luminance =
                (0.299 * red as f32 + 0.587 * green as f32 + 0.114 * blue as f32) / count as f32;
            cells[cy * cols + cx] = level_of(luminance);
        }
    }
    Ok(cells)
}

/// Bones drawn into the mock grid: the palm loop plus the five finger chains in
/// MediaPipe landmark order.
const BONES: [(usize, usize); 20] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];

/// The ASCII layer's per-frame state: the previous cell levels, so the art
/// settles instead of flickering.
pub struct AsciiLayer {
    smoothed: Vec<f32>,
}

impl AsciiLayer {
    pub fn new() -> Self {
        AsciiLayer {
            smoothed: Vec::new(),
        }
    }

    pub fn render(&mut self, frame: &Frame, cols: usize) -> Result<AsciiGrid> {
        let cols = cols.clamp(8, 400);
        let mut cells = frame_levels(frame, cols)?;
        if self.smoothed.len() != cells.len() {
            self.smoothed = cells.iter().map(|&level| level as f32).collect();
        } else {
            for (index, cell) in cells.iter_mut().enumerate() {
                let next = self.smoothed[index] * (1.0 - EMA_ALPHA) + *cell as f32 * EMA_ALPHA;
                self.smoothed[index] = next;
                *cell = next.round().clamp(0.0, (LEVELS - 1) as f32) as u8;
            }
        }
        Ok(AsciiGrid {
            cols,
            rows: rows_for(cols, frame.size()),
            cells,
        })
    }

    /// Mock mode has no camera, so the art comes from the synthetic skeleton:
    /// the bones are rasterised as lines into the grid.
    pub fn render_landmarks(
        &mut self,
        landmarks: &[Landmark],
        frame_size: (usize, usize),
        cols: usize,
    ) -> AsciiGrid {
        let cols = cols.clamp(8, 400);
        let rows = rows_for(cols, frame_size);
        let fw = frame_size.0.max(1) as f32;
        let fh = frame_size.1.max(1) as f32;
        let mut cells = vec![0u8; cols * rows];

        let stamp = |cells: &mut Vec<u8>, x: f32, y: f32| {
            let col = (x * cols as f32 / fw).clamp(0.0, (cols - 1) as f32) as usize;
            let row = (y * rows as f32 / fh).clamp(0.0, (rows - 1) as f32) as usize;
            cells[row * cols + col] = (LEVELS - 1) as u8;
            // One cell of shoulder on each side so a diagonal bone reads as a
            // stroke rather than a chain of single dots.
            for neighbour in [col.saturating_sub(1), (col + 1).min(cols - 1)] {
                let index = row * cols + neighbour;
                cells[index] = cells[index].max((LEVELS - 3) as u8);
            }
        };

        for &(a, b) in BONES.iter().filter(|&(_, b)| landmarks.len() > *b) {
            let from = &landmarks[a];
            let to = &landmarks[b];
            let dx = to.x - from.x;
            let dy = to.y - from.y;
            // Enough samples to cross every cell the segment passes through.
            let steps =
                ((dx * cols as f32 / fw).abs() + (dy * rows as f32 / fh).abs()).ceil() as usize;
            for step in 0..=steps {
                let t = step as f32 / steps.max(1) as f32;
                stamp(&mut cells, from.x + dx * t, from.y + dy * t);
            }
        }
        AsciiGrid { cols, rows, cells }
    }
}

impl Default for AsciiLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn solid(rows: usize, cols: usize, r: u8, g: u8, b: u8) -> Frame {
        Frame::new(cols, rows, [r, g, b].repeat(rows * cols))
    }

    #[test]
    fn rows_follow_the_frame_aspect() {
        assert_eq!(rows_for(140, (640, 480)), 105);
        assert_eq!(rows_for(100, (1920, 1080)), 56);
        assert_eq!(rows_for(100, (480, 640)), 133);
    }

    #[test]
    fn a_black_frame_is_the_empty_end_of_the_ramp() {
        let grid = AsciiLayer::new()
            .render(&solid(48, 64, 0, 0, 0), 16)
            .unwrap();
        assert_eq!((grid.cols, grid.rows), (16, 12));
        assert!(grid.cells.iter().all(|&level| level == 0));
    }

    #[test]
    fn a_white_frame_is_the_solid_end_of_the_ramp() {
        let grid = AsciiLayer::new()
            .render(&solid(48, 64, 255, 255, 255), 16)
            .unwrap();
        assert!(grid.cells.iter().all(|&level| level == (LEVELS - 1) as u8));
    }

    #[test]
    fn a_mid_grey_lands_in_the_middle_of_the_ramp() {
        let grid = AsciiLayer::new()
            .render(&solid(48, 64, 128, 128, 128), 16)
            .unwrap();
        let level = grid.level(0, 0);
        assert!(level > 0 && level < (LEVELS - 1) as u8);
        assert_eq!(Ramp::Blocks.glyph(level), '▌');
    }

    #[test]
    fn luminance_is_not_blue_weighted() {
        // Pure green (luma 149) sits in a higher bucket than pure blue (29),
        // which is exactly why a luminance ramp beats a raw channel.
        let green = AsciiLayer::new()
            .render(&solid(24, 24, 0, 255, 0), 12)
            .unwrap();
        let blue = AsciiLayer::new()
            .render(&solid(24, 24, 255, 0, 0), 12)
            .unwrap();
        assert!(green.level(0, 0) > blue.level(0, 0));
    }

    #[test]
    fn a_gradient_is_monotonic_across_the_grid() {
        let mut pixels = vec![0u8; 48 * 64 * 3];
        for y in 0..48 {
            for x in 0..64 {
                let value = (x * 255 / 63) as u8;
                let i = (y * 64 + x) * 3;
                pixels[i] = value;
                pixels[i + 1] = value;
                pixels[i + 2] = value;
            }
        }
        let grid = AsciiLayer::new()
            .render(&Frame::new(64, 48, pixels), 16)
            .unwrap();
        for row in 0..grid.rows {
            for col in 1..grid.cols {
                assert!(grid.level(row, col) >= grid.level(row, col - 1));
            }
        }
        assert!(grid.level(0, grid.cols - 1) > grid.level(0, 0));
    }

    #[test]
    fn smoothing_never_invents_brightness() {
        let mut layer = AsciiLayer::new();
        let first = layer.render(&solid(24, 24, 128, 128, 128), 12).unwrap();
        let second = layer.render(&solid(24, 24, 255, 255, 255), 12).unwrap();
        // The EMA pulls a mid-grey cell toward white, never past it, and the
        // ramp stays inside its bounds.
        assert!(second.level(0, 0) >= first.level(0, 0));
        assert!(second.level(0, 0) <= (LEVELS - 1) as u8);
    }

    #[test]
    fn every_ramp_covers_every_level() {
        for &ramp in Ramp::all() {
            for level in 0..LEVELS as u8 {
                let glyph = ramp.glyph(level);
                assert_ne!(glyph, '\0');
                assert!(ramp.glyph(200) == ramp.glyph((LEVELS - 1) as u8));
            }
        }
    }

    #[test]
    fn the_text_dump_has_one_line_per_row() {
        let grid = AsciiLayer::new()
            .render(&solid(48, 64, 255, 255, 255), 16)
            .unwrap();
        let text = grid.text(Ramp::Blocks);
        assert_eq!(text.lines().count(), grid.rows);
        assert_eq!(text.lines().next().unwrap().chars().count(), grid.cols);
    }

    #[test]
    fn ramp_names_round_trip() {
        assert_eq!(Ramp::parse("Braille"), Some(Ramp::Braille));
        assert_eq!(Ramp::parse("ascii"), Some(Ramp::Ascii));
        assert_eq!(Ramp::parse("blocks"), Some(Ramp::Blocks));
        assert_eq!(Ramp::parse("shades"), Some(Ramp::Shades));
        assert_eq!(Ramp::parse("halftone"), Some(Ramp::Dots));
        assert_eq!(Ramp::parse("kaomoji"), None);
        assert_eq!(Ramp::Blocks.name(), "blocks");
        for &ramp in Ramp::all() {
            assert_eq!(Ramp::parse(ramp.name()), Some(ramp));
            assert_eq!(ramp.preview().chars().count(), LEVELS);
        }
    }

    #[test]
    fn every_ramp_is_monotone_in_coverage() {
        // Each step of a ramp must draw at least as much ink as the one before it,
        // otherwise a brighter pixel would render a darker cell.
        for &ramp in Ramp::all() {
            for level in 1..LEVELS {
                let previous = ramp.glyph((level - 1) as u8);
                let current = ramp.glyph(level as u8);
                assert_ne!(
                    previous,
                    current,
                    "{} is flat at level {level}",
                    ramp.name()
                );
            }
        }
    }

    #[test]
    #[ignore]
    /// What downsampling a 640x480 frame to the dashboard's 140-column grid costs.
    /// Run with
    /// `cargo test --release -- --include-ignored bench_ascii_downsample --nocapture`.
    fn bench_ascii_downsample() {
        const N: u64 = 200;
        let frame = Frame::new(
            640,
            480,
            (0..640 * 480 * 3).map(|i| (i % 256) as u8).collect(),
        );
        let start = Instant::now();
        for _ in 0..N {
            let levels = frame_levels(&frame, 140).expect("a 640x480 frame has rows");
            std::hint::black_box(&levels);
        }
        println!(
            "frame_levels 640x480 -> 140x{}: {:.3} ms/frame",
            rows_for(140, (640, 480)),
            start.elapsed().as_secs_f64() * 1000.0 / N as f64
        );
    }

    #[test]
    fn the_mock_grid_draws_bones_not_just_joint_dots() {
        let landmarks: Vec<Landmark> = (0..5)
            .map(|i| Landmark {
                x: 320.0,
                y: 100.0 + i as f32 * 50.0,
                z: 0.0,
            })
            .collect();
        let grid = AsciiLayer::new().render_landmarks(&landmarks, (640, 480), 20);
        assert_eq!((grid.cols, grid.rows), (20, 15));
        // Every row between the first and last joint is lit: a long bone is
        // interpolated, so it does not leave a gap between its endpoints.
        for row in 3..=9 {
            assert!(grid.level(row, 10) > 0, "bone row {row} should be drawn");
        }
        assert_eq!(grid.level(0, 0), 0, "the background stays blank");
    }
}
