//! The degree x octave keyboard.
//!
//! A scale gives the notes; the layout decides where they sit on the stage.
//! Columns are scale degrees (low to high, left to right), rows are octaves,
//! with the **top row being the highest octave** - the hand reads "up = higher"
//! without thinking about it.
//!
//! The geometry here is shared with `dashboard/skeleton.rs`: both use the same
//! `NOTE_MARGIN` inset on each axis, so the cell the hand is over is exactly the
//! cell that lights up.

use crate::core::scale::Scale;
use crate::core::{NOTE_HIGH, NOTE_LOW};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GridLayout {
    /// Number of octave rows, top row first (highest octave).
    pub rows: usize,
    /// Number of degree columns, left to right.
    pub cols: usize,
    /// Row-major, top row first: `notes[row * cols + col]`.
    pub notes: Vec<u8>,
    /// Octave number of each row, top row first.
    pub octaves: Vec<u8>,
    /// Root pitch class of the grid, so rows can be labelled correctly.
    pub root: u8,
}

impl GridLayout {
    /// Build a grid of `octaves` octave rows rooted at `root`, placed as close to
    /// the centre of NOTE_LOW..=NOTE_HIGH as it will fit.
    pub fn new(root: u8, scale: &Scale, octaves: usize) -> GridLayout {
        let degrees = &scale.degrees;
        let rows = fit_rows(root, degrees, octaves.max(1));
        let base = best_base(root, degrees, rows);

        // Highest octave first, so row 0 is the top of the stage.
        let octaves_used: Vec<u8> = (0..rows).map(|row| base + row as u8).rev().collect();
        let notes: Vec<u8> = octaves_used
            .iter()
            .flat_map(|&octave| {
                degrees
                    .iter()
                    .map(move |&degree| root + octave * 12 + degree)
            })
            .collect();

        GridLayout {
            rows,
            cols: degrees.len(),
            notes,
            octaves: octaves_used,
            root,
        }
    }

    /// Cell index for a normalised position (0..1) inside the inset band.
    pub fn cell_at(&self, x_norm: f32, y_norm: f32) -> usize {
        let col = ((x_norm.clamp(0.0, 1.0) * self.cols as f32) as usize).min(self.cols - 1);
        let row = ((y_norm.clamp(0.0, 1.0) * self.rows as f32) as usize).min(self.rows - 1);
        row * self.cols + col
    }

    pub fn note_at(&self, x_norm: f32, y_norm: f32) -> u8 {
        self.notes[self.cell_at(x_norm, y_norm)]
    }

    pub fn len(&self) -> usize {
        self.notes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }
}

/// Does a `rows`-row grid rooted at `root` fit anywhere in the playable range?
fn fits(root: u8, degrees: &[u8], rows: usize) -> bool {
    let top_degree = *degrees.last().unwrap_or(&0);
    (NOTE_LOW / 12..=NOTE_HIGH / 12).any(|base| {
        let first = root + base * 12;
        let last = root + (base + rows as u8 - 1) * 12 + top_degree;
        first >= NOTE_LOW && last <= NOTE_HIGH
    })
}

/// Highest starting octave whose `rows` rows fit inside the playable range.
fn best_base(root: u8, degrees: &[u8], rows: usize) -> u8 {
    let top_degree = *degrees.last().unwrap_or(&0);
    let mut best: Option<(i32, u8)> = None;
    for base in (NOTE_LOW / 12)..=(NOTE_HIGH / 12) {
        let first = root + base * 12;
        let last = root + (base + rows as u8 - 1) * 12 + top_degree;
        if first < NOTE_LOW || last > NOTE_HIGH {
            continue;
        }
        let centre = ((first + last) as i32 - (NOTE_LOW + NOTE_HIGH) as i32).abs();
        if best.is_none_or(|(previous, _)| centre < previous) {
            best = Some((centre, base));
        }
    }
    // No base was perfectly centred; fall back to the lowest octave in range.
    best.map_or(NOTE_LOW / 12, |(_, base)| base)
}

/// Shrink the requested number of rows until the grid fits in the range.
fn fit_rows(root: u8, degrees: &[u8], requested: usize) -> usize {
    (1..=requested)
        .rev()
        .find(|&rows| fits(root, degrees, rows))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::scale::{preset, root_class};

    #[test]
    fn the_grid_is_centred_and_inside_the_range() {
        let grid = GridLayout::new(
            root_class("C").unwrap(),
            &preset("major pentatonic").unwrap(),
            3,
        );
        assert_eq!(grid.rows, 3);
        assert_eq!(grid.cols, 5);
        assert_eq!(grid.len(), 15);
        // Top row is the highest octave.
        assert_eq!(grid.octaves, vec![6, 5, 4]);
        assert_eq!(grid.notes[0], 72); // C6
        assert_eq!(grid.notes[grid.len() - 1], 57); // A3
        assert!(grid
            .notes
            .iter()
            .all(|&n| (NOTE_LOW..=NOTE_HIGH).contains(&n)));
    }

    #[test]
    fn cells_map_to_the_expected_notes() {
        let grid = GridLayout::new(
            root_class("C").unwrap(),
            &preset("major pentatonic").unwrap(),
            3,
        );
        assert_eq!(grid.note_at(0.0, 0.0), 72); // C6, top left
        assert_eq!(grid.note_at(1.0, 0.0), 81); // A6, top right
        assert_eq!(grid.note_at(0.0, 1.0), 48); // C4, bottom left
        assert_eq!(grid.note_at(1.0, 1.0), 57); // A4, bottom right
        assert_eq!(grid.note_at(0.5, 0.5), 64); // E5, middle cell
    }

    #[test]
    fn moving_right_and_up_always_rises() {
        let grid = GridLayout::new(0, &preset("major").unwrap(), 3);
        assert!(grid.note_at(1.0, 0.0) > grid.note_at(0.0, 1.0));
        assert!(grid.note_at(0.9, 0.5) >= grid.note_at(0.1, 0.5));
        assert!(grid.note_at(0.5, 0.1) >= grid.note_at(0.5, 0.9));
    }

    #[test]
    fn out_of_range_positions_clamp_to_the_nearest_cell() {
        let grid = GridLayout::new(
            root_class("A").unwrap(),
            &preset("minor pentatonic").unwrap(),
            4,
        );
        for x in [-1.0f32, 0.0, 0.3, 0.9, 1.0, 2.0] {
            for y in [-0.5, 0.0, 0.5, 1.0, 1.5] {
                let cell = grid.cell_at(x, y);
                assert!(cell < grid.len(), "cell {cell} out of bounds for {x},{y}");
            }
        }
    }

    #[test]
    fn an_impossible_grid_shrinks_instead_of_panicking() {
        // Six chromatic octaves cannot fit in a 48 semitone range.
        let grid = GridLayout::new(0, &preset("chromatic").unwrap(), 6);
        assert!(grid.rows < 6);
        assert!(grid
            .notes
            .iter()
            .all(|&n| (NOTE_LOW..=NOTE_HIGH).contains(&n)));
        assert!(!grid.is_empty());
    }
}
