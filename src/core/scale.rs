//! Scale presets and the notes they make playable.
//!
//! The pitch mapping is quantised to a scale: instead of every pixel column
//! being a semitone, the hand lands on one of the notes the scale permits. Any
//! two cells of a scale are consonant with each other, so sweeping the hand
//! across the stage cannot produce a clash - the reason the chromatic mapping
//! (the old behaviour) can sound random.

#[derive(Debug, Clone, PartialEq)]
pub struct Scale {
    pub name: &'static str,
    /// Semitone offsets from the root within one octave.
    pub degrees: Vec<u8>,
}

/// Available presets. `chromatic` is the pre-quantisation behaviour, kept for
/// comparison and for anyone who wants every semitone.
pub const PRESETS: [(&str, &[u8]); 7] = [
    ("chromatic", &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ("major pentatonic", &[0, 2, 4, 7, 9]),
    ("minor pentatonic", &[0, 3, 5, 7, 10]),
    ("major", &[0, 2, 4, 5, 7, 9, 11]),
    ("minor", &[0, 2, 3, 5, 7, 8, 10]),
    ("dorian", &[0, 2, 3, 5, 7, 9, 10]),
    ("whole tone", &[0, 2, 4, 6, 8, 10]),
];

/// Look up a preset by name, tolerating the spellings people actually type
/// (`penta`, `pent`, `ionian`, hyphens, capitals). An exact name wins over a
/// substring, so `minor` is the natural minor and not minor pentatonic.
pub fn preset(name: &str) -> Option<Scale> {
    let needle = name.trim().to_lowercase().replace(['-', ' '], "");
    let matches: Vec<&(&str, &[u8])> = PRESETS
        .iter()
        .filter(|(preset, _)| {
            let key = preset.replace(['-', ' '], "");
            key == needle || key.contains(&needle) || needle.contains(&key)
        })
        .collect();
    let chosen = matches
        .iter()
        .find(|(preset, _)| preset.replace(['-', ' '], "") == needle)
        .or_else(|| matches.first());
    chosen.map(|(preset, degrees)| Scale {
        name: preset,
        degrees: degrees.to_vec(),
    })
}

/// Pitch class (0..11) of a letter name such as `C`, `F#` or `Bb`.
pub fn root_class(name: &str) -> Option<u8> {
    let text = name.trim().to_lowercase();
    let letter = text.chars().next()?;
    let base = match letter {
        'c' => 0,
        'd' => 2,
        'e' => 4,
        'f' => 5,
        'g' => 7,
        'a' => 9,
        'b' => 11,
        _ => return None,
    };
    let accidental: String = text
        .chars()
        .skip(1)
        .filter(|c| !c.is_whitespace())
        .collect();
    let offset: i32 = match accidental.as_str() {
        "" | "natural" => 0,
        "#" | "sharp" => 1,
        "b" | "flat" => -1,
        _ => return None,
    };
    Some(((base + offset).rem_euclid(12)) as u8)
}

/// Is `note` inside `scale` rooted at `root`? Used by the tests and by any
/// consumer that wants to check a note against a scale.
#[allow(dead_code)]
pub fn is_in_scale(note: u8, root: u8, scale: &Scale) -> bool {
    let interval = (note as i32 - root as i32).rem_euclid(12) as u8;
    scale.degrees.contains(&interval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::grid::GridLayout;
    use crate::core::{NOTE_HIGH, NOTE_LOW};

    #[test]
    fn presets_round_trip_by_name() {
        assert_eq!(preset("Pentatonic").unwrap().name, "major pentatonic");
        assert_eq!(preset("major-pentatonic").unwrap().name, "major pentatonic");
        assert_eq!(preset("MINOR").unwrap().degrees, vec![0, 2, 3, 5, 7, 8, 10]);
        assert_eq!(preset("minor pentatonic").unwrap().name, "minor pentatonic");
        assert_eq!(preset("whole tone").unwrap().degrees.len(), 6);
        assert!(preset("lydian").is_none());
    }

    #[test]
    fn root_names_parse() {
        assert_eq!(root_class("C"), Some(0));
        assert_eq!(root_class("F#"), Some(6));
        assert_eq!(root_class("Bb"), Some(10));
        assert_eq!(root_class("c sharp"), Some(1));
        assert_eq!(root_class("H"), None);
    }

    #[test]
    fn a_pentatonic_grid_is_consonant_and_inside_the_range() {
        let scale = preset("major pentatonic").unwrap();
        let grid = GridLayout::new(root_class("C").unwrap(), &scale, 4);
        assert!(grid.notes.iter().all(|&n| is_in_scale(n, 0, &scale)));
        assert!(grid
            .notes
            .iter()
            .all(|&n| (NOTE_LOW..=NOTE_HIGH).contains(&n)));
        // C, D, E, G, A per octave - no F, no B, no semitone neighbours.
        assert!(grid.notes.iter().all(|&n| n % 12 != 5 && n % 12 != 11));
    }

    #[test]
    fn chromatic_keeps_every_note() {
        let scale = preset("chromatic").unwrap();
        let grid = GridLayout::new(0, &scale, 4);
        assert_eq!(grid.len(), 48);
        assert!(grid.notes.iter().all(|&n| is_in_scale(n, 0, &scale)));
    }

    #[test]
    fn transposing_the_root_shifts_the_grid() {
        let scale = preset("minor pentatonic").unwrap();
        let in_c = GridLayout::new(0, &scale, 3).notes;
        let in_a = GridLayout::new(root_class("A").unwrap(), &scale, 3).notes;
        assert_ne!(
            in_c, in_a,
            "a different root must produce a different keyboard"
        );
        assert!(
            in_a.iter().any(|&n| n % 12 == 9),
            "A should appear in A pentatonic"
        );
        assert!(
            !in_c.iter().any(|&n| n % 12 == 9),
            "A is not in C minor pentatonic"
        );
    }
}
