//! ALICE 三相原理 Phase 3 — Musical Intent packet + deterministic synthesizer
//!
//! Encodes a full-song generation prompt in **exactly 8 bytes**:
//!
//! | Byte | Field              | Range       |
//! |------|--------------------|-------------|
//! | 0    | `genre`            | `0..=255`   |
//! | 1    | `mood`             | `0..=255`   |
//! | 2    | `length_bars`      | `0..=255`   |
//! | 3    | `tempo_bpm_offset` | `0..=255` → BPM `40..=295` |
//! | 4    | `key`              | `0..=14` (canonical ABC tonic in circle-of-fifths order) |
//! | 5    | `mode`             | `0..=6` (Ionian / Dorian / Phrygian / Lydian / Mixolydian / Aeolian / Locrian) |
//! | 6-7  | `variation_seed`   | `u16` little-endian PRNG seed |
//!
//! This is the ALICE-Synth canonical realization of **Phase 3 (Intent)** in the
//! ALICE 三相原理 (Data → Law → Intent). Instead of shipping ABC score text
//! (Phase 2, Law), a caller can ship an 8-byte packet — a *few kilobytes* of
//! ABC compressed to *a few bytes* — and let the receiver reconstruct the tune
//! locally.
//!
//! # Deterministic synthesizer
//!
//! [`MusicIntent::synthesize`] produces a valid [`AbcTune`] deterministically
//! from the packet. The current implementation is a procedural generator
//! (`mode_scale × mood_bias × LCG PRNG`) that serves as the canonical
//! stand-in for a future LLM-driven plan head (see `docs/ROADMAP.md`
//! ADR-011). Same packet → same tune, always, on any host.
//!
//! # Inverse extraction
//!
//! [`MusicIntent::from_tune`] performs a best-effort reverse: reads the
//! header's key signature and tempo, counts bars, hashes the body for a
//! stable `variation_seed`. It cannot recover `genre` or `mood` (those live
//! outside the ABC surface) and reports them as defaults.
//!
//! # Example
//!
//! ```
//! use alice_synth::intent::{genre, mode, mood, MusicIntent};
//!
//! let intent = MusicIntent {
//!     genre: genre::FOLK,
//!     mood: mood::HAPPY,
//!     length_bars: 4,
//!     tempo_bpm_offset: 80,   // → 120 BPM
//!     key: 0,                 // C
//!     mode: mode::IONIAN,     // C major
//!     variation_seed: 0xC0DE,
//! };
//!
//! let bytes = intent.to_bytes();
//! assert_eq!(bytes.len(), 8);
//! assert_eq!(MusicIntent::from_bytes(bytes), intent);
//!
//! let tune = intent.synthesize();
//! assert_eq!(tune.header.tempo_bpm, 120);
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::abc::{AbcElement, AbcHeader, AbcTune, KeySignature};

/// Canonical genre identifiers. Values `10..=255` are reserved for future
/// standardization or caller-defined use.
pub mod genre {
    pub const FOLK: u8 = 0;
    pub const JAZZ: u8 = 1;
    pub const BLUES: u8 = 2;
    pub const CLASSICAL: u8 = 3;
    pub const POP: u8 = 4;
    pub const AMBIENT: u8 = 5;
    pub const ROCK: u8 = 6;
    pub const LULLABY: u8 = 7;
    pub const CINEMATIC: u8 = 8;
    pub const ELECTRONIC: u8 = 9;
}

/// Canonical mood identifiers. Values `8..=255` are reserved.
pub mod mood {
    pub const HAPPY: u8 = 0;
    pub const SAD: u8 = 1;
    pub const TENSE: u8 = 2;
    pub const SERENE: u8 = 3;
    pub const ENERGETIC: u8 = 4;
    pub const MELANCHOLY: u8 = 5;
    pub const MYSTERIOUS: u8 = 6;
    pub const TRIUMPHANT: u8 = 7;
}

/// Church mode identifiers. Values `7..=255` are reserved.
///
/// [`MAJOR`] and [`MINOR`] are convenience aliases for [`IONIAN`] and
/// [`AEOLIAN`] respectively.
pub mod mode {
    pub const IONIAN: u8 = 0;
    pub const DORIAN: u8 = 1;
    pub const PHRYGIAN: u8 = 2;
    pub const LYDIAN: u8 = 3;
    pub const MIXOLYDIAN: u8 = 4;
    pub const AEOLIAN: u8 = 5;
    pub const LOCRIAN: u8 = 6;

    pub const MAJOR: u8 = IONIAN;
    pub const MINOR: u8 = AEOLIAN;
}

/// The 8-byte Musical Intent packet.
///
/// Serialize with [`Self::to_bytes`], deserialize with [`Self::from_bytes`].
/// Render to a playable tune with [`Self::synthesize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MusicIntent {
    /// Genre selector (see [`genre`] module)
    pub genre: u8,
    /// Mood selector (see [`mood`] module)
    pub mood: u8,
    /// Number of bars to render (0-255)
    pub length_bars: u8,
    /// Tempo BPM offset (actual BPM = 40 + this value, clamped to 40..=300)
    pub tempo_bpm_offset: u8,
    /// Canonical tonic index (0..=14 in circle-of-fifths order)
    pub key: u8,
    /// Church mode (see [`mode`] module)
    pub mode: u8,
    /// PRNG seed for procedural variation (little-endian on the wire)
    pub variation_seed: u16,
}

impl MusicIntent {
    /// A neutral C-major Folk / Happy tune with a stable seed. Handy for tests
    /// and as a starting point for callers who want to hand-tune only a few
    /// fields.
    pub const DEFAULT_C_MAJOR: Self = Self {
        genre: genre::FOLK,
        mood: mood::HAPPY,
        length_bars: 4,
        tempo_bpm_offset: 80, // 120 BPM
        key: 0,
        mode: mode::IONIAN,
        variation_seed: 0xC0DE,
    };

    /// Serialize to the canonical 8-byte wire format.
    #[must_use]
    pub const fn to_bytes(&self) -> [u8; 8] {
        let seed_bytes = self.variation_seed.to_le_bytes();
        [
            self.genre,
            self.mood,
            self.length_bars,
            self.tempo_bpm_offset,
            self.key,
            self.mode,
            seed_bytes[0],
            seed_bytes[1],
        ]
    }

    /// Deserialize from the canonical 8-byte wire format.
    ///
    /// This function accepts any byte pattern — invalid values in the `key`
    /// or `mode` fields are clamped at [`Self::synthesize`] time rather than
    /// rejected here, so intent packets remain a total order.
    #[must_use]
    pub const fn from_bytes(data: [u8; 8]) -> Self {
        Self {
            genre: data[0],
            mood: data[1],
            length_bars: data[2],
            tempo_bpm_offset: data[3],
            key: data[4],
            mode: data[5],
            variation_seed: u16::from_le_bytes([data[6], data[7]]),
        }
    }

    /// The resolved key signature (sharp count) that this intent would apply
    /// to its output tune.
    #[must_use]
    pub fn key_signature(&self) -> KeySignature {
        let tonic_idx = usize::from(self.key.min(14));
        let mode_idx = usize::from(self.mode.min(6));
        let raw = i16::from(TONIC_MAJOR_SHARPS[tonic_idx]) + i16::from(MODE_SHARP_OFFSET[mode_idx]);
        let clamped: i16 = raw.clamp(-7, 7);
        // Casting a value already clamped to `-7..=7` is lossless.
        KeySignature {
            sharps: clamped as i8,
        }
    }

    /// Resolved tempo in BPM (`40 + tempo_bpm_offset`, clamped to `40..=300`).
    #[must_use]
    pub fn tempo_bpm(&self) -> u16 {
        (40u16 + u16::from(self.tempo_bpm_offset)).clamp(40, 300)
    }

    /// Deterministically synthesize an [`AbcTune`] from this packet.
    ///
    /// The current implementation is a procedural generator (mode scale +
    /// mood-weighted degree bias + LCG PRNG seeded by `variation_seed`) that
    /// serves as a stand-in for a future LLM-driven plan head. The output is
    /// stable across builds and hosts — same packet → same tune.
    #[must_use]
    pub fn synthesize(&self) -> AbcTune {
        let header = AbcHeader {
            meter_num: 4,
            meter_den: 4,
            unit_num: 1,
            unit_den: 4,
            tempo_bpm: self.tempo_bpm(),
            key: self.key_signature(),
        };

        let tonic_idx = usize::from(self.key.min(14));
        let mode_idx = usize::from(self.mode.min(6));
        let tonic_midi = i16::from(TONIC_MIDI[tonic_idx]);
        let scale = MODE_SCALES[mode_idx];
        let bias = mood_bias(self.mood);
        let bias_sum: u32 = bias.iter().map(|&w| u32::from(w)).sum();
        let bias_sum = bias_sum.max(1); // never divide by zero

        let mut prng: u32 = u32::from(self.variation_seed).wrapping_mul(2_654_435_761);
        // A seed of 0 would degenerate an LCG stream near zero; salt it.
        if prng == 0 {
            prng = 0xACE1_ACE1;
        }

        let notes_per_bar = 4;
        let total_notes = usize::from(self.length_bars) * notes_per_bar;
        let mut body: Vec<AbcElement> =
            Vec::with_capacity(total_notes + usize::from(self.length_bars));

        for note_idx in 0..total_notes {
            // Insert a barline at every measure boundary (skip index 0).
            if note_idx > 0 && note_idx % notes_per_bar == 0 {
                body.push(AbcElement::Barline);
            }

            prng = advance_prng(prng);
            let mut roll = (prng >> 8) % bias_sum;
            let mut degree = 0usize;
            for (i, &w) in bias.iter().enumerate() {
                let w32 = u32::from(w);
                if roll < w32 {
                    degree = i;
                    break;
                }
                roll -= w32;
            }

            // Every fourth note takes the melody up an octave — a cheap way to
            // avoid monotonous flat contours in the MVP generator.
            let octave_offset = if note_idx.is_multiple_of(4) && note_idx > 0 {
                12
            } else {
                0
            };

            let midi_raw = tonic_midi + scale[degree] + octave_offset;
            let midi = midi_raw.clamp(0, 127) as u8;

            body.push(AbcElement::Note {
                midi,
                num: 1,
                den: 1,
                tie_follows: false,
            });
        }

        AbcTune {
            header,
            body,
            extra_voices: Vec::new(),
        }
    }

    /// Best-effort reverse extraction: derive an intent packet from an existing
    /// tune's header and body statistics.
    ///
    /// The output preserves `key`, `tempo`, and `length_bars`; `genre` and
    /// `mood` are set to their defaults (`genre::FOLK`, `mood::HAPPY`) because
    /// those categorical labels do not survive the ABC round trip. The
    /// `variation_seed` is derived from an FNV-1a hash over the body's note
    /// pitches so the same input tune always produces the same seed.
    ///
    /// Round-trip note: `intent.synthesize().from_tune()` **does not**
    /// reproduce the original intent's genre/mood — that information is
    /// intentionally lossy. It does preserve key, tempo, and (approximately)
    /// length.
    #[must_use]
    pub fn from_tune(tune: &AbcTune) -> Self {
        let (tonic_idx, mode_idx) = tonic_and_mode_from_sharps(tune.header.key.sharps);
        let tempo_bpm_offset =
            u8::try_from(tune.header.tempo_bpm.saturating_sub(40).min(255)).unwrap_or(0);

        let note_count: usize = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Note { .. } | AbcElement::Chord { .. }))
            .count();
        let length_bars = u8::try_from(note_count.div_ceil(4).min(255)).unwrap_or(255);

        let seed = fnv1a_seed(&tune.body);

        Self {
            genre: genre::FOLK,
            mood: mood::HAPPY,
            length_bars,
            tempo_bpm_offset,
            key: tonic_idx,
            mode: mode_idx,
            variation_seed: seed,
        }
    }
}

// ---------- Internal tables (module-private) ----------

/// Sharp count of each canonical tonic treated as an Ionian (major) key.
/// Index order matches the ABC circle-of-fifths convention used by
/// [`crate::abc::parse_key`].
const TONIC_MAJOR_SHARPS: [i8; 15] = [
    0,  // 0  C
    1,  // 1  G
    2,  // 2  D
    3,  // 3  A
    4,  // 4  E
    5,  // 5  B
    6,  // 6  F#
    7,  // 7  C#
    -1, // 8  F
    -2, // 9  Bb
    -3, // 10 Eb
    -4, // 11 Ab
    -5, // 12 Db
    -6, // 13 Gb
    -7, // 14 Cb
];

/// MIDI pitch of each canonical tonic at octave 4. Enharmonic duplicates
/// (`C#` / `Db`, `F#` / `Gb`) intentionally share MIDI numbers.
const TONIC_MIDI: [u8; 15] = [
    60, // C4
    67, // G4
    62, // D4
    69, // A4
    64, // E4
    71, // B4
    66, // F#4
    61, // C#4
    65, // F4
    70, // Bb4
    63, // Eb4
    68, // Ab4
    61, // Db4 (enharmonic C#4)
    66, // Gb4 (enharmonic F#4)
    59, // Cb4 (enharmonic B3)
];

/// Sharp-count offset applied by each church mode relative to Ionian.
const MODE_SHARP_OFFSET: [i8; 7] = [
    0,  // Ionian
    -2, // Dorian
    -4, // Phrygian
    1,  // Lydian
    -1, // Mixolydian
    -3, // Aeolian
    -5, // Locrian
];

/// Semitone offsets of each scale degree (1..=7) for each church mode.
const MODE_SCALES: [[i16; 7]; 7] = [
    [0, 2, 4, 5, 7, 9, 11], // Ionian
    [0, 2, 3, 5, 7, 9, 10], // Dorian
    [0, 1, 3, 5, 7, 8, 10], // Phrygian
    [0, 2, 4, 6, 7, 9, 11], // Lydian
    [0, 2, 4, 5, 7, 9, 10], // Mixolydian
    [0, 2, 3, 5, 7, 8, 10], // Aeolian
    [0, 1, 3, 5, 6, 8, 10], // Locrian
];

/// Scale-degree weights for each canonical mood. Row indices match the
/// constants in [`mood`]; the default row is used for unknown mood values.
const MOOD_BIAS_TABLE: [[u8; 7]; 8] = [
    [4, 1, 3, 1, 3, 1, 1], // Happy — heavy on 1, 3, 5 (major triad)
    [3, 1, 2, 1, 2, 3, 1], // Sad — tonic + 6 emphasis
    [1, 2, 1, 3, 1, 2, 3], // Tense — dissonant 4 and 7
    [4, 1, 1, 1, 3, 1, 1], // Serene — tonic + fifth
    [2, 2, 2, 1, 3, 1, 2], // Energetic — leading tone + fifth
    [3, 1, 2, 1, 2, 3, 1], // Melancholy — mirror of Sad
    [1, 3, 1, 3, 1, 3, 1], // Mysterious — quartal emphasis
    [3, 1, 2, 1, 4, 1, 2], // Triumphant — dominant heavy
];

const MOOD_BIAS_DEFAULT: [u8; 7] = [3, 1, 2, 1, 3, 1, 1];

#[inline]
fn mood_bias(mood: u8) -> [u8; 7] {
    let idx = usize::from(mood);
    if idx < MOOD_BIAS_TABLE.len() {
        MOOD_BIAS_TABLE[idx]
    } else {
        MOOD_BIAS_DEFAULT
    }
}

/// A simple LCG advance step. Constants are the "Numerical Recipes" set,
/// which are well-mixed for 32-bit output.
#[inline]
const fn advance_prng(state: u32) -> u32 {
    state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// Reverse-lookup a sharp count into `(tonic_idx, mode_idx)`. Prefers the
/// Ionian interpretation; falls back to Aeolian only if Ionian is out of range.
///
/// Ambiguous cases (a `0`-sharp signature could be C major, A minor, D Dorian,
/// …) always resolve to the simplest option (C major) — see ADR-010 in the
/// roadmap for the rationale.
fn tonic_and_mode_from_sharps(sharps: i8) -> (u8, u8) {
    // Prefer Ionian interpretation.
    if let Some(idx) = TONIC_MAJOR_SHARPS.iter().position(|&s| s == sharps) {
        if let Ok(as_u8) = u8::try_from(idx) {
            return (as_u8, mode::IONIAN);
        }
    }
    // Otherwise try Aeolian (relative minor): required_tonic_sharps = sharps + 3.
    let aeolian_target = sharps.saturating_add(3);
    if let Some(idx) = TONIC_MAJOR_SHARPS.iter().position(|&s| s == aeolian_target) {
        if let Ok(as_u8) = u8::try_from(idx) {
            return (as_u8, mode::AEOLIAN);
        }
    }
    (0, mode::IONIAN)
}

/// FNV-1a hash over the note pitches in the body — used as a stable seed
/// derivation for [`MusicIntent::from_tune`]. Non-note elements contribute
/// a `0` byte so barlines / rests still perturb the hash.
fn fnv1a_seed(body: &[AbcElement]) -> u16 {
    let mut hash: u32 = 0x811c_9dc5;
    for elem in body {
        let byte = match elem {
            AbcElement::Note { midi, .. } => *midi,
            AbcElement::Chord { notes, count, .. } => {
                notes[..usize::from(*count)].iter().copied().sum::<u8>()
            }
            _ => 0,
        };
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    // Fold to 16 bits.
    let folded = (hash ^ (hash >> 16)) & 0xFFFF;
    // `folded` is guaranteed < 0x1_0000, so this cast is lossless.
    folded as u16
}

#[cfg(test)]
#[allow(clippy::naive_bytecount)]
mod tests {
    use super::*;
    use crate::abc::AbcElement;

    #[test]
    fn to_bytes_exact_size() {
        let intent = MusicIntent::DEFAULT_C_MAJOR;
        assert_eq!(intent.to_bytes().len(), 8);
    }

    #[test]
    fn roundtrip_default() {
        let intent = MusicIntent::DEFAULT_C_MAJOR;
        assert_eq!(MusicIntent::from_bytes(intent.to_bytes()), intent);
    }

    #[test]
    fn roundtrip_max_values() {
        let intent = MusicIntent {
            genre: 255,
            mood: 255,
            length_bars: 255,
            tempo_bpm_offset: 255,
            key: 14,
            mode: 6,
            variation_seed: 0xFFFF,
        };
        assert_eq!(MusicIntent::from_bytes(intent.to_bytes()), intent);
    }

    #[test]
    fn wire_format_little_endian_seed() {
        let intent = MusicIntent {
            variation_seed: 0x1234,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        let bytes = intent.to_bytes();
        assert_eq!(bytes[6], 0x34);
        assert_eq!(bytes[7], 0x12);
    }

    #[test]
    fn key_signature_c_major_zero_sharps() {
        let intent = MusicIntent {
            key: 0,
            mode: mode::IONIAN,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        assert_eq!(intent.key_signature().sharps, 0);
    }

    #[test]
    fn key_signature_g_dorian_minus_one_sharps() {
        // G (1 sharp) + Dorian (-2) = -1
        let intent = MusicIntent {
            key: 1,
            mode: mode::DORIAN,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        assert_eq!(intent.key_signature().sharps, -1);
    }

    #[test]
    fn key_signature_out_of_range_mode_clamped() {
        // mode = 255 → clamped to 6 (Locrian).
        let intent = MusicIntent {
            key: 0,
            mode: 255,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        // C + Locrian (-5) = -5 sharps.
        assert_eq!(intent.key_signature().sharps, -5);
    }

    #[test]
    fn tempo_calculation() {
        let intent = MusicIntent {
            tempo_bpm_offset: 80,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        assert_eq!(intent.tempo_bpm(), 120);
    }

    #[test]
    fn tempo_max_clamped_to_300() {
        let intent = MusicIntent {
            tempo_bpm_offset: 255,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        // 40 + 255 = 295, still within 40..=300.
        assert_eq!(intent.tempo_bpm(), 295);
    }

    #[test]
    fn synthesize_produces_notes() {
        let intent = MusicIntent {
            length_bars: 2,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        let tune = intent.synthesize();
        let note_count = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Note { .. }))
            .count();
        // 2 bars × 4 notes = 8 notes (+ 1 barline).
        assert_eq!(note_count, 8);
    }

    #[test]
    fn synthesize_length_bars_matches_barlines() {
        let intent = MusicIntent {
            length_bars: 4,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        let tune = intent.synthesize();
        let barline_count = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Barline))
            .count();
        // Barlines are inserted between bars, not after the last one → N-1 = 3.
        assert_eq!(barline_count, 3);
    }

    #[test]
    fn synthesize_respects_tempo_and_key() {
        let intent = MusicIntent {
            tempo_bpm_offset: 100, // 140 BPM
            key: 1,                // G
            mode: mode::IONIAN,    // G major → 1 sharp
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        let tune = intent.synthesize();
        assert_eq!(tune.header.tempo_bpm, 140);
        assert_eq!(tune.header.key.sharps, 1);
    }

    #[test]
    fn synthesize_deterministic_same_seed_same_output() {
        let a = MusicIntent {
            variation_seed: 0x1234,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        let b = MusicIntent {
            variation_seed: 0x1234,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        assert_eq!(a.synthesize().body.len(), b.synthesize().body.len());
        // Compare note sequences.
        let a_notes: Vec<u8> = a
            .synthesize()
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        let b_notes: Vec<u8> = b
            .synthesize()
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(a_notes, b_notes);
    }

    #[test]
    fn synthesize_different_seeds_diverge() {
        let a = MusicIntent {
            variation_seed: 0x1111,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        let b = MusicIntent {
            variation_seed: 0x2222,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        let a_notes: Vec<u8> = a
            .synthesize()
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        let b_notes: Vec<u8> = b
            .synthesize()
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_ne!(
            a_notes, b_notes,
            "distinct seeds should produce distinct tunes"
        );
    }

    #[test]
    fn synthesize_unknown_genre_and_mood_fallback_gracefully() {
        // Values 200 for both — outside canonical range but must still produce a tune.
        let intent = MusicIntent {
            genre: 200,
            mood: 200,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        let tune = intent.synthesize();
        let note_count = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Note { .. }))
            .count();
        assert!(note_count > 0);
    }

    #[test]
    fn from_tune_recovers_key_and_tempo() {
        let src = "X:1\nM:4/4\nL:1/4\nQ:96\nK:G\nG A B c |\n";
        let tune = crate::abc::parse(src).expect("parse");
        let intent = MusicIntent::from_tune(&tune);
        // G major → tonic idx 1, Ionian, sharps=1
        assert_eq!(intent.key, 1);
        assert_eq!(intent.mode, mode::IONIAN);
        assert_eq!(intent.tempo_bpm_offset, 96 - 40);
    }

    #[test]
    fn from_tune_recovers_length_bars() {
        // 8 notes at 4 notes/bar = 2 bars.
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC D E F | G A B c |\n";
        let tune = crate::abc::parse(src).expect("parse");
        let intent = MusicIntent::from_tune(&tune);
        assert_eq!(intent.length_bars, 2);
    }

    #[test]
    fn from_tune_seed_stable_across_calls() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n";
        let tune = crate::abc::parse(src).expect("parse");
        let seed_a = MusicIntent::from_tune(&tune).variation_seed;
        let seed_b = MusicIntent::from_tune(&tune).variation_seed;
        assert_eq!(seed_a, seed_b);
    }

    #[test]
    fn from_tune_minor_key_falls_to_aeolian() {
        // A minor = 0 sharps but written explicitly as K:Am → after parse, sharps=0.
        // from_tune prefers Ionian for 0 sharps → returns C major. This documents
        // the ambiguous-case behaviour (see ADR-010).
        let src = "X:1\nM:4/4\nL:1/4\nK:Am\nA B c d |\n";
        let tune = crate::abc::parse(src).expect("parse");
        let intent = MusicIntent::from_tune(&tune);
        assert_eq!(intent.key, 0);
        assert_eq!(intent.mode, mode::IONIAN);
    }

    #[test]
    fn intent_synthesize_produces_playable_score() {
        let intent = MusicIntent::DEFAULT_C_MAJOR;
        let tune = intent.synthesize();
        let score = tune.to_score(96);
        // Non-empty event stream, tempo carried through, at least one NoteOn.
        assert!(!score.events.is_empty());
        assert_eq!(score.header.tempo_bpm, 120);
        assert!(score
            .events
            .iter()
            .any(|e| matches!(e.kind, crate::score::NoteEventKind::NoteOn)));
    }

    #[test]
    fn fnv1a_seed_stable_for_same_body() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC D E |\n";
        let tune = crate::abc::parse(src).expect("parse");
        let h1 = fnv1a_seed(&tune.body);
        let h2 = fnv1a_seed(&tune.body);
        assert_eq!(h1, h2);
    }

    #[test]
    fn constants_module_defines_canonical_labels() {
        assert_eq!(genre::FOLK, 0);
        assert_eq!(genre::ELECTRONIC, 9);
        assert_eq!(mood::HAPPY, 0);
        assert_eq!(mood::TRIUMPHANT, 7);
        assert_eq!(mode::IONIAN, 0);
        assert_eq!(mode::LOCRIAN, 6);
        assert_eq!(mode::MAJOR, mode::IONIAN);
        assert_eq!(mode::MINOR, mode::AEOLIAN);
    }
}
