//! ABC notation parser — YuE2-style symbolic planning
//!
//! Parses a minimal subset of [ABC notation](https://abcnotation.com/wiki/abc:standard:v2.1)
//! into [`crate::score::Score`]. ABC is a compact text format widely used for folk and
//! traditional tunes, and it is the same symbolic representation adopted by `YuE2` (2026-09)
//! as the human-editable intermediate for its symbolic planning stage.
//!
//! # Supported subset (Phase 1 MVP)
//!
//! - Header fields: `X:` `T:` `M:` `L:` `Q:` `K:` (only `M` `L` `Q` `K` affect output)
//! - Body: `A-G` `a-g` note letters, `,` `'` octave modifiers, `^` `_` `=` `^^` `__` accidentals
//! - Durations: bare (=1×L), `N` (=N×L), `/N` (=L/N), `/` (=L/2), `N/M` (=N/M×L)
//! - Rest: `z` and `Z` with the same duration syntax
//! - Barlines: `|` `||` `|]` `[|` `:|` `|:` (all treated as boundary markers)
//! - Key signatures: 15 canonical major keys (C, G, D, A, E, B, F#, C#, F, Bb, Eb, Ab, Db, Gb, Cb)
//! - Inline fields `[K:...]` `[M:...]` — silently skipped
//! - Structural chars `-` `[` `]` `(` `)` `{` `}` `:` `*` — silently skipped
//! - Decorations `!...!` — silently skipped
//! - Line comments starting with `%`
//!
//! # Not supported (Phase 2 or later)
//!
//! Multi-voice (`V:`), chord expansion, ties (notes are not merged), grace notes `{}`,
//! tuplets `(3`, minor keys, mode names (dor/mix/lyd/...), microtonal accidentals.
//!
//! # Example
//!
//! ```
//! use alice_synth::abc::parse;
//!
//! let src = "X:1\nT:Twinkle\nM:4/4\nL:1/4\nQ:120\nK:C\nC C G G | A A G2 |\n";
//! let tune = parse(src).unwrap();
//! let score = tune.to_score(96);
//! assert_eq!(score.header.tempo_bpm, 120);
//! assert_eq!(score.events[0].note, 60); // middle C (C4)
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::score::{NoteEvent, NoteEventKind, Score, ScoreHeader};

const NATURAL_SEMITONE: [i16; 7] = [0, 2, 4, 5, 7, 9, 11];

/// Parse errors — no allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbcError {
    /// Character not recognized (line number 1-based, byte value)
    UnexpectedChar { line: u16, ch: u8 },
    /// `M:` must be `N/N` with both in 1..=64
    InvalidMeter,
    /// `L:` must be `N/N` with both in 1..=64
    InvalidUnitLength,
    /// `Q:` must be BPM in 40..=300, either `Q:120` or `Q:1/4=120`
    InvalidTempo,
    /// `K:` must be one of the 15 canonical major keys
    InvalidKey,
    /// Body content appeared before `K:` header
    MissingKey,
    /// Resulting MIDI pitch fell outside 0..=127
    NoteOutOfMidiRange,
    /// Duration modifier produced overflow or division by zero
    DurationOverflow,
    /// Input contained no notes
    EmptyTune,
}

/// Canonical major key signature
///
/// Represented as a signed sharp count in `-7..=7`. Positive values are sharps
/// (in the order F#, C#, G#, D#, A#, E#, B#), negative values are flats (in the
/// order Bb, Eb, Ab, Db, Gb, Cb, Fb).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeySignature {
    /// Sharps count in `-7..=7` (negative values indicate flats)
    pub sharps: i8,
}

impl KeySignature {
    /// C major (no accidentals)
    pub const C_MAJOR: Self = Self { sharps: 0 };

    /// Semitone offset that this key applies to a natural note letter.
    ///
    /// `letter_index` is 0=C, 1=D, 2=E, 3=F, 4=G, 5=A, 6=B. Returns `+1` if the
    /// note is sharpened by the key signature, `-1` if flattened, `0` otherwise.
    #[must_use]
    pub const fn accidental_for(&self, letter_index: u8) -> i8 {
        // Sharp order: F(3) C(0) G(4) D(1) A(5) E(2) B(6)
        // Flat  order: B(6) E(2) A(5) D(1) G(4) C(0) F(3)
        const SHARP_ORDER: [u8; 7] = [3, 0, 4, 1, 5, 2, 6];
        const FLAT_ORDER: [u8; 7] = [6, 2, 5, 1, 4, 0, 3];
        if letter_index >= 7 {
            return 0;
        }
        let n = self.sharps;
        if n > 0 {
            let count = n as usize;
            let mut i = 0;
            while i < count && i < 7 {
                if SHARP_ORDER[i] == letter_index {
                    return 1;
                }
                i += 1;
            }
            0
        } else if n < 0 {
            let count = (-n) as usize;
            let mut i = 0;
            while i < count && i < 7 {
                if FLAT_ORDER[i] == letter_index {
                    return -1;
                }
                i += 1;
            }
            0
        } else {
            0
        }
    }
}

/// A single body element of a parsed ABC tune.
///
/// Durations are stored as a rational `num/den` multiplier of the header's
/// unit-note length (`L:`); actual tick counts are computed lazily in
/// [`AbcTune::to_score`] so a single parse can be rendered at any tick division.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbcElement {
    /// A pitched note
    Note {
        /// MIDI note number `0..=127`
        midi: u8,
        /// Duration numerator (`num/den` × unit length)
        num: u16,
        /// Duration denominator
        den: u16,
    },
    /// A rest (silence)
    Rest {
        /// Duration numerator
        num: u16,
        /// Duration denominator
        den: u16,
    },
    /// A bar boundary — informational only in Phase 1
    Barline,
}

/// The header fields that affect audio playback.
///
/// Purely informational fields (`X:` reference, `T:` title, `C:` composer, …)
/// are recognized during parsing but not stored, keeping the type `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbcHeader {
    /// Time signature numerator (default 4)
    pub meter_num: u8,
    /// Time signature denominator (default 4)
    pub meter_den: u8,
    /// Unit note length numerator (default 1)
    pub unit_num: u8,
    /// Unit note length denominator (default 8, per ABC v2.1)
    pub unit_den: u8,
    /// Tempo in beats per minute (default 120)
    pub tempo_bpm: u16,
    /// Key signature (default C major)
    pub key: KeySignature,
}

impl Default for AbcHeader {
    fn default() -> Self {
        Self {
            meter_num: 4,
            meter_den: 4,
            unit_num: 1,
            unit_den: 8,
            tempo_bpm: 120,
            key: KeySignature::C_MAJOR,
        }
    }
}

/// A parsed ABC tune
#[derive(Debug, Clone)]
pub struct AbcTune {
    /// Playback-affecting header fields
    pub header: AbcHeader,
    /// Body element sequence
    pub body: Vec<AbcElement>,
}

impl AbcTune {
    /// Convert this tune to an ALICE-Synth [`Score`] at the given tick division.
    ///
    /// Each [`AbcElement::Note`] emits a `NoteOn` (with `delta_tick` equal to any
    /// preceding accumulated rest) followed by a `NoteOff` whose `delta_tick` is
    /// the note's own duration. Rests advance the running delta without emitting
    /// events. Barlines are silently dropped.
    ///
    /// The 4095-tick per-event delta limit imposed by [`NoteEvent`] is clamped;
    /// callers who need longer rests should split them at parse time.
    #[must_use]
    pub fn to_score(&self, tick_div: u16) -> Score {
        let mut score = Score {
            header: ScoreHeader {
                tempo_bpm: self.header.tempo_bpm,
                tracks: 1,
                tick_div,
            },
            events: Vec::with_capacity(self.body.len() * 2),
        };
        let unit_ticks = self.unit_ticks(tick_div);
        let mut pending_delta: u32 = 0;
        for elem in &self.body {
            match *elem {
                AbcElement::Note { midi, num, den } => {
                    let ticks = duration_to_ticks(unit_ticks, num, den);
                    score.events.push(NoteEvent {
                        delta_tick: u16::try_from(pending_delta.min(4095)).unwrap_or(4095),
                        channel: 0,
                        note: midi,
                        velocity: 80,
                        kind: NoteEventKind::NoteOn,
                    });
                    score.events.push(NoteEvent {
                        delta_tick: u16::try_from(ticks.min(4095)).unwrap_or(4095),
                        channel: 0,
                        note: midi,
                        velocity: 0,
                        kind: NoteEventKind::NoteOff,
                    });
                    pending_delta = 0;
                }
                AbcElement::Rest { num, den } => {
                    let ticks = duration_to_ticks(unit_ticks, num, den);
                    pending_delta = pending_delta.saturating_add(ticks);
                }
                AbcElement::Barline => {}
            }
        }
        score
    }

    /// Number of ticks per unit note at the given tick division
    #[must_use]
    pub fn unit_ticks(&self, tick_div: u16) -> u32 {
        // A whole note is `tick_div * 4` ticks; the unit note is `unit_num/unit_den` of that.
        let whole = u32::from(tick_div) * 4;
        whole * u32::from(self.header.unit_num) / u32::from(self.header.unit_den)
    }
}

fn duration_to_ticks(unit_ticks: u32, num: u16, den: u16) -> u32 {
    if den == 0 {
        return 0;
    }
    unit_ticks
        .saturating_mul(u32::from(num))
        .checked_div(u32::from(den))
        .unwrap_or(0)
}

/// Parse ABC source text into an [`AbcTune`].
///
/// # Errors
///
/// Returns [`AbcError`] on syntax errors, invalid header values, out-of-range
/// MIDI pitches, or duration overflow.
pub fn parse(input: &str) -> Result<AbcTune, AbcError> {
    let mut header = AbcHeader::default();
    let mut body: Vec<AbcElement> = Vec::new();
    let mut key_seen = false;
    let mut in_body = false;

    for (line_idx, raw_line) in input.lines().enumerate() {
        let line_no = u16::try_from(line_idx + 1).unwrap_or(u16::MAX);
        let line = raw_line.trim();

        if line.is_empty() || line.starts_with('%') {
            continue;
        }

        let is_header_line = !in_body
            && line.len() >= 2
            && line.as_bytes()[1] == b':'
            && line.as_bytes()[0].is_ascii_alphabetic();

        if is_header_line {
            let field = line.as_bytes()[0];
            let value = line[2..].trim();
            parse_header_field(field, value, line_no, &mut header)?;
            if field == b'K' {
                key_seen = true;
                in_body = true;
            }
        } else {
            if !key_seen {
                return Err(AbcError::MissingKey);
            }
            in_body = true;
            parse_body_line(line, line_no, header.key, &mut body)?;
        }
    }

    if !body.iter().any(|e| matches!(e, AbcElement::Note { .. })) {
        return Err(AbcError::EmptyTune);
    }

    Ok(AbcTune { header, body })
}

fn parse_header_field(
    field: u8,
    value: &str,
    line_no: u16,
    header: &mut AbcHeader,
) -> Result<(), AbcError> {
    match field {
        // Informational fields ignored in Phase 1 MVP
        b'X' | b'T' | b'C' | b'S' | b'N' | b'R' | b'B' | b'D' | b'H' | b'O' | b'P' | b'W'
        | b'w' | b'Z' | b'A' | b'F' | b'G' | b'I' => Ok(()),
        b'M' => {
            let (n, d) = parse_fraction(value).ok_or(AbcError::InvalidMeter)?;
            if n == 0 || d == 0 || n > 64 || d > 64 {
                return Err(AbcError::InvalidMeter);
            }
            header.meter_num = n;
            header.meter_den = d;
            Ok(())
        }
        b'L' => {
            let (n, d) = parse_fraction(value).ok_or(AbcError::InvalidUnitLength)?;
            if n == 0 || d == 0 || n > 64 || d > 64 {
                return Err(AbcError::InvalidUnitLength);
            }
            header.unit_num = n;
            header.unit_den = d;
            Ok(())
        }
        b'Q' => {
            let bpm = parse_tempo(value).ok_or(AbcError::InvalidTempo)?;
            if !(40..=300).contains(&bpm) {
                return Err(AbcError::InvalidTempo);
            }
            header.tempo_bpm = bpm;
            Ok(())
        }
        b'K' => {
            header.key = parse_key(value).ok_or(AbcError::InvalidKey)?;
            Ok(())
        }
        _ => Err(AbcError::UnexpectedChar {
            line: line_no,
            ch: field,
        }),
    }
}

fn parse_fraction(s: &str) -> Option<(u8, u8)> {
    let mut parts = s.splitn(2, '/');
    let n: u8 = parts.next()?.trim().parse().ok()?;
    let d: u8 = parts.next()?.trim().parse().ok()?;
    Some((n, d))
}

fn parse_tempo(s: &str) -> Option<u16> {
    // `Q:120` or `Q:1/4=120`
    if let Some(eq_pos) = s.find('=') {
        s[eq_pos + 1..].trim().parse().ok()
    } else {
        s.trim().parse().ok()
    }
}

fn parse_key(s: &str) -> Option<KeySignature> {
    let core = s.split_whitespace().next()?;
    let sharps = match core {
        "C" | "Cmaj" | "CMaj" => 0i8,
        "G" | "Gmaj" => 1,
        "D" | "Dmaj" => 2,
        "A" | "Amaj" => 3,
        "E" | "Emaj" => 4,
        "B" | "Bmaj" => 5,
        "F#" | "F#maj" => 6,
        "C#" | "C#maj" => 7,
        "F" | "Fmaj" => -1,
        "Bb" | "Bbmaj" => -2,
        "Eb" | "Ebmaj" => -3,
        "Ab" | "Abmaj" => -4,
        "Db" | "Dbmaj" => -5,
        "Gb" | "Gbmaj" => -6,
        "Cb" | "Cbmaj" => -7,
        _ => return None,
    };
    Some(KeySignature { sharps })
}

fn parse_body_line(
    line: &str,
    line_no: u16,
    key: KeySignature,
    body: &mut Vec<AbcElement>,
) -> Result<(), AbcError> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'%' => break, // rest of line is a comment
            b'|' => {
                body.push(AbcElement::Barline);
                i += 1;
                while i < bytes.len() && matches!(bytes[i], b'|' | b']' | b':') {
                    i += 1;
                }
            }
            b'[' => {
                // Inline field `[X:...]` — skip to `]`. Else (chord `[abc]`) skip only `[`.
                let inline_field = i + 2 < bytes.len()
                    && bytes[i + 1].is_ascii_alphabetic()
                    && bytes[i + 2] == b':';
                if inline_field {
                    i += 3;
                    while i < bytes.len() && bytes[i] != b']' {
                        i += 1;
                    }
                    if i < bytes.len() {
                        i += 1; // consume `]`
                    }
                } else {
                    i += 1;
                }
            }
            b' ' | b'\t' | b'\r' | b']' | b'-' | b'(' | b')' | b'{' | b'}' | b':' | b'*' | b'.'
            | b'~' | b'>' | b'<' => {
                i += 1;
            }
            b'!' => {
                // Decoration `!...!` — skip until closing `!`
                i += 1;
                while i < bytes.len() && bytes[i] != b'!' {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
            }
            b'"' => {
                // Chord/annotation `"..."` — skip until closing `"`
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
            }
            b'^' | b'_' | b'=' | b'A'..=b'G' | b'a'..=b'g' => {
                let (midi, consumed) = parse_note(bytes, i, line_no, key)?;
                let (num, den, dur_consumed) = parse_duration(bytes, i + consumed, line_no)?;
                body.push(AbcElement::Note { midi, num, den });
                i += consumed + dur_consumed;
            }
            b'z' | b'Z' | b'x' | b'X' => {
                i += 1;
                let (num, den, dur_consumed) = parse_duration(bytes, i, line_no)?;
                body.push(AbcElement::Rest { num, den });
                i += dur_consumed;
            }
            _ => {
                return Err(AbcError::UnexpectedChar {
                    line: line_no,
                    ch: c,
                });
            }
        }
    }
    Ok(())
}

/// Parse a note starting at position `start`. Returns `(midi, bytes_consumed)`.
fn parse_note(
    bytes: &[u8],
    start: usize,
    line_no: u16,
    key: KeySignature,
) -> Result<(u8, usize), AbcError> {
    let mut pos = start;
    let mut accidental: i16 = 0;
    let mut accidental_explicit = false;

    while pos < bytes.len() {
        match bytes[pos] {
            b'^' => {
                accidental += 1;
                accidental_explicit = true;
                pos += 1;
            }
            b'_' => {
                accidental -= 1;
                accidental_explicit = true;
                pos += 1;
            }
            b'=' => {
                accidental = 0;
                accidental_explicit = true;
                pos += 1;
            }
            _ => break,
        }
    }

    if pos >= bytes.len() {
        return Err(AbcError::UnexpectedChar {
            line: line_no,
            ch: 0,
        });
    }

    let (letter_index, base_octave) = match bytes[pos] {
        b'C' => (0u8, 4i16),
        b'D' => (1, 4),
        b'E' => (2, 4),
        b'F' => (3, 4),
        b'G' => (4, 4),
        b'A' => (5, 4),
        b'B' => (6, 4),
        b'c' => (0, 5),
        b'd' => (1, 5),
        b'e' => (2, 5),
        b'f' => (3, 5),
        b'g' => (4, 5),
        b'a' => (5, 5),
        b'b' => (6, 5),
        ch => {
            return Err(AbcError::UnexpectedChar { line: line_no, ch });
        }
    };
    pos += 1;

    let mut octave = base_octave;
    while pos < bytes.len() {
        match bytes[pos] {
            b',' => {
                octave -= 1;
                pos += 1;
            }
            b'\'' => {
                octave += 1;
                pos += 1;
            }
            _ => break,
        }
    }

    let key_offset = if accidental_explicit {
        accidental
    } else {
        i16::from(key.accidental_for(letter_index))
    };

    let midi_raw = 12 * (octave + 1) + NATURAL_SEMITONE[usize::from(letter_index)] + key_offset;
    if !(0..=127).contains(&midi_raw) {
        return Err(AbcError::NoteOutOfMidiRange);
    }
    Ok((midi_raw as u8, pos - start))
}

/// Parse an optional duration modifier. Returns `(num, den, bytes_consumed)`.
///
/// Forms recognized (relative to the header's unit length `L`):
///
/// | Input | (num, den) | Meaning         |
/// |-------|------------|-----------------|
/// | (none)| (1, 1)     | 1×L             |
/// | `2`   | (2, 1)     | 2×L             |
/// | `/2`  | (1, 2)     | L/2             |
/// | `/`   | (1, 2)     | L/2 (shorthand) |
/// | `3/2` | (3, 2)     | dotted L        |
fn parse_duration(bytes: &[u8], start: usize, line_no: u16) -> Result<(u16, u16, usize), AbcError> {
    let mut pos = start;

    let num_start = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    let num: u16 = if pos > num_start {
        parse_u16(&bytes[num_start..pos], line_no)?
    } else {
        1
    };

    let den: u16 = if pos < bytes.len() && bytes[pos] == b'/' {
        pos += 1;
        let den_start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos > den_start {
            parse_u16(&bytes[den_start..pos], line_no)?
        } else {
            2
        }
    } else {
        1
    };

    if num == 0 || den == 0 {
        return Err(AbcError::DurationOverflow);
    }
    Ok((num, den, pos - start))
}

fn parse_u16(bytes: &[u8], line_no: u16) -> Result<u16, AbcError> {
    let s = core::str::from_utf8(bytes).map_err(|_| AbcError::UnexpectedChar {
        line: line_no,
        ch: bytes.first().copied().unwrap_or(0),
    })?;
    s.parse::<u16>().map_err(|_| AbcError::DurationOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_c_major_scale() {
        let src = "X:1\nT:Scale\nM:4/4\nL:1/4\nK:C\nC D E F | G A B c |\n";
        let tune = parse(src).expect("parse");
        assert_eq!(tune.header.tempo_bpm, 120);
        assert_eq!(tune.header.meter_num, 4);
        assert_eq!(tune.header.unit_num, 1);
        assert_eq!(tune.header.unit_den, 4);
        assert_eq!(tune.header.key.sharps, 0);
        let notes: Vec<&AbcElement> = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Note { .. }))
            .collect();
        assert_eq!(notes.len(), 8);
        if let AbcElement::Note { midi, .. } = notes[0] {
            assert_eq!(*midi, 60);
        }
        if let AbcElement::Note { midi, .. } = notes[7] {
            assert_eq!(*midi, 72);
        }
    }

    #[test]
    fn parse_twinkle_first_line() {
        let src = "X:1\nT:Twinkle\nM:4/4\nL:1/4\nK:C\nC C G G | A A G2 |\n";
        let tune = parse(src).expect("parse");
        let notes: Vec<&AbcElement> = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Note { .. }))
            .collect();
        assert_eq!(notes.len(), 7);
        if let AbcElement::Note { midi, num, den } = notes[6] {
            assert_eq!(*midi, 67);
            assert_eq!(*num, 2);
            assert_eq!(*den, 1);
        }
    }

    #[test]
    fn key_signature_g_major_sharpens_f() {
        let src = "X:1\nM:4/4\nL:1/4\nK:G\nF G |\n";
        let tune = parse(src).expect("parse");
        let notes: Vec<u8> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(notes[0], 66); // F# (F=65 + 1)
        assert_eq!(notes[1], 67); // G unaffected
    }

    #[test]
    fn explicit_natural_overrides_key() {
        let src = "X:1\nM:4/4\nL:1/4\nK:G\n=F G |\n";
        let tune = parse(src).expect("parse");
        let notes: Vec<u8> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(notes[0], 65); // natural F
    }

    #[test]
    fn octave_modifiers() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC, C c c' |\n";
        let tune = parse(src).expect("parse");
        let notes: Vec<u8> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(notes.as_slice(), &[48u8, 60, 72, 84][..]);
    }

    #[test]
    fn duration_forms() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC C2 C/2 C/ C3/2 |\n";
        let tune = parse(src).expect("parse");
        let pairs: Vec<(u16, u16)> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { num, den, .. } => Some((*num, *den)),
                _ => None,
            })
            .collect();
        assert_eq!(
            pairs.as_slice(),
            &[(1u16, 1u16), (2, 1), (1, 2), (1, 2), (3, 2)][..]
        );
    }

    #[test]
    fn rests_recognized() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC z C z2 |\n";
        let tune = parse(src).expect("parse");
        let rest_count = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Rest { .. }))
            .count();
        assert_eq!(rest_count, 2);
    }

    #[test]
    fn tempo_bare_form() {
        let src = "X:1\nM:4/4\nL:1/4\nQ:90\nK:C\nC |\n";
        let tune = parse(src).expect("parse");
        assert_eq!(tune.header.tempo_bpm, 90);
    }

    #[test]
    fn tempo_note_equals_bpm_form() {
        let src = "X:1\nM:4/4\nL:1/4\nQ:1/4=140\nK:C\nC |\n";
        let tune = parse(src).expect("parse");
        assert_eq!(tune.header.tempo_bpm, 140);
    }

    #[test]
    fn error_on_missing_key() {
        let src = "X:1\nM:4/4\nL:1/4\nC C |\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::MissingKey);
    }

    #[test]
    fn error_on_invalid_key() {
        let src = "X:1\nM:4/4\nL:1/4\nK:H\nC |\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::InvalidKey);
    }

    #[test]
    fn error_on_empty_tune() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n| | |\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::EmptyTune);
    }

    #[test]
    fn error_on_out_of_range_pitch() {
        // C,,,,, is too low: octave = 4 - 5 = -1 → MIDI = 12*0 + 0 = 0, which is valid.
        // Force below: C,,,,,, → octave = -2 → MIDI = -12. Should error.
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC,,,,,, |\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::NoteOutOfMidiRange);
    }

    #[test]
    fn score_conversion_note_pairs() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC D |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);
        assert_eq!(score.events.len(), 4);
        assert_eq!(score.events[0].kind, NoteEventKind::NoteOn);
        assert_eq!(score.events[0].note, 60);
        assert_eq!(score.events[1].kind, NoteEventKind::NoteOff);
        assert_eq!(score.events[1].delta_tick, 96);
        assert_eq!(score.events[2].kind, NoteEventKind::NoteOn);
        assert_eq!(score.events[2].note, 62);
    }

    #[test]
    fn score_conversion_rest_becomes_delta() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC z D |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);
        assert_eq!(score.events[0].delta_tick, 0);
        assert_eq!(score.events[1].delta_tick, 96);
        assert_eq!(score.events[2].delta_tick, 96);
        assert_eq!(score.events[3].delta_tick, 96);
    }

    #[test]
    fn score_conversion_carries_tempo() {
        let src = "X:1\nM:4/4\nL:1/4\nQ:80\nK:C\nC |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);
        assert_eq!(score.header.tempo_bpm, 80);
        assert_eq!(score.header.tick_div, 96);
    }

    #[test]
    fn score_conversion_scales_with_tick_div() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC |\n";
        let tune = parse(src).expect("parse");
        let s96 = tune.to_score(96);
        let s48 = tune.to_score(48);
        assert_eq!(s96.events[1].delta_tick, 96);
        assert_eq!(s48.events[1].delta_tick, 48);
    }

    #[test]
    fn structural_chars_are_skipped() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC-D [C]D !ff!C \"Am\"D |\n";
        let tune = parse(src).expect("parse");
        let note_count = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Note { .. }))
            .count();
        assert!(note_count >= 4, "expected >= 4 notes, got {note_count}");
    }

    #[test]
    fn inline_field_is_skipped() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC [K:G] D |\n";
        let tune = parse(src).expect("parse");
        let note_count = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Note { .. }))
            .count();
        assert_eq!(note_count, 2);
    }

    #[test]
    fn all_15_major_keys_parse() {
        // Pre-built &str fixtures to keep this test compatible with `no_std + alloc`
        // (avoids depending on the `String` prelude).
        let fixtures: &[&str] = &[
            "X:1\nM:4/4\nL:1/4\nK:C\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:G\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:D\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:A\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:E\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:B\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:F#\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:C#\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:F\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Bb\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Eb\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Ab\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Db\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Gb\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Cb\nC |\n",
        ];
        for src in fixtures {
            assert!(parse(src).is_ok(), "fixture should parse: {src:?}");
        }
    }

    #[test]
    fn double_sharp_double_flat() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n^^C __B |\n";
        let tune = parse(src).expect("parse");
        let notes: Vec<u8> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(notes[0], 62); // C4 (60) + 2
        assert_eq!(notes[1], 69); // B4 (71) - 2
    }

    #[test]
    fn line_comment_percent_is_ignored() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC % this is a comment\nD |\n";
        let tune = parse(src).expect("parse");
        let note_count = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Note { .. }))
            .count();
        assert_eq!(note_count, 2);
    }

    #[test]
    fn key_signature_accidental_table_correct() {
        let g = KeySignature { sharps: 1 };
        assert_eq!(g.accidental_for(3), 1); // F sharpened
        assert_eq!(g.accidental_for(0), 0); // C unchanged

        let f = KeySignature { sharps: -1 };
        assert_eq!(f.accidental_for(6), -1); // B flattened
        assert_eq!(f.accidental_for(0), 0);

        let c_sharp = KeySignature { sharps: 7 };
        for i in 0..7 {
            assert_eq!(c_sharp.accidental_for(i), 1);
        }

        let c_flat = KeySignature { sharps: -7 };
        for i in 0..7 {
            assert_eq!(c_flat.accidental_for(i), -1);
        }

        let c_major = KeySignature::C_MAJOR;
        for i in 0..7 {
            assert_eq!(c_major.accidental_for(i), 0);
        }
    }

    #[test]
    fn end_to_end_c_major_ode_to_joy_prefix() {
        // Ode to Joy first phrase: E E F G G F E D C C D E E D D
        let src = "X:1\nT:Ode to Joy\nM:4/4\nL:1/4\nQ:100\nK:C\n\
                   E E F G | G F E D | C C D E | E3/2 D/2 D2 |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);
        assert_eq!(score.header.tempo_bpm, 100);
        assert!(score.events.len() >= 28); // 14+ notes × 2 events
    }

    #[test]
    fn end_to_end_abc_to_pcm_via_synthesizer() {
        // Full pipeline validation: ABC text → AbcTune → Score → Synthesizer → PCM samples.
        // Proves the whole "send the score, not the waveform" contract lands as audible audio.
        use crate::synth::Synthesizer;

        let src = "X:1\nM:4/4\nL:1/4\nQ:120\nK:C\nC E G c |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);

        let sample_rate: u32 = 44_100;
        let mut synth = Synthesizer::new(sample_rate);
        synth.load_score(&score);

        // Render 0.5 s (22050 samples) — enough to hear the first note fully.
        let mut buf = [0.0_f32; 4096];
        synth.render(&mut buf);

        // The score starts with a NoteOn at delta=0, so the first samples must be non-silent.
        let non_zero = buf.iter().filter(|s| s.abs() > 1e-6).count();
        assert!(
            non_zero > 100,
            "synth should produce non-silent audio from ABC input, got {non_zero} non-zero samples"
        );
    }

    #[test]
    fn unit_ticks_computes_correctly() {
        let header_quarter = AbcHeader {
            unit_num: 1,
            unit_den: 4,
            ..AbcHeader::default()
        };
        let tune = AbcTune {
            header: header_quarter,
            body: Vec::new(),
        };
        assert_eq!(tune.unit_ticks(96), 96); // quarter at 96 tpq = 96 ticks
        assert_eq!(tune.unit_ticks(48), 48);

        let header_eighth = AbcHeader {
            unit_num: 1,
            unit_den: 8,
            ..AbcHeader::default()
        };
        let tune8 = AbcTune {
            header: header_eighth,
            body: Vec::new(),
        };
        assert_eq!(tune8.unit_ticks(96), 48); // eighth = half of quarter
    }
}
