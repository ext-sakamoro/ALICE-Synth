//! ABC notation parser — YuE2-style symbolic planning
//!
//! Parses a minimal subset of [ABC notation](https://abcnotation.com/wiki/abc:standard:v2.1)
//! into [`crate::score::Score`]. ABC is a compact text format widely used for folk and
//! traditional tunes, and it is the same symbolic representation adopted by `YuE2` (2026-09)
//! as the human-editable intermediate for its symbolic planning stage.
//!
//! # Supported subset (through Phase 2a)
//!
//! - Header fields: `X:` `T:` `M:` `L:` `Q:` `K:` (only `M` `L` `Q` `K` affect output)
//! - Body: `A-G` `a-g` note letters, `,` `'` octave modifiers, `^` `_` `=` `^^` `__` accidentals
//! - Durations: bare (=1×L), `N` (=N×L), `/N` (=L/N), `/` (=L/2), `N/M` (=N/M×L)
//! - Rest: `z` and `Z` with the same duration syntax
//! - Barlines: `|` `||` `|]` `[|` (as boundaries)
//! - Chord notation: `[CEG]` up to [`MAX_CHORD_NOTES`] simultaneous notes, per-note accidentals
//! - Ties `-`: consecutive same-pitch notes merged into a single sustained note
//! - Repeats: `|:` … `:|` unrolled twice at parse time
//! - Voltas: `[1` first ending / `[2` second ending inside a repeat block
//! - Key signatures: 30 canonical keys — 15 major (C/G/D/A/E/B/F#/C#/F/Bb/Eb/Ab/Db/Gb/Cb) and
//!   15 relative minors (Am/Em/…/Abm, accepting both `m` and `min` suffixes)
//! - Inline fields `[K:...]` `[M:...]` — silently skipped
//! - Structural chars `(` `)` `{` `}` `*` `.` `~` `>` `<` — silently skipped
//! - Decorations `!...!` — silently skipped
//! - Annotations `"..."` — silently skipped
//! - Line comments starting with `%`
//!
//! # Added in Phase 2b (2026-09-13)
//!
//! - Church modes: Dor / Mix / Lyd / Phr / Loc + Ion / Aeo synonyms (30+ keys)
//! - Tuplets: `(P`, `(P:Q`, `(P:Q:R` — duration multiplier applied to next `r` musical elements
//! - Grace notes: `{gab}c` → each inner note emitted as a `1/32` prefix `Note`
//! - Multi-voice: `V:1` / `V:2` / … switch active voice; voice N renders to channel N-1
//!
//! # Not supported (Phase 2c or later)
//!
//! Microtonal accidentals, chord-level tie coalescing (chord `tie_follows` is parsed but not
//! merged), volta numbers ≥ 3, repeat shorthand `::` / `|:|`.
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
use alloc::{vec, vec::Vec};

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
    /// Chord exceeded the 8-note capacity
    ChordTooLarge,
    /// Chord `[...]` was not closed with a matching `]`
    UnbalancedChord,
    /// Nested `|:` inside another repeat block (not supported in Phase 2a)
    NestedRepeat,
    /// `|:` opened without a matching `:|`
    UnbalancedRepeat,
    /// `[N` volta indicator used with an unsupported number (Phase 2a supports 1..=2)
    UnsupportedVolta,
    /// `(N` tuplet size outside the supported range 2..=9, or malformed `(P:Q:R` spec
    UnsupportedTuplet,
    /// `V:N` voice number outside 1..=16, or missing / non-numeric
    InvalidVoice,
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

/// Maximum number of simultaneous notes in an [`AbcElement::Chord`].
///
/// Chosen to cover triads, seventh chords, ninth chords, and dense piano voicings
/// while keeping `AbcElement` `Copy` and cache-line friendly.
pub const MAX_CHORD_NOTES: usize = 8;

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
        /// `true` if the ABC source suffixed this note with `-` (tie)
        tie_follows: bool,
    },
    /// A rest (silence)
    Rest {
        /// Duration numerator
        num: u16,
        /// Duration denominator
        den: u16,
    },
    /// A chord — up to [`MAX_CHORD_NOTES`] simultaneous notes sharing one duration
    Chord {
        /// MIDI note numbers; only the first `count` entries are valid
        notes: [u8; MAX_CHORD_NOTES],
        /// Number of valid entries in `notes` (`1..=MAX_CHORD_NOTES`)
        count: u8,
        /// Duration numerator (`num/den` × unit length)
        num: u16,
        /// Duration denominator
        den: u16,
        /// `true` if the chord was suffixed with `-`
        ///
        /// Chord-level tie coalescing is not yet implemented (Phase 2b); the field
        /// is retained for round-trip fidelity and for future consumers.
        tie_follows: bool,
    },
    /// A bar boundary — informational only, dropped by [`AbcTune::to_score`]
    Barline,
    /// `|:` marker (consumed by repeat unrolling; not expected in body after parse)
    RepeatStart,
    /// `:|` marker (consumed by repeat unrolling; not expected in body after parse)
    RepeatEnd,
    /// `[N` volta start marker (consumed by repeat unrolling)
    VoltaStart(u8),
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
///
/// Voice 0 (the default / lead voice) lives in [`AbcTune::body`]. Additional
/// voices declared via `V:2`, `V:3`, … appear in [`AbcTune::extra_voices`]
/// (index `N-1` maps to `V:(N+1)`). Single-voice tunes leave `extra_voices`
/// empty and are byte-for-byte compatible with pre-Phase-2b output.
#[derive(Debug, Clone)]
pub struct AbcTune {
    /// Playback-affecting header fields
    pub header: AbcHeader,
    /// Voice 0 body element sequence (the default / lead voice)
    pub body: Vec<AbcElement>,
    /// Additional voices for `V:2` and above; each is an independent element sequence
    pub extra_voices: Vec<Vec<AbcElement>>,
}

impl AbcTune {
    /// Convert this tune to an ALICE-Synth [`Score`] at the given tick division.
    ///
    /// Emission rules:
    ///
    /// - [`AbcElement::Note`] → `NoteOn` (delta = accumulated pending rest ticks) +
    ///   `NoteOff` (delta = note duration). Consecutive same-pitch notes joined by
    ///   `-` (tie) are coalesced into a single `NoteOn`/`NoteOff` pair whose duration
    ///   is the sum of all tied notes' ticks.
    /// - [`AbcElement::Chord`] → N `NoteOn` events (first with pending delta, rest
    ///   with delta 0) followed by N `NoteOff` events (first with chord duration,
    ///   rest with delta 0). Chord-level ties are not yet coalesced.
    /// - [`AbcElement::Rest`] → advances the running delta only.
    /// - [`AbcElement::Barline`] → dropped.
    /// - Repeat / volta markers are consumed by parse-time unrolling and should
    ///   never reach this method.
    ///
    /// **Multi-voice** (Phase 2b): voice 0 renders to `channel = 0`; each extra
    /// voice renders to `channel = voice_index + 1`. Voice event streams are
    /// merged by absolute tick (stable-sorted) so simultaneous events across
    /// voices retain their per-voice emission order.
    ///
    /// The 4095-tick per-event delta limit imposed by [`NoteEvent`] is clamped;
    /// callers who need longer rests should split them at parse time.
    #[must_use]
    pub fn to_score(&self, tick_div: u16) -> Score {
        let unit_ticks = self.unit_ticks(tick_div);
        let mut absolute: Vec<AbsoluteEvent> = Vec::with_capacity(self.body.len() * 2);
        render_voice_to_absolute(&self.body, 0, unit_ticks, &mut absolute);
        for (idx, voice) in self.extra_voices.iter().enumerate() {
            let channel = u8::try_from((idx + 1) & 0x0F).unwrap_or(15);
            render_voice_to_absolute(voice, channel, unit_ticks, &mut absolute);
        }
        // Stable sort by absolute tick — preserves per-voice emission order for
        // events sharing a tick, which is what NoteEvent delta semantics need.
        absolute.sort_by_key(|e| e.tick);
        let mut prev_tick: u32 = 0;
        let mut events: Vec<NoteEvent> = Vec::with_capacity(absolute.len());
        for e in absolute {
            let delta_u32 = e.tick.saturating_sub(prev_tick);
            events.push(NoteEvent {
                delta_tick: u16::try_from(delta_u32.min(4095)).unwrap_or(4095),
                channel: e.channel,
                note: e.note,
                velocity: e.velocity,
                kind: e.kind,
            });
            prev_tick = e.tick;
        }
        let num_voices = 1 + self.extra_voices.len();
        Score {
            header: ScoreHeader {
                tempo_bpm: self.header.tempo_bpm,
                tracks: u8::try_from(num_voices.min(16)).unwrap_or(1),
                tick_div,
            },
            events,
        }
    }

    /// Number of ticks per unit note at the given tick division
    #[must_use]
    pub fn unit_ticks(&self, tick_div: u16) -> u32 {
        // A whole note is `tick_div * 4` ticks; the unit note is `unit_num/unit_den` of that.
        let whole = u32::from(tick_div) * 4;
        whole * u32::from(self.header.unit_num) / u32::from(self.header.unit_den)
    }
}

/// Internal absolute-tick event used while merging voices in [`AbcTune::to_score`].
#[derive(Debug, Clone, Copy)]
struct AbsoluteEvent {
    tick: u32,
    channel: u8,
    note: u8,
    velocity: u8,
    kind: NoteEventKind,
}

/// Render one voice's body into absolute-tick events on the given channel.
///
/// This is the multi-voice-aware core of [`AbcTune::to_score`]; the outer merge
/// step is responsible for sorting the resulting stream and converting back to
/// delta ticks.
#[allow(clippy::too_many_lines)]
fn render_voice_to_absolute(
    body: &[AbcElement],
    channel: u8,
    unit_ticks: u32,
    out: &mut Vec<AbsoluteEvent>,
) {
    let mut cursor: u32 = 0;
    let mut pending_delta: u32 = 0;
    let mut i = 0;
    while i < body.len() {
        match body[i] {
            AbcElement::Note {
                midi,
                num,
                den,
                tie_follows,
            } => {
                // Coalesce a chain of tied notes of the same pitch.
                let mut total_ticks = duration_to_ticks(unit_ticks, num, den);
                let mut cur_tie = tie_follows;
                let mut next_i = i + 1;
                while cur_tie && next_i < body.len() {
                    let mut probe = next_i;
                    while probe < body.len() && matches!(body[probe], AbcElement::Barline) {
                        probe += 1;
                    }
                    if probe >= body.len() {
                        break;
                    }
                    if let AbcElement::Note {
                        midi: m2,
                        num: n2,
                        den: d2,
                        tie_follows: tf2,
                    } = body[probe]
                    {
                        if m2 == midi {
                            total_ticks =
                                total_ticks.saturating_add(duration_to_ticks(unit_ticks, n2, d2));
                            cur_tie = tf2;
                            next_i = probe + 1;
                            continue;
                        }
                    }
                    break;
                }
                cursor = cursor.saturating_add(pending_delta);
                pending_delta = 0;
                out.push(AbsoluteEvent {
                    tick: cursor,
                    channel,
                    note: midi,
                    velocity: 80,
                    kind: NoteEventKind::NoteOn,
                });
                cursor = cursor.saturating_add(total_ticks);
                out.push(AbsoluteEvent {
                    tick: cursor,
                    channel,
                    note: midi,
                    velocity: 0,
                    kind: NoteEventKind::NoteOff,
                });
                i = next_i;
            }
            AbcElement::Rest { num, den } => {
                pending_delta =
                    pending_delta.saturating_add(duration_to_ticks(unit_ticks, num, den));
                i += 1;
            }
            AbcElement::Chord {
                notes,
                count,
                num,
                den,
                tie_follows: _,
            } => {
                cursor = cursor.saturating_add(pending_delta);
                pending_delta = 0;
                let cnt = usize::from(count);
                let ticks = duration_to_ticks(unit_ticks, num, den);
                for note in notes[..cnt].iter().copied() {
                    out.push(AbsoluteEvent {
                        tick: cursor,
                        channel,
                        note,
                        velocity: 80,
                        kind: NoteEventKind::NoteOn,
                    });
                }
                let off_tick = cursor.saturating_add(ticks);
                for note in notes[..cnt].iter().copied() {
                    out.push(AbsoluteEvent {
                        tick: off_tick,
                        channel,
                        note,
                        velocity: 0,
                        kind: NoteEventKind::NoteOff,
                    });
                }
                cursor = off_tick;
                i += 1;
            }
            AbcElement::Barline
            | AbcElement::RepeatStart
            | AbcElement::RepeatEnd
            | AbcElement::VoltaStart(_) => {
                i += 1;
            }
        }
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
    let mut voices: Vec<Vec<AbcElement>> = vec![Vec::new()];
    let mut current_voice: usize = 0;
    let mut key_seen = false;
    let mut in_body = false;
    let mut tuplet_state: Option<TupletState> = None;

    for (line_idx, raw_line) in input.lines().enumerate() {
        let line_no = u16::try_from(line_idx + 1).unwrap_or(u16::MAX);
        let line = raw_line.trim();

        if line.is_empty() || line.starts_with('%') {
            continue;
        }

        let is_field_line = line.len() >= 2
            && line.as_bytes()[1] == b':'
            && line.as_bytes()[0].is_ascii_alphabetic();

        // `V:N` is handled specially — voice switches may occur in either the
        // header block or between body lines, so we bypass the header/body split.
        if is_field_line && line.as_bytes()[0] == b'V' {
            let voice_num = parse_voice_field(&line[2..])?;
            let idx = usize::from(voice_num - 1);
            while voices.len() <= idx {
                voices.push(Vec::new());
            }
            current_voice = idx;
            tuplet_state = None;
            continue;
        }

        if !in_body && is_field_line {
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
            parse_body_line(
                line,
                line_no,
                header.key,
                &mut voices[current_voice],
                &mut tuplet_state,
            )?;
        }
    }

    for voice in &mut voices {
        unroll_repeats(voice)?;
    }

    let has_notes = voices
        .iter()
        .flatten()
        .any(|e| matches!(e, AbcElement::Note { .. } | AbcElement::Chord { .. }));
    if !has_notes {
        return Err(AbcError::EmptyTune);
    }

    let body = voices.remove(0);
    let extra_voices = voices;

    Ok(AbcTune {
        header,
        body,
        extra_voices,
    })
}

/// Extract the numeric voice index from a `V:` header value.
///
/// Accepts a bare integer (`V:2`) or an integer followed by decorative attributes
/// (`V:2 clef=bass`); only the numeric prefix is inspected. Voice numbers must
/// be in `1..=16` to match [`NoteEvent`]'s 4-bit channel field.
fn parse_voice_field(value: &str) -> Result<u8, AbcError> {
    let trimmed = value.trim_start();
    let digits: usize = trimmed.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return Err(AbcError::InvalidVoice);
    }
    let num: u8 = trimmed[..digits]
        .parse()
        .map_err(|_| AbcError::InvalidVoice)?;
    if !(1..=16).contains(&num) {
        return Err(AbcError::InvalidVoice);
    }
    Ok(num)
}

/// Expand `|:` `:|` blocks (with optional `[1` / `[2` voltas) into a linear body.
///
/// Simple case (no voltas): `|: A :|` → `A A`.
/// Volta case: `|: A [1 B :| [2 C |` → `A B A C |` (barlines / trailing markers are
/// preserved for downstream tooling).
///
/// The unroller processes one repeat block per iteration and re-scans after each
/// expansion, which keeps the algorithm linear in the number of repeat markers
/// even though it uses `splice`. Nested repeats are not supported and produce
/// [`AbcError::NestedRepeat`].
#[allow(clippy::too_many_lines)]
fn unroll_repeats(body: &mut Vec<AbcElement>) -> Result<(), AbcError> {
    let mut i = 0;
    while i < body.len() {
        if !matches!(body[i], AbcElement::RepeatStart) {
            i += 1;
            continue;
        }
        let start = i;

        // Locate matching RepeatEnd; error on nested RepeatStart.
        let mut end_idx: Option<usize> = None;
        for (offset, elem) in body.iter().enumerate().skip(start + 1) {
            match elem {
                AbcElement::RepeatStart => return Err(AbcError::NestedRepeat),
                AbcElement::RepeatEnd => {
                    end_idx = Some(offset);
                    break;
                }
                _ => {}
            }
        }
        let end = end_idx.ok_or(AbcError::UnbalancedRepeat)?;

        // Locate optional first-ending marker inside the repeat block.
        let volta1 = ((start + 1)..end).find(|&k| matches!(body[k], AbcElement::VoltaStart(1)));

        // Locate optional second-ending marker immediately after the RepeatEnd.
        // Skip a single barline between `:|` and `[2` if present.
        let volta2_marker = {
            let mut probe = end + 1;
            while probe < body.len() && matches!(body[probe], AbcElement::Barline) {
                probe += 1;
            }
            if probe < body.len() && matches!(body[probe], AbcElement::VoltaStart(2)) {
                Some(probe)
            } else {
                None
            }
        };

        // Determine slice boundaries.
        let common_end = volta1.unwrap_or(end);
        let common: Vec<AbcElement> = body[(start + 1)..common_end].to_vec();
        let first_ending: Vec<AbcElement> = if let Some(v1) = volta1 {
            body[(v1 + 1)..end].to_vec()
        } else {
            Vec::new()
        };

        // Second ending runs from after `[2` up to the next Barline / repeat marker
        // / volta / end of body. That terminator itself is not consumed here so it
        // stays in the body after splicing.
        let (replace_end, second_ending): (usize, Vec<AbcElement>) = match volta2_marker {
            Some(v2) => {
                let content_start = v2 + 1;
                let content_end = body
                    .iter()
                    .enumerate()
                    .skip(content_start)
                    .find(|(_, e)| {
                        matches!(
                            e,
                            AbcElement::Barline
                                | AbcElement::RepeatStart
                                | AbcElement::RepeatEnd
                                | AbcElement::VoltaStart(_)
                        )
                    })
                    .map_or(body.len(), |(idx, _)| idx);
                (content_end, body[content_start..content_end].to_vec())
            }
            None => (end + 1, Vec::new()),
        };

        let expanded_len = common.len() * 2 + first_ending.len() + second_ending.len();
        let mut expanded: Vec<AbcElement> = Vec::with_capacity(expanded_len);
        expanded.extend_from_slice(&common);
        expanded.extend_from_slice(&first_ending);
        expanded.extend_from_slice(&common);
        expanded.extend_from_slice(&second_ending);

        body.splice(start..replace_end, expanded);

        // Restart scan at `start` — the expansion may itself contain no repeat markers,
        // but this keeps the loop invariant simple.
        i = start;
    }

    // After unroll, any leftover markers (e.g. stray `[1` without a repeat block) are
    // dropped so `to_score()` never has to reason about them.
    body.retain(|e| {
        !matches!(
            e,
            AbcElement::RepeatStart | AbcElement::RepeatEnd | AbcElement::VoltaStart(_)
        )
    });
    Ok(())
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
    // Compute the sharp count as `tonic_major_sharps + mode_offset`. This keeps the
    // Phase 2b mode expansion table-driven (15 tonics × 7 modes = 105 combinations)
    // without exploding the match arm count.
    let core = s.split_whitespace().next()?;
    let (tonic, mode_suffix) = split_tonic_mode(core)?;
    let tonic_sharps = tonic_major_sharps(tonic)?;
    let mode_offset = mode_offset(mode_suffix)?;
    let sharps = tonic_sharps.checked_add(mode_offset)?;
    if !(-7..=7).contains(&sharps) {
        return None;
    }
    Some(KeySignature { sharps })
}

/// Split a key string like `"Ador"` into (tonic, mode-suffix). Tonic is 1-2 chars
/// (letter plus optional `#`/`b`), remainder is the mode suffix (may be empty).
fn split_tonic_mode(core: &str) -> Option<(&str, &str)> {
    let bytes = core.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_alphabetic() {
        return None;
    }
    let split = if bytes.len() > 1 && matches!(bytes[1], b'#' | b'b') {
        // Only treat `b` as an accidental when the tonic letter is A-G *and* the
        // following char isn't a mode letter — otherwise `Bmin` would be misread
        // as tonic "Bb". Cheap disambiguation: check if the char after `b` is a
        // vowel/consonant that starts a mode ("m" for minor is the only shared
        // prefix; luckily `Bb` is followed by `m`/`maj`/end, and `Bm` starts with
        // `m` directly). So we accept the naive 2-char tonic only when the tonic
        // letter is uppercase A-G and the sharp/flat sign is a canonical modifier.
        if bytes[1] == b'#' {
            2
        } else if bytes[1] == b'b' && bytes[0] >= b'A' && bytes[0] <= b'G' {
            // Reject the ambiguous case `Bm`/`Bmin` (mode-only) by checking that
            // the char after `b` (if any) is not a mode-continuation letter.
            // Mode strings all start with a lowercase letter (a-z). To keep this
            // simple: if the third char is `m` and the *fourth* char isn't `i`
            // (for `min`) or end, then it's really `Xb` + `major`-ish. But this
            // is getting fragile; the canonical solution is: `b` is only a flat
            // accidental for the seven letters that can be flatted (B, E, A, D,
            // G, C, F). B and E are the tricky ones because `Bm`/`Em` are valid
            // minor keys. The disambiguator: `Bb` is always followed by end or a
            // mode marker starting with `m` (`maj`/`min`) — so `Bm` vs `Bbm` we
            // detect by peeking bytes[2]: if it exists and is `a` or `i`, then
            // we have `Bmaj` or `Bmin` (so tonic is `B`, not `Bb`). Otherwise
            // treat as `Bb` + rest.
            if bytes.len() >= 3 && matches!(bytes[2], b'a' | b'i') {
                1 // tonic is just `B`
            } else {
                2 // tonic is `Bb`
            }
        } else {
            1
        }
    } else {
        1
    };
    Some(core.split_at(split))
}

/// Sharp count of the tonic treated as an Ionian (major) key.
///
/// Extra tonics `G#`/`D#`/`A#` (with theoretical major sharp counts 8/9/10) are
/// listed because they appear as valid *minor* / *modal* keys — the bounds check
/// in [`parse_key`] rejects the final `sharps` when it falls outside `-7..=7`.
fn tonic_major_sharps(t: &str) -> Option<i8> {
    Some(match t {
        "C" => 0,
        "G" => 1,
        "D" => 2,
        "A" => 3,
        "E" => 4,
        "B" => 5,
        "F#" => 6,
        "C#" => 7,
        "G#" => 8,
        "D#" => 9,
        "A#" => 10,
        "F" => -1,
        "Bb" => -2,
        "Eb" => -3,
        "Ab" => -4,
        "Db" => -5,
        "Gb" => -6,
        "Cb" => -7,
        _ => return None,
    })
}

/// Sharp-count offset applied by a mode suffix relative to Ionian.
///
/// Canonical circle-of-fifths mode positions:
/// Lyd(+1) Ion(0) Mix(-1) Dor(-2) Aeo(-3) Phr(-4) Loc(-5)
fn mode_offset(suffix: &str) -> Option<i8> {
    Some(match suffix {
        "" | "maj" | "Maj" | "major" | "Major" | "ion" | "Ion" | "ionian" | "Ionian" => 0,
        "m" | "min" | "aeo" | "Aeo" | "aeolian" | "Aeolian" | "minor" | "Minor" => -3,
        "dor" | "Dor" | "dorian" | "Dorian" => -2,
        "phr" | "Phr" | "phrygian" | "Phrygian" => -4,
        "lyd" | "Lyd" | "lydian" | "Lydian" => 1,
        "mix" | "Mix" | "mixolydian" | "Mixolydian" => -1,
        "loc" | "Loc" | "locrian" | "Locrian" => -5,
        _ => return None,
    })
}

#[allow(clippy::too_many_lines)]
fn parse_body_line(
    line: &str,
    line_no: u16,
    key: KeySignature,
    body: &mut Vec<AbcElement>,
    tuplet_state: &mut Option<TupletState>,
) -> Result<(), AbcError> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'%' => break, // rest of line is a comment
            b'(' => {
                // `(N` — tuplet spec. Else regular open-paren (slur/grouping) — skip.
                if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                    let (spec, consumed) = parse_tuplet(bytes, i, line_no)?;
                    *tuplet_state = Some(spec);
                    i += consumed;
                    continue;
                }
                i += 1;
            }
            b'{' => {
                // Grace-note group `{...}`. Expand each inner note as a normal Note
                // with duration (1, 32); rests/chords inside grace groups are not
                // supported and produce UnexpectedChar. Tuplet state is *not*
                // consumed by grace notes.
                i += 1;
                while i < bytes.len() && bytes[i] != b'}' {
                    match bytes[i] {
                        b' ' | b'\t' => {
                            i += 1;
                        }
                        b'^' | b'_' | b'=' | b'A'..=b'G' | b'a'..=b'g' => {
                            let (midi, consumed) = parse_note(bytes, i, line_no, key)?;
                            // Grace notes ignore any duration suffix in ABC MVP — they
                            // always render as a 1/32 note.
                            body.push(AbcElement::Note {
                                midi,
                                num: 1,
                                den: 32,
                                tie_follows: false,
                            });
                            i += consumed;
                        }
                        ch => {
                            return Err(AbcError::UnexpectedChar { line: line_no, ch });
                        }
                    }
                }
                if i >= bytes.len() {
                    return Err(AbcError::UnbalancedChord); // reuse: unbalanced `{`
                }
                i += 1; // consume `}`
            }
            b'|' => {
                // `|:` = RepeatStart, else Barline (with `||`/`|]` swallowed)
                if i + 1 < bytes.len() && bytes[i + 1] == b':' {
                    body.push(AbcElement::RepeatStart);
                    i += 2;
                } else {
                    body.push(AbcElement::Barline);
                    i += 1;
                    while i < bytes.len() && matches!(bytes[i], b'|' | b']') {
                        i += 1;
                    }
                }
            }
            b':' => {
                // `:|` = RepeatEnd, else no-op (stray `:` treated as structural)
                if i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                    body.push(AbcElement::RepeatEnd);
                    i += 2;
                    while i < bytes.len() && matches!(bytes[i], b'|' | b']') {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
            b'[' => {
                // Order: (1) volta `[N`, (2) inline field `[X:...]`, (3) chord `[<notes>]`
                if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                    let (volta, consumed) = parse_volta(bytes, i, line_no)?;
                    body.push(volta);
                    i += consumed;
                    continue;
                }
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
                    continue;
                }
                // Chord
                let (chord, consumed) = parse_chord(bytes, i, line_no, key)?;
                body.push(chord);
                apply_tuplet(body, tuplet_state);
                i += consumed;
            }
            b' ' | b'\t' | b'\r' | b']' | b'-' | b')' | b'*' | b'.' | b'~' | b'>' | b'<' => {
                // Structural / decoration characters — silently skipped.
                // `-` here handles stray ties (a `-` right after a note is instead absorbed
                // by the note branch below); `(`, `{`, `}` are handled by their own
                // dedicated branches for tuplets and grace notes.
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
                let mut cursor = i + consumed + dur_consumed;
                let tie_follows = cursor < bytes.len() && bytes[cursor] == b'-';
                if tie_follows {
                    cursor += 1;
                }
                body.push(AbcElement::Note {
                    midi,
                    num,
                    den,
                    tie_follows,
                });
                apply_tuplet(body, tuplet_state);
                i = cursor;
            }
            b'z' | b'Z' | b'x' | b'X' => {
                i += 1;
                let (num, den, dur_consumed) = parse_duration(bytes, i, line_no)?;
                body.push(AbcElement::Rest { num, den });
                apply_tuplet(body, tuplet_state);
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

/// Active tuplet coverage: `q/p` duration multiplier applied to the next
/// `r_remaining` musical elements (Note / Chord / Rest).
#[derive(Debug, Clone, Copy)]
pub struct TupletState {
    /// Tuplet size — how many notes are being played
    pub p: u8,
    /// Time-count — the notes take the space of `q` normal notes
    pub q: u8,
    /// Remaining notes still to receive the multiplier
    pub r_remaining: u8,
}

/// Default `q` value for a bare `(P` tuplet spec, per ABC 2.1.
const fn default_tuplet_q(p: u8) -> u8 {
    match p {
        2 | 4 | 8 => 3,
        _ => 2,
    }
}

/// Parse a tuplet spec `(P`, `(P:Q`, or `(P:Q:R` starting at `(`.
///
/// Returns the initial [`TupletState`] and the byte length consumed. Missing `Q`
/// falls back to [`default_tuplet_q`]; missing `R` defaults to `P`.
fn parse_tuplet(
    bytes: &[u8],
    start: usize,
    line_no: u16,
) -> Result<(TupletState, usize), AbcError> {
    let mut pos = start + 1;
    let (p_val, p_len) = read_digits(bytes, pos, line_no)?;
    if p_len == 0 {
        return Err(AbcError::UnsupportedTuplet);
    }
    pos += p_len;
    let p = u8::try_from(p_val).map_err(|_| AbcError::UnsupportedTuplet)?;
    if !(2..=9).contains(&p) {
        return Err(AbcError::UnsupportedTuplet);
    }

    let mut q = default_tuplet_q(p);
    let mut r = p;

    if pos < bytes.len() && bytes[pos] == b':' {
        pos += 1;
        let (q_val, q_len) = read_digits(bytes, pos, line_no)?;
        if q_len > 0 {
            q = u8::try_from(q_val).map_err(|_| AbcError::UnsupportedTuplet)?;
        }
        pos += q_len;

        if pos < bytes.len() && bytes[pos] == b':' {
            pos += 1;
            let (r_val, r_len) = read_digits(bytes, pos, line_no)?;
            if r_len > 0 {
                r = u8::try_from(r_val).map_err(|_| AbcError::UnsupportedTuplet)?;
            }
            pos += r_len;
        }
    }

    if q == 0 || r == 0 {
        return Err(AbcError::UnsupportedTuplet);
    }

    Ok((
        TupletState {
            p,
            q,
            r_remaining: r,
        },
        pos - start,
    ))
}

/// Read a run of ASCII digits starting at `pos`. Returns `(value, byte_length)`.
fn read_digits(bytes: &[u8], pos: usize, line_no: u16) -> Result<(u16, usize), AbcError> {
    let start = pos;
    let mut end = pos;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == start {
        return Ok((0, 0));
    }
    let value = parse_u16(&bytes[start..end], line_no)?;
    Ok((value, end - start))
}

/// Apply the active tuplet's `q/p` multiplier to the most recently pushed
/// musical element (Note / Chord / Rest) and decrement `r_remaining`.
fn apply_tuplet(body: &mut [AbcElement], state: &mut Option<TupletState>) {
    let Some(mut ts) = *state else {
        return;
    };
    let (p16, q16) = (u16::from(ts.p), u16::from(ts.q));
    if let Some(last) = body.last_mut() {
        match last {
            AbcElement::Note { num, den, .. }
            | AbcElement::Rest { num, den }
            | AbcElement::Chord { num, den, .. } => {
                *num = num.saturating_mul(q16);
                *den = den.saturating_mul(p16);
            }
            _ => {
                // Non-musical element (e.g. Barline) — no adjustment; also do not
                // decrement r_remaining so the multiplier still applies to the next
                // actual note.
                return;
            }
        }
    }
    ts.r_remaining -= 1;
    *state = if ts.r_remaining == 0 { None } else { Some(ts) };
}

/// Parse a volta marker `[N` starting at `[`. Returns the marker and bytes consumed.
///
/// Phase 2a accepts `[1` and `[2`; other numbers return `UnsupportedVolta`.
fn parse_volta(bytes: &[u8], start: usize, line_no: u16) -> Result<(AbcElement, usize), AbcError> {
    let mut pos = start + 1;
    let digit_start = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos == digit_start {
        return Err(AbcError::UnexpectedChar {
            line: line_no,
            ch: b'[',
        });
    }
    let value = parse_u16(&bytes[digit_start..pos], line_no)?;
    let volta_u8 = u8::try_from(value).map_err(|_| AbcError::UnsupportedVolta)?;
    if volta_u8 == 0 || volta_u8 > 2 {
        return Err(AbcError::UnsupportedVolta);
    }
    Ok((AbcElement::VoltaStart(volta_u8), pos - start))
}

/// Parse a chord `[<notes>]<duration>?-?` starting at `[`. Returns the chord element
/// and total bytes consumed (including any duration modifier and tie suffix).
fn parse_chord(
    bytes: &[u8],
    start: usize,
    line_no: u16,
    key: KeySignature,
) -> Result<(AbcElement, usize), AbcError> {
    let mut pos = start + 1;
    let mut notes = [0u8; MAX_CHORD_NOTES];
    let mut count: u8 = 0;

    while pos < bytes.len() && bytes[pos] != b']' {
        match bytes[pos] {
            b' ' | b'\t' => {
                pos += 1;
            }
            b'^' | b'_' | b'=' | b'A'..=b'G' | b'a'..=b'g' => {
                let (midi, consumed) = parse_note(bytes, pos, line_no, key)?;
                if usize::from(count) >= MAX_CHORD_NOTES {
                    return Err(AbcError::ChordTooLarge);
                }
                notes[usize::from(count)] = midi;
                count += 1;
                pos += consumed;
            }
            ch => {
                return Err(AbcError::UnexpectedChar { line: line_no, ch });
            }
        }
    }

    if pos >= bytes.len() {
        return Err(AbcError::UnbalancedChord);
    }
    pos += 1; // consume `]`

    if count == 0 {
        return Err(AbcError::UnbalancedChord);
    }

    let (num, den, dur_consumed) = parse_duration(bytes, pos, line_no)?;
    pos += dur_consumed;

    let tie_follows = pos < bytes.len() && bytes[pos] == b'-';
    if tie_follows {
        pos += 1;
    }

    Ok((
        AbcElement::Chord {
            notes,
            count,
            num,
            den,
            tie_follows,
        },
        pos - start,
    ))
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
        if let AbcElement::Note { midi, num, den, .. } = notes[6] {
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

    // ---------- Phase 2a tests: chord / repeat / volta / tie / minor keys ----------

    #[test]
    fn phase2a_chord_basic_triad() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n[CEG] |\n";
        let tune = parse(src).expect("parse");
        let chord = tune
            .body
            .iter()
            .find(|e| matches!(e, AbcElement::Chord { .. }))
            .expect("chord");
        if let AbcElement::Chord {
            notes: [n0, n1, n2, ..],
            count,
            num,
            den,
            ..
        } = *chord
        {
            assert_eq!(count, 3);
            assert_eq!([n0, n1, n2], [60u8, 64, 67]); // C E G
            assert_eq!(num, 1);
            assert_eq!(den, 1);
        }
    }

    #[test]
    fn phase2a_chord_with_duration_modifier() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n[CEG]2 |\n";
        let tune = parse(src).expect("parse");
        if let Some(AbcElement::Chord { num, den, .. }) = tune
            .body
            .iter()
            .copied()
            .find(|e| matches!(e, AbcElement::Chord { .. }))
        {
            assert_eq!(num, 2);
            assert_eq!(den, 1);
        }
    }

    #[test]
    fn phase2a_chord_with_individual_accidentals() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n[^Ce_g] |\n";
        let tune = parse(src).expect("parse");
        if let Some(AbcElement::Chord {
            notes: [n0, n1, n2, ..],
            count,
            ..
        }) = tune
            .body
            .iter()
            .copied()
            .find(|e| matches!(e, AbcElement::Chord { .. }))
        {
            assert_eq!(count, 3);
            assert_eq!(n0, 61); // ^C = C#4 = 61
            assert_eq!(n1, 76); // e = E5 = 76
            assert_eq!(n2, 78); // _g = Gb5 = 78
        }
    }

    #[test]
    fn phase2a_chord_too_large_errors() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n[CDEFGABcd] |\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::ChordTooLarge);
    }

    #[test]
    fn phase2a_chord_unbalanced_errors() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n[CEG C\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::UnbalancedChord);
    }

    #[test]
    fn phase2a_chord_score_conversion_emits_stacked_events() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n[CEG] |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);
        // 3 NoteOn (delta 0, 0, 0) + 3 NoteOff (delta 96, 0, 0)
        assert_eq!(score.events.len(), 6);
        assert_eq!(score.events[0].kind, NoteEventKind::NoteOn);
        assert_eq!(score.events[0].note, 60);
        assert_eq!(score.events[0].delta_tick, 0);
        assert_eq!(score.events[1].note, 64);
        assert_eq!(score.events[1].delta_tick, 0);
        assert_eq!(score.events[2].note, 67);
        assert_eq!(score.events[2].delta_tick, 0);
        assert_eq!(score.events[3].kind, NoteEventKind::NoteOff);
        assert_eq!(score.events[3].note, 60);
        assert_eq!(score.events[3].delta_tick, 96);
        assert_eq!(score.events[4].delta_tick, 0);
        assert_eq!(score.events[5].delta_tick, 0);
    }

    #[test]
    fn phase2a_repeat_simple_unroll() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n|: C D :|\n";
        let tune = parse(src).expect("parse");
        // After unroll: C D C D (barlines dropped, markers removed)
        let midis: Vec<u8> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(midis.as_slice(), &[60u8, 62, 60, 62][..]);
        assert!(!tune
            .body
            .iter()
            .any(|e| matches!(e, AbcElement::RepeatStart | AbcElement::RepeatEnd)));
    }

    #[test]
    fn phase2a_repeat_with_voltas() {
        // |: A [1 B :| [2 C |  →  A B A C |
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n|: A [1 B :| [2 c |\n";
        let tune = parse(src).expect("parse");
        let midis: Vec<u8> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        // A = 69, B = 71, c = 72
        assert_eq!(midis.as_slice(), &[69u8, 71, 69, 72][..]);
        assert!(!tune
            .body
            .iter()
            .any(|e| matches!(e, AbcElement::VoltaStart(_))));
    }

    #[test]
    fn phase2a_repeat_nested_errors() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n|: C |: D :| :|\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::NestedRepeat);
    }

    #[test]
    fn phase2a_repeat_unbalanced_errors() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n|: C D |\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::UnbalancedRepeat);
    }

    #[test]
    fn phase2a_volta_number_3_unsupported() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n|: A [1 B :| [3 C |\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::UnsupportedVolta);
    }

    #[test]
    fn phase2a_tie_merges_same_pitch_duration() {
        // Two quarter-notes tied → one half-note (2× ticks).
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC-C |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);
        assert_eq!(score.events.len(), 2); // 1 NoteOn + 1 NoteOff (merged)
        assert_eq!(score.events[0].kind, NoteEventKind::NoteOn);
        assert_eq!(score.events[0].note, 60);
        assert_eq!(score.events[1].kind, NoteEventKind::NoteOff);
        assert_eq!(score.events[1].note, 60);
        assert_eq!(score.events[1].delta_tick, 192); // 96 + 96
    }

    #[test]
    fn phase2a_tie_across_barline() {
        // Quarter-note tied across a barline to another quarter-note.
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC- | C |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);
        assert_eq!(score.events.len(), 2);
        assert_eq!(score.events[1].delta_tick, 192);
    }

    #[test]
    fn phase2a_tie_mismatched_pitch_does_not_merge() {
        // A `-` between different pitches: from-side note still records the tie flag
        // but coalescing must not proceed since MIDI numbers differ.
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC-D |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);
        assert_eq!(score.events.len(), 4); // 2 NoteOn + 2 NoteOff (no merge)
        assert_eq!(score.events[0].note, 60);
        assert_eq!(score.events[2].note, 62);
    }

    #[test]
    fn phase2a_tie_chain_of_three_notes() {
        // C-C-C — three quarter notes fused into one 3× duration.
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC-C-C |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);
        assert_eq!(score.events.len(), 2);
        assert_eq!(score.events[1].delta_tick, 288); // 3 × 96
    }

    #[test]
    fn phase2a_minor_key_am_zero_sharps() {
        let src = "X:1\nM:4/4\nL:1/4\nK:Am\nC |\n";
        let tune = parse(src).expect("parse");
        assert_eq!(tune.header.key.sharps, 0);
    }

    #[test]
    fn phase2a_minor_key_bm_two_sharps_via_amin_synonym() {
        let src1 = "X:1\nM:4/4\nL:1/4\nK:Bm\nC |\n";
        let src2 = "X:1\nM:4/4\nL:1/4\nK:Bmin\nC |\n";
        let t1 = parse(src1).expect("Bm");
        let t2 = parse(src2).expect("Bmin");
        assert_eq!(t1.header.key.sharps, 2);
        assert_eq!(t2.header.key.sharps, 2);
    }

    #[test]
    fn phase2a_minor_key_dm_flattens_b() {
        // D minor has 1 flat (Bb). Playing bare `B` in K:Dm must yield Bb (MIDI 70).
        let src = "X:1\nM:4/4\nL:1/4\nK:Dm\nB |\n";
        let tune = parse(src).expect("parse");
        if let Some(AbcElement::Note { midi, .. }) = tune
            .body
            .iter()
            .copied()
            .find(|e| matches!(e, AbcElement::Note { .. }))
        {
            assert_eq!(midi, 70); // Bb4
        } else {
            panic!("expected a Note");
        }
    }

    #[test]
    fn phase2a_all_15_minor_keys_parse() {
        let fixtures: &[&str] = &[
            "X:1\nM:4/4\nL:1/4\nK:Am\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Em\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Bm\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:F#m\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:C#m\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:G#m\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:D#m\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:A#m\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Dm\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Gm\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Cm\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Fm\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Bbm\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Ebm\nC |\n",
            "X:1\nM:4/4\nL:1/4\nK:Abm\nC |\n",
        ];
        for src in fixtures {
            assert!(
                parse(src).is_ok(),
                "minor key fixture should parse: {src:?}"
            );
        }
    }

    // ---------- Phase 2b tests: church modes / tuplets / grace / multi-voice ----------

    #[test]
    fn phase2b_mode_dorian_shifts_sharps() {
        // A Dorian = A major (3 sharps) - 2 = 1 sharp
        let src = "X:1\nM:4/4\nL:1/4\nK:Ador\nC |\n";
        let tune = parse(src).expect("parse");
        assert_eq!(tune.header.key.sharps, 1);
    }

    #[test]
    fn phase2b_mode_mixolydian_shifts_sharps() {
        // G Mixolydian = G major (1 sharp) - 1 = 0 sharps
        let src = "X:1\nM:4/4\nL:1/4\nK:Gmix\nC |\n";
        let tune = parse(src).expect("parse");
        assert_eq!(tune.header.key.sharps, 0);
    }

    #[test]
    fn phase2b_mode_lydian_shifts_sharps() {
        // F Lydian = F major (-1) + 1 = 0 sharps
        let src = "X:1\nM:4/4\nL:1/4\nK:Flyd\nC |\n";
        let tune = parse(src).expect("parse");
        assert_eq!(tune.header.key.sharps, 0);
    }

    #[test]
    fn phase2b_mode_phrygian_shifts_sharps() {
        // E Phrygian = E major (4) - 4 = 0 sharps
        let src = "X:1\nM:4/4\nL:1/4\nK:Ephr\nC |\n";
        let tune = parse(src).expect("parse");
        assert_eq!(tune.header.key.sharps, 0);
    }

    #[test]
    fn phase2b_mode_locrian_shifts_sharps() {
        // B Locrian = B major (5) - 5 = 0 sharps
        let src = "X:1\nM:4/4\nL:1/4\nK:Bloc\nC |\n";
        let tune = parse(src).expect("parse");
        assert_eq!(tune.header.key.sharps, 0);
    }

    #[test]
    fn phase2b_mode_ionian_synonym_equals_major() {
        let src_ion = "X:1\nM:4/4\nL:1/4\nK:Cion\nC |\n";
        let src_maj = "X:1\nM:4/4\nL:1/4\nK:C\nC |\n";
        assert_eq!(
            parse(src_ion).expect("Cion").header.key.sharps,
            parse(src_maj).expect("C").header.key.sharps
        );
    }

    #[test]
    fn phase2b_mode_out_of_range_rejected() {
        // G# major = 8 sharps; G# Lydian = 9 → out of range.
        let src = "X:1\nM:4/4\nL:1/4\nK:G#lyd\nC |\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::InvalidKey);
    }

    #[test]
    fn phase2b_triplet_shortens_notes() {
        // (3 C C C — three quarter notes take the space of 2 → each note = 2/3 L
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n(3 C C C |\n";
        let tune = parse(src).expect("parse");
        let pairs: Vec<(u16, u16)> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { num, den, .. } => Some((*num, *den)),
                _ => None,
            })
            .collect();
        assert_eq!(pairs.as_slice(), &[(2, 3), (2, 3), (2, 3)][..]);
    }

    #[test]
    fn phase2b_duplet_stretches_notes() {
        // (2 C C — two notes take the space of 3 → each = 3/2 L
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n(2 C C |\n";
        let tune = parse(src).expect("parse");
        let pairs: Vec<(u16, u16)> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { num, den, .. } => Some((*num, *den)),
                _ => None,
            })
            .collect();
        assert_eq!(pairs.as_slice(), &[(3, 2), (3, 2)][..]);
    }

    #[test]
    fn phase2b_tuplet_only_applies_to_r_notes() {
        // (3 in a row = 3 notes; the fourth is back to normal.
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n(3 C C C D |\n";
        let tune = parse(src).expect("parse");
        let pairs: Vec<(u16, u16)> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { num, den, .. } => Some((*num, *den)),
                _ => None,
            })
            .collect();
        assert_eq!(pairs.as_slice(), &[(2, 3), (2, 3), (2, 3), (1, 1)][..]);
    }

    #[test]
    fn phase2b_tuplet_full_form_p_q_r() {
        // (5:2:3 — 5-in-2 ratio applied to only 3 notes (uncommon but supported)
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n(5:2:3 C C C D E |\n";
        let tune = parse(src).expect("parse");
        let pairs: Vec<(u16, u16)> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { num, den, .. } => Some((*num, *den)),
                _ => None,
            })
            .collect();
        // First 3 notes get (2, 5) multiplier, last 2 are normal
        assert_eq!(
            pairs.as_slice(),
            &[(2, 5), (2, 5), (2, 5), (1, 1), (1, 1)][..]
        );
    }

    #[test]
    fn phase2b_tuplet_out_of_range_errors() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n(1 C |\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::UnsupportedTuplet);
    }

    #[test]
    fn phase2b_grace_notes_emit_short_notes() {
        // {ab}c — two grace notes (1/32 each) prepended to c.
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n{ab}c |\n";
        let tune = parse(src).expect("parse");
        let notes: Vec<(u8, u16, u16)> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, num, den, .. } => Some((*midi, *num, *den)),
                _ => None,
            })
            .collect();
        // a=69+12=81 (a is lowercase = C5 base), b=83, c=72.
        // Wait: lowercase a = A5 (MIDI 81), lowercase b = B5 (MIDI 83), lowercase c = C5 (MIDI 72).
        assert_eq!(notes.len(), 3);
        assert_eq!(notes[0], (81, 1, 32));
        assert_eq!(notes[1], (83, 1, 32));
        assert_eq!(notes[2], (72, 1, 1));
    }

    #[test]
    fn phase2b_grace_empty_group_is_allowed() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\n{}c |\n";
        let tune = parse(src).expect("parse");
        let note_count = tune
            .body
            .iter()
            .filter(|e| matches!(e, AbcElement::Note { .. }))
            .count();
        assert_eq!(note_count, 1);
    }

    #[test]
    fn phase2b_multi_voice_two_voices_populated() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nV:1\nC D |\nV:2\nc, d, |\n";
        let tune = parse(src).expect("parse");
        assert_eq!(tune.extra_voices.len(), 1);
        let v0_notes: Vec<u8> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        let v1_notes: Vec<u8> = tune.extra_voices[0]
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(v0_notes.as_slice(), &[60u8, 62][..]); // C4, D4
        assert_eq!(v1_notes.as_slice(), &[60u8, 62][..]); // c, and d, drop back to octave 4
    }

    #[test]
    fn phase2b_multi_voice_channels_in_score() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nV:1\nC |\nV:2\nG |\n";
        let tune = parse(src).expect("parse");
        let score = tune.to_score(96);
        assert_eq!(score.header.tracks, 2);
        // Both voices start at absolute tick 0, so events on both channels are
        // emitted at delta 0 (stable sort keeps voice-0 events first).
        let v0_on = score
            .events
            .iter()
            .find(|e| e.kind == NoteEventKind::NoteOn && e.channel == 0)
            .expect("voice 0 NoteOn");
        let v1_on = score
            .events
            .iter()
            .find(|e| e.kind == NoteEventKind::NoteOn && e.channel == 1)
            .expect("voice 1 NoteOn");
        assert_eq!(v0_on.note, 60);
        assert_eq!(v1_on.note, 67);
    }

    #[test]
    fn phase2b_multi_voice_invalid_number_errors() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nV:0\nC |\n";
        assert_eq!(parse(src).unwrap_err(), AbcError::InvalidVoice);
    }

    #[test]
    fn phase2b_multi_voice_switch_mid_tune() {
        // Voice 1 plays C, switch to V:2, play E, switch back to V:1, play D.
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nV:1\nC |\nV:2\nE |\nV:1\nD |\n";
        let tune = parse(src).expect("parse");
        let v0_notes: Vec<u8> = tune
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(v0_notes.as_slice(), &[60u8, 62][..]);
    }

    #[test]
    fn phase2b_single_voice_default_unchanged() {
        // Regression: single-voice tunes must produce identical output to Phase 2a.
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC D |\n";
        let tune = parse(src).expect("parse");
        assert!(tune.extra_voices.is_empty());
        let score = tune.to_score(96);
        assert_eq!(score.header.tracks, 1);
        assert_eq!(score.events[0].delta_tick, 0);
        assert_eq!(score.events[0].note, 60);
        assert_eq!(score.events[1].delta_tick, 96);
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
            extra_voices: Vec::new(),
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
            extra_voices: Vec::new(),
        };
        assert_eq!(tune8.unit_ticks(96), 48); // eighth = half of quarter
    }
}
