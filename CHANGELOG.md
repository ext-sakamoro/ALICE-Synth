# Changelog

All notable changes to ALICE-Synth will be documented in this file.

## [0.2.0-dev] - 2026-09-12

### Added — Phase 1 (initial ABC parser)
- `abc` module — ABC notation parser for YuE2-style symbolic planning (`feature = "abc"`)
  - Parses header fields `M:` `L:` `Q:` `K:` and body notes / accidentals / durations / rests / barlines
  - 15 canonical major key signatures (C, G, D, A, E, B, F#, C#, F, Bb, Eb, Ab, Db, Gb, Cb)
  - Rational duration storage (`num`/`den` × unit length) — one parse renders at any tick division
  - `AbcTune::to_score()` converts to native `Score` for playback via existing `Synthesizer`
  - `no_std + alloc` compatible, zero extra dependencies
  - 26 unit tests including Ode-to-Joy fixture + end-to-end PCM synthesis via `Synthesizer`
- Module doc reference to YuE2 (Multimodal Art Projection × HKUST, 2026-09-10) as the canonical prior art

### Added — Phase 2a (chord / repeat / tie / minor keys)
- `MAX_CHORD_NOTES = 8` public constant + `AbcElement::Chord { notes: [u8; 8], count, num, den, tie_follows }`
  variant — polyphonic chord support with per-note accidentals (`[^Ce_g]`) and duration modifier (`[CEG]2`)
- `AbcElement::Note.tie_follows: bool` field for asymmetric tie tracking; consecutive same-pitch tied
  notes are coalesced into a single `NoteOn`/`NoteOff` pair at score conversion time
- `AbcElement::{RepeatStart, RepeatEnd, VoltaStart(u8)}` markers, consumed by parse-time
  `unroll_repeats()` post-processing
- Repeat unrolling: `|: X :|` → `X X`, with volta support `|: A [1 B :| [2 C |` → `A B A C`
- 15 canonical minor key signatures (Am / Em / Bm / F#m / C#m / G#m / D#m / A#m / Dm / Gm / Cm / Fm /
  Bbm / Ebm / Abm; both `m` and `min` suffixes accepted; total 30 key signatures)
- 5 new `AbcError` variants: `ChordTooLarge`, `UnbalancedChord`, `NestedRepeat`,
  `UnbalancedRepeat`, `UnsupportedVolta`
- 19 additional unit tests (chord parsing / score conversion / repeat unroll / volta / tie chain /
  minor keys); **total 45 abc tests, 145 crate tests**

### Changed — Phase 2a (breaking, pre-1.0)
- `AbcElement::Note` gained a required `tie_follows: bool` field. Pattern matches must add either
  `tie_follows` or `..`. Documented in `docs/ROADMAP.md` ADR-005.

### Added — Phase 2b (church modes / tuplets / grace notes / multi-voice)
- **Church modes** — `parse_key` refactored to a tonic + mode-suffix computed table. Dor/Mix/Lyd/Phr/Loc
  + Ion/Aeo synonyms accepted; sharp counts derived as `tonic_major_sharps + mode_offset`. Out-of-range
  combinations return `AbcError::InvalidKey`.
- **Tuplets** — `(P`, `(P:Q`, `(P:Q:R` forms parsed. Duration multipliers `q/p` applied to the next `r`
  Note/Chord/Rest elements at parse time; bare `(P` uses ABC 2.1 default `q` values. New public struct
  `TupletState`; new error `AbcError::UnsupportedTuplet`.
- **Grace notes** — `{gab}c` groups parsed. Each inner note is emitted as a `1/32`-duration `Note`
  before the following main note; rests / chords inside grace groups are not supported (produce
  `UnexpectedChar`); grace notes do not consume tuplet slots.
- **Multi-voice** — `AbcTune` gained `extra_voices: Vec<Vec<AbcElement>>`; voice 0 stays in `body`.
  `V:N` field lines switch the active voice both in the header block and between body lines. Voice `N`
  renders to `channel = N-1` in the emitted `Score`; `Score.header.tracks` reflects the voice count.
  New error `AbcError::InvalidVoice`.
- `to_score` refactored to an absolute-tick merge pipeline (`render_voice_to_absolute` → stable sort →
  delta conversion) to enable correct cross-voice event ordering. Single-voice output is unchanged.
- 19 new unit tests (`phase2b_*`); **total 65 abc tests, 164 crate tests**. Clippy pedantic clean.

### Changed — Phase 2b (additive; pattern-match compatible)
- `AbcTune` gained the `extra_voices` field. Struct literals must include it (single-voice case:
  `extra_voices: Vec::new()`). Existing pattern matches using `AbcTune { header, body, .. }` remain
  unaffected. Documented in `docs/ROADMAP.md` ADR-007.

### Added — Phase 2c (chord tie / volta ≥ 3 / repeat shorthand)
- **Chord tie coalescing** — `render_voice_to_absolute` refactored to a `tied_in` state machine
  that unifies Note-tie and Chord-tie handling. Per-note ties across chords with mismatching
  pitch sets are supported: `[CEG]-[CEF]` ties C and E across chord boundaries while G ends
  normally and F starts fresh at the second onset. `[CEG]-C` (chord tying into a single note)
  and `[CEG]-z[CEG]` (rest breaking the chord tie) are also handled. Dangling ties at
  end-of-body emit a final `NoteOff` to keep the event stream well-formed.
- **Voltas `[1`..`[4`** — `unroll_repeats` refactored to a two-phase state machine
  (`extract_repeat_block` + iteration expansion). Any number of voltas from 1 to 4 supported;
  iteration count is `max(2, num_voltas_defined)`. Missing volta indices (e.g. `[1]` + `[3]`
  without `[2]`) leave the corresponding iteration to play the common section only.
  Volta numbers ≥ 5 continue to return `AbcError::UnsupportedVolta`.
- **Repeat shorthands** — `::` parsed as `RepeatEnd + RepeatStart` (end-then-start).
  `|:|` parsed as `Barline + RepeatStart` (barline immediately followed by a new repeat).
- 12 new unit tests (`phase2c_*`); **total 76 abc tests, 176 crate tests**. Clippy pedantic clean.

### Changed — Phase 2c (breaking, pre-1.0)
- **Volta acceptance range** raised from `[1..=2]` to `[1..=4]`. The Phase 2a test
  `phase2a_volta_number_3_unsupported` was renamed to `phase2a_volta_number_5_unsupported` to
  reflect the new boundary. Callers depending on the old rejection at `[3]` must update.
- Test `phase2c_volta_five_still_unsupported` documents the new upper bound.

### Refactored — Phase 2c (behaviourally identical)
- `render_voice_to_absolute` moved from an ad-hoc tie-chain loop to a unified `tied_in` state
  machine. All Phase 2a Note-tie tests still pass unchanged; the new mechanism now also handles
  chord-level ties. Documented in `docs/ROADMAP.md` ADR-009.

### Added — Phase 3 (ALICE 三相原理 Intent packet + procedural synthesizer)
- **`intent` feature** (`feature = "intent"`, implies `abc`) — new module `src/intent.rs`
  (~700 LoC + 22 tests).
- **`MusicIntent` — 8-byte packet** representing a musical intent: genre (u8) + mood (u8) +
  length_bars (u8) + tempo_bpm_offset (u8) + key (u8, tonic index 0..=14) + mode (u8) +
  variation_seed (u16 little-endian). `to_bytes` / `from_bytes` roundtrip is exact.
- **Canonical constants** in submodules `genre::` (Folk / Jazz / Blues / Classical / Pop /
  Ambient / Rock / Lullaby / Cinematic / Electronic), `mood::` (Happy / Sad / Tense / Serene /
  Energetic / Melancholy / Mysterious / Triumphant), `mode::` (Ionian / Dorian / Phrygian /
  Lydian / Mixolydian / Aeolian / Locrian, plus `MAJOR` / `MINOR` aliases).
- **`MusicIntent::synthesize()` → `AbcTune`** deterministic procedural generator: mode scale
  × mood-weighted degree bias × LCG PRNG seeded by `variation_seed`. Serves as the canonical
  stand-in for a future LLM-driven plan head (ADR-011); same packet → same tune, always.
- **`MusicIntent::from_tune()`** — best-effort inverse: extracts key (Ionian assumed, falls
  back to Aeolian), tempo, and length_bars; derives a stable `variation_seed` via FNV-1a hash
  of the body's note pitches. Genre / mood default to Folk / Happy (categorical labels do not
  survive the ABC round trip).
- **`MusicIntent::DEFAULT_C_MAJOR`** — a neutral starting point (C-major Folk / Happy at 120
  BPM, 4 bars, seed `0xC0DE`).
- ALICE 三相原理 realized: instead of shipping ABC score text (Phase 2, Law), a caller ships
  8 bytes and the receiver reconstructs the tune locally.
- 22 unit tests + 1 doctest; **total 198 crate tests**. Clippy pedantic clean, fmt clean,
  no_std + alloc build passes.

### Documented — Phase 3
- ADR-010 (docs/ROADMAP.md): Ambiguous key signature reverse-lookup in `from_tune` prefers
  Ionian over relative Aeolian for aesthetic simplicity.
- ADR-011 (docs/ROADMAP.md): Procedural synthesizer is the canonical MVP; a future
  LLM-driven plan head can slot in as an alternate `synthesize` implementation without
  changing the `MusicIntent` wire format.

### Added — Phase 3.1 (cross-crate integration + local editing / cover APIs)
- **`PlanHead` trait + `ProceduralPlanHead`** in `intent` module — canonical extension point
  for LLM-driven `synthesize` alternatives. Object-safe so downstream crates can swap in a
  `Box<dyn PlanHead>` at runtime. `ProceduralPlanHead` delegates to `MusicIntent::synthesize`.
- **`agentic` feature** (`feature = "agentic"`, implies `abc`) — new module `src/agentic.rs`
  (~330 LoC + 13 tests) with `AbcDiff::compute` / `AbcDiff::apply` and
  `AbcTune::diff` / `AbcTune::patch` convenience methods. Position-based diff (per-index
  additions / removals / replacements + optional header overrides for tempo and key).
  Preserves `extra_voices` verbatim through round trips.
- **`cover` feature** (`feature = "cover"`, implies `intent`) — new module `src/cover.rs`
  (~290 LoC + 6 tests) with `Transcriber` trait, `TranscribeError` (Copy enum), and
  `CoverPipeline<T, P>`. Zero-shot cover pipeline: audio → transcribe → extract
  `MusicIntent` → apply target genre/mood/mode → re-synthesize. Concrete transcribers live
  in downstream crates (external ML model wrappers).
- Cross-crate contract: ALICE-LOL side (commit `c41de9b` on the ALICE-LOL repo) added
  `IntentNode::Music { packet: [u8; 8] }` and `music_intent()` constructor. LOL programs can
  now carry `MusicIntent` packets alongside physical verbs (grasp / walk / etc.). No
  dependency edge added — both sides speak the 8-byte contract without a shared crate.
- 22 new unit tests (2 PlanHead + 13 agentic + 6 cover + 1 dyn-object) + 2 new doctests;
  **total 220 crate tests, 6 doctests**. Clippy pedantic clean, fmt clean, no_std + alloc
  build passes.

### Documented — Phase 3.1
- ADR-012 (docs/ROADMAP.md): Cross-crate MusicIntent protocol — the 8-byte packet is the
  single source of truth. ALICE-LOL wraps it in `IntentNode::Music` with no dep on
  ALICE-Synth; ALICE-Synth defines the canonical layout. Bidirectional interop through the
  8 bytes only.
- ADR-013 (docs/ROADMAP.md): `AbcDiff` is intentionally position-based rather than LCS-based.
  Handles the common LLM-revision case (localized in-place edits) with minimal complexity;
  a smarter algorithm can slot in later without changing the public surface.
- ADR-014 (docs/ROADMAP.md): `Transcriber` trait lives in ALICE-Synth but has no concrete
  implementations here. Keeps this crate free of ML dependencies while giving downstream
  transcribers a stable interface to target.

### Added — Phase 3.2 (LCS-based diff — shift-aware editing)
- **`AbcDiff::compute_lcs`** — new LCS (Longest Common Subsequence) backend for the
  agentic diff. Classic `O(N × M)` DP table with backtracking, emitting insert/remove
  edits only (no `body_replacements`). Detects element shifts as coherent single-edit
  operations rather than as chains of positional replacements.
- **`AbcTune::diff_lcs`** — convenience method paired with the existing `AbcTune::diff`.
- 12 new unit tests demonstrating shift detection (prepend, splice, deletion), edge
  cases (empty original / revised), and confirming header-change behaviour matches the
  positional backend. **Total 232 crate tests.**
- Position-based backend (`AbcDiff::compute`, `AbcTune::diff`) unchanged — callers keep
  choosing the trade-off (fast + coarse vs. thorough + shift-aware).

### Documented — Phase 3.2
- ADR-015 (docs/ROADMAP.md): `compute_lcs` co-exists with `compute` rather than replacing
  it. Callers pick per revision profile — small edits → positional (`O(N)`), structural
  edits → LCS (`O(N × M)`). Removes the "LCS is Phase 4 candidate" note from ADR-013.

## [0.1.1] - 2026-03-04

### Added
- `ffi` module — 20 `extern "C"` functions for C/C++/C# interop (`feature = "ffi"`)
- `python` module — PyO3 bindings: 4 classes + 2 functions (`feature = "python"`)
- Unity C# bindings (`bindings/unity/AliceSynth.cs`) — `IDisposable` wrappers
- UE5 C++ header (`bindings/ue5/AliceSynth.h`) — RAII `FSynthesizer` / `FOscillator`
- 19 FFI tests, PyO3 compilation check
- `#[repr(C)]` on `Adsr` struct for FFI safety

## [0.1.0] - 2026-02-23

### Added
- `Oscillator` — phase-accumulator generators: sine, saw, square, triangle, noise, wavetable
- `Adsr` / `AdsrState` — sample-accurate ADSR envelope (16-byte config)
- `Patch` — FM (32 B), Additive (64 B), Subtractive (24 B), Wavetable (256 B)
- `Score` / `NoteEvent` — compact score format (8-byte header + 4-byte events)
- `Synthesizer` — 64-voice polyphonic engine with 16-channel score playback
- `Delay`, `LowPassFilter`, `StateVariableFilter`, `Reverb` — audio effects
- `no_std` compatible with `alloc` fallback
- 100 unit tests covering all modules
