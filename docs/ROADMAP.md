# ALICE-Synth Roadmap

**Tagline**: *Procedural Audio Synthesizer — Don't send waveforms, send the score.*

## 現在位置

- **Version**: 0.2.0-dev (unreleased)
- **Branch**: main
- **Latest commit**: pending — ABC parser (Phase 1 MVP) 統合
- **License**: MIT OR Apache-2.0 (dual)

## Phase 一覧

### ✅ Phase 0 — Core synthesis primitives (v0.1.0, 2026-02-23)

- `oscillator` phase-accumulator generators (sine / saw / square / triangle / noise / wavetable)
- `envelope` ADSR (16-byte config, sample-accurate)
- `patch` instrument variants — FM (32 B) / Additive (64 B) / Subtractive (24 B) / Wavetable (256 B)
- `score` compact binary format — 8-byte header + 4-byte note events
- `synth` 64-voice / 16-channel polyphonic engine
- `effects` delay / low-pass / state-variable filter / reverb
- `no_std + alloc` compatible; 100 unit tests

### ✅ Phase 1a — FFI + Python bindings (v0.1.1, 2026-03-04)

- `ffi` 20 `extern "C"` functions for C / C++ / C# interop
- `python` PyO3 bindings — 4 classes + 2 functions
- Unity C# `bindings/unity/AliceSynth.cs`
- UE5 C++ `bindings/ue5/AliceSynth.h`

### ✅ Phase 1b — ABC notation parser (v0.2.0-dev, 2026-09-12)

**Motivation**: YuE2 (M-A-P × HKUST, 2026-09-10) established ABC notation as the canonical
human-editable intermediate representation for symbolic music planning. ALICE-Synth's founding
tagline *"send the score, not the waveform"* is philosophically identical to YuE2's
Symbolic Planning stage; adopting the same input surface closes the gap.

**Scope (MVP)**:

- Header fields: `X:` `T:` `M:` `L:` `Q:` `K:` (only `M`/`L`/`Q`/`K` affect audio)
- Body: `A-G a-g` note letters, `,`/`'` octave modifiers, `^`/`_`/`=`/`^^`/`__` accidentals
- Durations: rational `num/den` × unit length (`N` `/N` `/` `N/M` forms)
- Rest: `z` `Z` with duration modifier
- Barlines: `|` `||` `|]` `[|` `:|` `|:` (all as boundary markers)
- 15 canonical major key signatures
- Inline field `[K:...]` skip, decorations `!...!` skip, annotations `"..."` skip, `%` comments

**Non-goals (defer to Phase 2)**: multi-voice `V:`, chord expansion, ties, grace notes, tuplets,
minor keys, mode names.

**Deliverables**:

- `src/abc.rs` — parser + `AbcTune` + `AbcTune::to_score(tick_div)` conversion
- Feature gate `abc = []` (opt-in, `no_std + alloc` compatible, zero extra deps)
- 26 unit tests including "Ode to Joy" fixture + end-to-end PCM synthesis via `Synthesizer`

**DoD**:

- `cargo build --all-features` = pass
- `cargo test --features abc,std` = all tests pass (existing 100 + new 22)
- `cargo clippy --all-features --all-targets -- -D warnings` = warnings 0
- `cargo fmt --check` = OK
- 仮実装 grep (`todo!|unimplemented!|panic!.*stub|TODO|mock|dummy|placeholder`) = 0 hits in `src/abc.rs`
- End-to-end verified: ABC → Score → PCM via `Synthesizer` (existing pipeline)

### ✅ Phase 2a — Chord / repeat / tie / minor keys (v0.2.0-dev, 2026-09-12)

Landed in the same v0.2.0-dev milestone as Phase 1b (single commit follow-up).

- **Chord expansion `[CEG]`** — `AbcElement::Chord { notes: [u8; MAX_CHORD_NOTES=8], count, num, den, tie_follows }`;
  supports per-note accidentals (`[^Ce_g]`) and duration modifier (`[CEG]2`); score conversion
  emits N `NoteOn` (first with pending delta, rest with 0) + N `NoteOff` (first with chord
  duration, rest with 0).
- **Repeats `|: :|`** — parse-time `unroll_repeats()` post-processes markers into linear body;
  `|: X :|` → `X X`. Nested repeats return `AbcError::NestedRepeat`; unbalanced return
  `AbcError::UnbalancedRepeat`.
- **Voltas `[1` `[2`** — `|: A [1 B :| [2 C |` → `A B A C`. Numbers ≥ 3 return
  `AbcError::UnsupportedVolta`.
- **Ties `-`** — `Note.tie_follows: bool` (**breaking pattern-match change**, see ADR-005);
  consecutive same-pitch tied notes coalesce into one sustained `NoteOn`/`NoteOff` pair at
  score-conversion time (ticks summed, so cross-barline ties work).
- **30 canonical key signatures** — added 15 relative minors (Am/Em/…/Abm) accepting both `m`
  and `min` suffixes; sharp counts computed as `relative_major − 3` per canonical theory.
- 19 new tests (`phase2a_*`); total 45 abc tests / 145 crate tests. Clippy pedantic clean, fmt
  clean, doctest passes, no_std + alloc build passes.

### ✅ Phase 2b — Church modes / tuplets / grace notes / multi-voice (v0.2.0-dev, 2026-09-13)

- **Church modes** — `parse_key` refactored to compute `sharps = tonic_major_sharps + mode_offset`.
  Dor / Mix / Lyd / Phr / Loc + Ion / Aeo synonyms accepted (long and short forms). Out-of-range
  results return `AbcError::InvalidKey`.
- **Tuplets** — `(P`, `(P:Q`, `(P:Q:R` forms parsed. Duration multipliers `q/p` applied to the next
  `r` Note/Chord/Rest elements at parse time. New public struct `TupletState`; new error
  `AbcError::UnsupportedTuplet`.
- **Grace notes** — `{gab}c` groups expanded to `1/32`-duration `Note` events prepended to the main
  note. Rests / chords inside grace groups are not supported. Grace notes do not consume tuplet
  slots.
- **Multi-voice `V:`** — `AbcTune` gained `extra_voices: Vec<Vec<AbcElement>>`. `V:N` field lines
  switch the active voice both in the header block and between body lines. Voice N renders to
  channel `N-1`; `to_score` refactored to an absolute-tick merge pipeline for correct cross-voice
  ordering. New error `AbcError::InvalidVoice`.
- 19 new tests (`phase2b_*`); total 65 abc tests / 164 crate tests. Clippy pedantic clean, fmt
  clean, doctest passes, no_std + alloc build passes, stub grep clean.

### ✅ Phase 2c — Chord tie / volta ≥ 3 / repeat shorthand (v0.2.0-dev, 2026-09-13)

- **Chord tie coalescing** — `render_voice_to_absolute` refactored to a `tied_in` state machine.
  Per-note ties across chords with mismatching pitch sets: `[CEG]-[CEF]` → C and E extend, G ends
  at first-chord duration, F starts fresh at second-chord onset. Rest breaks the chord tie.
  Dangling ties at end-of-body emit final `NoteOff`s for well-formed event streams.
- **Voltas `[1`..`[4`** — `unroll_repeats` refactored to two-phase state machine
  (`extract_repeat_block` + iteration expansion). Iteration count = `max(2, num_voltas)`;
  missing indices leave that iteration to play the common section only. `[5]` still rejected.
- **Repeat shorthand** — `::` = `RepeatEnd + RepeatStart` (end-then-start); `|:|` =
  `Barline + RepeatStart`.
- 12 new tests (`phase2c_*`); total 76 abc tests / 176 crate tests. Clippy pedantic clean.

### ⏳ Phase 3 — Symbolic Intent DSL integration

- `AbcTune` ↔ ALICE-LOL Music IR round-trip (via `alice-compiler` AST)
- 8-byte Intent packet compression (ALICE 三相原理 Phase 3): "genre + mood + length" → server-side
  ABC synthesis (LLM plan head, delegated to ALICE-LLM)
- Zero-shot cover pipeline: audio → SheetSage2-style transcription → AbcTune → style rewrite
- Agentic editing: `AbcTune` diff / patch API for LLM-driven revision loops

### ⏳ Phase 4 — Audio quality parity with modern models

- Anti-aliased BLIT-based oscillators (vs current phase-accumulator basic waveforms)
- Physical modelling (Karplus-Strong / waveguide) for plucked / bowed / blown instruments
- Convolution reverb with impulse response library
- 48 kHz stereo output path (parity with YuE2 output spec)
- Optional `flow-matching` feature gate: neural residual atop procedural base (long-term R&D)

## Open Questions

- **OQ-1**: Should `AbcTune::to_score()` preserve barline positions as `ControlChange` events for
  downstream loop / section handling, or continue dropping them silently?
- **OQ-2**: Chord expansion (Phase 2) — expand to simultaneous NoteOn at delta=0 (current
  design) or introduce a dedicated `Chord` variant to preserve semantic grouping through the
  Score binary format?
- **OQ-3**: Feature gating strategy — keep `abc = []` (no deps) or split into `abc-core` (parser
  only, no `Score` conversion) + `abc-render` (with Score conversion, requires `std` for future
  file I/O)?
- **OQ-4**: License positioning — should the ABC parser be dual-licensed (MIT OR Apache-2.0)
  matching the rest of the crate, or contributed upstream to `abc-notation` community crates?

## 判断記録 (ADR)

### ADR-001 — ABC notation adoption over MusicXML / MIDI as primary text IR (2026-09-12)

**Context**: YuE2 established ABC as the industry-standard human-editable intermediate for open
music generation. Alternatives considered: MusicXML (verbose, XML-heavy, poor human-editability),
MIDI (binary, no text edit path), custom DSL (invents-new-standard risk).

**Decision**: Adopt ABC notation as the canonical text-based score input for ALICE-Synth.
MIDI import remains a separate `midi` feature. MusicXML is out of scope indefinitely.

**Consequences**: (+) Zero-cost adoption of a proven, widely-taught notation; interoperability
with abcnotation.com, EasyABC, Sibelius, MuseScore; alignment with YuE2 for future integration.
(−) MVP scope must explicitly document what's not supported (multi-voice / chord / ties etc.)
to prevent user confusion when their ABC file "partially works".

### ADR-002 — Rational duration storage instead of pre-computed ticks (2026-09-12)

**Context**: Two options for storing note durations in `AbcElement::Note`:

1. Store `ticks: u16` computed at parse time from a canonical `tick_div`.
2. Store `(num: u16, den: u16)` and compute ticks in `to_score(tick_div)`.

**Decision**: Option 2 (rational storage).

**Consequences**: (+) A single `AbcTune` can be rendered at any tick division without precision
loss or rescaling artifacts; (+) `to_score(48)` and `to_score(96)` produce mathematically
identical timings (just scaled). (−) `AbcElement::Note` grows from 3 bytes (u8 + u16) to 5 bytes
(u8 + u16 + u16), still `Copy` and cache-line friendly.

### ADR-003 — Feature gate `abc = []` with no extra dependencies (2026-09-12)

**Context**: The ABC parser needs `Vec<T>` (already available via alloc in existing `score`
module) but no other alloc types (no `String` in production paths — errors are `Copy` enums,
no heap-allocated error messages).

**Decision**: Keep `abc = []` with zero external deps. The feature is `no_std + alloc`
compatible and does not pull in `std`.

**Consequences**: (+) Preserves ALICE-Synth's embedded-friendly posture; no bloat when the
feature is disabled. (−) Error messages carry no dynamic context (only `line: u16` + `ch: u8`);
callers who want richer diagnostics must lift `AbcError` into their own std-flavored types.

### ADR-005 — `Note.tie_follows: bool` field is a breaking pattern-match change (2026-09-12)

**Context**: Phase 2a needs to represent tied notes. Two options:

1. Add a required field `tie_follows: bool` to `AbcElement::Note`.
2. Introduce a wrapper variant `AbcElement::TiedNote(Note)` that composes.

**Decision**: Option 1 (required field).

**Consequences**: (+) Direct — score conversion just reads the flag. (+) Keeps `AbcElement`
enum flat, no double-indirection when scanning. (−) **Breaking pre-1.0**: existing code that
destructured `AbcElement::Note { midi, num, den }` without `..` will fail to compile. Callers
must add either the new field or `..`. Since the crate is pre-1.0 and no external consumers
exist yet, we accept the churn. The alternative wrapper variant would also break patterns
(a wrapper `TiedNote(Note)` requires matching on both variants), so the churn is unavoidable.

### ADR-006 — Repeat unrolling at parse time, not at `to_score()` (2026-09-12)

**Context**: Repeats can be expanded either at parse time (fold `|: :|` into a linear body) or
at score conversion (stateful stack walking during iteration).

**Decision**: Unroll at parse time via `unroll_repeats()`.

**Consequences**: (+) `to_score()` stays linear and stateless w.r.t. repeats. (+) `AbcTune.body`
is a single source of truth of the played sequence; downstream tooling (visualization, export)
sees the same order the synth does. (−) Repeat structure is lost; round-tripping the body back
to ABC text would produce an unrolled version. Acceptable for MVP — round-trip export is not a
Phase 2 goal.

### ADR-007 — Multi-voice as `extra_voices` field, not `voices: Vec<Vec<>>` (2026-09-13)

**Context**: Multi-voice support needs a way to hold N parallel voice bodies. Two options:

1. Replace `body: Vec<AbcElement>` with `voices: Vec<Vec<AbcElement>>`. Cleaner symmetric API but
   forces every existing consumer to migrate from `tune.body` to `tune.voices[0]`.
2. Keep `body` as voice 0 and add `extra_voices: Vec<Vec<AbcElement>>` for voices 2+.

**Decision**: Option 2 (asymmetric).

**Consequences**: (+) Zero-churn for existing single-voice consumers (`tune.body` still works). (+)
Struct literals only need one new field (`extra_voices: Vec::new()`) — pattern matches with `..`
are unaffected. (−) API is asymmetric — callers iterating over all voices must special-case
voice 0. Provided as ergonomic trade-off; a helper method `voices_iter()` could be added later
without further breakage if the asymmetry becomes painful.

### ADR-008 — Grace notes rendered as `1/32` short notes, not a dedicated variant (2026-09-13)

**Context**: Grace notes `{gab}c` are ornaments played very briefly before the main note. Options:

1. New `AbcElement::Grace { notes, count }` variant, handled specially in `to_score`.
2. Emit each inner note as a normal `Note` with fixed duration `1/32` at parse time.

**Decision**: Option 2 (parse-time expansion).

**Consequences**: (+) No new enum variant → no additional pattern-match churn. (+) Downstream
consumers (visualizers, exporters) see grace notes as normal short notes; nothing special to
handle. (−) The ornament nature is lost after parsing — a round-trip back to ABC would produce
literal short notes instead of `{...}` syntax. Acceptable for MVP; if a future ABC exporter needs
fidelity, we can revisit with a dedicated variant.

### ADR-009 — `tied_in` state machine unifies Note-tie and Chord-tie handling (2026-09-13)

**Context**: Phase 2a implemented Note-tie coalescing with a `while cur_tie` scan-forward loop
that produced a single merged `NoteOn`/`NoteOff` pair with summed ticks. Extending this to
chord ties required tracking per-note continuations across chord boundaries with mismatched
pitch sets. Two options:

1. Keep the scan-forward pattern and grow it into a per-chord-note tracker.
2. Refactor to a state machine that carries `tied_in: [u8; MAX_CHORD_NOTES]` between elements,
   representing pitches currently sounding due to a tie into this position.

**Decision**: Option 2 (state machine).

**Consequences**: (+) Note-tie and Chord-tie share one implementation; both cases reduce to
"is this pitch already sounding? then skip NoteOn; is this pitch tying forward? then defer
NoteOff." (+) Cross-boundary partial overlap (`[CEG]-[CEF]`) falls out naturally — untied
notes emit their own `NoteOff` at the from-chord's duration; new notes emit `NoteOn` at the
to-chord's onset. (+) Rest handling (rests break ties) is explicit: flush `tied_in` at Rest.
(+) End-of-body dangling ties are trivially handled by a final flush loop. (−) The old
scan-forward loop was easier to reason about for the Note-only case; the new state machine
requires understanding a small piece of stateful arithmetic. All Phase 2a Note-tie tests
still pass unchanged.

### ADR-004 — Barlines emit no Score events in Phase 1 (2026-09-12)

**Context**: `AbcElement::Barline` is parsed but dropped during `to_score()`. Alternative:
emit `ControlChange` events at each bar for downstream tooling.

**Decision**: Drop barlines in Phase 1. Revisit in Phase 2 with concrete use case (loop
points, section markers, DAW alignment).

**Consequences**: (+) Simplest possible v0.2.0 shipped surface. (−) Downstream users who want
to loop by bar must maintain their own bar count.
