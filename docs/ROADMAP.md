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

### 🚧 Phase 1b — ABC notation parser (v0.2.0, 2026-09-12)

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

### ⏳ Phase 2 — ABC extended coverage

- Multi-voice `V:` support (multi-track output → Score channels)
- Chord expansion `[CEG]` → simultaneous NoteOn/NoteOff at delta=0
- Ties `-` → merge adjacent notes of the same pitch into one longer note
- Grace notes `{gab}` → very short prefix notes
- Tuplets `(3abc` → 3-in-the-time-of-2 duration adjustment
- Minor keys + mode names (dor / mix / lyd / phr / loc / aeo)
- Repeat structures `|:` `:|` `[1` `[2` — score-level unroll to native format

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

### ADR-004 — Barlines emit no Score events in Phase 1 (2026-09-12)

**Context**: `AbcElement::Barline` is parsed but dropped during `to_score()`. Alternative:
emit `ControlChange` events at each bar for downstream tooling.

**Decision**: Drop barlines in Phase 1. Revisit in Phase 2 with concrete use case (loop
points, section markers, DAW alignment).

**Consequences**: (+) Simplest possible v0.2.0 shipped surface. (−) Downstream users who want
to loop by bar must maintain their own bar count.
