# ALICE-Synth design decisions

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

### ADR-010 — Ambiguous key signature reverse-lookup prefers Ionian (2026-09-13)

**Context**: `MusicIntent::from_tune` needs to recover a `(tonic, mode)` pair from an
`AbcHeader`'s sharp count. A given sharp count is ambiguous — 0 sharps could be C major,
A minor, D Dorian, E Phrygian, F Lydian, G Mixolydian, or B Locrian. All are valid.

**Decision**: Prefer Ionian (major). If Ionian is out of the canonical `-7..=7` range, fall
back to Aeolian (natural minor). Other modes are never inferred.

**Consequences**: (+) Deterministic and unsurprising for the most common case. (+) Zero
external metadata required. (−) A tune actually written in D Dorian will be reported as
F major on the reverse trip. This is documented behaviour, not a bug — callers who need
mode fidelity should pass the intent packet around instead of the ABC text.

### ADR-011 — Procedural synthesizer is the canonical MVP; LLM plan head is a future drop-in (2026-09-13)

**Context**: ALICE 三相原理 Phase 3 is defined by an intent packet + a synthesizer that
turns intent into a tune. The synthesizer is the "smart" part. Two options:

1. Ship a stub that panics until an LLM plan head exists in ALICE-LLM.
2. Ship a deterministic procedural generator (`mode_scale × mood_bias × LCG PRNG`) that
   produces valid, listenable tunes today; treat it as the canonical MVP; allow a future
   LLM-driven implementation to slot in without changing the `MusicIntent` wire format.

**Decision**: Option 2 (procedural MVP).

**Consequences**: (+) The intent packet is usable end-to-end from day one — 8 bytes in, PCM
audio out via `synthesize().to_score(96)` → existing `Synthesizer`. (+) Downstream projects
(ALICE-LOL Music IR, ALICE-LLM audio head, ALICE-Cognitive agentic editing) can integrate
against a stable API. (+) Same packet always produces the same tune, on any host. (−) Musical
quality is limited by the procedural rules; a rich LLM plan head will produce more
compelling output. That is Phase 3.1 work, on ALICE-LLM's side.

### ADR-012 — Cross-crate `MusicIntent` protocol is the 8-byte packet, nothing else (2026-09-13)

**Context**: ALICE-LOL and ALICE-Synth both need to speak `MusicIntent`. Two options:

1. Make ALICE-LOL depend on ALICE-Synth (or a shared "protocol" crate).
2. Have both crates independently agree on the 8-byte wire layout; ALICE-LOL wraps the
   payload as opaque `[u8; 8]` in `IntentNode::Music`, and consumers deserialize via
   `alice_synth::intent::MusicIntent::from_bytes`.

**Decision**: Option 2 (opaque 8-byte payload; zero dependency edge).

**Consequences**: (+) ALICE-LOL takes no dependency edge on this crate.
(+) The 8-byte contract is the *only* thing crates must agree on; extensions to
`MusicIntent`'s Rust surface (methods, constants) do not ripple into LOL. (+) Other
consumers (network protocols, embedded firmware, non-Rust languages) can produce and
consume the packet without linking either crate. (−) Discoverability suffers: an
ALICE-LOL user staring at `IntentNode::Music { packet: [u8; 8] }` needs a comment pointer
to know what byte layout to write. Mitigated by the doc comment on the variant.

### ADR-013 — `AbcDiff` is position-based, not LCS/Myers (2026-09-13)

**Context**: LLM-driven revisions are usually localized: a phrase gets its ending changed,
a note is transposed, a section is added at the end. Full LCS/Myers diff catches shifts
too, at the cost of `O(N × M)` computation and much more complex output.

**Decision**: Ship a position-based diff for Phase 3.1. A per-index walk records
insertions, removals, and replacements against original body positions; unchanged
elements produce empty entries.

**Consequences**: (+) `O(max(N, M))` compute; small, predictable output structure that
LLMs can serialize back into ABC without ambiguity. (+) Trivial to understand and audit.
(−) Element shifts (e.g. "prepend a bar") produce a run of replacements + a tail
insertion instead of the semantically correct "one insertion at index 0". Acceptable for
MVP; a Phase 4 upgrade can drop in an LCS backend behind the same public surface.

### ADR-014 — `Transcriber` trait lives in ALICE-Synth, concrete impls do not (2026-09-13)

**Context**: The `cover` feature composes an audio → symbolic → audio pipeline, but the
transcription stage is a hard ML problem. Two options:

1. Ship a concrete `Transcriber` implementation in ALICE-Synth (would require ML deps).
2. Ship only the `Transcriber` trait + composition; require downstream crates to provide
   concrete implementations.

**Decision**: Option 2.

**Consequences**: (+) ALICE-Synth stays pure Rust + `alloc`; no `wgpu` / `pytorch` /
`ONNX Runtime` transitive weight. (+) Downstream implementers (e.g. an eventual
`alice-music-transcribe` crate wrapping `SheetSage2`) have a stable trait to target.
(−) A user opting into the `cover` feature cannot immediately run a cover; they need a
concrete transcriber. The doctest and unit tests use a `FixedTranscriber` stub to
demonstrate the composition pattern.

### ADR-015 — LCS backend co-exists with the positional backend (2026-09-13)

**Context**: Phase 3.1 shipped `AbcDiff::compute` as a position-based diff and deferred
LCS/Myers to Phase 4 (ADR-013). Phase 3.2 brought LCS forward as a bounded engineering
task. The question was whether to replace the positional backend or keep both.

**Decision**: Keep both. `AbcDiff::compute` remains position-based; `AbcDiff::compute_lcs`
adds an LCS backend. `AbcTune::diff` calls the positional backend; `AbcTune::diff_lcs`
calls the LCS backend. Both produce interchangeable `AbcDiff` values (the LCS backend
just leaves `body_replacements` empty).

**Consequences**: (+) Callers pick per revision profile: small localized edits → `diff`
(`O(N)`, no allocation for a DP table); structural edits → `diff_lcs` (`O(N × M)`, but
shift-aware). (+) Backward compatible — no existing caller of `compute` / `diff` sees
behaviour change. (−) Two backends to maintain. Acceptable because the LCS backend is
~80 lines and the positional backend is ~40 lines; both live in one module.

This ADR supersedes the "LCS is Phase 4 candidate" note in ADR-013 — LCS shipped in
Phase 3.2 instead.

### ADR-004 — Barlines emit no Score events in Phase 1 (2026-09-12)

**Context**: `AbcElement::Barline` is parsed but dropped during `to_score()`. Alternative:
emit `ControlChange` events at each bar for downstream tooling.

**Decision**: Drop barlines in Phase 1. Revisit in Phase 2 with concrete use case (loop
points, section markers, DAW alignment).

**Consequences**: (+) Simplest possible v0.2.0 shipped surface. (−) Downstream users who want
to loop by bar must maintain their own bar count.
