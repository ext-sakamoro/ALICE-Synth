# ALICE-Synth

**Procedural Audio Synthesizer — Don't send waveforms, send the score**

> "A symphony is not a WAV file. It's a recipe."

```
Traditional:  3-minute BGM = 30 MB (WAV) or 3 MB (MP3)
ALICE-Synth:  3-minute BGM = 2 KB (score) + 200 B (instrument patches)
```

## The Problem

ALICE-Animation compresses a full anime episode to 20-50 KB of SDF data. ALICE-Voice compresses dialogue to ~50 bytes per frame. But the moment you attach a BGM track or sound effects, the file balloons to megabytes — **the audio becomes 99.9% of the total file size**.

This is the last bastion of raw data in the ALICE pipeline.

## The Solution

Instead of encoding audio waveforms, encode **the instructions to generate them**:

- **Instruments** = mathematical oscillator definitions (FM, additive, subtractive, wavetable)
- **Score** = note events with timing, pitch, velocity, and expression
- **Effects** = DSP graph descriptions (delay, filter — all parametric, no impulse response samples)

The playback device synthesizes the audio in real-time from these descriptions — exactly like a MIDI synthesizer, but with ALICE-grade compression and deterministic output.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ALICE-Synth                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐     │
│  │  Score Parser │──▶│  Sequencer   │──▶│  Mixer / Output  │     │
│  │  (2 KB data)  │   │  tick-based  │   │  f32 PCM / i16   │     │
│  └──────────────┘   └──────┬───────┘   └──────────────────┘     │
│                             │                                     │
│              ┌──────────────┼──────────────┐                     │
│              ▼              ▼              ▼                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  FM Synth    │ │  Additive    │ │  Wavetable   │            │
│  │  2-op FM     │ │  Harmonic    │ │  Single-cycle│            │
│  │  (32 bytes)  │ │  (64 bytes)  │ │  (256 bytes) │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│              │              │              │                     │
│              ▼              ▼              ▼                     │
│  ┌─────────────────────────────────────────────────┐            │
│  │  DSP Effects Chain                               │            │
│  │  Delay | LowPassFilter | StateVariableFilter      │            │
│  │  All parametric — no impulse response samples    │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Synthesis Engines

### FM Synthesis (2-operator)

2-operator frequency modulation synthesis. Operator 1 modulates operator 0 (carrier). Each operator has a frequency ratio, modulation index, ADSR envelope, and output level. The patch struct holds 4 operator slots for forward compatibility, but only operators 0 and 1 are active in the current render path; operators 2 and 3 have zero level.

```
Patch size: 32 bytes (4 operator slots × 8 bytes each)
Active operators: 2 (operator[1] modulates operator[0])
Polyphony: 64 voices on Cortex-A76 (Pi 5)
Complexity: O(operators × samples) per voice
```

### Additive Synthesis

Sum of sine harmonics with individual amplitude envelopes. Ideal for organ, bell, and pad sounds.

```
Patch size: 64 bytes (16 harmonics × 4 bytes)
Quality: Arbitrary precision (more harmonics = richer)
```

### Subtractive Synthesis

Oscillator (saw/square/pulse/triangle/noise) → StateVariableFilter → Amplitude envelope. Classic analog synth model.

```
Patch size: 24 bytes (osc + filter + env)
Efficiency: Cheapest per-sample cost
```

### Wavetable Synthesis

Single-cycle waveform (256 samples) with linear interpolation.

```
Patch size: 256 bytes (single cycle, can be procedurally generated)
Flexibility: Any timbre from a single cycle
```

## Score Format

Compact binary score format inspired by MIDI but optimized for size:

```
┌──────────────────────────────────────────────────┐
│  ScoreHeader (8 bytes)                            │
│  ├─ magic: [u8; 4]  = "ASYN"                    │
│  ├─ tempo: u16       = BPM (40-300)              │
│  ├─ tracks: u8       = channel count (1-16)      │
│  └─ tick_div: u8     = ticks per beat (24-480)   │
├──────────────────────────────────────────────────┤
│  NoteEvent (4 bytes each)                         │
│  ├─ delta_tick: u12  = time offset (VLQ-like)    │
│  ├─ channel: u4      = instrument/track          │
│  ├─ note: u7         = MIDI note number          │
│  ├─ velocity: u7     = 0-127                     │
│  └─ flags: u2        = on/off/bend/cc            │
├──────────────────────────────────────────────────┤
│  InstrumentDef (32-256 bytes per instrument)      │
│  ├─ synth_type: u8   = FM/Additive/Sub/Wavetable│
│  └─ params: [u8; N]  = engine-specific params    │
└──────────────────────────────────────────────────┘
```

### Event Kinds

| Kind | Status | Notes |
|------|--------|-------|
| NoteOn | Implemented | velocity=0 treated as NoteOff |
| NoteOff | Implemented | Triggers release phase |
| PitchBend | Planned | Enum variant exists; not yet processed |
| ControlChange | Planned | Enum variant exists; not yet processed |

### Size Comparison

| Content | WAV | MP3 | MIDI | ALICE-Synth |
|---------|-----|-----|------|-------------|
| 3-min BGM | 30 MB | 3 MB | 20 KB | **2 KB** |
| Full anime episode audio | 50 MB | 5 MB | N/A | **~5 KB** |

## DSP Effects

All effects are mathematically defined — no sample data required.

### Delay

Circular buffer delay with configurable feedback and wet/dry mix.

```rust
let delay = Delay::from_ms(250.0, sample_rate, 0.4, 0.5);
```

### LowPassFilter

One-pole IIR low-pass filter. Efficient single-multiply-per-sample implementation.

```
y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
Size: 8 bytes
```

### StateVariableFilter (SVF)

Resonant 2-pole state-variable filter with simultaneous low-pass, band-pass, and high-pass outputs. Used internally for subtractive synthesis voice filtering.

```
Outputs: (low, band, high) simultaneously
Size: 12 bytes
Default Effect output: low-pass
```

```rust
let mut svf = StateVariableFilter::new(cutoff_hz, resonance, sample_rate);
let (low, band, high) = svf.process_svf(input_sample);
```

## API Design

```rust
use alice_synth::{Synthesizer, Score, Patch, FmPatch, AdditivePatch};

// Create synthesizer (no_std compatible)
let mut synth = Synthesizer::new(44100); // sample rate

// Control master volume [0.0, 1.0] (default: 0.8)
synth.master_volume = 0.8;

// Load instrument patches
let piano = Patch::Fm(FmPatch::electric_piano());
let strings = Patch::Additive(AdditivePatch::strings());
synth.load_patch(0, piano);
synth.load_patch(1, strings);

// Load score (2 KB)
let score = Score::from_bytes(&score_data)?;
synth.load_score(&score);

// Trigger notes directly
synth.note_on(0, 60, 100); // channel, MIDI note, velocity
synth.note_off(0, 60);

// Render to f32 buffer (real-time or offline)
let mut buffer = [0.0f32; 1024];
synth.render(&mut buffer); // Fill 1024 samples

// Render to i16 PCM buffer
let mut buffer_i16 = [0i16; 1024];
synth.render_i16(&mut buffer_i16);
```

## Voice Stealing

The synthesizer maintains a pool of 64 voices. When all voices are active and a new note-on arrives, the engine steals the voice with the lowest current amplitude (computed as `amp_env.level() * velocity`). This minimizes audible artifacts by silencing the quietest active note.

## Master Volume

`Synthesizer::master_volume` is a public `f32` field in `[0.0, 1.0]`. It scales the final mixed output before writing to the render buffer. Default value is `0.8`.

## Ecosystem Integration

```
ALICE-Animation ─── score data ───▶ ALICE-Synth ─── PCM ───▶ Speaker
     │                                    ▲
     │                                    │
ALICE-Voice ──── lip-sync timing ─────────┘
     │
ALICE-Streaming-Protocol ── multiplexed audio+video ──▶ Network

ALICE-LOL ──── IntentNode::Music { packet: [u8; 8] } ────▶ ALICE-Synth
                            (zero-dep 8-byte contract)
```

| Bridge | Direction | Data |
|--------|-----------|------|
| Animation → Synth | Score per cut | 2 KB score |
| Voice → Synth | Timing sync (dialogue ↔ BGM ducking) | Envelope follower |
| Synth → Streaming | Multiplexed in ASP packet | Score bytes (not PCM) |
| Edge → Synth | Sensor data → sonification | Pitch/volume mapping |
| LOL ↔ Synth | Musical Intent packet | 8-byte `MusicIntent` |
| LLM ↔ Synth | Future `PlanHead` impl | 8-byte packet + generated `AbcTune` |

## Target Platforms

| Platform | Polyphony | Latency | Memory |
|----------|-----------|---------|--------|
| Raspberry Pi 5 (A76) | 128 voices | < 3ms | < 1 MB |
| ESP32-S3 | 8 voices | < 10ms | < 64 KB |
| Cortex-M4 (STM32F4) | 4 voices | < 5ms | < 32 KB |
| RISC-V (GD32VF103) | 2 voices | < 10ms | < 16 KB |
| x86_64 (SIMD) | 512+ voices | < 1ms | < 4 MB |

## Feature Flags

| Feature | Dependencies | Description |
|---------|-------------|-------------|
| *(default)* | None | Core synth engine, no_std, zero alloc |
| `std` | std | File I/O, Vec-based buffers |
| `abc` | *(none)* | ABC notation parser — YuE2-style symbolic planning |
| `intent` | `abc` | 8-byte Musical Intent packet + deterministic synthesizer (ALICE 三相原理 Phase 3) |
| `agentic` | `abc` | `AbcDiff` / patch API — position-based and LCS backends for LLM revision loops |
| `cover` | `intent` | Zero-shot cover pipeline — `Transcriber` trait + `CoverPipeline` |
| `ffi` | std | C/C++/C# FFI — 20 `extern "C"` functions |
| `python` | pyo3, std | PyO3 Python bindings — 4 classes + 2 functions |
| `midi` | std | MIDI file import/export (Planned) |
| `streaming` | libasp | ALICE Streaming Protocol integration (Planned) |
| `animation` | std | ALICE-Animation bridge (Planned) |

Note: `midi`, `streaming`, and `animation` feature flags are declared in `Cargo.toml` but have no implementation code yet. Enabling them currently has no effect beyond enabling `std`.

## ABC Notation + Musical Intent (ALICE 三相原理 Phase 2 / 3)

The `abc` feature adds a full-featured ABC notation parser and the `intent` feature adds an 8-byte Musical Intent packet with a deterministic procedural synthesizer. Together they realize the **Data → Law → Intent** progression: instead of shipping audio waveforms (Phase 1, MB scale) or ABC score text (Phase 2, KB scale), a caller can ship **8 bytes** and let the receiver reconstruct the tune locally.

**ABC parser** (`abc` feature, ~2500 LoC + 76 tests):

- Headers: `X:` `T:` `M:` `L:` `Q:` `K:` + inline `[K:...]` / `[M:...]` fields
- Body: notes `A-G a-g`, octave `,` / `'`, accidentals `^ _ = ^^ __`, rational durations
- Chords: `[CEG]` up to 8 simultaneous notes, per-note accidentals, chord ties
- Repeats + voltas: `|:` `:|` with `[1` … `[4` endings (max iteration = `max(2, num_voltas)`)
- Ties: `C-C` (Note-to-Note) and `[CEG]-[CEF]` (Chord-to-Chord, per-note coalescing)
- Tuplets: `(P` `(P:Q` `(P:Q:R` (ABC 2.1 defaults for bare `(P`)
- Grace notes: `{gab}c` → each inner note as `1/32` prefix `Note`
- Multi-voice: `V:1` / `V:2` / … render to channels 0, 1, … with absolute-tick merge
- Church modes: Ionian / Dorian / Phrygian / Lydian / Mixolydian / Aeolian / Locrian
- Keys: 30 canonical major + minor + all 7 modes = 105 combinations
- Shorthand: `::` = end-then-start repeat, `|:|` = barline + repeat start

```rust
use alice_synth::abc::parse;

let src = "X:1\nT:Ode to Joy\nM:4/4\nL:1/4\nQ:100\nK:C\n\
           E E F G | G F E D | C C D E | E3/2 D/2 D2 |\n";
let tune = parse(src).unwrap();
let score = tune.to_score(96);  // → existing Synthesizer for PCM output
```

**Musical Intent packet** (`intent` feature, ~750 LoC + 24 tests):

| Byte | Field              | Range       |
|------|--------------------|-------------|
| 0    | `genre`            | `0..=255` (Folk / Jazz / Blues / Classical / Pop / Ambient / …) |
| 1    | `mood`             | `0..=255` (Happy / Sad / Tense / Serene / Energetic / …) |
| 2    | `length_bars`      | `0..=255`   |
| 3    | `tempo_bpm_offset` | `0..=255` → BPM `40..=295` |
| 4    | `key`              | `0..=14` canonical tonic in circle-of-fifths order |
| 5    | `mode`             | `0..=6` church mode |
| 6-7  | `variation_seed`   | `u16` little-endian PRNG seed |

```rust
use alice_synth::intent::{genre, mode, mood, MusicIntent};

// 8 bytes → full tune
let intent = MusicIntent {
    genre: genre::JAZZ,
    mood: mood::MYSTERIOUS,
    length_bars: 16,
    tempo_bpm_offset: 20,      // → 60 BPM
    key: 4,                    // E
    mode: mode::DORIAN,        // E Dorian
    variation_seed: 0xDEAD,
};
let bytes: [u8; 8] = intent.to_bytes();

// Deterministic procedural synthesizer — future LLM plan heads plug in via
// the `PlanHead` trait without changing the wire format.
let tune = intent.synthesize();
let score = tune.to_score(96);
```

**Agentic editing** (`agentic` feature, ~600 LoC + 25 tests):

Two diff backends — pick per revision profile:

- `AbcTune::diff(&other)` — **position-based**, `O(N)`. Fast, small output. Best for locally-scoped LLM edits (a note here, tempo bumped there).
- `AbcTune::diff_lcs(&other)` — **LCS-based**, `O(N × M)`. Shift-aware. Best for structural edits (prepending a bar, splicing a phrase).

```rust
let original = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
let revised = parse("X:1\nM:4/4\nL:1/4\nK:C\nA C D E F |\n").unwrap();

// LCS correctly identifies "prepend A" as a single insertion:
let diff = original.diff_lcs(&revised);
assert_eq!(diff.body_insertions.len(), 1);
assert!(diff.body_removals.is_empty());

// Positional would misdetect this as 4 replacements + 1 tail insertion;
// use it when edits are known to be in-place.

let patched = original.patch(&diff);
```

**Zero-shot cover pipeline** (`cover` feature, ~290 LoC + 6 tests):

```rust
use alice_synth::cover::{CoverPipeline, Transcriber, TranscribeError};
use alice_synth::intent::{mode, mood, MusicIntent};

// Concrete Transcriber implementations live in downstream crates
// (e.g. an external ML crate wrapping SheetSage2).
struct MyTranscriber;
impl Transcriber for MyTranscriber { /* … */ }

let pipeline = CoverPipeline::new(MyTranscriber);
let target = MusicIntent {
    genre: genre::AMBIENT,
    mood: mood::SERENE,
    mode: mode::LYDIAN,
    ..MusicIntent::DEFAULT_C_MAJOR
};
let cover = pipeline.cover(&audio_samples, 44_100, target)?;
// Source's key / tempo / length are preserved; genre / mood / mode / seed
// come from the target — a shift-in-style, same-shape cover.
```

**Cross-crate bridge** — ALICE-LOL v0.2.0 exposes `IntentNode::Music { packet: [u8; 8] }` so LOL programs can embed Musical Intent alongside physical verb intents (grasp / walk / etc.). Zero dependency edge between the two crates — the 8-byte packet is the entire contract.

## FFI & Language Bindings

| Target | Path | Functions |
|--------|------|-----------|
| C-ABI (FFI) | `src/ffi.rs` | 20 `extern "C"` functions |
| Python (PyO3) | `src/python.rs` | 4 classes + 2 functions |
| Unity (C#) | `bindings/unity/AliceSynth.cs` | `Synthesizer`, `OscillatorHandle` (IDisposable) |
| UE5 (C++) | `bindings/ue5/AliceSynth.h` | `FSynthesizer`, `FOscillator` (RAII) |

## Tests

The library ships with **232 unit tests + 6 doctests** covering every module:

| Module | Tests | Feature gate |
|--------|-------|--------------|
| `oscillator` | 23 | *(core)* |
| `envelope` | 16 | *(core)* |
| `score` | 16 | *(core)* |
| `effects` | 15 | *(core)* |
| `patch` | 15 | *(core)* |
| `synth` | 15 | *(core)* |
| `ffi` | 19 | `ffi` |
| `abc` | 76 | `abc` |
| `intent` | 24 | `intent` |
| `agentic` | 25 | `agentic` |
| `cover` | 6 | `cover` |
| doctests | 6 | mixed |

```bash
# Everything (recommended for CI)
cargo test --features intent,agentic,cover,ffi,std

# Just the ABC / Intent stack
cargo test --features intent,agentic,cover,std
```

## License

MIT

## Author

Moroya Sakamoto

---

*"The orchestra is in the equation, not the recording."*
