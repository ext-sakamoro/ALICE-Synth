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
```

| Bridge | Direction | Data |
|--------|-----------|------|
| Animation → Synth | Score per cut | 2 KB score |
| Voice → Synth | Timing sync (dialogue ↔ BGM ducking) | Envelope follower |
| Synth → Streaming | Multiplexed in ASP packet | Score bytes (not PCM) |
| Edge → Synth | Sensor data → sonification | Pitch/volume mapping |

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
| `midi` | std | MIDI file import/export (Planned) |
| `streaming` | libasp | ALICE Streaming Protocol integration (Planned) |
| `animation` | std | ALICE-Animation bridge (Planned) |

Note: `midi`, `streaming`, and `animation` feature flags are declared in `Cargo.toml` but have no implementation code yet. Enabling them currently has no effect beyond enabling `std`.

## Tests

The library ships with 100 unit tests covering all modules:

| Module | Tests |
|--------|-------|
| `oscillator` | 23 |
| `envelope` | 16 |
| `score` | 16 |
| `effects` | 15 |
| `patch` | 15 |
| `synth` | 15 |

```
cargo test
```

## License

MIT

## Author

Moroya Sakamoto

---

*"The orchestra is in the equation, not the recording."*
