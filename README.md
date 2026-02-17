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
- **Effects** = DSP graph descriptions (reverb IR as exponential decay, delay as recurrence)

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
│  │  4-op DX7    │ │  Harmonic    │ │  Single-cycle│            │
│  │  (32 bytes)  │ │  (64 bytes)  │ │  (256 bytes) │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│              │              │              │                     │
│              ▼              ▼              ▼                     │
│  ┌─────────────────────────────────────────────────┐            │
│  │  DSP Effects Chain                               │            │
│  │  Reverb (exp decay) │ Delay │ Filter │ Chorus   │            │
│  │  All parametric — no impulse response samples    │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Synthesis Engines

### FM Synthesis (4-operator)

Yamaha DX7-style frequency modulation. Each operator is a sine oscillator with frequency ratio, modulation index, envelope (ADSR), and feedback.

```
Patch size: 32 bytes (4 operators × 8 bytes each)
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

Oscillator (saw/square/pulse) → Filter (resonant LP/HP/BP) → Envelope. Classic analog synth model.

```
Patch size: 24 bytes (osc + filter + env)
Efficiency: Cheapest per-sample cost
```

### Wavetable Synthesis

Single-cycle waveform (256 samples) with interpolation and morphing.

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

### Size Comparison

| Content | WAV | MP3 | MIDI | ALICE-Synth |
|---------|-----|-----|------|-------------|
| 3-min BGM | 30 MB | 3 MB | 20 KB | **2 KB** |
| Explosion SE | 200 KB | 20 KB | — | **48 bytes** |
| Footstep SE | 50 KB | 5 KB | — | **24 bytes** |
| UI click | 10 KB | 1 KB | — | **8 bytes** |
| Full anime episode audio | 50 MB | 5 MB | N/A | **~5 KB** |

## Sound Effects (Procedural SE)

Sound effects are generated from mathematical descriptions rather than recorded samples:

| Effect | Formula | Params | Size |
|--------|---------|--------|------|
| Explosion | White noise → exp decay → LP filter sweep | decay_ms, filter_freq, resonance | 12 bytes |
| Footstep | Noise burst → band-pass → short decay | material, weight, surface | 8 bytes |
| Sword swing | Sine sweep (high→low) + noise burst | speed, length, material | 10 bytes |
| Rain | Filtered noise + random drop triggers | density, drop_size, surface | 8 bytes |
| Engine | FM synthesis + vibrato + harmonics | rpm, load, exhaust_type | 16 bytes |

## API Design

```rust
use alice_synth::{Synthesizer, Score, Patch, FmPatch};

// Create synthesizer (no_std compatible)
let mut synth = Synthesizer::new(44100); // sample rate

// Load instrument patches (32 bytes each)
let piano = FmPatch::electric_piano();
let strings = Patch::additive_strings();
synth.load_patch(0, piano);
synth.load_patch(1, strings);

// Load score (2 KB)
let score = Score::from_bytes(&score_data)?;
synth.load_score(&score);

// Render to buffer (real-time or offline)
let mut buffer = [0i16; 1024];
synth.render(&mut buffer); // Fill 1024 samples

// Procedural sound effect (one-shot)
let explosion = synth.trigger_se(SE::explosion(
    decay_ms: 800,
    filter_hz: 2000,
    resonance: 0.7,
));
```

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
| Animation → Synth | Score + SE triggers per cut | 2 KB score + event list |
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
| `midi` | std | MIDI file import/export |
| `streaming` | libasp | ALICE Streaming Protocol integration |
| `animation` | std | ALICE-Animation bridge (score + SE triggers) |

## License

MIT

## Author

Moroya Sakamoto

---

*"The orchestra is in the equation, not the recording."*
