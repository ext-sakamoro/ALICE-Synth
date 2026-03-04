//! # ALICE-Synth
//!
//! **Procedural Audio Synthesizer — Don't send waveforms, send the score.**
//!
//! ALICE-Synth generates audio from compact mathematical descriptions.
//! A 3-minute BGM fits in ~2 KB as a score; a sound effect in 48 bytes as a patch.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`oscillator`] | Phase-accumulator waveform generators (sine, saw, square, triangle, noise, wavetable) |
//! | [`envelope`] | ADSR envelope generator — 16-byte config, sample-accurate timing |
//! | [`patch`] | Instrument patches — FM (32 B), Additive (64 B), Subtractive (24 B), Wavetable (256 B) |
//! | [`score`] | Compact score format — 8-byte header + 4-byte note events, serializable |
//! | [`synth`] | Multi-voice polyphonic engine — 64-voice, 16-channel, score playback to PCM |
//! | [`effects`] | Audio effects — delay, low-pass, state-variable filter, reverb |
//!
//! ## Cargo Features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | no | Standard library support |
//! | `ffi` | no | C/C++/C# FFI — 20 `extern "C"` functions |
//! | `python` | no | PyO3 Python bindings — 4 classes + 2 functions |
//! | `midi` | no | MIDI file import/export (future) |
//! | `streaming` | no | ALICE Streaming Protocol integration (future) |
//! | `animation` | no | ALICE-Animation lip-sync bridge (future) |
//!
//! ## Quick Start
//!
//! ```
//! use alice_synth::{Oscillator, Waveform, Adsr, AdsrState};
//!
//! // Create a sine oscillator
//! let mut osc = Oscillator::new(Waveform::Sine);
//!
//! // Generate one sample at 440 Hz, 44100 Hz sample rate
//! let inv_sr = 1.0 / 44100.0_f32;
//! let sample = osc.next_sample(440.0, inv_sr);
//! assert!(sample >= -1.0 && sample <= 1.0);
//!
//! // ADSR envelope (attack=10ms, decay=50ms, sustain=0.7, release=100ms)
//! let env = Adsr::from_ms(10.0, 50.0, 0.7, 100.0, 44100.0);
//! let mut state = AdsrState::new();
//! state.note_on();
//! let amplitude = state.next(&env);
//! assert!(amplitude >= 0.0);
//! ```
//!
//! ## Patch Sizes
//!
//! | Patch Type | Size |
//! |------------|------|
//! | FM | 32 bytes (carrier + modulator frequencies, ratios, envelope) |
//! | Additive | 64 bytes (8 harmonic amplitudes + envelope) |
//! | Subtractive | 24 bytes (filter cutoff, resonance, envelope) |
//! | Wavetable | 256 bytes (64-sample table + envelope) |
//!
//! Author: Moroya Sakamoto

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::inline_always
)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod effects;
pub mod envelope;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod oscillator;
pub mod patch;
#[cfg(feature = "python")]
pub mod python;
pub mod score;
pub mod synth;

pub use effects::{Delay, Effect, LowPassFilter};
pub use envelope::{Adsr, AdsrState};
pub use oscillator::{Oscillator, Waveform};
pub use patch::{AdditivePatch, FmPatch, Patch, SubtractivePatch, WavetablePatch};
pub use score::{NoteEvent, NoteEventKind, Score, ScoreHeader};
pub use synth::{Synthesizer, Voice};
