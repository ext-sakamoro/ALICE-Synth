//! ALICE-Synth — Procedural Audio Synthesizer
//!
//! Don't send waveforms, send the score.
//!
//! Generates audio from mathematical descriptions:
//! - FM/Additive/Subtractive/Wavetable synthesis engines
//! - Compact score format (2 KB for 3-minute BGM)
//! - Procedural sound effects (48 bytes for an explosion)
//! - no_std compatible, zero allocation in core path
//!
//! Author: Moroya Sakamoto

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod oscillator;
pub mod envelope;
pub mod patch;
pub mod score;
pub mod synth;
pub mod effects;

pub use oscillator::{Oscillator, Waveform};
pub use envelope::{Adsr, AdsrState};
pub use patch::{Patch, FmPatch, AdditivePatch, SubtractivePatch, WavetablePatch};
pub use score::{Score, ScoreHeader, NoteEvent, NoteEventKind};
pub use synth::{Synthesizer, Voice};
pub use effects::{Effect, Delay, LowPassFilter};
