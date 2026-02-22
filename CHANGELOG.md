# Changelog

All notable changes to ALICE-Synth will be documented in this file.

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
