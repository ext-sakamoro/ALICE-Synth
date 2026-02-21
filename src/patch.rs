//! Instrument patch definitions
//!
//! Each patch type encodes an instrument's timbre as mathematical parameters.
//! Patch sizes: FM=32B, Additive=64B, Subtractive=24B, Wavetable=256B.
//!
//! Author: Moroya Sakamoto

use crate::envelope::Adsr;
use crate::oscillator::Waveform;

/// Synthesis engine type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SynthType {
    Fm = 0,
    Additive = 1,
    Subtractive = 2,
    Wavetable = 3,
}

/// Unified patch enum
#[derive(Debug, Clone)]
pub enum Patch {
    Fm(FmPatch),
    Additive(AdditivePatch),
    Subtractive(SubtractivePatch),
    Wavetable(WavetablePatch),
}

impl Patch {
    pub fn synth_type(&self) -> SynthType {
        match self {
            Patch::Fm(_) => SynthType::Fm,
            Patch::Additive(_) => SynthType::Additive,
            Patch::Subtractive(_) => SynthType::Subtractive,
            Patch::Wavetable(_) => SynthType::Wavetable,
        }
    }
}

/// FM Synthesis patch — 4-operator DX7-style
///
/// Each operator: frequency ratio + modulation index + ADSR
/// Total: 32 bytes (4 operators × 8 bytes)
#[derive(Debug, Clone)]
pub struct FmPatch {
    /// 4 FM operators
    pub operators: [FmOperator; 4],
    /// Algorithm: how operators connect (0-7)
    pub algorithm: u8,
    /// Feedback amount for operator 0 (0-7)
    pub feedback: u8,
}

/// Single FM operator
#[derive(Debug, Clone, Copy)]
pub struct FmOperator {
    /// Frequency ratio relative to carrier (e.g. 1.0, 2.0, 3.5)
    pub ratio: f32,
    /// Modulation index (depth of FM)
    pub mod_index: f32,
    /// ADSR envelope
    pub envelope: Adsr,
    /// Output level [0.0, 1.0]
    pub level: f32,
}

impl FmPatch {
    /// Classic electric piano patch (DX7 "E.PIANO 1" inspired)
    pub fn electric_piano() -> Self {
        let sr = 44100.0;
        Self {
            operators: [
                FmOperator {
                    ratio: 1.0,
                    mod_index: 0.0,
                    level: 1.0,
                    envelope: Adsr::from_ms(2.0, 800.0, 0.0, 200.0, sr),
                },
                FmOperator {
                    ratio: 1.0,
                    mod_index: 2.5,
                    level: 0.8,
                    envelope: Adsr::from_ms(2.0, 400.0, 0.0, 100.0, sr),
                },
                FmOperator {
                    ratio: 14.0,
                    mod_index: 0.0,
                    level: 0.3,
                    envelope: Adsr::from_ms(0.5, 50.0, 0.0, 50.0, sr),
                },
                FmOperator {
                    ratio: 1.0,
                    mod_index: 0.0,
                    level: 0.0,
                    envelope: Adsr::from_ms(1.0, 1.0, 0.0, 1.0, sr),
                },
            ],
            algorithm: 0,
            feedback: 0,
        }
    }

    /// Simple bell patch
    pub fn bell() -> Self {
        let sr = 44100.0;
        Self {
            operators: [
                FmOperator {
                    ratio: 1.0,
                    mod_index: 0.0,
                    level: 1.0,
                    envelope: Adsr::from_ms(1.0, 2000.0, 0.0, 500.0, sr),
                },
                FmOperator {
                    ratio: 3.5,
                    mod_index: 5.0,
                    level: 0.7,
                    envelope: Adsr::from_ms(1.0, 1500.0, 0.0, 400.0, sr),
                },
                FmOperator {
                    ratio: 0.0,
                    mod_index: 0.0,
                    level: 0.0,
                    envelope: Adsr::from_ms(1.0, 1.0, 0.0, 1.0, sr),
                },
                FmOperator {
                    ratio: 0.0,
                    mod_index: 0.0,
                    level: 0.0,
                    envelope: Adsr::from_ms(1.0, 1.0, 0.0, 1.0, sr),
                },
            ],
            algorithm: 0,
            feedback: 0,
        }
    }
}

/// Additive Synthesis patch — sum of sine harmonics
///
/// 16 harmonics × (amplitude f32) = 64 bytes
#[derive(Debug, Clone)]
pub struct AdditivePatch {
    /// Harmonic amplitudes [0.0, 1.0] for harmonics 1..=16
    pub harmonics: [f32; 16],
    /// ADSR envelope
    pub envelope: Adsr,
}

impl AdditivePatch {
    /// Organ-like sound (odd harmonics)
    pub fn organ() -> Self {
        let sr = 44100.0;
        Self {
            harmonics: [
                1.0, 0.0, 0.5, 0.0, 0.33, 0.0, 0.25, 0.0, 0.2, 0.0, 0.16, 0.0, 0.14, 0.0,
                0.12, 0.0,
            ],
            envelope: Adsr::organ(sr),
        }
    }

    /// String-like sound (all harmonics with 1/n rolloff)
    pub fn strings() -> Self {
        let sr = 44100.0;
        let mut harmonics = [0.0f32; 16];
        for (i, h) in harmonics.iter_mut().enumerate() {
            *h = 1.0 / (i + 1) as f32;
        }
        Self {
            harmonics,
            envelope: Adsr::from_ms(100.0, 50.0, 0.8, 300.0, sr),
        }
    }
}

/// Subtractive Synthesis patch — oscillator → filter → envelope
///
/// Total: 24 bytes
#[derive(Debug, Clone)]
pub struct SubtractivePatch {
    /// Oscillator waveform
    pub waveform: Waveform,
    /// Filter cutoff frequency (Hz)
    pub cutoff_hz: f32,
    /// Filter resonance [0.0, 1.0)
    pub resonance: f32,
    /// Filter envelope amount [0.0, 1.0]
    pub filter_env_amount: f32,
    /// Amplitude envelope
    pub amp_envelope: Adsr,
    /// Filter envelope
    pub filter_envelope: Adsr,
}

impl SubtractivePatch {
    /// Classic bass sound
    pub fn bass() -> Self {
        let sr = 44100.0;
        Self {
            waveform: Waveform::Saw,
            cutoff_hz: 800.0,
            resonance: 0.3,
            filter_env_amount: 0.6,
            amp_envelope: Adsr::from_ms(5.0, 100.0, 0.7, 100.0, sr),
            filter_envelope: Adsr::from_ms(5.0, 200.0, 0.2, 100.0, sr),
        }
    }

    /// Pluck/stab sound
    pub fn pluck() -> Self {
        let sr = 44100.0;
        Self {
            waveform: Waveform::Square,
            cutoff_hz: 2000.0,
            resonance: 0.5,
            filter_env_amount: 0.8,
            amp_envelope: Adsr::from_ms(1.0, 300.0, 0.0, 50.0, sr),
            filter_envelope: Adsr::from_ms(1.0, 150.0, 0.0, 50.0, sr),
        }
    }
}

/// Wavetable Synthesis patch — single-cycle waveform
///
/// 256 samples × f32 = 1024 bytes (but can store as i8 = 256 bytes)
#[derive(Debug, Clone)]
pub struct WavetablePatch {
    /// Single-cycle waveform (256 samples, [-1.0, 1.0])
    pub table: [f32; 256],
    /// ADSR envelope
    pub envelope: Adsr,
}

impl WavetablePatch {
    /// Generate from a mathematical function
    pub fn from_fn<F: Fn(f32) -> f32>(f: F, envelope: Adsr) -> Self {
        let mut table = [0.0f32; 256];
        for (i, sample) in table.iter_mut().enumerate() {
            let phase = i as f32 / 256.0;
            *sample = f(phase);
        }
        Self { table, envelope }
    }

    /// Look up value with linear interpolation
    #[inline]
    pub fn lookup(&self, phase: f32) -> f32 {
        let idx_f = phase * 256.0;
        let idx0 = idx_f as usize % 256;
        let idx1 = (idx0 + 1) % 256;
        let idx_floor = {
            let i = idx_f as i32;
            let f = i as f32;
            if idx_f < f { f - 1.0 } else { f }
        };
        let frac = idx_f - idx_floor;
        self.table[idx0] * (1.0 - frac) + self.table[idx1] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn test_fm_patch_creation() {
        let patch = FmPatch::electric_piano();
        assert_eq!(patch.operators[0].ratio, 1.0);
        assert!(patch.operators[1].mod_index > 0.0);
    }

    #[test]
    fn test_additive_organ() {
        let patch = AdditivePatch::organ();
        // Fundamental should be loudest
        assert_eq!(patch.harmonics[0], 1.0);
        // Even harmonics should be zero
        assert_eq!(patch.harmonics[1], 0.0);
    }

    #[test]
    fn test_wavetable_lookup() {
        let sr = 44100.0;
        let wt = WavetablePatch::from_fn(
            |phase| (phase * 2.0 * PI).sin(),
            Adsr::organ(sr),
        );
        // Phase 0.25 → sin(π/2) ≈ 1.0
        let val = wt.lookup(0.25);
        assert!((val - 1.0).abs() < 0.05, "expected ~1.0, got {val}");
    }

    #[test]
    fn test_subtractive_bass() {
        let patch = SubtractivePatch::bass();
        assert_eq!(patch.waveform, Waveform::Saw);
        assert!(patch.cutoff_hz > 0.0);
    }

    // --- NEW TESTS ---

    #[test]
    fn test_synth_type_from_patch_fm() {
        let patch = Patch::Fm(FmPatch::electric_piano());
        assert_eq!(patch.synth_type(), crate::patch::SynthType::Fm);
    }

    #[test]
    fn test_synth_type_from_patch_additive() {
        let patch = Patch::Additive(AdditivePatch::organ());
        assert_eq!(patch.synth_type(), crate::patch::SynthType::Additive);
    }

    #[test]
    fn test_synth_type_from_patch_subtractive() {
        let patch = Patch::Subtractive(SubtractivePatch::bass());
        assert_eq!(patch.synth_type(), crate::patch::SynthType::Subtractive);
    }

    #[test]
    fn test_synth_type_from_patch_wavetable() {
        let adsr = crate::envelope::Adsr::organ(44100.0);
        let patch = Patch::Wavetable(WavetablePatch::from_fn(|_| 0.0, adsr));
        assert_eq!(patch.synth_type(), crate::patch::SynthType::Wavetable);
    }

    #[test]
    fn test_additive_strings_harmonic_rolloff() {
        let patch = AdditivePatch::strings();
        // Harmonic amplitudes must follow 1/n rolloff: h[0]=1.0, h[1]=0.5, h[2]=0.333...
        assert!((patch.harmonics[0] - 1.0).abs() < 0.001);
        assert!((patch.harmonics[1] - 0.5).abs() < 0.001);
        assert!((patch.harmonics[2] - (1.0/3.0)).abs() < 0.001);
    }

    #[test]
    fn test_additive_organ_odd_harmonics() {
        let patch = AdditivePatch::organ();
        // Even-indexed positions (harmonics 2,4,6,...) should be zero
        assert_eq!(patch.harmonics[1], 0.0, "2nd harmonic should be 0");
        assert_eq!(patch.harmonics[3], 0.0, "4th harmonic should be 0");
        assert_eq!(patch.harmonics[5], 0.0, "6th harmonic should be 0");
    }

    #[test]
    fn test_fm_bell_patch_operators() {
        let patch = FmPatch::bell();
        assert_eq!(patch.operators[0].ratio, 1.0);
        assert!((patch.operators[1].ratio - 3.5).abs() < 0.001);
        assert!(patch.operators[1].mod_index > 0.0);
        // Unused operators should have zero level
        assert_eq!(patch.operators[2].level, 0.0);
        assert_eq!(patch.operators[3].level, 0.0);
    }

    #[test]
    fn test_subtractive_pluck_patch() {
        let patch = SubtractivePatch::pluck();
        assert_eq!(patch.waveform, Waveform::Square);
        assert!(patch.resonance >= 0.0 && patch.resonance < 1.0);
        assert!(patch.cutoff_hz > 0.0);
        assert_eq!(patch.amp_envelope.sustain, 0.0);
    }

    #[test]
    fn test_wavetable_from_fn_sine() {
        let adsr = crate::envelope::Adsr::organ(44100.0);
        let wt = WavetablePatch::from_fn(
            |phase| (phase * 2.0 * PI).sin(),
            adsr,
        );
        // Verify that all 256 samples are within [-1, 1]
        for &s in wt.table.iter() {
            assert!(s >= -1.001 && s <= 1.001, "wavetable sample out of range: {s}");
        }
    }

    #[test]
    fn test_wavetable_lookup_phase_zero() {
        let adsr = crate::envelope::Adsr::organ(44100.0);
        let wt = WavetablePatch::from_fn(|_| 0.5, adsr);
        // Constant wavetable: lookup at any phase must return 0.5
        let v0 = wt.lookup(0.0);
        let v1 = wt.lookup(0.5);
        let v2 = wt.lookup(0.999);
        assert!((v0 - 0.5).abs() < 0.001, "lookup(0.0) should be 0.5, got {v0}");
        assert!((v1 - 0.5).abs() < 0.001, "lookup(0.5) should be 0.5, got {v1}");
        assert!((v2 - 0.5).abs() < 0.001, "lookup(0.999) should be 0.5, got {v2}");
    }

    #[test]
    fn test_wavetable_interpolation_midpoint() {
        let adsr = crate::envelope::Adsr::organ(44100.0);
        // Ramp from 0 to 1 across the table
        let wt = WavetablePatch::from_fn(|phase| phase * 2.0 - 1.0, adsr);
        // At phase=0.5 (midpoint), value should be ~0.0
        let v = wt.lookup(0.5);
        assert!(v.abs() < 0.05, "midpoint ramp wavetable lookup should be near 0, got {v}");
    }
}
