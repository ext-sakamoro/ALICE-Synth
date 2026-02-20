//! Core oscillator — phase-accumulator waveform generators
//!
//! All oscillators use a 0.0..1.0 phase accumulator and
//! produce output in the -1.0..1.0 range.
//!
//! Author: Moroya Sakamoto

use core::f32::consts::PI;

/// 2*pi — avoid repeated 2.0 * PI multiplications in hot paths
const TWO_PI: f32 = 2.0 * PI;

/// Reciprocal of 32768 — noise normalization, avoids per-sample division
const RCP_32768: f32 = 1.0 / 32768.0;

/// Reciprocal of 12 — used in midi_to_freq semitone conversion
const RCP_12: f32 = 1.0 / 12.0;

/// no_std-compatible floor function
#[inline(always)]
fn floor_f32(x: f32) -> f32 {
    let i = x as i32;
    let f = i as f32;
    if x < f { f - 1.0 } else { f }
}

/// Waveform type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Waveform {
    Sine = 0,
    Saw = 1,
    Square = 2,
    Triangle = 3,
    Noise = 4,
}

impl Waveform {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Sine,
            1 => Self::Saw,
            2 => Self::Square,
            3 => Self::Triangle,
            4 => Self::Noise,
            _ => Self::Sine,
        }
    }
}

/// Phase-accumulator oscillator
///
/// Generates waveforms at arbitrary frequency with minimal state.
/// Total state: 8 bytes (phase f32 + noise_state u32).
pub struct Oscillator {
    /// Current phase [0.0, 1.0)
    phase: f32,
    /// Waveform type
    pub waveform: Waveform,
    /// LFSR state for noise generation
    noise_state: u32,
}

impl Oscillator {
    pub fn new(waveform: Waveform) -> Self {
        Self {
            phase: 0.0,
            waveform,
            noise_state: 0xACE1u32,
        }
    }

    /// Reset phase to zero
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Generate next sample at given frequency
    ///
    /// `freq_hz`: oscillator frequency in Hz
    /// `inv_sample_rate`: pre-computed `1.0 / sample_rate` — caller must hoist
    ///   this outside any per-sample loop to eliminate the division from the hot path
    ///
    /// Returns sample in [-1.0, 1.0]
    #[inline(always)]
    pub fn next_sample(&mut self, freq_hz: f32, inv_sample_rate: f32) -> f32 {
        let out = self.sample_at_phase(self.phase);
        self.phase += freq_hz * inv_sample_rate;
        // Wrap phase — branchless via fractional subtraction
        self.phase -= floor_f32(self.phase);
        out
    }

    /// Generate sample with phase modulation (for FM synthesis)
    ///
    /// `freq_hz`: oscillator frequency in Hz
    /// `inv_sample_rate`: pre-computed `1.0 / sample_rate`
    /// `phase_mod`: additional phase offset from modulator oscillator
    #[inline(always)]
    pub fn next_sample_fm(&mut self, freq_hz: f32, inv_sample_rate: f32, phase_mod: f32) -> f32 {
        let mod_phase = self.phase + phase_mod;
        let mod_phase = mod_phase - floor_f32(mod_phase);
        let out = self.sample_at_phase(mod_phase);
        self.phase += freq_hz * inv_sample_rate;
        self.phase -= floor_f32(self.phase);
        out
    }

    /// Evaluate waveform at given phase [0.0, 1.0)
    #[inline(always)]
    fn sample_at_phase(&mut self, phase: f32) -> f32 {
        match self.waveform {
            Waveform::Sine => sin_approx(phase * TWO_PI),
            Waveform::Saw => 2.0 * phase - 1.0,
            Waveform::Square => {
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Triangle => {
                if phase < 0.5 {
                    4.0 * phase - 1.0
                } else {
                    3.0 - 4.0 * phase
                }
            }
            Waveform::Noise => {
                // 16-bit LFSR noise; RCP_32768 replaces the / 32768.0 division
                let bit = ((self.noise_state >> 0) ^ (self.noise_state >> 2)
                    ^ (self.noise_state >> 3)
                    ^ (self.noise_state >> 5))
                    & 1;
                self.noise_state = (self.noise_state >> 1) | (bit << 15);
                (self.noise_state as f32 * RCP_32768) * 2.0 - 1.0
            }
        }
    }
}

/// Fast sine approximation (Bhaskara I, ~0.1% error)
///
/// Avoids libm dependency for no_std targets.
/// The rational division is unavoidable (one div per call), but TWO_PI
/// is a constant so no runtime multiply is needed for normalization.
#[inline(always)]
pub fn sin_approx(x: f32) -> f32 {
    // Normalize to [0, 2π)
    let x = x % TWO_PI;
    let x = if x < 0.0 { x + TWO_PI } else { x };

    // Map to [0, π] with sign
    let (x, sign) = if x > PI { (x - PI, -1.0) } else { (x, 1.0) };

    // Bhaskara I approximation: sin(x) ≈ 16x(π-x) / (5π² - 4x(π-x))
    let num = 16.0 * x * (PI - x);
    let den = 5.0 * PI * PI - 4.0 * x * (PI - x);
    sign * num / den
}

/// Convert MIDI note number to frequency (Hz)
///
/// A4 (note 69) = 440 Hz
/// RCP_12 replaces the / 12.0 division in semitone computation.
#[inline(always)]
pub fn midi_to_freq(note: u8) -> f32 {
    440.0 * pow2_approx((note as f32 - 69.0) * RCP_12)
}

/// Fast 2^x approximation for no_std
///
/// Uses integer bit manipulation + polynomial approximation.
#[inline(always)]
fn pow2_approx(x: f32) -> f32 {
    // 2^x = 2^int(x) * 2^frac(x)
    let floor = floor_f32(x);
    let frac = x - floor;
    let int_part = floor as i32;

    // 2^frac approximation (linear: good enough for ±6 semitones)
    let frac_approx = 1.0 + frac * (0.6931472 + frac * (0.2402265 + frac * 0.0558011));

    // 2^int via IEEE 754 exponent field manipulation
    let bits = ((127 + int_part) as u32) << 23;
    let int_pow = f32::from_bits(bits);

    int_pow * frac_approx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin_approx() {
        let result = sin_approx(PI / 2.0);
        assert!((result - 1.0).abs() < 0.002, "sin(π/2) ≈ 1.0, got {result}");

        let result = sin_approx(0.0);
        assert!(result.abs() < 0.001, "sin(0) ≈ 0.0, got {result}");

        let result = sin_approx(PI);
        assert!(result.abs() < 0.001, "sin(π) ≈ 0.0, got {result}");
    }

    #[test]
    fn test_oscillator_sine() {
        let mut osc = Oscillator::new(Waveform::Sine);
        let inv_sr = 1.0_f32 / 44100.0;
        let mut samples = [0.0f32; 100];
        for s in samples.iter_mut() {
            *s = osc.next_sample(440.0, inv_sr);
        }
        // First sample should be near zero (sin(0))
        assert!(samples[0].abs() < 0.1);
        // Should have values in [-1, 1]
        assert!(samples.iter().all(|&s| s >= -1.01 && s <= 1.01));
    }

    #[test]
    fn test_oscillator_saw() {
        let mut osc = Oscillator::new(Waveform::Saw);
        let inv_sr = 1.0_f32 / 4.0;
        let sample = osc.next_sample(1.0, inv_sr); // phase=0 → saw=-1
        assert!((sample - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_midi_to_freq() {
        let a4 = midi_to_freq(69);
        assert!((a4 - 440.0).abs() < 1.0, "A4 should be ~440 Hz, got {a4}");

        let a5 = midi_to_freq(81);
        assert!((a5 - 880.0).abs() < 5.0, "A5 should be ~880 Hz, got {a5}");
    }

    #[test]
    fn test_oscillator_noise() {
        let mut osc = Oscillator::new(Waveform::Noise);
        let inv_sr = 1.0_f32 / 44100.0;
        let mut samples = [0.0f32; 100];
        for s in samples.iter_mut() {
            *s = osc.next_sample(44100.0, inv_sr);
        }
        // Noise should have variety
        let min = samples.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 0.5, "noise should have spread");
    }
}
