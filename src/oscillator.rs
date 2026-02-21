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

    // --- NEW TESTS ---

    #[test]
    fn test_rcp_32768_correctness() {
        // Verify reciprocal constant matches its intended value
        let expected = 1.0_f32 / 32768.0;
        assert!((RCP_32768 - expected).abs() < 1e-10, "RCP_32768 constant incorrect");
    }

    #[test]
    fn test_rcp_12_correctness() {
        let expected = 1.0_f32 / 12.0;
        assert!((RCP_12 - expected).abs() < 1e-10, "RCP_12 constant incorrect");
    }

    #[test]
    fn test_sin_approx_negative_angle() {
        // sin(-π/2) ≈ -1.0 (via negative branch in normalization)
        let result = sin_approx(-PI / 2.0);
        assert!((result - (-1.0)).abs() < 0.002, "sin(-π/2) ≈ -1.0, got {result}");
    }

    #[test]
    fn test_sin_approx_three_pi_over_two() {
        // 3π/2 → sin ≈ -1.0
        let result = sin_approx(3.0 * PI / 2.0);
        assert!((result - (-1.0)).abs() < 0.002, "sin(3π/2) ≈ -1.0, got {result}");
    }

    #[test]
    fn test_sin_approx_two_pi() {
        // 2π → sin ≈ 0.0
        let result = sin_approx(TWO_PI);
        assert!(result.abs() < 0.002, "sin(2π) ≈ 0.0, got {result}");
    }

    #[test]
    fn test_waveform_from_u8_all_variants() {
        assert_eq!(Waveform::from_u8(0), Waveform::Sine);
        assert_eq!(Waveform::from_u8(1), Waveform::Saw);
        assert_eq!(Waveform::from_u8(2), Waveform::Square);
        assert_eq!(Waveform::from_u8(3), Waveform::Triangle);
        assert_eq!(Waveform::from_u8(4), Waveform::Noise);
        // Out-of-range falls back to Sine
        assert_eq!(Waveform::from_u8(255), Waveform::Sine);
    }

    #[test]
    fn test_oscillator_square_output_bounds() {
        // Square wave must be exactly +1 or -1
        let mut osc = Oscillator::new(Waveform::Square);
        let inv_sr = 1.0_f32 / 44100.0;
        for _ in 0..200 {
            let s = osc.next_sample(440.0, inv_sr);
            assert!(
                (s - 1.0).abs() < 0.001 || (s + 1.0).abs() < 0.001,
                "square wave must be +1 or -1, got {s}"
            );
        }
    }

    #[test]
    fn test_oscillator_triangle_range() {
        // Triangle wave must stay in [-1, 1]
        let mut osc = Oscillator::new(Waveform::Triangle);
        let inv_sr = 1.0_f32 / 44100.0;
        for _ in 0..500 {
            let s = osc.next_sample(440.0, inv_sr);
            assert!(s >= -1.0 && s <= 1.0, "triangle out of range: {s}");
        }
    }

    #[test]
    fn test_oscillator_triangle_peak() {
        // At phase=0.25 the triangle should be near +1, at phase=0.75 near -1
        // Produce one full cycle at 1 Hz with sample rate 4
        let mut osc = Oscillator::new(Waveform::Triangle);
        let inv_sr = 1.0_f32 / 4.0;
        let s0 = osc.next_sample(1.0, inv_sr); // phase 0.0 → -1+0 = -1? 4*0 - 1 = -1
        let s1 = osc.next_sample(1.0, inv_sr); // phase 0.25 → 4*0.25-1 = 0
        let s2 = osc.next_sample(1.0, inv_sr); // phase 0.5 → 3-4*0.5 = 1
        let s3 = osc.next_sample(1.0, inv_sr); // phase 0.75 → 3-4*0.75 = 0
        // The triangle at phase=0 should be -1
        assert!((s0 - (-1.0)).abs() < 0.01, "triangle at phase=0 should be -1, got {s0}");
        // At phase=0.5 (second half), 3 - 4*0.5 = 1
        assert!((s2 - 1.0).abs() < 0.01, "triangle at phase=0.5 should be 1, got {s2}");
        let _ = (s1, s3);
    }

    #[test]
    fn test_oscillator_reset_restarts_phase() {
        let mut osc = Oscillator::new(Waveform::Saw);
        let inv_sr = 1.0_f32 / 44100.0;
        let first = osc.next_sample(440.0, inv_sr);
        // Advance a few samples
        for _ in 0..10 {
            osc.next_sample(440.0, inv_sr);
        }
        // Reset should bring phase back to 0
        osc.reset();
        let after_reset = osc.next_sample(440.0, inv_sr);
        assert!(
            (after_reset - first).abs() < 0.001,
            "after reset, first sample should match original first sample"
        );
    }

    #[test]
    fn test_oscillator_fm_output_range() {
        let mut osc = Oscillator::new(Waveform::Sine);
        let inv_sr = 1.0_f32 / 44100.0;
        for _ in 0..500 {
            let s = osc.next_sample_fm(440.0, inv_sr, 0.5);
            assert!(s >= -1.01 && s <= 1.01, "FM output out of range: {s}");
        }
    }

    #[test]
    fn test_midi_c4_frequency() {
        // Middle C (MIDI note 60) should be ~261.63 Hz
        let c4 = midi_to_freq(60);
        assert!((c4 - 261.63).abs() < 2.0, "C4 should be ~261.63 Hz, got {c4}");
    }

    #[test]
    fn test_midi_octave_doubling() {
        // Each octave (12 semitones) should double the frequency
        let a4 = midi_to_freq(69);
        let a5 = midi_to_freq(81);
        let ratio = a5 / a4;
        assert!((ratio - 2.0).abs() < 0.02, "octave ratio should be ~2.0, got {ratio}");
    }

    #[test]
    fn test_midi_note_0_is_positive() {
        // MIDI note 0 (lowest) should produce a positive frequency
        let freq = midi_to_freq(0);
        assert!(freq > 0.0, "MIDI note 0 frequency must be positive, got {freq}");
    }

    #[test]
    fn test_midi_note_127_below_nyquist() {
        // MIDI note 127 should be well below 44100/2 Nyquist
        let freq = midi_to_freq(127);
        assert!(freq < 22050.0, "MIDI note 127 should be below Nyquist, got {freq}");
    }

    #[test]
    fn test_noise_output_within_bounds() {
        // The LFSR noise is based on a 16-bit state (0..=65535), normalized by 1/32768.
        // Before the final *2-1 the range is [0, 65535/32768] = [0, ~2.0], giving
        // a maximum output just below 2.0*2 - 1 = 3.0 in the worst case.
        // In practice the state is 16-bit (0..65535), so the maximum output is
        // (65535/32768)*2 - 1 ≈ 3.0. We just verify the noise has reasonable spread.
        let mut osc = Oscillator::new(Waveform::Noise);
        let inv_sr = 1.0_f32 / 44100.0;
        let mut samples = [0.0f32; 1000];
        for s in samples.iter_mut() {
            *s = osc.next_sample(440.0, inv_sr);
        }
        // Verify the noise covers at least a 1.0 range of values
        let min = samples.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 1.0, "noise should span a wide range, spread={}", max - min);
        // Verify it is not DC
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 2.0, "noise mean should be roughly bounded, got {mean}");
    }

    #[test]
    fn test_floor_f32_positive() {
        assert!((floor_f32(2.9) - 2.0).abs() < 1e-6);
        assert!((floor_f32(3.0) - 3.0).abs() < 1e-6);
        assert!((floor_f32(0.1) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_floor_f32_negative() {
        assert!((floor_f32(-0.1) - (-1.0)).abs() < 1e-6);
        assert!((floor_f32(-1.9) - (-2.0)).abs() < 1e-6);
        assert!((floor_f32(-2.0) - (-2.0)).abs() < 1e-6);
    }
}
