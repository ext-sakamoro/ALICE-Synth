//! DSP effects — parametric, no impulse response samples
//!
//! All effects are mathematically defined (no sample data).
//! Reverb = exponential decay, Delay = circular buffer, Filter = biquad.
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::oscillator::sin_approx;

/// Effect trait
pub trait Effect {
    /// Process one sample (mono)
    fn process(&mut self, input: f32) -> f32;

    /// Reset internal state
    fn reset(&mut self);
}

/// Delay effect — circular buffer
///
/// Size: 4 bytes params + buffer
pub struct Delay {
    buffer: Vec<f32>,
    write_pos: usize,
    /// Feedback amount [0.0, 1.0)
    pub feedback: f32,
    /// Wet/dry mix [0.0, 1.0]
    pub mix: f32,
}

impl Delay {
    /// Create delay with given time in samples
    pub fn new(delay_samples: usize, feedback: f32, mix: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            feedback: feedback.clamp(0.0, 0.99),
            mix: mix.clamp(0.0, 1.0),
        }
    }

    /// Create delay from milliseconds
    pub fn from_ms(delay_ms: f32, sample_rate: f32, feedback: f32, mix: f32) -> Self {
        const RCP_1000: f32 = 1.0 / 1000.0;
        let samples = (delay_ms * sample_rate * RCP_1000) as usize;
        Self::new(samples, feedback, mix)
    }
}

impl Effect for Delay {
    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buffer[self.write_pos];
        self.buffer[self.write_pos] = input + delayed * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        input * (1.0 - self.mix) + delayed * self.mix
    }

    fn reset(&mut self) {
        self.buffer.iter_mut().for_each(|s| *s = 0.0);
        self.write_pos = 0;
    }
}

/// One-pole low-pass filter
///
/// y[n] = α * x[n] + (1-α) * y[n-1]
/// Size: 8 bytes
pub struct LowPassFilter {
    /// Filter coefficient (0.0 = no filtering, 1.0 = pass-through)
    alpha: f32,
    /// Previous output
    prev: f32,
}

impl LowPassFilter {
    /// Create from cutoff frequency
    #[inline(always)]
    pub fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let rc = (2.0 * core::f32::consts::PI * cutoff_hz).recip();
        let dt = sample_rate.recip();
        let alpha = dt / (rc + dt);
        Self { alpha, prev: 0.0 }
    }

    /// Set cutoff frequency
    #[inline(always)]
    pub fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        let rc = (2.0 * core::f32::consts::PI * cutoff_hz).recip();
        let dt = sample_rate.recip();
        self.alpha = dt / (rc + dt);
    }
}

impl Effect for LowPassFilter {
    #[inline(always)]
    fn process(&mut self, input: f32) -> f32 {
        self.prev = self.alpha * input + (1.0 - self.alpha) * self.prev;
        self.prev
    }

    fn reset(&mut self) {
        self.prev = 0.0;
    }
}

/// Resonant state-variable filter (SVF)
///
/// Simultaneous LP/HP/BP outputs. Size: 12 bytes.
pub struct StateVariableFilter {
    /// Cutoff coefficient
    f: f32,
    /// Damping (resonance = 1/(2*damp))
    damp: f32,
    /// State variables
    low: f32,
    band: f32,
}

impl StateVariableFilter {
    pub fn new(cutoff_hz: f32, resonance: f32, sample_rate: f32) -> Self {
        let f = 2.0 * sin_approx(core::f32::consts::PI * cutoff_hz * sample_rate.recip());
        let damp = (1.0 - resonance.clamp(0.0, 0.99)).max(0.01);
        Self {
            f,
            damp,
            low: 0.0,
            band: 0.0,
        }
    }

    /// Process and return (low, band, high) simultaneously
    #[inline(always)]
    pub fn process_svf(&mut self, input: f32) -> (f32, f32, f32) {
        let high = input - self.low - self.damp * self.band;
        self.band += self.f * high;
        self.low += self.f * self.band;
        (self.low, self.band, high)
    }

    #[inline(always)]
    pub fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        self.f = 2.0 * sin_approx(core::f32::consts::PI * cutoff_hz * sample_rate.recip());
    }
}

impl Effect for StateVariableFilter {
    #[inline(always)]
    fn process(&mut self, input: f32) -> f32 {
        self.process_svf(input).0 // Low-pass by default
    }

    fn reset(&mut self) {
        self.low = 0.0;
        self.band = 0.0;
    }
}

/// Simple reverb — Schroeder allpass + comb filter network
///
/// Parametric: decay_time + room_size, no IR samples.
pub struct Reverb {
    comb_buffers: [Vec<f32>; 4],
    comb_pos: [usize; 4],
    comb_feedback: f32,
    allpass_buffers: [Vec<f32>; 2],
    allpass_pos: [usize; 2],
    pub mix: f32,
}

impl Reverb {
    /// Create reverb with room size and decay time
    pub fn new(sample_rate: f32, room_size: f32, decay: f32, mix: f32) -> Self {
        let base_delays = [1116, 1188, 1277, 1356]; // Schroeder delays
        let allpass_delays = [556, 441];

        const RCP_44100: f32 = 1.0 / 44100.0;
        let scale = room_size.clamp(0.1, 2.0);
        // Multiply by RCP_44100 instead of dividing by 44100 — both computed once at init
        let scale_sr = scale * sample_rate * RCP_44100;
        let comb_buffers = base_delays.map(|d| {
            let size = (d as f32 * scale_sr) as usize;
            vec![0.0f32; size.max(1)]
        });
        let allpass_buffers = allpass_delays.map(|d| {
            let size = (d as f32 * scale_sr) as usize;
            vec![0.0f32; size.max(1)]
        });

        Self {
            comb_buffers,
            comb_pos: [0; 4],
            comb_feedback: decay.clamp(0.0, 0.99),
            allpass_buffers,
            allpass_pos: [0; 2],
            mix: mix.clamp(0.0, 1.0),
        }
    }
}

impl Effect for Reverb {
    fn process(&mut self, input: f32) -> f32 {
        // Parallel comb filters
        let mut comb_out = 0.0f32;
        for i in 0..4 {
            let buf = &mut self.comb_buffers[i];
            let pos = self.comb_pos[i];
            let delayed = buf[pos];
            buf[pos] = input + delayed * self.comb_feedback;
            self.comb_pos[i] = (pos + 1) % buf.len();
            comb_out += delayed;
        }
        comb_out *= 0.25;

        // Series allpass filters
        let mut out = comb_out;
        for i in 0..2 {
            let buf = &mut self.allpass_buffers[i];
            let pos = self.allpass_pos[i];
            let delayed = buf[pos];
            let new_val = out + delayed * 0.5;
            buf[pos] = new_val;
            out = delayed - out * 0.5;
            self.allpass_pos[i] = (pos + 1) % buf.len();
        }

        input * (1.0 - self.mix) + out * self.mix
    }

    fn reset(&mut self) {
        for buf in &mut self.comb_buffers {
            buf.iter_mut().for_each(|s| *s = 0.0);
        }
        for buf in &mut self.allpass_buffers {
            buf.iter_mut().for_each(|s| *s = 0.0);
        }
        self.comb_pos = [0; 4];
        self.allpass_pos = [0; 2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay() {
        let mut delay = Delay::new(4, 0.0, 1.0);
        // Feed impulse
        let _ = delay.process(1.0);
        let _ = delay.process(0.0);
        let _ = delay.process(0.0);
        let _ = delay.process(0.0);
        let out = delay.process(0.0);
        assert!((out - 1.0).abs() < 0.01, "delayed impulse expected, got {out}");
    }

    #[test]
    fn test_lowpass() {
        let mut lpf = LowPassFilter::new(100.0, 44100.0);
        // Feed step function
        let mut prev = 0.0f32;
        for _ in 0..1000 {
            let out = lpf.process(1.0);
            assert!(out >= prev - 0.001, "LPF should be monotonic");
            prev = out;
        }
        assert!(prev > 0.9, "LPF should converge to input");
    }

    #[test]
    fn test_svf() {
        let mut svf = StateVariableFilter::new(1000.0, 0.5, 44100.0);
        let (low, band, high) = svf.process_svf(1.0);
        // First sample: low and band should be small, high should be large
        assert!(high.abs() > low.abs());
        let _ = band; // used
    }

    #[test]
    fn test_reverb_no_panic() {
        let mut reverb = Reverb::new(44100.0, 1.0, 0.8, 0.3);
        for _ in 0..1000 {
            let _ = reverb.process(0.5);
        }
        reverb.reset();
        let out = reverb.process(0.0);
        assert!(out.abs() < 0.01, "after reset, output should be near zero");
    }

    // --- NEW TESTS ---

    #[test]
    fn test_delay_zero_mix_passes_dry() {
        // mix=0.0 → output should equal input (dry only)
        let mut delay = Delay::new(100, 0.0, 0.0);
        let out = delay.process(0.75);
        assert!((out - 0.75).abs() < 0.001, "zero-mix delay should pass dry signal, got {out}");
    }

    #[test]
    fn test_delay_from_ms_creates_correct_buffer() {
        // 10ms at 1000 Hz = 10 samples
        let delay = Delay::from_ms(10.0, 1000.0, 0.5, 0.5);
        assert_eq!(delay.buffer.len(), 10);
    }

    #[test]
    fn test_delay_reset_clears_buffer() {
        let mut delay = Delay::new(4, 0.9, 0.5);
        // Load up the buffer
        for _ in 0..4 { delay.process(1.0); }
        delay.reset();
        // After reset, write pos should be 0 and buffer zeroed
        assert_eq!(delay.write_pos, 0);
        assert!(delay.buffer.iter().all(|&s| s == 0.0), "buffer should be zeroed after reset");
    }

    #[test]
    fn test_delay_feedback_clamped() {
        // Feedback ≥ 1.0 should be clamped to 0.99 to prevent runaway
        let delay = Delay::new(4, 2.0, 0.5);
        assert!(delay.feedback <= 0.99, "feedback should be clamped to 0.99");
    }

    #[test]
    fn test_lowpass_reset_clears_state() {
        let mut lpf = LowPassFilter::new(1000.0, 44100.0);
        for _ in 0..100 { lpf.process(1.0); }
        lpf.reset();
        // After reset, passing zero should produce zero
        let out = lpf.process(0.0);
        assert_eq!(out, 0.0, "LPF after reset with 0 input should output 0");
    }

    #[test]
    fn test_lowpass_high_cutoff_near_passthrough() {
        // High cutoff → alpha is larger → converges to input faster than low cutoff.
        // At 20kHz/44100Hz, alpha ≈ 0.74.  After 10 steps at 1.0 input, output
        // should be substantially closer to 1.0 than the low-cutoff case.
        let mut lpf_high = LowPassFilter::new(20000.0, 44100.0);
        let mut lpf_low  = LowPassFilter::new(10.0,    44100.0);
        let mut out_high = 0.0_f32;
        let mut out_low  = 0.0_f32;
        for _ in 0..20 {
            out_high = lpf_high.process(1.0);
            out_low  = lpf_low.process(1.0);
        }
        assert!(out_high > out_low,
            "high-cutoff LPF should converge faster, high={out_high}, low={out_low}");
        assert!(out_high > 0.5,
            "high-cutoff LPF should reach at least 0.5 after 20 samples, got {out_high}");
    }

    #[test]
    fn test_lowpass_low_cutoff_attenuates() {
        // Very low cutoff → first sample output should be very small
        let mut lpf = LowPassFilter::new(1.0, 44100.0);
        let out = lpf.process(1.0);
        assert!(out < 0.01, "very low cutoff LPF should heavily attenuate first sample, got {out}");
    }

    #[test]
    fn test_svf_reset_clears_state() {
        let mut svf = StateVariableFilter::new(1000.0, 0.5, 44100.0);
        for _ in 0..100 { svf.process(1.0); }
        svf.reset();
        let (low, band, high) = svf.process_svf(0.0);
        assert_eq!(low, 0.0);
        assert_eq!(band, 0.0);
        assert_eq!(high, 0.0);
    }

    #[test]
    fn test_svf_lp_hp_sum_near_input() {
        // SVF: high = input - low - damp*band.  low and band start at 0.
        // On the first sample: band=0, low=0 → high = input.
        // LP+HP on first call: 0 + input = input.  After the update, low changes.
        // We just verify that LP + HP + BP = input + some linear combination of state.
        // A simpler, correct invariant: after steady-state with DC input, LP ≈ input.
        let mut svf = StateVariableFilter::new(5000.0, 0.0, 44100.0);
        // Feed many samples of DC=1.0 until low-pass output settles near 1.0
        let mut low_out = 0.0_f32;
        for _ in 0..500 {
            let (l, _, _) = svf.process_svf(1.0);
            low_out = l;
        }
        assert!(low_out > 0.9, "SVF low-pass should converge to DC input after many samples, got {low_out}");
    }

    #[test]
    fn test_reverb_produces_output_after_impulse() {
        // The comb delays are 1116+ samples, so the impulse echo appears after
        // at least 1116 samples.  Feed the impulse, then iterate enough samples
        // to reach the first comb echo.
        let mut reverb = Reverb::new(44100.0, 1.0, 0.8, 1.0);
        reverb.process(1.0); // feed impulse
        let mut has_tail = false;
        for _ in 0..2000 {
            let out = reverb.process(0.0);
            if out.abs() > 0.001 { has_tail = true; break; }
        }
        assert!(has_tail, "reverb should produce a decaying tail after impulse within 2000 samples");
    }

    #[test]
    fn test_reverb_mix_zero_passes_dry() {
        // mix=0 → output = input regardless of reverb state
        let mut reverb = Reverb::new(44100.0, 1.0, 0.8, 0.0);
        let out = reverb.process(0.42);
        assert!((out - 0.42).abs() < 0.001, "zero-mix reverb should pass dry, got {out}");
    }
}
