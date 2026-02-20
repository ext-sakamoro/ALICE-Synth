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
}
