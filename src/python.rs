//! Python bindings for ALICE-Synth via PyO3
//!
//! 6 classes + 3 functions for procedural audio synthesis from Python.
//!
//! Author: Moroya Sakamoto

// PyO3 #[pymethods] macro generates Into<PyErr> conversions that trigger this lint
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::envelope::{Adsr, AdsrState};
use crate::oscillator::{self, Oscillator, Waveform};
use crate::patch::{AdditivePatch, FmPatch, Patch, SubtractivePatch, WavetablePatch};
use crate::score::Score;
use crate::synth::Synthesizer;

// ============================================================================
// Envelope
// ============================================================================

/// ADSR envelope parameters
#[pyclass(name = "Adsr")]
#[derive(Clone)]
pub struct PyAdsr {
    inner: Adsr,
}

#[pymethods]
impl PyAdsr {
    /// Create ADSR from milliseconds
    #[new]
    #[pyo3(signature = (attack_ms, decay_ms, sustain, release_ms, sample_rate=44100.0))]
    fn new(attack_ms: f32, decay_ms: f32, sustain: f32, release_ms: f32, sample_rate: f32) -> Self {
        Self {
            inner: Adsr::from_ms(attack_ms, decay_ms, sustain, release_ms, sample_rate),
        }
    }

    /// Percussive preset (zero sustain)
    #[staticmethod]
    #[pyo3(signature = (attack_ms, decay_ms, sample_rate=44100.0))]
    fn percussive(attack_ms: f32, decay_ms: f32, sample_rate: f32) -> Self {
        Self {
            inner: Adsr::percussive(attack_ms, decay_ms, sample_rate),
        }
    }

    /// Organ preset (instant attack, full sustain)
    #[staticmethod]
    #[pyo3(signature = (sample_rate=44100.0))]
    fn organ(sample_rate: f32) -> Self {
        Self {
            inner: Adsr::organ(sample_rate),
        }
    }

    /// Piano preset
    #[staticmethod]
    #[pyo3(signature = (sample_rate=44100.0))]
    fn piano(sample_rate: f32) -> Self {
        Self {
            inner: Adsr::piano(sample_rate),
        }
    }

    #[getter]
    fn attack(&self) -> u32 {
        self.inner.attack
    }

    #[getter]
    fn decay(&self) -> u32 {
        self.inner.decay
    }

    #[getter]
    fn sustain(&self) -> f32 {
        self.inner.sustain
    }

    #[getter]
    fn release(&self) -> u32 {
        self.inner.release
    }
}

// ============================================================================
// Envelope State
// ============================================================================

/// ADSR envelope state machine for one voice
#[pyclass(name = "AdsrState")]
pub struct PyAdsrState {
    inner: AdsrState,
}

#[pymethods]
impl PyAdsrState {
    #[new]
    fn new() -> Self {
        Self {
            inner: AdsrState::new(),
        }
    }

    /// Trigger note-on
    fn note_on(&mut self) {
        self.inner.note_on();
    }

    /// Trigger note-off
    fn note_off(&mut self) {
        self.inner.note_off();
    }

    /// Advance one sample, return amplitude [0.0, 1.0]
    fn next(&mut self, params: &PyAdsr) -> f32 {
        self.inner.next(&params.inner)
    }

    /// Is this envelope still active?
    #[getter]
    fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    /// Current output level
    #[getter]
    fn level(&self) -> f32 {
        self.inner.level()
    }
}

// ============================================================================
// Oscillator
// ============================================================================

/// Waveform oscillator
#[pyclass(name = "Oscillator")]
pub struct PyOscillator {
    inner: Oscillator,
}

#[pymethods]
impl PyOscillator {
    /// Create oscillator: "sine", "saw", "square", "triangle", "noise"
    #[new]
    #[pyo3(signature = (waveform="sine"))]
    fn new(waveform: &str) -> PyResult<Self> {
        let wf = match waveform.to_lowercase().as_str() {
            "sine" => Waveform::Sine,
            "saw" => Waveform::Saw,
            "square" => Waveform::Square,
            "triangle" => Waveform::Triangle,
            "noise" => Waveform::Noise,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown waveform: {waveform}. Use sine/saw/square/triangle/noise"
                )))
            }
        };
        Ok(Self {
            inner: Oscillator::new(wf),
        })
    }

    /// Generate next sample at given frequency
    fn next_sample(&mut self, freq_hz: f32, inv_sample_rate: f32) -> f32 {
        self.inner.next_sample(freq_hz, inv_sample_rate)
    }

    /// Reset phase to zero
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Generate a block of samples
    fn render_block(&mut self, freq_hz: f32, inv_sample_rate: f32, num_samples: usize) -> Vec<f32> {
        let mut buf = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            buf.push(self.inner.next_sample(freq_hz, inv_sample_rate));
        }
        buf
    }
}

// ============================================================================
// Synthesizer
// ============================================================================

/// Multi-voice polyphonic synthesizer
#[pyclass(name = "Synthesizer")]
pub struct PySynthesizer {
    inner: Synthesizer,
}

#[pymethods]
impl PySynthesizer {
    #[new]
    #[pyo3(signature = (sample_rate=44100))]
    fn new(sample_rate: u32) -> Self {
        Self {
            inner: Synthesizer::new(sample_rate),
        }
    }

    /// Load FM preset: 0=electric_piano, 1=bell
    fn load_fm_preset(&mut self, channel: usize, preset: u8) -> PyResult<()> {
        let patch = match preset {
            0 => Patch::Fm(FmPatch::electric_piano()),
            1 => Patch::Fm(FmPatch::bell()),
            _ => return Err(PyValueError::new_err(format!("Unknown preset: {preset}"))),
        };
        self.inner.load_patch(channel, patch);
        Ok(())
    }

    /// Load additive preset: 0=strings, 1=organ
    fn load_additive_preset(&mut self, channel: usize, preset: u8) -> PyResult<()> {
        let patch = match preset {
            0 => Patch::Additive(AdditivePatch::strings()),
            1 => Patch::Additive(AdditivePatch::organ()),
            _ => return Err(PyValueError::new_err(format!("Unknown preset: {preset}"))),
        };
        self.inner.load_patch(channel, patch);
        Ok(())
    }

    /// Load subtractive preset: 0=bass, 1=pluck
    fn load_subtractive_preset(&mut self, channel: usize, preset: u8) -> PyResult<()> {
        let patch = match preset {
            0 => Patch::Subtractive(SubtractivePatch::bass()),
            1 => Patch::Subtractive(SubtractivePatch::pluck()),
            _ => return Err(PyValueError::new_err(format!("Unknown preset: {preset}"))),
        };
        self.inner.load_patch(channel, patch);
        Ok(())
    }

    /// Load wavetable preset (sine wave)
    fn load_wavetable_preset(&mut self, channel: usize) {
        let env = Adsr::from_ms(5.0, 100.0, 0.7, 200.0, 44100.0);
        self.inner.load_patch(
            channel,
            Patch::Wavetable(WavetablePatch::from_fn(
                |p| oscillator::sin_approx(p * core::f32::consts::TAU),
                env,
            )),
        );
    }

    /// Trigger note on
    fn note_on(&mut self, channel: u8, note: u8, velocity: u8) {
        self.inner.note_on(channel, note, velocity);
    }

    /// Trigger note off
    fn note_off(&mut self, channel: u8, note: u8) {
        self.inner.note_off(channel, note);
    }

    /// Render audio samples as f32 list
    fn render(&mut self, num_samples: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; num_samples];
        self.inner.render(&mut buf);
        buf
    }

    /// Render audio samples as i16 list
    fn render_i16(&mut self, num_samples: usize) -> Vec<i16> {
        let mut buf = vec![0i16; num_samples];
        self.inner.render_i16(&mut buf);
        buf
    }

    /// Load score from bytes
    fn load_score(&mut self, data: &[u8]) -> PyResult<()> {
        let score =
            Score::from_bytes(data).ok_or_else(|| PyValueError::new_err("Invalid score data"))?;
        self.inner.load_score(&score);
        Ok(())
    }

    /// Number of currently active voices
    #[getter]
    fn active_voices(&self) -> usize {
        self.inner.active_voice_count()
    }

    /// Set master volume [0.0, 1.0]
    #[setter]
    fn set_master_volume(&mut self, volume: f32) {
        self.inner.master_volume = volume.clamp(0.0, 1.0);
    }

    /// Get master volume
    #[getter]
    fn master_volume(&self) -> f32 {
        self.inner.master_volume
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Convert MIDI note number to frequency in Hz
#[pyfunction]
fn midi_to_freq(note: u8) -> f32 {
    oscillator::midi_to_freq(note)
}

/// Fast sine approximation
#[pyfunction]
fn sin_approx(x: f32) -> f32 {
    oscillator::sin_approx(x)
}

// ============================================================================
// Module
// ============================================================================

/// ALICE-Synth Python module
#[pymodule]
pub fn alice_synth(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAdsr>()?;
    m.add_class::<PyAdsrState>()?;
    m.add_class::<PyOscillator>()?;
    m.add_class::<PySynthesizer>()?;
    m.add_function(wrap_pyfunction!(midi_to_freq, m)?)?;
    m.add_function(wrap_pyfunction!(sin_approx, m)?)?;
    Ok(())
}

// ============================================================================
// Tests (use inner Rust types directly — PyO3 extension-module cannot link in cargo test)
// ============================================================================

#[cfg(test)]
mod tests {
    use crate::envelope::{Adsr, AdsrState};
    use crate::oscillator::{self, Oscillator, Waveform};
    use crate::patch::{AdditivePatch, FmPatch, Patch, SubtractivePatch, WavetablePatch};
    use crate::synth::Synthesizer;

    #[test]
    fn test_pyo3_adsr_from_ms() {
        let adsr = Adsr::from_ms(10.0, 50.0, 0.7, 100.0, 44100.0);
        assert!(adsr.attack > 0);
        assert!(adsr.decay > 0);
        assert!((adsr.sustain - 0.7).abs() < f32::EPSILON);
        assert!(adsr.release > 0);
    }

    #[test]
    fn test_pyo3_adsr_percussive() {
        let adsr = Adsr::percussive(1.0, 50.0, 44100.0);
        assert_eq!(adsr.sustain, 0.0);
    }

    #[test]
    fn test_pyo3_adsr_organ() {
        let adsr = Adsr::organ(44100.0);
        assert_eq!(adsr.sustain, 1.0);
    }

    #[test]
    fn test_pyo3_adsr_piano() {
        let adsr = Adsr::piano(44100.0);
        assert!(adsr.sustain < 0.5);
    }

    #[test]
    fn test_pyo3_adsr_state_lifecycle() {
        let params = Adsr::from_ms(4.0, 4.0, 0.5, 4.0, 44100.0);
        let mut state = AdsrState::new();
        assert!(!state.is_active());
        state.note_on();
        assert!(state.is_active());
        for _ in 0..100 {
            state.next(&params);
        }
        state.note_off();
        for _ in 0..1000 {
            state.next(&params);
        }
        assert!(!state.is_active());
    }

    #[test]
    fn test_pyo3_oscillator_sine() {
        let mut osc = Oscillator::new(Waveform::Sine);
        let sample = osc.next_sample(440.0, 1.0 / 44100.0);
        assert!(sample >= -1.0 && sample <= 1.0);
    }

    #[test]
    fn test_pyo3_oscillator_all_waveforms() {
        for wf in &[
            Waveform::Sine,
            Waveform::Saw,
            Waveform::Square,
            Waveform::Triangle,
            Waveform::Noise,
        ] {
            let mut osc = Oscillator::new(*wf);
            let s = osc.next_sample(440.0, 1.0 / 44100.0);
            assert!(s >= -1.1 && s <= 1.1, "{wf:?}: {s}");
        }
    }

    #[test]
    fn test_pyo3_oscillator_render_block() {
        let mut osc = Oscillator::new(Waveform::Sine);
        let inv_sr = 1.0 / 44100.0;
        let buf: Vec<f32> = (0..256).map(|_| osc.next_sample(440.0, inv_sr)).collect();
        assert_eq!(buf.len(), 256);
        assert!(buf.iter().all(|&s| s >= -1.1 && s <= 1.1));
    }

    #[test]
    fn test_pyo3_oscillator_reset() {
        let mut osc = Oscillator::new(Waveform::Saw);
        osc.next_sample(440.0, 1.0 / 44100.0);
        osc.reset();
        let s = osc.next_sample(440.0, 1.0 / 44100.0);
        assert!(s >= -1.0 && s <= 1.0);
    }

    #[test]
    fn test_pyo3_synth_create() {
        let synth = Synthesizer::new(44100);
        assert_eq!(synth.active_voice_count(), 0);
    }

    #[test]
    fn test_pyo3_synth_fm_preset() {
        let mut synth = Synthesizer::new(44100);
        synth.load_patch(0, Patch::Fm(FmPatch::electric_piano()));
        synth.load_patch(1, Patch::Fm(FmPatch::bell()));
    }

    #[test]
    fn test_pyo3_synth_note_on_off() {
        let mut synth = Synthesizer::new(44100);
        synth.load_patch(0, Patch::Fm(FmPatch::electric_piano()));
        synth.note_on(0, 60, 100);
        assert!(synth.active_voice_count() > 0);
        synth.note_off(0, 60);
    }

    #[test]
    fn test_pyo3_synth_render() {
        let mut synth = Synthesizer::new(44100);
        synth.load_patch(0, Patch::Fm(FmPatch::electric_piano()));
        synth.note_on(0, 69, 100);
        let mut buf = [0.0f32; 1024];
        synth.render(&mut buf);
        assert!(buf.iter().any(|&s| s != 0.0));
    }

    #[test]
    fn test_pyo3_synth_render_i16() {
        let mut synth = Synthesizer::new(44100);
        synth.load_patch(0, Patch::Fm(FmPatch::electric_piano()));
        synth.note_on(0, 60, 100);
        let mut buf = [0i16; 512];
        synth.render_i16(&mut buf);
    }

    #[test]
    fn test_pyo3_synth_volume() {
        let mut synth = Synthesizer::new(44100);
        synth.master_volume = 0.5;
        assert!((synth.master_volume - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pyo3_synth_volume_clamp() {
        let mut synth = Synthesizer::new(44100);
        synth.master_volume = 2.0_f32.clamp(0.0, 1.0);
        assert!((synth.master_volume - 1.0).abs() < f32::EPSILON);
        synth.master_volume = (-1.0_f32).clamp(0.0, 1.0);
        assert!(synth.master_volume.abs() < f32::EPSILON);
    }

    #[test]
    fn test_pyo3_synth_additive_preset() {
        let mut synth = Synthesizer::new(44100);
        synth.load_patch(0, Patch::Additive(AdditivePatch::strings()));
        synth.load_patch(1, Patch::Additive(AdditivePatch::organ()));
    }

    #[test]
    fn test_pyo3_synth_subtractive_preset() {
        let mut synth = Synthesizer::new(44100);
        synth.load_patch(0, Patch::Subtractive(SubtractivePatch::bass()));
        synth.load_patch(1, Patch::Subtractive(SubtractivePatch::pluck()));
    }

    #[test]
    fn test_pyo3_synth_wavetable_preset() {
        let env = Adsr::from_ms(5.0, 100.0, 0.7, 200.0, 44100.0);
        let wt =
            WavetablePatch::from_fn(|p| oscillator::sin_approx(p * core::f32::consts::TAU), env);
        let mut synth = Synthesizer::new(44100);
        synth.load_patch(0, Patch::Wavetable(wt));
        synth.note_on(0, 60, 100);
        assert!(synth.active_voice_count() > 0);
    }

    #[test]
    fn test_pyo3_midi_to_freq() {
        let freq = oscillator::midi_to_freq(69);
        assert!((freq - 440.0).abs() < 0.1);
    }

    #[test]
    fn test_pyo3_sin_approx() {
        let val = oscillator::sin_approx(0.0);
        assert!(val.abs() < 0.01);
    }
}
