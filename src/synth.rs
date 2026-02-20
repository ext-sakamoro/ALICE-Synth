//! Synthesizer — multi-voice polyphonic engine with score playback
//!
//! Manages voices, patches, and sequencing. Renders audio
//! to f32 or i16 PCM buffers.
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::envelope::{Adsr, AdsrState};
use crate::oscillator::{midi_to_freq, Oscillator, Waveform};
use crate::patch::{AdditivePatch, FmPatch, Patch, SubtractivePatch, WavetablePatch};
use crate::score::{NoteEventKind, Score};
use crate::effects::{Effect, StateVariableFilter};

/// Maximum polyphony
const MAX_VOICES: usize = 64;
/// Maximum instrument channels
const MAX_CHANNELS: usize = 16;
/// Reciprocal of MIDI velocity range (127) — avoids per-note-on division
const RCP_127: f32 = 1.0 / 127.0;

/// Single voice state
pub struct Voice {
    /// Is this voice currently producing sound?
    active: bool,
    /// MIDI note number
    note: u8,
    /// Channel/instrument index
    channel: u8,
    /// Velocity [0.0, 1.0]
    velocity: f32,
    /// Primary oscillator
    osc: Oscillator,
    /// Modulator oscillator (for FM)
    mod_osc: Oscillator,
    /// Amplitude envelope
    amp_env: AdsrState,
    /// Modulator envelope (for FM)
    mod_env: AdsrState,
    /// Filter envelope (for subtractive)
    filter_env: AdsrState,
    /// Per-voice filter (for subtractive)
    filter: StateVariableFilter,
    /// Current frequency
    freq_hz: f32,
}

impl Voice {
    fn new() -> Self {
        Self {
            active: false,
            note: 0,
            channel: 0,
            velocity: 0.0,
            osc: Oscillator::new(Waveform::Sine),
            mod_osc: Oscillator::new(Waveform::Sine),
            amp_env: AdsrState::new(),
            mod_env: AdsrState::new(),
            filter_env: AdsrState::new(),
            filter: StateVariableFilter::new(20000.0, 0.0, 44100.0),
            freq_hz: 440.0,
        }
    }

    fn trigger(&mut self, note: u8, velocity: u8, channel: u8) {
        self.active = true;
        self.note = note;
        self.channel = channel;
        self.velocity = velocity as f32 * RCP_127;
        self.freq_hz = midi_to_freq(note);
        self.osc.reset();
        self.mod_osc.reset();
        self.amp_env.note_on();
        self.mod_env.note_on();
        self.filter_env.note_on();
    }

    fn release(&mut self) {
        self.amp_env.note_off();
        self.mod_env.note_off();
        self.filter_env.note_off();
    }
}

/// Polyphonic synthesizer
pub struct Synthesizer {
    /// Audio sample rate
    sample_rate: f32,
    /// Pre-computed reciprocal of sample_rate — eliminates per-sample divisions
    inv_sample_rate: f32,
    /// Voice pool
    voices: Vec<Voice>,
    /// Instrument patches per channel
    patches: Vec<Option<Patch>>,
    /// Master volume [0.0, 1.0]
    pub master_volume: f32,
    /// Score for playback
    score: Option<Score>,
    /// Current event index in score
    score_event_idx: usize,
    /// Tick accumulator for score playback
    tick_accum: f32,
    /// Samples per tick (computed from score header)
    samples_per_tick: f32,
    /// Pre-computed reciprocal of samples_per_tick — avoids per-sample division in sequencer
    inv_samples_per_tick: f32,
}

impl Synthesizer {
    pub fn new(sample_rate: u32) -> Self {
        let mut voices = Vec::with_capacity(MAX_VOICES);
        for _ in 0..MAX_VOICES {
            voices.push(Voice::new());
        }
        let mut patches = Vec::with_capacity(MAX_CHANNELS);
        for _ in 0..MAX_CHANNELS {
            patches.push(None);
        }
        let sr = sample_rate as f32;
        Self {
            sample_rate: sr,
            inv_sample_rate: sr.recip(),
            voices,
            patches,
            master_volume: 0.8,
            score: None,
            score_event_idx: 0,
            tick_accum: 0.0,
            samples_per_tick: 0.0,
            inv_samples_per_tick: 0.0,
        }
    }

    /// Load an instrument patch to a channel
    pub fn load_patch(&mut self, channel: usize, patch: Patch) {
        if channel < MAX_CHANNELS {
            self.patches[channel] = Some(patch);
        }
    }

    /// Load a score for playback
    pub fn load_score(&mut self, score: &Score) {
        let spt = score.header.samples_per_tick(self.sample_rate);
        self.samples_per_tick = spt;
        // Pre-compute reciprocal so advance_score multiplies instead of divides
        self.inv_samples_per_tick = if spt > 0.0 { spt.recip() } else { 0.0 };
        self.score = Some(score.clone());
        self.score_event_idx = 0;
        self.tick_accum = 0.0;
    }

    /// Trigger a note
    pub fn note_on(&mut self, channel: u8, note: u8, velocity: u8) {
        // Find free voice (or steal oldest)
        let voice_idx = self.find_free_voice();
        self.voices[voice_idx].trigger(note, velocity, channel);

        // Set oscillator waveform based on patch
        if let Some(Some(patch)) = self.patches.get(channel as usize) {
            match patch {
                Patch::Subtractive(sub) => {
                    self.voices[voice_idx].osc.waveform = sub.waveform;
                }
                _ => {
                    self.voices[voice_idx].osc.waveform = Waveform::Sine;
                }
            }
        }
    }

    /// Release a note
    pub fn note_off(&mut self, channel: u8, note: u8) {
        for voice in &mut self.voices {
            if voice.active && voice.channel == channel && voice.note == note {
                voice.release();
                break;
            }
        }
    }

    /// Render audio to f32 buffer (mono)
    ///
    /// Returns number of samples written.
    pub fn render(&mut self, buffer: &mut [f32]) -> usize {
        // Hoist inv_sample_rate: shared across all voices and all samples in this call
        let inv_sr = self.inv_sample_rate;
        let sr = self.sample_rate;

        for sample in buffer.iter_mut() {
            // Advance score sequencer
            self.advance_score();

            // Mix all active voices
            let mut mix = 0.0f32;
            for voice in &mut self.voices {
                if !voice.active {
                    continue;
                }
                let ch = voice.channel as usize;
                let s = match self.patches.get(ch).and_then(|p| p.as_ref()) {
                    Some(Patch::Fm(ref fm)) => render_fm_voice(voice, fm, inv_sr),
                    Some(Patch::Additive(ref add)) => {
                        render_additive_voice(voice, add, sr, inv_sr)
                    }
                    Some(Patch::Subtractive(ref sub)) => {
                        render_subtractive_voice(voice, sub, sr, inv_sr)
                    }
                    Some(Patch::Wavetable(ref wt)) => {
                        render_wavetable_voice(voice, wt, inv_sr)
                    }
                    None => {
                        // Default: simple sine
                        let env = voice.amp_env.next(&Adsr {
                            attack: 100,
                            decay: 1000,
                            sustain: 0.5,
                            release: 2000,
                        });
                        let s = voice.osc.next_sample(voice.freq_hz, inv_sr);
                        s * env * voice.velocity
                    }
                };
                mix += s;

                if !voice.amp_env.is_active() {
                    voice.active = false;
                }
            }

            *sample = mix * self.master_volume;
        }
        buffer.len()
    }

    /// Render to i16 PCM buffer
    pub fn render_i16(&mut self, buffer: &mut [i16]) -> usize {
        let mut f32_buf: Vec<f32> = vec![0.0f32; buffer.len()];
        self.render(&mut f32_buf);
        for (i, &s) in f32_buf.iter().enumerate() {
            let clamped = if s < -1.0 { -1.0f32 } else if s > 1.0 { 1.0 } else { s };
            buffer[i] = (clamped * 32767.0) as i16;
        }
        buffer.len()
    }

    /// Number of currently active voices
    pub fn active_voice_count(&self) -> usize {
        self.voices.iter().filter(|v| v.active).count()
    }

    /// Advance score sequencer by one sample
    fn advance_score(&mut self) {
        let Some(score) = self.score.as_ref() else {
            return;
        };

        let event_count = score.events.len();
        if self.score_event_idx >= event_count {
            return;
        }

        self.tick_accum += 1.0;
        // Multiply by pre-computed reciprocal instead of dividing per sample
        let ticks_elapsed = self.tick_accum * self.inv_samples_per_tick;

        // Process events whose delta has elapsed
        while self.score_event_idx < event_count {
            let event = self.score.as_ref().expect("score checked above").events[self.score_event_idx];
            if (event.delta_tick as f32) > ticks_elapsed {
                break;
            }
            self.tick_accum -= event.delta_tick as f32 * self.samples_per_tick;

            // Inline event processing to avoid borrow conflict
            match event.kind {
                NoteEventKind::NoteOn => {
                    if event.velocity > 0 {
                        self.note_on(event.channel, event.note, event.velocity);
                    } else {
                        self.note_off(event.channel, event.note);
                    }
                }
                NoteEventKind::NoteOff => {
                    self.note_off(event.channel, event.note);
                }
                _ => {} // PitchBend, CC: future
            }

            self.score_event_idx += 1;
        }
    }

    fn find_free_voice(&self) -> usize {
        // Find inactive voice
        for (i, v) in self.voices.iter().enumerate() {
            if !v.active {
                return i;
            }
        }
        // Voice stealing: find quietest voice
        let mut min_level = f32::MAX;
        let mut min_idx = 0;
        for (i, v) in self.voices.iter().enumerate() {
            let level = v.amp_env.level() * v.velocity;
            if level < min_level {
                min_level = level;
                min_idx = i;
            }
        }
        min_idx
    }
}

/// Render one sample from an FM voice
///
/// `inv_sample_rate`: pre-computed `1.0 / sample_rate`
#[inline(always)]
fn render_fm_voice(voice: &mut Voice, patch: &FmPatch, inv_sample_rate: f32) -> f32 {
    // Simple 2-op FM: operator[1] modulates operator[0]
    let op1 = &patch.operators[1];
    let op0 = &patch.operators[0];

    let mod_env = voice.mod_env.next(&op1.envelope);
    let mod_freq = voice.freq_hz * op1.ratio;
    let mod_sample = voice.mod_osc.next_sample(mod_freq, inv_sample_rate);
    let phase_mod = mod_sample * op1.mod_index * mod_env;

    let carrier_env = voice.amp_env.next(&op0.envelope);
    let carrier_freq = voice.freq_hz * op0.ratio;
    let carrier = voice.osc.next_sample_fm(carrier_freq, inv_sample_rate, phase_mod);

    carrier * carrier_env * voice.velocity * op0.level
}

/// Render one sample from an additive voice
///
/// `sample_rate`: used for Nyquist check (compare, not divide)
/// `inv_sample_rate`: pre-computed `1.0 / sample_rate` for oscillator phase step
#[inline(always)]
fn render_additive_voice(
    voice: &mut Voice,
    patch: &AdditivePatch,
    sample_rate: f32,
    inv_sample_rate: f32,
) -> f32 {
    let env = voice.amp_env.next(&patch.envelope);
    // Nyquist limit: half of sample_rate; pre-multiply once
    let nyquist = sample_rate * 0.5;
    let mut sum = 0.0f32;
    for (i, &amp) in patch.harmonics.iter().enumerate() {
        if amp < 0.001 {
            continue;
        }
        let harmonic = (i + 1) as f32;
        let freq = voice.freq_hz * harmonic;
        if freq > nyquist {
            break; // Nyquist limit
        }
        // Use main osc phase as base, compute harmonic phase
        let phase = voice.osc.next_sample(freq, inv_sample_rate);
        sum += phase * amp;
    }
    // Advance base oscillator for phase tracking
    let _ = voice.osc.next_sample(voice.freq_hz, inv_sample_rate);
    sum * env * voice.velocity
}

/// Render one sample from a subtractive voice
///
/// `sample_rate`: used for filter Nyquist clamp
/// `inv_sample_rate`: pre-computed `1.0 / sample_rate` for oscillator phase step
#[inline(always)]
fn render_subtractive_voice(
    voice: &mut Voice,
    patch: &SubtractivePatch,
    sample_rate: f32,
    inv_sample_rate: f32,
) -> f32 {
    let amp_env = voice.amp_env.next(&patch.amp_envelope);
    let filter_env = voice.filter_env.next(&patch.filter_envelope);

    // Oscillator
    let raw = voice.osc.next_sample(voice.freq_hz, inv_sample_rate);

    // Filter with envelope modulation
    let cutoff = patch.cutoff_hz + filter_env * patch.filter_env_amount * 10000.0;
    voice.filter.set_cutoff(cutoff.clamp(20.0, sample_rate * 0.45), sample_rate);
    let filtered = voice.filter.process(raw);

    filtered * amp_env * voice.velocity
}

/// Render one sample from a wavetable voice
///
/// `inv_sample_rate`: pre-computed `1.0 / sample_rate`
#[inline(always)]
fn render_wavetable_voice(voice: &mut Voice, patch: &WavetablePatch, inv_sample_rate: f32) -> f32 {
    let env = voice.amp_env.next(&patch.envelope);
    let sample = voice.osc.next_sample(voice.freq_hz, inv_sample_rate);
    // Map oscillator output [-1,1] to phase [0,1]
    let phase = (sample + 1.0) * 0.5;
    let wt_sample = patch.lookup(phase);
    wt_sample * env * voice.velocity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::FmPatch;
    use crate::score::{NoteEvent, NoteEventKind, Score};

    #[test]
    fn test_synth_basic() {
        let mut synth = Synthesizer::new(44100);
        synth.load_patch(0, Patch::Fm(FmPatch::electric_piano()));
        synth.note_on(0, 60, 100);
        assert_eq!(synth.active_voice_count(), 1);

        let mut buf = [0.0f32; 512];
        synth.render(&mut buf);

        // Should produce non-silent output
        let max_abs = buf.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.01, "should produce sound, max_abs={max_abs}");
    }

    #[test]
    fn test_synth_note_off() {
        let mut synth = Synthesizer::new(44100);
        synth.note_on(0, 60, 80);
        synth.note_off(0, 60);

        // Render enough for release to complete
        let mut buf = [0.0f32; 44100];
        synth.render(&mut buf);

        assert_eq!(synth.active_voice_count(), 0);
    }

    #[test]
    fn test_synth_polyphony() {
        let mut synth = Synthesizer::new(44100);
        synth.note_on(0, 60, 80);
        synth.note_on(0, 64, 80);
        synth.note_on(0, 67, 80);
        assert_eq!(synth.active_voice_count(), 3);
    }

    #[test]
    fn test_synth_score_playback() {
        let mut synth = Synthesizer::new(44100);
        synth.load_patch(0, Patch::Fm(FmPatch::bell()));

        let mut score = Score::new(120, 1);
        score.add_event(NoteEvent {
            delta_tick: 0,
            channel: 0,
            note: 72,
            velocity: 100,
            kind: NoteEventKind::NoteOn,
        });
        score.add_event(NoteEvent {
            delta_tick: 96,
            channel: 0,
            note: 72,
            velocity: 0,
            kind: NoteEventKind::NoteOff,
        });
        synth.load_score(&score);

        let mut buf = [0.0f32; 4096];
        synth.render(&mut buf);

        // Should have produced sound
        let max_abs = buf.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.001);
    }

    #[test]
    fn test_render_i16() {
        let mut synth = Synthesizer::new(44100);
        synth.note_on(0, 60, 100);
        let mut buf = [0i16; 256];
        synth.render_i16(&mut buf);
        let max_abs = buf.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(max_abs > 0, "should produce non-zero i16 samples");
    }

    #[test]
    fn test_subtractive_voice() {
        let mut synth = Synthesizer::new(44100);
        synth.load_patch(0, Patch::Subtractive(SubtractivePatch::bass()));
        synth.note_on(0, 36, 100); // Low C
        let mut buf = [0.0f32; 1024];
        synth.render(&mut buf);
        let max_abs = buf.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.01);
    }
}
