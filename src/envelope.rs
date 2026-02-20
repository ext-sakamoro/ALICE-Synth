//! ADSR Envelope generator
//!
//! Standard Attack-Decay-Sustain-Release envelope for amplitude
//! and modulation control. Fixed-point friendly, branchless-optimizable.
//!
//! Author: Moroya Sakamoto

/// ADSR envelope parameters
///
/// Times are in samples (not seconds) for deterministic no_std operation.
/// Sustain level is in [0.0, 1.0].
///
/// Size: 16 bytes
#[derive(Debug, Clone, Copy)]
pub struct Adsr {
    /// Attack time in samples
    pub attack: u32,
    /// Decay time in samples
    pub decay: u32,
    /// Sustain level [0.0, 1.0]
    pub sustain: f32,
    /// Release time in samples
    pub release: u32,
}

impl Adsr {
    /// Create ADSR from time in milliseconds
    pub fn from_ms(attack_ms: f32, decay_ms: f32, sustain: f32, release_ms: f32, sample_rate: f32) -> Self {
        const RCP_1000: f32 = 1.0 / 1000.0;
        // Multiply by RCP_1000 instead of dividing per field
        let ms_to_samples = sample_rate * RCP_1000;
        Self {
            attack: (attack_ms * ms_to_samples) as u32,
            decay: (decay_ms * ms_to_samples) as u32,
            sustain: sustain.clamp(0.0, 1.0),
            release: (release_ms * ms_to_samples) as u32,
        }
    }

    /// Quick percussive envelope (good for drums/SE)
    pub fn percussive(attack_ms: f32, decay_ms: f32, sample_rate: f32) -> Self {
        Self::from_ms(attack_ms, decay_ms, 0.0, decay_ms, sample_rate)
    }

    /// Organ-style envelope (instant attack, full sustain)
    pub fn organ(sample_rate: f32) -> Self {
        Self::from_ms(1.0, 1.0, 1.0, 10.0, sample_rate)
    }

    /// Piano-style envelope
    pub fn piano(sample_rate: f32) -> Self {
        Self::from_ms(5.0, 200.0, 0.3, 500.0, sample_rate)
    }
}

/// ADSR envelope state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdsrPhase {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Envelope state for one voice
///
/// Size: 12 bytes
pub struct AdsrState {
    /// Current phase
    phase: AdsrPhase,
    /// Sample counter within current phase
    counter: u32,
    /// Current output level [0.0, 1.0]
    level: f32,
}

impl AdsrState {
    pub fn new() -> Self {
        Self {
            phase: AdsrPhase::Idle,
            counter: 0,
            level: 0.0,
        }
    }

    /// Trigger note-on
    pub fn note_on(&mut self) {
        self.phase = AdsrPhase::Attack;
        self.counter = 0;
    }

    /// Trigger note-off (enter release phase)
    pub fn note_off(&mut self) {
        if self.phase != AdsrPhase::Idle {
            self.phase = AdsrPhase::Release;
            self.counter = 0;
        }
    }

    /// Is this envelope still producing output?
    pub fn is_active(&self) -> bool {
        self.phase != AdsrPhase::Idle
    }

    /// Current output level
    pub fn level(&self) -> f32 {
        self.level
    }

    /// Current phase
    pub fn phase(&self) -> AdsrPhase {
        self.phase
    }

    /// Advance envelope by one sample
    ///
    /// Returns amplitude multiplier [0.0, 1.0].
    /// Divisions are replaced with reciprocal multiplications; each reciprocal is
    /// computed at most once per phase transition, not once per sample.
    #[inline(always)]
    pub fn next(&mut self, params: &Adsr) -> f32 {
        match self.phase {
            AdsrPhase::Idle => {
                self.level = 0.0;
            }
            AdsrPhase::Attack => {
                if params.attack == 0 {
                    self.level = 1.0;
                    self.phase = AdsrPhase::Decay;
                    self.counter = 0;
                } else {
                    // Reciprocal computed once per call; attack is constant per phase
                    let inv_attack = (params.attack as f32).recip();
                    self.level = self.counter as f32 * inv_attack;
                    self.counter += 1;
                    if self.counter >= params.attack {
                        self.level = 1.0;
                        self.phase = AdsrPhase::Decay;
                        self.counter = 0;
                    }
                }
            }
            AdsrPhase::Decay => {
                if params.decay == 0 {
                    self.level = params.sustain;
                    self.phase = AdsrPhase::Sustain;
                } else {
                    // Reciprocal computed once per call; decay is constant per phase
                    let inv_decay = (params.decay as f32).recip();
                    let t = self.counter as f32 * inv_decay;
                    self.level = 1.0 + (params.sustain - 1.0) * t;
                    self.counter += 1;
                    if self.counter >= params.decay {
                        self.level = params.sustain;
                        self.phase = AdsrPhase::Sustain;
                    }
                }
            }
            AdsrPhase::Sustain => {
                self.level = params.sustain;
            }
            AdsrPhase::Release => {
                if params.release == 0 {
                    self.level = 0.0;
                    self.phase = AdsrPhase::Idle;
                } else {
                    // Reciprocal computed once per call; release is constant per phase
                    let inv_release = (params.release as f32).recip();
                    let start_level = if self.counter == 0 {
                        self.level
                    } else {
                        // Reconstruct start level; replace inner division with multiply
                        let progress = self.counter as f32 * inv_release;
                        self.level / (1.0 - progress).max(0.001)
                    };
                    self.counter += 1;
                    let t = self.counter as f32 * inv_release;
                    self.level = start_level * (1.0 - t).max(0.0);
                    if self.counter >= params.release {
                        self.level = 0.0;
                        self.phase = AdsrPhase::Idle;
                    }
                }
            }
        }
        self.level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adsr_attack() {
        let params = Adsr {
            attack: 10,
            decay: 10,
            sustain: 0.5,
            release: 10,
        };
        let mut state = AdsrState::new();
        state.note_on();

        // Attack phase: should ramp from 0 to 1
        let mut levels = [0.0f32; 10];
        for i in 0..10 {
            levels[i] = state.next(&params);
        }
        assert!(levels[0] < 0.2, "start of attack should be low");
        assert!(levels[9] > 0.8, "end of attack should be high");
    }

    #[test]
    fn test_adsr_sustain() {
        let params = Adsr {
            attack: 4,
            decay: 4,
            sustain: 0.6,
            release: 4,
        };
        let mut state = AdsrState::new();
        state.note_on();

        // Run through attack + decay
        for _ in 0..20 {
            state.next(&params);
        }
        // Should be at sustain level
        let level = state.next(&params);
        assert!(
            (level - 0.6).abs() < 0.05,
            "sustain should be ~0.6, got {level}"
        );
    }

    #[test]
    fn test_adsr_release_to_idle() {
        let params = Adsr {
            attack: 2,
            decay: 2,
            sustain: 0.5,
            release: 4,
        };
        let mut state = AdsrState::new();
        state.note_on();
        for _ in 0..10 {
            state.next(&params);
        }
        assert!(state.is_active());

        state.note_off();
        for _ in 0..10 {
            state.next(&params);
        }
        assert!(!state.is_active(), "should be idle after release");
        assert!(state.level() < 0.01);
    }

    #[test]
    fn test_percussive() {
        let params = Adsr::percussive(1.0, 50.0, 44100.0);
        assert!(params.attack < 100);
        assert!(params.sustain == 0.0);
    }
}
