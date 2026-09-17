//! Analytic oracles — closed-form checks for the acoustic laws in ALICE-Synth
//! (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms or an independent f64 DFT written
//! in this file, never from the crate function under test.
//!
//! Oracle sources:
//! - oscillator spectra (f64 DFT over an integer number of periods): a sine
//!   has one line at f with amplitude 1 and everything else ≤ −50 dB (Bhaskara
//!   sine, |Δ| ≤ 1.7e-3); saw harmonics fall as 2/(πn), square as 4/(πn)
//!   (odd only), triangle as 8/(π²n²) (odd only); DC = 0
//! - equal temperament: f = 440·2^((n−69)/12), A4 = 440, an octave doubles
//! - ADSR: linear ramps counter/attack, 1 → sustain over decay, level → 0
//!   over release; `from_ms` converts at sample_rate/1000
//! - one-pole low-pass: DC gain 1, |H(f)| = 1/√(1 + (f/fc)²) for f ≪ fs
//!   (within 1 dB at fc, ≥ 19 dB down at 10·fc), a delay line echoes an
//!   impulse after exactly `delay_samples`
//! - score header: ticks/s = bpm/60·tick_div, wire round trip

use alice_synth::envelope::{Adsr, AdsrPhase, AdsrState};
use alice_synth::oscillator::{midi_to_freq, sin_approx, Oscillator, Waveform};
use alice_synth::score::ScoreHeader;
use alice_synth::{Delay, Effect, LowPassFilter};

/// Amplitude of the DFT line at harmonic `k` for a signal spanning `periods`
/// whole periods (so every harmonic sits exactly on a bin).
fn harmonic_amplitude(samples: &[f32], periods: usize, k: usize) -> f64 {
    let n = samples.len() as f64;
    let bin = (periods * k) as f64;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &s) in samples.iter().enumerate() {
        let ang = 2.0 * std::f64::consts::PI * bin * i as f64 / n;
        re += s as f64 * ang.cos();
        im -= s as f64 * ang.sin();
    }
    2.0 * (re * re + im * im).sqrt() / n
}

fn render(waveform: Waveform, freq: f32, sample_rate: f32, n: usize) -> Vec<f32> {
    let mut osc = Oscillator::new(waveform);
    (0..n)
        .map(|_| osc.next_sample(freq, 1.0 / sample_rate))
        .collect()
}

// ───────────────────────── oscillators ────────────────────────────────────

#[test]
fn oscillator_spectra_follow_the_fourier_series_of_each_waveform() {
    // 48 kHz, 375 Hz ⇒ exactly 128 samples per period; 32 periods = 4096 samples
    let (sr, f, periods) = (48_000.0f32, 375.0f32, 32usize);
    let n = periods * 128;
    let sine = render(Waveform::Sine, f, sr, n);
    assert!(
        (harmonic_amplitude(&sine, periods, 1) - 1.0).abs() < 3e-3,
        "sine fundamental"
    );
    for k in 2..=9 {
        let a = harmonic_amplitude(&sine, periods, k);
        assert!(
            20.0 * a.log10() < -50.0,
            "sine harmonic {k}: {:.1} dB",
            20.0 * a.log10()
        );
    }
    let dc: f64 = sine.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
    assert!(dc.abs() < 1e-3, "sine DC {dc}");
    // saw: bₙ = 2/(πn) for every n (sign aside)
    let saw = render(Waveform::Saw, f, sr, n);
    for k in 1..=7 {
        let expected = 2.0 / (std::f64::consts::PI * k as f64);
        let a = harmonic_amplitude(&saw, periods, k);
        assert!(
            (a - expected).abs() < 0.02 * expected + 5e-3,
            "saw harmonic {k}: {a} vs {expected}"
        );
    }
    // square: 4/(πn) for odd n, 0 for even
    let square = render(Waveform::Square, f, sr, n);
    for k in 1..=7 {
        let a = harmonic_amplitude(&square, periods, k);
        if k % 2 == 1 {
            let expected = 4.0 / (std::f64::consts::PI * k as f64);
            assert!(
                (a - expected).abs() < 0.02 * expected + 5e-3,
                "square harmonic {k}: {a} vs {expected}"
            );
        } else {
            assert!(a < 5e-3, "square even harmonic {k}: {a}");
        }
    }
    // triangle: 8/(π²n²) for odd n
    let tri = render(Waveform::Triangle, f, sr, n);
    for k in [1usize, 3, 5, 7] {
        let expected = 8.0 / (std::f64::consts::PI.powi(2) * (k * k) as f64);
        let a = harmonic_amplitude(&tri, periods, k);
        assert!(
            (a - expected).abs() < 0.02 * expected + 5e-3,
            "triangle harmonic {k}: {a} vs {expected}"
        );
    }
    assert!(harmonic_amplitude(&tri, periods, 2) < 5e-3);
    // every waveform stays in [−1, 1] and is periodic with the set frequency
    for w in [
        Waveform::Sine,
        Waveform::Saw,
        Waveform::Square,
        Waveform::Triangle,
    ] {
        let s = render(w, f, sr, n);
        assert!(s.iter().all(|v| (-1.0..=1.0).contains(v)), "{w:?} range");
        for i in 0..128 {
            assert!((s[i] - s[i + 128]).abs() < 1e-4, "{w:?} periodic at {i}");
        }
    }
    // Bhaskara sine: |sin_approx − sin| ≤ 1.63e-3 on the whole circle
    for i in 0..=720 {
        let x = i as f32 * std::f32::consts::TAU / 720.0 - std::f32::consts::TAU;
        assert!(
            (sin_approx(x) as f64 - (x as f64).sin()).abs() <= 1.7e-3,
            "sin_approx({x})"
        );
    }
}

#[test]
fn midi_to_freq_is_equal_temperament_within_a_hundredth_of_a_cent() {
    assert_eq!(midi_to_freq(69), 440.0, "A4");
    for note in 0..=127u8 {
        let expected = 440.0 * 2f64.powf((note as f64 - 69.0) / 12.0);
        let got = midi_to_freq(note) as f64;
        let cents = 1200.0 * (got / expected).log2();
        assert!(
            cents.abs() < 0.01,
            "note {note}: {got} Hz vs {expected} ({cents:.3} cents)"
        );
    }
    assert!(
        (midi_to_freq(81) / midi_to_freq(69) - 2.0).abs() < 1e-5,
        "octave doubles"
    );
    assert!((midi_to_freq(60) - 261.6256).abs() < 0.01, "middle C");
}

// ───────────────────────── envelope ───────────────────────────────────────

#[test]
fn adsr_ramps_are_linear_with_the_configured_sample_counts() {
    let params = Adsr::from_ms(10.0, 20.0, 0.25, 40.0, 1000.0); // 10 / 20 / 40 samples at 1 kHz
    assert_eq!((params.attack, params.decay, params.release), (10, 20, 40));
    assert_eq!(params.sustain, 0.25);
    let mut env = AdsrState::new();
    assert!(!env.is_active());
    env.note_on();
    // attack: counter/attack, reaching 1.0 on the last attack sample
    for k in 0..10u32 {
        let l = env.next(&params);
        let expected = if k == 9 { 1.0 } else { k as f32 / 10.0 };
        assert!(
            (l - expected).abs() < 1e-6,
            "attack sample {k}: {l} vs {expected}"
        );
    }
    assert_eq!(env.phase(), AdsrPhase::Decay);
    // decay: 1 → sustain linearly over `decay` samples
    for k in 0..20u32 {
        let l = env.next(&params);
        let expected = if k == 19 {
            0.25
        } else {
            1.0 + (0.25 - 1.0) * k as f32 / 20.0
        };
        assert!(
            (l - expected).abs() < 1e-6,
            "decay sample {k}: {l} vs {expected}"
        );
    }
    assert_eq!(env.phase(), AdsrPhase::Sustain);
    for _ in 0..100 {
        assert_eq!(env.next(&params), 0.25, "sustain holds");
    }
    // release: sustain → 0 over `release` samples, then idle
    env.note_off();
    let mut last = 1.0f32;
    for k in 0..40u32 {
        let l = env.next(&params);
        assert!(l <= last + 1e-6 && l >= 0.0, "release sample {k} monotone");
        last = l;
    }
    assert_eq!(env.level(), 0.0);
    assert!(!env.is_active());
    // zero-length stages jump
    let instant = Adsr {
        attack: 0,
        decay: 0,
        sustain: 0.7,
        release: 0,
    };
    let mut e = AdsrState::new();
    e.note_on();
    e.next(&instant);
    assert_eq!(e.next(&instant), 0.7);
    e.note_off();
    assert_eq!(e.next(&instant), 0.0);
    // presets convert at sample_rate / 1000
    let p = Adsr::piano(44_100.0);
    assert_eq!((p.attack, p.decay, p.release), (220, 8_820, 22_050));
    assert!((p.sustain - 0.3).abs() < 1e-6);
}

// ───────────────────────── effects ────────────────────────────────────────

#[test]
fn one_pole_low_pass_and_delay_line_match_their_closed_forms() {
    let sr = 48_000.0f32;
    let fc = 480.0f32; // fs / 100 ⇒ the bilinear warp is negligible
    let mut lp = LowPassFilter::new(fc, sr);
    // DC: a unit step settles at 1
    let mut y = 0.0;
    for _ in 0..20_000 {
        y = lp.process(1.0);
    }
    assert!((y - 1.0).abs() < 1e-4, "DC gain {y}");
    // steady-state gain at f: |H| = 1/√(1 + (f/fc)²)  (analogue prototype)
    for (f, tol_db) in [(fc, 1.0f64), (10.0 * fc, 1.0), (0.1 * fc, 0.2)] {
        let mut lp = LowPassFilter::new(fc, sr);
        let n = 48_000usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x = (2.0 * std::f64::consts::PI * f as f64 * i as f64 / sr as f64).sin() as f32;
            out.push(lp.process(x));
        }
        // measure on the last half (transient gone), whole periods of f
        let tail = &out[n / 2..];
        let periods = (f as f64 * tail.len() as f64 / sr as f64).round() as usize;
        let gain = harmonic_amplitude(tail, periods, 1);
        let expected = 1.0 / (1.0 + (f as f64 / fc as f64).powi(2)).sqrt();
        let err_db = 20.0 * (gain / expected).log10();
        assert!(
            err_db.abs() < tol_db,
            "|H({f})| = {gain} vs {expected} ({err_db:.2} dB)"
        );
    }
    // delay: an impulse reappears after exactly d samples, scaled by mix, then
    // by feedback each further d
    let d = 37;
    let (fb, mix) = (0.5f32, 1.0f32);
    let mut delay = Delay::new(d, fb, mix);
    let mut out = Vec::new();
    for i in 0..(4 * d) {
        out.push(delay.process(if i == 0 { 1.0 } else { 0.0 }));
    }
    for (i, &v) in out.iter().enumerate() {
        let expected = if i > 0 && i % d == 0 {
            fb.powi((i / d - 1) as i32)
        } else {
            0.0
        };
        assert!(
            (v - expected).abs() < 1e-6,
            "delay sample {i}: {v} vs {expected}"
        );
    }
    assert_eq!(Delay::from_ms(10.0, 48_000.0, 0.0, 0.5).process(0.0), 0.0);
}

// ───────────────────────── score wire format ──────────────────────────────

#[test]
fn score_header_timing_and_wire_format_round_trip() {
    let h = ScoreHeader {
        tempo_bpm: 120,
        tracks: 3,
        tick_div: 480,
    };
    assert!(
        (h.ticks_per_second() - 960.0).abs() < 1e-4,
        "120 bpm × 480 ppq / 60"
    );
    assert!((h.samples_per_tick(48_000.0) - 50.0).abs() < 1e-4);
    // every common PPQ value must survive the 8-byte header
    for ppq in [24u16, 48, 96, 120, 192, 240, 384, 480, 960] {
        let h = ScoreHeader {
            tempo_bpm: 90,
            tracks: 1,
            tick_div: ppq,
        };
        let back = ScoreHeader::from_bytes(&h.to_bytes()).expect("valid header");
        assert_eq!(
            (back.tempo_bpm, back.tracks, back.tick_div),
            (90, 1, ppq),
            "ppq {ppq}"
        );
    }
    assert!(ScoreHeader::from_bytes(b"NOPE0000").is_none());
    assert!(ScoreHeader::from_bytes(&[0; 4]).is_none());
}
