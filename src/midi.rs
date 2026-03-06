//! MIDI 統合モジュール
//!
//! MIDI ノート番号と周波数の変換、MIDI イベントのパース、
//! ASYN Score との相互変換を提供する。
//!
//! # Feature
//!
//! `midi` feature を有効にすると利用可能。
//!
//! Author: Moroya Sakamoto

use crate::score::{NoteEvent, NoteEventKind, Score};

/// MIDI ノート番号から周波数 (Hz) へ変換。
///
/// A4 (note 69) = 440 Hz 基準。
#[inline]
#[must_use]
pub fn midi_to_freq(note: u8) -> f32 {
    // f = 440 * 2^((note - 69) / 12)
    440.0 * f32::powf(2.0, (note as f32 - 69.0) / 12.0)
}

/// 周波数 (Hz) から最も近い MIDI ノート番号へ変換。
#[inline]
#[must_use]
pub fn freq_to_midi(freq: f32) -> u8 {
    if freq <= 0.0 {
        return 0;
    }
    let note = 12.0f32.mul_add(f32::log2(freq / 440.0), 69.0);
    (note.round() as i32).clamp(0, 127) as u8
}

/// MIDI イベント種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidiEventKind {
    /// ノートオン (channel, note, velocity)。
    NoteOn,
    /// ノートオフ (channel, note, velocity)。
    NoteOff,
    /// コントロールチェンジ (channel, controller, value)。
    ControlChange,
    /// プログラムチェンジ (channel, program)。
    ProgramChange,
    /// ピッチベンド (channel, value)。
    PitchBend,
}

/// MIDI イベント。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MidiEvent {
    /// デルタタイム (ティック)。
    pub delta_time: u32,
    /// イベント種別。
    pub kind: MidiEventKind,
    /// チャネル (0-15)。
    pub channel: u8,
    /// データバイト1 (note/controller/program)。
    pub data1: u8,
    /// データバイト2 (velocity/value)。0 for `ProgramChange`。
    pub data2: u8,
}

impl MidiEvent {
    /// 新しい MIDI イベントを作成。
    #[must_use]
    pub const fn new(
        delta_time: u32,
        kind: MidiEventKind,
        channel: u8,
        data1: u8,
        data2: u8,
    ) -> Self {
        Self {
            delta_time,
            kind,
            channel,
            data1,
            data2,
        }
    }

    /// Note On イベント。
    #[must_use]
    pub const fn note_on(delta_time: u32, channel: u8, note: u8, velocity: u8) -> Self {
        Self::new(delta_time, MidiEventKind::NoteOn, channel, note, velocity)
    }

    /// Note Off イベント。
    #[must_use]
    pub const fn note_off(delta_time: u32, channel: u8, note: u8) -> Self {
        Self::new(delta_time, MidiEventKind::NoteOff, channel, note, 0)
    }
}

/// MIDI トラック。
#[derive(Debug, Clone)]
pub struct MidiTrack {
    /// トラック名。
    pub name: Vec<u8>,
    /// イベント列。
    pub events: Vec<MidiEvent>,
}

impl MidiTrack {
    /// 空のトラックを作成。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            name: Vec::new(),
            events: Vec::new(),
        }
    }

    /// イベント追加。
    pub fn add_event(&mut self, event: MidiEvent) {
        self.events.push(event);
    }

    /// トラック内の総ティック数。
    #[must_use]
    pub fn total_ticks(&self) -> u32 {
        self.events.iter().map(|e| e.delta_time).sum()
    }
}

impl Default for MidiTrack {
    fn default() -> Self {
        Self::new()
    }
}

/// MIDI トラック列を ASYN Score に変換。
///
/// # Arguments
///
/// - `tracks`: MIDI トラック列
/// - `tempo_bpm`: テンポ (BPM)
/// - `ticks_per_beat`: MIDI の ticks per beat (PPQ)
/// - `asyn_tick_div`: 変換先の ASYN tick division
#[must_use]
pub fn midi_to_score(
    tracks: &[MidiTrack],
    tempo_bpm: u16,
    ticks_per_beat: u16,
    asyn_tick_div: u16,
) -> Score {
    let track_count = tracks.len().min(16) as u8;
    let mut score = Score::new(tempo_bpm, track_count);
    score.header.tick_div = asyn_tick_div;

    // 全トラックのイベントを絶対時間付きで収集
    let mut abs_events: Vec<(u32, u8, u8, u8, NoteEventKind)> = Vec::new();

    for (ch, track) in tracks.iter().enumerate() {
        let channel = (ch & 0x0F) as u8;
        let mut abs_tick: u32 = 0;
        for ev in &track.events {
            abs_tick += ev.delta_time;
            let kind = match ev.kind {
                MidiEventKind::NoteOn => NoteEventKind::NoteOn,
                MidiEventKind::NoteOff => NoteEventKind::NoteOff,
                MidiEventKind::PitchBend => NoteEventKind::PitchBend,
                MidiEventKind::ControlChange | MidiEventKind::ProgramChange => {
                    NoteEventKind::ControlChange
                }
            };
            // ティック解像度変換
            let scaled_tick = if ticks_per_beat > 0 {
                (abs_tick as u64 * asyn_tick_div as u64 / ticks_per_beat as u64) as u32
            } else {
                abs_tick
            };
            abs_events.push((scaled_tick, channel, ev.data1 & 0x7F, ev.data2 & 0x7F, kind));
        }
    }

    // 絶対時間でソート
    abs_events.sort_by_key(|e| e.0);

    // デルタタイムに変換
    let mut prev_tick: u32 = 0;
    for (abs_tick, channel, note, velocity, kind) in abs_events {
        let delta = (abs_tick - prev_tick).min(4095) as u16;
        score.add_event(NoteEvent {
            delta_tick: delta,
            channel,
            note,
            velocity,
            kind,
        });
        prev_tick = abs_tick;
    }

    score
}

/// ASYN Score を MIDI イベント列に変換。
///
/// 1トラック分のフラットな MIDI イベント列を返す。
#[must_use]
pub fn score_to_midi(score: &Score) -> Vec<MidiEvent> {
    let mut midi_events = Vec::with_capacity(score.events.len());
    for ev in &score.events {
        let kind = match ev.kind {
            NoteEventKind::NoteOn => MidiEventKind::NoteOn,
            NoteEventKind::NoteOff => MidiEventKind::NoteOff,
            NoteEventKind::PitchBend => MidiEventKind::PitchBend,
            NoteEventKind::ControlChange => MidiEventKind::ControlChange,
        };
        midi_events.push(MidiEvent {
            delta_time: ev.delta_tick as u32,
            kind,
            channel: ev.channel,
            data1: ev.note,
            data2: ev.velocity,
        });
    }
    midi_events
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midi_to_freq_a4() {
        let freq = midi_to_freq(69);
        assert!((freq - 440.0).abs() < 0.01);
    }

    #[test]
    fn midi_to_freq_middle_c() {
        let freq = midi_to_freq(60);
        assert!((freq - 261.63).abs() < 0.1);
    }

    #[test]
    fn midi_to_freq_octave() {
        let f1 = midi_to_freq(60);
        let f2 = midi_to_freq(72);
        assert!((f2 / f1 - 2.0).abs() < 0.01);
    }

    #[test]
    fn freq_to_midi_a4() {
        assert_eq!(freq_to_midi(440.0), 69);
    }

    #[test]
    fn freq_to_midi_roundtrip() {
        for note in 0..=127u8 {
            let freq = midi_to_freq(note);
            let back = freq_to_midi(freq);
            assert_eq!(back, note, "roundtrip failed for note {note}");
        }
    }

    #[test]
    fn freq_to_midi_zero() {
        assert_eq!(freq_to_midi(0.0), 0);
        assert_eq!(freq_to_midi(-1.0), 0);
    }

    #[test]
    fn midi_event_note_on() {
        let ev = MidiEvent::note_on(96, 0, 60, 100);
        assert_eq!(ev.kind, MidiEventKind::NoteOn);
        assert_eq!(ev.data1, 60);
        assert_eq!(ev.data2, 100);
    }

    #[test]
    fn midi_event_note_off() {
        let ev = MidiEvent::note_off(48, 1, 60);
        assert_eq!(ev.kind, MidiEventKind::NoteOff);
        assert_eq!(ev.channel, 1);
    }

    #[test]
    fn midi_track_total_ticks() {
        let mut track = MidiTrack::new();
        track.add_event(MidiEvent::note_on(0, 0, 60, 100));
        track.add_event(MidiEvent::note_off(96, 0, 60));
        track.add_event(MidiEvent::note_on(0, 0, 62, 100));
        track.add_event(MidiEvent::note_off(96, 0, 62));
        assert_eq!(track.total_ticks(), 192);
    }

    #[test]
    fn midi_track_default() {
        let track = MidiTrack::default();
        assert!(track.events.is_empty());
    }

    #[test]
    fn midi_to_score_basic() {
        let mut track = MidiTrack::new();
        track.add_event(MidiEvent::note_on(0, 0, 60, 100));
        track.add_event(MidiEvent::note_off(480, 0, 60));

        let score = midi_to_score(&[track], 120, 480, 96);
        assert_eq!(score.header.tempo_bpm, 120);
        assert_eq!(score.header.tracks, 1);
        assert_eq!(score.events.len(), 2);
        // 480 ticks @ 480ppq → 96 ticks @ 96ppq
        assert_eq!(score.events[1].delta_tick, 96);
    }

    #[test]
    fn midi_to_score_multi_track() {
        let mut t1 = MidiTrack::new();
        t1.add_event(MidiEvent::note_on(0, 0, 60, 100));
        let mut t2 = MidiTrack::new();
        t2.add_event(MidiEvent::note_on(0, 0, 64, 80));

        let score = midi_to_score(&[t1, t2], 120, 480, 96);
        assert_eq!(score.header.tracks, 2);
        assert_eq!(score.events.len(), 2);
        // 両方 delta=0 なのでチャネルが異なるイベントが2つ
    }

    #[test]
    fn score_to_midi_roundtrip() {
        let mut score = Score::new(120, 1);
        score.add_event(NoteEvent {
            delta_tick: 0,
            channel: 0,
            note: 60,
            velocity: 100,
            kind: NoteEventKind::NoteOn,
        });
        score.add_event(NoteEvent {
            delta_tick: 96,
            channel: 0,
            note: 60,
            velocity: 0,
            kind: NoteEventKind::NoteOff,
        });

        let midi = score_to_midi(&score);
        assert_eq!(midi.len(), 2);
        assert_eq!(midi[0].kind, MidiEventKind::NoteOn);
        assert_eq!(midi[0].data1, 60);
        assert_eq!(midi[1].kind, MidiEventKind::NoteOff);
        assert_eq!(midi[1].delta_time, 96);
    }

    #[test]
    fn midi_to_score_empty_track() {
        let track = MidiTrack::new();
        let score = midi_to_score(&[track], 120, 480, 96);
        assert!(score.events.is_empty());
    }

    #[test]
    fn midi_to_score_tick_scaling() {
        let mut track = MidiTrack::new();
        track.add_event(MidiEvent::note_on(0, 0, 60, 100));
        track.add_event(MidiEvent::note_off(960, 0, 60));

        let score = midi_to_score(&[track], 120, 480, 96);
        // 960 ticks @ 480ppq = 2 beats → 192 ticks @ 96ppq
        assert_eq!(score.events[1].delta_tick, 192);
    }

    #[test]
    fn midi_to_score_delta_clamped() {
        let mut track = MidiTrack::new();
        track.add_event(MidiEvent::note_on(0, 0, 60, 100));
        // Very large delta
        track.add_event(MidiEvent::note_off(100_000, 0, 60));

        let score = midi_to_score(&[track], 120, 480, 96);
        // delta_tick は 12ビット (0-4095) に制限
        assert!(score.events[1].delta_tick <= 4095);
    }

    #[test]
    fn control_change_mapping() {
        let mut track = MidiTrack::new();
        track.add_event(MidiEvent::new(0, MidiEventKind::ControlChange, 0, 1, 64));

        let score = midi_to_score(&[track], 120, 480, 96);
        assert_eq!(score.events[0].kind, NoteEventKind::ControlChange);
    }
}
