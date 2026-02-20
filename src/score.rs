//! Compact score format
//!
//! Binary format: 8-byte header + 4-byte note events.
//! Encodes a 3-minute BGM in ~2 KB.
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Score header — 8 bytes
///
/// Magic "ASYN" + tempo + track count + tick division
#[derive(Debug, Clone, Copy)]
pub struct ScoreHeader {
    /// Tempo in BPM (40-300)
    pub tempo_bpm: u16,
    /// Number of tracks/channels (1-16)
    pub tracks: u8,
    /// Ticks per beat (24-480, default 96)
    pub tick_div: u16,
}

impl ScoreHeader {
    pub const MAGIC: [u8; 4] = *b"ASYN";

    /// Ticks per second at current tempo
    #[inline(always)]
    pub fn ticks_per_second(&self) -> f32 {
        const RCP_60: f32 = 1.0 / 60.0;
        self.tempo_bpm as f32 * RCP_60 * self.tick_div as f32
    }

    /// Samples per tick at given sample rate
    pub fn samples_per_tick(&self, sample_rate: f32) -> f32 {
        sample_rate / self.ticks_per_second()
    }

    /// Serialize to 8 bytes
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0..4].copy_from_slice(&Self::MAGIC);
        buf[4..6].copy_from_slice(&self.tempo_bpm.to_le_bytes());
        buf[6] = self.tracks;
        buf[7] = (self.tick_div / 2) as u8; // Store as half (fits u8 for common values)
        buf
    }

    /// Deserialize from 8 bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 8 || data[0..4] != Self::MAGIC {
            return None;
        }
        let tempo_bpm = u16::from_le_bytes([data[4], data[5]]);
        let tracks = data[6];
        let tick_div = data[7] as u16 * 2;
        Some(Self {
            tempo_bpm,
            tracks,
            tick_div,
        })
    }
}

/// Note event kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NoteEventKind {
    NoteOn = 0,
    NoteOff = 1,
    PitchBend = 2,
    ControlChange = 3,
}

impl NoteEventKind {
    pub fn from_u8(v: u8) -> Self {
        match v & 0x03 {
            0 => Self::NoteOn,
            1 => Self::NoteOff,
            2 => Self::PitchBend,
            _ => Self::ControlChange,
        }
    }
}

/// Compact note event — 4 bytes
///
/// Layout: [delta_tick:12][channel:4][note:7][velocity:7][kind:2]
/// = 32 bits = 4 bytes per event
#[derive(Debug, Clone, Copy)]
pub struct NoteEvent {
    /// Delta time in ticks from previous event (0-4095)
    pub delta_tick: u16,
    /// Channel/instrument (0-15)
    pub channel: u8,
    /// MIDI note number (0-127)
    pub note: u8,
    /// Velocity (0-127)
    pub velocity: u8,
    /// Event kind
    pub kind: NoteEventKind,
}

impl NoteEvent {
    /// Pack into 4 bytes
    pub fn to_bytes(&self) -> [u8; 4] {
        let mut bits: u32 = 0;
        bits |= (self.delta_tick as u32 & 0xFFF) << 20;
        bits |= ((self.channel & 0x0F) as u32) << 16;
        bits |= ((self.note & 0x7F) as u32) << 9;
        bits |= ((self.velocity & 0x7F) as u32) << 2;
        bits |= (self.kind as u32) & 0x03;
        bits.to_le_bytes()
    }

    /// Unpack from 4 bytes
    pub fn from_bytes(data: &[u8; 4]) -> Self {
        let bits = u32::from_le_bytes(*data);
        Self {
            delta_tick: ((bits >> 20) & 0xFFF) as u16,
            channel: ((bits >> 16) & 0x0F) as u8,
            note: ((bits >> 9) & 0x7F) as u8,
            velocity: ((bits >> 2) & 0x7F) as u8,
            kind: NoteEventKind::from_u8((bits & 0x03) as u8),
        }
    }
}

/// Complete score: header + events
#[derive(Debug, Clone)]
pub struct Score {
    pub header: ScoreHeader,
    pub events: Vec<NoteEvent>,
}

impl Score {
    pub fn new(tempo_bpm: u16, tracks: u8) -> Self {
        Self {
            header: ScoreHeader {
                tempo_bpm,
                tracks,
                tick_div: 96,
            },
            events: Vec::new(),
        }
    }

    /// Add a note event
    pub fn add_event(&mut self, event: NoteEvent) {
        self.events.push(event);
    }

    /// Total size in bytes
    pub fn size_bytes(&self) -> usize {
        8 + self.events.len() * 4
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.size_bytes());
        buf.extend_from_slice(&self.header.to_bytes());
        for event in &self.events {
            buf.extend_from_slice(&event.to_bytes());
        }
        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        let header = ScoreHeader::from_bytes(data)?;
        let event_data = &data[8..];
        let event_count = event_data.len() / 4;
        let mut events = Vec::with_capacity(event_count);
        for i in 0..event_count {
            let offset = i * 4;
            let bytes: [u8; 4] = [
                event_data[offset],
                event_data[offset + 1],
                event_data[offset + 2],
                event_data[offset + 3],
            ];
            events.push(NoteEvent::from_bytes(&bytes));
        }
        Some(Self { header, events })
    }

    /// Total duration in ticks
    pub fn total_ticks(&self) -> u32 {
        self.events.iter().map(|e| e.delta_tick as u32).sum()
    }

    /// Total duration in seconds
    #[inline(always)]
    pub fn duration_secs(&self) -> f32 {
        let tps = self.header.ticks_per_second();
        if tps > 0.0 {
            self.total_ticks() as f32 * tps.recip()
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_event_roundtrip() {
        let event = NoteEvent {
            delta_tick: 96,
            channel: 0,
            note: 60,
            velocity: 100,
            kind: NoteEventKind::NoteOn,
        };
        let bytes = event.to_bytes();
        let decoded = NoteEvent::from_bytes(&bytes);
        assert_eq!(decoded.delta_tick, 96);
        assert_eq!(decoded.channel, 0);
        assert_eq!(decoded.note, 60);
        assert_eq!(decoded.velocity, 100);
        assert_eq!(decoded.kind, NoteEventKind::NoteOn);
    }

    #[test]
    fn test_score_header_roundtrip() {
        let header = ScoreHeader {
            tempo_bpm: 120,
            tracks: 4,
            tick_div: 96,
        };
        let bytes = header.to_bytes();
        let decoded = ScoreHeader::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.tempo_bpm, 120);
        assert_eq!(decoded.tracks, 4);
        assert_eq!(decoded.tick_div, 96);
    }

    #[test]
    fn test_score_size() {
        let mut score = Score::new(120, 2);
        // Add C major chord
        for note in [60, 64, 67] {
            score.add_event(NoteEvent {
                delta_tick: 0,
                channel: 0,
                note,
                velocity: 80,
                kind: NoteEventKind::NoteOn,
            });
        }
        // Note off after 1 beat
        for note in [60, 64, 67] {
            score.add_event(NoteEvent {
                delta_tick: if note == 60 { 96 } else { 0 },
                channel: 0,
                note,
                velocity: 0,
                kind: NoteEventKind::NoteOff,
            });
        }
        assert_eq!(score.size_bytes(), 8 + 6 * 4); // header + 6 events
        assert_eq!(score.size_bytes(), 32);
    }

    #[test]
    fn test_score_roundtrip() {
        let mut score = Score::new(140, 1);
        score.add_event(NoteEvent {
            delta_tick: 0,
            channel: 0,
            note: 72,
            velocity: 127,
            kind: NoteEventKind::NoteOn,
        });
        let bytes = score.to_bytes();
        let decoded = Score::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.header.tempo_bpm, 140);
        assert_eq!(decoded.events.len(), 1);
        assert_eq!(decoded.events[0].note, 72);
    }

    #[test]
    fn test_ticks_per_second() {
        let header = ScoreHeader {
            tempo_bpm: 120,
            tracks: 1,
            tick_div: 96,
        };
        // 120 BPM = 2 beats/sec, 96 ticks/beat = 192 ticks/sec
        let tps = header.ticks_per_second();
        assert!((tps - 192.0).abs() < 0.1);
    }
}
