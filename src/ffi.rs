//! FFI bindings for ALICE-Synth (C/C++/C# interop)
//!
//! 20 exported functions for cross-language audio synthesis.
//!
//! Author: Moroya Sakamoto

extern crate alloc;
use alloc::boxed::Box;
use core::ptr;

use crate::envelope::Adsr;
use crate::oscillator::{Oscillator, Waveform};
use crate::patch::{FmPatch, Patch};
use crate::score::Score;
use crate::synth::Synthesizer;

// ============================================================================
// Types
// ============================================================================

/// Opaque handle to a Synthesizer
pub type SynthHandle = *mut core::ffi::c_void;

/// Opaque handle to an Oscillator
pub type OscHandle = *mut core::ffi::c_void;

/// Null handle constant
pub const SYNTH_HANDLE_NULL: SynthHandle = ptr::null_mut();

/// Result code for FFI operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthResult {
    /// Operation succeeded
    Ok = 0,
    /// Invalid handle provided
    InvalidHandle = 1,
    /// Null pointer provided
    NullPointer = 2,
    /// Invalid parameter value
    InvalidParameter = 3,
    /// Unknown error
    Unknown = 99,
}

/// Version information
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VersionInfo {
    /// Major version number
    pub major: u16,
    /// Minor version number
    pub minor: u16,
    /// Patch version number
    pub patch: u16,
}

// ============================================================================
// Info
// ============================================================================

/// Get version info
#[no_mangle]
pub extern "C" fn alice_synth_version() -> VersionInfo {
    VersionInfo {
        major: 0,
        minor: 1,
        patch: 0,
    }
}

// ============================================================================
// Synthesizer
// ============================================================================

/// Create a new synthesizer
#[no_mangle]
pub extern "C" fn alice_synth_new(sample_rate: u32) -> SynthHandle {
    let synth = Box::new(Synthesizer::new(sample_rate));
    Box::into_raw(synth) as SynthHandle
}

/// Free a synthesizer
///
/// # Safety
///
/// `handle` must be a valid handle returned by `alice_synth_new`.
#[no_mangle]
pub unsafe extern "C" fn alice_synth_free(handle: SynthHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle as *mut Synthesizer) });
    }
}

/// Load an FM preset patch to a channel
#[no_mangle]
pub extern "C" fn alice_synth_load_fm_preset(
    handle: SynthHandle,
    channel: u8,
    preset: u8,
) -> SynthResult {
    if handle.is_null() {
        return SynthResult::InvalidHandle;
    }
    let synth = unsafe { &mut *(handle as *mut Synthesizer) };
    let patch = match preset {
        0 => Patch::Fm(FmPatch::electric_piano()),
        1 => Patch::Fm(FmPatch::bell()),
        _ => return SynthResult::InvalidParameter,
    };
    synth.load_patch(channel as usize, patch);
    SynthResult::Ok
}

/// Trigger note on
#[no_mangle]
pub extern "C" fn alice_synth_note_on(
    handle: SynthHandle,
    channel: u8,
    note: u8,
    velocity: u8,
) -> SynthResult {
    if handle.is_null() {
        return SynthResult::InvalidHandle;
    }
    let synth = unsafe { &mut *(handle as *mut Synthesizer) };
    synth.note_on(channel, note, velocity);
    SynthResult::Ok
}

/// Trigger note off
#[no_mangle]
pub extern "C" fn alice_synth_note_off(handle: SynthHandle, channel: u8, note: u8) -> SynthResult {
    if handle.is_null() {
        return SynthResult::InvalidHandle;
    }
    let synth = unsafe { &mut *(handle as *mut Synthesizer) };
    synth.note_off(channel, note);
    SynthResult::Ok
}

/// Render audio into a float buffer
///
/// Returns the number of samples written.
///
/// # Safety
///
/// `buffer` must point to an array of at least `buffer_len` f32 elements.
#[no_mangle]
pub unsafe extern "C" fn alice_synth_render(
    handle: SynthHandle,
    buffer: *mut f32,
    buffer_len: u32,
) -> i32 {
    if handle.is_null() || buffer.is_null() {
        return -1;
    }
    let synth = unsafe { &mut *(handle as *mut Synthesizer) };
    let buf = unsafe { core::slice::from_raw_parts_mut(buffer, buffer_len as usize) };
    synth.render(buf) as i32
}

/// Render audio into an i16 buffer
///
/// # Safety
///
/// `buffer` must point to an array of at least `buffer_len` i16 elements.
#[no_mangle]
pub unsafe extern "C" fn alice_synth_render_i16(
    handle: SynthHandle,
    buffer: *mut i16,
    buffer_len: u32,
) -> i32 {
    if handle.is_null() || buffer.is_null() {
        return -1;
    }
    let synth = unsafe { &mut *(handle as *mut Synthesizer) };
    let buf = unsafe { core::slice::from_raw_parts_mut(buffer, buffer_len as usize) };
    synth.render_i16(buf) as i32
}

/// Get the number of active voices
#[no_mangle]
pub extern "C" fn alice_synth_active_voices(handle: SynthHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let synth = unsafe { &*(handle as *const Synthesizer) };
    synth.active_voice_count() as i32
}

/// Set master volume (0.0 - 1.0)
#[no_mangle]
pub extern "C" fn alice_synth_set_volume(handle: SynthHandle, volume: f32) -> SynthResult {
    if handle.is_null() {
        return SynthResult::InvalidHandle;
    }
    let synth = unsafe { &mut *(handle as *mut Synthesizer) };
    synth.master_volume = volume;
    SynthResult::Ok
}

// ============================================================================
// Score
// ============================================================================

/// Load a score from binary data
///
/// # Safety
///
/// `data` must point to an array of at least `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn alice_synth_load_score(
    handle: SynthHandle,
    data: *const u8,
    len: u32,
) -> SynthResult {
    if handle.is_null() || data.is_null() {
        return SynthResult::InvalidHandle;
    }
    let bytes = unsafe { core::slice::from_raw_parts(data, len as usize) };
    let score = match Score::from_bytes(bytes) {
        Some(s) => s,
        None => return SynthResult::InvalidParameter,
    };
    let synth = unsafe { &mut *(handle as *mut Synthesizer) };
    synth.load_score(&score);
    SynthResult::Ok
}

// ============================================================================
// Oscillator (standalone)
// ============================================================================

/// Create a new oscillator (0=Sine, 1=Saw, 2=Square, 3=Triangle, 4=Noise)
#[no_mangle]
pub extern "C" fn alice_osc_new(waveform: u8) -> OscHandle {
    let osc = Box::new(Oscillator::new(Waveform::from_u8(waveform)));
    Box::into_raw(osc) as OscHandle
}

/// Free an oscillator
///
/// # Safety
///
/// `handle` must be a valid handle returned by `alice_osc_new`.
#[no_mangle]
pub unsafe extern "C" fn alice_osc_free(handle: OscHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle as *mut Oscillator) });
    }
}

/// Generate the next sample
#[no_mangle]
pub extern "C" fn alice_osc_next(handle: OscHandle, freq_hz: f32, inv_sample_rate: f32) -> f32 {
    if handle.is_null() {
        return 0.0;
    }
    let osc = unsafe { &mut *(handle as *mut Oscillator) };
    osc.next_sample(freq_hz, inv_sample_rate)
}

/// Reset oscillator phase
#[no_mangle]
pub extern "C" fn alice_osc_reset(handle: OscHandle) {
    if !handle.is_null() {
        let osc = unsafe { &mut *(handle as *mut Oscillator) };
        osc.reset();
    }
}

// ============================================================================
// Envelope (standalone)
// ============================================================================

/// Create an ADSR envelope from milliseconds
#[no_mangle]
pub extern "C" fn alice_adsr_from_ms(
    attack_ms: f32,
    decay_ms: f32,
    sustain: f32,
    release_ms: f32,
    sample_rate: f32,
) -> Adsr {
    Adsr::from_ms(attack_ms, decay_ms, sustain, release_ms, sample_rate)
}

// ============================================================================
// Utility
// ============================================================================

/// Convert MIDI note number to frequency
#[no_mangle]
pub extern "C" fn alice_synth_midi_to_freq(note: u8) -> f32 {
    crate::oscillator::midi_to_freq(note)
}

/// Fast sine approximation
#[no_mangle]
pub extern "C" fn alice_synth_sin_approx(x: f32) -> f32 {
    crate::oscillator::sin_approx(x)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = alice_synth_version();
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 1);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn test_synth_create_and_free() {
        let handle = alice_synth_new(44100);
        assert!(!handle.is_null());
        unsafe { alice_synth_free(handle) };
    }

    #[test]
    fn test_synth_free_null() {
        unsafe { alice_synth_free(ptr::null_mut()) };
    }

    #[test]
    fn test_load_fm_preset() {
        let handle = alice_synth_new(44100);
        assert_eq!(alice_synth_load_fm_preset(handle, 0, 0), SynthResult::Ok);
        assert_eq!(alice_synth_load_fm_preset(handle, 1, 1), SynthResult::Ok);
        assert_eq!(
            alice_synth_load_fm_preset(handle, 0, 99),
            SynthResult::InvalidParameter
        );
        unsafe { alice_synth_free(handle) };
    }

    #[test]
    fn test_note_on_off() {
        let handle = alice_synth_new(44100);
        alice_synth_load_fm_preset(handle, 0, 0);
        assert_eq!(alice_synth_note_on(handle, 0, 60, 100), SynthResult::Ok);
        assert!(alice_synth_active_voices(handle) > 0);
        assert_eq!(alice_synth_note_off(handle, 0, 60), SynthResult::Ok);
        unsafe { alice_synth_free(handle) };
    }

    #[test]
    fn test_note_on_null_handle() {
        assert_eq!(
            alice_synth_note_on(ptr::null_mut(), 0, 60, 100),
            SynthResult::InvalidHandle
        );
    }

    #[test]
    fn test_render() {
        let handle = alice_synth_new(44100);
        alice_synth_load_fm_preset(handle, 0, 0);
        alice_synth_note_on(handle, 0, 69, 100);
        let mut buffer = [0.0f32; 1024];
        let n = unsafe { alice_synth_render(handle, buffer.as_mut_ptr(), 1024) };
        assert_eq!(n, 1024);
        // At least some samples should be non-zero
        assert!(buffer.iter().any(|&s| s != 0.0));
        unsafe { alice_synth_free(handle) };
    }

    #[test]
    fn test_render_null() {
        assert_eq!(
            unsafe { alice_synth_render(ptr::null_mut(), ptr::null_mut(), 0) },
            -1
        );
    }

    #[test]
    fn test_render_i16() {
        let handle = alice_synth_new(44100);
        alice_synth_load_fm_preset(handle, 0, 0);
        alice_synth_note_on(handle, 0, 60, 100);
        let mut buffer = [0i16; 512];
        let n = unsafe { alice_synth_render_i16(handle, buffer.as_mut_ptr(), 512) };
        assert_eq!(n, 512);
        unsafe { alice_synth_free(handle) };
    }

    #[test]
    fn test_set_volume() {
        let handle = alice_synth_new(44100);
        assert_eq!(alice_synth_set_volume(handle, 0.5), SynthResult::Ok);
        assert_eq!(
            alice_synth_set_volume(ptr::null_mut(), 0.5),
            SynthResult::InvalidHandle
        );
        unsafe { alice_synth_free(handle) };
    }

    #[test]
    fn test_active_voices() {
        let handle = alice_synth_new(44100);
        assert_eq!(alice_synth_active_voices(handle), 0);
        assert_eq!(alice_synth_active_voices(ptr::null_mut()), -1);
        unsafe { alice_synth_free(handle) };
    }

    #[test]
    fn test_osc_create_and_free() {
        let handle = alice_osc_new(0); // Sine
        assert!(!handle.is_null());
        unsafe { alice_osc_free(handle) };
    }

    #[test]
    fn test_osc_next_sample() {
        let handle = alice_osc_new(0); // Sine
        let sample = alice_osc_next(handle, 440.0, 1.0 / 44100.0);
        // First sample of sine at 440Hz should be a small positive value
        assert!(sample >= -1.0 && sample <= 1.0);
        unsafe { alice_osc_free(handle) };
    }

    #[test]
    fn test_osc_reset() {
        let handle = alice_osc_new(1); // Saw
        alice_osc_next(handle, 440.0, 1.0 / 44100.0);
        alice_osc_reset(handle);
        // After reset, should behave as newly created
        let sample = alice_osc_next(handle, 440.0, 1.0 / 44100.0);
        assert!(sample >= -1.0 && sample <= 1.0);
        unsafe { alice_osc_free(handle) };
    }

    #[test]
    fn test_osc_null_returns_zero() {
        assert_eq!(alice_osc_next(ptr::null_mut(), 440.0, 1.0 / 44100.0), 0.0);
    }

    #[test]
    fn test_adsr_from_ms() {
        let adsr = alice_adsr_from_ms(10.0, 50.0, 0.7, 100.0, 44100.0);
        assert!(adsr.attack > 0);
        assert!(adsr.decay > 0);
        assert!((adsr.sustain - 0.7).abs() < f32::EPSILON);
        assert!(adsr.release > 0);
    }

    #[test]
    fn test_midi_to_freq() {
        let freq = alice_synth_midi_to_freq(69); // A4
        assert!((freq - 440.0).abs() < 0.1);
    }

    #[test]
    fn test_sin_approx() {
        let val = alice_synth_sin_approx(0.0);
        assert!(val.abs() < 0.01);
    }

    #[test]
    fn test_full_lifecycle() {
        let synth = alice_synth_new(44100);
        alice_synth_load_fm_preset(synth, 0, 0);
        alice_synth_set_volume(synth, 0.8);
        alice_synth_note_on(synth, 0, 60, 100);
        alice_synth_note_on(synth, 0, 64, 80);
        alice_synth_note_on(synth, 0, 67, 80);
        assert!(alice_synth_active_voices(synth) >= 3);

        let mut buf = [0.0f32; 4096];
        let n = unsafe { alice_synth_render(synth, buf.as_mut_ptr(), 4096) };
        assert_eq!(n, 4096);
        assert!(buf.iter().any(|&s| s != 0.0));

        alice_synth_note_off(synth, 0, 60);
        alice_synth_note_off(synth, 0, 64);
        alice_synth_note_off(synth, 0, 67);

        unsafe { alice_synth_free(synth) };
    }
}
