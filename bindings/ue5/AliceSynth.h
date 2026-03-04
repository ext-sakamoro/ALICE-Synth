// ALICE-Synth UE5 C++ Header
// 20 FFI functions for procedural audio synthesis
//
// Author: Moroya Sakamoto

#pragma once

#include <cstdint>
#include <utility>

// ============================================================================
// C API
// ============================================================================

extern "C" {

// Types
typedef void* SynthHandle;
typedef void* OscHandle;

enum SynthResult : int32_t {
    SYNTH_OK              = 0,
    SYNTH_INVALID_HANDLE  = 1,
    SYNTH_NULL_POINTER    = 2,
    SYNTH_INVALID_PARAM   = 3,
    SYNTH_UNKNOWN         = 99,
};

struct AliceSynthVersionInfo {
    uint16_t major;
    uint16_t minor;
    uint16_t patch;
};

struct AliceAdsr {
    uint32_t attack;
    uint32_t decay;
    float    sustain;
    uint32_t release;
};

// Info
AliceSynthVersionInfo alice_synth_version();

// Synthesizer lifecycle
SynthHandle  alice_synth_new(uint32_t sample_rate);
void         alice_synth_free(SynthHandle handle);

// Patch
SynthResult  alice_synth_load_fm_preset(SynthHandle handle, uint8_t channel, uint8_t preset);

// Note control
SynthResult  alice_synth_note_on(SynthHandle handle, uint8_t channel, uint8_t note, uint8_t velocity);
SynthResult  alice_synth_note_off(SynthHandle handle, uint8_t channel, uint8_t note);

// Render
int32_t      alice_synth_render(SynthHandle handle, float* buffer, uint32_t buffer_len);
int32_t      alice_synth_render_i16(SynthHandle handle, int16_t* buffer, uint32_t buffer_len);

// State
int32_t      alice_synth_active_voices(SynthHandle handle);
SynthResult  alice_synth_set_volume(SynthHandle handle, float volume);

// Score
SynthResult  alice_synth_load_score(SynthHandle handle, const uint8_t* data, uint32_t len);

// Oscillator lifecycle
OscHandle    alice_osc_new(uint8_t waveform);
void         alice_osc_free(OscHandle handle);
float        alice_osc_next(OscHandle handle, float freq_hz, float inv_sample_rate);
void         alice_osc_reset(OscHandle handle);

// Envelope
AliceAdsr    alice_adsr_from_ms(float attack_ms, float decay_ms, float sustain, float release_ms, float sample_rate);

// Utility
float        alice_synth_midi_to_freq(uint8_t note);
float        alice_synth_sin_approx(float x);

} // extern "C"

// ============================================================================
// RAII C++ Wrappers
// ============================================================================

namespace AliceSynth {

/// RAII wrapper for the synthesizer
class FSynthesizer {
public:
    explicit FSynthesizer(uint32_t SampleRate)
        : Handle(alice_synth_new(SampleRate)) {}

    ~FSynthesizer() {
        if (Handle) alice_synth_free(Handle);
    }

    // Move only
    FSynthesizer(FSynthesizer&& Other) noexcept : Handle(Other.Handle) { Other.Handle = nullptr; }
    FSynthesizer& operator=(FSynthesizer&& Other) noexcept {
        if (this != &Other) {
            if (Handle) alice_synth_free(Handle);
            Handle = Other.Handle;
            Other.Handle = nullptr;
        }
        return *this;
    }
    FSynthesizer(const FSynthesizer&) = delete;
    FSynthesizer& operator=(const FSynthesizer&) = delete;

    SynthResult LoadFmPreset(uint8_t Channel, uint8_t Preset) {
        return alice_synth_load_fm_preset(Handle, Channel, Preset);
    }

    SynthResult NoteOn(uint8_t Channel, uint8_t Note, uint8_t Velocity) {
        return alice_synth_note_on(Handle, Channel, Note, Velocity);
    }

    SynthResult NoteOff(uint8_t Channel, uint8_t Note) {
        return alice_synth_note_off(Handle, Channel, Note);
    }

    int32_t Render(float* Buffer, uint32_t Len) {
        return alice_synth_render(Handle, Buffer, Len);
    }

    int32_t RenderI16(int16_t* Buffer, uint32_t Len) {
        return alice_synth_render_i16(Handle, Buffer, Len);
    }

    int32_t ActiveVoices() const {
        return alice_synth_active_voices(Handle);
    }

    SynthResult SetVolume(float Volume) {
        return alice_synth_set_volume(Handle, Volume);
    }

    SynthResult LoadScore(const uint8_t* Data, uint32_t Len) {
        return alice_synth_load_score(Handle, Data, Len);
    }

    bool IsValid() const { return Handle != nullptr; }

private:
    SynthHandle Handle = nullptr;
};

/// RAII wrapper for the oscillator
class FOscillator {
public:
    /// @param Waveform 0=Sine, 1=Saw, 2=Square, 3=Triangle, 4=Noise
    explicit FOscillator(uint8_t Waveform)
        : Handle(alice_osc_new(Waveform)) {}

    ~FOscillator() {
        if (Handle) alice_osc_free(Handle);
    }

    // Move only
    FOscillator(FOscillator&& Other) noexcept : Handle(Other.Handle) { Other.Handle = nullptr; }
    FOscillator& operator=(FOscillator&& Other) noexcept {
        if (this != &Other) {
            if (Handle) alice_osc_free(Handle);
            Handle = Other.Handle;
            Other.Handle = nullptr;
        }
        return *this;
    }
    FOscillator(const FOscillator&) = delete;
    FOscillator& operator=(const FOscillator&) = delete;

    float Next(float FreqHz, float InvSampleRate) {
        return alice_osc_next(Handle, FreqHz, InvSampleRate);
    }

    void Reset() { alice_osc_reset(Handle); }

    bool IsValid() const { return Handle != nullptr; }

private:
    OscHandle Handle = nullptr;
};

} // namespace AliceSynth
