// ALICE-Synth Unity C# Bindings
// 20 FFI functions for procedural audio synthesis
//
// Author: Moroya Sakamoto

using System;
using System.Runtime.InteropServices;

namespace AliceSynth
{
    // ========================================================================
    // Types
    // ========================================================================

    public enum SynthResult : int
    {
        Ok = 0,
        InvalidHandle = 1,
        NullPointer = 2,
        InvalidParameter = 3,
        Unknown = 99,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VersionInfo
    {
        public ushort Major;
        public ushort Minor;
        public ushort Patch;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Adsr
    {
        public uint Attack;
        public uint Decay;
        public float Sustain;
        public uint Release;
    }

    // ========================================================================
    // Native Imports
    // ========================================================================

    public static class Native
    {
#if UNITY_IOS && !UNITY_EDITOR
        private const string Lib = "__Internal";
#else
        private const string Lib = "alice_synth";
#endif

        // Info
        [DllImport(Lib)] public static extern VersionInfo alice_synth_version();

        // Synthesizer lifecycle
        [DllImport(Lib)] public static extern IntPtr alice_synth_new(uint sampleRate);
        [DllImport(Lib)] public static extern void alice_synth_free(IntPtr handle);

        // Patch
        [DllImport(Lib)] public static extern SynthResult alice_synth_load_fm_preset(IntPtr handle, byte channel, byte preset);

        // Note control
        [DllImport(Lib)] public static extern SynthResult alice_synth_note_on(IntPtr handle, byte channel, byte note, byte velocity);
        [DllImport(Lib)] public static extern SynthResult alice_synth_note_off(IntPtr handle, byte channel, byte note);

        // Render
        [DllImport(Lib)] public static extern int alice_synth_render(IntPtr handle, IntPtr buffer, uint bufferLen);
        [DllImport(Lib)] public static extern int alice_synth_render_i16(IntPtr handle, IntPtr buffer, uint bufferLen);

        // State
        [DllImport(Lib)] public static extern int alice_synth_active_voices(IntPtr handle);
        [DllImport(Lib)] public static extern SynthResult alice_synth_set_volume(IntPtr handle, float volume);

        // Score
        [DllImport(Lib)] public static extern SynthResult alice_synth_load_score(IntPtr handle, IntPtr data, uint len);

        // Oscillator lifecycle
        [DllImport(Lib)] public static extern IntPtr alice_osc_new(byte waveform);
        [DllImport(Lib)] public static extern void alice_osc_free(IntPtr handle);
        [DllImport(Lib)] public static extern float alice_osc_next(IntPtr handle, float freqHz, float invSampleRate);
        [DllImport(Lib)] public static extern void alice_osc_reset(IntPtr handle);

        // Envelope
        [DllImport(Lib)] public static extern Adsr alice_adsr_from_ms(float attackMs, float decayMs, float sustain, float releaseMs, float sampleRate);

        // Utility
        [DllImport(Lib)] public static extern float alice_synth_midi_to_freq(byte note);
        [DllImport(Lib)] public static extern float alice_synth_sin_approx(float x);
    }

    // ========================================================================
    // Managed Wrappers
    // ========================================================================

    /// <summary>
    /// RAII wrapper for the ALICE-Synth synthesizer.
    /// </summary>
    public sealed class Synthesizer : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public Synthesizer(uint sampleRate)
        {
            _handle = Native.alice_synth_new(sampleRate);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create synthesizer");
        }

        public SynthResult LoadFmPreset(byte channel, byte preset)
            => Native.alice_synth_load_fm_preset(_handle, channel, preset);

        public SynthResult NoteOn(byte channel, byte note, byte velocity)
            => Native.alice_synth_note_on(_handle, channel, note, velocity);

        public SynthResult NoteOff(byte channel, byte note)
            => Native.alice_synth_note_off(_handle, channel, note);

        public unsafe int Render(float[] buffer)
        {
            fixed (float* p = buffer)
            {
                return Native.alice_synth_render(_handle, (IntPtr)p, (uint)buffer.Length);
            }
        }

        public unsafe int RenderI16(short[] buffer)
        {
            fixed (short* p = buffer)
            {
                return Native.alice_synth_render_i16(_handle, (IntPtr)p, (uint)buffer.Length);
            }
        }

        public int ActiveVoices => Native.alice_synth_active_voices(_handle);

        public SynthResult SetVolume(float volume)
            => Native.alice_synth_set_volume(_handle, volume);

        public unsafe SynthResult LoadScore(byte[] data)
        {
            fixed (byte* p = data)
            {
                return Native.alice_synth_load_score(_handle, (IntPtr)p, (uint)data.Length);
            }
        }

        public void Dispose()
        {
            if (!_disposed && _handle != IntPtr.Zero)
            {
                Native.alice_synth_free(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~Synthesizer() => Dispose();
    }

    /// <summary>
    /// RAII wrapper for the ALICE-Synth oscillator.
    /// </summary>
    public sealed class OscillatorHandle : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <param name="waveform">0=Sine, 1=Saw, 2=Square, 3=Triangle, 4=Noise</param>
        public OscillatorHandle(byte waveform)
        {
            _handle = Native.alice_osc_new(waveform);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create oscillator");
        }

        public float Next(float freqHz, float invSampleRate)
            => Native.alice_osc_next(_handle, freqHz, invSampleRate);

        public void Reset() => Native.alice_osc_reset(_handle);

        public void Dispose()
        {
            if (!_disposed && _handle != IntPtr.Zero)
            {
                Native.alice_osc_free(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~OscillatorHandle() => Dispose();
    }
}
