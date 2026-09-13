//! Zero-shot cover pipeline — `audio` → `AbcTune` → style rewrite.
//!
//! ALICE-Synth ships the *composition* of the pipeline. The *transcription*
//! stage (audio waveform → symbolic ABC) is a hard ML problem that lives in
//! external crates (e.g. an eventual `alice-music-transcribe` wrapping
//! `SheetSage2` or a similar model). This module provides:
//!
//! - The [`Transcriber`] trait — a contract for concrete transcription
//!   backends. Zero-sized dependencies here: any crate can implement it
//!   without pulling in ALICE-Synth's other features.
//! - [`CoverPipeline`] — a composition helper that pairs a transcriber with
//!   the ALICE-Synth intent packet + procedural synthesizer to produce a
//!   "cover" of the input audio in a target style.
//! - [`TranscribeError`] — a `Copy` error enum with no dynamic context, so
//!   `no_std + alloc` targets don't need to allocate for error propagation.
//!
//! # Example
//!
//! ```
//! use alice_synth::abc::parse;
//! use alice_synth::cover::{CoverPipeline, TranscribeError, Transcriber};
//! use alice_synth::intent::{mode, mood, MusicIntent};
//!
//! /// A stub transcriber that always returns a fixed tune — useful for
//! /// exercising the pipeline without a real ML model.
//! struct StubTranscriber;
//!
//! impl Transcriber for StubTranscriber {
//!     fn transcribe(
//!         &self,
//!         _samples: &[f32],
//!         _sample_rate: u32,
//!     ) -> Result<alice_synth::abc::AbcTune, TranscribeError> {
//!         parse("X:1\nM:4/4\nL:1/4\nQ:120\nK:C\nC D E F |\n")
//!             .map_err(|_| TranscribeError::TranscriptionFailed)
//!     }
//! }
//!
//! let pipeline = CoverPipeline::new(StubTranscriber);
//! let target = MusicIntent {
//!     mood: mood::MYSTERIOUS,
//!     mode: mode::DORIAN,
//!     ..MusicIntent::DEFAULT_C_MAJOR
//! };
//! let cover = pipeline.cover(&[0.0_f32; 256], 44_100, target).unwrap();
//! // The cover preserves the source's structural properties but re-generates
//! // with the target style.
//! assert!(!cover.body.is_empty());
//! ```
//!
//! Author: Moroya Sakamoto

use crate::abc::AbcTune;
use crate::intent::{MusicIntent, PlanHead, ProceduralPlanHead};

/// Errors that a [`Transcriber`] can report.
///
/// The variants are deliberately coarse so all sensible implementations can
/// map their internal error codes onto them without needing to expose
/// implementation details in a public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscribeError {
    /// The input PCM was empty or shorter than the minimum required duration.
    InputTooShort,
    /// The sample rate is not supported by the concrete transcriber.
    UnsupportedSampleRate,
    /// The transcriber's underlying model or resources are not available in
    /// this environment (e.g. GPU absent, weights not loaded).
    ModelNotAvailable,
    /// The transcription itself failed (model produced no output or produced
    /// invalid ABC).
    TranscriptionFailed,
}

/// Turns raw PCM audio into a symbolic [`AbcTune`].
///
/// Concrete implementations belong in downstream crates (e.g. an
/// `alice-music-transcribe` crate wrapping `SheetSage2` weights, or a network
/// client that delegates to a hosted service). ALICE-Synth only defines the
/// trait so the [`CoverPipeline`] can compose against a stable interface.
///
/// **Sample layout**: interleaved stereo `[L, R, L, R, …]` at `sample_rate`
/// Hz. Mono is accepted if the implementation supports it — see each
/// implementation's docs for details.
pub trait Transcriber {
    /// Transcribe `samples` (interleaved stereo at `sample_rate` Hz) into an
    /// [`AbcTune`].
    ///
    /// # Errors
    ///
    /// Returns [`TranscribeError`] variants depending on the failure mode.
    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<AbcTune, TranscribeError>;
}

/// Composition of a [`Transcriber`] + a [`PlanHead`] that produces "covers":
/// an input audio file re-rendered in a target style.
///
/// The default plan head is [`ProceduralPlanHead`]. Downstream crates that
/// have an LLM-backed plan head can construct a pipeline with
/// [`CoverPipeline::with_plan_head`] to swap in that alternative.
#[derive(Debug, Clone, Copy)]
pub struct CoverPipeline<T: Transcriber, P: PlanHead = ProceduralPlanHead> {
    transcriber: T,
    plan_head: P,
}

impl<T: Transcriber> CoverPipeline<T, ProceduralPlanHead> {
    /// Construct a pipeline with the default [`ProceduralPlanHead`].
    #[must_use]
    pub fn new(transcriber: T) -> Self {
        Self {
            transcriber,
            plan_head: ProceduralPlanHead,
        }
    }
}

impl<T: Transcriber, P: PlanHead> CoverPipeline<T, P> {
    /// Construct a pipeline with a custom plan head.
    ///
    /// Use this to slot in an LLM-backed generator produced by a downstream
    /// crate (see [`crate::intent::PlanHead`]).
    #[must_use]
    pub fn with_plan_head(transcriber: T, plan_head: P) -> Self {
        Self {
            transcriber,
            plan_head,
        }
    }

    /// Run the pipeline: transcribe `source_audio`, extract its structural
    /// intent, override the categorical fields with `target_style`, and
    /// re-render.
    ///
    /// The rewrite policy is intentionally conservative: `key`, `tempo`, and
    /// `length_bars` come from the source (so the cover has the same broad
    /// shape as the input); `genre`, `mood`, `mode`, and `variation_seed`
    /// come from the target (so the cover has the intended new style).
    ///
    /// # Errors
    ///
    /// Propagates any error from the transcriber.
    pub fn cover(
        &self,
        source_audio: &[f32],
        sample_rate: u32,
        target_style: MusicIntent,
    ) -> Result<AbcTune, TranscribeError> {
        let source_tune = self.transcriber.transcribe(source_audio, sample_rate)?;
        let source_intent = MusicIntent::from_tune(&source_tune);
        let rewritten = MusicIntent {
            genre: target_style.genre,
            mood: target_style.mood,
            mode: target_style.mode,
            variation_seed: target_style.variation_seed,
            // Preserve source key, tempo, and length so the cover shares the
            // input's structural properties.
            key: source_intent.key,
            length_bars: source_intent.length_bars,
            tempo_bpm_offset: source_intent.tempo_bpm_offset,
        };
        Ok(self.plan_head.plan(&rewritten))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abc::parse;
    use crate::intent::{genre, mode, mood};

    /// Stub transcriber that always returns the same tune. Used to exercise
    /// the pipeline logic without a real ML backend.
    struct FixedTranscriber(&'static str);

    impl Transcriber for FixedTranscriber {
        fn transcribe(
            &self,
            _samples: &[f32],
            _sample_rate: u32,
        ) -> Result<AbcTune, TranscribeError> {
            parse(self.0).map_err(|_| TranscribeError::TranscriptionFailed)
        }
    }

    /// Stub transcriber that always fails, for negative testing.
    struct FailingTranscriber;

    impl Transcriber for FailingTranscriber {
        fn transcribe(
            &self,
            _samples: &[f32],
            _sample_rate: u32,
        ) -> Result<AbcTune, TranscribeError> {
            Err(TranscribeError::ModelNotAvailable)
        }
    }

    #[test]
    fn cover_pipeline_runs_end_to_end() {
        let pipeline = CoverPipeline::new(FixedTranscriber("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n"));
        let target = MusicIntent::DEFAULT_C_MAJOR;
        let cover = pipeline.cover(&[0.0_f32; 256], 44_100, target).unwrap();
        assert!(!cover.body.is_empty());
    }

    #[test]
    fn cover_preserves_source_key_and_tempo() {
        let pipeline = CoverPipeline::new(FixedTranscriber(
            "X:1\nM:4/4\nL:1/4\nQ:180\nK:G\nG A B c |\n",
        ));
        // Target style asks for D major, but the pipeline should keep the source's G.
        let target = MusicIntent {
            key: 2, // D
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        let cover = pipeline.cover(&[0.0_f32; 256], 44_100, target).unwrap();
        assert_eq!(cover.header.key.sharps, 1); // G major from source
        assert_eq!(cover.header.tempo_bpm, 180);
    }

    #[test]
    fn cover_applies_target_mood_and_mode() {
        // Set an unusual target mood/mode and confirm the resulting tune's
        // key signature reflects the mode offset applied on top of the
        // source key.
        let pipeline = CoverPipeline::new(FixedTranscriber("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n"));
        let target = MusicIntent {
            mood: mood::MYSTERIOUS,
            mode: mode::DORIAN,
            genre: genre::JAZZ,
            variation_seed: 0xBEEF,
            ..MusicIntent::DEFAULT_C_MAJOR
        };
        let cover = pipeline.cover(&[0.0_f32; 256], 44_100, target).unwrap();
        // Source key is C (0 sharps); Dorian offset is -2. Cover key = -2.
        assert_eq!(cover.header.key.sharps, -2);
    }

    #[test]
    fn cover_propagates_transcriber_error() {
        let pipeline = CoverPipeline::new(FailingTranscriber);
        let err = pipeline
            .cover(&[0.0_f32; 256], 44_100, MusicIntent::DEFAULT_C_MAJOR)
            .unwrap_err();
        assert_eq!(err, TranscribeError::ModelNotAvailable);
    }

    #[test]
    fn cover_pipeline_supports_custom_plan_head() {
        struct DoublingPlanHead;
        impl PlanHead for DoublingPlanHead {
            fn plan(&self, intent: &MusicIntent) -> AbcTune {
                // Double the length before calling through to the procedural head.
                let mut intent = *intent;
                intent.length_bars = intent.length_bars.saturating_mul(2);
                ProceduralPlanHead.plan(&intent)
            }
        }
        let pipeline = CoverPipeline::with_plan_head(
            FixedTranscriber("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n"),
            DoublingPlanHead,
        );
        let cover = pipeline
            .cover(&[0.0_f32; 256], 44_100, MusicIntent::DEFAULT_C_MAJOR)
            .unwrap();
        // Doubling the intent's length_bars must yield more notes than the
        // default plan head would produce for the same source.
        let bare = CoverPipeline::new(FixedTranscriber("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n"))
            .cover(&[0.0_f32; 256], 44_100, MusicIntent::DEFAULT_C_MAJOR)
            .unwrap();
        assert!(cover.body.len() >= bare.body.len());
    }

    #[test]
    fn transcribe_error_variants_are_all_copyable() {
        // Compile-time check: TranscribeError must be `Copy` so it can be
        // passed by value without heap allocation.
        fn assert_copy<T: Copy>() {}
        assert_copy::<TranscribeError>();
    }
}
