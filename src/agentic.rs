//! Agentic editing API — position-based diff and patch for [`AbcTune`].
//!
//! Enables LLM-driven revision loops (or human review workflows) that operate
//! on parsed ABC trees rather than raw text. The design is intentionally
//! minimal for the Phase 3.1 MVP:
//!
//! - **Position-based**: edits are recorded per body index rather than as an
//!   LCS/Myers-style script. Works well when revisions are localised (a
//!   handful of note changes in an otherwise unchanged tune) but does not
//!   detect element shifts. Sophisticated diff algorithms are a Phase 4
//!   candidate.
//! - **Header changes** (tempo, key) are tracked as optional overrides.
//! - **Extra voices** and tuplet / grace state are treated as opaque: only
//!   `body` and `header` participate in the diff.
//!
//! # Example
//!
//! ```
//! use alice_synth::abc::parse;
//! use alice_synth::agentic::AbcDiff;
//!
//! let original = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
//! let revised = parse("X:1\nM:4/4\nL:1/4\nQ:140\nK:C\nC D E G |\n").unwrap();
//! let diff = AbcDiff::compute(&original, &revised);
//!
//! assert_eq!(diff.tempo_change, Some(140));
//! assert!(!diff.body_replacements.is_empty());
//!
//! // Apply the diff to produce the revised tune from the original.
//! let patched = diff.apply(&original);
//! assert_eq!(patched.header.tempo_bpm, revised.header.tempo_bpm);
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::abc::{AbcElement, AbcHeader, AbcTune, KeySignature};

/// A structural diff between two [`AbcTune`]s.
///
/// Records header changes (tempo, key) as optional overrides and body changes
/// as a per-index list of insertions, removals, and replacements. See the
/// module-level documentation for the caveats of position-based diffing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AbcDiff {
    /// New tempo BPM if the original and revised values differ
    pub tempo_change: Option<u16>,
    /// New key signature if the original and revised keys differ
    pub key_change: Option<KeySignature>,
    /// Elements to insert. Each `(index, element)` means "at the given index
    /// in the *revised* body, insert this element." The list is
    /// canonicalised to ascending order.
    pub body_insertions: Vec<(usize, AbcElement)>,
    /// Indices (in the *original* body) whose elements are removed in the
    /// revised body.
    pub body_removals: Vec<usize>,
    /// Elements that appear at the same index in both bodies but differ.
    /// Recorded as `(index, new_element)`.
    pub body_replacements: Vec<(usize, AbcElement)>,
}

impl AbcDiff {
    /// Compute the diff from `original` to `revised`.
    ///
    /// The algorithm is `O(max(N, M))` — a single parallel walk through both
    /// bodies. When one side is shorter, remaining elements are recorded as
    /// insertions or removals as appropriate.
    #[must_use]
    pub fn compute(original: &AbcTune, revised: &AbcTune) -> Self {
        let tempo_change = (original.header.tempo_bpm != revised.header.tempo_bpm)
            .then_some(revised.header.tempo_bpm);
        let key_change = (original.header.key != revised.header.key).then_some(revised.header.key);

        let mut body_insertions: Vec<(usize, AbcElement)> = Vec::new();
        let mut body_removals: Vec<usize> = Vec::new();
        let mut body_replacements: Vec<(usize, AbcElement)> = Vec::new();

        let max_len = original.body.len().max(revised.body.len());
        for i in 0..max_len {
            match (original.body.get(i), revised.body.get(i)) {
                (Some(a), Some(b)) if a != b => {
                    body_replacements.push((i, *b));
                }
                (None, Some(b)) => {
                    body_insertions.push((i, *b));
                }
                (Some(_), None) => {
                    body_removals.push(i);
                }
                // Both present and equal, or both absent — nothing to record.
                (Some(_), Some(_)) | (None, None) => {}
            }
        }

        Self {
            tempo_change,
            key_change,
            body_insertions,
            body_removals,
            body_replacements,
        }
    }

    /// Apply this diff to `original` and return the resulting [`AbcTune`].
    ///
    /// The revision preserves `original.extra_voices` and every header field
    /// not touched by the diff. `body_removals` are processed first (in
    /// descending index order to keep earlier indices valid), then
    /// `body_replacements`, then `body_insertions` (in ascending order so
    /// each insertion sees the position it was recorded against).
    #[must_use]
    pub fn apply(&self, original: &AbcTune) -> AbcTune {
        let mut header = AbcHeader {
            tempo_bpm: self.tempo_change.unwrap_or(original.header.tempo_bpm),
            key: self.key_change.unwrap_or(original.header.key),
            ..original.header
        };
        // `..original.header` above copies meter_num/den, unit_num/den intact.
        // Rewriting `tempo_bpm` and `key` explicitly ensures the diff wins.
        header.tempo_bpm = self.tempo_change.unwrap_or(original.header.tempo_bpm);
        header.key = self.key_change.unwrap_or(original.header.key);

        let mut body = original.body.clone();

        // 1) Removals — descending so earlier indices stay valid.
        let mut removals = self.body_removals.clone();
        removals.sort_unstable_by(|a, b| b.cmp(a));
        for idx in removals {
            if idx < body.len() {
                body.remove(idx);
            }
        }

        // 2) Replacements — index refers to the (already-shrunk) body.
        for &(idx, elem) in &self.body_replacements {
            if idx < body.len() {
                body[idx] = elem;
            } else {
                body.push(elem);
            }
        }

        // 3) Insertions — ascending so each `idx` points to its intended slot.
        let mut insertions = self.body_insertions.clone();
        insertions.sort_unstable_by_key(|(idx, _)| *idx);
        for (idx, elem) in insertions {
            let clamped = idx.min(body.len());
            body.insert(clamped, elem);
        }

        AbcTune {
            header,
            body,
            extra_voices: original.extra_voices.clone(),
        }
    }

    /// `true` when the diff records no changes at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tempo_change.is_none()
            && self.key_change.is_none()
            && self.body_insertions.is_empty()
            && self.body_removals.is_empty()
            && self.body_replacements.is_empty()
    }

    /// Compute a diff using a **classic LCS (Longest Common Subsequence) backend**.
    ///
    /// The upgrade over [`Self::compute`] is that element shifts are detected as
    /// coherent insertions or removals rather than as a run of position-based
    /// replacements plus a tail insertion. This matters when LLM-produced
    /// revisions prepend or splice bars into a tune.
    ///
    /// # Algorithm
    ///
    /// - Classic dynamic-programming LCS table, `O(N × M)` time and space.
    /// - Backtracks the table to emit an edit script consisting only of
    ///   insertions (with revised-body indices) and removals (with original-body
    ///   indices). No `body_replacements` are produced — an in-place change is
    ///   represented as a removal followed by an insertion at the same index.
    /// - Header changes (tempo, key) are computed identically to
    ///   [`Self::compute`].
    ///
    /// # When to use which
    ///
    /// - [`Self::compute`] (position-based) — fast, small output. Best for
    ///   locally-scoped LLM revisions (a note changed here, tempo bumped there).
    /// - [`Self::compute_lcs`] (this method) — resilient to element shifts.
    ///   Best for structural edits (prepending a bar, splicing a phrase).
    ///
    /// The resulting `AbcDiff` values are interchangeable in [`Self::apply`].
    #[must_use]
    pub fn compute_lcs(original: &AbcTune, revised: &AbcTune) -> Self {
        let tempo_change = (original.header.tempo_bpm != revised.header.tempo_bpm)
            .then_some(revised.header.tempo_bpm);
        let key_change = (original.header.key != revised.header.key).then_some(revised.header.key);

        let (body_removals, body_insertions) = lcs_edit_script(&original.body, &revised.body);

        Self {
            tempo_change,
            key_change,
            body_insertions,
            body_removals,
            body_replacements: Vec::new(),
        }
    }
}

/// Compute an LCS-derived edit script: `(removals_in_original_indices,
/// insertions_at_revised_indices)`.
///
/// Both output vectors are sorted in ascending index order to keep the
/// `AbcDiff` value canonical.
fn lcs_edit_script(a: &[AbcElement], b: &[AbcElement]) -> (Vec<usize>, Vec<(usize, AbcElement)>) {
    let n = a.len();
    let m = b.len();

    // Fast paths for empty inputs — avoid allocating the DP table.
    if n == 0 {
        let insertions: Vec<(usize, AbcElement)> =
            b.iter().enumerate().map(|(i, e)| (i, *e)).collect();
        return (Vec::new(), insertions);
    }
    if m == 0 {
        let removals: Vec<usize> = (0..n).collect();
        return (removals, Vec::new());
    }

    // Classic O(N × M) LCS table. `table[i][j]` = LCS length for a[..i] vs b[..j].
    let mut table: Vec<Vec<usize>> = vec![vec![0; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            table[i][j] = if a[i - 1] == b[j - 1] {
                table[i - 1][j - 1] + 1
            } else {
                table[i - 1][j].max(table[i][j - 1])
            };
        }
    }

    // Backtrack. Prefer removals over insertions on ties for determinism.
    let mut removals: Vec<usize> = Vec::new();
    let mut insertions: Vec<(usize, AbcElement)> = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            i -= 1;
            j -= 1;
        } else if table[i - 1][j] >= table[i][j - 1] {
            removals.push(i - 1);
            i -= 1;
        } else {
            insertions.push((j - 1, b[j - 1]));
            j -= 1;
        }
    }
    while i > 0 {
        removals.push(i - 1);
        i -= 1;
    }
    while j > 0 {
        insertions.push((j - 1, b[j - 1]));
        j -= 1;
    }

    removals.reverse();
    insertions.reverse();
    (removals, insertions)
}

impl AbcTune {
    /// Convenience method: [`AbcDiff::compute`](AbcDiff::compute) with `self`
    /// as the original (position-based backend).
    #[must_use]
    pub fn diff(&self, other: &AbcTune) -> AbcDiff {
        AbcDiff::compute(self, other)
    }

    /// Convenience method: [`AbcDiff::compute_lcs`](AbcDiff::compute_lcs) with
    /// `self` as the original (LCS backend — shift-aware).
    #[must_use]
    pub fn diff_lcs(&self, other: &AbcTune) -> AbcDiff {
        AbcDiff::compute_lcs(self, other)
    }

    /// Convenience method: [`AbcDiff::apply`](AbcDiff::apply) treating `self`
    /// as the original.
    #[must_use]
    pub fn patch(&self, diff: &AbcDiff) -> AbcTune {
        diff.apply(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abc::parse;

    #[test]
    fn empty_diff_when_tunes_are_equal() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let diff = AbcDiff::compute(&a, &b);
        assert!(diff.is_empty());
    }

    #[test]
    fn detects_tempo_change() {
        let a = parse("X:1\nM:4/4\nL:1/4\nQ:120\nK:C\nC |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nQ:140\nK:C\nC |\n").unwrap();
        let diff = AbcDiff::compute(&a, &b);
        assert_eq!(diff.tempo_change, Some(140));
        assert!(diff.key_change.is_none());
    }

    #[test]
    fn detects_key_change() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:G\nC |\n").unwrap();
        let diff = AbcDiff::compute(&a, &b);
        assert_eq!(diff.key_change.map(|k| k.sharps), Some(1));
    }

    #[test]
    fn detects_body_replacement() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E G |\n").unwrap();
        let diff = AbcDiff::compute(&a, &b);
        assert!(!diff.body_replacements.is_empty());
        assert!(diff.body_insertions.is_empty());
        assert!(diff.body_removals.is_empty());
    }

    #[test]
    fn detects_body_insertion() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let diff = AbcDiff::compute(&a, &b);
        assert!(!diff.body_insertions.is_empty());
    }

    #[test]
    fn detects_body_removal() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D |\n").unwrap();
        let diff = AbcDiff::compute(&a, &b);
        assert!(!diff.body_removals.is_empty());
    }

    #[test]
    fn patch_reproduces_target_tempo() {
        let a = parse("X:1\nM:4/4\nL:1/4\nQ:120\nK:C\nC |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nQ:180\nK:C\nC |\n").unwrap();
        let diff = a.diff(&b);
        let patched = a.patch(&diff);
        assert_eq!(patched.header.tempo_bpm, b.header.tempo_bpm);
    }

    #[test]
    fn patch_reproduces_target_key() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:D\nC |\n").unwrap();
        let diff = a.diff(&b);
        let patched = a.patch(&diff);
        assert_eq!(patched.header.key.sharps, b.header.key.sharps);
    }

    #[test]
    fn patch_reproduces_body_after_replacement() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E G |\n").unwrap();
        let diff = a.diff(&b);
        let patched = a.patch(&diff);
        // Extract note pitches from both for comparison.
        let expected: Vec<u8> = b
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        let actual: Vec<u8> = patched
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn patch_extends_body_via_insertions() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let diff = a.diff(&b);
        let patched = a.patch(&diff);
        assert_eq!(patched.body.len(), b.body.len());
    }

    #[test]
    fn patch_shrinks_body_via_removals() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D |\n").unwrap();
        let diff = a.diff(&b);
        let patched = a.patch(&diff);
        assert_eq!(patched.body.len(), b.body.len());
    }

    #[test]
    fn empty_diff_produces_clone() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let diff = AbcDiff::default();
        let cloned = diff.apply(&a);
        assert_eq!(a.body.len(), cloned.body.len());
        assert_eq!(a.header.tempo_bpm, cloned.header.tempo_bpm);
    }

    #[test]
    fn diff_is_empty_reports_correctly() {
        assert!(AbcDiff::default().is_empty());
        let d = AbcDiff {
            tempo_change: Some(120),
            ..AbcDiff::default()
        };
        assert!(!d.is_empty());
    }

    #[test]
    fn round_trip_through_diff_preserves_extra_voices() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nV:1\nC D |\nV:2\nE F |\n";
        let a = parse(src).unwrap();
        let b = parse(src).unwrap();
        let diff = a.diff(&b);
        let patched = a.patch(&diff);
        assert_eq!(patched.extra_voices.len(), a.extra_voices.len());
    }

    // ---------- Phase 3.2 LCS-based diff tests ----------

    #[test]
    fn lcs_empty_diff_when_tunes_are_equal() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let diff = AbcDiff::compute_lcs(&a, &b);
        assert!(diff.is_empty());
    }

    #[test]
    fn lcs_diff_never_emits_replacements() {
        // LCS represents in-place changes as delete+insert, never replacement.
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E G |\n").unwrap();
        let diff = a.diff_lcs(&b);
        assert!(diff.body_replacements.is_empty());
        // Instead we expect one removal (F) and one insertion (G).
        assert_eq!(diff.body_removals.len(), 1);
        assert_eq!(diff.body_insertions.len(), 1);
    }

    #[test]
    fn lcs_detects_prepend_as_single_insertion() {
        // Prepending 'A' to `C D E F` should look like ONE insertion at
        // position 0, not four replacements + one trailing insertion.
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nA C D E F |\n").unwrap();
        let lcs_diff = a.diff_lcs(&b);
        let positional_diff = a.diff(&b);
        // LCS: 1 insertion + 0 removals.
        assert_eq!(lcs_diff.body_insertions.len(), 1);
        assert_eq!(lcs_diff.body_removals.len(), 0);
        // Positional: multiple replacements + 1 insertion at the end.
        assert!(
            positional_diff.body_replacements.len() >= 4,
            "positional diff should misdetect prepend as many replacements"
        );
    }

    #[test]
    fn lcs_detects_splice_in_middle_as_pair_of_edits() {
        // `C D E F` → `C D X E F` — splice X between D and E.
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D X E F |\n").unwrap();
        let diff = a.diff_lcs(&b);
        // Expect one insertion for X, no removals.
        assert_eq!(diff.body_removals.len(), 0);
        assert_eq!(diff.body_insertions.len(), 1);
    }

    #[test]
    fn lcs_detects_deletion_in_middle() {
        // `C D E F G` → `C D F G` — remove E from the middle.
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F G |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D F G |\n").unwrap();
        let diff = a.diff_lcs(&b);
        assert_eq!(diff.body_removals.len(), 1);
        assert_eq!(diff.body_insertions.len(), 0);
    }

    #[test]
    fn lcs_patch_reproduces_target_after_prepend() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nA C D E F |\n").unwrap();
        let diff = a.diff_lcs(&b);
        let patched = a.patch(&diff);
        let expected: Vec<u8> = b
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        let actual: Vec<u8> = patched
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn lcs_patch_reproduces_target_after_splice_in_middle() {
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D X E F |\n").unwrap();
        let diff = a.diff_lcs(&b);
        let patched = a.patch(&diff);
        let actual: Vec<u8> = patched
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        let expected: Vec<u8> = b
            .body
            .iter()
            .filter_map(|e| match e {
                AbcElement::Note { midi, .. } => Some(*midi),
                _ => None,
            })
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn lcs_full_replacement_is_all_removes_plus_all_inserts() {
        // Bodies share nothing → every original element removed, every revised
        // element inserted.
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nG A B |\n").unwrap();
        let diff = a.diff_lcs(&b);
        // 3 note removals + 3 note insertions (barlines share, one each).
        // Actual counts depend on how the parser tags Barline elements — count
        // only note edits to be robust.
        let removed_notes = diff
            .body_removals
            .iter()
            .filter(|&&idx| matches!(a.body.get(idx), Some(AbcElement::Note { .. })))
            .count();
        let inserted_notes = diff
            .body_insertions
            .iter()
            .filter(|(_, e)| matches!(e, AbcElement::Note { .. }))
            .count();
        assert_eq!(removed_notes, 3);
        assert_eq!(inserted_notes, 3);
    }

    #[test]
    fn lcs_indices_are_ascending_and_canonical() {
        // The public API guarantees ascending indices — LLM consumers can
        // stream the edit script sequentially. (Uses `A`/`B` as the "new"
        // notes because `Y` is not a valid ABC pitch letter.)
        let a = parse("X:1\nM:4/4\nL:1/4\nK:C\nC D E F G |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nK:C\nC A D E B G |\n").unwrap();
        let diff = a.diff_lcs(&b);
        for w in diff.body_removals.windows(2) {
            assert!(w[0] < w[1], "removals must be ascending");
        }
        for w in diff.body_insertions.windows(2) {
            assert!(w[0].0 < w[1].0, "insertions must be ascending");
        }
    }

    #[test]
    fn lcs_handles_empty_original() {
        let empty_src = "X:1\nM:4/4\nL:1/4\nK:C\nC |\n";
        let a = parse(empty_src).unwrap();
        let mut empty = a.clone();
        empty.body.clear();
        let diff = AbcDiff::compute_lcs(&empty, &a);
        // Every element in `a` is a fresh insertion.
        assert_eq!(diff.body_insertions.len(), a.body.len());
        assert!(diff.body_removals.is_empty());
    }

    #[test]
    fn lcs_handles_empty_revised() {
        let src = "X:1\nM:4/4\nL:1/4\nK:C\nC |\n";
        let a = parse(src).unwrap();
        let mut empty = a.clone();
        empty.body.clear();
        let diff = AbcDiff::compute_lcs(&a, &empty);
        // Every element removed.
        assert_eq!(diff.body_removals.len(), a.body.len());
        assert!(diff.body_insertions.is_empty());
    }

    #[test]
    fn lcs_and_positional_agree_on_header_changes() {
        let a = parse("X:1\nM:4/4\nL:1/4\nQ:120\nK:C\nC |\n").unwrap();
        let b = parse("X:1\nM:4/4\nL:1/4\nQ:180\nK:G\nC |\n").unwrap();
        let lcs = AbcDiff::compute_lcs(&a, &b);
        let pos = AbcDiff::compute(&a, &b);
        assert_eq!(lcs.tempo_change, pos.tempo_change);
        assert_eq!(
            lcs.key_change.map(|k| k.sharps),
            pos.key_change.map(|k| k.sharps)
        );
    }
}
