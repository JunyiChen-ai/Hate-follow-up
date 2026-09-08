# RCD-Witness: matched real-context deletion pilot

## Scope and wording

The method identifies evidence that is sufficient and necessary **for the
frozen MLLM under matched real-context deletion interventions**.  It does not
claim causal necessity in the world or causal identification of a human event.

## Modules (preprocessing is not a module)

1. **Four-Arm Matched Local Deletion Field.** For each candidate interval `I`,
   freeze the real outside neighborhood `C`, carrier geometry, focus timestamps,
   policy question, and context speech.  Query:
   - `zF`: target visual and target speech observed;
   - `z-V`: target visual slots replaced by real neighboring frames;
   - `z-T`: target speech explicitly removed while target frames remain;
   - `z0`: both target modalities removed/replaced.
   All arms use the same neutral slot labels and prompt template. Every frame is
   labeled only by its true sampling timestamp; substituted frames therefore
   remain visibly outside `I` without an answer-bearing "removed" cue. The
   target-speech field always exists and is the empty string in text-deletion
   arms; context speech is never relabeled as target speech.
2. **Sufficient-and-Necessary Witness Test.** An interval is a witness iff
   `zF > 0 and z0 <= 0`.  The witness score is `min(zF, -z0)`.  Modality deltas
   `dV=zF-z-V`, `dT=zF-z-T`, and interaction
   `s=zF-z-V-z-T+z0` are recorded as typed provenance, never fused into the gate.
3. **Fixed-Budget Recursive Minimal-Witness Search.** Four-arm score U8, select
   exactly four parents by a frozen label-free ordering, split each into four,
   and four-arm score the 16 children.  If any child is a witness, use witness
   children; otherwise retain a witness parent.  Adjacent outputs merge only at
   an exact shared boundary.  Total: 96 semantic branches/video.

Frame decoding/sampling, ASR extraction/alignment, time-grid construction, and
canvas rendering are preprocessing and are excluded from module count.

## Frozen selection rule

Lexicographic descending priority:

1. parent satisfies the witness test;
2. modality deletion arms disagree in sign;
3. smaller absolute witness margin (closer to a boundary);
4. larger factual margin;
5. earlier timestamp as deterministic tie break.

## Controls at the same 96-branch budget

- `rcd_witness`: proposed method;
- `rcd_no_necessity`: same tree, factual-positive leaves only;
- `rcd_hash_zoom`: hash-selected parents, same deletion field and extraction;
- `rcd_uniform24`: 24 uniform intervals, four deletion arms each;
- `factual_uniform96`: 96 uniform intervals, factual arm only.

Mechanism controls after the main gate passes: text-only, visual-only, temporal
frame shuffle, within-video ASR circular shift, left-only/right-only context
substitution, blank deletion, and generic-hate policy wording.

## Pilot staging

1. Eight-video label-blind mechanism sanity, two hash-frozen videos per dataset.
2. Open GT only after all eight videos finish atomically.
3. Kill immediately if joint-deletion sign flips occur in fewer than 10% of
   factual-positive evaluated windows, or the witness gate retains/kills nearly
   every factual-positive window.
4. If the mechanism is non-degenerate, run the disjoint balanced pilot63.

## Claim gates

- `rcd_witness` beats `rcd_no_necessity`, `rcd_uniform24`, and
  `factual_uniform96` on macro interval F1@0.5; Module 2 must independently
  improve F1@0.5 or F1@0.7.
- Positive F1@0.5 delta on at least three of four datasets.
- Joint beats text-only on at least two datasets; frame shuffle reduces visual
  contribution and localization performance.
- Left/right substitution does not reverse the main result.
- The full relation-aware policy outperforms a generic-hate question.
- Neither modality has non-positive contribution on more than 85% of positive
  witnesses if a balanced multimodal claim is made.

Passing these gates supports a provisional task-specific novelty score of
6.5--7/10.  Failure of the necessity ablation kills RCD-Witness rather than
triggering test-label tuning.
