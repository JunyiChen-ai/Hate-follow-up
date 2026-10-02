# M1 Grounder independent code review

Date: 2026-10-02. Reviewer: independent agent instance
`/root/m1_grounder_code_review`, same inherited model as the implementation agent.
Scope: rule 6 of `RESEARCH_ITERATION_RULES.md`; implementation correctness only.
Reviewed `experiments/20261002_m1_grounder/grounder.py`, `measure.py`, the frozen
Judge/input helpers, and relevant installed Qwen3-VL attention/cache code. No GPU
was launched by the reviewer; no test annotations or performance metrics were read.

**Decision: PASS. No unresolved bug blocking the complete paired experiment.**
This is permission to measure this candidate, not evidence of a performance gain
or a validated explanation for one.

## Findings and checks

1. **The intervention reaches attention and changes the reader scores.** The
   hooks replace the `Qwen3VLTextAttention` mask immediately before its forward
   call. The SDPA implementation consumes this mask. `late`, `shifted`, and
   `verdict_only` restrict the last `ceil(L/4)` layers; `early` restricts exactly
   the first `ceil(L/4)`; `all_local` restricts all layers; `base` none. On the
   36-layer deployed model these are 9 layers, beginning at layer 27 for `late`.
   Independently tested the actual Qwen3 text-layer implementation on a small
   8-layer CPU model and captured the masks after the intervention hooks. The
   expected layer selections match exactly. All 136 branch reads in the four
   GPU smoke videos differ between `base` and each intervention arm, so no arm is
   accidentally bypassed. Magnitudes are plumbing evidence, not quality evidence.

2. **Causal and numerical paths match across arms.** Every arm receives the same
   four-dimensional additive causal-mask representation in every query layer.
   All cached positions are earlier than the query; current query positions see
   only themselves and earlier query tokens. Future query tokens are forbidden
   in every inspected layer. Restricted masks change only the cached-key access
   pattern. `verdict_only` leaves prefix media accessible and removes the cached
   global question/answer extension. The other restricted arms keep scaffolding
   plus their selected prefix media and remove the same global extension.

3. **Prefix and temporal alignment are correct for the reviewed inputs.** The
   production code reconstructs the processor's image-token expansion and asserts
   exact equality with encoded input IDs before using tokenizer character
   offsets. Frame spans include timestamps and vision delimiters; image-token
   counts are checked per frame. Speech uses the shared proportional word slicing
   rule. Independently encoded the first two videos of each corpus on CPU using
   the actual cached images and processor: prefix lengths 2343, 2748, 3022, 3083,
   matching the deployed smoke records. A separate check over all 333 manifest
   records confirms the reference proportional word slice agrees with
   `window_text` for every window. Zero-frame windows are retained without invented
   frames. Four-fps curves in all 24 smoke records exactly match the intended
   8-second window mapping and `ceil(duration * 4)` lengths.

4. **Branch independence and baseline parity hold.** The cache is cropped back
   to the prefix-plus-global-turn length after each branch. On the independent
   CPU model, crop versus deep-copy scores, reversed arm order, retained prefix
   key/value tensors, and hooked `base` versus ordinary attention are all exactly
   equal. In the actual lab2 GPU smoke, every crop/copy check and every hooked
   base/ordinary check is also exactly zero across all four videos. The global
   logit and stance are exactly shared across the six arms. The four GPU parity
   checks sample one branch per video, as declared; they are not exhaustive
   equivalence tests over all future inputs.

5. **The shifted control preserves the declared token budget.** Visual and
   speech positions are rotated within their respective token-position lists;
   per-modality cardinality and total kept-media cardinality are preserved.
   Scaffolding and causal query tokens are unchanged. True/shifted overlap is
   logged, including cases with complete overlap because all available tokens
   from a modality belong to one window. The control may retain partial image
   spans or wordpieces and preserves the local speech copied into the question.
   These are declared properties: interpret it as a wrong-prefix-access control,
   not complete removal or semantic replacement of local information.

6. **No labels enter measurement; cached inputs are matched.** The reader loads
   the fixed manifest, frame paths, repaired ASR segments, and frozen model. It
   does not call the GT-reading helper also present in `src/video_inputs.py`,
   read any annotation array, fit to labels, or use prior logits as model inputs.
   All 215 HateMM and 118 HateClipSeg manifest records have cached frames. Six
   arms are recomputed from the same live per-video prefix, so there is no input
   version mixing among arms. The existing `r6_bma` CLI reads the unchanged
   `z_video`, `z_visual`, `z_speech`, start/end schema and calls the canonical
   evaluator; the measurement file itself intentionally does not evaluate.
   Full-run coverage and the actual downstream invocation still need ordinary
   completion checks before reporting results.

## Corrected issue

The initial `calls: 2` output field inherited from the old reader undercounted
actual forward calls. The implementation agent corrected it to
`3 + n_branches`, with a breakdown for prefix encoding, global query, answer
extension, and local branches. Confirmed in the final reviewed source. The smoke
records retain the old metadata and must not be used to claim a two-call
deployment cost. Experimental arms share the first three calls; deployed cost
must count them for each method. Verification calls are additional.

## Evidence and limits

- `runs/20261002_m1_grounder/independent_review/check_mask.py` and
  `check_mask.json`: executable independent CPU checks; Torch 2.7.1,
  Transformers 4.57.6, no GPU.
- `runs/20261002_m1_grounder/independent_review/token_mapping.json`: actual
  processor/image expansion checks for the four smoke videos.
- `runs/20261002_m1_grounder/independent_review/smoke_integrity.json`: global
  parity, 4-fps mapping, effective-score changes, and all-record ASR slice check.
- `runs/20261002_m1_grounder/r1_smoke/checks.jsonl` and `config.json`: inspected
  runtime evidence supplied by the implementation agent, host `sc474399`, Torch
  2.11.0, Transformers 5.15.1. This provides deployment-version confirmation in
  addition to the reviewer's different-version CPU checks.

No claim of complete causal isolation is justified: cached prefix tokens,
allowed scaffolding, and early-layer query states can already contain global
context. The README correctly limits the mechanism to direct attention access.
No implementation change is required for that declared mechanism. The full
six-arm results and the predeclared controls must determine whether it improves
localization and whether any mechanism explanation is supported.

## Analysis-path supplement, same review

Reviewed the subsequently added
`experiments/20261002_m1_grounder/analyze.py` while full GPU measurement was still
running, without reading its performance results. **PASS.**

- `prepare` requires all 333 manifest videos, exact paired keys, identical global
  logits/window grids, four-fps lengths, and finite raw curves. The final reviewed
  `report` also requires every decoded arm to retain exactly the raw keys, global
  logits, rate, and lengths, with finite curves. The latter assertions were added
  following this review. Four synthetic corruptions—missing decoded video,
  altered global logit, shortened curve, and NaN—are each rejected.
- The diagnostic shift is one label-free constant per video, added to every
  available modality read and to the raw max curve, with the original global
  logit retained. Independent checks on all four smoke videos confirm that full
  curves and every individual modality preserve all ranks and ties. This control
  is correctly identified as a reconstruction using two experimental caches,
  not a deployed method or an ensemble candidate.
- Raw metrics use `src.eval.evaluate_four_datasets`. Decoder calls exactly match
  the current r6 configuration: `--noleak --transform nscore --key calib
  --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`.
  The existing r6 CLI produces final metrics through the same evaluator. Fit
  parameters are refitted without labels for each arm; the output explicitly
  states this distinction from freezing parameters.
- Paired intervals bootstrap per-video differences, with fixed seed 0 and
  10,000 resamples. Window diagnostics first average each stratum within each
  video and bootstrap those video means. They do not treat correlated windows
  as independent samples. Correct-Yes/wrong-No comparisons are restricted to
  mixed-label videos, where a video does contain positive frames. GT loads and
  window-group thresholds occur only in post-inference evaluation/reporting.
- No-drop and main improvement gates match the declared `.005` pooled, `.01`
  within, and `.01` within improvement on both corpora. Component gates require
  a common main metric to improve by `.01` on both corpora. The shifted-prefix
  and shift-only checks are at least as strict as the README requirements. The
  raw-ordering gate checks only the sign of paired mean improvement; its size
  and confidence interval must still be reported, and this boolean alone is not
  evidence of a statistically reliable raw-reader improvement.
- The initial analysis omitted the promised with/without sampled-frame window
  breakdown. The implementation agent added both strata for each window group,
  based on `kept_visual_tokens > 0`; confirmed in the final reviewed code and a
  structural execution test. No scoring logic was changed.

Structural test evidence is under
`runs/20261002_m1_grounder/independent_review/analysis_fixture/REVIEW_ONLY.json`.
The fixture uses the four smoke inputs, **synthetic alternating GT and dummy
metrics**, intercepts all subprocess calls, and never reads real GT. Its generated
tables are test artifacts and must never be cited as experimental results. The
prepare/report path, 14 assembled canonical CLI calls, deterministic bootstrap,
rank preservation, frame strata, and failure assertions all passed. No new GPU
call or additional performance-based decision was made in this review.
