# M1 Selector independent code review

Date: 2026-10-02. Reviewer: independent agent instance
`/root/m1_grounder_code_review`, same inherited model as the main session.
Scope: rule 6, implementation correctness of
`experiments/20261002_m1_selector/{selector,measure}.py`. Also inspected the
shared Judge/token-region helpers and the installed Qwen3-VL attention/SDPA
path. No method code changed; no GT, empirical performance, GPU job, or new
sub-agent was used by the reviewer.

**Decision: PASS. Static review, independent CPU checks, and deployment-version
smoke confirmation passed. No unresolved scoring bug found.** The complete
paired experiment may proceed. This review does not predict effectiveness.

## Selection and attention computation

- Native Qwen3-VL calls the attention registry after Q/K normalization, rotary
  position encoding, and KV-cache update. The wrapper therefore receives the
  intended Q/K tensors. It delegates all inactive calls and all non-target
  modules to the original SDPA function. Prefix encoding, the global question,
  and answer extension run while routing is inactive. Closing the router
  restores the original registry function.
- GQA grouping is correct: contiguous query heads belonging to each KV head are
  reshaped as `(batch, KV heads, groups, head dim)`. Independent synthetic checks
  compared the result with explicitly repeated KV heads and ordinary pairwise
  dot products. The head decisions agree for both default and non-default
  native attention scaling.
- The criterion is the declared **mean attention probability per token**, not
  mean raw QK affinity. For local set L and remote set R, it compares
  `logsumexp(s_L) - log |L|` with `logsumexp(s_R) - log |R|`. The shared attention
  softmax denominator cancels. Independently compared this decision against
  the average probabilities from the complete last-query-token softmax,
  including scaffolding, global context, and query tokens. Decisions agree.
  Empty local or remote sets disable selection and the all-head restriction.
- Only a single query row is materialized for selection. The ordinary native
  SDPA function still computes the complete query forward once. Every arm uses
  the same head-specific additive causal-mask representation. Selected heads
  block only remote prefix media. Local media, non-media scaffolding, global
  Q/A, and causal query values remain accessible. All future query-value edges
  remain blocked in the independent mask checks.
- Routing uses the complete observed query to select a mask for all its
  positions. The code and README explicitly acknowledge this; earlier query
  states are not claimed to be strictly autoregressively independent of later
  query text through routing. Later-layer selection is computed along the
  current intervention trajectory, not along a separately run baseline.

## Replay controls and branch isolation

- In a paired call, `select` stores its layer-by-head Boolean trajectory before
  replay controls run. `permuted_heads` rotates that exact trajectory by half
  the query-head count; `shifted_support` keeps the trajectory and substitutes
  the declared token-count-matched wrong support. Neither replay control
  recomputes its selection statistic. Independent checks confirm identities,
  head counts, blocked positions, and the unchanged causal/global masks.
- Saved replay inputs are checked against freshly computed global logits,
  prefix token count, and window intervals. Their per-layer selected indices
  reconstruct the intended Boolean head matrix. Main argument validation
  requires an earlier `select` arm or an explicit replay run for these controls.
  Using the same completed select run, model, and input caches remains required;
  the explicit replay path is recorded in config.
- Every branch crops its dynamic cache to the retained prefix/global length.
  A real small Qwen3 text model on CPU gave exact equality for crop versus
  deep-copy outputs and trajectories, reversed arm order, and every retained
  key/value tensor. Base through the wrapper equals ordinary SDPA exactly.
  Selected routing changes the toy output, demonstrating that it reaches the
  scoring path rather than merely producing a diagnostic trace.

## Inputs, output, and labels

- `src/window_token_regions.py` is the previously reviewed exact token-span
  implementation, now shared without cross-experiment imports. It verifies
  expanded input-token equality and image-token counts, uses the existing
  proportional transcript-word slicing, and records empty local visual support.
  No input resolution, timestamps, prompts, or model weights are changed here.
- All arms share freshly computed prefix/global context and identical window
  questions. Visual and speech branches remain independent; absent speech
  windows do not acquire invented reads. Head-index arrays are explicitly
  excluded from the max-logit operation used to construct the raw curve.
- Output uses the existing `z_video`, `z_visual`, `z_speech`, and window start/end
  fields. Curves use `ceil(duration * 4)` positions and the established
  center-time-to-8-second-window mapping. The reader iterates the same fixed
  215/118 manifest rows and fails on missing cached frames instead of silently
  removing videos. Resume requires matched arm completion sets.
- No annotation, GT array, GT-dependent subset selector, fit, or threshold is
  read by the measurement or routing code. The fixed seed and equality
  comparison introduce no label-based choice. Evaluation is intentionally
  outside these two files and must use the canonical evaluator and current r6
  CLI when added. Complete output coverage will need verification on completion.
- Model-forward metadata correctly counts `3 + observed modality/window
  branches`. QK-row selection and head-mask overhead must be measured even
  though they add no forward call. Prefix sharing in the experiment must not
  hide per-method deployment cost.

## Independent evidence and limits

Executable checks and results:
`runs/20261002_m1_selector/independent_review/check_selector.py` and
`check_selector.json`. Environment: CPU only, Torch 2.7.1, Transformers 4.57.6;
8 randomly initialized Qwen3 text layers, with four query heads and two KV heads.
No pretrained weights or empirical labels were needed. Selected head counts
vary across layers, and no-head as well as nonempty-head cases occur.

The deployment environment is lab2 HateVLM, Torch 2.11.0 / Transformers 5.15.1.
The implementation agent inspected that version's actual attention forward
signature and ran the GPU smoke; the reviewer independently parsed its returned
records at `runs/20261002_m1_selector/r1_smoke/`. Four videos and all five arms
passed with every crop/copy difference zero and ordinary-versus-wrapper base
difference zero over every observed branch. Global logits are identical across
arms, all 4-fps curves match their window scores exactly, and call counts are
correct. All 136 replayed branch trajectories match the original selection or
its exact half-head rotation, as appropriate. These checks are recorded in
`runs/20261002_m1_selector/independent_review/smoke_integrity.json`.

The smoke's summed select branch time is 1.1513 times ordinary branch time;
charging the shared prefix once gives a 1.1008 estimated standalone ratio.
These are four-video synchronized wall-time measurements, not full-corpus
cost claims or evidence of a performance gain.

Shared contextualized keys/values, global Q/A, unselected heads, and the query's
own copied speech remain available. Thus this implements selective direct
attention access, not complete information removal. Replayed wrong support may
overlap true support; rotated heads may overlap selected heads. The declared
overlap and all-/no-head frequency diagnostics are necessary to interpret those
controls, not reasons to change the already declared mechanism.

## Main-pair analysis supplement, same review

The newly added `analyze.py` and `launch/run_analysis.sh` also pass this review.
Only the initial base/select comparison is implemented at this stage; the
summary correctly fixes `mechanism_supported` to false until the declared
controls provide evidence.

- Preparation checks the complete fixed manifest, paired keys, same global
  judgement, finite/equal-length 4-fps curves, matching window boundaries, and
  speech-branch availability. It compares against the current `base_gridA`
  reader and records each video's maximum baseline-window discrepancy.
- Raw metrics call the canonical evaluator. Decoding uses the current r6 flags
  `--noleak --transform nscore --key calib --duration bma --bma-prior length
  --min-windows 2 --bma-grid 6 --arm m2`, and that CLI calls the same evaluator.
  Report assertions require decoded keys, globals, rates, lengths, and finite
  scores to match raw inputs. Parameters are label-free refits for each arm,
  explicitly distinguished from frozen decoder parameters.
- The continuation gate tests any final main-metric gain of at least `.01`.
  The performance gate requires within-video improvement of at least `.01`
  on both corpora and no drop beyond `.005` pooled / `.01` within. It checks
  both the paired base and the historical current r6, making promotion stricter
  than checking only the paired baseline. No mechanism promotion is inferred
  from this initial pair alone.
- Paired confidence intervals resample video-level differences with fixed seed
  0 and 10,000 resamples. Correct-Yes/wrong-No breakdowns are computed only on
  mixed-label videos. GT is accessed only after measurement, in the evaluator
  or report, and cannot affect head selection or fitting.
- Trajectory checks require a trace for each observed modality read, the right
  number of layers, unique head indices, and indices within bounds. Head
  frequency and all-/no-head statistics count layer/branch decisions; overlap
  compares the actual set with the same half-head rotation used by the control.
  This is an overlap fraction of selected heads, not a Jaccard index or a
  semantic correctness measure.
- Cost totals retain prefix and branch time separately and include actual
  model-forward counts. The two CPU arms have separate raw metric files,
  decoded output directories, and logs. Preparation finishes first; the launch
  script waits for both exit statuses and reports only after both succeed.

No new method-performance output or true GT was read for this supplement.
Full-pair coverage, baseline discrepancy, cost, and the conditional mechanism
controls remain the required empirical follow-through after this code PASS.
