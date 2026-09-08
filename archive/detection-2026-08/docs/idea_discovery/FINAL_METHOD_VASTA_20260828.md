# VASTA: Visual-Authority Selective Temporal Arbitration

## Status

VASTA is the strongest current development candidate for label-free hateful
video localization. It is a training-free, multimodal authority-routing method,
not a conventional feature-fusion method. The current 611-video evaluation is
exploratory/test-developed and must not be described as an untouched holdout.

## Core claim

Different modalities should receive state-dependent decision authority. Frozen
visual experts own temporal extent and retain final authority when they agree.
Transcript evidence may intervene only when the visual experts disagree, and
only when its negative evidence is stronger than the language model's explicit
content-free null response.

## Modules

Data decoding, frame sampling, ASR loading, timestamp alignment, proposal
export, and curve rasterization are preprocessing and are not counted as
modules.

### Module 1: Proposal-set Temporal Support Recovery

Given the top K=8 frozen A10/Vid-Group proposals, form the minimum temporal
envelope covering their support. This repairs the systematic under-coverage of
the rank-1 proposal. The mechanism is adaptive extent calibration; it must not
be sold as a multi-component topology decoder because 609/611 envelopes are a
single connected component and same-length dilation explains most of the gain.

### Module 2: Visual Authority State Estimation

Two heterogeneous frozen visual experts, A08/VideoTGB and A12/TimeLens, emit
event-existence decisions e8 and e12. They induce three authority states:

- e8=e12=1: visual-positive consensus; keep the Module-1 interval.
- e8=e12=0: visual-negative consensus; emit EMPTY.
- e8!=e12: visual epistemic disagreement; defer only this existence decision
  to Module 3. Temporal boundaries remain visual.

This is asymmetric authority routing, not score averaging or symmetric fusion.

### Module 3: Deployment-time Null-Referenced Selective Arbitration

Query the frozen text expert once with the exact task prompt and an explicit
empty transcript. Its Yes-minus-No next-token logit is the null anchor z0. For
each real transcript, compute z. In the visual-disagreement state only, veto the
visual-positive decision iff z<z0. The current deterministic batch-1 run gives
z0=-12.0. The anchor uses one content-free query, no labels, no corpus
statistics, and no naturally empty videos.

The final existence rule is

    keep = e8                         if e8 == e12
           1[z(transcript) >= z0]     otherwise.

Text never moves boundaries and never overrides visual consensus.

## Current performance (611-video exploratory cohort, 4 FPS)

| Dataset | pooled ROC | PR | within-video ROC | F1@.3 | F1@.5 | F1@.7 |
|---|---:|---:|---:|---:|---:|---:|
| HateMM | .569 | .300 | .602 | .281 | .235 | .144 |
| HateClipSeg | .550 | .509 | .557 | .214 | .109 | .062 |
| MHC | .616 | .302 | .665 | .413 | .381 | .222 |
| MHC-zh | .611 | .238 | .567 | .432 | .400 | .256 |
| Macro | .586 | .337 | .598 | .335 | .281 | .171 |

The strongest reproduced interval baseline, A10, has macro F1@.5=.163. The
visual-OR predecessor has .273. VASTA therefore improves A10 by .118 absolute
and the matched visual-OR system by .008 absolute.

## Mechanism evidence

- 279/611 videos are visual disagreements.
- Null arbitration triggers 48 vetoes: 43 correct FP removals and 5 wrong
  removals at tIoU=.5.
- The two retrospective hash halves have net corrections +18 and +20.
- Against visual OR, paired dataset-balanced bootstrap deltas are:
  - F1@.3 +.0100, 95% CI [.0019,.0187], p=.0064;
  - F1@.5 +.0080, 95% CI [.0002,.0162], p=.0228;
  - F1@.7 +.0035, 95% CI [-.0031,.0097], p=.1436.
- Matched-count controls at F1@.5: shortest transcript .267, hash random .263,
  and empty-only .176, versus VASTA .281.
- A generic hate prompt scores .2809 versus the stance prompt .2814. Therefore
  the evidence supports semantic negative-evidence arbitration, not a strong
  stance/use-mention claim.
- Corrected arbitration improves all three tested extent sources at F1@.5:
  A10 .1994->.2035, global 1.49x dilation .2655->.2730, and adaptive top-8
  envelope .2734->.2814.
- A08 cached answers are parseable in 611/611 cases and decisions match their
  anchored first Yes/No token in 611/611 cases. Twenty-five contradictory tails
  occur only after the first EOS and do not alter decisions.

## Novelty assessment and safe claims

Independent task-specific novelty score: 6.3/10 (general-method novelty about
4.5/10). The defensible paradigm claim is constrained multimodal authority for
label-free temporal localization: visual evidence owns localization; text has
a null-calibrated, disagreement-only veto.

Safe current claims:

- best interval F1@.5 among the currently reproduced label-free baselines on
  this common 611-video four-dataset cohort;
- both temporal support recovery and selective arbitration contribute;
- the text gate is label-free and rejects simpler length/emptiness/random
  explanations.

Unsafe current claims:

- broad SOTA generalization on an untouched benchmark;
- proven stance, quotation, condemnation, or use/mention reasoning;
- proposal topology as the source of the extent gain;
- significance at F1@.7;
- superiority on every dataset and every metric.

## Remaining validation gate

The complete rule, prompts, K=8, explicit-null query, batch size 1, and extent
source are now frozen. A genuinely sealed external cohort is still required for
a strict SOTA/generalization claim because the current four test sets were seen
during method development. The internal 32-video pilot is not sealed and must
not be recycled as such.

## Authoritative artifacts

- Canonical predictions: `results/idea_discovery/melt/VASTA_confirmation_v3.jsonl`
- Metrics: `results/idea_discovery/melt/VASTA_confirmation_v3_metrics.json`
- Canonical text scores: `results/idea_discovery/melt/transcript_existence_qwen3vl8b_vasta_v3.jsonl`
- Routing audit: `results/idea_discovery/melt/VASTA_routing_audit_v2.json`
- Informativeness controls: `results/idea_discovery/melt/vasta_informativeness_controls_confirmation_v1_metrics.json`
- A08 parser audit: `results/idea_discovery/melt/videotgb_answer_parser_audit_v1.json`
- Independent story review: `docs/idea_discovery/VASTA_STORY_REVIEW_20260828.md`
