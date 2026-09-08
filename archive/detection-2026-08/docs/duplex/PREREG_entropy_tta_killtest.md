# Pre-registration — Label-free entropy adaptation kill test (HateMM test)

**Frozen:** 2026-08-09, before any adaptation step runs.
**Compute:** one RTX 5090; norm-parameter adaptation of Qwen3-VL-8B on 215
unlabeled videos, then one rescoring pass. Projected well under 8 GPU-hours.
**Status:** kill test opening the label-free adaptation family — the first
training-based family in this project. Every prior family (evidence-side,
prompt-side, scalar and hidden-state readout, threshold geometry, dual-call
factorization) has a preregistered death; none of those deaths covers
training, and the project charter explicitly permits adaptation without
human hate labels, including on the target corpus's own unlabeled data.

## Phenomenon

On HateMM the judge's ranking is strong (AUC 0.923) but the label-free
operating point fails (valley macro-F1 0.6562 vs oracle 0.888): the z
distribution's middle is congested — violent-but-not-hateful normals and
mid-confidence hatefuls overlap — so the KDE valley lands off the class
boundary. Separately, the readout-bottleneck probe showed the model's
representation carries distinctions its frozen scalar readout mis-spends.

## Mechanism

Entropy minimization on the model's own binary {Yes, No} posterior, over the
corpus's unlabeled videos, updates only normalization parameters
(TENT-style). It pushes each video's posterior away from 0.5 along the
model's own decision boundary, draining the congested middle of the z
distribution and restoring a bimodal gap. The label-free valley then has a
real valley to find. Deployment stays a single call per video; adaptation is
a one-time offline pass with no labels of any kind.

Falsifiable asymmetry, stated in advance: this sharpens the MODEL'S OWN
construct boundary (protected-group hate). It should therefore help where
the annotation boundary coincides with that construct (HateMM) and should
NOT rescue construct-mismatched collapses (MHClip union) — per the
ill-posedness result, no label-free procedure can. HateMM is thus the
decisive corpus; MHClip is out of scope here.

## Frozen protocol

- Corpus: HateMM test, all 215 videos, existing frozen inputs (uniform-16
  frames, restored fresh transcripts, identical prompt).
- Trainable parameters: all normalization-layer weights (RMSNorm gains)
  of the language model and vision tower; everything else frozen. No LoRA,
  no head retraining.
- Objective: Shannon entropy of the renormalized two-way {Yes, No}
  distribution at the answer position, minimized. No other loss terms.
- Optimizer AdamW, lr 1e-5, weight decay 0, batch size 1, exactly 1 epoch
  over the corpus in a fixed shuffled order (seed 20260808), gradient
  checkpointing as needed. No early stopping, no lr search: the primary
  arm is this configuration alone. (A descriptive sensitivity arm at lr
  1e-6 may be reported but carries no confirmatory weight.)
- After adaptation: rescore all 215 videos with the adapted model in a
  single pass each; readout, KDE-valley convention, and macro-F1 machinery
  identical to the existing test_c2 evaluation.
- Placebo arm (matched compute): identical training run except each
  optimization step pairs one video's frames with a different video's
  transcript (fixed derangement, seed 20260808), destroying audio-visual
  evidence coherence while keeping data statistics and update count. Then
  rescore with coherent inputs as above.

## Frozen decision rule

The family **SURVIVES** only if all three clauses hold:

1. **Operating point recovers:** adapted valley macro-F1 ≥ 0.75 (baseline
   0.6562; this closes ≥ 40% of the 0.232 oracle gap).
2. **Ranking preserved:** adapted AUC ≥ 0.90 (baseline 0.9232). Sharpening
   that destroys ranking is mode collapse, not mechanism.
3. **Evidence-driven, not update-driven:** the placebo arm's valley
   macro-F1 gain over baseline is < 50% of the real arm's gain.

Reported descriptively either way: adapted z histogram summary vs baseline
(bimodality/trough depth), valley location vs labeled oracle threshold,
per-stratum movement of the 70 baseline false positives and the baseline
false negatives, entropy trajectory during training.

## Interpretation boundaries

- Pass: licenses the full method preregistration (second occupied-boundary
  corpus HateClipSeg-strict, ImpliHateVid regression check, deployment
  story). Not itself a cross-corpus claim.
- Clause 1 fails: entropy sharpening does not move the operating point;
  the adaptation family loses its cheapest, most favorable case — retire
  the entropy objective on this judge.
- Clause 2 fails: collapse; retire and report the failure mode.
- Clause 3 fails: gains come from update dynamics, not evidence — the
  mechanism story is false even if numbers improve; do not promote.
- No human labels, no benchmark labels, no external data touch any
  training step. Labels appear only in post-hoc evaluation, as everywhere
  else in this project.
