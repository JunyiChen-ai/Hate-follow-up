# Pre-registration — Blind annotation-validity audit (HateClipSeg normals, MHClip-EN positives)

**Frozen:** 2026-08-08, before any blind coding output exists.
**Compute:** CPU only; no judge calls. Coding is done by Claude Opus agents —
a different model family from the Qwen3-VL judge under study.
**Status:** measurement study. It hardens or retracts the two central claims
of `RANKING_AUTOPSY_NOTE.md`, both of which were derived score-aware and are
therefore vulnerable to the objection that we relabeled whatever our model
got wrong.

## Claims under test

1. **HateClipSeg negative-class contamination.** The autopsy coded 17 of the
   50 clean-normal videos as containing explicit protected-group hostility
   or extremist-movement glorification (worth +0.147 AUC if true).
2. **MHClip-EN positive-class construct mismatch.** The autopsy coded 17 of
   25 low-scoring positives — and by extension an estimated majority of the
   union positive class — as containing no protected-group target at all.

## Blinding protocol (the load-bearing control)

- Coders are fresh agent instances that receive ONLY: the benchmark's own
  written label definitions, the video's fresh transcript, and 4 uniformly
  spaced frames. Nothing else.
- Coders never see: judge scores, prior taxonomy codes, the autopsy note,
  anything under `results/`, or this preregistration's motivation sections.
- The orchestrating agent must not open any score file until all coding
  outputs are collected.
- Videos are presented in random order (seed 20260808) with neutral
  identifiers carrying no stratum information.
- The EN batch mixes positives with randomly sampled normals so a coder
  cannot infer that every item is a shipped positive.

## Frozen samples

- HateClipSeg: ALL 50 clean-normal videos, plus 25 randomly sampled
  (seed 20260808) shipped-positive videos as blinding fillers.
- MHClip-EN: ALL 49 union positives, plus 40 randomly sampled Normals
  (seed 20260808) as fillers.
- Fillers are coded identically; their results are reported but no frozen
  claim depends on them.

## Frozen coding scheme

- HateClipSeg items, per the corpus README definitions: does the video
  contain (a) hateful content toward a protected group, (b) extremist or
  hate-movement glorification, (c) other offensive content (insulting /
  sexual / violent / harmful), or (d) none. Multiple allowed; "cannot
  determine" allowed and excluded from rates (count reported).
- MHClip-EN items, per the MHClip annotation definitions (as recorded in
  `DATASET_AND_POLICY_REFERENCE.md`): 3-class {Hateful, Offensive, Normal},
  plus one boolean: "is a protected group targeted (explicitly or
  implicitly)?" — with the same "cannot determine" rule.
- Each item is coded once. No adjudication rounds, no majority voting (the
  single-pass rule mirrors the judge-side single-call discipline and avoids
  ensemble-style laundering of the verdict).

## Model-independent evidence (no coding involved)

Enumerate all exact-duplicate fresh-transcript groups in HateClipSeg
(nonempty transcripts, byte equality) and report shipped-label agreement
within each group. This subset is entirely independent of both models.

## Frozen decision rule

- Claim 1 **CONFIRMED** iff the blind-coded rate of (a)-or-(b) among the 50
  HateClipSeg clean normals is ≥ 20% (exact binomial CI reported). Else
  RETRACTED, and the +0.147 corrected-AUC arithmetic is withdrawn.
- Claim 2 **CONFIRMED** iff the blind-coded rate of "no protected group
  targeted" among the 49 EN union positives is ≥ 40%. Else RETRACTED.
- Secondary, reported either way: judge AUC recomputed against blind-coded
  labels (both corpora, coded items only), unblinded only after all coding
  is complete.

## Interpretation boundaries

- Confirmation licenses the autopsy's reframing: the residual performance
  gap on these corpora is substantially a property of the benchmark labels,
  and the follow-up's evaluation must report label-corrected ranges
  alongside shipped-label numbers. It does not by itself add any method
  component.
- Retraction of either claim reverts that corpus to "unexplained judge
  limitation" and the closeout note must be amended to say so.
- Known residual weakness, stated now: coders are LLMs, not trained human
  annotators, and share no family with the judge but do share general
  pretraining biases. The blindness to scores and the model-independent
  duplicate audit are the controls; a human-subjects audit is out of scope.
