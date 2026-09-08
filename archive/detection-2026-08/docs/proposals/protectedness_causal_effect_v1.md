# Proposal v1 — Protectedness Causal Effect (PCE)

**Status:** **premise audit failed; do not run the proposed PCE pilot as written.**
Only 15/70 (21.4%) corrected-8B HateMM false positives were generic
hostility/violence without a protected target, below the preregistered 30%
threshold. Quoted/reported/countered hate (11/70) additionally shows that
protectedness alone misses stance. See
`docs/duplex/HATEMM_FP_PREMISE_AUDIT.md`. **Date:** 2026-08-08.

## One sentence

Measure the causal contribution of *protected target status* to a frozen
multimodal judge by comparing the factual video against a minimally edited
counterfactual in which protected-group mentions are replaced by a generic
person/group, then use `(factual hate logit, protectedness effect)` as a
label-free two-dimensional decision geometry.

## Four-part story

### Phenomenon

Hate policy is not generic offensiveness: hostility becomes hate when it is
directed at a protected target. HateMM's normal class is violence-saturated,
and the corrected judge ranks it well (AUC 0.9232) while its one-dimensional
score distribution places 70 normals in the high-score mode and bends the KDE
valley to macro-F1 0.6562. On ImpliHateVid, 79% of false positives carry
hate-adjacent surface cues. These errors are compatible with a judge that sees
hostility but does not distinguish whether protected status is causally
load-bearing.

### Mechanism

For video `v`, obtain the frozen factual score `z(v)`. Construct `do(P=0)` by
replacing transcript/OCR spans that denote a policy-protected group with a
grammatically matched generic referent (`people`, `a person`, `a group`) while
leaving predicates, negation, quotation, frames, timing and all other evidence
unchanged. Score the counterfactual with the same joint judge:

`Delta_P(v) = z(v) - z(do(P=0, v))`.

Generic violence should remain offensive under `do(P=0)` and have small
`Delta_P`; protected-target hate should lose the attribute that makes the same
hostility a hate-policy violation and have positive `Delta_P`. A two-component
model over `(z, Delta_P)`, fitted to the target corpus without labels, can
separate high-z generic violence from high-z protected-target hate without a
hand-tuned fusion weight.

The span detector must be a frozen general-purpose entity/semantic tagger
conditioned only on the platform policy's protected-attribute categories. It
may not use a hate lexicon, group-frequency statistics, dataset labels or a
task-specific external corpus.

### Predictions

On cue-matched TP vs FP cohorts from ImpliHateVid and HateMM:

1. `Delta_P` AUC is at least 0.65 on each arm.
2. `(z, Delta_P)` improves TP-vs-FP separation over factual `z` by at least
   0.05 AUC on the primary HateMM arm.
3. The gain is larger on HateMM violent-normal false positives than on ordinary
   true negatives.
4. Full-corpus label-free 2-D clustering reduces HateMM's operating-point gap
   without lowering ImpliHateVid macro-F1 by more than 0.01.

Disconfirmation: if `Delta_P` is reproduced by generic token removal, or true
hate and violent normal move equally, protectedness is not a causal coordinate
available to the judge and the proposal dies.

### Counterfactual ablations

- **P→generic (mechanism):** replace protected target with generic person/group.
- **P→P placebo:** replace it with a different protected target while preserving
  number and syntax. This should preserve policy status and produce a much
  smaller effect.
- **Random-NP placebo:** replace a non-protected noun phrase of matched token
  length with a generic referent. This measures deletion/fluency sensitivity.
- **Predicate placebo:** leave the target intact and replace a matched neutral
  predicate span. This bounds generic semantic-edit sensitivity.
- **Text-only diagnostic:** determines whether any effect is merely a language
  classifier artifact; it is not a candidate method arm.

## Cheapest decisive pilot

Use existing corrected-baseline diagnostic cohorts, labels only to name cells:

- HateMM test: all 70 FP + 70 cue-matched TP; primary arm because generic
  violence is the located confound.
- ImpliHateVid train: 70 FP + 70 cue-matched TP sampled with frozen seed;
  directional replication.

Freeze counterfactual texts before any new score is read. Manually audit only
the span detector's intervention validity on a label-blind random 20-video
sample. Score factual values from cached source runs; the pilot therefore needs
three diagnostic counterfactual calls per video, one per intervention. A
surviving deployed method uses factual + P→generic only: two MLLM calls.

Primary pilot pass requires all:

- HateMM `AUC(Delta_P; TP,FP) >= 0.65`;
- HateMM AUC of the frozen, label-free 2-D recipe exceeds factual-z AUC by
  >=0.05;
- `AUC(Delta_P) - AUC(Delta_randomNP) >= 0.05`;
- median `|Delta_PtoP|` is no more than half median `|Delta_P|` on TP;
- ImpliHateVid directions agree (positive deltas), with no separate tuning.

Before scoring, the exact 2-D estimator, component-to-label mapping, span
detector, substitutions, cohort seed, and bootstrap bars must be pre-registered.

## Novelty boundary

Counterfactual token fairness swaps one social-group token for another and
trains text classifiers to be invariant. PCE instead removes *protectedness*
while preserving hostility, uses the induced inference-time causal effect as a
detection coordinate, operates on a frozen multimodal video judge, and chooses
the decision geometry without hate labels. The relation must be cited rather
than claiming counterfactual identity editing itself as novel.

## Main risks

1. Protected spans are often visual or implicit, so transcript intervention has
   incomplete coverage.
2. Generic replacement can create unnatural language; the random-NP and P→P
   placebos are load-bearing.
3. The judge may respond to hostility independently of protected status, making
   `Delta_P` near zero even on true hate.
4. A two-dimensional mixture can become another selector artifact. Its recipe
   must be frozen on unlabeled geometry and tested unchanged across datasets.
