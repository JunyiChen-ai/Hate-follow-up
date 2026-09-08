# Readout-bottleneck kill test: result note

**Date:** 2026-08-09. **Verdict:** **DEAD** under the frozen rule.
**Compute:** CPU only, 5 seconds. No model call, no rescoring.
**Preregistration:** `docs/duplex/PREREG_readout_bottleneck_killtest.md`
**Analysis:** `scripts/duplex/readout_bottleneck_killtest.py`
**Machine-readable result:** `results/readout_bottleneck/results.json`

The information is there and it is not reachable without labels. A supervised
probe on layer-27 hidden states separates no-protected-target positives from
protected-target positives at leave-one-out AUC 0.837, well above the 0.75
floor. Not one of the 24 unsupervised axes beats the scalar readout it was meant
to widen: the best reaches 0.746 against z's 0.749, and the 0.799 floor is
missed by every candidate. Clause 1 passes, clauses 2 and 3 fail, so the
readout-bottleneck family does not proceed.

## What ran

The hidden states written by the two frozen Qwen3-VL-8B judge runs were read
from disk and nothing was recomputed. MHClip-EN contributes 161 files and
HateClipSeg 394, each a 37 × 4096 float16 array of final-token states, one row
per layer with row 0 the embedding output. Every file was checked for that exact
shape and for finite values, and every id matched its corpus's `scores.jsonl`
exactly. No HateClipSeg file differed structurally.

Layers 18, 27 and 36 were analysed, as frozen. Each layer's 4096 features were
standardized per dimension using the mean and standard deviation of that
corpus's own hidden states over all videos, with no labels involved; no
dimension had zero variance in either corpus. PCA was computed by singular value
decomposition on the standardized, re-centred matrix, taking the top 8
components, and each component's sign was flipped when its Spearman correlation
with that corpus's z was negative. The probe is L2-penalised logistic regression
at a penalty of lambda = 1.0, fixed in advance and never tuned, fitted by L-BFGS
with an unpenalised intercept; leave-one-out cross-validation refits it 49 times
and scores each held-out video with the model that never saw it. All 49 fits
converged.

The strata rebuilt to their expected sizes exactly: 34 no-protected-target
positives, 15 protected-target positives and 112 shipped Normals on MHClip-EN;
59 insulting-only videos and 50 clean normals on HateClipSeg. HateClipSeg's
`lexicons.json` was never opened. As an independent check that the strata are
the same ones the earlier work used, the scalar readout's AUCs on them
reproduce the preregistered reference values to three decimals: 0.749 on
no-target versus Normals, 0.866 on protected-target versus Normals, 0.637 on
insulting-only versus clean normals.

## Frozen clauses

| Clause | Rule | Result |
|---|---|---|
| 1. Information exists | Logistic probe, EN layer 27, no-target (34) vs protected-target (15), leave-one-out, AUC ≥ 0.75 | **PASS** at 0.837 |
| 2. Label-free access | Best of 24 unsupervised EN axes reaches AUC ≥ 0.799 on no-target vs Normals | **FAIL** at 0.746 |
| 3. Cross-corpus replication | One of HateClipSeg's top-3 axes at that layer reaches AUC ≥ 0.687 on insulting-only vs clean normals | **FAIL** at 0.617 |

All three were required. Clause 2 fails, so the family is dead; clause 3 was
evaluated anyway at layer 27, the layer of the strongest EN axis, and also
fails.

## The supervised probe

| Measure | Value |
|---|---:|
| Leave-one-out AUC, no-target vs protected-target | 0.837 |
| In-sample AUC on the same 49 videos | 1.000 |
| Scalar readout z on the same pair, same direction | 0.321 |

The in-sample fit is perfect because 4096 features and 48 training points always
separate, which is why only the leave-one-out number is read. That number is
0.837, so the layer-27 representation carries the distinction linearly.

The third row is the point of the test. Read in the same direction, the scalar
readout puts no-target positives *below* protected-target positives at 0.321,
which is 0.679 in the reverse direction. The judge's one logit contrast does not
merely lose the distinction; it spends it on severity, ranking protected-group
hostility above target-free abuse. The representation at the same depth holds
the distinction in the other direction at 0.837. That is the bottleneck the
preregistration described, and it is real.

## Unsupervised axes on MHClip-EN

AUC on no-target positives versus shipped Normals, with each axis's Spearman
correlation with z and its share of variance. Reference: z = 0.749, floor = 0.799.

| Layer | PC | Variance share | Spearman with z | AUC no-target vs Normal | AUC protected vs Normal |
|---:|---:|---:|---:|---:|---:|
| 18 | 1 | 0.240 | 0.073 | 0.624 | 0.499 |
| 18 | 3 | 0.058 | 0.154 | 0.695 | 0.771 |
| 18 | 5 | 0.037 | 0.379 | 0.567 | 0.701 |
| 18 | 7 | 0.029 | 0.291 | 0.543 | 0.732 |
| 27 | 1 | 0.459 | 0.992 | **0.746** | 0.860 |
| 27 | 2 | 0.155 | 0.147 | 0.553 | 0.518 |
| 27 | 3 | 0.048 | 0.067 | 0.474 | 0.679 |
| 36 | 1 | 0.413 | 0.995 | 0.741 | 0.859 |
| 36 | 2 | 0.225 | 0.043 | 0.522 | 0.451 |
| 36 | 6 | 0.023 | 0.083 | 0.593 | 0.503 |

Rows are the strongest and most informative of the 24; the full table for all
three layers is in the JSON. Zero axes reach the 0.799 floor and zero exceed z's
0.749. The best two axes, layer-27 PC1 and layer-36 PC1, are the z axis wearing
a different name: they correlate with z at Spearman 0.992 and 0.995 and carry 46
and 41 percent of the variance respectively. Their AUCs, 0.746 and 0.741, sit
just below z's 0.749, and their protected-target AUCs, 0.860 and 0.859, sit just
below z's 0.866. Unsupervised variance at the late layers rediscovers the
bottleneck rather than widening it.

No axis qualified for the construct-specificity table the preregistration asked
for, since the qualifying set is empty. The protected-target column above is
reported for the whole candidate set instead. It shows the same conflation
everywhere: axes that do better on the no-target stratum also do better on the
protected-target stratum, and no axis inverts the ordering the way the
supervised probe does.

Layer 18 rules out the obvious rescue. Its components are almost uncorrelated
with z, PC1 at 0.073 and PC2 at 0.000, so they are genuinely different
directions rather than copies of the readout. They are also uninformative: the
best of the eight reaches 0.695. The failure is not that PCA keeps finding z. It
is that the directions which are not z carry nothing on this stratum.

## Layer 36 alone, against the z bottleneck

Layer 36 is the row the scalar readout is taken from, so the gap between its
best unsupervised axis and z measures what the single logit contrast discards at
its own depth.

| Quantity | MHClip-EN | HateClipSeg |
|---|---:|---:|
| Best layer-36 unsupervised axis | 0.741 (PC1) | 0.616 (PC1) |
| Scalar readout z | 0.749 | 0.637 |
| Difference | −0.008 | −0.021 |

The gap is negative in both corpora. At the depth where the readout is taken,
the best label-free direction is slightly worse than the readout itself. There
is no discarded variance to recover at layer 36 by this route.

## HateClipSeg replication

At layer 27, the three axes clause 3 admits, scored on insulting-only versus
clean normals. Reference: z = 0.637, floor = 0.687.

| PC | Variance share | Spearman with z | AUC insulting-only vs clean normal |
|---:|---:|---:|---:|
| 1 | 0.444 | 0.966 | 0.617 |
| 2 | 0.137 | 0.480 | 0.327 |
| 3 | 0.050 | 0.148 | 0.605 |

The best is 0.617, below both the floor and z. The pattern matches MHClip-EN
component for component: PC1 is the z direction at Spearman 0.966 and reproduces
z's AUC to within 0.02, and the remaining components are at or below chance. The
same PC1 reaches 0.836 on hateful videos versus clean normals while sitting at
0.617 on insulting-only ones, which is the severity conflation again, measured
in a second corpus with different annotators and a different label scheme.

Across all 24 HateClipSeg axes the highest insulting-only AUC is 0.683, at layer
18 PC4, which correlates with z at 0.043. That single number is the only hint in
the whole test that a z-free direction might carry construct information, and it
still falls below the 0.687 floor, sits at a layer the frozen rule did not
select, and has no MHClip-EN counterpart. It is the sort of isolated maximum a
24-candidate search produces by chance, which is exactly what clause 3's
top-3 restriction was written to prevent from being read as a result.

## Interpretation

The preregistration named this outcome in advance: information exists but is not
reachable unsupervised, so the family is blocked under the label-free
constraint. The test cleanly separates two questions that the dual-axis failure
left tangled. Does the model internally distinguish targeted hate from generic
offensiveness? Yes, at 0.837 with labels. Can that distinction be picked up
without labels, from the corpus's own hidden states? No, at 0.746 against a
0.799 bar, in two corpora.

The reason the unsupervised route fails is visible rather than inferred. At
layers 27 and 36 the dominant direction of variance *is* the readout direction,
correlated with z above 0.96 in both corpora and carrying over 40 percent of the
variance. Unsupervised variance in a judge's late layers is dominated by the
thing the judge was asked to compute. Principal components therefore rediscover
the scalar the method was supposed to replace, and the components that escape it
are not about the construct at all. Widening the readout by looking for
label-free structure in the same states does not work, because the structure
that is easy to find is the readout and the structure that is not the readout is
noise on this task.

This also settles where the earlier dual-axis failure lives. Rewording the
question did not move the measurement, and now the internal state shows why: the
separating direction exists but is not the direction the model's own variance
points along, so no amount of prompt rewording will surface it and no
unsupervised summary of the states will find it. The distinction is a minority
direction in a representation whose majority direction is the answer to the
question that was asked.

## Decision

The readout-bottleneck method family does not proceed. No follow-on
preregistration for per-axis label-free thresholds or task-spec composition is
written.

The supervised rescue is explicitly not taken. A probe at 0.837 is a measurement
instrument, and the preregistration forbids promoting it to a method component;
doing so would trade the label-free claim for a number. Recorded and stopped.

What survives is a boundary result worth stating plainly. For this judge, on
these two corpora, the targeted-hate versus generic-offensiveness distinction is
linearly present in the mid-late representation and absent from every
unsupervised summary of it. Any future attempt on this distinction must obtain
its direction from somewhere other than the corpus's own unlabeled variance —
which, under the label-free constraint, means somewhere other than the hidden
states of a single judge call.
