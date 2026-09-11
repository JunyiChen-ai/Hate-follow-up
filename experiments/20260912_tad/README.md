# TAD — separating topic from act in label-free hateful video localization

Status: 2026-09-12 proposal + E0 kill test. Development-selected numbers only (rule 10); test reads
logged in §6. Built on SPVL-r2 (`experiments/20260910_spvl/`); predecessor `experiments/20260912_pwc/`
is archived as a negative result and supplies the diagnosis below.

## 1. The mechanism problem

Measured on the within subsets from `runs/20260910_spvl/full2_dual_evid_stance/predictions.jsonl`
(`experiments/20260912_pwc/diagnose_topic_confound.py`, PWC README §7c). Windows cross-tabulated by GT
label and by whether the window's own transcript mentions the video's target group:

| corpus | effect on the window log-odds |
|---|---|
| HateMM | mentioning the target group **+8.1**; being a GT-positive window **+4.3** |
| HCS | mentioning the target group **+7.0**; being a GT-positive window **+4.3** |

The nuisance factor is 1.6–1.9× the signal. On HateMM, inside the "mentions target" stratum the GT effect
nearly disappears (mean z +16.09 for positive windows vs +14.28 for negative ones).

**Diagnosis: the frozen MLLM's per-window judgement is largely topic detection — does this window talk
about the group the video is hostile to — not act detection — does this window attack them.** Hate videos
are topically homogeneous, so the confound is present in every window of every hateful video.

This explains three earlier results that otherwise look unrelated:

- PWC E0: comparison, absolute and joint-context read-outs all land in the same accuracy band
  (.55–.69) and agree with each other on .71–.83 of pairs. A read-out over topic-dominated scores is
  still topic-dominated, whatever the question form.
- The within ceiling of .70 / .60 on all seven MLLMs (SPVL README §11): the confound is a property of the
  task presentation, not of model capacity.
- The whole-video verdict being strong (pooled ROC .88 / .67) while the per-window ordering is weak: at
  video level, topic and act coincide; inside a hateful video they do not.

## 2. Method

Hate is an *attack on a group defined by a protected characteristic*. The task therefore has a nuisance
dimension that is definitionally present in every positive instance: reference to a protected group. The
method measures that dimension explicitly and removes it from the window score.

Everything in SPVL-r2 is unchanged except that each window gets one additional short read and the
residual is computed from the corrected score.

```
1-3. prefix, whole-video verdict z_video, stance turn                       [unchanged]
4.   per window: visual branch + speech branch, evidence question -> a_i = max(visual, speech)   [unchanged]
4b.  NEW per window: one topic branch -> t_i
5.   corrected window score  r_i = a_i - beta * t_i
6.   frame score = [z_video + mean_i(a_i)] + centred rank of r_i     (intercept unchanged)
```

Topic question (joint presentation, one branch, no modality split, no self-output):

```
Consider only window {i} of {n}, from {t1}s to {t2}s of this video. Transcript in this window: {...}
Look at the frames whose timestamps fall inside this window.

Does THIS window refer to, mention, or depict a group defined by a protected characteristic (such as
race, ethnicity, national origin, religion, gender, sexual orientation, disability, or immigration
status)? Ignore whether the reference is hostile, neutral, or supportive; answer only about whether
such a group is referred to.

Answer "Yes" or "No".
```

The group is never named: naming it would require the model's own whole-video hypothesis, which
`experiments/20260911_hvl/` falsified (a hypothesis in context makes the model mark 83 % of windows
positive, and another video's hypothesis works as well as its own).

`beta` is a per-video ordinary least-squares coefficient, `beta = cov(a, t) / var(t)`, clipped to
[0, 2]; `beta = 0` when `var(t) < 1e-6`. It uses only this video's own two score vectors — no labels, no
corpus statistics — which is the class of quantity rule 3 allows (same class as the rank transform and
the zero-mean centring already in the method). Declared alternatives, reported alongside: fixed
`beta = 1.0` and `beta = 0.5`.

### Why this is a mechanism and not a prompt change

The contribution is not the wording of the topic question; it is that the window judgement is decomposed
into a nuisance dimension and a signal dimension, where the nuisance dimension is fixed by the definition
of the task (hate = attack **on a protected group**, so reference to a protected group is present in every
positive and in much of the negative material of a hateful video) and is measured by the same frozen model
in the same forward. The ablation makes the claim falsifiable: `a` alone, `t` alone, and `a - beta t` are
three separate rows, and `t` must not be a renaming of `a`.

## 3. Inputs and cost

No new cache, no new preprocessing, no new model, no resolution change. Per video: prefix once, 1 verdict
read, 2N window reads (unchanged), plus **N topic reads**, i.e. 1.5× the window reads of SPVL-r2 and about
1.3× its wall time. Measured seconds/video reported per run (rule 14j).

## 4. Constants (declared before any run; identical for both corpora, rule 13)

Inherited unchanged from SPVL-r2: K = 20 uniform frames, pixel cap 100352 / 65536, S = 8 s fixed windows,
`SYSTEM_MESSAGE`, `YOUTUBE_RULES`, `READER_PRAG`, `VIDEO_QUESTION`, evidence-style window question, dual
branches with max, stance = verdict, Yes/No token sets, greedy read-out, seed 0, intercept
`z_video + mean_i(a_i)`, centred-rank residual.

New: the topic question above, verbatim, both corpora; one joint topic branch per window (no modality
split); `beta` per video by OLS clipped to [0, 2], with `1.0` and `0.5` as declared alternatives.

## 5. Plan and gates

### E0 — is the topic read a separate measurement? (kill test)

Run the method on the within subsets (HateMM 84 / HCS 99). Declared before the run:

1. **Degeneracy check.** If the median per-video Spearman(a, t) ≥ .9, `t` is `a` renamed (the 2026-08
   second-question result, Spearman .92–.98) and the direction stops immediately.
2. **Validity check.** `t` must track the topic and not the act: report within-video AUC of `t` against
   the GT window labels (expected: above chance but clearly below `a`, since positive windows do mention
   the group), and the mean `t` in each of the four cells of the §1 table (expected: driven by the mention
   factor, not by the GT factor).
3. **Gate.** Within from the corrected residual `r = a - beta t` must exceed within from `a` by **≥ .01 on
   both corpora**, for at least one of the three declared `beta` choices, with that choice then fixed for
   E1. Reference on these subsets: SPVL-r2 within .6968 / .6020 (cache path,
   `runs/20260910_spvl/mllm_table.md`).

### E1 — full run and gates

Both corpora, all 333 videos, one run of the final code, three metrics through the shared evaluator.
Rule 8 comparison gate vs T3AL and promotion gate vs SPVL-r2 (HateMM .8919 / .6831 / .6976; HCS
.7119 / .6664 / .6001), noise floor pooled .005 / within .01. Pooled is **not** frozen by the unchanged
intercept (PWC README §5e correction) and is reported at every stage. Rule 14g ablation: `a` alone
(= SPVL-r2), `t` alone, `a - beta t` for the three `beta` values, and a control where `t` is replaced by
the topic scores of a **different video**'s windows (the analogue of HVL's permuted-hypothesis control:
if the permuted correction works as well, the correction is not carrying window-specific information).

### E2 — cross-model check

If E1 passes, repeat on at least three further MLLMs from the §11 family study, with the pre-declared
reading rule (same sign beyond noise on ≥ 5 of 7 if all seven are run).

### Secondary evaluation (user ruling, 2026-09-12)

HCS main-table numbers stay on the full corpus. A secondary evaluation on the hate-labelled subset of HCS
is reported separately, to separate "the method is weak" from "the label definition differs from the
prompt". No change to the prompt, the method or the constants (rule 13).

## 6. Test-read log (rule 10)

- 2026-09-12: the §1 cross-tabulation (PWC README §7c) — GT window labels read to build the four cells;
  no model input touched. Design decision taken from it: measure and remove the protected-group reference
  dimension.
