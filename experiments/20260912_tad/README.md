# TAD — separating topic from act in label-free hateful video localization

Status: 2026-09-12 proposal + E0 kill test. Development-selected numbers only (rule 10); test reads
logged in §6. Built on SPVL-r2 (`experiments/20260910_spvl/`); predecessor `experiments/20260912_pwc/`
is archived as a negative result and supplies the diagnosis below.

## 1. The mechanism problem

Measured on the within subsets from `runs/20260910_spvl/full2_dual_evid_stance/predictions.jsonl`
(`experiments/20260912_pwc/diagnose_topic_confound.py`, PWC README §7c). Windows cross-tabulated by GT
label and by whether the window's own transcript mentions the video's target group:

| corpus | effect on the window log-odds (unweighted cell means / count-weighted additive fit) |
|---|---|
| HateMM | mentioning the target group **+8.1 / +7.2**; being a GT-positive window **+4.3 / +6.3** |
| HCS | mentioning the target group **+7.0 / +6.9**; being a GT-positive window **+4.3 / +4.7** |

Correction (rule-4 review, 2026-09-12): the cells are very unbalanced (mention-yes is 52 + 112 of 1558
HateMM windows, 127 + 180 of 3006 HCS windows), so the unweighted difference of cell means overstates the
nuisance. Count-weighted, the nuisance is **comparable to** the signal on HateMM (ratio 1.15) and about
1.5× on HCS — not the 1.6–1.9× an earlier draft claimed. Inside the "mentions target" stratum the GT
effect is still small on HateMM (mean z +16.09 positive vs +14.28 negative).

Two caveats carried over from PWC §7c and both material: (i) the target string is the model's own
whole-video hypothesis, so "the window contains the target string" and "the model scores the window high"
are correlated partly by construction; (ii) the lexical mention marker fires on only 10.5 % / 10.2 % of
windows, so "the confound is present in every window" is a **hypothesis about the model's graded topic
response**, not an established fact — it is what the topic read `t` is meant to test.

**Diagnosis: the frozen MLLM's per-window judgement is substantially topic detection — does this window
talk about the group the video is hostile to — rather than act detection — does this window attack them.**

Premise mismatch declared before the run: the nuisance is definitionally present in every *hate* positive,
but HateClipSeg's GT positives are the offensive union (Hateful, Insulting, Sexual, Violence, Self-Harm),
so an HCS positive frame need not refer to a protected group at all. The correction is therefore predicted
to behave differently on the two corpora, and the hate-only secondary GT (`data/gt_4fps_hate_only/`) is the
place where the premise actually holds.

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

No new cache, no new preprocessing, no new model, no resolution change. Per video: prefix once, 1 verdict read, the
unchanged act reads (2 per window, but the speech branch is skipped on windows without speech: 3285 of
3768 HateMM windows and 2979 of 3591 HCS windows have speech), plus **one topic read per window**. Measured
over the test set that is 7053 → 10821 reads on HateMM and 6570 → 10161 on HCS, i.e. **×1.53 / ×1.55**,
not the "2N → 3N" an earlier draft stated. Measured wall time on the within subsets: 3.05 s/video
(`runs/20260912_tad/e0/run.log`) against SPVL-r2's ~1.5 s. Measured seconds/video reported per run (rule 14j).

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
Rule 8 comparison gate vs T3AL and promotion gate vs the promoted SPVL-r2 numbers (HateMM
.8919 / .6831 / .6976; HCS .7119 / .6664 / .6001, mask path). The *internal* comparison is against this
run's own `--curve act` row, because `tad.py` uses the KV-cache path whose SPVL-r2 numbers are
.8920 / .6825 / .6968 and .7132 / .6675 / .6020 (`runs/20260910_spvl/mllm_table.md`); all differences are
inside the noise floor, so the rule-8 gate is unaffected. Noise floor pooled .005 / within .01. Pooled is **not** frozen by the unchanged
intercept (PWC README §5e correction) and is reported at every stage. Rule 14g ablation: `a` alone
(= SPVL-r2), `t` alone, `a - beta t` for the three `beta` values, and a control where `t` is replaced by
the topic scores of a **different video**'s windows (the analogue of HVL's permuted-hypothesis control:
if the permuted correction works as well, the correction is not carrying window-specific information).

### E2 — cross-model check

If E1 passes, repeat on at least three further MLLMs from the SPVL §11 family study (which covers eight
models: Qwen3-VL 2B/4B/8B/32B, Qwen2.5-VL-7B, InternVL3.5-8B, LLaVA-OneVision-7B, Gemma-3-12B — its prose
says "seven", the table has eight rows), with the pre-declared reading rule (same sign beyond noise on at
least five of the models run).

### Secondary evaluation (user ruling, 2026-09-12)

HCS main-table numbers stay on the full corpus. A secondary evaluation on the hate-labelled subset of HCS
is reported separately, to separate "the method is weak" from "the label definition differs from the
prompt". No change to the prompt, the method or the constants (rule 13).

## 5b. E0 result, round 1 — subtraction is falsified (2026-09-12, lab-server sc448960)

Run `runs/20260912_tad/e0/` (183 videos, 559 s, 3.05 s/video, one topic read per window). Analysis
`analyze_e0.py`. Window-level within-video AUC on the videos that contain both classes:

Frame-level within through the shared evaluator (`runs/20260912_tad/e0/metrics_<tag>.json`), which is what
the gate is read on. The act-arm parity check demanded by the review passed: `--curve act` gives
.6926 / .6007, identical to HVL's same-code-path baseline (`runs/20260911_hvl/p_base`) and within the noise
floor of SPVL-r2's .6968 / .6020 (`runs/20260910_spvl/mllm_table.md`, cache path). An earlier compose bug
(proportional resize of the window curve instead of the frame-centre mapping) inflated it to .7086; fixed.

| curve | HateMM within | HCS within |
|---|---|---|
| act `a` (= SPVL-r2 window score, baseline) | **.6926** | **.6007** |
| topic `t` alone | .6567 (−.036) | .5467 (−.054) |
| `a − beta_OLS · t` (mean beta .80, median .71) | .6159 (**−.077**) | .5793 (−.021) |
| `a − 1.0 · t` | .5749 (−.118) | .5764 (−.024) |
| `a − 0.5 · t` | .6419 (−.051) | .5970 (−.004) |
| control: `a − beta_OLS · t'` from another video (mean beta .19, median 0) | .6830 (−.010) | .5933 (−.007) |

Window-level AUC on the same videos (`analyze_e0.py`, not the evaluator) tells the same story:
act .7581 / .6212, topic .6981 / .5630, corrected-OLS .6528 / .5983.

Degeneracy check **passes**: median Spearman(a, t) = .695 (HateMM) / .483 (HCS), far below the .9 stop
threshold, and the fitted coefficient is substantial (median beta .93 / .60, zero on 3 % / 2 % of videos).
So the topic read is a genuinely separate measurement, not the act read renamed — the 2026-08
second-question failure mode does not apply here.

**The gate fails: every subtraction lowers within on both corpora, monotonically in beta.** The reason is
visible in the second row: the topic read **alone** orders windows at .657 / .547, well above chance.
Referring to a protected group is not a nuisance variable in these corpora — it is positively predictive
of the GT label (positive windows mention the group more often: 13 % vs 8 % on HateMM). Subtracting a
predictive variable removes signal, not noise. The §1 cross-tabulation was right that the topic factor
dominates the *magnitude* of the score; the inference "therefore remove it" was wrong, because the factor
is correlated with the target as well as with the score.

The monotone decrease in beta (0 → .5 → .8 → 1.0 gives .693 → .642 → .616 → .575 on HateMM) is the
signature the reviewer named for "subtraction adds variance rather than removing a confound": a genuine
confound removal would show an interior optimum. The permuted control behaves as predicted too — its OLS
beta collapses towards the clip at 0 (median 0, mean .19), so it degenerates towards the act curve and
loses only .010 / .007.

Because the gate failed, the further controls the review asked for (a null auxiliary read, a second act
read in the rules wording, the beta profile beyond 1.0, the 2×2 table recomputed on the residual) would
only explain a gain that does not exist; they are not run. They are recorded here in case a later variant
of this idea produces a gain that needs explaining.

Disposition: round 1 (subtraction) is archived as a negative result. The diagnosis survives — the window
judgement is largely a topic judgement — but the correction has to change what is *asked*, not subtract
what was answered. That is round 2.

## 5c. Round 2 — the window decision as a speech act

Instead of correcting a binary "does this window violate the rules", the decision is re-specified as a
five-way choice over what the video **does** towards the group:

```
attacks   - it attacks, dehumanises, threatens, excludes or stereotypes such a group, and the video endorses this
reports   - it neutrally describes or reports on such a group or on someone else's attack
quotes    - it quotes or shows someone else's words without endorsing them
condemns  - it criticises or condemns such an attack
unrelated - it does not refer to such a group at all
```

read as a restricted softmax over the five first tokens (disjointness checked at start-up), with
`act_margin = log P(attacks) − logsumexp(log P(other four))`.

Why these five: they are exactly the carve-outs the project's own hate definition states — "Quotation,
neutral reporting, counterspeech, satire, and condemnation are not endorsement" (`LEGACY_POLICY`,
2026-08 judge). A binary violation question forces all five into one axis, which is why "mentions the
group" and "attacks the group" collapse onto the same score. `unrelated` absorbs the topic dimension that
round 1 tried to subtract, but inside the same decision rather than after it.

Cost: one extra read per window instead of one Yes/No read (`--acts 1 --topic 0`), same as round 1.

Declared gate (before the run): `act_margin` as the residual curve must beat the act curve by ≥ .01 within
on both corpora, either alone (`--curve actmargin`) or as an equal-weight rank sum with it
(`--curve act_plus_margin`). Diagnostics recorded: the distribution over the five acts, the share of
windows whose argmax is `unrelated`, and Spearman(act_margin, a).

### Round 2 result — also falsified (`runs/20260912_tad/e0_acts/`, 573 s, 3.13 s/video)

| curve | HateMM within | HCS within |
|---|---|---|
| act `a` (baseline) | **.6926** | **.6007** |
| `act_margin` alone | .6800 (−.013) | .5998 (−.001) |
| rank sum of `a` and `act_margin` | .6945 (+.002) | .6076 (+.007) |

**The gate (≥ .01 on both) fails.** The rank sum is the better of the two and is inside the noise floor on
both corpora.

Mechanism reading, from the recorded diagnostics: **the model never uses the carve-outs.** Argmax over the
five acts is `attacks` on 79.6 % of HateMM windows and 56.0 % of HCS windows, `unrelated` on 18.6 % / 34.9 %,
and the three exemption options together take 1.9 % / 9.2 % (`reports` .2 % / 5.3 %, `quotes` 1.6 % / 3.4 %,
`condemns` .1 % / .5 %). The five-way decision therefore collapses to `attacks` versus `unrelated`, which is
the topic axis again — median Spearman(a, act_margin) = .725 / .704, and the GT gap in `act_margin`
(+7.16 / +5.72) is the same size as in `a`. Re-specifying the decision does not make the model apply the
distinction; it re-labels the same axis.

### Disposition (rule 9)

Two modification rounds, neither reaching +.01 on both corpora: subtraction (round 1) and re-specifying
the decision (round 2). The directory is archived as a negative result. What survives is the measurement,
not the correction: the per-window judgement is substantially a topic judgement, the model can report the
topic dimension separately (Spearman .70 / .48), and neither removing it nor re-framing the decision
recovers ordering.

## 5d. Oracle ceiling of the per-window feature family (2026-09-12, diagnostic only)

`oracle_ceiling.py`, run on `runs/20260912_tad/e0/`. A classifier is fitted **on the test GT** with
grouped 5-fold cross-validation by video, and its out-of-fold score is evaluated — an upper bound on what
any label-free combination of the same features could reach. It is a rule-10 error-analysis read; the
fitted model is never a method arm and never enters a gate.

Window-level within, HateMM (75 videos, 1522 windows) / HCS (96, 2912):

| features | logistic regression | gradient boosting |
|---|---|---|
| the act read alone, **no fitting** | **.7581 / .6212** | — |
| MLLM reads (a, t, act_margin, visual, speech) | .7422 / .6250 | .6140 / .5582 |
| + window position, length, speech presence, transcript length | .7310 / .6258 | .6883 / .5628 |
| + ImageBind audio embedding of the window (32 PCA dims) | .7442 / .6141 | **.7928** / .5525 |
| + the neighbouring windows' reads | — | .6921 / .5476 |

**With real labels, a fitted combination of everything cached does not reliably beat the raw act read.**
The single exception is gradient boosting with the audio embedding on HateMM (+.035), which does not
transfer to HCS (−.069). Temporal context (the neighbours' reads) does not help either, which matches
HVL's neighbour-context arm (−.017).

**Representation ceiling** (`runs/20260912_tad/e0_hidden/`, 371 s, 53 MB of fp16 vectors): the same oracle
fitted on the *hidden state* at the window-branch read position — i.e. everything the Yes/No projection
discards — does not beat the scalar read either.

| features | HateMM within | HCS within |
|---|---|---|
| act read alone, no fitting | **.7581** | **.6212** |
| hidden state, 64 PCA dims, logistic regression | .7470 | .6078 |
| hidden state, 256 PCA dims, logistic regression | .6980 | .5520 |
| hidden state, 128 PCA dims, gradient boosting | .7460 | .5942 |

Limitation of this test, stated: the vector is read at the answer position of the window branch, which is
already downstream of the model's decision. The model's *pre-decision* contextual representation of the
window inside the prefix was not probed; that would need a token-to-time mapping through the image
expansion and was not built.

Consequence for the direction of the project: the per-window feature family is at its ceiling. Any method
that reweights, calibrates, combines or re-ranks these reads — including a self-training or pseudo-label
head over them — is bounded by a number the frozen read already reaches. The remaining levers are what the
model is *asked* to compute per window (round 2) or changing the model's own representation (adaptation),
not another combination of the current outputs.

## 6. Test-read log (rule 10)

- 2026-09-12: `oracle_ceiling.py` fitted a logistic regression and a gradient-boosting classifier on the
  **test** window labels with grouped cross-validation, to bound what any combination of the cached
  per-window features could reach (§5d). Finding: the fitted models do not reliably beat the unfitted act
  read. Design decision taken from it: do not pursue a learned head or a pseudo-label self-training stage
  over these features; the lever has to be what the model computes, not how its outputs are combined.
- 2026-09-12: the §1 cross-tabulation (PWC README §7c) — GT window labels read to build the four cells;
  no model input touched. Design decision taken from it: measure and remove the protected-group reference
  dimension.
