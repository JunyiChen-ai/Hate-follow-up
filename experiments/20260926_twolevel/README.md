# Two-level probabilistic model over the SPVL-r2 reads (2026-09-26)

Status: declared before any run. CPU only, on cached reads. This redesigns existing components; it adds no new
module. User direction (2026-09-26): the method works but is not novel. Keep each working component's mechanism,
make its form less ad hoc, and require that no metric drops and that the component still does its job
(ablation). Gains are secondary.

Process note: this is a redesign of components of the current method, not a new candidate method. The rule-4
proposal review agent was not run. In place of the rule-6 code review, the plumbing checks of §6 are used, as for
`experiments/20260926_glr` at the user's request.

## 1. What changes and what stays

Module 1, the MLLM reads, is unchanged. The reads come from the cached run `runs/20260926_glr/base_gridA`: SPVL-r2
grid A on the fixed ASR, with the whole-video verdict `z_video` and the visual and speech reads of each 8 s window.
A later step will read every window under the "violating" hypothesis; this is not part of this CPU run.

| current part (ablation evidence, `experiments/20260922_til/README.md` §9) | mechanism | redesigned form here |
|---|---|---|
| duration prior: one two-state chain, stay 1 − 4/80 per 4 s cell | hate persists; weak gaps fill, strong changes stay sharp | one chain per modality; stay probabilities estimated from the unlabeled reads |
| dual branches, window score = max of visual and speech (scaled by corpus std) | either modality is enough | a moment is hateful if either modality's chain is in the hate state |
| read scaling by the corpus std | reads become evidence | Gaussian emissions (one variance per modality) estimated from the unlabeled reads; the non-hate mean may differ between violating and non-violating videos (the stance leak) |
| intercept `z_video + mean window z` | the cross-video order comes from the verdict and the extent | P(video violates \| verdict, all reads) by Bayes; the verdict is one observation of V |
| centred within-video rank | the within-video order comes from the window evidence | P(hateful at t \| V = 1, reads, at least one hateful cell) |
| (none) | — | frame probability = product of the two; intervals = runs of frames with probability ≥ 0.5 |

Model per video. V ∈ {0, 1}. The verdict read `z_video` ~ N(m_V, τ²). If V = 0, every chain state is 0. If V = 1,
each modality chain (visual, speech) is a two-state Markov chain on 4 s cells. It has its own initial hate
probability ι and stay probabilities a (non-hate) and b (hate), and it must be in the hate state in at least one
cell of at least one chain.

Each window's read of modality m observes whether chain m is hateful anywhere in the window's cells: an 8 s window
covers two 4 s cells, observed through pair states as in TIL. The read is distributed as:

- N(μ_11, σ²) if it is hateful there;
- N(μ_10, σ²) if it is not, in a violating video;
- N(μ_00, σ²) in a non-violating video.

Windows without speech force the speech chain to 0 there (speech hate needs speech). The two chains are
independent given V = 1, apart from the at-least-one constraint. Exact inference: forward–backward over pair states
per chain. The constraint is applied in closed form from each chain's all-zero path.

## 2. Parameter estimation (label-free)

EM over all test videos of one corpus, using the reads only. Parameters:

- π = P(V = 1);
- m_0, m_1, τ for the verdict;
- per modality: μ_00, μ_10, μ_11, σ, ι, a, b.

Initial values:

- π = .5;
- m_0 and m_1 = the 10th and 90th percentiles of `z_video`; τ = the std of `z_video`;
- μ_00, μ_10, μ_11 = the 10th, 50th and 90th percentiles of the modality's reads; σ = the std of those reads;
- ι = .5; a = b = 1 − 4/80.

Stopping rule: at most 300 iterations; stop when the relative gain of the total log-likelihood is below 1e-7. The
log-likelihood must not decrease by more than 1e-6 relative in any iteration (asserted). If a modality ends with
μ_11 ≤ μ_10, that arm is reported as failed, with no rescue. Scope: per corpus (primary, like the corpus std of the
current method), and both corpora pooled (declared alternative, for the rule-13 question).

## 3. Arms (all on `runs/20260926_glr/base_gridA`)

| arm | what |
|---|---|
| `current` | replica of the current method (`til_infer.py --model average --fusion max --dwell 80`); must equal `runs/20260926_glr/infer/base_gridA_d80/metrics.json` |
| `m2` | new time level; composition as now: intercept + centred rank of logit P(hateful at t \| V = 1) |
| `full` | new time level + new video level: score = log P(V = 1 \| ·) + log P(hateful at t \| V = 1, ·), plus intervals |
| `full_pooled` | `full` with EM over both corpora together |
| `abl_nocoupling` | `full` with independent cells (a = 1 − ι, b = ι): is the chain still doing its job? |
| `abl_sharedchain` | `full` with one chain whose windows emit both reads: do the per-modality chains matter? |
| `abl_noleak` | `full` with μ_10 = μ_00: does the leak term matter? |
| `abl_noatleast` | `full` without the at-least-one constraint |
| `abl_vverdict` / `abl_vreads` | `full` with the video posterior from the verdict only / from the reads only |

Constants: 4 s cells; the cached 8 s windows (grid A); interval threshold .5 on the frame probability; frames take
their cell's probability. Seed: not used (EM is deterministic).

## 4. Decision rule (declared before the run)

Noise floor as in rule 7: pooled .005, within .01.

- **No drop** (the user's criterion): on both corpora, no metric of `full` falls below `current` by more than the
  noise floor. Reported separately for `m2`, to isolate the time level.
- **Mechanism still at work**: `abl_nocoupling` costs ≥ .01 within on both corpora against `full`. The other
  ablations are reported, not gated.
- **New output**: interval F1@.3 / .5 / .7 of `full`. The current method outputs no intervals, so its F1 is 0.
- If `full` fails but `m2` passes, only the time level is kept. No rescue tuning in this round.

Per-video paired bootstrap intervals (4000, seed 0) are reported for within, as in the GLR analysis.

## 5. Cost

No model calls. CPU: EM plus inference over 333 videos. Reads reused from `runs/20260926_glr/base_gridA`.

## 6. Plumbing checks (instead of the code-review agent)

1. `current` reproduces `base_gridA_d80` to 4 decimals on all six numbers.
2. Pair-state forward–backward (with forced cells and the all-zero path) equals brute-force enumeration on random
   sequences of up to 8 cells, to 1e-8.
3. The scoring code never opens a GT file. GT is read only by the evaluator and by the bootstrap analysis.
4. The EM log-likelihood is monotone (asserted).

## 7. How to run

```
bash experiments/20260926_twolevel/launch/run_all.sh
```

## 8. Test-read log (rule 10)

- Before this design (2026-09-26): `runs/20260926_glr/base_gridA/predictions.jsonl` with `data/gt_4fps/*.npz`
  (error types of the current window score), and `data/frames_k20` frame times. Found: visual reads persist more
  than speech reads across adjacent windows, label-free: median rank correlation .35 / .56 vs .20 / .40, using
  only windows that hold their own frame. GT boundaries are no closer to Whisper chunk edges or to shot cuts than
  random points. Changed: one chain per modality; no structure-dependent switching.

## 9. Results (2026-09-26, uoa-lab1, CPU)

Plumbing: forward–backward vs brute force max difference 3.6e-15; `current` reproduces `base_gridA_d80` on all six
numbers; EM log-likelihood monotone in every arm; no arm had μ_11 ≤ μ_10. Source
`runs/20260926_twolevel/analysis/table.txt` and `runs/20260926_twolevel/<arm>/metrics.json` (pooled ROC / pooled
PR / within; interval F1@.3 / .5 / .7).

| arm | HateMM | HateClipSeg |
|---|---|---|
| `current` | .8956 / .6888 / .7546; F1 0 / 0 / 0 | .7136 / .6671 / .6364; F1 0 / 0 / 0 |
| `m2` (new time level, current composition) | .8952 / .6855 / .7122 | .7134 / .6661 / .6190 |
| `full` | .8470 / .5969 / .7122; F1 .298 / .246 / .183 | .7249 / .6583 / .6190; F1 .313 / .156 / .088 |
| `full_pooled` | .8460 / .5955 / .7061 | .7246 / .6512 / .6210 |
| `abl_nocoupling` | .8602 / .5923 / .6370 | .6928 / .6324 / .5908 |
| `abl_sharedchain` | .8578 / .6043 / .6985 | .6804 / .6315 / .5733 |
| `abl_noleak` | .8387 / .5923 / .7168 | .7245 / .6513 / .6305 |
| `abl_noatleast` | .8485 / .5978 / .7147 | .7253 / .6581 / .6192 |
| `abl_vverdict` | .8509 / .6008 / .7122 | .7198 / .6653 / .6190 |

Paired bootstrap of within: `m2` − `current` −.042 [−.078, −.003] / −.018 [−.035, −.001]; `abl_nocoupling` −
`full` −.075 [−.119, −.033] / −.028 [−.063, +.003]; `abl_sharedchain` − `full` −.014 [−.057, +.025] / −.046
[−.082, −.011].

**Decision (§4): fails.** `full` and `m2` drop beyond the noise floor on both corpora (within; HateMM pooled for
`full`). The mechanism check passes: without temporal coupling, within falls by .075 / .028. The only new
capability is interval output (F1@.3 about .30 on both corpora, where the current method has none).

Diagnostics (after the decision, not method versions; `runs/20260926_twolevel/diag_*`, within HateMM / HCS):

| diagnostic | within | reading |
|---|---|---|
| `m2` with dwell fixed at 80 s | .7148 / .6192 | the estimated dwell is not the cause |
| `m2` without forcing the speech chain in silent windows | .7143 / .6279 | forcing costs a little on HCS |
| current single chain with interval (pair-state) observations, max fusion (`diag_til_interval_d80`) | .7601 / .6303 | counting each read once is fine |
| same with sum fusion / average + sum | .7255 / .5999, .7203 / .6088 | adding the two modalities' evidence hurts |
| two chains + OR, current evidence y / corpus std, dwell 80, speech forced | .7375 / .6094 | |
| same without forcing | .7539 / .6173 | back to the current level on HateMM; OR costs .013 on HCS vs max |
| Bayes video posterior as the intercept, within ranks as now (`diag_lexi_bayesV`) | pooled .8230 / .5576, .7041 / .6311 | the Bayes video level ranks videos worse than `z_video + mean window z` |

Cause: the EM-calibrated Gaussian emissions treat every window read as strong, independent evidence. The
evidence per read is several times that of the current y / corpus std, so the persistence prior barely smooths.
The same overconfidence makes the Bayes video posterior (a sum over windows) a poor ranker of videos. The current
method works because it treats each read as weak evidence and lets the prior dominate. A principled version has
to model why reads are weak evidence: reads of one segment are strongly correlated, since they share the video's
context. Independent Gaussian draws do not capture this. Secondary: forcing speech to 0 in silence costs
.016 / .008; OR of two chains matches max on HateMM and is .013 worse on HCS.

Test-read log addition: the diagnostics above read only evaluator outputs (`metrics.json`), and the bootstrap read
`data/gt_4fps/*.npz`. Changed: nothing in this round.

## 10. Round 2 (2026-09-26): explicit segment durations

Status: declared before the declared runs; pilots listed in §10.2 were run first (development reads, rule 10).

### 10.1 Why a second round

Rule 9: round 1 failed the no-drop rule, but `full` raised HCS pooled ROC by .011, so the same method may be revised
(at most 3 rounds). Round 2 targets the cause found in §9: the EM-fitted time level does not smooth.

### 10.2 Error analysis and pilots before round 2 (test-read log)

Files read: `runs/20260926_glr/base_gridA/predictions.jsonl`, `data/gt_4fps/{HateMM,HateClipSeg}.npz`,
`runs/20260926_twolevel/*/{metrics.json,run.log}`. The analysis scripts ran from the session scratchpad; their
numbers are given here. Pilot outputs: `runs/20260926_twolevel/diag_r2_*`. Numbers are HateMM / HCS.

1. **Structure of the reads (label-free).**
   - Most read variance is between videos: visual 86% / 75%, speech 69% / 49%.
   - After removing each video's mean read, the correlation of adjacent windows is .50 / .59 (visual) and
     .40 / .32 (speech). It is about 0 by 32 s.
   - Visual–speech correlation of the video-centred reads is .18 / .09.
2. **How strong a read is as evidence (test-read).** Logistic slope of window GT (hate fraction ≥ .5) on the
   read, in videos with both classes:
   - visual .055 / .101 per unit, speech .068 / .044;
   - on video-centred reads: visual .067 / .066, speech .056 / .035.
   For comparison, the current evidence 1 / corpus std gives .150 / .177 and .079 / .093, and round 1's EM gives
   .785 / .957 and .748 / .319. So a read is weak evidence, and EM overstates it 5–14 times; the current scaling
   overstates it 1.3–2.7 times.
3. **Pilot a: fixed video effect.** Reads centred within each video, round-1 model, `m2`, no speech forcing
   (`twolevel.py --center`): .7030 / .6138. EM then fits rare short bursts as the hate state (visual hate dwell
   16 s, initial hate probability .12).
4. **Pilot b: evidence tempering and the leave-one-read-out criterion.** The time-level emissions are multiplied by κ
   at inference (EM unchanged; `m2`, no forcing).

   | κ | 1 | .5 | .25 | .125 | .0625 |
   |---|---|---|---|---|---|
   | within | .7143 / .6279 | .7369 / .6284 | .7423 / .6197 | .7490 / .6100 | .7429 / .6021 |
   | leave-one-read-out log density per read | −3.06 / −3.01 | −3.14 / −3.21 | −3.21 / −3.24 | −3.23 / −3.26 | — |

   The label-free criterion is best at κ = 1 on both corpora, which is the round-1 setting. The reason: a read is
   predictable from its neighbours whether or not they share the GT state. This refutes the round-2 idea proposed
   after round 1 (choose the evidence strength by predicting held-out reads). Not used.
5. **Pilot c: explicit durations (this round's change).** Settings: `m2`, no forcing, no constraint, mean durations
   80 / 80 s, shape k (sub-states per segment).

   | k | 1 | 2 | 4 | 8 |
   |---|---|---|---|---|
   | within | .7192 / .6297 | .7178 / .6261 | .7548 / .6296 | .7497 / .6236 |
   | with video-centred reads | — | .6889 / .6047 | .6931 / .5927 | .6777 / .5739 |

   The fitted evidence barely changes with k (visual slope .83 at k = 1, .81 at k = 4). The gain comes from the
   duration shape. With k = 1 (geometric), a single window's strong read can form its own hate segment. With
   k = 4, a segment lasts at least 4 cells (16 s, two windows), with its most likely length about 64 s.
6. **Pilot d: video level with k = 4.** Pooled ROC / PR:

   | video level | as the intercept (+ centred rank) | as the product P(V) × P(h \| V) |
   |---|---|---|
   | joint posterior (round 1) | .8235 / .5499, .7060 / .6317 | .8662 / .6385, .7259 / .6611 |
   | reads counted as ICC-effective reads (residual ICC .53 / .23 visual / speech on HateMM, .31 / .27 on HCS) | .8939 / .6756, .7154 / .6384 | .8657 / .6401, .7332 / .6648 |
   | two-component mixture over verdict + mean read per modality | .8961 / .6725, .7109 / .6490 | .8730 / .6604, .7287 / .6668 |

   The current intercept `z_video + mean window z` gives .8956 / .6888, .7136 / .6671. Every principled video level
   loses at least .013 pooled PR on at least one corpus. The video level stays unchanged in this round.
7. **Pilot e: carrier fusion.** One shared chain. A hateful window shows hate in visual only, speech only, or both,
   with carrier probabilities fitted by EM. Result: k = 4 .7418 / .6347, k = 1 .6984 / .6296. It is better on HCS
   and worse on HateMM than per-modality chains + OR.

Changed for round 2:
- Explicit durations with k = 4, development-selected from {1, 2, 4, 8}.
- Per-modality chains + OR, development-selected over carrier fusion.
- Speech forcing and the at-least-one constraint dropped (§9 diagnostics).
- Composition unchanged.

### 10.3 Model changes against §1

- **Durations.** Each hate segment and each gap lasts a negative-binomial number of 4 s cells: k sub-states in a
  row, each left with probability 4k / D per cell, where D is the declared mean in seconds. k = 1 is round 1's
  geometric chain. A segment shorter than k cells has probability 0. Exact inference: forward–backward over
  (previous cell's hate bit, sub-state).
- **Silent windows.** A window without speech has no speech observation; nothing is forced.
- **Constraint.** No at-least-one constraint.
- **Composition (primary, `r2_m2`).** Intercept `z_video + mean window z`, plus the centred rank of
  logit P(hateful at t | V = 1, reads). This is round 1's `m2`.
- **Intervals (`r2_full`).** Frame probability = P(V = 1 | ·) × P(hateful at t | V = 1, ·), with the joint video
  posterior of round 1; intervals are runs of frames with probability ≥ .5.

The persistence prior is still the only temporal coupling. The new shape states what the prior assumes: segments
have a typical length, and one window cannot make a segment. The emissions are fitted without labels instead of
being scaled by the corpus std.

### 10.4 Constants and arms

Constants, the same for both corpora:
- cell 4 s;
- k = 4 (development-selected, §10.2 item 5);
- D_gap = D_hate = 80 s (the current method's dwell);
- EM initialisation and stopping as §2 (without the constraint);
- interval threshold .5.

| arm | what |
|---|---|
| `r2_m2` | primary: new time level, current composition |
| `r2_full` | same time level, joint video posterior, product composition, intervals |
| `r2_k1` | ablation of the duration shape: geometric (k = 1) |
| `r2_nocoupling` | ablation of persistence: independent cells, P(hate) fitted by EM |
| `r2_noleak` | ablation of the stance-leak term: μ10 = μ00 |
| `r2_carrier` | fusion alternative (§10.2 item 7) |
| `r2_k2`, `r2_k8`, `r2_d40`, `r2_d160` | declared scans: k = 2, 8; D = 40, 160 s (both durations) |

`current` is round 1's run `runs/20260926_twolevel/current` (not re-run).

### 10.5 Decision rule

- **No drop:** `r2_m2` against `current`. None of the six numbers may fall by more than the noise floor
  (pooled .005, within .01).
- **Mechanism at work:** `r2_nocoupling` − `r2_m2` ≤ −.01 within on both corpora.
- **Duration shape as a claim (rule 14g):** `r2_k1` − `r2_m2` ≤ −.01 on at least one main metric on both corpora.
  Otherwise it is reported as a HateMM-only effect, not a claim.
- **Sensitivity:** the worst value over the scans is reported.
- **New output:** interval F1 of `r2_full`.
- If the no-drop rule fails, a third (last) round may follow.

Paired bootstrap of within as in §4 (`analyze.py --round 2`, output `runs/20260926_twolevel/analysis_r2/`).

### 10.6 Plumbing checks

1. The explicit-duration forward–backward must equal brute-force enumeration (k ≤ 2, up to 5 cells). For k = 1 it
   must also equal round 1's pair-state forward–backward. Tolerance 1e-8; logged in `r2_m2/run.log`.
2. Declared runs whose settings match a pilot must reproduce the pilot's numbers. The pairs are `r2_m2` /
   `diag_r2_k4`, `r2_k1` / `diag_r2_k1`, `r2_carrier` / `diag_r2_k4_carrier`, `r2_full` /
   `diag_r2_k4_full_vnone`.
3. Scoring never opens a GT file. EM is monotone (asserted).

### 10.7 How to run

```
bash experiments/20260926_twolevel/launch/run_r2.sh
```

### 10.8 Results (2026-09-26, uoa-lab1, CPU)

Source: `runs/20260926_twolevel/analysis_r2/table.txt` and `runs/20260926_twolevel/r2_*/metrics.json`. Numbers are
pooled ROC / pooled PR / within.

Plumbing checks:
- The self-test passes: 2.1e-14 against brute force and against round 1 at k = 1.
- The four declared runs that repeat a pilot reproduce it exactly.
- EM is monotone in every arm.

| arm | HateMM | HateClipSeg |
|---|---|---|
| `current` | .8956 / .6888 / .7546 | .7136 / .6671 / .6364 |
| `r2_m2` | .8954 / .6882 / .7548 | .7134 / .6664 / .6296 |
| `r2_full` | .8662 / .6385 / .7548; F1@.3/.5/.7 .331 / .265 / .221 | .7259 / .6611 / .6296; F1 .267 / .141 / .055 |
| `r2_k1` (geometric) | .8952 / .6859 / .7192 | .7136 / .6661 / .6297 |
| `r2_nocoupling` | .8949 / .6829 / .6278 | .7134 / .6661 / .6096 |
| `r2_noleak` | .8955 / .6888 / .7579 | .7137 / .6664 / .6428 |
| `r2_carrier` | .8954 / .6878 / .7418 | .7135 / .6667 / .6347 |
| scans k = 2 / 8, D = 40 / 160 s (within) | .7178 / .7497 / .7415 / .7525 | .6261 / .6236 / .6310 / .6247 |

Paired bootstrap of within (HateMM; HCS):

| comparison | HateMM | HCS |
|---|---|---|
| `r2_m2` − `current` | +.000 [−.031, +.036] | −.007 [−.029, +.017] |
| `r2_nocoupling` − `r2_m2` | −.127 [−.181, −.076] | −.020 [−.049, +.008] |
| `r2_k1` − `r2_m2` | −.036 [−.065, −.009] | +.000 [−.018, +.018] |
| `r2_noleak` − `r2_m2` | +.003 [−.009, +.015] | +.013 [−.003, +.030] |
| `r2_carrier` − `r2_m2` | −.013 [−.038, +.009] | +.005 [−.009, +.019] |

**Decision (§10.5):**
- **No drop: passes.** All six numbers of `r2_m2` are within the noise floor of `current`.
- **Mechanism at work: passes.** Without temporal coupling, within falls by .127 / .020.
- **Duration shape: HateMM only.** Geometric durations cost .036 on HateMM and nothing on HCS, so the shape is not
  a rule-14g claim.
- **Sensitivity.** Worst within over the scans: .7178 (HateMM, k = 2) / .6236 (HCS, k = 8).
- **New output.** `r2_full` gives interval F1@.3 of .331 / .267.
- **Leak term fails its ablation.** Removing μ10 ≠ μ00 raises within by .003 / .013, so under rule 14g the term is
  dropped. The reduced model (`r2_noleak`) is the round-2 result: .8955 / .6888 / .7579 and .7137 / .6664 / .6428,
  development-selected. Its own ablations are re-run in §10.9 so that the ablation evidence refers to the final
  model.

### 10.9 Ablations of the reduced model (declared before running)

The reduced model is `r2_m2` with μ10 = μ00, i.e. one non-hate mean per modality for all videos. Its arms repeat
§10.4 with `--noleak` added:

- `r2nl_full` (intervals);
- `r2nl_k1` (geometric);
- `r2nl_nocoupling`;
- `r2nl_carrier`;
- scans `r2nl_k2`, `r2nl_k8`, `r2nl_d40`, `r2nl_d160`.

Decision rules are as in §10.5, with `r2_noleak` in place of `r2_m2`. The no-drop check is `r2_noleak` against
`current`. Bootstrap: `analyze.py --round 3` (output `analysis_r2nl/`).

### 10.10 Results of §10.9 (2026-09-26, uoa-lab1, CPU)

Source: `runs/20260926_twolevel/analysis_r2nl/table.txt` and `runs/20260926_twolevel/r2nl_*/metrics.json`. Numbers
are pooled ROC / pooled PR / within.

| arm | HateMM | HateClipSeg |
|---|---|---|
| `current` | .8956 / .6888 / .7546 | .7136 / .6671 / .6364 |
| **`r2_noleak`** (round-2 result) | **.8955 / .6888 / .7579** | **.7137 / .6664 / .6428** |
| `r2nl_full` | .8686 / .6448 / .7579; F1@.3/.5/.7 .321 / .265 / .220 | .7354 / .6631 / .6428; F1 .283 / .157 / .079 |
| `r2nl_k1` (geometric) | .8952 / .6861 / .7126 | .7139 / .6668 / .6406 |
| `r2nl_nocoupling` | .8949 / .6828 / .6108 | .7129 / .6660 / .6002 |
| `r2nl_carrier` | .8954 / .6880 / .7422 | .7129 / .6659 / .5983 |
| scans k = 2 / 8, D = 40 / 160 s (within) | .7234 / .7475 / .7413 / .7495 | .6372 / .6303 / .6204 / .6327 |

Paired bootstrap of within (HateMM; HCS):

| comparison | HateMM | HCS |
|---|---|---|
| `r2_noleak` − `current` | +.003 [−.031, +.041] | +.006 [−.015, +.031] |
| `r2nl_nocoupling` − `r2_noleak` | −.147 [−.207, −.091] | −.043 [−.073, −.013] |
| `r2nl_k1` − `r2_noleak` | −.045 [−.080, −.015] | −.002 [−.019, +.014] |
| `r2nl_carrier` − `r2_noleak` | −.016 [−.042, +.010] | −.045 [−.073, −.017] |

**Decision:**
- **No drop: passes.** All six numbers of `r2_noleak` are within the noise floor of `current`. Within is higher by
  .003 / .006; that is inside the noise floor, so it is not a gain.
- **Mechanism at work: passes.** Without temporal coupling, within falls by .147 / .043, and both intervals exclude 0.
- **Duration shape: HateMM only.** Geometric durations cost .045 on HateMM and nothing on HCS, so the shape is not a
  rule-14g claim.
- **Fusion.** One shared chain with carrier fusion is below per-modality chains + OR on both corpora (−.016 / −.045).
- **Sensitivity.** Worst within over the scans: .7234 (HateMM, k = 2) / .6204 (HCS, D = 40 s).
- **New output.** Interval F1@.3 .321 / .283 (`r2nl_full`). The current method outputs no intervals.

What changed against the current method, in the time level only (composition unchanged):
- **Evidence.** Emissions fitted by EM on the unlabeled reads (one non-hate mean, one hate mean and one variance per
  modality) replace the corpus-std scaling.
- **Durations.** Negative-binomial durations (shape 4, mean 80 s) replace the geometric chain.
- **Fusion.** Per-modality chains, combined by "either chain is in the hate state", replace the max of scaled reads.

Mechanism finding: with fitted evidence, a geometric chain lets a single window's strong read form its own
hate segment, and smoothing stops (§9). A duration model in which one window cannot make a segment restores it,
without scaling the evidence down by hand. The current method gets the same effect differently: it keeps the
geometric chain and weakens the evidence by the corpus std.

Development-selected (rule 10): k = 4, OR fusion, removing the leak term (§10.2, §10.8).

## 11. Composition: calibrated video key (2026-09-26, declared before the declared runs)

### 11.1 What was seen first (test-read log)

These diagnostics use the reduced round-2 time level. Outputs are in `runs/20260926_twolevel/diag_r2v_*`; the code is
`diag_vlevel.py`.

Composition as now: score = K + centred rank, where K = z_video + mean window z in raw MLLM logits and the rank term
spans 1. Multiplying K by a scale s changes how much frames of different videos interleave. Pooled ROC / PR:

| s | 100 (pure video-first order) | 4 | 1 (now) | .5 | .25 | .125 |
|---|---|---|---|---|---|---|
| HateMM | .8940 / .6840 | .8946 / .6858 | .8955 / .6888 | .8965 / .6938 | .8978 / .6984 | .8993 / .7010 |
| HCS | .7113 / .6633 | .7118 / .6638 | .7137 / .6664 | .7155 / .6689 | .7185 / .6720 | .7217 / .6749 |

Readings:
- Pooled rises steadily as s falls, on both corpora. The raw key is overweighted against the within-video order.
- The label-free calibration below gives s = .357 / .341 and pooled .8970 / .6959, .7170 / .6706.
- Replacing K by a model posterior was worse (§10.2 item 6). A posterior over the verdict and the mean reads, used as
  the key at s = 1, gives .8960 / .6735 and .7114 / .6490. Used video-first, it gives .8940 / .6688 and .7084 /
  .6464.
- The expected-hate-fraction key (log P(V) + log mean P(h | V)) gives .8632 / .6180 and .7272 / .6722.

The unit of K against the rank term's span of 1 is an undeclared constant of the current composition.

### 11.2 Change

The key becomes a log-odds: logit P(V = 1 | K) = a K + b. Here a and b come from a two-component 1-D Gaussian mixture
with a shared variance, fitted by EM to the corpus's keys without labels (`key_calibration` in `twolevel_r2.py`).
The composition becomes logit P(V = 1 | K) + the centred rank of logit P(hateful at t | V = 1), where the rank term
is a within-video adjustment of at most ±.5 nats. For intervals, P(V = 1 | K) replaces the joint video posterior in
the product. The time level is `r2_noleak`.

The test-best scale (s ≤ .125) is not used. The declared primary is the label-free calibration.

### 11.3 Arms

| arm | what |
|---|---|
| `c_m2` | primary: calibrated key + centred rank |
| `c_full` | P(V = 1 \| K) × P(hateful at t \| V = 1), intervals |
| `c_norank` | ablation: calibrated key only (no within-video term) |
| `c_nokey` | ablation: centred rank only (no video term) |
| `c_s0.5`, `c_s0.25`, `c_s0.125` | declared scan: raw key × s (sensitivity; values already seen in §11.1) |

Baseline: `r2_noleak` (raw key, s = 1).

### 11.4 Decision rule

- **No drop:** `c_m2` against `r2_noleak` and against `current`. None of the six numbers may fall by more than the
  noise floor.
- **The video term still does its job:** `c_nokey` − `c_m2` pooled ROC ≤ −.01 on both corpora.
- **The within term still does its job:** `c_norank` − `c_m2` within ≤ −.01 on both corpora.
- Gains are reported, and are development evidence only (§11.1 was seen first).

### 11.5 Results (2026-09-26, uoa-lab1, CPU)

Source: `runs/20260926_twolevel/analysis_c/table.txt` and `runs/20260926_twolevel/c_*/metrics.json`. The first
launch crashed on a bug: the calibration was not stored. It was fixed in the next commit with the design unchanged,
then re-run. Label-free calibration: logit P(V = 1 | K) = .3575 K + .399 (HateMM), .3406 K + 3.508 (HCS).

| arm | HateMM | HateClipSeg |
|---|---|---|
| `current` | .8956 / .6888 / .7546 | .7136 / .6671 / .6364 |
| `r2_noleak` (raw key) | .8955 / .6888 / .7579 | .7137 / .6664 / .6428 |
| **`c_m2`** (calibrated key) | **.8970 / .6959 / .7579** | **.7170 / .6706 / .6428** |
| `c_full` | .8829 / .7010 / .7579; F1@.3/.5/.7 .318 / .262 / .217 | .7375 / .6775 / .6428; F1 .316 / .180 / .079 |
| `c_norank` | .8935 / .6813 / .5000 | .7107 / .6631 / .5000 |
| `c_nokey` | .5696 / .2778 / .7579 | .5699 / .5238 / .6428 |
| scan s = .5 / .25 / .125 (pooled) | .8965 / .6938, .8978 / .6984, .8993 / .7010 | .7155 / .6689, .7185 / .6720, .7217 / .6749 |

**Decision (§11.4):**
- **No drop: passes.** Against `current`, `c_m2` changes by +.0014 / +.0071 / +.0033 (HateMM) and +.0034 / +.0035 /
  +.0064 (HCS). Only HateMM pooled PR is above the noise floor.
- **The video term does its job:** without it, pooled ROC falls by .327 / .147.
- **The within term does its job:** without it, within falls to .5 (−.258 / −.143), and pooled falls by .004–.015.
- **The product composition (`c_full`) is mixed.** HCS rises by +.024 / +.010, HateMM PR rises by +.012, but HateMM
  ROC falls by .013. It is kept only as the source of intervals (F1@.3 .318 / .316).

Composition after this step: logit P(V = 1 | K) + centred rank of logit P(hateful at t | V = 1). Here K = z_video +
mean window z, and the rank term is a within-video adjustment of at most ±.5 nats. The remaining constant is the
rank term's span. The scan says a wider span (equivalently a smaller s) raises pooled on both corpora. No label-free
rule for it was found, so it stays at the rank's natural range.

## 12. Robustness checks of the redesigned method (declared before running; CPU, cached reads)

"New" = `r2_noleak` time level + calibrated key (`c_m2` settings). "Current" = `til_infer.py --model average
--fusion max --dwell 80` with the raw key (the current method). Both run on the same cached reads. These runs predate
the 2026-09-26 ASR loader fix, so their numbers differ slightly from the main runs.

### 12.1 Other MLLMs

These are the eight `full` runs of the family study (`runs/20260910_spvl/mllm/<model>/full`):
Qwen3-VL 2B / 4B / 8B / 32B, Qwen2.5-VL-7B, InternVL3.5-8B, LLaVA-OneVision-7B and Gemma-3-12B. For each model the
report gives the six numbers of both methods and their differences. Summary: for each metric, the number of models
where "new" is not below "current" by more than the noise floor.

### 12.2 Reading-module components under the new method

These are the Qwen3-VL-8B cache-path ablation runs (`runs/20260910_spvl/mllm/q3vl-8b/<arm>`):
- `full`;
- `nostance` (no stance turn);
- `noctx` (no transcript context);
- `noframes` (no frames; branches are joint and speech);
- `joint` (one joint branch instead of two);
- `winonly` (all of the above removed).

The ASR-segment run is skipped: its windows are longer than two cells, and the observation model needs fixed 8 s
windows. The new model takes the branch keys of each run, one chain per branch.

The report gives, for each method, each arm's change against `full`. A reading component counts as still doing
its job under the new method if removing it costs ≥ .01 on at least one main metric on both corpora (rule 14g).

Outputs: `runs/20260926_twolevel/robust/<model or arm>_{cur,new}/`, table
`runs/20260926_twolevel/robust/table.txt` (`summarize_robust.py`). Launch: `launch/run_robust.sh`.

## 13. Interval output by MAP decoding (declared before running)

`c_full` makes intervals by thresholding P(V = 1 | K) × P(hateful at t | V = 1) at .5. The explicit-duration model
can instead give its most probable segmentation. `decode.py` does this:
- The time level and the key are as in `c_m2`.
- A video gets intervals only if P(V = 1 | K) ≥ .5.
- For such a video, each modality chain's Viterbi path gives its hate cells, and a cell is hateful if any chain's
  path is in the hate phase.
- Intervals are runs of hateful cells, in seconds.

The score curve is the `c_m2` composition, so frame metrics equal `c_m2`. The only constant is the .5 gate, the same
value as `c_full`'s threshold.

Arm: `c_viterbi`. It is compared with `c_full` on interval F1@.3 / .5 / .7. There is no gate: this is a new output,
and the current method has none.

Result (2026-09-26, `runs/20260926_twolevel/c_viterbi/metrics.json`; 274 intervals). Interval F1@.3 / .5 / .7:

| arm | HateMM | HCS |
|---|---|---|
| `c_viterbi` | .324 / .276 / .235 | .245 / .131 / .072 |
| `c_full` | .318 / .262 / .217 | .316 / .180 / .079 |

Viterbi is slightly better on HateMM and worse on HCS. HCS has more and shorter GT segments: 3.3 per video, mean
37 s. A single most probable path with a mean duration of 80 s merges them. The thresholded product (`c_full`)
stays the interval output.

### 12.3 Results (2026-09-26, uoa-lab1, CPU)

Source: `runs/20260926_twolevel/robust/table.txt`.

- The first launch stopped at the window-only run, whose verdict is a constant (z_video = −5.53 for every video):
  the initial verdict variance was 0.
- Fix: a variance floor of 1e-6 (other runs are unchanged). The window-only arm was re-run.

**12.1 Other MLLMs.** Within, new − current:

| model | HateMM | HCS |
|---|---|---|
| Qwen3-VL-2B | −.042 | +.004 |
| Qwen3-VL-4B | +.021 | +.024 |
| Qwen3-VL-8B | +.014 | +.015 |
| Qwen3-VL-32B | +.006 | +.031 |
| Qwen2.5-VL-7B | +.060 | +.047 |
| InternVL3.5-8B | +.031 | +.038 |
| LLaVA-OV-7B | −.060 | −.026 |
| Gemma-3-12B | −.015 | +.014 |

- Models not below current beyond the noise floor: ROC 8/8 and 8/8, PR 6/8 and 6/8, within 5/8 and 7/8.
- The drops go with extreme fitted evidence. The EM evidence per read, as a multiple of the current y / std, is
  13–20 for the LLaVA and Gemma visual branches and 2.5–8 elsewhere. LLaVA's visual chain also gets an initial hate
  probability of 1.00.

**12.2 Reading components under the new method.** Within, arm − full; current method in brackets:

| removed | HateMM | HCS |
|---|---|---|
| stance turn | −.021 (−.005) | −.010 (−.035) |
| transcript context | −.045 (−.043) | −.011 (−.010) |
| frames | −.076 (−.033) | −.104 (−.043) |
| dual branches (one joint branch) | −.016 (+.011) | −.075 (−.056) |
| all of these | −.063 (−.057) | −.087 (−.062) |

Under the new method every reading component costs ≥ .01 within on both corpora. Under the current method, the
stance turn and the dual branches do not reach .01 on HateMM. HateMM pooled also falls without the transcript
(−.147 / −.185).

**Diagnostics after 12.1** (development reads; `runs/20260926_twolevel/robust/*_{iotast,nscore}`,
`runs/20260926_twolevel/diag_{lin_*,c_m2_iotast,c_m2_nscore}`):
- **Start distribution fixed to the stationary share (.5).** LLaVA .7239 / .5692, Qwen3-VL-2B .7338 / .5785,
  main run .7545 / .6426. Not a fix.
- **Current evidence (y / std, max fusion, one chain) with the new durations.** k = 1 reproduces the pair-state
  TIL arm (.7601 / .6303). k = 4 gives .7513 / .6147, so with weak evidence the negative-binomial shape hurts. The
  two routes to smoothing (weak evidence + geometric, fitted evidence + no one-window segments) are alternatives,
  not additive.
- **Normal scores before EM.** Each modality's reads are replaced by the normal score of their rank within the
  corpus; the key keeps the raw reads. Within:
  - main run .7525 / .6397;
  - across the 8 MLLMs, against current: HateMM −.027 (2B), +.027, +.005, −.001, +.069, +.032, −.006, +.067;
    HCS −.002, +.022, +.011, +.024, +.046, +.036, +.019, +.042.
  This leads to round 3 (§14).

## 14. Round 3 of the time level: normal-score reads (declared before the declared runs)

### 14.1 Change

Before EM, each modality's window reads are replaced by Φ⁻¹((rank − .5) / N), with the rank taken over all windows
of that modality in the corpus. Only the order of the reads within a corpus is used, not the MLLM's logit scale,
which differs between models and saturates differently (§12.3). Everything else is `c_m2`:
- EM Gaussian emissions without the leak term;
- negative-binomial durations (k = 4, 80 / 80 s);
- per-modality chains + OR;
- calibrated key with the raw reads, plus the centred rank.

This is the third and last revision of the time level (rule 9). Normal scores were chosen after seeing §12.3,
so they are development-selected.

### 14.2 Arms

| arm | what |
|---|---|
| `r3_m2` | primary (main run `runs/20260926_glr/base_gridA`) |
| `r3_full` | product composition, intervals |
| `r3_k1`, `r3_nocoupling` | ablations: geometric durations; independent cells |
| `r3_k2`, `r3_k8`, `r3_d40`, `r3_d160` | declared scans |
| `robust/<model>_r3` | the eight family-study runs (§12.1) |
| `robust/<arm>_r3` | the Qwen3-VL-8B reading ablations (§12.2) |

### 14.3 Decision rule

- **No drop:** `r3_m2` against `current`, all six numbers within the noise floor.
- **Mechanism at work:** `r3_nocoupling` − `r3_m2` ≤ −.01 within on both corpora.
- **Robustness (reported, compared with §12.1):** the number of MLLMs not below current beyond the noise floor on
  each metric, and the mean within change.
- **Reading components (reported):** each must cost ≥ .01 on a main metric on both corpora.
- **Choice between `c_m2` (round 2) and `r3_m2`.** Both must pass the no-drop rule. The one with more MLLMs not
  below current on within (summed over the two corpora) is kept. On a tie, the one with the higher mean within
  change is kept.

Launch: `launch/run_r3.sh`; bootstrap `analyze.py --round 5` (`analysis_r3/`); robustness table
`summarize_robust.py --suffix r3`.

### 14.4 Results (2026-09-26, uoa-lab1, CPU)

Sources: `runs/20260926_twolevel/analysis_r3/table.txt`, `runs/20260926_twolevel/robust/table_r3.txt`.
- The self-test passes (2.1e-14).
- `r3_m2` reproduces the pilot `diag_c_m2_nscore`.
- The robustness runs reproduce the `*_new_nscore` pilots.

| arm | HateMM | HateClipSeg |
|---|---|---|
| `current` | .8956 / .6888 / .7546 | .7136 / .6671 / .6364 |
| `c_m2` (round 2 + calibrated key) | .8970 / .6959 / .7579 | .7170 / .6706 / .6428 |
| **`r3_m2`** | **.8971 / .6953 / .7525** | **.7168 / .6705 / .6397** |
| `r3_full` | .8903 / .7070 / .7525; F1@.3/.5/.7 .322 / .282 / .230 | .7321 / .6709 / .6397; F1 .273 / .158 / .085 |
| `r3_k1` (geometric) | .8967 / .6925 / .7389 | .7166 / .6695 / .6296 |
| `r3_nocoupling` | .8961 / .6881 / .6367 | .7144 / .6689 / .5801 |
| scans k = 2 / 8, D = 40 / 160 s (within) | .7426 / .7469 / .7281 / .7530 | .6344 / .6277 / .6196 / .6308 |

Paired bootstrap of within (HateMM; HCS):

| comparison | HateMM | HCS |
|---|---|---|
| `r3_m2` − `current` | −.002 [−.030, +.028] | +.003 [−.021, +.029] |
| `r3_nocoupling` − `r3_m2` | −.116 [−.179, −.050] | −.060 [−.095, −.024] |
| `r3_k1` − `r3_m2` | −.014 [−.034, +.006] | −.010 [−.028, +.008] |

Robustness (the eight MLLMs, against current):
- HateMM: ROC 8/8, PR 6/8, within 7/8; mean within change +.021.
- HCS: ROC 7/8, PR 7/8, within 8/8; mean within change +.025.
- Drops beyond the noise floor:
  - Qwen3-VL-2B: HateMM within −.027; HCS ROC −.005, PR −.007.
  - Qwen2.5-VL-7B: HateMM PR −.005, while its within is +.069.
  - LLaVA-OV-7B: HateMM PR −.014.

Reading components under `r3` (within, arm − full; current method in brackets):

| removed | HateMM | HCS |
|---|---|---|
| stance turn | −.014 (−.005) | −.011 (−.035) |
| transcript context | −.039 (−.043) | +.001 (−.010) |
| frames | −.060 (−.033) | −.109 (−.043) |
| dual branches | +.009 (+.011) | −.074 (−.056) |

HCS pooled without the stance turn: −.010 / −.013.

**Decision (§14.3):**
- **No drop: passes.** All six numbers are within the noise floor of `current`.
- **Mechanism at work: passes.** Without coupling, within falls by .116 / .060; both intervals exclude 0.
- **Duration shape.** Geometric durations now cost ≥ .01 on both corpora (−.014 / −.010), so the shape meets the
  rule-14g point threshold. The intervals still include 0.
- **Choice.** The tie-break rule keeps `r3_m2`: its within count over the eight MLLMs is 7 + 8 = 15, against 5 + 7
  = 12 for `c_m2`. The price:
  - `c_m2` is .005 / .003 higher on within in the main run (inside noise);
  - under `r3`, the transcript context no longer reaches .01 on HCS and the dual branches do not on HateMM, while
    under `c_m2` all four reading components did.
- **Development-selected:** normal scores, k = 4, OR fusion, no leak term, the calibrated key.

**Redesigned method (`r3_m2`), changes against the current method:**
1. **Reads to evidence.** Each modality's reads become normal scores of their rank within the corpus. Gaussian
   emissions are fitted by EM without labels. This replaces y / corpus std.
2. **Duration prior.** Negative-binomial segment lengths (shape 4, mean 80 s), so one window cannot form a segment.
   This replaces the geometric chain.
3. **Modalities.** One chain per modality; a moment is hateful if either chain is in the hate state. This replaces
   the max of scaled reads.
4. **Composition.** The key K = z_video + mean window z becomes logit P(V = 1 | K) from a label-free two-component
   mixture, plus the centred rank. This replaces raw K plus the rank.
5. **New output.** Intervals from P(V = 1 | K) × P(hateful at t | V = 1) ≥ .5.

## 15. Three phases with learned durations: normal, hate, slip (declared before the declared runs)

### 15.1 Why (test-read log, 2026-09-27)

Files read:
- `runs/20260926_glr/base_gridA/predictions.jsonl`;
- `data/gt_4fps/*.npz`;
- `runs/20260926_twolevel/{current,r3_m2}/predictions.jsonl`;
- `runs/20260926_glr/infer/base_gridA_gauss8/predictions.jsonl`;
- `runs/20260910_spvl/mllm/q3vl-8b/{full,nostance}/predictions.jsonl`.

The analysis scripts ran from the session scratchpad; their numbers are given here. Numbers are HateMM / HCS.

1. **Wrong high reads are brief.**
   - Method: a window is "high" when its fused read (max of the corpus-std-scaled branches) is above the corpus
     median. A run of consecutive high windows with no GT-hateful window is a false alarm.
   - Inside violating videos, false-alarm runs last one window in 55% / 74% of cases and ≥ 4 windows in 21% / 5%.
     With the corpus 70th percentile as the threshold: 53% / 79% and 6% / 2%.
   - GT hate segments last a median of 4 / 3 windows (mean 61 / 43 s). 22% / 25% of them are a single window.
   - In benign videos, with the median threshold, false alarms are longer on HateMM (24% ≥ 4 windows). These are
     whole-video errors.
2. **Where lone false alarms (single-window false-alarm runs) end up.** Mean within-video percentile, 1 = first:

   | order by | HateMM | HCS |
   |---|---|---|
   | raw reads | .63 | .78 |
   | Gaussian smoothing, 8 s | .47 | .59 |
   | current method | .45 | .54 |
   | `r3_m2` | .34 | .49 |
   | GT-hateful windows (all methods) | .55–.56 | .54–.55 |

   False-alarm runs of two or more windows barely move (.65 to .63, .72 to .66).
3. **The stance shifts every window.** With the stance turn, all window reads of videos with a Yes verdict rise by
   2.39 / 2.08 logits; with a No verdict, by .68 / .60. The within-video order hardly changes: median Spearman
   between reads with and without the stance .93 / .92.

Change motivated by item 1: the time level gets an explicit phase for brief wrong reads. The durations of all
phases are learned instead of the declared shape 4 and mean 80 s of rounds 2–3.

### 15.2 Model (`slip.py`)

Per modality, each 4 s cell is in one of three phases:
- **normal** (read distribution N(μ0, σ²));
- **hate** (N(μ1, σ²));
- **slip** (N(μ1, σ²)): a window the MLLM reads as hateful although it is not.

Rules:
- Hate and slip windows share the same read distribution. They differ only in two respects: slip can occur in any
  video but hate only in a violating video (V = 1), and their durations differ.
- Every phase lasts a geometric number of cells with a learned mean. There is no declared shape or mean.
- Transitions: normal to hate with rate h (V = 1 only); normal to slip with rate s (any video); hate to normal;
  slip to normal.
- Each 8 s window's read observes "hate or slip in any of the window's cells", through pair states as in rounds 1–3.
- Videos judged benign therefore show what slips look like, and the fitted slip duration is then used inside
  violating videos.

Unchanged from `r3_m2`:
- normal-score reads;
- one chain per modality;
- P(hateful at t | V = 1) = 1 − Π over modalities of (1 − P(hate phase));
- the verdict model within EM;
- the calibrated key plus the centred rank;
- intervals from the product.

Inference is exact forward–backward over (previous cell high, phase). The M-step has a closed form:
- s = (normal-to-slip counts under V = 1 and V = 0) / (all transitions out of normal, V = 1 and V = 0);
- h = (normal-to-hate counts) · (1 − s) / (normal-to-normal + normal-to-hate counts under V = 1);
- the stay probabilities of hate and slip are their self-transition shares.

EM initialisation:
- h = s = .025, i.e. normal lasts 80 s on average in a violating video;
- hate stay .95 (80 s); slip stay .5 (8 s);
- start distribution (.5, .25, .25) under V = 1 and (.75, 0, .25) under V = 0;
- μ0 and μ1 at the 10th and 90th percentiles of the normal scores; σ² = their variance.

Stopping: 300 iterations or a relative gain < 1e-7. Monotonicity is asserted. Rates are clipped to [1e-6, .5].

### 15.3 Arms

| arm | what |
|---|---|
| `s_m2` | primary |
| `s_full` | product composition, intervals |
| `s_noslip` | ablation: no slip phase (normal / hate with learned durations) |
| `s_nocoupling` | ablation: independent cells |
| `robust/<model>_s` | the eight family-study runs (§12.1) |
| `robust/<arm>_s` | the Qwen3-VL-8B reading ablations (§12.2) |

Also reported for each corpus and modality: the learned mean durations (normal, hate, slip) and the start rates.

### 15.4 Decision rule

1. **No drop:** `s_m2` against `current`, all six numbers within the noise floor.
2. **The slip phase does its job:** `s_noslip` − `s_m2` ≤ −.01 within on both corpora.
3. **Persistence does its job:** `s_nocoupling` − `s_m2` ≤ −.01 within on both corpora.
4. **Story check (reported, not gated):** the learned slip duration is shorter than the hate duration for every
   corpus and modality.
5. **Robustness:** over the eight MLLMs, the within count not below current (summed over both corpora) must be
   ≥ 14 (`r3_m2` has 15).
6. **Outcome:** if 1, 2, 3 and 5 pass, `s_m2` replaces `r3_m2` as the candidate, since it removes the declared
   shape and mean durations. Otherwise `r3_m2` stays.

Paired bootstrap as before: `analyze.py --round 6` (`analysis_s/`). Robustness table: `summarize_robust.py --suffix s`.

### 15.5 Plumbing checks

1. The three-phase forward–backward equals brute-force enumeration over phase paths (up to 5 cells) in likelihood,
   hate marginals and expected transition counts, to 1e-8. Logged in `s_m2/run.log`. The value was 8.9e-15 before
   this declaration.
2. Scoring never opens a GT file. EM is monotone (asserted).

Launch: `launch/run_slip.sh`.

### 15.6 Results (2026-09-27, uoa-lab1, CPU)

Sources: `runs/20260926_twolevel/analysis_s/table.txt` and `runs/20260926_twolevel/robust/table_s.txt`. The self-test
passes (8.9e-15).

| arm | HateMM | HateClipSeg |
|---|---|---|
| `current` | .8956 / .6888 / .7546 | .7136 / .6671 / .6364 |
| `r3_m2` | .8971 / .6953 / .7525 | .7168 / .6705 / .6397 |
| `s_m2` | .8968 / .6923 / .7567 | .7160 / .6719 / .6178 |
| `s_full` | .8940 / .7131 / .7567; F1@.3/.5/.7 .335 / .293 / .250 | .7286 / .6490 / .6178; F1 .215 / .131 / .072 |
| `s_noslip` | .8970 / .6930 / .7586 | .7163 / .6694 / .6284 |
| `s_nocoupling` | .8958 / .6848 / .6347 | .7155 / .6689 / .6045 |

Learned mean durations of `s_m2`:

| corpus, modality | normal | hate | slip |
|---|---|---|---|
| HateMM visual | 682 s | 1429 s | 73 s |
| HateMM speech | 101 s | 315 s | 104 s |
| HCS visual | 216 s | 890 s | 89 s |
| HCS speech | 99 s | 246 s | 48 s |

Robustness over the eight MLLMs:
- within not below current: 6/8 on HateMM, 5/8 on HCS (11 in total; the rule needs 14);
- mean within change −.009 / +.001.

**Decision (§15.4): fails.**
- **No drop fails:** HCS within falls by .019.
- **The slip phase does no work:** removing it changes within by +.002 / +.011.
- **Robustness fails:** 11 < 14.
- **Story check:** slip is shorter than hate in every corpus and modality, but it is not brief (48–104 s).
- **Outcome:** `r3_m2` stays the candidate.

**Why.** Without labels, a brief wrong high read and a brief real hate segment look the same: 22–25% of GT hate
segments are one window. EM uses the extra phase for medium-length high stretches instead, and treats hate as
lasting most of a violating video. So the finding "wrong high reads are brief" (§15.1) cannot be learned as a
separate phase from unlabeled reads. "Hate lasts" has to enter as a declared assumption, which is what the minimum
duration of rounds 2–3 does.

Side result: `s_noslip` (normal / hate, geometric durations learned by EM, no declared shape or mean) gives
.7586 / .6284. Against current that is +.004 / −.008 within, inside the noise floor. Its learned durations are long
(hate 160–980 s), so strong persistence also comes from learned geometric durations when the reads are normal
scores. Round 3's `r3_k1` used geometric durations fixed at 80 s. It was not a declared candidate; see §15.7.

### 15.7 Follow-up: learned durations without a slip phase (declared before running)

The candidate is `s_noslip`: normal-score reads, two phases (normal / hate), geometric durations and start
probabilities learned by EM, composition as `r3_m2`. It has no declared duration constant.

Runs:
- `s_noslip_nocoupling` (persistence ablation);
- `robust/<model>_sn` for the eight family-study runs;
- `robust/<arm>_sn` for the reading ablations.

Decision:
- **No drop:** against `current` (already measured: passes).
- **Persistence:** `s_noslip_nocoupling` − `s_noslip` ≤ −.01 within on both corpora.
- **Robustness:** within count not below current ≥ 14 over the eight MLLMs.
- **Outcome:** if all pass, it is reported as the alternative with no declared durations, and the user chooses between
  it and `r3_m2`. If not, `r3_m2` stays.
