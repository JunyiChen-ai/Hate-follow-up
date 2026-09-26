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
