# PWC — pairwise window comparison for label-free hateful video localization

Status: 2026-09-12 proposal. No run yet. Development-selected numbers only (rule 10); every test read is
logged in §7.

Starting point: SPVL-r2 (`experiments/20260910_spvl/`), the current method. This experiment changes exactly
one thing in it — where the within-video ordering comes from — and leaves the prefix, the verdict and the
video intercept untouched.

## 1. Mechanism problem this experiment targets

SPVL-r2 produces the within-video ordering from N independent absolute judgements ("does THIS window
violate the rules, Yes/No"), one per 8 s window, each read as a Yes/No log-odds. The ordering metric
(within-video macro ROC) asks a relative question, and the three diagnostics below say the absolute
judgements carry almost no relative information once the yes/no boundary is crossed.

Diagnostics computed 2026-09-12 from `runs/20260910_spvl/full2_dual_evid_stance/predictions.jsonl` and
`data/gt_4fps/*.npz` (test read, logged in §7). HateMM / HateClipSeg:

The authoritative baseline is the shared evaluator: **within .6976 / .6001** for this run
(`runs/20260910_spvl/full2_dual_evid_stance/metrics_izv_plus_mean_rrank.json`; the cache path gives
.6968 / .6020, `runs/20260910_spvl/mllm_table.md`). The rows below come from an ad-hoc script, not from
`src/eval/evaluate.py`, and are marked as such: they differ from the evaluator on HateMM by .0024
(grid-length handling), which is why they are used only for the *relative* statements in this section and
never as a gate value.

| quantity (ad-hoc script, not the shared evaluator) | HateMM | HCS |
|---|---|---|
| frame-level within if every 8 s window took its true positive fraction (granularity ceiling) | .9596 | .9860 |
| frame-level within from the real-valued window log-odds (= the method) | .7000 | .6001 |
| frame-level within from the *sign* of the window log-odds only (yes/no, no magnitude) | .5715 | .5685 |
| window-level ordering among windows the model judged positive (z > 0) | .6851 | **.5662** |
| fraction of windows judged positive | 45.8 % | 50.7 % |

Reading: the 8 s grid is not the bottleneck (ceiling .96 / .99). On HateClipSeg the whole curve is
essentially a binary decision — the magnitude adds .032 over the sign, and among positively judged windows
the ordering is near chance. On HateMM the magnitude is worth more (.13) but the same ceiling gap remains.

Two explanations are consistent with these numbers and they imply different methods:

- **H1 no common scale.** Each window is judged in isolation, so its log-odds encodes "does this look like a
  violation at all", not "more than that other window". The model could separate the windows but is never
  asked to.
- **H2 no discriminative signal.** The model genuinely cannot tell these windows apart, and no read-out
  will recover an ordering that is not there.

E0 (§6) is designed only to separate H1 from H2, before any method is built.

A third problem, specific to hate and diagnosed earlier in this repo, is what a comparison should ask
about. Hate is a relation (statement + target + endorsement). The target is frequently named in a window
that carries no attack itself (SPVL README §1 problem A; removing the full transcript costs HateMM pooled
ROC .148 in the final configuration — `runs/20260910_spvl/mllm/q3vl-8b/full` .8920 vs `.../noctx` .7439 —
and .227 in the round-1 configuration, `runs/20260910_spvl/ablation_table.md`; either way the verdict
depends on material outside the window carrying the slur). An absolute
per-window question cannot distinguish *the window that carries the attack* from *the window that supplies
the context needed to understand it*; both look like "content related to a violation". A comparison can be
asked to make exactly that distinction, and that distinction is the hate-specific part of this proposal.

## 2. Method

Everything up to and including the window branches is SPVL-r2, unchanged. One new stage is inserted, and
one line of the composition changes.

```
1. shared prefix (system + 20 timestamped frames + full timestamped transcript + rules + reader)   [unchanged]
2. whole-video question -> z_video                                                                 [unchanged]
3. the model's own Yes/No appended as a completed assistant turn (stance)                          [unchanged]
4. per window: visual branch and speech branch, evidence question -> z_i = max(visual, speech)     [unchanged;
   z_i is now used ONLY for the video intercept]
5. NEW  pairwise comparison over windows -> window utility u_i
6. frame score = [z_video + mean_i(z_i)]  +  centred rank of u_i        (intercept unchanged; residual new)
```

### 5a. Which pairs are asked (comparison graph)

All pairs would be N(N-1)/2 (435 at the HCS median N = 30). Instead a fixed sparse graph:

- order the N windows around a ring by a permutation drawn from a generator seeded with `SEED` and `N`
  only (no labels, no scores, reproducible);
- connect every ring position to the positions at ring distance 1, 2 and 3 → 3N edges, every node of
  degree 6, diameter O(log N);
- if N ≤ 7 use the complete graph instead.

The permutation is deliberately not temporal order: a temporally local graph would tie distant parts of the
video together only through long chains, and the least-squares solution would accumulate error along them.

### 5b. What is asked (the comparison question)

One query per edge; both windows in the same query; the frames and the transcript of both windows are
given together (no modality split here). The prefix is shared, so the model still sees the whole video.

```
Consider two segments of this video.

Segment A: from {t1}s to {t2}s. Transcript in A: {...}
Segment B: from {t3}s to {t4}s. Transcript in B: {...}
Look at the frames whose timestamps fall inside each segment.

Use the rest of the video to resolve who is being referred to, whether a statement is
quoted or endorsed, and who is speaking.

Which segment contains a greater proportion of content that itself expresses an endorsed
attack on a protected group, as defined by the rules above? Merely naming a group, or
providing context that explains another segment, does not itself count as an attack.

Answer "A", "B", or "C", where C means they are equal or there is not enough evidence.
```

Read-out: log-probabilities of the first tokens of {A, " A", a, " a"} etc. at the answer position, no
generation. With `p_A, p_B, p_C` renormalised over the three sets,

    d(i, j) = log( (p_A + p_C/2) / (p_B + p_C/2) )

### 5c. Position-bias symmetrisation

Every edge is asked twice, the second time with the two windows swapped, and

    l_ij = ( d(i, j) - d(j, i) ) / 2

is kept. This is part of the estimator, not post-processing: the two queries measure the same quantity and
only the antisymmetric part is the comparison; the symmetric part is the position preference.

### 5d. Aggregation

    u* = argmin_{sum_i u_i = 0}  sum_{(i,j) in E} ( u_i - u_j - l_ij )^2

a linear least-squares problem on a connected graph, solved with the graph Laplacian pseudo-inverse. It is
a deterministic function of the video's own comparison results; no labels, no fitted hyper-parameters. Rule
3 ruling (user, 2026-09-12): repeated reads of one frozen model aggregated by a fixed deterministic rule are
not an ensemble; the comparison stage is still an explicit component and goes into the ablation.

### 5e. Composition

Residual = centred rank of `u_i` inside the video (frame takes its window's value). Intercept unchanged.

Correction (rule-4 review, 2026-09-12): an earlier draft claimed pooled metrics cannot move in round 1.
That is **false** — pooled ROC/PR are computed over frames pooled across videos, so changing which window
gets which centred rank changes which label sits at which score. The repo's own numbers bound the effect:
intercept only (no residual) vs the SPVL-r2 residual is .8829 / .6427 → .8919 / .6831 (HateMM) and
.6718 / .6157 → .7119 / .6664 (HCS), `runs/20260910_spvl/ablation_table.md`. The correct statement is:
the intercept is unchanged, so **any** pooled movement is attributable to the reordering. A worse ordering
can therefore fail the pooled promotion gate, and pooled must be reported at E1 as well as E2.
(`compose.py --residual` gains a source switch; the arithmetic is the same.)

## 3. Inputs and cost

No new cache, no new preprocessing. Same `data/frames_k20`, `data/asr_whisper_large_v3`,
`Qwen/Qwen3-VL-8B-Instruct`.

Per video: prefix encoded once (median 2839 tokens), 1 verdict read, 2N window reads (unchanged), plus
**6N comparison reads** (3N edges × 2 orderings), each ~120-180 tokens on the shared KV cache. Median N is
15 (HateMM) and 30 (HCS), i.e. 90 / 180 extra short reads. Expected 4-8× the current 1.5 s/video; the
measured seconds/video and total GPU time are reported per run (rule 14j). No extra frame encoding.

## 4. Constants (declared before any run; identical for both corpora, rule 13)

Inherited from SPVL-r2 unchanged: K = 20 uniform frames, pixel cap 100352 / 65536, S = 8 s fixed windows,
`SYSTEM_MESSAGE`, `YOUTUBE_RULES`, `READER_PRAG`, `VIDEO_QUESTION`, evidence-style window question, dual
branches, stance = verdict, Yes/No token sets, greedy read-out, seed 0, `FILL_UNCOVERED = -12`.

New:

| constant | value |
|---|---|
| comparison graph | ring over a permutation from `numpy.random.default_rng(SEED * 1000 + N)`, edges at ring distance 1, 2, 3; complete graph if N ≤ 7 |
| orderings per edge | 2 (A/B swapped) |
| answer token sets | first tokens of {A, " A", a, " a"}, {B, " B", b, " b"}, {C, " C", c, " c"}; the three sets must be disjoint (checked at start-up) |
| tie handling | `d = log((p_A + p_C/2)/(p_B + p_C/2))` |
| symmetrisation | `l_ij = (d(i,j) - d(j,i))/2` |
| aggregation | least squares with `sum u = 0`, Laplacian pseudo-inverse |
| residual | centred rank of `u` |
| intercept | `z_video + mean_i(z_i)` (unchanged) |
| comparison question | the wording in §5b, verbatim, both corpora |

Declared ablation switches: `--question {carrier,generic}` (generic = "which segment more clearly violates
the rules above", i.e. the carrier/context clause removed), `--graph {ring3,complete,temporal3}`,
`--swap {on,off}`, `--edges-per-node k`.

## 5. Plumbing checks (must pass before any number is read)

Written to `verify.json` on the first video of every run:

- the A/B/C token sets are disjoint and each is non-empty;
- string and token seam checks for the comparison branch, as in SPVL §5 (`seam_check_text`,
  `seam_check_tokens`);
- one comparison read through the cache path vs the same question as a plain independent call:
  `|Δd| < 3` nats (same gate as SPVL's branch check);
- swap identity: asking edge (i,j) and (j,i) must give `d(i,j) + d(j,i)` ≈ 0 only if the model has no
  position preference — this is *recorded*, not gated;
- the Laplacian solve reproduces `l_ij` exactly on a synthetic consistent input (unit test).

## 6. Plan and gates

### E0 — does the model compare better than it scores? (kill test, no method yet)

Purpose: separate H1 from H2. Cheapest possible form: no graph, no aggregation, no composition change.

- Videos: the within-defined subsets (HateMM 84, HCS 99). GT is read to build and score the pairs; the
  model never sees a label.
- Pairs: inside each video, window labels from the 4 fps GT (a window is positive if > 50 % of its frames
  are positive). Sample up to 5 pairs per video where the two windows have **different** labels and
  **both were judged positive by SPVL-r2** (`z_i > 0`) — that is exactly the population where the current
  score is near chance on HCS. Fixed seed, no selection by error.
- For each pair: the §5b comparison, both orderings, `l_ij`.
- Measures: (a) accuracy of `sign(l_ij)` against the GT label difference; (b) accuracy of
  `sign(z_i - z_j)` on the same pairs; (c) swap agreement rate; (d) share of probability on C.

**Decision rule, declared before reading:** proceed to E1 only if (a) exceeds (b) by **≥ 5 points on both
corpora**. Otherwise H2 stands, the direction is archived as a negative result with the numbers, and no
comparison method is built.

### E1 — pilot on the within subsets

Full method (§2) on HateMM 84 / HCS 99, intercept fixed to the SPVL-r2 value. Reference on the same
videos: within .6968 / .6020 (cache path). Also run, in the same pilot:

- control A: `--question generic` (is the gain from comparison, or from the carrier/context distinction?);
- control B: same pairs, same budget, but each window asked absolutely and `l_ij` replaced by `z_i - z_j`
  (is the gain from relative judgement, or from spending more compute?);
- record: cycle inconsistency over the graph's triangles, swap agreement, share of C.

Go to E2 if within improves by ≥ .01 on both corpora over SPVL-r2 and control B.

### E2 — full run and gates

Both corpora, all 333 videos, one run of the final code, three metrics. Rule 8 comparison gate vs T3AL and
promotion gate vs SPVL-r2 (HateMM .8919 / .6831 / .6976; HCS .7119 / .6664 / .6001), noise floor pooled
.005 / within .01. Rule 14g per-component ablation for every part claimed as novelty: the comparison stage,
the carrier/context wording, the symmetrisation, the graph. Rule 14 checklist filled in this README.

### Rounds (rule 9)

Modification rounds are counted from E1. Three at most; if none reaches the promotion gate the directory
moves to `archive/experiments/` with the best numbers and the diagnosis.

## 7. Test-read log (rule 10)

- **2026-09-12**, `runs/20260910_spvl/full2_dual_evid_stance/predictions.jsonl` + `data/gt_4fps/*.npz`, read
  by ad-hoc analysis scripts (kept in the scratchpad, results transcribed into §1). Computed: (i) frame
  within from the window scores, from their sign only, and from the oracle window positive fraction;
  (ii) window-level AUC restricted to windows with `z > 0`; (iii) the share of windows judged positive;
  (iv) the number of prefix frames falling inside each window; (v) GT span length statistics.
  Findings: the granularity ceiling is .96 / .99; the curve is close to a binary decision on HCS; ordering
  among positively judged windows is .685 / .566; 27 % / 34 % of windows contain none of the 20 prefix
  frames; GT spans have median length 13.0 s / 21.0 s.
  Design decisions taken from this: (a) the within-video ordering, not the window length or the frame
  budget, is the target of this experiment; (b) E0 is restricted to pairs of positively judged windows,
  because that is where the current score fails; (c) multi-scale windows and per-window frames are not
  pursued (the ceiling and the two earlier negative results make them uninteresting).

## 7b. E0 result (2026-09-12, lab-server sc448960) — the direction is falsified

Runs: `runs/20260912_pwc/e0/` (carrier + generic, 192 s) and `runs/20260912_pwc/e0b/` (adds the three
controls asked for by the rule-4 review, 382 s). 704 pairs over 147 videos (83 HCS, 64 HateMM); the other
36 videos of the within subsets have no pair with different GT labels and both windows judged positive.
Same prefix, stance turn and cache path as SPVL-r2; `verify.json` gives cache-vs-plain |Δz| 0.115.

Accuracy of the sign against the GT window-label difference, with a 95 % bootstrap CI over **videos**:

| arm | HateMM (295 pairs, 64 videos) | HCS (409 pairs, 83 videos) |
|---|---|---|
| `sign(z_i - z_j)`, the SPVL-r2 window scores (baseline) | **.664** | **.565** |
| carrier comparison (A/B/C, swap-symmetrised) | .654 (gain −.010, CI [−.075, +.045]) | .570 (+.005, CI [−.061, +.066]) |
| generic comparison ("which more clearly violates") | .661 (−.003, CI [−.060, +.050]) | .550 (−.015, CI [−.081, +.045]) |
| content-blind control (timestamps only) | .546 (−.119, CI [−.225, −.017]) | .511 (−.054, CI [−.141, +.032]) |
| joint-context absolute (Yes/No per segment, same two-segment prompt) | .685 (+.020, CI [−.041, +.078]) | .575 (+.010, CI [−.051, +.071]) |

**Declared rule: proceed only if the comparison beats the baseline by ≥ 5 points on both corpora. It does
not (−1.0 / +0.5 points). The direction is archived.**

The read-out itself is sound, so this is not an implementation failure: swap agreement .82–.86, letter
marginals P(A)/P(B)/P(C) ≈ .45 / .50 / .04 (no degenerate letter preference), and removing the two
transcripts costs .119 / .054, i.e. the comparison is content-driven, not a temporal prior. Every arm —
absolute, relative, joint-context — lands in the same band (.55–.69), and each agrees with the sign of
`z_i - z_j` on .71–.83 of pairs. The comparison recovers the same information the absolute score already
has, and no more: .664 / .565 matches the window-level AUC among positively judged windows computed in
§1 (.685 / .566).

**Conclusion: H2, not H1.** The within-video ordering is not limited by the read-out (no common scale) but
by what the model can discriminate. Asking differently cannot recover an ordering that is not in the
model's judgement.

Two further mechanisms were killed for free from existing runs, with no new inference:

- **Context contrast** ("what this window adds beyond the global context"): within of
  `z(full context) − z(window alone)`, using `runs/20260910_spvl/mllm/q3vl-8b/{full,winonly}` on the same
  videos, is .467 / .494 versus .699 / .602 for `z(full context)` alone (per-video correlation of the two
  conditionings .74 / .64). Subtracting the global conditioning destroys the signal rather than isolating it.
- Per-corpus recall of GT-positive windows splits sharply by speech: .89 (with speech) vs .38 (without) on
  HateMM, .70 vs .45 on HCS.

## 7c. What the diagnosis found instead (2026-09-12, no GPU)

Windows were cross-tabulated by their GT label and by whether the window's own transcript mentions the
video's target group (target string from the HVL E0 hypotheses, `runs/20260911_hvl/e0_hypothesis/`; plain
word match after dropping generic words like "people"/"group"). Mean window log-odds and the share judged
positive, on the within subsets:

| corpus | GT | mentions target | n windows | judged positive | mean z |
|---|---|---|---|---|---|
| HateMM | 0 | no | 628 | .60 | +3.75 |
| HateMM | 0 | **yes** | 52 | **.96** | **+14.28** |
| HateMM | 1 | no | 766 | .84 | +10.49 |
| HateMM | 1 | yes | 112 | 1.00 | +16.09 |
| HCS | 0 | no | 1298 | .38 | −1.15 |
| HCS | 0 | **yes** | 127 | **.69** | **+6.33** |
| HCS | 1 | no | 1401 | .65 | +3.63 |
| HCS | 1 | yes | 180 | .87 | +10.10 |

Main effects (averaged over the other factor): mentioning the target is worth **+8.1** log-odds on HateMM
and **+7.0** on HCS; actually being a GT-positive window is worth **+4.3** on both. The nuisance factor is
1.6–1.9× the size of the signal. Inside the "mentions target" stratum the GT effect nearly vanishes on
HateMM (+16.09 vs +14.28).

Reading: the frozen MLLM's per-window judgement is largely **topic detection** — does this window talk
about the group the video is hostile to — rather than **act detection** — does this window attack them.
Hate videos are topically homogeneous, so this confound is present in every window and no read-out over
those scores can remove it. Caveat: the target string is the model's own whole-video hypothesis, so
"mentions target" is not an independent annotation; and GT-negative windows that mention the target may
genuinely be borderline. The effect sizes are large enough that this does not explain them away, but the
next experiment must not depend on the model's own hypothesis (rule: no feedback of self-output).

Disposition: this directory is archived as a negative result. The diagnosis moves to
`experiments/20260912_tad/` (topic/act decomposition).

## 8. Relation to other work

- Pairwise / setwise ranking prompting and rank aggregation (Bradley–Terry, HodgeRank) are established in
  text retrieval and LLM-as-a-judge; transferring them is allowed under rule 4 but is not by itself the
  novelty claim. The claim has to be the carrier/context distinction and the fact that it is what produces
  a usable within-video ordering for hate.
- LELA (arXiv 2602.09637), the closest label-free hate localization method, scores every frame
  independently through five captioners and an LLM and explicitly does no cross-segment calibration — the
  failure this experiment targets is present there in a stronger form.
- NOVA (arXiv 2609.06360) supplies the idea of writing the non-target side into the prompt explicitly
  (there: forbidding confusable verbs in the "normal" descriptions); here it is the sentence "merely naming
  a group, or providing context that explains another segment, does not itself count as an attack". Its
  other device, a normality anchor built from the video's first frames, is **not** adopted: a hateful video
  may open with the attack.
- Falsified in this repo and not revisited: feedback of the model's own verbalised output (hypotheses,
  chains, revision summaries — `experiments/20260911_hvl/`), neighbour-window context, per-window frames,
  multi-scale windows, ASR-defined windows.
