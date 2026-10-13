# M1 candidate 42: evidence-bound pairwise comparison of windows

**归档（2026-10-13，第 33 项）**：两窗帧并排的比较式读取没有超过同样帧的逐点读取（within −.0099 / −.0033；`l + u` +.0017 / −.0068），HateClipSeg 上连最一致的边也只与逐点读数持平；配对比较这一族（时间戳引用版 PWC 与本次展示帧版）在该模型上关闭。`ROOT=parents[2]` 以 experiments/ 位置为准。

Status: declared 2026-10-13 before any GPU run (user instruction 2026-10-13: "run the proposal, and do not stop thinking
about improvements if it fails"). Development-selected (rule 10). Fixed r6 reads, fixed r6 decoder, sole evaluator.

## Evidence and hypothesis

- The within metric is the ordering of windows inside a video. r6 orders them from independent absolute reads
  ("is THIS window one of the segments ...", Yes/No log-odds, one per 8 s window).
- PWC (`experiments/20260912_pwc/README.md` §1, 2026-09-12, SPVL-r2 reads): among windows the model judged positive
  the within-video ordering was near chance on HateClipSeg (.566); the sign alone gives within .5715 / .5685 against
  .70 / .60 with the magnitude; the 8 s grid ceiling is .96 / .99. The bottleneck is the ordering, not the grid.
- PWC E0 (§7b) falsified pairwise comparison in the form "look at the frames whose timestamps fall inside each
  segment": accuracy against the GT label difference .654 / .570 versus .664 / .565 for `sign(z_i − z_j)`, and the
  content-blind control (timestamps only) fell to .546 / .511, so that comparison was carried by the two transcripts.
  The frames were referenced by timestamp, never shown.
- 2026-10-10/11 (`experiments/20261007_m1_streamingtom/README.md`, `archive/experiments/20261010_m1_grounded_adjacent/README.md`):
  the model finds the frames of a window by their timestamp labels in only 36 % / 44 % of the windows (probe A);
  showing the in-window frames right before the question gives within +.0224 / +.0128 with no new frame; every
  label-free way of deciding per window whether to take the extra read stays inside its random band.
- Hypothesis H1: with the frames of both windows shown next to the question, a relative judgement between two windows
  of the same video carries within-video ordering information that the pointwise read with the same frames (`l`,
  `adjacent_local`) does not. Alternative H2: the comparison reproduces `sign(l_i − l_j)`; then the comparison family
  is closed for this model and the candidate is archived.

## Mechanism (exact, both corpora)

1. Native prefix (20 frames, full ASR, rules), G, own hard stance, V and S per window: r6, unchanged (re-read in the
   job and bound to the placement records and to r6's predictions).
2. Comparison graph per video, label-free and reproducible: windows placed on a ring by a permutation from
   `numpy.random.default_rng([0, video_index])` (video_index = position in the fixed 333 list); edges at ring
   distances 1 and 2 (2N edges, degree 4, connected); the complete graph when N ≤ 4; no edge when N = 1.
3. Each edge (i, j) is asked twice on the shared cache as one user turn, i as A then j as A:
   `Window A, from {start}s to {end}s of this video. Frames inside window A:` followed by that window's LOCAL 0.5 fps
   frames (the stored frames of `placement_controls_main`, `data/temporal_retrieved_kv`, all frames inside the
   window, labelled `[t=..s]` like the prefix frames), the same for B, then
   `Consider the two windows above, A and B. Judge the visual content alone (imagery, gestures, symbols, on-screen
   text), ignoring the speech. Use the rest of the video only to resolve who or what is being referred to.`
   `Which window more clearly shows visual content that violates the above rules?`
   `Answer "A", "B", or "C", where C means they are equal or there is not enough evidence.`
   A window without LOCAL frames (7 of 7359) is introduced with `(No frames are available for this window.)`.
4. Read-out (as PWC §5b–5c): log-probabilities of the first tokens of A / B / C under the softmax restricted to
   them; `d = log((pA + pC/2) / (pB + pC/2))`; `l_ij = (d(i as A) − d(j as A)) / 2` (the antisymmetric part; the
   symmetric part is the position preference and is reported).
5. Aggregation per video: `u = argmin_{Σu = 0} Σ_edges (u_i − u_j − l_ij)²`, the Laplacian pseudo-inverse (exact
   recovery on a consistent input is checked by `compare_analyze.py --stage selfcheck`).
6. Composition into the visual channel; S, G, stance, fusion max(V, S) and the r6 decoder unchanged:
   - primary `cmp_order`: `z_visual_i = mean_video(l) + u_i` (level from the pointwise reads, ordering from the
     comparisons only);
   - secondary `cmp_sum`: `z_visual_i = l_i + u_i`.
   `l_i` is the stored `adjacent_local` read of the window (the native V where the window has no LOCAL frames), the
   pointwise read with exactly the frames the comparison sees.

Constants: seed 0; ring distances 1, 2; complete graph up to 4 windows; the three texts above; both orders per edge;
`d` and `l_ij` as above; nothing else. No labels in 1–6.

## Difference from earlier attempts (checked 2026-10-13)

- PWC (2026-09-12): same estimator family, but the windows' frames were referenced by timestamp, which the model
  cannot resolve (probe A); the content-blind control shows its comparison was transcript-driven. Here the frames are
  shown and the question is visual-only, so the comparison slots into the V channel the decoder already reads.
- Candidate 41 and its revision, the pre-read rules and the combination simulations: per-window selection or
  aggregation of pointwise reads; no relative question was asked.
- HVL, ReKV, StreamingTOM, ordered slots: memory or state across windows; no comparison.
- Explorer 18: acquisition of new frames gated by entropy; no comparison.
Rule 4 review: `docs/reviews/20261013_m1_pairwise_evidence_novelty.md`. Rule 6 review: `docs/reviews/20261013_m1_pairwise_evidence_code.md`.

## Inputs and cost

No new extraction: the LOCAL frames and the prefix frames exist. Per video: r6's calls, plus 2 × (number of edges)
comparison forwards on the shared cache, each with the frames of two windows (about 8 images). Estimate from the
placement run (`adjacent_local`, 4 images per forward, 0.11 s per forward on a 5090): 0.2 to 0.25 s per comparison,
HateMM 3768 windows → about 15 000 comparisons → 50 to 65 min; HateClipSeg 3591 windows → 50 to 60 min; plus the
native reads (5 / 4 min). Peak memory about 19 GiB. The method would cost about 4 to 5 native reads per window.

## Declared gates, controls and readings (before any run)

- Binding: the job's native reads equal the placement records (asserted per window) and r6's predictions (`prepare`);
  `full_l` composed from the stored reads must decode to the `adjacent_local` metrics exactly.
- Performance (rule 8, vs r6): the primary `cmp_order` loses no metric beyond the noise floor (.005 pooled, .01
  within) in either corpus and gains ≥ .01 on at least one metric in both corpora.
- Mechanism: `cmp_order` − `full_l` within ≥ .01 in both corpora with no pooled loss beyond .005 in either
  ("comparison supported"). `cmp_sum` is reported with the same criteria as the secondary.
- Permutation controls, six seeds: the edge values `l_ij` of each video permuted among its edges
  (`default_rng([100 + seed, video_index])`), then the same aggregation and compositions. A composition is not read as
  working unless it is above every seed in both corpora; the band is reported.
- Pilot on the fixed five videos first: proceed to the 333 only if the plumbing asserts hold (forward and vision
  counts, native reads equal, A/B/C token sets disjoint), swap agreement ≥ .6 and mean pC ≤ .8 on the pilot edges.
  Otherwise one wording revision of the question is allowed before the main run, recorded here; a second failure
  stops the candidate.
- Rule 10 diagnostics (labels, in `report` only): on graph edges whose windows carry different GT labels (window
  positive if more than half of its 4 fps frames are positive), accuracy of `sign(l_ij)` against the label
  difference versus `sign(l_i − l_j)`, `sign(v_i − v_j)`, `sign(E_i − E_j)`, `sign(a_i − a_j)` on the same pairs, with a
  95 % bootstrap interval over videos for the gain over `l`; the subset where both windows are fused-positive under
  r6; Spearman correlation of `u` with `l` inside videos; within-video standard deviations of `u` and `l`.
- Reading: mechanism supported → the comparison is the component (ablation: `full_l` is the comparison removed);
  performance passes but mechanism not → the gain is the input change, the comparison is not credited; neither →
  archive. Revisions (rule 9, at most three) only with an evidence-based change from the diagnostics.

## Run

```
# fixed five (pilot), sc474397
sbatch --export=SCOPE=smoke experiments/20261013_m1_pairwise_evidence/launch/lab1_compare.sbatch
# main, one corpus per machine is allowed (CLAUDE.md dispatch rule): e.g. HateMM on sc474397, HateClipSeg on sc474398
sbatch --export=SCOPE=main,DATASETS=HateMM experiments/20261013_m1_pairwise_evidence/launch/lab1_compare.sbatch
sbatch --export=SCOPE=main,DATASETS=HateClipSeg experiments/20261013_m1_pairwise_evidence/launch/lab3_compare.sbatch
# analysis (CPU, sc474397) after rsync of runs/20261013_m1_pairwise_evidence/main/records
STAGE=prepare bash experiments/20261013_m1_pairwise_evidence/launch/run_compare_analysis.sh
STAGE=build …; STAGE=evaluate …; STAGE=report …
```
Outputs: `runs/20261013_m1_pairwise_evidence/{smoke,main}/records/<ds>/<vid>.json`, `main_analysis/alignment.json`,
`selection/<name>/predictions.jsonl`, `selection_decoded/<name>/metrics.json`, `analysis/summary.json`.

## Files read (rule 10 log)

`runs/20261007_m1_streamingtom/placement_controls_main/records` (frames, stored reads; no labels),
`runs/20260926_glr/base_gridA/predictions.jsonl` (r6 reads for binding), `data/gt_4fps/*.npz` (sole evaluator and the
`report` diagnostics only).

## Pilot on the fixed five (2026-10-13, sc474397 / Slurm 406; no GT read)

Reading job complete for the five videos (158 windows, 316 edges, 632 comparisons, all with images); plumbing asserts
held (forward and vision counts, native V equal to the placement records per window, A/B/C first-token sets disjoint:
A {32, 64, 264, 362}, B {33, 65, 293, 425}, C {34, 66, 272, 356}); `prepare --smoke` PASS after one fix in the
analysis only (a numpy integer in the JSON summary; the reading job is unchanged). Numbers
(`runs/20261013_m1_pairwise_evidence/smoke_analysis/plumbing_summary.json`):

| | HateMM (3 videos, 192 edges) | HateClipSeg (2 videos, 124 edges) |
|---|---|---|
| swap agreement (sign of d flips with the order; edges with d ≠ 0) | .599 | .766 |
| mean pC | .554 | .173 |
| letter marginals P(A) / P(B) | .328 / .118 | .490 / .337 |
| mean position preference s = (d_ij + d_ji)/2 | +1.82 | +2.65 |
| mean \|l_ij\| | 3.03 | 7.54 |
| seconds per comparison forward | 0.21 | 0.20 |
| peak memory | 17.8 GiB | 17.2 GiB |

Pilot rule (declared above, on the pilot edges together): swap agreement 210 / 316 = .66 ≥ .6, mean pC .40 ≤ .8 →
proceed to the 333 with the declared wording. Noted for the reading of the results: the model prefers position A
(s ≈ +2 log-odds; the symmetrisation removes it), and on the long HateMM video `hate_video_114` (79 windows,
158 edges) it answers C in 66 % of the comparisons with swap agreement .54. Cost projection from 0.2 s per
comparison: HateMM about 50 min, HateClipSeg about 48 min, plus the native reads.

## Full run (2026-10-13; HateMM 215 on sc474397 / Slurm 407, HateClipSeg 118 on sc474399 / Slurm 408; development-selected)

Binding (`runs/20261013_m1_pairwise_evidence/main_analysis/alignment.json`, PASS on all 333): the job's native reads
equal r6's predictions (z_video, every window's V and S, score_curve) and the placement records per window; `full_l`
composed from the stored reads decodes to the `adjacent_local` metrics exactly. Comparisons: HateMM 7423 edges /
14 846 forwards, HateClipSeg 7182 / 14 362, all with images; comparison time 48.2 / 48.0 min per corpus (0.19 s per
forward), native reads 5.1 / 4.1 min, peak 17.8 / 17.5 GiB. Swap agreement .601 / .669; mean pC .485 / .312;
P(A) .316 / .426 against P(B) .198 / .262 (position preference s = +1.66 / +2.61 log-odds, removed by the
symmetrisation); mean |l_ij| 4.30 / 6.32.

Metrics (sole evaluator, fixed r6 decoder; `runs/20261013_m1_pairwise_evidence/analysis/summary.json`):

| arm | HateMM ROC / PR / within | HateClipSeg ROC / PR / within |
|---|---|---|
| r6 | .8971 / .6942 / .7508 | .7168 / .6711 / .6373 |
| `full_l` (pointwise, same frames; = `adjacent_local`) | .8977 / .6887 / .7690 | .7383 / .6893 / .6637 |
| `cmp_order` (primary: ordering from the comparisons only) | .8989 / .6950 / .7591 | .7384 / .6871 / .6605 |
| `cmp_sum` (secondary: `l + u`) | .8969 / .6931 / .7707 | .7394 / .6866 / .6570 |

| reading | `cmp_order` | `cmp_sum` |
|---|---|---|
| within vs r6 | +.0083 / +.0231 | +.0199 / +.0196 |
| pooled vs r6 (ROC, PR) | +.0018, +.0008 / +.0216, +.0161 | −.0002, −.0011 / +.0225, +.0155 |
| within vs `full_l` (mechanism gate, ≥ +.01 both) | **−.0099 / −.0033** | **+.0017 / −.0068** |
| permutation band, within vs r6 (six seeds) | −.0158 to +.0009 / −.0518 to −.0124 | +.0101 to +.0255 / −.0078 to +.0094 |
| above every seed | +.0074 / +.0355 | −.0056 / +.0102 |
| rule 8 vs r6 | FAIL (no metric ≥ .01 in both corpora) | PASS (within) |
| mechanism | not supported | not supported |

Reading per the declaration: the primary fails both gates. The secondary passes the performance gate against r6, but
not the mechanism gate: it is `full_l` plus a term that changes within by +.0017 / −.0068, inside its permutation band
on HateMM. The comparisons do carry ordering information (the primary is far above its permuted versions, +.007 /
+.036), but less than the pointwise read with the same frames already carries.

Diagnostics (rule 10, labels read in `report` only; graph edges whose two windows carry different labels):

| | HateMM (901 pairs, 75 videos) | HateClipSeg (1996 pairs, 96 videos) |
|---|---|---|
| accuracy of the comparison `sign(l_ij)` | .630 | .554 |
| `sign(l_i − l_j)` (same frames, pointwise) | .586 | .575 |
| `sign(v_i − v_j)` / `sign(E_i − E_j)` / `sign(a_i − a_j)` | .592 / .597 / .616 | .563 / .584 / .556 |
| gain of the comparison over `l`, 95 % interval over videos | +.044 [−.010, +.104] | −.021 [−.055, +.010] |
| both windows fused-positive under r6: comparison / `l` | .578 / .516 (531 pairs) | .520 / .540 (661 pairs) |
| Spearman(u, l) inside videos | .39 (197 videos) | .57 (118) |
| within-video SD of u / of l | 3.56 / 2.70 | 5.56 / 4.10 |

## Test error analysis after the run (rule 10, 2026-10-13; `error_analysis.py`, `runs/20261013_m1_pairwise_evidence/error_analysis/summary.json`)

Files read: the main records, the decoded predictions of `full_l` / `cmp_order` / `cmp_sum` / `cmp_order_d1`,
`data/gt_4fps`, the placement records. Findings:

- Reliability does not predict benefit. Decoded per-video within on the within videos, split at the median swap
  agreement: HateMM above the median (40 videos, agreement .83) `cmp_order − full_l` = −.0132, at or below (44, .30)
  −.0069; HateClipSeg +.0027 / −.0089. The videos where the model compares consistently are not where the
  comparison ordering helps.
- Edge quality: on HateMM the comparison beats `l` on consistent edges (.675 vs .612, 508 pairs) and on low-pC edges
  (.666 vs .624); on HateClipSeg even consistent edges are only equal (.582 vs .580) and inconsistent edges are at
  chance (.482). So on HateClipSeg no edge filtering can lift the comparison above the pointwise read.
- Temporal distance: comparisons between neighbouring windows are useless. Gap 1: swap agreement .42 / .51, accuracy
  .48 / .42 on label-different pairs (pointwise `l` .48 / .54); gap > 5: agreement .57 / .73, accuracy .667 / .560
  (`l` .608 / .580). Adjacent windows look alike and the model answers inconsistently; only distant pairs are
  compared reliably. This rules out a "difference between consecutive windows" reading for this model.
- Aggregated ordering before the decoder, all label-different window pairs of the within videos, macro over videos:
  HateMM u .629, l .588, v .652, l+u .610; HateClipSeg u .570, l .608, v .566, l+u .603. With the max(V, S) fusion:
  HateMM r6 .745, `full_l` .719, `cmp_order` .711; HateClipSeg `full_l` .654, `cmp_sum` .652, r6 .631, `cmp_order`
  .628. On HateMM the window-level accuracies order the arms the opposite way from the decoded within (r6 highest
  before decoding, lowest after), so differences of this size at the window level are not what the decoded metric
  reflects; on HateClipSeg the comparison is below the pointwise read at every level.
- Graph density: half the edges (distance-1 ring only, `cmp_order_d1`) gives within +.0098 / +.0161 against
  +.0083 / +.0231 for the full graph; doubling the edges would not change the HateMM picture.

## Decision (2026-10-13)

The comparison does not add within-video ordering information beyond the pointwise read with the same frames: on
HateClipSeg it is below that read at the edge level, at the aggregated level and after decoding, even on its most
consistent edges; on HateMM it is at most equal and loses after decoding. Together with PWC (timestamp-referenced
comparison, 2026-09-12) this closes the pairwise family for this model under both evidence presentations: H2 holds.
No comparison-side revision can reach +.01 over `full_l` on HateClipSeg (its best edges are equal to `l`), so the
three rule-9 revisions are not spent. The candidate is archived (item 33); `cmp_sum`'s passing of the performance
gate is the frame input change already measured as `adjacent_local`, not the comparison, and is not credited.
What the run adds for the next step: neighbouring windows cannot be told apart by this model (no difference reading),
distant pairs can; the model's within-video relative judgements and its absolute ones carry the same information.
