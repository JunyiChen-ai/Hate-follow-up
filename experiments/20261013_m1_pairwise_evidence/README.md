# M1 candidate 42: evidence-bound pairwise comparison of windows

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
