# TIL: temporal inference over interval measurements (2026-09-22)

Host for measurement runs: written in each `runs/20260922_til/<run>/run.log` first line. Code: this directory +
`src/mllm_judge.py` + `src/video_inputs.py`. Evaluator: `src/eval/evaluate_four_datasets.py` (unchanged).

## 1. Problem this candidate addresses

SPVL-r2 (`experiments/20260910_spvl/`) asks a frozen MLLM one Yes/No question per fixed 8 s window and
ranks frames by the centred rank of those answers. Every window is an independent measurement; nothing
relates neighbouring windows, and the 8 s grid fixes boundary resolution at 8 s. Two weeks of candidates
that changed *what is asked* (pairwise, onset, counterfactual, hypothesis chains, speech-act readouts; run on
2026-09-11/12) all stayed inside the per-window noise, and asking the model itself to carry temporal state
in-context lost .03 within on HateMM and .04–.07 on HCS (HVL). The one lever not yet used is *inference over the measurements after they are taken*.

## 2. Mechanism

The MLLM is used only as an interval sensor: it answers "does the violating content occur inside this
interval" for intervals of fixed geometry. Localization is posterior inference of a latent hate timeline
`h(t)` from those interval measurements under an explicit temporal model.

Measurement (GPU, `til_measure.py`): identical to SPVL-r2 (shared prefix = rules + 20 frames + transcript;
whole-video verdict `z_v`; stance turn; per-window isolated visual and speech branches, evidence question,
prefix KV cache), run on **two 8 s grids**: grid A starts at 0 s (0–8, 8–16, …), grid B starts at 4 s
(4–12, 12–20, …). Every 4 s cell except the first (0–4 s) and possibly the last is covered by one window of each grid, and each window
by two branches. The 8 s question length is kept because it is the length at which the model answers best
(4 s and 16 s are worse, `runs/20260910_spvl/ablation_table.md`); the shifted grid supplies boundary
information that a single grid cannot (a boundary at 10 s is "somewhere in 8–16" on grid A alone; grid B's
4–12 window answers weakly and its 12–20 window strongly, which places the boundary in 8–12).

Inference (CPU, `til_infer.py`): latent cells of 4 s (the endpoint grid of the two window grids).
- Observation model: a window covering cells `I` with modality read `y_m` contributes a log-likelihood
  ratio `y_m / s_m` for "some cell in `I` is hateful" against "no cell in `I` is hateful" (deterministic OR
  support with a likelihood ratio, no leak term); `s_m` is the standard deviation of modality `m`'s reads over
  the corpus, computed from the predictions only (label-free but transductive at corpus level; the declared
  alternative pools both corpora, arm A4p). The window's ratio is the max over its modalities (SPVL-r2 semantics:
  either modality suffices); the sum of the modality ratios is the declared alternative.
- Duration prior: two-state Markov chain on the 4 s cells with stay probability `1 − 4/D`, `D` the mean
  dwell in seconds, the same for both states and both corpora.
- Because every window covers at most two consecutive cells, the chain state at cell `c` is the pair
  `(h_{c−1}, h_c)` (four states) and exact forward–backward gives the posterior `p(h_c = 1 | all reads)`.
- Frame score = [`z_v` + mean of grid-A window reads] (the SPVL-r2 intercept, unchanged) + centred
  within-video rank of the cell posterior log-odds expanded to 4 fps.

What is new information: grid B is a second set of MLLM calls on different intervals (not a re-reading of
existing scores, not the model's own outputs fed back). What is new computation: the interval-support
observation model and the duration prior, both outside the model. No labels anywhere.

Source / relation to prior work: interval measurements + explicit temporal inference is the classic
noisy-sensor / HMM setting; in video it is closest to LAVAD and T3AL (training-free per-segment scoring
followed by temporal refinement) and to weakly supervised action segmentation with duration models
(NN-Viterbi, semi-Markov). Their refinement operates on one grid of independent scores; the claim here is
that (i) the measurement geometry (fixed length, isolation, modality separation, shifted coverage) and
(ii) an observation model that respects the interval support carry information that smoothing of one grid
does not. Whether (ii) holds is exactly what the arms below test.

Proposal review (rule 4, 2026-09-22, independent agent): 放行. Literature record and required fixes are in
`docs/reviews/20260922_til_proposal_review.md`; the fixes are applied in this README (pooled metrics can move;
A1b and A4p arms added; grid-B call count; D development-selected; fallback wording; s_m transductive).
Code review (rule 6, 2026-09-22): no blocking finding; `run_measure.sh` marker fix applied; the pair-state
forward–backward was verified against brute-force enumeration (n ≤ 10, max error 1.6e-10).

## 3. Inputs and cost

Cached inputs reused: `data/frames_k20`, `data/asr_whisper_large_v3`, `data/omsl_v6_inputs/manifests`.
New MLLM calls per video for grid B: one prefix encoding, one whole-video verdict (needed for the stance turn;
its value is not used downstream, grid A's `z_v` is the intercept), one visual branch per grid-B window and
one speech branch per grid-B window with speech. In total the branch count doubles relative to SPVL-r2. SPVL-r2 costs about 1.5 s/video on one 5090 with Qwen3-VL-8B; TIL
costs about 3 s/video (two prefix encodings, twice the branches): about 17 min GPU for both corpora (333
videos) on one 5090, i.e. about 11 min HateMM + 6 min HCS per grid. Inference is milliseconds on CPU.
Grid A is re-run with the same code so that the baseline is on the same code path (HVL `p_base` showed the
`src/mllm_judge.py` path reproduces the SPVL-r2 mask path within noise: within −.004 / −.001).

## 4. Constants (declared before any run)

| constant | value |
|---|---|
| model | Qwen/Qwen3-VL-8B-Instruct, bf16, greedy readout, seed 0 |
| prefix | rules + 20 uniform frames (91 tokens each) + full timestamped transcript; frozen prompt text in `src/mllm_judge.py` |
| window length S | 8 s (both grids) |
| grid offsets | A: 0 s; B: 4 s (B's first window is 4–12 s; the 0–4 s cell is covered by grid A only) |
| branches | visual + speech, isolated (deep-copied prefix cache); speech branch skipped when the window has no speech, as in SPVL-r2 |
| window question | the SPVL-r2 evidence wording (HVL `yesno_question`, `has_hyp=False`), unchanged |
| cell | 4 s |
| modality scale `s_m` | std of all reads of modality `m` in the measurement runs of that corpus |
| fusion across modalities | max of scaled reads (declared scan: sum) |
| dwell `D` | 80 s, development-selected (the p_stay .9 at 8 s of the 2026-09-22 CPU check, §6); declared scan: 40, 160 s; worst reported |
| uncovered window | `FILL_UNCOVERED = −12` (inherited from SPVL; cannot trigger with 20 frames) |
| chain initial state | `h_{−1} = 0`; `p(h_0 = 1) = .5` |
| intercept | `z_v` + mean of grid-A raw window reads (SPVL-r2 definition) |
| residual | centred within-video rank of the 4 fps curve |

Both corpora use the same constants (rule 13).

## 5. Arms (all from the same two measurement runs; each is a `til_infer.py` call)

| arm | grid(s) | observation model | duration prior | purpose |
|---|---|---|---|---|
| A0 | A | raw max over branches, no scaling | none | SPVL-r2 replicate on this code path (baseline) |
| A1b | A | per-window scaled max | none | scaling alone (isolates the `s_m` scaling from the prior) |
| A1 | A | per-window scaled max | D = 80 s | prior on one grid (A1 − A1b = the prior alone; A1 − A0 mixes scaling and prior) |
| A2 | A + B | cell = mean of covering windows' scaled reads | none | more measurements, no model |
| A3 | A + B | cell mean as A2 | D = 80 s | averaging + smoothing (the null the method must beat) |
| **A4** | A + B | interval-support (pair-state chain) | D = 80 s | **the method** |
| A5 | A + B | interval-support | none (p_stay = .5) | observation model without the prior |
| A6 | A + B | interval-support, sum fusion | D = 80 s | fusion ablation |
| B1 | B | as A1 | D = 80 s | grid B measures as well as grid A (within only: its intercept comes from grid B's own reads, so pooled is not comparable to A1) |
| A4p | A + B | as A4, `s_m` pooled over both corpora | D = 80 s | rule-13 check: the claim must not depend on per-corpus scaling |

Decision rule (declared): A4 is the candidate. Gate (rule 8) against the SPVL-r2 STATUS row
(HateMM .8919 / .6831 / .6976, HCS .7119 / .6664 / .6001; A0 only checks the code path): no metric down by
more than noise on either corpus; at least one metric up ≥ .01 on both. All three metrics are reported:
the residual changes which frames carry which score, so pooled ROC / PR can move even though the intercept is
unchanged (PWC README §2 records this). The *mechanism* claim additionally requires A4 − A3 ≥ .01 within on
both corpora (the observation model adds information beyond averaging + smoothing). If A4 passes the gate
but A4 ≈ A3, the shifted grid is only a second sample: the result is then reported as an explicit temporal
prior over isolated reads, which under rule 4 case (3) is not a new method but an explicit component of
SPVL-r2 with its ablation (A1 vs A1b), and the paradigm claim is withdrawn. If A4 fails the gate: rule 9
(one metric up ≥ .01 on one corpus → up to three revision rounds; otherwise archive). A1 − A1b measures the
prior alone; A1b − A0 the scaling alone.

Expected numbers (from the CPU checks below, single-scale smoke of §6 item 6): A1 ≈ .748 / .625 within;
A4 is predicted ≥ .76 / .63 if the interval model adds boundary information.

## 6. Development reads of test before this proposal (rule 10; all CPU, on cached SPVL runs)

Files read: `runs/20260910_spvl/full2_dual_evid_stance/predictions.jsonl`, `runs/20260910_spvl/mllm/*/full/`,
`runs/20260910_spvl/abl_s4`, `abl_s16`, `full`, `data/gt_4fps/*.npz` (labels only in the evaluator and in the
GT statistics).

1. **Two-state HMM over cached window reads** (emission `z/s`, `s` = corpus std, p_stay .9 at 8 s steps),
   intercept unchanged, within-video AUC base → HMM: HateMM 8B .698→.749, 2B .655→.764, 4B .675→.728,
   32B .701→.747, InternVL3.5-8B .660→.733, LLaVA-OV-7B .646→.730, Gemma-3-12B .667→.720, Qwen2.5-VL-7B
   .680→.673; HCS .600→.617, .567→.583, .581→.589, .602→.595, .604→.582, .565→.600, .576→.588, .567→.531.
   Pooled ROC/PR unchanged (±.003). A [.5, 1, .5] kernel gives about half the gain.
2. **Controls (8B, window level):** interior windows only (no GT boundary inside) HateMM .792→.807, HCS
   .633→.655 — not only a boundary effect; HMM applied to within-video *shuffled* reads .677 / .583 (below
   base) — the gain needs the true adjacency. Frame-level gain on HateMM (+.05) is about half boundary
   handling, half interior re-ordering.
3. **GT structure:** HateMM 1.49 positive runs per video, mean run 7.7 windows (62 s); HCS 3.34 runs per
   video, mean 4.6 windows (37 s). A long-dwell prior fits HateMM and fits HCS less; this is the predicted
   reason for the smaller HCS gain and is reported, not fitted.
4. **Window length with the same prior** (round-1 SPVL config, HateMM within): 4 s .657→.711, 8 s .678→.730,
   16 s .636→.711. Shorter windows do not reach the 8 s grid even with the prior, which motivates keeping
   8 s questions and getting resolution from the shifted grid instead.
5. **Rejected on the same cached data:** selecting or weighting the branch by transcript-only / frames-only
   whole-video verdicts (HateMM +.009, HCS −.010); mean of branches instead of max (−.003 / −.005).

6. **Smoke test of `til_infer.py` on the cached SPVL-r2 run** (single grid, so no new measurements): the
   replicate arm reproduces SPVL-r2 exactly (.8919/.6831/.6976, .7119/.6664/.6001). With the prior (D = 80 s),
   fusion by *sum* of scaled modality reads gives within .7193 / .6032, fusion by *max* .7482 / .6249; dwell
   scan with max: D = 40 s .7452 / .6229, D = 160 s .7448 / .6229. The interval model on the single grid is
   at the same level as the cell chain (.7518 / .6189), as expected without a second grid. Decision: max
   fusion is the declared default, sum the scan (this is a development-selected choice, made before any
   grid-B measurement exists).

Design decisions taken from these reads: cell = 4 s, scale by corpus std, dwell in seconds, max fusion as
the declared default with sum as the scan, grid B offset 4 s.

## 7. How to run

```
# measurement (one corpus pair per machine; lab machines, HateVLM env)
bash experiments/20260922_til/launch/run_measure.sh gridA 0
bash experiments/20260922_til/launch/run_measure.sh gridB 4
# inference + evaluation (CPU, all arms) after both runs are back in runs/20260922_til/
bash experiments/20260922_til/launch/run_infer.sh
```

## 8. Results

(filled after the runs)
