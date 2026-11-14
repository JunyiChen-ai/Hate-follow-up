# Experiment-section gates: baselines, DeHate margin, ablation rows (started 2026-11-14)

Host: sc474397 (uoa-lab1). Status: **in progress, no gate passed yet.** All numbers are on the test split and are
development-selected. Nothing in the method, STATUS or the paper has been changed.

## 1. Gates (set by the user, 2026-11-14)

1. Published label-free baselines vs the naive Qwen3-VL-8B row (HateMM 81.0/55.3/61.8, HCS 70.6/67.4/61.2, DeHate
   69.1/17.7/60.5): best published cell >= naive on at least 7 of 9 cells, the rest at most 1 point below.
   Work lives in `experiments/20261113_lf_baselines/` (a sub-agent started a §3 there; it was stopped when the
   session ran out of usage, see that README and `squeue` for what it left, e.g. Slurm job 2302 `lf-q3vl`).
2. DeHate: our pooled ROC and pooled PR each >= +2 points over the best label-free row. HateMM / HCS six numbers not
   below the current method by more than .005 pooled / .01 within. Method stays label-free.
3. Ablation: all 14 rows of the appendix remove-one table, every cell (3 corpora x 3 metrics) strictly below the
   full method; cells equal by construction exempt. No per-variant seeds (the pipeline is deterministic).

## 2. Findings so far (CPU, cached reads; scratch scripts, not yet moved into this directory)

Test-read log (rule 10): read `runs/20261101_paper_redecode/{main,dehate}/*/predictions.jsonl`,
`runs/20261113_lf_baselines/naive/vlm/*/{predictions,units}.jsonl`, `runs/20260927_dvd/reads_{main,dehate}/reads.jsonl`,
`runs/20261101_reader_ablation/decoded/*`, and the test GT, to compare video-level terms and decoder settings.

- **DeHate pooled is decided by the order across videos.** With a perfect within-video order the current method
  would reach only .700 / .177; with a perfect video-level order .926 / .378. Video AUC of the current video term
  (verdict + mean window read) is .723; the naive row's per-video maximum reaches .758.
- **The cached target-group read helps DeHate strongly** (`z_target` of `experiments/20260927_dvd`, one extra Yes/No
  branch per video on the same prefix, 0.5 s per video). Frame score = mean window read + a x verdict + b x target
  read + lambda x centred rank (current method: a = 1, b = 0, lambda = 1; 89.9/69.2, 73.3/68.9, 69.4/16.4):

  | a, b, lambda | HateMM ROC/PR | HCS ROC/PR | DeHate ROC/PR |
  |---|---|---|---|
  | .5, .5, 8 | 90.2 / 70.1 | 73.2 / 69.3 | 73.8 / 19.9 |
  | 0, .75, 8 | 90.0 / 70.1 | 72.8 / 69.0 | 76.0 / 21.6 |
  | each term divided by its corpus s.d., lambda = 1 (no constants) | 90.1 / 70.2 | 72.8 / 69.0 | 75.8 / 21.3 |

  These meet gate 2 against the naive row (+6.7 / +3.6 for the last row) if the baselines of gate 1 do not rise.
  HCS pooled ROC sits at the −.5 limit. Bootstrap intervals were running when the session stopped.
- **A larger within-video term (lambda about 8 in raw units) raises pooled ROC / PR on all three corpora** and makes
  the decoder ablation rows move the pooled metrics (they did not at lambda = 1).
- **Conflict that is not solved.** The target read hurts HCS (its GT counts offensive content that has no protected
  target) and the mean window read hurts DeHate once the target read is present. So "without window aggregate" is
  not below the full method on DeHate, and a "without target read" row is above the full method on HCS, for every
  fixed-weight or mixture-weighted form tried. A scope question written from each benchmark's own label taxonomy
  (HCS: hateful, insulting, sexual, violent, self-harm) is the untested way out; it needs the user's decision.
- **Decoder scans** (`runs/20261114_exp_gates/decoder_scan/{main,dehate}/<tag>/`, within HateMM / HCS / DeHate;
  current .7799 / .6866 / .6603): emission tempering .3 → .8029 / .6484 / .6245; .5 → .7942 / .6691 / .6407;
  .7 → .7876 / .6785 / .6500; tempering by residual lag-1 autocorrelation (`--ktemper ar1`) .7938 / .6535 / .6420;
  minimum 3 windows .7774 / .6760 / .6531; log-uniform duration prior .7775 / .7002 / .6544; fixed 80 s
  .7781 / .6735 / .6761; linear evidence .7958 / .6530 / .6573. No setting is at least as good as "fixed 80 s" on
  DeHate and "linear evidence" on HateMM at once. `--bma-prior learned` (corpus prior over durations fitted by EM)
  was still running (`learned`, `learned_g10`).
- Ablation table under the candidate integrator (.5, .5, 8), 16 cells not below the full method: Joint Query
  (DeHate ROC +0.8, within +1.4), Verdict Turn (HateMM ROC/PR +0.0/+0.1, DeHate +0.3/+0.3/+0.4), Window Evidence
  (DeHate ROC +0.2), Fixed Mean Duration (HateMM PR +0.1, DeHate within +1.6), Linear Evidence (HateMM
  +0.1/+0.3/+1.6), Global Transcript (HCS ROC +0.6, computed with the target read of the full prefix).

## 3. Code added

- `experiments/20260926_twolevel/twolevel_r2.py`: `--bma-prior learned`, `--ktemper ar1` (both off by default; the
  default flags reproduce `runs/20261101_paper_redecode/main/m2` exactly, run `decoder_scan/main/repro`).
- `decode.sh`: one decoder run on one read set. `scope_read.py` + `launch/scope_lab1.sbatch`: target read on the
  `no_transcript` and `no_prefix_frames` prefixes (needed for those two ablation rows). Slurm job 2300 is queued on
  uoa-lab1 behind another project's jobs; output `runs/20261114_exp_gates/scope_reads/`.

## 4. Next steps

1. Decide the scope question for HCS (universal protected-group question, or one per benchmark taxonomy).
2. Read the `learned` decoder runs; continue the decoder search for a setting that beats fixed 80 s and linear
   evidence on every corpus.
3. Re-decode every ablation arm under the chosen decoder and integrator and count the cells that do not drop.
4. Resume the baseline work of gate 1.

## 5. Gate table script and Integrator / decoder candidates (implementation session, 2026-11-15)

All numbers on the test split, **development-selected**. Host uoa-lab1 (sc474397), CPU. Nothing in the method,
STATUS, the paper or its tables was changed.

### 5.1 `gate_table.py`

`python experiments/20261114_exp_gates/gate_table.py --decoder <component[,component…]> [--combine prob|logit]
--form raw|sd|cdf --a <z_video weight> --b <scope weight> --lam <rank weight>` → `runs/20261114_exp_gates/gate_table/tables/<name>.{md,json}`.
Frame score `K_v + lam * centred rank`, `K_v = f_w(mean window read) + a f_v(z_video) + b f_t(scope read)`; `f` is the
identity (`raw`), division by the corpus s.d. (`sd`) or the corpus empirical distribution function (`cdf`), each
fitted on the full method's reads (no label). Rows: the 14 appendix rows plus `w/o Scope Read` (apart, not counted).
All rows share one decoder and one Integrator; no per-row tuning, no seeds (deterministic).
- Reader rows use their own reads. `w/o Global Transcript` / `w/o Global Frames` use the scope read on their own
  prefix (Slurm 2300, `runs/20261114_exp_gates/scope_reads/`); the other reader rows keep the full prefix and its
  scope read (`runs/20260927_dvd/reads_{main,dehate}`).
- `w/ ASR Windows` keeps its original composition (ASR windows cannot enter the time level): both sides use the
  SPVL composition's within-video term (`runs/20261101_reader_ablation/compose/{fixed,asr}_*`), with this run's
  Integrator term on each side.
- Decoder rows (`nocoupling`, `sharedchain`, `fixed80`, `lin`) add their flag to every decoder component. Missing
  decodes are run once into `runs/20261114_exp_gates/gate_table/decodes/`; existing decodes are reused (table EXISTING).
- Metrics: `src/eval/evaluate.py` `pooled` / `within_video_macro` in memory. Paired bootstrap over videos, 1000
  draws, seed 0, every cell. Count: a cell is "not below" when the change is > −0.05 points; within cells of the
  Integrator rows are equal by construction and exempt.
- Plumbing: `--check` (current method, `raw a=1 b=0 lam=1`) equals `runs/20261101_paper_redecode/{main,dehate}/m2/metrics.json`
  exactly (max |diff| 0.0 on all three corpora); the `current` table reproduces the appendix table (ROC and within
  columns, to rounding; e.g. w/o Global Transcript HateMM −14.4 / −1.6, w/ ASR Windows within −11.0 / −13.4 / −8.1).
- Gate 2 is checked against the best label-free cell per metric (naive row or the best published row of
  `runs/20261113_lf_baselines/gate/gate.json`, set A all published rows and set C peer-reviewed only). Since Probe-VAD
  (§3.1 of the baseline README), the best DeHate pooled ROC is Probe-VAD 71.47, so gate 2 needs DeHate ROC ≥ 73.47 and
  PR ≥ 19.69 (naive 17.69). CoReVAD (Slurm 2305) was still running when these tables were made.

### 5.2 Integrator candidates (decoder = current m2)

Tables `runs/20261114_exp_gates/gate_table/tables/m2_prob__*.md` and `current.md`. Not-below count over the 14 rows
(126 cells, 15 exempt); gate 3 needs 0.

| Integrator | full HateMM / HCS / DeHate (ROC / PR) | not below (14 rows) | `w/o Scope Read` not below | DeHate − best label-free ROC / PR | HateMM, HCS − current (ROC / PR) |
|---|---|---|---|---|---|
| current (raw 1, 0, 1) | 89.9/69.2, 73.3/68.9, 69.4/16.4 | 24 | 6 (= by construction) | −2.1 / −1.3 | 0 |
| (i) raw a .5, b .5, λ 8 | 90.2/70.1, 73.2/69.3, 73.8/19.9 | 19 | 3 | +2.4 / +2.2 | +0.34/+0.93, −0.02/+0.32 |
| (ii) raw a 0, b .75, λ 8 | 90.0/70.1, 72.8/69.0, 76.0/21.6 | 19 | 2 | +4.5 / +3.9 | +0.12/+0.84, −0.45/+0.09 |
| (iii) sd a 1, b 1, λ 1 | 90.2/70.0, 71.9/67.8, 73.8/20.2 | 18 | 4 | +2.3 / +2.5 | +0.36/+0.77, −1.38/−1.10 |
| (iv) cdf a 1, b 1, λ 1 | 89.5/68.1, 72.5/67.6, 73.5/20.4 | 12 | 2 | +2.1 / +2.7 | −0.34/−1.12, −0.81/−1.39 |

DeHate gate-2 bootstrap (`runs/20261114_exp_gates/gate_table/gate2_boot_dehate.txt`, 1000 draws): (i) vs Probe-VAD
ROC +2.4 [−0.5, +5.4], vs naive PR +2.2 [−1.5, +7.0]; (ii) vs Probe-VAD ROC +4.5 [+1.8, +7.3], vs naive PR +3.9
[−0.0, +8.8]; (iii) +2.3 [−0.6, +5.4] / +2.5 [−1.2, +7.2]; (iv) +2.1 [−0.9, +5.1] / +2.7 [−1.2, +6.8]. Only (ii) has a
ROC interval above 0 against Probe-VAD; every PR interval against the naive row contains 0 (or touches it, (ii)).

The two cells the brief asked about:
- `w/o Window Aggregate` on DeHate (ROC / PR): (i) +0.1 / +0.5 (not below, interval contains 0); (ii) +0.6 / +1.1
  (not below); (iii) −0.1 / +0.1; (iv) −0.9 / −1.4 (both below).
- `w/o Scope Read` on HCS (ROC / PR): (i) +1.8 / +2.0; (ii) +2.7 / +3.0; (iii) +2.3 / +2.4; (iv) +1.7 / +2.3. Removing
  the scope read raises HCS pooled under every form; no form tried makes both cells drop.
- Only (ii) meets gate 2 on all limits (HCS ROC −0.45, inside the −0.5 limit); (iii) and (iv) break the HCS limit.
- Cells not below under (iv), the lowest count: w/o Global Transcript HCS ROC / PR (+1.2 / +1.5), Joint Query DeHate
  ROC and within, Verdict Turn DeHate ROC and within, Fixed Mean Duration HateMM PR and DeHate within, Linear Evidence
  HateMM ROC / PR / within and DeHate ROC. The within cells (Joint Query, Verdict Turn, Fixed Mean Duration and Linear
  Evidence on DeHate / HateMM) do not depend on the Integrator; they need a decoder change (§5.3).

### 5.3 Decoder: averaging cell posteriors over evidence scales and duration priors

`runs/20261114_exp_gates/decoder_avg/scan.{json,log}` (script kept in the session scratchpad; the logic is
`gate_table.Store.rank`). All 502 subsets of the eight existing full-method decodes (m2, kappa .3 / .5 / .7, fixed
80 s, log-uniform prior, linear evidence, learned prior), averaged as probabilities or as logits (single
components once), centred rank, within only. Target: within ≥ .7958 (HateMM, linear evidence), ≥ .6866 (HCS,
current), ≥ .6761 (DeHate, fixed 80 s) at once.

- **No subset meets the three targets (0 of 502).** Closest by worst shortfall: fixed80 + log prior + linear
  (logit) .7873 / .6831 / .6705 (−.0085 on HateMM); m2 + fixed80 + linear (logit) .7901 / .6776 / .6723; m2 +
  fixed80 (logit) .7862 / .6805 / .6760.
- Named groups: evidence scales m2 + kappa .3/.5/.7 (logit) .7901 / .6779 / .6493; duration priors m2 + fixed80 +
  log + learned (logit) .7836 / .6875 / .6643; all eight (logit) .7894 / .6815 / .6553; (prob) .7970 / .6682 / .6454.
- Averaging moves each corpus toward the middle of its components; it does not reach the best component on all
  corpora at once. Not written into `twolevel_r2.py`.

### 5.4 Test-read record (rule 10)

Read: test GT `data/gt_4fps/{HateMM,HateClipSeg,DeHate}.npz`; predictions of every decode listed in `gate_table.py`
EXISTING, the reader-ablation reads and compositions, `runs/20261114_exp_gates/scope_reads/*`,
`runs/20260927_dvd/reads_*`, the naive and Probe-VAD predictions. Found: §5.2–§5.3. Design changed: none (candidates
are reported, not adopted). The Integrator forms (iii)/(iv) and the averaging scan were chosen after seeing the
earlier test numbers of §2; every configuration run is listed above.
