# Experiment-section gates: baselines, DeHate margin, ablation rows (started 2026-11-14)

Host: sc474397 (uoa-lab1). Status: **gate 1 passed 2026-11-16 (§12); gates 2 and 3 open (§13).** All numbers are on the test split and are
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
- Commit bd969f2 also carries the default-off `--bma-grid-spacing linear` option of experiments/20261108_duration_grid
  (uncommitted work of an earlier session).

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

## 6. Exploratory read: a HateClipSeg scope question from the dataset's own label taxonomy (G1 / B4; written before the run)

Status: exploratory, waits for the user's decision; the method is not changed. HateClipSeg labels each segment with
six categories, in this order (`scripts/build_gt_4fps_hate_only.py` line 10, from the HateClipSeg paper, arXiv
2508.01712; the CSV `~/data/HateClipSeg/Dataset/segment_level_annotation.csv` stores only the six-bit vectors): Normal,
Hateful, Insulting, Sexual, Violence, Self-Harm. The protocol GT counts a frame positive when any of the five
non-normal categories is set.

**Mechanism hypothesis** (main session, brief 3): HateClipSeg's ground truth counts all five kinds as positive; the
protected-group question covers only the first. A video-level read that states the benchmark's positive class should
order HateClipSeg videos better than the protected-group read and at least as well as the mean window read.

**Question** (text fixed by the main session, brief 3, 2026-11-15, before any read; `scope_read.py`
`HCS_TAXONOMY_QUESTION`, `--question hcs_taxonomy`; category names as the label dimensions, no glosses):

> Does this video contain content of any of these kinds: hateful, insulting, sexual, violence, or self-harm?
>
> Answer "No" if the video contains none of these.
>
> Answer "Yes" or "No".

An earlier text with a gloss per category was committed on 2026-11-15 (commits 2640be9, then a three-prefix variant)
and submitted as Slurm 2309 / 2310; both jobs were cancelled while pending, before any read, and the text was replaced
by the one above. No read with the earlier text exists.

**Run.** One isolated Yes/No branch per HateClipSeg test video (118) on three prefixes (full, `no_transcript`,
`no_prefix_frames`; the last two feed the two reader ablation rows), stored as `z_target`, with the verdict
`z_video` on the same prefix as a check against the stored reads. `QUESTION=hcs_taxonomy sbatch
experiments/20261114_exp_gates/launch/scope_lab1.sbatch` → `runs/20261114_exp_gates/scope_reads/tax_<prefix>_hcs/`.
Check: `z_video` of `tax_full_hcs` against the stored verdicts of the full reads (same tolerance as Slurm 2300).
Evaluation: G3 / F4 / F5 (§7). Cost if adopted: one Yes/No branch per video on the cached prefix (about 0.5 s).

**Result (Slurm 2311, uoa-lab1 sc474397, 2026-11-15, 3 min).** 118 videos on each of the three prefixes, no errors.
`z_video` on all three prefixes equals the stored verdicts value by value (max |diff| 0, 118 / 118 each;
`runs/20261114_exp_gates/scope_reads/tax_zvideo_check.txt`). G3 extras (`video_terms_eval.py taxonomy`, decoder m2,
full prefix; `runs/20261114_exp_gates/gate_table/g3_hcs.md`):

| HateClipSeg term | video AUC | Spearman vs positive-frame share |
|---|---|---|
| mw | .762 | +.616 |
| zv | .794 | +.441 |
| z_target (protected) | .771 | +.378 |
| z_tax | .825 | +.500 |

| K (raw, rank weight 8) | pooled ROC | pooled PR |
|---|---|---|
| mw | 75.5 | 72.0 |
| mw + .75 z_tax | 76.0 | 72.0 |
| .75 z_tax | 71.5 | 66.7 |
| mw + .5 zv + .5 z_tax | 75.0 | 70.9 |
| reference: mw + .75 z_target, Integrator (ii) | 72.8 | 69.0 |

The taxonomy read separates hateful from non-hateful HateClipSeg videos better than the protected-group read (.825
against .771) and better than mw and zv; its Spearman with the positive share (+.500) is below mw (+.616). In the
frame score, mw + .75 z_tax is +3.2 ROC / +3.0 PR over the (ii) reference and +0.5 / 0.0 over mw alone.
Development-selected (test labels read for the evaluation only).

## 7. Integrator forms F1–F5 (main-session brief 2026-11-15)

Forms fixed by the main session; no form chosen here. zT = protected-group scope read, mw = mean over windows of
max(z_visual, z_speech), σ(x) = s.d. of x over all videos of the corpus (full method's reads; the same constants for
every row), crank = the decoder's within-video centred rank (current m2). `gate_table.py --form F1..F5 --lam <λ>`.
- F1: K = mw + 0.5 zv + 0.5 zT, score K + 8 crank. F2: K = mw/σ(mw) + zT/σ(zT), λ 1. F3: A = mean_w min(z_w, zT),
  K = A/σ(A) + zT/σ(zT), λ 1. F4 / F5: F2 / F3 with HateClipSeg's zT replaced by the taxonomy read of §6 (Slurm 2311).
- Integrator rows (fixed): w/o Video-Level Term (K = 0); w/o Window Aggregate (K = the zT term only: F1 0.5 zT,
  F2–F5 zT/σ(zT)); w/o Scope Read (F1 mw + 0.5 zv; F2–F5 mw/σ(mw)), listed apart and not counted in the 14; Mean over
  Modalities (per window the mean of the available modality reads replaces the max; σ constants unchanged).
- Reader and decoder rows as in §5.1.

| form | full HateMM / HCS / DeHate (ROC / PR) | not below (14 rows) | w/o Scope Read not below | DeHate − best label-free ROC / PR | HateMM, HCS − current (ROC / PR) |
|---|---|---|---|---|---|
| F1 | 90.2/70.1, 73.2/69.3, 73.8/19.9 | 19 | 3 | +2.4 / +2.2 | +0.34/+0.93, −0.02/+0.32 |
| F2 | 90.1/70.2, 72.8/69.0, 75.8/21.3 | 19 | 2 | +4.4 / +3.6 | +0.17/+0.96, −0.47/+0.04 |
| F3 | 90.0/69.9, 71.4/68.0, 76.7/23.3 | 16 | 2 | +5.2 / +5.6 | +0.10/+0.66, −1.89/−0.95 |

Tables: `runs/20261114_exp_gates/gate_table/tables/m2_prob__F{1,2,3}_l*.md`. Gate 3 fails for all three.
- F1 equals Integrator (i) of §5.2 except for the w/o Window Aggregate row (now K = 0.5 zT only): HateMM −5.1 / −10.7,
  HCS −5.5 / −6.0, DeHate +1.9 / +1.6 (not below).
- w/o Window Aggregate on DeHate: F1 +1.9 / +1.6, F2 +0.6 / +1.1 (not below), F3 −0.2 / −0.9 (below, both intervals
  contain 0). w/o Scope Read on HCS: F1 +1.8 / +2.0, F2 +2.6 / +3.0, F3 +4.0 / +4.0 (all not below).
- Gate 2: F1 and F2 within all limits (F2 HCS ROC −0.47, limit −0.5); F3 breaks the HCS limit (−1.89 ROC).
- Remaining not-below cells common to F1–F3: Joint Query DeHate ROC and within; Verdict Turn HateMM ROC / PR and
  DeHate; Window Evidence DeHate ROC; Fixed Mean Duration HateMM ROC / PR and DeHate within; Linear Evidence HateMM
  ROC / PR / within and DeHate ROC; Mean over Modalities DeHate (F1, F2) or HCS ROC and DeHate ROC (F3).

**F4, F5, F6 (after Slurm 2311; HateClipSeg scope read = taxonomy read on each row's own prefix, HateMM and DeHate keep
the protected-group read).** F6 = `--form raw --a 0 --b .75 --lam 8 --hcs-taxonomy` (Integrator (ii) with the taxonomy
read on HateClipSeg). Tables `runs/20261114_exp_gates/gate_table/tables/{m2_prob__F4_l1,m2_prob__F5_l1,F6}.md`
(1000 bootstrap draws; the two decoder rows of §9.1 included).

| form | full HateMM / HCS / DeHate (ROC / PR) | not below (14 rows) | w/o Scope Read | w/o Duration Posterior | w/o EM Evidence | DeHate − best label-free ROC / PR | HateMM, HCS − current (ROC / PR) |
|---|---|---|---|---|---|---|---|
| F4 | 90.1/70.2, 75.6/71.4, 75.8/21.3 | 19 | 1 | 0 | 8 | +6.7 / +3.6 | +0.17/+0.96, +2.30/+2.42 |
| F5 | 90.0/69.9, 75.3/70.8, 76.7/23.3 | 15 | 2 | 0 | 8 | +7.5 / +5.6 | +0.10/+0.66, +2.07/+1.86 |
| F6 | 90.0/70.1, 76.0/72.0, 76.0/21.6 | 19 | 1 | 0 | 8 | +6.8 / +3.9 | +0.12/+0.84, +2.69/+3.00 |

- Gate 2 holds for all three (DeHate margins ≥ +2; HateMM and HCS not below current; within unchanged by construction).
- Gate 3 fails for all three. F5 has the fewest not-below cells (15; 8 of 14 rows below on every cell). Remaining in
  F5: Joint Query DeHate ROC / within; Verdict Turn HateMM ROC / PR, DeHate ROC / within; Window Evidence DeHate ROC;
  Fixed Mean Duration HateMM ROC / PR, DeHate within; Linear Evidence HateMM ROC / PR / within, DeHate ROC; Mean over
  Modalities DeHate ROC. These are reader and decoder rows; the Integrator does not change their within cells.
- With the taxonomy read, `w/o Scope Read` on HateClipSeg drops in ROC under F4–F6 (−0.1 / −0.1 / −0.4, intervals
  contain 0); under F2 / F3 it rose by 2.6 / 4.0.

## 8. Decoder D1–D4 (main-session brief 2026-11-15)

Base flags = the current method. Outputs `runs/20261114_exp_gates/decoder_d/`. Code: `d1_ties.py` (new) and two
options in `experiments/20260926_twolevel/twolevel_r2.py` (`--exact-logodds`, `--bma-kappa`; both off by default).

**D1, saturation and ties of `cell_prob`** (`d1_ties.py`; cell GT = majority of its 16 4-fps frames).

| decode | corpus | (a) at the 1e-12 clip | (b) tied within-video pairs | (c) GT-mixed among tied | min(p, 1−p) < 1e-6 / < 1e-9 |
|---|---|---|---|---|---|
| m2 | HateMM | 0 % | 0 % | — | 15.31 % / 1.86 % |
| m2 | HCS | 0 % | 0 % | — | 7.83 % / 0.58 % |
| m2 | DeHate | 0 % | 0 % | — | 4.58 % / 0.37 % |
| kap0.3 | all three | 0 % | 0 % | — | ≤ 0.03 % / 0 % |

No cell reaches the clip and no two cells of a video have the same value; a sizeable share is within 1e-6 of 0 or 1.

**D2, exact log-odds** (`--exact-logodds`: per chain the non-hate mass = sum of g over hate == 0 states, log miss
summed over chains, cell log-odds = log(−expm1(log miss)) − log miss, no clip). Flag off: `repro_off_1t` (one thread)
equals `decoder_scan/main/repro` value by value (score_curve and cell_prob, max |diff| 0); with four threads the
cell_prob differs by ≤ 8.8e-15 in 323 videos from summation order while score_curve stays identical. Flag on:
within m2 .7799 / .6866 / .6603, fixed 80 s .7781 / .6735 / .6761, linear evidence .7958 / .6530 / .6573, identical to
the clipped versions; on the main corpora no cell's exact log-odds differs from logit(clip(p)) by more than 1e-3. D2
does not change any number, so D3 uses the stored cell_prob.

**D3, logit average of decodes** (from `runs/20261114_exp_gates/decoder_avg/scan.json`; `d3.txt`).

**D4, per-video integration over the read noise scale** (`--bma-kappa "1,0.5,0.25"`: read variance s2/κ, uniform prior
over κ, joint (gap, hate, κ) model averaging per chain at inference, EM unchanged). Self-test against brute force over
(κ, chain, path) inside the existing self-test: max abs difference 4.26e-14 (all tests together). `--bma-kappa 1`
equals m2 value by value. Posterior mean κ per video (mean / median): HateMM visual .783 / .852, speech .776 / .827;
HCS visual .885 / .990, speech .876 / .975; DeHate visual .773 / .795, speech .756 / .756. The video-level log-odds
`lo` keeps the κ = 1 likelihood under V = 0 (it is not used by the m2 score).

| decoder | HateMM | HCS | DeHate |
|---|---|---|---|
| reference m2 | .7799 | .6866 | .6603 |
| reference fixed 80 s | .7781 | .6735 | .6761 |
| reference linear evidence | .7958 | .6530 | .6573 |
| D2 exact, m2 | .7799 | .6866 | .6603 |
| D2 exact, fixed 80 s | .7781 | .6735 | .6761 |
| D2 exact, linear | .7958 | .6530 | .6573 |
| D3 S1 m2 + kap0.3 | .7887 | .6797 | .6499 |
| D3 S2 m2 + fixed80 | .7862 | .6805 | .6760 |
| D3 S3 m2 + kap0.3 + fixed80 | .7902 | .6773 | .6676 |
| D3 S4 m2 + kap0.3 + fixed80 + plog | .7861 | .6848 | .6593 |
| D4 κ ∈ {1, .5, .25} | .7931 | .6780 | .6536 |

Members of D3 (within): m2 .7799 / .6866 / .6603; kap0.3 .8029 / .6484 / .6245; fixed80 .7781 / .6735 / .6761; plog
.7775 / .7002 / .6544. No D2–D4 setting is at least m2 on HCS and at least fixed 80 s on DeHate and at least linear
evidence on HateMM. D4 pooled ROC / PR: HateMM .8989 / .6927, HCS .7325 / .6893, DeHate .6940 / .1635.

## 9. Decoder rows: no duration posterior, no EM

`twolevel_r2.py --bma-weights prior` (default `posterior` = unchanged): at the final inference pass the per-video
average over the (gap, hate) grid uses the prior weights only (uniform in length, same grid and bounds); the pair
likelihoods do not enter the weights; EM unchanged (posterior weights). Runs `runs/20261114_exp_gates/decoder_d/
{main,dehate}/wprior` (decode.sh base flags + `--bma-weights prior`; one thread). Plumbing (`g2.txt`): the default
flag equals `decoder_scan/main/repro` value by value; `--bma-weights prior --bma-fixed 80` equals the stored fixed80
decodes value by value (main and DeHate).

| decoder | HateMM ROC / PR / within | HCS | DeHate |
|---|---|---|---|
| m2 (current) | .8989 / .6921 / .7799 | .7327 / .6895 / .6866 | .6940 / .1636 / .6603 |
| fixed 80 s | .8987 / .6921 / .7781 | .7324 / .6887 / .6735 | .6938 / .1633 / .6761 |
| prior weights only | .8985 / .6901 / .7316 | .7325 / .6890 / .6787 | .6939 / .1636 / .6476 |

With prior weights only, within drops on all three corpora against m2 (−.048 / −.008 / −.013) and is below fixed 80 s
on DeHate. As an ablation row ("w/o duration posterior") it would drop every within cell; pooled moves by at most .002.

### 9.1 No EM (planning-session item J1, 2026-11-15)

`twolevel_r2.py --em-iters N` (default 300 = unchanged). With `--em-iters 0` the decoder uses `init_params` as they
are (verdict means at the 10th / 90th percentile of z_video, corpus variance, pi .5; per emitter the window-read
percentiles as level means and the corpus variance; uniform start distribution) and no EM update; everything else is
m2, the per-video duration posterior included. Runs `runs/20261114_exp_gates/decoder_d/{main,dehate}/em0`
(decode.sh base flags + `--em-iters 0`; one thread). Plumbing (`decoder_d/j1.txt`): the default flag
(`decoder_d/main/j1_default`) equals `decoder_scan/main/repro` value by value (score_curve and cell_prob, max |diff| 0,
333 videos).

| decoder | HateMM ROC / PR / within | HCS | DeHate |
|---|---|---|---|
| m2 (current) | .8989 / .6921 / .7799 | .7327 / .6895 / .6866 | .6940 / .1636 / .6603 |
| linear evidence | .8990 / .6926 / .7958 | .7321 / .6888 / .6530 | .6940 / .1635 / .6573 |
| prior weights only | .8985 / .6901 / .7316 | .7325 / .6890 / .6787 | .6939 / .1636 / .6476 |
| no EM (`em0`) | .8989 / .6923 / .7857 | .7326 / .6895 / .6858 | .6940 / .1637 / .6715 |

Without EM, within is higher than m2 on HateMM (+.006) and DeHate (+.011) and .0008 lower on HCS; pooled moves by at
most .0002. As an ablation row ("w/o EM Evidence") it is not below the full method on most cells.
`gate_table.py` lists both rows (`w/o Duration Posterior` = `--bma-weights prior`, `w/o EM Evidence` = `--em-iters 0`)
apart from the 14, like `w/o Scope Read`, in every table from now on.
Current Integrator with these rows (`gate_table/tables/current_j1.md`, 1000 bootstrap draws): 14-row count unchanged
(24); cells not below: `w/o Duration Posterior` 5 of 9 (within drops −4.8 / −0.8 / −1.3, HCS and DeHate intervals
contain 0), `w/o EM Evidence` 8 of 9 (only HCS within drops, −0.1, interval contains 0).

## 10. Video-level ordinal read on our Reader's prefix (brief 4 H2–H3; written before the run)

Origin: Probe-VAD (arXiv 2609.17211) was excluded as a baseline by the user on 2026-11-15 ("不要这个 Probe-VAD 但是记下它的方法
它可以用来refine我们的method"); its probing (ten ordered "at least tau" Yes/No questions, monotone projection, mean) is
tested here on the whole video, inside our method. (The brief called this §7; §7 already holds F1–F5.)

**Mechanism hypothesis** (main session, brief 4): one Yes / No margin measures confidence at a single implicit
threshold; ten ordered "at least tau" probes give the expected severity. A graded severity read should order videos
better across a corpus than the verdict margin, and it gives offensive-but-not-protected-target content a middle value
instead of a No, which fits HateClipSeg's ground truth and DeHate's at once.

**Read** (`ordinal_read.py`, `launch/ordinal_lab1.sbatch`; uoa-lab1). Prefix = the adopted Reader's prefix, unchanged
(rules + 20 frames + transcript), same prefix cache as `z_video`. Per video, isolated branches, each on its own copy
of the cache, read one by one (no batching, so no batched-vs-single check is needed): `z_video` (verdict question,
checked against the stored verdicts with |diff| ≤ 1e-3, the tolerance of Slurm 2300, where the observed difference was
0.0), `z_target` (protected-group scope question) and ten ordinal branches. Question text, fixed by the main session
before the run, tau = 0.1, 0.2, …, 1.0 printed with one decimal:

> Use an ordered hatefulness scale from 0 to 1: 0 means ordinary non-hateful content; 0.5 means clearly concerning or
> plausibly hateful content; 1 means unmistakably severe, explicit hateful content. Intermediate values preserve this
> order.
>
> On this scale, is the hatefulness of this video at least {tau}?
>
> Answer "Yes" or "No".

Output `runs/20261114_exp_gates/ordinal_reads/ord_full_{main,dehate}/reads.jsonl`: `m_ord` (ten Yes − No margins),
`z_video`, `z_target`. Derived on CPU without labels: p_tau = sigmoid(m_tau); non-increasing projection over tau
(pool adjacent violators); s = mean of the ten projected values; z_ord = logit(clip(s, 1e-4, 1 − 1e-4)); s_raw = mean
of the unprojected p_tau; z_mean = mean of the ten margins. Cost if adopted: ten more Yes/No branches per video on the
cached prefix (about 1–2 s per video).

**Result (Slurm 2312, uoa-lab1 sc474397, 2026-11-15, 25 min: 333 videos in 312 s, 1341 in 1182 s; 0.9 s per video).**
No errors. `z_video` equals the stored verdicts on every video (max |diff| 0, 333 + 1341); `z_target` equals the
stored protected-group read (max |diff| 0). Before the projection the ten p_tau are out of order (some rise with tau)
in 89.8 % (HateMM), 97.5 % (HateClipSeg) and 88.4 % (DeHate) of videos.

H3 video-level terms (`video_terms_eval.py ordinal`, decoder m2; `runs/20261114_exp_gates/gate_table/h3_ordinal.md`),
video AUC / Spearman against the share of positive frames:

| term | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| zv | .935 / +.727 | .794 / +.441 | .734 / +.321 |
| mw | .885 / +.680 | .762 / +.616 | .692 / +.271 |
| z_target | .922 / +.708 | .771 / +.378 | .816 / +.435 |
| z_ord | .935 / +.730 | .808 / +.468 | .742 / +.330 |
| logit(s_raw) | .935 / +.730 | .808 / +.468 | .742 / +.330 |
| z_mean | .937 / +.731 | .810 / +.461 | .746 / +.336 |

Frame score K + λ crank, pooled ROC / PR:

| Integrator | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| O1 sd: mw + z_ord (λ 1) | 89.5 / 68.7 | 73.8 / 69.8 | 70.3 / 16.4 |
| O2 sd: z_ord | 86.4 / 60.9 | 70.1 / 65.5 | 67.4 / 15.1 |
| O3 sd: mw + z_ord + z_target | 89.5 / 68.6 | 72.1 / 68.0 | 74.0 / 19.5 |
| O4 raw: mw + .75 z_ord (λ 8) | 89.6 / 69.1 | 74.7 / 71.0 | 70.6 / 16.6 |
| O5 sd: mw + zv + z_ord | 89.6 / 68.8 | 72.8 / 68.7 | 69.7 / 16.5 |
| reference: current (raw mw + zv, λ 1) | 89.9 / 69.2 | 73.3 / 68.9 | 69.4 / 16.4 |

Full gate tables, O1 (`--form sd --a 0 --b 1 --lam 1 --scope ordinal`) and O4 (`--form raw --a 0 --b .75 --lam 8
--scope ordinal`), `runs/20261114_exp_gates/gate_table/tables/{O1,O4}.md` (1000 bootstrap draws; `w/o Global
Transcript` and `w/o Global Frames` use the full-prefix z_ord, borrowed, marked in the tables):

| form | not below (14 rows) | w/o Scope Read | w/o Duration Posterior | w/o EM Evidence | DeHate − best label-free ROC / PR | HateMM, HCS − current (ROC / PR) |
|---|---|---|---|---|---|---|
| O1 | 19 | 4 | 1 | 8 | +1.1 / −1.3 | −0.40/−0.52, +0.51/+0.90 |
| O4 | 19 | 3 | 0 | 8 | +1.5 / −1.1 | −0.25/−0.13, +1.44/+2.01 |

- The ordinal read orders videos about as well as the verdict on HateMM (.935 both), slightly better on HateClipSeg
  (.808 against .794), and below the protected-group read on DeHate (.742 against .816). The projection does not
  change the ranking (z_ord and logit(s_raw) give the same AUC and Spearman); the mean margin z_mean is within .004.
- Gate 2 fails for O1 and O4 (DeHate +1.1 / +1.5 ROC, −1.3 / −1.1 PR against the best label-free row; O1 also breaks
  the HateMM PR limit, −0.52). O3, which keeps z_target, reaches 74.0 / 19.5 on DeHate; its DeHate gain comes with
  z_target.
- Gate 3 fails for both (19 cells not below). `w/o Scope Read` (removing z_ord) is not below on HateClipSeg ROC / PR in
  both forms.

Test-read record (rule 10): read the ordinal reads, the GT arrays of the three corpora and the m2 decodes for the
tables above; no design was changed from them.
- Code fix found while running O1: `gate_table.py` `scales()` (forms `sd` and `cdf`) built its array from the whole
  `terms` tuple, which since §7 also holds the per-window arrays; it now takes the four video-level terms only. The
  `sd` / `cdf` tables of §5.2 were produced before the tuple was extended, so their numbers are unaffected.

## 11. Ablation tables with the no-EM decoder as the full method's decoder (brief 5 K2, 2026-11-15)

Reason (planning session): `w/o EM Evidence` (§9.1) is not below the full method on 8 of 9 cells; a simpler decoder
that is not worse should be the method's decoder. Development-selected (the decision rests on test tables).

`gate_table.py --decoder em0`: every row's decode uses `--em-iters 0` on top of the base flags. Reader rows decode
their own reads with `--em-iters 0`; decoder rows `w/o Temporal Coupling`, `w/ Shared Chain`, `w/ Fixed Mean
Duration`, `w/o Duration Posterior` add their flag on top of `--em-iters 0`; `w/ Linear Evidence` reuses the current
linear decode (`twolevel_r2.py` runs no EM under `--evidence linear`); `w/ ASR Windows` uses the composition on both
sides, so the decoder does not enter it. The apart row `w/o EM Evidence` is replaced by `w/ EM Evidence` = the current
m2 decode (`--em-iters 300`). The full decode is `decoder_d/{main,dehate}/em0` (§9.1). 20 new decodes in
`runs/20261114_exp_gates/gate_table/decodes/{main,dehate}/em0__*` (one run each). Code: `gate_table.py` (COMPONENTS
`em0`, ROW_FLAGS `em300`, EXISTING entries, `ROWS_EM0`). Run: uoa-lab1, CPU, `gate_table/logs/k2_run.sh`, logs
`gate_table/K2_F5.log`, `K2_F6.log`; 1000 paired bootstrap draws. Tables `gate_table/tables/em0__F5.md`,
`em0__F6.md` (and `.json`); every not-below cell with its interval in `gate_table/K2_notbelow.txt`.

Full method, absolute (ROC / PR / within):

| decoder, form | HateMM | HCS | DeHate | not below, 14 rows | rows below on every cell |
|---|---|---|---|---|---|
| em0, F5 | 90.0 / 70.0 / 78.6 | 75.3 / 70.8 / 68.6 | 76.7 / 23.4 / 67.2 | 12 | 8 of 14 |
| em0, F6 | 90.0 / 70.2 / 78.6 | 75.9 / 71.9 / 68.6 | 76.0 / 21.6 / 67.2 | 17 | 6 of 14 |
| m2, F5 (§7) | 90.0 / 69.9 / — | 75.3 / 70.8 / — | 76.7 / 23.3 / — | 15 | 8 of 14 |
| m2, F6 (§7) | 90.0 / 70.1 / — | 76.0 / 72.0 / — | 76.0 / 21.6 / — | 19 | 7 of 14 |

Not-below cells among the 14 rows (change in points, 95 % paired interval):
- F5 (12): Verdict Turn HateMM ROC +0.06, PR +0.27, within +0.02, DeHate ROC +0.20; Linear Evidence HateMM ROC +0.05,
  PR +0.15, within +1.01, DeHate ROC −0.03; Global Transcript HCS within +0.38; Joint Query DeHate ROC +0.50; Window
  Evidence DeHate ROC +0.04; Mean over Modalities DeHate ROC −0.03. Every interval contains 0.
- F6 (17): Verdict Turn HateMM ROC +0.10, PR +0.14, within +0.02, DeHate ROC +0.58, PR +0.44; Linear Evidence HateMM
  ROC +0.04, PR +0.11, within +1.01, DeHate ROC −0.03; Fixed Mean Duration HateMM PR −0.01; Global Transcript HCS
  within +0.38; Joint Query DeHate ROC +0.49; Window Evidence DeHate ROC +0.20; Window Aggregate DeHate ROC +0.64, PR
  +1.08; Mean over Modalities DeHate ROC +0.37, PR +0.72. Every interval contains 0.

Apart rows (not counted):
- `w/o Scope Read`: F5 2 not below (HCS ROC +0.09, PR +1.15); F6 1 (HCS PR +0.10). Intervals contain 0.
- `w/o Duration Posterior`: F5 1 (HCS PR +0.01); F6 1 (HCS PR −0.02). Within drops −3.9 / −0.5 / −1.2 in both.
- `w/ EM Evidence`: F5 4 not below (HCS ROC +0.02, PR +0.00, within +0.08; DeHate ROC −0.03); F6 6 (adds HateMM ROC
  −0.04, interval [−0.08, −0.01] excludes 0, and DeHate PR −0.03). DeHate within −1.1 (interval excludes 0) and HateMM
  within −0.6 in both: adding EM back lowers within.

Gate 2: DeHate full − best label-free row ROC / PR: F5 +7.5 (MARS) / +5.7 (naive row); F6 +6.9 / +3.9. Passes for both.
HateMM / HCS against the current method (ROC / PR / within; limits −0.5 pooled, −1.0 within): F5 +0.15 / +0.82 / +0.58,
+2.05 / +1.86 / −0.08; F6 +0.16 / +0.97 / +0.58, +2.67 / +2.99 / −0.08. Within limits for both.

Gate 3 fails for both (12 and 17 not-below cells; needs 0). With em0 instead of m2 the count drops by 3 (F5) and 2
(F6). Every not-below cell of the 14 rows has an interval containing 0; the only not-below interval that excludes 0
is in the apart row `w/ EM Evidence` (HateMM ROC under F6, −0.04).

Test-read record (rule 10): read the GT arrays of the three corpora, the em0 decodes and the two tables; no design was
changed from them.

## 12. Gate 1 passed (2026-11-16)

From 2026-11-16 the planning session does all work itself (user instruction). Baseline work:
`experiments/20261113_lf_baselines/README.md` §3.4.2 (Cog-VADU: 0 of 9 cells, no change) and §3.5 (Gaussian
smoothing of stored baseline scores). `runs/20261113_lf_baselines/gate/gate.md`: sets A and C **9 of 9, gate
passes**; sets B and D 3 of 9. The two cells that were missing come from "MARS + Gaussian smoothing, sigma 5 s"
(82.2 / 55.5 / 62.3, 69.5 / 65.4 / 61.2, 71.6 / 18.4 / 62.1): HateClipSeg within 61.24 vs 61.18, DeHate pooled PR
18.44 vs 17.69. Limits (stated in that README): the filter is our addition to the MARS port; the naive row with the
same filter (83.7 / 58.2 / 62.3, 73.1 / 69.7 / 63.6, 71.3 / 18.9 / 62.4) is above MARS + smoothing on 7 of 9 cells.

Consequence for gate 2: the best label-free DeHate row is now MARS + smoothing, pooled ROC 71.62 / PR 18.44, so
gate 2 needs DeHate ROC >= 73.62 and PR >= 20.44. The current method (69.40 / 16.36) is below that row.

## 13. Gates 2 and 3: analysis on cached reads and decodes (2026-11-16; CPU; development-selected)

Test-read log (rule 10): the reads and decodes of §5–§11 and the test GT, through `gate_table.py`'s loaders;
scripts `video_terms_auc.py`, `integrator_scan.py` (no bootstrap). Looked at: video-level AUC of each video term,
and the 17-row table for the forms below. Design changes: none adopted.

**13.1 Which video-level term orders videos** (`video_terms_auc.py`; video label = at least one hateful frame;
decoder em0):

| term | HateMM (85 of 215) | HateClipSeg (103 of 118) | DeHate (234 of 1151) |
|---|---|---|---|
| verdict `z_video` | .935 | .794 | .734 |
| protected-group scope read `z_target` | .922 | .771 | **.816** |
| HateClipSeg taxonomy read | — | .825 | — |
| mean window read | .885 | .762 | .692 |
| mean of the top quarter of the windows | .909 | .825 | .702 |
| A = mean of min(window read, scope read) | .911 | .755 | .758 |
| decoder's own video log-odds | .863 | .732 | .716 |
| scope + mean window (each / s.d.) | .924 | .781 | .777 |
| scope + A | .926 | .772 | .797 |
| scope + verdict + mean window | .932 | .798 | .770 |

On DeHate the scope read alone is the best term; every window aggregate and the verdict lower it. On HateClipSeg
almost every video is positive, so the video order matters little there and the pooled metrics follow the share of
hateful frames (correlation with that share: mean window read +.61, verdict +.46, scope +.41).

**13.2 The no-EM decoder's evidence is a linear evidence.** `init_params` puts the two emission means at the 10th
and 90th percentile of the normal-score reads with the corpus variance (1), so the log-likelihood ratio of a read y
is 2.56 y; `--evidence linear` uses y / s.d. = y. Under em0 the row `w/ Linear Evidence` therefore only changes the
slope from 2.56 to 1 (the same change as `--kappa 0.39`). It is not a different evidence model.

**13.3 Integrator scan** (`integrator_scan.py`, decoder em0, universal protected-group question on all corpora;
full = pooled ROC / PR / within; not-below over the 14 rows, all 9 cells / only the ROC and within cells that the
appendix table prints):

| video term (each / s.d.) + lambda x centred rank | HateMM | HateClipSeg | DeHate | not below |
|---|---|---|---|---|
| mean window + scope, lambda 1 (= F2 with em0) | 90.1 / 70.3 / 78.6 | 72.8 / 69.0 / 68.6 | 75.8 / 21.4 / 67.2 | 16 / 11 |
| mean window + scope, lambda 2 | 89.4 / 69.0 / 78.6 | 73.4 / 69.6 / 68.6 | 75.6 / 21.2 / 67.2 | 11 / 8 |
| mean window + scope, lambda 3.46 | 86.8 / 64.5 / 78.6 | 73.3 / 69.4 / 68.6 | 74.3 / 20.0 / 67.2 | 8 / 6 |
| A + scope, lambda 1 | 90.0 / 70.0 / 78.6 | 71.4 / 68.0 / 68.6 | 76.7 / 23.4 / 67.2 | 13 / 9 |
| A + scope, lambda 2 | 89.2 / 68.5 / 78.6 | 72.1 / 68.7 / 68.6 | 76.5 / 22.8 / 67.2 | 8 / 6 |
| A + scope, lambda 3.46 | 86.4 / 63.7 / 78.6 | 72.1 / 68.4 / 68.6 | 75.2 / 21.2 / 67.2 | 6 / 4 |
| top quarter + scope, lambda 1 | 89.6 / 68.5 / 78.6 | 70.3 / 66.3 / 68.6 | 75.4 / 21.7 / 67.2 | 18 / 11 |
| mean window + verdict + scope, lambda 1 | 90.3 / 70.1 / 78.6 | 71.9 / 67.8 / 68.6 | 73.8 / 20.2 / 67.2 | 16 / 10 |
| product: corpus rank of (mean window + scope) x within-video quantile | 79.6 / 58.0 / 78.6 | 71.7 / 67.7 / 68.6 | 70.2 / 17.6 / 67.2 | 11 / 7 |

- Gate 2 against the new best label-free row (71.62 / 18.44): mean window + scope, lambda 1 gives +4.2 / +3.0;
  A + scope +5.1 / +5.0. Both pass. HateClipSeg pooled ROC against the current method (73.27): −0.5 and −1.9; the
  A form needs the taxonomy read on HateClipSeg (F5, 75.3 / 70.8) to stay inside the limit.
- A larger within-video weight makes the reader and decoder rows move the pooled cells, but above lambda 2 it costs
  pooled ROC / PR on HateMM (−3.3 / −5.8 at 3.46). The product form loses 10 points on HateMM.
- Three within-video cells do not depend on the Integrator and stay above −0.05 under em0: `w/o Global Transcript`
  HateClipSeg +0.38, `w/o Verdict Turn` HateMM +0.02, `w/ Linear Evidence` HateMM +1.01. Under the EM decoder (m2)
  the set is different (`w/o Verdict Turn` DeHate +0.4, `w/ Joint Query` DeHate +1.4, `w/ Fixed Mean Duration`
  DeHate +1.6, `w/ Linear Evidence` HateMM +1.6). All of these have bootstrap intervals that contain 0 (§11).
- DeHate pooled ROC rises for rows that change only the window reads (`w/ Joint Query`, `w/o Verdict Turn`,
  `w/o Window Evidence`) and for `w/o Window Aggregate`, `w/ Mean over Modalities`, because on DeHate the mean
  window read lowers the video order that the scope read gives (13.1). The A form removes most of these cells.

**13.4 Reading.** No choice of decoder and Integrator among the forms above makes every cell of the 14 rows drop.
Two parts have no measurable effect and are the source of most remaining cells: the verdict turn (three-corpus
mean change when removed: pooled ROC −0.1, within −0.5, no cell significant on HateMM or DeHate) and EM (§9.1).
Removing them from the method removes their rows; the other reader rows must then be re-read without the verdict
turn (the six arms took 7.4 GPU-hours on RTX 5090s in `experiments/20261101_reader_ablation/README.md` §5).
Not started; it changes the method description, so it waits for the user.

## 14. Ablation table without the verdict turn and without EM (design written 2026-11-16, before the reads)

User, 2026-11-16: go ahead with removing the two parts and re-running; **do not change the default method yet**. So
this section evaluates a candidate configuration; `research-wiki/STATUS.md`, the paper and the method page keep the
current method (verdict turn, EM, `K = z_video + mean window read`).

**Configuration under test.**
- Reader: no verdict turn. Full-method reads = the stored `no_stance` reads; reader rows = the `ns_<arm>` reads of
  `experiments/20261101_reader_ablation/README.md` §8 (GPU, Slurm on uoa-lab1).
- Decoder: `--em-iters 0` (§9.1, §11).
- Integrator, three variants fixed before the reads are seen, each with λ = 1: (i) the current
  `K = z_video + mean window read` (`--form raw`); (ii) `F2`: `mw/σ + zT/σ`, the protected-group scope read on all
  three corpora; (iii) `F3`: `A/σ + zT/σ`. Any other form tried later is post-hoc and is labelled so.
- The scope read does not depend on the verdict turn (`scope_read.py` and `experiments/20260927_dvd/dvd_measure.py`
  read it as a branch on the prefix cache, `in_place=False`), so the stored scope reads are used.

**Rows** (`gate_table.py --family ns --decoder em0`). Two paper rows no longer exist in this configuration and are
listed apart instead of counted: `w/o Verdict Turn` becomes `w/ Verdict Turn` (the reads with the verdict turn), and
`w/ Linear Evidence` becomes `w/ Linear Evidence (slope 1)` (without EM the evidence is already linear, §13.2). That
leaves **12 counted rows**: Reader 6 (`ns_no_transcript`, `ns_no_prefix_frames`, `ns_no_isolation`, `ns_joint`,
`ns_no_window_evidence`, `ns_asr_windows` under the composition on both sides), Decoder 3, Integrator 3. Apart rows:
the two above, `w/ EM Evidence`, `w/o Scope Read`, `w/o Duration Posterior`.

**Criteria** (unchanged). Gate 3: every counted cell strictly below the full method on the three metrics (nine cells
per row; within-video cells of Integrator rows are equal by construction and exempt); the count on the pooled ROC and
within columns alone (the columns the paper's appendix table prints) is reported next to it. Gate 2: DeHate pooled
ROC ≥ 73.62 and pooled PR ≥ 20.44 (§12); HateMM / HateClipSeg against the current method within −0.5 pooled, −1.0
within.

**Order of work.** Decoder and Integrator rows need only the stored `no_stance` reads and are computed on CPU first
(`--skip-missing`); the reader rows follow when the reads are back.

### 14.1 Runs (2026-11-16 / 17; uoa-lab1 sc474397)

- Reads (GPU, Slurm jobs 2404–2407): `experiments/20261101_reader_ablation/README.md` §8.4. 0 errors; every in-job
  check 0.0; coverage 215 / 118 / 1341 lines per arm.
- Tables (CPU): `bash experiments/20261114_exp_gates/ns_tables.sh main`, then `… all`. Outputs
  `runs/20261114_exp_gates/gate_table/tables/ns_all__em0__{raw,F2,F3,F5}.{md,json}` (main corpora alone:
  `ns_main__em0__*`; decoder and Integrator rows before the reads: `ns_pre__em0__*`); logs `gate_table/logs/`;
  compositions `runs/20261101_reader_ablation/compose/ns_{fixed,asr}_{main,DeHate}`. 1000 paired bootstrap draws.
- Like-for-like tables that keep the verdict turn (same decoder, same Integrators):
  `gate_table.py --family current --decoder em0 --form {raw,F2,F3} --name em0__{raw,F2,F3}`; `em0__F5` is the table
  of §11.
- Counts over the stored tables: `table_summary.py` → `gate_table/tables/summary_ns_vs_verdict_turn.md`.

### 14.2 Results (development-selected; Qwen3-VL-8B; ROC / PR / within in points)

Full method, DeHate margin over the best label-free row (MARS + Gaussian smoothing, 71.62 / 18.44, §12), change
against the current method, and cells not below the full method.

| verdict turn | Integrator | HateMM | HateClipSeg | DeHate | DeHate − best label-free (ROC / PR) | HateMM − current | HateClipSeg − current | counted rows | cells not below | ROC and within only | on the 12 shared rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| removed | raw | 89.8 / 69.1 / 78.6 | 71.9 / 67.1 / 67.3 | 69.6 / 16.6 / 66.9 | −2.0 / −1.8 | −0.07 / −0.08 / +0.60 | −1.40 / −1.83 / −1.40 | 12 | 20 | 14 | 20 |
| removed | F2 | 90.2 / 70.4 / 78.6 | 72.0 / 67.7 / 67.3 | 76.2 / 21.5 / 66.9 | +4.6 / +3.0 | +0.29 / +1.24 / +0.60 | −1.27 / −1.29 / −1.40 | 12 | 13 | 9 | 13 |
| removed | F3 | 90.1 / 70.4 / 78.6 | 71.4 / 67.4 / 67.3 | 76.8 / 22.3 / 66.9 | +5.2 / +3.8 | +0.23 / +1.15 / +0.60 | −1.87 / −1.58 / −1.40 | 12 | 9 | 7 | 9 |
| removed | F5 (post-hoc here) | 90.1 / 70.4 / 78.6 | 74.2 / 69.3 / 67.3 | 76.8 / 22.3 / 66.9 | +5.2 / +3.8 | +0.23 / +1.15 / +0.60 | +0.93 / +0.39 / −1.40 | 12 | 9 | 7 | 9 |
| kept | raw | 89.9 / 69.2 / 78.6 | 73.3 / 68.9 / 68.6 | 69.4 / 16.4 / 67.2 | −2.2 / −2.1 | +0.00 / +0.03 / +0.58 | −0.00 / −0.00 / −0.08 | 14 | 22 | 13 | 14 |
| kept | F2 | 90.1 / 70.3 / 78.6 | 72.8 / 69.0 / 68.6 | 75.8 / 21.4 / 67.2 | +4.2 / +2.9 | +0.21 / +1.11 / +0.58 | −0.48 / +0.03 / −0.08 | 14 | 16 | 11 | 7 |
| kept | F3 | 90.0 / 70.0 / 78.6 | 71.4 / 68.0 / 68.6 | 76.7 / 23.4 / 67.2 | +5.1 / +4.9 | +0.15 / +0.82 / +0.58 | −1.89 / −0.95 / −0.08 | 14 | 13 | 11 | 5 |
| kept | F5 | 90.0 / 70.0 / 78.6 | 75.3 / 70.8 / 68.6 | 76.7 / 23.4 / 67.2 | +5.1 / +4.9 | +0.15 / +0.82 / +0.58 | +2.05 / +1.86 / −0.08 | 14 | 12 | 10 | 4 |

"Counted rows": 12 without the verdict turn (§14 design); 14 with it (the 12, `w/o Verdict Turn`, `w/ Linear Evidence`).
"On the 12 shared rows": the count restricted to the rows that exist in both families. The `em0__F5` table file of §11
was written before the smoothing rows of §12 existed; its DeHate margin above is recomputed against 71.62 / 18.44.

Cells not below, counted rows (every one has a paired bootstrap interval containing 0):

| verdict turn removed, F3 (9 cells) | verdict turn kept, F3 (13 cells) |
|---|---|
| `w/o Global Transcript`: HateClipSeg within +0.8 | `w/o Global Transcript`: HateClipSeg within +0.4 |
| `w/o Global Frames`: HateMM within +0.3 | `w/ Joint Query`: DeHate ROC +0.5 |
| `w/o Query Isolation`: HateClipSeg ROC +0.6, PR +0.2 | `w/o Verdict Turn`: HateMM +0.1 / +0.3 / +0.0, DeHate ROC +0.2 |
| `w/ Joint Query`: HateMM ROC +0.0, DeHate ROC +0.2, within +0.2 | `w/o Window Evidence`: DeHate ROC +0.0 |
| `w/ Fixed Mean Duration`: HateMM ROC −0.0, PR +0.0 | `w/ Linear Evidence`: HateMM +0.0 / +0.2 / +1.0, DeHate ROC −0.0 |
| | `w/ Mean over Modalities`: HateClipSeg ROC −0.0, DeHate ROC −0.0 |

The verdict turn itself, row `w/ Verdict Turn` of the family without it (change when the verdict turn is put back):
raw HateMM +0.1 / +0.1 / −0.0, HateClipSeg +1.4 / +1.8 / +1.3, DeHate −0.2 / −0.2 / +0.3; F3 HateMM −0.1 / −0.3 / −0.0,
HateClipSeg +0.1 / +0.7 / +1.3, DeHate −0.2 / +1.0 / +0.3. Only the HateClipSeg pooled ROC change under raw has an
interval that excludes 0.

Post-hoc reading (added 2026-11-17 after the tables were seen; not the gate's definition): on cells averaged over the
three corpora (one cell per metric per row), no counted row is not below the full method in six of the eight tables
(both families under F2 and F5, F3 without the verdict turn, and F3 with it on the 12 shared rows); the exceptions are
the two raw tables (4 and 5 cells) and `w/o Verdict Turn` under F3 (pooled ROC +0.07).

### 14.3 Reading

1. **Gate 3 is not passed in any configuration.** Removing the verdict turn and EM leaves 9 cells not below in 12 rows
   (F3 / F5). Keeping the verdict turn and removing EM leaves 5 (F3) or 4 (F5) on the same 12 rows, plus 4 in the
   `w/o Verdict Turn` row and 4 in the `w/ Linear Evidence` row.
2. **Removing the verdict turn does not help the table and costs HateClipSeg.** The three or four not-below cells of
   its own row disappear, and four to five new ones appear in the other reader rows (`w/o Global Frames`,
   `w/o Query Isolation`, `w/ Joint Query`). HateClipSeg falls by 1.3–1.4 within and 1.3–1.9 pooled against the current
   method under raw / F2 / F3 (my limit: −1.0 within, −0.5 pooled). §13.4's statement that the verdict turn has no
   measurable effect holds for HateMM and DeHate, not for HateClipSeg.
3. **Removing EM stands** (§9.1, §11): `w/ EM Evidence` is not below on 6–7 of 9 cells in the family without the
   verdict turn and 4–7 with it; the no-EM decoder changes HateMM within by +0.6 and HateClipSeg within by −0.08.
   Without EM the `w/ Linear Evidence` row is only a slope change and is not a component row.
4. **Gate 2 passes with the scope read in both families** (F2, F3, F5: DeHate +4.2 to +5.2 ROC, +2.9 to +4.9 PR over
   the best label-free row) and fails with the current Integrator in both. The HateClipSeg limits against the current
   method hold only when the verdict turn is kept and the Integrator is F5 (+2.05 / +1.86 / −0.08) or F2 (−0.48 /
   +0.03 / −0.08); F3 with the protected-group read on HateClipSeg is −1.89 on pooled ROC.
5. **What the remaining cells are.** All are within ±1 point with intervals containing 0, and they move between rows
   when the configuration changes (e.g. `w/o Query Isolation` on HateClipSeg pooled: −0.8 / −1.6 with the verdict
   turn, +0.6 / +0.2 without, F3). Four of the cells with the verdict turn kept are DeHate pooled ROC: under F3 / F5
   the DeHate video-level term is carried by the scope read (`w/o Scope Read` −5.7 / −7.1, `w/o Window Aggregate`
   −0.2 / −1.0), so a row that only changes window reads cannot move DeHate pooled ROC by more than noise.

Nothing is adopted. `research-wiki/STATUS.md`, the paper tables and the method page are unchanged (user instruction of
2026-11-16).

### 14.4 Test-read record (rule 10)

Read: the test metrics in the twelve `ns_*` tables, `em0__{raw,F2,F3,F5}` and `summary_ns_vs_verdict_turn.md` (all
under `runs/20261114_exp_gates/gate_table/tables/`); no per-video predictions were inspected. Found: §14.2–14.3.
Design changed: none. The Integrators raw, F2, F3 were fixed before the reads (§14); F5 and the corpus-averaged
reading are post-hoc.

## 15. Merging small components into one row (user request 2026-11-17; design written before the runs)

User, 2026-11-17: try merging small components and see whether a table with every cell below the full method can be
reached, with as few merges as possible. Exploratory; nothing is adopted (the default method is unchanged).

**Base.** Verdict turn kept, decoder without EM (`em0`), Integrators F3 (protected-group read on all corpora) and F5
(taxonomy read on HateClipSeg); both are reported because the HateClipSeg question is not decided. Rows: the 14 rows
minus `w/ Linear Evidence` (without EM it is a slope change, not a component, §13.2) = 13 rows. Rows with a not-below
cell in that base (§14.2): `w/o Global Transcript` (HateClipSeg within), `w/o Verdict Turn` (HateMM three cells,
DeHate ROC), `w/ Joint Query` (DeHate ROC), `w/o Window Evidence` (DeHate ROC), `w/ Mean over Modalities` (DeHate ROC;
HateClipSeg ROC under F3).

**What a merged row is.** Two parts are removed (or replaced) at the same time and the result is compared with the
same full method, on all three corpora with the same constants. A merged row replaces the rows of its parts.

**Rules, fixed before the numbers.**
1. Only rows with a not-below cell are merged; rows that are below on every cell stay as they are.
2. The two parts must belong to one design unit that can be named in a sentence of the method section.
3. Fewest merges; among equal counts, the set that needs no new reads.
4. No choice per corpus or per cell; the convention `not below` = change > −0.05 point stays.
5. Selection among merge sets is made on test tables: development-selected, and the table says so.

**Stage 1 (CPU, stored reads).** `merge_rows.py` (imports `gate_table.py`, same evaluator calls, 1000 paired
bootstrap draws):
- part + verdict turn, from the `ns_<arm>` reads of §14 compared with the full method that keeps the verdict turn:
  transcript, global frames, query isolation, joint query, window evidence;
- `w/ Mean over Modalities` folded into `w/o Window Aggregate` (the max over modalities is how the aggregate is
  built; the merged row is the existing `w/o Window Aggregate` row);
- window evidence + mean over modalities (older reader's reads with the mean in the aggregate).
A joint query has one read per window, so it already has one chain in the decoder and no max in the Integrator; the
`w/ Joint Query` row is unchanged by merging `w/ Shared Chain` or `w/ Mean over Modalities` into it.

**Stage 2 (GPU, only if stage 1 leaves not-below cells).** New reader arms that change two reader parts at once
(`read_arms.py`): joint query + no query isolation, joint query + older window inputs, no query isolation + older
window inputs. Each about 1.5–2 h on one RTX 5090 for the three corpora; which ones are run is decided from stage 1
and written here before submission.

### 15.1 Stage 1 results (2026-11-17, uoa-lab1, CPU; development-selected)

`merge_rows.py --form {F5,F3} --name merge__em0__{F5,F3}`; tables `gate_table/tables/merge__em0__{F5,F3}.md`. Change
against the full method (ROC / PR / within; `!` not below; under F5 unless stated):

| merged row | HateMM | HateClipSeg | DeHate | not below (F5 / F3) |
|---|---|---|---|---|
| w/o Global Transcript + Verdict Turn | −21.3 / −20.7 / −1.4 | −1.5 / −1.7 / −0.5 | −11.7 / −6.5 / −2.9 | 0 / 0 |
| w/o Global Frames + Verdict Turn | −2.3 / −2.2 / +0.4! | −8.5 / −4.7 / −2.9 | −1.6 / −2.8 / −2.2 | 1 / 1 |
| w/o Query Isolation + Verdict Turn | −1.7 / −4.4 / −5.4 | −0.6 / −0.3 / −2.1 | −0.6 / −3.1 / −3.4 | 0 / 1 (HateClipSeg ROC +0.5) |
| w/ Joint Query, w/o Verdict Turn | +0.1! / −0.3 / −1.1 | −4.6 / −4.3 / −7.9 | +0.4! / −1.6 / −0.1 | 2 / 2 |
| w/o Window Evidence + Verdict Turn | −0.3 / −0.2 / −3.1 | −3.0 / −3.8 / −6.2 | −0.0! / −2.4 / −3.4 | 1 / 1 |
| w/o Window Evidence, w/ Mean over Modalities | −0.5 / −1.4 / −2.5 | −1.9 / −3.5 / −4.9 | −0.1 / −2.1 / −2.4 | 0 / 0 |

- **Transcript + verdict turn** is below on every cell under both Integrators and removes the five not-below cells
  of its two rows. It is the only pairing of the verdict turn that does.
- **Mean over Modalities folded into Window Aggregate**: the remaining row `w/o Window Aggregate` is below on every
  cell (DeHate −0.2 / −1.0, intervals contain 0).
- After these two merges the not-below cells left are `w/ Joint Query` DeHate ROC +0.5 and `w/o Window Evidence`
  DeHate ROC +0.0. Pairing either with the verdict turn does not help, and the verdict turn is used above.
- `w/o Window Evidence, w/ Mean over Modalities` is below on every cell, but the two parts are in different modules
  (Reader input, Integrator aggregate) and do not form one design unit (rule 2); kept as a number, not proposed.

**Stage 2 decision (before submission).** All three combined reader arms are read (uoa-lab1, about 3.3 GPU-h;
`experiments/20261101_reader_ablation/README.md` §9): joint query + older window inputs (one merge would cover both
remaining rows), joint query + no query isolation, no query isolation + older window inputs. Candidate merge sets,
fixed now: (a) {transcript + verdict turn, joint query + window evidence, modalities into window aggregate} — three
merges, 10 rows; (b) {transcript + verdict turn, joint query + query isolation + window evidence as one row,
modalities into window aggregate} — four merges, 9 rows, needs a further arm and is read only if (a) fails.
