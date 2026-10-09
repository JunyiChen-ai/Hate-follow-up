# Oracle test selection for the weakly supervised baselines (2026-10-09)

**The checkpoint (epoch) and the output branch of every weakly supervised baseline are selected on the TEST set,
using test labels.** User decision of 2026-10-09: give the baselines the most favourable selection, so that the
comparison cannot be called unfair to them. These numbers are an **upper bound for the baselines only** and must be
labelled as test-selected (oracle) wherever they appear. Our own method never reads test labels; nothing here changes
it. The baselines' training itself is unchanged: each still trains with video-level labels of the train split only.

Mechanism hypothesis: none (a measurement of how far the baselines can go under test-set selection).

## 1. Candidates

Methods: BERT + MIL, wav2vec2 + MIL, CLIP + MIL, VadCLIP, DSANet, AVadCLIP (audio-visual teacher), MultiHateLoc
(reimplementation), SAGE, CLARA; corpora HateMM (215), HateClipSeg (118), DeHate (1151), the exact test cohorts;
seeds 2025 / 234 / 3407.

A **candidate** is one (checkpoint, output branch) pair of one seed:
- **checkpoints**: the weights after every epoch of a retraining with the identical recipe, splits and seed (§2),
  plus the current run's selected checkpoint ("current"; its saved outputs, read only);
- **branches**: every output the method's own inference produces:

| method | epochs per run | branches (current table branch first) |
|---|---|---|
| BERT / wav2vec2 / CLIP + MIL | 50 | score_mil |
| VadCLIP | 10 | score_align, score_mlp |
| DSANet | 10 | score_mlp, score_refined (= score_mlp up to rounding with two classes), score_align |
| AVadCLIP | 10 | score_align, score_mlp |
| MultiHateLoc | as tuned: HateMM 100, HateClipSeg 30, DeHate 30 | score_fused, score_visual, score_audio, score_text, score_dms, score_union (0/1 top-K set) |
| SAGE | 30 | fusion (softmax of logits_fusion), text, audio, vision (each expert's own classifier) |
| CLARA | up to 50; the recipe's early stopping (patience 10 on val accuracy) ends the run | P(hateful) |

Hyper-parameters are the current runs' (MultiHateLoc: the val-tuned winner; no new search). A candidate whose test
curve fails the exact-cohort coverage check (e.g. an epoch after training diverged, with non-finite scores) is not
evaluated and is listed as invalid; there is no fallback.

## 2. How the per-epoch test predictions were obtained

No per-epoch checkpoint of the current runs was kept (only the selected or last model; VadCLIP / DSANet last-epoch
models on uoa-lab3), so **every method was retrained** with its identical recipe and seed. The training scripts gained
a default-off flag that saves each epoch's weights or scores (each change is recorded in the method README). Saving
or scoring draws no random number, so the retrained trajectory is the current one. Any difference comes only from GPU
non-determinism, and the summary measures it (§4, "does the retraining reproduce the current run?").

| method | retraining | per-epoch test scoring |
|---|---|---|
| MIL heads | `retrain.py mil` → `mil/train_mil.py` `run_seed` | hook after every epoch: `train_mil.predict` on the test cohort |
| VadCLIP, DSANet, AVadCLIP | `retrain.py fixed` → `weaksup_common/run_method.py` `train_cmd` + `--save-every-epoch` | each `model_eNN.pth` scored by the port's own `infer_cmd`, then deleted |
| MultiHateLoc | `retrain.py mhl` → `train_cmd` (tuned params) + `--save-every-epoch` | each `epoch_states/eNNN.pt` scored by `multihateloc.train.predict`, then deleted |
| SAGE | `sage/sage_run.py train --save-every-epoch` (`DETWIN_RUNS` = this run root) | `sage_epochs.py`: every epoch checkpoint of every seed, plus the current run's selected checkpoint, on every test window (inputs of `cmd_infer`), four heads |
| CLARA | `clara/clara_train.py --save-every-epoch` (`DETWIN_RUNS` = this run root) | after the normal run, each epoch's weights through the same `trainer.predict` path |

Per-epoch predictions are kept at the method's native rate (1 fps score arrays or 8-s window scores) in
`runs/20261008_baselines/oracle_test_selection/<method>/<Dataset>/seed<k>/epochs/`; the 4 fps curve is the
deterministic expansion of §3. No feature or KV tensor is saved.

## 3. Evaluation

`evaluate_candidates.py`: every candidate passes the coverage check of the current runs, then the canonical evaluator.
- 1 fps methods: `weaksup_common/common.py` `build_predictions`, i.e. ×4 repeat, last-value pad to ceil(4D), every
  cohort video finite on every GT frame.
- SAGE, CLARA: the checks of `detwin/common.py` `finalize`, i.e. one score per 8-s window, frame i takes window
  i // 32, every cohort video finite on every GT frame.
- Then `python -m src.eval.evaluate_four_datasets` on chunks of candidates, one method name per candidate
  (`<method>@<checkpoint>@<branch>`). `weaksup_common/common.py` `evaluate` checks that every candidate covers the
  whole cohort and frame pool.

Kept per seed: `candidates.json` (every candidate, its three metrics, the evaluator `metrics.json` it comes from) and
`eval/chunk_NNN/metrics.json`. The chunk's 4 fps `predictions.jsonl` is deleted after evaluation; it can be rebuilt
from the native-rate files. Check before any run: re-evaluating the current runs' saved outputs through this code
reproduces their `metrics.json` exactly (VadCLIP both branches, CLARA, SAGE on HateClipSeg: Δ = 0 on all three
metrics).

## 4. Summaries (`summarize.py`)

Per method × corpus, mean ± sd (n − 1) over the three seeds:
- **oracle B, single-checkpoint oracle (headline; user decision 2026-10-09)**: for each seed, the one candidate with
  the highest mean of pooled ROC-AUC, pooled PR-AUC and within-video ROC-AUC; its three metrics.
- oracle A, per-metric oracle (for the record): for each seed and each metric separately, the maximum over all
  candidates. The three numbers can come from three different candidates, so A is not one model.
- current: the existing val-selected / last-epoch / authors'-rule row (declared branch), copied unchanged from
  `runs/20261008_baselines/<method>/<Dataset>/seed<k>/metrics.json`.
- reproduction: the retrained run's candidate under the current rule (declared branch) minus the current number.

Outputs: `runs/20261008_baselines/oracle_test_selection/oracle_summary.json` and `oracle_table.md`.

## 5. How to run

```bash
# GPU, inside Slurm (one task list per job; a finished seed is kept on resubmission)
mkdir -p runs/20261008_baselines/oracle_test_selection/slurm
sbatch experiments/20261008_baselines/launch/oracle_lab1.sbatch mil:clip:hatemm fixed:vadclip:hatemm mhl:hatemm sage:HateMM clara:HateClipSeg ...
# SAGE needs the test window frames first (CPU): DETWIN_RUNS=$PWD/runs/20261008_baselines/oracle_test_selection \
#   .cache/envs/hv_cv2/bin/python experiments/20261008_baselines/sage/sage_run.py prep --dataset <DS> --workers 8
# CLARA HateMM was trained on CPU in the current run; same here (setsid nohup):
#   bash experiments/20261008_baselines/oracle_test_selection/run_tasks.sh clara:HateMM:cpu
# CPU, after the per-epoch scores exist
python experiments/20261008_baselines/oracle_test_selection/evaluate_candidates.py --workers 6
python experiments/20261008_baselines/oracle_test_selection/summarize.py
```

## 6. Results (state of 2026-10-09 21:10; SAGE pending)

TEST-selected upper bound for the baselines. Cells: pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro
ROC-AUC, mean over seeds 2025 / 234 / 3407. Transcribed from
`runs/20261008_baselines/oracle_test_selection/oracle_table.md` (built by `summarize.py` from each seed's
`candidates.json`, whose numbers are the evaluator's `metrics.json` outputs); sd, the chosen candidate per seed and
the reproduction check are in that file and in `oracle_summary.json`. "current" = the existing val / last / authors'
row, unchanged (CLARA DeHate: the run of Slurm 329, finished 2026-10-09 19:53, outputs copied from uoa-lab2).

| method | corpus | current (val / last) | oracle B (headline) | oracle A (per metric) | candidates per seed |
|---|---|---|---|---|---|
| BERT + MIL (text) | HateMM | 0.6362 / 0.3835 / 0.5253 | **0.6684 / 0.4302 / 0.5467** | 0.6696 / 0.4313 / 0.5509 | 51/51/51 |
| BERT + MIL (text) | HateClipSeg | 0.5442 / 0.5138 / 0.5044 | **0.5554 / 0.5231 / 0.5138** | 0.5558 / 0.5245 / 0.5142 | 51/51/51 |
| BERT + MIL (text) | DeHate | 0.6456 / 0.1730 / 0.5526 | **0.6533 / 0.1770 / 0.5538** | 0.6544 / 0.1785 / 0.5608 | 51/51/51 |
| wav2vec2 + MIL (audio) | HateMM | 0.6919 / 0.4422 / 0.5570 | **0.6847 / 0.4459 / 0.5750** | 0.6932 / 0.4461 / 0.5842 | 51/51/51 |
| wav2vec2 + MIL (audio) | HateClipSeg | 0.4973 / 0.4718 / 0.4775 | **0.5164 / 0.4878 / 0.5151** | 0.5164 / 0.4889 / 0.5152 | 51/51/51 |
| wav2vec2 + MIL (audio) | DeHate | 0.5418 / 0.0821 / 0.5293 | **0.5443 / 0.0811 / 0.5443** | 0.5458 / 0.0826 / 0.5456 | 51/51/51 |
| CLIP + MIL (visual) | HateMM | 0.7191 / 0.4585 / 0.5229 | **0.7220 / 0.4637 / 0.5226** | 0.7226 / 0.4660 / 0.5494 | 51/51/51 |
| CLIP + MIL (visual) | HateClipSeg | 0.5583 / 0.5182 / 0.5249 | **0.5589 / 0.5176 / 0.5260** | 0.5593 / 0.5187 / 0.5267 | 51/51/51 |
| CLIP + MIL (visual) | DeHate | 0.6418 / 0.1312 / 0.5202 | **0.6483 / 0.1291 / 0.5238** | 0.6511 / 0.1325 / 0.5367 | 51/51/51 |
| VadCLIP | HateMM | 0.6111 / 0.3588 / 0.4783 | **0.6813 / 0.3991 / 0.4796** | 0.6822 / 0.4034 / 0.4939 | 22/22/22 |
| VadCLIP | HateClipSeg | 0.5308 / 0.4825 / 0.5124 | **0.5302 / 0.4821 / 0.5136** | 0.5310 / 0.4826 / 0.5205 | 22/22/22 |
| VadCLIP | DeHate | 0.6035 / 0.1091 / 0.5120 | **0.6151 / 0.1164 / 0.4990** | 0.6279 / 0.1198 / 0.5245 | 22/22/22 |
| DSANet | HateMM | 0.7005 / 0.4136 / 0.5290 | **0.7032 / 0.4578 / 0.5722** | 0.7075 / 0.4629 / 0.5748 | 33/33/33 |
| DSANet | HateClipSeg | 0.5080 / 0.4622 / 0.5024 | **0.5372 / 0.5111 / 0.5076** | 0.5384 / 0.5113 / 0.5245 | 33/33/33 |
| DSANet | DeHate | 0.6404 / 0.1363 / 0.4776 | **0.6691 / 0.1434 / 0.5244** | 0.6709 / 0.1523 / 0.5624 | 33/33/33 |
| AVadCLIP (audio-visual) | HateMM | 0.6369 / 0.3503 / 0.5077 | **0.6627 / 0.4166 / 0.5116** | 0.6772 / 0.4346 / 0.5319 | 22/22/22 |
| AVadCLIP (audio-visual) | HateClipSeg | 0.4996 / 0.4596 / 0.5291 | **0.5580 / 0.5166 / 0.5342** | 0.5585 / 0.5169 / 0.5374 | 22/22/22 |
| AVadCLIP (audio-visual) | DeHate | 0.6104 / 0.1112 / 0.5343 | **0.6419 / 0.1386 / 0.5058** | 0.6425 / 0.1392 / 0.5355 | 22/22/22 |
| MultiHateLoc (reimpl.) | HateMM | 0.7441 / 0.4946 / 0.6228 | **0.7733 / 0.5361 / 0.6273** | 0.7832 / 0.5449 / 0.6340 | 606/606/606 |
| MultiHateLoc (reimpl.) | HateClipSeg | 0.5185 / 0.4787 / 0.4983 | **0.5650 / 0.5284 / 0.5189** | 0.5674 / 0.5296 / 0.5262 | 186/186/186 |
| MultiHateLoc (reimpl.) | DeHate | 0.6102 / 0.1289 / 0.5420 | **0.6416 / 0.1622 / 0.5408** | 0.6457 / 0.1633 / 0.5730 | 186/186/186 |
| SAGE (8-s windows) | HateMM | 0.7306 / 0.5035 / 0.6126 | **pending** | pending | -/-/- |
| SAGE (8-s windows) | HateClipSeg | 0.5453 / 0.5098 / 0.5156 | **pending** | pending | -/-/- |
| SAGE (8-s windows) | DeHate | 0.5970 / 0.1125 / 0.5698 | **pending** | pending | -/-/- |
| CLARA (8-s windows) | HateMM | 0.8722 / 0.6307 / 0.5417 | **0.8706 / 0.6283 / 0.5754** | 0.8754 / 0.6346 / 0.5767 | 16/25/18 |
| CLARA (8-s windows) | HateClipSeg | 0.5413 / 0.5165 / 0.5050 | **0.5820 / 0.5408 / 0.5139** | 0.5846 / 0.5424 / 0.5226 | 12/12/12 |
| CLARA (8-s windows) | DeHate | 0.7738 / 0.1943 / 0.4774 | **0.7766 / 0.2047 / 0.5050** | 0.7809 / 0.2086 / 0.5181 | 39/22/22 |

5182 candidates evaluated so far, none invalid. Oracle B picks a current-run checkpoint for only 2 of the 72 seed
runs (DSANet HateMM seed 234, AVadCLIP DeHate seed 3407); every other pick is a retrained epoch.

Reading the table:
- Oracle B is the selection the paper uses: one checkpoint and branch per seed, chosen on test labels. Oracle A is
  higher by construction, because its three metrics may come from three different candidates.
- Within one seed, oracle B's mean of the three metrics is never below the current mean, because the current
  checkpoint is itself a candidate. A single metric can still drop below the current number. For example,
  wav2vec2 + MIL on HateMM has pooled ROC-AUC .6847 under B against .6919 current.
- Branch changes chosen by B: VadCLIP HateMM uses score_mlp; DSANet uses score_align (except one HateMM seed);
  AVadCLIP uses score_mlp; MultiHateLoc HateMM uses score_dms. The per-seed choices are in `oracle_table.md`.

## 7. Run record

| step | host | job | wall time |
|---|---|---|---|
| MIL ×3, VadCLIP, DSANet, AVadCLIP, MultiHateLoc (all corpora), CLARA HateClipSeg | uoa-lab1 (sc474397) | Slurm 331, 18:20-19:55 | 1.58 GPU-h (its two SAGE tasks failed at start: the lab1 HF cache no longer held the SAGE encoders) |
| CLARA HateMM (CPU, as the current run) | uoa-lab1 | setsid nohup (`cpu/cpu_chain.sh`), 18:22-18:47 | CPU only |
| SAGE test-window frames HateMM / HateClipSeg (`sage_run.py prep`) | uoa-lab1 | same CPU chain | 2 min CPU |
| CLARA DeHate | uoa-lab2 (sc474399) | Slurm 339, 20:05-20:52 | 0.79 GPU-h |
| SAGE DeHate (train 3 seeds, score all epochs) | uoa-lab3 (sc474398) | Slurm 340, from 20:05 | running; about 5.8 h training + 2.5 h scoring expected (Slurm 332 failed at start: lab3 lacked `data/asr_whisper_large_v3/DeHate/all_splits_chunks.jsonl`, copied from uoa-lab2) |
| SAGE HateClipSeg, HateMM | uoa-lab1 | Slurm 341, from 20:52 | running; about 3 h expected |

Inputs added for these runs:
- `data/weaksup_1fps/{vit_b16_imagenet_1fps,vggish_1s}/dehate/`: the DeHate MultiHateLoc rows (addendum in that
  PROVENANCE.md). With them, the DeHate MultiHateLoc retraining reproduces the reused 2026-09-26 run exactly.
- uoa-lab3 `data/{sage_frames16,sage_frames16_win8,wav16k_mono}/DeHate/` copied from uoa-lab2, with a PROVENANCE.md
  each. The current SAGE DeHate checkpoints were copied to uoa-lab3 `runs/20261008_baselines/sage/DeHate/`.
- uoa-lab1 `.cache/hf/hub`: `MCG-NJU/videomae-base` and
  `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` copied from uoa-lab3. The lab1 copies had been removed
  on 2026-10-09 around 05:34. The SAGE HateClipSeg retraining with these encoders gives the current run's epoch-1
  loss and val macro-F1 to 5 decimals.

Reproduction of the current runs (retrained candidate under the current rule minus the current number, max over the
three metrics):
- MIL and MultiHateLoc: 0 on every seed.
- VadCLIP and AVadCLIP: ≤ .001.
- DSANet: ≤ .019 (GPU non-determinism compounds over 10 epochs).
- CLARA: .006 to .038 on every corpus, both on CPU (HateMM) and on GPU. Its bf16 training is not run-to-run
  deterministic, and the HateMM CPU retraining already differs at epoch 1.
- SAGE HateClipSeg: identical for the first 3 epochs, then drifts (GPU non-determinism); current seed 2025 selected
  epoch 13, the retraining epoch 12. The current SAGE DeHate seed 2025 ran with a different data loader (README of
  `../sage/`), so its trajectory cannot be reproduced.

In every case the current run's selected checkpoint is itself a candidate, so the oracle pool always contains it.

## 8. Remaining steps (when Slurm 340 and 341 finish)

```bash
cd ~/Hate-follow-up                    # uoa-lab1
rsync -a uoa-lab3:Hate-follow-up/runs/20261008_baselines/oracle_test_selection/ runs/20261008_baselines/oracle_test_selection/
source ~/miniconda3/bin/activate HateVideo
python experiments/20261008_baselines/oracle_test_selection/evaluate_candidates.py --workers 6
python experiments/20261008_baselines/oracle_test_selection/summarize.py
```
Check `runs/20261008_baselines/oracle_test_selection/sage/<DS>/score_epochs/check.json`. The current checkpoints'
fusion scores, re-scored by `sage_epochs.py`, should match the current window scores up to GPU rounding.
