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

## 6. Results

Pending (filled in from `oracle_table.md` when the runs finish).
