# Baselines on the HateClipSeg split of the Retrieval-hate repository (2026-11-15)

Request (coordinator of `~/Retrieval-hate`, 2026-11-15): that repository's main table needs HateClipSeg cells for the
baselines of this repository, but its HateClipSeg split differs from ours: train 251 / val 63 / test 79 videos
(`hier_evidence_common.load_fixed_cohort('hateclipseg')` there) against p11's 237 / 39 / 118 here; the two test sets
share 23 videos. So the weakly supervised baselines are retrained on that split and the label-free baselines are run on
its 79 test videos, with the settings of the existing HateClipSeg runs. A naive VLM row with a non-Qwen model
(InternVL3.5-8B) is added on that repository's three test cohorts.

Mechanism hypothesis: none. These are baseline outputs for another repository's table. Nothing here is evaluated: the
canonical evaluator and `data/gt_4fps` know only our 118-video cohort, so every run stops after `predictions.jsonl`
(our schema, 4 fps `score_curve` of length ceil(4 · duration)); the Retrieval-hate repository evaluates them with its
own 1 fps evaluator (`experiments/20261115_hf_baselines/` there). No number of this directory enters our tables.

## 1. How the split is switched (no existing result is touched)

Every baseline script keeps its code path; a set of environment variables (all in `env.sh`; unset = behaviour as
before) replaces the split, the test cohort and the run root:

| variable | read by | effect |
|---|---|---|
| `WEAKSUP_ALT_DIR`, `WEAKSUP_RUNS` | `20261008_baselines/weaksup_common/common.py` (MIL heads, AVadCLIP) | splits, train/val labels and test durations from `data/rh_hcs_split/weaksup/`; outputs under the new run root; `evaluate()` stops after `predictions.jsonl`; `train_mil.py` / `run_method.py` skip the seed summary |
| `DETWIN_ALT_COHORT=1`, `DETWIN_SPLIT_DIR`, `DETWIN_RUNS`, `SAGE_DATA`, `CLARA_DATA` | `detwin/common.py`, `sage/sage_run.py`, `clara/*.py` (SAGE, CLARA) | split file `data/rh_hcs_split/video_splits/HateClipSeg.json`; frame / clip / embedding caches under `data/rh_hcs_split/`; `finalize()` stops after `predictions.jsonl` |
| `LF_ALT_COHORT_DIR`, `LF_BASELINES_RUNS`, `LF_RUN_ROOT` | `src/lf_baselines.py`, `20261008_baselines/lf_common.py`, the `OUT_ROOT`s of `prism_hate.py`, `videomind_hate.py`, `eventvad_4fps.py`, `EVENTS` of `eventvad_q3vl.py` | cohort list and manifest from `data/rh_hcs_split/cohort/`; outputs under the new run root; `finalize()` stops after `predictions.jsonl` |

`prepare.py` builds `data/rh_hcs_split/` (provenance there) from the split exported by the Retrieval-hate repository
(`runs/20261115_hf_baselines/hcs_split/split.json` there). Test durations are the ffprobe format durations of
`~/data/HateClipSeg/videos`, which equal the durations of our test manifest on the 23 shared test videos (max
difference 0). Train / val labels are that repository's; no test label and no frame label is read anywhere.

Label check (`data/rh_hcs_split/split_check.json` and a second check against `data/gt_4fps`): of the 314 train / val
videos, 219 have a label in our p11 train / val split and all 219 agree; the other 95 are test videos of p11, and
their label there (any positive GT frame) agrees for all 95. 0 disagreements.

Reuse of existing per-video caches and outputs (same scripts, same settings, same videos):
- SAGE 16-frame inputs, CLARA clips / frames / OCR / embeddings: videos whose role is the same in both splits (219
  train / val, 23 test) are linked from `data/`; the other 95 train / val videos (whole-video inputs) and 56 test videos
  (window inputs) are built by the same prep scripts. CLARA rationales exist for all 394 videos and are read as is.
- PRISM window embeddings and EventVAD event boundaries of the 23 shared test videos are copied from the existing
  runs (`reuse_existing.py`); the other 56 videos are computed here.
- Everything else is computed for all 79 videos.

## 2. What is run

| row | script, variant | setting |
|---|---|---|
| BERT + MIL, wav2vec2 + MIL, CLIP + MIL | `20261008_baselines/mil/train_mil.py` | as `mil/README.md`: 50 epochs, epoch chosen on val video AP, seeds 2025 / 234 / 3407 |
| AVadCLIP | `weaksup_common/run_method.py --method avadclip` | fixed XD preset, last of 10 epochs, val not read; the file holds `avadclip` (score_align, our headline) and `avadclip_mlp` |
| SAGE | `sage/sage_run.py` | as `sage/README.md`: epoch on val macro-F1, 8 s windows |
| CLARA | `clara/clara_{prep,ocr,embed,train}.py`, rationale mode `both` | as `clara/README.md`; the w/o-rationale check is not rerun |
| Qwen3-VL-8B naive | `20261113_lf_baselines/naive_prompt.py --arm vlm` | as its README §1 |
| MARS on Qwen3-VL-8B | `mars_q3vl.py`, read-out `prob` | as README §2 |
| EventVAD on Qwen3-VL-8B | `eventvad_4fps.py segment` + `eventvad_q3vl.py`, read-out `prob` | as README §2 |
| PRISM | `prism/prism_hate.py --av` | as `prism/README.md` (transcript variant) |
| VideoMind | `videomind/videomind_hate.py --av` | as `videomind/README.md` (transcript variant) |
| InternVL3.5-8B naive | `naive_prompt.py --arm vlm_internvl` (new arm) | the `vlm` arm with `OpenGVLab/InternVL3_5-8B-HF`: same 8 s windows, 1 fps frames (at most 8, decoded with longer side 640), Whisper words, question and fp32 Yes / No read-out through `src/mllm_judge.py` `Judge`; one 448 tile per frame (`FAMILY_IMAGE_KW`, as `experiments/20261101_backbones`). HateMM 215 and DeHate 1151 (our cohorts; the Retrieval-hate HateMM cohort is ours minus `hate_video_427`), HateClipSeg 79 |

## 3. How to run

```bash
# uoa-lab1, repository root
python experiments/20261115_rh_hcs_split/prepare.py                                   # HateVideo env
source experiments/20261115_rh_hcs_split/env.sh
.cache/envs/hv_cv2/bin/python experiments/20261008_baselines/sage/sage_run.py prep --dataset HateClipSeg --workers 6
.cache/envs/hv_cv2/bin/python experiments/20261008_baselines/clara/clara_prep.py --dataset HateClipSeg --workers 6
# OCR of the new clip folders: uoa-lab3 (paddleocr env), folders copied there and the ocr_text files copied back
python experiments/20261115_rh_hcs_split/reuse_existing.py prism; python experiments/20261115_rh_hcs_split/reuse_existing.py eventvad
sbatch experiments/20261115_rh_hcs_split/launch/weaksup_lab1.sbatch mil      # then: sage, clara
sbatch experiments/20261115_rh_hcs_split/launch/lf_lab1.sbatch naive:vlm:HateClipSeg mars:HateClipSeg naive:vlm_internvl:HateClipSeg naive:vlm_internvl:HateMM naive:vlm_internvl:DeHate
sbatch experiments/20261115_rh_hcs_split/launch/videomind_lab1.sbatch
sbatch experiments/20261115_rh_hcs_split/launch/eventvad_lab1.sbatch
sbatch experiments/20261115_rh_hcs_split/launch/prism_lab3.sbatch            # uoa-lab3
```

One GPU job at a time (the account's other jobs hold the second GPU of QoS `gpu2`). Outputs:
`runs/20261115_rh_hcs_split/<method>/HateClipSeg/...` and `runs/20261115_rh_hcs_split/lf/{naive,mars,eventvad}/...`.

## 4. Runs

All on 2026-11-15 / 16 (NZDT), RTX 5090 nodes, one GPU job at a time. `run.log` first line = host in every run directory.

| Slurm | host | what | state, elapsed | code |
|---|---|---|---|---|
| 2319 | uoa-lab1 (sc474397) | `weaksup_lab1.sbatch mil`: BERT / wav2vec2 / CLIP + MIL, AVadCLIP, 3 seeds each | completed, 0:14 | de3b361 |
| 2320 | uoa-lab1 | `weaksup_lab1.sbatch sage`: SAGE 3 seeds + window scores | completed, 0:35 | 54bdd9b |
| 2322 | uoa-lab1 | `weaksup_lab1.sbatch clara` | failed after 14 s: `openai/whisper-large-v3`, the ViT and the BERT encoder are no longer in uoa-lab1's `.cache/hf` (the job had waited 7.5 h for a GPU behind other jobs of the account) | 54bdd9b |
| 2333 | uoa-lab3 (sc474398) | `weaksup_lab3.sbatch clara`: CLARA embeddings (95 train / val videos, 56 test videos) + 3 seeds | completed, 0:10 | 338f7dd |
| 2334 | uoa-lab3 | `lf_lab3.sbatch naive:vlm:HateClipSeg naive:vlm_internvl:HateClipSeg mars:HateClipSeg naive:vlm_internvl:HateMM naive:vlm_internvl:DeHate` | completed, 2:26 (6 / 10 / 52 / 16 / 62 min) | 338f7dd |
| 2337 | uoa-lab3 | `prism_lab3.sbatch`: 54 of the 56 new videos embedded; 2 webm files failed in decord; the score stage stopped (2 F2 videos > 1 % of 79) | failed, 3:26 | 338f7dd |
| 2341 | uoa-lab3 | same file, retry of the 2 videos: same decord error | failed, 0:00:25 | 338f7dd |
| 2342 | uoa-lab3 | same file with `PRISM_TRANSCODE_ON_READ_ERROR=1`: the 2 videos re-encoded and embedded, score, predictions | completed, 0:10 | 1dcbd9d |
| 2343 | uoa-lab1 | `videomind_lab1.sbatch` | completed, 0:18 | 1dcbd9d |
| 2344 | uoa-lab1 | `eventvad_lab1.sbatch`: segmentation of 56 videos (161 min), then EventVAD on Qwen3-VL-8B (40 min) | completed, 3:21 | 1dcbd9d |

GPU time 10.7 h in all (the 54 PRISM videos embedded by the failed job 2337
were kept). CPU: SAGE and CLARA prep on uoa-lab1 (5 min each), OCR of the 2583 new clip folders on uoa-lab3 CPU
(30 min, 5945 images), copied back to uoa-lab1.

Facts from the logs:
- MIL heads: every seed scored 79 / 79 videos; tail padding 30 frames in 3 videos.
- SAGE: selected epochs 15 / 16 / 16, val macro-F1 .7237 for all three seeds (63 val videos, 8 not hateful).
- CLARA: embeddings train 251 / val 63 / 2389 test windows, none skipped; val accuracy of the selected checkpoint
  .873 / .873 / .905.
- Naive Qwen3-VL-8B and InternVL3.5-8B: 2389 windows on HateClipSeg, no failed video, no F2. InternVL3.5-8B on HateMM
  215 / 215 and DeHate 1151 / 1151 videos, coverage ok.
- MARS: 2389 windows, every window has a turn-4 label; 32 turns hit 2048 tokens; turn-1 JSON parse failures 24.
- EventVAD: 2895 events; 56 answers hit 2048 tokens, 53 without a stated score; no F2.
- VideoMind: 79 / 79, 395 verifier calls, no F2.
- PRISM: 79 / 79, 34,049 windows, no F2; `yt_hkNgzMgCA9E` and `yt_p_qvu46d1AU` transcoded.

## 5. Deviations from the existing HateClipSeg runs

- PRISM: two webm files pass `decord_ok` but fail in the full decord read (`threaded_decoder.cc:104`). The existing
  run had one such video (`yt_8sIY4W1hBQM`, F2). With 79 videos two F2 videos exceed the 1 % stop rule, so these two
  take the script's own decode-only H.264 re-encode (`PRISM_TRANSCODE_ON_READ_ERROR=1`, new, unset = unchanged). The
  embeddings of the 23 shared test videos are the existing run's (uoa-campus3 H200); the other 56 are from uoa-lab3.
- EventVAD: event boundaries of the 23 shared test videos are the existing run's (lab-server); the other 56 were
  segmented on uoa-lab1 with the same released config. All 79 videos were scored here.
- CLARA / SAGE: caches of role-unchanged videos reused (section 1); CLARA GPU stages on uoa-lab3 instead of uoa-lab1.
- VideoMind: `Qwen/Qwen2-VL-7B-Instruct` was downloaded again to uoa-lab1 (`runs/_setup_uoa-lab1/`; the snapshot
  directory has the same name as the one recorded in the existing run's config) and `VideoMind-7B` copied from
  uoa-lab2, whose copy of the base model is a dangling link.
- **LAVAD (`lavad_av`) was not run.** It needs the visual-only LAVAD chain for the 56 new videos first (BLIP-2
  captions, ImageBind cleaning, about 13k Llama-2 summaries and 13k scores, refinement; about 3 to 4 GPU-h) and then
  the AV scoring. BLIP-2 (`blip2-opt-6.7b-coco`) and Llama-2-13b-chat are no longer on any lab machine (uoa-lab1 holds
  a 540 K stub, uoa-lab2's `~/.cache/huggingface` was emptied), the `lavad_tf449` env exists only on uoa-lab2, whose GPU
  was held by other jobs of the account the whole time, and four more scripts (`exact_cohort.py`, `blip2_caption.py`,
  `lavad_chain.py`, `lavad_av.py`) would need the cohort / run-root override. Not improvised.
- The canonical evaluator was not run on anything here (section "Mechanism hypothesis").

Outputs were copied to uoa-lab1 with `rsync -a`. The Retrieval-hate repository collects them with
`experiments/20261115_hf_baselines/collect_sources.py` there.

Left on disk by this experiment: uoa-lab1 `.cache/hf/hub/models--Qwen--Qwen2-VL-7B-Instruct` (17 G) and
`models--yeliudev--VideoMind-7B` (4.4 G), `data/rh_hcs_split/` (about 4 G; the SAGE window frames of the 56 new test
videos are most of it), `runs/20261115_rh_hcs_split/eventvad/HateClipSeg/features` (CLIP / RAFT features);
uoa-lab3 `data/rh_hcs_split/` and `runs/20261115_rh_hcs_split/`.
