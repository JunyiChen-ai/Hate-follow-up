# Baseline run plan (2026-10-08)

This is a plan only. Nothing here has been launched. No GPU job, long CPU job, commit or push was made while writing
it. `README.md` in this directory belongs to the agent re-scoring the existing Retrieval-hate (RH) outputs on
HateMM/HateClipSeg; this file does not repeat or change that work.

Main table groups:
- **Weakly supervised:** trained with video-level labels of the target corpus's train split.
- **Label-free:** no target-corpus labels in training, adaptation or threshold choice. Rows that were pretrained on
  other corpora's labels (hate-speech classifiers, temporal-grounding or anomaly models) carry a footnote.

Abbreviations: HMM = HateMM, HCS = HateClipSeg, RH = `/home/jehc223/Retrieval-hate` (read only), HFU = this repo.
GPU-hours are RTX 5090 hours unless marked A100. Estimates scaled from logged runs are marked "scaled".

---

## 1. Rules every run follows

### 1.1 Test cohorts (user rule, 2026-10-08: exactly our test sets, no subsets)

| corpus | videos | ID list source | test hours |
|---|---|---|---|
| HateMM | 215 | `runs/20260926_twolevel/r6_bma/predictions.jsonl`, `dataset == HateMM` (= all 215 GT test IDs) | 8.1 h |
| HateClipSeg | 118 | same file, `dataset == HateClipSeg` (GT has 119; `yt_NzvfkIYS5Yg` is excluded, as for the current method, see `data/gt_4fps/PROVENANCE.md`) | 7.9 h |
| DeHate | 1151 | GT test IDs in `data/gt_4fps/DeHate.npz` (= the scored subset of the 1341-video test split; the current method's `runs/20260927_dehate_external/r6_bma/predictions.jsonl` covers all 1341) | 30.7 h (1341 videos: 35.2 h) |

Before any run, one CPU step (`experiments/20261008_baselines/cohort.py`, to be written) writes
`runs/20261008_baselines/cohort/{HateMM,HateClipSeg,DeHate}.txt` from those sources. Every run ends with
`coverage_check.py` (to be written), which fails the run unless:
- the predictions contain exactly the cohort IDs;
- no other IDs are present (for HCS, a stray `yt_NzvfkIYS5Yg` row is dropped before evaluation; otherwise the
  evaluator would score 119 videos);
- each row has `error: null` and a finite score for all `ceil(duration * 4)` frames;
- the evaluator's `n_videos_overlap` equals 215 / 118 / 1151.

This check is required because `src/eval/evaluate_four_datasets.py` silently drops videos with an error or an empty
curve. A missing video would otherwise shrink the cohort without notice.

**DeHate inference scope.**
- Methods whose output depends on the set of videos processed run on all 1341 test videos, exactly as the current
  method did. Example: LAVAD's caption index is built over the whole set.
- Cheap methods also run on all 1341.
- Per-video methods that are expensive (EventVAD, VADTree, LELA) run only on the 1151 cohort videos. Their per-video
  output does not depend on other videos, so the result is the same.
- Evaluation is always on the 1151.

### 1.2 Output format and the 4 fps grid

- Shared schema rows (`schema_version, method, dataset, video_id, duration, native_rate, score_curve, intervals,
  error, calls, seed, code_path, extra`), the same as `scripts/reproduction_baselines/zs_imagebind.py` writes.
- Curve length is `ceil(duration * 4)`. Duration comes from `data/omsl_v6_inputs/manifests/all_test.jsonl`
  (HMM/HCS) or `data/manifests/DeHate_test.jsonl`. The evaluator truncates to the GT length, as for the current method.
- Mapping from the native rate to the 4 fps grid:
  - 1 fps: repeat each value 4 times, then pad with the last value or cut to `ceil(duration * 4)`. This is the rule of
    `experiments/20260927_dehate_external/convert_weaksup.py`.
  - Coarser windows (0.5 fps, 0.1 fps, segments, events, intervals): frame `i` (centre `(i + 0.5) / 4` s) takes the
    value of the unit containing that centre. Frames past the last unit take the last unit's value.
  - Native 4 fps: used as is.
- Seeded methods run seeds 234 / 2025 / 3407, the DeHate weakly supervised setup. Each seed is evaluated separately
  and the table reports the seed mean (sd).
- Evaluator: only `src/eval/evaluate_four_datasets.py --datasets <corpus>`. Nothing is copied or changed.

### 1.3 Fallbacks (every frame of every cohort video gets a score)

| code | situation | fallback |
|---|---|---|
| F1 | Part of a video is not covered by the method's unit: no ASR segment there, silence, frames after the last decoded frame, windows the method skips | The method's own output for **empty input** at those frames. Text methods: the empty string, or `(no speech)` in an LLM prompt (as `experiments/20260910_spvl/spvl.py` already writes). Audio methods: digital silence of the same length. Feature-MIL heads: the zero feature row that `bert_sentence_1fps` already holds for silence. Tails: last-value padding (§1.2). |
| F2 | Whole video fails: decode error, missing stream after re-decode, OOM after one retry at half batch, model exception | Constant curve equal to the **median of that method's frame scores over the successfully scored videos of the same corpus and seed**. This uses test predictions only, no labels. Every F2 video is listed in `run.log` and in `extra.fallback`. If more than 1 % of a cohort needs F2, the run is stopped and the cause fixed instead. |
| F3 | Generative scorer gives no parseable number (EventVAD, LAVAD, URF, VADTree, LELA) | The method's declared rule is kept, and its rate is reported. EventVAD keeps its legacy `UNPARSED_FILL = 0.0` (`scripts/reproduction_baselines/eventvad/rasterize_and_eval.py`). LAVAD refusals: nearest valid frame in the same video, then F2. |
| F4 | Video without a transcript, for text-only methods (cohort counts with no timed text: HMM 14, HCS 3, DeHate 36) | F1 over the whole video: a constant curve at the method's empty-input score. |

Untimed Whisper chunks (`end` is null; the 2026-09-26 diagnosis found these were being dropped) are kept. Their end
is set to the next chunk's start or to the video duration.

### 1.4 Prompts and queries (kept consistent within each method family)

| family | methods | text used |
|---|---|---|
| Yes/No policy question, log-odds of "Yes" vs "No" | Qwen3-8B text, MLLM per-window (winonly), our method | `SYSTEM_MESSAGE` + "Based on this platform's hate speech policy, the following types of content are prohibited:" + the 9 `YOUTUBE_RULES` + `READER_PRAG`, copied verbatim from `experiments/20260910_spvl/spvl.py`. Per unit: "Consider only window i of n, from a s to b s of this video. Transcript in this window: <text>. Does THIS window contain content that violates any of the above rules? Answer "Yes" or "No"." For Qwen3-8B the unit is a Whisper segment and only "window" becomes "segment". |
| Natural-language moment query | UniTime, SeViLA, T3AL main label, LaGoVAD | RH `MAIN_QUERY` "hateful content targeting a person or group" (SeViLA wraps it as "Does this frame contain hateful content targeting a person or group?"). Kept unchanged, so the DeHate numbers match the RH HMM/HCS setup. |
| Two text anchors, similarity difference | ZS-CLIP, ZS-ImageBind (vision and audio), LAION-CLAP | ZS-CLIP: "a normal video frame" / "a hateful video frame" (RH). ImageBind: "normal" / "hateful" (`data/assets/imagebind/text_embeddings_normal_hateful.npy`). CLAP (new): "a normal audio clip" / "a hateful audio clip", mirroring ZS-CLIP's template. |
| Method-native anomaly prompts | LAVAD, EventVAD, URF-HVAA, Vad-R1 | Each paper's own prompt, verbatim or as reconstructed in `scripts/reproduction_baselines/eventvad/prompt.py`. Rewording them for hate needs owner approval (decision D7). |
| Classifier | ToxiGen-RoBERTa / HateBERT / CardiffNLP | No prompt. The score is the logit of the hate class. |
| Must be rewritten | VADTree (its prompts carry UCF/XD category priors) | Replace the anomaly-category list with the 9 `YOUTUBE_RULES` (decision D7). |

### 1.5 Other constraints

- **Code location.**
  - RH campaign scripts hard-code `ROOT=/home/jehc223/Retrieval-hate` and write into RH. They are ported, never run
    in place. The port goes to `scripts/reproduction_baselines/rh_campaign/`, the baseline-reproduction location in
    CLAUDE.md. Changes:
    - `ROOT` points to HFU;
    - the video path comes from the manifest;
    - the gold comes from `data/gt_4fps`;
    - outputs go to `runs/20261008_baselines/<method>/<corpus>/`;
    - new caches go to `data/<type>/` with a `PROVENANCE.md`.
  - Upstream repos are cloned into HFU `third_party/` on the target machine at the RH-recorded commit, and the RH
    patch is applied.
  - Weights are symlinked into HFU `.cache/`, never copied.
  - RH venvs (`_venv/av2a`, `_venv/sevila`) are rebuilt under HFU `.cache/envs/`.
- **Hash ban.**
  - RH `run_t3al.py` derives per-video seeds from `zlib.crc32(video_id)`. The port replaces this with
    `seed + index in the sorted cohort list`.
  - Vad-R1's script checks pinned prompt digests; the port removes that check.
  - No digest is computed or recorded anywhere.
- **Slurm.**
  - Every GPU stage is an sbatch file in `experiments/20261008_baselines/launch/`, submitted on the target machine
    to its local partition (`local-sc474397/8/9`, `local-sc448960`, `--gres=gpu:1`, 4 CPU / 32G, mem ≤ 56000 MB).
  - Jobs on one node queue in its partition without `--dependency`.
  - Pure CPU steps may use `setsid nohup`.
- **Labels.** Label-free rows read no target label at any point. Weakly supervised rows read train/val video labels
  only; test labels go only to the evaluator.
- **No cross-imports.** The plan directory does not import from other experiment directories. Running another
  experiment's script unchanged with a new `--exp-id` is allowed, as `20260927_dehate_external` did with
  `til_measure.py`.

### 1.6 Machine snapshot (2026-10-08 00:52 NZDT; `bash scripts/check_machines.sh`, `sinfo`, `squeue`)

| machine | state | relevant assets |
|---|---|---|
| uoa-lab1 (sc474397) | Slurm 282 `m1-stom-ctl-unif` running (another agent) | EventVAD and Vad-R1 weights and venv (`~/data/checkpoints/{videollama2,raft,vad_r1}`, `~/venvs/SafetyContradiction`); CoCa, Qwen2.5-VL-7B, CLIP-L/336, CLIP-B/16 in HF cache; DeHate test videos; all 1 fps weak-sup feature caches (HMM/HCS in `runs/legacy_1fps/*/reproduction/features/`, DeHate in RH `results/reproduction/features/*/dehate`); RH `data/AV2A_wav` (16 kHz wav for all splits of all three corpora); RH `data/frames_1fps/DeHate` |
| uoa-lab2 (sc474399) | idle; commit f384697, behind lab1 (42c8622), needs `git pull` | DeHate all splits (train 4680, val 668, test 1341; 31G); RH `third_party/{lavad,AV2A,SeViLA,UniTime,URF-HVAA,LaGoVAD-PreVAD,MULDE,CLAP,T3AL,VADTree,fed_wsvad}`, `_ckpt`, `_venv`; HF cache: Qwen3-8B, InternVL3_5-8B-HF, Qwen2.5-VL-7B, Qwen2-VL-7B + UniTime LoRA, BLIP-2 opt-6.7b-coco, Llama-2-13b-chat, Llama-3.1-8B, VideoLLaMA3-7B, LanguageBind, flan-t5-xl, cardiffnlp hate classifier; conda HateVLM (transformers 5.15.1); 564G free |
| uoa-lab3 (sc474398) | Slurm 281 `m1-stom-ctl-dual` running (another agent) | Qwen2.5-VL / Qwen3-VL caches; HMM/HCS videos; no DeHate videos; 1.1T free |
| lab-server (sc448960) | idle, shared; commit f384697 | 125 GB RAM; HateVLM env; HMM/HCS videos; `tomh/toxigen_hatebert`, `facebook/wav2vec2-base-960h`; no DeHate; 700G free |
| uoa-campus1/2/3 | ControlMaster sockets dead | need the user to run `ssh uoa-campusN true` with a token; at most 2 jobs across the three |

---

## 2. Problems found while planning (fix or decide before numbers enter the table)

1. **The HCS MultiHateLoc number in the main table is leaked.**
   - `runs/20260829_omsl_v6/multihateloc_frozen_current4fps_v1_metrics.json` comes from the lab1 checkpoint
     `runs/legacy_1fps/lab1/reproduction/baselines/multihateloc_reimpl/hateclipseg/model.pt`.
   - That checkpoint was trained on the legacy lab1 HCS train list (315 videos). The list contains **95 of our 118
     HCS test videos**; I checked this against `runs/legacy_1fps/lab1/reproduction/splits/hateclipseg_train.txt`.
   - Fix: retrain on the split that defines our HCS test set, `RH data/gt/HateClipSeg/p11_split.json` (train 237 /
     val 39 / test 119, seed 0). See W6.
2. **Every legacy HCS weakly supervised run used the wrong split.**
   - lab2 official-val used 251 / 63 / 79, and lab1 used 315 / 79.
   - Only 23 of our 118 HCS test videos are in either legacy test list; 95 are in legacy train/val.
   - No legacy HCS weakly supervised output can be reused. All are retrained on p11.
   - p11 val is small: 39 videos, 5 of them normal under the offensive-union label. That makes validation-based
     selection noisy.
3. **HMM legacy weakly supervised outputs miss one video.**
   - `runs/legacy_1fps/lab2/reproduction/official_val/final/<method>/hatemm/seed_*/scores.jsonl` has 214 of 215
     videos; `hate_video_427` is missing.
   - Its features exist and the `model.pth` files exist, so all 215 are re-inferred from the saved checkpoints.
4. **T3AL HMM/HCS do not cover our test sets.**
   - `runs/20260829_omsl_v6/t3al_anchor_s20250819_metrics.json` predicted only 207 of 215 HMM and 110 of 118 HCS
     videos.
   - Its prediction file (`results/idea_discovery/less/t3al_anchor_s20250819.jsonl`) no longer exists, and the RH
     caption cache `idea-stage/repro_t3al/captions/*` is empty.
   - The original per-video seeds were derived with crc32 and cannot be reproduced under the hash ban.
   - T3AL is therefore rerun on all three corpora (L7), not only DeHate.
5. **MultiHateLoc uses different branches across corpora.**
   - The main-table HMM/HCS rows are the DMS branch, single seed, default hyperparameters, linear 1→4 fps
     interpolation.
   - DeHate is `score_fused`, 5-trial Optuna + 3 seeds, ×4 repeat.
   - Plan: use one setup on all three corpora (W6, decision D3).
6. **MULDE and CLAP (CVPR'24) are not label-free.**
   - MULDE fits only on train videos labelled non-hateful and picks its configuration on val frame labels.
   - CLAP (CVPR'24) trains on an unlabelled train pool but picks its FedAvg round count on val frame ROC.
   - They cannot sit in the Label-free group as they were run (decision D6).
7. **Some label-free rows were pretrained on other labelled data.** Footnote them like the hate classifiers:
   - SeViLA, UniTime and LaGoVAD: other temporal-grounding labels.
   - Vad-R1: anomaly data.
   - T3AL's preset `D_anet` was chosen on HMM/HCS/MHC val frame PR-AUC in 2026-08. It is kept fixed and is not
     re-chosen on DeHate.
8. **HFU baseline code points at a missing directory.**
   - `scripts/reproduction_baselines/hate_common/data.py` (and `multihateloc/data.py`, the extractors) read
     `REPO/results/reproduction/...`, which no longer exists. The files moved to
     `runs/legacy_1fps/lab{1,2}/reproduction/` (`LEGACY_PATHS.md`).
   - HFU's `hate_common` has no `dehate` corpus; RH's version does.
   - Fed-WSVAD upstream is not in HFU `third_party/`; it is only in RH `third_party/fed_wsvad`.
   - These are setup items for W4–W6.

---

## 3. Weakly supervised runs

Shared inputs, all checked present:
- **Train/val splits:**
  - HMM: RH `results/reproduction/splits/hatemm_{train,val,test}.txt`, 744 / 109 / 215. These equal
    `~/data/HateMM/splits/{train_clean,valid,test_clean}.csv`. Train hate 298 / non-hate 446.
  - HCS: p11, 237 / 39 / 119 (118 scored). Label = offensive union, as in the HCS GT: train 207 offensive /
    30 normal.
  - DeHate: `~/data/DeHate/DeHate_labels.csv` `Split`, train 4680 (1484 hate) / val 668 (212) / test 1341. The
    train/val videos are on uoa-lab2 only.
- **Train-split video hours:** HMM 31.5 + 3.8 val; HCS 15.6 + 2.5 val; DeHate 126.5 + 18.1 val.
- **1 fps features covering train, val and test of all three corpora:**
  - `clip_b16_1fps` (512-d);
  - `bert_sentence_1fps` (768-d BERT CLS of each Whisper segment, repeated over the seconds it covers, zeros in
    silence);
  - `vggish_1s` (128-d);
  - `vit_b16_imagenet_1fps`;
  - `i3d_rgb_5crop`.
  - HMM/HCS: `runs/legacy_1fps/lab2/reproduction/features/`. DeHate: RH `results/reproduction/features/*/dehate`
    (lab1). Copy the DeHate ones used (CLIP, BERT, VGGish; 3.5 GB) into `data/weaksup_1fps/<feature>/DeHate/` with a
    `PROVENANCE.md`.
- **16 kHz mono wav for all splits of all three corpora:** RH `data/AV2A_wav/{HateMM,HateClipSeg,DeHate}` on lab1 and
  lab2 (HMM val misses the 2 videos that `validation_clean` already drops).

### W1–W3. One MIL head on three single-modality features

**Proposed MIL head** (decision D1):
- **Loss:** top-k MIL with binary cross-entropy (Wu et al., ECCV 2020 XD-Violence; also VadCLIP's classification
  loss). Video score = mean of the top `k = floor(T/16) + 1` instance scores; BCE against the video label.
- Why this over the Sultani et al. (CVPR 2018) ranking loss:
  - it works on variable-length 1 fps bags without resampling to 32 segments. DeHate's median video is 64 s, so 32
    segments would be about 2 s each;
  - it uses positive and negative videos independently, so no 30+30 pair batches are needed;
  - it has no feature-magnitude or BatchNorm tricks that would favour one modality.
- **Sultani as a one-flag alternative:** hinge `1 − max f(pos) + max f(neg)` + 8e-5 smoothness + 8e-5 sparsity.
- **Head (same for all modalities):** Sultani's MLP `d → 512 → 32 → 1`, ReLU, dropout 0.6, sigmoid.
- **Training:**
  - Adam, lr 1e-4, weight decay 5e-4, batch 64 videos with a padding mask, at most 50 epochs.
  - Epoch chosen by val video AP (top-k mean).
  - Seeds 234 / 2025 / 3407.
  - No hyperparameter search, so every modality and corpus gets the same budget.
- **Inference:** instance score per 1 s row → ×4 → pad (§1.2). No fallback is needed: every second has a feature row
  (zeros in silence for text).

| run | instance feature | ready? |
|---|---|---|
| W1 BERT+MIL (text) | `bert_sentence_1fps`: bert-base-uncased CLS of each Whisper large-v3 segment, the MultiHateLoc text branch input | yes, all splits, all corpora |
| W2 CLIP+MIL (visual) | `clip_b16_1fps` (`image_embeds`, 512-d) | yes, all splits, all corpora |
| W3 audio+MIL | (a) **wav2vec2-base** (`facebook/wav2vec2-base`, pretrained not ASR-finetuned, last layer, mean over each 1 s, 768-d), or (b) MFCC (40 coeffs + Δ, mean/std per 1 s, 160-d, CPU), or (c) existing `vggish_1s` (no extraction) | (a) and (b) need extraction; (c) ready |

**W3 extraction (option a):**
- Audio hours, all splits: HMM 43.5 h, HCS 26 h, DeHate 180 h.
- The wavs already exist.
- About 1 GPU-h including I/O on lab1 (wav2vec2-base, under 2 GB). The model is a 360 MB download.
- Output: `data/wav2vec2_base_1s/<corpus>/` + `PROVENANCE.md`.
- MFCC: about 2 h of CPU on lab1.

**Cost.** Training is about 5–10 min per corpus × modality × seed on a 5090 (CPU is also feasible). Total for
3 modalities × 3 corpora × 3 seeds is about **1.5 GPU-h**. Memory is under 4 GB.

**Machine:** uoa-lab1, where all features and wavs are, or lab3 after an rsync of about 10 GB of features. One job
per corpus.

**Risks:**
- Text instances repeat a long Whisper segment over many seconds. Segments can exceed 100 s on HMM/HCS, so the text
  row localizes only as finely as Whisper segments.
- The HMM BERT features use the legacy lab2 `hatemm_all` transcripts. Those may differ slightly from
  `data/asr_whisper_large_v3/HateMM` (test), which W1 does not use.

### W4. MACIL-SD, DSANet, Fed-WSVAD on HMM and HCS (reuse the DeHate setup)

**DeHate setup reused** (RH `experiments/20260926_dehate_external/launch/baseline.sh`):
- `scripts/reproduction_baselines/tune_official_val.py --trials 5`: Optuna, sampler seed 234, objective val video AP.
- Then `confirm_baselines.py`: retrain the winner at seeds 234 / 2025 / 3407 and score test once.
- Branches:
  - MACIL-SD `score_av`;
  - DSANet `score_mlp`;
  - Fed-WSVAD 3-client `score_align`.
- 4 fps by `convert_weaksup.py`'s ×4 rule.
- Features:
  - MACIL-SD: I3D 5-crop + VGGish;
  - DSANet and Fed-WSVAD: CLIP B/16 1 fps.

**HMM:**
- Reuse `runs/legacy_1fps/lab2/reproduction/official_val/final/{macilsd,dsanet,fed_wsvad_3client}/hatemm/seed_*/model.pth`.
  These used the same tune-then-3-seed protocol, but with 40 trials instead of 5.
- Re-infer all 215 test videos (this adds `hate_video_427`) and convert to 4 fps.
- Cost: about 0.2 GPU-h total.
- Alternative for strict parity with DeHate (decision D4): retrain with 5 trials, about 2 h per method.

**HCS:** retrain on p11 with the DeHate protocol. Setup work:
- p11 split files in `runs/20261008_baselines/splits/hateclipseg_{train,val,test}.txt`; test = our 118;
- the `hate_common` path override;
- clone `third_party/fed_wsvad` at RH commit 287747f (and install `ftfy`, whose absence broke the first DeHate
  launch).

Scaled cost: DeHate took 2.4 h, 2.4 h and 3.8 h for 455k train rows, and HCS train has 56k rows. So about
**0.5–0.8 GPU-h per method, about 2 GPU-h for the three**. Memory is under 16 GB.

**Machine:** uoa-lab1 (features local, `~/venvs/SafetyContradiction` and HateVideo envs present).

**Fallback:** F2 only. Every cohort video has features: I3D is missing only for DeHate `8bAfN6vXoIZp`, and DeHate
MACIL-SD is already done.

**Risks:**
- p11 val has 5 normal videos, so the Optuna objective is noisy.
- The HCS label is offensive-union. That is the HCS GT definition, which is an open protocol question in STATUS.

### W5. VadCLIP on all three corpora

- **Code:** HFU `scripts/reproduction_baselines/vadclip/{model,option,train,infer}.py` with
  `train_vadclip_hatemm.py` / `test_vadclip_hatemm.py`. Upstream is `third_party/VadCLIP` (github.com/nwpu-zxr/VadCLIP
  @c41067f, Apache-2.0). It is already supported by `tune_official_val.py`.
- **Prompts:** "normal content" / "hateful content" (`hate_common.PROMPT_TEXT`).
- **Branch:** `score_align`, VadCLIP's fine-grained alignment branch. `score_mlp` is reported alongside as a check.
- **Features:** `clip_b16_1fps`, present for all splits of all three corpora.
- **Upstream checkpoint selection:** the upstream repo picks the checkpoint by test AP every epoch. The HFU port
  already selects on val; this must be kept.

| corpus | how | GPU-h |
|---|---|---|
| HMM | reuse `official_val/final/vadclip/hatemm/seed_*/model.pth`; re-infer 215 | 0.1 |
| HCS | retrain on p11, 5 trials + 3 seeds | about 0.6 |
| DeHate | 5 trials + 3 seeds on train/val, same as the other DeHate baselines | about 2.5 (scaled from DSANet's 2 h 27 m on DeHate) |

**Machine:** uoa-lab1. **Memory:** under 12 GB. **Fallback:** F2 only.

**Risk:** DeHate support must be added to HFU `hate_common` (copy the RH version's `dehate` entries).

### W6. MultiHateLoc (listed as done; it is not, see §2.1 and §2.5)

- **DeHate:** done. `runs/20260927_dehate_external/weaksup/metrics.json`, `score_fused`, 3 seeds.
- **HMM:** the current row covers 215 videos but uses a different setup from DeHate. Replace it with
  `official_val/final/multihateloc/hatemm/seed_*` re-inferred on 215, `score_fused`, ×4. Cost about 0.1 GPU-h.
- **HCS:** must be retrained on p11. Features are ViT-ImageNet 1 fps + VGGish + BERT sentence, all present. Cost about
  0.5 GPU-h.
- **Machine:** uoa-lab1.

---

## 4. Label-free runs

### L1. DeHate: ZS-CLIP, ZS-ImageBind audio, LAVAD, AV²A, SeViLA, UniTime

**What exists:**
- No RH campaign method was ever run on DeHate.
- Only two DeHate baseline outputs exist:
  - HFU `runs/20260927_dehate_external/zs_imagebind` (vision channel);
  - the four weakly supervised methods.

**What the ports need:**
- Every RH campaign script hard-codes RH paths and a `VIDEO_DIR` dict with no DeHate entry, so all are ported (§1.5).
- Most upstream code, checkpoints and venvs are on uoa-lab2, so L1 runs on **uoa-lab2**. DeHate test videos, 1 fps
  frames and 16 kHz wavs are already there.
- Scope: all 1341 test videos, evaluated on 1151.
- Each method is one sbatch job, queued in `local-sc474399`.

| method | code (RH, ported) and upstream | inputs for DeHate | query / prompt | native rate → 4 fps | GPU mem | DeHate GPU-h | fallback |
|---|---|---|---|---|---|---|---|
| ZS-CLIP | `zs_clip.py`, `extract_dense.py`. No upstream: it is LAVAD's baseline row | CLIP-L/336 `pooler_output` at 4 fps (`dense4fps_clipL336`). **Missing for DeHate; extract.** Weights on lab1 and lab2 | "a normal video frame" / "a hateful video frame"; score = sigmoid(100·Δsim) | 4 fps, as is | about 3 GB | about 1 (scaled from HMM 3095 s for 1081 videos) | F2 (decode failure) |
| ZS-ImageBind audio | `extract_imagebind.py` (audio channel), `imagebind_text_emb.py`; ImageBind-huge (`third_party/lavad/libs/ImageBind`, `data/assets/imagebind/imagebind_huge.pth`). HFU `zs_imagebind.py` is vision-only, so add an audio path | 16 kHz wav (ready) → 2 s clips, 128-bin mel | "normal" / "hateful" (same anchors as the vision row) | 0.5 fps, piecewise constant | about 2.3 GB | about 0.5 | F1: videos or segments without audio get the silence embedding's score. The RH campaign dropped such videos (15 on HMM); here they are kept |
| LAVAD | `run_lavad_wave1.sh`, `blip2_caption.py`, `lavad_chain.py` + `patches/lavad.patch`, `shim/llama_hf`; upstream github.com/lucazanella/lavad @1ad46c66 | 1 fps frames (**ready**: RH `data/frames_1fps/DeHate`, 33G). BLIP-2 opt-6.7b-coco, Llama-2-13b-chat in NF4 and ImageBind on lab2 | LAVAD verbatim: "If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1 …" | 1 fps → ×4 | 14.4 GB (batch 48) | **about 15.5** (scaled: 8.3 GPU-h for 18.9 h of video) | F3 then F2 |
| AV²A | `run_av2a.py`, `run_av2a_supervised.sh` (a restart supervisor, not label supervision), `eval_av2a.sh`; upstream github.com/eitan159/AV2A @b0d6db8b, unpatched; own venv (transformers 4.31) | mp4 + 16 kHz wav (ready); LanguageBind Video_FT, Audio_FT, Image (lab2) | 12 hate labels + 20 background labels, template "A {label}" (RH list); headline `sim_combined` | 1 fps (audio half 0.1 fps) → ×4 | about 18.6 GB | about 4.5 (scaled: 1.2 s per 10 s window) | F1 silence audio for no-audio videos; F2 |
| SeViLA Localizer | `run_sevila.py` (`--selftest`); upstream github.com/Yui010206/SeViLA @419e7281; own venv | 1 fps frames (ready); `sevila_pretrained.pth`, flan-t5-xl (lab2), `eva_vit_g.pth` | "Question: Does this frame contain hateful content targeting a person or group? Option A: yes. Option B: no. …"; yes-logit | 1 fps → ×4 | 8.5 GB | about 1 | F2 |
| UniTime | `run_unitime.py` + `patches/UniTime.patch`, `unitime_to_curves.py`; upstream github.com/lzq5/UniTime @a557bbd5 | raw mp4 (ready); Qwen2-VL-7B + `zeqianli/UniTime` LoRA (lab2) | "the moment containing hateful content targeting a person or group" | intervals → binary 0/1 at 4 fps (`window` variant) | **28 GB peak**, tight on 32 GB | about 2–6 (scaled from 5.6–6.1 s/video; longer DeHate videos take longer) | F2. A video with no predicted interval is all zeros; that is a valid output, not a fallback |

**L1 total:** about **27 GPU-h on uoa-lab2**.

Setup, about half a day:
- port the scripts;
- clone 4 upstreams into HFU `third_party/` on lab2 and apply the RH patches;
- rebuild the av2a and sevila venvs;
- symlink the weights;
- run a 5-video smoke test per method.

Risks:
- UniTime's 28 GB peak leaves almost no headroom; the longest DeHate video is 299 s.
- The LAVAD deviations already recorded in RH stay: single BLIP-2 captioner, greedy decoding, 1.22 % Llama-2 refusals.
- AV²A's `evt_*` variants are stochastic (a random flip), so only `sim_combined` is used.
- Licences: LAVAD, AV²A, SeViLA and UniTime are research code; Llama-2 is under Meta's licence.

**Optional on DeHate** (same machine, after L1; decision D6):

| method | notes | DeHate GPU-h |
|---|---|---|
| LaGoVAD | `run_lagovad.py` + patch; CLIP B/16 on every 8th native frame (extract); main query is RH's long hate definition; pretrained on other labels | about 1.2 |
| URF-HVAA | `urf_chain.py` + patch; VideoLLaMA3-7B + Llama-3.1-8B (lab2); 10 s windows, 0.1 fps | about 9–10 |
| Vad-R1 | HFU `scripts/reproduction_baselines/run_all_vadr1.sh` (lab1 venv, vLLM); add DeHate to `VIDEO_DIRS`; one interval per video, so within-video ROC is .5 by construction (as on HMM/HCS) | about 1.3, on lab1 |
| MULDE | **not label-free** (normal-only train set, val-frame-label config); needs CLIP-L/336 4 fps for DeHate train/val/test | about 4 |
| CLAP (CVPR'24, AnasEmad11/CLAP) | **not label-free as run** (val frame ROC picks rounds); needs the train pool's dense features | about 3–5 |

### L1b. Reruns of ZS-CLIP, ZS-ImageBind audio, LAVAD, AV²A, SeViLA, UniTime on HMM and HCS (only if marked "needs rerun")

Known coverage gaps:
- The RH campaign dropped videos without audio (15 HMM videos for the ImageBind audio and AV²A paths).
- It had one failed SeViLA video and 21 UniTime errors.
- HCS `yt_NzvfkIYS5Yg` was a zero array in the dense CLIP cache.

The re-evaluation agent decides which methods need a rerun. Costs if needed, using the same ported code on uoa-lab2
(test videos only, scaled from the RH logs):

| method | HMM GPU-h | HCS GPU-h | can only the missing videos be rerun? |
|---|---|---|---|
| ZS-CLIP | 0.3 | 0.3 | yes, if the original cache is used for the rest |
| ZS-ImageBind audio | 0.15 | 0.15 | yes (F1 silence for no-audio videos) |
| LAVAD | 3.6 | 3.5 | **no**: the caption index spans the corpus, so rerun the whole corpus |
| AV²A | 1.0 | 1.0 | yes (F1 silence) |
| SeViLA | 0.2 | 0.2 | yes |
| UniTime | 0.4 | 1.0 (31.7 s median per video) | yes |
| **total (full reruns)** | **5.7** | **6.2** | |

Mixing old outputs with new rows for the missing videos is allowed only when the ported code reproduces the old
scores on 5 overlapping videos within float tolerance. Otherwise the whole corpus is rerun.

### L2. Text-only LLM (Qwen3-8B) on transcript segments, all three corpora

- **Code:** new `experiments/20261008_baselines/text_llm.py`. It copies the prompt strings verbatim from
  `spvl.py` (no import) and uses HF transformers in the HateVLM env.
- **Model:** `Qwen/Qwen3-8B` (Apache-2.0, 16.4 GB bf16, cached on lab2).
- **Thinking off:** `enable_thinking=False` in `apply_chat_template`. Otherwise the first generated token is
  `<think>` and the Yes/No log-odds mean nothing.
- **Unit:** one Whisper large-v3 segment from `data/asr_whisper_large_v3/<corpus>/timestamped_chunks.jsonl`, scored
  with no other context. This cache is ready for all three test sets (HMM 215 rows, HCS 393 covering all 118, DeHate
  1341).
- **Score:** log P("Yes") − log P("No") at the first answer token, using the same Yes/No token sets as `spvl.py`.
- **Prompt:** §1.4. With this, Qwen3-8B text and the MLLM winonly arm (L8) differ only in the model and in segment
  vs 8 s window.
- **Mapping:** each frame takes the score of the segment containing its centre. Where segments overlap, the larger
  score wins.
- **Fallback:** F1 for frames outside any segment, using the prompt with `(no speech)`; F4 for videos without a
  transcript.
- **Size:** segments in the cohorts: HMM 2308, HCS 1582, DeHate about 8700. One short forward each.
- **Cost:** about **0.5 GPU-h for all three corpora**. Memory 18 GB.
- **Machine:** uoa-lab2 (weights cached).

### L3. Off-the-shelf hate speech classifier on transcript segments, all three corpora

- **Code:** new `experiments/20261008_baselines/text_classifier.py`.
- **Unit and fallback:** the same unit, mapping and fallback as L2. The empty-input score is the classifier on `""`.
- **Long segments:** segments over 512 tokens are split into ≤512-token pieces, and the segment score is the maximum
  over pieces.
- **Score:** the logit of the hate class.

Candidate models (decision D2; all were trained on other hate-labelled corpora; footnote in the table):

| model | what it is | ready? |
|---|---|---|
| `tomh/toxigen_roberta` | RoBERTa fine-tuned on ToxiGen (implicit hate, 13 target groups); outputs LABEL_0/LABEL_1 with no names, so check on two fixed sentences which label means toxic; checkpoints are research use only | download about 500 MB |
| `tomh/toxigen_hatebert` | HateBERT fine-tuned on ToxiGen; the only off-the-shelf "HateBERT" classifier | cached on lab-server |
| `GroNLP/hateBERT` | a masked-LM base model, **not** a classifier; its HatEval/OffensEval fine-tunes are on OSF, not HF | not usable as is |
| `cardiffnlp/twitter-roberta-base-hate-latest` | binary HATE/NOT-HATE trained on 13 English hate datasets; CC-BY-4.0 | cached on lab2 |

- **Cost:** under 0.1 GPU-h per model for all three corpora; CPU is also feasible.
- **Machine:** uoa-lab2, in the same job as L2.
- **Risk:** max length 512 tokens; English only, while HMM has a minority of non-English speech.

### L4. LAION-CLAP audio–text zero-shot, all three corpora (optional)

- **Code:** new `experiments/20261008_baselines/clap_zeroshot.py`, using HF `laion/clap-htsat-unfused` (Apache-2.0).
  Using the HF version avoids the `laion_clap` package's `position_ids` loading bug. The pip checkpoint
  `630k-audioset-best.pt` (CC0) is the alternative.
- **Input:** 48 kHz audio decoded from the mp4 with ffmpeg. The cached 16 kHz wavs are band-limited and CLAP expects
  48 kHz.
- **Unit:** 2 s non-overlapping clips, the same unit as the ImageBind audio row; each clip is repeat-padded by CLAP
  to 10 s. 0.5 fps, piecewise constant.
- **Anchors:** "a normal audio clip" / "a hateful audio clip". Score = cosine difference.
- **Fallback:** F1 silence for no-audio videos.
- **Cost:** about 1 GPU-h for all three corpora (DeHate 1341: about 63k clips) plus about 1 h of CPU decoding.
  Memory 2 GB.
- **Machine:** uoa-lab1 or lab3.
- **Risk:** CLAP represents sound events, not spoken words, so it is expected near chance on speech-borne hate.

### L5. EventVAD at 4 fps, all three corpora (full rerun; the 1 fps runs are not reused)

- **Code:** HFU `scripts/reproduction_baselines/eventvad/` (`segment_events.py`, `score_events.py`,
  `rasterize_and_eval.py`, `run_all_eventvad.sh`).
  - It reconstructs the release's missing graph propagation and prompt (`DESIGN_EVENTVAD.md`).
  - Upstream is github.com/YihuaJerry/EventVAD (no licence), in `third_party/EventVAD`.
- **Models:** CLIP B/16 + RAFT (stage 1, about 0.3 GB) and VideoLLaMA2.1-7B-16F fp16 (stage 2, about 16 GB).
  Weights and venv are on lab1 only.
- **Port work:**
  - add `hateclipseg` and `dehate` to the corpus list, with the manifest as the video source;
  - change the rasterizer to the 4 fps grid: frame `i` takes the event containing `(i + 0.5) / 4` s, the same
    midpoint rule as before.
- **Native rate:** events in decoded native frames (FPS capped at 30, as in the paper).
- **Prompt:** the `paper` arm (`prompt.py`).
- **Fallback:** F3 = unparsed event → 0.0 (declared legacy rule). On HMM in 2026-08, 41 % of events were unparsed.
  The bounded-range arm would fix much of this but needs owner approval (decision D7). F2 for decode failures.

| corpus | scope | GPU-h (scaled from legacy HMM: segmentation 388.8 min + scoring 174.3 min for 8.1 h of video) | machine |
|---|---|---|---|
| HMM | 215 | about 9.5 | uoa-lab1 (assets local) |
| HCS | 118 | about 9 | uoa-lab1, after HMM |
| DeHate | 1151 | **about 36** (segmentation about 25, scoring about 11) | lab-server: 125 GB RAM; needs the venv, weights (about 17 GB) and DeHate test videos (6 GB) copied in, about 2 h of setup |

Risks:
- RAFT on every native frame pair is the main cost.
- Whole videos are decoded into RAM; lab-server's RAM avoids that risk for DeHate.
- Capping decode at 10 fps would cut DeHate to about 12–15 GPU-h, but departs from the paper's FPS = 30
  (decision D5).
- The upstream release never ran as published; the reconstruction is documented.

### L6. VADTree, all three corpora (decision D8)

- **Code:** github.com/wenlongli10/VADTree (NeurIPS 2025, arXiv 2510.22693; no licence file). Already cloned at
  uoa-lab2 RH `third_party/VADTree` @04dc1df with no weights; clone into HFU `third_party/`.
- **Models:**
  - EfficientGEBD ResNet50 boundary detector. **Its weight may not be downloadable.** VADTree only points to the
    EfficientGEBD repo, which releases a different backbone. Check this first; it is a blocker.
  - LLaVA-Video-7B-Qwen2 (16 GB).
  - DeepSeek-R1-Distill-Qwen-14B (29.5 GB bf16, thinking on; **does not fit a 32 GB 5090 with a KV cache**, so
    campus A100/H200 or 8-bit).
  - ImageBind-huge.
  - Two conda envs.
- **Inputs:** raw video only. It is visual only, with no speech channel.
- **Prompt:** must be rewritten. Its prompts list UCF/XD anomaly categories; replace them with the 9 `YOUTUBE_RULES`.
- **Native rate:** per tree node (coarse and fine), fused into per-frame scores, sampled at the 4 fps frame centres.
- **Cost.** The paper reports 62.3 GPU-h on two 3090s for 10.3 h of UCF-Crime video. Hate videos with frequent cuts
  are probably nearer XD-Violence's node density (about 2.5× UCF). Scaled, on A100:

  | scope | A100-h |
  |---|---|
  | HMM | 25–60 |
  | HCS | 25–60 |
  | DeHate (1151) | 90–230 |
  | **total** | **140–350** |
  | coarse-level-only variant (the paper's own, 27 % of the cost) | 40–95 |

- **Machine:** campus (A100 80G; H200 campus3 for the 14B model). Two jobs maximum, so one corpus per job.
  Prerequisite: the user renews the ControlMaster sockets.
- **Fallback:** F2.
- **Risks:**
  - the GEBD weight;
  - setup is multi-day (RH's 2026-08 survey already marked it "not recommended");
  - by far the largest cost of any baseline.

### L7. T3AL on DeHate, and a rerun on HMM/HCS (§2.4)

- **Code:** port RH `scripts/repro_campaign/{run_t3al.py, extract_coca_4fps.py, t3al_aggregate.py}`. Upstream is
  HFU `third_party/T3AL` (benedettaliberatori/T3AL @dfbbbc1c, CVPR 2024, no licence).
  - Upstream `T3ALNet.forward` runs unchanged; four hooks are overridden as in RH.
  - Per-video seed = `seed + cohort index` (no crc32).
- **Model:** open_clip `coca_ViT-L-14` / `mscoco_finetuned_laion2B-s13B-b90k`, cached on lab1 and lab2.
- **Inputs:**
  - CoCa 4 fps features: HMM/HCS test ready in RH `data/CLIP_Embedding/{HateMM,HateClipSeg}/coca_vitL14_4fps`;
    **DeHate must be extracted**.
  - 1 fps CoCa captions for all three corpora must be regenerated (the RH caption cache is empty).
- **Classes:** the RH frozen list. Main label "hateful content targeting a person or group".
- **Preset:** `D_anet`, fixed from 2026-08, not re-chosen.
- **Output:** 4 fps curves (native), variant `main`, seeds 20250819/20/21 as before or 234/2025/3407 (decision D4).
- **Cost:**
  - DeHate (1341): features about 0.5 h, captions about 0.5 h, adaptation 3 seeds about 1 h, so **about 2 GPU-h**.
  - HMM + HCS rerun: about 1.5 GPU-h.
  - Memory about 8 GB.
- **Machine:** uoa-lab1 (CoCa and DeHate videos local).
- **Fallback:** F2.
- **Risk:** upstream `get_indices` sampling degenerates for videos under 400 frames (100 s at 4 fps). The DeHate
  median is 64 s, so many DeHate videos may hit it. The smoke test must include short videos. If the method cannot
  produce output there, those videos fall to F2 and the >1 % stop rule applies.

### L8. MLLM per-window scoring without context on DeHate (Qwen2.5-VL-7B, InternVL3.5-8B)

- **Code:** `experiments/20260910_spvl/spvl.py` unchanged, arm `winonly`:
  `--isolation cache --frames 0 --no-transcript-context --branches joint --window-question rules --windows fixed --window-seconds 8`.
  This is the same flag set as `launch/run_mllm.sh`.
  - Add `--exp-id 20261008_baselines --datasets DeHate --manifest data/manifests/DeHate_test.jsonl`.
  - Then `compose.py --run-dir … --intercept spvl --residual rank --datasets DeHate`. This gives `ispvl_rrank`, the
    per-window-alone row of `runs/20260910_spvl/mllm_table.md`.
- **What winonly is:** per 8 s window, rules + the window's transcript only. No frames, no whole-video transcript,
  no stance. This is how it was defined for HMM/HCS. The DeHate Qwen3-VL reads used a different script
  (`experiments/20260922_til/til_measure.py`, the full SPVL-r2 reads).
- **Inputs:** `data/asr_whisper_large_v3/DeHate` (ready). Frames are not used.
- **Models:** `Qwen/Qwen2.5-VL-7B-Instruct` (lab1, lab2, lab3) and `OpenGVLab/InternVL3_5-8B-HF` (lab2 only).
  HateVLM env (transformers 5.15.1).
- **Native rate:** 8 s windows → frames, as `compose.py` does.
- **Fallback:** none needed. Windows without speech are asked with `(no speech)`, so every window has a score.
- **Cost:** HMM+HCS took 162–180 s for 333 videos, so DeHate 1341 is about 15 min plus loading per model. **About
  0.7 GPU-h for both.** Memory about 18 GB.
- **Machine:** uoa-lab2 (only place with InternVL3.5).

### L9. LELA (decision D9)

- **Paper:** "Towards Training-free Multimodal Hate Localisation with Large Language Models", arXiv 2602.09637
  (2026-02). **No code is released** (searched the arXiv page, the paper and GitHub). Its HMM Tables 1 and 3 disagree,
  with ROC and PR swapped.
- **Pipeline (from the paper):**
  - five text captioners: BLIP-2 (images), Whisper (speech), EasyOCR, LP-MusicCaps, PDVC (dense video captions);
  - for each frame, speech is combined with each of the other four;
  - per combination, a summary call, a role + rationale call and a 0–1 score call go to GPT-4o-mini, text only;
  - frame score = max over the four;
  - about 12 API calls per scored frame. The frame rate is not stated ("follows LAVAD", i.e. 1 fps).
- **Reusable inputs:**
  - Whisper transcripts (ready);
  - BLIP-2 1 fps captions (from L1 LAVAD on DeHate; the RH campaign's on HMM/HCS);
  - EasyOCR (lab2 has an `ocr_easy` env; no DeHate OCR cache).
- **New inputs:** LP-MusicCaps and PDVC. PDVC is the heaviest, with its own feature pipeline. About 10–20 GPU-h of
  captioning in total.
- **API cost** (gpt-4o-mini $0.15 / 1M input, $0.60 / 1M output; Batch API half price). Assumed per scored frame:
  about 4,600 input + 1,640 output tokens, about $0.00167.

  | scoring rate | HMM | HCS | DeHate (1151 ≈ 30.7 h) | total | total (Batch API) |
  |---|---|---|---|---|---|
  | 1 per second (paper-like) | $54, 387k calls | $24, 170k calls | about $185, 1.3M calls | **about $260** | about $130 |
  | 4 per second | $216 | $95 | about $740 | about $1,050 | about $525 |
  | 1 per Whisper segment | $11 | $5 | about $37 | about $52 | about $26 |

- **API key:**
  - The HateVideo env has the `openai` package (2.30.0).
  - `OPENAI_API_KEY` is not set in the lab1 shell.
  - No active HFU code reads a key; only an archived README under `archive/base-paper/` mentions one.
  - A reimplementation would read `OPENAI_API_KEY` from the environment in the Slurm job. The key is never written to
    a file in git or to a log.
- **Risks:**
  - It is a reimplementation, not the authors' code.
  - Several details are unstated: frame rate, alignment, temperature, model snapshot.
  - API rate limits; data-use terms for sending DeHate-derived text to OpenAI. The DeHate terms forbid
    redistributing derived data; whether sending captions or transcripts to a third-party API is allowed must be
    checked.

---

## 5. Schedule

Order of work:
1. **CPU work on uoa-lab1, about 1–1.5 days of engineering:**
   - `cohort.py` and `coverage_check.py`;
   - the `rh_campaign` ports;
   - the `hate_common` fixes (paths, `dehate`, p11);
   - the new scripts (MIL head, text LLM, classifier, CLAP);
   - the EventVAD 4 fps rasterizer and corpus entries;
   - the T3AL port;
   - copy the DeHate 1 fps features into `data/` with `PROVENANCE.md`;
   - an independent code review of the ports;
   - commit and push, then `git pull` on lab2 and lab-server.
2. **Environment and asset setup, in parallel with step 1, about 2–3 h:**
   - lab2: clone upstreams into HFU `third_party/`, rebuild the av2a/sevila venvs, symlink weights.
   - lab-server: EventVAD venv, VideoLLaMA2 + RAFT weights, DeHate test videos.
3. **5-video smoke test per method** on its target machine, with the coverage check on that subset.
4. **Full jobs.** Each corpus of each method runs whole on one machine. Machines run in parallel. Jobs on one node
   queue in its partition.

| machine | job queue, in order | GPU-h |
|---|---|---|
| uoa-lab2 (idle now) | L1 DeHate: ZS-CLIP → ZS-ImageBind audio → SeViLA → UniTime → AV²A → LAVAD (6 jobs, about 27 h); then L8 winonly Qwen2.5-VL + InternVL3.5 (0.7); then L2 + L3 on all three corpora (0.6) | **about 28.5** |
| lab-server (idle now) | L5 EventVAD DeHate, 1151 videos | **about 36** (long pole) |
| uoa-lab1 (after Slurm 282) | L5 EventVAD HMM → L5 EventVAD HCS | about 18.5 |
| uoa-lab3 (after Slurm 281) | W3 wav2vec2 extraction (1) → W1–W3 MIL (1.5) → W4 HMM re-infer + HCS retrain (2.2) → W5 VadCLIP (3.2) → W6 MultiHateLoc (0.6) → L7 T3AL all three (3.5) → L4 CLAP (1) | about 13. Needs about 45 GB rsync'd from lab1 first (DeHate wavs 20G, 1 fps features, DeHate test videos 6G, HMM/HCS legacy checkpoints). **Or swap with lab1:** if 282 frees first, run this queue on lab1 (everything local) and EventVAD HMM/HCS on lab3 after copying the EventVAD venv and weights there. |

Wall-clock estimates:
- **Required set** (W1–W6, L1, L2, L3, L5, L7, L8; L4 included): about 1–1.5 days of setup, then about 36 h for the
  long pole (EventVAD DeHate on lab-server). In total, **about 2.5–3 days**, assuming lab1/lab3 free up within the
  first day.
- **If L1b HMM/HCS reruns are needed:** about 12 GPU-h, put on lab2 after its queue. That makes lab2 about 40 h; it
  stays within about 3 days if L1b moves to whichever lab node frees first (weights copied).
- **Optional DeHate label-free** (LaGoVAD, URF-HVAA, Vad-R1): about 12 GPU-h on lab2 after the required queue, so
  **+0.5 day**.
- **VADTree:** campus only.
  - Full, about 140–350 A100-h over 2 concurrent jobs: **+3–7 days** plus 1–2 days of setup.
  - Coarse-level only on HMM + HCS: about 1–2 days.
- **LELA:** +1–2 days for the PDVC/LP-MusicCaps setup and captioning (10–20 GPU-h), plus a Batch-API turnaround of
  about 24 h; $26–$1,050.

---

## 6. Decisions the user must make

- **D1 MIL loss for W1–W3.** Recommended: top-k BCE (Wu et al. 2020). Alternative: the Sultani 2018 ranking loss.
  The head and training budget are the same either way.
- **D2 Text classifier.**
  - `tomh/toxigen_roberta`: recommended as the off-the-shelf model; research-use checkpoint.
  - `tomh/toxigen_hatebert`: the only ready "HateBERT" classifier.
  - `cardiffnlp/twitter-roberta-base-hate-latest`: trained on 13 hate datasets, CC-BY-4.0.
  - Raw `GroNLP/hateBERT` is not a classifier.
  - All cost minutes, so running two and reporting one is cheap.
- **D3 MultiHateLoc.** Use one setup on all corpora: `score_fused`, Optuna + 3 seeds. Retrain HCS on p11 (the
  current HCS number is leaked) and replace the HMM DMS single-seed row.
- **D4 HMM weakly supervised.** Reuse the 2026-08 official-val checkpoints (40-trial Optuna, 3 seeds; re-infer 215;
  minutes), or retrain with 5 trials for strict parity with DeHate (about 2 h per method). Also choose the T3AL seeds:
  the old 2025-08 seeds or 234/2025/3407.
- **D5 EventVAD decode rate.** Paper FPS = 30 (DeHate about 36 GPU-h) or capped at 10 fps (about 12–15 GPU-h, a
  stated deviation).
- **D6 Rows that use labels.** Whether MULDE and CLAP (CVPR'24) appear at all, and in which group:
  - MULDE uses normal-only train labels and val frame labels;
  - CLAP (CVPR'24) uses val frame labels for its round count.
  - Also whether the footnoted pretrained-on-other-labels rows stay in Label-free: SeViLA, UniTime, LaGoVAD, Vad-R1,
    the hate classifiers.
- **D7 Prompt wording.**
  - Keep the paper-native anomaly prompts (LAVAD, EventVAD, URF, Vad-R1) as the main rows; recommended for fidelity.
  - EventVAD's `bounded` arm (fixes some of the 34–44 % unparsed outputs) needs approval.
  - VADTree's category prompts must be rewritten; proposed: the 9 policy rules.
- **D8 VADTree.** Run it in full (140–350 A100-h, campus, GEBD weight unverified), coarse-level only, or only on
  HMM + HCS.
- **D9 LELA.**
  - Approve the reimplementation and the API spend: about $130–260 at 1 per second with or without the Batch API;
    about $26–52 at 1 per Whisper segment.
  - Provide the key as an environment variable.
  - Confirm that sending DeHate-derived text to OpenAI is allowed under the DeHate terms.
- **D10 Campus access.** Only needed for VADTree. Renew the ControlMaster sockets (`ssh uoa-campusN true`).

---

## 7. Summary

| run | corpora | code ready? | inputs ready? | GPU-h | machine | blocker |
|---|---|---|---|---|---|---|
| W1 BERT+MIL | all 3 | new script (small) | yes | about 0.5 | lab3/lab1 | D1 |
| W2 CLIP+MIL | all 3 | new script | yes | about 0.5 | lab3/lab1 | D1 |
| W3 audio+MIL | all 3 | new script + extractor | wav yes; wav2vec2/MFCC no (VGGish yes) | about 1.5 | lab3/lab1 | choose feature |
| W4 MACIL-SD/DSANet/Fed-WSVAD | HMM, HCS | yes, needs path/p11/dehate fixes + Fed-WSVAD clone | yes | about 2.2 | lab3/lab1 | HCS must use p11; D4 |
| W5 VadCLIP | all 3 | yes (port exists) | yes | about 3.2 | lab3/lab1 | none |
| W6 MultiHateLoc | HMM, HCS redo (DeHate done) | yes | yes | about 0.6 | lab3/lab1 | HCS number leaked; D3 |
| L1 ZS-CLIP | DeHate | port needed | features to extract | about 1 | lab2 | none |
| L1 ZS-ImageBind audio | DeHate | port (add audio path) | wav yes | about 0.5 | lab2 | none |
| L1 LAVAD | DeHate | port needed | frames yes | about 15.5 | lab2 | Llama-2 licence; long |
| L1 AV²A | DeHate | port + venv rebuild | yes | about 4.5 | lab2 | none |
| L1 SeViLA | DeHate | port + venv rebuild | frames yes | about 1 | lab2 | none |
| L1 UniTime | DeHate | port needed | yes | 2–6 | lab2 | 28 GB peak |
| L1b six reruns | HMM, HCS | same ports | test media yes | up to 12 | lab2 or free node | only if flagged |
| L1 optional (LaGoVAD, URF, Vad-R1, MULDE, CLAP-CVPR) | DeHate | ports | mostly to extract | about 12–21 | lab2 (Vad-R1 lab1) | D6 (MULDE/CLAP-CVPR use labels) |
| L2 Qwen3-8B text | all 3 | new script | yes | about 0.5 | lab2 | none |
| L3 hate classifier | all 3 | new script | yes (one model to download) | about 0.1 | lab2 | D2 |
| L4 LAION-CLAP | all 3 | new script | decode 48 kHz | about 1 | lab3/lab1 | optional |
| L5 EventVAD 4 fps | all 3 | yes; add corpora + 4 fps rasterizer | yes | about 55 (HMM 9.5, HCS 9, DeHate 36) | lab1 + lab-server | D5; 41 % unparsed (D7) |
| L6 VADTree | all 3 | cloned, not ported | weights missing | 140–350 A100 (coarse 40–95) | campus | GEBD weight; 14B model > 32 GB; D8 |
| L7 T3AL | DeHate + HMM/HCS rerun | port (remove crc32 seed) | CoCa HMM/HCS yes; DeHate + captions to extract | about 3.5 | lab3/lab1 | short-video `get_indices` risk; old HMM/HCS rows incomplete |
| L8 MLLM winonly | DeHate | yes (`spvl.py` unchanged) | yes | about 0.7 | lab2 | none |
| L9 LELA | all 3 | no code; reimplement | partly (PDVC, LP-MusicCaps, OCR missing) | 10–20 + API | lab2 + API | D9: about $130–260 at 1/s |

Required set: about 2.5–3 days wall clock, about 95 GPU-h on 4 lab GPUs. Optional reruns and extras add about
0.5 day. VADTree and LELA are separate decisions with multi-day costs.
