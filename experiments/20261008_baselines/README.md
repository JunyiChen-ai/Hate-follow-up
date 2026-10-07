# Label-free baselines re-evaluated with this repo's evaluator (2026-10-08)

Status: done, CPU only, on uoa-lab1 (sc474397). No model was run. This directory reads the per-video outputs of the
Retrieval-hate 2026-08 reproduction campaign, puts them on our 4 fps grid with the campaign's own rasterisation
rule, checks them against our fixed test cohorts, and passes them unchanged to
`src/eval/evaluate_four_datasets.py`. `run_plan.md` in this directory belongs to another agent and is not described
here.

Mechanism hypothesis: none. This is a measurement of existing baseline outputs, not a method change.

## 1. Cohort and the exact-test-set rule

- Cohort = the (dataset, video_id) set of the current method's predictions,
  `runs/20260926_twolevel/final_rawkey/predictions.jsonl`: HateMM 215 videos, HateClipSeg 118 videos
  (the 119 GT test ids minus `yt_NzvfkIYS5Yg`, see `data/gt_4fps/PROVENANCE.md`). Within-video macro is defined on
  84 / 99 of them. GT = `data/gt_4fps/<dataset>.npz`, split `test`; these arrays are identical (ids, lengths,
  labels) to Retrieval-hate's `data/gt/frame_gt_4fps/` on both corpora, checked by array comparison.
- Rule (user, 2026-10-08): a corpus is evaluated for a method only if every cohort video has a finite score on
  every GT frame. One missing video, or any unscored (NaN) frame inside the GT length, means no metrics for that
  corpus and the run is marked "needs rerun". Videos outside the cohort are dropped before evaluation.
- Every evaluated corpus has the same frame pool as the current method: HateMM 116,975 frames / 215 videos,
  HateClipSeg 113,002 frames / 118 videos (`n_frames`, `n_videos_predicted` in each `metrics.json`, equal to
  `runs/20260926_twolevel/final_rawkey/metrics.json`). The evaluator therefore dropped no video and no frame.

## 2. What was reused, and from where

| method | per-video output used | where it lives |
|---|---|---|
| ZS-CLIP | `scores_visual_<DS>.npz`, prompt set `main` | uoa-lab2 `~/Retrieval-hate/archive/idea-stage/repro_zs_clip/` → copied |
| ZS-ImageBind (audio) | `imagebind_audio/<vid>.npy` embeddings + `imagebind_text_normal_hateful.npy` | uoa-lab2 `~/Retrieval-hate/data/CLIP_Embedding/` → copied |
| LAVAD | `curves/<DS>/<vid>.npz`, key `base` | `/home/jehc223/Retrieval-hate/archive/idea-stage/repro_lavad/` (local, read in place) |
| AV²A | `curves/<DS>/<vid>.npz`, key `sim_combined` | uoa-lab2 `.../repro_av2a/curves/` → copied |
| SeViLA Localizer | `curves/<DS>/<vid>.npz`, key `main` | uoa-lab2 `.../repro_sevila/curves/` → copied |
| UniTime | `curves/main/<DS>/<vid>.npz`, key `window`, + `<DS>_intervals_window.json` | uoa-lab2 `.../repro_unitime/curves/main/` → copied |
| URF-HVAA | `curves/<DS>/<vid>.npz`, key `base` | `/home/jehc223/Retrieval-hate/archive/idea-stage/repro_urf/` (local) |
| MULDE | `curves/<DS>/<vid>.npz`, keys `clipL336_s0/1/2` | uoa-lab2 `.../repro_mulde/curves/` → copied |
| CLAP | `curves/<DS>/<vid>.npz`, keys `fedavg11_s0/1/2` | uoa-lab2 `.../repro_clap/curves/` → copied |
| LaGoVAD | `curves/<DS>/<vid>.npz`, key `main` | uoa-lab2 `.../repro_lagovad/curves/` → copied |
| Qwen2.5-VL-7B grounding | `raw/qwen_<DS>_main.jsonl` (one interval per video) | `/home/jehc223/Retrieval-hate/archive/idea-stage/repro_qwen_ground/` (local) |

The uoa-lab1 copy of Retrieval-hate holds only the git-tracked part of the campaign; the per-video curves of
ZS-CLIP, ImageBind, AV²A, SeViLA, UniTime, MULDE, CLAP and LaGoVAD exist only on uoa-lab2. They were copied
read-only to `data/retrieval_hate_repro/` (215 MB, HateMM and HateClipSeg only; provenance in
`data/retrieval_hate_repro/PROVENANCE.md`). Nothing was written to either Retrieval-hate checkout. Variant choice =
the campaign's pre-declared main variant for each method (`summary_test.csv`, "main variants only; no post-hoc
best-variant selection"). Method descriptions: `Retrieval-hate/archive/idea-stage/REPRO_CAMPAIGN_RESULTS.md`
(§ZS-CLIP, §H, §I, §J, §K, §L, §M) and `archive/idea-stage/repro_campaign/sections/{O_mulde,P_clap,R_sevila}.md`.

## 3. Conversion rule per method

All rules are Retrieval-hate's `scripts/repro_campaign/eval_frame.py` arithmetic, re-implemented in
`convert_rh_baselines.py` (no import from Retrieval-hate).

- **Score curves** (ZS-CLIP 4 fps, ImageBind-audio 0.5 fps, LAVAD 1 fps, AV²A 1 fps, SeViLA 1 fps, URF 0.1 fps,
  MULDE 4 fps, CLAP 2 fps, LaGoVAD fps/8 per video): piecewise-constant broadcast, `broadcast_to_4fps` — 4 fps
  frame i takes native sample `floor(i / 4 * rate)`, index clipped to the last sample. Output length = GT length.
  - One change from the campaign: Retrieval-hate scored `min(T_gt, T_feat)` frames, i.e. it dropped GT frames past
    the method's last native sample. To keep the exact frame pool, the same function is called with `T = T_gt`, so
    those frames hold the last native sample's value (the function's own clip branch). Count per method in
    `coverage.json` (`tail_hold`): ZS-CLIP and MULDE 67 frames / 10 videos (HateMM), 76 / 3 (HCS); LAVAD and SeViLA
    134 / 72, 107 / 31; CLAP 143 / 83, 110 / 35; ImageBind-audio 3 / 2, 23 / 1; AV²A, LaGoVAD, UniTime, Qwen 0.
    At most 0.13 % of a corpus's frames. Most are under 1 s (a 1 fps or 2 fps grid ending before `floor(4D)`).
    The largest is `bit_AxrVklzh9Cyf` (HCS): its decodable video stream ends about 18.5 s before the annotated
    duration, so frame-based methods have no output for the last 73–75 frames, all GT-positive (ZS-CLIP, MULDE, CLAP,
    SeViLA). A rerun would hit the same media limit.
  - ZS-CLIP stores `cos(img, "a hateful video frame") − cos(img, "a normal video frame")`; the campaign reported
    `sigmoid(100·x)`. The map is strictly increasing, so all three metrics are identical.
  - ImageBind-audio score = softmax over (normal, hateful) of the unit-normalised audio embedding times the raw text
    embedding (`imagebind_curves`), computed here on CPU from the cached embeddings.
- **Interval outputs** (Qwen2.5-VL, UniTime): the interval is clipped to `[0, D]` and frames
  `[ceil(4a), ceil(4b))` are set to 1, all others 0 (`qwen_curves`, `unitime_to_curves.raster`). UniTime's curves
  were already rasterised this way by the campaign at the GT length and are used as stored; Qwen's are rasterised
  here from the raw JSONL. With a 0/1 score, pooled ROC-AUC equals balanced accuracy at one operating point. The
  intervals are also passed to the evaluator, so its interval-F1 fields are filled for these two methods only.
- D = `duration` in `data/omsl_v6_inputs/manifests/all_test.jsonl` (identical to Retrieval-hate's GT durations).

## 4. Coverage against the fixed cohorts

Source: `runs/20261008_baselines/<method>/coverage.json`.

| method | HateMM | HateClipSeg |
|---|---|---|
| ZS-CLIP | 215/215, exact | 118/118, exact |
| ZS-ImageBind (audio) | 215/215, exact | 118/118, exact |
| LAVAD | 215/215, exact | 118/118 videos, **not exact**: `bit_ZaY0S1anrdep` has 12 unscored frames (the stored `base` curve is NaN at the 1 fps samples for 147–149 s; the campaign reported HCS coverage .9999 for this reason) |
| AV²A | 215/215, exact | 118/118, exact |
| SeViLA Localizer | 215/215, exact | 118/118, exact |
| UniTime | 215/215, exact | **117/118**: `yt_8sIY4W1hBQM` has no output (`FileNotFoundError` in UniTime's video loader; decord also fails on this file, campaign §J) |
| URF-HVAA | **213/215**: `non_hate_video_32`, `non_hate_video_441` have every window NaN (unscored; the campaign's refusal log counts only 4 refused HateMM windows, so most of these are not refusals); 4 more videos with one unscored 10 s window (`hate_video_277`, `hate_video_44`, `non_hate_video_11`, `non_hate_video_348`) | **115/118**: `bit_AxrVklzh9Cyf` no output (frame-sampler `IndexError`), `bit_O4ybgRijNK8x`, `yt_Xtics6JrHKc` every window NaN |
| MULDE (3 seeds) | 215/215, exact | 118/118, exact |
| CLAP (3 seeds) | 215/215, exact | 118/118, exact |
| LaGoVAD | 215/215, exact | 118/118, exact |
| Qwen2.5-VL-7B grounding | 215/215, exact | 118/118, exact |

Not evaluated (no `predictions.jsonl` rows, no metrics): LAVAD HateClipSeg, UniTime HateClipSeg, URF-HVAA both
corpora. URF also has a structural gap: its 10 s windows tile only full windows, so the last partial window of 203
HateMM and 109 HCS videos is never scored (up to 9.75 s each, 3.4 % / 1.9 % of frames). The campaign's own HCS
numbers for ZS-CLIP, MULDE and CLAP included `yt_NzvfkIYS5Yg` (scored on an all-zero feature array, campaign §H);
that video is outside our cohort and was dropped.

## 5. How each method was run, and label use

No method selected any knob, prompt, threshold or checkpoint on test labels (campaign freeze §10 red line 1; MULDE's
and CLAP's upstream test-label selection loops were disabled by the campaign, `REPRO_CAMPAIGN_SUMMARY.md` §6.6).

| method | hate query / prompt | selection | supervision |
|---|---|---|---|
| ZS-CLIP | CLIP ViT-L/14-336; "a normal video frame" vs "a hateful video frame", fixed by the campaign brief; three other prompt pairs reported in an appendix, none selected | none | label-free |
| ZS-ImageBind (audio) | ImageBind-Huge; text "normal" vs "hateful"; 2 s audio clips at 0.5 fps. The audio channel is the campaign's extension beyond LAVAD's visual baseline | none | label-free |
| LAVAD | LAVAD's own anomaly-scoring prompt (law-enforcement wording; no hate wording); BLIP-2 opt-6.7b-coco only (1 of 5 captioners), Llama-2-13b-chat in NF4, greedy, 1 fps centres | none | label-free |
| AV²A | LanguageBind; hyper-parameters = the AV²A README's LanguageBind row (authors' values for the external LLP set); vocabulary of 12 hate + 20 background event words written once by the campaign; read-out `sim_combined` = per-second max over the 12 hate labels of the continuous similarity (the campaign's read-out; the published pipeline outputs events, `evt_*`) | none on hate data | label-free w.r.t. hate; external hyper-parameters |
| SeViLA Localizer | "Does this frame contain hateful content targeting a person or group?" in the repo's QVHighlights template; yes-logit per 1 fps frame | none, no val run | **external temporal supervision**: Localizer pre-trained on QVHighlights |
| UniTime | Qwen2-VL-7B + UniTime LoRA; query "the moment containing hateful content targeting a person or group" | none | **external temporal supervision** (temporal-grounding training data) |
| URF-HVAA | URF's anomaly prompts; VideoLLaMA3-7B captions, Llama-3.1-8B scorer; 10 s centres (vs 0.53 s upstream) | none | label-free |
| MULDE | none (density model on CLIP-L/336 features) | **val split, frame ROC-AUC over a 144-cell grid** (width, σ_high, epochs, read-out, aggregation), per dataset; chosen HateMM `[4096,4096]/1.0/50/log_density/gmm1`, HCS `[4096,4096]/0.5/50/score_norm/gmm5` (`data/retrieval_hate_repro/repro_mulde/run_record_*.json`) | **uses target labels**: trained only on train-split videos whose video-level label is non-hateful (HateMM 446, HCS 30 videos), plus val frame labels for selection |
| CLAP | none (pseudo-labels from feature-norm clustering, `--train_mode US`) | **val split, frame ROC-AUC** over FedAvg rounds {1,2,5,10}; chosen 1 on both HateMM and HCS, which equals the published `train.sh` default (`knob_rule.json`, `eval_clap_knobsel_val.json`) | training pool = train split, no labels; val labels used only for the round count |
| LaGoVAD | free-text definition `main` ("Hateful content: speech, imagery or on-screen text that attacks, dehumanizes, …"), frozen before the run; raw logit instead of the demo's saturating sigmoid | none | **external supervision**: released checkpoint trained on the PreVAD anomaly corpus (campaign class `aux-temporal-pretrain`) |
| Qwen2.5-VL-7B grounding | lmms-eval Charades grounding prompt, query "the moment containing hateful content targeting a person or group"; 32 frames, NF4 (deviation from bf16) | none | label-free |

Hash-ban notes (CLAUDE.md). CLAP's 11 FedAvg clients are assigned by `crc32(video_id) mod 11`
(`Retrieval-hate/scripts/repro_campaign/run_clap.py:128`), so its curves depend on a checksum-derived partition.
AV²A re-seeds each video with `seed XOR crc32(video_id)`; per campaign §J this affects only the event outputs
(`evt_*`, random flip in the video transform), not the `sim_combined` curve used here. No hash, checksum or commit
identifier was computed or recorded in this directory; copied metadata files that carry commit ids were deleted
(see `data/retrieval_hate_repro/PROVENANCE.md`).

## 6. Results

Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC, canonical evaluator. Every number is
transcribed from the `per_dataset` block of the cited `metrics.json` (written by `src/eval/evaluate_four_datasets.py`).
MULDE and CLAP: mean ± sd (n−1) over the three seed runs, each seed in its own `metrics.json`. The `MACRO` block in
`lavad/metrics.json` and `unitime/metrics.json` covers HateMM only (`n_datasets` = 1) and must not be read as a
two-corpus mean. These are baseline outputs selected without test labels; they are not development-selected.

| method | HateMM ROC | PR | within | HCS ROC | PR | within | source |
|---|---:|---:|---:|---:|---:|---:|---|
| ZS-CLIP | .5368 | .2774 | .5156 | .4984 | .4627 | .5201 | `runs/20261008_baselines/zs_clip/metrics.json` |
| ZS-ImageBind (audio) | .5654 | .2905 | .5256 | .5652 | .5122 | .5218 | `runs/20261008_baselines/zs_imagebind_audio/metrics.json` |
| LAVAD | .5588 | .2906 | .4814 | — | — | — | `runs/20261008_baselines/lavad/metrics.json` (HCS not exact) |
| AV²A | .5393 | .2520 | .5087 | .4860 | .4680 | .4947 | `runs/20261008_baselines/av2a/metrics.json` |
| SeViLA Localizer † | .6271 | .3319 | .5209 | .5762 | .5614 | .5430 | `runs/20261008_baselines/sevila/metrics.json` |
| UniTime † | .4778 | .2345 | .4715 | — | — | — | `runs/20261008_baselines/unitime/metrics.json` (HCS missing 1 video) |
| URF-HVAA | — | — | — | — | — | — | not evaluated (`runs/20261008_baselines/urf_hvaa/coverage.json`) |
| MULDE ‡ | .5989 ± .0031 | .3076 ± .0048 | .5365 ± .0030 | .5245 ± .0066 | .4896 ± .0095 | .5153 ± .0182 | `runs/20261008_baselines/mulde_s{0,1,2}/metrics.json` |
| CLAP § | .6013 ± .0147 | .3563 ± .0050 | .5246 ± .0145 | .4730 ± .0016 | .4641 ± .0029 | .5015 ± .0022 | `runs/20261008_baselines/clap_s{0,1,2}/metrics.json` |
| LaGoVAD † | .5579 | .3047 | .4785 | .5000 | .4666 | .5185 | `runs/20261008_baselines/lagovad/metrics.json` |
| Qwen2.5-VL-7B grounding | .5185 | .2522 | .5161 | .5030 | .4750 | .5031 | `runs/20261008_baselines/qwen25vl_grounding/metrics.json` |

† external temporal / anomaly supervision. ‡ trained on target videos labelled non-hateful; val-label selection.
§ val-label selection of the round count (landed on the published default on both corpora); crc32 client split.

Per-seed values: MULDE HateMM .6002/.3089/.5395, .6011/.3117/.5364, .5954/.3023/.5336; HCS .5295/.5005/.5278,
.5170/.4833/.4945, .5269/.4849/.5237. CLAP HateMM .5857/.3515/.5181, .6147/.3614/.5412, .6037/.3560/.5146; HCS
.4748/.4668/.5030, .4723/.4644/.5026, .4720/.4610/.4990 (seeds 0/1/2, same files).

Consistency with the campaign: on HateMM (same 215-video pool) every pooled number matches the campaign's
`summary_test.csv` within .0006; on HateClipSeg, AV²A, SeViLA, LaGoVAD, Qwen and ImageBind-audio match within
.0001 (same 118 videos); ZS-CLIP, MULDE s0 and CLAP s0 move by up to .014 because the campaign pool also held
`yt_NzvfkIYS5Yg`.

## 7. What the numbers say, and what is open

- Every evaluated baseline has within-video macro ROC-AUC between .47 and .55 on both corpora, i.e. close to
  chance for ordering frames inside a video. The current method reads .7508 / .6373
  (`runs/20260926_twolevel/final_rawkey/metrics.json`).
- Best pooled baseline rows: SeViLA (.6271 / .5762 ROC), which uses external temporal supervision; among strictly
  label-free rows, ZS-ImageBind-audio (.5654 / .5652) and LAVAD on HateMM (.5588).
- Open, needs the user's decision:
  1. LAVAD HCS: 12 unscored frames in one video. Options: rerun LAVAD stage 04–06 for `bit_ZaY0S1anrdep`, or
     leave the HCS cell empty.
  2. UniTime HCS: rerun one video (`yt_8sIY4W1hBQM`, needs a decoder that opens it), or leave empty.
  3. URF-HVAA: needs a rerun on both corpora (7 missing or all-unscored videos, 4 partly unscored, and a 0.1 fps
     grid that leaves the last partial window unscored); or drop it.
  4. Tail hold (§3): accept holding the last native value for frames past a method's native coverage (≤ 0.13 % of
     frames; largest case `bit_AxrVklzh9Cyf`, 18.5 s of undecodable tail), or treat it as partial coverage.
  5. MULDE uses target video-level labels for training and val frame labels for selection, so it is not label-free
     under this project's definition; CLAP's round count was chosen with val labels (same value as the published
     default) and its client split is checksum-derived. Decide whether these two enter the paper table, a separate
     "uses target labels" block, or neither.
  6. Our own ZS-ImageBind (image) run, `runs/20260912_baselines/zs_imagebind/`, covers only 115/118 HCS videos
     (3 decode failures), so it fails the exact-test-set rule. The campaign's ImageBind-image embeddings on
     uoa-lab2 (`~/Retrieval-hate/data/CLIP_Embedding/<DS>/imagebind_image/`) cover all 215 + 118 cohort videos and
     could be converted the same way. T3AL (campaign's best label-free mean) has curves on uoa-lab2 but seed
     20250819 covers 214/215 HateMM videos; its preset was chosen on val PR-AUC. Neither was in this task's list.

## 8. How to run

```bash
source ~/miniconda3/bin/activate HateVideo
cd ~/Hate-follow-up
python3 experiments/20261008_baselines/convert_rh_baselines.py            # all methods, ~15 s CPU
python3 experiments/20261008_baselines/convert_rh_baselines.py --methods zs_clip sevila
```

Per method the script writes `runs/20261008_baselines/<method>/`: `run.log` (first line = hostname),
`run.pid`, `config.json` (inputs, variant, conversion rule, date), `coverage.json`, and, only for corpora that
pass the exact-test-set rule, `predictions.jsonl` (shared schema) plus `metrics.json`, which the script obtains by
calling `python3 -m src.eval.evaluate_four_datasets --predictions … --gt-dir data/gt_4fps --out …/metrics.json
--datasets <exact corpora>`. Inputs: `data/retrieval_hate_repro/` (copied, see its `PROVENANCE.md`) and the
read-only Retrieval-hate checkout on uoa-lab1.
