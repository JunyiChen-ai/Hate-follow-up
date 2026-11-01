# Backbone study with the adopted reader (2026-11-01)

## 1. What and why

The paper's backbone table (`paper/overleaf/sections/experiments.tex`, Table `tab:mllms`) and its appendix ablations
(`paper/overleaf/sections/appendix.tex`, Table `tab:mllm_ablation`) still hold the r6-era reader: the 2026-09-10
family study (`experiments/20260910_spvl/README.md` §11, reads `runs/20260910_spvl/mllm/<m>/`) decoded with the r6 time
level and the calibrated key (`experiments/20260926_twolevel/README.md` §12.1, §19.3, `robust/<m>_r6`). Since
2026-10-31 the method uses the window-evidence reader (`experiments/20261031_window_evidence/`, `src/window_evidence.py`)
and the raw key (calibration removed 2026-10-07). This experiment re-runs both tables with the adopted reader on the
same eight checkpoints, HateMM and HateClipSeg only (the two tables cover only these corpora). It is a reporting run,
not a candidate: nothing is promoted or decided here.

Backbones (same Hugging Face checkpoints as the 2026-09-10 study):

| tag | checkpoint | family (`model_type`) | path in `src/window_evidence_family.py` |
|---|---|---|---|
| q3vl-2b / q3vl-4b / q3vl-8b / q3vl-32b | `Qwen/Qwen3-VL-{2B,4B,8B,32B}-Instruct` | qwen3_vl | mRoPE (adopted code unchanged) |
| q25vl-7b | `Qwen/Qwen2.5-VL-7B-Instruct` | qwen2_5_vl | mRoPE (adopted code unchanged) |
| internvl35-8b | `OpenGVLab/InternVL3_5-8B-HF` | internvl | 1D |
| llava-ov-7b | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | llava_onevision | 1D |
| gemma3-12b | `google/gemma-3-12b-it` | gemma3 | 1D |

## 2. Definitions (declared 2026-11-01, before any GPU run)

**Reader ("full").** The adopted reader, unchanged in content: shared prefix (system message, intro, the 20 uniform
frames of `data/frames_k20` with their timestamps, the whole timestamped Whisper transcript, the 9 rules, the reader
note; `Judge.prefix_messages`) encoded once → whole-video question → `z_video` → the model's own Yes/No written back as
the stance turn → per 8 s window two isolated branches on the shared cache: visual (the window's 0.5 fps frames of
`data/temporal_retrieved_kv/<ds>`, `t1 <= t < t2`, each labelled `[t=..s]`, then the r6 visual question, one user turn;
a window without a 0.5 fps frame asks the visual question alone) and speech (the r6 speech question with the words whose
forced-aligned midpoint lies in the window, `data/asr_whisper_aligned/<ds>`; proportional slice for videos without an
alignment; read only when non-empty). Window read `z = max(z_visual, z_speech)`; record `src/window_evidence.window_prediction`.
Per-family image settings `FAMILY_IMAGE_KW` (Qwen pixel cap, InternVL one 448 tile, Gemma 3 without pan-and-scan,
LLaVA-OneVision base 384 grid), as the 2026-09-10 study.

**Main table columns** (pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC, 4 fps, test split,
`src/eval/evaluate_four_datasets.py`):
- **Verdict**: the full run's `z_video` painted on every frame (`experiments/20260910_spvl/compose.py --intercept spvl
  --residual none`, as the old column; within = .500 by construction).
- **Per window**: each window read alone. Per window one ordinary uncached call: the old per-window-alone prompt
  without whole-video prefix (system message + "Observe this video." + the 9 rules + the reader note, no frames, no
  transcript, `prefix_messages([], [], with_context=False, with_frames=False)`), no verdict, no stance; then one user
  turn with only that window's 0.5 fps frames (labelled as in the visual branch) and the joint question
  `yesno_question(i, n, t1, t2, words, kind="joint")` carrying the window's aligned words ("(no speech)" when empty).
  The window score is that read. Scored as the old column (`summarize_mllm.py`: `winonly` composed with
  `--intercept spvl --residual rank`): a per-video constant plus the zero-mean centred rank of the window reads inside
  the video. The old `winonly` `z_video` was one constant per model (whole-video question on the rules-only prefix:
  q3vl-8b −5.526, gemma3-12b −5.734, llava-ov-7b −0.498 on every video), so the composition here is
  `--intercept none --residual rank`, which gives the same three numbers. Differences from the old column: the old read
  had no frames at all and the rules question; this one has the window's own frames, its aligned words and the joint
  question wording (as requested).
- **\ours{}**: the full reads decoded with the current time level and raw key:
  `python experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key raw --duration bma --bma-prior
  length --min-windows 2 --bma-grid 6 --arm m2`, HateMM + HateClipSeg in one decode (EM per corpus).

**Appendix table** (within change, arm − full, both measured "under the reader's own composition", no temporal model:
`compose.py --intercept zv_plus_mean --residual rank`, i.e. verdict + mean window read + centred rank of the window
reads; within depends only on the centred rank; as `summarize_mllm.py`). Columns and arms:

| paper column | arm here | definition |
|---|---|---|
| alone | the Per-window read above | composed `--intercept none --residual rank` (within identical to any per-video intercept) |
| −ctx | A1 `no_transcript` | `experiments/20261101_reader_ablation/README.md` §2 |
| −frames | A2 `no_prefix_frames` | same (removes the 20 prefix frames; window branches keep their own frames) |
| ASR | A4 `asr_windows`, frame cap | same (Whisper segments as windows; gaps −12), except that a segment window's visual branch sees at most 4 of its 0.5 fps frames, uniformly spaced over the segment (see below) |
| −stance | A3 `no_stance` | same |
| joint | A6 `joint` | same (one branch per window: window frames + joint question with aligned words) |

**A4 frame cap (declared 2026-11-01 after the LLaVA-OneVision smoke ran out of memory on one A4 window, before any full arm run).** Whisper segments reach 479 s on HateMM and 326 s on HateClipSeg (99th percentile 146 s / 199 s; 232 / 690 segments carry ≥ 15 frames at 0.5 fps, the longest 240). With every frame in the branch (reader-ablation A4) the longest window is 22k image tokens on Qwen (91 per frame), 61k on InternVL / Gemma (256) and 175k on LLaVA-OneVision (730), past LLaVA-OneVision's 32k context and past a 32 GB GPU. So in this study A4 gives a segment window at most 4 in-window frames (`np.linspace` over the segment's frames, rounded), the number of 0.5 fps frames in one fixed 8 s window, i.e. the fixed grid's per-read visual budget; segments with ≤ 4 frames are unchanged. The cap applies to every backbone, Qwen3-VL-8B included, so this A4 is not the reader-ablation A4 on long segments. The smoke A4 records (`runs/20261101_backbones/smoke/<tag>/records/asr_windows`, q3vl-2b / q3vl-4b / q25vl-7b / q3vl-8b / internvl35-8b) predate the cap.

The other arm definitions are the reader-ablation agent's, unchanged; for the mRoPE families the context is built by its
`src/reader_arms.build_context`, so a Qwen3-VL-8B arm here is the same computation as its arm. The 1D families build the
same context in `src/window_evidence_family.build_arm`.

## 3. Porting (`src/window_evidence_family.py`) and checks

- mRoPE families (Qwen3-VL, Qwen2.5-VL): the adopted functions are called unchanged (`stance_cache.build / margin`,
  `window_evidence.frames_margin`; explicit 3D positions continued after the cached conversation). Qwen2.5-VL has the
  same `get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, ...)` interface in transformers 5.15.1.
- 1D families (InternVL3.5, LLaVA-OneVision, Gemma 3): the same conversation through each chat template; positions
  continue from the cache length; every branch is a forward on a deep copy of the shared cache (Gemma 3's
  sliding-window cache layers cannot be cropped back once full; the 2026-09-10 study also used copies). Template
  adaptations (as the 2026-09-10 study): Gemma's template rejects consecutive user turns, so a turn asked with nothing
  answered before it joins the prefix's user turn (`Judge.same_turn`; applies to the Per-window read and to A3);
  LLaVA-OneVision renders a turn's images before its text, so a window's frame times are given as a list after its
  frames ("The k images above are frames sampled from this window, in order; frame k was taken at: ..."), its stance
  turn keeps the generation header (`Judge.loose_stance_seam`) and its branches are rendered without the stance
  history and appended after the cached stance tokens. LLaVA-OneVision reads a turn with exactly one image in its
  single-image (AnyRes) mode, which costs more tokens than a prefix frame; this only happens for a window with one
  0.5 fps frame (a last window of at most 2 s).
- Gemma 3 fixes found by the smoke (transformers 5.15.1; both before any full Gemma run): (i) the model's own cache keeps only the last 1023 tokens in its sliding-window layers, and a cached suffix with images then fails with a mask shape error; 1D families now encode the prefix on an explicit full-length `DynamicCache` (for InternVL / LLaVA-OneVision this is the cache they build anyway: their HateClipSeg runs with this code equal their smoke reads with the old code, 0.0); (ii) Gemma 3 lets an image's tokens attend to each other both ways, and the image-block ids are indexed by absolute position, so ids covering only the suffix left the window frames causal (plain-call check 7.4 nats); the ids now cover the cached tokens too (check 1.16 nats). Gemma 3 reads branches in place and crops back (memory: 12B weights + a 2.4 GB full-length cache).
- Check (a), Qwen3-VL-8B: on the first 2 + 2 HateMM / HateClipSeg test videos (manifest order) the full reads must equal
  the stored adopted reads `runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned` (`z_video`, every `z_visual`,
  every `z_speech`, speech presence), max |Δ| ≤ 1e-3 (the window_evidence tolerance; the adopted path is expected to
  give 0.0).
- Check (b), every family: on the same 2 + 2 videos (smoke) and on the first video of every full run, the verdict and
  the first four windows' visual and speech branches read on the shared cache are compared with the same token string
  read in one ordinary uncached forward with all images (prefix frames + the branch's frames). Gate: max |Δz| < 3 nats,
  the 2026-09-10 family study's gate (README §11 there: bf16 cached-decode vs prefill noise reaches ≈ 1 nat on rare
  branches, a position or mask error shifts every branch by more), and the token ids of the plain call must equal the
  cached ids + the branch ids (1D) or have the cached length + the branch ids (mRoPE). Spearman over the compared reads
  is logged. A failure ends the job with `FAILED`.
- Arms A3, A4, A6 share the prefix and verdict with the full reader, so their `z_video` must equal the main record's
  (≤ 1e-3) on every video.

## 4. How to run

Inputs on the machine (rsync from uoa-lab1 when missing, no delete): `data/frames_k20`, `data/temporal_retrieved_kv`,
`data/asr_whisper_aligned`, `data/asr_whisper_large_v3` (HateMM, HateClipSeg), `data/omsl_v6_inputs/manifests`, and for
the q3vl-8b smoke `runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl`. Weights in the repo HF
cache `.cache/hf/hub` (copied or symlinked; logs `runs/_setup_uoa-lab1/`).

1. Smoke (checks a, b; every arm on 2 + 2 videos):
   `TAGS="q3vl-8b q3vl-2b ..." SCOPE=smoke sbatch experiments/20261101_backbones/launch/<lab1|lab23|labserver>.sbatch`
2. Main reads, one backbone × one corpus per job:
   `TAG=<tag> DATASET=<HateMM|HateClipSeg> sbatch experiments/20261101_backbones/launch/<...>.sbatch`
3. Reader arms, one backbone × one corpus per job:
   `TAG=<tag> DATASET=<ds> ARMS="no_transcript no_prefix_frames no_stance asr_windows joint" sbatch ...`
4. Results back to uoa-lab1: `rsync -a <host>:Hate-follow-up/runs/20261101_backbones/<tag>/ runs/20261101_backbones/<tag>/`
5. Decode / compose / table (uoa-lab1, CPU, HateVideo env):
   `python experiments/20261101_backbones/evaluate.py` → `runs/20261101_backbones/table.md`, `table.json`.

## 5. Results (development-selected; pooled ROC / pooled PR / within, test split, 4 fps)

Source of every new number: `runs/20261101_backbones/table.md` / `table.json`, transcribed from the evaluator outputs
`runs/20261101_backbones/<tag>/merged/full/metrics_ispvl_rnone.json` (Verdict),
`runs/20261101_backbones/<tag>/merged/alone/metrics_inone_rrank.json` (Per window),
`runs/20261101_backbones/decoded/<tag>/metrics.json` (\ours{}). Old = the paper's `tab:mllms` (r6-era reader, calibrated key).
Within on 84 HateMM / 99 HateClipSeg videos.

### 5.1 Main table

| backbone | corpus | Verdict old → new | Per window old → new | \ours{} old → new |
|---|---|---|---|---|
| Qwen3-VL-2B | HateMM | .847 / .571 → .851 / .580 | .544 / .271 / .628 → .543 / .271 / .632 | .854 / .629 / .738 → .864 / .617 / .752 |
| | HateClipSeg | .613 / .585 → .605 / .579 | .525 / .494 / .542 → .558 / .524 / .611 | .623 / .614 / .575 → .634 / .632 / .632 |
| Qwen3-VL-4B | HateMM | .871 / .627 → .871 / .626 | .549 / .275 / .631 → .551 / .276 / .658 | .883 / .660 / .751 → .882 / .659 / .777 |
| | HateClipSeg | .683 / .652 → .681 / .652 | .538 / .500 / .568 → .573 / .535 / .632 | .699 / .672 / .618 → .707 / .676 / .660 |
| Qwen3-VL-8B | HateMM | .882 / .646 → .880 / .638 | .550 / .271 / .628 → .548 / .274 / .636 | .894 / .691 / .764 → .899 / .692 / .780 |
| | HateClipSeg | .674 / .617 → .674 / .616 | .544 / .505 / .576 → .572 / .538 / .627 | .716 / .672 / .637 → .733 / .689 / .687 |
| Qwen3-VL-32B | both | not run (§5.4) | | |
| Qwen2.5-VL-7B | HateMM | .871 / .639 → .870 / .638 | .549 / .272 / .641 → .550 / .275 / .671 | .887 / .671 / .756 → .887 / .672 / .750 |
| | HateClipSeg | .690 / .647 → .693 / .648 | .540 / .502 / .566 → .566 / .524 / .621 | .699 / .668 / .588 → .707 / .679 / .629 |
| InternVL3.5-8B | HateMM | .877 / .626 → .878 / .635 | .543 / .271 / .618 → .553 / .277 / .651 | .888 / .673 / .739 → .889 / .672 / .756 |
| | HateClipSeg | .654 / .625 → .653 / .625 | .538 / .500 / .575 → .569 / .530 / .622 | .678 / .653 / .632 → .694 / .659 / .657 |
| LLaVA-OneVision-7B | HateMM | .878 / .641 → .877 / .637 | .548 / .274 / .632 → .538 / .268 / .610 | .881 / .656 / .733 → .886 / .674 / .750 |
| | HateClipSeg | .682 / .649 → .679 / .648 | .531 / .502 / .545 → .554 / .520 / .586 | .679 / .666 / .613 → .692 / .678 / .638 |
| Gemma-3-12B | HateMM | .831 / .497 → .836 / .515 | .551 / .271 / .640 → .545 / .268 / .646 | .863 / .629 / .781 → .858 / .610 / .783 |
| | HateClipSeg | .661 / .627 → .664 / .626 | .533 / .501 / .554 → .564 / .518 / .613 | .720 / .696 / .625 → .737 / .679 / .677 |

(Verdict within = .500 on every row.) The Qwen3-VL-8B \ours{} row equals the adopted decode
(`runs/20261031_window_evidence/decoded_main/adopted_rawkey/metrics.json`, .8989 / .6921 / .7799 and .7327 / .6895 / .6866):
this run's full reads equal the stored adopted reads on all 333 videos (max |Δ| 0.0, `table.json` consistency).

Reading (within, new): \ours{} − Per window = +.120 / +.119 / +.144 / +.079 / +.105 / +.140 / +.137 on HateMM and
+.021 / +.028 / +.060 / +.008 / +.035 / +.052 / +.064 on HateClipSeg (2B, 4B, 8B, Qwen2.5, InternVL, LLaVA, Gemma).
The Per-window read now carries the window's own frames and aligned words, so on HateClipSeg it is .59–.63 (old
.54–.58); on HateMM it stays .61–.67. \ours{} within rises over the old column on every backbone except Qwen2.5-VL
HateMM (−.006, inside the .01 floor); pooled PR falls on Qwen3-VL-2B HateMM (−.012) and Gemma (−.019 / −.017).

### 5.2 Porting checks

| backbone | check (b) on the smoke: max \|Δz\| (nats), min Spearman, token seam | first video of each full run (HateMM / HateClipSeg) | check (a) |
|---|---|---|---|
| Qwen3-VL-2B | .166, .95, identical | .148 / .102 | |
| Qwen3-VL-4B | .616, .983, identical | .616 / .394 | |
| Qwen3-VL-8B | .872 (verdict of non_hate_video_4), .983, identical | .197 / .275 | 0.0 on the 4 smoke videos; 0.0 on all 333 in the full run |
| Qwen2.5-VL-7B | .094, .95, identical | .094 / .073 | |
| InternVL3.5-8B | .203, .967, identical | .127 / .116 | |
| LLaVA-OneVision-7B | .068, .943, identical | .068 / .046 | |
| Gemma-3-12B | 1.158 (after the two fixes of §3), .983, identical | 1.158 / .184 | |

All below the 3-nat gate. Every smoke read equals the same video's read in the full run (max |Δ| 0.0 for every
backbone; InternVL / LLaVA-OneVision HateClipSeg ran after the full-length-cache change of §3, their smoke before it).
Smoke and arms smoke records: `runs/20261101_backbones/smoke/<tag>/`.

### 5.3 Runs (main table)

All on lab RTX 5090s through Slurm (`launch/lab23.sbatch` on sc474398 = uoa-lab3, `launch/labserver.sbatch` on
sc448960 = lab-server); results rsynced to uoa-lab1. Jobs (elapsed): smokes 1365 (q3vl-2b, 4b, q25vl; q3vl-8b failed on a
missing reference file) 4.4 min, 1368 (q3vl-8b, internvl, llava) 11.7 min, 1407 (gemma) 6.2 min; q3vl-2b 1369 / 1370
(8.9 / 8.2 min), q3vl-4b 1371 / 1372 (12.6 / 11.5), q25vl-7b 1373 / 1374 (19.8 / 18.0), internvl35-8b 1380 / 1381
(30.5 / 27.7), llava-ov-7b 1378 / 1379 (65.6 / 60.5), gemma3-12b 1408 / 1409 (69.8 / 63.2), q3vl-8b 1412 / 1413
(17.6 / 16.2). Main table ≈ 7.6 GPU-hours including smokes. Cost per video (full + alone): 2B ≈ 2.5 s (HateMM),
8B ≈ 4.9 s, InternVL ≈ 8.5 s, LLaVA-OneVision ≈ 18 s, Gemma ≈ 19 s; the alone read is ≈ 35–40 % of it.

### 5.4 Not run: Qwen3-VL-32B

bf16 weights are 66 GB, larger than a 32 GB RTX 5090; the 2026-09-10 study ran it on a campus A100. All three campus
ControlMaster sockets were dead for the whole session (`ssh -O check uoa-campus1/2/3`: no socket), and campus login is
token-only, so no campus job could be submitted. The table keeps the old 32B row until a campus run (one job per corpus,
`launch/` needs a campus sbatch) or a decision to use a quantized 32B.

### 5.5 Appendix table

(pending)

## 6. Test-read log (rule 10)

(pending)
