# LAVAD (CVPR 2024): HateClipSeg completion and DeHate

Label-free baseline. No training, no tuning, no label read. HateMM is the re-evaluated Retrieval-hate campaign
output (`runs/20261008_baselines/lavad/metrics.json`, see `../README.md`). This directory completes HateClipSeg
(one video, `bit_ZaY0S1anrdep`, had 12 unscored frames) and adds DeHate with the campaign's setup.

## How the campaign ran LAVAD (the setup reproduced here)

Source: Retrieval-hate `archive/idea-stage/REPRO_CAMPAIGN_RESULTS.md` §K and
`archive/idea-stage/repro_campaign/MODEL_ASSETS_STATUS.md` (uoa-lab2, read only).

- Code: `scripts/repro_campaign/{blip2_caption.py, lavad_chain.py, run_lavad_wave1.sh}` on top of
  `third_party/lavad` @ 1ad46c66. Machine: one RTX 5090 (uoa-lab2), 2026-08-19..22, shared `HateVideo` env
  (torch 2.7.1+cu128, **transformers 4.49.0**, bitsandbytes 0.49.2, accelerate 1.5.2).
- The official LLM step (`torchrun --nproc_per_node 2`, Meta's model-parallel loader) was not used. Both
  `Llama.build` calls go through the `llama_hf` shim, which loads `NousResearch/Llama-2-13b-chat-hf` (ungated
  mirror of Llama-2-13b-chat) with HF transformers in **NF4** (bitsandbytes 4-bit, double quantisation, bf16
  compute; `LLAMA_HF_4BIT=1`) on one GPU, chat layout `[INST] <<SYS>>…`, `max_seq_len` 512, **greedy** decoding,
  batch 48 with `expandable_segments`.
- Captioner: `Salesforce/blip2-opt-6.7b-coco` only (1 of LAVAD's 5), fp16, unconditional, batch 48, on 1 fps
  frames (`ffmpeg -vf fps=1 -q:v 2`); every 1 fps frame is a centre. ImageBind-Huge (fp32) for caption cleaning
  and refinement; the caption and summary indices are built **per video** (official `create_index.py`), so a
  video's output does not depend on the other videos processed.
- Prompts, verbatim from LAVAD's `04_query_llm.sh` (no hate wording):
  - summary: "Please summarize what happened in few sentences, based on the following temporal description of a
    scene. Do not include any unnecessary details or descriptions."
  - score (system): "If you were a law enforcement agency, how would you rate the scene described on a scale from
    0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities? Please provide
    the response in the form of a Python list and respond with only one number in the provided list below [0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '['
    and end with  ']'."
- Reported curve: `base` = stage-06 refined score (softmax(similarity)-weighted mean of the scores of the 10
  nearest summaries of the same video), 1 fps. Llama-2 refusals are masked, not interpolated; a centre whose 10
  neighbours all refused stays unscored (that is what happened at 147–149 s of `bit_ZaY0S1anrdep`).

## The port (this directory)

`blip2_caption.py`, `lavad_chain.py`, `shim/llama_hf/__init__.py` are the campaign files with paths changed
(repo root, `data/frames_1fps/`, `data/blip2_captions_1fps/`, `runs/20261008_baselines/lavad/{work,curves}/`),
the video list read from `--ids-file`, and the `curves` stage no longer reading GT labels (the campaign split its
refusal count by label). `run_chain.sh` = `run_lavad_wave1.sh` with the same stage order, batch sizes and prompt.
The HateVideo env on uoa-lab2 now has transformers 4.57.6; a first 6-video rerun with it (Slurm 286) gave BLIP-2
captions that differ from the campaign's on 3–11 of 214–289 frames per video (kept in
`runs/20261008_baselines/lavad/captions_tf4.57.6_job286/`). The runs therefore use
`.cache/envs/lavad_tf449` (venv with `--system-site-packages` on HateVideo, only transformers 4.49.0 and tokenizers
0.21.4 installed; log `runs/_setup_uoa-lab2/lavad_tf449_venv.log`).

Grid and fallbacks (`finalize.py`): 1 fps → 4 fps, frame i takes sample floor(i/4), frames past the last sample
hold its value. F3 (run_plan.md §1.3): an unscored 1 fps sample takes the nearest scored sample of the same video
(ties: the earlier one); F2 if a video has no scored sample.

## HateClipSeg completion

`lab2_lavad_hcs.sbatch`: rerun `bit_ZaY0S1anrdep` and 5 overlap videos (the first 5 cohort ids), compare every
stage with the campaign's files (`compare_rerun.py` → `runs/20261008_baselines/lavad/HateClipSeg_rerun_check.json`).
If the overlap videos reproduce (same NaN pattern, max |diff| ≤ 1e-6), HateClipSeg = campaign curves for 117
videos + the rerun curve; otherwise all 118 are rerun (`lab2_lavad_hcs_full.sbatch`, run_plan.md L1b).

Check result (Slurm 294, uoa-lab2, transformers 4.49.0 venv): **not reproduced**. Per video, rerun vs campaign:

| video | captions differing | clean / summary / score entries differing | `base` max abs diff | `base` Pearson |
|---|---|---|---|---|
| bit_ZaY0S1anrdep (target) | 3 / 277 | 22 / 35 / 6 | .020; unscored samples 147–149 and 223–225 (campaign: 147–149) | .948 |
| bit_0EHvMSiEHVoc | 7 / 227 | 19 / 19 / 4 | .091 | .940 |
| bit_0nXuyV2rypaf | 1 / 259 | 0 / 0 / 0 | .0099 | .999 |
| bit_1R6aAMuJnH2t | 3 / 214 | 21 / 23 / 7 | .010 | .969 |
| bit_3rLvnyheeW6V | 4 / 289 | 0 / 0 / 0 | .055 | .996 |
| bit_4BVcDK677Zp7 | 3 / 234 | 21 / 34 / 12 | .071 | .612 |

Two sources: BLIP-2 captions differ on 1–7 frames per video even with the campaign's transformers version
(with 4.57.6: 3–11), and the ImageBind refinement differs even where captions, summaries and scores are identical
(bit_0nXuyV2rypaf, bit_3rLvnyheeW6V). The cause below the library level (CUDA kernels, driver) was not isolated.
So the campaign's 117 curves are not mixed with rerun curves: all 118 HateClipSeg videos come from the port
(`lab2_lavad_hcs_full.sbatch`). The rerun still leaves `bit_ZaY0S1anrdep` with unscored samples (Llama-2 refused
all 10 neighbours); F3 fills them. Full record: `runs/20261008_baselines/lavad/HateClipSeg_rerun_check.json`.

Status: the full rerun of all 118 videos runs on uoa-campus2 in Slurm 24597 (section "Campus run" below). The 6
lab2 videos of the check are redone there as well, so that every HateClipSeg curve comes from one machine and GPU
type. The job writes `runs/20261008_baselines/lavad/HateClipSeg/metrics.json` with `finalize.py --dataset
HateClipSeg --all-rerun`.

## DeHate

The 1151 cohort videos (30.7 h of video), second half of Slurm 24597 on uoa-campus2 (section "Campus run"). The
job ends with `finalize.py --dataset DeHate`. Estimated cost before the run: 14–20 GPU-h on a 5090 (campaign rate
0.44 GPU-h per video hour; the 6-video check ran at 0.69 with less prompt-cache reuse).

## Campus run (2026-10-08)

Moved from uoa-lab2 to **uoa-campus2 (foscsmlprd02), one NVIDIA A100-SXM4-80GB**, because the lab account may run
only 2 GPU jobs at a time (QOS `gpu2`) and 7 jobs were queued. `lab2_lavad_hcs_full.sbatch` and
`lab2_lavad_dehate.sbatch` were never submitted. uoa-campus1 was not usable: its `/data` quota for jehc223 is over the
soft limit with the grace period expired (296G of 290G), so no file can be written there.

- Job: `cd /data/jehc223/Hate-follow-up && sbatch experiments/20261008_baselines/launch/campus_lavad.sbatch` on
  uoa-campus2, Slurm 24597 (started 2026-10-08 10:31 NZDT, commit d9db913). One job runs HateClipSeg (all 118)
  and then DeHate (1151), each `run_chain.sh` + `finalize.py`, the commands of the two lab2 files. Log
  `runs/20261008_baselines/lavad/slurm_24597.out`.
- Same setup as lab2: the port in this directory, the verbatim LAVAD prompts, `Salesforce/blip2-opt-6.7b-coco` fp16,
  `NousResearch/Llama-2-13b-chat-hf` in NF4 (bitsandbytes 0.49.2, double quantisation, bf16 compute) through the
  `llama_hf` shim, greedy decoding, batch 48, ImageBind-Huge (`data/assets/imagebind/imagebind_huge.pth`,
  `third_party/lavad/libs/ImageBind` copied from uoa-lab1), 1 fps frames copied from uoa-lab2 (real files, see
  `data/frames_1fps/PROVENANCE.md` on uoa-campus2). The only differences are the GPU (A100 instead of RTX 5090) and the env build.
- Env: campus2 has no HateVideo env, so `launch/campus_lavad_env.sh` builds `.cache/envs/lavad_tf449` as a standalone
  Python 3.11.8 env with lab2's HateVideo package versions as pip constraints
  (`launch/campus_lavad_env_constraints.txt`; torch 2.7.1+cu128, torchvision 0.22.1, torchaudio 2.7.1, accelerate
  1.5.2, bitsandbytes 0.49.2, numpy 1.26.4, pytorchvideo 0.1.5, opencv-python 4.11.0.86, scikit-learn 1.5.2) plus
  the lab2 venv's transformers 4.49.0 and tokenizers 0.21.4, and writes the same 4-line torchvision
  `functional_tensor.py` shim that lab2's HateVideo carries for pytorchvideo. Logs and `pip freeze` in
  `runs/_setup_uoa-campus2/`.
- Models: downloaded on campus2 into `.cache/hf` with `hf download` (Llama-2 and BLIP-2 without the `.bin` shards,
  as in lab2's cache). File names and sizes equal lab2's cache; config, tokenizer and index files are byte-identical.
- Port change for the campus run: `blip2_caption.py` no longer needs the video file when the video's 1 fps frame
  directory already holds JPEGs (the video is read only to extract frames; campus2 holds frames, not videos). The
  captions are computed from the same JPEGs as before.
- Smoke test on HateClipSeg `bit_0nXuyV2rypaf` and DeHate `4PmH5EgjyduF`: Slurm 24586 found the missing torchvision
  shim and the video-file check, 24595 ran stages 01–04 and found the missing opencv, 24596 finished stages 05–06
  and the curves for both videos. Outputs are in
  `runs/20261008_baselines/_smoke_campus/lavad/` (moved there so that the real run recomputes them). A100 versus the
  lab2 5090 rerun (Slurm 294) on `bit_0nXuyV2rypaf`: BLIP-2 captions differ on 17 of 259 frames (lab2 versus the
  campaign: 1 of 259); clean / summary / score entries differ on 13 / 22 / 36 of 259; `base` curve max abs diff
  .033, Pearson .931. This is the same kind of hardware-level difference the lab2 check found, so the HateClipSeg and
  DeHate curves all come from the A100 run and are not mixed with lab2 or campaign curves.
- Speed on the A100 (HateClipSeg): BLIP-2 captions 20.9 frames/s (28,242 frames in 22.5 min), caption cleaning 26 min,
  Llama-2 summaries 1.17 generations/s (the lab2 5090 ran 3.2/s in Slurm 294). The A100 is shared with small
  processes of other users. Estimate from these rates: HateClipSeg done on the evening of 2026-10-08 (NZDT), DeHate
  about 40 h later (around midday 2026-10-10); about 50 GPU-h in total instead of the 18–25 estimated for the 5090.
- On uoa-lab1, the lab2 check's per-stage files were moved to
  `runs/20261008_baselines/lavad/lab2_rerun_check_job294/` before the campus outputs were copied back (see the
  README.txt there); `HateClipSeg_rerun_check.json` and `slurm_294.out` stay where they were.

Remaining steps after Slurm 24597 ends (check `slurm_24597.out` for `JOB_DONE`, `FAILED`, `Traceback`; if it stopped,
resubmitting the same sbatch continues, because every stage skips finished videos):

```
# on uoa-lab1
rsync -a uoa-campus2:/data/jehc223/Hate-follow-up/runs/20261008_baselines/lavad/ runs/20261008_baselines/lavad/
rsync -a uoa-campus2:/data/jehc223/Hate-follow-up/data/blip2_captions_1fps/ data/blip2_captions_1fps/
python experiments/20261008_baselines/campus_lab1_check.py \
  runs/20261008_baselines/lavad/HateClipSeg:HateClipSeg runs/20261008_baselines/lavad/DeHate:DeHate
# then write data/blip2_captions_1fps/PROVENANCE.md (generated on uoa-campus2 by blip2_caption.py in
# .cache/envs/lavad_tf449, Slurm 24597) and fill the results below
```

## Results (HateClipSeg, DeHate)

(pending: Slurm 24597 on uoa-campus2)

## Transcript (audio-visual) variant, 2026-10-09

User decision 2026-10-09: the label-free baselines must see the picture and the speech. LAVAD-AV
(`lavad_av.py`) changes only LAVAD's scoring stage (04b); the visual-only rows above are unchanged and keep their
directories.

### What the LLM sees

Llama-2-13b-chat (NF4, greedy, max_seq_len 512, batch 48, `lavad_chain.Scorer`) scores every 1 fps centre c of a
video. LAVAD's law-enforcement question is replaced by the hate rules of `hate_query.md` §1 (verbatim); LAVAD's
rating sentence and its format instruction are kept. The reported run uses prompt version `moderator_prefill`
(`lavad_av.py` `PROMPT_VERSIONS`; why, in "Attempt 1 and the refusal pilot" below): the system turn starts with the
system message of the shared hate query's Yes/No family (`experiments/20260910_spvl/spvl.py` `SYSTEM_MESSAGE`), and
the assistant turn is started with ` [` after `[/INST]`, so the model writes the number that LAVAD's format
instruction asks for (the parsed answer is ` [` + generation). System turn (exact text):

```
You are a content moderation analyst. Answer based strictly on observable evidence.

Based on this platform's hate speech policy, the following types of content are prohibited:
1. Encourage violence against groups based on protected status
2. Incite hatred against groups based on protected status
3. Dehumanize groups by comparing to non-human entities
4. Praise or glorify violence against protected groups
5. Use slurs and stereotypes that incite hatred (including through humor, irony, or coded language)
6. Claim groups are inferior based on protected status
7. Promote hateful supremacism
8. Make conspiratorial claims about groups based on protected status
9. Deny or minimize major violent events

How would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with content that violates any of the above rules? Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'.
```

User turn: LAVAD's `f"{summary}."` (the temporal summary of the clip's BLIP-2 captions) plus a second block with the
clip's speech; then the prefilled start of the answer:

```
[user]      <temporal summary>.

            Speech in this clip: <Whisper large-v3 text of [max(c-5, 0), min(c+5, n)) s, or "(no speech)">
[assistant]  [
```

The span is LAVAD's 10 s clip of centre c on the 1 fps grid (`lavad_chain.window_frames`, n = last centre + 1).
Transcript: `data/asr_whisper_large_v3/<DS>/timestamped_chunks.jsonl`, segments loaded with `qwen3_text/text_llm.py`
`load_segments` (untimed chunks kept: missing end = next chunk's start or the video duration), cut to the span with
`src/video_inputs.py` `window_text` (`../transcript_windows.py`). The answer is parsed with LAVAD's `[x]` rule;
an unparsed answer (refusal) is masked, as in the visual-only run.

Prompt budget. LAVAD's max_seq_len is 512, and `Scorer` truncates a longer prompt from the right (cutting the
`[/INST]` tag; the official Meta loader would stop with an assertion). The rules make the system turn about 150 tokens
longer than LAVAD's, so a prompt over 496 tokens (16 left for the answer) keeps the first speech words that fit,
followed by ` ...`; if it is still too long with the speech cut to `...`, the summary is cut the same way. Dry run with
the Llama-2 tokenizer on uoa-lab1 (`lavad_av.py lengths`): median prompt 381 / 381 / 385 tokens (HateMM / HateClipSeg
/ DeHate); speech cut in 934 of 29,243 / 544 of 28,242 / 2,244 of 110,449 prompts; summary cut in 1 / 10 / 33. This
dry run used prompt version `rules`; the reported version's system message makes prompts 19 tokens longer, and its
actual counts are under Results (`runs/20261008_baselines/lavad_av/prompt_stats_<DS>.json`).

### Attempt 1 and the refusal pilot

Attempt 1 used prompt version `rules`: the same system turn without the first sentence, and no prefill.
- HateMM (uoa-lab2, Slurm 366, 2026-10-10 22:09–23:45): Llama-2 refused 4,387 of 29,243 centres (15 %), mostly
  "I apologize, but I cannot provide a rating for the scene ... as it contains hate speech and derogatory
  language". In 11 of 215 videos every refined sample was unscored (for every centre, all 10 neighbour summaries
  were refused), so those videos had no score at all. F2 would be needed for 5.1 % of the cohort (> 1 %), so the
  exact-cohort rule stopped the run (run_plan.md §1.3); no metrics.
- HateClipSeg (uoa-campus2, Slurm 24651, 22:00–00:11): 1,648 of 28,242 centres refused (5.8 %), no video fully
  refused, evaluated .5884 / .5883 / .5116. Not a reported number: the three corpora must share one prompt.
- Outputs kept in `runs/20261008_baselines/lavad_av/attempt1_rules/` (HateMM: `slurm_366.out`; HateClipSeg:
  `slurm_24651.out`).

Refusal pilot (uoa-lab2, Slurm 367, 2026-10-10 23:50–00:09; `launch/lab2_lavad_av_pilot.sbatch`): 17 HateMM videos =
the 11 fully refused ones + the first 6 other cohort ids in sorted order; only refusal counts were read, no label
and no metric (`runs/20261008_baselines/lavad_av/pilot/<version>/prompt_stats_HateMM.json`).

| version | refused centres (of 3,081) | unscored refined samples | videos with no scored sample |
|---|---:|---:|---:|
| `rules` (attempt 1, same 17 videos) | 1,034 | 1,702 | 11 |
| `moderator` (system message added) | 500 | 441 | 3 |
| `moderator_prefill` (system message + answer started with ` [`) | 11 | 0 | 0 |

`moderator_prefill` was fixed for all three corpora before any of them was scored with it. Its 11 remaining refusals
are " [I apologize, but I cannot provide a rating ...". Answer values in the pilot: 0, .2, .4 and .8 dominate.

### What is reused, and the decision on the summaries

Decision: the speech goes into the scoring prompt only. The summary prompt sees captions only, so the captions,
caption cleaning, temporal summaries (stage 04a) and the summary index with each centre's 10 nearest summaries and
their ImageBind similarities (stages 05/06, `refined`) are the visual-only run's files, unchanged. Only stage 04b is
rerun; the refined curve `base` is LAVAD's softmax(similarity)-weighted mean of the **new** scores of the same 10
neighbours (refusals masked), then F3 (nearest scored 1 fps sample) and 1 fps → 4 fps as in `finalize.py`.

| corpus | reused summaries + neighbours | origin |
|---|---|---|
| HateMM | `data/retrieval_hate_repro/repro_lavad_work/HateMM/{summary,refined}/` | Retrieval-hate campaign, uoa-lab2 RTX 5090, 2026-08 (the visual-only HateMM row is that campaign's curve); copied read-only from uoa-lab2 `~/Retrieval-hate/data/lavad/`, see `data/retrieval_hate_repro/PROVENANCE.md` |
| HateClipSeg | `runs/20261008_baselines/lavad/work/{summary,refined}/HateClipSeg/` (on uoa-lab2 read from a copy, `runs/20261008_baselines/lavad_av/visual_from_campus2/`, made with `rsync -a` from uoa-lab1; uoa-lab2's own `runs/20261008_baselines/lavad/work/*/HateClipSeg/` hold the 6 videos of Slurm 294 and were not touched) | uoa-campus2 A100, Slurm 24602 (visual-only row above) |
| DeHate | `runs/20261008_baselines/lavad/work/{summary,refined}/DeHate/` | uoa-campus2 A100, Slurm 24603 (visual-only row above); BLIP-2 captions of that job are in campus2 `data/blip2_captions_1fps/DeHate/` |

Reuse check (`lavad_av.py check-reuse`): recomputing the visual-only `base` curve from the reused visual-only
`score` + `refined` files with this file's arithmetic reproduces the stored visual-only curve on every cohort video:
HateMM 215/215 against the campaign curves, max |diff| 0 (uoa-lab1, `reuse_check_HateMM.json`); HateClipSeg 118/118,
max |diff| 2.2e-16 on uoa-lab2 (`reuse_check_HateClipSeg.json`, the machine of the AV run; uoa-lab1
`reuse_check_HateClipSeg_lab1.json`: 2.2e-16; uoa-campus2 inside Slurm 24651: 0); DeHate 1151/1151, max |diff| 0 on
uoa-campus2 inside Slurm 24651 (uoa-lab1 `reuse_check_DeHate_lab1.json`: 3.3e-16). Files in
`runs/20261008_baselines/lavad_av/`.

Caveat on ties. LAVAD's scores take 11 values and the refined curve averages them, so many frames tie exactly. Float
differences at the 1e-16 level change how such ties are broken: on uoa-lab1, the HateClipSeg visual-only predictions
rebuilt from the same files with this arithmetic (curves identical within 2.2e-16) evaluate to .5741 / .5469 / .5160
instead of the stored .5737 / .5464 / .5129. Each corpus's LAVAD-AV curves are computed on the machine that scored
it: DeHate on uoa-campus2 (the machine of its visual-only curve), HateMM on uoa-lab2 (the machine type of the
campaign's visual-only curve), HateClipSeg on uoa-lab2 (its visual-only curve is from uoa-campus2). Differences of
this size between the AV and visual-only rows can come from tie breaking alone. The scores themselves also depend
on the GPU: Llama-2 NF4 answers differ for a few % of prompts between the A100 and the RTX 5090 (section "Campus run"
above).

### Host, job, commands

Env on both machines: `.cache/envs/lavad_tf449` (transformers 4.49.0, tokenizers 0.21.4, torch 2.7.1, bitsandbytes
0.49.2; the env of the visual-only runs), `NousResearch/Llama-2-13b-chat-hf` from the machine's HF cache. No job read a
label; the evaluator ran once per corpus at the end of the job (`lavad_av.py finalize`).

| corpus | machine, GPU | job | when (NZDT) | code |
|---|---|---|---|---|
| HateMM | uoa-lab2 (sc474399), RTX 5090 | Slurm 369, `sbatch experiments/20261008_baselines/launch/lab2_lavad_av.sbatch HateMM` | 2026-10-11 00:10–01:07 | score 2c8dc1b, curves + finalize 027e7cd (only `--visual-run` added) |
| HateClipSeg | uoa-lab2 (sc474399), RTX 5090 | Slurm 370, `sbatch experiments/20261008_baselines/launch/lab2_lavad_av.sbatch HateClipSeg runs/20261008_baselines/lavad_av/visual_from_campus2` | 2026-10-11 01:07–02:06 | 027e7cd |
| DeHate | uoa-campus2 (foscsmlprd02), A100-SXM4-80GB | Slurm 24651, `sbatch experiments/20261008_baselines/launch/campus_av.sbatch imagebind HateClipSeg HateMM DeHate` | 2026-10-10 21:27 – 2026-10-11 05:12 (DeHate scoring 00:13–05:11) | job started at 27e1c27; the DeHate scoring process waited on `hold_score.txt` for 1 min and restarted at 2c8dc1b |

How the runs ended up split. Slurm 24651 (submitted 2026-10-09 20:33, started 2026-10-10 21:27 after about 25 h in
the campus2 queue) first extracted the ZS-ImageBind-AV DeHate vision embeddings (see
`../zs_imagebind_audio/README.md`), then scored HateClipSeg with attempt 1. HateMM was moved to uoa-lab2 while that
ran (`elsewhere_HateMM.txt`, a lab GPU had become free; HateMM's visual-only curve is the campaign's lab2 RTX 5090
run). When attempt 1 failed on HateMM, the campus job's DeHate scoring was held (`hold_score.txt`) until the prompt
was fixed, and restarted with `moderator_prefill`. The plan to restart it as HateClipSeg + DeHate (`after_hold.json`)
was not read, because the waiting process was still running the code from before that option existed (file kept as
`after_hold.unused.json`); HateClipSeg was therefore rescored with `moderator_prefill` on uoa-lab2 (Slurm 370).
Pilot: Slurm 367 (uoa-lab2). The visual-only inputs copied to uoa-campus2 and uoa-lab2 for these jobs were deleted
afterwards; the results were copied back to uoa-lab1 with `rsync -a` and re-checked there
(`python experiments/20261008_baselines/campus_lab1_check.py runs/20261008_baselines/lavad_av/<DS>:<DS>`: exact cohort,
canonical evaluator into `metrics_lab1.json`, identical to the job's `metrics.json` on all three corpora).

GPU time: uoa-campus2 A100 7.75 h for Slurm 24651 (ZS-ImageBind-AV extraction 0.55 h, attempt-1 HateClipSeg 2.2 h,
DeHate 5.0 h); uoa-lab2 RTX 5090 3.82 h (Slurm 366 attempt-1 HateMM 1.59 h, 367 pilot 0.32 h, 369 HateMM 0.94 h, 370
HateClipSeg 0.98 h). Llama-2 rate with `moderator_prefill`: 8.0 / 7.4 generations/s on the RTX 5090 (HateMM /
HateClipSeg), 5.9 on the A100 (DeHate).

Outputs (`runs/20261008_baselines/lavad_av/`): `<DS>/` (`predictions.jsonl`, `coverage.json`, `config.json`,
`metrics.json`, `metrics_lab1.json`, `lab1_check.json`, `run.log`); `work/{score,score_refusals,speech}/<DS>/` (per
centre: score, refusal text, the speech text that was shown); `curves/<DS>/` (1 fps `raw` and `base`);
`prompt_stats_<DS>.json`, `refusal_stats_<DS>.json`, `reuse_check_*.json`; job logs `slurm_{24651,366,367,369,370}.out`;
`attempt1_rules/`, `pilot/`.

### Results

Canonical evaluator, exact cohorts; pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC. Nothing
selected on labels; the prompt version was chosen on refusal counts only (pilot above). Not development-selected.

| corpus | LAVAD-AV (speech + hate rules) | LAVAD visual-only (rows above) | videos / frames (within defined) | refused centres | F3 / F2 videos | source |
|---|---|---|---|---:|---|---|
| HateMM | .7297 / .5021 / .5152 | .5588 / .2906 / .4814 | 215 / 116,975 (84) | 77 of 29,243 | 0 / 0 | `runs/20261008_baselines/lavad_av/HateMM/metrics.json`; visual `runs/20261008_baselines/lavad/metrics.json` |
| HateClipSeg | .6145 / .6040 / .5348 | .5737 / .5464 / .5129 | 118 / 113,002 (99) | 6 of 28,242 | 0 / 0 | `.../lavad_av/HateClipSeg/metrics.json`; visual `.../lavad/HateClipSeg/metrics.json` |
| DeHate | .6080 / .1244 / .4978 | .5311 / .0823 / .5138 | 1151 / 441,345 (222) | 3 of 110,449 | 0 / 0 | `.../lavad_av/DeHate/metrics.json`; visual `.../lavad/DeHate/metrics.json` |

No refined sample was left unscored, so neither F3 nor F2 was used. Prompts whose speech was cut to fit 496 tokens:
1,174 / 740 / 3,342 (51,179 / 38,770 / 172,626 words dropped); summaries cut: 9 / 53 / 101; none truncated by
`Scorer`. Curve frames past the last 1 fps sample (holding it): 253 / 170 / 946.

Reading. The AV row differs from the visual-only row in two things at once: the scoring prompt (hate rules,
moderation system message, prefilled answer, instead of the law-enforcement question) and the speech. No row with the
new prompt and without speech was run, so the gain cannot be split between the two. Pooled ROC and PR rise on all
three corpora; within-video ROC rises on HateMM and HateClipSeg and falls on DeHate (.5138 → .4978).
