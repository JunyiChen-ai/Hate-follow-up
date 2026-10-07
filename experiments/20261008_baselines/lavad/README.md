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

Status (2026-10-08 06:50): the full HateClipSeg rerun is not submitted yet (one pending job per partition; Slurm 303
is pending on uoa-lab2). Next: `sbatch experiments/20261008_baselines/launch/lab2_lavad_hcs_full.sbatch` on uoa-lab2
once 303 is running. Estimated 4–5 GPU-h (the 6-video check took 17.5 min for 0.42 h of video; the 6 videos are
not redone). The job ends with `finalize.py --dataset HateClipSeg --all-rerun`, which writes
`runs/20261008_baselines/lavad/HateClipSeg/metrics.json`.

## DeHate

`lab2_lavad_dehate.sbatch`: the 1151 cohort videos (30.7 h of video). Frames are hard-linked into
`data/frames_1fps/DeHate/` on uoa-lab2. Status (2026-10-08 06:50): not submitted. Submit on uoa-lab2 once the full
HateClipSeg job is running. Estimated 14–20 GPU-h (campaign rate 0.44 GPU-h per video hour; the 6-video check ran
at 0.69 with less prompt-cache reuse). The job is resumable (every stage skips finished videos). It ends with
`finalize.py --dataset DeHate`. Then rsync `runs/20261008_baselines/lavad/` and `data/blip2_captions_1fps/` back to
uoa-lab1, and write `data/blip2_captions_1fps/PROVENANCE.md` (generated on uoa-lab2 by `blip2_caption.py` in the
`lavad_tf449` venv).
