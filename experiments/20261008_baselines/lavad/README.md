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
