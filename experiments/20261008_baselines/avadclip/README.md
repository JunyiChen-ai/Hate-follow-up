# AVadCLIP (IEEE TMM 2026), weakly supervised, 2026-10-08

Upstream: github.com/RowanSu/AVadCLIP @ d3f6e16, cloned into `third_party/AVadCLIP` (2026-10-08). The repo ships the
model, training and test scripts for XD-Violence and CCTV-Fights, but no CLIP or Wav2CLIP extraction script.
Port: `avadclip_port.py` (this directory) imports the audio-visual model `model_t.AVADCLIP` and the losses `CLAS2`,
`CLASM` unchanged from `third_party/AVadCLIP/src` and replaces only the data plumbing and the training loop's checkpoint
handling. The reported model is the audio-visual one (the repo's "teacher", `xd_train_t.py`); the distilled
visual-only "student" is for audio-absent input and is not run. Shared splits, labels, 4 fps rule and evaluation:
`../weaksup_common/README.md`.

Mechanism hypothesis: none (reference baseline; video-level labels of the target corpus's train split).

## Setup

| | |
|---|---|
| visual input | `clip_b16_1fps`: CLIP ViT-B/16 image embedding of the frame at t = i s, one row per second; the same rows VadCLIP and DSANet use |
| audio input | `data/wav2clip_1s`: Wav2CLIP (descriptinc/lyrebird-wav2clip v0.1.0-alpha checkpoint, encoder + audio_transform, frozen) on the 1 s audio window [i, i+1) s that matches visual row i; i.e. `wav2clip.get_model(frame_length=16000, hop_length=16000)` on the whole 16 kHz waveform (spectrogram normalisation over the video). Extracted by `../weaksup_common/extract_audio.py` |
| classes / prompts | 2: "normal content", "hateful content" (the VadCLIP port's prompts); text-orthogonality term divided by classes − 1 instead of 6 |
| hyper-parameters | the upstream XD-Violence preset (`src/xd_option.py`, teacher) on all three corpora: embed 512, width 512, 1 head, 2 layers, visual and audio length 256, attn-window 4, prompt 10/10, 10 epochs, batch 96, lr 1e-5, AdamW, MultiStepLR [3, 6, 10] × 0.1, loss3 weight 1e-4; classes 7 → 2 |
| selection | none: the last of the 10 epochs is kept (`--select last`); the validation split is not read |
| score | `score_align` = 1 − softmax(logits2)[:, 0] (upstream's AP2, the number its training loop selects on), headline; `score_mlp` = sigmoid(logits1) reported as `avadclip_mlp` |
| seeds | 2025 / 234 / 3407 |
| environment | conda HateVideo, torch 2.7.1 + cu128, RTX 5090 (Slurm); AVadCLIP's code runs unchanged on torch 2.7 (its `DistanceAdj` hard-codes `cuda`) |

**What the original script did and what changed.**
- Upstream `xd_train_t.py` calls `test()` on the test set after every epoch, saves the checkpoint with the best TEST
  AP (alignment branch) and reloads it at the end of every epoch. Here the test set is never opened during
  training, nothing is reloaded, and the last epoch of the fixed 10-epoch XD schedule is kept (coordinator
  instruction 2026-10-08).
- Upstream reads features from CSV lists (`path`, `audio_path`, label string); here from `data/weaksup_1fps` and
  `data/wav2clip_1s` by id, with integer video labels turned into one-hot [normal, hateful].
- Snippet unit. Upstream has one CLIP and one Wav2CLIP vector per 16-frame snippet. Here the unit is the 1 s row
  shared by VadCLIP, DSANet and AVadCLIP (one CLIP frame per second), and Wav2CLIP is computed on the matching 1 s
  audio window, so the two modalities line up row by row as the model requires. Re-extracting both modalities at
  16-frame snippets would give AVadCLIP different visual input from VadCLIP and DSANet; that was not done.
- Upstream's test-time ×16 upsampling to frames is dropped (one row is one second); scores go to the 4 fps grid by
  the shared rule.

Splits: HateMM 744 / 109 / 215, HateClipSeg (p11) 237 / 39 / 118, DeHate 4680 / 668 / 1151 (val unused);
(train ∪ val) ∩ test = ∅ (`runs/20261008_baselines/weaksup_inputs/split_check.json`).

## How to run

```bash
# uoa-lab1, inside Slurm: launch/lab1_mil_mhl_avadclip.sbatch (after launch/lab1_audio_features.sbatch)
python experiments/20261008_baselines/weaksup_common/run_method.py --method avadclip --corpus {hatemm,hateclipseg,dehate}
```

Outputs: `runs/20261008_baselines/avadclip/<Dataset>/seed<k>/` and `.../<Dataset>/summary.json`.

## Results

Seed mean ± sd (n − 1) over seeds 2025 / 234 / 3407, transcribed from `runs/20261008_baselines/avadclip/<Dataset>/summary.json`, which is built from each seed's `metrics.json` (canonical evaluator). Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC.

| row | corpus | ROC | PR | within | per seed (ROC / PR / within; 2025, 234, 3407) | host |
|---|---|---:|---:|---:|---|---|
| AVadCLIP (score_align, headline) | HateMM | 0.6369 ± 0.0225 | 0.3503 ± 0.0398 | 0.5077 ± 0.0358 | 0.6226/0.3087/0.5233; 0.6629/0.3879/0.5331; 0.6252/0.3545/0.4667 | sc474397 |
| AVadCLIP (score_align, headline) | HateClipSeg | 0.4996 ± 0.0096 | 0.4596 ± 0.0091 | 0.5291 ± 0.0184 | 0.5008/0.4578/0.5417; 0.4896/0.4515/0.5376; 0.5086/0.4694/0.5080 | sc474397 |
| AVadCLIP (score_align, headline) | DeHate | 0.6104 ± 0.0168 | 0.1112 ± 0.0017 | 0.5343 ± 0.0314 | 0.6008/0.1094/0.5134; 0.6006/0.1114/0.5190; 0.6298/0.1128/0.5704 | sc474397 |
| AVadCLIP score_mlp | HateMM | 0.6772 ± 0.0214 | 0.4345 ± 0.0359 | 0.4764 ± 0.0268 | 0.6757/0.4226/0.5048; 0.6992/0.4748/0.4727; 0.6566/0.4060/0.4517 | sc474397 |
| AVadCLIP score_mlp | HateClipSeg | 0.5574 ± 0.0229 | 0.5163 ± 0.0274 | 0.5326 ± 0.0065 | 0.5377/0.4863/0.5377; 0.5825/0.5401/0.5348; 0.5520/0.5224/0.5253 | sc474397 |
| AVadCLIP score_mlp | DeHate | 0.6419 ± 0.0345 | 0.1385 ± 0.0231 | 0.5057 ± 0.0172 | 0.6031/0.1219/0.5144; 0.6539/0.1288/0.5167; 0.6688/0.1649/0.4859 | sc474397 |

Host: uoa-lab1 (sc474397), Slurm 291, 2026-10-08. Every cohort video scored on every GT frame (`coverage.json`).

## Change for oracle test selection (2026-10-09)

`avadclip_port.py train` accepts `--save-every-epoch` (default off): it also writes `model_eNN.pth` (non-CLIP weights, as the final save) after every epoch. Saving draws no random number. Used only by `../oracle_test_selection/` (checkpoint and branch chosen on TEST labels, an upper bound for the baselines; see that README). The rows above are unchanged.
