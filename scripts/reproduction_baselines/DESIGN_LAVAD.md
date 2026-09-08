# LAVAD baseline port (CVPR 2024)

Upstream: <https://github.com/lucazanella/lavad>, commit
`1ad46c666d1b3cfb262f3dd84769acf873285056` (audited 2026-08-21).

## Status

The cohort adapter, strict 1 fps score packer and shared-evaluator handoff are
implemented and CPU-smoke-tested. No full inference has been launched. A full
run is not a single-model inference: it is the released seven-stage pipeline
below and requires a separate legacy environment plus gated Llama-2 weights.

## Scientific role

LAVAD is training-free: it uses no examples or labels from the target corpus.
It is a surveillance-*anomaly* detector as released, not a hate detector. The
primary reproducibility arm therefore keeps the published law-enforcement /
suspicious-activity prompt verbatim. A prompt replacing anomaly with
hateful/offensive content would be a clearly named task-prompt adaptation,
not the released baseline, and must not silently replace the primary arm.

This is an external baseline and violates this project's own deployment cap:
for every temporal point it makes one Llama call to summarize retrieved frame
captions and a second call to score the summary, after five captioning-model
passes. That is reported as cost, not presented as our method design.

## Exact pipeline and why it is expensive

1. Run **five** BLIP-2 captioners independently on every sampled frame:
   OPT-6.7B (base and COCO), Flan-T5-XL (base and COCO), and Flan-T5-XXL.
2. Embed all candidate captions with ImageBind and build one FAISS index per
   video.
3. For each time point, decode a 10 s / 10-frame clip, embed it with ImageBind,
   retrieve the visually closest candidate caption, and call that the clean
   caption.
4. Run Llama-2-13B-chat **twice per time point**: first temporal summarization,
   then anomaly scoring in eleven discrete levels (0.0,...,1.0).
5. Embed every temporal summary with ImageBind and build a second FAISS index.
6. Decode/embed the local video clip again, retrieve the ten nearest temporal
   summaries, and similarity-weight their scores.
7. Rasterize and evaluate.

At 1 fps, `N` gold seconds mean approximately `5N` BLIP generations, `2N`
Llama generations, two video-ImageBind passes, two text-ImageBind passes, and
two per-video FAISS indexes. Intermediate JSON stores five raw captions,
cleaned captions, summaries, raw scores, neighbor identities and similarities.
The official README reports **2 x 64 GB A100**, and its Llama implementation
uses tensor-model-parallel checkpoint shards via `torchrun --nproc_per_node 2`.
The current host has one 32 GB RTX 5090, so the released command cannot run
unchanged. Its pinned stack (PyTorch 1.13, torchvision 0.14, transformers
4.31) also predates this GPU generation and conflicts with the project's
vLLM/transformers environment. A modern HF/vLLM rewrite could fit the 13B
model on one GPU, but would cease to be an exact released-code reproduction
and needs a separately documented equivalence check.

## Temporal port

The frozen gold grid is one frame at `t=0,1,...` while `t < wav_duration`.
`lavad/prepare.py` makes exactly one numbered JPEG per gold frame and writes
the four-column `VideoRecord` manifest. Run every upstream stage with
`frame_interval=1`; the 10 s clip and 10 sampled-frame defaults retain their
physical meaning. When audio outlives visual media, the final decodable image
is held, matching the already frozen CLIP feature extractor.

`lavad/pack_scores.py` rejects rather than crops/repeats a score vector whose
length differs from gold. It reproduces upstream's softmax weighting over the
ten refinement neighbors and writes `score_raw` and `score_refined` branches.
Then use `eval_baseline_scores.py`, so all frame metrics come from
`frame_eval_common`.

## Reproducible commands

```bash
# CPU audit / plumbing
/home/jehc223/venvs/SafetyContradiction/bin/python \
  scripts/reproduction_baselines/smoke_cpu_lavad.py

# Prepare one corpus (decoding only; resumable when complete)
CORPUS=hateclipseg bash scripts/reproduction_baselines/run_lavad.sh

# After the seven upstream stages have populated raw/refined artefacts
python scripts/reproduction_baselines/lavad/pack_scores.py \
  --corpus hateclipseg --raw-scores RAW_DIR \
  --refined-scores REFINED_DIR --similarities SIMILARITY_DIR \
  --output runs/legacy_1fps/lab1/reproduction/baselines/lavad/hateclipseg/scores.jsonl
python scripts/reproduction_baselines/eval_baseline_scores.py \
  --corpus hateclipseg \
  --scores runs/legacy_1fps/lab1/reproduction/baselines/lavad/hateclipseg/scores.jsonl \
  --json-out runs/legacy_1fps/lab1/reproduction/baselines/lavad/hateclipseg/frame_eval.json
```

## Deviations from upstream

- Input cohort and labels are replaced by frozen study manifests; labels are
  never exposed to LAVAD and are read only by the evaluator.
- Sampling is 1 fps with `frame_interval=1`, rather than original-video frames
  with an interval of 16, to land exactly on the common gold grid.
- Upstream sklearn evaluation is replaced by the shared evaluator.
- The released anomaly prompt remains the primary arm. No hate-adapted prompt
  has been run or selected on these test labels.
- No model architecture, retrieval rule, number of captioners, number of
  neighbors, score levels, or temporal window has been changed.
