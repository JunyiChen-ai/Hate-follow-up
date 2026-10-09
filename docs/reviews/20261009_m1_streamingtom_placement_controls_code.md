# Code review (rule 6): StreamingTOM placement controls, 2026-10-09

Reviewer: one independent agent (same model as the main session), before any GPU run. Files:
`experiments/20261007_m1_streamingtom/{placement_controls,placement_analyze,placement_selfcheck}.py`,
`launch/placement_lab3.sbatch`, `launch/run_placement_analysis.sh`. Checked against `src/stance_cache.py`,
`src/mllm_judge.py`, `reader.py` (arm E's path), `measure.py`, `local_analyze.py` and the installed
`modeling_qwen3_vl.py` / `masking_utils.py` / `cache_utils.py`. The reviewer also ran a CPU text-level check
with the processor on HateMM hate_video_1: the standalone text equals head + branch text and tokenizes to
prefix + whole-video question + stance + branch ids; the replay suffix text equals E's; `adjacent_local`
differs from the replay only in the labels; `adjacent_native` uses the prefix's exact frame wrapper;
`prefix_local` renders 20 + k frames with the intro count, the whole-video question once, the stance answer
and the window question once.

## Findings

1. BLOCKER. `placement_controls.read` asserted exactly 20 native frames, and `placement_analyze.prepare`
   assumed 20 in two places. HateClipSeg `bit_AxrVklzh9Cyf` has 18 files in `data/frames_k20`; r6 and E used
   18 for it. Position 232 of 333 in the main order, not in the fixed five, so the full run would have aborted
   after about two thirds of the work. Fixed: `0 < len(native) <= 20` in the job; the analysis uses the
   record's own native frame count. `prefix_local`'s intro states 18 + k for that video, as r6's prefix has 18.
2. MINOR. The per-window replay distance was only in `alignment.json`. Fixed: `report()` copies the binding
   numbers (replay max/mean |Δz|, exact count, covered windows, standalone-native max |Δz|, prefix_local
   token counts, seconds, peak memory) into `placement_analysis/summary.json`.
3. MINOR (operational). The placement files were uncommitted at review time (committed since: 0fc039a,
   dbaaf00, and the fix commit). On sc474397 the Qwen3-VL-8B hub snapshot was incomplete at review time (it
   was being copied back from sc474398); the lab1 launch file is only usable after that copy finishes.

## Answers

- Suffix path: explicit 3-axis position ids make `Qwen3VLModel.forward` skip `compute_3d_position_ids`, so
  `rope_deltas` is untouched (and saved/restored anyway). With `attention_mask=None` and the cache,
  `create_causal_mask` builds the bottom-right causal mask over cache + suffix, the same path as r6's `_step`.
  Text positions equal r6's (arange + logical start); images go through the model's own
  `get_image_features` / scatter / DeepStack on the suffix rows only, the same rows and layers as E's injection.
  `cache.crop(n)` restores the cache length, as `stance_cache.margin` does.
- Standalone path: text equals r6's cached conversation; `same_turn` is False; `rope_deltas` is reset and
  recomputed, then restored.
- Arms match the README (frames, placement, labels, which windows keep the native V).
- No label or GT array on the scoring path; `data/gt_4fps` is read only by the evaluator subprocesses.
- `measure.prediction` is reused unchanged; the forward / vision call-count assertions match the code.
- Binding checks test what they claim; the half-open boundary rule is the same in the job, the analysis and
  the coverage analysis. Only the 20-frame assumption failed on real data.
- `report()` computes the declared differences, floors, "supported" rule and position verdict as declared.

Verdict before the fix: FAIL (finding 1). After the fix the CPU fixture check passes again
(`runs/20261007_m1_streamingtom/placement_cpu_checks/summary.json`); no further review round was opened,
per rule 6 (a found bug only needs its fix confirmed).
