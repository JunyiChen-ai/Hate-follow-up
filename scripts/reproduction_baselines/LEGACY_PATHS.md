# Legacy paths (1 fps reproduction toolchain)

These scripts date from the 1 fps protocol (2026-08-18 to 08-26) and still reference
paths that moved in the 2026-09-09 cleanup. Fix a reference when you touch the script.

| old path in scripts | now |
|---|---|
| `docs/duplex/FRAME_EVAL_PROTOCOL.md`, `BASELINE_RESULTS.md`, `OFFICIAL_VAL_RESULTS.md`, `official_val_results.json` | `docs/protocol_1fps_legacy/` |
| other `docs/duplex/*` | `archive/detection-2026-08/docs/duplex/` |
| `results/reproduction/` | `runs/legacy_1fps/lab1/reproduction/` (uoa-lab1 outputs) and `runs/legacy_1fps/lab2/reproduction/` (uoa-lab2 outputs) |
| `results/label_free_adapt/` | `runs/legacy_label_free_adapt_2026-08/` (GT arrays now `data/gt_4fps/`) |
| `scripts/duplex/<analysis scripts>` | `archive/detection-2026-08/scripts/duplex/` (feature extraction, GT building, ASR and `frame_eval_common.py` stay in `scripts/duplex/`) |
| `scripts/idea_discovery/`, `scripts/label_free_adapt/` | `archive/experiments/idea_discovery-2026-08/` |

The 1 fps evaluator `eval_baseline_scores.py` + `scripts/duplex/frame_eval_common.py`
is kept for reading the legacy tables only. New numbers use `src/eval/` on the 4 fps
protocol (CLAUDE.md).
