# Spatial-search crop interface failure — independent diagnosis

2026-10-06. Independent GPT-6-astra reviewer; same-family provisional. Diagnosis only: no production code, source records, boxes, crops, completed readings or configuration were changed. This does not reopen proposal review or evaluate the method's performance.

**Confirmed execution defect:** `measure.read_video` sends every non-null FOUND crop through `memory` → `source_image_branch.encode_branch` → frozen `Judge.encode`. Legal nonempty pixel geometry does not guarantee that the image processor supports the image. The installed Qwen3-VL processor uses `Qwen2VLImageProcessor`; its `smart_resize` explicitly rejects `max(height,width)/min(height,width) > 200`, before resizing. Current acquisition validates real source geometry/pixels but does not make that extra reader-interface distinction.

Actual whole-source CPU inventory: **333 videos, 7,359 windows, 6,767 FOUND crops; 6,759 encode successfully, 8 fail across 7 videos**. Every FOUND crop was opened, checked against the recorded integer box and original image size, compared pixel-for-pixel against the original image crop, and passed through the installed frozen image processor with the production image kwargs. All eight failures were additionally reproduced through the actual rendered original-frame-plus-crop `Judge.encode` path. All failures are the same intrinsic aspect-ratio ValueError; no other crop processor failure was observed.

| Dataset | Video | Window (zero-based) | Actual crop W×H |
|---|---|---:|---:|
| HateMM | hate_video_299 | 4 | 366×1 |
| HateMM | non_hate_video_528 | 5 | 730×1 |
| HateMM | non_hate_video_590 | 10 | 366×1 |
| HateClipSeg | bit_4BVcDK677Zp7 | 12 | 730×1 |
| HateClipSeg | bit_H6MYrA7ZTokV | 18 | 231×1 |
| HateClipSeg | bit_IVZ7jXxCeokA | 29 | 726×1 |
| HateClipSeg | bit_XkQGFhrIWgVG | 10 | 730×1 |
| HateClipSeg | bit_XkQGFhrIWgVG | 14 | 730×1 |

The first failing video is manifest ordinal 138, `HateMM/hate_video_299`. Its window-4 PNG is `data/temporal_spatial_search/HateMM/hate_video_299/crops/w000004.png`, box `[364,230,730,231]`. CPU production encoding raises exactly `absolute aspect ratio must be smaller than 200, got 366.0`, matching Slurm 192. The returned log ends with successful 137/333, and the local paired-output directory contains 137 completed record filenames; the earlier 125 count was an intermediate snapshot. No completed record body was read. None of the seven affected videos has a completed paired record.

Boundary confirmation used synthetic RGB images in both orientations: ratio 199 and **exactly 200** pass; 201 fails. Despite the error wording, the implementation's boundary is strictly greater than 200, not greater-than-or-equal. Production kwargs remain `size.shortest_edge=65536`, `size.longest_edge=100352`; changing those limits does not remove the earlier aspect check.

Minimal execution fix recommended:

1. Preserve acquired FOUND status, original box/PNG, search feedback, UNKNOWN outcomes and source-generation costs. Do not expand the box, pad/stretch the PNG, regenerate sources or substitute a fabricated observation.
2. At the reader interface, distinguish acquired FOUND from actually usable input. For a crop rejected by this specific frozen processor constraint, mark an explicit intrinsic-unsupported reason including actual dimensions/boundary and follow the already-existing unavailable-source native-V path. Keep native G/own stance/S and source cost unchanged. A narrow dimension eligibility check tied to the installed processor is sufficient for this demonstrated failure; do not broadly catch unrelated exceptions as unavailable.
3. Count `found_windows` separately from actually used image-source windows and unsupported windows. Vision-forward assertions must use actual used windows, not acquired FOUND count. Validation must recompute eligibility and require native-V equality/no source branch for unavailable crops. Analysis and the execution guard must distinguish actual image-source use from acquisition success.
4. Preserve the exact existing valid-source content, tokens, positions and numeric path. Existing completed supported-source readings should remain intact; if their metadata schema needs new usage fields, record the migration separately and avoid silently overwriting scientific readings. Verify a supported case unchanged, ratio-200 accepted, ratio-201 unavailable and the real failing crop fallback before resuming.

This is a model-input support omission and resulting execution failure. The proposed fallback is not yet implemented or independently confirmed here; it is not grounds for interpreting this partial run as scientific success or failure.

Evidence: `runs/20261005_m1_spatial_search/crop_interface_fix/independent/diagnose.py`, `inventory.json`, `summary.json`, `run.log`, plus `boundary.py`, `boundary_summary.json`, `boundary.log`. Executed with the local HateVLM Python and `CUDA_VISIBLE_DEVICES=''`. Metadata and crop bytes/mtime were checked unchanged after use. No model weights, GPU work, GT, real prediction values, metrics, hashes or new dataset were used; source metadata, actual PNGs, processor implementation and failure/progress logs were the evidence.
