# data/gt_4fps_hate_only — provenance

Secondary evaluation GT for HateClipSeg, restricted to the **Hateful** category instead of the offensive
union used by the main table. Built 2026-09-12 on uoa-lab1 by `scripts/build_gt_4fps_hate_only.py`
(conda HateVideo) from `/home/jehc223/Retrieval-hate/data/gt/HateClipSeg/gold_segments.json`, whose
per-segment 6-dim label follows the HateClipSeg paper's order
`[Normal, Hateful, Insulting, Sexual, Violence, Self-Harm]` (arXiv 2508.01712).

A segment is positive iff dimension 1 (Hateful) is set. The 4 fps grid, split and video set are taken
from `data/gt_4fps/HateClipSeg.npz` frame for frame, so the two arrays are aligned index by index.

Counts (`report.json`): 119 videos, 114,097 frames, 22,623 positive frames (base rate **.198**) against
53,766 (base rate .471) in the main GT; 51 videos contain both classes here, against 99 in the main GT.

Purpose (user ruling 2026-09-12): the main table stays on the full HateClipSeg GT. This array is used only
for a **secondary** evaluation reported next to it, to separate "the method is weak on HateClipSeg" from
"the label definition (offensive union) differs from the prompt (hate rules)". No method, prompt or
constant depends on it (rule 13), and it is never used for selection.
