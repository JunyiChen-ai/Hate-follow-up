#!/usr/bin/env python3
"""Post-hoc GT diagnostics only; no scoring or fitting parameters are changed."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").is_file())
sys.path.insert(0,str(ROOT))
from src.eval.evaluate import pooled

OUT=ROOT/"runs/20261002_revisable_prior"
run=OUT/"r1_full_px"
params=json.load((run/"params.json").open())
rows=list(map(json.loads,(run/"predictions.jsonl").open()))
summary={}
for ds in ("HateMM","HateClipSeg"):
    gt=np.load(ROOT/f"data/gt_4fps/{ds}.npz",allow_pickle=True)
    ys={str(vid):np.array([np.asarray(gt["y4"][i]).any()],np.int8) for i,vid in enumerate(gt["video_ids"]) if str(gt["split"][i])=="test"}
    rr={r["video_id"]:r["extra"] for r in rows if r["dataset"]==ds and r["video_id"] in ys}
    yy={vid:ys[vid] for vid in rr}
    fields=("z_video","global_nscore","video_logodds","offset_mean")
    diag={k:pooled(yy,{vid:np.array([r[k]]) for vid,r in rr.items()}) for k in fields}
    p=params[ds]
    summary[ds]={"video_count":len(rr),"one_observation_per_video_ROC_diagnostic":{k:v["frame_ROC_AUC"] for k,v in diag.items()},
        "offset_vs_global_spearman":float(spearmanr([r["offset_mean"] for r in rr.values()],[r["global_nscore"] for r in rr.values()]).statistic),
        "shared_variance_fraction_global_visual_speech":(p["tau"]**2/(p["tau"]**2+np.array(p["var"]))).tolist()}
out=OUT/"analysis";out.mkdir(exist_ok=True)
(out/"failure_diagnostics.json").write_text(json.dumps({"datasets":summary,"GT_used":"test, after fitting; diagnostic only",
    "metric_note":"canonical pooled evaluator with one binary presence observation per video; not a replacement frame metric",
    "interpretation":"Tests whether inferred shared offset retains discriminative video information that revised regime odds lose."},indent=2)+"\n")
print(json.dumps(summary,indent=2))
