#!/usr/bin/env python3
"""Post-inference comparisons. GT is used only here and by the canonical evaluator."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").is_file())
sys.path.insert(0,str(ROOT))
from src.eval.evaluate import within_video_macro

OUT=ROOT/"runs/20261002_revisable_prior"
PATHS={"append_r6":"runs/20261002_verdict_analysis/main_full",
       "nostance_r6":"runs/20261002_verdict_analysis/main_nostance",
       **{k:"runs/20261002_revisable_prior/r1_"+k for k in ("full","independent","no_global")}}


def read(path):
    return {(r["dataset"],r["video_id"]):r for r in map(json.loads,(ROOT/path/"predictions.jsonl").open()) if not r.get("error")}


def bootstrap(d):
    d=np.asarray(d,float)
    if not len(d):return {"n":0}
    rng=np.random.default_rng(0);means=d[rng.integers(len(d),size=(10000,len(d)))].mean(axis=1)
    return {"n":len(d),"mean":float(d.mean()),"ci95":np.quantile(means,[.025,.975]).tolist(),
            "improved":int((d>1e-10).sum()),"worse":int((d< -1e-10).sum())}


def main():
    out=OUT/"analysis";out.mkdir(exist_ok=True)
    selection=json.load((OUT/"numerical_selection.json").open())
    for arm in ("full","independent","no_global"):
        if not selection[arm]["numerically_acceptable"]:
            raise RuntimeError(f"{arm}: unresolved numerical/semantic fit; do not interpret method metrics")
        PATHS[arm]=selection[arm]["path"]
    pred={k:read(p) for k,p in PATHS.items()}
    keys=pred["append_r6"].keys()
    assert all(p.keys()==keys for p in pred.values())
    metrics={k:{r["dataset"]:r for r in json.load((ROOT/p/"metrics.json").open())["per_dataset"]} for k,p in PATHS.items()}
    allrows=[];summaries={};metric_names=("frame_ROC_AUC","frame_PR_AUC","within_video_macro_ROC_AUC")
    for ds in ("HateMM","HateClipSeg"):
        gt=np.load(ROOT/f"data/gt_4fps/{ds}.npz",allow_pickle=True)
        ys={str(v):np.asarray(gt["y4"][i],np.int8) for i,v in enumerate(gt["video_ids"]) if str(gt["split"][i])=="test"}
        rows=[]
        for vid,y in ys.items():
            key=ds,vid
            if key not in pred["full"]:continue
            rawz=pred["full"][key]["extra"]["z_video"]
            row={"dataset":ds,"video_id":vid,"z_video":rawz,"positive_fraction":float(y.mean()),
                 "global_class":("TP" if y.any() else "FP") if rawz>0 else ("FN" if y.any() else "TN")}
            for method,p in pred.items():
                row[method]=within_video_macro({vid:y},{vid:np.asarray(p[key]["score_curve"])})["within_video_macro_ROC_AUC"]
            row["delta"]=row["full"]-row["append_r6"] if row["full"] is not None else None
            row["candidate_video_logodds"]=pred["full"][key]["extra"]["video_logodds"]
            row["candidate_offset"]=pred["full"][key]["extra"]["offset_mean"]
            rows.append(row)
        allrows+=rows
        s={"metrics":{k:{m:metrics[k][ds][m] for m in metric_names} for k in PATHS},"sources":PATHS,"comparisons":{}}
        for ref in ("append_r6","nostance_r6","independent","no_global"):
            mm=[r for r in rows if r["full"] is not None]
            delta={m:metrics["full"][ds][m]-metrics[ref][ds][m] for m in metric_names}
            b=bootstrap([r["full"]-r[ref] for r in mm])
            assert abs(b["mean"]-delta["within_video_macro_ROC_AUC"])<1e-12
            s["comparisons"][ref]={"metric_differences":delta,"within_paired":b,
                "by_global":{c:bootstrap([r["full"]-r[ref] for r in mm if r["global_class"]==c]) for c in ("TP","FN")}}
        s["largest_gains"]=sorted([r for r in rows if r["delta"] is not None],key=lambda r:r["delta"],reverse=True)[:5]
        s["largest_losses"]=sorted([r for r in rows if r["delta"] is not None],key=lambda r:r["delta"])[:5]
        summaries[ds]=s
    valid={}
    for arm in ("full","independent","no_global"):
        ps=json.load((ROOT/PATHS[arm]/"params.json").open())
        qs=json.load((ROOT/PATHS[arm]/"quadrature.json").open())
        valid[arm]={ds:{"ordered":p["ordered_means"],"converged":p["converged"],"tau":p["tau"],"pi":p["pi"],
                       "quadrature":qs[ds]} for ds,p in ps.items()}
    no_drop=all(all(s["comparisons"]["append_r6"]["metric_differences"][m]>=(-.01 if m.startswith("within") else -.005) for m in metric_names) for s in summaries.values())
    improvement=any(all(s["comparisons"]["append_r6"]["metric_differences"][m]>=.01 for s in summaries.values()) for m in metric_names)
    numerical_ok=all(v["ordered"] and v["converged"] and not v["quadrature"]["needs_refit"] for v in valid["full"].values())
    output={"datasets":summaries,"fits":valid,"gate":{"no_drop":no_drop,"same_metric_gain_both":improvement,"fit_valid":numerical_ok,
             "passes_performance_gate":bool(no_drop and improvement and numerical_ok)},"scope":"development-selected; old matched Reader family; not a main-table replacement"}
    (out/"summary.json").write_text(json.dumps(output,indent=2)+"\n")
    with (out/"per_video.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    text=["dataset\tmethod\tROC\tPR\twithin\n"]
    for ds,s in summaries.items():
        for k,mm in s["metrics"].items():text.append(ds+"\t"+k+"\t"+"\t".join(f"{mm[m]:.6f}" for m in metric_names)+"\n")
    (out/"table.tsv").write_text("".join(text));print("".join(text));print(json.dumps(output["gate"]));print("ANALYSIS_DONE")


if __name__=="__main__":main()
