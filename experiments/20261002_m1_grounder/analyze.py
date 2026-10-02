#!/usr/bin/env python3
"""Post-reading evaluation/diagnostics. GT never enters the reading module."""
import argparse
import copy
import csv
import json
import socket
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.eval.evaluate import within_video_macro

ARMS=("base","late","verdict_only","all_local","shifted","early","shift_only")
METRICS=("frame_ROC_AUC","frame_PR_AUC","within_video_macro_ROC_AUC")
DATASETS=("HateMM","HateClipSeg")


def read(path):
    result={}
    for r in map(json.loads,path.open()):
        key=r["dataset"],r["video_id"]
        assert key not in result and not r.get("error"),key
        result[key]=r
    return result


def boot(values):
    v=np.asarray(values,float)
    if not len(v):return {"n":0}
    rng=np.random.default_rng(0)
    means=v[rng.integers(len(v),size=(10000,len(v)))].mean(axis=1)
    return {"n":len(v),"mean":float(v.mean()),"ci95":np.quantile(means,[.025,.975]).tolist(),
            "improved":int((v>1e-10).sum()),"worse":int((v< -1e-10).sum())}


def prepare(readroot):
    rows={arm:read(readroot/arm/"predictions.jsonl") for arm in ARMS if arm!="shift_only"}
    base=rows["base"]
    manifest=read(ROOT/"data/omsl_v6_inputs/manifests/all_test.jsonl")
    expected={k for k in manifest if k[0] in DATASETS}
    assert set(base)==expected,(len(base),len(expected))
    for arm,data in rows.items():
        assert data.keys()==base.keys(),arm
        for k,r in data.items():
            ref=base[k]
            assert r["extra"]["z_video"]==ref["extra"]["z_video"]
            assert r["native_rate"]==4 and len(r["score_curve"])==len(ref["score_curve"])
            assert np.isfinite(r["score_curve"]).all()
            assert [(w["start"],w["end"]) for w in r["extra"]["windows"]]==[(w["start"],w["end"]) for w in ref["extra"]["windows"]]
    out=readroot/"shift_only";out.mkdir(exist_ok=True)
    with (out/"predictions.jsonl").open("w") as f:
        for key,r in base.items():
            shift=float(np.mean([w["z"] for w in rows["late"][key]["extra"]["windows"]])-np.mean([w["z"] for w in r["extra"]["windows"]]))
            s=copy.deepcopy(r);s["method"]="m1_grounder_shift_only_diagnostic"
            s["score_curve"]=(np.array(s["score_curve"])+shift).tolist()
            for w in s["extra"]["windows"]:
                for name in ("z","z_visual","z_speech"):
                    if name in w:w[name]+=shift
            s["extra"]["diagnostic_constant_shift"]=shift
            f.write(json.dumps(s)+"\n")
    (out/"config.json").write_text(json.dumps({"source":str(readroot),"rule":"same constant per video, no labels; diagnostic only"},indent=2)+"\n")
    print("PREPARED",len(base),"paired videos",flush=True)


def evaluate(readroot,decoded):
    for arm in ARMS:
        p=readroot/arm
        subprocess.run([sys.executable,"-m","src.eval.evaluate_four_datasets","--predictions",str(p/"predictions.jsonl"),
                        "--gt-dir","data/gt_4fps","--datasets",*DATASETS,"--out",str(p/"metrics.json")],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable,"experiments/20260926_twolevel/twolevel_r2.py","--run",str(p),
            "--datasets",*DATASETS,"--noleak","--transform","nscore","--key","calib","--duration","bma",
            "--bma-prior","length","--min-windows","2","--bma-grid","6","--arm","m2","--out-root",str(decoded),"--tag",arm],cwd=ROOT,check=True)


def report(readroot,decoded,out):
    out.mkdir(exist_ok=True)
    raw={a:read(readroot/a/"predictions.jsonl") for a in ARMS}
    pred={a:read(decoded/a/"predictions.jsonl") for a in ARMS}
    metrics={a:{r["dataset"]:r for r in json.load((decoded/a/"metrics.json").open())["per_dataset"]} for a in ARMS}
    rawmetrics={a:{r["dataset"]:r for r in json.load((readroot/a/"metrics.json").open())["per_dataset"]} for a in ARMS}
    results={};allrows=[]
    for ds in DATASETS:
        gt=np.load(ROOT/f"data/gt_4fps/{ds}.npz",allow_pickle=True)
        ys={str(vid):np.asarray(gt["y4"][i]) for i,vid in enumerate(gt["video_ids"]) if str(gt["split"][i])=="test"}
        rows=[]
        for vid,y in ys.items():
            key=ds,vid
            if key not in pred["base"]:continue
            r={"dataset":ds,"video_id":vid,"positive_fraction":float(y.mean()),"global_positive":raw["base"][key]["extra"]["z_video"]>0}
            for arm in ARMS:
                r[arm]=within_video_macro({vid:y},{vid:np.asarray(pred[arm][key]["score_curve"])})["within_video_macro_ROC_AUC"]
                r[arm+"_raw"]=within_video_macro({vid:y},{vid:np.asarray(raw[arm][key]["score_curve"])})["within_video_macro_ROC_AUC"]
            r["delta"]=r["late"]-r["base"] if r["base"] is not None else None
            xx,yy=raw["base"][key]["score_curve"],raw["late"][key]["score_curve"]
            r["raw_rank_spearman"]=float(spearmanr(xx,yy).statistic) if np.ptp(xx)>0 and np.ptp(yy)>0 else None
            rows.append(r)
        mixed=[r for r in rows if r["delta"] is not None];allrows+=rows
        comp={}
        for ref in ARMS:
            if ref=="late":continue
            comp[ref]={"delta":{m:metrics["late"][ds][m]-metrics[ref][ds][m] for m in METRICS},
                       "within_paired":boot([r["late"]-r[ref] for r in mixed]),
                       "raw_within_paired":boot([r["late_raw"]-r[ref+"_raw"] for r in mixed]),
                       "correct_yes":boot([r["late"]-r[ref] for r in mixed if r["global_positive"]]),
                       "wrong_no":boot([r["late"]-r[ref] for r in mixed if not r["global_positive"]]),
                       "sparse_positive":boot([r["late"]-r[ref] for r in mixed if r["positive_fraction"]<.25])}
        results[ds]={"final":{a:{m:metrics[a][ds][m] for m in METRICS} for a in ARMS},
                     "raw":{a:{m:rawmetrics[a][ds][m] for m in METRICS} for a in ARMS},"comparisons":comp,
                     "largest_gains":sorted(mixed,key=lambda r:r["delta"],reverse=True)[:5],
                     "largest_losses":sorted(mixed,key=lambda r:r["delta"])[:5]}
    gates={"no_drop":all(results[d]["comparisons"]["base"]["delta"][m]>=(-.01 if m.startswith("within") else -.005) for d in DATASETS for m in METRICS),
           "within_gain_both":all(results[d]["comparisons"]["base"]["delta"][METRICS[-1]]>=.01 for d in DATASETS),
           "component_gates":{a:any(all(results[d]["comparisons"][a]["delta"][m]>=.01 for d in DATASETS) for m in METRICS)
                              for a in ("shifted","all_local","verdict_only","early")}}
    gates["performance_pass"]=gates["no_drop"] and gates["within_gain_both"]
    gates["staged_mechanism_supported"]=gates["performance_pass"] and all(gates["component_gates"].values())
    result={"datasets":results,"gates":gates,"scope":"development-selected; frozen r6 decoder algorithm; parameters refit label-free per read arm",
            "metric_sources":{a:str((decoded/a/"metrics.json").relative_to(ROOT)) for a in ARMS}}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"per_video.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    lines=["dataset\tarm\tROC\tPR\twithin\traw_within\n"]
    for ds,r in results.items():
        for arm,mm in r["final"].items():lines.append(ds+"\t"+arm+"\t"+"\t".join(f"{mm[m]:.6f}" for m in METRICS)+f"\t{r['raw'][arm][METRICS[-1]]:.6f}\n")
    (out/"table.tsv").write_text("".join(lines));print("".join(lines));print(json.dumps(gates));print("ANALYSIS_DONE",flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--read-run",default="r1_full")
    ap.add_argument("--stage",choices=["all","prepare","evaluate","report"],default="all");a=ap.parse_args()
    root=ROOT/"runs/20261002_m1_grounder";readroot=root/a.read_run;decoded=root/(a.read_run+"_decoded");out=root/(a.read_run+"_analysis")
    print("host",socket.gethostname(),flush=True)
    if a.stage in ("all","prepare"):prepare(readroot)
    if a.stage in ("all","evaluate"):evaluate(readroot,decoded)
    if a.stage in ("all","report"):report(readroot,decoded,out)


if __name__=="__main__":main()
