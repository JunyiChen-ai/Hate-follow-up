#!/usr/bin/env python3
"""Canonical full-pair evaluation; labels enter only the post-inference report."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"CLAUDE.md").is_file())
sys.path.insert(0,str(ROOT))
from src.eval.evaluate import within_video_macro

DATASETS=("HateMM","HateClipSeg")
METRICS=("frame_ROC_AUC","frame_PR_AUC","within_video_macro_ROC_AUC")


def read(path):
    out={}
    for r in map(json.loads,path.open()):
        key=r["dataset"],r["video_id"]
        assert key not in out and not r.get("error"),key
        out[key]=r
    return out


def boot(values):
    v=np.array(values,float);rng=np.random.default_rng(0)
    if not len(v):return {"n":0}
    means=v[rng.integers(len(v),size=(10000,len(v)))].mean(axis=1)
    return {"n":len(v),"mean":float(v.mean()),"ci95":np.quantile(means,[.025,.975]).tolist(),
            "improved":int((v>1e-10).sum()),"worse":int((v< -1e-10).sum())}


def prepare(root,out):
    raw={a:read(root/a/"predictions.jsonl") for a in ("base","select")}
    expected={k for k in read(ROOT/"data/omsl_v6_inputs/manifests/all_test.jsonl") if k[0] in DATASETS}
    old=read(ROOT/"runs/20260926_glr/base_gridA/predictions.jsonl")
    assert raw["base"].keys()==raw["select"].keys()==expected
    parity=[]
    for key,b in raw["base"].items():
        s=raw["select"][key];h=old[key]
        assert b["extra"]["z_video"]==s["extra"]["z_video"]==h["extra"]["z_video"]
        assert b["native_rate"]==s["native_rate"]==4
        assert len(b["score_curve"])==len(s["score_curve"])==len(h["score_curve"])
        assert np.isfinite(b["score_curve"]).all() and np.isfinite(s["score_curve"]).all()
        bounds=lambda r:[(w["start"],w["end"]) for w in r["extra"]["windows"]]
        assert bounds(b)==bounds(s)==bounds(h)
        diffs=[]
        for wb,ws,wh in zip(b["extra"]["windows"],s["extra"]["windows"],h["extra"]["windows"]):
            assert ("z_speech" in wb)==("z_speech" in ws)==("z_speech" in wh)
            diffs.extend(abs(wb[m]-wh[m]) for m in ("z_visual","z_speech") if m in wb)
        parity.append({"dataset":key[0],"video_id":key[1],"window_max_abs_diff":max(diffs,default=0.)})
    (out/"alignment.json").write_text(json.dumps({"videos":len(expected),"global_exact":True,"current_base_parity":parity},indent=2)+"\n")
    print("PREPARED",len(expected),"base maximum difference",max(r["window_max_abs_diff"] for r in parity),flush=True)


def evaluate(root,decoded,arm):
    subprocess.run([sys.executable,"-m","src.eval.evaluate_four_datasets","--predictions",str(root/arm/"predictions.jsonl"),
        "--gt-dir","data/gt_4fps","--datasets",*DATASETS,"--out",str(root/arm/"metrics.json")],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,"experiments/20260926_twolevel/twolevel_r2.py","--run",str(root/arm),
        "--datasets",*DATASETS,"--noleak","--transform","nscore","--key","calib","--duration","bma",
        "--bma-prior","length","--min-windows","2","--bma-grid","6","--arm","m2","--out-root",str(decoded),"--tag",arm],cwd=ROOT,check=True)


def metrics(path):
    return {r["dataset"]:r for r in json.load(path.open())["per_dataset"]}


def report(root,decoded,out):
    raw={a:read(root/a/"predictions.jsonl") for a in ("base","select")}
    pred={a:read(decoded/a/"predictions.jsonl") for a in raw}
    mm={a:metrics(decoded/a/"metrics.json") for a in raw};rm={a:metrics(root/a/"metrics.json") for a in raw}
    historical=metrics(ROOT/"runs/20260926_twolevel/r6_bma/metrics.json")
    for a in raw:
        assert pred[a].keys()==raw[a].keys()==raw["base"].keys()
        for k,r in pred[a].items():
            assert r["extra"]["z_video"]==raw["base"][k]["extra"]["z_video"] and r["native_rate"]==4
            assert len(r["score_curve"])==len(raw["base"][k]["score_curve"]) and np.isfinite(r["score_curve"]).all()
    result={"datasets":{},"scope":"development-selected; r6 algorithm unchanged, unsupervised parameters refit per arm",
            "metric_sources":{a:str((decoded/a/"metrics.json").relative_to(ROOT)) for a in raw},"mechanism_supported":False}
    video_rows=[];head_report={};cost={};lines=["dataset\tarm\tROC\tPR\twithin\traw_within\n"]
    for ds in DATASETS:
        g=np.load(ROOT/f"data/gt_4fps/{ds}.npz",allow_pickle=True)
        ys={str(v):np.asarray(g["y4"][i]) for i,v in enumerate(g["video_ids"]) if str(g["split"][i])=="test"}
        rows=[];trajectories=[]
        for key,r in raw["select"].items():
            if key[0]!=ds:continue
            H=r["extra"]["n_heads"];L=r["extra"]["text_layers"]
            for w in r["extra"]["windows"]:
                for mod in ("visual","speech"):
                    assert ("heads_"+mod in w)==("z_"+mod in w)
                    if "heads_"+mod not in w:continue
                    assert len(w["heads_"+mod])==L
                    t=np.zeros((L,H),bool)
                    for i,ids in enumerate(w["heads_"+mod]):
                        assert len(ids)==len(set(ids)) and all(0<=h<H for h in ids)
                        t[i,ids]=True
                    trajectories.append(t)
            vid=key[1]
            if vid not in ys:continue
            y=ys[vid];v={"dataset":ds,"video_id":vid,"positive_fraction":float(y.mean()),"global_positive":r["extra"]["z_video"]>0}
            for a in raw:
                v[a]=within_video_macro({vid:y},{vid:np.asarray(pred[a][key]["score_curve"])})[METRICS[-1]]
                v[a+"_raw"]=within_video_macro({vid:y},{vid:np.asarray(raw[a][key]["score_curve"])})[METRICS[-1]]
            if v["base"] is not None:
                v["delta"]=v["select"]-v["base"];v["raw_delta"]=v["select_raw"]-v["base_raw"];rows.append(v)
        video_rows.extend(rows)
        t=np.stack(trajectories);counts=t.sum(axis=-1);overlap=(t&np.roll(t,H//2,axis=-1)).sum(axis=-1)
        head_report[ds]={"branches":len(t),"mean_selected_fraction_per_layer":t.mean(axis=(0,2)).tolist(),
            "all_selected_fraction":float((counts==H).mean()),"none_selected_fraction":float((counts==0).mean()),
            "mean_rotated_overlap_when_nonempty":float((overlap[counts>0]/counts[counts>0]).mean())}
        delta={m:mm["select"][ds][m]-mm["base"][ds][m] for m in METRICS}
        result["datasets"][ds]={"final":{a:{m:mm[a][ds][m] for m in METRICS} for a in raw},
            "raw":{a:{m:rm[a][ds][m] for m in METRICS} for a in raw},"delta":delta,
            "delta_vs_current":{m:mm["select"][ds][m]-historical[ds][m] for m in METRICS},
            "within_paired":boot([r["delta"] for r in rows]),"raw_within_paired":boot([r["raw_delta"] for r in rows]),
            "correct_yes":boot([r["delta"] for r in rows if r["global_positive"]]),
            "wrong_no":boot([r["delta"] for r in rows if not r["global_positive"]]),
            "largest_gains":sorted(rows,key=lambda r:r["delta"],reverse=True)[:5],"largest_losses":sorted(rows,key=lambda r:r["delta"])[:5]}
        cost[ds]={}
        for a in raw:
            rr=[r for k,r in raw[a].items() if k[0]==ds]
            cost[ds][a]={"prefix_seconds":sum(r["extra"]["prefix_seconds"] for r in rr),
                "branch_seconds":sum(r["extra"]["branch_seconds"] for r in rr),"mean_calls":float(np.mean([r["calls"] for r in rr]))}
            lines.append(ds+"\t"+a+"\t"+"\t".join(f"{mm[a][ds][m]:.6f}" for m in METRICS)+f"\t{rm[a][ds][METRICS[-1]]:.6f}\n")
    result["gates"]={"any_qualifying_gain":any(r["delta"][m]>=.01 for r in result["datasets"].values() for m in METRICS),
        "performance_pass":all(r["delta"][METRICS[-1]]>=.01 and r["delta_vs_current"][METRICS[-1]]>=.01 and
            all(r[d][m]>=(-.01 if m==METRICS[-1] else -.005) for d in ("delta","delta_vs_current") for m in METRICS) for r in result["datasets"].values())}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    (out/"head_selection.json").write_text(json.dumps(head_report,indent=2)+"\n")
    (out/"cost.json").write_text(json.dumps(cost,indent=2)+"\n")
    (out/"per_video.json").write_text(json.dumps(video_rows,indent=2)+"\n")
    (out/"table.tsv").write_text("".join(lines));print("".join(lines));print(result["gates"]);print("ANALYSIS_DONE",flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("prepare","evaluate","report"),required=True)
    ap.add_argument("--arm",choices=("base","select"));a=ap.parse_args()
    if a.stage=="evaluate" and a.arm is None:ap.error("evaluate requires --arm")
    parent=ROOT/"runs/20261002_m1_selector";root=parent/"r1_main";decoded=parent/"r1_main_decoded";out=parent/"r1_main_analysis"
    out.mkdir(parents=True,exist_ok=True);print("host",socket.gethostname(),flush=True)
    if a.stage=="prepare":prepare(root,out)
    elif a.stage=="evaluate":evaluate(root,decoded,a.arm)
    else:report(root,decoded,out)


if __name__=="__main__":main()
