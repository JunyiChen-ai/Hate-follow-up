#!/usr/bin/env python3
"""Exploratory cached-feature kill-test for a prequential T3AL write gate.

This is deliberately a T3AL-like prototype-update proxy, not yet a reproduction claim.
It uses frozen CLIP frame features and frozen per-ASR-chunk Qwen margins.  Gold is read
only after every curve for a configuration has been produced.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "results/label_free_adapt/features/vidgroup_clip_l14"
GTROOT = Path("/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps")
ASR = {
    "HateMM": ROOT / "results/hatemm_localization/per_chunk.jsonl",
    "HateClipSeg": ROOT / "results/reproduction/ours/hateclipseg/per_chunk.jsonl",
    "MHC": ROOT / "results/reproduction/ours/mhclip_en/per_chunk.jsonl",
    "MHC_zh": ROOT / "results/reproduction/ours/mhclip_zh/per_chunk.jsonl",
}


def rows(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def ecdf(x):
    if len(x) < 2: return np.full_like(x, .5, dtype=float)
    return (rankdata(x, method="average") - .5) / len(x)


def interp(x, n):
    if len(x) == n: return np.asarray(x, float)
    if len(x) == 1: return np.full(n, float(x[0]))
    return np.interp((np.arange(n)+.5)/n, (np.arange(len(x))+.5)/len(x), x)


def text_curve(chunks, duration, n, temp):
    accum=np.zeros(n); weight=np.zeros(n)
    for r in chunks:
        s,e=map(float,r["span"]); z=float(r.get("z_masked",r.get("z_isolated",-20)))
        val=1/(1+math.exp(-np.clip(z/temp,-30,30)))
        a=max(0,int(np.floor(s/duration*n))); b=min(n,max(a+1,int(np.ceil(e/duration*n))))
        accum[a:b]+=val; weight[a:b]+=1
    out=np.full(n,.5); ok=weight>0; out[ok]=accum[ok]/weight[ok]
    return out,ok


def query_vector(prompt):
    import torch, clip
    model,_=clip.load("ViT-L/14",device="cpu",jit=False)
    toks=clip.tokenize(["normal content",prompt])
    with torch.no_grad(): q=model.encode_text(toks).float().numpy()
    q/=np.linalg.norm(q,axis=1,keepdims=True)+1e-12
    d=q[1]-q[0]; return d/(np.linalg.norm(d)+1e-12)


def proposed_curve(X,q,tcurve,known,eta,radius,lam,quantile,mode):
    base=X@q; n=len(base); k=max(1,int(round(n*(1-quantile))))
    pos=np.argsort(base)[-k:]; neg=np.argsort(base)[:k]
    negproto=X[neg].mean(0)
    js=[]; local_parts=[]; far_parts=[]
    for i in pos:
        qp=q+eta*(X[i]-negproto); qp/=np.linalg.norm(qp)+1e-12
        delta=X@qp-base
        dist=np.abs(np.arange(n)-i); local=dist<=radius; far=dist>2*radius
        # Held-out ASR is a signed target, not merely positive corroboration.
        # A useful write raises the local visual margin where ASR is hateful and
        # lowers it where ASR is benign.  Confidence controls weight; missing ASR
        # contributes exactly zero.
        target=2*tcurve-1
        w=np.abs(target)*known
        den=w[local].sum()
        lg=float((delta[local]*target[local]*w[local]).sum()/den) if den>1e-9 else 0.
        fd=float(np.mean(np.abs(delta[far]))) if far.any() else 0.
        js.append(lg-lam*fd); local_parts.append(lg); far_parts.append(fd)
    js=np.asarray(js); coverage=.5
    m=max(1,int(round(k*coverage)))
    if mode=="prequential": chosen=pos[np.argsort(js)[-m:]]
    elif mode=="confidence": chosen=pos[np.argsort(base[pos])[-m:]]
    elif mode=="agreement": chosen=pos[np.argsort(tcurve[pos])[-m:]]
    elif mode=="ungated": chosen=pos
    elif mode=="frozen": chosen=np.array([],dtype=int)
    else: raise ValueError(mode)
    if len(chosen):
        qp=q+eta*(X[chosen].mean(0)-negproto); qp/=np.linalg.norm(qp)+1e-12
        curve=X@qp
    else: curve=base.copy()
    return curve,{"j":js,"conf":base[pos],"coverage":len(chosen)/max(len(pos),1),
                  "local":local_parts,"far":far_parts}


def spans(binary):
    d=np.diff(np.pad(np.asarray(binary,np.int8),(1,1)))
    return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))


def interval_counts(y,s,thr=.5):
    ps=spans(s>=np.quantile(s,.8)); gs=spans(y>0)
    pairs=[]
    for i,(a,b) in enumerate(ps):
        for j,(c,d) in enumerate(gs):
            inter=max(0,min(b,d)-max(a,c)); union=max(b,d)-min(a,c)
            pairs.append((inter/max(union,1),i,j))
    tp=0; up=set(); ug=set()
    for v,i,j in sorted(pairs,reverse=True):
        if v<thr: break
        if i not in up and j not in ug: tp+=1; up.add(i); ug.add(j)
    return tp,len(ps)-tp,len(gs)-tp


def metrics(items):
    yy=[]; ss=[]; within=[]; tp=fp=fn=0
    for y,s in items:
        yy.extend(y); ss.extend(s)
        if np.unique(y).size==2: within.append(roc_auc_score(y,s))
        a,b,c=interval_counts(y,s); tp+=a; fp+=b; fn+=c
    pr=average_precision_score(yy,ss) if len(set(yy))==2 else float("nan")
    roc=roc_auc_score(yy,ss) if len(set(yy))==2 else float("nan")
    f1=2*tp/max(2*tp+fp+fn,1)
    return {"pooled_roc":roc,"pooled_pr":pr,"within_macro_roc":float(np.mean(within)) if within else float("nan"),
            "interval_f1_05":f1,"n_videos":len(items),"n_within":len(within)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--prompt",default="hateful content")
    ap.add_argument("--eta",type=float,default=.15); ap.add_argument("--radius-sec",type=float,default=6)
    ap.add_argument("--lambda-drift",type=float,default=1.); ap.add_argument("--candidate-quantile",type=float,default=.8)
    ap.add_argument("--text-temp",type=float,default=4.); ap.add_argument("--out",required=True)
    a=ap.parse_args(); q=query_vector(a.prompt)
    modes=["frozen","ungated","confidence","agreement","prequential"]
    result={"config":vars(a),"datasets":{},"diagnostics":{}}
    all_by_mode={m:[] for m in modes}; all_j=[]; all_c=[]
    for ds,apath in ASR.items():
        byvid={}
        for r in rows(apath): byvid.setdefault(r["video_id"],[]).append(r)
        z=np.load(GTROOT/(ds+".npz"),allow_pickle=True)
        ids=list(map(str,z["video_ids"])); split=list(map(str,z["split"])); durations=z["duration"]
        gt={v:(split[i],float(durations[i]),np.asarray(z["y4"][i],int)) for i,v in enumerate(ids)}
        cur={m:[] for m in modes}; diagnostics=[]
        for fp in sorted((FEAT/ds).glob("*.npy")):
            vid=fp.stem
            if vid not in gt or gt[vid][0] != "test" or vid not in byvid: continue
            _,dur,y=gt[vid]; X=np.load(fp).astype(float); X/=np.linalg.norm(X,axis=1,keepdims=True)+1e-12
            tc,known=text_curve(byvid[vid],dur,len(X),a.text_temp)
            for m in modes:
                score,d=proposed_curve(X,q,tc,known,a.eta,max(1,round(a.radius_sec/dur*len(X))),a.lambda_drift,a.candidate_quantile,m)
                cur[m].append((y,interp(ecdf(score),len(y))))
                if m=="prequential":
                    all_j.extend(d["j"]); all_c.extend(d["conf"]); diagnostics.append(d)
        result["datasets"][ds]={m:metrics(cur[m]) for m in modes}
        for m in modes: all_by_mode[m].extend(cur[m])
        result["diagnostics"][ds]={"n":len(diagnostics),"mean_coverage":float(np.mean([d["coverage"] for d in diagnostics])) if diagnostics else None,
          "mean_local":float(np.mean([x for d in diagnostics for x in d["local"]])) if diagnostics else None,
          "mean_far":float(np.mean([x for d in diagnostics for x in d["far"]])) if diagnostics else None}
    result["macro"]={m:{k:float(np.nanmean([result["datasets"][d][m][k] for d in result["datasets"]]))
                           for k in ["pooled_roc","pooled_pr","within_macro_roc","interval_f1_05"]} for m in modes}
    result["diagnostics"]["j_conf_spearman"]=float(spearmanr(all_j,all_c).statistic) if len(all_j)>2 else None
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result["macro"],indent=2)); print("diag",json.dumps(result["diagnostics"],indent=2))

if __name__=="__main__": main()
