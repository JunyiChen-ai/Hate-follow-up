"""Topic/act cross-tabulation for any run: do GT-negative windows that merely MENTION the target score high?

Same computation as experiments/20260912_pwc/diagnose_topic_confound.py, parameterised by --predictions so
it can be run on an adapted SDL run. The falsifiable prediction of SDL is that after adaptation the mention
main effect shrinks while the GT main effect is retained. Rule-10 test read; log it in the README.
"""
import json, re, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,".")
from src.video_inputs import load_asr, fixed_windows, window_text

import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument("--predictions", default="runs/20260910_spvl/full2_dual_evid_stance/predictions.jsonl")
FULL=_ap.parse_args().predictions
HYP="runs/20260911_hvl/e0_hypothesis/predictions.jsonl"
STOP={"people","person","group","men","women","community","the","of","and","white","black"}
GEN_STOP={"people","person","group","community","the","of","and"}

def target_words(t):
    if not t: return []
    return [x for x in re.findall(r"[a-z]+", t.lower()) if len(x)>2 and x not in GEN_STOP]

hyp={}
for line in open(HYP):
    r=json.loads(line); h=(r.get("extra") or {}).get("hypothesis") or {}
    hyp[(r["dataset"],r["video_id"])]=h.get("target")

rows=[json.loads(l) for l in open(FULL)]
for ds in ["HateMM","HateClipSeg"]:
    g=np.load(f"data/gt_4fps/{ds}.npz",allow_pickle=True)
    gt={v:np.asarray(y,float) for v,y in zip(g["video_ids"],g["y4"])}
    asr=load_asr(ds)
    R=[r for r in rows if r["dataset"]==ds and r.get("extra") and r["extra"].get("windows")]
    # cells: [judged positive, total] keyed by (gt label, mentions target)
    cell={(l,m):[0,0] for l in (0,1) for m in (0,1)}
    zmean={(l,m):[] for l in (0,1) for m in (0,1)}
    nvid=0
    for r in R:
        vid=r["video_id"]; y=gt.get(vid)
        tw=target_words(hyp.get((ds,vid)))
        if y is None or not tw: continue
        segs=asr.get(vid,[])
        wins=fixed_windows(float(r["duration"]), 8.0)
        W={w["i"]:w for w in r["extra"]["windows"]}
        ok=False
        for i,(a,b) in enumerate(wins):
            if i not in W: continue
            ia,ib=int(round(a*4)),min(int(round(b*4)),len(y))
            seg=y[ia:ib]
            if not seg.size: continue
            lab=1 if seg.mean()>0.5 else 0
            txt=(window_text(segs,a,b) or "").lower()
            men=1 if any(w in txt for w in tw) else 0
            z=float(W[i]["z"])
            cell[(lab,men)][1]+=1; cell[(lab,men)][0]+=int(z>0)
            zmean[(lab,men)].append(z)
            ok=True
        nvid+=ok
    print(f"\n=== {ds} ({nvid} videos with a target string)")
    print("  GT label | mentions target | n windows | judged positive | mean z")
    for lab in (0,1):
        for men in (0,1):
            c=cell[(lab,men)]
            if c[1]==0: continue
            print(f"     {lab}     |        {men}        |   {c[1]:5d}   |     {c[0]/c[1]:.2f}      | {np.mean(zmean[(lab,men)]):+.2f}")
    fp_m=cell[(0,1)]; fp_n=cell[(0,0)]
    if fp_m[1] and fp_n[1]:
        print(f"  false-positive rate on GT-negative windows: mentions target {fp_m[0]/fp_m[1]:.2f} "
              f"vs no mention {fp_n[0]/fp_n[1]:.2f}   (gap {fp_m[0]/fp_m[1]-fp_n[0]/fp_n[1]:+.2f})")
    tp_m=cell[(1,1)]; tp_n=cell[(1,0)]
    if tp_m[1] and tp_n[1]:
        print(f"  recall on GT-positive windows:              mentions target {tp_m[0]/tp_m[1]:.2f} "
              f"vs no mention {tp_n[0]/tp_n[1]:.2f}")
