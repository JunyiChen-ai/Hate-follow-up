import json, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score

def load(p):
    out={}
    for line in open(p):
        r=json.loads(line)
        if r.get("extra") and r["extra"].get("windows"):
            out[(r["dataset"],r["video_id"])]={w["i"]:float(w["z"]) for w in r["extra"]["windows"]}
    return out

FULL=load("runs/20260910_spvl/mllm/q3vl-8b/full/predictions.jsonl")
WIN =load("runs/20260910_spvl/mllm/q3vl-8b/winonly/predictions.jsonl")
print("videos: full",len(FULL),"winonly",len(WIN))

for ds in ["HateMM","HateClipSeg"]:
    g=np.load(f"data/gt_4fps/{ds}.npz",allow_pickle=True)
    gt={v:np.asarray(y,float) for v,y in zip(g["video_ids"],g["y4"])}
    rows={"z_full":[], "z_winonly":[], "contrast":[], "contrast_z":[]}
    sp=[]
    for (d,vid),zf in FULL.items():
        if d!=ds or (d,vid) not in WIN: continue
        y=gt.get(vid)
        if y is None: continue
        zw=WIN[(d,vid)]
        idx=sorted(set(zf)&set(zw))
        if len(idx)<2: continue
        a=np.array([zf[i] for i in idx]); b=np.array([zw[i] for i in idx])
        n=len(y); lab=[]
        curves={"z_full":a, "z_winonly":b, "contrast":a-b,
                "contrast_z":(a-a.mean())/(a.std()+1e-9)-(b-b.mean())/(b.std()+1e-9)}
        frame={k:np.full(n,np.nan) for k in curves}
        for k_,i in enumerate(idx):
            s,e=int(round(i*8*4)),min(int(round((i+1)*8*4)),n)
            if e<=s: continue
            for k in curves: frame[k][s:e]=curves[k][k_]
        m=~np.isnan(frame["z_full"]); yy=y[m]
        if yy.min()==yy.max(): continue
        for k in curves: rows[k].append(roc_auc_score(yy,frame[k][m]))
        if a.std()>0 and b.std()>0:
            sp.append(np.corrcoef(a,b)[0,1])
    print(f"\n{ds}  n={len(rows['z_full'])} videos   mean corr(z_full, z_winonly) = {np.mean(sp):.3f}")
    for k in ["z_full","z_winonly","contrast","contrast_z"]:
        print(f"  within from {k:12s} = {np.mean(rows[k]):.4f}")
