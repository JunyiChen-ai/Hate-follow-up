#!/usr/bin/env python3
"""Cached modality-knockout stability gates for fused endpoint proposals."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

PAIR_METHODS = {
    "va": "less_3v_no_language_majority_global_aligned_within_video_v2",
    "vt": "less_3v_no_audio_majority_global_aligned_within_video_v2",
    "at": "less_3v_no_visual_majority_global_aligned_within_video_v2",
}

def load(path, method=None):
    out={}
    for line in Path(path).open():
        r=json.loads(line)
        if method is None or r.get("method")==method: out[(r["dataset"],r["video_id"])]=r
    return out

def interval(r): return tuple(map(float,r["intervals"][0][:2])) if r and r.get("intervals") else None
def logit(x):
    x=np.clip(np.asarray(x,float),1e-6,1-1e-6); return np.log(x/(1-x))
def resize(x,n):
    ix=np.minimum((np.arange(n)*len(x)/n).astype(int),len(x)-1); return x[ix]
def sample(field,times,duration):
    ix=np.clip(np.round(np.asarray(times)/max(duration,1e-9)*(len(field)-1)).astype(int),0,len(field)-1)
    x=field[ix]; scale=np.median(np.abs(field-np.median(field)))+1e-6
    return np.tanh(x/(2*scale))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--zoom",type=Path,required=True); ap.add_argument("--coalitions",type=Path,required=True)
    ap.add_argument("--base",type=Path,required=True); ap.add_argument("--base-method",required=True)
    ap.add_argument("--bank",type=Path,required=True); ap.add_argument("--tight-method",default="fact_less_t3al_dualgeo_shorter_v5")
    ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    if args.out.exists(): raise RuntimeError(f"refusing existing output: {args.out}")
    zoom=load(args.zoom,"ambi_zoom_v1"); fused=load(args.zoom,"ambi_fused_transition_control_v1")
    base=load(args.base,args.base_method); tight=load(args.bank,args.tight_method)
    pairs={k:load(args.coalitions,v) for k,v in PAIR_METHODS.items()}
    methods=("knockout_all_pairs_direction_v1","knockout_all_pairs_localcell_v1","knockout_two_of_three_direction_v1")
    counts={m:0 for m in methods}
    with args.out.open('w') as h:
      for key in sorted(zoom):
        bi,fi,ti=interval(base.get(key)),interval(fused.get(key)),interval(tight.get(key))
        decisions={m:[False,False] for m in methods}; diagnostics=[None,None]
        if bi is not None and fi is not None and all(key in x for x in pairs.values()):
          n=len(zoom[key]['score_curve']); duration=float(zoom[key]['duration'])
          fields={name:resize(logit(rows[key]['score_curve']),n) for name,rows in pairs.items()}
          fields={name:x-x.mean() for name,x in fields.items()}
          audit=zoom[key].get('raw',{}).get('boundary_audit',{})
          for side,name in enumerate(('left','right')):
            a=audit.get(name,{}); times=a.get('times',[]); idx=a.get('control_transition')
            preserves=ti is None or (fi[side]<=ti[0] if side==0 else fi[side]>=ti[1])
            if len(times)!=8 or not isinstance(idx,int) or not preserves or abs(fi[side]-bi[side])<=1e-8: continue
            sign=1 if side==0 else -1
            values={p:sample(field,times,duration) for p,field in fields.items()}
            signed={p:float(sign*(v[idx+1]-v[idx])) for p,v in values.items()}
            best={p:int(np.argmax(sign*np.diff(v))) for p,v in values.items()}
            positive=sum(x>0 for x in signed.values())
            all_direction=positive==3
            localcell=all_direction and all(abs(j-idx)<=1 for j in best.values())
            decisions[methods[0]][side]=all_direction
            decisions[methods[1]][side]=localcell
            decisions[methods[2]][side]=positive>=2
            diagnostics[side]={"fused_cell":idx,"pair_signed_delta":signed,"pair_best_cell":best}
        for method in methods:
          source=base.get(key) or zoom[key]; out=dict(source); out['method']=method; accepted=[False,False]; result=bi
          if bi is not None and fi is not None:
            v=list(bi)
            for side in (0,1):
              if decisions[method][side]: v[side]=fi[side]; accepted[side]=True
            if v[0]<v[1]: result=tuple(v); counts[method]+=sum(accepted)
            else: accepted=[False,False]
          out['intervals']=[] if result is None else [[result[0],result[1],1.0]]
          out['raw']={**out.get('raw',{}),"gt_access":False,"selector":"modality_knockout_stability",
                      "accepted_sides":accepted,"endpoint_diagnostics":diagnostics}
          h.write(json.dumps(out,separators=(',',':'))+'\n')
    print(json.dumps({"n":len(zoom),"accepted_endpoint_actions":counts},indent=2))

if __name__=='__main__': main()
