#!/usr/bin/env python3
"""Assemble matched 2x2 TRACE factorial and additive/order controls."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def load(path, method):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, Path(path).open())
            if r.get("method") == method}

def main():
    p=argparse.ArgumentParser()
    for name in ("base","text","smooth","sequential","reverse"):
        p.add_argument(f"--{name}",type=Path,required=True);p.add_argument(f"--{name}-method",required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    if a.out.exists(): raise RuntimeError(f"refusing existing output: {a.out}")
    sources={name:load(getattr(a,name),getattr(a,f"{name}_method")) for name in ("base","text","smooth","sequential","reverse")}
    keys=set.intersection(*(set(x) for x in sources.values())); methods={"base":"trace_factor_base_v1","text":"trace_factor_text_v1","smooth":"trace_factor_smooth_v1","sequential":"trace_factor_sequential_v1","reverse":"trace_factor_reverse_v1"}
    with a.out.open("w") as h:
      for key in sorted(keys):
       curves={name:np.asarray(rows[key]["score_curve"],float) for name,rows in sources.items()}
       additive=curves["text"]+curves["smooth"]-curves["base"]
       if np.min(additive)<0 or np.max(additive)>1: raise RuntimeError(f"additive out of range: {key}")
       for name,method in methods.items():
        row=dict(sources[name][key]);row["method"]=method;row["raw"]={**row.get("raw",{}),"factorial_arm":name,"gt_access":False};h.write(json.dumps(row,separators=(",",":"))+"\n")
       row=dict(sources["base"][key]);row["method"]="trace_factor_additive_v1";row["score_curve"]=additive.tolist();row["raw"]={**row.get("raw",{}),"factorial_arm":"additive_probability_corrections","gt_access":False,"video_mean_preserved":True};h.write(json.dumps(row,separators=(",",":"))+"\n")
    print(json.dumps({"n":len(keys),"methods":list(methods.values())+["trace_factor_additive_v1"]},indent=2))
if __name__=="__main__":main()
