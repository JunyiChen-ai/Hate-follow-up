#!/usr/bin/env python3
"""Symmetric consensus between predictive-risk and stability diffusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_lcurve_diffusion import select as lcurve_select
from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_trace_continuous_gcv import select as gcv_select
from scripts.idea_discovery.project_trace_gcv import reconstruct, smooth


METHODS = ("consensus_diffusion_geometric_v1", "consensus_diffusion_harmonic_control_v1",
           "consensus_diffusion_arithmetic_control_v1")


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--base-method",required=True);ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise RuntimeError(f"refusing existing output: {args.out}")
    rows=load(args.base,args.base_method); eps=np.finfo(float).tiny
    with args.out.open("w") as handle:
        for _,row in sorted(rows.items()):
            scores=np.asarray(row["score_curve"],float);c=np.clip(scores,1e-6,1-1e-6)
            logits=np.log(c/(1-c));signal=logits-logits.mean()
            _,g,g_audit=gcv_select(signal);_,l,l_audit=lcurve_select(signal)
            g=max(eps,g);l=max(eps,l)
            strengths={METHODS[0]:float(np.sqrt(g*l)),
                       METHODS[1]:float(2/(1/g+1/l)),
                       METHODS[2]:float((g+l)/2)}
            for method,strength in strengths.items():
                fitted=smooth(signal,np.ones(len(signal)-1),strength)
                output=dict(row);output["method"]=method
                output["score_curve"]=reconstruct(scores,fitted).tolist()
                output["raw"]={**output.get("raw",{}),"gt_access":False,
                    "module":"symmetric_predictive_stability_consensus",
                    "dataset_parameters":0,"label_selected_parameters":0,
                    "video_mean_preserved":True,"intervals_preserved":True,
                    "consensus":{"gcv_strength":g,"lcurve_strength":l,
                                 "consensus_strength":strength,
                                 "gcv":g_audit,"lcurve":l_audit,
                                 "rule":method}}
                handle.write(json.dumps(output,separators=(",",":"))+"\n")


if __name__=="__main__":main()
