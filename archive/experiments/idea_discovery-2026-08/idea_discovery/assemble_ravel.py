#!/usr/bin/env python3
"""Assemble RAVEL's identity-grounded intervals and dense label-free readout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", type=Path, required=True)
    ap.add_argument("--dense", type=Path, required=True)
    ap.add_argument("--dense-method", default="melt_adaptive")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise RuntimeError(f"refusing existing output: {args.out}")
    def load(path, method=None):
        out={}
        for line in path.open(encoding="utf-8"):
            row=json.loads(line)
            if method is None or row.get("method")==method:
                out[(row["dataset"],row["video_id"])]=row
        return out
    ext,dense=load(args.intervals),load(args.dense,args.dense_method)
    for key in sorted(set(ext)&set(dense)):
        er,dr=ext[key],dense[key]
        error=er.get("error") or dr.get("error")
        if error:
            append_jsonl(args.out,Prediction("RAVEL",key[0],key[1],er["duration"],error=error));continue
        intervals=[Interval(float(x[0]),float(x[1]),float(x[2]) if len(x)>2 else 1.)
                   for x in er.get("intervals",[])]
        append_jsonl(args.out,Prediction(
            "RAVEL",key[0],key[1],float(er["duration"]),
            score_curve=[float(x) for x in dr["score_curve"]],intervals=intervals,
            calls=3,modality_evidence={
                "interval_readout":"joint_anchored_modality_responsibility",
                "dense_readout":args.dense_method,
                "interval_source":str(args.intervals),"dense_source":str(args.dense)},
            raw={"prediction_object":"relation_identity_plus_temporal_extent"}))


if __name__=="__main__": main()
