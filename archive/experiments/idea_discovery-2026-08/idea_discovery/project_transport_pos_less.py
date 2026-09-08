#!/usr/bin/env python3
"""Lag-robust transport PoS projection for multimodal LESS.

Each timestamped transcript chunk is represented by a duration-adaptive soft
temporal support rather than an exact box. Sparse confidence orders are
qualified against all circular transcript reassignments, then imposed through
a minimum-change convex projection of the dense LESS logits. No GT is read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from scripts.idea_discovery.project_pos_less import load_chunks, sigmoid


def transport_kernels(spans, nframes):
    """Return chunk-normalized measurement kernels and frame-normalized transport."""
    kernels = np.zeros((len(spans), nframes), dtype=float)
    grid = np.arange(nframes, dtype=float) + .5
    for k, (lo, hi) in enumerate(spans):
        center = .5 * (lo + hi); half = max(.5, .5 * (hi - lo))
        # Unit support on the observed chunk; linear uncertainty shoulders of
        # one half-chunk duration model timestamp/ASR boundary imprecision.
        distance = np.maximum(np.maximum(lo - grid, grid - hi), 0.0)
        kernels[k] = np.maximum(0.0, 1.0 - distance / half)
        if not np.any(kernels[k]):
            kernels[k, min(nframes - 1, max(0, int(center)))] = 1.0
    measure = kernels / np.maximum(kernels.sum(axis=1, keepdims=True), 1e-12)
    transport = kernels.T
    transport /= np.maximum(transport.sum(axis=1, keepdims=True), 1.0)
    return measure, transport


def empirical_edges(conf, margin, alpha):
    proposed = [(i, j) for i in range(len(conf)) for j in range(len(conf))
                if conf[i] - conf[j] >= margin]
    if len(conf) <= 2:
        return proposed
    rotations = [np.roll(conf, offset) for offset in range(1, len(conf))]
    output = []
    for i, j in proposed:
        observed = conf[i] - conf[j]
        null = np.asarray([values[i] - values[j] for values in rotations])
        pvalue = (1 + int(np.sum(null >= observed))) / (1 + len(null))
        if pvalue <= alpha:
            output.append((i, j))
    return output


def project(base, spans, conf, margin, alpha):
    edges = empirical_edges(conf, margin, alpha)
    if not edges:
        return base.copy(), 0, True, 0.0
    measure, transport = transport_kernels(spans, len(base))
    means = measure @ base
    # A chunk offset is transported through its uncertainty kernel; the same
    # measurement operator reads the post-projection chunk evidence.
    chunk_op = measure @ transport
    edge_op = np.stack([chunk_op[i] - chunk_op[j] for i, j in edges])
    edge_base = np.asarray([means[i] - means[j] for i, j in edges])
    result = minimize(lambda x: .5 * float(x @ x), np.zeros(len(spans)), jac=lambda x: x,
        constraints={"type": "ineq", "fun": lambda x: edge_base + edge_op @ x,
                     "jac": lambda x: edge_op}, method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 500})
    score = base + transport @ result.x
    post = measure @ score
    feasible = bool(result.success and all(post[i] >= post[j] - 1e-6 for i, j in edges))
    return score if feasible else base.copy(), len(edges), feasible, float(np.linalg.norm(result.x))


def deterministic_shift(values, dataset, video_id):
    if len(values) < 2: return values.copy()
    import hashlib
    token = hashlib.sha256(f"transport-pos/{dataset}/{video_id}".encode()).digest()
    offset = 1 + int.from_bytes(token[:8], "little") % (len(values) - 1)
    return np.roll(values, offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=.2)
    ap.add_argument("--relation-alpha", type=float, default=.5)
    ap.add_argument("--null-gate", action="store_true")
    args = ap.parse_args()
    if args.out.exists(): raise RuntimeError(f"refusing existing output: {args.out}")
    chunks = load_chunks(); audit = {"rows": 0, "with_text": 0, "changed_aligned": 0,
        "changed_shift": 0, "edges_aligned": 0, "edges_shift": 0, "failures": 0}
    with args.base.open() as source, args.out.open("w") as sink:
        for row in map(json.loads, source):
            if row["method"] != "fact_less_t3al_dualgeo_midpoint_v5": continue
            audit["rows"] += 1; key = row["dataset"], row["video_id"]
            p = np.clip(np.asarray(row["score_curve"], float), 1e-5, 1-1e-5)
            base = np.log(p/(1-p)); spans=[]; confidence=[]
            for chunk in chunks.get(key, []):
                lo=max(0,min(len(p)-1,int(float(chunk["span"][0])*4)))
                hi=min(len(p),max(lo+1,int(math.ceil(float(chunk["span"][1])*4))))
                spans.append((lo,hi));confidence.append(sigmoid(float(
                    chunk.get("z_masked",chunk.get("z_isolated",-20)))))
            conf=np.asarray(confidence);audit["with_text"]+=int(bool(spans))
            aligned_result=(project(base,spans,conf,args.margin,args.relation_alpha)
                            if spans else (base.copy(),0,True,0.0))
            null_norms=[]
            if args.null_gate and len(conf)>2:
                for offset in range(1,len(conf)):
                    null_norms.append(project(base,spans,np.roll(conf,offset),
                        args.margin,args.relation_alpha)[3])
            aligned_authorized=(not args.null_gate or not null_norms or
                                aligned_result[3] < float(np.median(null_norms)))
            shift_values=deterministic_shift(conf,*key)
            shift_result=(project(base,spans,shift_values,args.margin,args.relation_alpha)
                          if spans else (base.copy(),0,True,0.0))
            for control,result in (("aligned",aligned_result),("shift",shift_result)):
                score,nedges,feasible,norm=result
                if control=="aligned" and not aligned_authorized:
                    score=base.copy()
                posterior=1/(1+np.exp(-np.clip(score,-30,30)))
                changed=bool(np.max(np.abs(posterior-p))>1e-8)
                audit[f"changed_{control}"]+=int(changed);audit[f"edges_{control}"]+=nedges
                audit["failures"]+=int(not feasible)
                suffix="_nullgate" if args.null_gate else ""
                output=dict(row);output["method"]=f"transport_pos_less_{control}{suffix}_v1"
                output["score_curve"]=posterior.tolist();output["raw"]={**row.get("raw",{}),
                    "gt_access":False,"dense_authority":"LESS","text_role":"lag_robust_soft_transport_poset",
                    "uncertainty_radius":"half_chunk_duration","poset_margin":args.margin,
                    "relation_alpha":args.relation_alpha,"n_edges":nedges,
                    "null_gate":args.null_gate,"aligned_authorized":aligned_authorized,
                    "null_projection_norm_median":float(np.median(null_norms)) if null_norms else None,
                    "projection_norm":norm,"feasible":feasible,"exact_fallback":not changed}
                sink.write(json.dumps(output,separators=(",",":"))+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
