#!/usr/bin/env python3
"""Parameter-free dual-geometry controls for factorized LESS."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path, method=None):
    out = {}
    for row in map(json.loads, path.open()):
        if method is None or row["method"] == method:
            out[(row["dataset"], row["video_id"])] = row
    return out


def choose(cca, t3al, rule):
    if not cca:
        return []
    c = list(map(float, cca[0][:2]))
    if not t3al:
        return c
    t = list(map(float, t3al[0][:2]))
    if rule == "cca": return c
    if rule == "t3al": return t
    if rule == "shorter": return c if c[1] - c[0] <= t[1] - t[0] else t
    if rule == "midpoint": return [(c[0] + t[0]) / 2, (c[1] + t[1]) / 2]
    if rule == "intersection":
        overlap = [max(c[0], t[0]), min(c[1], t[1])]
        return overlap if overlap[1] > overlap[0] else c
    if rule == "union": return [min(c[0], t[0]), max(c[1], t[1])]
    raise ValueError(rule)


def main():
    p=argparse.ArgumentParser();p.add_argument('--posterior',type=Path,required=True);p.add_argument('--posterior-method',required=True);p.add_argument('--cca',type=Path,required=True);p.add_argument('--t3al',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists(): raise RuntimeError(f'refusing existing output: {a.out}')
    post=load(a.posterior,a.posterior_method);cca=load(a.cca);t3al=load(a.t3al);keys=sorted(set(post)&set(cca)&set(t3al))
    for rule in ('cca','t3al','shorter','midpoint','intersection','union'):
        for key in keys:
            row=post[key];pair=choose(cca[key]['intervals'],t3al[key]['intervals'],rule);ivs=[]
            if pair and pair[1]>pair[0]: ivs=[Interval(pair[0],min(float(row['duration']),pair[1]),1.)]
            append_jsonl(a.out,Prediction(f'fact_less_t3al_dualgeo_{rule}_v5',key[0],key[1],float(row['duration']),score_curve=[float(x) for x in row['score_curve']],intervals=ivs,calls=0,raw={'gt_access':False,'existence_gate':'cca','boundary_rule':rule,'numeric_parameters':0}))
    print(json.dumps({'n':len(keys),'rules':6}))
if __name__=='__main__':main()
