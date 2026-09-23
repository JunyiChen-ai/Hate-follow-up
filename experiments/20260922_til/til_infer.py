#!/usr/bin/env python3
"""TIL inference: posterior hate timeline from interval measurements of one or two window grids. CPU only.

Cells of `--cell` seconds. Each window (from any grid) covers the cells it overlaps; its modality reads y_m are
scaled by the corpus std s_m of that modality (from the predictions only) and fused (sum or max) into one
log-likelihood ratio r for "some covered cell is hateful" vs "none is".
  --model interval : exact forward-backward over pair states (h_{c-1}, h_c); each window is an interval-support
                     (noisy-OR) observation on the cells it covers (at most two consecutive cells).
  --model average  : cell value = mean of the fused reads of the windows covering it; two-state chain on cells.
  --model none     : cell value = raw max over branches of the (single) grid's window, no scaling (SPVL-r2 replicate).
Duration prior: stay probability 1 - cell/D (D in seconds); --dwell 0 means no temporal coupling (stay = .5).
Frame score = [z_video + mean of grid-A raw window reads] + centred within-video rank of the cell log-odds at 4 fps.
Evaluation only through src/eval/evaluate_four_datasets.py. No labels are read here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
FPS = 4.0


def mods_of(windows):
    """Modality keys present in a run's windows: z_visual / z_speech / z_joint ...; a run with only "z" has one modality "z"."""
    keys = sorted({k for w in windows for k in w if k.startswith("z_")})
    return keys or ["z"]


def load_run(path):
    rows = [json.loads(l) for l in open(Path(path) / "predictions.jsonl") if l.strip()]
    return {(r["dataset"], r["video_id"]): r for r in rows if not r.get("error")}


def centered_rank(v):
    v = np.asarray(v, float)
    if len(v) <= 1 or np.ptp(v) <= 1e-12:
        return np.zeros_like(v)
    r = (rankdata(v, method="average") - 0.5) / len(v) - 0.5
    return r - r.mean()


def cover(start, end, n_cells, cell):
    ks = [k for k in range(n_cells) if min(end, (k + 1) * cell) - max(start, k * cell) > 1e-6]
    if not ks:
        ks = [min(n_cells - 1, int(start // cell))]
    return ks


def log_trans(p_stay):
    return np.log(np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]]))


def chain2(v, p_stay):
    """Two-state chain on cells with log-likelihood-ratio emissions v (potential exp(v) for state 1); posterior log-odds."""
    n = len(v)
    E = np.stack([np.zeros(n), np.asarray(v, float)], 1)
    T = log_trans(p_stay)
    f = np.zeros((n, 2)); f[0] = np.log(0.5) + E[0]
    for c in range(1, n):
        f[c] = E[c] + np.logaddexp(f[c - 1, 0] + T[0], f[c - 1, 1] + T[1])
    b = np.zeros((n, 2))
    for c in range(n - 2, -1, -1):
        b[c] = np.logaddexp(T[:, 0] + E[c + 1, 0] + b[c + 1, 0], T[:, 1] + E[c + 1, 1] + b[c + 1, 1])
    q = f + b
    return q[:, 1] - q[:, 0]


def chain_pair(n, obs, p_stay):
    """Pair-state chain: state at cell c is s = 2*h_{c-1} + h_c (h_{-1} = 0). obs[c] = list of (r, kind) with
    kind 'pair' (any of h_{c-1}, h_c) or 'single' (h_c). Returns posterior log-odds of h_c = 1."""
    NEG = -1e30
    T = log_trans(p_stay)
    E = np.zeros((n, 4))
    for c in range(n):
        for r, kind in obs[c]:
            for s in range(4):
                hp, hc = s >> 1, s & 1
                on = (hp or hc) if kind == "pair" else hc
                if on:
                    E[c, s] += r
    # transitions s=(a,b) -> s'=(b',c') allowed iff b' == b
    A = np.full((4, 4), NEG)
    for s in range(4):
        b_ = s & 1
        for s2 in range(4):
            if (s2 >> 1) == b_:
                A[s, s2] = T[b_, s2 & 1]
    f = np.full((n, 4), NEG)
    f[0, 0] = np.log(0.5) + E[0, 0]
    f[0, 1] = np.log(0.5) + E[0, 1]
    for c in range(1, n):
        m = f[c - 1][:, None] + A
        f[c] = E[c] + np.logaddexp.reduce(m, axis=0)
    b = np.zeros((n, 4))
    for c in range(n - 2, -1, -1):
        m = A + (E[c + 1] + b[c + 1])[None, :]
        b[c] = np.logaddexp.reduce(m, axis=1)
    q = f + b
    q -= q.max(1, keepdims=True)
    p = np.exp(q); p /= p.sum(1, keepdims=True)
    p1 = np.clip(p[:, 1] + p[:, 3], 1e-9, 1 - 1e-9)
    return np.log(p1) - np.log(1 - p1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="measurement run dirs; the first is grid A (intercept source)")
    ap.add_argument("--model", choices=["interval", "average", "none", "smooth"], default="interval")
    ap.add_argument("--fusion", choices=["sum", "max"], default="max")
    ap.add_argument("--dwell", type=float, default=80.0, help="mean dwell in seconds; 0 = no temporal coupling")
    ap.add_argument("--cell", type=float, default=4.0)
    ap.add_argument("--scale", choices=["corpus", "pooled"], default="corpus", help="modality std per corpus (transductive) or pooled over all corpora")
    ap.add_argument("--shuffle", action="store_true", help="control: chain applied to a within-video shuffled window order (seed 0), scores mapped back")
    ap.add_argument("--gt-dir", default=str(ROOT / "data/gt_4fps"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default=str(ROOT / "runs/20260922_til/infer"))
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()
    runs = [load_run(p) for p in a.runs]
    keys = [k for k in runs[0] if k[0] in a.datasets and all(k in r for r in runs)]
    p_stay = 0.5 if a.dwell <= 0 else max(0.5, 1.0 - a.cell / a.dwell)
    # label-free modality scales per corpus from the reads themselves
    MODS = mods_of([w for r in runs for rec in r.values() for w in rec["extra"]["windows"]])
    scale = {}
    for ds in a.datasets:
        for m in MODS:
            vals = [w[m] for r in runs for k, rec in r.items() if (a.scale == "pooled" or k[0] == ds)
                    for w in rec["extra"]["windows"] if m in w]
            scale[(ds, m)] = float(np.std(vals)) if len(vals) > 1 else 1.0
    rng = np.random.RandomState(0)
    out_dir = Path(a.out_root) / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    n_ok = 0
    with open(pred_path, "w") as fh:
        for key in keys:
            ds, vid = key
            recA = runs[0][key]
            dur = float(recA["duration"]); L = len(recA["score_curve"])
            n_cells = max(1, int(np.ceil(dur / a.cell - 1e-9)))
            wins = [w for r in (runs if a.model != "none" else runs[:1]) for w in r[key]["extra"]["windows"]]
            if not wins:  # ASR-window run, video without transcript: no measurement at all
                cell_lo = np.zeros(n_cells)
            elif a.model == "none":
                v = np.full(n_cells, np.nan)
                for w in wins:
                    for k in cover(w["start"], w["end"], n_cells, a.cell):
                        v[k] = w["z"] if np.isnan(v[k]) else max(v[k], w["z"])
                v = np.where(np.isnan(v), np.nanmin(v), v)
                cell_lo = v
            else:
                fused = []
                for w in wins:
                    reads = [w[m] / scale[(ds, m)] for m in MODS if m in w]
                    if not reads:
                        continue
                    fused.append((sum(reads) if a.fusion == "sum" else max(reads), cover(w["start"], w["end"], n_cells, a.cell)))
                if a.model in ("average", "smooth"):
                    if a.shuffle or a.model == "smooth":  # window-level operations (single fixed grid only)
                        if len(runs) != 1:
                            raise SystemExit("--shuffle / --model smooth need a single run")
                        wv = np.array([r for r, _ in fused]); n = len(wv)
                        if a.model == "smooth":
                            q = np.pad(wv, 1, mode="edge"); wv2 = (0.5 * q[:-2] + q[1:-1] + 0.5 * q[2:]) / 2.0 if n >= 3 else wv
                        else:
                            perm = rng.permutation(n); rep = np.repeat(wv[perm], 2)
                            post = chain2(rep, p_stay)[0::2]; wv2 = np.empty(n); wv2[perm] = post
                        cell_lo = np.zeros(n_cells)
                        for (r, ks), val in zip(fused, wv2):
                            for k in ks:
                                cell_lo[k] = val
                    else:
                        acc = np.zeros(n_cells); cnt = np.zeros(n_cells)
                        for r, ks in fused:
                            for k in ks:
                                acc[k] += r; cnt[k] += 1
                        v = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
                        cell_lo = chain2(v, p_stay)
                else:
                    obs = [[] for _ in range(n_cells)]
                    for r, ks in fused:
                        if len(ks) > 2 or (len(ks) == 2 and ks[1] != ks[0] + 1):
                            raise SystemExit(f"{key}: window covers cells {ks}; the pair-state chain needs <= 2 consecutive cells")
                        obs[max(ks)].append((r, "pair" if len(ks) == 2 else "single"))
                    cell_lo = chain_pair(n_cells, obs, p_stay)
            centers = (np.arange(L) + 0.5) / FPS
            idx = np.clip((centers // a.cell).astype(int), 0, n_cells - 1)
            curve = np.asarray(cell_lo, float)[idx]
            zv = float(recA["extra"]["z_video"])
            intercept = zv + (float(np.mean([w["z"] for w in recA["extra"]["windows"]])) if recA["extra"]["windows"] else 0.0)
            final = intercept + centered_rank(curve)
            fh.write(json.dumps({**recA, "method": f"til__{a.tag}", "score_curve": [float(x) for x in final],
                                 "extra": {"z_video": zv, "intercept": intercept, "cell_logodds": [float(x) for x in cell_lo]}}) + "\n")
            n_ok += 1
    metrics_path = out_dir / "metrics.json"
    subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(pred_path),
                    "--gt-dir", a.gt_dir, "--out", str(metrics_path), "--datasets", *a.datasets],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL)
    cfg = {**vars(a), "p_stay": p_stay, "scale": {f"{k[0]}/{k[1]}": v for k, v in scale.items()}, "n_videos": n_ok}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    d = json.load(open(metrics_path))
    print(f"{a.tag:28s} n={n_ok} " + "  ".join(f"{p['dataset'][:6]} {p['frame_ROC_AUC']:.4f}/{p['frame_PR_AUC']:.4f}/{p['within_video_macro_ROC_AUC']:.4f}({p['n_videos_defined']})" for p in d["per_dataset"]))


if __name__ == "__main__":
    main()
