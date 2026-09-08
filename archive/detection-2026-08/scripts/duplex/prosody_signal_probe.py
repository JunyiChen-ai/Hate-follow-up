#!/usr/bin/env python3
"""Frozen general-purpose emotion feature probe on the 90-video cohort."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.stats import rankdata
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/prosody_signal"
MODEL_ID = "superb/wav2vec2-base-superb-er"


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    n1, n0 = np.sum(y == 1), np.sum(y == 0)
    if not n1 or not n0:
        return None
    r = rankdata(s, method="average")
    return float((np.sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def valid_intervals(chunks, duration):
    out = []
    for c in chunks:
        a, b = c.get("start"), c.get("end")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or b <= a:
            continue
        a, b = max(0.0, float(a)), min(duration, float(b))
        n = max(1, int(np.ceil((b - a) / 10.0)))
        for x, y in zip(np.linspace(a, b, n + 1)[:-1], np.linspace(a, b, n + 1)[1:]):
            if y - x >= 0.5:
                out.append((float(x), float(y)))
    return out


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    cohort = json.load(open(ROOT / "results/temporal_attribution/cohorts.json"))
    cell = {v: c for c, ids in cohort["cohorts"].items() for v in ids}
    marker = cohort["marker_positive"]
    joint = cohort["joint_z"]
    ids = sorted(cell)
    asr = {r["video_id"]: r for r in map(json.loads, open(ROOT / "results/temporal_attribution/timestamped_asr.jsonl")) if r.get("video_id") in set(ids)}

    feat = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to("cuda").eval()
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    angry = [k for k, v in labels.items() if v in {"ang", "angry", "anger"} or "ang" in v]
    if len(angry) != 1:
        raise SystemExit(f"Expected exactly one angry label, got {labels}")
    angry = angry[0]

    rows = []
    for i, vid in enumerate(ids, 1):
        wav, sr = torchaudio.load(str(ROOT / "results/c2_fullcorpus/wav" / f"{vid}.wav"))
        audio = torch.mean(wav, dim=0).numpy(); duration = len(audio) / sr
        ints = valid_intervals(asr.get(vid, {}).get("chunks", []), duration)
        ps, ds = [], []
        for a, b in ints:
            clip = audio[int(a * sr):int(b * sr)]
            inp = feat(clip, sampling_rate=sr, return_tensors="pt", padding=True)
            inp = {k: v.to("cuda") for k, v in inp.items()}
            with torch.inference_mode():
                p = torch.softmax(model(**inp).logits.float(), dim=-1)[0, angry].item()
            ps.append(p); ds.append(b - a)
        row = {"video_id": vid, "cell": cell[vid], "marker": marker[vid],
               "joint_z": joint[vid], "duration": duration, "n_intervals": len(ps),
               "angry_mean": float(np.average(ps, weights=ds)) if ps else None,
               "angry_max": float(max(ps)) if ps else None,
               "angry_top4": float(np.mean(sorted(ps)[-4:])) if ps else None}
        rows.append(row)
        if i == 1 or i % 10 == 0: print(f"[{i}/90] {vid} intervals={len(ps)} mean={row['angry_mean']}", flush=True)

    with (RUN / "features.jsonl").open("w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    valid = [r for r in rows if r["angry_mean"] is not None]
    def pair(pos, neg, score="angry_mean", subset=lambda r: True):
        rr = [r for r in valid if r["cell"] in {pos, neg} and subset(r)]
        return auc([int(r["cell"] == pos) for r in rr], [r[score] for r in rr])
    x = np.array([[1.0, r["joint_z"]] for r in valid]); q = np.array([r["angry_mean"] for r in valid])
    residual = q - x @ np.linalg.lstsq(x, q, rcond=None)[0]
    for r, v in zip(valid, residual): r["prosody_residual"] = float(v)
    a_tp_fp = pair("tp", "fp")
    a_marker = pair("tp", "fp", subset=lambda r: r["marker"])
    a_tp_tn = pair("tp", "tn")
    a_res = pair("tp", "fp", score="prosody_residual")
    a_dur = pair("tp", "fp", score="duration")
    clauses = {"coverage": len(valid) >= 75, "tp_fp_auc": a_tp_fp >= .65,
               "marker_tp_fp_auc": a_marker >= .65, "tp_tn_auc": a_tp_tn >= .60,
               "residual_auc": a_res >= .60, "duration_control": a_dur < .60}
    out = {"model_labels": labels, "coverage": len(valid),
           "auc": {"tp_fp": a_tp_fp, "marker_tp_fp": a_marker, "tp_tn": a_tp_tn,
                   "residual_tp_fp": a_res, "duration_tp_fp": a_dur},
           "clauses": clauses, "verdict": "PASS" if all(clauses.values()) else "FAIL"}
    (RUN / "results.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__": main()
