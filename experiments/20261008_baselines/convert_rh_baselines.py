#!/usr/bin/env python3
"""Re-evaluate Retrieval-hate (2026-08) baseline outputs with this repo's evaluator.

No model is run here. Each method's per-video output from the Retrieval-hate
reproduction campaign is read, put on the 4 fps grid with the rule that campaign
used, checked against our fixed test cohort, written in the shared prediction
schema, and passed to `src/eval/evaluate_four_datasets.py` unchanged.

Fixed cohort = the (dataset, video_id) set in the current method's predictions
(`runs/20260926_twolevel/final_rawkey/predictions.jsonl`): HateMM 215,
HateClipSeg 118. Exact-test-set rule (user, 2026-10-08): a corpus is evaluated
for a method only if the method has a finite score for every frame of every
cohort video. One missing video, or any unscored (NaN) frame inside the GT
length, means no metrics for that corpus; the run records it as needing a rerun.

Rasterisation (Retrieval-hate `scripts/repro_campaign/eval_frame.py`):
  * score curves: piecewise-constant broadcast, frame i of the 4 fps grid takes
    native sample floor(i / 4 * rate), index clipped to the last sample
    (`broadcast_to_4fps`). The output length is the GT length T_gt. Retrieval-hate
    truncated both sides to min(T_gt, T_feat); here the same function is called
    with T = T_gt, so frames past the method's last native sample hold that
    sample's value (the function's own clip/pad branch). Those frames are counted
    in coverage.json.
  * interval outputs (Qwen2.5-VL, UniTime): interval clipped to [0, D], frames
    i in [ceil(4a), ceil(4b)) set to 1, others 0 (`qwen_curves`, `raster`).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP_ID = "20261008_baselines"
OUT_ROOT = REPO / "runs" / EXP_ID
COHORT_PRED = REPO / "runs/20260926_twolevel/final_rawkey/predictions.jsonl"
MANIFEST = REPO / "data/omsl_v6_inputs/manifests/all_test.jsonl"
GT_DIR = REPO / "data/gt_4fps"
COPY = REPO / "data/retrieval_hate_repro"          # copied from uoa-lab2, see PROVENANCE.md
RH_LOCAL = Path("/home/jehc223/Retrieval-hate/archive/idea-stage")  # read-only, this machine
DATASETS = ("HateMM", "HateClipSeg")
FPS = 4.0


# ------------------------------------------------------------ cohort / GT ---
def load_cohort():
    cohort = {}
    with COHORT_PRED.open() as fh:
        for line in fh:
            row = json.loads(line)
            cohort.setdefault(row["dataset"], set()).add(row["video_id"])
    dur = {}
    with MANIFEST.open() as fh:
        for line in fh:
            row = json.loads(line)
            dur[(row["dataset"], row["video_id"])] = float(row["duration"])
    t_gt = {}
    for ds in DATASETS:
        z = np.load(GT_DIR / f"{ds}.npz", allow_pickle=True)
        for i, vid in enumerate(z["video_ids"]):
            if str(z["split"][i]) == "test":
                t_gt[(ds, str(vid))] = len(z["y4"][i])
    return {ds: sorted(cohort[ds]) for ds in DATASETS}, dur, t_gt


# ------------------------------------------------- rasterisation (RH rules) ---
def broadcast_to_4fps(curve: np.ndarray, native_rate: float, T: int) -> np.ndarray:
    """Retrieval-hate eval_frame.broadcast_to_4fps, unchanged arithmetic."""
    if native_rate == FPS:
        s = curve
    else:
        idx = np.floor(np.arange(T) / FPS * native_rate).astype(int)
        idx = np.clip(idx, 0, len(curve) - 1)
        s = curve[idx]
    if len(s) < T:
        s = np.concatenate([s, np.full(T - len(s), s[-1] if len(s) else 0.0)])
    return s[:T].astype(np.float64)


def raster_intervals(spans, D: float, T: int) -> np.ndarray:
    """Retrieval-hate qwen_curves / unitime_to_curves.raster, unchanged arithmetic."""
    c = np.zeros(T, dtype=np.float64)
    for a, b in spans:
        a, b = float(a), float(b)
        if b < a:
            a, b = b, a
        a, b = max(0.0, min(a, D)), max(0.0, min(b, D))
        i0 = int(np.ceil(a * FPS - 1e-9))
        i1 = int(np.ceil(b * FPS - 1e-9))
        c[max(0, i0):max(0, min(T, i1))] = 1.0
    return c


# ---------------------------------------------------------- method loaders ---
# Each loader returns (curve, native_rate, intervals) or (None, reason).
def npz_loader(root: Path, key: str):
    def load(ds, vid, D, T):
        p = root / ds / f"{vid}.npz"
        if not p.exists():
            return None, "no_output_file"
        z = np.load(p, allow_pickle=False)
        c = np.asarray(z[key], dtype=np.float64).reshape(-1)
        if c.size == 0 or not np.isfinite(c).any():
            return None, "empty_or_all_nan"
        return (c, float(z["rate"]), []), None
    return load


_ZS = {}


def zs_clip_loader(ds, vid, D, T):
    if ds not in _ZS:
        _ZS[ds] = np.load(COPY / f"repro_zs_clip/scores_visual_{ds}.npz", allow_pickle=True)
    z = _ZS[ds]
    names = [str(x) for x in z["set_names"]]
    if vid not in z.files:
        return None, "no_output"
    # row `main` = cos(img, "a hateful video frame") - cos(img, "a normal video frame");
    # Retrieval-hate reports sigmoid(100 * this), a strictly increasing map.
    return (np.asarray(z[vid][names.index("main")], dtype=np.float64), FPS, []), None


_IB_TEXT = None


def imagebind_audio_loader(ds, vid, D, T):
    """Retrieval-hate eval_frame.imagebind_curves, channel `audio`, rate 0.5."""
    global _IB_TEXT
    if _IB_TEXT is None:
        _IB_TEXT = np.load(COPY / "imagebind_audio/imagebind_text_normal_hateful.npy")
    p = COPY / "imagebind_audio" / ds / f"{vid}.npy"
    if not p.exists():
        return None, "no_embedding"
    e = np.load(p).astype(np.float32)
    if e.size == 0:
        return None, "empty_embedding"
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8)
    logits = e @ _IB_TEXT.T
    m = logits.max(axis=1, keepdims=True)
    q = np.exp(logits - m)
    return ((q[:, 1] / q.sum(axis=1)).astype(np.float64), 0.5, []), None


_QWEN = {}


def qwen_loader(ds, vid, D, T):
    if ds not in _QWEN:
        last = {}
        with open(RH_LOCAL / f"repro_qwen_ground/raw/qwen_{ds}_main.jsonl") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                last[r["video_id"]] = r          # last record per id, as Retrieval-hate
        _QWEN[ds] = last
    r = _QWEN[ds].get(vid)
    if r is None:
        return None, "not_run"
    if r.get("span") is None:
        return None, r.get("error") or "unparsed"
    a, b = float(r["span"][0]), float(r["span"][1])
    if b < a:
        a, b = b, a
    a, b = max(0.0, min(a, D)), max(0.0, min(b, D))
    return (raster_intervals([(a, b)], D, T), FPS, [[a, b, 1.0]]), None


_UNI_IV = {}


def unitime_loader(ds, vid, D, T):
    """Retrieval-hate curves/main/<ds>/<vid>.npz key `window` (already rasterised by
    unitime_to_curves.py at the GT length) + its interval file."""
    if ds not in _UNI_IV:
        _UNI_IV[ds] = json.loads((COPY / f"repro_unitime/curves/main/{ds}_intervals_window.json").read_text())
    p = COPY / f"repro_unitime/curves/main/{ds}/{vid}.npz"
    if not p.exists():
        return None, "no_output_file"
    z = np.load(p, allow_pickle=False)
    c = np.asarray(z["window"], dtype=np.float64)
    iv = [[max(0.0, min(float(a), D)), max(0.0, min(float(b), D)), 1.0]
          for a, b, *_ in _UNI_IV[ds].get(vid, [])]
    return (c, float(z["rate"]), iv), None


def _src(path: Path) -> str:
    return str(path)


METHODS = {
    # name: (loader, variant, input description, native-rate note, supervision note)
    "zs_clip": dict(
        loader=zs_clip_loader, variant="main (prompt pair 'a normal video frame' / 'a hateful video frame')",
        inputs=[_src(COPY / "repro_zs_clip/scores_visual_{HateMM,HateClipSeg}.npz")],
        native="4 fps (CLIP ViT-L/14-336 dense cache)",
        score="cos(img, hateful prompt) - cos(img, normal prompt); RH reports sigmoid(100*x), same ranking"),
    "zs_imagebind_audio": dict(
        loader=imagebind_audio_loader, variant="audio channel, text ['normal', 'hateful']",
        inputs=[_src(COPY / "imagebind_audio/{HateMM,HateClipSeg}/<vid>.npy"),
                _src(COPY / "imagebind_audio/imagebind_text_normal_hateful.npy")],
        native="0.5 fps (2 s audio clips)",
        score="softmax over (normal, hateful) of unit-normalised audio embedding @ raw text embedding (RH imagebind_curves)"),
    "lavad": dict(
        loader=npz_loader(RH_LOCAL / "repro_lavad/curves", "base"), variant="base (LAVAD stage-06 refined score)",
        inputs=[_src(RH_LOCAL / "repro_lavad/curves/{HateMM,HateClipSeg}/<vid>.npz")],
        native="1 fps", score="npz key `base`"),
    "av2a": dict(
        loader=npz_loader(COPY / "repro_av2a/curves", "sim_combined"), variant="sim_combined",
        inputs=[_src(COPY / "repro_av2a/curves/{HateMM,HateClipSeg}/<vid>.npz")],
        native="1 fps grid (audio half is constant per 10 s window)", score="npz key `sim_combined`"),
    "sevila": dict(
        loader=npz_loader(COPY / "repro_sevila/curves", "main"), variant="main (Localizer yes-logit)",
        inputs=[_src(COPY / "repro_sevila/curves/{HateMM,HateClipSeg}/<vid>.npz")],
        native="1 fps", score="npz key `main`"),
    "unitime": dict(
        loader=unitime_loader, variant="main query, `window` read-out (binary)",
        inputs=[_src(COPY / "repro_unitime/curves/main/{HateMM,HateClipSeg}/<vid>.npz"),
                _src(COPY / "repro_unitime/curves/main/{HateMM,HateClipSeg}_intervals_window.json")],
        native="interval output, rasterised by RH at 4 fps", score="npz key `window`"),
    "urf_hvaa": dict(
        loader=npz_loader(RH_LOCAL / "repro_urf/curves", "base"), variant="base",
        inputs=[_src(RH_LOCAL / "repro_urf/curves/{HateMM,HateClipSeg}/<vid>.npz")],
        native="0.1 fps", score="npz key `base`"),
    "lagovad": dict(
        loader=npz_loader(COPY / "repro_lagovad/curves", "main"), variant="main (free-text hate definition, raw logit)",
        inputs=[_src(COPY / "repro_lagovad/curves/{HateMM,HateClipSeg}/<vid>.npz")],
        native="fps/8 per video (2.9-7.5 samples/s)", score="npz key `main`"),
    "qwen25vl_grounding": dict(
        loader=qwen_loader, variant="query=main",
        inputs=[_src(RH_LOCAL / "repro_qwen_ground/raw/qwen_{HateMM,HateClipSeg}_main.jsonl")],
        native="one interval per video", score="binary raster of the predicted interval"),
}
for s in (0, 1, 2):
    METHODS[f"mulde_s{s}"] = dict(
        loader=npz_loader(COPY / "repro_mulde/curves", f"clipL336_s{s}"), variant=f"clipL336_s{s}",
        inputs=[_src(COPY / "repro_mulde/curves/{HateMM,HateClipSeg}/<vid>.npz")],
        native="4 fps", score=f"npz key `clipL336_s{s}`", group="mulde")
    METHODS[f"clap_s{s}"] = dict(
        loader=npz_loader(COPY / "repro_clap/curves", f"fedavg11_s{s}"), variant=f"fedavg11_s{s}",
        inputs=[_src(COPY / "repro_clap/curves/{HateMM,HateClipSeg}/<vid>.npz")],
        native="2 fps (0.5 s snippets)", score=f"npz key `fedavg11_s{s}`", group="clap")


# ------------------------------------------------------------------- main ---
def convert(method: str, cohort, dur, t_gt):
    spec = METHODS[method]
    rows, report = [], {}
    for ds in DATASETS:
        missing, nan_videos, tail = {}, {}, []
        ds_rows = []
        for vid in cohort[ds]:
            D, T = dur[(ds, vid)], t_gt[(ds, vid)]
            got, reason = spec["loader"](ds, vid, D, T)
            if got is None:
                missing[vid] = reason
                continue
            c, rate, iv = got
            T_feat = int(np.ceil(len(c) * FPS / rate)) if rate != FPS else len(c)
            s = broadcast_to_4fps(c, rate, T)
            n_nan = int((~np.isfinite(s)).sum())
            if n_nan:
                nan_videos[vid] = n_nan
            if T_feat < T:
                tail.append({"video_id": vid, "frames": T - T_feat, "seconds": (T - T_feat) / FPS})
            ds_rows.append({"schema_version": 1, "method": method, "dataset": ds,
                            "video_id": vid, "duration": D, "native_rate": rate,
                            "score_curve": [float(x) for x in s], "intervals": iv})
        exact = not missing and not nan_videos
        n_frames = sum(t_gt[(ds, v)] for v in cohort[ds])
        report[ds] = {
            "cohort_videos": len(cohort[ds]),
            "videos_with_output": len(cohort[ds]) - len(missing),
            "missing_videos": missing,
            "videos_with_unscored_frames": nan_videos,
            "exact_test_set": exact,
            "tail_hold": {"n_videos": len(tail),
                          "n_frames": int(sum(t["frames"] for t in tail)),
                          "share_of_gt_frames": round(sum(t["frames"] for t in tail) / n_frames, 6),
                          "videos_over_1s": sorted([t for t in tail if t["frames"] > 4],
                                                   key=lambda t: -t["frames"])},
        }
        if exact:
            rows.extend(ds_rows)
    return rows, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    args = ap.parse_args()
    cohort, dur, t_gt = load_cohort()
    host = socket.gethostname()
    today = datetime.date.today().isoformat()
    for method in args.methods:
        spec = METHODS[method]
        out = OUT_ROOT / method
        out.mkdir(parents=True, exist_ok=True)
        log = open(out / "run.log", "w")

        def say(msg):
            log.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
            log.flush()
        log.write(f"host {host}\n")
        say(f"code experiments/{EXP_ID}/convert_rh_baselines.py (uncommitted working copy, {today})")
        say("command python3 " + " ".join(sys.argv) + f"   (this run: {method})")
        (out / "run.pid").write_text(f"{os.getpid()}\n")
        rows, report = convert(method, cohort, dur, t_gt)
        (out / "coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        config = {
            "exp_id": EXP_ID, "run_name": method, "date": today, "host": host,
            "code": f"experiments/{EXP_ID}/convert_rh_baselines.py",
            "source": "Retrieval-hate 2026-08 reproduction campaign outputs (no model rerun)",
            "variant": spec["variant"], "inputs": spec["inputs"], "native_rate": spec["native"],
            "score": spec["score"],
            "cohort": f"{COHORT_PRED.relative_to(REPO)} (HateMM 215, HateClipSeg 118)",
            "gt": f"{GT_DIR.relative_to(REPO)}/<dataset>.npz, split test",
            "durations": str(MANIFEST.relative_to(REPO)),
            "conversion_rule": ("curve: frame i <- native sample floor(i/4*rate), clipped to last sample "
                                "(RH eval_frame.broadcast_to_4fps), length = GT length; interval: "
                                "clip to [0,D], frames [ceil(4a), ceil(4b)) = 1"),
            "exact_test_set_rule": "evaluate a corpus only if every cohort video has a finite score on every GT frame",
            "evaluator": "python3 -m src.eval.evaluate_four_datasets --predictions predictions.jsonl "
                         "--gt-dir data/gt_4fps --out metrics.json --datasets <exact corpora>",
        }
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for ds in DATASETS:
            r = report[ds]
            say(f"{ds}: {r['videos_with_output']}/{r['cohort_videos']} videos, missing={r['missing_videos']}, "
                f"unscored-frame videos={r['videos_with_unscored_frames']}, exact={r['exact_test_set']}, "
                f"tail-hold frames={r['tail_hold']['n_frames']} in {r['tail_hold']['n_videos']} videos")
        exact = [ds for ds in DATASETS if report[ds]["exact_test_set"]]
        pred = out / "predictions.jsonl"
        metrics = out / "metrics.json"
        if not exact:
            for p in (pred, metrics):
                if p.exists():
                    p.unlink()
            say("no corpus covers the exact test set; not evaluated (needs rerun)")
            log.close()
            continue
        with pred.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        cmd = [sys.executable, "-m", "src.eval.evaluate_four_datasets", "--predictions", str(pred),
               "--gt-dir", str(GT_DIR), "--out", str(metrics), "--datasets", *exact]
        say("evaluator " + " ".join(cmd[1:]))
        res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        log.write(res.stdout + res.stderr)
        if res.returncode != 0:
            say(f"FAILED evaluator rc={res.returncode}")
            log.close()
            return res.returncode
        m = json.loads(metrics.read_text())
        for r in m["per_dataset"]:
            say(f"{r['dataset']}: ROC {r['frame_ROC_AUC']:.4f} PR {r['frame_PR_AUC']:.4f} "
                f"within {r['within_video_macro_ROC_AUC']:.4f} n_videos {r['n_videos_predicted']} "
                f"n_frames {r['n_frames']} n_within {r['n_videos_defined']}")
        say("DONE")
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
