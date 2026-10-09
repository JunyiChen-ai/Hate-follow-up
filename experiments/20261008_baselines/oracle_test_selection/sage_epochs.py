#!/usr/bin/env python3
"""Oracle test selection for SAGE: train each seed with per-epoch checkpoints, then score every checkpoint on every
test window with all four output heads.

    DETWIN_RUNS=runs/20261008_baselines/oracle_test_selection \
        python sage/sage_run.py train --dataset DS --seed S --save-every-epoch      # identical recipe, 30 epochs
    python sage_epochs.py --dataset DS                                              # this file

Scoring is cmd_infer of sage/sage_run.py, unchanged in every input: the same window frame cache (16 frames per window
by the authors' sampler), window audio, window transcript, Encoders, window batches of 16 inside each video and the
F1 tail rule (a window with fewer than 16 readable frames takes the previous window's value). What differs: instead
of the three selected checkpoints, every per-epoch checkpoint of every seed (epochs/eNN.pth) plus the current run's
selected checkpoint (runs/20261008_baselines/sage/<DS>/seed<k>/sage_best_model.pth, tag "current") is loaded, and four
heads are read per model: softmax(logits)[1] of logits_fusion (the current run's output), logits_text,
logits_audio and logits_vision. Each window batch is encoded once and passed through every model.
A checkpoint whose weights are not finite (training diverged) is not scored and is recorded in nonfinite.json.

Output: runs/20261008_baselines/oracle_test_selection/sage/<DS>/seed<k>/epochs/<tag>.json.gz =
{head: {video_id: [window scores]}}; progress in sage/<DS>/score_epochs/partial.jsonl (resumable, deleted at the
end); check of the "current" fusion scores against the current run's infer/window_scores.jsonl in check.json.
The per-epoch .pth files are deleted after their scores are written. No test label is read.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "sage"))
import sage_run as SR  # noqa: E402  (imports detwin/common.py as SR.C)

C = SR.C
CUR = C.REPO / "runs" / "20261008_baselines"
ORACLE = CUR / "oracle_test_selection"
HEADS = ("fusion", "text", "audio", "vision")


def finite_state(sd) -> bool:
    return all(torch.isfinite(v).all().item() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-current", action="store_true", help="do not score the current run's selected checkpoints")
    args = ap.parse_args()
    from Sage import SAGE
    ds = args.dataset
    work = ORACLE / "sage" / ds / "score_epochs"
    work.mkdir(parents=True, exist_ok=True)
    log = C.RunLog(work / "run.log")
    cfg = SR.load_config()
    cfg["device"] = args.device
    log(f"score SAGE {ds} per epoch; code {C.code_version()}")
    models, nonfinite = {}, {}
    for s in C.SEEDS:
        sd_dir = ORACLE / "sage" / ds / f"seed{s}"
        if not (sd_dir / "run.log").is_file() or "DONE train" not in (sd_dir / "run.log").read_text():
            raise SystemExit(f"FAILED: seed {s} not trained ({sd_dir})")
        cks = sorted((sd_dir / "epochs").glob("e*.pth"))
        if not args.no_current:
            cks.append(CUR / "sage" / ds / f"seed{s}" / "sage_best_model.pth")
        for ck in cks:
            tag = "current" if ck.name == "sage_best_model.pth" else ck.stem
            sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]  # own checkpoints
            if not finite_state(sd):
                nonfinite[f"{s}/{tag}"] = str(ck.relative_to(C.REPO))
                continue
            m = SR.no_attention_weights(SAGE(cfg["model"]).to(args.device))
            m.load_state_dict(sd)
            m.eval()
            models[(s, tag)] = m
    (work / "nonfinite.json").write_text(json.dumps(nonfinite, indent=1) + "\n")
    log(f"{len(models)} models loaded; non-finite weights (not scored): {sorted(nonfinite)}")
    enc = SR.Encoders(cfg, args.device)
    test = C.load_split(ds)["test"]
    dur = {x["video_id"]: float(x["duration"]) for x in test}
    segs = C.load_asr(ds, dur)
    part = work / "partial.jsonl"
    done = set()
    if part.is_file():
        for line in part.open():
            done.add(json.loads(line)["video_id"])
    fh = part.open("a")
    t0 = time.time()
    for k, x in enumerate(test):
        v = x["video_id"]
        if v in done:
            continue
        meta = json.loads((SR.WIN_DIR / ds / f"{v}.json").read_text())
        warr = np.load(SR.WIN_DIR / ds / f"{v}.npy", mmap_mode="r")
        wave = C.load_wav(C.wav_path(ds, v))
        wins = C.windows(dur[v])
        ok_w = [j for j, okj in enumerate(meta["window_ok"]) if okj]
        scores = {key: {h: [None] * len(wins) for h in HEADS} for key in models}
        for i0 in range(0, len(ok_w), args.batch):
            js = ok_w[i0:i0 + args.batch]
            frames = [torch.from_numpy(np.array(warr[j])) for j in js]
            vfe = enc.vision(frames, train=False)
            tfe = enc.texts([C.window_text(segs.get(v, []), *wins[j]) for j in js])
            afe = torch.stack([enc.audio_from_wave(None if wave is None else
                                                   wave[int(round(wins[j][0] * 16000)):int(round(wins[j][1] * 16000))])
                               for j in js])
            with torch.no_grad():
                for key, m in models.items():
                    lt, la, lv, lf, _, _ = m(tfe, afe, vfe)
                    for h, lg in zip(HEADS, (lf, lt, la, lv)):
                        p = torch.softmax(lg.float(), 1)[:, 1].tolist()
                        for j, pj in zip(js, p):
                            scores[key][h][j] = pj
        f1 = [j for j in range(len(wins)) if j not in ok_w]
        for key in models:
            for h in HEADS:
                sc = scores[key][h]
                for j in range(len(wins)):
                    if sc[j] is None and j > 0 and sc[j - 1] is not None:
                        sc[j] = sc[j - 1]
        fh.write(json.dumps({"video_id": v, "f1_windows": f1,
                             "scores": {f"{s}/{tag}": scores[(s, tag)] for (s, tag) in models}}) + "\n")
        fh.flush()
        done.add(v)
        if len(done) % 20 == 0:
            log(f"  {len(done)}/{len(test)} videos ({time.time() - t0:.0f}s)")
    fh.close()
    if len(done) != len(test):
        raise SystemExit(f"FAILED: {len(done)}/{len(test)} videos scored")
    # split into one file per (seed, tag)
    out = {key: {h: {} for h in HEADS} for key in models}
    f1w = {}
    for line in part.open():
        r = json.loads(line)
        f1w[r["video_id"]] = r["f1_windows"]
        for key in models:
            for h in HEADS:
                out[key][h][r["video_id"]] = r["scores"][f"{key[0]}/{key[1]}"][h]
    for (s, tag), d in out.items():
        dst = ORACLE / "sage" / ds / f"seed{s}" / "epochs" / f"{tag}.json.gz"
        with gzip.open(dst.with_name(dst.name + ".part"), "wt") as g:
            json.dump(d, g)
        dst.with_name(dst.name + ".part").rename(dst)
    (work / "f1_windows.json").write_text(json.dumps({v: x for v, x in f1w.items() if x}) + "\n")
    # check: the current run's selected checkpoints, fusion head, against the current run's window scores
    check = {}
    cur_ws = CUR / "sage" / ds / "infer" / "window_scores.jsonl"
    if cur_ws.is_file() and not args.no_current:
        ref = {json.loads(l)["video_id"]: json.loads(l)["scores"] for l in cur_ws.open()}
        for s in C.SEEDS:
            if (s, "current") not in out:
                continue
            diffs = [abs(a - b) for v, r in ref.items() for a, b in zip(r[str(s)], out[(s, "current")]["fusion"][v])]
            check[str(s)] = {"max_abs_diff_vs_current_window_scores": max(diffs), "n_windows": len(diffs)}
    (work / "check.json").write_text(json.dumps(check, indent=1) + "\n")
    log(f"check vs current window scores: {check}")
    for s in C.SEEDS:
        for ck in (ORACLE / "sage" / ds / f"seed{s}" / "epochs").glob("e*.pth"):
            ck.unlink()
    part.unlink()
    log(f"DONE score_epochs {ds} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
