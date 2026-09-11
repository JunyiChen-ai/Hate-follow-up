#!/usr/bin/env python3
"""E0 kill test for PWC (README section 6): does the model compare two windows better than it scores them?

No method, no composition change. For each video of the within-defined subsets, pairs of 8 s windows are
selected that (a) have different GT window labels and (b) were BOTH judged positive by SPVL-r2 (z > 0) --
the population where the current absolute score is near chance. Each pair is put to the model as one
A/B/C comparison, in both orderings, under the same prefix and stance turn as SPVL-r2.

Measured: accuracy of sign(l_ij) against the GT label difference, versus accuracy of sign(z_i - z_j) on
the same pairs; swap agreement; probability mass on C. The GT is used only to build and score the pairs;
the model never sees a label (rule 10 test read, logged in the experiment README).

Outputs runs/<exp_id>/<run_name>/{pairs.jsonl,summary.json,config.json,run.log,verify.json}.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE_PATH = "experiments/20260912_pwc/pilot0.py"
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION  # noqa: E402
from src.video_inputs import (FPS, fixed_windows, frame_paths, load_asr, load_manifest,  # noqa: E402
                              window_text, within_defined_ids)

SEED = 0
WINDOW_SECONDS = 8.0
PAIRS_PER_VIDEO = 5
CHOICES = ("A", "B", "C")
SPVL_RUN = ROOT / "runs/20260910_spvl/full2_dual_evid_stance/predictions.jsonl"

CARRIER_QUESTION = (
    "Which segment contains a greater proportion of content that itself expresses an endorsed "
    "attack on a protected group, as defined by the rules above? Merely naming a group, or "
    "providing context that explains another segment, does not itself count as an attack."
)
GENERIC_QUESTION = (
    "Which segment more clearly contains content that violates any of the rules above?"
)


def compare_question(idx_a, idx_b, wins, wtexts, style):
    """One A/B/C comparison of two windows under the shared prefix."""
    def seg(letter, k):
        t1, t2 = wins[k]
        body = (wtexts[k] or "").strip() or "(no speech)"
        return f"Segment {letter}: from {t1:.1f}s to {t2:.1f}s of this video. Transcript in {letter}: {body}"

    core = CARRIER_QUESTION if style == "carrier" else GENERIC_QUESTION
    return (
        "Consider two segments of this video.\n\n"
        f"{seg('A', idx_a)}\n\n{seg('B', idx_b)}\n\n"
        "Look at the frames whose timestamps fall inside each segment. Use the rest of the video to "
        "resolve who is being referred to, whether a statement is quoted or endorsed, and who is "
        "speaking.\n\n"
        f"{core}\n\n"
        'Answer "A", "B", or "C", where C means they are equal or there is not enough evidence.'
    )


def window_labels(y4, wins):
    """GT label per window: positive if more than half of its 4 fps frames are positive."""
    out = []
    for t1, t2 in wins:
        a, b = int(round(t1 * FPS)), min(int(round(t2 * FPS)), len(y4))
        seg = y4[a:b]
        out.append(1 if seg.size and float(seg.mean()) > 0.5 else 0)
    return out


def load_gt(datasets):
    gt = {}
    for ds in datasets:
        z = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        for vid, y in zip(z["video_ids"], z["y4"]):
            gt[(ds, str(vid))] = np.asarray(y, dtype=float)
    return gt


def load_spvl_windows():
    """SPVL-r2 per-window scores, used only to select the pairs and as the baseline comparator."""
    out = {}
    for line in open(SPVL_RUN):
        r = json.loads(line)
        if r.get("extra") and r["extra"].get("windows"):
            out[(r["dataset"], r["video_id"])] = {w["i"]: float(w["z"]) for w in r["extra"]["windows"]}
    return out


def select_pairs(labels, zmap, rng, k):
    """Pairs with different GT labels where both windows were judged positive by SPVL-r2."""
    cand = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                continue
            zi, zj = zmap.get(i), zmap.get(j)
            if zi is None or zj is None or zi <= 0 or zj <= 0:
                continue
            cand.append((i, j))
    if not cand:
        return []
    if len(cand) > k:
        pick = rng.choice(len(cand), size=k, replace=False)
        cand = [cand[p] for p in sorted(pick.tolist())]
    return cand


def d_from_logprobs(lp):
    """log((p_A + p_C/2) / (p_B + p_C/2)) from log-probs over the restricted A/B/C softmax."""
    p = np.exp(np.asarray(lp, dtype=float))
    p = p / p.sum()
    return float(np.log((p[0] + p[2] / 2) / (p[1] + p[2] / 2))), float(p[2])


def run_video(judge, row, segments, zmap, y4, args, choice_ids, verify=False):
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    wins = fixed_windows(dur, args.window_seconds)
    wtexts = [window_text(segments, a, b) for a, b in wins]
    labels = window_labels(y4, wins)
    rng = np.random.default_rng(SEED * 100003 + args.video_index)
    pairs = select_pairs(labels, zmap, rng, args.pairs_per_video)
    if not pairs:
        return [], {"n_pairs": 0, "n_windows": len(wins)}

    frames = frame_paths(ds, vid, args.frames, "k20") if args.frames > 0 else []
    if args.frames > 0 and not frames:
        return [], {"error": "no frames cached"}
    msgs, image_files = judge.prefix_messages(frames, segments, with_context=True, with_frames=args.frames > 0)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    prefix_ids = enc["input_ids"][0].tolist()
    cache = judge.prefix_cache(enc)

    b0, b0_text = judge.branch_ids(msgs, VIDEO_QUESTION)
    z_video = judge.cached_margin(cache, b0, in_place=True)
    stance = "Yes" if z_video > 0 else "No"
    a0, a0_text = judge.answer_ids(msgs, VIDEO_QUESTION, stance)
    judge.extend_cache(cache, a0)
    history = [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]},
               judge.turn("assistant", stance)]
    head = prefix_text + b0_text + a0_text

    info = {"prefix_tokens": len(prefix_ids), "n_windows": len(wins), "n_pairs": len(pairs),
            "z_video": z_video, "stance": stance, "img_tokens": judge.img_tokens[:1]}
    if verify:
        from PIL import Image
        z_ref = judge.plain_margin(msgs, image_files, VIDEO_QUESTION)
        q_probe = compare_question(pairs[0][0], pairs[0][1], wins, wtexts, "carrier")
        bp, _ = judge.branch_ids(msgs, q_probe, history, head_text=head)
        # token seam including the stance turn: prefix + Q0 + answer + comparison branch must equal the
        # tokenization of the template's own four-turn rendering (seam_check_tokens takes no history).
        full = judge.render(judge.conv(msgs, q_probe, history), add_generation_prompt=True)
        images = [Image.open(p).convert("RGB") for p in image_files]
        enc_full = judge.encode(full, images)
        for im in images:
            im.close()
        stance_ids = judge.tok(b0_text + a0_text, add_special_tokens=False)["input_ids"]
        want = prefix_ids + stance_ids + bp
        got = enc_full["input_ids"][0].tolist()
        if got != want:
            raise AssertionError(f"token seam mismatch: {len(prefix_ids)}+{len(stance_ids)}+{len(bp)} vs {len(got)}")
        info["verify"] = {"video_q_cache_vs_plain_dz": abs(z_video - z_ref),
                          "seam_tokens": [len(prefix_ids), len(stance_ids), len(bp)],
                          "choice_ids": {c: choice_ids[k] for k, c in enumerate(CHOICES)},
                          "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
        if abs(z_video - z_ref) >= 3.0:
            raise SystemExit(f"VERIFY GATE FAILED: {info['verify']}")

    rows = []
    choices = [choice_ids[k] for k in range(len(CHOICES))]
    for (i, j) in pairs:
        rec = {"dataset": ds, "video_id": vid, "i": i, "j": j, "n_windows": len(wins),
               "label_i": labels[i], "label_j": labels[j], "z_i": zmap[i], "z_j": zmap[j],
               "t_i": wins[i], "t_j": wins[j], "z_video": z_video, "stance": stance}
        for style in args.questions:
            ds_ij = {}
            for orient, (a, b) in (("ij", (i, j)), ("ji", (j, i))):
                q = compare_question(a, b, wins, wtexts, style)
                bids, _ = judge.branch_ids(msgs, q, history, head_text=head)
                lp = judge.cached_choices(cache, bids, choices)
                d, pc = d_from_logprobs(lp)
                ds_ij[orient] = d
                rec[f"{style}_lp_{orient}"] = lp
                rec[f"{style}_pC_{orient}"] = pc
            # d("ij") scores window i as A; d("ji") scores window j as A. The antisymmetric part is the comparison.
            rec[f"{style}_d_ij"] = ds_ij["ij"]
            rec[f"{style}_d_ji"] = ds_ij["ji"]
            rec[f"{style}_l"] = (ds_ij["ij"] - ds_ij["ji"]) / 2.0
        rows.append(rec)
    del cache
    return rows, info


def summarize(rows, styles):
    """Accuracy of the comparison against the GT label difference, versus the SPVL-r2 score difference."""
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        R = [r for r in rows if r["dataset"] == ds]
        truth = np.array([r["label_i"] - r["label_j"] for r in R], dtype=float)  # +1: i positive, -1: j positive
        zdiff = np.array([r["z_i"] - r["z_j"] for r in R], dtype=float)
        e = {"n_pairs": len(R), "n_videos": len({r["video_id"] for r in R}),
             "acc_z_diff": float(np.mean(np.sign(zdiff) == np.sign(truth)))}
        for style in styles:
            l = np.array([r[f"{style}_l"] for r in R], dtype=float)
            dij = np.array([r[f"{style}_d_ij"] for r in R], dtype=float)
            dji = np.array([r[f"{style}_d_ji"] for r in R], dtype=float)
            pc = np.array([r[f"{style}_pC_ij"] for r in R] + [r[f"{style}_pC_ji"] for r in R], dtype=float)
            e[f"acc_{style}"] = float(np.mean(np.sign(l) == np.sign(truth)))
            e[f"acc_{style}_single_order"] = float(np.mean(np.sign(dij) == np.sign(truth)))
            e[f"swap_agreement_{style}"] = float(np.mean(np.sign(dij) == -np.sign(dji)))
            e[f"mean_pC_{style}"] = float(pc.mean())
            e[f"gain_over_z_{style}"] = e[f"acc_{style}"] - e["acc_z_diff"]
        out[ds] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="e0")
    ap.add_argument("--exp-id", default="20260912_pwc")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    ap.add_argument("--pairs-per-video", type=int, default=PAIRS_PER_VIDEO)
    ap.add_argument("--questions", nargs="+", default=["carrier", "generic"], choices=["carrier", "generic"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(SEED)

    out_dir = ROOT / "runs" / args.exp_id / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    cfg = dict(vars(args))
    cfg.update({"code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(),
                "seed": SEED, "carrier_question": CARRIER_QUESTION, "generic_question": GENERIC_QUESTION,
                "video_question": VIDEO_QUESTION, "spvl_run": str(SPVL_RUN)})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    rows_all = load_manifest(args.manifest, args.datasets)
    keep = within_defined_ids(args.datasets)
    rows_all = [r for r in rows_all if (r["dataset"], r["video_id"]) in keep]
    rows_all.sort(key=lambda r: (r["dataset"], r["video_id"]))
    if args.limit:
        rows_all = rows_all[:args.limit]
    gt = load_gt(args.datasets)
    spvl = load_spvl_windows()
    asr = {ds: load_asr(ds) for ds in args.datasets}

    judge = Judge(model_id=args.model)
    choice_ids = [judge.label_ids(c) for c in CHOICES]
    flat = [t for c in choice_ids for t in c]
    if len(flat) != len(set(flat)):
        raise SystemExit(f"A/B/C token sets are not disjoint: {choice_ids}")
    logging.info("choice ids %s", {c: choice_ids[k] for k, c in enumerate(CHOICES)})

    pf = open(out_dir / "pairs.jsonl", "w")
    rows, t0, skipped = [], time.time(), 0
    for n, row in enumerate(rows_all):
        key = (row["dataset"], row["video_id"])
        if key not in spvl or key not in gt:
            skipped += 1
            continue
        args.video_index = n
        try:
            recs, info = run_video(judge, row, asr[row["dataset"]].get(row["video_id"], []),
                                   spvl[key], gt[key], args, choice_ids, verify=(len(rows) == 0))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.exception("video %s failed: %s", row["video_id"], exc)
            skipped += 1
            continue
        if info.get("verify"):
            (out_dir / "verify.json").write_text(json.dumps(info["verify"], indent=2))
        for r in recs:
            pf.write(json.dumps(r, ensure_ascii=False) + "\n")
        pf.flush()
        rows.extend(recs)
        if n % 10 == 0:
            logging.info("%d/%d %s pairs=%d elapsed=%.0fs", n + 1, len(rows_all), row["video_id"],
                         len(rows), time.time() - t0)
    pf.close()

    summary = {"per_dataset": summarize(rows, args.questions), "n_pairs_total": len(rows),
               "n_videos_skipped": skipped, "seconds": time.time() - t0, "config": cfg}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info("DONE %s", json.dumps(summary["per_dataset"], indent=2))


if __name__ == "__main__":
    main()
