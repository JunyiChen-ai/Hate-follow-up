#!/usr/bin/env python3
"""HVL: hypothesis–verification localization (label-free, one frozen MLLM, prefix KV cache).

Per video (README §2):
  S1 verdict + hypothesis : whole-video Yes/No log-odds z_video; then a greedy, fixed-format hypothesis
                            (TARGET / FORM / EVIDENCE with timestamps), appended to the context.
  S2 verification         : per 8 s window, visual and speech branches; either independent Yes/No
                            ("is this window part of the evidence?") or a sequential four-state chain
                            (start / continue / stop / none) where window i follows the model's own answer
                            for window i-1. Window score = log-odds of "present".
  S3 revision             : the verification summary is shown and the whole-video question asked again -> z_rev.
  S4 composition          : compose.py (intercept + centred-rank residual), shared evaluator.
No labels are read here except, under --only-within-defined, to pick the pilot subset (rule 10).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION  # noqa: E402
from src.video_inputs import (FPS, fixed_windows, frame_paths, load_asr, load_manifest,  # noqa: E402
                              window_text, within_defined_ids)

CODE_PATH = "experiments/20260911_hvl/hvl.py"
SEED = 0
FILL_UNCOVERED = -12.0
MAX_HYP_TOKENS = 220

HYP_QUESTION = (
    "Summarise your judgement as a hypothesis in exactly this format, three lines and nothing else:\n"
    "TARGET: <the group or groups the content targets, or none>\n"
    "FORM: <the rule numbers from the list above that are violated, or none>\n"
    "EVIDENCE: <up to 5 items separated by ';', each written as start-end: what is said or shown there, "
    "using the transcript or frame timestamps in seconds; or none>"
)
HYP_QUESTION_TF = (
    "Summarise your judgement as a hypothesis in exactly this format, two lines and nothing else:\n"
    "TARGET: <the group or groups the content targets, or none>\n"
    "FORM: <the rule numbers from the list above that are violated, or none>"
)
STATES = ["start", "continue", "stop", "none"]


def state_question(i, n, t1, t2, text, kind, first, has_hyp=True):
    body = text.strip() if text and text.strip() else "(no speech)"
    head = f"Window {i + 1} of {n}, from {t1:.1f}s to {t2:.1f}s."
    if kind == "visual":
        head += " Judge only the visual content of this window (frames whose timestamps fall inside it), ignoring the speech."
    elif kind == "speech":
        head += f" Judge only what is spoken in this window: {body}"
    else:
        head += f" Transcript in this window: {body}"
    prev = "There is no previous window." if first else "Take your answer for the previous window into account."
    ref = "the hypothesis above" if has_hyp else "content that violates the above rules"
    return (f"{head}\n\n{prev} Relative to {ref}, which one describes THIS window: "
            "\"start\" (the violating content is present here and was not present in the previous window), "
            "\"continue\" (present here and also in the previous window), "
            "\"stop\" (not present here but present in the previous window), or "
            "\"none\" (not present here and not in the previous window)?\n\nAnswer with one word.")


def yesno_question(i, n, t1, t2, text, kind, has_hyp=True, neighbours=None):
    body = text.strip() if text and text.strip() else "(no speech)"
    head = f"Consider only window {i + 1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video."
    if neighbours:  # local temporal context (gap G1 without a chain): what is said just before and after
        prev_t, next_t = neighbours
        head += (f" For context only: the previous window says \"{prev_t or '(no speech)'}\"; "
                 f"the next window says \"{next_t or '(no speech)'}\". Judge THIS window, not the context.")
    if kind == "visual":
        ctx = (f"{head} Look only at the frames whose timestamps fall inside this window and judge the visual "
               f"content alone (imagery, gestures, symbols, on-screen text), ignoring the speech.\n\n")
        q = ("Is THIS window one of the segments where the visual evidence for the hypothesis above occurs?" if has_hyp
             else "Is THIS window one of the segments where visual content that violates the above rules occurs?")
    elif kind == "speech":
        ctx = f"{head} Judge only what is spoken in this window: {body}\n\n"
        q = ("Is THIS window one of the segments where the spoken evidence for the hypothesis above occurs?" if has_hyp
             else "Is THIS window one of the segments where speech that violates the above rules occurs?")
    else:
        ctx = f"{head} Transcript in this window: {body}\n\n"
        q = ("Is THIS window one of the segments where the evidence for the hypothesis above occurs?" if has_hyp
             else "Is THIS window one of the segments where the content that violates the above rules occurs?")
    return ctx + q + '\n\nAnswer "Yes" or "No".'


def parse_hypothesis(text):
    out = {"target": None, "form": [], "evidence": [], "raw": text}
    for line in text.splitlines():
        low = line.strip()
        if low.upper().startswith("TARGET:"):
            out["target"] = low.split(":", 1)[1].strip()
        elif low.upper().startswith("FORM:"):
            out["form"] = [int(x) for x in re.findall(r"\b([1-9])\b", low.split(":", 1)[1])]
        elif low.upper().startswith("EVIDENCE:"):
            body = low.split(":", 1)[1]
            for item in re.split(r";", body):
                m = re.search(r"(\d+(?:\.\d+)?)\s*s?\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*s?\s*:?\s*(.*)", item)
                if m:
                    a, b = float(m.group(1)), float(m.group(2))
                    if b > a:
                        out["evidence"].append({"start": a, "end": b, "note": m.group(3).strip()[:200]})
    t = (out["target"] or "").lower()
    out["is_none"] = (t in ("", "none", "n/a", "no", "not applicable")) and not out["evidence"]
    return out


def score_video(judge, row, segments, args, verify=False):
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    wins = fixed_windows(dur, args.window_seconds)
    frames = frame_paths(ds, vid, args.frames, "k20") if args.frames > 0 else []
    if args.frames > 0 and not frames:
        return None, {"error": "no frames cached"}
    if args.fill_frames and frames:  # windows without a uniform frame inside get their centre frame (data/frames_w8)
        have = {int(t // args.window_seconds) for t, _ in frames}
        extra = [(t, f) for t, f in frame_paths(ds, vid, 0, "w8") if int(t // args.window_seconds) not in have]
        frames = sorted(frames + extra, key=lambda x: x[0])
    msgs, image_files = judge.prefix_messages(frames, segments, with_context=not args.no_transcript_context,
                                              with_frames=args.frames > 0)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    prefix_ids = enc["input_ids"][0].tolist()
    info = {"prefix_tokens": len(prefix_ids), "n_windows": len(wins), "img_tokens": judge.img_tokens[:1],
            "n_frames": len(frames)}
    cache = judge.prefix_cache(enc)
    # ---- S1 verdict
    b0, b0_text = judge.branch_ids(msgs, VIDEO_QUESTION)
    if verify:
        judge.seam_check_tokens(msgs, image_files, prefix_ids, VIDEO_QUESTION, b0)
    z_video = judge.cached_margin(cache, b0, in_place=True)
    stance = "Yes" if z_video > 0 else "No"
    a0, a0_text = judge.answer_ids(msgs, VIDEO_QUESTION, stance)
    judge.extend_cache(cache, a0)
    history = [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]}, judge.turn("assistant", stance)]
    head = prefix_text + b0_text + a0_text
    hyp = None
    if args.hypothesis in ("structured", "target_form"):
        hq = HYP_QUESTION if args.hypothesis == "structured" else HYP_QUESTION_TF
        b1, b1_text = judge.branch_ids(msgs, hq, history, head_text=head)
        if args.hyp_permute is not None:  # control: another video's hypothesis is written as this model's answer
            hyp_text = args.hyp_permute[(ds, vid)]
            judge.extend_cache(cache, b1)
            judge.extend_cache(cache, judge.tok(hyp_text, add_special_tokens=False)["input_ids"])
        else:
            hyp_text, gen_ids = judge.cached_generate(cache, b1, MAX_HYP_TOKENS, in_place=True)
        hyp = parse_hypothesis(hyp_text)
        a1, a1_text = judge.answer_ids(msgs, hq, hyp_text, history)
        # the generated tokens are already in the cache; append only the end-of-turn that the template adds
        eot = judge.tok(a1_text[len(hyp_text):] if a1_text.startswith(hyp_text) else a1_text, add_special_tokens=False)["input_ids"] \
            if a1_text.startswith(hyp_text) else None
        if eot is None:  # template re-rendered the answer differently: replay the whole answer turn text instead
            raise AssertionError("hypothesis turn rendering does not start with the generated text")
        judge.extend_cache(cache, eot)
        history = history + [{"role": "user", "content": [{"type": "text", "text": hq}]}, judge.turn("assistant", hyp_text)]
        head = head + b1_text + a1_text
    info["hypothesis"] = hyp
    # ---- optional context filtering (gap 5): rebuild the prefix with only the cited sentences (+-1 segment)
    if args.context == "cited" and hyp and hyp["evidence"]:
        keep = set()
        for k, (s, e, _) in enumerate(segments):
            for ev in hyp["evidence"]:
                if e > ev["start"] - 1e-6 and s < ev["end"] + 1e-6:
                    keep.update({k - 1, k, k + 1})
        segs2 = [seg for k, seg in enumerate(segments) if k in keep]
        msgs2, image_files2 = judge.prefix_messages(frames, segs2, with_context=True, with_frames=args.frames > 0)
        prefix_text2, enc2 = judge.encode_prefix(msgs2, image_files2)
        del cache
        cache = judge.prefix_cache(enc2)
        head = prefix_text2
        for q, a in ((VIDEO_QUESTION, stance), (hq, hyp["raw"])):
            hq, _ = judge.branch_ids(msgs2, q, None if q == VIDEO_QUESTION else history[:2], head_text=head)
            judge.extend_cache(cache, hq)
            ha, ha_text = judge.answer_ids(msgs2, q, a, None if q == VIDEO_QUESTION else history[:2])
            judge.extend_cache(cache, ha)
            head = judge.render(judge.conv(msgs2, q, None if q == VIDEO_QUESTION else history[:2]) + [judge.turn("assistant", a)], False)
        msgs, prefix_text = msgs2, prefix_text2
        info["context_segments"] = len(segs2)
    # ---- S2 verification
    wtexts = [window_text(segments, a, b) for a, b in wins]
    kinds = {"joint": ["joint"], "dual": ["visual", "speech"]}[args.branches]
    per = [dict() for _ in wins]
    state_lp = [dict() for _ in wins]
    n_branch = 0
    if args.verify != "none":
        choices = [judge.label_ids(s) for s in STATES]
        flat = [t for c in choices for t in c]
        if len(set(flat)) != len(flat):
            raise SystemExit(f"state token sets overlap: {choices}")
        for kind in kinds:
            if kind == "visual" and args.frames == 0:
                continue
            chain = copy.deepcopy(cache) if args.verify == "sequential" else cache
            chain_head, chain_hist = head, list(history)
            prev_present = None
            order = list(range(len(wins)))
            if args.shuffle_windows:  # control: same chain, windows visited in a fixed pseudo-random order
                rng = np.random.RandomState(SEED + len(wins)); order = list(rng.permutation(len(wins)))
            for i in order:
                (a, b), t = wins[i], wtexts[i]
                if kind == "speech" and not (t and t.strip()) and args.verify != "sequential":
                    continue  # independent mode (SPVL semantics): no speech branch without speech
                if args.states == "four":
                    q = state_question(i, len(wins), a, b, t, kind, first=(prev_present is None), has_hyp=hyp is not None)
                else:
                    nb = None
                    if args.neighbours:
                        nb = (wtexts[i - 1].strip()[:300] if i > 0 else "(start of video)",
                              wtexts[i + 1].strip()[:300] if i + 1 < len(wins) else "(end of video)")
                    q = yesno_question(i, len(wins), a, b, t, kind, has_hyp=hyp is not None, neighbours=nb)
                bids, btext = judge.branch_ids(msgs, q, chain_hist, head_text=chain_head)
                n_branch += 1
                if args.states == "four":
                    lp = judge.cached_choices(chain, bids, choices, in_place=(args.verify == "sequential"))
                    present = float(np.logaddexp(lp[0], lp[1]) - np.logaddexp(lp[2], lp[3]))
                    state_lp[i][kind] = lp
                    ans = STATES[int(np.argmax(lp))]
                else:
                    present = judge.cached_margin(chain, bids, in_place=(args.verify == "sequential"))
                    ans = "Yes" if present > 0 else "No"
                per[i][kind] = present
                if args.verify == "sequential":  # the model's own answer stays in the chain
                    aids, atext = judge.answer_ids(msgs, q, ans, chain_hist)
                    judge.extend_cache(chain, aids)
                    chain_hist = chain_hist + [{"role": "user", "content": [{"type": "text", "text": q}]}, judge.turn("assistant", ans)]
                    chain_head = chain_head + btext + atext
                    prev_present = present > 0
            if args.verify == "sequential":
                del chain
    z_win = [max(d.values()) if d else FILL_UNCOVERED for d in per]
    # ---- S3 revision
    z_rev = None
    summary = None
    if args.revise == "on" and args.verify != "none":
        pres = [i for i, z in enumerate(z_win) if z > 0]
        best = int(np.argmax(z_win)) if z_win else None
        parts = [f"Verification summary: {len(pres)} of {len(wins)} windows were judged to contain the violating content"]
        if pres:
            parts.append("(" + ", ".join(f"{wins[i][0]:.0f}-{wins[i][1]:.0f}s" for i in pres[:12]) + (", ..." if len(pres) > 12 else "") + ")")
        if best is not None and z_win[best] > FILL_UNCOVERED:
            snippet = (wtexts[best] or "(no speech)").strip()[:300]
            parts.append(f". Strongest window: {wins[best][0]:.0f}-{wins[best][1]:.0f}s: \"{snippet}\"")
        summary = " ".join(parts) + ".\n\nTaking this verification into account, " + VIDEO_QUESTION[0].lower() + VIDEO_QUESTION[1:]
        rq, _ = judge.branch_ids(msgs, summary, history, head_text=head)
        z_rev = judge.cached_margin(cache, rq, in_place=False)
    del cache
    info.update({"n_branches": n_branch, "z_video": z_video, "z_rev": z_rev, "stance": stance})
    if verify:
        z_ref = judge.plain_margin(msgs, image_files, VIDEO_QUESTION)
        info["verify"] = {"video_q_cache_vs_plain_dz": abs(z_video - z_ref),
                          "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
        if abs(z_video - z_ref) >= 3.0:
            raise SystemExit(f"VERIFY GATE FAILED: {info['verify']}")
    L = int(math.ceil(dur * FPS))
    centers = (np.arange(L) + 0.5) / FPS
    idx = np.minimum((centers // args.window_seconds).astype(int), len(wins) - 1)
    curve = np.asarray(z_win, dtype=float)[idx]
    pred = {"schema_version": 1, "method": args.method_name, "dataset": ds, "video_id": vid, "duration": dur,
            "native_rate": FPS, "score_curve": [float(x) for x in curve], "intervals": [], "error": None,
            "calls": 3 if args.revise == "on" else 2, "seed": SEED, "code_path": CODE_PATH,
            "extra": {"z_video": z_video, "z_rev": z_rev, "stance": stance, "hypothesis": hyp, "summary": summary,
                      "windows": [{"i": i, "start": a, "end": b, "z": z, **{f"z_{k}": v for k, v in per[i].items()},
                                   **{f"lp_{k}": v for k, v in state_lp[i].items()}}
                                  for i, ((a, b), z) in enumerate(zip(wins, z_win))],
                      **{k: v for k, v in info.items() if k not in ("verify", "hypothesis")}}}
    return pred, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--exp-id", default="20260911_hvl")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--no-transcript-context", action="store_true")
    ap.add_argument("--window-seconds", type=float, default=8.0)
    ap.add_argument("--hypothesis", choices=["none", "structured", "target_form"], default="structured",
                    help="structured: TARGET/FORM/EVIDENCE; target_form: TARGET/FORM only (no self-cited timestamps)")
    ap.add_argument("--verify", choices=["none", "independent", "sequential"], default="sequential")
    ap.add_argument("--states", choices=["yesno", "four"], default="four")
    ap.add_argument("--revise", choices=["off", "on"], default="on")
    ap.add_argument("--context", choices=["full", "cited"], default="full")
    ap.add_argument("--fill-frames", action="store_true")
    ap.add_argument("--branches", choices=["joint", "dual"], default="dual")
    ap.add_argument("--neighbours", action="store_true", help="independent mode: show the previous and next window transcripts as context")
    ap.add_argument("--shuffle-windows", action="store_true", help="control: sequential chain visits windows in a fixed shuffled order")
    ap.add_argument("--hyp-from", default=None, help="control: predictions.jsonl of an earlier run; each video gets the NEXT video's hypothesis")
    ap.add_argument("--only-within-defined", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--method-name", default=None)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    out_dir = ROOT / "runs" / args.exp_id / (f"{args.model_tag}/{args.run_name}" if args.model_tag else args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    tag = args.method_name or "hvl_" + "_".join([
        f"f{args.frames}" + ("fill" if args.fill_frames else ""), "noctx" if args.no_transcript_context else f"ctx{args.context}",
        f"w{args.window_seconds:g}", f"hyp{args.hypothesis}", f"v{args.verify}", f"s{args.states}",
        f"rev{args.revise}", f"br{args.branches}"] + (["nb1"] if args.neighbours else []) + (["shuf"] if args.shuffle_windows else []) + (["hyppermute"] if args.hyp_from else []))
    if args.model_tag and not args.method_name:
        tag = f"{args.model_tag}__{tag}"
    args.method_name = tag
    cfg = {k: v for k, v in vars(args).items() if k != "hyp_permute"}
    cfg.update({"code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(),
                "seed": SEED, "fill_uncovered": FILL_UNCOVERED, "max_hyp_tokens": MAX_HYP_TOKENS,
                "hyp_question": HYP_QUESTION, "states": STATES, "video_question": VIDEO_QUESTION})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    rows = load_manifest(args.manifest, args.datasets)
    if args.only_within_defined:
        keep = within_defined_ids(args.datasets)
        rows = [r for r in rows if (r["dataset"], r["video_id"]) in keep]
    if args.limit:
        rows = rows[:args.limit]
    asr = {ds: load_asr(ds) for ds in args.datasets}
    logging.info("videos %d  config %s", len(rows), tag)
    args.hyp_permute = None
    if args.hyp_from:
        src = [json.loads(l) for l in open(args.hyp_from) if l.strip()]
        src = [r for r in src if not r.get("error") and (r["extra"].get("hypothesis") or {}).get("raw")]
        by_ds = {}
        for r in src:
            by_ds.setdefault(r["dataset"], []).append(r)
        args.hyp_permute = {}
        for ds, lst in by_ds.items():
            for k, r in enumerate(lst):
                args.hyp_permute[(ds, r["video_id"])] = lst[(k + 1) % len(lst)]["extra"]["hypothesis"]["raw"]
        rows = [r for r in rows if (r["dataset"], r["video_id"]) in args.hyp_permute]
        logging.info("hypothesis permutation control: %d videos", len(rows))
    judge = Judge(model_id=args.model)
    import transformers
    cfg.update({"family": judge.family, "same_turn": judge.same_turn, "transformers": transformers.__version__})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    pred_path = out_dir / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in open(pred_path):
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["dataset"], r["video_id"]))
    n_err, t0 = 0, time.time()
    with open(pred_path, "a") as fh:
        for n, row in enumerate(rows):
            key = (row["dataset"], row["video_id"])
            if key in done:
                continue
            segments = asr[row["dataset"]].get(row["video_id"], [])
            try:
                pred, info = score_video(judge, row, segments, args, verify=(args.verify_only or n == 0))
                if pred is None:
                    raise RuntimeError(info.get("error", "unknown"))
                if "verify" in info:
                    logging.info("VERIFY %s %s", row["video_id"], json.dumps(info["verify"]))
                    (out_dir / "verify.json").write_text(json.dumps({"video_id": row["video_id"], **info}, indent=2, default=str))
                    if args.verify_only:
                        logging.info("hypothesis: %s", json.dumps(info.get("hypothesis"), ensure_ascii=False))
                        logging.info("DONE verify-only")
                        return
                fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
                fh.flush()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                n_err += 1
                logging.error("OOM %s %s", row["video_id"], str(exc)[:200])
                fh.write(json.dumps({"method": tag, "dataset": row["dataset"], "video_id": row["video_id"],
                                     "score_curve": [], "intervals": [], "error": "OOM"}) + "\n")
            except Exception as exc:
                n_err += 1
                logging.exception("FAILED %s: %s", row["video_id"], exc)
                fh.write(json.dumps({"method": tag, "dataset": row["dataset"], "video_id": row["video_id"],
                                     "score_curve": [], "intervals": [], "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            if (n + 1) % 10 == 0:
                logging.info("progress %d/%d  %.1fs/video  errors %d", n + 1, len(rows), (time.time() - t0) / (n + 1), n_err)
    logging.info("DONE videos=%d errors=%d elapsed=%.0fs", len(rows), n_err, time.time() - t0)


if __name__ == "__main__":
    main()
