#!/usr/bin/env python3
"""GLR measurement pass: generative likelihood ratio of what is actually said in each 8 s window.

The frozen MLLM is used as a conditional language model, not as a judge. Per video: prefix = 20 timestamped
frames + a transcript header, left as an open user turn. For every 8 s window with speech, the summed
log-probability of the window's words (teacher forcing, no generation) is read under

  2 context variants  full: the transcript lines of all earlier windows; none: no transcript
  2 framings          assistant: the condition is an instruction, the words are the assistant's reply
                      document:  the condition is a note on the window's transcript line, the words follow it
                                 inside the user turn
  4 conditions        H1 (the speaker violates the hate policy) and H0_discuss / H0_report / H0_crude

The window's own words never appear before the point where they are scored (they would be copied otherwise).
Labels are never read here. The analysis (glr_analyze.py) turns the reads into
LLR = [log p(x|H1) - log mean_k p(x|H0_k)] / n_tokens.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL  # noqa: E402
from src.video_inputs import FPS, fixed_windows, frame_paths, load_asr, load_manifest, window_text  # noqa: E402

CODE_PATH = "experiments/20260926_glr/glr_measure.py"
SEED = 0
MAX_TOKENS_X = 256  # cap on the scored words of one window (Whisper repetition loops can be very long)

CONDITIONS = {
    "H1": ("promotes hatred against people because of a protected attribute (race, ethnicity, religion, "
           "nationality, sexual orientation, gender identity or disability): encouraging violence against them, "
           "inciting hatred, dehumanizing them, calling them inferior, using slurs or hateful stereotypes "
           "(including as jokes), glorifying violence against them or hateful supremacism, spreading conspiracy "
           "claims about them, or denying major violent events"),
    "H0_discuss": ("talks about people or topics without promoting hatred against any group: ordinary "
                   "conversation, opinion, commentary or discussion"),
    "H0_report": ("reports on, quotes or criticizes the hateful words or acts of others without endorsing them, "
                  "as in news, documentary, education or counter-speech"),
    "H0_crude": "is crude or rude and may use profanity, but does not attack anyone because of a protected attribute",
}
CONTEXTS = ("full", "none")
FRAMINGS = ("assistant", "document")
TRANSCRIPT_HEADER = "\nTranscript of the speech in the video, in 8-second parts:\n"
NOT_SHOWN = "(not shown)\n"


def clean(text):
    return " ".join((text or "").split())


def past_line(a, b, text):
    return f"[{a:.1f}s-{b:.1f}s] {text}\n"


def head_text(framing, cond, a, b, close):
    if framing == "assistant":
        return (f"In the part of the video from {a:.1f}s to {b:.1f}s, the speaker {CONDITIONS[cond]}.\n"
                f"Write exactly what is said in that part.{close}")
    return f"[{a:.1f}s-{b:.1f}s] (in this part the speaker {CONDITIONS[cond]})"


def cont_text(framing, text):
    return text if framing == "assistant" else " " + text


class Scorer:
    def __init__(self, judge):
        self.j = judge
        probe = judge.render([{"role": "user", "content": [{"type": "text", "text": "QZX"}]}], add_generation_prompt=True)
        self.close = probe[probe.index("QZX") + 3:]  # end of the user turn + assistant header, from the template

    def ids(self, text):
        return self.j.tok(text, add_special_tokens=False)["input_ids"]

    def prefix(self, frames):
        from PIL import Image
        content = [{"type": "text", "text": f"Below are {len(frames)} frames sampled from a video, each preceded by its timestamp.\n"}]
        for t, _ in frames:
            content.append({"type": "text", "text": f"[t={t:.1f}s]\n"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": TRANSCRIPT_HEADER})
        msgs = [{"role": "user", "content": content}]
        full = self.j.render(self.j._append_text(msgs, self.j.SENTINEL), add_generation_prompt=False)
        text = full[:full.index(self.j.SENTINEL)]  # open user turn, ends with the header's newline
        images = [Image.open(p).convert("RGB") for _, p in frames]
        enc = self.j.encode(text, images)
        for im in images:
            im.close()
        return text, enc, [p for _, p in frames]

    def read(self, cache, h_ids, x_ids):
        """Summed log p(x | cache, head) with an in-place branch that is cropped back afterwards."""
        n0 = cache.get_seq_length()
        lp = self.j.cached_logprobs(cache, h_ids, x_ids, in_place=True)
        cache.crop(n0)
        return float(np.sum(lp))


def plain_logprob(scorer, text_before, image_files, x_text):
    """Reference: one ordinary forward over the whole string (no cache)."""
    from PIL import Image
    j = scorer.j
    images = [Image.open(p).convert("RGB") for p in image_files]
    enc = j.encode(text_before + x_text, images)
    for im in images:
        im.close()
    x_ids = scorer.ids(x_text)
    ids = enc["input_ids"][0].tolist()
    with torch.no_grad():
        out = j.model.model(**j.model_inputs(enc), use_cache=False)
        n = len(x_ids)
        h = out.last_hidden_state[0, len(ids) - n - 1:len(ids) - 1]
        del out
    return float(np.sum(j.token_logprobs(h, ids[-n:]))), ids


def score_video(scorer, row, segments, args, verify=False):
    j = scorer.j
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    wins = fixed_windows(dur, args.window_seconds)
    frames = frame_paths(ds, vid, args.frames, "k20")
    if not frames:
        return None, {"error": "no frames cached"}
    prefix_text, enc, image_files = scorer.prefix(frames)
    prefix_ids = enc["input_ids"][0].tolist()
    cache_full = j.prefix_cache(enc)
    cache_none = copy.deepcopy(cache_full)
    j.extend_cache(cache_none, scorer.ids(NOT_SHOWN))
    texts = [clean(window_text(segments, a, b)) for a, b in wins]
    info = {"prefix_tokens": len(prefix_ids), "n_windows": len(wins), "n_frames": len(frames)}
    per, past_ids, past_text, n_branch = [], [], "", 0
    vcheck = None
    speech_idx = [i for i, t in enumerate(texts) if t]
    check_i = speech_idx[1] if len(speech_idx) > 1 else (speech_idx[0] if speech_idx else None)
    for i, ((a, b), t) in enumerate(zip(wins, texts)):
        rec = {"i": i, "start": a, "end": b, "has_speech": bool(t)}
        if t:
            lp = {}
            for fr in FRAMINGS:
                x_ids = scorer.ids(cont_text(fr, t))[:MAX_TOKENS_X]
                rec[f"ntok_{fr}"] = len(x_ids)
                for ctx, cache in (("full", cache_full), ("none", cache_none)):
                    for cond in CONDITIONS:
                        h_ids = scorer.ids(head_text(fr, cond, a, b, scorer.close))
                        lp[f"{ctx}|{fr}|{cond}"] = scorer.read(cache, h_ids, x_ids)
                        n_branch += 1
            rec["lp"] = lp
            if verify and i == check_i:
                vcheck = {"window": i}
                for fr in FRAMINGS:
                    # token seam: prefix + past lines + head + words must equal the tokenization of the whole string
                    h_txt = head_text(fr, "H1", a, b, scorer.close)
                    x_txt = cont_text(fr, t)
                    if len(scorer.ids(x_txt)) > MAX_TOKENS_X:
                        continue
                    ref_lp, ref_ids = plain_logprob(scorer, prefix_text + past_text + h_txt, image_files, x_txt)
                    mine = prefix_ids + past_ids + scorer.ids(h_txt) + scorer.ids(x_txt)
                    if mine != ref_ids:
                        k = next((q for q, (u, v) in enumerate(zip(mine, ref_ids)) if u != v), min(len(mine), len(ref_ids)))
                        raise SystemExit(f"TOKEN SEAM MISMATCH ({fr}) at {k}: {len(mine)} vs {len(ref_ids)}")
                    ntok = rec[f"ntok_{fr}"]
                    vcheck[f"{fr}_cache_vs_plain_dlp_per_token"] = abs(lp[f"full|{fr}|H1"] - ref_lp) / max(ntok, 1)
                    # crop path vs an independent deep-copied branch
                    dc = j.cached_logprobs(cache_full, scorer.ids(h_txt), scorer.ids(x_txt)[:MAX_TOKENS_X], in_place=False)
                    vcheck[f"{fr}_crop_vs_deepcopy_dlp"] = abs(lp[f"full|{fr}|H1"] - float(np.sum(dc)))
                if any(v > 0.25 for k, v in vcheck.items() if k.endswith("per_token")) or \
                        any(v > 1e-2 for k, v in vcheck.items() if k.endswith("deepcopy_dlp")):
                    raise SystemExit(f"VERIFY GATE FAILED: {vcheck}")
            line = past_line(a, b, t)
            ids = scorer.ids(line)
            j.extend_cache(cache_full, ids)
            past_ids += ids
            past_text += line
        per.append(rec)
    del cache_full, cache_none
    info["n_branches"] = n_branch
    if verify:
        info["verify"] = {**(vcheck or {"window": None}), "close_string": scorer.close,
                          "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
    llr = []
    for r in per:
        if r["has_speech"]:
            lp = r["lp"]
            h0 = [lp[f"full|assistant|{c}"] for c in CONDITIONS if c.startswith("H0")]
            llr.append((lp["full|assistant|H1"] - (np.logaddexp.reduce(h0) - math.log(len(h0)))) / max(r["ntok_assistant"], 1))
        else:
            llr.append(None)
    vals = [v for v in llr if v is not None]
    fill = (min(vals) - 1.0) if vals else 0.0
    L = int(math.ceil(dur * FPS))
    idx = np.clip(np.floor(((np.arange(L) + 0.5) / FPS) / args.window_seconds).astype(int), 0, len(wins) - 1)
    curve = np.asarray([fill if v is None else v for v in llr], float)[idx]
    pred = {"schema_version": 1, "method": "glr_measure_w8", "dataset": ds, "video_id": vid, "duration": dur,
            "native_rate": FPS, "score_curve": [float(x) for x in curve], "intervals": [], "error": None,
            "calls": 1, "seed": SEED, "code_path": CODE_PATH,
            "extra": {"windows": per, **{k: v for k, v in info.items() if k != "verify"}}}
    return pred, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--exp-id", default="20260926_glr")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--window-seconds", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    out_dir = ROOT / "runs" / args.exp_id / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    cfg = dict(vars(args))
    cfg.update({"code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(), "seed": SEED,
                "conditions": CONDITIONS, "contexts": CONTEXTS, "framings": FRAMINGS, "max_tokens_x": MAX_TOKENS_X,
                "transcript_header": TRANSCRIPT_HEADER, "not_shown": NOT_SHOWN,
                "head_assistant_example": head_text("assistant", "H1", 8.0, 16.0, "<CLOSE>"),
                "head_document_example": head_text("document", "H1", 8.0, 16.0, "")})
    rows = load_manifest(args.manifest, args.datasets)
    if args.limit:
        rows = rows[:args.limit]
    asr = {ds: load_asr(ds) for ds in args.datasets}
    logging.info("videos %d", len(rows))
    judge = Judge(model_id=args.model)
    scorer = Scorer(judge)
    import transformers
    cfg.update({"family": judge.family, "transformers": transformers.__version__, "torch": torch.__version__,
                "close_string": scorer.close})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    pred_path = out_dir / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in open(pred_path):
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["dataset"], r["video_id"]))
    n_err, t0, n_new = 0, time.time(), 0
    with open(pred_path, "a") as fh:
        for n, row in enumerate(rows):
            key = (row["dataset"], row["video_id"])
            if key in done:
                continue
            segments = asr[row["dataset"]].get(row["video_id"], [])
            try:
                pred, info = score_video(scorer, row, segments, args, verify=(args.verify_only or n_new == 0))
                if pred is None:
                    raise RuntimeError(info.get("error", "unknown"))
                if "verify" in info:
                    logging.info("VERIFY %s %s", row["video_id"], json.dumps(info["verify"]))
                    (out_dir / "verify.json").write_text(json.dumps({"video_id": row["video_id"], **info}, indent=2, default=str))
                    if args.verify_only:
                        logging.info("DONE verify-only")
                        return
                fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
                fh.flush()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                n_err += 1
                logging.error("OOM %s %s", row["video_id"], str(exc)[:200])
                fh.write(json.dumps({"method": "glr_measure_w8", "dataset": row["dataset"], "video_id": row["video_id"],
                                     "score_curve": [], "intervals": [], "error": "OOM"}) + "\n")
            except SystemExit:
                raise
            except Exception as exc:
                n_err += 1
                logging.exception("FAILED %s: %s", row["video_id"], exc)
                fh.write(json.dumps({"method": "glr_measure_w8", "dataset": row["dataset"], "video_id": row["video_id"],
                                     "score_curve": [], "intervals": [], "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            n_new += 1
            if n_new % 10 == 0:
                logging.info("progress %d/%d  %.1fs/video  errors %d", n + 1, len(rows), (time.time() - t0) / n_new, n_err)
    logging.info("DONE videos=%d errors=%d elapsed=%.0fs", len(rows), n_err, time.time() - t0)


if __name__ == "__main__":
    main()
