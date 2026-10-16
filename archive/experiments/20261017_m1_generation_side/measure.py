#!/usr/bin/env python3
"""Candidate 45 GPU stage (README §2): per 8-s window with speech, on the r6 stance cache of the video,
  arm A  verbatim repetition under teacher forcing -> per-token -log p of the window text (copy_mean, copy_max, first_loss,
         refuse_mass = log probability mass of refusal starters at the first answer position);
  arm B  minimal compliant rewrite, greedy decoding -> rewritten text and word-level edit_rate.
The prefix, stance conditioning, frames and the proportional window text are exactly r6's; visual reads are not recomputed
(analyze.py takes z_visual from the stored r6 run). Replay check: on the first two videos of each corpus the r6 speech read is
recomputed and compared with the stored z_speech. GPU inside Slurm only; no label file is read.
Output: runs/20261017_m1_generation_side/<name>/{records/<ds>/<vid>.json, run.log, run.pid, config.json}.
"""
import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (A_SCORES, EXP, IMPLEMENTATION, MAX_NEW_CAP, NATIVE_FRAMES, REFUSAL_STARTERS, WINDOW_S, edit_scores,  # noqa: E402
                    norm_body, question_repeat, question_rewrite, selected_rows, stored_r6)
from src.mllm_judge import MODEL, Judge, yesno_question  # noqa: E402
from src.stance_cache import build, margin  # noqa: E402
from src.video_inputs import fixed_windows, frame_paths, load_asr, window_text  # noqa: E402


def tick():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def lm_head_fp32(j):
    if getattr(j, "_W32", None) is None:
        j._W32 = j.model.get_output_embeddings().weight.float()
    return j._W32


def refusal_ids(j):
    ids = set()
    for s in REFUSAL_STARTERS:
        for v in (s, " " + s):
            t = j.tok.encode(v, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


class Branch:
    """Independent branch on the stance cache: tokens appended in place, cropped back on exit (as src.stance_cache.margin)."""

    def __init__(self, j, cache, ctx):
        self.j, self.cache, self.ctx = j, cache, ctx

    def __enter__(self):
        self.n = self.cache.get_seq_length()
        self.j.model.model.rope_deltas = self.ctx["rope"].clone()
        return self

    def __exit__(self, *exc):
        self.cache.crop(self.n)
        self.j.model.model.rope_deltas = self.ctx["rope"].clone()
        return False


@torch.no_grad()
def repeat_read(j, cache, ctx, question, body, refuse):
    """Teacher forcing of `body` as the assistant answer to `question`; full-vocabulary fp32 log-softmax per position."""
    bids, _ = j.branch_ids(ctx["msgs"], question, ctx["history"], head_text=ctx["head"])
    aid, _ = j.answer_ids(ctx["msgs"], question, body, ctx["history"])
    cont = j.tok(body, add_special_tokens=False)["input_ids"]
    if aid[:len(cont)] != cont:
        raise AssertionError(f"answer tokens differ from the body tokens: {aid[:8]} vs {cont[:8]}")
    with Branch(j, cache, ctx):
        ids = bids + cont
        out = j.model.model(input_ids=torch.tensor([ids], device=j.device), past_key_values=cache, use_cache=True)
        m = len(bids)
        h = out.last_hidden_state[0, m - 1:m - 1 + len(cont)]
        del out
        lp = torch.log_softmax(h.float() @ lm_head_fp32(j).T, -1)
        tok_lp = lp.gather(1, torch.tensor(cont, device=j.device)[:, None])[:, 0]
        own = [t for t in refuse if t != cont[0]]          # README §2: the body's own first token never counts as refusal
        refuse_mass = float(torch.logsumexp(lp[0, own], 0)) if own else -50.0
        losses = (-tok_lp).tolist()
        del lp, h
    return dict(copy_mean=sum(losses) / len(losses), copy_max=max(losses), first_loss=losses[0], refuse_mass=refuse_mass,
                n_tokens=len(cont), token_losses=[round(x, 4) for x in losses], tokens=[j.tok.decode([t]) for t in cont])


@torch.no_grad()
def rewrite_read(j, cache, ctx, question, body):
    """Greedy decoding of the rewrite; stops at an end-of-turn token or at min(256, 2 * body tokens + 16) tokens."""
    bids, _ = j.branch_ids(ctx["msgs"], question, ctx["history"], head_text=ctx["head"])
    n_body = len(j.tok(body, add_special_tokens=False)["input_ids"])
    max_new = min(MAX_NEW_CAP, 2 * n_body + 16)
    W = lm_head_fp32(j)
    gen = []
    with Branch(j, cache, ctx):
        h = j._step(cache, bids)
        for _ in range(max_new):
            nxt = int((h.float() @ W.T).argmax())
            if nxt in j.eos_ids:
                break
            gen.append(nxt)
            h = j._step(cache, [nxt])
    text = j.tok.decode(gen, skip_special_tokens=True).strip()
    return dict(rewrite=text, **edit_scores(body, text), n_gen=len(gen), hit_max=len(gen) >= max_new, max_new=max_new)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="main")
    ap.add_argument("--arms", default="AB", help="which arms to run (A, B or AB)")
    a = ap.parse_args()
    out = EXP / a.name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    import transformers
    cfg = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d"), model=MODEL, implementation=IMPLEMENTATION, arms=a.arms,
               window_seconds=WINDOW_S, native_frames=NATIVE_FRAMES, max_new_cap=MAX_NEW_CAP, refusal_starters=list(REFUSAL_STARTERS),
               question_repeat=question_repeat(0, 1, 0.0, 8.0, "<text>"), question_rewrite=question_rewrite(0, 1, 0.0, 8.0, "<text>"),
               GT_read=False, smoke=a.smoke, datasets=a.datasets, torch=torch.__version__, transformers=transformers.__version__,
               seed=0, decoding="greedy / teacher forcing", command="python -u " + " ".join(sys.argv),
               source="experiments/20261017_m1_generation_side/measure.py; 2026-10-17")
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    rows = selected_rows(a.datasets, a.smoke)
    r6 = stored_r6()
    asr = {ds: load_asr(ds) for ds in a.datasets}
    torch.manual_seed(0)
    j = Judge(MODEL)
    refuse = refusal_ids(j)
    logging.info("refusal starter token ids %s -> %s", refuse, [j.tok.decode([t]) for t in refuse])
    replay_videos = {ds: [x["video_id"] for x in rows if x["dataset"] == ds][:2] for ds in a.datasets}
    done = 0
    for ordinal, row in enumerate(rows, 1):
        ds, vid = row["dataset"], row["video_id"]
        path = out / "records" / ds / f"{vid}.json"
        if path.exists():
            done += 1
            continue
        begin = tick()
        segments = asr[ds].get(vid, [])
        native = frame_paths(ds, vid, NATIVE_FRAMES)
        cache, ctx = build(j, native, segments)
        wins = fixed_windows(float(row["duration"]), WINDOW_S)
        n = len(wins)
        ref = r6[(ds, vid)]["extra"]["windows"]
        assert len(ref) == n, (vid, len(ref), n)
        traces, replay, t_a, t_b = [], [], 0.0, 0.0
        for i, (t1, t2) in enumerate(wins):
            raw_text = window_text(segments, t1, t2)
            assert (ref[i].get("z_speech") is not None) == bool(raw_text.strip()), (vid, i)
            body = norm_body(raw_text)
            t = dict(i=i, bounds=[t1, t2], text=body, z_visual_r6=ref[i]["z_visual"], z_speech_r6=ref[i].get("z_speech"))
            if body:
                if "A" in a.arms:
                    s = tick()
                    t["A"] = repeat_read(j, cache, ctx, question_repeat(i, n, t1, t2, body), body, refuse)
                    t_a += tick() - s
                if "B" in a.arms:
                    s = tick()
                    t["B"] = rewrite_read(j, cache, ctx, question_rewrite(i, n, t1, t2, body), body)
                    t_b += tick() - s
                if vid in replay_videos[ds]:
                    zp = margin(j, cache, ctx, yesno_question(i, n, t1, t2, raw_text, "speech"))   # r6 wording on r6's raw slice
                    t["z_speech_replay"] = zp
                    replay.append(abs(zp - ref[i]["z_speech"]))
            traces.append(t)
        secs = tick() - begin
        bundle = dict(implementation=IMPLEMENTATION, model=MODEL, host=socket.gethostname(), GT_read=False, dataset=ds, video_id=vid,
                      duration=float(row["duration"]), stance=ctx["stance"], stance_r6=r6[(ds, vid)]["extra"]["stance"],
                      global_margin=ctx["global_margin"], z_video_r6=r6[(ds, vid)]["extra"]["z_video"], prefix_tokens=ctx["prefix_tokens"],
                      stance_cache_tokens=ctx["stance_cache_tokens"], windows=n, speech_windows=sum(bool(t["text"]) for t in traces),
                      seconds=secs, seconds_A=t_a, seconds_B=t_b, replay_max_abs=max(replay) if replay else None,
                      peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, traces=traces)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle, allow_nan=False) + "\n")
        done += 1
        ex = next((t for t in traces if t.get("A")), None)
        logging.info("%d/%d %s/%s windows=%d speech=%d %.0fs (A %.0fs, B %.0fs) stance=%s/%s replay_max_abs=%s copy_mean=%s edit_rate=%s",
                     ordinal, len(rows), ds, vid, n, bundle["speech_windows"], secs, t_a, t_b, ctx["stance"], bundle["stance_r6"],
                     None if bundle["replay_max_abs"] is None else round(bundle["replay_max_abs"], 4),
                     None if ex is None else round(ex["A"]["copy_mean"], 3), None if ex is None or "B" not in ex else round(ex["B"]["edit_rate"], 3))
        del cache
    logging.info("DONE coverage=%d/%d", done, len(rows))


if __name__ == "__main__":
    main()
