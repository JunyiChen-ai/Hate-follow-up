#!/usr/bin/env python3
"""Candidate 46 stage 2 (README §2): speech reads with the topic direction removed from the residual stream.
Per video the r6 stance cache is rebuilt (prefix + verdict, untouched by the hooks); per window with speech the r6 speech
question is read (a) with the topic direction projected out at the embedding output and every decoder-layer output of the
branch forward (z_abl), and (b) five times with random unit directions instead (z_rand[k], seeds 0-4). The mean projection
of the branch residuals onto the topic direction per layer is kept for the report. First two videos per corpus also replay
the plain r6 speech read as a check. Visual reads come from the stored r6 run. GPU inside Slurm only; no label file read.
Output: runs/20261017_m1_topic_removal/<name>/{records/<ds>/<vid>.json, run.log, run.pid, config.json}.
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
from common import DIRECTIONS, EXP, IMPLEMENTATION, N_RAND, NATIVE_FRAMES, WINDOW_S, selected_rows, stored_r6  # noqa: E402
from src.mllm_judge import MODEL, Judge, yesno_question  # noqa: E402
from src.stance_cache import build, margin  # noqa: E402
from src.video_inputs import fixed_windows, frame_paths, load_asr, window_text  # noqa: E402


def tick():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class Ablator:
    """Directional ablation h <- h - (h.u) u at the embedding output and every decoder-layer output, all positions of the
    current forward. Inactive (plain forward) while `dirs` is None. `proj[l]` = mean projection before removal."""

    def __init__(self, j):
        self.lm = j.model.model.language_model
        self.points = [self.lm.embed_tokens] + list(self.lm.layers)
        self.dirs, self.proj = None, None
        self.hooks = [m.register_forward_hook(self._make(i)) for i, m in enumerate(self.points)]

    def _make(self, i):
        def hook(module, args, output):
            if self.dirs is None:
                return None
            h = output[0] if isinstance(output, tuple) else output
            assert h.ndim == 3 and h.shape[0] == 1, h.shape
            u = self.dirs[i]
            p = h.float() @ u                                  # (1, T)
            self.proj[i] = float(p[0].mean())
            h2 = (h.float() - p[..., None] * u).to(h.dtype)
            return (h2, *output[1:]) if isinstance(output, tuple) else h2
        return hook


@torch.no_grad()
def read(j, cache, ctx, question, abl, dirs):
    """Yes/No log-odds of `question` on the stance cache with `dirs` removed (None = plain read); cache restored."""
    n = cache.get_seq_length()
    j.model.model.rope_deltas = ctx["rope"].clone()
    ids, _ = j.branch_ids(ctx["msgs"], question, ctx["history"], head_text=ctx["head"])
    abl.dirs, abl.proj = dirs, [None] * len(abl.points)
    try:
        z = j.cached_margin(cache, ids, in_place=True)
        proj = list(abl.proj)
    finally:
        abl.dirs = None
        cache.crop(n)
        j.model.model.rope_deltas = ctx["rope"].clone()
    return z, proj


def random_directions(L, H, seed, device):
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(L, H, generator=g)
    return (R / R.norm(dim=-1, keepdim=True)).to(device)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="main")
    a = ap.parse_args()
    out = EXP / a.name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    import transformers
    cfg = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d"), model=MODEL, implementation=IMPLEMENTATION, directions=str(DIRECTIONS.relative_to(DIRECTIONS.parents[3])),
               n_random=N_RAND, random_seeds=list(range(N_RAND)), ablated_points="embedding output + every decoder-layer output, all positions of the branch forward",
               window_seconds=WINDOW_S, native_frames=NATIVE_FRAMES, GT_read=False, smoke=a.smoke, datasets=a.datasets, torch=torch.__version__,
               transformers=transformers.__version__, seed=0, command="python -u " + " ".join(sys.argv), source="experiments/20261017_m1_topic_removal/measure.py; 2026-10-17")
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    rows = selected_rows(a.datasets, a.smoke)
    r6 = stored_r6()
    asr = {ds: load_asr(ds) for ds in a.datasets}
    torch.manual_seed(0)
    j = Judge(MODEL)
    abl = Ablator(j)
    L, H = len(abl.points), j.model.config.text_config.hidden_size
    topic = torch.load(DIRECTIONS).float()
    assert topic.shape == (L, H) and torch.isfinite(topic).all() and torch.allclose(topic.norm(dim=-1), torch.ones(L), atol=1e-4), topic.shape
    topic = topic.to(j.device)
    rand = [random_directions(L, H, k, j.device) for k in range(N_RAND)]
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
        assert abl.dirs is None
        cache, ctx = build(j, native, segments)
        wins = fixed_windows(float(row["duration"]), WINDOW_S)
        n = len(wins)
        ref = r6[(ds, vid)]["extra"]["windows"]
        assert len(ref) == n, (vid, len(ref), n)
        traces, replay = [], []
        for i, (t1, t2) in enumerate(wins):
            body = window_text(segments, t1, t2)
            assert (ref[i].get("z_speech") is not None) == bool(body.strip()), (vid, i)
            t = dict(i=i, bounds=[t1, t2], text=body.strip(), z_visual_r6=ref[i]["z_visual"], z_speech_r6=ref[i].get("z_speech"))
            if body.strip():
                q = yesno_question(i, n, t1, t2, body, "speech")
                z, proj = read(j, cache, ctx, q, abl, topic)
                t["z_abl"], t["proj_topic"] = z, [round(p, 4) for p in proj]
                t["z_rand"] = [read(j, cache, ctx, q, abl, R)[0] for R in rand]
                if vid in replay_videos[ds]:
                    zp = margin(j, cache, ctx, q)
                    t["z_speech_replay"] = zp
                    replay.append(abs(zp - ref[i]["z_speech"]))
            traces.append(t)
        secs = tick() - begin
        bundle = dict(implementation=IMPLEMENTATION, model=MODEL, host=socket.gethostname(), GT_read=False, dataset=ds, video_id=vid,
                      duration=float(row["duration"]), stance=ctx["stance"], stance_r6=r6[(ds, vid)]["extra"]["stance"], global_margin=ctx["global_margin"],
                      z_video_r6=r6[(ds, vid)]["extra"]["z_video"], prefix_tokens=ctx["prefix_tokens"], stance_cache_tokens=ctx["stance_cache_tokens"],
                      windows=n, speech_windows=sum(bool(t["text"]) for t in traces), seconds=secs, replay_max_abs=max(replay) if replay else None,
                      peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, traces=traces)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle, allow_nan=False) + "\n")
        done += 1
        ex = [t for t in traces if "z_abl" in t]
        logging.info("%d/%d %s/%s windows=%d speech=%d %.0fs stance=%s/%s replay_max_abs=%s  mean z_speech %.2f -> z_abl %.2f (rand %.2f)",
                     ordinal, len(rows), ds, vid, n, bundle["speech_windows"], secs, ctx["stance"], bundle["stance_r6"],
                     None if bundle["replay_max_abs"] is None else round(bundle["replay_max_abs"], 4),
                     sum(t["z_speech_r6"] for t in ex) / max(1, len(ex)), sum(t["z_abl"] for t in ex) / max(1, len(ex)),
                     sum(sum(t["z_rand"]) / N_RAND for t in ex) / max(1, len(ex)))
        del cache
    logging.info("DONE coverage=%d/%d", done, len(rows))


if __name__ == "__main__":
    main()
