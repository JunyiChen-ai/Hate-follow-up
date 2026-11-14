#!/usr/bin/env python3
"""Probe-VAD (arXiv 2609.17211, preprint under review; github.com/yvestine/COVAS-VAD, docs/METHOD.md) on
Qwen3-VL-8B with frames + Whisper words. README §3.

Probe-VAD is training-free: a frozen video LLM looks at a fixed-length clip and answers ten ordered YES/NO questions,
"is the severity at least tau", tau = 0.1 .. 1.0. p_tau = softmax over the YES and NO next-token logits; the ten
probabilities are projected onto a non-increasing profile (pool-adjacent-violators); the clip score is their mean;
the clip-score sequence of a video is Gaussian-smoothed; a clip's score is given to the frames from its centre to
the next clip's centre. The clip is encoded once and the ten questions share that prefix.

Kept from the release: the window definition (centres c_i = i * stride, window [c_i - 5 s, c_i + 5 s] clipped to the
video), 2 fps sampling with at most 10 frames, the ten thresholds, single-token YES / NO candidates at T = 1, the
projection, the mean, Gaussian smoothing over the clip sequence, the expansion rule.
Changed (README §3 lists every deviation):
- backbone Qwen3-VL-8B (the release's default is VideoLLaMA3-7B; the paper also reports Qwen3-VL-8B), frames at the
  image size of the naive row; YES / NO logits from an fp32 output layer;
- the window's Whisper words are added to the user turn ("Transcript: <words>"), as in every row of §1-§2;
- the system prompt and the question are the release's sentences with the surveillance anomaly-severity scale
  replaced by a hatefulness scale (SYSTEM, QUESTION below; fixed before any run);
- stride: `--stride` seconds between clip centres (the release uses 16 frames, about 0.53 s at 30 fps), for cost;
  the smoothing sigma is given in seconds and converted to clips (the release's sigma = 10 clips = 160 frames,
  5.33 s at 30 fps).

Stages: `infer` (GPU, HateVLM env) -> runs/20261113_lf_baselines/probevad/<DS>/s<stride>/units.jsonl (one line per
video, resumable; the ten YES-NO logit differences of every clip); `raster` (CPU, HateVideo env) -> one directory
per smoothing sigma with predictions.jsonl and the canonical evaluator's metrics.json. No label is read.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_common as Q         # noqa: E402
from vllm_common import L       # noqa: E402

WINDOW = 10.0
SAMPLE_FPS = 2.0
MAX_FRAMES = 10
THRESHOLDS = [k / 10 for k in range(1, 11)]
RELEASE_SIGMA_S = 160 / 30.0     # the release's sigma (10 clips of 16 frames) in seconds at 30 fps
NO_SPEECH = "(no speech)"
# The release's system prompt (research_code/src/prompt_configs.py DEFAULT_SYSTEM_PROMPT) with the anomaly-severity
# scale of surveillance scenes replaced by a hatefulness scale, sentence by sentence.
SYSTEM = (
    "You are a careful hateful video detector. "
    "Judge only the evidence in the current video segment, its frames and its transcript. Use an ordered "
    "hatefulness scale from 0 to 1: 0 means ordinary non-hateful content; 0.5 "
    "means clearly concerning or plausibly hateful content; and 1 means "
    "unmistakably severe, explicit hateful content. "
    "Intermediate thresholds preserve this order. Do not infer content that is "
    "not supported by the segment."
)
Q_HEAD = "On the hatefulness scale defined by the system, is the hatefulness of this video segment at least"
Q_TAIL = " %.1f? Judge only the current segment's evidence. Answer exactly YES or NO."


def out_dir(ds, stride):
    return os.path.join(Q.RUNS, "probevad", ds, "s%g" % stride)


def clips(duration, stride):
    """(centre, start, end) of every clip: centres i * stride while < duration (docs/METHOD.md section 1)."""
    out, i = [], 0
    while i * stride < duration - 1e-9 or i == 0:
        c = i * stride
        out.append((c, max(0.0, c - WINDOW / 2), min(duration, c + WINDOW / 2)))
        i += 1
    return out


def pava_nonincreasing(p):
    """L2 projection of p onto non-increasing sequences (pool adjacent violators)."""
    vals, wts = [], []
    for x in p:
        vals.append(float(x))
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals[-2:] = [v]
            wts[-2:] = [w]
    out = []
    for v, w in zip(vals, wts):
        out.extend([v] * int(w))
    return np.asarray(out)


def clip_score(z):
    p = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
    return float(pava_nonincreasing(p).mean())


class Prober:
    """Ten threshold questions on one clip: the system message, the frames, the transcript and the question up to
    'at least' are forwarded once (prefix KV cache); the ten endings are forwarded as one batch on the expanded
    cache. YES - NO logit differences from an fp32 output layer."""

    def __init__(self, log):
        import torch
        from src.mllm_judge import Judge
        self.torch = torch
        self.j = Judge(Q.MODEL)
        self.log = log
        ids = {}
        for w in ("YES", "NO"):
            t = self.j.tok.encode(w, add_special_tokens=False)
            if len(t) != 1:
                raise SystemExit("%s is not one token: %s" % (w, t))
            ids[w] = t[0]
        self.cand = torch.tensor([ids["YES"], ids["NO"]], device=self.j.device)
        self.tails = None
        self.batched = True
        self.checked = 0

    def render(self, n_images, text, tau=None):
        user = "Transcript: %s\n\n%s" % (text or NO_SPEECH, Q_HEAD) + (Q_TAIL % tau if tau is not None else "")
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image"}] * n_images + [{"type": "text", "text": user}]}]
        return self.j.render(msgs, add_generation_prompt=True)

    def split(self, n_images, text):
        """(head string, list of ten tail strings): the rendered prompts differ only after Q_HEAD."""
        fulls = [self.render(n_images, text, tau) for tau in THRESHOLDS]
        cut = fulls[0].rindex(Q_HEAD) + len(Q_HEAD)
        if any(f[:cut] != fulls[0][:cut] for f in fulls):
            raise AssertionError("rendered prompts differ before the threshold")
        return fulls[0][:cut], [f[cut:] for f in fulls], fulls

    def _margin(self, hidden_rows):
        lg = self.j._logits_fp32(hidden_rows, self.cand)
        return (lg[:, 0] - lg[:, 1]).tolist()

    def plain(self, full, images):
        torch = self.torch
        with torch.inference_mode():
            enc = self.j.encode(full, images)
            out = self.j.model.model(**self.j.model_inputs(enc), use_cache=False)
            z = self._margin(out.last_hidden_state[0, -1:])[0]
            del out
        return z

    def score(self, image_files, text):
        torch = self.torch
        images = Q.open_images(image_files)
        head, tails, fulls = self.split(len(images), text)
        if self.tails is None:
            self.tails = [self.j.tok(t, add_special_tokens=False)["input_ids"] for t in tails]
            if len({len(t) for t in self.tails}) != 1:
                self.batched = False
                self.log("threshold endings have different token lengths: one forward per threshold")
        with torch.inference_mode():
            enc = self.j.encode(head, images)
            if self.checked < 2:      # token seam: head + ending must equal the tokens of the full prompt
                for k, full in enumerate(fulls):
                    got = self.j.encode(full, images)["input_ids"][0].tolist()
                    if got != enc["input_ids"][0].tolist() + self.tails[k]:
                        raise AssertionError("token seam mismatch at threshold %d" % k)
            cache = self.j.prefix_cache(enc)
            check = self.checked < 2

            def batched_z():
                c = copy.deepcopy(cache) if check else cache
                c.batch_repeat_interleave(len(self.tails))
                out = self.j.model.model(input_ids=torch.tensor(self.tails, device=self.j.device),
                                         past_key_values=c, use_cache=True)
                return self._margin(out.last_hidden_state[:, -1])

            def looped_z():
                z = []
                for t in self.tails:
                    c = copy.deepcopy(cache)
                    z.append(self._margin(self.j._step(c, t)[None])[0])
                    del c
                return z

            if not check:
                z = batched_z() if self.batched else looped_z()
            else:                     # both cache paths against ordinary full forwards, on the first two clips
                ref = [self.plain(full, images) for full in fulls]
                fmt = lambda xs: " ".join("%.2f" % x for x in xs)
                zl = looped_z()
                dl = max(abs(a - b) for a, b in zip(zl, ref))
                zb, db = None, None
                if self.batched:
                    try:
                        zb = batched_z()
                        db = max(abs(a - b) for a, b in zip(zb, ref))
                    except Exception as e:  # noqa: BLE001
                        self.log("batched endings failed (%r)" % (e,))
                self.log("check %d: full forward %s | per-threshold cache %s (max |dz| %.3f) | batched cache %s (%s)" % (
                    self.checked, fmt(ref), fmt(zl), dl, fmt(zb) if zb else "-", "%.3f" % db if zb else "-"))
                if zb is not None and db <= 0.75:
                    z = zb
                elif dl <= 0.75:
                    if self.batched:
                        self.log("batched cache path not used: one forward per threshold")
                    self.batched = False
                    z = zl
                else:
                    raise SystemExit("cache paths differ from the full forward (%.3f)" % dl)
                self.checked += 1
            del cache
        return z


def cmd_infer(args):
    ds = args.dataset
    dest = out_dir(ds, args.stride)
    L.start_run_log(dest)
    log = Q.log_to(os.path.join(dest, "infer.log"))
    path = os.path.join(dest, "units.jsonl")
    man = L.manifest(ds)
    vids = L.cohort(ds)
    if args.limit:
        vids = vids[:args.limit]
    done = set()
    if os.path.isfile(path):
        with open(path) as fh:
            done = {json.loads(x)["video_id"] for x in fh if x.strip()}
    todo = [v for v in vids if v not in done]
    log("probevad %s stride %g s: %d cohort videos, %d to do; code %s" % (
        ds, args.stride, len(vids), len(todo), L.git_version()))
    if not todo:
        log("DONE infer (nothing to do)")
        return
    segs = L.transcript_segments(ds, {v: man[v]["duration"] for v in vids})
    pr = Prober(log)
    with open(os.path.join(dest, "prompt.txt"), "w") as fh:
        fh.write(pr.render(2, "<words>", 0.5))
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def decode(v):
        try:
            return Q.decode_frames(man[v]["video_path"], SAMPLE_FPS, "probe_%s_%s" % (ds, v)), None
        except Exception as e:  # noqa: BLE001
            return [], repr(e)

    ahead = args.workers * 2
    futs = [pool.submit(decode, v) for v in todo[:ahead]]
    t0, n = time.time(), 0
    for i, v in enumerate(todo):
        if i + ahead < len(todo):
            futs.append(pool.submit(decode, todo[i + ahead]))
        frames, err = futs[i].result()
        us = []
        if not err and frames:
            for k, (c, t1, t2) in enumerate(clips(man[v]["duration"], args.stride)):
                files = Q.uniform(Q.frames_in(frames, t1, t2), MAX_FRAMES)
                text = L.span_text(segs.get(v, []), t1, t2)
                us.append({"i": k, "centre": c, "start": t1, "end": t2, "n_frames": len(files), "text": text,
                           "z": pr.score(files, text)})
            n += len(us)
        Q.drop_frames("probe_%s_%s" % (ds, v))
        with open(path, "a") as fh:
            fh.write(json.dumps({"dataset": ds, "video_id": v, "duration": man[v]["duration"],
                                 "error": err or (None if frames else "no frames decoded"), "units": us}) + "\n")
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            log("%d / %d videos, %d clips, %.0f s" % (i + 1, len(todo), n, time.time() - t0))
    log("DONE infer %s stride %g: %d clips in %.0f s (batched endings: %s)" % (
        ds, args.stride, n, time.time() - t0, pr.batched))


def cmd_raster(args):
    from scipy.ndimage import gaussian_filter1d
    ds = args.dataset
    dest = out_dir(ds, args.stride)
    recs = {}
    with open(os.path.join(dest, "units.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["video_id"]] = r
    stats = {"videos": len(recs), "clips": sum(len(r["units"]) for r in recs.values()), "configs": {}}
    raw = {v: np.asarray([clip_score(u["z"]) for u in r["units"]]) for v, r in recs.items()}
    viol = [float(np.any(np.diff(1 / (1 + np.exp(-np.asarray(u["z"])))) > 1e-9))
            for r in recs.values() for u in r["units"]]
    stats["clips_with_order_violation"] = float(np.mean(viol)) if viol else None
    for sig in args.sigma:
        name = "sig%g" % sig
        method = "probevad_q3vl_s%g_%s" % (args.stride, name)
        rows = {}
        for v, r in recs.items():
            n = L.n_frames(r["duration"])
            us = r["units"]
            if r.get("error") or not us:
                rows[v] = L.row(method, ds, v, r["duration"], [], None, __file__, error=r.get("error") or "no clips")
                continue
            s = raw[v]
            if sig > 0:
                s = gaussian_filter1d(s, sigma=sig / args.stride)     # the release's scipy default (mode reflect)
            curve = L.units_to_4fps([u["centre"] for u in us], None, s, n)
            rows[v] = L.row(method, ds, v, r["duration"], curve, 1.0 / args.stride, __file__)
        med, failed = L.apply_f2(rows, ds)
        stats["configs"][name] = {"sigma_s": sig, "sigma_clips": sig / args.stride, "f2_videos": failed}
        L.finalize(ds, list(rows.values()), os.path.join(dest, name), method)
    with open(os.path.join(dest, "raster_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("infer", "raster"))
    ap.add_argument("--dataset", required=True, choices=L.DATASETS)
    ap.add_argument("--stride", type=float, default=4.0, help="seconds between clip centres")
    ap.add_argument("--sigma", type=float, nargs="+", default=[RELEASE_SIGMA_S],
                    help="raster: Gaussian smoothing sigma in seconds (0 = none); one output directory each")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="ffmpeg decoders")
    args = ap.parse_args()
    {"infer": cmd_infer, "raster": cmd_raster}[args.stage](args)


if __name__ == "__main__":
    main()
