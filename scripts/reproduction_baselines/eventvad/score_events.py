#!/usr/bin/env python
"""EventVAD stage 2: score each event with VideoLLaMA2.1-7B-16F. Needs a GPU.

Replaces upstream `src/score/event_score.py`, whose prompt is the placeholder
string `"prompt"`. See `prompt.py` for the reconstruction and its sources.

Two departures from upstream, both forced.

**Attention (patch E1).** `videollama2/model/encoder.py` sets
`config._attn_implementation = 'flash_attention_2'` unconditionally on the
SigLIP vision tower, regardless of `load_pretrained_model`'s `use_flash_attn`
flag, which defaults to False and governs only the language model. flash-attn
2.5.8 does not build against torch 2.8 / CUDA 12.8 on Blackwell in any
reasonable time, and it is not needed: SigLIP's encoder is a plain bidirectional
transformer over 729 patch tokens and `sdpa` computes the same attention. The
tower's `__init__` is replaced at import time with a copy that differs in that
one string.

**Frames (patch E6).** Upstream hands `processor['video']` a path to the
re-encoded segment file, and `process_video` opens it with decord. Stage 1
writes boundaries rather than segment files, so the 16 frames are pulled from
the source video instead. The index rule is VideoLLaMA2's own
`mm_utils.frame_sample(duration, mode='uniform', num_frames=16)` applied to the
event's frame range, so the model sees the frames upstream's sampler would have
picked out of that segment, and `process_video` accepts the resulting
`np.ndarray` on a documented branch of its own dispatch. Pulling from the
source also avoids the mp4v re-encode, which cannot round-trip the AV1 inputs.

Output, one JSON object per line, in
`results/reproduction/baselines/eventvad/<corpus>/event_scores[_<arm>].jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASELINES))
for _p in (HERE, BASELINES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hate_common import data as hdata            # noqa: E402
import config as cfgmod                          # noqa: E402
import prompt as pmod                            # noqa: E402
import segment_events as seg                     # noqa: E402
import video_io                                  # noqa: E402

VIDEOLLAMA2_CLONE = os.path.join(PROJECT_ROOT, "third_party", "VideoLLaMA2")
DEFAULT_MODEL = "/home/jehc223/data/checkpoints/videollama2"
NUM_FRAMES = 16


# ------------------------------------------------------------- patch E1
def patch_siglip_attention(impl="sdpa"):
    """Replace SiglipVisionTower.__init__ so the tower does not demand
    flash-attn. Body is upstream's, `encoder.py` lines 86-100, with
    `'flash_attention_2'` replaced by `impl`."""
    from transformers import SiglipImageProcessor, SiglipVisionConfig
    from transformers import SiglipVisionModel
    import videollama2.model.encoder as enc

    def __init__(self, vision_tower, args, load_pretrained=False):
        import torch.nn as nn
        nn.Module.__init__(self)
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature",
                                      "patch")
        self.image_processor = SiglipImageProcessor.from_pretrained(
            self.vision_tower_name)
        config = SiglipVisionConfig.from_pretrained(self.vision_tower_name)
        config._attn_implementation = impl          # PORT PATCH (patch E1)
        if not load_pretrained:
            self.vision_tower = SiglipVisionModel(config=config)
        else:
            self.vision_tower = SiglipVisionModel.from_pretrained(
                self.vision_tower_name)

    enc.SiglipVisionTower.__init__ = __init__
    return enc.SiglipVisionTower


def load_model(model_path=DEFAULT_MODEL, attn_impl="sdpa", device="cuda:0"):
    if VIDEOLLAMA2_CLONE not in sys.path:
        sys.path.insert(0, VIDEOLLAMA2_CLONE)
    patch_siglip_attention(attn_impl)
    from videollama2 import model_init
    from videollama2.utils import disable_torch_init
    disable_torch_init()                                    # upstream
    model, processor, tokenizer = model_init(model_path, device_map=device)
    return model.half().eval(), processor, tokenizer


# ------------------------------------------------------------- frame plan
def frame_sample_uniform(duration, num_frames=NUM_FRAMES):
    """VideoLLaMA2 `mm_utils.frame_sample(mode='uniform')`, verbatim."""
    seg_size = float(duration - 1) / num_frames
    ids = []
    for i in range(num_frames):
        ids.append((seg_size * i + seg_size * (i + 1)) / 2)
    return np.round(np.array(ids) + 1e-6).astype(int)


def event_frame_indices(start, end, num_frames=NUM_FRAMES):
    """Absolute frame indices for one `[start, end)` event."""
    duration = max(1, end - start)
    rel = frame_sample_uniform(duration, num_frames)
    rel = np.clip(rel, 0, duration - 1)
    return (start + rel).astype(int)


def collect_event_frames(pr, events, num_frames=NUM_FRAMES):
    """One decode pass over the video; yields (event_index, frames array).

    Events partition the frame range in order, so at most a bounded number of
    events are open at once and nothing needs the whole video resident.
    """
    plans = [event_frame_indices(s, e, num_frames) for s, e in events]
    pending = {}
    for idx, plan in enumerate(plans):
        for want in set(int(i) for i in plan):
            pending.setdefault(want, []).append(idx)

    have = {i: {} for i in range(len(plans))}
    remaining = {i: len(set(int(x) for x in plans[i]))
                 for i in range(len(plans))}
    highest = None
    for pos, frame in enumerate(video_io.iter_frames(pr)):
        highest = frame
        owners = pending.pop(pos, None)
        if owners is None:
            continue
        for idx in owners:
            have[idx][pos] = np.ascontiguousarray(frame)
            remaining[idx] -= 1
            if remaining[idx] == 0:
                yield idx, np.stack([have[idx][int(i)] for i in plans[idx]])
                have[idx] = None
        if not pending:
            break
    # Anything still pending asked for a frame past the end of the decoded
    # stream, which happens when ffprobe's frame count over-reports. Hold the
    # last frame, as `video_io.read_frame_range` does.
    for idx, need in list(remaining.items()):
        if need > 0 and have[idx] is not None:
            if highest is None:
                raise RuntimeError("decoded no frames from %s" % pr.path)
            frames = [have[idx].get(int(i), highest) for i in plans[idx]]
            yield idx, np.stack(frames)
            have[idx] = None


# -------------------------------------------------------------------- run
def out_paths(corpus, arm, root=None):
    dest = seg.out_dir(corpus, root)
    suffix = "" if arm == pmod.DEFAULT_ARM else "_" + arm
    return dest, os.path.join(dest, "events.jsonl"), \
        os.path.join(dest, "event_scores%s.jsonl" % suffix)


def load_events(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "%s not found -- run segment_events.py first" % path)
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            out[rec["video_id"]] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=list(hdata.CORPORA))
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--arm", default=pmod.DEFAULT_ARM,
                    choices=sorted(pmod.ARMS))
    ap.add_argument("--attn", default="sdpa",
                    choices=("sdpa", "eager", "flash_attention_2"))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    cfgmod.add_config_args(ap)
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.corpus:
        ap.error("--corpus is required unless --selftest")

    cfg = cfgmod.config_from_args(args)
    dest, ev_path, sc_path = out_paths(args.corpus, args.arm, args.out_root)
    os.makedirs(dest, exist_ok=True)
    if args.restart and os.path.isfile(sc_path):
        os.remove(sc_path)

    events_by_id = load_events(ev_path)
    ids, _ = seg.cohort(args.corpus, args.split, args.limit)
    ids = [v for v in ids if v in events_by_id]
    done = seg.already_done(sc_path)
    todo = [v for v in ids if v not in done]

    instruct = pmod.build_prompt(args.arm)
    n_events = sum(len(events_by_id[v]["events"]) for v in todo)
    print("%s arm=%s: %d videos with events, %d done, %d to do (%d events)"
          % (args.corpus, args.arm, len(ids), len(done), len(todo), n_events))
    print("prompt:\n%s\n" % instruct)
    if not todo:
        print("nothing to do")
        return 0

    model, processor, tokenizer = load_model(args.model, args.attn)
    from videollama2 import mm_infer
    import torch

    with open(os.path.join(dest, "score_meta.json"), "w") as fh:
        json.dump({"corpus": args.corpus, "arm": args.arm,
                   "prompt": instruct, "model": args.model,
                   "attn_implementation": args.attn,
                   "max_new_tokens": args.max_new_tokens,
                   "frames_per_event": cfg.frames_per_event,
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime())},
                  fh, indent=2)

    started = time.time()
    scored = 0
    with open(sc_path, "a", encoding="utf-8") as out:
        for k, vid in enumerate(todo, 1):
            meta = events_by_id[vid]
            events = [(int(a), int(b)) for a, b in meta["events"]]
            rec = {"video_id": vid, "arm": args.arm,
                   "n_frames": meta["n_frames"],
                   "decode_fps": meta["decode_fps"],
                   "events": [None] * len(events)}
            t0 = time.time()
            try:
                path = seg.video_path(args.corpus, vid)
                pr = video_io.probe(path, cfg)
                for idx, frames in collect_event_frames(
                        pr, events, cfg.frames_per_event):
                    tensor = processor["video"](frames)
                    with torch.autocast("cuda", dtype=torch.float16), \
                            torch.no_grad():
                        text = mm_infer(tensor, instruct, model=model,
                                        tokenizer=tokenizer, do_sample=False,
                                        modal="video",
                                        max_new_tokens=args.max_new_tokens)
                    raw, status, _ = pmod.parse_score(text)
                    score, rule = pmod.normalise_score(raw)
                    rec["events"][idx] = {
                        "start": events[idx][0], "end": events[idx][1],
                        "score_raw": raw, "score": score,
                        "parse_status": status, "range_rule": rule,
                        "text": text,
                    }
                    scored += 1
                missing = [i for i, e in enumerate(rec["events"]) if e is None]
                if missing:
                    raise RuntimeError("no frames for events %s" % missing[:5])
            except Exception as exc:                     # noqa: BLE001
                rec["error"] = "%s: %s" % (type(exc).__name__, exc)
            rec["wall_s"] = round(time.time() - t0, 3)
            out.write(json.dumps(rec) + "\n")
            out.flush()

            elapsed = time.time() - started
            eta = (len(todo) - k) * (elapsed / k) / 60.0
            print("[%4d/%4d] %-24s %3d events %7.1fs | %d scored | eta %.1f min"
                  % (k, len(todo), vid, len(events), rec["wall_s"], scored,
                     eta), flush=True)

    print("done in %.1f min, %d events scored"
          % ((time.time() - started) / 60.0, scored))
    return 0


# ------------------------------------------------------------------- tests
def _check(name, ok, detail=""):
    print("  %s %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return bool(ok)


def selftest():
    print("score_events selftest")
    ok = True

    # frame_sample_uniform against VideoLLaMA2's own implementation.
    if os.path.isdir(VIDEOLLAMA2_CLONE):
        sys.path.insert(0, VIDEOLLAMA2_CLONE)
        try:
            from videollama2.mm_utils import frame_sample as upstream_sample
            same = all(
                np.array_equal(frame_sample_uniform(d),
                               upstream_sample(d, mode="uniform",
                                               num_frames=NUM_FRAMES))
                for d in (1, 2, 5, 16, 17, 100, 3001))
            ok &= _check("frame_sample matches VideoLLaMA2's", same)
        except Exception as exc:                         # noqa: BLE001
            ok &= _check("frame_sample matches VideoLLaMA2's", False, str(exc))

    idxs = event_frame_indices(100, 116)
    ok &= _check("16 indices inside a 16-frame event",
                 len(idxs) == 16 and idxs.min() >= 100 and idxs.max() < 116,
                 str(idxs))
    idxs = event_frame_indices(50, 53)
    ok &= _check("short event repeats rather than overruns",
                 len(idxs) == 16 and idxs.min() >= 50 and idxs.max() < 53,
                 str(idxs))
    idxs = event_frame_indices(0, 3000)
    ok &= _check("long event spreads across the range",
                 len(idxs) == 16 and idxs.min() >= 0 and idxs.max() < 3000,
                 "%d..%d" % (idxs.min(), idxs.max()))

    for arm in sorted(pmod.ARMS):
        text = pmod.build_prompt(arm)
        ok &= _check("arm %s mentions a score" % arm, "score" in text.lower())
    ok &= _check("paper arm quotes Figure 2 verbatim",
                 pmod.PAPER_QUESTION in pmod.ARMS["paper"]
                 and pmod.PAPER_INSTRUCTION in pmod.ARMS["paper"])

    cases = [
        ("0.8", 0.8, "bare_float"),
        ("A man is seen with a gun ... Therefore, the final score is 0.8.",
         0.8, "sentence"),
        ("The anomaly score is 0.35", 0.35, "sentence"),
        ("Final score: 0.9", 0.9, "sentence"),
        ("Nothing unusual happens here. 0.1", 0.1, "trailing_number"),
        ("I cannot tell.", None, "unparsed"),
    ]
    for text, want, want_status in cases:
        got, status, _ = pmod.parse_score(text)
        ok &= _check("parse %r" % text[:40],
                     got == want and status == want_status,
                     "got %r/%s" % (got, status))
    for value, want, rule in [(0.8, 0.8, "in_range"), (8.0, 0.8, "div10"),
                              (80.0, 0.8, "div100"), (-1.0, 0.0, "clamped_low"),
                              (500.0, 1.0, "clamped_high")]:
        got, got_rule = pmod.normalise_score(value)
        ok &= _check("normalise %s" % value,
                     abs(got - want) < 1e-9 and got_rule == rule,
                     "%r/%s" % (got, got_rule))

    print("  %s" % ("all passed" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
