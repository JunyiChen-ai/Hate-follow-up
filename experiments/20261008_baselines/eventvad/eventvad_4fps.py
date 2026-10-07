#!/usr/bin/env python3
"""EventVAD (ACM MM 2025, arXiv 2504.13092) on the 4 fps protocol: HateMM, HateClipSeg, DeHate.

The method is the reconstruction in scripts/reproduction_baselines/eventvad/ (config, graph, boundary, video_io,
prompt, features; see DESIGN_EVENTVAD.md there). Those modules are imported unchanged. This driver adds:

  1. Corpora and cohorts: video paths from the test manifests, the exact cohorts of hate_query.md section 3.
  2. Decode at min(native, 30) fps, the paper's FPS = 30 (run_plan.md L5, decision D5 = paper rate).
  3. Throughput only, same arithmetic: RAFT is run on batches of adjacent frame pairs (RAFT is a per-pair function;
     its encoders use instance norm and eval-mode batch norm, so a batch gives each pair the result it gets alone,
     up to floating-point reduction order) and CLIP preprocessing runs in a thread pool. CLIP is loaded from the
     local checkpoint file, so the vendored loader's download/checksum branch is never entered (hash ban).
  4. Score parsing with every event scored (the legacy run left 41 % of HateMM events unparsed):
       A. the legacy parser (prompt.parse_score) and range rule (prompt.normalise_score), unchanged, after one
          added rule: "X out of N" / "X/N" next to the word score/rate is read as X / N;
       B. if A finds no number, answer extraction: the same model, the same 16 frames and prompt, and its own
          answer followed by "\nTherefore, the final score is" -- the closing sentence of the paper's Figure 2
          example ("Therefore, the final score is 0.8.") and the second stage of zero-shot chain-of-thought
          prompting (Kojima et al., 2022), whose "Let's think step by step" the paper's instruction uses. Greedy,
          at most 8 new tokens; the first number (or "X out of N") is read and range-normalised as in A;
       C. if B also yields no number, the legacy fill 0.0 (run_plan.md F3). The rate of every rule is reported.
  5. 4 fps rasterisation: an event covers decoded frames [s, e), i.e. seconds [s/fps, e/fps); 4 fps frame i takes
     the event containing its centre (i + 0.5) / 4 s, and frames past the last decoded frame take the last event.
  6. Fallback F2 for a video that fails as a whole (decode error): constant curve at the median frame score of the
     scored videos of the corpus. More than 1 % F2 stops the corpus.

Stages: `segment` (GPU: CLIP + RAFT), `score` (GPU: VideoLLaMA2.1-7B-16F), `raster` (CPU; writes predictions.jsonl
and calls src/eval/evaluate_four_datasets.py). `segment` and `score` append one line per video and skip videos
already recorded without an error, so a job killed by the partition's 1-day limit resumes when resubmitted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(BASE))
EV = os.path.join(REPO, "scripts", "reproduction_baselines", "eventvad")
BASELINES = os.path.dirname(EV)
for _p in (BASE, EV, BASELINES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lf_common as L                              # noqa: E402
import boundary as bnd                             # noqa: E402
import config as cfgmod                            # noqa: E402
import graph as gmod                               # noqa: E402
import prompt as pmod                              # noqa: E402
import video_io                                    # noqa: E402

METHOD = "eventvad"
OUT_ROOT = os.path.join(REPO, "runs", "20261008_baselines", "eventvad")
CODE_PATH = "experiments/20261008_baselines/eventvad/eventvad_4fps.py"
EXTRACT_SUFFIX = "\nTherefore, the final score is"
UNPARSED_FILL = 0.0


def first_existing(*paths):
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.exists(p):
            return p
    return os.path.expanduser(paths[0])


DEFAULT_MODEL = first_existing(os.path.join(REPO, ".cache/checkpoints/videollama2"), "~/data/checkpoints/videollama2")
DEFAULT_RAFT = first_existing(os.path.join(REPO, ".cache/checkpoints/raft/raft-things.pth"),
                              "~/data/checkpoints/raft/raft-things.pth")
DEFAULT_CLIP = first_existing(os.path.join(REPO, ".cache/clip/ViT-B-16.pt"), "~/.cache/clip/ViT-B-16.pt")


def out_dir(ds):
    return os.path.join(OUT_ROOT, ds)


def done_ids(path):
    done = set()
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("error"):
                        done.discard(rec["video_id"])
                    else:
                        done.add(rec["video_id"])
    return done


def load_jsonl_ok(path):
    out = {}
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("error"):
                        out.setdefault(rec["video_id"], rec)
                    else:
                        out[rec["video_id"]] = rec
    return out


# ============================================================================ stage 1: segmentation
class BatchedExtractor:
    """features.FeatureExtractor with batched RAFT pairs and threaded CLIP preprocessing (same outputs)."""

    def __init__(self, device, raft_ckpt, clip_path, raft_iters=20, chunk_size=500, raft_batch=8, threads=8):
        import torch
        import features as fmod
        from hate_common.clip import clip as clip_mod
        self.torch = torch
        self.device = torch.device(device)
        self.raft_iters = raft_iters
        self.chunk_size = chunk_size
        self.raft_batch = raft_batch
        model, preprocess = clip_mod.load(clip_path, device=self.device, jit=False)   # file path: no download
        self.clip_model = model.float().eval()
        self.preprocess = preprocess
        if fmod.RAFT_CORE not in sys.path:
            sys.path.insert(0, fmod.RAFT_CORE)
        from argparse import Namespace
        from raft import RAFT
        raft = RAFT(Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0.0))
        state = torch.load(raft_ckpt, map_location="cpu", weights_only=True)
        raft.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()}, strict=True)
        self.raft_model = raft.to(self.device).eval().float()
        self.flow_proj = fmod.init_random_ortho(2, fmod.FLOW_DIM)
        self.pool = ThreadPoolExecutor(threads)

    def _clip_chunk(self, frames):
        from PIL import Image
        torch = self.torch
        tensors = list(self.pool.map(lambda f: self.preprocess(Image.fromarray(f)), frames))
        with torch.no_grad():
            batch = torch.stack(tensors).to(self.device).float()
            return self.clip_model.encode_image(batch).cpu().numpy()

    def _flow(self, prevs, currs):
        torch = self.torch
        from utils.utils import InputPadder
        try:
            with torch.no_grad():
                p = torch.from_numpy(np.stack(prevs)).permute(0, 3, 1, 2).to(self.device).float()
                c = torch.from_numpy(np.stack(currs)).permute(0, 3, 1, 2).to(self.device).float()
                padder = InputPadder(p.shape)
                p, c = padder.pad(p, c)
                flow = self.raft_model(p, c, iters=self.raft_iters)[-1]
                flow = padder.unpad(flow)
                return torch.mean(flow, dim=[2, 3]).cpu().numpy().reshape(len(prevs), 2)
        except torch.cuda.OutOfMemoryError:
            if len(prevs) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(prevs) // 2
            return np.concatenate([self._flow(prevs[:h], currs[:h]), self._flow(prevs[h:], currs[h:])])

    def extract(self, frame_iter):
        clip_out, flow_out = [], []
        state = {"prev": None}

        def flush(buf):
            clip_out.append(self._clip_chunk(buf))
            if state["prev"] is None:
                flow_out.append(np.zeros((1, 2), dtype=np.float32))     # row 0: no frame before the first
                seq = buf
            else:
                seq = [state["prev"]] + buf
            pairs = list(zip(seq[:-1], seq[1:]))
            for i in range(0, len(pairs), self.raft_batch):
                chunk = pairs[i:i + self.raft_batch]
                flow_out.append(self._flow([a for a, _ in chunk], [b for _, b in chunk]))
            state["prev"] = buf[-1]

        buf = []
        for frame in frame_iter:
            buf.append(frame)
            if len(buf) >= self.chunk_size:
                flush(buf)
                buf = []
        if buf:
            flush(buf)
        if not clip_out:
            raise RuntimeError("no frames reached the feature extractor")
        clip = np.concatenate(clip_out).astype(np.float32)
        flow_raw = np.concatenate(flow_out).astype(np.float32)
        if clip.shape[0] != flow_raw.shape[0]:
            raise AssertionError("clip %d rows vs flow %d rows" % (clip.shape[0], flow_raw.shape[0]))
        return clip, (flow_raw @ self.flow_proj).astype(np.float32)


def segment_one(path, extractor, cfg):
    timings = {}
    t0 = time.time()
    pr = video_io.probe(path, cfg)
    t1 = time.time()
    clip_feats, flow_feats = extractor.extract(video_io.iter_frames(pr))
    t2 = time.time()
    n = clip_feats.shape[0]
    adj = gmod.build_dynamic_graph(clip_feats, flow_feats, pr.decode_fps, cfg)
    nodes = gmod.fuse_node_features(clip_feats, flow_feats, cfg)
    propagated = gmod.graph_propagation(nodes, adj, cfg)
    bounds, bdiag = bnd.detect_boundaries(propagated, pr.decode_fps, cfg)
    events, ediag = bnd.events_from_boundaries(bounds, n, pr.decode_fps, cfg)
    t3 = time.time()
    timings.update(probe=t1 - t0, features=t2 - t1, graph_and_cut=t3 - t2)
    return {"probe": pr.as_dict(), "n_frames": int(n), "decode_fps": float(pr.decode_fps), "n_edges": int(adj.nnz),
            "events": [[int(a), int(b)] for a, b in events], "boundaries": [int(b) for b in bounds],
            "boundary": bdiag, "event_diag": ediag, "timings": {k: round(v, 3) for k, v in timings.items()}}


def cmd_segment(args):
    cfg = cfgmod.config_from_args(args)
    dest = out_dir(args.dataset)
    L.start_run_log(dest)
    ev_path = os.path.join(dest, "events.jsonl")
    man = L.manifest(args.dataset)
    ids = L.cohort(args.dataset)
    if args.limit:
        ids = ids[:args.limit]
    done = done_ids(ev_path)
    todo = [v for v in ids if v not in done]
    print("%s segment: %d in cohort, %d done, %d to do" % (args.dataset, len(ids), len(done), len(todo)), flush=True)
    print("config: %s" % json.dumps(cfg.as_dict(), sort_keys=True), flush=True)
    if not todo:
        return 0
    extractor = BatchedExtractor(args.device, args.raft_ckpt, args.clip_ckpt, cfg.raft_iters, cfg.chunk_size,
                                 args.raft_batch, args.threads)
    with open(os.path.join(dest, "segment_config.json"), "w") as fh:
        json.dump({"dataset": args.dataset, "config": cfg.as_dict(), "raft_ckpt": args.raft_ckpt,
                   "clip_ckpt": args.clip_ckpt, "raft_batch": args.raft_batch, "threads": args.threads,
                   "code_version": L.git_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
    started, frames_seen = time.time(), 0
    with open(ev_path, "a") as out:
        for k, vid in enumerate(todo, 1):
            path = man[vid]["video_path"]
            rec = {"video_id": vid, "video_path": path}
            t0 = time.time()
            try:
                rec.update(segment_one(path, extractor, cfg))
                frames_seen += rec["n_frames"]
            except Exception as exc:                                       # noqa: BLE001
                rec["error"] = "%s: %s" % (type(exc).__name__, exc)
            rec["wall_s"] = round(time.time() - t0, 3)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            el = time.time() - started
            print("[%4d/%4d] %-22s %6d fr @%5.2f %4d ev %7.1fs | %5.1f fr/s | eta %6.1f min%s" % (
                k, len(todo), vid, rec.get("n_frames", -1), rec.get("decode_fps", 0), len(rec.get("events", [])),
                rec["wall_s"], frames_seen / max(el, 1e-9), (len(todo) - k) * el / k / 60.0,
                ("  ERROR " + rec["error"][:200]) if rec.get("error") else ""), flush=True)
    print("SEGMENT_DONE %s in %.1f min" % (args.dataset, (time.time() - started) / 60), flush=True)
    return 0


# ============================================================================ stage 2: scoring
_NUM = r"[-+]?\d*\.?\d+"
_RATIO = re.compile(r"(?:score|rate|rating|rated)[^.\n]{0,80}?(%s)\s*(?:out of|/)\s*(%s)" % (_NUM, _NUM), re.I)
_FIRST = re.compile(r"(%s)(\s*(?:out of|/)\s*(%s))?" % (_NUM, _NUM), re.I)


def _ratio(num, den):
    try:
        num, den = float(num), float(den)
    except ValueError:
        return None
    if den <= 0 or num < 0 or num > den:
        return None
    return num / den


def parse_answer(text):
    """Stage A. Returns (score in [0,1] or None, status, raw number, range rule)."""
    hits = _RATIO.findall(text or "")
    if hits:
        value = _ratio(*hits[-1])
        if value is not None:
            return value, "ratio", "%s/%s" % hits[-1], "ratio"
    raw, status, _ = pmod.parse_score(text)
    score, rule = pmod.normalise_score(raw)
    return score, status, raw, rule


def parse_extraction(text):
    """Stage B: the continuation of '... Therefore, the final score is'."""
    m = _FIRST.search(text or "")
    if not m:
        return None, None, "none"
    if m.group(3):
        value = _ratio(m.group(1), m.group(3))
        if value is not None:
            return value, "%s/%s" % (m.group(1), m.group(3)), "ratio"
    try:
        raw = float(m.group(1))
    except ValueError:
        return None, None, "none"
    score, rule = pmod.normalise_score(raw)
    return score, raw, rule


def mm_continue(tensor, instruct, prefix, model, tokenizer, max_new_tokens=8):
    """mm_infer's prompt construction (videollama2/__init__.py), with the assistant turn pre-filled by `prefix`."""
    import torch
    from videollama2.constants import DEFAULT_VIDEO_TOKEN
    from videollama2.mm_utils import tokenizer_multimodal_token
    message = [{"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + instruct}]
    if model.config.model_type in ["videollama2", "videollama2_mistral", "videollama2_mixtral"]:
        raise RuntimeError("system-message model types are not used here")
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + prefix
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, DEFAULT_VIDEO_TOKEN,
                                           return_tensors="pt").unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    with torch.inference_mode():
        out = model.generate(input_ids, attention_mask=attention_masks, images=[(tensor.half().cuda(), "video")],
                             do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens, top_p=0.9,
                             use_cache=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def cmd_score(args):
    import score_events as sev                     # load_model (SigLIP sdpa patch E1), collect_event_frames (E6)
    cfg = cfgmod.config_from_args(args)
    dest = out_dir(args.dataset)
    L.start_run_log(dest)
    ev_path = os.path.join(dest, "events.jsonl")
    sc_path = os.path.join(dest, "event_scores.jsonl")
    events_by_id = {v: r for v, r in load_jsonl_ok(ev_path).items() if not r.get("error")}
    man = L.manifest(args.dataset)
    ids = [v for v in L.cohort(args.dataset) if v in events_by_id]
    if args.limit:
        ids = ids[:args.limit]
    done = done_ids(sc_path)
    todo = [v for v in ids if v not in done]
    instruct = pmod.build_prompt("paper")
    print("%s score: %d videos with events, %d done, %d to do (%d events)" % (
        args.dataset, len(ids), len(done), len(todo), sum(len(events_by_id[v]["events"]) for v in todo)), flush=True)
    print("prompt:\n%s\nextraction suffix: %r" % (instruct, EXTRACT_SUFFIX), flush=True)
    if not todo:
        return 0
    model, processor, tokenizer = sev.load_model(args.model, "sdpa")
    from videollama2 import mm_infer
    import torch
    with open(os.path.join(dest, "score_config.json"), "w") as fh:
        json.dump({"dataset": args.dataset, "arm": "paper", "prompt": instruct, "extraction_suffix": EXTRACT_SUFFIX,
                   "extraction_max_new_tokens": 8, "model": args.model, "attn_implementation": "sdpa",
                   "max_new_tokens": args.max_new_tokens, "do_sample": False,
                   "frames_per_event": cfg.frames_per_event, "unparsed_fill": UNPARSED_FILL,
                   "code_version": L.git_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
    started, n_ev, n_extract = time.time(), 0, 0
    with open(sc_path, "a") as out:
        for k, vid in enumerate(todo, 1):
            meta = events_by_id[vid]
            events = [(int(a), int(b)) for a, b in meta["events"]]
            rec = {"video_id": vid, "n_frames": meta["n_frames"], "decode_fps": meta["decode_fps"],
                   "events": [None] * len(events)}
            t0 = time.time()
            try:
                pr = video_io.probe(man[vid]["video_path"], cfg)
                for idx, frames in sev.collect_event_frames(pr, events, cfg.frames_per_event):
                    tensor = processor["video"](frames)
                    with torch.autocast("cuda", dtype=torch.float16), torch.no_grad():
                        text = mm_infer(tensor, instruct, model=model, tokenizer=tokenizer, do_sample=False,
                                        modal="video", max_new_tokens=args.max_new_tokens)
                    score, status, raw, rule = parse_answer(text)
                    ext = None
                    if score is None:
                        with torch.autocast("cuda", dtype=torch.float16):
                            ext = mm_continue(tensor, instruct, text + EXTRACT_SUFFIX, model, tokenizer)
                        score, raw, rule = parse_extraction(ext)
                        status = "extracted" if score is not None else "unparsed"
                        n_extract += 1
                    rec["events"][idx] = {"start": events[idx][0], "end": events[idx][1], "score": score,
                                          "score_raw": raw, "parse_status": status, "range_rule": rule,
                                          "text": text, "extraction": ext}
                    n_ev += 1
                missing = [i for i, e in enumerate(rec["events"]) if e is None]
                if missing:
                    raise RuntimeError("no frames for events %s" % missing[:5])
            except Exception as exc:                                       # noqa: BLE001
                rec["error"] = "%s: %s" % (type(exc).__name__, exc)
            rec["wall_s"] = round(time.time() - t0, 3)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            el = time.time() - started
            print("[%4d/%4d] %-22s %3d ev %7.1fs | %d ev scored, %d extraction calls | eta %6.1f min%s" % (
                k, len(todo), vid, len(events), rec["wall_s"], n_ev, n_extract, (len(todo) - k) * el / k / 60.0,
                ("  ERROR " + rec["error"][:200]) if rec.get("error") else ""), flush=True)
    print("SCORE_DONE %s in %.1f min" % (args.dataset, (time.time() - started) / 60), flush=True)
    return 0


# ============================================================================ stage 3: 4 fps + evaluation
def cmd_raster(args):
    ds = args.dataset
    dest = out_dir(ds)
    L.start_run_log(dest)
    man = L.manifest(ds)
    scores = load_jsonl_ok(os.path.join(dest, "event_scores.jsonl"))
    segs = load_jsonl_ok(os.path.join(dest, "events.jsonl"))
    rows, stats = {}, {"parse_status": {}, "range_rule": {}, "n_events": 0, "n_unparsed_filled": 0,
                       "frames_unparsed_filled": 0, "frames_total": 0}
    for vid in L.cohort(ds):
        dur = man[vid]["duration"]
        n = L.n_frames(dur)
        rec = scores.get(vid)
        err = None
        if rec is None:
            err = (segs.get(vid) or {}).get("error") or "no event_scores row"
        elif rec.get("error"):
            err = rec["error"]
        if err:
            rows[vid] = L.row(METHOD, ds, vid, dur, [], "events", CODE_PATH, error=err)
            continue
        fps = float(rec["decode_fps"])
        starts = [e["start"] / fps for e in rec["events"]]
        ends = [e["end"] / fps for e in rec["events"]]
        vals = [UNPARSED_FILL if e["score"] is None else float(e["score"]) for e in rec["events"]]
        filled = np.array([e["score"] is None for e in rec["events"]], dtype=float)
        curve = L.units_to_4fps(starts, ends, vals, n)
        fill_curve = L.units_to_4fps(starts, ends, filled, n)
        for e in rec["events"]:
            stats["parse_status"][e["parse_status"]] = stats["parse_status"].get(e["parse_status"], 0) + 1
            stats["range_rule"][e["range_rule"]] = stats["range_rule"].get(e["range_rule"], 0) + 1
        stats["n_events"] += len(rec["events"])
        stats["n_unparsed_filled"] += int(filled.sum())
        stats["frames_unparsed_filled"] += int(fill_curve.sum())
        stats["frames_total"] += n
        rows[vid] = L.row(METHOD, ds, vid, dur, curve, "events", CODE_PATH,
                          extra={"n_events": len(rec["events"]), "decode_fps": fps,
                                 "n_unparsed_filled": int(filled.sum()),
                                 "n_extracted": sum(e["parse_status"] == "extracted" for e in rec["events"])},
                          calls=len(rec["events"]) + sum(e.get("extraction") is not None for e in rec["events"]))
    med, failed = L.apply_f2(rows, ds)
    stats.update(n_videos=len(rows), n_f2=len(failed), f2=failed, f2_median=med)
    with open(os.path.join(dest, "raster_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    with open(os.path.join(dest, "run.log"), "a") as fh:
        fh.write("raster: %s\n" % json.dumps({k: v for k, v in stats.items() if k != "f2"}))
        for vid, why in failed:
            fh.write("F2 %s: %s\n" % (vid, why))
    print(json.dumps({k: v for k, v in stats.items() if k != "f2"}, indent=1))
    if len(failed) > 0.01 * len(rows):
        print("STOP: %d F2 videos > 1%% of %d" % (len(failed), len(rows)))
        return 1
    with open(os.path.join(dest, "config.json"), "w") as fh:
        json.dump({"method": METHOD, "dataset": ds, "code": CODE_PATH, "code_version": L.git_version(),
                   "cohort": "runs/20261008_baselines/cohort/%s.txt" % ds,
                   "segment_config": "segment_config.json", "score_config": "score_config.json",
                   "native_unit": "EventVAD events on the min(native, 30) fps decoded stream",
                   "to_4fps": "frame i takes the event containing (i+0.5)/4 s; past the last event: last event",
                   "parse_rules": "A legacy parser + ratio; B answer extraction; C fill %.1f" % UNPARSED_FILL,
                   "fallback_F2": "median frame score of scored videos", "date": time.strftime("%Y-%m-%d")},
                  fh, indent=2)
    L.finalize(ds, list(rows.values()), dest, METHOD)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("segment", "score", "raster", "selftest"))
    ap.add_argument("--dataset", choices=L.DATASETS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--raft-ckpt", default=DEFAULT_RAFT)
    ap.add_argument("--clip-ckpt", default=DEFAULT_CLIP)
    ap.add_argument("--raft-batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    cfgmod.add_config_args(ap)
    args = ap.parse_args(argv)
    if args.stage == "selftest":
        return selftest()
    if not args.dataset:
        ap.error("--dataset is required")
    return {"segment": cmd_segment, "score": cmd_score, "raster": cmd_raster}[args.stage](args)


def selftest():
    ok = True
    cases = [("Therefore, the final score is 0.8.", 0.8, "sentence"),
             ("I would assign an anomaly score of 2 out of 5, indicating", 0.4, "ratio"),
             ("Overall, the anomaly score for this video is low.", None, "unparsed"),
             ("Therefore, the anomaly score for this video is 0.", 0.0, "trailing_number"),
             ("Anomaly score: 0.5", 0.5, "sentence")]
    for text, want, st in cases:
        got, status, _, _ = parse_answer(text)
        good = (got == want or (got is not None and want is not None and abs(got - want) < 1e-9)) and status == st
        ok &= good
        print("%s parse %r -> %r %s" % ("PASS" if good else "FAIL", text[:50], got, status))
    for text, want in [(" 0.2.", 0.2), ("0", 0.0), (" 3 out of 10.", 0.3), (": 7", 0.7), ("low.", None), (" low, about 0.1.", 0.1),
                       (" 1/5", 0.2)]:
        got, _, _ = parse_extraction(text)
        good = (got is None and want is None) or (got is not None and want is not None and abs(got - want) < 1e-9)
        ok &= good
        print("%s extract %r -> %r" % ("PASS" if good else "FAIL", text, got))
    curve = L.units_to_4fps([0.0, 1.5], [1.5, 2.0], [0.1, 0.9], 10)
    good = np.allclose(curve, [0.1] * 6 + [0.9] * 4)
    ok &= good
    print("%s raster midpoint and tail hold %s" % ("PASS" if good else "FAIL", curve))
    print("all passed" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
