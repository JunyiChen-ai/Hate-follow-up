#!/usr/bin/env python3
"""EventVAD (ACM MM 2025, arXiv 2504.13092) on the 4 fps protocol: HateMM, HateClipSeg, DeHate.

The method is the reconstruction in scripts/reproduction_baselines/eventvad/ (config, graph, boundary, video_io,
prompt, features; see DESIGN_EVENTVAD.md there). Those modules are imported unchanged. This driver adds:

  1. Corpora and cohorts: video paths from the test manifests, the exact cohorts of hate_query.md section 3.
  2. Decode at min(native, 30) fps, the paper's FPS = 30 (run_plan.md L5, decision D5 = paper rate).
     Graph constants: the released code's (config `--preset upstream`: CLIP weight alpha = 0.8, time decay
     gamma = 0.05 per frame index, upstream moving-average arithmetic, raw CLIP magnitudes in the node features);
     the paper states alpha = 0.75 and gamma = 0.6. CLIP and RAFT features are cached per video
     (<dataset>/features/<id>.npz), so the graph stage can be rerun without RAFT.
     Prompt: the release has the placeholder string "prompt"; the reconstruction of prompt.py (Figure 2) is used,
     with the nine hate rules of hate_query.md in its anomaly slot (`build_prompt`). Label: prompt reconstructed.
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

Transcript (audio-visual) variant, `--transcript` on `score`, `raster` and `prompts` (default off; 2026-10-09): the
scoring prompt of each event starts with the line "Speech during this segment: <text>", where <text> is the Whisper
large-v3 transcript of the event's time span [s/fps, e/fps) (data/asr_whisper_large_v3/<DS>/timestamped_chunks.jsonl;
untimed chunks kept as in qwen3_text/text_llm.py; span text by src/video_inputs.py `window_text`; whitespace
collapsed; "(no speech)" when empty; the longest whole-word prefix of at most 512 tokens of the model's tokenizer).
Everything else is the visual-only run's: the events of runs/20261008_baselines/eventvad/<DS>/events.jsonl (not
recomputed), the 16 frames, the nine rules in the anomaly slot, greedy decoding, rules A/B/C. Outputs go to
runs/20261008_baselines/eventvad_av/<DS>/; method name "eventvad_av".

Stages: `segment` (GPU: CLIP + RAFT), `score` (GPU: VideoLLaMA2.1-7B-16F), `raster` (CPU; writes predictions.jsonl
and calls src/eval/evaluate_four_datasets.py), `prompts` (CPU; prints the transcript-variant prompts of the first
--limit videos, writes nothing). `segment` and `score` append one line per video and skip videos
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
if REPO not in sys.path:
    sys.path.append(REPO)                          # for src.video_inputs (appended: shadows nothing)

import lf_common as L                              # noqa: E402
import boundary as bnd                             # noqa: E402
import config as cfgmod                            # noqa: E402
import graph as gmod                               # noqa: E402
import prompt as pmod                              # noqa: E402
import video_io                                    # noqa: E402

METHOD = "eventvad"
METHOD_AV = "eventvad_av"
OUT_ROOT = os.path.join(REPO, "runs", "20261008_baselines", "eventvad")
OUT_ROOT_AV = os.path.join(REPO, "runs", "20261008_baselines", "eventvad_av")
CODE_PATH = "experiments/20261008_baselines/eventvad/eventvad_4fps.py"
EXTRACT_SUFFIX = "\nTherefore, the final score is"
UNPARSED_FILL = 0.0
ASR_DIR = os.path.join(REPO, "data", "asr_whisper_large_v3")
SPEECH_PREFIX = "Speech during this segment: "
NO_SPEECH = "(no speech)"                 # run_plan.md 1.3 F1, the empty-input text of the LLM prompts
SPEECH_CAP_TOKENS = 512                   # fixed before any run (README); about 0.1 % of events exceed it


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


def out_dir(ds, transcript=False):
    return os.path.join(OUT_ROOT_AV if transcript else OUT_ROOT, ds)


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


def segment_one(path, extractor, cfg, feat_path):
    """Decode + CLIP/RAFT features (cached to feat_path, so the graph stage can be rerun without RAFT), then the
    dynamic graph, propagation and boundary detection."""
    timings = {}
    t0 = time.time()
    pr = video_io.probe(path, cfg)
    t1 = time.time()
    if feat_path and os.path.isfile(feat_path):
        z = np.load(feat_path)
        clip_feats, flow_feats = z["clip"], z["flow"]
        if abs(float(z["decode_fps"]) - pr.decode_fps) > 1e-6:
            raise RuntimeError("cached features were decoded at %s fps, probe says %s" % (z["decode_fps"],
                                                                                          pr.decode_fps))
    else:
        clip_feats, flow_feats = extractor().extract(video_io.iter_frames(pr))
        if feat_path:
            os.makedirs(os.path.dirname(feat_path), exist_ok=True)
            tmp = feat_path + ".tmp.npz"
            np.savez(tmp, clip=clip_feats, flow=flow_feats, decode_fps=pr.decode_fps)
            os.replace(tmp, feat_path)
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
    holder = {}

    def extractor():          # built on first use, so a rerun on cached features never loads CLIP/RAFT
        if "x" not in holder:
            holder["x"] = BatchedExtractor(args.device, args.raft_ckpt, args.clip_ckpt, cfg.raft_iters,
                                           cfg.chunk_size, args.raft_batch, args.threads)
        return holder["x"]
    with open(os.path.join(dest, "segment_config.json"), "w") as fh:
        json.dump({"dataset": args.dataset, "preset": args.preset, "config": cfg.as_dict(),
                   "paper_states": {"alpha": 0.75, "gamma": 0.6, "note": "paper values; the released config.py has "
                                    "clip_weight 0.8 and time_decay 0.05 (per frame), which are used here"},
                   "raft_ckpt": args.raft_ckpt,
                   "clip_ckpt": args.clip_ckpt, "raft_batch": args.raft_batch, "threads": args.threads,
                   "code_version": L.git_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
    started, frames_seen = time.time(), 0
    with open(ev_path, "a") as out:
        for k, vid in enumerate(todo, 1):
            path = man[vid]["video_path"]
            rec = {"video_id": vid, "video_path": path}
            t0 = time.time()
            try:
                rec.update(segment_one(path, extractor, cfg, os.path.join(dest, "features", vid + ".npz")))
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


# hate_query.md section 1, verbatim (YOUTUBE_RULES of experiments/20260910_spvl/spvl.py)
HATE_RULES = (
    "1. Encourage violence against groups based on protected status",
    "2. Incite hatred against groups based on protected status",
    "3. Dehumanize groups by comparing to non-human entities",
    "4. Praise or glorify violence against protected groups",
    "5. Use slurs and stereotypes that incite hatred (including through humor, irony, or coded language)",
    "6. Claim groups are inferior based on protected status",
    "7. Promote hateful supremacism",
    "8. Make conspiratorial claims about groups based on protected status",
    "9. Deny or minimize major violent events",
)


def build_prompt(speech=None):
    """The reconstructed Figure 2 prompt (prompt.py, `paper` arm) with the hate rules in its anomaly slot: the
    #Question line is followed by the definition of an anomaly as content violating any of the rules; the
    #Question and #Instruction sentences are unchanged. Transcript variant: `speech` (the event's span text, or
    NO_SPEECH) goes on one line before the question; with speech=None the prompt is the visual-only one."""
    question = (pmod.PAPER_QUESTION + " An anomaly is content that violates any of the following rules:\n"
                + "\n".join(HATE_RULES))
    body = "#Question: %s\n#Instruction: %s" % (question, pmod.PAPER_INSTRUCTION)
    if speech is None:
        return body
    return SPEECH_PREFIX + speech + "\n" + body


def load_speech_segments(ds, man):
    """video id -> [(start, end, text)] Whisper large-v3 segments of the cohort videos. Untimed chunks are kept as in
    qwen3_text/text_llm.py `load_segments` (run_plan.md 1.3): a missing start is the previous segment's end, a
    missing end the next chunk's start (if later) or the video duration. Videos without a row have no segment."""
    out = {}
    with open(os.path.join(ASR_DIR, ds, "timestamped_chunks.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            v = r["video_id"]
            if v not in man:
                continue
            ch = r.get("chunks") or []
            segs, prev_end = [], 0.0
            for i, c in enumerate(ch):
                st = c.get("start")
                en = c.get("end")
                st = float(st) if st is not None else prev_end
                if en is None:
                    nxt = next((float(x["start"]) for x in ch[i + 1:] if x.get("start") is not None), None)
                    en = nxt if nxt is not None and nxt > st else man[v]["duration"]
                en = float(en)
                segs.append((st, en, c.get("text") or ""))
                prev_end = max(prev_end, en)
            out[v] = segs
    return out


def speech_text(segments, t1, t2, n_tokens, cap=SPEECH_CAP_TOKENS):
    """Transcript of [t1, t2] (src/video_inputs.py window_text: proportional word slicing of the Whisper segments),
    whitespace collapsed; NO_SPEECH if empty; the longest whole-word prefix with at most `cap` tokens
    (n_tokens(text) counts the model's tokens). Returns (text, tokens, truncated)."""
    import src.video_inputs as vi
    if not os.path.abspath(vi.__file__).startswith(REPO + os.sep):
        raise RuntimeError("src.video_inputs resolved to %s, not this repo" % vi.__file__)
    text = " ".join(vi.window_text(segments, t1, t2).split())
    if not text:
        return NO_SPEECH, 0, False
    n = n_tokens(text)
    if n <= cap:
        return text, n, False
    words = text.split()
    lo, hi = 0, len(words)                      # largest k with n_tokens(words[:k]) <= cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_tokens(" ".join(words[:mid])) <= cap:
            lo = mid
        else:
            hi = mid - 1
    text = " ".join(words[:lo])
    if not text:
        return NO_SPEECH, 0, True
    return text, n_tokens(text), True


def cmd_score(args):
    import score_events as sev                     # load_model (SigLIP sdpa patch E1), collect_event_frames (E6)
    cfg = cfgmod.config_from_args(args)
    av = args.transcript
    dest = out_dir(args.dataset, av)
    L.start_run_log(dest)
    ev_path = os.path.join(out_dir(args.dataset), "events.jsonl")     # the variant reuses the visual-only events
    sc_path = os.path.join(dest, "event_scores.jsonl")
    seg_rows = load_jsonl_ok(ev_path)
    if av and not args.limit:
        missing = [v for v in L.cohort(args.dataset) if v not in seg_rows]
        if missing:
            print("FAILED: the visual-only segmentation of %s lacks %d cohort videos (e.g. %s); finish it first" % (
                args.dataset, len(missing), missing[:3]), flush=True)
            return 1
    events_by_id = {v: r for v, r in seg_rows.items() if not r.get("error")}
    man = L.manifest(args.dataset)
    ids = [v for v in L.cohort(args.dataset) if v in events_by_id]
    if args.limit:
        ids = ids[:args.limit]
    done = done_ids(sc_path)
    todo = [v for v in ids if v not in done]
    instruct = build_prompt(speech="<text>" if av else None)
    speech = load_speech_segments(args.dataset, man) if av else None
    print("%s score%s: %d videos with events, %d done, %d to do (%d events)" % (
        args.dataset, " (transcript variant)" if av else "", len(ids), len(done), len(todo),
        sum(len(events_by_id[v]["events"]) for v in todo)), flush=True)
    print("prompt:\n%s\nextraction suffix: %r" % (instruct, EXTRACT_SUFFIX), flush=True)
    if av:
        print("transcripts: %d of %d cohort videos have Whisper segments; cap %d tokens" % (
            sum(1 for v in ids if speech.get(v)), len(ids), args.speech_cap_tokens), flush=True)
    if not todo:
        return 0
    model, processor, tokenizer = sev.load_model(args.model, "sdpa")
    from videollama2 import mm_infer
    import torch

    def n_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    conf = {"dataset": args.dataset, "arm": "paper_reconstructed_with_hate_rules",
            "prompt_note": "prompt reconstructed (release has the placeholder string 'prompt')",
            "prompt": instruct, "extraction_suffix": EXTRACT_SUFFIX,
            "extraction_max_new_tokens": 8, "model": args.model, "attn_implementation": "sdpa",
            "max_new_tokens": args.max_new_tokens, "do_sample": False,
            "frames_per_event": cfg.frames_per_event, "unparsed_fill": UNPARSED_FILL,
            "code_version": L.git_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if av:
        conf.update(arm="paper_reconstructed_with_hate_rules_plus_transcript", events=ev_path,
                    transcript=os.path.join(ASR_DIR, args.dataset, "timestamped_chunks.jsonl"),
                    transcript_rule="Whisper segments (untimed chunks kept as qwen3_text/text_llm.py); span "
                                    "[s/fps, e/fps) text by src/video_inputs.py window_text; whitespace collapsed; "
                                    "'%s' if empty; longest whole-word prefix of <= %d tokens of the model "
                                    "tokenizer" % (NO_SPEECH, args.speech_cap_tokens),
                    speech_cap_tokens=args.speech_cap_tokens)
    with open(os.path.join(dest, "score_config.json"), "w") as fh:
        json.dump(conf, fh, indent=2)
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
                spans = [None] * len(events)
                if av:
                    fps = float(meta["decode_fps"])
                    spans = [speech_text(speech.get(vid, []), a / fps, b / fps, n_tokens, args.speech_cap_tokens)
                             for a, b in events]
                for idx, frames in sev.collect_event_frames(pr, events, cfg.frames_per_event):
                    tensor = processor["video"](frames)
                    if av:
                        instruct = build_prompt(speech=spans[idx][0])
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
                    if av:
                        rec["events"][idx].update(speech=spans[idx][0], speech_tokens=spans[idx][1],
                                                  speech_truncated=spans[idx][2])
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
    av = args.transcript
    method = METHOD_AV if av else METHOD
    dest = out_dir(ds, av)
    L.start_run_log(dest)
    man = L.manifest(ds)
    scores = load_jsonl_ok(os.path.join(dest, "event_scores.jsonl"))
    segs = load_jsonl_ok(os.path.join(out_dir(ds), "events.jsonl"))
    rows, stats = {}, {"parse_status": {}, "range_rule": {}, "n_events": 0, "n_unparsed_filled": 0,
                       "frames_unparsed_filled": 0, "frames_total": 0}
    if av:
        stats.update(n_events_no_speech=0, n_events_speech_truncated=0, speech_tokens_max=0)
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
            rows[vid] = L.row(method, ds, vid, dur, [], "events", CODE_PATH, error=err)
            continue
        if av and [[e["start"], e["end"]] for e in rec["events"]] != (segs.get(vid) or {}).get("events"):
            raise RuntimeError("%s: scored events differ from the visual-only events.jsonl" % vid)
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
            if av:
                stats["n_events_no_speech"] += int(e["speech"] == NO_SPEECH)
                stats["n_events_speech_truncated"] += int(bool(e["speech_truncated"]))
                stats["speech_tokens_max"] = max(stats["speech_tokens_max"], int(e["speech_tokens"]))
        stats["n_events"] += len(rec["events"])
        stats["n_unparsed_filled"] += int(filled.sum())
        stats["frames_unparsed_filled"] += int(fill_curve.sum())
        stats["frames_total"] += n
        rows[vid] = L.row(method, ds, vid, dur, curve, "events", CODE_PATH,
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
        json.dump({"method": method, "dataset": ds, "code": CODE_PATH, "code_version": L.git_version(),
                   "events": os.path.relpath(os.path.join(out_dir(ds), "events.jsonl"), REPO),
                   "transcript_variant": bool(av),
                   "cohort": "runs/20261008_baselines/cohort/%s.txt" % ds,
                   "segment_config": "segment_config.json", "score_config": "score_config.json",
                   "native_unit": "EventVAD events on the min(native, 30) fps decoded stream",
                   "to_4fps": "frame i takes the event containing (i+0.5)/4 s; past the last event: last event",
                   "parse_rules": "A legacy parser + ratio; B answer extraction; C fill %.1f" % UNPARSED_FILL,
                   "fallback_F2": "median frame score of scored videos", "date": time.strftime("%Y-%m-%d")},
                  fh, indent=2)
    L.finalize(ds, list(rows.values()), dest, method)
    return 0


def cmd_prompts(args):
    """CPU check of the transcript variant: prints the prompts of the first --limit videos' events (tokenizer only,
    no model, no output file)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def n_tokens(text):
        return len(tok.encode(text, add_special_tokens=False))
    man = L.manifest(args.dataset)
    speech = load_speech_segments(args.dataset, man)
    evs = {v: r for v, r in load_jsonl_ok(os.path.join(out_dir(args.dataset), "events.jsonl")).items()
           if not r.get("error")}
    ids = [v for v in L.cohort(args.dataset) if v in evs][:args.limit or 2]
    for vid in ids:
        fps = float(evs[vid]["decode_fps"])
        for k, (a, b) in enumerate(evs[vid]["events"]):
            text, n, cut = speech_text(speech.get(vid, []), a / fps, b / fps, n_tokens, args.speech_cap_tokens)
            print("=== %s event %d [%.2f, %.2f) s: %d speech tokens%s" % (vid, k, a / fps, b / fps, n,
                                                                       " (truncated)" if cut else ""))
            print(build_prompt(speech=text))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("segment", "score", "raster", "prompts", "selftest"))
    ap.add_argument("--dataset", choices=L.DATASETS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--raft-ckpt", default=DEFAULT_RAFT)
    ap.add_argument("--clip-ckpt", default=DEFAULT_CLIP)
    ap.add_argument("--raft-batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--transcript", action="store_true",
                    help="transcript (audio-visual) variant of score/raster/prompts; outputs in eventvad_av/<DS>/")
    ap.add_argument("--speech-cap-tokens", type=int, default=SPEECH_CAP_TOKENS)
    cfgmod.add_config_args(ap)
    ap.set_defaults(preset="upstream")      # released code constants (coordinator 2026-10-08), see README
    args = ap.parse_args(argv)
    if args.stage == "selftest":
        return selftest()
    if not args.dataset:
        ap.error("--dataset is required")
    if args.stage == "prompts":
        args.transcript = True
    if args.transcript and args.stage == "segment":
        ap.error("--transcript reuses the visual-only segmentation; run segment without it")
    return {"segment": cmd_segment, "score": cmd_score, "raster": cmd_raster, "prompts": cmd_prompts}[args.stage](args)


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
    good = build_prompt() == build_prompt(None) and build_prompt().startswith("#Question: ") and \
        build_prompt("hi there").startswith(SPEECH_PREFIX + "hi there\n#Question: ") and \
        build_prompt("x").endswith(build_prompt())
    ok &= good
    print("%s prompt: visual-only unchanged, speech line before the question" % ("PASS" if good else "FAIL"))
    segs = [(0.0, 4.0, " a b c d"), (4.0, 6.0, " e  f"), (10.0, 12.0, "")]
    ntok = lambda t: len(t.split())                                   # noqa: E731  (word count as token count)
    for (t1, t2, cap), want in [((0.0, 2.0, 512), ("a b", 2, False)), ((3.0, 6.0, 512), ("d e f", 3, False)),
                                ((0.0, 6.0, 4), ("a b c d", 4, True)), ((7.0, 12.0, 512), (NO_SPEECH, 0, False))]:
        got = speech_text(segs, t1, t2, ntok, cap)
        good = got == want
        ok &= good
        print("%s speech_text [%s, %s] cap %d -> %r" % ("PASS" if good else "FAIL", t1, t2, cap, got))
    curve = L.units_to_4fps([0.0, 1.5], [1.5, 2.0], [0.1, 0.9], 10)
    good = np.allclose(curve, [0.1] * 6 + [0.9] * 4)
    ok &= good
    print("%s raster midpoint and tail hold %s" % ("PASS" if good else "FAIL", curve))
    print("all passed" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
