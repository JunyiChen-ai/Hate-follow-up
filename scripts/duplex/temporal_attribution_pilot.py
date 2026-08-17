"""Temporal attribution pilot: single-pass evidence localization on HateClipSeg.

Pre-registration: docs/duplex/PREREG_temporal_attribution_pilot.md (frozen
2026-08-12, commit 26ccde5). Nothing in the protocol is re-decided here.

The frozen judge's video-level decision

    z = logsumexp(logits[Yes ids]) - logsumexp(logits[No ids])

is attributed back onto the judge's own input tokens with two pre-named
variants, and the resulting per-second attribution curve is scored against
HateClipSeg's segment-level gold:

  A1 (primary)   gradient x input: |grad_e z . e| per input token.
  A2 (secondary) mean over the last third of layers and all heads of the
                 last prompt position's attention row.
  control        identical pipeline with every token weight set to 1.

The judge prompt is not rebuilt here: `build_messages`, `resolve_frames`,
`resolve_transcript` and the pixel budget are imported from the frozen
extractor `src/duplex/extract_duplex_readout.py`, which imports the prompt
skeleton, the rules and the Yes/No id sets from the frozen
`src/duplex/score_duplex_probe.py`. The frozen HateClipSeg configuration is
`--split test --transcript-limit 0 --transcript-override-json
results/hateclipseg/c2_overrides.json` (scripts/duplex/run_hateclipseg.sh).

Deployment cost: one judge call per video. The call is executed as two
passes of the same input: pass A (eager attention, no grad) reads z and the
attention rows, pass B (sdpa, grad enabled) re-runs the same input to obtain
the gradient of the same z. Splitting the single call into two passes is a
memory measure only: an eager attention graph retained for backward costs
~23 GiB at the corpus's longest prompt, which does not fit beside the bf16
weights on one 32 GiB card.

Output: results/temporal_attribution_pilot/{per_video.jsonl, report.json,
STATUS, DONE}; the driver redirects stdout to run.log.
"""

import argparse
import ast
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
from scipy import stats

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "duplex"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "our_method"))

# Frozen judge material. Imported, never copied.
from extract_duplex_readout import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    READER,
    build_messages,
    resolve_frames,
    resolve_transcript,
)
from score_duplex_probe import (  # noqa: E402
    DUPLEX_PROMPT,
    READER_BLOCKS,
    YOUTUBE_RULES,
    build_binary_token_ids,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

DATASET = "HateClipSeg"
SPLIT = "test"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
NUM_FRAMES = 16
TRANSCRIPT_LIMIT = 0
OVERRIDES_JSON = os.path.join(PROJECT_ROOT, "results", "hateclipseg", "c2_overrides.json")
FROZEN_SCORES = os.path.join(PROJECT_ROOT, "results", "hateclipseg", "judge_8b", "scores.jsonl")
GOLD_CSV = os.path.join(
    PROJECT_ROOT, "idea-stage", "pilots", "b1_coverage_audit", "data",
    "segment_level_annotation.csv")
CHUNKS_JSONL = os.path.join(
    PROJECT_ROOT, "results", "interleaved_timeline", "hateclipseg", "timestamped_chunks.jsonl")
SEGMENTS_JSON = os.path.join(
    PROJECT_ROOT, "results", "interleaved_timeline", "hateclipseg", "segments.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "temporal_attribution_pilot")

POSITIONAL_SEED = 20260812
POSITIONAL_N = 20

CHANNELS = ("all", "transcript", "frame")
WEIGHTINGS = ("a1", "a2", "control")
COLLAPSES = ("union", "strict")


# --------------------------------------------------------------------- gold

def load_gold():
    """video_id -> (labels [n_seg][6], spans [n_seg][2] float seconds)."""
    gold = {}
    with open(GOLD_CSV) as f:
        for row in csv.DictReader(f):
            labels = ast.literal_eval(row["Segment-Level Label"])
            spans = [[float(a), float(b)]
                     for a, b in ast.literal_eval(row["Segment Timestamp"])]
            if len(labels) != len(spans):
                continue
            gold[row["Video Id"]] = (labels, spans)
    return gold


def is_offensive_union(lab):
    return any(int(x) == 1 for x in lab[1:6])


def is_normal_only(lab):
    return int(lab[0]) == 1 and not is_offensive_union(lab)


def is_hateful_strict(lab):
    return int(lab[1]) == 1


def sparse_video(labels):
    return (any(is_offensive_union(l) for l in labels)
            and any(is_normal_only(l) for l in labels))


# ------------------------------------------------------------------ cohort

def build_cohort():
    ids = load_clean_split_ids(DATASET, SPLIT)
    gold = load_gold()
    routes = json.load(open(SEGMENTS_JSON))
    overrides = json.load(open(OVERRIDES_JSON))
    chunks = {}
    with open(CHUNKS_JSONL) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                chunks[r["video_id"]] = r

    counts = {"locally_complete": len(ids)}
    have_gold = [v for v in ids if v in gold]
    counts["has_segment_gold"] = len(have_gold)
    sparse = [v for v in have_gold if sparse_video(gold[v][0])]
    counts["sparse"] = len(sparse)
    routed = [v for v in sparse if routes.get(v, {}).get("route") == "timestamped"]
    counts["route_timestamped"] = len(routed)
    counts["prereg_cohort"] = len(routed)

    cohort, excl = [], {"no_override": [], "alignment_failed": [],
                        "no_chunks": [], "no_duration": []}
    for v in routed:
        t = overrides.get(v)
        if t is None:
            excl["no_override"].append(v)
            continue
        rec = chunks.get(v)
        if not rec or not rec.get("chunks"):
            excl["no_chunks"].append(v)
            continue
        joined = "".join(c.get("text") or "" for c in rec["chunks"])
        if joined.strip() != t:
            excl["alignment_failed"].append(v)
            continue
        dur = rec.get("container_duration") or rec.get("wav_duration")
        if not dur or dur <= 0:
            excl["no_duration"].append(v)
            continue
        cohort.append(v)
    counts["scored"] = len(cohort)
    return cohort, counts, excl, gold, overrides, chunks


# ------------------------------------------------------------ time mapping

def chunk_spans_in_transcript(rec, transcript):
    """Character span in `transcript` and [start, end] seconds for each chunk.

    The chunk texts concatenate to the transcript up to leading/trailing
    whitespace (verified by the cohort filter), so a single leading offset
    maps chunk character positions into transcript coordinates.
    """
    joined = "".join(c.get("text") or "" for c in rec["chunks"])
    lead = len(joined) - len(joined.lstrip())
    out, pos = [], 0
    n = len(transcript)
    for c in rec["chunks"]:
        txt = c.get("text") or ""
        c0, c1 = pos - lead, pos + len(txt) - lead
        pos += len(txt)
        c0, c1 = max(0, min(n, c0)), max(0, min(n, c1))
        if c1 <= c0:
            continue
        s, e = c.get("start"), c.get("end")
        if s is None or e is None:
            continue
        out.append((c0, c1, float(s), float(e)))
    return out


def add_interval(bins, t0, t1, w):
    """Add weight w spread over [t0, t1) into per-second bins."""
    nb = len(bins)
    if nb == 0 or w == 0.0:
        return
    t0 = max(0.0, min(t0, nb - 1e-9))
    t1 = max(0.0, min(t1, nb - 1e-9))
    if t1 <= t0:
        bins[int(t0)] += w
        return
    b0, b1 = int(t0), int(t1)
    span = t1 - t0
    if b0 == b1:
        bins[b0] += w
        return
    for b in range(b0, b1 + 1):
        lo, hi = max(t0, b), min(t1, b + 1.0)
        if hi > lo:
            bins[b] += w * (hi - lo) / span


def segment_density(bins, s, e):
    """Overlap-weighted mean of the per-second bin values over [s, e]."""
    nb = len(bins)
    if nb == 0:
        return 0.0
    s = max(0.0, min(s, nb - 1e-9))
    e = max(0.0, min(e, float(nb)))
    if e - s < 1e-6:
        return float(bins[int(s)])
    tot, wsum = 0.0, 0.0
    for b in range(int(s), min(nb, int(math.ceil(e)))):
        lo, hi = max(s, b), min(e, b + 1.0)
        if hi > lo:
            tot += bins[b] * (hi - lo)
            wsum += (hi - lo)
    return float(tot / wsum) if wsum > 0 else 0.0


# ----------------------------------------------------------------- statistic

def rank_auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    r = stats.rankdata(allv)
    rpos = r[:len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


# --------------------------------------------------------------------- model

class AttnRowCapture:
    """Forward hooks on the last third of text layers.

    Each hook keeps only the last query position's attention row, averaged
    over heads, and drops the full [1, H, q, kv] tensor immediately.
    """

    def __init__(self, layers):
        self.rows = {}
        self.handles = []
        self.enabled = True
        for idx, layer in layers:
            self.handles.append(layer.self_attn.register_forward_hook(self._make(idx)))

    def _make(self, idx):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            w = output[1]
            if w is None:
                raise RuntimeError("attention weights are None; eager attention required")
            self.rows[idx] = w[0, :, -1, :].detach().float().mean(0).cpu().numpy()
        return hook

    def clear(self):
        self.rows = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    status_path = os.path.join(args.out_dir, "STATUS")
    per_video_path = os.path.join(args.out_dir, "per_video.jsonl")
    report_path = os.path.join(args.out_dir, "report.json")

    def status(s):
        with open(status_path, "w") as f:
            f.write(s + "\n")

    status("cohort")
    cohort, counts, excl, gold, overrides, chunks = build_cohort()
    logging.info(f"cohort counts: {counts}")
    logging.info("exclusions: " + json.dumps({k: len(v) for k, v in excl.items()}))
    if args.limit:
        cohort = cohort[:args.limit]
        logging.info(f"--limit: {len(cohort)} videos")

    if not args.analyze_only:
        run_forward(cohort, gold, overrides, chunks, per_video_path, status)

    status("analyze")
    report = analyze(per_video_path, counts, excl, cohort)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(json.dumps(report["headline"], indent=2))
    status("DONE")
    with open(os.path.join(args.out_dir, "DONE"), "w") as f:
        f.write(time.strftime("%F %T") + "\n")


def run_forward(cohort, gold, overrides, chunks, per_video_path, status):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    done = set()
    if os.path.exists(per_video_path):
        with open(per_video_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["video_id"])
    remaining = [v for v in cohort if v not in done]
    logging.info(f"resume: {len(done)} done, {len(remaining)} remaining")
    if not remaining:
        return

    annotations = load_annotations(DATASET)
    frozen_z = {}
    with open(FROZEN_SCORES) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                frozen_z[r["video_id"]] = r.get("z")

    processor = AutoProcessor.from_pretrained(MODEL)
    tokenizer = processor.tokenizer
    ids = build_binary_token_ids(tokenizer)
    yes_ids, no_ids = sorted(ids["Yes"]), sorted(ids["No"])
    logging.info(f"Yes ids {yes_ids} No ids {no_ids}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Activation checkpointing for the backward pass only. Every dropout
    # probability in this checkpoint is 0.0, so the train() flag the
    # checkpointing path requires does not change any value; the run asserts
    # this by comparing the grad pass's z with the frozen judge's z.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    text_model = model.model.language_model
    n_layers = len(text_model.layers)
    late = list(range(n_layers - n_layers // 3, n_layers))
    logging.info(f"text layers={n_layers}; late-layer set={late[0]}..{late[-1]} "
                 f"({len(late)} layers)")
    cap = AttnRowCapture([(i, text_model.layers[i]) for i in late])

    yes_idx = torch.tensor(yes_ids, device=model.device)
    no_idx = torch.tensor(no_ids, device=model.device)
    image_token_id = int(model.config.image_token_id)
    size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    rules_text = YOUTUBE_RULES

    # Gradient entry point: replace the merged inputs_embeds handed to the
    # text model with a leaf copy so that z.backward() populates its .grad.
    grad_slot = {}

    def pre_hook(module, pargs, pkwargs):
        e = pkwargs.get("inputs_embeds")
        if e is None or not grad_slot.get("arm"):
            return None
        e = e.detach().requires_grad_(True)
        grad_slot["e"] = e
        pkwargs["inputs_embeds"] = e
        return pargs, pkwargs

    text_model.register_forward_pre_hook(pre_hook, with_kwargs=True)

    prompt_prefix_tpl = DUPLEX_PROMPT[:DUPLEX_PROMPT.index("{transcript}")]
    t0 = time.time()
    n_done = 0
    fout = open(per_video_path, "a")

    for i, vid in enumerate(remaining):
        if i % 10 == 0:
            status(f"forward {i}/{len(remaining)} {time.strftime('%F %T')}")
        ann = annotations.get(vid)
        frame_paths = resolve_frames(vid, DATASET, NUM_FRAMES)
        if ann is None or not frame_paths:
            raise SystemExit(f"{vid}: missing annotation or frames")
        transcript = resolve_transcript(ann, vid, overrides, TRANSCRIPT_LIMIT)
        if transcript != overrides[vid]:
            raise SystemExit(f"{vid}: transcript is not the c2 override")

        messages = build_messages(ann, frame_paths, rules_text, transcript)
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        # The prompt slot the transcript occupies, recomputed from the frozen
        # skeleton: everything before it is title-only, so its character span
        # inside the prompt text is exact.
        prefix = prompt_prefix_tpl.format(title=ann.get("title", "") or "")
        prompt_text = DUPLEX_PROMPT.format(
            title=ann.get("title", "") or "", transcript=transcript,
            rules=rules_text, reader_block=READER_BLOCKS[READER])
        if prompt_text not in text:
            raise SystemExit(f"{vid}: frozen prompt text not found in chat template output")
        if not prompt_text.startswith(prefix) or \
                prompt_text[len(prefix):len(prefix) + len(transcript)] != transcript:
            raise SystemExit(f"{vid}: transcript slot mismatch")
        c0, c1 = len(prefix), len(prefix) + len(transcript)

        images = [Image.open(p).convert("RGB") for p in frame_paths]
        inputs = processor(text=[text], images=images, return_tensors="pt",
                           size=size_kwarg)
        for im in images:
            im.close()
        del images
        inputs = inputs.to(model.device)
        input_ids = inputs["input_ids"][0]
        seq = int(input_ids.shape[0])

        # --- token -> prompt character offsets for the text tail
        enc = tokenizer(prompt_text, add_special_tokens=False,
                        return_offsets_mapping=True)
        ptoks = torch.tensor(enc["input_ids"], device=input_ids.device)
        start = find_subsequence(input_ids, ptoks)
        if start < 0:
            raise SystemExit(f"{vid}: prompt tokens not a contiguous block of input_ids")

        # --- frame token spans
        img_pos = (input_ids == image_token_id).nonzero(as_tuple=True)[0].tolist()
        runs = contiguous_runs(img_pos)
        if len(runs) != len(frame_paths):
            raise SystemExit(f"{vid}: {len(runs)} image runs != {len(frame_paths)} frames")

        # ------------------------------------------------ pass A: z + attention
        cap.clear()
        cap.enabled = True
        grad_slot["arm"] = False
        with torch.no_grad():
            out = model(**inputs, use_cache=False, logits_to_keep=1)
            last = out.logits[0, -1, :].float()
            z = float(torch.logsumexp(last[yes_idx], 0) - torch.logsumexp(last[no_idx], 0))
        del out, last
        rows = np.stack([cap.rows[k] for k in late], 0)
        a2 = rows.mean(0)
        cap.clear()
        if a2.shape[0] != seq:
            raise SystemExit(f"{vid}: attention row {a2.shape} != seq {seq}")

        # ------------------------------------------------ pass B: gradient
        cap.enabled = False
        grad_slot["arm"] = True
        grad_slot.pop("e", None)
        model.set_attn_implementation("sdpa")
        model.train()
        out = model(**inputs, use_cache=False, logits_to_keep=1)
        last = out.logits[0, -1, :].float()
        zg = torch.logsumexp(last[yes_idx], 0) - torch.logsumexp(last[no_idx], 0)
        zg.backward()
        e = grad_slot["e"]
        a1 = (e.grad[0].float() * e[0].detach().float()).sum(-1).abs().cpu().numpy()
        z_grad = float(zg.detach())
        del out, last, zg, e
        grad_slot.pop("e", None)
        grad_slot["arm"] = False
        model.eval()
        model.set_attn_implementation("eager")
        del inputs
        torch.cuda.empty_cache()

        rec = score_video(vid, gold[vid], chunks[vid], transcript, enc, start,
                          c0, c1, runs, seq, a1, a2, z, z_grad,
                          frozen_z.get(vid), len(frame_paths))
        rec["prompt_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        n_done += 1
        if n_done % 20 == 0 or i == 0:
            el = time.time() - t0
            logging.info(f"  [{n_done}/{len(remaining)}] {vid} z={z:+.3f} "
                         f"seq={seq} {el / max(n_done, 1):.2f}s/video "
                         f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    fout.close()
    cap.remove()
    logging.info(f"forward done: {n_done} videos in {time.time() - t0:.1f}s")


def find_subsequence(hay, needle):
    n, m = int(hay.shape[0]), int(needle.shape[0])
    if m == 0 or m > n:
        return -1
    first = int(needle[0])
    cand = (hay[:n - m + 1] == first).nonzero(as_tuple=True)[0].tolist()
    for s in cand:
        if torch.equal(hay[s:s + m], needle):
            return s
    return -1


def contiguous_runs(positions):
    runs = []
    for p in positions:
        if runs and p == runs[-1][1] + 1:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    return runs


def score_video(vid, goldv, chunk_rec, transcript, enc, tok_start, c0, c1,
                runs, seq, a1, a2, z, z_grad, z_frozen, n_frames):
    labels, spans = goldv
    duration = float(chunk_rec.get("container_duration")
                     or chunk_rec.get("wav_duration"))
    max_end = max([e for _, e in spans] + [duration])
    nb = int(math.ceil(max_end)) + 1

    # --- transcript tokens -> chunk -> [start, end]
    cspans = chunk_spans_in_transcript(chunk_rec, transcript)
    tok_time = []          # (token_index, t0, t1, channel)
    for j, (o0, o1) in enumerate(enc["offset_mapping"]):
        if o1 <= o0:
            continue
        s, e = o0, o1
        if e <= c0 or s >= c1:
            continue
        s, e = max(s, c0), min(e, c1)
        rs, re = s - c0, e - c0
        best, bov = None, 0
        for (a, b, ts, te) in cspans:
            ov = min(re, b) - max(rs, a)
            if ov > bov:
                bov, best = ov, (ts, te)
        if best is None:
            continue
        tok_time.append((tok_start + j, best[0], best[1], "transcript"))
    n_transcript_tokens = len(tok_time)

    # --- frame tokens -> nominal frame timestamp
    n_frame_tokens = 0
    for fi, (a, b) in enumerate(runs):
        t = (fi + 0.5) / float(n_frames) * duration
        for p in range(a, b + 1):
            tok_time.append((p, t, t, "frame"))
            n_frame_tokens += 1

    weights = {"a1": a1, "a2": a2, "control": np.ones(seq, dtype=np.float64)}
    # Tokens sharing a time interval (same ASR chunk, or same frame) are
    # pooled before binning; this is exactly equivalent to binning them one
    # by one and keeps the per-video cost independent of transcript length.
    groups = {}
    for (p, t0, t1, c) in tok_time:
        groups.setdefault((t0, t1, c), []).append(p)
    curves = {}
    for wname, w in weights.items():
        for ch in CHANNELS:
            bins = np.zeros(nb, dtype=np.float64)
            for (t0, t1, c), idxs in groups.items():
                if ch != "all" and c != ch:
                    continue
                add_interval(bins, t0, t1, float(np.asarray(w)[idxs].sum()))
            curves[(wname, ch)] = bins

    pos_idx = [k for k, l in enumerate(labels) if is_offensive_union(l)]
    neg_idx = [k for k, l in enumerate(labels) if is_normal_only(l)]
    strict_idx = [k for k, l in enumerate(labels) if is_hateful_strict(l)]

    aucs = {}
    for wname in WEIGHTINGS:
        for ch in CHANNELS:
            bins = curves[(wname, ch)]
            dens = [segment_density(bins, s, e) for s, e in spans]
            for coll, pidx in (("union", pos_idx), ("strict", strict_idx)):
                key = f"{wname}|{ch}|{coll}"
                aucs[key] = rank_auc([dens[k] for k in pidx],
                                     [dens[k] for k in neg_idx])

    # --- positional quartiles (share of total attribution over all prompt
    # positions, not only the curve's tokens)
    quart = {}
    for wname in WEIGHTINGS:
        w = np.asarray(weights[wname], dtype=np.float64)
        tot = float(w.sum())
        qs = []
        for q in range(4):
            a = int(round(seq * q / 4.0))
            b = int(round(seq * (q + 1) / 4.0))
            qs.append(float(w[a:b].sum() / tot) if tot > 0 else 0.0)
        quart[wname] = qs

    return {
        "video_id": vid,
        "z": z,
        "z_grad_pass": z_grad,
        "z_frozen": z_frozen,
        "seq_len": seq,
        "duration": duration,
        "n_segments": len(labels),
        "n_pos_union": len(pos_idx),
        "n_pos_strict": len(strict_idx),
        "n_neg": len(neg_idx),
        "n_transcript_tokens": n_transcript_tokens,
        "n_frame_tokens": n_frame_tokens,
        "n_chunks": len(chunk_rec.get("chunks", [])),
        "auc": aucs,
        "quartiles": quart,
    }


def analyze(per_video_path, counts, excl, cohort):
    recs = []
    with open(per_video_path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    keys = sorted({k for r in recs for k in r["auc"]})
    macro, dist = {}, {}
    for k in keys:
        vals = [r["auc"][k] for r in recs if r["auc"].get(k) is not None]
        macro[k] = float(np.mean(vals)) if vals else None
        dist[k] = {
            "n_videos": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
            "min": float(np.min(vals)) if vals else None,
            "p25": float(np.percentile(vals, 25)) if vals else None,
            "median": float(np.median(vals)) if vals else None,
            "p75": float(np.percentile(vals, 75)) if vals else None,
            "max": float(np.max(vals)) if vals else None,
            "frac_above_0.5": float(np.mean([v > 0.5 for v in vals])) if vals else None,
        }

    # correlation of per-video AUC with the video-level decision z
    corr = {}
    for k in ("a1|all|union", "a2|all|union", "control|all|union"):
        pairs = [(r["auc"][k], r["z"]) for r in recs if r["auc"].get(k) is not None]
        if len(pairs) > 2:
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            corr[k] = {"pearson_r": float(stats.pearsonr(a, b)[0]),
                       "pearson_p": float(stats.pearsonr(a, b)[1]),
                       "spearman_r": float(stats.spearmanr(a, b)[0]),
                       "spearman_p": float(stats.spearmanr(a, b)[1]),
                       "n": len(pairs)}

    # positional artifact check on 20 videos drawn with the frozen seed
    rng = random.Random(POSITIONAL_SEED)
    ids = sorted(r["video_id"] for r in recs)
    sample = sorted(rng.sample(ids, min(POSITIONAL_N, len(ids))))
    byid = {r["video_id"]: r for r in recs}
    positional = {"seed": POSITIONAL_SEED, "n": len(sample), "video_ids": sample}
    for wname in WEIGHTINGS:
        q = np.array([byid[v]["quartiles"][wname] for v in sample], dtype=float)
        positional[wname] = {"mean_share_by_quartile": [float(x) for x in q.mean(0)],
                             "std_share_by_quartile": [float(x) for x in q.std(0)]}

    control = macro.get("control|all|union")
    decision = {}
    survive = False
    for arm in ("a1", "a2"):
        m = macro.get(f"{arm}|all|union")
        c1_ok = m is not None and m >= 0.65
        c2_ok = m is not None and control is not None and m >= control + 0.05
        decision[arm] = {"macro_auc": m, "clause1_ge_0.65": c1_ok,
                         "clause2_ge_control_plus_0.05": c2_ok,
                         "passes": bool(c1_ok and c2_ok)}
        survive = survive or (c1_ok and c2_ok)
    decision["control_macro_auc"] = control
    decision["verdict"] = "SURVIVE" if survive else "DIE"

    zs = [(r["z"], r["z_frozen"]) for r in recs if r.get("z_frozen") is not None]
    zdiff = [abs(a - b) for a, b in zs]
    zpass = [abs(r["z"] - r["z_grad_pass"]) for r in recs]

    return {
        "cohort_counts": counts,
        "exclusions": {k: {"n": len(v), "video_ids": v} for k, v in excl.items()},
        "n_scored": len(recs),
        "headline": {
            "cohort_prereg": counts["prereg_cohort"],
            "scored": len(recs),
            "macro_auc": {k: macro[k] for k in
                          ("a1|all|union", "a2|all|union", "control|all|union",
                           "a1|all|strict", "a2|all|strict", "control|all|strict",
                           "a1|transcript|union", "a2|transcript|union",
                           "a1|frame|union", "a2|frame|union",
                           "control|transcript|union", "control|frame|union")
                          if k in macro},
            "decision": decision,
        },
        "macro_auc_all": macro,
        "per_video_auc_distribution": dist,
        "auc_vs_z_correlation": corr,
        "positional_check": positional,
        "z_agreement": {
            "n_vs_frozen": len(zs),
            "max_abs_diff_vs_frozen": float(max(zdiff)) if zdiff else None,
            "mean_abs_diff_vs_frozen": float(np.mean(zdiff)) if zdiff else None,
            "max_abs_diff_eager_vs_sdpa_pass": float(max(zpass)) if zpass else None,
        },
        "token_counts": {
            "transcript_tokens_mean": float(np.mean([r["n_transcript_tokens"] for r in recs])),
            "frame_tokens_mean": float(np.mean([r["n_frame_tokens"] for r in recs])),
            "seq_len_mean": float(np.mean([r["seq_len"] for r in recs])),
        },
    }


if __name__ == "__main__":
    main()
