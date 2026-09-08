"""Sentinel logit-lens localization pilot (direction 2) on HateClipSeg.

Pre-registration: docs/duplex/PREREG_sentinel_localization_pilot.md (frozen
2026-08-12, commit 99d492c). Nothing in that protocol is re-decided here.

Direction 1 attributed the global verdict back onto the input and died. This
pilot does not attribute anything: it ELICITS a local judgment at every point
of the timeline inside one pass, and reads each judgment off the model's own
answer direction.

Arm A (the claim, "locator" call). One prompt: the frozen judge's union rules
block, a fixed locator instruction, the 16 frames, the title, then the
timestamped ASR chunks in temporal order, each written as

    [k | <start>-<end>s] <chunk text> ⟦ASSESS⟧

No text is generated. At the final token position of each sentinel the late
layers' hidden states are pushed through the model's final RMSNorm and
unembedding, and

    m_k = logsumexp(logits[Yes ids]) - logsumexp(logits[No ids])

is read there. Frozen aggregate: median of the last four decoder layers'
margins. The final layer's margin is kept descriptively.

Arm B (diagnostic capability ceiling, separate single pass). Same content with
no sentinels, followed by one numbered question per chunk
("Q<k>: Does segment <k> violate the rules? Answer:"); the margin is the
next-token margin at each answer position, read from the final layer.

Curve and statistic are direction 1's machinery, for comparability: each
chunk's margin is spread uniformly over the chunk's [start, end], binned per
second, averaged over each gold segment's span, and the per-video AUC of
offensive-union segments against normal-only segments is macro-averaged.

Frozen material is imported, never copied: YOUTUBE_RULES, DUPLEX_PROMPT,
READER_BLOCKS, SYSTEM_MESSAGE and build_binary_token_ids come from the frozen
judge; resolve_frames and the pixel budget from the frozen extractor.

Output: results/sentinel_localization/{per_video.jsonl, report.json, STATUS,
DONE}; the driver redirects stdout to run.log.
"""

import argparse
import ast
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time

import numpy as np
import torch
from scipy import stats

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "duplex"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "our_method"))

from extract_duplex_readout import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    resolve_frames,
)
from score_duplex_probe import (  # noqa: E402
    DUPLEX_PROMPT,
    READER_BLOCKS,
    SYSTEM_MESSAGE,
    YOUTUBE_RULES,
    build_binary_token_ids,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

DATASET = "HateClipSeg"
SPLIT = "test"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
NUM_FRAMES = 16
MAX_PROMPT_TOKENS = 24000

GOLD_CSV = os.path.join(
    PROJECT_ROOT, "idea-stage", "pilots", "b1_coverage_audit", "data",
    "segment_level_annotation.csv")
CHUNKS_JSONL = os.path.join(
    PROJECT_ROOT, "results", "interleaved_timeline", "hateclipseg",
    "timestamped_chunks.jsonl")
FROZEN_SCORES = os.path.join(
    PROJECT_ROOT, "results", "hateclipseg", "judge_8b", "scores.jsonl")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "sentinel_localization")

SENTINEL = "⟦ASSESS⟧"

# --------------------------------------------------------------- frozen text
# Written once, before any margin was computed, and hashed into the report.
# The rules block is the frozen judge's YOUTUBE_RULES, inserted verbatim, under
# the frozen judge's own lead-in sentence (DUPLEX_PROMPT).

RULES_LEAD_IN = ("Based on this platform's hate speech policy, the following "
                 "types of content are prohibited:")

LOCATOR_INSTRUCTION = (
    "The video's frames and its title follow, and then its transcript in "
    "temporal order, split into segments. Each segment is written on its own "
    "line as \"[k | start-end s] segment text\" and is followed by the marker "
    + SENTINEL + ". At each " + SENTINEL + " marker, judge whether the "
    "segment just shown violates the rules above, and answer \"Yes\" or "
    "\"No\" in your head before reading on. Write nothing."
)

ARM_B_INSTRUCTION = (
    "The video's frames and its title follow, and then its transcript in "
    "temporal order, split into numbered segments. Each segment is written on "
    "its own line as \"[k | start-end s] segment text\". After the transcript "
    "there is one question per segment, asking whether that segment violates "
    "the rules above. Answer each question with \"Yes\" or \"No\"."
)

LOCATOR_PREFIX = RULES_LEAD_IN + "\n" + YOUTUBE_RULES + "\n\n" + LOCATOR_INSTRUCTION
ARM_B_PREFIX = RULES_LEAD_IN + "\n" + YOUTUBE_RULES + "\n\n" + ARM_B_INSTRUCTION


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


FROZEN_TEXT_SHA = {
    "rules_block": sha(YOUTUBE_RULES),
    "locator_instruction": sha(LOCATOR_INSTRUCTION),
    "locator_prefix": sha(LOCATOR_PREFIX),
    "arm_b_instruction": sha(ARM_B_INSTRUCTION),
    "arm_b_prefix": sha(ARM_B_PREFIX),
    "sentinel": sha(SENTINEL),
}


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

def usable_spans(rec):
    """[(start, end)] for every chunk, or None if the record is unusable.

    A missing final end time is replaced by wav_duration (container_duration as
    a fallback). This touches no model input: the spans only place the chunk on
    the timeline for scoring.
    """
    chunks = rec.get("chunks") or []
    if not chunks:
        return None
    dur = rec.get("wav_duration") or rec.get("container_duration")
    spans = []
    for i, c in enumerate(chunks):
        s, e = c.get("start"), c.get("end")
        if s is None:
            return None
        if e is None:
            if i != len(chunks) - 1:
                return None
            if not dur or float(dur) <= float(s):
                return None
            e = float(dur)
        spans.append((float(s), float(e)))
    return spans


def build_cohort():
    ids = load_clean_split_ids(DATASET, SPLIT)
    gold = load_gold()
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

    cohort, excl = [], {"no_chunks": [], "unusable_spans": []}
    for v in sparse:
        rec = chunks.get(v)
        if not rec or not rec.get("chunks"):
            excl["no_chunks"].append(v)
            continue
        if usable_spans(rec) is None:
            excl["unusable_spans"].append(v)
            continue
        cohort.append(v)
    counts["prereg_cohort"] = len(cohort)
    return cohort, counts, excl, gold, chunks


# ------------------------------------------------------------ time mapping

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


def add_interval_flat(bins, t0, t1, w):
    """Descriptive variant: hold w constant over every second of [t0, t1)."""
    nb = len(bins)
    if nb == 0:
        return
    t0 = max(0.0, min(t0, nb - 1e-9))
    t1 = max(0.0, min(t1, nb - 1e-9))
    if t1 <= t0:
        bins[int(t0)] += w
        return
    for b in range(int(t0), int(t1) + 1):
        lo, hi = max(t0, b), min(t1, b + 1.0)
        if hi > lo:
            bins[b] += w * (hi - lo)


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


def rank_auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    r = stats.rankdata(allv)
    rpos = r[:len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


# ------------------------------------------------------------------ prompts

_WS = re.compile(r"\s+")


def clean_chunk_text(t):
    """One-line chunk text with the sentinel's bracket characters removed.

    Removing U+27E6/U+27E7 guarantees that every occurrence of the sentinel
    string in the prompt is a marker the script emitted.
    """
    t = (t or "").replace("⟦", "").replace("⟧", "")
    return _WS.sub(" ", t).strip()


def chunk_lines(spans, texts):
    return ["[%d | %.0f-%.0fs] %s" % (k, spans[k][0], spans[k][1], texts[k])
            for k in range(len(texts))]


def locator_tail(title, lines):
    body = "\n".join("%s %s" % (ln, SENTINEL) for ln in lines)
    return "Title: %s\n\nTranscript segments:\n%s" % (title, body)


def arm_b_tail(title, lines):
    body = "\n".join(lines)
    qs = "\n".join("Q%d: Does segment %d violate the rules? Answer:" % (k, k)
                   for k in range(len(lines)))
    return ("Title: %s\n\nTranscript segments:\n%s\n\nQuestions:\n%s"
            % (title, body, qs))


def build_messages_split(prefix, frame_paths, tail):
    """System turn, then one user turn: prefix text, frames, tail text.

    The frozen judge puts all images first; the pre-registration fixes this
    prompt's order as rules -> instruction -> frames -> title -> chunks, so the
    instruction precedes the images here.
    """
    content = [{"type": "text", "text": prefix}]
    content += [{"type": "image"} for _ in frame_paths]
    content.append({"type": "text", "text": tail})
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]


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


def marker_positions(tokenizer, input_ids, tail, marker, search_from=0):
    """Absolute input_ids index of the final token of every `marker` in `tail`.

    The tail is a contiguous block of input_ids (it follows the last image
    placeholder and precedes the chat template's closing special tokens), so it
    is tokenized on its own and located by exact subsequence match.
    """
    enc = tokenizer(tail, add_special_tokens=False, return_offsets_mapping=True)
    tail_ids = torch.tensor(enc["input_ids"], device=input_ids.device)
    start = find_subsequence(input_ids, tail_ids)
    if start < 0:
        raise RuntimeError("tail tokens are not a contiguous block of input_ids")
    offs = enc["offset_mapping"]

    char_ends, i = [], search_from
    while True:
        j = tail.find(marker, i)
        if j < 0:
            break
        char_ends.append(j + len(marker))
        i = j + len(marker)

    out = []
    for ce in char_ends:
        hit = None
        for j, (a, b) in enumerate(offs):
            if b <= a:
                continue
            if a < ce <= b:
                hit = j
                break
        if hit is None:
            raise RuntimeError("marker end %d has no covering token" % ce)
        out.append(start + hit)
    return out, start, len(enc["input_ids"])


# ------------------------------------------------------------------- readout

class Lens:
    """final RMSNorm -> unembedding -> Yes/No margin, at chosen positions."""

    def __init__(self, model, yes_idx, no_idx):
        self.norm = model.model.language_model.norm
        self.head = model.lm_head
        self.yes = yes_idx
        self.no = no_idx

    def margins(self, h, positions, already_normed=False):
        """h: [1, seq, d] hidden state. Returns list of float margins."""
        x = h[0, positions, :]
        if not already_normed:
            x = self.norm(x)
        logits = self.head(x).float()
        m = (torch.logsumexp(logits[:, self.yes], dim=-1)
             - torch.logsumexp(logits[:, self.no], dim=-1))
        return [float(v) for v in m]


def layer_margins(lens, hidden_states, positions, n_last=4):
    """Margins at `positions` for the last `n_last` decoder layers.

    transformers 4.57.1 returns L+1 hidden states for an L-layer text model:
    index 0 is the embedding and index i (1..L) is the RAW output of decoder
    layer i-1. None of them has the final norm applied (verified at run time
    against the model's own logits), so the final RMSNorm is applied here to
    every layer read, including the last.
    """
    L = len(hidden_states) - 1
    out = {}
    for k in range(n_last):
        layer = L - n_last + k          # decoder layer index
        out[layer] = lens.margins(hidden_states[layer + 1], positions, False)
    return out


# ---------------------------------------------------------------------- main

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
    cohort, counts, excl, gold, chunks = build_cohort()
    logging.info("cohort counts: %s" % json.dumps(counts))
    logging.info("exclusions: %s" % json.dumps({k: len(v) for k, v in excl.items()}))
    logging.info("frozen text sha256: %s" % json.dumps(FROZEN_TEXT_SHA, indent=2))
    logging.info("LOCATOR_INSTRUCTION:\n%s" % LOCATOR_INSTRUCTION)
    logging.info("ARM_B_INSTRUCTION:\n%s" % ARM_B_INSTRUCTION)
    if args.limit:
        cohort = cohort[:args.limit]
        logging.info("--limit: %d videos" % len(cohort))

    if not args.analyze_only:
        run_forward(cohort, gold, chunks, per_video_path, status)

    status("analyze")
    report = analyze(per_video_path, counts, excl, cohort)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(json.dumps(report["headline"], indent=2))
    status("DONE")
    with open(os.path.join(args.out_dir, "DONE"), "w") as f:
        f.write(time.strftime("%F %T") + "\n")


def run_forward(cohort, gold, chunks, per_video_path, status):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    done = set()
    if os.path.exists(per_video_path):
        with open(per_video_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["video_id"])
    remaining = [v for v in cohort if v not in done]
    logging.info("resume: %d done, %d remaining" % (len(done), len(remaining)))
    if not remaining:
        return

    annotations = load_annotations(DATASET)
    frozen_z = {}
    if os.path.exists(FROZEN_SCORES):
        with open(FROZEN_SCORES) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    frozen_z[r["video_id"]] = r.get("z")
        logging.info("frozen judge z: %d videos from %s"
                     % (len(frozen_z), FROZEN_SCORES))
    else:
        logging.info("frozen judge scores absent at %s; z correlation skipped"
                     % FROZEN_SCORES)

    processor = AutoProcessor.from_pretrained(MODEL)
    tokenizer = processor.tokenizer
    ids = build_binary_token_ids(tokenizer)
    yes_ids, no_ids = sorted(ids["Yes"]), sorted(ids["No"])
    logging.info("Yes ids %s No ids %s" % (yes_ids, no_ids))

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = len(model.model.language_model.layers)
    logging.info("text layers=%d; last-4 set=%d..%d"
                 % (n_layers, n_layers - 4, n_layers - 1))
    lens = Lens(model,
                torch.tensor(yes_ids, device=model.device),
                torch.tensor(no_ids, device=model.device))
    size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}

    t0 = time.time()
    n_done = 0
    fout = open(per_video_path, "a")
    verified = {"lens": False}

    for i, vid in enumerate(remaining):
        if i % 10 == 0:
            status("forward %d/%d %s" % (i, len(remaining), time.strftime("%F %T")))
        ann = annotations.get(vid)
        frame_paths = resolve_frames(vid, DATASET, NUM_FRAMES)
        if ann is None or not frame_paths:
            raise SystemExit("%s: missing annotation or frames" % vid)
        title = ann.get("title", "") or ""
        rec = chunks[vid]
        spans = usable_spans(rec)
        texts = [clean_chunk_text(c.get("text")) for c in rec["chunks"]]
        n_chunks = len(texts)
        lines = chunk_lines(spans, texts)
        ntok_chunk = [len(tokenizer(t, add_special_tokens=False)["input_ids"])
                      for t in texts]

        images = [Image.open(p).convert("RGB") for p in frame_paths]

        out_rec = {
            "video_id": vid,
            "n_chunks": n_chunks,
            "n_frames": len(frame_paths),
            "chunk_spans": spans,
            "chunk_tokens": ntok_chunk,
            "z_frozen": frozen_z.get(vid),
            "duration": float(rec.get("wav_duration")
                              or rec.get("container_duration") or spans[-1][1]),
        }

        # ------------------------------------------------------------- arm A
        tail_a = locator_tail(title, lines)
        msgs_a = build_messages_split(LOCATOR_PREFIX, frame_paths, tail_a)
        text_a = processor.apply_chat_template(msgs_a, tokenize=False,
                                               add_generation_prompt=True)
        inputs = processor(text=[text_a], images=images, return_tensors="pt",
                           size=size_kwarg)
        seq_a = int(inputs["input_ids"].shape[1])
        out_rec["seq_len_a"] = seq_a
        out_rec["prompt_sha256_a"] = hashlib.sha256(text_a.encode()).hexdigest()
        if seq_a > MAX_PROMPT_TOKENS:
            out_rec["excluded_too_long"] = "arm_a"
            for im in images:
                im.close()
            fout.write(json.dumps(out_rec) + "\n")
            fout.flush()
            logging.info("  %s SKIPPED arm A: %d tokens > %d"
                         % (vid, seq_a, MAX_PROMPT_TOKENS))
            continue

        inputs = inputs.to(model.device)
        input_ids = inputs["input_ids"][0]
        pos_a, _, _ = marker_positions(tokenizer, input_ids, tail_a, SENTINEL)
        if len(pos_a) != n_chunks:
            raise SystemExit("%s: %d sentinels located, %d chunks"
                             % (vid, len(pos_a), n_chunks))
        with torch.no_grad():
            out = model(**inputs, use_cache=False, output_hidden_states=True,
                        logits_to_keep=1)
            hs = out.hidden_states
            if len(hs) != n_layers + 1:
                raise SystemExit("%s: %d hidden states, %d layers"
                                 % (vid, len(hs), n_layers))
            if not verified["lens"]:
                ref = out.logits[0, -1, :].float()
                chk = lens.margins(hs[-1], [seq_a - 1], False)[0]
                ref_m = float(torch.logsumexp(ref[lens.yes], 0)
                              - torch.logsumexp(ref[lens.no], 0))
                d = abs(chk - ref_m)
                logging.info("lens check: final-layer margin %.6f vs model "
                             "logits %.6f (|d|=%.2e)" % (chk, ref_m, d))
                if d > 0.5:
                    raise SystemExit("lens does not reproduce model logits")
                verified["lens"] = True
            lm = layer_margins(lens, hs, pos_a, 4)
        del out, hs, inputs
        torch.cuda.empty_cache()

        layers = sorted(lm)
        arr = np.array([lm[l] for l in layers], dtype=np.float64)  # [4, n_chunks]
        out_rec["arm_a"] = {
            "layers": layers,
            "per_layer_margin": {str(l): lm[l] for l in layers},
            "median4_margin": [float(x) for x in np.median(arr, axis=0)],
            "final_layer_margin": [float(x) for x in arr[-1]],
        }

        # ------------------------------------------------------------- arm B
        tail_b = arm_b_tail(title, lines)
        msgs_b = build_messages_split(ARM_B_PREFIX, frame_paths, tail_b)
        text_b = processor.apply_chat_template(msgs_b, tokenize=False,
                                               add_generation_prompt=True)
        inputs = processor(text=[text_b], images=images, return_tensors="pt",
                           size=size_kwarg)
        seq_b = int(inputs["input_ids"].shape[1])
        out_rec["seq_len_b"] = seq_b
        out_rec["prompt_sha256_b"] = hashlib.sha256(text_b.encode()).hexdigest()
        for im in images:
            im.close()
        del images

        if seq_b > MAX_PROMPT_TOKENS:
            out_rec["excluded_too_long_b"] = True
            logging.info("  %s SKIPPED arm B: %d tokens > %d"
                         % (vid, seq_b, MAX_PROMPT_TOKENS))
        else:
            inputs = inputs.to(model.device)
            input_ids = inputs["input_ids"][0]
            qstart = tail_b.rindex("\nQuestions:\n")
            pos_b, _, _ = marker_positions(tokenizer, input_ids, tail_b,
                                           "Answer:", search_from=qstart)
            if len(pos_b) != n_chunks:
                raise SystemExit("%s: %d answer slots located, %d chunks"
                                 % (vid, len(pos_b), n_chunks))
            with torch.no_grad():
                out = model(**inputs, use_cache=False,
                            output_hidden_states=True, logits_to_keep=1)
                mb = lens.margins(out.hidden_states[-1], pos_b, False)
            del out, inputs
            torch.cuda.empty_cache()
            out_rec["arm_b"] = {"margin": mb}

        score_video(out_rec, gold[vid])
        fout.write(json.dumps(out_rec) + "\n")
        fout.flush()
        n_done += 1
        if n_done % 20 == 0 or i == 0:
            el = time.time() - t0
            logging.info("  [%d/%d] %s n_chunks=%d seq_a=%d seq_b=%d "
                         "%.2fs/video peak_vram=%.2f GiB"
                         % (n_done, len(remaining), vid, n_chunks, seq_a, seq_b,
                            el / max(n_done, 1),
                            torch.cuda.max_memory_allocated() / 2 ** 30))
    fout.close()
    logging.info("forward done: %d videos in %.1fs" % (n_done, time.time() - t0))


# ------------------------------------------------------------------- scoring

def curves_and_auc(spans, weights, gold_spans, labels, duration, flat=False):
    max_end = max([e for _, e in gold_spans] + [e for _, e in spans] + [duration])
    nb = int(math.ceil(max_end)) + 1
    bins = np.zeros(nb, dtype=np.float64)
    add = add_interval_flat if flat else add_interval
    for (s, e), w in zip(spans, weights):
        add(bins, s, e, float(w))
    dens = [segment_density(bins, s, e) for s, e in gold_spans]
    pos = [k for k, l in enumerate(labels) if is_offensive_union(l)]
    neg = [k for k, l in enumerate(labels) if is_normal_only(l)]
    strict = [k for k, l in enumerate(labels) if is_hateful_strict(l)]
    return {
        "union": rank_auc([dens[k] for k in pos], [dens[k] for k in neg]),
        "strict": rank_auc([dens[k] for k in strict], [dens[k] for k in neg]),
    }


def score_video(rec, goldv):
    labels, gold_spans = goldv
    spans = [tuple(s) for s in rec["chunk_spans"]]
    dur = rec["duration"]
    aucs = {}
    series = {}
    if "arm_a" in rec:
        series["a_median4"] = rec["arm_a"]["median4_margin"]
        series["a_final"] = rec["arm_a"]["final_layer_margin"]
    if "arm_b" in rec:
        series["b_final"] = rec["arm_b"]["margin"]
    series["density"] = [float(x) for x in rec["chunk_tokens"]]
    for name, w in series.items():
        for coll, v in curves_and_auc(spans, w, gold_spans, labels, dur).items():
            aucs["%s|%s" % (name, coll)] = v
        for coll, v in curves_and_auc(spans, w, gold_spans, labels, dur,
                                      flat=True).items():
            aucs["%s_flat|%s" % (name, coll)] = v
    rec["auc"] = aucs
    rec["n_pos_union"] = sum(1 for l in labels if is_offensive_union(l))
    rec["n_pos_strict"] = sum(1 for l in labels if is_hateful_strict(l))
    rec["n_neg"] = sum(1 for l in labels if is_normal_only(l))
    rec["n_segments"] = len(labels)
    # position drift: Spearman(margin, chunk index)
    drift = {}
    for name in ("a_median4", "a_final", "b_final"):
        if name in series and len(series[name]) >= 4:
            r = stats.spearmanr(np.arange(len(series[name])),
                                np.asarray(series[name], float))
            if np.isfinite(r.statistic):
                drift[name] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    rec["position_drift"] = drift
    for name in ("a_median4", "a_final", "b_final"):
        if name in series:
            rec["mean_margin_" + name] = float(np.mean(series[name]))
    return rec


def dist(vals):
    if not vals:
        return None
    v = np.asarray(vals, float)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "p25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "p75": float(np.percentile(v, 75)),
        "max": float(v.max()),
        "frac_above_0.5": float(np.mean(v > 0.5)),
    }


def analyze(per_video_path, counts, excl, cohort):
    recs = []
    with open(per_video_path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    scored = [r for r in recs if "auc" in r]
    too_long_a = [r["video_id"] for r in recs if r.get("excluded_too_long")]
    too_long_b = [r["video_id"] for r in recs if r.get("excluded_too_long_b")]

    keys = sorted({k for r in scored for k in r["auc"]})
    macro, dists = {}, {}
    for k in keys:
        vals = [r["auc"][k] for r in scored if r["auc"].get(k) is not None]
        macro[k] = float(np.mean(vals)) if vals else None
        dists[k] = dist(vals)

    # position drift distribution
    drift = {}
    for name in ("a_median4", "a_final", "b_final"):
        rhos = [r["position_drift"][name]["rho"] for r in scored
                if name in r.get("position_drift", {})]
        ps = [r["position_drift"][name]["p"] for r in scored
              if name in r.get("position_drift", {})]
        if rhos:
            d = dist(rhos)
            d["frac_p_lt_0.05"] = float(np.mean(np.asarray(ps) < 0.05))
            d["frac_rho_negative"] = float(np.mean(np.asarray(rhos) < 0))
            drift[name] = d

    # margin level summaries
    levels = {}
    for name in ("a_median4", "a_final", "b_final"):
        vals = [r["mean_margin_" + name] for r in scored
                if "mean_margin_" + name in r]
        if vals:
            levels[name] = dist(vals)

    # correlation of the video's mean margin with the frozen detection z
    corr = {}
    for name in ("a_median4", "a_final", "b_final"):
        pairs = [(r["mean_margin_" + name], r["z_frozen"]) for r in scored
                 if r.get("z_frozen") is not None and "mean_margin_" + name in r]
        if len(pairs) > 2:
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            corr[name] = {"n": len(pairs),
                          "pearson_r": float(stats.pearsonr(a, b)[0]),
                          "pearson_p": float(stats.pearsonr(a, b)[1]),
                          "spearman_r": float(stats.spearmanr(a, b).statistic),
                          "spearman_p": float(stats.spearmanr(a, b).pvalue)}

    control = macro.get("density|union")
    a_macro = macro.get("a_median4|union")
    c1 = a_macro is not None and a_macro >= 0.65
    c2 = (a_macro is not None and control is not None
          and a_macro >= control + 0.05)
    b_macro = macro.get("b_final|union")
    if b_macro is None:
        branch = "arm B not scored"
    elif c1 and c2:
        branch = ("A passes -> licenses the full method preregistration")
    elif b_macro >= 0.65:
        branch = ("A fails, Arm B >= 0.65 -> family survives only as Arm B "
                  "(novelty penalty; owner decision, not promoted unilaterally)")
    elif a_macro is not None and a_macro < 0.60 and b_macro < 0.60:
        branch = ("A and B both < 0.60 -> elicited-local-judgment family dies; "
                  "goal loop re-enters idea discovery")
    else:
        branch = ("A fails and Arm B in [0.60, 0.65) -> A dies; B does not "
                  "reach the 0.65 clause and the 'both < 0.60' death clause "
                  "does not fire either")
    decision = {
        "arm_a_macro_auc_median4_union": a_macro,
        "arm_a_macro_auc_final_layer_union": macro.get("a_final|union"),
        "arm_b_macro_auc_union": b_macro,
        "token_density_control_macro_auc_union": control,
        "clause1_ge_0.65": bool(c1),
        "clause2_ge_control_plus_0.05": bool(c2),
        "verdict": "SURVIVE" if (c1 and c2) else "DIE",
        "interpretation_branch": branch,
    }

    return {
        "frozen_text": {
            "sentinel": SENTINEL,
            "locator_instruction": LOCATOR_INSTRUCTION,
            "arm_b_instruction": ARM_B_INSTRUCTION,
            "rules_lead_in": RULES_LEAD_IN,
            "sha256": FROZEN_TEXT_SHA,
        },
        "cohort_counts": counts,
        "exclusions": {k: {"n": len(v), "video_ids": v} for k, v in excl.items()},
        "excluded_too_long_arm_a": {"n": len(too_long_a), "video_ids": too_long_a},
        "excluded_too_long_arm_b": {"n": len(too_long_b), "video_ids": too_long_b},
        "n_scored": len(scored),
        "headline": {
            "cohort_prereg": counts["prereg_cohort"],
            "scored": len(scored),
            "macro_auc": {k: macro[k] for k in
                          ("a_median4|union", "a_final|union", "b_final|union",
                           "density|union", "a_median4|strict", "a_final|strict",
                           "b_final|strict", "density|strict")
                          if k in macro},
            "decision": decision,
        },
        "macro_auc_all": macro,
        "per_video_auc_distribution": dists,
        "position_drift": drift,
        "margin_levels": levels,
        "mean_margin_vs_frozen_z": corr,
        "prompt_size": {
            "seq_len_a": dist([r["seq_len_a"] for r in recs if "seq_len_a" in r]),
            "seq_len_b": dist([r["seq_len_b"] for r in recs if "seq_len_b" in r]),
            "n_chunks": dist([r["n_chunks"] for r in recs]),
        },
    }


if __name__ == "__main__":
    main()
