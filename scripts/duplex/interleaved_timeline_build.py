"""Interleaved-timeline kill test, stage B (CPU): assign transcript to frame slots.

Takes the transcript the frozen judge actually saw -- the C2-gated fresh
Whisper text where the gate accepted it, the dataset transcript otherwise --
and splits it across the sixteen frame slots without changing a character of it.

Which characters survive the frozen repetition collapse is decided by the frozen
`split_units` and `collapse_repeats` of
`scripts/duplex/channel_restoration_gate.py`, re-implemented here with character
offsets attached; the offsets are the only addition, and equivalence to the
frozen routines is asserted by rebuilding the judged text.

The surviving text is then cut into atomic pieces at clause boundaries, Whisper
chunk boundaries, and whitespace runs, and each piece is placed by its own
midpoint. Two routes to a piece's time, per the pre-registration:

  timestamped -- the video's judged text came from the accepted fresh route and
    stage A reproduced the frozen ASR text exactly, so chunk timestamps apply.
    A character position maps to a time by linear interpolation inside its
    chunk; repetition-collapsed videos keep their surviving units' offsets in
    the pre-collapse text, so the map still holds.

  proportional -- everything else (gate fell back to the dataset transcript, no
    usable audio, or stage A did not reproduce the frozen text). A unit at
    character midpoint p of a transcript of length L lands in slot
    floor(16 p / L).

Frame i covers [i D / 16, (i+1) D / 16) of the container duration D, which is
how frames_16 was extracted.

Pre-registration: docs/duplex/PREREG_interleaved_timeline_killtest.md.

Output: results/interleaved_timeline/<slug>/segments.json (never committed; it
carries transcript text) and a route census on stdout.

Usage:
  python scripts/duplex/interleaved_timeline_build.py --corpus mhclip_zh
"""

import argparse
import json
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from channel_restoration_gate import CLAUSE_DELIM, MAX_NGRAM, clause_key  # noqa: E402
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402
from interleaved_timeline_asr import CORPORA, load_jsonl  # noqa: E402

N_SLOTS = 16
ROTATION = 8


# ------------------------------------------------------- frozen split, traced
def split_units_offsets(text):
    """`channel_restoration_gate.split_units` with (start, end) char offsets.

    Returns [(key, piece, start, end)]. The pieces concatenate back to `text`.
    """
    parts = CLAUSE_DELIM.split(text)
    units = []
    buf, buf_start, pos = "", 0, 0
    for i in range(0, len(parts), 2):
        core = parts[i]
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        if buf == "":
            buf_start = pos
        piece = buf + core + delim
        pos += len(core) + len(delim)
        key = clause_key(core)
        if not key:
            buf = piece
            continue
        buf = ""
        units.append((key, piece, buf_start, pos))
    if buf:
        if units:
            k, p, s, _ = units[-1]
            units[-1] = (k, p + buf, s, pos)
        else:
            units.append((clause_key(buf), buf, buf_start, pos))
    return units


def collapse_repeats_traced(text, max_ngram=MAX_NGRAM):
    """`collapse_repeats` with survivors carrying their offsets in `text`."""
    units = split_units_offsets(text)
    changed = True
    while changed:
        changed = False
        out = []
        i, n = 0, len(units)
        while i < n:
            hit = False
            for k in range(1, max_ngram + 1):
                if i + 2 * k > n:
                    break
                block = [u[0] for u in units[i:i + k]]
                if block != [u[0] for u in units[i + k:i + 2 * k]]:
                    continue
                j = i + k
                while j + k <= n and block == [u[0] for u in units[j:j + k]]:
                    j += k
                out.extend(units[i:i + k])
                i = j
                hit = changed = True
                break
            if not hit:
                out.append(units[i])
                i += 1
        units = out
    return units


# ------------------------------------------------------------- time mapping --
def chunk_char_time_map(chunks, text, duration):
    """(fn: char index in `text` -> seconds) or None if the chunks do not fit.

    Chunk texts concatenate to `text` up to the strip the ASR stage applied, so
    a leading-whitespace offset is all that separates the two coordinate
    systems.
    """
    joined = "".join(c["text"] for c in chunks)
    if joined.strip() != text:
        return None
    lead = len(joined) - len(joined.lstrip())
    spans = []
    pos = 0
    prev_end = 0.0
    for c in chunks:
        n = len(c["text"])
        s = c.get("start")
        e = c.get("end")
        s = prev_end if s is None else float(s)
        e = duration if e is None else float(e)
        if e < s:
            e = s
        spans.append((pos, pos + n, s, e))
        prev_end = e
        pos += n
    if not spans:
        return None

    def to_time(p_text):
        q = p_text + lead
        for k, (a, b, s, e) in enumerate(spans):
            if q < b or k == len(spans) - 1:
                if b <= a:
                    return s
                frac = min(max((q - a) / (b - a), 0.0), 1.0)
                return s + frac * (e - s)
        return spans[-1][3]

    return to_time


def slot_of(t, duration):
    if duration is None or duration <= 0:
        return 0
    return min(max(int(t / duration * N_SLOTS), 0), N_SLOTS - 1)


def atomic_pieces(text, survivor_ranges, chunk_bounds=()):
    """Cut the surviving text as finely as the evidence allows.

    Cut positions are clause-unit boundaries, Whisper chunk boundaries where
    they exist, and the end of every whitespace run. Trailing whitespace stays
    attached to its piece, so the pieces concatenate back to the surviving text
    exactly. Without this the granularity would be the clause, and a Chinese
    Whisper transcript with no punctuation is a single clause however long it
    runs -- interleaving would have nothing to interleave.
    """
    cuts = set(chunk_bounds)
    for s, e in survivor_ranges:
        cuts.add(s)
        cuts.add(e)
    for m in re.finditer(r"\s+", text):
        cuts.add(m.end())
    pieces = []
    for s, e in survivor_ranges:
        pts = sorted({p for p in cuts if s <= p <= e} | {s, e})
        for a, b in zip(pts, pts[1:]):
            if b > a:
                pieces.append((a, b))
    return pieces


# --------------------------------------------------------------------- build --
def build_corpus(corpus, out_root):
    dataset, work_rel = CORPORA[corpus]
    work = os.path.join(ROOT, work_rel)
    out_dir = os.path.join(out_root, corpus)
    os.makedirs(out_dir, exist_ok=True)

    ann = load_annotations(dataset)
    split_ids = load_clean_split_ids(dataset, "test")
    seen, ids = set(), []
    for v in split_ids:
        if v not in seen:
            seen.add(v)
            ids.append(v)

    overrides = json.load(open(os.path.join(work, "c2_overrides.json")))
    meta = load_jsonl(os.path.join(work, "audio_meta.jsonl"))
    stamped = load_jsonl(os.path.join(out_dir, "timestamped_chunks.jsonl"))

    out = {}
    census = {"timestamped": 0, "proportional": 0, "empty": 0,
              "proportional_reason": {"dataset_transcript": 0,
                                      "no_timestamps": 0,
                                      "asr_text_mismatch": 0,
                                      "collapse_rebuild": 0,
                                      "no_chunk_map": 0,
                                      "no_duration": 0},
              "collapse_rebuild_mismatch": 0}
    for vid in ids:
        judged = overrides.get(vid)
        from_override = judged is not None
        if judged is None:
            judged = (ann.get(vid, {}).get("transcript") or "")
        if not judged.strip():
            out[vid] = {"slots": [""] * N_SLOTS, "route": "empty",
                        "n_units": 0, "n_chars": 0}
            census["empty"] += 1
            continue

        st = stamped.get(vid)
        duration = (meta.get(vid, {}).get("container_duration")
                    or meta.get(vid, {}).get("wav_duration"))

        route = None
        units = None
        if not from_override:
            census["proportional_reason"]["dataset_transcript"] += 1
        elif st is None:
            census["proportional_reason"]["no_timestamps"] += 1
        elif not st.get("matches_frozen_text"):
            census["proportional_reason"]["asr_text_mismatch"] += 1
        elif duration is None:
            census["proportional_reason"]["no_duration"] += 1
        else:
            raw = st["text"]
            survivors = collapse_repeats_traced(raw)
            rebuilt = "".join(u[1] for u in survivors).strip()
            if rebuilt != judged:
                census["collapse_rebuild_mismatch"] += 1
                census["proportional_reason"]["collapse_rebuild"] += 1
            else:
                to_time = chunk_char_time_map(st["chunks"], raw, duration)
                if to_time is None:
                    census["proportional_reason"]["no_chunk_map"] += 1
                else:
                    route = "timestamped"
                    joined = "".join(c["text"] for c in st["chunks"])
                    lead = len(joined) - len(joined.lstrip())
                    bounds, p = [], 0
                    for c in st["chunks"]:
                        p += len(c["text"])
                        bounds.append(max(p - lead, 0))
                    ranges = [(u[2], u[3]) for u in survivors]
                    units = [(raw[a:b],
                              slot_of(to_time((a + b) / 2.0), duration))
                             for a, b in atomic_pieces(raw, ranges, bounds)]

        if route is None:
            route = "proportional"
            ranges = [(u[2], u[3]) for u in split_units_offsets(judged)]
            L = max(len(judged), 1)
            units = [(judged[a:b],
                      min(int(((a + b) / 2.0) / L * N_SLOTS), N_SLOTS - 1))
                     for a, b in atomic_pieces(judged, ranges)]

        # Monotone repair: Whisper chunk spans occasionally overlap, which could
        # place a later clause in an earlier slot and scramble reading order.
        # A running maximum forbids that; it is a no-op wherever the timestamps
        # are already ordered.
        slots = [""] * N_SLOTS
        floor = 0
        for piece, s in units:
            s = max(s, floor)
            floor = s
            slots[s] += piece
        if "".join(slots).strip() != judged.strip():
            raise SystemExit(f"ABORT: {corpus}: slot concatenation does not "
                             "rebuild the judged transcript")
        census[route] += 1
        out[vid] = {"slots": slots, "route": route, "n_pieces": len(units),
                    "n_chars": len(judged)}

    with open(os.path.join(out_dir, "segments.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)
    census["n_videos"] = len(ids)
    census["n_nonempty_slots_mean"] = round(
        sum(sum(1 for s in r["slots"] if s.strip()) for r in out.values())
        / max(len(out), 1), 3)
    return census


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="all",
                    choices=sorted(CORPORA) + ["all"])
    ap.add_argument("--out-root", default=os.path.join(
        ROOT, "results", "interleaved_timeline"))
    args = ap.parse_args()

    corpora = sorted(CORPORA) if args.corpus == "all" else [args.corpus]
    report = {}
    for c in corpora:
        report[c] = build_corpus(c, args.out_root)
        print(f"[{c}] {json.dumps(report[c], ensure_ascii=False)}", flush=True)
    with open(os.path.join(args.out_root, "segmentation_census.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
