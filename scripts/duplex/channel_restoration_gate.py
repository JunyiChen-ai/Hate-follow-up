"""Channel restoration, stage C: the frozen C2 degeneracy gate.

Two deterministic parts, both fixed by the pre-registration from Step-0
measurements and neither tuned afterwards.

1. Repetition collapse. Consecutive repeated clauses and consecutive repeated
   n-grams of clauses are collapsed to a single instance, where a clause
   boundary is sentence-terminal punctuation or a comma. The collapse is
   order-preserving, keeps the first instance verbatim, and applies to every
   fresh transcript whether or not it is later rejected.
2. Rejection. A fresh transcript is discarded, and that video falls back to its
   dataset transcript, if and only if the silero-VAD speech fraction is below
   0.05 AND the gzip compression ratio of the RAW fresh transcript exceeds 7.

Output: c2_overrides.json, the video_id -> transcript map the judge consumes
under C2. Videos that fall back are absent from the map, which leaves the
dataset transcript in place. Per-video gate outcomes go to gate_outcomes.jsonl.

Pre-registration: docs/duplex/PREREG_channel_restoration.md.
"""

import gzip
import json
import os
import re
import sys
import unicodedata

VAD_BOUND = 0.05
GZIP_BOUND = 7.0
MAX_NGRAM = 8

# Sentence-terminal punctuation or a comma, in the Latin and CJK forms that
# Whisper emits, optionally repeated and followed by whitespace.
CLAUSE_DELIM = re.compile(r"([.!?…。！？,，、]+\s*|\n+)")


def clause_key(text):
    """Match key for two clauses: case-folded, punctuation-stripped, respaced."""
    s = unicodedata.normalize("NFKC", text).casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_units(text):
    """[(key, verbatim_text_with_its_delimiter)], order-preserving.

    Fragments whose key is empty carry no content and are glued onto the
    preceding unit so that punctuation runs never become match targets.
    """
    parts = CLAUSE_DELIM.split(text)
    units = []
    buf = ""
    for i in range(0, len(parts), 2):
        core = parts[i]
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        piece = buf + core + delim
        key = clause_key(core)
        if not key:
            buf = piece
            continue
        buf = ""
        units.append((key, piece))
    if buf:
        if units:
            units[-1] = (units[-1][0], units[-1][1] + buf)
        else:
            units.append((clause_key(buf), buf))
    return units


def collapse_repeats(text, max_ngram=MAX_NGRAM):
    """Collapse consecutive repeated clause n-grams to a single instance."""
    units = split_units(text)
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
    return "".join(u[1] for u in units).strip()


def gzip_ratio(text):
    raw = (text or "").encode("utf-8")
    if not raw:
        return None
    return len(raw) / len(gzip.compress(raw, 9))


def main():
    ids_path, out_dir = sys.argv[1], sys.argv[2]
    ids = json.load(open(ids_path))["dismissed_all"]
    asr = {}
    with open(os.path.join(out_dir, "fresh_transcripts.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            asr[r["video_id"]] = r

    overrides, outcomes = {}, []
    counts = {"total": len(ids), "no_fresh_pass": 0, "asr_error": 0,
              "rejected": 0, "accepted": 0}
    for vid in ids:
        r = asr.get(vid)
        if r is None or r.get("error"):
            counts["no_fresh_pass" if r is None else "asr_error"] += 1
            outcomes.append({"video_id": vid, "outcome": "no_fresh_pass",
                             "reason": "no usable audio" if r is None else "asr_error"})
            continue
        raw = r["fresh_text"]
        collapsed = collapse_repeats(raw)
        ratio = r.get("gzip_ratio_raw")
        vad = r.get("vad_speech_frac")
        reject = (vad is not None and vad < VAD_BOUND
                  and ratio is not None and ratio > GZIP_BOUND)
        o = {"video_id": vid, "outcome": "rejected" if reject else "accepted",
             "vad_speech_frac": vad, "gzip_ratio_raw": ratio,
             "raw_chars": len(raw), "collapsed_chars": len(collapsed),
             "collapse_shrink": (round(1 - len(collapsed) / len(raw), 6) if raw else None),
             "old_chars": r["old_chars"]}
        outcomes.append(o)
        if reject:
            counts["rejected"] += 1
        else:
            counts["accepted"] += 1
            overrides[vid] = collapsed

    with open(os.path.join(out_dir, "c2_overrides.json"), "w") as f:
        json.dump(overrides, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "gate_outcomes.jsonl"), "w") as f:
        for o in outcomes:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(json.dumps(counts, indent=1))
    print(f"override map: {len(overrides)} ids")


if __name__ == "__main__":
    main()
