"""Cross-benchmark stage C: the frozen C2 degeneracy gate, per dataset.

Nothing about the gate is re-decided here. `collapse_repeats` (clause-level
repetition collapse, max_ngram 8) and the two rejection bounds
(`VAD_BOUND` 0.05, `GZIP_BOUND` 7.0) are imported from
`scripts/duplex/channel_restoration_gate.py`, which is unmodified. This module
only replaces the hard-wired ImpliHateVid id list with the requested dataset's
`train_clean` split.

A fresh transcript is discarded, and that video falls back to its dataset
transcript, if and only if the silero-VAD speech fraction is below 0.05 AND the
gzip compression ratio of the raw fresh transcript exceeds 7. Videos with no
usable audio are recorded as `no_fresh_pass` and also fall back, so a dataset
with no source media degrades to the dataset transcript everywhere rather than
failing.

Output: c2_overrides.json (video_id -> gated fresh transcript) and
gate_outcomes.jsonl.

Usage:
  python scripts/duplex/crossbench_gate.py --dataset HateMM \
      --out-dir results/crossbench/hatemm
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from channel_restoration_gate import (  # noqa: E402
    GZIP_BOUND, MAX_NGRAM, VAD_BOUND, collapse_repeats,
)
from data_utils import load_clean_split_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ids = load_clean_split_ids(args.dataset, args.split)
    asr_path = os.path.join(args.out_dir, "fresh_transcripts.jsonl")
    asr = {}
    if os.path.exists(asr_path):
        with open(asr_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
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
        outcomes.append({
            "video_id": vid, "outcome": "rejected" if reject else "accepted",
            "vad_speech_frac": vad, "gzip_ratio_raw": ratio,
            "raw_chars": len(raw), "collapsed_chars": len(collapsed),
            "collapse_shrink": (round(1 - len(collapsed) / len(raw), 6)
                                if raw else None),
            "old_chars": r["old_chars"]})
        if reject:
            counts["rejected"] += 1
        else:
            counts["accepted"] += 1
            overrides[vid] = collapsed

    with open(os.path.join(args.out_dir, "c2_overrides.json"), "w") as f:
        json.dump(overrides, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "gate_outcomes.jsonl"), "w") as f:
        for o in outcomes:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"gate [{args.dataset}] bounds: vad<{VAD_BOUND} AND gzip>{GZIP_BOUND}, "
          f"max_ngram={MAX_NGRAM}")
    print(json.dumps(counts, indent=1))
    print(f"override map: {len(overrides)} ids")


if __name__ == "__main__":
    main()
