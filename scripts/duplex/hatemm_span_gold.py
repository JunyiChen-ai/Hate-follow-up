"""HateMM hate-span gold: fetch, parse and describe the upstream annotation.

Status: this is a DESCRIPTIVE utility for a descriptive diagnostic. It has no
pre-registration, no decision bar and no interpretation grid.

What it provides. HateMM ships one extra annotation layer that the local
`annotation(new).json` copy dropped: for every hate video, the time span (or
spans) the annotators marked as the hateful part. The upstream file is
`HateMM_annotation.csv`, published openly with the dataset on Zenodo record
7799469 (the file the GitHub repo hate-alert/HateMM points at), four columns:

    video_file_name, label, hate_snippet, target

`hate_snippet` is a Python-literal list of [start, end] wall-clock strings, e.g.
"[['00:00:34', '00:01:34']]", empty for non-hate videos. This module downloads
that file once into <data>/HateMM/upstream_spans/, parses the snippets into
seconds, and hands the evaluation-side gold to the isolated-chunk diagnostic.

Nothing here is used as model input: the spans only place chunks on the
timeline after scoring.

Usage:
  python scripts/duplex/hatemm_span_gold.py            # download if absent, report
  python scripts/duplex/hatemm_span_gold.py --no-fetch
"""

import argparse
import ast
import csv
import json
import os
import sys
import urllib.request

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "our_method"))

DATA_ROOT = os.path.join(os.path.expanduser("~"), "data", "HateMM")
SPAN_DIR = os.path.join(DATA_ROOT, "upstream_spans")
SPAN_CSV = os.path.join(SPAN_DIR, "HateMM_annotation.csv")
SPAN_URL = ("https://zenodo.org/api/records/7799469/files/"
            "HateMM_annotation.csv/content")
TEST_SPLIT = os.path.join(DATA_ROOT, "splits", "test_clean.csv")


def fetch(force=False):
    """Download the upstream annotation CSV if it is not already on disk."""
    if os.path.exists(SPAN_CSV) and not force:
        return SPAN_CSV
    os.makedirs(SPAN_DIR, exist_ok=True)
    urllib.request.urlretrieve(SPAN_URL, SPAN_CSV)
    return SPAN_CSV


def parse_clock(t):
    """'HH:MM:SS' (or 'MM:SS', or 'SS') -> seconds. None if unparseable."""
    parts = str(t).strip().split(":")
    if not parts or len(parts) > 3:
        return None
    try:
        vals = [int(float(p)) for p in parts]
    except ValueError:
        return None
    while len(vals) < 3:
        vals = [0] + vals
    return vals[0] * 3600 + vals[1] * 60 + vals[2]


def parse_snippet(cell):
    """Parse one `hate_snippet` cell into [(start_s, end_s)] plus a problem list."""
    cell = (cell or "").strip()
    if not cell:
        return [], []
    try:
        raw = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return [], ["unparseable_cell"]
    spans, bad = [], []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            bad.append("malformed_pair:%r" % (item,))
            continue
        s, e = parse_clock(item[0]), parse_clock(item[1])
        if s is None or e is None:
            bad.append("unparseable_time:%r" % (item,))
            continue
        if e <= s:
            bad.append("non_positive_duration:%r" % (item,))
            continue
        spans.append((float(s), float(e)))
    return merge(spans), bad


def merge(spans):
    """Union of possibly overlapping spans, sorted."""
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return [(a, b) for a, b in out]


def read_split(path=TEST_SPLIT):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_span_gold(ids=None, fetch_if_missing=True):
    """{video_id: [(start_s, end_s), ...]} for hate videos, plus a problem dict.

    Only rows labelled Hate carry spans. Restricted to `ids` when given.
    """
    if fetch_if_missing:
        fetch()
    keep = set(ids) if ids is not None else None
    gold, problems, labels = {}, {}, {}
    with open(SPAN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            vid = row["video_file_name"].rsplit(".", 1)[0]
            if keep is not None and vid not in keep:
                continue
            labels[vid] = row["label"]
            if row["label"].strip().lower() != "hate":
                continue
            spans, bad = parse_snippet(row["hate_snippet"])
            gold[vid] = spans
            if bad:
                problems[vid] = bad
    return gold, problems, labels


def describe(gold, labels, ids, meta=None):
    hate_ids = [v for v in ids if v.startswith("hate_video_")]
    with_spans = [v for v in hate_ids if gold.get(v)]
    n_spans = [len(gold[v]) for v in with_spans]
    durs = [e - s for v in with_spans for s, e in gold[v]]
    tot = {v: sum(e - s for s, e in gold[v]) for v in with_spans}
    cov = []
    if meta:
        for v in with_spans:
            d = (meta.get(v) or {}).get("wav_duration")
            if d:
                cov.append(min(tot[v] / float(d), 1.0))

    def stat(a):
        if not a:
            return {"n": 0}
        a = sorted(a)
        n = len(a)
        return {"n": n, "mean": sum(a) / n, "median": a[n // 2],
                "min": a[0], "max": a[-1],
                "q25": a[int(0.25 * n)], "q75": a[int(0.75 * n)]}

    return {
        "split_videos": len(ids),
        "hate_videos_in_split": len(hate_ids),
        "hate_videos_with_spans": len(with_spans),
        "hate_videos_without_spans": len(hate_ids) - len(with_spans),
        "csv_label_matches_id_prefix": sum(
            (labels.get(v, "").strip().lower() == "hate")
            == v.startswith("hate_video_") for v in ids if v in labels),
        "csv_rows_seen_for_split": len([v for v in ids if v in labels]),
        "spans_per_video": stat(n_spans),
        "span_duration_sec": stat(durs),
        "hate_seconds_per_video": stat(list(tot.values())),
        "span_coverage_fraction_of_audio": stat(cov),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "results", "hatemm_localization", "span_gold.json"))
    args = ap.parse_args()

    ids = read_split()
    gold, problems, labels = load_span_gold(ids, fetch_if_missing=not args.no_fetch)

    meta_path = os.path.join(PROJECT_ROOT, "results", "testruns", "hatemm",
                             "audio_meta.jsonl")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    meta[r["video_id"]] = r

    rep = describe(gold, labels, ids, meta)
    rep["parse_problems"] = problems
    rep["source"] = {"url": SPAN_URL, "path": SPAN_CSV}
    rep["spans"] = {v: gold[v] for v in sorted(gold)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    head = {k: v for k, v in rep.items() if k != "spans"}
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
