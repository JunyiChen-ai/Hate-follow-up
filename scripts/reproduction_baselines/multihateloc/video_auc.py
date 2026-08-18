"""Video-level ROC-AUC of a scores.jsonl, the column BASELINE_RESULTS.md adds.

Each video's frame score array is max-pooled to one number and those numbers
are ranked against the corpus video label, over the same cohort the frame
evaluation covers (the gold cohort, so the two columns of a row describe one
set of videos). Rank AUC via the Mann-Whitney identity with mid-ranks for
ties; no scikit-learn.

  python video_auc.py --corpus hatemm --scores .../scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..")))

from hate_common import data as hdata  # noqa: E402


def roc_auc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=list(hdata.CORPORA))
    ap.add_argument("--scores", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    records = hdata.load_scores_jsonl(args.scores)
    gt = hdata.gt_arrays(args.corpus, args.split)
    labels = hdata.load_labels(args.corpus)
    cohort = sorted(set(records) & set(gt))
    branches = sorted(records[cohort[0]].keys())
    out = {}
    for b in branches:
        y = [labels[v] for v in cohort]
        s = [float(np.max(records[v][b])) for v in cohort]
        out[b] = roc_auc(y, s)
        print("%s / %s / %s  video AUC (max-pool)  %.4f  (n=%d, %d hateful)"
              % (args.corpus, args.split, b, out[b], len(cohort), int(sum(y))))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"corpus": args.corpus, "n_videos": len(cohort),
                       "video_auc_maxpool": out}, fh, indent=1)
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
