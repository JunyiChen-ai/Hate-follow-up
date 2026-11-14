#!/usr/bin/env python3
"""Carry over per-video intermediate outputs of the existing HateClipSeg runs for the test videos both cohorts share
(README.md). CPU, uoa-lab1, after `source experiments/20261115_rh_hcs_split/env.sh`.

    python experiments/20261115_rh_hcs_split/reuse_existing.py prism      # window embeddings of prism_av
    python experiments/20261115_rh_hcs_split/reuse_existing.py eventvad   # event boundaries of the EventVAD segmentation

Both stages are per video and resumable in the original scripts (a video with a row in extract_meta.jsonl /
events.jsonl is skipped), so the copied rows are exactly what the existing runs computed for those videos and the
reruns compute only the other videos. Only error-free rows are copied. Nothing is copied twice.
"""
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src import lf_baselines as L            # noqa: E402

DS = "HateClipSeg"
OLD = os.path.join(REPO, "runs", "20261008_baselines")
NEW = os.environ["LF_BASELINES_RUNS"]


def rows(path):
    out = {}
    if os.path.isfile(path):
        for line in open(path):
            r = json.loads(line)
            if not r.get("error"):
                out[r["video_id"]] = line if line.endswith("\n") else line + "\n"
    return out


def carry(sub, files, per_video=None):
    assert L.alt_cohort(DS), "source env.sh first"
    ids = L.cohort(DS)
    src, dst = os.path.join(OLD, sub, DS), os.path.join(NEW, sub, DS)
    os.makedirs(dst, exist_ok=True)
    main = rows(os.path.join(src, files[0]))
    have = rows(os.path.join(dst, files[0]))
    take = [v for v in ids if v in main and v not in have]
    for name in files:
        old = rows(os.path.join(src, name))
        with open(os.path.join(dst, name), "a") as fh:
            for v in take:
                fh.write(old[v])
    if per_video:
        os.makedirs(os.path.join(dst, os.path.dirname(per_video)), exist_ok=True)
        for v in take:
            shutil.copy2(os.path.join(src, per_video % v), os.path.join(dst, per_video % v))
    print("%s: %d of %d cohort videos carried over from %s (%d were already there)" % (
        sub, len(take), len(ids), src, len([v for v in ids if v in have])))


if __name__ == "__main__":
    if sys.argv[1:] == ["prism"]:
        carry("prism_av", ["extract_meta.jsonl", "window_texts.jsonl"], "features/%s.npy")
    elif sys.argv[1:] == ["eventvad"]:
        carry("eventvad", ["events.jsonl"])
    else:
        raise SystemExit(__doc__)
