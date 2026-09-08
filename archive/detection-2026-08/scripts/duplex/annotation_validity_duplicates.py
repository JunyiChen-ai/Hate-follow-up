"""Model-independent part of the annotation-validity audit.

Group the 394 scored HateClipSeg videos by exact byte equality of their fresh
transcript, keep the nonempty groups of size >= 2, and report whether the
shipped video-level labels agree inside each group. Two videos whose audio
transcribes to the identical byte string are either the same upload twice or
the same speech re-cut; a shipped label that differs between them is an
annotation inconsistency that no model is involved in.

Reads only the transcript file and the shipped annotation csv. No score file.
Writes results/annotation_validity/duplicates.json.
"""

import ast
import csv
import json
import os
from collections import defaultdict

ROOT = "/home/jehc223/Hate-follow-up"
DATA = "/home/jehc223/data"
OUT = os.path.join(ROOT, "results", "annotation_validity")
IDX = ["normal", "hateful", "insulting", "sexual", "violence", "harm"]
OFFENSIVE = set(IDX[1:])


def main():
    labs = {}
    p = os.path.join(ROOT, "idea-stage", "pilots", "b1_coverage_audit",
                     "data", "video_level_annotation.csv")
    with open(p) as f:
        for row in csv.DictReader(f):
            labs[row["Video Id"].strip()] = ast.literal_eval(row["Video-Level Label"])
    clean = {l.strip() for l in
             open(os.path.join(DATA, "HateClipSeg", "splits", "test_clean.csv"))
             if l.strip()}

    groups = defaultdict(list)
    n_rows = n_empty = 0
    for line in open(os.path.join(ROOT, "results", "hateclipseg",
                                  "fresh_transcripts.jsonl")):
        r = json.loads(line)
        if r["video_id"] not in clean:
            continue
        n_rows += 1
        t = (r.get("fresh_text") or "")
        if not t.strip():
            n_empty += 1
            continue
        groups[t].append(r["video_id"])

    dups = [v for v in groups.values() if len(v) > 1]
    dups.sort(key=len, reverse=True)

    rows = []
    for k, vids in enumerate(dups, 1):
        union = [int(bool(set(labs[v]) & OFFENSIVE)) for v in vids]
        strict = [int("hateful" in labs[v]) for v in vids]
        exact = [tuple(sorted(labs[v])) for v in vids]
        rows.append({
            "group": f"DUP-{k:02d}",
            "size": len(vids),
            "video_ids": vids,
            "union_labels": union,
            "union_agree": len(set(union)) == 1,
            "strict_labels": strict,
            "strict_agree": len(set(strict)) == 1,
            "exact_multilabel_agree": len(set(exact)) == 1,
            "transcript_chars": len(next(t for t, v in groups.items() if v == vids)),
        })

    n_groups = len(rows)
    n_videos = sum(r["size"] for r in rows)
    res = {
        "corpus": "HateClipSeg",
        "n_scored_videos": n_rows,
        "n_empty_transcripts": n_empty,
        "n_duplicate_groups": n_groups,
        "n_videos_in_duplicate_groups": n_videos,
        "n_groups_union_disagree": sum(1 for r in rows if not r["union_agree"]),
        "n_groups_strict_disagree": sum(1 for r in rows if not r["strict_agree"]),
        "n_groups_exact_multilabel_disagree":
            sum(1 for r in rows if not r["exact_multilabel_agree"]),
        "groups": rows,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "duplicates.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "groups"}, indent=1))
    for r in rows:
        print(f"  {r['group']} n={r['size']} chars={r['transcript_chars']} "
              f"union={r['union_labels']} strict={r['strict_labels']} "
              f"exact_agree={r['exact_multilabel_agree']}")


if __name__ == "__main__":
    main()
