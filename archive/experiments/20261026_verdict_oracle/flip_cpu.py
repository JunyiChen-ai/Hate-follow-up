"""Oracle diagnostic (rule 10, reads test GT): what frame-level metrics r6_bma would reach if the video verdict were
right more often, with the window reads unchanged.

The r6 frame score is K + centred rank, with K = z_video + mean window z (raw key). A "wrong" video is one whose
verdict sign disagrees with the video's GT (any positive 4 fps frame). Arms:
  base           unchanged r6 predictions (must reproduce the reference metrics);
  flip_all       wrong videos get -z_video in K (same confidence, corrected sign): score -= 2 z_video;
  flip_fp / fn   only the false-positive / false-negative videos are flipped;
  flip_pXX_sY    a random XX % of the wrong videos are flipped (seed Y);
  perfect_split  every GT-negative video is shifted below every GT-positive video (headroom A1).
Window reads, M2 and the within-video rank are untouched, so within can change only through videos whose frames are
not all of one class. Output: <out>/predictions.jsonl, then src/eval/evaluate_four_datasets.py on it."""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--reads", required=True)
ap.add_argument("--gt-dir", default="data/gt_4fps")
ap.add_argument("--out", required=True)
a = ap.parse_args()
zv = {}
for line in open(a.reads):
    r = json.loads(line)
    zv[(r["dataset"], r["video_id"])] = float(r["extra"]["z_video"])
rows = [json.loads(l) for l in open(a.pred)]
gt = {}
for ds in {r["dataset"] for r in rows}:
    g = np.load(Path(a.gt_dir) / f"{ds}.npz", allow_pickle=True)
    gt.update({(ds, str(v)): bool(np.asarray(y).any()) for v, s, y in zip(g["video_ids"], g["split"], g["y4"]) if s == "test"})
rows = [r for r in rows if (r["dataset"], r["video_id"]) in gt]
wrong = {k for k in ((r["dataset"], r["video_id"]) for r in rows) if (zv[k] > 0) != gt[k]}
fp = {k for k in wrong if not gt[k]}
fn = wrong - fp
arms = {"base": set(), "flip_all": wrong, "flip_fp": fp, "flip_fn": fn}
rng = np.random.default_rng(0)
wl = sorted(wrong)
for frac in (25, 50):
    for s in range(5):
        pick = rng.choice(len(wl), size=round(len(wl) * frac / 100), replace=False)
        arms[f"flip_p{frac}_s{s}"] = {wl[i] for i in pick}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
summary = {"n_videos": len(rows), "n_wrong": len(wrong), "n_fp": len(fp), "n_fn": len(fn),
           "per_dataset": {ds: {"wrong": sum(k[0] == ds for k in wrong), "fp": sum(k[0] == ds for k in fp),
                                "fn": sum(k[0] == ds for k in fn), "videos": sum(r["dataset"] == ds for r in rows)}
                           for ds in sorted({r["dataset"] for r in rows})}}
with open(out / "predictions.jsonl", "w") as f:
    for name, flip in arms.items():
        for r in rows:
            k = (r["dataset"], r["video_id"])
            s = np.asarray(r["score_curve"], float)
            if k in flip:
                s = s - 2 * zv[k]
            f.write(json.dumps({**r, "method": f"oracle__{name}", "score_curve": s.tolist(), "intervals": []}) + "\n")
        print(name, len(flip), flush=True)
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        lo_pos = min(min(r["score_curve"]) for r in sub if gt[(ds, r["video_id"])])
        hi_neg = max(max(r["score_curve"]) for r in sub if not gt[(ds, r["video_id"])])
        c = max(0.0, hi_neg - lo_pos + 1.0)
        for r in sub:
            s = np.asarray(r["score_curve"], float) - (0.0 if gt[(ds, r["video_id"])] else c)
            f.write(json.dumps({**r, "method": "oracle__perfect_split", "score_curve": s.tolist(), "intervals": []}) + "\n")
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print(json.dumps(summary))
