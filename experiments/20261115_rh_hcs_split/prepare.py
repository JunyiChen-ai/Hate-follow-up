#!/usr/bin/env python3
"""Inputs for the baseline reruns on the HateClipSeg split of the Retrieval-hate repository (README.md). CPU, uoa-lab1.

    python experiments/20261115_rh_hcs_split/prepare.py            # HateVideo env; writes data/rh_hcs_split/

Source of the split: /home/jehc223/Retrieval-hate/runs/20261115_hf_baselines/hcs_split/split.json, written there by
experiments/20261115_hf_baselines/export_hcs_split.py from hier_evidence_common.load_fixed_cohort('hateclipseg'):
train / val / test ids and the video-level labels of the train and val ids. No test label and no frame label is read.

Written under data/rh_hcs_split/ (PROVENANCE.md there):
  weaksup/splits/hateclipseg_{train,val,test}.txt, weaksup/labels/hateclipseg.json (train + val only),
  weaksup/durations/hateclipseg.json                              -> WEAKSUP_ALT_DIR (weaksup_common/common.py)
  video_splits/HateClipSeg.json                                    -> DETWIN_SPLIT_DIR (detwin/common.py)
  cohort/<DS>.txt, cohort/<DS>_manifest.jsonl                      -> LF_ALT_COHORT_DIR (src/lf_baselines.py);
      HateClipSeg = the new 79 test videos; HateMM and DeHate = the unchanged cohorts and manifests, listed here only so
      that runs made with this directory stop after predictions.jsonl
  sage_frames16/, sage_frames16_win8/, clara_raw/, clara_emb/, clara_rationale   -> SAGE_DATA / CLARA_DATA: links to
      the existing per-video caches of videos whose role is unchanged (train or val before and now: whole-video
      inputs; test before and now: window inputs); the prep scripts fill in the rest.
Durations: ffprobe format duration of ~/data/HateClipSeg/videos/<id>.<ext>, the rule that reproduces the durations of
data/omsl_v6_inputs/manifests/all_test.jsonl (checked here on the test videos both splits share).
The label comparison with data/weaksup_video_splits/HateClipSeg.json is printed and written to split_check.json.
"""
from __future__ import annotations

import datetime
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path("/home/jehc223/Retrieval-hate/runs/20261115_hf_baselines/hcs_split/split.json")
OUT = REPO / "data" / "rh_hcs_split"
DATA = REPO / "data"
VIDEOS = Path.home() / "data" / "HateClipSeg" / "videos"
DS, KEY = "HateClipSeg", "hateclipseg"


def video_file(v):
    for ext in ("mp4", "webm", "mkv"):
        p = VIDEOS / f"{v}.{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(v)


def ffprobe_duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not (dst.exists() or dst.is_symlink()):
        os.symlink(src, dst)


def main():
    s = json.loads(SRC.read_text())
    tr, va, te = s["train"], s["val"], s["test"]
    lab = {v: int(x) for v, x in s["train_val_labels"].items()}
    assert set(lab) == set(tr) | set(va) and not (set(tr) | set(va)) & set(te) and not set(tr) & set(va)
    old = json.loads((DATA / "weaksup_video_splits" / f"{DS}.json").read_text())
    old_lab = {x["video_id"]: int(x["label"]) for k in ("train", "val") for x in old[k]}
    old_test = {x["video_id"]: float(x["duration"]) for x in old["test"]}
    both = sorted(set(lab) & set(old_lab))
    disagree = [(v, lab[v], old_lab[v]) for v in both if lab[v] != old_lab[v]]

    # durations of the new test videos
    dur = {v: ffprobe_duration(video_file(v)) for v in te}
    shared = sorted(set(te) & set(old_test))
    dur_diff = max((abs(dur[v] - old_test[v]) for v in shared), default=0.0)
    if dur_diff > 1e-6:
        raise SystemExit(f"ffprobe durations differ from the existing manifest by up to {dur_diff}")

    w = OUT / "weaksup"
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        p = w / "splits" / f"{KEY}_{name}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(ids) + "\n")
    for sub, obj in (("labels", lab), ("durations", dur)):
        p = w / sub / f"{KEY}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1) + "\n")

    p = OUT / "video_splits" / f"{DS}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"train": [{"video_id": v, "label": lab[v]} for v in tr],
                             "val": [{"video_id": v, "label": lab[v]} for v in va],
                             "test": [{"video_id": v, "duration": dur[v]} for v in te]}) + "\n")

    c = OUT / "cohort"
    c.mkdir(parents=True, exist_ok=True)
    (c / f"{DS}.txt").write_text("\n".join(sorted(te)) + "\n")
    with (c / f"{DS}_manifest.jsonl").open("w") as fh:
        for v in sorted(te):
            fh.write(json.dumps({"dataset": DS, "video_id": v, "duration": dur[v],
                                 "video_path": str(video_file(v))}) + "\n")
    sys.path.insert(0, str(REPO))
    from src import lf_baselines as L
    for ds in ("HateMM", "DeHate"):                      # unchanged cohorts, copied
        ids = [x.strip() for x in open(os.path.join(L.COHORT_DIR, ds + ".txt")) if x.strip()]
        assert len(ids) == L.EXPECTED[ds]
        (c / f"{ds}.txt").write_text("\n".join(ids) + "\n")
        n = 0
        with open(L.MANIFESTS[ds]) as fi, (c / f"{ds}_manifest.jsonl").open("w") as fo:
            for line in fi:
                if json.loads(line).get("dataset") == ds:
                    fo.write(line)
                    n += 1
        assert n >= len(ids), (ds, n)

    # links to existing per-video caches whose content does not depend on the split role
    keep_tv = sorted((set(tr) | set(va)) & set(old_lab))        # whole-video inputs exist
    keep_te = sorted(set(te) & set(old_test))                   # window inputs exist
    n_link = {"sage_frames16": 0, "sage_frames16_win8": 0, "clara_raw": 0, "clara_emb": 0}
    for v in keep_tv:
        src = DATA / "sage_frames16" / DS / f"{v}.npy"
        if src.is_file():
            link(src, OUT / "sage_frames16" / DS / f"{v}.npy")
            n_link["sage_frames16"] += 1
        src = DATA / "clara_emb" / DS / "video_embeddings" / f"{v}.pt"
        if src.is_file():
            link(src, OUT / "clara_emb" / DS / "video_embeddings" / f"{v}.pt")
            n_link["clara_emb"] += 1
    for v in keep_te:
        for ext in ("npy", "json"):
            src = DATA / "sage_frames16_win8" / DS / f"{v}.{ext}"
            if src.is_file():
                link(src, OUT / "sage_frames16_win8" / DS / f"{v}.{ext}")
                n_link["sage_frames16_win8"] += ext == "npy"
        for src in sorted((DATA / "clara_emb" / DS / "video_embeddings").glob(f"{v}__w*.pt")):
            link(src, OUT / "clara_emb" / DS / "video_embeddings" / src.name)
            n_link["clara_emb"] += 1
    for v in keep_tv + keep_te:
        src = DATA / "clara_raw" / DS / v
        if (src / "_PREP_DONE.json").is_file():
            link(src, OUT / "clara_raw" / DS / v)
            n_link["clara_raw"] += 1
    link(DATA / "clara_rationale", OUT / "clara_rationale")
    rat_missing = [v for v in tr + va + te
                   if not (DATA / "clara_rationale" / DS / "Qwen" / f"{v}_rationale.json").is_file()]

    check = {"date": datetime.date.today().isoformat(), "host": socket.gethostname(),
             "source": str(SRC), "n": {"train": len(tr), "val": len(va), "test": len(te)},
             "hateful": {"train": sum(lab[v] for v in tr), "val": sum(lab[v] for v in va)},
             "labels_compared_with_weaksup_video_splits": len(both), "label_disagreements": disagree,
             "train_val_ids_without_label_in_old_split": len(set(lab) - set(old_lab)),
             "test_shared_with_old_test": len(shared), "max_duration_diff_on_shared_test": dur_diff,
             "test_seconds": sum(dur.values()), "links": n_link, "rationale_missing": rat_missing}
    (OUT / "split_check.json").write_text(json.dumps(check, indent=1) + "\n")
    (OUT / "PROVENANCE.md").write_text(
        "# data/rh_hcs_split - provenance\n\n"
        f"Built {check['date']} on {check['host']} by `experiments/20261115_rh_hcs_split/prepare.py` "
        "(code version: the commit that adds that directory, 2026-11-15).\n\n"
        f"Upstream: `{SRC}` (Retrieval-hate `load_fixed_cohort('hateclipseg')`: train 251 / val 63 / test 79 and the "
        "video-level labels of train and val), `~/data/HateClipSeg/videos` (ffprobe durations), and for HateMM / DeHate "
        "the unchanged cohort lists and test manifests of this repository.\n\n"
        "`sage_frames16*`, `clara_raw`, `clara_emb`, `clara_rationale` hold symbolic links to the existing caches in "
        "`data/` for videos whose role (train/val or test) is the same in both splits; the other entries are written by "
        "`sage_run.py prep`, `clara_prep.py`, `clara_ocr.py`, `clara_embed.py` run with "
        "`experiments/20261115_rh_hcs_split/env.sh` (same scripts, same settings).\n")
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
