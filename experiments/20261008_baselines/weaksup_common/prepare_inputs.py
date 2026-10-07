#!/usr/bin/env python3
"""Build data/weaksup_1fps/: splits, train/val video labels and the 1 fps feature copies (CPU, uoa-lab1).

Splits (test = the exact cohort of hate_query.md §3):
  hatemm       train 744 / val 109 from Retrieval-hate results/reproduction/splits/hatemm_{train,val}.txt
               (= ~/data/HateMM/splits/{train_clean,validation_clean}.csv); test = 215 cohort ids
  hateclipseg  train 237 / val 39 from Retrieval-hate data/gt/HateClipSeg/p11_split.json; test = 118 cohort ids
  dehate       train 4680 / val 668 from Retrieval-hate results/reproduction/splits/dehate_{train,val}.txt
               (= the `Split` column of ~/data/DeHate/DeHate_labels.csv, checked below); test = 1151 cohort ids
Video labels (train and val ids only):
  hatemm       HateMM_annotation.csv `label` (Hate -> 1), via scripts/reproduction_baselines/hate_common
  hateclipseg  segment_level_annotation.csv, 1 iff any segment is positive on dimensions 1..5 (the offensive
               union that also defines the HCS GT), via hate_common; cross-checked against p11's gold_segments.json
  dehate       DeHate_labels.csv `Hate`
Features copied (only ids in train, val or the cohort):
  clip_b16_1fps, bert_sentence_1fps: all three corpora; vggish_1s, vit_b16_imagenet_1fps: hatemm, hateclipseg
  sources: hatemm runs/legacy_1fps/lab2/reproduction/features (complete, 1068 ids),
           hateclipseg runs/legacy_1fps/lab1/reproduction/features (lab2 lacks val id yt_DnrYK1FXKgk),
           dehate Retrieval-hate results/reproduction/features (read-only, this machine)

    python experiments/20261008_baselines/weaksup_common/prepare_inputs.py
"""
from __future__ import annotations

import ast
import csv
import datetime
import json
import shutil
import socket
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "scripts" / "reproduction_baselines"))
from hate_common import data as hdata  # noqa: E402

RH = Path("/home/jehc223/Retrieval-hate")
RH_SPLITS = RH / "results/reproduction/splits"
P11 = RH / "data/gt/HateClipSeg/p11_split.json"
P11_GOLD = RH / "data/gt/HateClipSeg/gold_segments.json"
DEHATE_CSV = Path("/home/jehc223/data/DeHate/DeHate_labels.csv")
COHORT_PRED = C.REPO / "runs/20260926_twolevel/final_rawkey/predictions.jsonl"
FEATURE_SRC = {"hatemm": C.REPO / "runs/legacy_1fps/lab2/reproduction/features",
               "hateclipseg": C.REPO / "runs/legacy_1fps/lab1/reproduction/features",
               "dehate": RH / "results/reproduction/features"}
FEATURES = {"clip_b16_1fps": C.CORPORA, "bert_sentence_1fps": C.CORPORA,
            "vggish_1s": ("hatemm", "hateclipseg"), "vit_b16_imagenet_1fps": ("hatemm", "hateclipseg")}


def cohorts():
    out = {}
    with COHORT_PRED.open() as fh:
        for line in fh:
            row = json.loads(line)
            out.setdefault(C.CORPUS[row["dataset"]], set()).add(row["video_id"])
    z = np.load(C.GT_DIR / "DeHate.npz", allow_pickle=True)
    out["dehate"] = {str(v) for v, s in zip(z["video_ids"], z["split"]) if str(s) == "test"}
    return {k: sorted(v) for k, v in out.items()}


def dehate_rows():
    csv.field_size_limit(1 << 30)
    with DEHATE_CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    log = C.RunLog(C.RUNS / "weaksup_inputs" / "run.log", mode="w")
    coh = cohorts()
    p11 = json.loads(P11.read_text())
    splits = {
        "hatemm": {"train": C.read_ids(RH_SPLITS / "hatemm_train.txt"),
                   "val": C.read_ids(RH_SPLITS / "hatemm_val.txt")},
        "hateclipseg": {"train": list(p11["train"]), "val": list(p11["val"])},
        "dehate": {"train": C.read_ids(RH_SPLITS / "dehate_train.txt"),
                   "val": C.read_ids(RH_SPLITS / "dehate_val.txt")},
    }
    for c in C.CORPORA:
        splits[c]["test"] = coh[c]

    # ---- DeHate split files == the CSV Split column
    rows = dehate_rows()
    csv_split = {}
    for r in rows:
        csv_split.setdefault(r["Split"], set()).add(r["Video ID"].strip())
    for k in ("train", "val"):
        if set(splits["dehate"][k]) != csv_split[k]:
            raise RuntimeError(f"DeHate {k} split file differs from the CSV Split column")
    if not set(coh["dehate"]) <= csv_split["test"]:
        raise RuntimeError("DeHate cohort is not inside the CSV test split")

    # ---- disjointness (actual id comparison)
    check = {}
    for c in C.CORPORA:
        s = {k: set(v) for k, v in splits[c].items()}
        if any(len(s[k]) != len(splits[c][k]) for k in s):
            raise RuntimeError(f"{c}: duplicate ids inside a split")
        inter = {"train&test": sorted(s["train"] & s["test"]), "val&test": sorted(s["val"] & s["test"]),
                 "train&val": sorted(s["train"] & s["val"])}
        check[c] = {"n_train": len(s["train"]), "n_val": len(s["val"]), "n_test_cohort": len(s["test"]),
                    "train_and_test": len(inter["train&test"]), "val_and_test": len(inter["val&test"]),
                    "train_and_val": len(inter["train&val"]),
                    "train_union_val_and_test": len((s["train"] | s["val"]) & s["test"])}
        if any(inter.values()):
            raise RuntimeError(f"{c}: split overlap {({k: v[:5] for k, v in inter.items() if v})}")
        if len(s["test"]) != C.COHORT_SIZE[c]:
            raise RuntimeError(f"{c}: cohort {len(s['test'])} != {C.COHORT_SIZE[c]}")
        log(f"{c}: train {len(s['train'])} val {len(s['val'])} test(cohort) {len(s['test'])}; "
            f"(train ∪ val) ∩ test = {check[c]['train_union_val_and_test']}, train ∩ val = {check[c]['train_and_val']}")
    hcs_out = sorted(set(p11["test"]) - set(coh["hateclipseg"]))
    check["hateclipseg"]["p11_test_not_in_cohort"] = hcs_out
    log(f"hateclipseg: p11 test ids outside the cohort (never scored, never trained on): {hcs_out}")

    # ---- labels (train + val only)
    labels = {}
    hm = hdata.load_labels("hatemm")
    hc = hdata.load_labels("hateclipseg")
    dh = {r["Video ID"].strip(): int(r["Hate"]) for r in rows}
    src = {"hatemm": hm, "hateclipseg": hc, "dehate": dh}
    for c in C.CORPORA:
        ids = splits[c]["train"] + splits[c]["val"]
        miss = [v for v in ids if v not in src[c]]
        if miss:
            raise RuntimeError(f"{c}: {len(miss)} train/val ids without a label, e.g. {miss[:5]}")
        labels[c] = {v: int(src[c][v]) for v in ids}
        for k in ("train", "val"):
            pos = sum(labels[c][v] for v in splits[c][k])
            check[c][f"{k}_hateful"] = pos
            check[c][f"{k}_normal"] = len(splits[c][k]) - pos
        log(f"{c}: labels train {check[c]['train_hateful']}+/{check[c]['train_normal']}-, "
            f"val {check[c]['val_hateful']}+/{check[c]['val_normal']}-")
    # HCS cross-check on train/val: segment CSV union rule vs p11 gold_segments.json (dims 1..5 of each segment)
    gold = json.loads(P11_GOLD.read_text())
    alt = {v: int(any(any(int(x) == 1 for x in seg[2][1:6]) for seg in gold[v]["segments"]))
           for v in splits["hateclipseg"]["train"] + splits["hateclipseg"]["val"]}
    dis = sorted(v for v in alt if alt[v] != labels["hateclipseg"][v])
    check["hateclipseg"]["label_crosscheck_disagreements"] = dis
    log(f"hateclipseg: segment-CSV labels vs gold_segments.json on train+val: {len(dis)} disagreements")

    # ---- write splits and labels
    (C.INPUTS / "splits").mkdir(parents=True, exist_ok=True)
    (C.INPUTS / "labels").mkdir(parents=True, exist_ok=True)
    for c in C.CORPORA:
        for k, ids in splits[c].items():
            (C.INPUTS / "splits" / f"{c}_{k}.txt").write_text("\n".join(ids) + "\n")
        (C.INPUTS / "labels" / f"{c}.json").write_text(json.dumps(labels[c], indent=0, sort_keys=True) + "\n")

    # ---- feature copies
    feat_report = {}
    for feat, corpora in FEATURES.items():
        for c in corpora:
            src_dir = FEATURE_SRC[c] / feat / c
            dst = C.INPUTS / feat / c
            dst.mkdir(parents=True, exist_ok=True)
            ids = splits[c]["train"] + splits[c]["val"] + splits[c]["test"]
            miss = [v for v in ids if not (src_dir / f"{v}.npy").is_file()]
            if miss:
                raise RuntimeError(f"{feat}/{c}: {len(miss)} ids without a feature file, e.g. {miss[:5]}")
            n_copied, dims = 0, set()
            for v in ids:
                target = dst / f"{v}.npy"
                if not target.is_file():
                    shutil.copy2(src_dir / f"{v}.npy", target)
                    n_copied += 1
                arr = np.load(target, mmap_mode="r")
                dims.add(arr.shape[1])
                if arr.ndim != 2 or arr.shape[0] < 1:
                    raise RuntimeError(f"{feat}/{c}/{v}: bad shape {arr.shape}")
            feat_report[f"{feat}/{c}"] = {"source": str(src_dir), "n_ids": len(ids), "copied_now": n_copied,
                                          "dims": sorted(dims)}
            log(f"{feat}/{c}: {len(ids)} ids present (copied now {n_copied}), dim {sorted(dims)}, from {src_dir}")

    report = {"date": datetime.date.today().isoformat(), "host": socket.gethostname(),
              "code": "experiments/20261008_baselines/weaksup_common/prepare_inputs.py, " + C.code_version(),
              "split_check": check, "features": feat_report,
              "sources": {"hatemm_splits": str(RH_SPLITS / "hatemm_{train,val}.txt"), "hateclipseg_splits": str(P11),
                          "dehate_splits": str(RH_SPLITS / "dehate_{train,val}.txt"),
                          "cohort_hatemm_hateclipseg": str(COHORT_PRED.relative_to(C.REPO)),
                          "cohort_dehate": "data/gt_4fps/DeHate.npz (test ids)"}}
    out = C.RUNS / "weaksup_inputs" / "split_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(out, C.INPUTS / "split_check.json")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
