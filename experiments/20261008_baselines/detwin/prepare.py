#!/usr/bin/env python3
"""CPU preparation for the SAGE / CLARA window runs (2026-10-08). Run on uoa-lab1.

    python experiments/20261008_baselines/detwin/prepare.py splits   # data/weaksup_video_splits/<ds>.json + check
    python experiments/20261008_baselines/detwin/prepare.py asr      # data/asr_whisper_large_v3/<ds>/all_splits_chunks.jsonl

Splits (train / val with video-level labels; test = the exact cohort, ids and durations only):
- HateMM: Retrieval-hate `results/reproduction/splits/hatemm_{train,val}.txt` (744 / 109), the split of the earlier
  weakly supervised baselines (run_plan.md §3). Label from the id (hate_video_* = 1, non_hate_video_* = 0).
- HateClipSeg: Retrieval-hate `data/gt/HateClipSeg/p11_split.json` train / val (237 / 39). Label = 1 when any
  annotated segment has any non-normal dimension (offensive union), the rule of the HCS GT
  (`build_gt_4fps.py` `dataset_spans`: `any(dims[1:])`).
- DeHate: `~/data/DeHate/DeHate_labels.csv` column `Split` (train 4680 / val 668), label = column `Hate`.
- Test cohorts: HateMM / HateClipSeg = the ids of `runs/20260926_twolevel/final_rawkey/predictions.jsonl`
  (215 / 118); DeHate = the test ids of `data/gt_4fps/DeHate.npz` (1151). Durations from the test manifests.
The disjointness of train ∪ val and the test cohort is checked by id comparison and written to
`runs/20261008_baselines/sage_clara_splits/split_check.json`.
"""
from __future__ import annotations

import csv
import datetime
import json
import shutil
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RH = Path("/home/jehc223/Retrieval-hate")
HOME_DATA = Path.home() / "data"
COHORT_PRED = C.REPO / "runs/20260926_twolevel/final_rawkey/predictions.jsonl"
MANIFEST = {"HateMM": C.REPO / "data/omsl_v6_inputs/manifests/all_test.jsonl",
            "HateClipSeg": C.REPO / "data/omsl_v6_inputs/manifests/all_test.jsonl",
            "DeHate": C.REPO / "data/manifests/DeHate_test.jsonl"}
ASR_SOURCES = {
    "HateMM": C.REPO / "runs/legacy_1fps/lab2/reproduction/asr/hatemm_all/timestamped_chunks.jsonl",
    "HateClipSeg": C.REPO / "data/asr_whisper_large_v3/HateClipSeg/timestamped_chunks.jsonl",
    "DeHate": RH / "results/reproduction/asr/dehate_all/timestamped_chunks.jsonl",
}


def read_lines(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def cohort_ids(ds: str) -> list[str]:
    if ds == "DeHate":
        ids = set(C.gt_lengths(ds))
    else:
        ids = {json.loads(l)["video_id"] for l in COHORT_PRED.open() if json.loads(l)["dataset"] == ds}
    return sorted(ids)


def durations(ds: str) -> dict[str, float]:
    out = {}
    for line in MANIFEST[ds].open():
        r = json.loads(line)
        if r["dataset"] == ds:
            out[r["video_id"]] = float(r["duration"])
    return out


def build_splits() -> None:
    out_dir = C.SPLIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    check = {"date": datetime.date.today().isoformat(), "host": socket.gethostname(),
             "code": "experiments/20261008_baselines/detwin/prepare.py splits", "datasets": {}}
    for ds in C.DATASETS:
        if ds == "HateMM":
            tr = read_lines(RH / "results/reproduction/splits/hatemm_train.txt")
            va = read_lines(RH / "results/reproduction/splits/hatemm_val.txt")
            lab = lambda v: 1 if v.startswith("hate_video_") else (0 if v.startswith("non_hate_video_") else None)
            train = [{"video_id": v, "label": lab(v)} for v in tr]
            val = [{"video_id": v, "label": lab(v)} for v in va]
        elif ds == "HateClipSeg":
            p11 = json.loads((RH / "data/gt/HateClipSeg/p11_split.json").read_text())
            gold = json.loads((RH / "data/gt/HateClipSeg/gold_segments.json").read_text())
            lab = lambda v: int(any(any(dims[1:]) for _, _, dims in gold[v]["segments"]))
            train = [{"video_id": v, "label": lab(v)} for v in p11["train"]]
            val = [{"video_id": v, "label": lab(v)} for v in p11["val"]]
        else:
            train, val = [], []
            with (HOME_DATA / "DeHate/DeHate_labels.csv").open(newline="") as fh:
                for r in csv.DictReader(fh):
                    row = {"video_id": r["Video ID"], "label": int(r["Hate"])}
                    if r["Split"] == "train":
                        train.append(row)
                    elif r["Split"] == "val":
                        val.append(row)
        assert all(x["label"] in (0, 1) for x in train + val), ds
        coh = cohort_ids(ds)
        assert len(coh) == C.COHORT_SIZE[ds], (ds, len(coh))
        dur = durations(ds)
        test = [{"video_id": v, "duration": dur[v]} for v in coh]
        s_tr, s_va, s_te = {x["video_id"] for x in train}, {x["video_id"] for x in val}, set(coh)
        info = {"n_train": len(train), "n_val": len(val), "n_test_cohort": len(coh),
                "train_and_test": len(s_tr & s_te), "val_and_test": len(s_va & s_te),
                "train_and_val": len(s_tr & s_va), "train_union_val_and_test": len((s_tr | s_va) & s_te),
                "train_hateful": sum(x["label"] for x in train), "train_normal": sum(1 - x["label"] for x in train),
                "val_hateful": sum(x["label"] for x in val), "val_normal": sum(1 - x["label"] for x in val),
                "label_rule": {"HateMM": "id prefix hate_video_ = 1",
                               "HateClipSeg": "any segment with any(dims[1:]) (offensive union, = HCS GT rule)",
                               "DeHate": "DeHate_labels.csv Hate"}[ds]}
        assert info["train_union_val_and_test"] == 0 and info["train_and_val"] == 0, (ds, info)
        if ds == "HateClipSeg":
            info["p11_test_not_in_cohort"] = sorted(set(p11["test"]) - s_te)
            assert s_te <= set(p11["test"])
        # cross-check with the other weakly supervised agent's split files, if present (read only)
        other = C.REPO / "data/weaksup_1fps/splits"
        key = {"HateMM": "hatemm", "HateClipSeg": "hateclipseg", "DeHate": "dehate"}[ds]
        if (other / f"{key}_train.txt").is_file():
            info["equal_to_data_weaksup_1fps_splits"] = {
                sp: set(read_lines(other / f"{key}_{sp}.txt")) == ids
                for sp, ids in (("train", s_tr), ("val", s_va), ("test", s_te))}
        (out_dir / f"{ds}.json").write_text(json.dumps({"train": train, "val": val, "test": test}, indent=1) + "\n")
        check["datasets"][ds] = info
        print(ds, json.dumps(info))
    run_dir = C.RUNS / "sage_clara_splits"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split_check.json").write_text(json.dumps(check, indent=2) + "\n")
    (out_dir / "PROVENANCE.md").write_text(
        "# data/weaksup_video_splits\n\n"
        "`<dataset>.json`: train / val video ids with video-level labels (0/1), and the exact test cohort (ids and\n"
        "durations, no labels), for the SAGE / CLARA window baselines of `experiments/20261008_baselines/`.\n"
        f"Built {datetime.date.today().isoformat()} on {socket.gethostname()} (uoa-lab1) by\n"
        "`experiments/20261008_baselines/detwin/prepare.py splits`.\n\n"
        "Sources: HateMM train/val = Retrieval-hate `results/reproduction/splits/hatemm_{train,val}.txt` (744/109),\n"
        "label from the id prefix; HateClipSeg train/val = Retrieval-hate `data/gt/HateClipSeg/p11_split.json`\n"
        "(237/39), label = any segment with a non-normal dimension (offensive union, the HCS GT rule); DeHate\n"
        "train/val = `~/data/DeHate/DeHate_labels.csv` `Split` (4680/668), label = `Hate`. Test cohorts: HateMM /\n"
        "HateClipSeg ids of `runs/20260926_twolevel/final_rawkey/predictions.jsonl` (215/118), DeHate test ids of\n"
        "`data/gt_4fps/DeHate.npz` (1151); durations from `data/omsl_v6_inputs/manifests/all_test.jsonl` and\n"
        "`data/manifests/DeHate_test.jsonl`. Disjointness check: `runs/20261008_baselines/sage_clara_splits/"
        "split_check.json`.\n")


def build_asr() -> None:
    """Copy the Whisper large-v3 rows of every train / val / cohort video into
    data/asr_whisper_large_v3/<ds>/all_splits_chunks.jsonl (one row format, see that directory's PROVENANCE.md)."""
    report = {}
    for ds in C.DATASETS:
        sp = C.load_split(ds)
        need = [x["video_id"] for x in sp["train"] + sp["val"] + sp["test"]]
        rows = {}
        for line in ASR_SOURCES[ds].open():
            r = json.loads(line)
            if r["video_id"] in set(need):
                rows[r["video_id"]] = r
        # Test-cohort rows must be the transcripts our own method reads (data/asr_whisper_large_v3/<ds>/
        # timestamped_chunks.jsonl). DeHate and HateClipSeg: same source, checked identical. HateMM: the all-split
        # source is a separate 2026-08 run of the same script on uoa-lab2 whose test texts differ, so the test rows
        # are taken from the data/ file and only train/val rows from the all-split source.
        test_ids = {x["video_id"] for x in sp["test"]}
        n_replaced = 0
        for line in (C.ASR_DIR / ds / "timestamped_chunks.jsonl").open():
            r = json.loads(line)
            if r["video_id"] in test_ids:
                if ds == "HateMM":
                    n_replaced += int(rows.get(r["video_id"], {}).get("chunks") != r["chunks"])
                    rows[r["video_id"]] = r
                else:
                    assert r["chunks"] == rows[r["video_id"]]["chunks"], r["video_id"]
        out = C.ASR_DIR / ds / "all_splits_chunks.jsonl"
        with out.open("w") as fh:
            for v in need:
                if v in rows:
                    fh.write(json.dumps(rows[v]) + "\n")
        missing = [v for v in need if v not in rows]
        errors = [v for v in need if v in rows and rows[v].get("error")]
        untimed = sum(1 for v in rows for c in rows[v].get("chunks") or [] if c.get("end") is None)
        report[ds] = {"needed": len(need), "rows": len(need) - len(missing), "missing": missing,
                      "test_rows_from_data_file_that_differed_from_all_split_source": n_replaced,
                      "error_rows": errors, "untimed_chunks": untimed, "source": str(ASR_SOURCES[ds])}
        print(ds, json.dumps(report[ds])[:600])
    run_dir = C.RUNS / "sage_clara_splits"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "asr_check.json").write_text(json.dumps(report, indent=2) + "\n")
    prov = C.ASR_DIR / "PROVENANCE.md"
    text = prov.read_text()
    marker = "**All-split files for the SAGE / CLARA window baselines (2026-10-08).**"
    if marker not in text:
        prov.write_text(text.rstrip() + "\n\n" + marker + " `<dataset>/all_splits_chunks.jsonl`: the rows of every\n"
            "train / val / test-cohort video of `data/weaksup_video_splits/<dataset>.json`, copied unchanged on uoa-lab1 by\n"
            "`experiments/20261008_baselines/detwin/prepare.py asr`. Sources: HateMM train/val rows from\n"
            "`runs/legacy_1fps/lab2/reproduction/asr/hatemm_all/timestamped_chunks.jsonl` (same script as the HateMM test\n"
            "file above, separate uoa-lab2 run), HateMM test rows = the HateMM test file above (the lab2 run's test texts\n"
            "differ, so the test file our method reads is kept); HateClipSeg = this directory's `HateClipSeg/timestamped_chunks.jsonl`; DeHate =\n"
            "`~/Retrieval-hate/results/reproduction/asr/dehate_all/timestamped_chunks.jsonl` (the source of the DeHate test\n"
            "file above; test rows checked identical). Missing rows and error rows: "
            "`runs/20261008_baselines/sage_clara_splits/asr_check.json`.\n")


if __name__ == "__main__":
    {"splits": build_splits, "asr": build_asr}[sys.argv[1]]()
