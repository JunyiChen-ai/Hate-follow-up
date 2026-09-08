from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

try:
    from .common import DATASETS, PROJECT_ROOT, read_jsonl, write_jsonl
except ImportError:
    from common import DATASETS, PROJECT_ROOT, read_jsonl, write_jsonl

DATA_ROOT = PROJECT_ROOT / "datasets" / "harmful_meme"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
B2_ROOT = "b2:junyi-data/harmful meme"


def _record(dataset: str, split: str, sid: str, image_path: Path, text: str, label: int, metadata: dict | None = None) -> dict:
    return {
        "id": f"{dataset}:{split}:{sid}",
        "dataset": dataset,
        "split": split,
        "image_path": str(image_path),
        "text": text or "",
        "label": int(label),
        "source_id": sid,
        "metadata": metadata or {},
    }


def sync_raw(dataset: str, dry_run: bool = False, exclude_zips: bool = True, transfers: int = 16, checkers: int = 32) -> None:
    datasets = DATASETS if dataset == "all" else (dataset,)
    for ds in datasets:
        dest = RAW_ROOT / ds
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rclone",
            "copy",
            f"{B2_ROOT}/{ds}",
            str(dest),
            "--progress",
            "--transfers",
            str(transfers),
            "--checkers",
            str(checkers),
            "--exclude",
            ".cache/**",
        ]
        if exclude_zips:
            cmd.extend(["--exclude", "*.zip"])
        if dry_run:
            cmd.append("--dry-run")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


def build_fhm() -> dict[str, int]:
    root = RAW_ROOT / "FHM"
    out_root = PROCESSED_ROOT / "FHM"
    train_rows = []
    dropped_train = 0
    for row in read_jsonl(root / "train.jsonl"):
        sid = str(row["id"])
        image_path = root / row["img"]
        if not image_path.is_file():
            dropped_train += 1
            continue
        train_rows.append(_record("FHM", "train", sid, image_path, row.get("text", ""), int(row["label"])))

    test_rows = []
    dropped_test = 0
    for name in ("test_seen.jsonl", "test_unseen.jsonl"):
        for row in read_jsonl(root / name):
            sid = str(row["id"])
            image_path = root / row["img"]
            if not image_path.is_file():
                dropped_test += 1
                continue
            meta = {"fhm_test_partition": name.replace(".jsonl", "")}
            test_rows.append(_record("FHM", "test", sid, image_path, row.get("text", ""), int(row["label"]), meta))

    write_jsonl(out_root / "train.jsonl", train_rows)
    write_jsonl(out_root / "test.jsonl", test_rows)
    return {"train": len(train_rows), "test": len(test_rows), "dropped_missing_train": dropped_train, "dropped_missing_test": dropped_test}


def build_mami() -> dict[str, int]:
    root = RAW_ROOT / "MAMI"
    out_root = PROCESSED_ROOT / "MAMI"
    train_rows = []
    with (root / "TRAINING" / "training.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row["file_name"]
            label = int(row["misogynous"])
            text = row.get("Text Transcription", "") or ""
            meta = {k: row[k] for k in ("shaming", "stereotype", "objectification", "violence") if k in row}
            train_rows.append(_record("MAMI", "train", sid, root / "TRAINING" / sid, text, label, meta))

    test_rows = []
    with (root / "test_labels.txt").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            sid = row[0]
            label = int(row[1])
            meta = {
                "shaming": row[2] if len(row) > 2 else None,
                "stereotype": row[3] if len(row) > 3 else None,
                "objectification": row[4] if len(row) > 4 else None,
                "violence": row[5] if len(row) > 5 else None,
                "text_missing_in_release": True,
            }
            test_rows.append(_record("MAMI", "test", sid, root / "test" / sid, "", label, meta))

    write_jsonl(out_root / "train.jsonl", train_rows)
    write_jsonl(out_root / "test.jsonl", test_rows)
    return {"train": len(train_rows), "test": len(test_rows)}


def _load_toxicn_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"{path} did not contain a JSON list")
    return obj


def build_toxicn() -> dict[str, int]:
    root = RAW_ROOT / "ToxiCN_MM"
    out_root = PROCESSED_ROOT / "ToxiCN_MM"
    counts = {}
    for split, filename in (("train", "train_data_discription_2.0.json"), ("test", "test_data_discription_2.0.json")):
        rows = []
        for i, row in enumerate(_load_toxicn_json(root / filename)):
            rel_path = row.get("path") or ""
            sid = rel_path or str(i)
            image_path = root / rel_path
            if not image_path.is_file():
                image_path = root / "meme" / rel_path
            meta = {
                "type": row.get("type"),
                "text_modal": row.get("text_modal"),
                "image_modal": row.get("image_modal"),
                "target": row.get("target"),
                "meme_discription": row.get("meme_discription"),
                "text_discription": row.get("text_discription"),
            }
            rows.append(_record("ToxiCN_MM", split, sid, image_path, row.get("text", ""), int(row["label"]), meta))
        write_jsonl(out_root / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    return counts


def build_processed(dataset: str) -> dict[str, dict[str, int]]:
    builders = {"FHM": build_fhm, "MAMI": build_mami, "ToxiCN_MM": build_toxicn}
    datasets = DATASETS if dataset == "all" else (dataset,)
    return {ds: builders[ds]() for ds in datasets}


def audit_processed(dataset: str) -> dict[str, dict]:
    datasets = DATASETS if dataset == "all" else (dataset,)
    summary = {}
    for ds in datasets:
        ds_summary = {}
        for split in ("train", "test"):
            path = PROCESSED_ROOT / ds / f"{split}.jsonl"
            rows = read_jsonl(path)
            ids = [r.get("id") for r in rows]
            missing_images = [r["id"] for r in rows if not Path(r["image_path"]).is_file()]
            bad_labels = [r["id"] for r in rows if r.get("label") not in (0, 1)]
            dup = len(ids) - len(set(ids))
            ds_summary[split] = {
                "path": str(path),
                "n": len(rows),
                "n_pos": sum(int(r["label"]) for r in rows if r.get("label") in (0, 1)),
                "n_neg": sum(1 for r in rows if r.get("label") == 0),
                "missing_images": len(missing_images),
                "bad_labels": len(bad_labels),
                "duplicate_ids": dup,
            }
        summary[ds] = ds_summary
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync/build/audit harmful meme datasets")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("sync", "build", "audit", "ensure"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", default="all", choices=("all", *DATASETS))
        if name in ("sync", "ensure"):
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--include-zips", action="store_true")
            p.add_argument("--transfers", type=int, default=16)
            p.add_argument("--checkers", type=int, default=32)
    args = parser.parse_args()

    if args.cmd == "sync":
        sync_raw(args.dataset, dry_run=args.dry_run, exclude_zips=not args.include_zips, transfers=args.transfers, checkers=args.checkers)
    elif args.cmd == "build":
        print(json.dumps(build_processed(args.dataset), ensure_ascii=False, indent=2))
    elif args.cmd == "audit":
        audit_processed(args.dataset)
    elif args.cmd == "ensure":
        sync_raw(args.dataset, dry_run=args.dry_run, exclude_zips=not args.include_zips, transfers=args.transfers, checkers=args.checkers)
        if not args.dry_run:
            print(json.dumps(build_processed(args.dataset), ensure_ascii=False, indent=2))
            audit_processed(args.dataset)


if __name__ == "__main__":
    main()
