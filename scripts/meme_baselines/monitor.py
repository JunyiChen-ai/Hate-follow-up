from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP2")
RESULT = ROOT / "results" / "meme_baselines"
DATA = ROOT / "datasets" / "harmful_meme" / "processed"
DATASETS = ("FHM", "MAMI", "ToxiCN_MM")
OUTPUTS = [
    ("naive_2b", "test_naive.jsonl"),
    ("mars_32b_awq", "test_mars.jsonl"),
    ("alarm_7b", "test_alarm.jsonl"),
    ("lorehm_32b_awq", "test_lorehm.jsonl"),
    ("mod_hate", "test_mod_hate_4shot.jsonl"),
    ("mod_hate", "test_mod_hate_8shot.jsonl"),
]


def n_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def invalid(path: Path) -> int:
    bad = 0
    if not path.exists():
        return bad
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            if r.get("pred") not in (0, 1, -1, None) or r.get("label") not in (0, 1):
                bad += 1
    return bad


def main():
    print("== squeue ==")
    try:
        out = subprocess.check_output(["squeue", "-u", subprocess.check_output(["whoami"], text=True).strip(), "-o", "%.18i %.22j %.8T %.10M %.20R"], text=True)
        print(out.strip())
    except Exception as exc:
        print(f"squeue failed: {exc}")
    print("\n== outputs ==")
    for ds in DATASETS:
        expected = n_jsonl(DATA / ds / "test.jsonl")
        for base, fname in OUTPUTS:
            path = RESULT / base / ds / fname
            rows = n_jsonl(path)
            bad = invalid(path)
            print(f"{base:16s} {ds:10s} {rows:5d}/{expected:<5d} invalid={bad} {path}")
    print("\n== recent log tails ==")
    for p in sorted((ROOT / "logs" / "meme_baselines" / "slurm").glob("*.err"))[-4:]:
        print(f"-- {p} --")
        try:
            print("\n".join(p.read_text(errors="replace").splitlines()[-8:]))
        except Exception as exc:
            print(exc)


if __name__ == "__main__":
    main()

