#!/usr/bin/env python3
"""Supervise 16-frame Qwen2.5-VL-72B-AWQ HateMM offline judging.

The vLLM engine can be left unusable after a CUDA illegal-memory-access
inside the visual encoder. This supervisor keeps the experiment at the
canonical 16-frame setting, but restarts a fresh process after a crash and
records the offending video id in a run-local crash-skip list.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP3")
OUT = ROOT / "results/boundary_rescue/HateMM/offline_test_qwen2.5-vl-72b-awq.jsonl"
SKIP_FILE = ROOT / "results/boundary_rescue/HateMM/qwen2.5-vl-72b-awq_16f_crash_skip.txt"
LOG_DIR = ROOT / "results/boundary_rescue/HateMM/qwen2.5-vl-72b-awq_16f_attempt_logs"
TEST_SPLIT = ROOT / "datasets/HateMM/splits/test_clean.csv"


def load_test_ids() -> list[str]:
    return [line.strip() for line in TEST_SPLIT.read_text().splitlines() if line.strip()]


def load_done_ids() -> set[str]:
    if not OUT.exists():
        return set()
    done: set[str] = set()
    with OUT.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("pred") in (0, 1) and row.get("video_id"):
                done.add(row["video_id"])
    return done


def load_skip_ids() -> set[str]:
    if not SKIP_FILE.exists():
        return set()
    return {line.strip() for line in SKIP_FILE.read_text().splitlines() if line.strip()}


def append_skip(video_id: str) -> None:
    skips = load_skip_ids()
    if video_id in skips:
        return
    with SKIP_FILE.open("a") as f:
        f.write(video_id + "\n")
        f.flush()
        os.fsync(f.fileno())


def status() -> tuple[list[str], set[str], set[str], list[str]]:
    test_ids = load_test_ids()
    done = load_done_ids()
    skips = load_skip_ids()
    pending = [v for v in test_ids if v not in done and v not in skips]
    return test_ids, done, skips, pending


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SKIP_FILE.parent.mkdir(parents=True, exist_ok=True)
    SKIP_FILE.touch(exist_ok=True)

    attempt = 0
    while True:
        test_ids, done, skips, pending = status()
        print(
            f"[supervisor] status: valid={len(done)} skip={len(skips)} "
            f"pending={len(pending)} total={len(test_ids)}",
            flush=True,
        )
        if not pending:
            print("[supervisor] complete: no pending unprocessed ids", flush=True)
            return 0

        first_pending = pending[0]
        before = len(done)
        attempt += 1
        log_path = LOG_DIR / f"attempt_{attempt:03d}_{first_pending}.log"
        cmd = [
            sys.executable,
            "src/boundary_rescue/judge_offline.py",
            "--model",
            "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
            "--dataset",
            "HateMM",
            "--batch-size",
            "1",
            "--gpu-mem",
            "0.88",
            "--max-model-len",
            "16384",
        ]
        env = os.environ.copy()
        env["NUM_FRAMES"] = "16"
        env["EXTRA_SKIP_VIDEOS_FILE"] = str(SKIP_FILE)
        print(f"[supervisor] attempt {attempt}: first_pending={first_pending}", flush=True)
        t0 = time.time()
        with log_path.open("w") as log:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elapsed = time.time() - t0
        _, done_after, skips_after, pending_after = status()
        print(
            f"[supervisor] attempt {attempt} exit={proc.returncode} "
            f"elapsed={elapsed:.1f}s valid {before}->{len(done_after)} "
            f"skip={len(skips_after)} pending={len(pending_after)} "
            f"log={log_path}",
            flush=True,
        )

        log_text = log_path.read_text(errors="replace")
        if "CUDA error: an illegal memory access" in log_text:
            print("[supervisor] detected CUDA illegal memory access", flush=True)

        if len(done_after) == before:
            append_skip(first_pending)
            print(f"[supervisor] no progress; crash-skipping {first_pending}", flush=True)
        elif proc.returncode != 0:
            print("[supervisor] child returned nonzero after progress; continuing with fresh engine", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
