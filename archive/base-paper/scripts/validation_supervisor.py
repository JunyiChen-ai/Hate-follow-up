#!/usr/bin/env python3
"""Restart JSONL-producing validation tasks after recoverable video failures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/data/jehc223/EMNLP3")
sys.path.insert(0, str(PROJECT_ROOT / "src" / "our_method"))

from data_utils import load_clean_split_ids  # noqa: E402

CUDA_FATAL_PATTERNS = (
    "CUDA error: an illegal memory access",
    "illegal memory access",
    "device-side assert",
)
FRAME_RETRY_PATTERNS = (
    "longer than the maximum model length",
)


def parse_frame_fallbacks(raw: str) -> list[int]:
    vals: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val <= 0:
            raise ValueError("frame fallbacks must be positive")
        if val not in vals:
            vals.append(val)
    if not vals:
        raise ValueError("empty frame fallback list")
    return vals


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = row.get("video_id")
            if vid:
                done.add(vid)
    return done


def load_skip_ids(skip_file: Path) -> set[str]:
    if not skip_file.exists():
        return set()
    return {line.strip() for line in skip_file.read_text().splitlines() if line.strip()}


def append_skip(skip_file: Path, video_id: str) -> None:
    skips = load_skip_ids(skip_file)
    if video_id in skips:
        return
    skip_file.parent.mkdir(parents=True, exist_ok=True)
    with skip_file.open("a") as f:
        f.write(video_id + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_frame_event(log_dir: Path, event: dict) -> None:
    path = log_dir / "frame_fallbacks.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-file", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument(
        "--adaptive-frames",
        action="store_true",
        help="Run pending videos at the highest frame count, then retry only the failing video at lower counts.",
    )
    parser.add_argument(
        "--frame-fallbacks",
        default="16,8,4,2,1",
        help="Comma-separated frame counts to try from highest to lowest when --adaptive-frames is set.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.command[0] != "--":
        parser.error("pass child command after --")
    command = args.command[1:]
    if not command:
        parser.error("empty child command")
    try:
        frame_fallbacks = parse_frame_fallbacks(args.frame_fallbacks)
    except ValueError as exc:
        parser.error(str(exc))

    out_path = Path(args.output)
    skip_file = Path(args.skip_file)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    skip_file.parent.mkdir(parents=True, exist_ok=True)
    skip_file.touch(exist_ok=True)

    expected = load_clean_split_ids(args.dataset, args.split)
    attempt = 0
    while attempt < args.max_attempts:
        done = load_done_ids(out_path)
        skips = load_skip_ids(skip_file)
        pending = [vid for vid in expected if vid not in done and vid not in skips]
        print(
            f"[supervisor] {args.dataset}/{args.split} done={len(done)} "
            f"skip={len(skips)} pending={len(pending)} total={len(expected)}",
            flush=True,
        )
        if not pending:
            print("[supervisor] complete", flush=True)
            return 0

        first_pending = pending[0]
        before = len(done)
        attempt += 1
        log_path = log_dir / f"attempt_{attempt:04d}_{first_pending}.log"
        env = os.environ.copy()
        env["EXTRA_SKIP_VIDEOS_FILE"] = str(skip_file)
        if args.adaptive_frames:
            env["NUM_FRAMES"] = str(frame_fallbacks[0])
        print(f"[supervisor] attempt={attempt} first_pending={first_pending}", flush=True)
        t0 = time.time()
        with log_path.open("w") as log:
            proc = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elapsed = time.time() - t0
        done_after = load_done_ids(out_path)
        skips_after = load_skip_ids(skip_file)
        log_text = log_path.read_text(errors="replace")
        fatal_cuda = any(pat in log_text for pat in CUDA_FATAL_PATTERNS)
        frame_retryable = fatal_cuda or any(pat in log_text for pat in FRAME_RETRY_PATTERNS)
        print(
            f"[supervisor] attempt={attempt} exit={proc.returncode} "
            f"elapsed={elapsed:.1f}s done {before}->{len(done_after)} "
            f"skip={len(skips_after)} cuda_fatal={fatal_cuda} log={log_path}",
            flush=True,
        )
        if not args.adaptive_frames and len(done_after) > before:
            continue
        if args.adaptive_frames:
            if len(done_after) > before and frame_retryable:
                skips_now = load_skip_ids(skip_file)
                pending_after = [
                    vid for vid in expected
                    if vid not in done_after and vid not in skips_now
                ]
                if pending_after:
                    first_pending = pending_after[0]
                    print(
                        f"[supervisor] progress before frame-retryable failure; "
                        f"frame-fallback next_pending={first_pending}",
                        flush=True,
                    )
                else:
                    continue
            elif len(done_after) > before:
                continue
            recovered = False
            for frames in frame_fallbacks[1:]:
                done_before_frame = load_done_ids(out_path)
                frame_log = log_dir / f"attempt_{attempt:04d}_{first_pending}_frames{frames}.log"
                frame_env = os.environ.copy()
                frame_env["EXTRA_SKIP_VIDEOS_FILE"] = str(skip_file)
                frame_env["ONLY_VIDEO_ID"] = first_pending
                frame_env["NUM_FRAMES"] = str(frames)
                print(
                    f"[supervisor] frame-fallback video={first_pending} frames={frames}",
                    flush=True,
                )
                t_frame = time.time()
                with frame_log.open("w") as log:
                    frame_proc = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        env=frame_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                frame_elapsed = time.time() - t_frame
                done_after_frame = load_done_ids(out_path)
                frame_text = frame_log.read_text(errors="replace")
                frame_cuda = any(pat in frame_text for pat in CUDA_FATAL_PATTERNS)
                print(
                    f"[supervisor] frame-fallback frames={frames} exit={frame_proc.returncode} "
                    f"elapsed={frame_elapsed:.1f}s done {len(done_before_frame)}->{len(done_after_frame)} "
                    f"cuda_fatal={frame_cuda} log={frame_log}",
                    flush=True,
                )
                append_frame_event(
                    log_dir,
                    {
                        "video_id": first_pending,
                        "frames": frames,
                        "exit_code": frame_proc.returncode,
                        "elapsed_sec": round(frame_elapsed, 3),
                        "done_before": len(done_before_frame),
                        "done_after": len(done_after_frame),
                        "cuda_fatal": frame_cuda,
                        "log": str(frame_log),
                    },
                )
                if len(done_after_frame) > len(done_before_frame):
                    recovered = True
                    break
            if recovered:
                continue
            append_skip(skip_file, first_pending)
            print(f"[supervisor] all frame fallbacks failed; crash-skipping {first_pending}", flush=True)
            continue
        if proc.returncode != 0 and fatal_cuda:
            append_skip(skip_file, first_pending)
            print(f"[supervisor] crash-skipping {first_pending}", flush=True)
            continue
        if proc.returncode == 0:
            print("[supervisor] child exited cleanly but made no progress; stopping for audit", flush=True)
            return 3
        print("[supervisor] non-CUDA child failure; stopping", flush=True)
        return proc.returncode or 1
    print("[supervisor] max attempts exceeded", flush=True)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
