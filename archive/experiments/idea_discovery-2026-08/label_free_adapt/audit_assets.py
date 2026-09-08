#!/usr/bin/env python3
"""Evidence-based official asset audit for the twelve adaptations."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ASSETS = {
    "A01": "https://github.com/yongliang-wu/NumPro.git",
    "A02": "https://github.com/zhangjinglei168/VTimeCoT.git",
    "A03": "https://github.com/minjoong507/Consistency-of-Video-LLM.git",
    "A04": "https://github.com/oceanflowlab/OmniVTG.git",
    "A05": "https://github.com/THUNLP-MT/MUSEG.git",
    "A06": "https://github.com/josephzpng/DisTime.git",
    "A07": None,
    "A08": "https://github.com/bigai-nlco/VideoTGB.git",
    "A09": "https://github.com/lzw-lzw/GroundingGPT.git",
    "A10": "https://github.com/baopj/Vid-Group.git",
    "A11": None,
    "A12": "https://github.com/TencentARC/TimeLens.git",
}


def probe(url: str | None) -> dict:
    if url is None:
        return {"status": "PARTIAL", "reason": "official repository URL not verified"}
    try:
        proc = subprocess.run(["git", "ls-remote", "--heads", url], capture_output=True,
                              text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "BLOCKED", "reason": "network timeout", "url": url}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"status": "BLOCKED", "reason": proc.stderr.strip()[-300:], "url": url}
    return {"status": "READY", "reason": "official repository reachable", "url": url,
            "n_heads": len(proc.stdout.splitlines())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/label_free_adapt/assets/audit.json")
    args = ap.parse_args()
    result = {key: probe(url) for key, url in ASSETS.items()}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

