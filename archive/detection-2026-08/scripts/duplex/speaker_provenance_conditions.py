#!/usr/bin/env python3
"""Freeze P/S/R transcript interventions from Sortformer provenance."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/speaker_provenance"


def permute(tag):
    if tag == "SPEAKER_UNKNOWN": return tag
    tag = tag.replace("PRIMARY", "TMP").replace("SECONDARY", "PRIMARY").replace("TMP", "SECONDARY")
    return tag.replace("ONSCREEN", "TMP").replace("VOICEOVER", "ONSCREEN").replace("TMP", "VOICEOVER")


def main():
    rows = [json.loads(x) for x in open(RUN / "sortformer_provenance.jsonl")]
    arms = {"P": {}, "S": {}, "R": {}}
    for r in rows:
        lines = {k: [] for k in arms}
        for c in r["chunks"]:
            text = (c.get("text") or "").strip()
            if not text: continue
            tag = c["tag"]
            lines["P"].append(f"[{tag}]: {text}")
            lines["S"].append(f"[SPEAKER]: {text}")
            lines["R"].append(f"[{permute(tag)}]: {text}")
        for arm in arms: arms[arm][r["video_id"]] = "\n".join(lines[arm])
    for arm, obj in arms.items():
        (RUN / f"override_{arm}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    (RUN / "video_ids.txt").write_text(",".join(sorted(arms["P"])) + "\n")
    print({k: len(v) for k, v in arms.items()})


if __name__ == "__main__": main()
