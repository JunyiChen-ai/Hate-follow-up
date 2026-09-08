#!/usr/bin/env python3
"""Stereotype-stratum check on HateClipSeg, stage 1: build the coding packet.

Question. The five-corpus error attribution left one model-side weakness open:
the frozen single-call Qwen3-VL-8B judge ranks gender and sexuality stereotype
content poorly on the pooled MHClip implicit stratum (fifteen of eighteen videos
of that type, AUC 0.71 flat across evidence-length bins) while ranking race and
nationality dog-whistles well on ImpliHateVid (0.9255). That sample is eighteen
videos. This packet supplies a third corpus for the same question.

Three HateClipSeg cohorts are packed for manual content coding:

  INS  the 59 videos whose only shipped category is `insulting` (AUC 0.637)
  NRM  the 50 videos that carry no offensive category at all (the negative class)
  HAT  a 40-video random sample, seed 20260808, of the 180 `hateful` videos

The packet carries video identifiers, transcript text and frame paths, so it
stays under results/ (gitignored). Only aliases of the form HCS-INS-07 and
aggregate statistics reach the committed note.

No model call, no GPU, CPU only.
"""

import json
import os
import random

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
OUT = os.path.join(ROOT, "results", "stereotype_stratum")
ITEMS = os.path.join(ROOT, "results", "ranking_autopsy", "hcs", "items.json")
FRESH = os.path.join(ROOT, "results", "hateclipseg", "fresh_transcripts.jsonl")
OVERRIDES = os.path.join(ROOT, "results", "hateclipseg", "c2_overrides.json")
FRAME_DIR = "/home/jehc223/data/HateClipSeg/frames_16"

SEED = 20260808
N_HATEFUL_SAMPLE = 40
PICK = [2, 6, 10, 14]     # same frame indices the ranking autopsy montaged
TILE = 336
EXCERPT = 1200            # transcript characters shown to the coder


def load_jsonl(path, key="video_id"):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r[key]] = r
    return out


def cohorts(items):
    """(alias prefix, ordered video records) for the three coded populations."""
    ins = [i for i in items if set(i["labels"]) == {"insulting"}]
    nrm = [i for i in items if i["union"] == 0]
    hat = [i for i in items if "hateful" in i["labels"]]
    rng = random.Random(SEED)
    hat_sample = rng.sample(sorted(hat, key=lambda x: x["video_id"]),
                            N_HATEFUL_SAMPLE)
    key = lambda r: r["video_id"]  # noqa: E731  deterministic alias order
    return [("INS", sorted(ins, key=key)),
            ("NRM", sorted(nrm, key=key)),
            ("HAT", sorted(hat_sample, key=key))]


def frames_for(vid):
    d = os.path.join(FRAME_DIR, vid)
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.endswith(".jpg")]


def montage(rec, out_dir):
    fr = rec["frames"]
    if not fr:
        return None
    idx = [min(i, len(fr) - 1) for i in PICK]
    canvas = Image.new("RGB", (TILE * 2, TILE * 2), (20, 20, 20))
    for k, i in enumerate(idx):
        im = Image.open(fr[i]).convert("RGB")
        im.thumbnail((TILE, TILE))
        x = (k % 2) * TILE + (TILE - im.width) // 2
        y = (k // 2) * TILE + (TILE - im.height) // 2
        canvas.paste(im, (x, y))
    p = os.path.join(out_dir, rec["alias"] + ".jpg")
    canvas.save(p, quality=80)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    mdir = os.path.join(OUT, "montage")
    os.makedirs(mdir, exist_ok=True)

    items = json.load(open(ITEMS))
    fresh = load_jsonl(FRESH)
    over = json.load(open(OVERRIDES)) if os.path.exists(OVERRIDES) else {}

    packet, lines = [], []
    for prefix, rows in cohorts(items):
        for k, it in enumerate(rows, 1):
            vid = it["video_id"]
            ftxt = (fresh.get(vid) or {}).get("fresh_text", "") or ""
            jtxt = over.get(vid, "") or ""
            rec = {
                "alias": f"HCS-{prefix}-{k:02d}",
                "cohort": prefix,
                "video_id": vid,
                "z": it["z"],
                "labels": it["labels"],
                "victims": it["victims"],
                "fresh_chars": len(ftxt),
                "judge_chars": len(jtxt),
                "judge_saw_transcript": bool(jtxt),
                "vad_speech_frac": it.get("vad_speech_frac"),
                "wav_duration": it.get("wav_duration"),
                "fresh_text": ftxt,
                "frames": frames_for(vid),
            }
            rec["montage"] = montage(rec, mdir)
            packet.append(rec)
            lines.append(
                "=" * 70 + "\n"
                f"{rec['alias']}  labels={','.join(it['labels'])}  "
                f"victims={','.join(it['victims']) or '-'}  "
                f"fresh_chars={rec['fresh_chars']}  "
                f"judge_saw={'Y' if jtxt else 'N'}  "
                f"dur={rec['wav_duration']}  vad={rec['vad_speech_frac']}\n"
                f"  T: {ftxt[:EXCERPT] if ftxt else '(no speech recognised)'}\n")

    with open(os.path.join(OUT, "packet.json"), "w") as f:
        json.dump(packet, f, indent=1)
    with open(os.path.join(OUT, "packet.txt"), "w") as f:
        f.write("".join(lines))
    counts = {}
    for r in packet:
        counts[r["cohort"]] = counts.get(r["cohort"], 0) + 1
    print(json.dumps({"n": len(packet), "by_cohort": counts,
                      "montages": sum(1 for r in packet if r["montage"])},
                     indent=1))


if __name__ == "__main__":
    main()
