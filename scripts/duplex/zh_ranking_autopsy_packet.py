"""Ranking-error autopsy for MHClip-ZH, stage 2: cohorts for manual coding.

Builds two packets under results/ranking_autopsy/zh/ (gitignored, they carry
video ids and transcript text):

  POS  all 45 union positives, ordered by ascending raw z, so that alias
       ZH-POS-01 is the worst-ranked positive. Used both for the false-negative
       taxonomy (the first twenty) and for the protected-target construct
       measurement that mirrors MHClip-EN.
  FP   the twenty highest-scoring Normals.

Also renders one 2x2 frame montage per video from the same uniform-16 grid the
judge read, so a coder sees four moments of a video in a single image read.
"""

import json
import os

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "ranking_autopsy", "zh")
RUN = os.path.join(ROOT, "results", "testruns", "mhclip_zh")
FRAMES = "/home/jehc223/data/Multihateclip/Chinese/frames_16"
N_FP = 20
PICK = [1, 5, 9, 13]
TILE = 400


def main():
    items = json.load(open(os.path.join(OUT, "items.json")))
    ov = json.load(open(os.path.join(RUN, "c2_overrides.json")))

    pos = sorted([i for i in items if i["binary"] == 1], key=lambda i: i["z"])
    neg = sorted([i for i in items if i["binary"] == 0], key=lambda i: -i["z"])

    packet = []
    for side, rows in (("POS", pos), ("FP", neg[:N_FP])):
        for k, it in enumerate(rows, 1):
            v = it["video_id"]
            fd = os.path.join(FRAMES, v)
            frames = sorted(os.listdir(fd)) if os.path.isdir(fd) else []
            rec = dict(it)
            rec.update({"alias": f"ZH-{side}-{k:02d}", "side": side, "rank_in_side": k,
                        "judge_saw_override": v in ov, "judge_text": ov.get(v, ""),
                        "frame_dir": fd,
                        "frames": [os.path.join(fd, f) for f in frames]})
            packet.append(rec)

    with open(os.path.join(OUT, "packet.json"), "w") as f:
        json.dump(packet, f, indent=2, ensure_ascii=False)

    lines = []
    for r in packet:
        lines.append("=" * 78)
        lines.append(f"{r['alias']}  z={r['z']}  label={r['fine']}")
        lines.append(f"  TITLE: {r['title']}")
        lines.append(f"  dur={r.get('wav_duration')}s vad={r.get('vad_speech_frac')} "
                     f"fresh_chars={r.get('fresh_chars')} "
                     f"judge_chars={r.get('n_transcript_chars')} "
                     f"src={r.get('transcript_source')} "
                     f"gate={r.get('gate_outcome')} "
                     f"script={r['script_fresh']['dominant_script']} "
                     f"cjk={r['script_fresh']['frac_cjk']}")
        lines.append(f"  frames: {r['frame_dir']} n={len(r['frames'])}")
        lines.append(f"  JUDGE_TEXT: {(r['judge_text'] or r['fresh_text'])[:2500]}")
        lines.append(f"  DATASET_TX: {r['dataset_transcript'][:1200]}")
        lines.append("")
    with open(os.path.join(OUT, "packet.txt"), "w") as f:
        f.write("\n".join(lines))

    d = os.path.join(OUT, "montage")
    os.makedirs(d, exist_ok=True)
    for r in packet:
        fr = r["frames"]
        if not fr:
            continue
        idx = [min(i, len(fr) - 1) for i in PICK]
        canvas = Image.new("RGB", (TILE * 2, TILE * 2), (20, 20, 20))
        for k, i in enumerate(idx):
            im = Image.open(fr[i]).convert("RGB")
            im.thumbnail((TILE, TILE))
            x = (k % 2) * TILE + (TILE - im.width) // 2
            y = (k // 2) * TILE + (TILE - im.height) // 2
            canvas.paste(im, (x, y))
        canvas.save(os.path.join(d, f"{r['alias']}.jpg"), quality=84)
    print(f"packet {len(packet)} rows, montages in {d}")


if __name__ == "__main__":
    main()
