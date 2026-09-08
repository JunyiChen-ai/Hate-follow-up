"""Ranking-error autopsy helper: 2x2 frame montages for manual inspection.

Takes frames 2, 6, 10 and 14 of the frozen uniform-16 grid the judge saw and
tiles them into one image per video, so a coder can view four moments of a
video in a single read. Writes to results/ranking_autopsy/<corpus>/montage/.
"""

import json
import os
import sys

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "ranking_autopsy")
PICK = [2, 6, 10, 14]
TILE = 384


def montage(corpus, aliases=None):
    packet = json.load(open(os.path.join(OUT, corpus, "packet.json")))
    d = os.path.join(OUT, corpus, "montage")
    os.makedirs(d, exist_ok=True)
    made = []
    for r in packet:
        if aliases and r["alias"] not in aliases:
            continue
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
        p = os.path.join(d, f"{r['alias']}.jpg")
        canvas.save(p, quality=82)
        made.append(p)
    return made


if __name__ == "__main__":
    c = sys.argv[1]
    al = sys.argv[2:] or None
    for p in montage(c, al):
        print(p)
