"""Build a blinded packet and contact sheets for the HateMM FP premise audit."""

import json
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "duplex"))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
from c2_fullcorpus_analyze import kde_valley, load_z
from data_utils import DATASET_ROOTS, load_annotations, load_clean_split_ids

SEED = 20260808
OUT = os.path.join(ROOT, "results", "hatemm_fp_audit")


def contact_sheet(vid, alias):
    frame_dir = os.path.join(DATASET_ROOTS["HateMM"], "frames_16", vid)
    paths = sorted(os.path.join(frame_dir, f) for f in os.listdir(frame_dir)
                   if f.lower().endswith(".jpg"))[:16]
    thumbs = []
    for path in paths:
        with Image.open(path) as im:
            x = im.convert("RGB"); x.thumbnail((256, 144)); thumbs.append(x.copy())
    sheet = Image.new("RGB", (4 * 256, 4 * 174), "white")
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(thumbs):
        x, y = (i % 4) * 256, (i // 4) * 174
        sheet.paste(im, (x, y)); draw.text((x + 4, y + 146), f"frame {i:02d}", fill="black")
    draw.text((4, 4), alias, fill="red")
    sheet.save(os.path.join(OUT, "contact_sheets", alias + ".jpg"), quality=88)


def main():
    os.makedirs(os.path.join(OUT, "contact_sheets"), exist_ok=True)
    z = load_z(os.path.join(ROOT, "results/testruns/hatemm/judge_8b/scores.jsonl"))
    ann = load_annotations("HateMM")
    ids = [v for v in load_clean_split_ids("HateMM", "test") if v in z]
    thr = kde_valley([z[v] for v in ids])["value"]
    fp = [v for v in ids if ann[v]["label"] == "Non Hate" and z[v] >= thr]
    if len(fp) != 70 or abs(thr - (-2.3455)) > 1e-6:
        raise SystemExit(f"cohort mismatch: n={len(fp)} threshold={thr}")
    rng = random.Random(SEED); shuffled = sorted(fp); rng.shuffle(shuffled)
    aliases = {vid: f"HFP-{i+1:03d}" for i, vid in enumerate(shuffled)}
    overrides = json.load(open(os.path.join(ROOT, "results/testruns/hatemm/c2_overrides.json")))
    packet = []
    for vid in shuffled:
        alias = aliases[vid]
        packet.append({"alias": alias, "title": ann[vid].get("title", "") or "",
                       "transcript": overrides.get(vid, ann[vid].get("transcript", "") or ""),
                       "contact_sheet": f"contact_sheets/{alias}.jpg"})
        contact_sheet(vid, alias)
    with open(os.path.join(OUT, "packet.json"), "w") as f:
        json.dump(packet, f, indent=2, ensure_ascii=False)
    # Mapping is isolated from the review packet and is not needed for coding.
    with open(os.path.join(OUT, "alias_map.json"), "w") as f:
        json.dump({aliases[v]: v for v in shuffled}, f, indent=2)
    template = {r["alias"]: {"primary": None, "confidence": None, "note": None,
                              "visual_reviewed": False} for r in packet}
    with open(os.path.join(OUT, "coding.json"), "w") as f:
        json.dump(template, f, indent=2)
    print(f"Wrote blinded packet for {len(packet)} items at threshold {thr}")


if __name__ == "__main__":
    main()
