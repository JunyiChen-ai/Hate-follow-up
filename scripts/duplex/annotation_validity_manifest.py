"""Build the blinded item manifest for the annotation-validity audit.

Frozen by docs/duplex/PREREG_annotation_validity_audit.md. Seed 20260808.

Two strata per corpus:
  HateClipSeg  : all 50 clean normals (target) + 25 sampled shipped positives
                 (blinding fillers).
  MHClip-EN    : all 49 union positives (target) + 40 sampled Normals
                 (blinding fillers).

Outputs, all under results/annotation_validity/ (gitignored):
  manifest.json        item_id -> corpus, video_id, stratum  (the unblinding key)
  packets/<batch>.md   one coder-facing batch: item ids, transcript text, frame
                       paths. Carries no stratum information and no video id.
Nothing here reads a judge score file.
"""

import csv
import ast
import json
import os
import random

ROOT = "/home/jehc223/Hate-follow-up"
DATA = "/home/jehc223/data"
OUT = os.path.join(ROOT, "results", "annotation_validity")
SEED = 20260808

HCS_IDX = ["normal", "hateful", "insulting", "sexual", "violence", "harm"]
HCS_OFFENSIVE = set(HCS_IDX[1:])
FRAME_PICKS = [0, 5, 10, 15]  # 4 uniformly spaced of the 16 shipped frames


def hcs_strata():
    labs = {}
    p = os.path.join(ROOT, "idea-stage", "pilots", "b1_coverage_audit",
                     "data", "video_level_annotation.csv")
    with open(p) as f:
        for row in csv.DictReader(f):
            labs[row["Video Id"].strip()] = ast.literal_eval(row["Video-Level Label"])
    clean = [l.strip() for l in
             open(os.path.join(DATA, "HateClipSeg", "splits", "test_clean.csv"))
             if l.strip()]
    normals = [v for v in clean if not (set(labs[v]) & HCS_OFFENSIVE)]
    positives = [v for v in clean if set(labs[v]) & HCS_OFFENSIVE]
    return normals, positives, labs, clean


def en_strata():
    ann = {x["Video_ID"]: x for x in json.load(
        open(os.path.join(DATA, "Multihateclip", "English", "annotation(new).json")))}
    clean = [l.strip() for l in
             open(os.path.join(DATA, "Multihateclip", "English",
                               "splits", "test_clean.csv")) if l.strip()]
    pos = [v for v in clean if ann[v]["Label"] in ("Hateful", "Offensive")]
    norm = [v for v in clean if ann[v]["Label"] == "Normal"]
    return pos, norm, ann


def transcripts(path):
    return {r["video_id"]: (r.get("fresh_text") or "")
            for r in (json.loads(l) for l in open(path))}


def frames(corpus, vid):
    root = (os.path.join(DATA, "HateClipSeg", "frames_16") if corpus == "HCS"
            else os.path.join(DATA, "Multihateclip", "English", "frames_16"))
    return [os.path.join(root, vid, f"frame_{i:03d}.jpg") for i in FRAME_PICKS]


def main():
    rng = random.Random(SEED)
    os.makedirs(os.path.join(OUT, "packets"), exist_ok=True)

    hcs_norm, hcs_pos, hcs_labs, hcs_clean = hcs_strata()
    en_pos, en_norm, en_ann = en_strata()
    hcs_norm, hcs_pos = sorted(hcs_norm), sorted(hcs_pos)
    en_pos, en_norm = sorted(en_pos), sorted(en_norm)

    hcs_fill = rng.sample(hcs_pos, 25)
    en_fill = rng.sample(en_norm, 40)

    items = ([{"corpus": "HCS", "video_id": v, "stratum": "target_clean_normal"}
              for v in hcs_norm]
             + [{"corpus": "HCS", "video_id": v, "stratum": "filler_shipped_positive"}
                for v in hcs_fill]
             + [{"corpus": "EN", "video_id": v, "stratum": "target_union_positive"}
                for v in en_pos]
             + [{"corpus": "EN", "video_id": v, "stratum": "filler_shipped_normal"}
                for v in en_fill])

    # Neutral ids assigned over the shuffled pooled list, so an id carries no
    # corpus or stratum information.
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["item_id"] = f"ITEM-{i:03d}"

    hcs_tx = transcripts(os.path.join(ROOT, "results", "hateclipseg",
                                      "fresh_transcripts.jsonl"))
    en_tx = transcripts(os.path.join(ROOT, "results", "testruns", "mhclip_en",
                                     "fresh_transcripts.jsonl"))

    for it in items:
        tx = hcs_tx if it["corpus"] == "HCS" else en_tx
        it["transcript"] = tx[it["video_id"]]
        it["frames"] = frames(it["corpus"], it["video_id"])
        it["shipped_label"] = (
            hcs_labs[it["video_id"]] if it["corpus"] == "HCS"
            else en_ann[it["video_id"]]["Label"])
        missing = [p for p in it["frames"] if not os.path.isfile(p)]
        if missing:
            raise SystemExit(f"missing frames for {it['item_id']}: {missing[0]}")

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump([{k: it[k] for k in
                    ("item_id", "corpus", "video_id", "stratum", "shipped_label")}
                   for it in items], f, indent=1)

    # Batches are corpus-homogeneous (the two coding schemes differ) but the
    # within-batch order is the shuffled pooled order, so strata stay mixed.
    batches = []
    for corpus, size in (("HCS", 13), ("EN", 13)):
        pool = [it for it in items if it["corpus"] == corpus]
        for k in range(0, len(pool), size):
            batches.append((corpus, pool[k:k + size]))

    index = []
    for n, (corpus, batch) in enumerate(batches, 1):
        name = f"batch_{n:02d}_{corpus}"
        lines = []
        for it in batch:
            t = " ".join(it["transcript"].split())
            if len(t) > 6000:
                t = t[:6000] + " [transcript truncated at 6000 characters]"
            lines.append(f"### {it['item_id']}\n\nFrames (read all four, in order):\n"
                         + "\n".join(f"- {p}" for p in it["frames"])
                         + f"\n\nTranscript (automatic speech recognition of the "
                           f"video's audio):\n\n{t if t.strip() else '[no speech detected]'}\n")
        with open(os.path.join(OUT, "packets", name + ".md"), "w") as f:
            f.write("\n".join(lines))
        index.append({"batch": name, "corpus": corpus,
                      "item_ids": [it["item_id"] for it in batch]})

    with open(os.path.join(OUT, "packets", "index.json"), "w") as f:
        json.dump(index, f, indent=1)

    print(f"items={len(items)} hcs_target={len(hcs_norm)} hcs_filler={len(hcs_fill)} "
          f"en_target={len(en_pos)} en_filler={len(en_fill)} batches={len(batches)}")
    for b in index:
        print(f"  {b['batch']}: {len(b['item_ids'])} items")


if __name__ == "__main__":
    main()
