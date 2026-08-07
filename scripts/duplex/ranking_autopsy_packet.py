"""Ranking-error autopsy, stage 2: build the misranking cohorts for manual coding.

Selects, per corpus and under the corpus's primary label collapse, the lowest-z
positives (false-negative side) and the highest-z negatives (false-positive
side), and writes a packet carrying the fresh transcript, the fine-grained
label, the label-free covariates and the frame paths for each.

The packet carries video ids and transcript text and therefore stays under
results/ (gitignored). Only aliases of the form EN-FN-07 reach the note.
"""

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
OUT = os.path.join(ROOT, "results", "ranking_autopsy")

FRAME_DIRS = {
    "en": "/home/jehc223/data/Multihateclip/English/frames_16",
    "hcs": "/home/jehc223/data/HateClipSeg/frames_16",
}
N = 25


def transcripts(corpus):
    path = {"en": os.path.join(ROOT, "results/testruns/mhclip_en/fresh_transcripts.jsonl"),
            "hcs": os.path.join(ROOT, "results/hateclipseg/fresh_transcripts.jsonl")}[corpus]
    d = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                d[o["video_id"]] = o
    return d


def overrides(corpus):
    """The transcript the judge actually saw, when the degeneracy gate accepted it."""
    path = {"en": os.path.join(ROOT, "results/testruns/mhclip_en/c2_overrides.json"),
            "hcs": os.path.join(ROOT, "results/hateclipseg/c2_overrides.json")}[corpus]
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def build(corpus, pos_key):
    items = json.load(open(os.path.join(OUT, corpus, "items.json")))
    tr, ov = transcripts(corpus), overrides(corpus)
    fdir = FRAME_DIRS[corpus]

    pos = sorted([i for i in items if i[pos_key] == 1], key=lambda i: i["z"])
    neg = sorted([i for i in items if i[pos_key] == 0], key=lambda i: -i["z"])
    tag = corpus.upper()

    packet = []
    for side, rows in (("FN", pos[:N]), ("FP", neg[:N])):
        for k, it in enumerate(rows, 1):
            v = it["video_id"]
            t = tr.get(v, {})
            fd = os.path.join(fdir, v)
            frames = sorted(os.listdir(fd)) if os.path.isdir(fd) else []
            rec = dict(it)
            rec.update({
                "alias": f"{tag}-{side}-{k:02d}",
                "side": side,
                "rank_in_side": k,
                "fresh_text": t.get("fresh_text", ""),
                "judge_saw_override": v in ov,
                "judge_text": ov.get(v, ""),
                "languages": t.get("languages"),
                "frame_dir": fd,
                "frames": [os.path.join(fd, f) for f in frames],
            })
            packet.append(rec)

    with open(os.path.join(OUT, corpus, "packet.json"), "w") as f:
        json.dump(packet, f, indent=2)

    # A compact human-readable view for coding.
    lines = []
    for r in packet:
        lines.append("=" * 78)
        lab = r.get("fine") or ",".join(r.get("labels", []))
        lines.append(f"{r['alias']}  z={r['z']}  label={lab}  "
                     f"victims={r.get('victims', '')}")
        lines.append(f"  dur={r.get('wav_duration')}s vad={r.get('vad_speech_frac')} "
                     f"fresh_chars={r.get('fresh_chars')} "
                     f"judge_chars={r.get('n_transcript_chars')} "
                     f"src={r.get('transcript_source')} lang={r.get('top_language')} "
                     f"langs={r.get('languages')}")
        if "segfrac_union" in r:
            lines.append(f"  segfrac_union={r['segfrac_union']:.3f} "
                         f"segfrac_hateful={r.get('segfrac_hateful', 0):.3f} "
                         f"nseg={r.get('n_segments')}")
        lines.append(f"  frames: {r['frame_dir']}  n={len(r['frames'])}")
        lines.append(f"  TRANSCRIPT: {r['fresh_text'][:3000]}")
        lines.append("")
    with open(os.path.join(OUT, corpus, "packet.txt"), "w") as f:
        f.write("\n".join(lines))
    return packet


if __name__ == "__main__":
    p = build("en", "binary")
    print("en packet", len(p))
    p = build("hcs", "union")
    print("hcs packet", len(p))
