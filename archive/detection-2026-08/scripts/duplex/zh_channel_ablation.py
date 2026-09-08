"""Ranking-error autopsy for MHClip-ZH, optional check: which text channel carries
the ranking?

Diagnostic, not preregistered and not a method. The judge's prompt carries three
inputs: sixteen frames, the shipped title and the transcript. The autopsy found
that 42 percent of MHClip-ZH titles carry the harvester's highlighted query term
and that the term is an offensive word, so the title is a candidate shortcut. It
also found that a fifth of transcripts are recogniser artefacts, so the
transcript is a candidate dead channel. Two one-call arms settle both:

  NOTITLE      the title field is blank; frames and transcript unchanged
  NOTRANSCRIPT the transcript field is blank; frames and title unchanged
  NOMARKUP     the title keeps every word but loses the harvester's HTML
               highlighting, which separates "the judge reads the title" from
               "the judge reads an artefact of how the corpus was collected"
  FRAMESONLY   both text fields blank, so the sixteen frames stand alone

Everything else is the frozen object: src/duplex/extract_duplex_readout.py is
imported and driven, never reimplemented, and the only change is made by
replacing what its annotation loader returns.

Usage:  python scripts/duplex/zh_channel_ablation.py notitle
        python scripts/duplex/zh_channel_ablation.py notranscript
        python scripts/duplex/zh_channel_ablation.py nomarkup
"""

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "duplex"))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

RUN = os.path.join(ROOT, "results", "testruns", "mhclip_zh")
OUT = os.path.join(ROOT, "results", "ranking_autopsy", "zh", "arms")


def main():
    arm = sys.argv[1]
    if arm not in ("notitle", "notranscript", "nomarkup", "framesonly"):
        raise SystemExit("arm must be notitle, notranscript, nomarkup or framesonly")

    import extract_duplex_readout as ed
    from data_utils import load_clean_split_ids

    real_loader = ed.load_annotations
    if arm == "notitle":
        def patched(dataset):
            ann = real_loader(dataset)
            for v in ann:
                ann[v] = dict(ann[v], title="")
            return ann
        ed.load_annotations = patched
        overrides = os.path.join(RUN, "c2_overrides.json")
    elif arm == "nomarkup":
        tag = re.compile(r"</?em[^>]*>")

        def patched(dataset):
            ann = real_loader(dataset)
            for v in ann:
                ann[v] = dict(ann[v], title=tag.sub("", ann[v].get("title", "") or ""))
            return ann
        ed.load_annotations = patched
        overrides = os.path.join(RUN, "c2_overrides.json")
    else:
        if arm == "framesonly":
            def patched(dataset):
                ann = real_loader(dataset)
                for v in ann:
                    ann[v] = dict(ann[v], title="")
                return ann
            ed.load_annotations = patched
        ids = load_clean_split_ids("MHClip_ZH", "test")
        overrides = os.path.join(OUT, "blank_overrides.json")
        os.makedirs(OUT, exist_ok=True)
        with open(overrides, "w") as f:
            json.dump({v: "" for v in ids}, f)

    out_dir = os.path.join(OUT, arm)
    sys.argv = ["extract_duplex_readout.py",
                "--dataset", "MHClip_ZH", "--split", "test",
                "--model", "Qwen/Qwen3-VL-8B-Instruct",
                "--transcript-limit", "0",
                "--transcript-override-json", overrides,
                "--out-dir", out_dir]
    ed.main()


if __name__ == "__main__":
    main()
