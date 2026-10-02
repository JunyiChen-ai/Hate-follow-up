#!/usr/bin/env python3
"""Export selected case evidence and static plots; no method changes."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from analyze import ROOT, OUT, READS, FINALS, read, auc, curve, dump
from src.eval.evaluate import binary_intervals
from src.video_inputs import load_asr, frame_paths, window_text

CASES = [
    ("HateMM", "hate_video_299", "Largest final gain in HateMM: neutral conversation followed by explicit racial abuse."),
    ("HateClipSeg", "bit_O1kfCUve96Vz", "Second-largest HCS final gain: speech carries group-directed derogatory claims; visual reads fall."),
    ("HateClipSeg", "bit_Jl81Ars8alKV", "Largest HCS final gain, but news/topic example must not be called proof of improved hateful-intent understanding."),
    ("HateMM", "hate_video_279", "Largest HateMM final loss and incorrect global No; two-window clip with nearly empty ASR."),
    ("HateClipSeg", "bit_EH4buPGuFok7", "Largest HCS final loss despite correct global Yes; verbal abuse and intervening conversation."),
    ("HateMM", "hate_video_318", "Largest HateMM raw max AUC gain, but no final AUC gain: downstream model already separates boundaries."),
    ("DeHate", "wX1cU2jIm6sL", "Second-largest DeHate final gain: sex/gender-related commentary transitions between speakers."),
    ("DeHate", "jljPd8Ce1kGe", "Largest DeHate final loss among incorrect global No; raw ordering improves while final ordering worsens."),
]


def main():
    out = OUT / "analysis/figures"
    out.mkdir(exist_ok=True)
    reads = {k: tuple(read(p) for p in v) for k, v in READS.items()}
    finals = {k: tuple(read(p) for p in v) for k, v in FINALS.items()}
    evidence = []
    for ds, vid, selection in CASES:
        fam = "DeHate" if ds == "DeHate" else "main"
        a, b = (d[(ds, vid)] for d in reads[fam])
        fa, fb = (d[(ds, vid)] for d in finals[fam])
        gt = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        ix = list(map(str, gt["video_ids"])).index(vid)
        assert str(gt["split"][ix]) == "test"
        y = np.asarray(gt["y4"][ix], np.int8)
        n = min(len(y), len(fa["score_curve"]), len(fb["score_curve"]))
        y = y[:n]; t = np.arange(n) / 4
        spans = binary_intervals(y)
        asr = load_asr(ds, fill_untimed=ds == "DeHate").get(vid, [])
        ws, wn = a["extra"]["windows"], b["extra"]["windows"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        for ax, m, label in zip(axes[:2], ("z_visual", "z_speech"), ("Visual log odds", "Speech log odds")):
            ax.plot(t, curve(wn, m, n), color="#68737D", label="Without global turn", lw=1.5)
            ax.plot(t, curve(ws, m, n), color="#1673B1", label="With global turn", lw=1.5)
            ax.axhline(0, lw=.5, color="black"); ax.set_ylabel(label)
        for r, color, label in ((fb, "#68737D", "Without global turn"), (fa, "#1673B1", "With global turn")):
            s = np.array(r["score_curve"][:n]); s -= s.mean()
            axes[2].plot(t, s, color=color, label=label, lw=1.5)
        axes[2].set_ylabel("Final score\n(video mean removed)")
        for ax in axes:
            for start, end in spans:
                ax.axvspan(start, end, color="#B53527", alpha=.10)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[2].set_xlabel("Time (s); pale red = annotated positive interval")
        af, an = auc(y, fa["score_curve"][:n]), auc(y, fb["score_curve"][:n])
        yes = "Yes" if a["extra"]["z_video"] > 0 else "No"
        fig.suptitle(f"{ds} / {vid} | global {yes} | final within AUC {an:.3f} -> {af:.3f}")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{ds}_{vid}_scores.{ext}", dpi=160)
        plt.close(fig)
        fs = frame_paths(ds, vid, 20)
        sheet = Image.new("RGB", (1400, 740), "white"); dr = ImageDraw.Draw(sheet)
        for i, (tm, p) in enumerate(fs):
            im = Image.open(p).convert("RGB"); im.thumbnail((280, 160))
            x, yy = (i % 5) * 280, (i // 5) * 185
            sheet.paste(im, (x + (280-im.width)//2, yy)); dr.text((x+5, yy+163), f"{vid} | {tm:.1f}s", fill="black")
        sheet.save(out / f"{ds}_{vid}_frames.jpg")
        ew = []
        for w, z in zip(ws, wn):
            mask = (t >= w["start"]) & (t < w["end"])
            ew.append({"start": w["start"], "end": w["end"], "gt_fraction": float(y[mask].mean()) if mask.any() else None,
                       "full": w, "no": z, "transcript": window_text(asr, w["start"], w["end"])})
        evidence.append({"dataset": ds, "video_id": vid, "selection": selection, "global": yes,
                         "z_video": a["extra"]["z_video"], "final_auc_full": af, "final_auc_no": an,
                         "raw_auc_full": auc(y, curve(ws, "z", n)), "raw_auc_no": auc(y, curve(wn, "z", n)),
                         "GT_intervals": spans, "transcript": asr, "windows": ew,
                         "frames": [{"time": tm, "path": str(p.relative_to(ROOT))} for tm, p in fs]})
    dump(OUT / "analysis/selected_cases.json", evidence)
    print("CASE_EXPORT_DONE", len(evidence))


if __name__ == "__main__":
    main()
