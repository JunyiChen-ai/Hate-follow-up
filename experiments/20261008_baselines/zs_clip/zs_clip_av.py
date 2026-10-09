#!/usr/bin/env python3
"""ZS-CLIP-AV: zero-shot CLIP on the frame and on the speech transcript (user decision 2026-10-09). The frame-only
run (`zs_clip.py`, runs/20261008_baselines/zs_clip/) is unchanged.

  frame       the frame-only run's per-frame value x = cos(img, "a hateful video frame") - cos(img, "a normal video
              frame") (CLIP ViT-L/14-336, 4 fps), read from its predictions.jsonl (HateMM / HateClipSeg:
              runs/20261008_baselines/zs_clip/predictions.jsonl, DeHate: runs/20261008_baselines/zs_clip/DeHate/
              predictions.jsonl), so the frame part is exactly the frame-only row. CLIP's zero-shot softmax over the
              two prompts with logit scale 100: p_frame = softmax(100 [cos_n, cos_h])_h = sigmoid(100 x).
  transcript  CLIP text embedding (`get_text_features`, same tokenizer and model as the campaign's prompt
              embeddings, truncation to 77 tokens, unit-normalised) of the Whisper transcript of the 8 s window
              containing the frame (windows [8w, 8w + 8) s, frame i -> window floor(i / 32); text from
              `transcript_windows.Transcripts`), against the text embeddings of "normal content" and the short
              positive query of hate_query.md section 2, "hateful content that attacks or demeans people based on
              protected attributes": p_text = softmax(100 [cos(t, normal), cos(t, hateful)])_hateful.
  score       (p_frame + p_text) / 2 in a window with speech; p_frame alone in a window without speech.
              No corpus statistic, no label.

Stages: `text` (CLIP text tower on the transcript windows; CPU is enough, ~20k short texts) and `score`
(combination + exact-cohort finalisation) -> runs/20261008_baselines/zs_clip_av/<DS>/. No labels are read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/20261008_baselines"))
import exact_cohort as ec  # noqa: E402
from transcript_windows import Transcripts  # noqa: E402

CODE_PATH = "experiments/20261008_baselines/zs_clip/zs_clip_av.py"
CLIP_MODEL = "openai/clip-vit-large-patch14-336"
PROMPT_EMB = REPO / "data/retrieval_hate_repro/repro_zs_clip/prompt_emb.npz"
FRAME_PRED = {"HateMM": ec.OUT_ROOT / "zs_clip/predictions.jsonl",
              "HateClipSeg": ec.OUT_ROOT / "zs_clip/predictions.jsonl",
              "DeHate": ec.OUT_ROOT / "zs_clip/DeHate/predictions.jsonl"}
OUT_ROOT = ec.OUT_ROOT / "zs_clip_av"
DATASETS = ("HateMM", "HateClipSeg", "DeHate")
NEG_TEXT = "normal content"
POS_TEXT = "hateful content that attacks or demeans people based on protected attributes"
LOGIT_SCALE = 100.0
FPS = 4.0
WIN_S = 8.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def frame_scores(ds: str) -> dict[str, np.ndarray]:
    out = {}
    with FRAME_PRED[ds].open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["dataset"] == ds:
                out[r["video_id"]] = np.asarray(r["score_curve"], dtype=np.float64)
    return out


class TextEncoder:
    """Retrieval-hate zs_clip.encode_texts: CLIPTokenizerFast (padding, truncation, max_length 77),
    CLIPModel.get_text_features, unit-normalised; float32."""

    def __init__(self, device: str):
        import torch
        from transformers import CLIPModel, CLIPTokenizerFast
        self.torch, self.device = torch, device
        self.tok = CLIPTokenizerFast.from_pretrained(CLIP_MODEL)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL, torch_dtype=torch.float32).to(device).eval()
        self.logit_scale = float(self.model.logit_scale.exp())

    def __call__(self, texts: list[str], bs: int = 256) -> tuple[np.ndarray, int]:
        torch = self.torch
        out = np.zeros((len(texts), self.model.config.projection_dim), dtype=np.float32)
        n_cut = 0
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i:i + bs]
                n_cut += sum(len(x) > 77 for x in self.tok(batch, truncation=False)["input_ids"])
                enc = self.tok(batch, padding=True, truncation=True, max_length=77, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                f = self.model.get_text_features(**enc)
                f = f / f.norm(dim=-1, keepdim=True)
                out[i:i + len(batch)] = f.float().cpu().numpy()
        return out, n_cut


def stage_text(a, log) -> int:
    import torch
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    enc = TextEncoder(dev)
    log(f"CLIP {CLIP_MODEL} on {dev}; logit scale exp(logit_scale) = {enc.logit_scale:.6f}")
    # pipeline check: the campaign's frame prompts re-encoded here must equal prompt_emb.npz
    pr = np.load(PROMPT_EMB, allow_pickle=True)
    chk, _ = enc([str(t) for t in pr["texts"][:2]])
    d = float(np.abs(chk - pr["emb"][:2]).max())
    log(f"text check vs campaign prompt_emb.npz ({list(pr['texts'][:2])}): max |diff| {d:.2e}")
    if d > 1e-4:
        log("FAILED text pipeline does not reproduce the campaign prompt embeddings")
        return 7
    anchors, _ = enc([NEG_TEXT, POS_TEXT])
    np.save(OUT_ROOT / "anchor_text_emb.npy", anchors)
    for ds in a.datasets:
        out_dir = OUT_ROOT / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        tr = Transcripts(ds)
        fr = frame_scores(ds)
        rows = []
        for v in ec.cohort(ds):
            T = len(fr[v])
            for w in range(int(np.ceil(T / (WIN_S * FPS)))):
                s = tr.span(v, w * WIN_S, (w + 1) * WIN_S)
                if s:
                    rows.append({"video_id": v, "window": w, "start": w * WIN_S, "end": (w + 1) * WIN_S, "text": s})
        t0 = time.time()
        E, n_cut = enc([r["text"] for r in rows])
        sims = E @ anchors.T
        p = sigmoid(LOGIT_SCALE * (sims[:, 1] - sims[:, 0])) if len(rows) else np.zeros(0)
        with (out_dir / "transcript_windows.jsonl").open("w") as fh:
            for r, x, sm in zip(rows, p, sims):
                fh.write(json.dumps({**r, "cos_normal": float(sm[0]), "cos_hateful": float(sm[1]),
                                     "p_hateful": float(x)}) + "\n")
        log(f"text {ds}: {len(rows)} windows with speech, {n_cut} longer than 77 tokens (truncated), "
            f"{time.time() - t0:.0f}s")
    return 0


def stage_score(a, log) -> int:
    rc = 0
    for ds in a.datasets:
        out = OUT_ROOT / ds
        dlog = ec.RunLog(out / "run.log", append=True)
        (out / "run.pid").write_text(f"{os.getpid()}\n")
        dlog(f"stage score; code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
        tw: dict[str, dict[int, float]] = {}
        with (out / "transcript_windows.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                tw.setdefault(r["video_id"], {})[int(r["window"])] = float(r["p_hateful"])
        fr = frame_scores(ds)
        curves, extra, fail = {}, {}, {}
        tot = {"frames": 0, "frames_with_transcript": 0, "videos_without_speech": 0}
        for v in ec.cohort(ds):
            if v not in fr:
                fail[v] = "no frame-only row"
                continue
            pf = sigmoid(LOGIT_SCALE * fr[v])
            T = len(pf)
            c = pf.copy()
            win = tw.get(v, {})
            w_idx = np.arange(T) // int(WIN_S * FPS)
            pt = np.array([win.get(int(w), np.nan) for w in w_idx])
            has = np.isfinite(pt)
            c[has] = 0.5 * (pf[has] + pt[has])
            curves[v] = c
            extra[v] = {"frames_with_transcript": int(has.sum())}
            tot["frames"] += T
            tot["frames_with_transcript"] += int(has.sum())
            tot["videos_without_speech"] += not win
        rep = ec.finalize("zs_clip_av", ds, out, curves, native_rate=FPS, code_path=CODE_PATH, log=dlog,
                          extra=extra, failures=fail, notes=tot,
                          config={"variant": "CLIP ViT-L/14-336 zero-shot, frame + transcript",
                                  "frame": f"{FRAME_PRED[ds].relative_to(REPO)} (x = cos_h - cos_n of 'a hateful "
                                           "video frame' / 'a normal video frame'); p_frame = sigmoid(100 x)",
                                  "transcript": "Whisper large-v3 text of the 8 s window [8w, 8w+8) containing the "
                                                "frame; CLIP get_text_features, truncation 77 tokens; windows: "
                                                "transcript_windows.jsonl",
                                  "transcript_anchors": [NEG_TEXT, POS_TEXT],
                                  "score": "mean(p_frame, p_text) with speech, p_frame without",
                                  "logit_scale": LOGIT_SCALE})
        rc |= 0 if rep.get("exact_test_set") else 6
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["text", "score"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log = ec.RunLog(OUT_ROOT / "run.log", append=True)
    (OUT_ROOT / "run.pid").write_text(f"{os.getpid()}\n")
    log(f"stage {a.stage}; code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
    rc = {"text": stage_text, "score": stage_score}[a.stage](a, log)
    log(f"stage {a.stage} {'DONE' if rc == 0 else f'FAILED rc={rc}'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
