#!/usr/bin/env python3
"""Qwen3-8B reading Whisper transcript segments (run_plan.md L2), HateMM, HateClipSeg, DeHate.

Unit: one Whisper large-v3 segment of `data/asr_whisper_large_v3/<DS>/timestamped_chunks.jsonl`, scored alone (no
other segment, no frames, no title). Untimed segments are kept (run_plan.md §1.3): a missing end becomes the next
segment's start or the video duration, a missing start the previous segment's end (0 for the first).

Prompt (shared hate query, `hate_query.md` §1; rules copied verbatim from `experiments/20260910_spvl/spvl.py`
`YOUTUBE_RULES`, no import; system message = spvl.py `SYSTEM_MESSAGE`):
  system: You are a content moderation analyst. Answer based strictly on observable evidence.
  user:   Based on this platform's hate speech policy, the following types of content are prohibited:
          <9 rules>

          Transcript of a speech segment from a video: <segment text>

          Does this segment contain content that violates any of the above rules? Answer "Yes" or "No".
Chat template with `enable_thinking=False` (Qwen3), so the first generated token is the answer.
Score: logsumexp over the Yes token ids minus logsumexp over the No token ids of the next-token logits, Yes/No id
sets built as in spvl.py `binary_ids` (Yes, " Yes", yes, " yes", YES, " YES"), lm_head applied in fp32 on the last
hidden state (spvl.py `margins_fp32`). One forward per segment, no padding, bf16 weights.

4 fps grid: frame i (centre (i + 0.5) / 4 s) takes the score of the segment containing its centre; overlapping
segments -> maximum. F1: a frame outside every segment takes the score of the same prompt with "(no speech)" as
the segment text (one constant). F4: a video with no segment is constant at that score.
No labels are read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/20261008_baselines"))
import exact_cohort as ec  # noqa: E402

CODE_PATH = "experiments/20261008_baselines/qwen3_text/text_llm.py"
MODEL = "Qwen/Qwen3-8B"
FPS = 4.0
OUT_ROOT = ec.OUT_ROOT / "qwen3_8b_text"

SYSTEM_MESSAGE = "You are a content moderation analyst. Answer based strictly on observable evidence."
YOUTUBE_RULES = """1. Encourage violence against groups based on protected status
2. Incite hatred against groups based on protected status
3. Dehumanize groups by comparing to non-human entities
4. Praise or glorify violence against protected groups
5. Use slurs and stereotypes that incite hatred (including through humor, irony, or coded language)
6. Claim groups are inferior based on protected status
7. Promote hateful supremacism
8. Make conspiratorial claims about groups based on protected status
9. Deny or minimize major violent events"""
QUESTION = 'Does this segment contain content that violates any of the above rules? Answer "Yes" or "No".'
EMPTY = "(no speech)"


def user_text(segment: str) -> str:
    body = segment.strip() if segment and segment.strip() else EMPTY
    return ("Based on this platform's hate speech policy, the following types of content are prohibited:\n"
            f"{YOUTUBE_RULES}\n\nTranscript of a speech segment from a video: {body}\n\n{QUESTION}")


def load_segments(ds: str) -> dict[str, list[tuple[float, float, str]]]:
    dur = ec.durations(ds)
    out = {}
    with open(REPO / f"data/asr_whisper_large_v3/{ds}/timestamped_chunks.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            v = r["video_id"]
            if v not in dur:
                continue
            ch = r.get("chunks") or []
            segs, prev_end = [], 0.0
            for i, c in enumerate(ch):
                s = c.get("start")
                e = c.get("end")
                s = float(s) if s is not None else prev_end
                if e is None:
                    nxt = next((float(x["start"]) for x in ch[i + 1:] if x.get("start") is not None), None)
                    e = nxt if nxt is not None and nxt > s else dur[v]
                e = float(e)
                text = c.get("text") or ""
                segs.append((s, e, text))
                prev_end = max(prev_end, e)
            out[v] = segs
    return out


class Scorer:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0",
                                                          attn_implementation="sdpa").eval()
        self.dev = self.model.device

        def first(s):
            ids = self.tok.encode(s, add_special_tokens=False)
            return ids[0] if ids else None
        sets = []
        for label in ("Yes", "No"):
            s = set()
            for v in (label, f" {label}", label.lower(), f" {label.lower()}", label.upper(), f" {label.upper()}"):
                t = first(v)
                if t is not None:
                    s.add(t)
            sets.append(sorted(s))
        self.yes_ids, self.no_ids = sets
        self.ids = torch.tensor(self.yes_ids + self.no_ids, device=self.dev)
        self.W = self.model.get_output_embeddings().weight[self.ids].float()

    def prompt(self, segment: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user_text(segment)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def __call__(self, segment: str) -> tuple[float, int]:
        torch = self.torch
        enc = self.tok(self.prompt(segment), return_tensors="pt", add_special_tokens=False).to(self.dev)
        with torch.no_grad():
            out = self.model.model(**enc, use_cache=False)
            h = out.last_hidden_state[0, -1:].float()
            lg = h @ self.W.T
        ny = len(self.yes_ids)
        z = float(torch.logsumexp(lg[0, :ny], 0) - torch.logsumexp(lg[0, ny:], 0))
        return z, int(enc["input_ids"].shape[1])


def paint(segs_scores, duration: float, empty_z: float) -> tuple[np.ndarray, int]:
    T = ec.curve_length(duration)
    c = np.full(T, -np.inf)
    centres = (np.arange(T) + 0.5) / FPS
    for (s, e, _), z in segs_scores:
        m = (centres >= s) & (centres < e)
        c[m] = np.maximum(c[m], z)
    n_f1 = int(np.isneginf(c).sum())
    c[np.isneginf(c)] = empty_z
    return c, n_f1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg", "DeHate"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N cohort videos, no evaluation")
    a = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    glog = ec.RunLog(OUT_ROOT / "run.log", append=True)
    glog(f"code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
    sc = Scorer()
    import transformers
    glog(f"model {MODEL} transformers {transformers.__version__} torch {sc.torch.__version__} "
         f"yes ids {sc.yes_ids} no ids {sc.no_ids}")
    glog("prompt example:\n" + sc.prompt("<segment text>"))
    empty_z, _ = sc(EMPTY)
    glog(f"empty-input score (F1/F4 constant) {empty_z:.6f}")
    rc = 0
    for ds in a.datasets:
        out = OUT_ROOT / ds
        out.mkdir(parents=True, exist_ok=True)
        log = ec.RunLog(out / "run.log")
        (out / "run.pid").write_text(f"{os.getpid()}\n")
        log(f"code {CODE_PATH} ({ec.code_version()}); command {' '.join(sys.argv)}")
        segs = load_segments(ds)
        ids = ec.cohort(ds)
        if a.limit:
            ids = ids[:a.limit]
        dur = ec.durations(ds)
        raw_path = out / "segment_scores.jsonl"
        curves, extra = {}, {}
        n_seg = n_f1 = n_f4 = n_untimed = 0
        t0 = time.time()
        with raw_path.open("w") as fh:
            for k, v in enumerate(ids, 1):
                ss = []
                for s, e, text in segs.get(v, []):
                    z, ntok = sc(text)
                    ss.append(((s, e, text), z))
                    fh.write(json.dumps({"video_id": v, "start": s, "end": e, "z": z, "n_tokens": ntok}) + "\n")
                n_seg += len(ss)
                c, nf = paint(ss, dur[v], empty_z)
                curves[v] = c
                extra[v] = {"n_segments": len(ss), "f1_frames": nf}
                n_f1 += nf
                if not ss:
                    n_f4 += 1
                    extra[v]["fallback"] = {"code": "F4", "reason": "no transcript segment", "value": empty_z}
                if k % 50 == 0:
                    log(f"progress {k}/{len(ids)} segments {n_seg} {time.time() - t0:.0f}s")
        log(f"{ds}: {len(ids)} videos, {n_seg} segments, F1 frames {n_f1}, F4 videos {n_f4}, "
            f"{time.time() - t0:.0f}s")
        if a.limit:
            log("smoke test: not evaluated")
            continue
        rep = ec.finalize("qwen3_8b_text_segments", ds, out, curves, native_rate=FPS, code_path=CODE_PATH, log=log,
                          extra=extra, notes={"segments": n_seg, "f1_frames": n_f1, "f4_videos": n_f4,
                                              "empty_input_score": empty_z},
                          config={"model": MODEL, "thinking": "off (enable_thinking=False)",
                                  "unit": "one Whisper large-v3 segment, scored alone",
                                  "transcripts": f"data/asr_whisper_large_v3/{ds}/timestamped_chunks.jsonl",
                                  "untimed_segments": "kept; end <- next start or duration; start <- previous end",
                                  "system": SYSTEM_MESSAGE, "user_template": user_text("<segment text>"),
                                  "score": "logsumexp(Yes ids) - logsumexp(No ids) at the first answer token",
                                  "yes_ids": sc.yes_ids, "no_ids": sc.no_ids,
                                  "grid": "frame centre inside a segment -> its score; overlaps -> max",
                                  "fallback_F1_F4": f"empty-input score {empty_z:.6f} ('{EMPTY}')"})
        rc |= 0 if rep.get("exact_test_set") else 6
    glog("DONE" if rc == 0 else f"FAILED rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
