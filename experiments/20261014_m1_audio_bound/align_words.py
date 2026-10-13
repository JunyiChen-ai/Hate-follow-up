#!/usr/bin/env python3
"""Word-level timestamps for the Whisper transcripts by CTC forced alignment (candidate 43 control `speech_aligned`).

For every test video: the words of the Whisper chunks (in order, original strings) are aligned to the 16 kHz audio
(`data/audio16k/<dataset>/<video_id>.wav`) with torchaudio's MMS_FA bundle (wav2vec2 CTC, 20 ms frames). Emissions
are computed on 30 s chunks of the waveform and concatenated; the alignment runs once over the whole sequence.
Words without a letter a–z (numbers, symbols) are aligned as the wildcard token `*`. Output
`data/asr_whisper_aligned/<dataset>/words.jsonl`: one record per video with `words: [{w, s, e, star}]`,
`status` (`aligned`, `no_text`, `failed`) and the chunk constant. No labels read. Rule 5 input.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.video_inputs import load_manifest  # noqa: E402

DATASETS = ("HateMM", "HateClipSeg")
CHUNK_S = 30.0
SR = 16000
OUT = ROOT / "data/asr_whisper_aligned"


def words_of(dataset):
    raw = {}
    for line in open(ROOT / f"data/asr_whisper_large_v3/{dataset}/timestamped_chunks.jsonl"):
        r = json.loads(line)
        raw[r["video_id"]] = [w for c in (r.get("chunks") or []) for w in (c.get("text") or "").split()]
    return raw


def normalise(word):
    s = re.sub(r"[^a-z']", "", word.lower().replace("’", "'"))
    return s if re.search(r"[a-z]", s) else "*"


def emissions(model, wav, device):
    out = []
    step = int(CHUNK_S * SR)
    with torch.inference_mode():
        for a in range(0, wav.shape[1], step):
            piece = wav[:, a:a + step]
            if piece.shape[1] < 400:  # shorter than one analysis window: pad with zeros
                piece = torch.nn.functional.pad(piece, (0, 400 - piece.shape[1]))
            e, _ = model(piece.to(device))
            out.append(e[0].cpu())
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N videos per dataset")
    ap.add_argument("--only", default="", help="smoke: one video id")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=True).to(device).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    rows = load_manifest(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl", a.datasets)
    if a.only:
        rows = [r for r in rows if r["video_id"] == a.only]
    elif a.limit:
        rows = [r for ds in a.datasets for r in [x for x in rows if x["dataset"] == ds][:a.limit]]
    raw = {ds: words_of(ds) for ds in a.datasets}
    records = {ds: [] for ds in a.datasets}
    t0 = time.perf_counter()
    for k, r in enumerate(rows, 1):
        ds, vid = r["dataset"], r["video_id"]
        words = raw[ds].get(vid, [])
        rec = dict(video_id=vid, dataset=ds, n_words=len(words), chunk_seconds=CHUNK_S, frame_seconds=0.02, words=[])
        if not words or all(normalise(w) == "*" for w in words):
            rec["status"] = "no_text"
        else:
            try:
                wav, sr = torchaudio.load(str(ROOT / "data/audio16k" / ds / f"{vid}.wav"))
                assert sr == SR and wav.shape[0] == 1
                em = emissions(model, wav, device)
                tokens = tokenizer([normalise(w) for w in words])
                spans = aligner(em, tokens)
                frame = wav.shape[1] / em.shape[0] / SR  # seconds per emission frame (about 0.02)
                assert len(spans) == len(words)
                rec["frame_seconds"] = frame
                rec["words"] = [dict(w=w, s=round(sp[0].start * frame, 3), e=round(sp[-1].end * frame, 3), star=normalise(w) == "*")
                                for w, sp in zip(words, spans)]
                rec["status"] = "aligned"
                rec["audio_seconds"] = wav.shape[1] / SR
            except Exception as ex:  # noqa: BLE001
                rec["status"] = "failed"
                rec["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
        records[ds].append(rec)
        if k % 20 == 0 or a.only or a.limit:
            print(k, len(rows), ds, vid, rec["status"], rec["n_words"], f"{time.perf_counter() - t0:.0f}s", flush=True)
    tag = "" if not (a.only or a.limit) else "_smoke"
    for ds in a.datasets:
        d = OUT / ds
        d.mkdir(parents=True, exist_ok=True)
        (d / f"words{tag}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records[ds]))
        st = {}
        for x in records[ds]:
            st[x["status"]] = st.get(x["status"], 0) + 1
        print(ds, st, flush=True)
    print("ALIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
