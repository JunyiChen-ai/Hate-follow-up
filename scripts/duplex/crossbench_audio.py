"""Cross-benchmark stage A (CPU): audio extraction and voice activity.

The ImpliHateVid channel-restoration stage A, generalized over the dataset and
the mp4 directory. `ffprobe`, `extract_wav` and the 1800-second cap are imported
from `scripts/duplex/channel_restoration_audio.py`, which is unmodified, so the
decode path and the VAD measurement are identical across benchmarks.

Videos whose mp4 is absent are still recorded, with `mp4_exists` false, so the
coverage gap is visible in the output rather than silently shrinking the
denominator.

Usage:
  python scripts/duplex/crossbench_audio.py --dataset HateMM \
      --mp4-dir /path/to/mp4 --out-dir results/crossbench/hatemm
"""

import argparse
import json
import os
import sys
import time
import wave

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from channel_restoration_audio import CAP_SECONDS, extract_wav, ffprobe  # noqa: E402
from data_utils import load_clean_split_ids  # noqa: E402

MP4_EXTS = (".mp4", ".mkv", ".webm", ".flv", ".avi", ".mov", ".m4v")


def find_media(mp4_dir, vid):
    """First existing <mp4_dir>/<vid><ext> over the accepted container types."""
    if not mp4_dir:
        return None
    for ext in MP4_EXTS:
        p = os.path.join(mp4_dir, vid + ext)
        if os.path.isfile(p) and os.path.getsize(p) > 1000:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--mp4-dir", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    wav_dir = os.path.join(args.out_dir, "wav")
    meta_path = os.path.join(args.out_dir, "audio_meta.jsonl")
    os.makedirs(wav_dir, exist_ok=True)

    ids = load_clean_split_ids(args.dataset, args.split)
    done = set()
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video_id"])
                except Exception:
                    continue
    remaining = [v for v in ids if v not in done]
    present = [v for v in remaining if find_media(args.mp4_dir, v)]
    print(f"stage A [{args.dataset}]: {len(ids)} ids, {len(done)} done, "
          f"{len(remaining)} remaining, {len(present)} of those have media "
          f"under {args.mp4_dir or '(no mp4 dir configured)'}", flush=True)
    if not remaining:
        return

    model = None
    if present:
        import torch
        torch.set_num_threads(8)
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                      force_reload=False, trust_repo=True)
        get_speech_timestamps, _, read_audio = utils[0], utils[1], utils[2]

    t0 = time.time()
    fh = open(meta_path, "a")
    for i, vid in enumerate(remaining, 1):
        src = find_media(args.mp4_dir, vid)
        r = {"video_id": vid, "mp4_exists": src is not None,
             "source_path_basename": os.path.basename(src) if src else None}
        if src:
            r.update(ffprobe(src))
            r["hit_duration_cap"] = bool(
                r["container_duration"] and r["container_duration"] > CAP_SECONDS + 1)
            wav = os.path.join(wav_dir, vid + ".wav")
            if r["has_audio"]:
                ok, err = extract_wav(src, wav)
                r["wav_ok"] = ok
                if err:
                    r["ffmpeg_stderr"] = err
            else:
                r["wav_ok"] = False
            if r.get("wav_ok"):
                with wave.open(wav) as w:
                    dur = w.getnframes() / w.getframerate()
                r["wav_duration"] = round(dur, 2)
                audio = read_audio(wav, sampling_rate=16000)
                ts = get_speech_timestamps(audio, model, sampling_rate=16000)
                speech = sum(t["end"] - t["start"] for t in ts) / 16000.0
                r["vad_segments"] = len(ts)
                r["vad_speech_sec"] = round(speech, 2)
                r["vad_speech_frac"] = round(speech / dur, 6) if dur else None
        else:
            r["wav_ok"] = False
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        if i % 50 == 0 or i == 1:
            el = time.time() - t0
            print(f"  [{i}/{len(remaining)}] {vid} mp4={r['mp4_exists']} "
                  f"dur={r.get('wav_duration')} vad={r.get('vad_speech_frac')} "
                  f"{i / max(el, 1e-9):.2f} clip/s", flush=True)
    fh.close()
    print(f"stage A [{args.dataset}] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
