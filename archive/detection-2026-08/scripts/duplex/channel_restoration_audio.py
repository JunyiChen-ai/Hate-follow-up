"""Channel restoration, stage A (CPU): audio extraction and voice activity.

For each of the 662 dismissed videos: ffprobe the container, decode a 16 kHz
mono wav capped at 30 minutes, and measure the silero-VAD speech fraction that
the frozen C2 degeneracy gate reads. Output is one JSON object per video,
appended and fsynced, so the stage resumes after a crash.

Pre-registration: docs/duplex/PREREG_channel_restoration.md.
"""

import json
import os
import subprocess
import sys
import time
import wave

CAP_SECONDS = 1800

OUT_DIR = os.environ.get(
    "CR_OUT_DIR", "/home/jehc223/Hate-follow-up/results/channel_restoration")
MP4_DIR = os.environ.get(
    "CR_MP4_DIR", "/home/jehc223/data/ImpliHateVid_video_dismissed")
WAV_DIR = os.path.join(OUT_DIR, "wav")
META_PATH = os.path.join(OUT_DIR, "audio_meta.jsonl")


def ffprobe(path):
    p = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True)
    d = json.loads(p.stdout or "{}")
    out = {"container_duration": None, "has_audio": False,
           "sample_rate": None, "channels": None, "codec": None}
    try:
        out["container_duration"] = float(d["format"]["duration"])
    except Exception:
        pass
    for s in d.get("streams", []):
        if s.get("codec_type") == "audio":
            out["has_audio"] = True
            out["sample_rate"] = s.get("sample_rate")
            out["channels"] = s.get("channels")
            out["codec"] = s.get("codec_name")
            break
    return out


def extract_wav(path, wav):
    p = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-t", str(CAP_SECONDS), "-i", path,
         "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav],
        capture_output=True, text=True)
    ok = p.returncode == 0 and os.path.isfile(wav) and os.path.getsize(wav) > 1000
    return ok, (p.stderr or "").strip()[:300]


def main():
    ids = json.load(open(sys.argv[1]))["dismissed_all"]
    os.makedirs(WAV_DIR, exist_ok=True)

    done = set()
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video_id"])
                except Exception:
                    continue
    remaining = [v for v in ids if v not in done]
    print(f"stage A: {len(ids)} total, {len(done)} done, {len(remaining)} remaining",
          flush=True)
    if not remaining:
        return

    import torch
    torch.set_num_threads(8)
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                  force_reload=False, trust_repo=True)
    get_speech_timestamps, _, read_audio = utils[0], utils[1], utils[2]

    t0 = time.time()
    fh = open(META_PATH, "a")
    for i, vid in enumerate(remaining, 1):
        mp4 = os.path.join(MP4_DIR, vid + ".mp4")
        r = {"video_id": vid, "mp4_exists": os.path.isfile(mp4)}
        if r["mp4_exists"]:
            r.update(ffprobe(mp4))
            r["hit_duration_cap"] = bool(
                r["container_duration"] and r["container_duration"] > CAP_SECONDS + 1)
            wav = os.path.join(WAV_DIR, vid + ".wav")
            if r["has_audio"]:
                ok, err = extract_wav(mp4, wav)
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
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        if i % 25 == 0 or i == 1:
            el = time.time() - t0
            print(f"  [{i}/{len(remaining)}] {vid} dur={r.get('wav_duration')} "
                  f"vad={r.get('vad_speech_frac')} {i/el:.2f} clip/s", flush=True)
    fh.close()
    print(f"stage A done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
