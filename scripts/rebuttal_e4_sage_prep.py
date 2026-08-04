#!/usr/bin/env python
"""Rebuttal E4b — build SAGE-format inputs on OUR fixed splits.

Produces, under results/rebuttal/E4_sage/<ds>/:
  splits/{train,val,test}.csv   (HateMM, comma; cols video_file_name,label,video_path)
  splits/{train,val,test}.tsv   (MHClip,  tab;   cols Video_ID,Majority_Voting,Label)
  frames/<vid>_<i>.jpg          (i=1..16, uniform sampling — SAGE frame-extract logic)
  audios_16k/<vid>.wav          (HateMM only; 44.1k -> 16k for SAGE's MFCC)

Transcripts reuse each dataset's annotation(new).json in place (already the
list-of-dicts format SAGE's Runner expects); MHClip audio (already 16k) is used
in place. Nothing outside results/rebuttal/E4_sage/ is written.
"""
import os
import sys
import csv
import cv2
import numpy as np

REPO = "/data/jehc223/EMNLP2"
sys.path.insert(0, os.path.join(REPO, "src"))
from our_method import data_utils as du  # noqa: E402

OUT_ROOT = os.path.join(REPO, "results/rebuttal/E4_sage")
N_FRAMES = 16

# our-dataset-name -> (SAGE key, kind). ImpliHateVid is already binary
# {Hateful, Normal}, so it rides the mhclip loader path (TSV) with no collapse.
DATASETS = {
    "HateMM": "hatemm",
    "MHClip_EN": "mhclip",
    "MHClip_ZH": "mhclip",
    "ImpliHateVid": "mhclip",
}
# MHClip binary collapse: Hateful+Offensive -> positive. (IH has no Offensive.)
MHCLIP_POS = {"Hateful", "Offensive"}
# Audio prep per dataset: "resample" existing wavs to 16k, or "extract" from mp4.
# Datasets absent here already ship 16k wavs used in place (MHClip EN/ZH).
AUDIO_BUILD = {"HateMM": "resample", "ImpliHateVid": "extract"}


def sample_frames_uniform(video_path, num_frames):
    """SAGE preprocess/frame-extract.py sample_frames_uniform, verbatim logic."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < num_frames:
        num_frames = frame_count if frame_count > 0 else 0
    interval = frame_count / num_frames if num_frames else 0
    frames, next_idx = [], 0
    for i in range(frame_count):
        success, frame = cap.read()
        if not success:
            continue
        if i + 1 >= next_idx:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            next_idx += interval
        if len(frames) == num_frames:
            break
    cap.release()
    return frames


def frames_from_dir(frames_dir, num_frames):
    """Fallback: uniform pick of num_frames from pre-extracted frames/<vid>/*.jpg."""
    jpgs = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]
    )
    if not jpgs:
        return []
    idx = np.linspace(0, len(jpgs) - 1, num=num_frames).round().astype(int)
    out = []
    for k in idx:
        img = cv2.imread(jpgs[int(k)])
        if img is None:
            continue
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out


def write_frames(vid, frames, out_dir):
    """Write 16 frames as <vid>_1.jpg..<vid>_16.jpg; pad by repeating last frame."""
    if not frames:
        return 0
    while len(frames) < N_FRAMES:  # pad short videos so the loader keeps them
        frames.append(frames[-1])
    for i, fr in enumerate(frames[:N_FRAMES]):
        p = os.path.join(out_dir, f"{vid}_{i + 1}.jpg")
        cv2.imwrite(p, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    return N_FRAMES


def build_frames(ds, ids, out_dir):
    root, sub = du.DATASET_ROOTS[ds], du.MP4_SUBDIRS[ds]
    os.makedirs(out_dir, exist_ok=True)
    made = from_mp4 = from_dir = failed = 0
    for vid in ids:
        # resume: skip if all 16 already present
        if all(os.path.exists(os.path.join(out_dir, f"{vid}_{i}.jpg")) for i in range(1, N_FRAMES + 1)):
            made += 1
            continue
        mp4 = os.path.join(root, sub, f"{vid}.mp4")
        frames = []
        if os.path.isfile(mp4) and os.path.getsize(mp4) > 1000:
            frames = sample_frames_uniform(mp4, N_FRAMES)
            src = "mp4"
        if not frames:
            fdir = os.path.join(root, "frames", vid)
            if os.path.isdir(fdir):
                frames = frames_from_dir(fdir, N_FRAMES)
                src = "dir"
        n = write_frames(vid, frames, out_dir)
        if n == N_FRAMES:
            made += 1
            from_mp4 += src == "mp4"
            from_dir += src == "dir"
        else:
            failed += 1
            print(f"  [FRAME-FAIL] {ds} {vid} (got {len(frames)} frames)", flush=True)
    print(f"  frames: {made}/{len(ids)} ok (mp4={from_mp4}, dir={from_dir}, fail={failed})", flush=True)


def audio_dir_for(ds):
    """Where SAGE's loader should look for <vid>.wav (16k) for this dataset."""
    if ds in AUDIO_BUILD:
        return os.path.join(OUT_ROOT, ds, "audios_16k")
    return os.path.join(du.DATASET_ROOTS[ds], "audios")


def build_audio_16k(ds, ids, out_dir):
    """Resample existing wavs -> 16k mono for SAGE's MFCC (sr=16000)."""
    import soundfile as sf
    import librosa
    os.makedirs(out_dir, exist_ok=True)
    root = du.DATASET_ROOTS[ds]
    done = miss = 0
    for vid in ids:
        dst = os.path.join(out_dir, f"{vid}.wav")
        if os.path.exists(dst):
            done += 1
            continue
        srcw = os.path.join(root, "audios", f"{vid}.wav")
        if not os.path.isfile(srcw):
            miss += 1
            continue
        wav, sr = sf.read(srcw)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
        sf.write(dst, wav, 16000)
        done += 1
    print(f"  audio_16k: {done} written, {miss} missing wav", flush=True)


def build_audio_extract(ds, ids, out_dir):
    """Extract 16k mono wav from each mp4 (no audios dir ships with IH). Videos
    with no audio track get a 1s silent wav so the file exists and SAGE's loader
    keeps the video (its AudioEncoder falls back to zeros on silence)."""
    import subprocess
    import numpy as np
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    root, sub = du.DATASET_ROOTS[ds], du.MP4_SUBDIRS[ds]
    ok = silent = miss = 0
    for vid in ids:
        dst = os.path.join(out_dir, f"{vid}.wav")
        if os.path.exists(dst):
            ok += 1
            continue
        mp4 = os.path.join(root, sub, f"{vid}.mp4")
        if not (os.path.isfile(mp4) and os.path.getsize(mp4) > 1000):
            miss += 1
            continue
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", mp4, "-vn", "-ac", "1", "-ar", "16000",
             "-f", "wav", "-loglevel", "error", dst],
            capture_output=True, text=True)
        if r.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 44:
            ok += 1
        else:  # no audio track / decode failure -> 1s silence so the file exists
            sf.write(dst, np.zeros(16000, dtype="float32"), 16000)
            silent += 1
    print(f"  audio_extract: {ok} from-track, {silent} silent-fallback, {miss} missing mp4", flush=True)


def build_splits(ds, key):
    ann = du.load_annotations(ds)
    sp_dir = os.path.join(OUT_ROOT, ds, "splits")
    os.makedirs(sp_dir, exist_ok=True)
    all_ids = set()
    for our_split, fname in [("train", "train"), ("validation", "val"), ("test", "test")]:
        ids = [v for v in du.load_clean_split_ids(ds, our_split)
               if v not in du.SKIP_VIDEOS.get(ds, set())]
        all_ids.update(ids)
        if key == "hatemm":
            path = os.path.join(sp_dir, f"{fname}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["video_file_name", "label", "video_path"])
                for vid in ids:
                    lbl = ann[vid]["label"]  # "Hate" / "Non Hate"
                    mp4 = os.path.join(du.DATASET_ROOTS[ds], du.MP4_SUBDIRS[ds], f"{vid}.mp4")
                    w.writerow([f"{vid}.mp4", lbl, mp4])
        else:
            path = os.path.join(sp_dir, f"{fname}.tsv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["Video_ID", "Majority_Voting", "Label"])
                for vid in ids:
                    raw = ann[vid]["label"]  # Hateful / Offensive / Normal
                    mv = "Hateful" if raw in MHCLIP_POS else "Normal"
                    w.writerow([vid, mv, raw])
        print(f"  split {ds}/{fname}: {len(ids)} rows -> {os.path.basename(path)}", flush=True)
    return sorted(all_ids)


def verify(ds, key):
    """Re-load each split through SAGE's own loaders; every split video must
    survive (all 16 frames + wav present) so the fixed test N is preserved."""
    sys.path.insert(0, os.path.join(REPO, "external_repos/SAGE/model"))
    from utils import load_split_hatemm, load_split_mhclip  # noqa: E402
    import json
    ann_path = os.path.join(du.DATASET_ROOTS[ds], "annotation(new).json")
    trans = {}
    for item in json.load(open(ann_path)):
        if item.get("Video_ID"):
            trans[item["Video_ID"]] = item
    frame_dir = os.path.join(OUT_ROOT, ds, "frames")
    audio_dir = audio_dir_for(ds)
    ext = "csv" if key == "hatemm" else "tsv"
    ok = True
    for fname, expect in [("train", None), ("val", None), ("test", None)]:
        path = os.path.join(OUT_ROOT, ds, "splits", f"{fname}.{ext}")
        raw = sum(1 for _ in open(path)) - 1  # minus header
        if key == "hatemm":
            df = load_split_hatemm(path, trans, audio_dir, frame_dir, N_FRAMES)
        else:
            df = load_split_mhclip(path, trans, audio_dir, frame_dir, N_FRAMES, num_classes=2)
        n_nan = int(df["Label"].isna().sum())
        flag = "" if (len(df) == raw and n_nan == 0) else "  <<< DROP/NAN"
        if flag:
            ok = False
        print(f"  verify {ds}/{fname}: raw={raw} loaded={len(df)} nan_label={n_nan}{flag}", flush=True)
    print(f"  verify {ds}: {'OK' if ok else 'MISMATCH — investigate'}", flush=True)


def main():
    # optional argv: dataset names to prep (default all). Lets the director prep
    # only ImpliHateVid without re-touching the finished HateMM/MHClip assets.
    wanted = sys.argv[1:] or list(DATASETS)
    for ds in wanted:
        key = DATASETS[ds]
        print(f"===== {ds} ({key}) =====", flush=True)
        ids = build_splits(ds, key)
        build_frames(ds, ids, os.path.join(OUT_ROOT, ds, "frames"))
        mode = AUDIO_BUILD.get(ds)
        if mode == "resample":
            build_audio_16k(ds, ids, audio_dir_for(ds))
        elif mode == "extract":
            build_audio_extract(ds, ids, audio_dir_for(ds))
        verify(ds, key)
    print("PREP DONE", flush=True)


if __name__ == "__main__":
    main()
