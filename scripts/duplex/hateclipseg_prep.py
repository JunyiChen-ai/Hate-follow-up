"""HateClipSeg preparation: dataset root, annotation, split, uniform-16 frames.

Builds the standard benchmark layout that `src/our_method/data_utils.py`
expects, so the frozen c2 test pipeline (audio -> Whisper -> degeneracy gate ->
uniform-16 frames -> single 8B judge call) runs on HateClipSeg without any
change to its stages.

Three things happen here and nothing else:

1. `annotation(new).json` is written from the shipped video-level annotation
   csv. Title and Transcript are empty strings: HateClipSeg ships neither, and
   two of the four corpora already measured (HateMM, ImpliHateVid) also carry
   an empty Title. The binary label uses the collapse frozen in the B1 pilot
   code: primary `Label` is the offensive union (hateful, insulting, sexual,
   violence, harm -> Offensive), secondary `Label_hateful_strict` is the
   hateful-only collapse. `lexicons.json` is never read.
2. `video/<id>.mp4` symlinks point at the pilot's media directory, and
   `splits/test.csv` lists every annotated id. `load_clean_split_ids` then
   drops the ids with no media, which is where the media attrition shows up.
3. `frames_16/<id>/frame_NNN.jpg` is produced by the repo's own
   `src/match_repro/extract_frames.py` rule: 16 indices from
   `np.linspace(0, n_frames - 1, 16)` over the decoded frame sequence, written
   as JPEG quality 85. Videos whose codec the bundled decord build cannot open
   (AV1 and VP9 here) are re-done with the system ffmpeg under the identical
   index rule, so the frame grid is the same one every other corpus got.

Usage:
  python scripts/duplex/hateclipseg_prep.py annotate
  python scripts/duplex/hateclipseg_prep.py prune
  python scripts/duplex/hateclipseg_prep.py frames-fallback
  python scripts/duplex/hateclipseg_prep.py verify
"""

import ast
import csv
import json
import os
import subprocess
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

from data_utils import DATASET_ROOTS, load_clean_split_ids  # noqa: E402

PILOT = os.path.join(ROOT, "idea-stage", "pilots", "b1_coverage_audit", "data")
ANN_CSV = os.path.join(PILOT, "video_level_annotation.csv")
PILOT_VIDEOS = os.path.join(PILOT, "videos")
DST = DATASET_ROOTS["HateClipSeg"]

# Frozen from the B1 pilot code. Index 0 is `normal`; 1..5 are the offensive
# categories whose union is the primary collapse.
IDX = ["normal", "hateful", "insulting", "sexual", "violence", "harm"]
OFFENSIVE = set(IDX[1:])
NUM_FRAMES = 16
JPEG_QUALITY = 85


def video_labels():
    """video_id -> (offensive_union, hateful_strict), annotation order kept."""
    out = {}
    with open(ANN_CSV) as f:
        for row in csv.DictReader(f):
            vid = row["Video Id"].strip()
            labs = ast.literal_eval(row["Video-Level Label"])
            unknown = set(labs) - set(IDX)
            if unknown:
                raise SystemExit(f"unknown label token(s) {unknown}")
            out[vid] = (int(bool(set(labs) & OFFENSIVE)), int("hateful" in labs))
    return out


def stage_annotate():
    os.makedirs(os.path.join(DST, "splits"), exist_ok=True)
    os.makedirs(os.path.join(DST, "video"), exist_ok=True)
    labs = video_labels()
    ann = [{"Video_ID": v, "Title": "", "Transcript": "",
            "Label": "Offensive" if u else "Normal",
            "Label_hateful_strict": "Hateful" if h else "Normal"}
           for v, (u, h) in labs.items()]
    with open(os.path.join(DST, "annotation(new).json"), "w") as f:
        json.dump(ann, f, ensure_ascii=False)
    with open(os.path.join(DST, "splits", "test.csv"), "w") as f:
        for e in ann:
            f.write(e["Video_ID"] + "\n")
    for s in ("train", "validation"):
        open(os.path.join(DST, "splits", f"{s}.csv"), "w").close()

    n_link = 0
    for fn in sorted(os.listdir(PILOT_VIDEOS)):
        stem = os.path.splitext(fn)[0]
        dst = os.path.join(DST, "video", stem + ".mp4")
        if not os.path.exists(dst):
            os.symlink(os.path.join(PILOT_VIDEOS, fn), dst)
            n_link += 1
    n_u = sum(u for u, _ in labs.values())
    n_h = sum(h for _, h in labs.values())
    print(f"annotated={len(ann)} offensive_union={n_u} hateful_strict={n_h} "
          f"normal={len(ann) - n_u}; symlinked {n_link} new media files")


def stage_prune_undecodable():
    """Drop media whose video stream yields no decodable frame.

    One shipped file is a truncated download: the container advertises a
    274-second h264 stream but carries 138 kB and no decodable video packet, so
    decord and the system ffmpeg alike return nothing. The copy in the bucket is
    byte-identical, so the source file is broken rather than the transfer.
    Removing the link makes `generate_clean_splits` exclude it through the
    ordinary missing-media path, which keeps the exclusion reproducible without
    naming any video in tracked source.
    """
    vdir = os.path.join(DST, "video")
    removed = 0
    for fn in sorted(os.listdir(vdir)):
        p = os.path.join(vdir, fn)
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", p, "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True)
        if not r.stdout:
            os.remove(p)
            removed += 1
    print(f"prune: removed {removed} undecodable media link(s); "
          f"{len(os.listdir(vdir))} remain")


def _probe(path):
    """(n_frames, width, height). n_frames is counted exactly when the
    container does not carry nb_frames, which is the case for the AV1 and VP9
    files."""
    q = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames,width,height", "-of", "json", path]
    st = json.loads(subprocess.run(q, capture_output=True, text=True).stdout)
    st = (st.get("streams") or [{}])[0]
    w, h = int(st.get("width") or 0), int(st.get("height") or 0)
    try:
        n = int(st["nb_frames"])
    except (KeyError, TypeError, ValueError):
        q = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path]
        n = int(subprocess.run(q, capture_output=True, text=True).stdout.strip())
    return n, w, h


def _extract_ffmpeg(path, out_dir, n, w, h):
    """The extract_frames.py index rule, decoded by the system ffmpeg."""
    from PIL import Image
    if n <= NUM_FRAMES:
        indices = list(range(n)) + [n - 1] * (NUM_FRAMES - n)
    else:
        indices = np.linspace(0, n - 1, NUM_FRAMES, dtype=int).tolist()
    wanted = sorted(set(indices))
    sel = "+".join(f"eq(n\\,{i})" for i in wanted)
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='{sel}'",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True)
    buf = np.frombuffer(p.stdout, dtype=np.uint8)
    fsz = w * h * 3
    got = len(buf) // fsz
    if got < len(wanted):
        return f"only {got}/{len(wanted)} frames decoded"
    frames = {idx: buf[k * fsz:(k + 1) * fsz].reshape(h, w, 3)
              for k, idx in enumerate(wanted[:got])}
    os.makedirs(out_dir, exist_ok=True)
    for i, idx in enumerate(indices):
        Image.fromarray(frames[idx]).save(
            os.path.join(out_dir, f"frame_{i:03d}.jpg"), quality=JPEG_QUALITY)
    return None


def incomplete_ids():
    ids = load_clean_split_ids("HateClipSeg", "test")
    return [v for v in ids
            if not os.path.exists(os.path.join(
                DST, "frames_16", v, f"frame_{NUM_FRAMES - 1:03d}.jpg"))]


def stage_frames_fallback():
    todo = incomplete_ids()
    print(f"frames fallback: {len(todo)} videos without a complete frame set")
    failed = []
    for i, v in enumerate(todo, 1):
        src = os.path.join(DST, "video", v + ".mp4")
        out_dir = os.path.join(DST, "frames_16", v)
        try:
            n, w, h = _probe(src)
            if n <= 0 or w <= 0 or h <= 0:
                raise ValueError(f"probe gave n={n} w={w} h={h}")
            err = _extract_ffmpeg(src, out_dir, n, w, h)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"[:200]
        if err:
            failed.append((v, err))
        if i % 5 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} done, {len(failed)} failed", flush=True)
    print(f"frames fallback finished: {len(todo) - len(failed)} recovered, "
          f"{len(failed)} still failing")
    for _, e in failed:
        print(f"  failure: {e}")


def stage_verify():
    """Decode-check every frame of every split id with PIL open+load.

    A truncated JPEG once killed a judge run partway while the pipeline still
    reported success, so this must pass before any GPU stage starts.
    """
    from PIL import Image
    ids = load_clean_split_ids("HateClipSeg", "test")
    ok, bad, missing = [], [], []
    for v in ids:
        d = os.path.join(DST, "frames_16", v)
        paths = [os.path.join(d, f"frame_{i:03d}.jpg") for i in range(NUM_FRAMES)]
        if not all(os.path.isfile(p) for p in paths):
            missing.append(v)
            continue
        try:
            for p in paths:
                with Image.open(p) as im:
                    im.load()
        except Exception as exc:
            bad.append((v, f"{type(exc).__name__}: {exc}"[:120]))
            continue
        ok.append(v)
    res = {"n_split_ids": len(ids), "n_frame_sets_ok": len(ok),
           "n_frame_sets_missing": len(missing), "n_frame_sets_corrupt": len(bad),
           "frames_per_video": NUM_FRAMES}
    print(json.dumps(res, indent=1))
    for _, e in bad:
        print(f"  corrupt: {e}")
    out = os.path.join(ROOT, "results", "hateclipseg", "frame_precheck.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    if bad or missing:
        raise SystemExit("ABORT: frame pre-flight did not pass")
    print("frame pre-flight PASSED")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("annotate", "all"):
        stage_annotate()
    if stage in ("prune", "all"):
        stage_prune_undecodable()
    if stage in ("frames-fallback", "all"):
        stage_frames_fallback()
    if stage in ("verify", "all"):
        stage_verify()
