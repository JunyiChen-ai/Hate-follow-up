#!/usr/bin/env python
"""CPU-only checks for the EventVAD port. Touches no GPU.

    CUDA_VISIBLE_DEVICES="" /home/jehc223/venvs/SafetyContradiction/bin/python \
        scripts/reproduction_baselines/smoke_cpu_eventvad.py

Five groups.

1. **The gaps are still there.** The port rests on three claims about the
   released code -- `graph_propagation` is imported and never defined, the
   scoring prompt is the literal string "prompt", the RAFT path is a
   placeholder -- plus the observation that `src/evaluate.py` does not compile.
   Those are checked against the pinned clone, so a clone at the wrong commit
   or an upstream that quietly fixed itself is caught rather than assumed.
2. **The substitutions are exact.** LAVIS's `clip_image_eval` transform is
   rebuilt and compared, step by step, with the vendored CLIP transform the
   port uses instead; the RAFT checkpoint's sha256 is verified; the flow
   projection is shown to be an isometry.
3. **The numerics.** Graph, propagation and boundary detection on synthetic
   features, through `segment_events.selftest`, plus the event-to-1 fps
   mapping and the prompt/parser through their own selftests.
4. **The cohorts.** Every gold video has a readable file, and the manifest
   counts are the 214 / 158 / 153 this study fixed.
5. **Real decode.** A handful of videos per corpus, chosen to include AV1 and
   HEVC, are decoded through the ffmpeg path and their frame counts checked
   against the gold length and the probed rate.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
EVENTVAD = os.path.join(HERE, "eventvad")
for _p in (HERE, EVENTVAD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hate_common import data as hdata            # noqa: E402
import boundary as bnd                           # noqa: E402
import config as cfgmod                          # noqa: E402
import features as feat                          # noqa: E402
import prompt as pmod                            # noqa: E402
import rasterize_and_eval as rast                # noqa: E402
import score_events as scr                       # noqa: E402
import segment_events as seg                     # noqa: E402
import video_io                                  # noqa: E402

UPSTREAM = os.path.join(PROJECT_ROOT, "third_party", "EventVAD")
UPSTREAM_SHA = "25cacd88a82af389776d2b397239f39961ac2d27"
EXPECTED_COHORT = {"hatemm": 214, "mhclip_en": 158, "mhclip_zh": 153}

FAILURES = []


def check(name, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print("  %s %s%s" % (mark, name, ("  -- " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)
    return bool(ok)


def head(title):
    print("\n=== %s ===" % title)


# ------------------------------------------------------------------ group 1
def check_gaps():
    head("1. the released gaps, against the pinned clone")
    if not os.path.isdir(os.path.join(UPSTREAM, ".git")):
        check("clone present", False, "%s missing; run clone_upstream.sh"
              % UPSTREAM)
        return
    sha = subprocess.run(["git", "-C", UPSTREAM, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    check("clone pinned at %s" % UPSTREAM_SHA[:7], sha == UPSTREAM_SHA, sha[:7])

    src = os.path.join(UPSTREAM, "src")
    uniseg = open(os.path.join(src, "event_seg", "uniseg_processor.py")).read()
    graphops = open(os.path.join(src, "event_seg", "graph_operations.py")).read()
    vidproc = open(os.path.join(src, "event_seg", "video_processing.py")).read()
    scorer = open(os.path.join(src, "score", "event_score.py")).read()
    extractor = open(os.path.join(src, "event_seg", "feature_extractor.py")).read()

    check("G1a: uniseg imports graph_propagation",
          "from graph_operations import graph_propagation" in uniseg)
    defined = any("def graph_propagation" in open(os.path.join(dp, f)).read()
                  for dp, _, fs in os.walk(src) for f in fs
                  if f.endswith(".py"))
    check("G1b: graph_propagation is defined nowhere in the release",
          not defined)
    check("G1c: graph_operations.py duplicates video_processing.py",
          graphops == vidproc)

    check("G2: the scoring prompt is the placeholder \"prompt\"",
          'abnormal_prompt = "prompt"' in scorer)
    check("G2b: the released parser is float(output.strip())",
          "float(output.strip())" in scorer)

    check("G3: the RAFT path is the placeholder /path/raft-things.pth",
          "'/path/raft-things.pth'" in extractor)

    ev = os.path.join(src, "evaluate.py")
    proc = subprocess.run([sys.executable, "-m", "py_compile", ev],
                          capture_output=True, text=True)
    check("G4: src/evaluate.py does not compile", proc.returncode != 0,
          (proc.stderr or proc.stdout).strip().splitlines()[-1][:80]
          if (proc.stderr or proc.stdout) else "")

    # The module the import names defines only process_video, so the import
    # cannot resolve and the released pipeline cannot start. Read with ast
    # rather than imported, because importing it would fail earlier on LAVIS,
    # which this port deliberately does not install.
    import ast
    top = [n.name for n in ast.parse(graphops).body
           if isinstance(n, ast.FunctionDef)]
    check("G1d: graph_operations defines only process_video",
          top == ["process_video"], str(top))


# ------------------------------------------------------------------ group 2
def check_substitutions():
    head("2. the substitutions the port makes for LAVIS and the RAFT path")

    # LAVIS clip_image_eval(224) == vendored clip._transform(224).
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    from hate_common.clip.clip import _transform

    lavis = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        lambda im: im.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    ours = _transform(224)
    names = lambda c: [type(t).__name__ for t in c.transforms]      # noqa: E731
    check("LAVIS and vendored CLIP transforms have the same step names",
          names(lavis) == names(ours), "%s" % names(ours))

    from PIL import Image
    rng = np.random.RandomState(0)
    img = Image.fromarray(rng.randint(0, 255, (321, 457, 3), dtype=np.uint8))
    a, b = lavis(img), ours(img)
    check("LAVIS and vendored CLIP transforms produce identical tensors",
          a.shape == b.shape and float((a - b).abs().max()) == 0.0,
          "max|diff|=%.3g shape=%s" % (float((a - b).abs().max()), tuple(a.shape)))

    exists, matches, sha = feat.check_raft_checkpoint()
    check("raft-things.pth present", exists, feat.DEFAULT_RAFT_CKPT)
    check("raft-things.pth sha256 matches princeton-vl's release", matches,
          sha or "")

    clip_ckpt = os.path.expanduser("~/.cache/clip/ViT-B-16.pt")
    ok = os.path.isfile(clip_ckpt)
    if ok:
        h = hashlib.sha256()
        with open(clip_ckpt, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        ok = h.hexdigest().startswith("5806e77c")
    check("OpenAI CLIP ViT-B/16 cached (the weights LAVIS resolves to)", ok,
          clip_ckpt)

    # The flow projection is an isometry, so the 128-d flow branch carries
    # exactly the two degrees of freedom of the mean flow vector.
    proj = feat.init_random_ortho(2, 128)
    u = rng.randn(200, 2).astype(np.float32)
    lifted = u @ proj
    d_lo = np.linalg.norm(u[:, None] - u[None, :], axis=2)
    d_hi = np.linalg.norm(lifted[:, None] - lifted[None, :], axis=2)
    check("the 2 -> 128 flow projection preserves distances",
          np.abs(d_lo - d_hi).max() < 1e-4,
          "max|diff|=%.3g" % np.abs(d_lo - d_hi).max())

    # VideoLLaMA2 is importable and does not need flash-attn to be built.
    clone = os.path.join(PROJECT_ROOT, "third_party", "VideoLLaMA2")
    ok = os.path.isdir(clone)
    check("VideoLLaMA2 clone present", ok, clone)
    if ok:
        proc = subprocess.run(
            [sys.executable, "-c",
             "import sys, warnings; warnings.filterwarnings('ignore'); "
             "sys.path.insert(0, %r); import videollama2; "
             "print(videollama2.model_init.__name__)" % clone],
            capture_output=True, text=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        check("videollama2 imports under torch 2.8 / transformers 4.57",
              proc.returncode == 0,
              proc.stderr.strip().splitlines()[-1][:90] if proc.stderr else "")
    ckpt = scr.DEFAULT_MODEL
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    ok = os.path.isfile(idx)
    check("VideoLLaMA2.1-7B-16F checkpoint present", ok, ckpt)
    if ok:
        wmap = json.load(open(idx))["weight_map"]
        n_vt = sum(1 for k in wmap if "vision_tower" in k)
        check("the SigLIP tower ships inside the checkpoint", n_vt > 0,
              "%d vision_tower tensors of %d" % (n_vt, len(wmap)))


# ------------------------------------------------------------------ group 3
def check_numerics():
    head("3. numerics: graph, propagation, boundaries, mapping, prompt")
    print(" -- segment_events.selftest --")
    if seg.selftest() != 0:
        FAILURES.append("segment_events.selftest")
    print(" -- rasterize_and_eval.selftest --")
    if rast.selftest() != 0:
        FAILURES.append("rasterize_and_eval.selftest")
    print(" -- score_events.selftest --")
    if scr.selftest() != 0:
        FAILURES.append("score_events.selftest")

    head("3b. the event grid maps onto the gold grid exactly")
    cfg = cfgmod.build_config("paper")
    # A whole video as one event, at each rate the corpora actually use.
    for fps in (24.0, 25.0, 29.97, 30.0):
        n_frames = int(round(100 * fps))
        events, _ = bnd.events_from_boundaries([], n_frames, fps, cfg)
        arr, filled = rast.rasterise(events, [0.42], 100, fps)
        check("fps %.2f: one event fills 100 gold seconds" % fps,
              arr.shape == (100,) and np.allclose(arr, 0.42) and filled == 0)
    # Every gold second must land inside some event, never past the last.
    rng = np.random.RandomState(1)
    for _ in range(200):
        fps = float(rng.choice([24.0, 25.0, 29.97, 30.0]))
        n_gold = int(rng.randint(5, 400))
        n_frames = int(round(n_gold * fps))
        cuts = sorted(rng.choice(np.arange(1, max(2, n_frames)),
                                 size=min(6, max(1, n_frames - 2)),
                                 replace=False))
        events, _ = bnd.events_from_boundaries(cuts, n_frames, fps, cfg)
        vals = list(rng.rand(len(events)))
        arr, _ = rast.rasterise(events, vals, n_gold, fps)
        if arr.shape != (n_gold,) or not np.isfinite(arr).all():
            check("randomised event/gold mapping", False,
                  "fps=%s n_gold=%s" % (fps, n_gold))
            return
        if not set(np.round(arr, 12)).issubset(set(np.round(vals, 12))):
            check("randomised event/gold mapping", False,
                  "score not drawn from the event set")
            return
    check("200 randomised event/gold mappings are total and finite", True)


# ------------------------------------------------------------------ group 4
def check_cohorts():
    head("4. cohorts and media")
    for corpus in hdata.CORPORA:
        ids, dropped = seg.cohort(corpus, "test")
        check("%s cohort is %d videos" % (corpus, EXPECTED_COHORT[corpus]),
              len(ids) == EXPECTED_COHORT[corpus],
              "%d ids, %d split ids without gold" % (len(ids), len(dropped)))
        missing = [v for v in ids
                   if not os.path.isfile(seg.video_path(corpus, v))]
        check("%s: every gold video has media" % corpus, not missing,
              "missing %s" % missing[:5])
        gt = hdata.gt_arrays(corpus, "test")
        bad = [v for v in ids if len(gt[v]) < 1]
        check("%s: every gold array is non-empty" % corpus, not bad,
              str(bad[:5]))


# ------------------------------------------------------------------ group 5
#: RAFT's correlation pyramid halves the H/8 feature map four times, so a side
#: under this collapses the coarsest level to width 1 and `bilinear_sampler`
#: divides by `W - 1`. See patch E10.
RAFT_MIN_SIDE = 128


def check_decode(per_corpus=4):
    head("5. real decode through the ffmpeg path")
    cfg = cfgmod.build_config("paper")
    for corpus in hdata.CORPORA:
        ids, _ = seg.cohort(corpus, "test")
        gt = hdata.gt_arrays(corpus, "test")
        picks = [ids[i] for i in
                 np.linspace(0, len(ids) - 1, per_corpus * 3, dtype=int)]
        codecs = {}
        for vid in picks:
            path = seg.video_path(corpus, vid)
            res = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name", "-of",
                 "default=nw=1:nk=1", path], capture_output=True, text=True)
            codecs.setdefault(res.stdout.strip(), []).append(vid)
        # one video per distinct codec, so AV1 and HEVC are always exercised
        chosen = [v[0] for v in codecs.values()]
        chosen += [v for v in picks if v not in chosen][:max(0, per_corpus
                                                             - len(chosen))]
        print("  %s codecs seen: %s"
              % (corpus, {k: len(v) for k, v in codecs.items()}))
        for vid in chosen:
            path = seg.video_path(corpus, vid)
            try:
                pr = video_io.probe(path, cfg)
                n = 0
                first = None
                for frame in video_io.iter_frames(pr):
                    if first is None:
                        first = frame
                    n += 1
                    if n >= 400:
                        break
                expect = min(400, int(pr.duration * pr.decode_fps) + 8)
                ok = (n > 0 and first is not None
                      and first.shape == (pr.out_height, pr.out_width, 3)
                      and first.dtype == np.uint8 and n <= expect + 8)
                ok = ok and min(pr.out_width, pr.out_height) >= RAFT_MIN_SIDE
                check("%s/%s decodes (%s)" % (corpus, vid,
                                              pr.decode_fps and "%dx%d @%.2f"
                                              % (pr.out_width, pr.out_height,
                                                 pr.decode_fps)),
                      ok, "%d frames, gold %ds" % (n, len(gt[vid])))
            except Exception as exc:                         # noqa: BLE001
                check("%s/%s decodes" % (corpus, vid), False,
                      "%s: %s" % (type(exc).__name__, exc))


# ------------------------------------------------------------------ group 6
def check_models():
    head("6. CLIP and RAFT load and run on real frames (CPU)")
    import torch
    try:
        fx = feat.FeatureExtractor(device="cpu", raft_iters=4, chunk_size=8)
    except Exception as exc:                                 # noqa: BLE001
        check("FeatureExtractor loads", False, "%s: %s" % (type(exc).__name__, exc))
        return
    check("FeatureExtractor loads CLIP ViT-B/16 and RAFT", True)
    check("CLIP runs in fp32, as Config.fp16_enabled = False",
          next(fx.clip_model.parameters()).dtype == torch.float32)

    # Regression for patch E10. RAFT sizes its coordinate grid H//8 x W//8 but
    # its encoder produces ceil(H/8) x ceil(W/8), so without InputPadder it
    # raises on any side that is not a multiple of 8 -- including 854x480, the
    # most common HateMM resolution. Upstream calls it unpadded.
    # RAFT also has a floor. Its correlation pyramid halves four times from
    # the H/8 feature map, so a side under 128 px collapses the coarsest level
    # to width 1, where `bilinear_sampler` normalises by `W - 1` and returns
    # NaN. The three sizes below are the real decoded sizes of this study's
    # corpora; `check_decode` asserts every video clears the floor.
    rng = np.random.RandomState(0)
    for h, w in ((480, 854), (720, 404), (480, 308)):
        frames = [rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
                  for _ in range(2)]
        try:
            flow = fx._mean_flow(fx._to_tensor(frames[0]),
                                 fx._to_tensor(frames[1]))
            ok = flow.shape == (2,) and np.isfinite(flow).all()
            detail = "mean flow %s" % np.round(flow, 4)
        except Exception as exc:                             # noqa: BLE001
            ok, detail = False, "%s: %s" % (type(exc).__name__, str(exc)[:90])
        check("RAFT runs at %dx%d (%s multiple of 8)"
              % (h, w, "not a" if (h % 8 or w % 8) else "a"), ok, detail)

    # Real frames, one per corpus, through the whole extractor.
    cfg = cfgmod.build_config("paper")
    for corpus in hdata.CORPORA:
        ids, _ = seg.cohort(corpus, "test")
        vid = ids[0]
        try:
            pr = video_io.probe(seg.video_path(corpus, vid), cfg)
            got = []
            for i, frame in enumerate(video_io.iter_frames(pr)):
                got.append(frame)
                if i >= 3:
                    break
            clip, flow = fx.extract(iter(got))
            ok = (clip.shape == (len(got), feat.CLIP_DIM)
                  and flow.shape == (len(got), feat.FLOW_DIM)
                  and np.isfinite(clip).all() and np.isfinite(flow).all()
                  and not flow[0].any())
            detail = ("clip norms %.1f-%.1f, flow row L2 %s"
                      % (np.linalg.norm(clip, axis=1).min(),
                         np.linalg.norm(clip, axis=1).max(),
                         np.round(np.linalg.norm(flow[1:], axis=1), 3)))
        except Exception as exc:                             # noqa: BLE001
            ok, detail = False, "%s: %s" % (type(exc).__name__, str(exc)[:90])
        check("%s/%s: CLIP 512-d and RAFT 128-d, finite, row 0 zero"
              % (corpus, vid), ok, detail)


def main():
    print(__doc__.strip().splitlines()[0])
    check_gaps()
    check_substitutions()
    check_numerics()
    check_cohorts()
    check_decode()
    check_models()
    print("\n%s" % ("=" * 60))
    if FAILURES:
        print("%d FAILURES: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
