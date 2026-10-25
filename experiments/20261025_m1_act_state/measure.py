#!/usr/bin/env python3
"""Candidate 50, act-state reading, mask form A1 (falsifier pre-check): GPU reads. Labels and GT are never read.

Per video, computed once and shared by all arms, exactly as in r6 (experiments/20260922_til/til_measure.py):
shared prefix (rules + 20 frames + timestamped transcript) -> verdict z_video -> the model's one-word stance turn.
Then for every 8 s window an isolated visual branch and (when the window has words) an isolated speech branch, read as
Yes/No log-odds, in every arm. Every arm uses the same explicit attention mask at all 36 text layers:

  base       every cached token visible. Plumbing replica: must equal runs/20260926_glr/base_gridA exactly.
  all_local  Grounder's all_local access set: scaffold + this window's prefix frame tokens and transcript tokens + the
             branch's own query tokens. Replica: within 1e-3 logit of runs/20261002_m1_grounder/r1_full/all_local.
  A1         all_local + the verdict span (verdict question turn, assistant header, one-word stance answer).
  A1_other   A1 on a cache copy whose verdict span holds the donor video's verdict-span keys and values: the donor's
             pre-rotary keys re-rotated (src/pre_rotary_memory.rotate) to THIS video's span positions, and the donor's
             values. Donor = next video in sorted id order (cyclic) in the same corpus with the same verdict sign
             (signs from the cached r6 reads, re-checked in job), so the span tokens are identical.

Videos of a corpus are processed in descending sorted-id order, so a donor's states are captured during its own main
pass (no extra forward). A donor that was not processed in this session (cyclic wrap, the end of a --first subset, a
resume) gets one donor pass: prefix + verdict + stance, nothing else.

Outputs: runs/20261025_m1_act_state/<run>/{run.log (first line = hostname), run.pid, config.json, checks.jsonl,
<arm>/predictions.jsonl, <arm>/config.json}. Per-window reads use the r6 read schema (extra.windows[i] has
start, end, z = max(z_visual, z_speech), z_visual, z_speech), so experiments/20260926_twolevel/twolevel_r2.py reads them.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src.video_inputs import FPS, fixed_windows, frame_paths, load_asr, load_manifest, window_text  # noqa: E402
from act_state import ARMS, donor_map  # noqa: E402

EXP = "20261025_m1_act_state"
IMPL = "c50-a1-v1"
CODE = "experiments/20261025_m1_act_state/{measure,act_state}.py"
MANIFESTS = {"HateMM": "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "HateClipSeg": "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "DeHate": "data/manifests/DeHate_test.jsonl"}
R6_READS = {"HateMM": "runs/20260926_glr/base_gridA/predictions.jsonl",
            "HateClipSeg": "runs/20260926_glr/base_gridA/predictions.jsonl",
            "DeHate": "runs/20260927_dehate_external/reads_gridA/predictions.jsonl"}
GROUNDER_ALL_LOCAL = {"HateMM": "runs/20261002_m1_grounder/r1_full/all_local/predictions.jsonl",
                      "HateClipSeg": "runs/20261002_m1_grounder/r1_full/all_local/predictions.jsonl"}
# In-job check tolerances (declared before running; README "In-job checks").
BASE_TOL = 0.0          # base replica and z_video vs the r6 reads: exact
ALL_LOCAL_TOL = 1e-3    # all_local replica vs Grounder all_local, logit
KEY_ROT_REL_TOL = 1e-2  # own pre-rotary span keys re-rotated vs cached keys, max |dk| / max |k|
VALUE_TOL = 0.0         # own captured span values vs cached values: exact
SELF_SWAP_TOL = 1e-3    # A1 on a cache whose span is replaced by its own re-rotated states vs A1, logit
CROP_COPY_TOL = 1e-2    # in-place + crop vs deep copy, logit (Grounder's tolerance)
HOOK_VS_PLAIN_TOL = .25  # explicit-mask base vs ordinary SDPA branch, logit (Grounder's tolerance)


class CheckFailed(SystemExit):
    pass


def jl(path):
    out = {}
    for r in map(json.loads, open(ROOT / path)):
        key = (r["dataset"], r["video_id"])
        assert key not in out and not r.get("error"), key
        out[key] = r
    return out


def corpus_rows(ds):
    rows = load_manifest(ROOT / MANIFESTS[ds], [ds])
    byid = {r["video_id"]: r for r in rows}
    assert len(byid) == len(rows), (ds, "duplicate ids in manifest")
    return byid


def plan(datasets, first, arms, done=None):
    """Per corpus: processing order (descending sorted id over the first `first` ids), donors, donor passes."""
    out = {}
    for ds in datasets:
        byid = corpus_rows(ds)
        ref = {k[1]: r for k, r in jl(R6_READS[ds]).items() if k[0] == ds}
        assert set(ref) == set(byid), (ds, "r6 reads do not cover the manifest")
        corpus = sorted(byid)
        sign = {v: float(ref[v]["extra"]["z_video"]) > 0 for v in corpus}
        assert all(float(ref[v]["extra"]["z_video"]) != 0 for v in corpus)
        todo = corpus[:first] if first else corpus
        donors = donor_map(sign, corpus) if "A1_other" in arms else {}
        if donors:
            assert all(donors[v] is not None for v in todo), (ds, "a sign group of size 1: no donor")
        skip = set(done.get(ds, ())) if done else set()
        order = [v for v in reversed(todo) if v not in skip]
        donors = {v: donors[v] for v in todo} if donors else {}
        users = {d: v for v, d in donors.items()}      # cyclic shift within a sign group: each donor has one user
        assert len(users) == len(donors)
        passes = []
        if donors:                                       # simulate the main loop's store (see main)
            pending, store = set(order), set()
            for v in order:
                if donors[v] not in store:
                    passes.append(donors[v])
                store.discard(donors[v])
                pending.discard(v)
                if users.get(v) in pending:
                    store.add(v)
        out[ds] = {"n_corpus": len(corpus), "todo": todo, "order": order, "sign": {v: sign[v] for v in todo},
                   "donors": donors, "users": users, "donor_passes": passes,
                   "n_yes": sum(sign.values()), "n_no": len(corpus) - sum(sign.values())}
    return out


# ---------------------------------------------------------------------------------------------- GPU part

def verdict_state(j, row, segments):
    """Prefix -> verdict -> stance turn, with the verdict span's pre-rotary keys and values captured."""
    import torch
    from act_state import capture, join_forwards
    from src.mllm_judge import VIDEO_QUESTION
    ds, vid = row["dataset"], row["video_id"]
    frames = frame_paths(ds, vid, 20, "k20")
    assert frames, (ds, vid, "missing frames")
    msgs, files = j.prefix_messages(frames, segments)
    prefix, enc = j.encode_prefix(msgs, files)
    cache = j.prefix_cache(enc)
    b0, b0text = j.branch_ids(msgs, VIDEO_QUESTION)
    with capture(j) as part0:
        zv = j.cached_margin(cache, b0, in_place=True)
    stance = "Yes" if zv > 0 else "No"
    a0, a0text = j.answer_ids(msgs, VIDEO_QUESTION, stance)
    with capture(j) as part1:
        j.extend_cache(cache, a0)
    span = join_forwards([part0, part1])
    P = int(enc["input_ids"].shape[1])
    assert cache.get_seq_length() == P + len(b0) + len(a0)
    assert span[0][0].shape[2] == len(b0) + len(a0)
    torch.cuda.synchronize()
    return {"frames": frames, "msgs": msgs, "files": files, "prefix": prefix, "enc": enc, "cache": cache, "b0": b0,
            "b0text": b0text, "a0": a0, "a0text": a0text, "z_video": float(zv), "stance": stance,
            "span_ids": list(b0) + list(a0), "span": span, "prefix_len": P}


def donor_pass(j, row, segments, ref_z):
    """Donor states only (no branches). Its z_video must equal the cached r6 read exactly."""
    st = verdict_state(j, row, segments)
    if abs(st["z_video"] - ref_z) > BASE_TOL:
        raise CheckFailed(f"CHECK_FAILED donor z_video {row['dataset']} {row['video_id']}: {st['z_video']} vs r6 {ref_z}")
    out = {"video_id": row["video_id"], "span_ids": st["span_ids"], "span": st["span"], "z_video": st["z_video"],
           "stance": st["stance"], "source": "donor_pass"}
    del st
    return out


def read_video(j, access, row, segments, arms, donor, ref, ref_al, verify):
    import torch
    from act_state import RESTRICTED, rotate_states, rotation_check, span_positions, swapped_cache, window_keep
    from src.mllm_judge import VIDEO_QUESTION, yesno_question
    from src.window_token_regions import token_regions
    started = time.perf_counter()
    ds, vid, dur = row["dataset"], row["video_id"], float(row["duration"])
    wins = fixed_windows(dur, 8)
    texts = [window_text(segments, a, b) for a, b in wins]
    st = verdict_state(j, row, segments)
    checks = {}
    # z_video and stance: the global path is unchanged, so they must equal the r6 read exactly.
    checks["z_video_abs_diff_vs_r6"] = abs(st["z_video"] - float(ref["extra"]["z_video"]))
    if checks["z_video_abs_diff_vs_r6"] > BASE_TOL:
        raise CheckFailed(f"CHECK_FAILED z_video {ds} {vid}: {st['z_video']} vs r6 {ref['extra']['z_video']}")
    if verify:
        j.seam_check_tokens(st["msgs"], st["files"], st["enc"]["input_ids"][0].tolist(), VIDEO_QUESTION, st["b0"])
    regions = token_regions(j, st["prefix"], st["enc"], st["frames"], segments, wins)
    P, cache = regions["prefix_len"], st["cache"]
    assert P == st["prefix_len"]
    ncache = cache.get_seq_length()
    assert ncache == P + len(st["span_ids"])
    pos = span_positions(j, st["enc"], st["span_ids"])
    rotary = j.model.model.language_model.rotary_emb
    own_rot = rotate_states(st["span"], pos, rotary)
    dk, dkr, dv = rotation_check(cache, P, own_rot)
    checks.update({"own_span_key_abs_diff": dk, "own_span_key_rel_diff": dkr, "own_span_value_abs_diff": dv})
    if dkr > KEY_ROT_REL_TOL or dv > VALUE_TOL:
        raise CheckFailed(f"CHECK_FAILED span re-rotation {ds} {vid}: key {dk} (rel {dkr}) value {dv}")
    caches = {arm: cache for arm in arms}
    donor_info = None
    if "A1_other" in arms:
        if donor["span_ids"] != st["span_ids"]:
            raise CheckFailed(f"CHECK_FAILED donor span tokens differ {ds} {vid} <- {donor['video_id']}")
        if donor["stance"] != st["stance"]:
            raise CheckFailed(f"CHECK_FAILED donor stance differs {ds} {vid} <- {donor['video_id']}")
        caches["A1_other"] = swapped_cache(cache, P, rotate_states(donor["span"], pos, rotary))
        donor_info = {"video_id": donor["video_id"], "z_video": donor["z_video"], "stance": donor["stance"],
                      "source": donor["source"]}
    history = [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]}, j.turn("assistant", st["stance"])]
    head = st["prefix"] + st["b0text"] + st["a0text"]
    torch.cuda.synchronize()
    prefix_seconds = time.perf_counter() - started
    per = {arm: [{} for _ in wins] for arm in arms}
    timing = {arm: 0. for arm in arms}
    branches = 0
    for kind in ("visual", "speech"):
        for i, ((a, b), txt) in enumerate(zip(wins, texts)):
            if kind == "speech" and not txt.strip():
                continue
            q = yesno_question(i, len(wins), a, b, txt, kind)
            bids, _ = j.branch_ids(st["msgs"], q, history, head_text=head)
            for arm in arms:
                keep = window_keep(regions, i, ncache, arm)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                z = access.margin(caches[arm], bids, keep, RESTRICTED[arm])
                torch.cuda.synchronize()
                timing[arm] += time.perf_counter() - tick
                per[arm][i]["z_" + kind] = float(z)
                if verify and branches == 0:
                    zc = access.margin(caches[arm], bids, keep, RESTRICTED[arm], copy_cache=True)
                    checks[arm + "_crop_copy_abs_diff"] = abs(z - zc)
                    if abs(z - zc) >= CROP_COPY_TOL:
                        raise CheckFailed(f"CHECK_FAILED crop/copy {arm} {ds} {vid}: {z} vs {zc}")
                    if arm == "base":
                        plain = j.cached_margin(cache, bids, in_place=False)
                        checks["base_hook_vs_ordinary_abs_diff"] = abs(z - plain)
                        if abs(z - plain) >= HOOK_VS_PLAIN_TOL:
                            raise CheckFailed(f"CHECK_FAILED mask plumbing changes base {ds} {vid}: {z} vs {plain}")
                    if arm == "A1":
                        selfc = swapped_cache(cache, P, own_rot)
                        zs = access.margin(selfc, bids, keep, True)
                        del selfc
                        checks["A1_self_swap_abs_diff"] = abs(z - zs)
                        if abs(z - zs) > SELF_SWAP_TOL:
                            raise CheckFailed(f"CHECK_FAILED self swap {ds} {vid}: {z} vs {zs}")
            branches += 1
    # replica checks against the cached reads
    ref_w = ref["extra"]["windows"]
    assert [(float(w["start"]), float(w["end"])) for w in ref_w] == [(float(a), float(b)) for a, b in wins], (ds, vid)
    if "base" in arms:
        diffs = []
        for w, v in zip(per["base"], ref_w):
            assert set(w) == {k for k in ("z_visual", "z_speech") if k in v}, (ds, vid, "branch set differs from r6")
            diffs.extend(abs(w[k] - float(v[k])) for k in w)
        checks["base_max_abs_diff_vs_r6"] = max(diffs, default=0.)
        if checks["base_max_abs_diff_vs_r6"] > BASE_TOL:
            raise CheckFailed(f"CHECK_FAILED base replica {ds} {vid}: max |dz| {checks['base_max_abs_diff_vs_r6']}")
    if "all_local" in arms and ref_al is not None:
        diffs = []
        for w, v in zip(per["all_local"], ref_al["extra"]["windows"]):
            assert set(w) == {k for k in ("z_visual", "z_speech") if k in v}, (ds, vid, "branch set differs from Grounder")
            diffs.extend(abs(w[k] - float(v[k])) for k in w)
        checks["all_local_max_abs_diff_vs_grounder"] = max(diffs, default=0.)
        if checks["all_local_max_abs_diff_vs_grounder"] > ALL_LOCAL_TOL:
            raise CheckFailed(f"CHECK_FAILED all_local replica {ds} {vid}: max |dz| {checks['all_local_max_abs_diff_vs_grounder']}")
    stats = []
    for i, (a, b) in enumerate(wins):
        m = regions["local"][i]
        stats.append({"kept_visual_tokens": int((m & regions["visual"]).sum()),
                      "kept_speech_tokens": int((m & regions["speech"]).sum()),
                      "kept_media_tokens": int(m.sum()), "scaffold_tokens": int(regions["scaffolding"].sum()),
                      "verdict_span_tokens": int(ncache - P),
                      "prefix_media_tokens": int((regions["visual"] | regions["speech"]).sum())})
    L = int(math.ceil(dur * FPS))
    index = np.clip(((np.arange(L) + .5) / FPS // 8).astype(int), 0, len(wins) - 1)
    records = {}
    for arm in arms:
        z = np.array([max(w.values()) for w in per[arm]])
        records[arm] = {
            "schema_version": 1, "method": "c50_act_state_" + arm, "dataset": ds, "video_id": vid, "duration": dur,
            "native_rate": FPS, "score_curve": z[index].tolist(), "intervals": [], "error": None, "seed": 0,
            "code_path": "experiments/20261025_m1_act_state/measure.py", "calls": 3 + branches,
            "extra": {"z_video": st["z_video"], "stance": st["stance"], "n_branches": branches,
                      "calls_breakdown": {"prefix": 1, "global_query": 1, "answer_extension": 1, "local_branches": branches},
                      "prefix_tokens": P, "cached_tokens": ncache, "verdict_span_tokens": ncache - P,
                      "text_layers": len(access.layers), "restricted_layers": len(access.layers) if RESTRICTED[arm] else 0,
                      "donor": donor_info if arm == "A1_other" else None,
                      "prefix_seconds": prefix_seconds, "branch_seconds": timing[arm],
                      "windows": [{"i": i, "start": a, "end": b, "z": float(z[i]), **per[arm][i], **stats[i]}
                                  for i, (a, b) in enumerate(wins)]}}
    own = {"video_id": vid, "span_ids": st["span_ids"], "span": st["span"], "z_video": st["z_video"],
           "stance": st["stance"], "source": "main_pass"}
    del caches, cache, st
    return records, {"dataset": ds, "video_id": vid, "checks": checks, "donor": donor_info,
                     "prefix_seconds": prefix_seconds, "branch_seconds": timing}, own


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-name", help="output: runs/20261025_m1_act_state/<run-name>/ (required unless --plan-only)")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"], choices=list(MANIFESTS))
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--first", type=int, default=0, help="first N videos by sorted id per corpus (0 = all)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--plan-only", action="store_true", help="CPU: print the processing / donor plan and exit")
    a = ap.parse_args()
    arms = [x for x in ARMS if x in a.arms]
    if not a.plan_only and not a.run_name:
        ap.error("--run-name is required unless --plan-only")
    if a.plan_only:
        p = plan(a.datasets, a.first, arms)
        print(json.dumps(p, indent=1))
        print("PLAN_DONE", {ds: {"todo": len(d["todo"]), "donor_passes": len(d["donor_passes"])} for ds, d in p.items()})
        return
    import torch
    import transformers
    from act_state import QueryAccess
    from src.mllm_judge import MODEL, VIDEO_QUESTION, Judge, yesno_question
    assert a.model == MODEL, "the reader is fixed to the r6 reader"
    out = ROOT / "runs" / EXP / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run.log", "a") as fh:
        fh.write(socket.gethostname() + "\n")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    torch.manual_seed(0)
    refs = {ds: {k[1]: r for k, r in jl(R6_READS[ds]).items() if k[0] == ds} for ds in a.datasets}
    ref_al = {ds: {k[1]: r for k, r in jl(GROUNDER_ALL_LOCAL[ds]).items() if k[0] == ds}
              for ds in a.datasets if ds in GROUNDER_ALL_LOCAL}
    rows = {ds: corpus_rows(ds) for ds in a.datasets}
    asr = {ds: load_asr(ds) for ds in a.datasets}
    handles, done = {}, {}
    for arm in arms:
        d = out / arm
        d.mkdir(exist_ok=True)
        path = d / "predictions.jsonl"
        done[arm] = {(r["dataset"], r["video_id"]) for r in map(json.loads, path.open())} if path.exists() else set()
    assert all(v == done[arms[0]] for v in done.values()), "partial per-arm record; inspect and repair before resume"
    done_by_ds = {ds: {v for d_, v in done[arms[0]] if d_ == ds} for ds in a.datasets}
    schedule = plan(a.datasets, a.first, arms, done_by_ds)
    j = Judge(a.model)
    access = QueryAccess(j)
    config = {**vars(a), "arms": arms, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(), "seed": 0,
              "torch": torch.__version__, "transformers": transformers.__version__, "GT_in_reader": False,
              "code": CODE + "; implementation tag " + IMPL + "; written 2026-10-25",
              "shared_code": ["src/window_token_regions.py", "src/pre_rotary_memory.py", "src/stance_cache.py",
                              "src/mllm_judge.py", "src/video_inputs.py"],
              "mask_source": "archive/experiments/20261002_m1_grounder/grounder.py (QueryAccess, window_keep), copied with changes",
              "text_layers": len(access.layers), "mask_plumbing": "explicit bf16 additive mask in every text layer, every arm",
              "access_sets": {"base": "all cached tokens", "all_local": "scaffold + own window prefix frame/transcript tokens",
                              "A1": "all_local + verdict span [prefix_len, cached_len)",
                              "A1_other": "A1 keep on a cache copy with the donor's verdict-span states re-rotated to this video's span positions"},
              "donor_rule": "next video in sorted id order (cyclic) of the same corpus with the same verdict sign (sign from the r6 reads)",
              "processing_order": "descending sorted id within each corpus",
              "tolerances": {"base_and_z_video": BASE_TOL, "all_local_vs_grounder": ALL_LOCAL_TOL,
                             "own_span_key_rel": KEY_ROT_REL_TOL, "own_span_value": VALUE_TOL, "self_swap": SELF_SWAP_TOL,
                             "crop_copy": CROP_COPY_TOL, "hook_vs_plain": HOOK_VS_PLAIN_TOL},
              "references": {"r6_reads": R6_READS, "grounder_all_local": GROUNDER_ALL_LOCAL}, "manifests": MANIFESTS,
              "plan": {ds: {"todo": len(s["todo"]), "order_first": s["order"][:3], "donor_passes": s["donor_passes"]}
                       for ds, s in schedule.items()},
              "video_question": VIDEO_QUESTION, "visual_question": yesno_question(0, 1, 0, 8, "", "visual"),
              "speech_question": yesno_question(0, 1, 0, 8, "<text>", "speech"),
              "calls_note": "calls = actual forwards per deployed arm (3 + n_branches); A1 adds no forward to r6; "
                            "donor passes and verification calls are extra and not counted"}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for arm in arms:
        (out / arm / "config.json").write_text(json.dumps({**config, "arm": arm}, indent=2) + "\n")
        handles[arm] = (out / arm / "predictions.jsonl").open("a")
    logchecks = (out / "checks.jsonl").open("a")
    start = time.time()
    total = sum(len(s["order"]) for s in schedule.values())
    k = 0
    n_donor_passes = 0
    for ds in a.datasets:
        s = schedule[ds]
        store = {}                       # donor id -> its verdict-span states (GPU, about 4 MB each; never written)
        pending = set(s["order"])
        first_video = True
        for vid in s["order"]:
            donor = None
            if "A1_other" in arms:
                d = s["donors"][vid]
                if d not in store:
                    logging.info("donor pass %s %s (for %s)", ds, d, vid)
                    store[d] = donor_pass(j, rows[ds][d], asr[ds].get(d, []), float(refs[ds][d]["extra"]["z_video"]))
                    n_donor_passes += 1
                donor = store.pop(d)
            recs, checks, own = read_video(j, access, rows[ds][vid], asr[ds].get(vid, []), arms, donor, refs[ds][vid],
                                           ref_al.get(ds, {}).get(vid), verify=first_video)
            pending.discard(vid)
            if "A1_other" in arms and s["users"].get(vid) in pending:
                store[vid] = own         # this video is the donor of a video processed later
            del own, donor
            first_video = False
            for arm, r in recs.items():
                handles[arm].write(json.dumps(r) + "\n")
                handles[arm].flush()
            logchecks.write(json.dumps(checks) + "\n")
            logchecks.flush()
            k += 1
            logging.info("progress %d/%d %s %s elapsed=%.1f", k, total, ds, vid, time.time() - start)
        assert not store, ("donor states left unused", list(store))
        torch.cuda.empty_cache()
    for f in handles.values():
        f.close()
    logchecks.close()
    access.close()
    logging.info("donor passes %d", n_donor_passes)
    logging.info("RUN_DONE videos=%d elapsed=%.1f", k, time.time() - start)


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        raise
