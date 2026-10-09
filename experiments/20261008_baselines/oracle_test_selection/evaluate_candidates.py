#!/usr/bin/env python3
"""Oracle test selection, step 2: evaluate every candidate (checkpoint x output branch) of every seed with the
canonical evaluator, after the exact-cohort coverage check every current run uses.

    python evaluate_candidates.py [--methods ...] [--datasets ...] [--workers N]

Per (method, dataset, seed) the candidates are
  - every per-epoch score file of the retrained run (oracle_test_selection/<method>/<DS>/seed<k>/epochs/), and
  - the current run's selected checkpoint ("current"), for every branch its saved outputs hold,
times every output branch of the method (oracle_common.METHODS). Coverage check:
  - 1 fps methods: weaksup_common/common.py build_predictions (x4 repeat, last-value pad to ceil(4D); every cohort
    video finite on every GT frame), the function every current weakly supervised run used;
  - 8-s window methods (SAGE, CLARA): the checks of detwin/common.py finalize (one score per window, curve covers
    the GT length, finite on every GT frame), window -> 4 fps by detwin window_curve. A candidate that fails (e.g.
    a diverged epoch with NaN scores) is not evaluated and is listed as invalid; there is no F2 fallback.
Evaluation: python -m src.eval.evaluate_four_datasets on chunks of candidates (one method name per candidate,
"<method>@<checkpoint>@<branch>"), through weaksup_common/common.py evaluate (checks that the evaluator saw the whole
cohort and frame pool for every candidate). The evaluator's metrics.json of each chunk is kept in
<seed dir>/eval/chunk_NNN/metrics.json; the chunk's 4 fps predictions.jsonl is deleted after evaluation (it is
re-derivable from the kept native-rate score files). Result per seed: <seed dir>/candidates.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_common as O  # noqa: E402

C, D = O.C, O.D
MAX_FRAMES_PER_CHUNK = 12_000_000
CODE = "experiments/20261008_baselines/oracle_test_selection/evaluate_candidates.py"

_cache = {}


def load_scores_any(path):
    """common.load_scores, also for .jsonl.gz; small cache so one file is parsed once for all its branches."""
    path = str(path)
    if path in _cache:
        return _cache[path]
    out = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if rec["video_id"] in out:
                    raise ValueError(f"{path}: duplicate {rec['video_id']}")
                out[rec["video_id"]] = rec
    _cache.clear()
    _cache[path] = out
    return out


C.load_scores = load_scores_any  # build_predictions looks the function up in its module at call time


class _Quiet:
    def __init__(self, log):
        self.log = log

    def __call__(self, msg):
        self.log.fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        self.log.fh.flush()


def load_window_file(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def window_rows(ds, name, seed, ws):
    """detwin finalize's exact-cohort checks and rows, without the F2 fallback. Returns (rows, None) or (None, why)."""
    split = D.load_split(ds)
    dur = {r["video_id"]: float(r["duration"]) for r in split["test"]}
    ids = sorted(dur)
    t_gt = D.gt_lengths(ds)
    if len(ids) != D.COHORT_SIZE[ds] or not set(ids) <= set(t_gt):
        raise RuntimeError(f"{ds}: cohort mismatch")
    rows, bad = [], {}
    for v in ids:
        w = ws.get(v)
        if w is None:
            bad[v] = "no_output"
            continue
        nw = len(D.windows(dur[v]))
        if len(w) != nw or any(x is None for x in w):
            bad[v] = f"{len(w)} window scores for {nw} windows"
            continue
        c = D.window_curve(w, dur[v])
        T = t_gt[v]
        if len(c) < T:
            bad[v] = f"curve shorter than GT ({len(c)} < {T})"
            continue
        if not np.isfinite(c[:T]).all():
            bad[v] = f"{int((~np.isfinite(c[:T])).sum())} non-finite GT frames"
            continue
        rows.append({"schema_version": 1, "method": name, "dataset": ds, "video_id": v, "duration": dur[v],
                     "native_rate": 1.0 / D.WIN, "score_curve": [float(x) for x in c], "intervals": [],
                     "error": None, "calls": None, "seed": seed, "code_path": CODE, "extra": {}})
    if bad:
        return None, f"{len(bad)} cohort videos fail the coverage check, e.g. {list(bad.items())[:2]}"
    return rows, None


def candidate_sources(method, ds, seed):
    """[(tag, path, branches)] for this seed: retrained epochs, then the current run's selected checkpoint."""
    spec = O.METHODS[method]
    sd = O.seed_dir(method, ds, seed)
    out = []
    if spec["kind"] == "frame":
        for p in sorted((sd / "epochs").glob("e*.jsonl.gz")):
            out.append((p.name.split(".")[0], p, spec["branches"]))
        cur = O.current_frame_scores(method, ds, seed)
        if cur.is_file():
            with open(cur) as fh:
                keys = set(json.loads(fh.readline()))
            out.append(("current", cur, [b for b in spec["branches"] if b in keys]))
    elif method == "sage":
        for p in sorted((sd / "epochs").glob("e*.json.gz")):
            out.append((p.name.split(".")[0], p, spec["branches"]))
        cur = sd / "epochs" / "current.json.gz"
        if cur.is_file():
            out.append(("current", cur, spec["branches"]))
        else:  # current run's fusion head only
            out.append(("current", O.CUR / "sage" / ds / "infer" / "window_scores.jsonl", ["fusion"]))
    elif method == "clara":
        for p in sorted((sd / "epochs").glob("e*.json")):
            out.append((p.stem, p, ["prob"]))
        cur = O.current_seed_dir(method, ds, seed) / "window_scores.json"
        if cur.is_file():
            out.append(("current", cur, ["prob"]))
    return out


def window_scores_for(method, tag, path, branch, seed):
    if method == "sage":
        if path.name == "window_scores.jsonl":
            return {json.loads(l)["video_id"]: json.loads(l)["scores"][str(seed)] for l in open(path)}
        return load_window_file(path)[branch]
    return load_window_file(path)  # clara: {video_id: [window scores]}


def evaluate_seed(method, ds, seed, force=False):
    spec = O.METHODS[method]
    corpus = C.CORPUS[ds]
    sd = O.seed_dir(method, ds, seed)
    sources = candidate_sources(method, ds, seed)
    if not any(t != "current" for t, _, _ in sources):
        return f"{method} {ds} {seed}: no retrained epochs yet"
    wanted = [(t, b) for t, _, bs in sources for b in bs]
    res_path = sd / "candidates.json"
    if res_path.is_file() and not force:
        old = json.loads(res_path.read_text())
        if sorted((c["checkpoint"], c["branch"]) for c in old["candidates"]) == sorted(wanted):
            return f"{method} {ds} {seed}: up to date ({len(wanted)} candidates)"
    sd.mkdir(parents=True, exist_ok=True)
    log = C.RunLog(sd / "eval.log", mode="w")
    quiet = _Quiet(log)
    log(f"evaluate {method} {ds} seed {seed}: {len(wanted)} candidates; code {C.code_version()}")
    if (sd / "eval").exists():
        shutil.rmtree(sd / "eval")
    cands, chunk, chunk_frames, n_chunk = [], [], 0, 0
    gt_len = C.gt_lengths(corpus)
    frames_per = sum(gt_len[v] for v in C.split_ids(corpus, "test"))

    def flush():
        nonlocal chunk, chunk_frames, n_chunk
        if not chunk:
            return
        cdir = sd / "eval" / f"chunk_{n_chunk:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        rows = [r for _, rs in chunk for r in rs]
        result = C.evaluate(corpus, rows, cdir, quiet)
        (cdir / "predictions.jsonl").unlink()
        by = {r["method"]: r for r in result["per_dataset"]}
        for cand, _ in chunk:
            r = by[cand["name"]]
            cand.update({k: float(r[k]) for k in O.METRICS})
            cand["n_videos"] = r["n_videos_overlap"]
            cand["n_frames"] = r.get("n_frames")
            cand["metrics_json"] = str((cdir / "metrics.json").relative_to(C.REPO))
        n_chunk += 1
        chunk, chunk_frames = [], 0

    for tag, path, branches in sources:
        for b in branches:
            name = f"{spec['current_name']}@{tag}@{b}"
            cand = {"checkpoint": tag, "branch": b, "name": name, "source": str(Path(path).relative_to(C.REPO)),
                    "from": "current run" if tag == "current" else "retrained run"}
            try:
                if spec["kind"] == "frame":
                    rows, _cov = C.build_predictions(corpus, path, [b], [name], seed, CODE, quiet)
                    why = None
                else:
                    rows, why = window_rows(ds, name, seed, window_scores_for(method, tag, Path(path), b, seed))
            except RuntimeError as exc:  # coverage check failed (missing video or non-finite frame)
                rows, why = None, str(exc)[:300]
            cands.append(cand)
            if rows is None:
                cand.update({"valid": False, "reason": why})
                log(f"{name}: not evaluated: {why}")
                continue
            cand["valid"] = True
            chunk.append((cand, rows))
            chunk_frames += frames_per
            if chunk_frames >= MAX_FRAMES_PER_CHUNK:
                flush()
    flush()
    n_valid = sum(c["valid"] for c in cands)
    res_path.write_text(json.dumps({"method": method, "dataset": ds, "seed": seed, "n_candidates": len(cands),
                                    "n_valid": n_valid, "evaluator": "src/eval/evaluate_four_datasets.py",
                                    "code": f"{CODE}, {C.code_version()}", "candidates": cands}, indent=1) + "\n")
    log(f"done: {n_valid}/{len(cands)} candidates evaluated in {n_chunk} chunks")
    return f"{method} {ds} {seed}: {n_valid}/{len(cands)} evaluated"


def _log_has(path, marker):
    return path.is_file() and marker in path.read_text()


def ready(method, ds, seed):
    """All per-epoch score files of this seed are written (completion marker of the step that writes them)."""
    sd = O.seed_dir(method, ds, seed)
    if method == "sage":
        return _log_has(O.ORACLE / "sage" / ds / "score_epochs" / "run.log", "DONE score_epochs")
    if method == "clara":
        return _log_has(sd / "run.log", "DONE clara")
    return (sd / "DONE").is_file()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(O.METHODS))
    ap.add_argument("--datasets", nargs="+", default=list(O.DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(O.SEEDS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    jobs = [(m, ds, s) for m in args.methods for ds in args.datasets for s in args.seeds if ready(m, ds, s)]
    print(f"{len(jobs)} seed runs with retrained epochs", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(evaluate_seed, *j, args.force): j for j in jobs}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {futs[f]}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
