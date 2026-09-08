"""Helper for the Slurm monitor: sanity-check output files of running
jobs. Invoked per-tick; emits tagged single-line messages for the
Monitor tool to surface as notifications.

Rules:
  IH judge job → offline_test_ih_<tag>.jsonl
    - each line JSON
    - pred ∈ {0, 1, -2}   (-2 is legit abstention; -1 is parse fail, never written)
    - rationale present and non-empty for >=90% of recent lines
  Stage-1 holistic job → results/holistic_<slug>/<ds>/<split>_binary.jsonl
    - each line JSON, field 'score' must be a float in [0,1] (or explicit null)
    - ratio of null 'score' >20% is suspicious
Emits one line per *problematic* file. Normal lines are silent to
avoid spam.
"""
from __future__ import annotations
import json, os, sys, re, glob


def check_judge_file(path):
    try:
        recs = [json.loads(l) for l in open(path) if l.strip()]
    except Exception as e:
        return f"PARSE_ERR {path} {e}"
    if not recs:
        return None
    n = len(recs)
    bad_pred = sum(1 for r in recs if r.get("pred") not in (0, 1, -2))
    empty_rat = sum(1 for r in recs if not (r.get("rationale") or "").strip())
    msgs = []
    if bad_pred / n > 0.05:
        msgs.append(f"judge_pred_bad={bad_pred}/{n}")
    if empty_rat / n > 0.2:
        msgs.append(f"judge_empty_rationale={empty_rat}/{n}")
    # last 3 lines
    tail = recs[-3:]
    for r in tail:
        if r.get("pred") not in (0, 1, -2):
            msgs.append(f"tail_bad_pred={r.get('pred')}")
            break
    if msgs:
        return f"{path} n={n} " + " ".join(msgs)
    return None


def check_holistic_file(path):
    try:
        recs = [json.loads(l) for l in open(path) if l.strip()]
    except Exception as e:
        return f"PARSE_ERR {path} {e}"
    if not recs:
        return None
    n = len(recs)
    nulls = sum(1 for r in recs if r.get("score") is None)
    bad_range = sum(1 for r in recs
                    if r.get("score") is not None and
                       (not isinstance(r["score"], (int, float))
                        or r["score"] < 0 or r["score"] > 1))
    msgs = []
    if nulls / n > 0.2:
        msgs.append(f"null_score={nulls}/{n}")
    if bad_range > 0:
        msgs.append(f"bad_range_score={bad_range}/{n}")
    if msgs:
        return f"{path} n={n} " + " ".join(msgs)
    return None


def tail_check(slurm_log):
    """Look at tail of slurm log for actionable error patterns."""
    if not os.path.isfile(slurm_log):
        return None
    try:
        tail = open(slurm_log, "rb").read()[-8000:].decode("utf-8", errors="replace")
    except Exception:
        return None
    patterns = [
        (r"consecutive failures", "CONSEC_FAIL"),
        (r"CUDA out of memory", "OOM"),
        (r"RuntimeError.*CUDA", "CUDA_ERR"),
        (r"Killed", "KILLED"),
        (r"Traceback \(most recent", "TRACEBACK"),
    ]
    hits = []
    for pat, tag in patterns:
        if re.search(pat, tail):
            hits.append(tag)
    if hits:
        return ",".join(hits)
    return None


def main():
    # inputs: list of "<jobid>:<type>:<relative_out_path>" strings on stdin
    # type in {judge, holistic}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 3:
            continue
        jobid, typ, path = parts[0], parts[1], ":".join(parts[2:])
        issue = None
        if typ == "judge":
            issue = check_judge_file(path)
        elif typ == "holistic":
            issue = check_holistic_file(path)
        slurm_log = f"/data/jehc223/EMNLP3/slurm-{jobid}.out"
        log_hit = tail_check(slurm_log)
        if issue:
            print(f"ALERT jobid={jobid} {issue}")
        if log_hit:
            print(f"ALERT jobid={jobid} log={log_hit}")


if __name__ == "__main__":
    main()
