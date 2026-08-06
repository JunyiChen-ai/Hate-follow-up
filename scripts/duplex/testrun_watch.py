"""Detached watcher: fire each benchmark's test-split run when its media lands.

The owner uploads source mp4s to B2 at paths and times this process does not
know in advance, possibly one dataset at a time and possibly in pieces. This
watcher polls the bucket, matches basenames against each dataset's test_clean id
list, and launches that dataset's pipeline once its media is there.

Trigger, per dataset, evaluated independently every poll:

  * coverage of the test-split ids reaches 100%, or
  * coverage is >= 90% and the found-count has not changed for 3 consecutive
    polls, which is what a finished upload with a few genuinely missing files
    looks like. The gap is recorded rather than waited on forever.

The launched run is `scripts/duplex/run_testrun.sh <slug>`, detached with setsid
so it outlives this watcher and every ssh session. Only one watcher-launched run
is active at a time, to bound CPU contention; the GPU itself is serialised
separately by a flock that every GPU stage takes, so a run launched here queues
behind whatever Mission A is doing rather than double-booking the card.

Transient B2 failures are logged and retried on the next poll. Nothing here ever
exits on a listing error.

Launch:
  setsid nohup python -u scripts/duplex/testrun_watch.py \
    > results/testruns/logs/watcher.log 2>&1 < /dev/null &

Self-test (no network, no GPU, no launch):
  python scripts/duplex/testrun_watch.py --self-test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

import testrun_media as tm  # noqa: E402

WATCHED = ["hatemm", "mhclip_en", "mhclip_zh"]
RUNNER = os.path.join(ROOT, "scripts", "duplex", "run_testrun.sh")
TESTRUNS = os.path.join(ROOT, "results", "testruns")
STATUS = os.path.join(TESTRUNS, "STATUS")
STATE = os.path.join(TESTRUNS, "watcher_state.json")
LOGDIR = os.path.join(TESTRUNS, "logs")

POLL_SECONDS = 600
MIN_FRAC = 0.90
STALL_POLLS = 3


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def say(msg):
    print(f"[{now()}] {msg}", flush=True)


def load_state():
    if os.path.exists(STATE):
        try:
            with open(STATE) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            say("watcher_state.json unreadable; starting from a clean state")
    return {s: {"state": "waiting", "n_found": -1, "stall": 0,
                "n_ids": None, "launched_at": None, "gap": None}
            for s in WATCHED}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def run_status(slug):
    """The dataset run's own STATUS file, or None if it never started."""
    p = os.path.join(TESTRUNS, slug, "STATUS")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def write_status(st, poll_n, note=""):
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    L = [f"watcher heartbeat: {now()}",
         f"poll: {poll_n}  interval: {POLL_SECONDS}s  pid: {os.getpid()}",
         f"trigger: coverage == 100%, or coverage >= {int(MIN_FRAC * 100)}% "
         f"unchanged for {STALL_POLLS} consecutive polls",
         ""]
    for slug in WATCHED:
        d = st[slug]
        cov = ("--" if d["n_ids"] in (None, 0)
               else f"{d['n_found']}/{d['n_ids']} "
                    f"({100.0 * d['n_found'] / d['n_ids']:.1f}%)")
        line = f"  {slug:12s} {d['state']:10s} coverage {cov}"
        rs = run_status(slug)
        if rs:
            line += f"  run-stage: {rs}"
        if d.get("gap"):
            line += f"  [{d['gap']} ids had no media in the bucket]"
        L.append(line)
    if note:
        L += ["", note]
    L.append("")
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(L))
    os.replace(tmp, STATUS)


def busy():
    """Slug of the watcher-launched run that is still going, or None."""
    for slug in WATCHED:
        rs = run_status(slug)
        if rs and not rs.startswith(("DONE", "FAILED")):
            if os.path.exists(os.path.join(TESTRUNS, slug, "STATUS")):
                # only counts if a process is actually alive for it
                p = subprocess.run(["pgrep", "-f", f"run_testrun.sh {slug}"],
                                   capture_output=True)
                if p.returncode == 0:
                    return slug
    return None


def launch(slug, dry_run=False):
    os.makedirs(LOGDIR, exist_ok=True)
    log = os.path.join(LOGDIR, f"{slug}_run.log")
    cmd = ["setsid", "nohup", "bash", RUNNER, slug]
    if dry_run:
        say(f"DRY RUN: would launch {' '.join(cmd)} > {log} 2>&1 < /dev/null")
        return True
    say(f"launching {' '.join(cmd)} > {log}")
    with open(log, "a") as lf, open(os.devnull) as devnull:
        subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                         stdin=devnull, start_new_session=True, cwd=ROOT)
    return True


def poll_once(st, poll_n, dry_run=False):
    ok, msg = tm.refresh_listing()
    say(f"poll {poll_n}: listing refresh {'ok' if ok else 'FAILED'}: {msg}")
    if not ok:
        say("keeping the previous listing and retrying next poll")
    idx = tm.load_index()
    if not idx:
        say("no usable listing on disk yet; nothing to match against")
        return

    for slug in WATCHED:
        d = st[slug]
        if d["state"] in ("DONE", "FAILED"):
            continue

        rs = run_status(slug)
        if d["state"] == "running":
            if rs and rs.startswith("DONE"):
                d["state"] = "DONE"
                say(f"{slug}: run reports DONE")
            elif rs and rs.startswith("FAILED"):
                d["state"] = "FAILED"
                say(f"{slug}: run reports {rs}")
            continue

        cov = tm.coverage(slug, idx)
        prev = d["n_found"]
        d["n_ids"], d["n_found"] = cov["n_ids"], cov["n_found"]
        d["stall"] = d["stall"] + 1 if cov["n_found"] == prev else 0
        say(f"{slug}: {cov['n_found']}/{cov['n_ids']} "
            f"({100 * cov['frac']:.1f}%), unchanged for {d['stall']} poll(s)")

        complete = cov["frac"] >= 1.0
        stalled = cov["frac"] >= MIN_FRAC and d["stall"] >= STALL_POLLS
        if not (complete or stalled):
            continue

        b = busy()
        if b and b != slug:
            say(f"{slug}: trigger condition met but {b} is still running; "
                f"holding until it finishes")
            d["state"] = "ready"
            continue

        d["gap"] = cov["n_missing"] if cov["n_missing"] else None
        why = ("100% coverage" if complete else
               f"coverage {100 * cov['frac']:.1f}% unchanged for "
               f"{d['stall']} polls, {cov['n_missing']} ids with no media")
        say(f"{slug}: TRIGGER ({why})")
        launch(slug, dry_run=dry_run)
        d["state"] = "running"
        d["launched_at"] = now()
        # One launch per poll, so two datasets landing together still start in
        # sequence rather than fighting for CPU.
        return


def self_test():
    """Verify the trigger path without touching the network or the GPU."""
    say("SELF-TEST: begin")
    fails = []

    if not os.path.exists(RUNNER):
        fails.append(f"runner missing: {RUNNER}")
    else:
        p = subprocess.run(["bash", "-n", RUNNER], capture_output=True)
        if p.returncode != 0:
            fails.append(f"runner is not valid bash: "
                         f"{p.stderr.decode('utf-8', 'replace')}")
        else:
            say("SELF-TEST: runner parses as bash")

    for slug in WATCHED:
        try:
            n = len(tm.test_ids(tm.DATASETS[slug]))
            say(f"SELF-TEST: {slug} test_clean id list loads, n={n}")
        except Exception as e:
            fails.append(f"{slug} id list failed: {e}")

    # A fake dataset whose coverage is forced complete, driven through the same
    # decision code, with launching stubbed out to a dry run.
    fake = "selftest_fake"
    real_watched = list(WATCHED)
    real_cov, real_idx = tm.coverage, tm.load_index
    real_refresh = tm.refresh_listing

    def stub(n_ids, n_found, frac, n_missing):
        tm.DATASETS[fake] = "HateMM"
        WATCHED[:] = [fake]
        tm.coverage = lambda s, i: {"slug": s, "dataset": "FAKE",
                                    "n_ids": n_ids, "n_found": n_found,
                                    "n_missing": n_missing, "frac": frac}
        tm.load_index = lambda *a, **k: {"x": [("x.mp4", 2000)]}
        tm.refresh_listing = lambda *a, **k: (True, "self-test: not refreshed")

    def unstub():
        tm.coverage, tm.load_index, tm.refresh_listing = \
            real_cov, real_idx, real_refresh
        WATCHED[:] = real_watched
        tm.DATASETS.pop(fake, None)

    st = {fake: {"state": "waiting", "n_found": -1, "stall": 0, "n_ids": None,
                 "launched_at": None, "gap": None}}
    stub(10, 10, 1.0, 0)
    try:
        poll_once(st, 0, dry_run=True)
    finally:
        unstub()

    if st[fake]["state"] == "running":
        say("SELF-TEST: forced-complete coverage drove the fake dataset to "
            "'running' and the launch command was constructed")
    else:
        fails.append(f"fake dataset did not trigger; state={st[fake]['state']}")

    # Negative case: 89% must never fire, however long it sits unchanged.
    st2 = {fake: {"state": "waiting", "n_found": 89, "stall": 0, "n_ids": 100,
                  "launched_at": None, "gap": None}}
    stub(100, 89, 0.89, 11)
    try:
        for i in range(4):
            poll_once(st2, i, dry_run=True)
    finally:
        unstub()
    if st2[fake]["state"] == "waiting":
        say("SELF-TEST: 89% coverage correctly refused to trigger even after "
            "4 unchanged polls")
    else:
        fails.append(f"89% coverage wrongly triggered; state={st2[fake]['state']}")

    # Positive stall case: 92% unchanged must fire on the third stalled poll.
    st3 = {fake: {"state": "waiting", "n_found": 92, "stall": 0, "n_ids": 100,
                  "launched_at": None, "gap": None}}
    stub(100, 92, 0.92, 8)
    fired_at = None
    try:
        for i in range(1, 6):
            poll_once(st3, i, dry_run=True)
            if st3[fake]["state"] == "running" and fired_at is None:
                fired_at = i
    finally:
        unstub()
    if fired_at == STALL_POLLS:
        say(f"SELF-TEST: 92% coverage fired on stalled poll {fired_at}, "
            f"recording a gap of {st3[fake]['gap']} ids")
    else:
        fails.append(f"92% stall case fired at poll {fired_at}, "
                     f"expected {STALL_POLLS}")

    if fails:
        for x in fails:
            say(f"SELF-TEST FAIL: {x}")
        return 1
    say("SELF-TEST: all checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--once", action="store_true", help="one poll, then exit")
    ap.add_argument("--interval", type=int, default=POLL_SECONDS)
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    os.makedirs(LOGDIR, exist_ok=True)
    say(f"watcher starting; pid={os.getpid()} "
        f"sid={os.getsid(0)} interval={args.interval}s")
    say(f"watching: {', '.join(WATCHED)}")

    st = load_state()
    poll_n = 0
    while True:
        poll_n += 1
        try:
            poll_once(st, poll_n)
        except Exception as e:  # a bad poll must never kill the watcher
            say(f"poll {poll_n} raised {type(e).__name__}: {e}; continuing")
        save_state(st)
        write_status(st, poll_n)

        if all(st[s]["state"] in ("DONE", "FAILED") for s in WATCHED):
            done = [s for s in WATCHED if st[s]["state"] == "DONE"]
            bad = [s for s in WATCHED if st[s]["state"] == "FAILED"]
            say(f"all watched datasets settled: DONE={done} FAILED={bad}; "
                f"watcher exiting")
            write_status(st, poll_n, note="watcher exited: all datasets settled")
            return 0
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
