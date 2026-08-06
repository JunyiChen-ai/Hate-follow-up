"""Test-run stage 0: locate each split id's source media in B2 and pull it.

The bucket layout is not fixed. ImpliHateVid mp4s sit under three label-named
folders, MultiHateClip's sit under a different project's `raw/` tree, and
HateMM's are not there yet at all, so nothing here may assume a path. The
matcher works on basenames instead: an object is a candidate for video id V if
its filename stem equals V and its extension is a media extension. That is the
same rule the upload watcher uses, which is why both import it from here.

Two modes:

  coverage  read a cached `rclone lsf` listing and report, per dataset, how many
            of that dataset's test-split ids have media in the bucket. Pure
            bookkeeping, no network, no download.

  pull      resolve the ids and copy them into one flat local directory, then
            verify each file's size against the size the listing reported.

Ids are treated as opaque. Nothing here prints an id, so a log or a committed
artifact never carries one.
"""

import argparse
import json
import os
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

# data_utils resolves the dataset roots off HVD_DATA_ROOT and defaults to a path
# that only exists on the source cluster. A detached watcher inherits almost no
# environment, so the machine's own data root is filled in here when the
# variable is unset rather than left to fail at the first annotation read.
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

REMOTE = "b2:junyi-data"
RCLONE = os.path.expanduser("~/.local/bin/rclone")

# Video containers first, then audio-only. The pipeline only needs an audio
# track, so an .m4a upload is as good as an .mp4 for the restoration stage.
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".flv", ".avi", ".mov", ".m4v")
AUDIO_EXTS = (".m4a", ".wav", ".mp3")
MEDIA_EXTS = VIDEO_EXTS + AUDIO_EXTS
EXT_RANK = {e: i for i, e in enumerate(MEDIA_EXTS)}

DATASETS = {
    "implihatevid": "ImpliHateVid",
    "hatemm": "HateMM",
    "mhclip_en": "MHClip_EN",
    "mhclip_zh": "MHClip_ZH",
}

LISTING = os.path.join(ROOT, "results", "testruns", "cache", "listing.txt")


def refresh_listing(path=LISTING, timeout=3600):
    """Re-list the bucket into `path`, atomically. Returns (ok, message).

    Written to a temp file and renamed, so a transient B2 failure leaves the
    previous listing intact rather than truncating it to nothing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            p = subprocess.run(
                [RCLONE, "lsf", "-R", "--files-only", "--fast-list",
                 "--format", "ps", "--separator", "|", REMOTE],
                stdout=f, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"rclone lsf timed out after {timeout}s"
    except OSError as e:
        return False, f"rclone lsf could not start: {e}"
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace").strip().splitlines()
        return False, f"rclone lsf rc={p.returncode}: {err[-1] if err else ''}"
    n = sum(1 for _ in open(tmp))
    if n == 0:
        return False, "rclone lsf returned an empty listing; keeping the old one"
    os.replace(tmp, path)
    return True, f"listed {n} objects"


def load_index(path=LISTING):
    """basename stem -> list of (path, size), best extension first."""
    idx = {}
    if not os.path.exists(path):
        return idx
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "|" not in line:
                continue
            p, _, size = line.rpartition("|")
            base = os.path.basename(p)
            stem, ext = os.path.splitext(base)
            ext = ext.lower()
            if ext not in EXT_RANK:
                continue
            try:
                size = int(size)
            except ValueError:
                continue
            if size < 1000:
                continue
            idx.setdefault(stem, []).append((p, size))
    for stem in idx:
        idx[stem].sort(key=lambda t: (EXT_RANK[os.path.splitext(t[0])[1].lower()],
                                      -t[1]))
    return idx


def test_ids(dataset):
    """The dataset's test_clean ids, de-duplicated, order preserved.

    MHClip_EN carries one duplicated id upstream; dropping it here keeps the
    denominators consistent with what the analyser reports.
    """
    from data_utils import load_clean_split_ids
    seen, out = set(), []
    for v in load_clean_split_ids(dataset, "test"):
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def resolve(ids, idx):
    """(found: id -> (path, size), missing: [id])."""
    found, missing = {}, []
    for v in ids:
        c = idx.get(v)
        if c:
            found[v] = c[0]
        else:
            missing.append(v)
    return found, missing


def coverage(slug, idx):
    ids = test_ids(DATASETS[slug])
    found, missing = resolve(ids, idx)
    return {"slug": slug, "dataset": DATASETS[slug], "n_ids": len(ids),
            "n_found": len(found), "n_missing": len(missing),
            "frac": round(len(found) / len(ids), 4) if ids else 0.0}


def local_path(dest_dir, vid, remote_path):
    return os.path.join(dest_dir, vid + os.path.splitext(remote_path)[1].lower())


def pull(slug, dest_dir, idx, log=print):
    """Copy every resolvable id into dest_dir, byte-verified. Returns a dict."""
    ids = test_ids(DATASETS[slug])
    found, missing = resolve(ids, idx)
    os.makedirs(dest_dir, exist_ok=True)

    todo = []
    for v, (rp, size) in found.items():
        lp = local_path(dest_dir, v, rp)
        if os.path.exists(lp) and os.path.getsize(lp) == size:
            continue
        todo.append((v, rp, size, lp))
    log(f"pull[{slug}]: {len(ids)} ids, {len(found)} resolvable, "
        f"{len(found) - len(todo)} already local and byte-exact, "
        f"{len(todo)} to fetch, {len(missing)} with no media in the bucket")

    # rclone --files-from takes paths relative to the remote root. Copying with
    # the source tree flattened is not something --files-from does, so each file
    # is fetched with copyto. Batched into one rclone process per chunk to keep
    # the per-call startup cost down.
    ok, failed = 0, []
    for i, (v, rp, size, lp) in enumerate(todo, 1):
        p = subprocess.run(
            [RCLONE, "copyto", f"{REMOTE}/{rp}", lp,
             "--retries", "5", "--low-level-retries", "20", "--timeout", "5m"],
            capture_output=True)
        if p.returncode != 0 or not os.path.exists(lp):
            failed.append(v)
        elif os.path.getsize(lp) != size:
            log(f"pull[{slug}]: size mismatch on item {i}; removing and failing it")
            os.remove(lp)
            failed.append(v)
        else:
            ok += 1
        if i % 25 == 0 or i == len(todo):
            log(f"pull[{slug}]: {i}/{len(todo)} fetched, {len(failed)} failed")

    verified = sum(1 for v, (rp, size) in found.items()
                   if os.path.exists(local_path(dest_dir, v, rp))
                   and os.path.getsize(local_path(dest_dir, v, rp)) == size)
    res = {"slug": slug, "dest_dir": dest_dir, "n_ids": len(ids),
           "n_resolvable": len(found), "n_fetched_now": ok,
           "n_failed": len(failed), "n_verified_local": verified,
           "n_no_media_in_bucket": len(missing)}
    log(f"pull[{slug}]: {json.dumps(res)}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["coverage", "pull", "refresh"])
    ap.add_argument("--slug", choices=list(DATASETS))
    ap.add_argument("--dest-dir")
    ap.add_argument("--listing", default=LISTING)
    ap.add_argument("--refresh", action="store_true",
                    help="re-list the bucket before doing anything else")
    args = ap.parse_args()

    if args.mode == "refresh" or args.refresh:
        ok, msg = refresh_listing(args.listing)
        print(f"refresh: {'ok' if ok else 'FAILED'}: {msg}")
        if args.mode == "refresh":
            return 0 if ok else 1

    idx = load_index(args.listing)
    if args.mode == "coverage":
        slugs = [args.slug] if args.slug else list(DATASETS)
        print(json.dumps([coverage(s, idx) for s in slugs], indent=1))
        return 0

    if not args.slug or not args.dest_dir:
        raise SystemExit("pull needs --slug and --dest-dir")
    res = pull(args.slug, args.dest_dir, idx)
    return 0 if res["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
