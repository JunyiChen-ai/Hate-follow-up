#!/bin/bash
# Stage and commit one dataset's test-run deliverables.
#
#   testrun_commit.sh <slug> <message-file>
#
# Split out of run_testrun.sh so the git index work happens in its own script
# rather than inside a quoted -c string, and so the caller can hold a flock
# across it: several dataset runs can finish at once and must not interleave
# their staging. Exit 3 means there was nothing to commit.

set -u
SLUG=${1:?slug}
MSGFILE=${2:?message file}
cd /home/jehc223/Hate-follow-up || exit 1

git add "docs/duplex/reports/test_c2_${SLUG}_8b.json" \
        "docs/duplex/reports/test_c2_${SLUG}_2b.json" \
        docs/duplex/TEST_RUNS_NOTE.md \
        scripts/duplex/testrun_media.py \
        scripts/duplex/testrun_note.py \
        scripts/duplex/testrun_watch.py \
        scripts/duplex/testrun_commit.sh \
        scripts/duplex/run_testrun.sh \
        scripts/duplex/crossbench_analyze.py 2>/dev/null

git diff --cached --quiet && exit 3

git -c user.name="Junyi Chen" -c user.email="jehc223@aucklanduni.ac.nz" \
    commit -q -F "$MSGFILE"
