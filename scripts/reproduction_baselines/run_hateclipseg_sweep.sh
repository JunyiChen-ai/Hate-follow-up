#!/usr/bin/env bash
# The whole HateClipSeg sweep, one GPU job at a time, in the owner's run order.
#
# HateClipSeg is the fourth corpus. Nothing here is a new harness: every stage
# is the existing runner for that method with CORPORA restricted to the new
# corpus, so a HateClipSeg cell is produced by the same code that produced the
# three corpora already in docs/duplex/BASELINE_RESULTS.md. The only new script
# is the method's own locator, scripts/duplex/masked_parallel_isolation_hateclipseg.py.
#
# Strictly serial: one stage finishes before the next starts, so the machine
# never holds more than one GPU job. Detached-friendly:
#
#     cd /home/jehc223/Hate-follow-up
#     setsid nohup bash scripts/reproduction_baselines/run_hateclipseg_sweep.sh \
#         > results/reproduction/hateclipseg_sweep/run.log 2>&1 < /dev/null &
#
# Progress: results/reproduction/hateclipseg_sweep/{run.log,STATUS}
# Finished: results/reproduction/hateclipseg_sweep/DONE (absent means running
#           or dead; STATUS then names the stage that was in flight).
#
# Restrict the sweep with STAGES:
#     STAGES="ours vadclip" bash .../run_hateclipseg_sweep.sh
#
# EventVAD is deliberately absent: it read the floor (0.50 to 0.52) on all
# three prior corpora at 6.4k MLLM calls and 13.4 GPU-hours, and the owner's
# default is not to spend that again.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/venvs/SafetyContradiction/bin/python}"
CORPUS=hateclipseg
STAGES="${STAGES:-ours vadclip dsanet macilsd multihateloc vadr1_anomaly vadr1_hateful}"
OUT="${REPO_ROOT}/results/reproduction/hateclipseg_sweep"

cd "${REPO_ROOT}"
mkdir -p "${OUT}"
rm -f "${OUT}/DONE"

say () { echo "[$(date '+%F %T')] $*"; }
status () { echo "$(date '+%F %T')  $*" > "${OUT}/STATUS"; }

fail () {
    status "FAILED: $1"
    say "STAGE FAILED: $1 (exit $2)"
    exit "$2"
}

run_stage () {
    local name="$1"; shift
    status "running ${name}"
    say "=== stage ${name} start ==="
    "$@" 2>&1 | tee "${OUT}/${name}.log"
    local rc="${PIPESTATUS[0]}"
    if [ "${rc}" -ne 0 ]; then fail "${name}" "${rc}"; fi
    say "=== stage ${name} done ==="
}

say "run_hateclipseg_sweep.sh starting; pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ')"
say "stages: ${STAGES}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

for stage in ${STAGES}; do
    case "${stage}" in
    ours)
        run_stage ours env HVD_DATA_ROOT=/home/jehc223/data "${PYTHON}" -u \
            scripts/duplex/masked_parallel_isolation_hateclipseg.py
        ;;
    vadclip|dsanet)
        run_stage "${stage}" env METHODS="${stage}" CORPORA="${CORPUS}" \
            PYTHON="${PYTHON}" bash scripts/reproduction_baselines/run_all.sh
        ;;
    macilsd)
        run_stage macilsd env CORPORA="${CORPUS}" PYTHON="${PYTHON}" \
            bash scripts/reproduction_baselines/run_all_macilsd.sh
        ;;
    multihateloc)
        run_stage multihateloc env CORPORA="${CORPUS}" PYTHON="${PYTHON}" \
            bash scripts/reproduction_baselines/multihateloc/run_all.sh
        ;;
    vadr1_anomaly)
        run_stage vadr1_anomaly env CORPORA="${CORPUS}" ARM=anomaly \
            PYTHON="${PYTHON}" bash scripts/reproduction_baselines/run_all_vadr1.sh
        ;;
    vadr1_hateful)
        run_stage vadr1_hateful env CORPORA="${CORPUS}" ARM=hateful \
            PYTHON="${PYTHON}" bash scripts/reproduction_baselines/run_all_vadr1.sh
        ;;
    *)
        fail "unknown stage ${stage}" 2
        ;;
    esac
done

status "DONE"
date '+%F %T' > "${OUT}/DONE"
say "=== sweep done ==="
