#!/bin/bash
# Run weakly supervised baseline tasks one after another inside one Slurm job (no job chaining).
# Each task is <kind>:<name>:<corpus>; kind = method (run_method.py) or mil (mil/train_mil.py).
#   bash experiments/20261008_baselines/weaksup_common/run_tasks.sh method:vadclip:hatemm mil:bert:dehate ...
# A failed task prints "TASK FAILED <task>" and the next task still runs; the last line is "ALL TASKS DONE" or
# "FAILED: <tasks>".
set -uo pipefail
cd "$(dirname "$0")/../../.."
failed=()
for task in "$@"; do
  IFS=: read -r kind name corpus <<< "$task"
  echo "$(date -Is) TASK START $task"
  if [ "$kind" = method ]; then
    python -u experiments/20261008_baselines/weaksup_common/run_method.py --method "$name" --corpus "$corpus"
  else
    python -u experiments/20261008_baselines/mil/train_mil.py --feature "$name" --corpus "$corpus"
  fi
  rc=$?
  if [ $rc -eq 0 ]; then echo "$(date -Is) TASK DONE $task"; else echo "$(date -Is) TASK FAILED $task rc=$rc"; failed+=("$task"); fi
done
if [ ${#failed[@]} -eq 0 ]; then echo "$(date -Is) ALL TASKS DONE"; else echo "$(date -Is) FAILED: ${failed[*]}"; exit 1; fi
