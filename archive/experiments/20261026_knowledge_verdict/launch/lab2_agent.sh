#!/usr/bin/env bash
# Candidate 51 agent (DashScope API + Wikipedia, CPU only) on uoa-lab2, detached from the SSH session. From the repo
# root on uoa-lab2, after git pull:
#   bash experiments/20261026_knowledge_verdict/launch/lab2_agent.sh [agent_retrieve.py args, e.g. --limit 2]
# Output: runs/20261026_knowledge_verdict/agent/<dataset>/<video_id>.json; run.log, run.pid, usage.jsonl, nohup.out.
# Markers in run.log: "DONE spent_in=" at the end, "FAILED" per failed video, "BUDGET reached" on the cost cap.
set -euo pipefail
cd ~/Hate-follow-up
mkdir -p runs/20261026_knowledge_verdict/agent
setsid nohup ~/miniconda3/envs/HateVideo/bin/python -u experiments/20261026_knowledge_verdict/agent_retrieve.py "$@" \
  > runs/20261026_knowledge_verdict/agent/nohup.out 2>&1 < /dev/null &
echo "started pid $!"
