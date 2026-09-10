#!/bin/bash
# Cross-machine check (CLAUDE.md "多机运行"): code revision on every machine, dirty /
# untracked counts, campus ControlMaster sockets, GPU load, disk, and stray project
# files outside the repository. Run before launching and before reporting.
# Usage: bash scripts/check_machines.sh
# Comparing git commit ids here is the single permitted exception to the hash ban.
LAB="uoa-lab1 uoa-lab2 uoa-lab3 lab-server"
CAMPUS="uoa-campus1 uoa-campus2 uoa-campus3"
LOCAL_HOST=$(hostname)
LAB_ALLOW='^(data|miniconda3|Hate-follow-up|Retrieval-hate|snap|venvs|AgentDebugX|Auto-claude-code-research-in-sleep|MemoryAgen|MDDL|miniconda\.sh|Miniconda3-latest-Linux-x86_64\.sh|CLAUDE\.md|nltk_data)$'

repo_line() {  # $1 = repo path
  cat <<EOS
cd $1 2>/dev/null || { echo "  NO REPO at $1"; exit 0; }
echo "  commit: \$(git rev-parse --short HEAD) \$(git log -1 --format=%cd --date=short)  dirty: \$(git status --short | grep -v '^??' | wc -l)  untracked: \$(git status --short | grep '^??' | wc -l)"
echo "  gpu: \$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';')"
EOS
}

echo "== lab machines (direct run, setsid nohup)"
for h in $LAB; do
  echo "-- $h"
  CMD="$(repo_line '~/Hate-follow-up')
echo \"  disk: \$(df -h ~ | tail -1 | awk '{print \$4\" free\"}')\"
stray=\$(ls ~ | grep -vE '$LAB_ALLOW'); [ -n \"\$stray\" ] && echo \"  STRAY in ~: \$(echo \$stray | tr '\n' ' ')\" || echo \"  ~ clean\""
  if ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" true 2>/dev/null; then
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "$CMD" 2>/dev/null || echo "  ssh command failed"
  elif [ "$h" = "uoa-lab1" ] && [ "$LOCAL_HOST" = "sc474397" ]; then
    bash -c "$CMD"
  else
    echo "  UNREACHABLE"
  fi
done

echo "== campus servers (Slurm; token login; needs a live ControlMaster socket)"
for h in $CAMPUS; do
  echo "-- $h"
  if ! ssh -O check "$h" >/dev/null 2>&1; then echo "  SOCKET DEAD: run 'ssh $h true' in a terminal and enter the token"; continue; fi
  CMD="$(repo_line '/data/jehc223/Hate-follow-up')
echo \"  jobs: \$(squeue -u \$USER -h 2>/dev/null | wc -l) mine / \$(squeue -h 2>/dev/null | wc -l) total; free gpus: \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '\$1<1000' | wc -l)\"
echo \"  disk: \$(df -h /data | tail -1 | awk '{print \$4\" free of \"\$2}'); quota: \$(quota -v 2>/dev/null | grep -A1 data-data | tail -1 | awk '{print \$1\"/\"\$2\" KB\"(\$1 ~ /\\*/ ? \" OVER SOFT LIMIT\" : \"\")}')\"
stray=\$(ls /data/jehc223/Hate-follow-up 2>/dev/null | grep -vE '^(AGENTS\.md|CLAUDE\.md|Readme\.md|RESEARCH_ITERATION_RULES\.md|LICENSE|environment_HateVideo\.yml|\.gitignore|\.git|\.cache|archive|configs|data|docs|experiments|research-wiki|runs|scripts|src|third_party)$'); [ -n \"\$stray\" ] && echo \"  STRAY in repo root: \$stray\" || true"
  ssh -o BatchMode=yes -o ConnectTimeout=8 -o RequestTTY=no "$h" "$CMD" 2>/dev/null | grep -v libmamba || echo "  ssh command failed"
done
