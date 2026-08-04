#!/usr/bin/env bash
set -u
cd /data/jehc223/EMNLP3 || exit 1

LOG=logs/meme_baselines/resume_monitor.log
mkdir -p logs/meme_baselines

while true; do
  ts=$(date "+%F %T %Z")
  {
    echo "===== ${ts} ====="
    squeue -u "$USER" -o "%.18i %.22j %.8T %.10M %.20R %.10b %.6D %.8C %.12m"
    echo "rows:"
    find results/meme_baselines -path "*/test_*.jsonl" -type f -exec wc -l {} + 2>/dev/null | sort -k2
    echo "latest_logs:"
    tail -n 8 logs/meme_baselines/mars_32b_awq.log 2>/dev/null || true
    tail -n 8 logs/meme_baselines/lorehm_32b_awq.log 2>/dev/null || true
    tail -n 8 logs/meme_baselines/mod_hate.log 2>/dev/null || true
    echo "recent_errors:"
    grep -RniE "traceback|out of memory|illegal memory|exception|failed|error|gated|token" \
      logs/meme_baselines/slurm logs/meme_baselines/*.log 2>/dev/null \
      | grep -viE "cuda graph|automatically detected platform cuda|conda entry point|monitor_sleep|chunked prefill|tokenizer|tokens per request" \
      | tail -n 25 || true
  } >> "$LOG" 2>&1

  if squeue -u "$USER" -h -o "%i %T" | grep -E "(9749|9750)[[:space:]]+RUNNING" >/dev/null; then
    echo "monitor_sleep=300 reason=resume_running" >> "$LOG"
    sleep 300
  else
    echo "monitor_sleep=3600 reason=resume_not_running" >> "$LOG"
    sleep 3600
  fi
done
