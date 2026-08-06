#!/bin/bash
# Detached end-to-end driver for the full-corpus C2 measurement.
#
# Owns the whole remaining pipeline: resume Whisper, apply the frozen gate, run
# the 8B judge over all 1283 videos, analyse, write the tracked artifacts, and
# commit them. Every stage is idempotent and resume-safe, so re-running this
# script after any interruption picks up where it stopped.
#
# Launch:
#   setsid nohup bash results/c2_fullcorpus/run_remaining.sh \
#     > results/c2_fullcorpus/logs/run_remaining.log 2>&1 < /dev/null &
#
# Progress: results/c2_fullcorpus/logs/run_remaining.log
# Stage:    results/c2_fullcorpus/STATUS  (final value DONE)
#
# results/ is gitignored, so a copy of this file is committed at
# scripts/duplex/run_c2_fullcorpus.sh.

set -u
set -o pipefail

ROOT=/home/jehc223/Hate-follow-up
OUT=$ROOT/results/c2_fullcorpus
IDS=$OUT/ids_full1283.json
STATUS=$OUT/STATUS
LOGDIR=$OUT/logs
SELF=$OUT/run_remaining.sh
mkdir -p "$LOGDIR"

cd "$ROOT" || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export CR_OUT_DIR=$OUT
export CR_MP4_DIR=/home/jehc223/data/ImpliHateVid_video_train
export TOKENIZERS_PARALLELISM=false
# shellcheck disable=SC1091
source ~/venvs/SafetyContradiction/bin/activate

say() { echo "[$(date '+%F %T')] $*"; }
stage() { echo "$1" > "$STATUS"; say "=== STAGE: $1 ==="; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

say "run_remaining.sh starting; pid=$$ pgid=$(ps -o pgid= -p $$ | tr -d ' ')"

# ---------------------------------------------------------------- sanitising
# Drop unparseable trailing lines (a kill mid-append leaves a partial record)
# and de-duplicate by video_id, keeping the first complete record. For the
# judge scores, also drop rows whose z is null or non-finite: the resume logic
# already ignores them, but leaving them in makes the line count misleading.
sanitize() {
  local path="$1" mode="$2"
  [ -f "$path" ] || return 0
  python - "$path" "$mode" <<'PY'
import json, math, os, sys
path, mode = sys.argv[1], sys.argv[2]
kept, seen, dropped = [], set(), 0
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        vid = r.get("video_id")
        if not vid or vid in seen:
            dropped += 1
            continue
        if mode == "scores":
            z = r.get("z")
            if not isinstance(z, (int, float)) or not math.isfinite(z):
                dropped += 1
                continue
        seen.add(vid)
        kept.append(json.dumps(r, ensure_ascii=False))
if dropped:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))
    os.replace(tmp, path)
print(f"sanitize {os.path.basename(path)}: kept {len(kept)}, dropped {dropped}")
PY
}

# ------------------------------------------------------------- A: transcribe
# CUDA is asserted inside channel_restoration_asr.py: it aborts rather than
# falling back to CPU. Nothing here relaxes that.
stage "asr"
sanitize "$OUT/fresh_transcripts.jsonl" transcripts
say "transcripts on disk: $(nlines "$OUT/fresh_transcripts.jsonl")/1283"

for attempt in 1 2 3 4; do
  have=$(nlines "$OUT/fresh_transcripts.jsonl")
  [ "$have" -ge 1283 ] && break
  say "asr attempt $attempt (have $have/1283)"
  python -u scripts/duplex/channel_restoration_asr.py "$IDS" 2>&1 \
    | tee -a "$LOGDIR/asr.log"
  sanitize "$OUT/fresh_transcripts.jsonl" transcripts
  now=$(nlines "$OUT/fresh_transcripts.jsonl")
  if [ "$now" -le "$have" ]; then
    say "asr attempt $attempt added nothing ($now); stopping retries"
    break
  fi
done
say "transcripts complete: $(nlines "$OUT/fresh_transcripts.jsonl")/1283"

# -------------------------------------------------------------------- B: gate
# The frozen degeneracy gate. Videos with no fresh transcript are recorded as
# no_fresh_pass and simply keep their dataset transcript, so a gap here
# degrades coverage rather than breaking the run.
stage "gate"
python -u scripts/duplex/channel_restoration_gate.py "$IDS" "$OUT" 2>&1 \
  | tee "$LOGDIR/gate.log"
say "override map: $(python -c "import json;print(len(json.load(open('$OUT/c2_overrides.json'))))") ids"

# ------------------------------------------------------------------- C: judge
run_judge() {
  local model="$1" dir="$2" tag="$3"
  mkdir -p "$dir"
  sanitize "$dir/scores.jsonl" scores
  say "judge $tag: $(nlines "$dir/scores.jsonl")/1283 already scored"
  python -u src/duplex/extract_duplex_readout.py \
    --dataset ImpliHateVid --split train \
    --model "$model" \
    --transcript-limit 0 \
    --transcript-override-json "$OUT/c2_overrides.json" \
    --out-dir "$dir" 2>&1 | tee -a "$LOGDIR/judge_$tag.log"
  local rc=${PIPESTATUS[0]}
  sanitize "$dir/scores.jsonl" scores
  say "judge $tag: now $(nlines "$dir/scores.jsonl")/1283 (rc=$rc)"
  return "$rc"
}

stage "judge_8b"
for attempt in 1 2 3; do
  run_judge "Qwen/Qwen3-VL-8B-Instruct" "$OUT/judge_8b" "8b" && break
  say "judge 8b attempt $attempt returned nonzero; resuming after 20s"
  sleep 20
done
N8=$(nlines "$OUT/judge_8b/scores.jsonl")
say "judge 8b final coverage: $N8/1283"
if [ "$N8" -lt 1283 ]; then
  say "WARNING: 8B judge coverage is incomplete ($N8/1283). Proceeding to"
  say "WARNING: analysis anyway; the report records coverage and n_missing."
fi

# ---------------------------------------------------------------- D: analysis
stage "analysis_8b"
mkdir -p docs/duplex/reports
if python -u scripts/duplex/c2_fullcorpus_analyze.py \
     --judge-dir "$OUT/judge_8b" \
     --out docs/duplex/reports/c2_fullcorpus_8b.json \
     --model-name "Qwen3-VL-8B-Instruct" \
     --cr-dir "$OUT" > "$LOGDIR/analyze_8b.log" 2>&1; then
  say "wrote docs/duplex/reports/c2_fullcorpus_8b.json"
else
  stage "FAILED: analysis_8b"
  say "analysis failed; see $LOGDIR/analyze_8b.log"
  tail -30 "$LOGDIR/analyze_8b.log"
  exit 1
fi

stage "note"
python -u scripts/duplex/c2_fullcorpus_note.py \
  docs/duplex/reports/c2_fullcorpus_8b.json \
  docs/duplex/C2_FULLCORPUS_NOTE.md 2>&1 | tee "$LOGDIR/note.log"

# ------------------------------------------------------------------- E: commit
commit_and_push() {
  local msg="$1"; shift
  git add "$@" || { say "git add failed"; return 1; }
  if git diff --cached --quiet; then
    say "nothing staged to commit"
    return 0
  fi
  git -c user.name="Junyi Chen" -c user.email="jehc223@aucklanduni.ac.nz" \
      commit -m "$msg" || { say "git commit failed"; return 1; }
  say "committed $(git rev-parse --short HEAD)"
  if git push origin follow-up >> "$LOGDIR/push.log" 2>&1; then
    say "pushed"
  else
    say "push failed; retrying with VSCODE_GIT_IPC_HANDLE"
    local sock
    sock=$(ls -t /run/user/135258174/vscode-git-*.sock 2>/dev/null | head -1)
    if [ -n "$sock" ] && \
       VSCODE_GIT_IPC_HANDLE="$sock" git push origin follow-up \
         >> "$LOGDIR/push.log" 2>&1; then
      say "pushed via vscode git sock"
    else
      say "PUSH FAILED; the commit is local only. See $LOGDIR/push.log"
    fi
  fi
}

stage "commit_8b"
cp "$SELF" scripts/duplex/run_c2_fullcorpus.sh

read -r -d '' MSG8B <<'EOM'
Full-corpus C2 measurement: 8B, all 1283 train_clean videos

Extends the channel-restoration C2 condition from the 662 dismissed videos to
the whole train_clean corpus, so the methodized pipeline --- fresh full-length
Whisper transcript, one 8B judge call, label-free KDE-valley threshold --- can
be read without conditioning on the stratum the C0 scores themselves selected.

Every component is inherited unmodified: the audio, ASR and gate stages from
the frozen channel-restoration scripts, the judge and raw-z readout from
extract_duplex_readout.py, and the threshold recipe from
rawz_detector_train.json, which this run reproduces on the C0 scores as a
self-check before applying it to C2.

Measurement on train only; the test split is untouched. Labels are used for
evaluation alone. The threshold is computed from the score distribution, and
the oracle threshold appears only as a reported gap.
EOM

commit_and_push "$MSG8B" \
  docs/duplex/reports/c2_fullcorpus_8b.json \
  docs/duplex/C2_FULLCORPUS_NOTE.md \
  scripts/duplex/c2_fullcorpus_analyze.py \
  scripts/duplex/c2_fullcorpus_note.py \
  scripts/duplex/run_c2_fullcorpus.sh

# --------------------------------------------------- F: optional 2B contrast
# Secondary. Anything that fails here leaves the 8B deliverables, already
# committed above, untouched.
stage "contrast_2b"
if run_judge "Qwen/Qwen3-VL-2B-Instruct" "$OUT/judge_2b" "2b"; then
  N2=$(nlines "$OUT/judge_2b/scores.jsonl")
  if [ "$N2" -ge 1283 ]; then
    stage "analysis_2b"
    if python -u scripts/duplex/c2_fullcorpus_analyze.py \
         --judge-dir "$OUT/judge_2b" \
         --out docs/duplex/reports/c2_fullcorpus_2b.json \
         --model-name "Qwen3-VL-2B-Instruct" \
         --cr-dir "$OUT" \
         --c0-scores "$ROOT/results/duplex_readout/ImpliHateVid_Qwen3-VL-2B-Instruct/scores.jsonl" \
         --frozen-valley 0.374625 \
         --killtest-scores "" > "$LOGDIR/analyze_2b.log" 2>&1; then
      python -u scripts/duplex/c2_fullcorpus_note.py \
        docs/duplex/reports/c2_fullcorpus_8b.json \
        docs/duplex/C2_FULLCORPUS_NOTE.md \
        docs/duplex/reports/c2_fullcorpus_2b.json 2>&1 | tee -a "$LOGDIR/note.log"
      stage "commit_2b"
      commit_and_push \
"Full-corpus C2: 2B contrast arm

The same full-corpus C2 inputs judged by Qwen3-VL-2B-Instruct, as a boundary
condition on where the pipeline stops working. Secondary to the 8B arm; the
note is regenerated so it carries both." \
        docs/duplex/reports/c2_fullcorpus_2b.json \
        docs/duplex/C2_FULLCORPUS_NOTE.md
    else
      say "2B analysis failed; 8B deliverables stand. See $LOGDIR/analyze_2b.log"
    fi
  else
    say "2B judge incomplete ($N2/1283); skipping 2B analysis"
  fi
else
  say "2B judge failed; skipping. 8B deliverables stand."
fi

stage "DONE"
say "=== DONE ==="
say "8B report: docs/duplex/reports/c2_fullcorpus_8b.json"
say "note:      docs/duplex/C2_FULLCORPUS_NOTE.md"
say "HEAD:      $(git rev-parse HEAD)"
