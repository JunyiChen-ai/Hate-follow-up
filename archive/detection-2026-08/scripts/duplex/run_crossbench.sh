#!/bin/bash
# Detached end-to-end driver for the cross-benchmark measurement.
#
# Owns the whole pipeline on HateMM, MHClip_EN and MHClip_ZH: extract audio and
# voice activity, re-transcribe with Whisper large-v3 on cuda, apply the frozen
# degeneracy gate, run the 8B judge and then the 2B contrast judge over each
# benchmark's train_clean split, analyse, write the tracked artifacts, and
# commit them. One GPU job at a time. Every stage is idempotent and resume-safe,
# so re-running this script after any interruption picks up where it stopped.
#
# Stages A-C degrade rather than fail when a benchmark's source media is not
# reachable: every video then takes the gate's no_fresh_pass branch and the
# judge reads the dataset transcript. The reports record the restoration
# coverage, so a zero shows up as a zero instead of disappearing.
#
# Launch:
#   setsid nohup bash scripts/duplex/run_crossbench.sh \
#     > results/crossbench/logs/run.log 2>&1 < /dev/null &
#
# Progress: results/crossbench/logs/run.log
# Stage:    results/crossbench/STATUS  (final value DONE, or FAILED: <stage>)

set -u
set -o pipefail

ROOT=/home/jehc223/Hate-follow-up
OUT=$ROOT/results/crossbench
STATUS=$OUT/STATUS
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"

cd "$ROOT" || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
export PATH=$HOME/.local/bin:$PATH
# shellcheck disable=SC1091
source ~/venvs/SafetyContradiction/bin/activate

DATASETS=(HateMM MHClip_EN MHClip_ZH)
SLUGS=(hatemm mhclip_en mhclip_zh)
# Conventional mp4 locations, matching data_utils.MP4_SUBDIRS. Absent today;
# dropping the videos here and re-running exercises the restoration route.
MP4DIRS=(
  /home/jehc223/data/HateMM/video
  /home/jehc223/data/Multihateclip/English/video_mp4
  /home/jehc223/data/Multihateclip/Chinese/video
)

say() { echo "[$(date '+%F %T')] $*"; }
stage() { echo "$1" > "$STATUS"; say "=== STAGE: $1 ==="; }
fail() { stage "FAILED: $1"; say "ABORTING at $1"; exit 1; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

say "run_crossbench.sh starting; pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ')"
say "gpu: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>&1 | head -1)"

# ---------------------------------------------------------------- sanitising
# Drop unparseable trailing lines (a kill mid-append leaves a partial record)
# and de-duplicate by video_id, keeping the first complete record. For judge
# scores, also drop rows whose z is null or non-finite: the resume logic already
# ignores them, but leaving them in makes the line count misleading.
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

# ------------------------------------------------------- push the backlog first
push_branch() {
  if git push origin follow-up >> "$LOGDIR/push.log" 2>&1; then
    say "pushed"
    return 0
  fi
  local sock
  sock=$(ls -t /run/user/135258174/vscode-git-*.sock 2>/dev/null | head -1)
  if [ -n "$sock" ] && \
     VSCODE_GIT_IPC_HANDLE="$sock" git push origin follow-up \
       >> "$LOGDIR/push.log" 2>&1; then
    say "pushed via vscode git sock $sock"
    return 0
  fi
  say "PUSH FAILED; commits are local only. See $LOGDIR/push.log"
  return 1
}

stage "push_backlog"
say "local HEAD $(git rev-parse --short HEAD); pushing any backlog"
push_branch || true

# ------------------------------------------------- A-C: audio, ASR, gate
for i in "${!DATASETS[@]}"; do
  DS=${DATASETS[$i]}
  SLUG=${SLUGS[$i]}
  WORK=$OUT/$SLUG
  mkdir -p "$WORK"

  stage "audio_$SLUG"
  sanitize "$WORK/audio_meta.jsonl" meta
  python -u scripts/duplex/crossbench_audio.py \
    --dataset "$DS" --out-dir "$WORK" --mp4-dir "${MP4DIRS[$i]}" 2>&1 \
    | tee -a "$LOGDIR/audio_$SLUG.log" || fail "audio_$SLUG"

  NWAV=$(python - "$WORK/audio_meta.jsonl" <<'PY'
import json, sys
n = 0
try:
    for line in open(sys.argv[1]):
        line = line.strip()
        if line and json.loads(line).get("wav_ok"):
            n += 1
except FileNotFoundError:
    pass
print(n)
PY
)
  say "$SLUG: $NWAV videos with usable audio"

  stage "asr_$SLUG"
  if [ "$NWAV" -gt 0 ]; then
    sanitize "$WORK/fresh_transcripts.jsonl" transcripts
    for attempt in 1 2 3; do
      HAVE=$(nlines "$WORK/fresh_transcripts.jsonl")
      [ "$HAVE" -ge "$NWAV" ] && break
      say "asr $SLUG attempt $attempt (have $HAVE/$NWAV)"
      python -u scripts/duplex/crossbench_asr.py \
        --dataset "$DS" --out-dir "$WORK" 2>&1 | tee -a "$LOGDIR/asr_$SLUG.log"
      sanitize "$WORK/fresh_transcripts.jsonl" transcripts
      NOW=$(nlines "$WORK/fresh_transcripts.jsonl")
      if [ "$NOW" -le "$HAVE" ]; then
        say "asr $SLUG attempt $attempt added nothing ($NOW); stopping retries"
        break
      fi
    done
  else
    say "$SLUG: no usable audio, skipping ASR. Every video will take the "
    say "$SLUG: gate's no_fresh_pass branch and keep its dataset transcript."
  fi

  stage "gate_$SLUG"
  python -u scripts/duplex/crossbench_gate.py \
    --dataset "$DS" --out-dir "$WORK" 2>&1 \
    | tee "$LOGDIR/gate_$SLUG.log" || fail "gate_$SLUG"
done

# ------------------------------------------------------------------- D: judge
# The judge input is the overrides map plus the uncapped transcript setting. If
# that input changes between runs -- which is exactly what happens when the
# source videos arrive and the ASR route finally fires -- the old scores are
# stale and must not be resumed into. A fingerprint file in each judge directory
# enforces that: on mismatch the directory is moved aside and rebuilt.
run_judge() {
  local ds="$1" slug="$2" model="$3" arm="$4"
  local work=$OUT/$slug
  local dir=$work/judge_$arm
  mkdir -p "$dir"

  local fp
  fp=$(python - "$work/c2_overrides.json" "$model" <<'PY'
import hashlib, sys
h = hashlib.sha256()
h.update(open(sys.argv[1], "rb").read())
h.update(b"|limit=0|model=" + sys.argv[2].encode())
print(h.hexdigest())
PY
)
  if [ -f "$dir/inputs.sha256" ] && [ "$(cat "$dir/inputs.sha256")" != "$fp" ]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    say "judge $slug/$arm: input fingerprint changed; parking old scores at ${dir}.stale.$ts"
    mv "$dir" "${dir}.stale.$ts"
    mkdir -p "$dir"
  fi
  echo "$fp" > "$dir/inputs.sha256"

  sanitize "$dir/scores.jsonl" scores
  say "judge $slug/$arm: $(nlines "$dir/scores.jsonl") already scored"
  python -u src/duplex/extract_duplex_readout.py \
    --dataset "$ds" --split train \
    --model "$model" \
    --transcript-limit 0 \
    --transcript-override-json "$work/c2_overrides.json" \
    --out-dir "$dir" 2>&1 | tee -a "$LOGDIR/judge_${slug}_$arm.log"
  local rc=${PIPESTATUS[0]}
  sanitize "$dir/scores.jsonl" scores
  say "judge $slug/$arm: now $(nlines "$dir/scores.jsonl") (rc=$rc)"
  return "$rc"
}

# 8B first, all three datasets, then the 2B contrast arm. One GPU job at a time.
for ARM_SPEC in "8b:Qwen/Qwen3-VL-8B-Instruct" "2b:Qwen/Qwen3-VL-2B-Instruct"; do
  ARM=${ARM_SPEC%%:*}
  MODEL=${ARM_SPEC#*:}
  for i in "${!DATASETS[@]}"; do
    DS=${DATASETS[$i]}
    SLUG=${SLUGS[$i]}
    stage "judge_${SLUG}_$ARM"
    for attempt in 1 2 3; do
      run_judge "$DS" "$SLUG" "$MODEL" "$ARM" && break
      say "judge $SLUG/$ARM attempt $attempt returned nonzero; resuming after 20s"
      sleep 20
    done
    N=$(nlines "$OUT/$SLUG/judge_$ARM/scores.jsonl")
    say "judge $SLUG/$ARM final coverage: $N"
  done
done

# ---------------------------------------------------------------- E: analysis
stage "analysis"
mkdir -p docs/duplex/reports
N_OK=0
for ARM in 8b 2b; do
  for i in "${!DATASETS[@]}"; do
    DS=${DATASETS[$i]}
    SLUG=${SLUGS[$i]}
    if python -u scripts/duplex/crossbench_analyze.py \
         --dataset "$DS" \
         --judge-dir "$OUT/$SLUG/judge_$ARM" \
         --work-dir "$OUT/$SLUG" \
         --out "docs/duplex/reports/crossbench_${SLUG}_$ARM.json" \
         --model-name "Qwen3-VL-${ARM^^}-Instruct" \
         > "$LOGDIR/analyze_${SLUG}_$ARM.log" 2>&1; then
      say "wrote docs/duplex/reports/crossbench_${SLUG}_$ARM.json"
      N_OK=$((N_OK + 1))
    else
      say "analysis failed for $SLUG/$ARM; see $LOGDIR/analyze_${SLUG}_$ARM.log"
      tail -20 "$LOGDIR/analyze_${SLUG}_$ARM.log"
    fi
  done
done
say "analysis: $N_OK/6 reports written"
[ "$N_OK" -gt 0 ] || fail "analysis"

stage "note"
python -u scripts/duplex/crossbench_note.py \
  docs/duplex/CROSSBENCH_NOTE.md 2>&1 | tee "$LOGDIR/note.log" || fail "note"

# ------------------------------------------------------------------- F: commit
stage "commit"
git add docs/duplex/reports/crossbench_*.json \
        docs/duplex/CROSSBENCH_NOTE.md \
        scripts/duplex/crossbench_audio.py \
        scripts/duplex/crossbench_asr.py \
        scripts/duplex/crossbench_gate.py \
        scripts/duplex/crossbench_analyze.py \
        scripts/duplex/crossbench_note.py \
        scripts/duplex/run_crossbench.sh || fail "commit"

if git diff --cached --quiet; then
  say "nothing staged to commit"
else
  read -r -d '' MSG <<'EOM'
Cross-benchmark measurement: HateMM, MHClip-EN, MHClip-ZH

Carries the channel-restoration method off ImpliHateVid and onto the other three
benchmarks' own train_clean splits, 8B plus the 2B contrast arm, one judge call
per video, label-free KDE-valley threshold recomputed per dataset on its own
raw-z distribution.

Every component is inherited: the audio, ASR and gate stages import their
constants and their text handling from the frozen channel-restoration scripts,
the judge and the raw-z readout come from extract_duplex_readout.py unmodified,
and the threshold recipe is the one frozen in rawz_detector_train.json, which
this run reproduces on the ImpliHateVid C0 scores as a self-check before
applying it anywhere new.

The restoration stage's coverage is reported per dataset rather than assumed.
Where the source media was not reachable, the gate takes its no_fresh_pass
branch, the judge reads the dataset transcript, and the report says so: the
numbers then measure the judge and the threshold travelling across benchmarks,
not the restoration itself.

Measurement on train only; no test split of any dataset was read. Labels are
used for evaluation alone. The threshold is computed from the score
distribution, and the oracle threshold appears only as a reported gap.
EOM
  git -c user.name="Junyi Chen" -c user.email="jehc223@aucklanduni.ac.nz" \
      commit -m "$MSG" || fail "commit"
  say "committed $(git rev-parse --short HEAD)"
fi
push_branch || true

stage "DONE"
say "=== DONE ==="
say "reports: docs/duplex/reports/crossbench_*.json"
say "note:    docs/duplex/CROSSBENCH_NOTE.md"
say "HEAD:    $(git rev-parse HEAD)"
