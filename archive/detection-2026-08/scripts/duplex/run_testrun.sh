#!/bin/bash
# Detached end-to-end driver for one benchmark's held-out test-split measurement.
#
#   bash scripts/duplex/run_testrun.sh <slug>
#   slug in: implihatevid | hatemm | mhclip_en | mhclip_zh
#
# Owns the whole pipeline for that one dataset: pull the test-split source media
# from B2 and byte-verify it, extract audio and voice activity, re-transcribe
# with Whisper large-v3 on cuda, apply the frozen degeneracy gate, run the 8B
# judge and then the 2B contrast judge over the test_clean split, analyse both
# arms, regenerate the note, commit and push.
#
# Every stage is idempotent and resume-safe: re-running after any interruption
# picks up where it stopped. Every GPU stage is wrapped in a flock on a single
# repo-wide lock, so several of these can be launched at once -- by hand or by
# the upload watcher -- and still only one will touch the GPU at a time.
#
# Launch (detached, survives every disconnect):
#   setsid nohup bash scripts/duplex/run_testrun.sh implihatevid \
#     > results/testruns/logs/implihatevid_run.log 2>&1 < /dev/null &
#
# Progress: results/testruns/logs/<slug>_run.log
# Stage:    results/testruns/<slug>/STATUS   (final value DONE, or FAILED: <stage>)

set -u
set -o pipefail

SLUG=${1:?usage: run_testrun.sh <implihatevid|hatemm|mhclip_en|mhclip_zh>}

case "$SLUG" in
  implihatevid) DS=ImpliHateVid ;;
  hatemm)       DS=HateMM ;;
  mhclip_en)    DS=MHClip_EN ;;
  mhclip_zh)    DS=MHClip_ZH ;;
  *) echo "unknown slug $SLUG" >&2; exit 2 ;;
esac

ROOT=/home/jehc223/Hate-follow-up
OUT=$ROOT/results/testruns/$SLUG
MEDIA=$OUT/media
STATUS=$OUT/STATUS
LOGDIR=$ROOT/results/testruns/logs
GPULOCK=$ROOT/results/testruns/gpu.lock
mkdir -p "$OUT" "$MEDIA" "$LOGDIR"
: > "$GPULOCK" 2>/dev/null || true

cd "$ROOT" || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
export PATH=$HOME/.local/bin:$PATH
# shellcheck disable=SC1091
source ~/venvs/SafetyContradiction/bin/activate

say() { echo "[$(date '+%F %T')] [$SLUG] $*"; }
stage() { echo "$1" > "$STATUS"; say "=== STAGE: $1 ==="; }
fail() { stage "FAILED: $1"; say "ABORTING at $1"; exit 1; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

# Serialise every GPU stage behind one lock. flock waits rather than failing, so
# a run that arrives while another dataset holds the GPU simply queues.
gpu() {
  local what="$1"; shift
  say "waiting for the GPU lock ($what)"
  flock "$GPULOCK" bash -c '"$@"' _ "$@"
  local rc=$?
  say "released the GPU lock ($what, rc=$rc)"
  return $rc
}

say "run_testrun.sh starting; dataset=$DS pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ')"
say "gpu: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>&1 | head -1)"

# ---------------------------------------------------------------- sanitising
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

push_branch() {
  if git push origin follow-up >> "$LOGDIR/push.log" 2>&1; then
    say "pushed"; return 0
  fi
  local sock
  sock=$(ls -t /run/user/135258174/vscode-git-*.sock 2>/dev/null | head -1)
  if [ -n "$sock" ] && \
     VSCODE_GIT_IPC_HANDLE="$sock" git push origin follow-up \
       >> "$LOGDIR/push.log" 2>&1; then
    say "pushed via vscode git sock"; return 0
  fi
  say "PUSH FAILED; commits are local only. See $LOGDIR/push.log"
  return 1
}

N_IDS=$(python -c "
import sys; sys.path.insert(0,'scripts/duplex')
from testrun_media import test_ids, DATASETS
print(len(test_ids(DATASETS['$SLUG'])))")
say "test_clean ids: $N_IDS"

# ------------------------------------------------------------ 0: pull media
# CPU and network only. The bucket listing is refreshed first so a run launched
# by hand picks up anything uploaded since the watcher last looked.
stage "pull"
python -u scripts/duplex/testrun_media.py pull --slug "$SLUG" \
  --dest-dir "$MEDIA" --refresh 2>&1 | tee -a "$LOGDIR/${SLUG}_pull.log"
NMEDIA=$(find "$MEDIA" -type f -size +1k | wc -l)
say "media on disk: $NMEDIA/$N_IDS"

# -------------------------------------------------------- A: audio (CPU)
stage "audio"
sanitize "$OUT/audio_meta.jsonl" meta
python -u scripts/duplex/crossbench_audio.py \
  --dataset "$DS" --split test --out-dir "$OUT" --mp4-dir "$MEDIA" 2>&1 \
  | tee -a "$LOGDIR/${SLUG}_audio.log" || fail "audio"

NWAV=$(python - "$OUT/audio_meta.jsonl" <<'PY'
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
say "$NWAV videos with usable audio"

# --------------------------------------------------------- B: ASR (GPU)
stage "asr"
if [ "$NWAV" -gt 0 ]; then
  sanitize "$OUT/fresh_transcripts.jsonl" transcripts
  for attempt in 1 2 3; do
    HAVE=$(nlines "$OUT/fresh_transcripts.jsonl")
    [ "$HAVE" -ge "$NWAV" ] && break
    say "asr attempt $attempt (have $HAVE/$NWAV)"
    gpu asr python -u scripts/duplex/crossbench_asr.py \
      --dataset "$DS" --split test --out-dir "$OUT" 2>&1 \
      | tee -a "$LOGDIR/${SLUG}_asr.log"
    sanitize "$OUT/fresh_transcripts.jsonl" transcripts
    NOW=$(nlines "$OUT/fresh_transcripts.jsonl")
    if [ "$NOW" -le "$HAVE" ]; then
      say "asr attempt $attempt added nothing ($NOW); stopping retries"
      break
    fi
  done
else
  say "no usable audio; every video takes the gate's no_fresh_pass branch"
fi

# -------------------------------------------------------------- C: gate (CPU)
stage "gate"
python -u scripts/duplex/crossbench_gate.py \
  --dataset "$DS" --split test --out-dir "$OUT" 2>&1 \
  | tee "$LOGDIR/${SLUG}_gate.log" || fail "gate"

# ------------------------------------------------------------- D: judge (GPU)
# The judge input is the overrides map plus the uncapped transcript setting. A
# fingerprint guards resume: if the overrides change -- which is what happens
# when more media arrives and the ASR route finally fires -- the old scores are
# stale and the directory is parked rather than resumed into.
run_judge() {
  local model="$1" arm="$2"
  local dir=$OUT/judge_$arm
  mkdir -p "$dir"
  local fp
  fp=$(python - "$OUT/c2_overrides.json" "$model" <<'PY'
import hashlib, sys
h = hashlib.sha256()
h.update(open(sys.argv[1], "rb").read())
h.update(b"|limit=0|split=test|model=" + sys.argv[2].encode())
print(h.hexdigest())
PY
)
  if [ -f "$dir/inputs.sha256" ] && [ "$(cat "$dir/inputs.sha256")" != "$fp" ]; then
    local ts; ts=$(date +%Y%m%d_%H%M%S)
    say "judge $arm: input fingerprint changed; parking old scores at ${dir}.stale.$ts"
    mv "$dir" "${dir}.stale.$ts"; mkdir -p "$dir"
  fi
  echo "$fp" > "$dir/inputs.sha256"

  sanitize "$dir/scores.jsonl" scores
  say "judge $arm: $(nlines "$dir/scores.jsonl")/$N_IDS already scored"
  gpu "judge_$arm" python -u src/duplex/extract_duplex_readout.py \
    --dataset "$DS" --split test \
    --model "$model" \
    --transcript-limit 0 \
    --transcript-override-json "$OUT/c2_overrides.json" \
    --out-dir "$dir" 2>&1 | tee -a "$LOGDIR/${SLUG}_judge_$arm.log"
  local rc=${PIPESTATUS[0]}
  sanitize "$dir/scores.jsonl" scores
  say "judge $arm: now $(nlines "$dir/scores.jsonl")/$N_IDS (rc=$rc)"
  return "$rc"
}

for ARM_SPEC in "8b:Qwen/Qwen3-VL-8B-Instruct" "2b:Qwen/Qwen3-VL-2B-Instruct"; do
  ARM=${ARM_SPEC%%:*}
  MODEL=${ARM_SPEC#*:}
  stage "judge_$ARM"
  for attempt in 1 2 3; do
    run_judge "$MODEL" "$ARM" && break
    say "judge $ARM attempt $attempt returned nonzero; resuming after 20s"
    sleep 20
  done
  say "judge $ARM final coverage: $(nlines "$OUT/judge_$ARM/scores.jsonl")/$N_IDS"
done

# ---------------------------------------------------------------- E: analysis
stage "analysis"
mkdir -p docs/duplex/reports
N_OK=0
for ARM in 8b 2b; do
  [ "$(nlines "$OUT/judge_$ARM/scores.jsonl")" -gt 0 ] || \
    { say "no $ARM scores; skipping its analysis"; continue; }
  if python -u scripts/duplex/crossbench_analyze.py \
       --dataset "$DS" --split test \
       --judge-dir "$OUT/judge_$ARM" \
       --work-dir "$OUT" \
       --out "docs/duplex/reports/test_c2_${SLUG}_$ARM.json" \
       --model-name "Qwen3-VL-${ARM^^}-Instruct" \
       > "$LOGDIR/${SLUG}_analyze_$ARM.log" 2>&1; then
    say "wrote docs/duplex/reports/test_c2_${SLUG}_$ARM.json"
    N_OK=$((N_OK + 1))
  else
    say "analysis failed for $ARM; see $LOGDIR/${SLUG}_analyze_$ARM.log"
    tail -20 "$LOGDIR/${SLUG}_analyze_$ARM.log"
  fi
done
[ "$N_OK" -gt 0 ] || fail "analysis"

# -------------------------------------------------------------------- F: note
# The note is regenerated from every test report on disk, so datasets that
# finish later simply extend it.
stage "note"
python -u scripts/duplex/testrun_note.py docs/duplex/TEST_RUNS_NOTE.md 2>&1 \
  | tee "$LOGDIR/${SLUG}_note.log" || fail "note"

# ------------------------------------------------------------------ G: commit
# Serialised behind the same lock the GPU stages use, so two datasets finishing
# together cannot interleave a git index.
stage "commit"
MSGFILE=$OUT/commit_msg.txt
cat > "$MSGFILE" <<EOM
Held-out test measurement: $DS test_clean, 8B and 2B

Runs the channel-restoration method unchanged on the held-out test split of
$DS: source media pulled and byte-verified, audio re-transcribed with Whisper
large-v3 on cuda, the frozen degeneracy gate applied, one judge call per video,
and a label-free KDE-valley threshold computed on this run's own raw-z
distribution with no labels.

Nothing was refitted for the test split. The threshold recipe is the one frozen
in rawz_detector_train.json, which the analysis reproduces on the ImpliHateVid
C0 scores as a self-check before applying it here. Train and test scores are
never pooled into one distribution. Labels enter the analysis alone; the oracle
threshold appears only as a reported gap.
EOM

flock "$GPULOCK" bash "$ROOT/scripts/duplex/testrun_commit.sh" "$SLUG" "$MSGFILE"
RC=$?
if [ "$RC" = 3 ]; then say "nothing staged to commit"
elif [ "$RC" != 0 ]; then say "commit failed rc=$RC"
else say "committed $(git rev-parse --short HEAD)"; fi
push_branch || true

stage "DONE"
say "=== DONE ==="
say "reports: docs/duplex/reports/test_c2_${SLUG}_{8b,2b}.json"
say "note:    docs/duplex/TEST_RUNS_NOTE.md"
