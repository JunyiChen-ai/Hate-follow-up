#!/bin/bash
# MHClip_ZH diagnostic, optional arm: Whisper with the language forced to zh.
#
# Pre-registered trigger, from the diagnostic brief: run this only if at least
# 10 percent of the 149 clips auto-detected a non-Chinese language. Whisper's own
# per-chunk language vote came back empty on the frozen run, so the trigger was
# read off the independent Unicode script profile instead: 24 of 149 fresh
# transcripts (16.1 percent) are not Han-dominant. The trigger fires.
#
# Everything is inherited from the frozen path. The only change is
# generate_kwargs["language"] = "zh", and the output is written to
# fresh_transcripts_zh.jsonl so the frozen fresh_transcripts.jsonl is untouched.
# The same frozen gate then produces a separate override map, and the same 8B
# judge scores the same 149 videos under it.
#
#   bash scripts/duplex/zh_forced_lang_arm.sh
#
# Log: results/testruns/logs/zh_diag.log (appended)

set -u
set -o pipefail

ROOT=/home/jehc223/Hate-follow-up
OUT=$ROOT/results/testruns/mhclip_zh
FORCED=$OUT/forced_zh
GPULOCK=$ROOT/results/testruns/gpu.lock

cd "$ROOT" || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
# shellcheck disable=SC1091
source ~/venvs/SafetyContradiction/bin/activate

say() { echo "[$(date '+%F %T')] [zh_forced] $*"; }

# The gate and the ASR resume logic both key off files inside one directory, so
# the forced arm gets its own directory with the wav and audio metadata shared
# by symlink -- the audio is identical, only the decoding language differs.
mkdir -p "$FORCED"
[ -e "$FORCED/wav" ] || ln -s "$OUT/wav" "$FORCED/wav"
[ -e "$FORCED/audio_meta.jsonl" ] || ln -s "$OUT/audio_meta.jsonl" "$FORCED/audio_meta.jsonl"

say "=== ASR, language forced to zh ==="
flock "$GPULOCK" python -u scripts/duplex/crossbench_asr.py \
  --dataset MHClip_ZH --split test --out-dir "$FORCED" --language zh || exit 1
say "fresh transcripts: $(wc -l < "$FORCED/fresh_transcripts.jsonl")"

say "=== frozen gate over the forced transcripts ==="
python -u scripts/duplex/crossbench_gate.py \
  --dataset MHClip_ZH --split test --out-dir "$FORCED" || exit 1

say "=== 8B judge under the forced-zh override map ==="
mkdir -p "$OUT/judge_8b_forcedzh"
flock "$GPULOCK" python -u src/duplex/extract_duplex_readout.py \
  --dataset MHClip_ZH --split test \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --transcript-limit 0 \
  --transcript-override-json "$FORCED/c2_overrides.json" \
  --out-dir "$OUT/judge_8b_forcedzh" || exit 1
say "scored: $(wc -l < "$OUT/judge_8b_forcedzh/scores.jsonl")"
say "=== DONE ==="
