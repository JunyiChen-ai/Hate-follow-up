#!/bin/bash
# Detached driver for the 4B completion sweep.
#
# Pre-registration: docs/duplex/PREREG_4b_completion.md (frozen, committed
# ba00da1, before a single new score was written).
#
# Scores Qwen3-VL-4B-Instruct on the three corpora it has never seen -- MHClip
# EN test (161), MHClip ZH test (149) and HateClipSeg (394) -- with
# src/duplex/extract_duplex_readout.py unmodified, under exactly the overrides
# the 8B arm read on each corpus. One GPU job at a time, strictly sequential.
# The weights are already cached; nothing is downloaded.
#
# Launch:
#   setsid nohup bash scripts/duplex/run_4b_completion.sh \
#     > results/scale_emergence/fourb_run.log 2>&1 < /dev/null &
#
# Progress: results/scale_emergence/fourb_run.log
# Stage:    results/scale_emergence/FOURB_STATUS  (final DONE, or FAILED: <stage>)

set -u
set -o pipefail

ROOT=/home/jehc223/Hate-follow-up
OUT=$ROOT/results/scale_emergence
STATUS=$OUT/FOURB_STATUS
MODEL=Qwen/Qwen3-VL-4B-Instruct
ARM=4b
STOP_AFTER=3600          # frozen stop rule: one hour of wall clock
T0=$(date +%s)
mkdir -p "$OUT"
rm -f "$OUT/FOURB_DONE"

cd "$ROOT" || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
# shellcheck disable=SC1091
source ~/venvs/SafetyContradiction/bin/activate

say()   { echo "[$(date '+%F %T')] $*"; }
stage() { echo "$1" > "$STATUS"; say "=== STAGE: $1 ==="; }
fail()  { stage "FAILED: $1"; say "ABORTING at $1"; exit 1; }
nlines(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }
elapsed(){ echo $(( $(date +%s) - T0 )); }
check_clock(){
  if [ "$(elapsed)" -gt "$STOP_AFTER" ]; then
    stage "STOPPED: wall clock $(elapsed)s exceeded the frozen ${STOP_AFTER}s rule"
    say "STOP RULE fired at $(elapsed)s"
    exit 2
  fi
}

say "run_4b_completion.sh starting; pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ')"
say "gpu: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1)"

# ---------------------------------------------------------------- pre-flight
# The frozen Yes and No id sets must come back from the 4B tokenizer unchanged
# and the checkpoint must be the cached 36-layer, 2560-wide one. The readout is
# meaningless otherwise.
stage "preflight"
python - "$MODEL" <<'PY' || fail "preflight"
import sys
sys.path.insert(0, "src/our_method")
from transformers import AutoProcessor, AutoConfig
from score_holistic_2b import build_binary_token_ids

FROZEN_YES = [7414, 9454, 9693, 9834, 14004, 14080]
FROZEN_NO = [902, 2152, 2308, 2753, 5664, 8996]

model = sys.argv[1]
proc = AutoProcessor.from_pretrained(model)
ids = build_binary_token_ids(proc.tokenizer)
yes, no = sorted(ids["Yes"]), sorted(ids["No"])
print("Yes", yes, [proc.tokenizer.decode([t]) for t in yes])
print("No ", no, [proc.tokenizer.decode([t]) for t in no])
assert yes == FROZEN_YES, f"Yes ids drifted: {yes}"
assert no == FROZEN_NO, f"No ids drifted: {no}"
assert not set(yes) & set(no)
cfg = AutoConfig.from_pretrained(model).text_config
print("layers", cfg.num_hidden_layers, "hidden", cfg.hidden_size)
assert (cfg.num_hidden_layers, cfg.hidden_size) == (36, 2560), "4B config drifted"
print("PREFLIGHT OK")
PY

# --------------------------------------------------------------------- score
# Strictly one GPU job at a time: the loop body blocks until the judge returns.
# $1 log slug, $2 working directory, $3 --dataset value, $4 expected coverage.
score_one() {
  local slug="$1" wd="$2" ds="$3" expected="$4"
  local dir=$wd/judge_$ARM
  mkdir -p "$dir"
  [ -f "$wd/c2_overrides.json" ] || fail "overrides missing for $slug"

  local fp
  fp=$(python - "$wd/c2_overrides.json" "$MODEL" <<'PY'
import hashlib, sys
h = hashlib.sha256()
h.update(open(sys.argv[1], "rb").read())
h.update(b"|limit=0|split=test|model=" + sys.argv[2].encode())
print(h.hexdigest())
PY
)
  if [ -f "$dir/inputs.sha256" ] && [ "$(cat "$dir/inputs.sha256")" != "$fp" ]; then
    say "$slug: input fingerprint changed; parking old scores"
    mv "$dir" "${dir}.stale.$(date +%Y%m%d_%H%M%S)"; mkdir -p "$dir"
  fi
  echo "$fp" > "$dir/inputs.sha256"

  for attempt in 1 2 3; do
    check_clock
    local have
    have=$(nlines "$dir/scores.jsonl")
    [ "$have" -ge "$expected" ] && break
    say "$slug attempt $attempt: $have/$expected scored"
    python -u src/duplex/extract_duplex_readout.py \
      --dataset "$ds" --split test \
      --model "$MODEL" \
      --transcript-limit 0 \
      --transcript-override-json "$wd/c2_overrides.json" \
      --out-dir "$dir" 2>&1 | tee -a "$OUT/${slug}_judge_${ARM}.log"
    local rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && break
    say "$slug attempt $attempt returned $rc; retrying in 20s"
    sleep 20
  done

  # Coverage asserts, frozen at 161 / 149 / 394, and the 8B arm must agree.
  local got nh have8
  got=$(nlines "$dir/scores.jsonl")
  nh=$(find "$dir/hidden" -name '*.npy' 2>/dev/null | wc -l)
  have8=$(nlines "$wd/judge_8b/scores.jsonl")
  say "$slug coverage: scores $got, hidden $nh, expected $expected, 8B reference $have8"
  [ "$got" -eq "$expected" ] || fail "coverage $slug (scores $got != $expected)"
  [ "$nh"  -eq "$expected" ] || fail "coverage $slug (hidden $nh != $expected)"
  [ "$have8" -eq "$expected" ] || fail "coverage $slug (8B $have8 != $expected)"
  say "$slug done at $(elapsed)s"
}

stage "score_mhclip_en"
score_one mhclip_en "$ROOT/results/testruns/mhclip_en" MHClip_EN 161

stage "score_mhclip_zh"
score_one mhclip_zh "$ROOT/results/testruns/mhclip_zh" MHClip_ZH 149

stage "score_hateclipseg"
score_one hateclipseg "$ROOT/results/hateclipseg" HateClipSeg 394

stage "DONE"
: > "$OUT/FOURB_DONE"
say "=== DONE in $(elapsed)s ==="
