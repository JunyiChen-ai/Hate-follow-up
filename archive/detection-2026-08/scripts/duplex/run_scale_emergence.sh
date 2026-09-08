#!/bin/bash
# Detached driver for the scale-emergence confirmatory arm.
#
# Pre-registration: docs/duplex/PREREG_scale_emergence.md (frozen, committed
# d0941a0, before this script downloaded a single weight).
#
# Downloads Qwen3-VL-4B-Instruct, checks that its tokenizer yields the frozen
# Yes and No id sets, then scores ImpliHateVid test_clean and HateMM test_clean
# with src/duplex/extract_duplex_readout.py unmodified under the same overrides
# the 8B and 2B arms read. One GPU job at a time, strictly sequential.
#
# Launch:
#   setsid nohup bash scripts/duplex/run_scale_emergence.sh \
#     > results/scale_emergence/run.log 2>&1 < /dev/null &
#
# Progress: results/scale_emergence/run.log
# Stage:    results/scale_emergence/STATUS  (final DONE, or FAILED: <stage>)

set -u
set -o pipefail

ROOT=/home/jehc223/Hate-follow-up
OUT=$ROOT/results/scale_emergence
STATUS=$OUT/STATUS
MODEL=Qwen/Qwen3-VL-4B-Instruct
ARM=4b
mkdir -p "$OUT"
rm -f "$OUT/DONE"

cd "$ROOT" || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
# shellcheck disable=SC1091
source ~/venvs/SafetyContradiction/bin/activate

say()   { echo "[$(date '+%F %T')] $*"; }
stage() { echo "$1" > "$STATUS"; say "=== STAGE: $1 ==="; }
fail()  { stage "FAILED: $1"; say "ABORTING at $1"; exit 1; }
nlines(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }

say "run_scale_emergence.sh starting; pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ')"
say "gpu: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1)"

# ------------------------------------------------------------------ download
stage "download"
python - "$MODEL" <<'PY' || exit 1
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=[
    "*.json", "*.safetensors", "*.txt", "*.py"])
print("snapshot:", p)
PY
[ $? -eq 0 ] || fail "download"

# ---------------------------------------------------------------- pre-flight
# The frozen Yes and No id sets must come back from the 4B tokenizer unchanged;
# the readout is meaningless otherwise.
stage "preflight"
python - "$MODEL" <<'PY' || exit 1
import sys, os
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
[ $? -eq 0 ] || fail "preflight"

# --------------------------------------------------------------------- score
# Strictly one GPU job at a time: the loop body blocks until the judge returns.
score_one() {
  local slug="$1" ds="$2" expected="$3"
  local wd=$ROOT/results/testruns/$slug
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
    say "$slug attempt $attempt: $(nlines "$dir/scores.jsonl")/$expected scored"
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

  # Coverage assert. The 4B arm must match the 8B arm's coverage exactly.
  local got have8
  got=$(nlines "$dir/scores.jsonl")
  have8=$(nlines "$wd/judge_8b/scores.jsonl")
  local nh
  nh=$(find "$dir/hidden" -name '*.npy' | wc -l)
  say "$slug coverage: scores $got, hidden $nh, 8B reference $have8"
  [ "$got" -eq "$have8" ] || fail "coverage $slug (scores $got != 8B $have8)"
  [ "$nh" -eq "$have8" ] || fail "coverage $slug (hidden $nh != 8B $have8)"
}

stage "score_implihatevid"
score_one implihatevid ImpliHateVid 400

stage "score_hatemm"
score_one hatemm HateMM 215

stage "DONE"
: > "$OUT/DONE"
say "=== DONE ==="
