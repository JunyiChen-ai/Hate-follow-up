#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jehc223/Hate-follow-up"
PY="/home/jehc223/venvs/SafetyContradiction/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT/src/our_method${PYTHONPATH:+:$PYTHONPATH}"
export HVD_DATA_ROOT=/home/jehc223/data

"$PY" scripts/duplex/temporal_attribution_cohorts.py
"$PY" scripts/duplex/temporal_attribution_asr.py
"$PY" scripts/duplex/temporal_attribution_probe.py
"$PY" scripts/duplex/temporal_attribution_analyze.py
