#!/bin/bash
set -euo pipefail

source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction

read -rsp "HF token: " HF_TOKEN_INPUT
echo

if [[ -z "$HF_TOKEN_INPUT" ]]; then
  echo "empty token" >&2
  exit 2
fi

python - <<'PY' "$HF_TOKEN_INPUT"
import sys
from pathlib import Path

token = sys.argv[1].strip()
path = Path.home() / ".cache" / "huggingface" / "token"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(token + "\n")
path.chmod(0o600)
print(f"wrote HF token to {path} with mode 0600")
PY

unset HF_TOKEN_INPUT

python - <<'PY'
from huggingface_hub import whoami

info = whoami()
print("HF auth ok:", info.get("name") or info.get("email") or "authenticated")
PY
