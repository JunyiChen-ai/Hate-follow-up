#!/usr/bin/env python
"""Score a trained VadCLIP checkpoint and write per-video frame scores.

Thin launcher over vadclip/infer.py. Writes
results/reproduction/baselines/vadclip/<corpus>/scores.jsonl, one JSON object per
video, each score array on the 1 fps gold grid.

    python scripts/reproduction_baselines/test_vadclip_hatemm.py --corpus hatemm
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vadclip.infer import main

if __name__ == "__main__":
    sys.exit(main())
