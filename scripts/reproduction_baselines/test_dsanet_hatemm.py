#!/usr/bin/env python
"""Score a trained DSANet checkpoint and write per-video frame scores.

Thin launcher over dsanet/infer.py. Writes
results/reproduction/baselines/dsanet/<corpus>/scores.jsonl, one JSON object per
video, each score array on the 1 fps gold grid.

    python scripts/reproduction_baselines/test_dsanet_hatemm.py --corpus hatemm
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsanet.infer import main

if __name__ == "__main__":
    sys.exit(main())
