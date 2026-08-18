#!/usr/bin/env python
"""Score a trained MACIL-SD checkpoint and write per-video frame scores.

Thin launcher over macilsd/infer.py. Writes
results/reproduction/baselines/<method>/<corpus>/scores.jsonl, one JSON object
per video, each score array on the 1 fps gold grid. --modality must match the
one the checkpoint was trained with, since it selects both the architecture and
the default output directory.

    python scripts/reproduction_baselines/test_macilsd_hatemm.py --corpus hatemm
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from macilsd.infer import main

if __name__ == "__main__":
    sys.exit(main())
