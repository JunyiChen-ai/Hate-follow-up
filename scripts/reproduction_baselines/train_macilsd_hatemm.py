#!/usr/bin/env python
"""Train MACIL-SD on a hateful-video corpus (HateMM by default).

Thin launcher. The loop lives in macilsd/train.py; every option is documented
in macilsd/option.py. --corpus takes hatemm, mhclip_en or mhclip_zh, and
--modality selects the audio-visual model or one of the two uni-modal
ablations.

    python scripts/reproduction_baselines/train_macilsd_hatemm.py --corpus hatemm
    python scripts/reproduction_baselines/train_macilsd_hatemm.py --corpus hatemm \
        --modality audio
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from macilsd.train import main

if __name__ == "__main__":
    sys.exit(main())
