#!/usr/bin/env python
"""Train DSANet on a hateful-video corpus (HateMM by default).

Thin launcher. The loop lives in dsanet/train.py; every option is documented in
dsanet/option.py. --corpus takes hatemm, mhclip_en or mhclip_zh.

    python scripts/reproduction_baselines/train_dsanet_hatemm.py --corpus hatemm
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsanet.train import main

if __name__ == "__main__":
    sys.exit(main())
