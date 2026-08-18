#!/usr/bin/env python
"""Train VadCLIP on a hateful-video corpus (HateMM by default).

Thin launcher. The loop lives in vadclip/train.py; every option is documented in
vadclip/option.py. --corpus takes hatemm, mhclip_en or mhclip_zh.

    python scripts/reproduction_baselines/train_vadclip_hatemm.py --corpus hatemm
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vadclip.train import main

if __name__ == "__main__":
    sys.exit(main())
