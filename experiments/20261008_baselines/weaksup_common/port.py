#!/usr/bin/env python3
"""Run one scripts/reproduction_baselines port (VadCLIP, DSANet, MultiHateLoc) on this experiment's inputs.

The ports are used unchanged.  Before their entry point runs, `hate_common.data` is pointed at data/weaksup_1fps/:

  FEATURE_ROOT -> data/weaksup_1fps/clip_b16_1fps         (VadCLIP, DSANet)
  multihateloc data FEATURE_ROOT -> data/weaksup_1fps     (vit_b16_imagenet_1fps, vggish_1s, bert_sentence_1fps)
  SPLIT_ROOT   -> data/weaksup_1fps/splits                (train / val, and test = the exact cohort)
  load_labels  -> train/val video labels; every test id gets the placeholder -1 (the ports' dataset classes need a
                  label slot at inference; the value is never used, and no test label is read)
  gt_arrays    -> placeholder arrays whose length is the feature row count (the ports use the gold only to pick
                  ids and to check lengths; no frame label is read)
  CORPORA      -> hatemm, hateclipseg, dehate (DeHate uses the HateMM window 256 / 64, as in Retrieval-hate)
  load_train_val with legacy_resplit=True is refused, so validation is always the split file.

    python experiments/20261008_baselines/weaksup_common/port.py vadclip.train --corpus hatemm --out-dir ... [opts]
    entries: vadclip.train vadclip.infer dsanet.train dsanet.infer multihateloc.train
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PORTS = C.REPO / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(PORTS))
from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402

ENTRIES = ("vadclip.train", "vadclip.infer", "dsanet.train", "dsanet.infer", "multihateloc.train")


def patch():
    hdata.FEATURE_ROOT = str(C.INPUTS / "clip_b16_1fps")
    hdata.SPLIT_ROOT = str(C.INPUTS / "splits")
    hdata.GT_ROOT = None
    hdata.CORPORA = C.CORPORA
    runtime.CORPORA = C.CORPORA
    runtime.VISUAL_LENGTH.setdefault("dehate", 256)
    runtime.ATTN_WINDOW.setdefault("dehate", 64)
    cache = {}

    def load_labels(corpus):
        if corpus not in cache:
            cache[corpus] = C.port_labels(corpus)
        return dict(cache[corpus])

    def gt_arrays(corpus, split="test"):
        if split != "test":
            raise RuntimeError("only the test cohort is scored")
        return {v: np.empty(np.load(hdata.feature_path(corpus, v), mmap_mode="r").shape[0], dtype=np.int8)
                for v in C.split_ids(corpus, "test")}

    original = hdata.load_train_val

    def load_train_val(corpus, labels=None, val_frac=0.1, seed=234, legacy_resplit=False):
        if legacy_resplit:
            raise RuntimeError("legacy re-split is disabled: validation is the fixed val split")
        return original(corpus, labels, val_frac, seed, False)

    hdata.load_labels = load_labels
    hdata.gt_arrays = gt_arrays
    hdata.load_train_val = load_train_val


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ENTRIES:
        raise SystemExit(f"usage: port.py <{'|'.join(ENTRIES)}> [args]")
    entry, argv = sys.argv[1], sys.argv[2:]
    patch()
    module = importlib.import_module(entry)
    if entry == "multihateloc.train":
        module.mdata.FEATURE_ROOT = str(C.INPUTS)
        if module.mdata.hdata is not hdata:
            raise RuntimeError("multihateloc sees a different hate_common.data module")
    return module.main(argv)


if __name__ == "__main__":
    sys.exit(main())
