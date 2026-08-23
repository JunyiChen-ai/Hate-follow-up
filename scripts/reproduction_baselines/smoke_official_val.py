#!/usr/bin/env python3
"""Fast protocol checks for the official-validation reproduction track."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from hate_common import data as hdata  # noqa: E402

EXPECTED_HCS_TEST = "0d6486438a27493322ffdc862cbcc079448a9b7530fd53b0203564992f800a2b"


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for corpus in hdata.CORPORA:
        train, val = hdata.load_train_val(corpus)
        test = hdata.load_split(corpus, "test")
        assert train and val and test, (corpus, len(train), len(val), len(test))
        assert not set(train) & set(val)
        assert not set(train) & set(test)
        assert not set(val) & set(test)
        labels = hdata.load_labels(corpus)
        assert all(v in labels for v in train + val + test)
        print(corpus, len(train), len(val), len(test),
              "val pos", sum(labels[v] for v in val))
    path = REPO / "results/reproduction/splits/hateclipseg_test.txt"
    assert digest(path) == EXPECTED_HCS_TEST
    print("official-validation protocol smoke: PASS")


if __name__ == "__main__": main()
