#!/usr/bin/env python
"""Dependency-free checks for the LAVAD cohort and score adapter."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hate_common import data as hdata
from lavad.pack_scores import align, weighted_scores
from lavad.prepare import cohort


def main():
    n = 0
    for corpus in hdata.CORPORA:
        ids, gt = cohort(corpus)
        assert ids == [v for v in hdata.load_split(corpus, "test") if v in gt]
        assert ids and all(len(gt[v]) > 0 for v in ids)
        n += len(ids)
        print("%-12s %d videos, %d frames" %
              (corpus, len(ids), sum(len(gt[v]) for v in ids)))
    raw = {"0": 0.2, "1": 0.8}
    assert np.allclose(weighted_scores(raw, {}, 1), [0.2, 0.8])
    refined = {"0": {"0": 0.0, "1": 1.0}}
    sim = {"0": {"0": 0.0, "1": 0.0}}
    assert np.allclose(weighted_scores(refined, sim, 2), [0.5])
    assert np.array_equal(align(np.array([.1, .2]), 2, "x"), [.1, .2])
    try:
        align(np.array([.1]), 2, "x")
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch was accepted")
    print("PASS: %d frozen-cohort videos; raw/refined packing and strict alignment" % n)


if __name__ == "__main__":
    main()
