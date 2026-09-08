import numpy as np

from scripts.idea_discovery.project_melt_dense_evidence import canonical_resample, peak_basin


def test_canonical_grid_and_peak_basin():
    dense = canonical_resample(np.asarray([0., 1., 0.]), 1.0)
    assert len(dense) == 4
    assert peak_basin(np.asarray([0., .6, .9, .7, .1]), 0, 5, .5) == (1, 4)
