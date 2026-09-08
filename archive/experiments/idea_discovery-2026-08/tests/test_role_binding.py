import numpy as np

from scripts.idea_discovery.run_role_binding import binding_decode


def test_counterfactual_margin_components():
    scores = np.asarray([
        [.1, .1, .1, .6, .1],
        [.6, .1, .1, .1, .1],
        [.7, .05, .05, .1, .1],
        [.1, .6, .1, .1, .1],
    ])
    intervals, _, paths = binding_decode(scores, 4.0)
    assert paths == [[1, 3]]
    assert [x.as_list()[:2] for x in intervals] == [[1.0, 3.0]]
