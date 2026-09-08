import numpy as np

from scripts.idea_discovery.redecode_role_binding import close_and_filter


def test_close_then_filter():
    active = np.asarray([0, 1, 1, 0, 1, 0, 0, 1, 0], bool)
    assert close_and_filter(active, 1, 2).astype(int).tolist() == [0,1,1,1,1,0,0,0,0]
