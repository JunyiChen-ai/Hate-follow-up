import numpy as np

from scripts.idea_discovery.decode_melt_phases import phase_viterbi, structured_intervals


def test_legal_lifecycle_path():
    # columns: OUT, ENTER, SUPPORT, EXIT, UNKNOWN
    phase = np.asarray([
        [.9, .02, .02, .02, .04],
        [.1, .8, .04, .03, .03],
        [.1, .05, .78, .04, .03],
        [.1, .03, .04, .8, .03],
        [.9, .02, .02, .02, .04],
    ])
    path, _ = phase_viterbi(phase)
    assert path.tolist() == [0, 1, 2, 3, 0]
    intervals, _, meta = structured_intervals(phase, 5.0, [2])
    assert [x.as_list()[:2] for x in intervals] == [[1.0, 4.0]]
    assert meta["kept_paths"] == [[1, 4]]


def test_incomplete_event_loses_to_all_out():
    phase = np.asarray([
        [.8, .1, .03, .02, .05],
        [.8, .1, .03, .02, .05],
    ])
    path, _ = phase_viterbi(phase)
    assert path.tolist() == [0, 0]
