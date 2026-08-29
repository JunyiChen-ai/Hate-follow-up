from scripts.label_free_adapt.evaluate import within_video_macro
from scripts.label_free_adapt.mechanisms import (
    boundary_distributions, consistency_curve, parse_numbered_intervals,
)
from scripts.label_free_adapt.schema import Interval, Prediction, intervals_to_curve

import numpy as np


def test_prediction_and_curve():
    intervals = [Interval(0.5, 1.5, 0.8)]
    curve = intervals_to_curve(intervals, duration=2.0)
    assert curve == [0.0, 0.0, 0.8, 0.8, 0.8, 0.8, 0.0, 0.0]
    Prediction("x", "d", "v", 2.0, score_curve=curve,
               intervals=intervals, calls=3).validate()


def test_prediction_serializes_call_cost():
    row = Prediction("x", "d", "v", 1.0, score_curve=[0.0], calls=7).to_dict()
    assert row["calls"] == 7


def test_numbered_parser_and_merge():
    text = '{"intervals":[{"start_frame":1,"end_frame":2,"confidence":0.7}],"evidence":{}}'
    spans, _ = parse_numbered_intervals(text, [0.0, 1.0, 2.0, 3.0], 4.0)
    assert spans == [Interval(1.0, 3.0, 0.7)]


def test_consistency_penalizes_instability():
    stable = consistency_curve([[0.8, 0.2], [0.8, 0.2]])
    unstable = consistency_curve([[1.0, 0.2], [0.0, 0.2]])
    assert stable[0] > unstable[0]
    assert stable[1] == unstable[1]


def test_boundary_distributions_are_normalized():
    starts, ends = boundary_distributions([[Interval(1.0, 2.0)]], 3.0)
    assert np.isclose(sum(starts), 1.0)
    assert np.isclose(sum(ends), 1.0)
    assert int(np.argmax(starts)) == 4
    assert int(np.argmax(ends)) == 8


def test_within_video_skips_single_class():
    result = within_video_macro(
        {"a": np.array([0, 1]), "b": np.array([1, 1])},
        {"a": np.array([0.1, 0.9]), "b": np.array([0.2, 0.8])},
    )
    assert result["within_video_macro_ROC_AUC"] == 1.0
    assert result["n_videos_defined"] == 1
    assert result["n_videos_skipped"] == 1
