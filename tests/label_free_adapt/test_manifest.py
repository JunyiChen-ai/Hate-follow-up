from scripts.label_free_adapt.build_manifest import duration_balanced


def test_duration_balanced_is_deterministic_and_spans_range():
    rows = [{"video_id": str(i), "duration": float(i)} for i in range(10)]
    got = duration_balanced(rows, 4)
    assert [x["video_id"] for x in got] == ["0", "3", "6", "9"]
