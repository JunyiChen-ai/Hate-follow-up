from scripts.idea_discovery.run_cwa_text_warrants import matched_windows, text_in_window


def test_matched_windows_requires_full_native_flank():
    assert matched_windows(4.0, 7.0, 12.0) == ([1.0, 4.0], [7.0, 10.0])
    assert matched_windows(1.0, 4.0, 12.0) == (None, [4.0, 7.0])


def test_text_in_window_uses_overlap_and_order():
    chunks = [
        {"span": [0.0, 2.0], "text": "a"},
        {"span": [2.0, 4.0], "text": "b"},
        {"span": [4.0, 6.0], "text": "c"},
    ]
    assert text_in_window(chunks, 1.0, 4.0) == "a b"
