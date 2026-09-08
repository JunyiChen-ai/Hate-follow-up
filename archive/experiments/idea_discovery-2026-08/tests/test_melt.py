import numpy as np

from scripts.idea_discovery.run_melt import (
    PHASES,
    lifecycle_decode,
    normalize_phase_rows,
    parse_audit,
    sanitized_cohort,
)


def test_normalize_phase_rows_requires_exact_bins_and_simplex():
    value = {str(i): {p: 1 for p in PHASES} for i in range(16)}
    phase = normalize_phase_rows(value)
    assert phase.shape == (16, 5)
    assert np.allclose(phase.sum(1), 1)
    columns = {p: [i + 1 for i in range(16)] for p in PHASES}
    phase = normalize_phase_rows(columns)
    assert phase.shape == (16, 5)
    assert np.allclose(phase.sum(1), 1)


def test_lifecycle_decode_keeps_all_out_and_finds_legal_event():
    all_out = np.full((4, 5), .01)
    all_out[:, 0] = .96
    intervals, _, meta = lifecycle_decode(all_out, 4.0)
    assert intervals == []
    assert meta["path"] == "all_OUT"

    phase = np.full((4, 5), .01)
    phase[0, 0] = .96
    phase[1, 1] = .96
    phase[2, 3] = .96
    phase[3, 0] = .96
    phase /= phase.sum(1, keepdims=True)
    intervals, _, meta = lifecycle_decode(phase, 4.0)
    assert [(x.start, x.end) for x in intervals] == [(1.0, 3.0)]
    assert meta["paths"] == [[1, 3]]


def test_unknown_dominant_bins_remain_no_event():
    phase = np.full((4, 5), .01)
    phase[:, PHASES.index("UNKNOWN")] = .90
    phase[:, PHASES.index("SUPPORT")] = .06
    phase /= phase.sum(1, keepdims=True)
    intervals, _, meta = lifecycle_decode(phase, 4.0)
    assert intervals == []
    assert meta["path"] == "all_OUT"


def test_cohort_rejects_ground_truth_fields(tmp_path):
    path = tmp_path / "unsafe.jsonl"
    path.write_text('{"dataset":"x","duration":1,"video_id":"v",'
                    '"video_path":"x.mp4","label":1}\n')
    try:
        sanitized_cohort(path)
    except RuntimeError as exc:
        assert "unsafe cohort" in str(exc)
    else:
        raise AssertionError("ground-truth-bearing cohort was accepted")


def test_parse_audit_computes_paired_certificate():
    raw = ('{"start":1.0,"end":3.0,"entailment":{"full":90,'
           '"left_trim":50,"right_trim":60,"expanded":88,"shifted":40,'
           '"cited_removed":20,"matched_control":85}}')
    interval, _, cert = parse_audit(raw, [1.0], [3.0], 4.0)
    assert (interval.start, interval.end) == (1.0, 3.0)
    assert cert["left_necessity"] == 40
    assert cert["alignment_effect"] == 50
    assert cert["citation_selectivity"] == 65
