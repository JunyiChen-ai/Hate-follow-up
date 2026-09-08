import numpy as np

from scripts.idea_discovery.run_counterfactual_evidence import candidates, ecdf, poset_curve


def test_candidate_generation_is_label_free_and_bounded():
    spans = candidates((np.arange(40), np.arange(40)[::-1]), duration=10.0, top_k=3)
    assert len(spans) == 3
    assert all(0 <= start < end <= 10 for start, end in spans)


def test_poset_fallback_is_finite():
    curve = poset_curve(np.linspace(-1, 1, 20), [], duration=5.0)
    assert curve.shape == (20,)
    assert np.isfinite(curve).all()
    assert np.all((ecdf(curve) > 0) & (ecdf(curve) < 1))
