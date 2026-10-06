"""Tests for the pre-registered statistics (scripts/canonical/stats.py).

Reference values are computed independently of the module: exact binomial sums with
fractions for McNemar, the textbook Wilson bounds, and a hand-worked Connor sample size.
"""

from __future__ import annotations

import builtins
import math
from fractions import Fraction

import numpy as np
import pytest

from scripts.canonical import stats


def _exact_mcnemar_reference(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(Fraction(math.comb(n, i), 2 ** n) for i in range(min(b, c) + 1))
    return min(1.0, float(2 * tail))


def test_wilson_matches_textbook_bounds():
    assert stats.wilson_interval(0, 10) == pytest.approx((0.0, 0.2775327998628892), abs=1e-12)
    lo, hi = stats.wilson_interval(10, 10)
    assert lo == pytest.approx(0.7224672001371107, abs=1e-12) and hi == pytest.approx(1.0, abs=1e-12)
    lo, hi = stats.wilson_interval(5, 10)
    assert lo + hi == pytest.approx(1.0, abs=1e-12)  # symmetric around one half
    assert all(math.isnan(v) for v in stats.wilson_interval(0, 0))
    r = stats.rate(1595, 1800)
    assert (r.successes, r.n) == (1595, 1800) and r.lo < r.rate < r.hi
    assert 0.87 < r.lo < r.hi < 0.90
    assert math.isnan(stats.rate(0, 0).rate)
    assert set(r.as_dict()) == {"successes", "n", "rate", "lo", "hi"}


@pytest.mark.parametrize("b,c", [(25, 8), (40, 24), (3, 0), (7, 7), (0, 0), (1, 30), (120, 95)])
def test_exact_mcnemar_matches_exact_binomial(b, c):
    assert stats.exact_mcnemar(b, c) == pytest.approx(_exact_mcnemar_reference(b, c), rel=1e-9, abs=1e-15)


def test_exact_mcnemar_without_scipy(monkeypatch):
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("scipy hidden for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    for b, c in [(25, 8), (40, 24), (3, 0), (7, 7), (1, 30)]:
        assert stats.exact_mcnemar(b, c) == pytest.approx(_exact_mcnemar_reference(b, c), rel=1e-9)


def test_seed_average_refuses_dropped_or_duplicated_seeds():
    rows = [{"arm": "A", "query_uid": u, "train_seed": s, "solved": v}
            for u, s, v in [("q1", 1, 1), ("q1", 2, 0), ("q2", 1, 1), ("q2", 2, 1)]]
    assert stats.seed_average(rows) == {"A": {"q1": 0.5, "q2": 1.0}}
    with pytest.raises(ValueError, match="duplicate observation"):
        stats.seed_average(rows + [rows[0]])
    with pytest.raises(ValueError, match="different replicate sets"):
        stats.seed_average(rows[:3])


def test_paired_compare_counts_and_intervals():
    a = {f"q{i}": 1.0 for i in range(40)}
    b = {f"q{i}": (0.0 if i < 10 else 1.0) for i in range(40)}
    res = stats.paired_compare(a, b, n_boot=4000, n_perm=4000, seed=3)
    assert (res.n_units, res.wins, res.losses, res.ties) == (40, 10, 0, 30)
    assert res.diff == pytest.approx(0.25)
    assert 0 < res.lo <= res.diff <= res.hi
    assert res.p_value < 0.01
    assert stats.superior(res) and stats.non_inferior(res, margin=0.0)
    again = stats.paired_compare(a, b, n_boot=4000, n_perm=4000, seed=3)
    assert again == res  # seeded, so reproducible
    with pytest.raises(ValueError, match="different unit sets"):
        stats.paired_compare(a, {"q0": 1.0})


def test_sign_flip_and_bootstrap_edge_cases():
    assert stats.sign_flip_pvalue(np.zeros(12)) == 1.0
    assert stats.sign_flip_pvalue(np.array([1.0, -1.0] * 20), n_perm=2000) > 0.5
    assert stats.paired_bootstrap_ci(np.full(9, 0.4), n_boot=500) == pytest.approx((0.4, 0.4))
    assert all(math.isnan(v) for v in stats.paired_bootstrap_ci(np.array([])))
    lo, hi = stats.paired_bootstrap_ci(np.arange(10.0), n_boot=2000, stat=np.median)
    assert lo <= 4.5 <= hi


def test_non_inferiority_is_decided_by_the_lower_bound():
    res = stats.PairedResult(n_units=100, mean_a=0.8, mean_b=0.82, diff=-0.02, lo=-0.04, hi=0.0,
                             p_value=0.2, wins=5, losses=7, ties=88)
    assert stats.non_inferior(res, margin=0.05)
    assert not stats.non_inferior(res, margin=0.03)
    assert stats.non_inferior(res, margin=-0.05)  # the sign of the margin is ignored
    assert not stats.superior(res)


def test_hierarchical_bootstrap_pairs_units_and_resamples_seeds():
    units = [f"q{i}" for i in range(30)]
    table_a = {"s1": {u: 1.0 for u in units}, "s2": {u: 1.0 for u in units}}
    table_b = {"s1": {u: 0.0 for u in units}, "s2": {u: 1.0 for u in units}}
    point, lo, hi = stats.hierarchical_bootstrap_diff(table_a, table_b, n_boot=2000, seed=1)
    assert point == pytest.approx(0.5)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(1.0)  # seed spread dominates with two seeds
    same = stats.hierarchical_bootstrap_diff(table_a, table_a, n_boot=500, seed=1)
    assert same == pytest.approx((0.0, 0.0, 0.0))
    bad = {"s1": {u: 1.0 for u in units[:-1]}}
    with pytest.raises(ValueError, match="different unit set"):
        stats.hierarchical_bootstrap_diff(table_a, bad, n_boot=10)


def test_latency_helpers():
    point, lo, hi = stats.quantile_with_ci([1.0, 2.0, 3.0, 4.0, 5.0], 0.5, n_boot=1000)
    assert point == 3.0 and lo <= point <= hi
    assert all(math.isnan(v) for v in stats.quantile_with_ci([], 0.5))
    a = {f"q{i}": 2.0 * (i + 1) for i in range(20)}
    b = {f"q{i}": float(i + 1) for i in range(20)}
    out = stats.paired_log_ratio(a, b, n_boot=500)
    assert out["gm_ratio"] == pytest.approx(2.0) and out["median_ratio"] == pytest.approx(2.0)
    assert out["lo"] == pytest.approx(2.0) and out["hi"] == pytest.approx(2.0)
    d = [0.3, -0.1, 0.2, 0.4, -0.2, 0.1]  # the permutation count governs the p-value
    a, b = {f"q{i}": math.exp(x) for i, x in enumerate(d)}, {f"q{i}": 1.0 for i in range(len(d))}
    out = stats.paired_log_ratio(a, b, n_boot=500, n_perm=999, seed=3)
    assert out["p_value"] == stats.sign_flip_pvalue(np.log([a[f"q{i}"] for i in range(len(d))]), n_perm=999, seed=4)
    with pytest.raises(ValueError, match="strictly positive"):
        stats.paired_log_ratio({"q": 0.0}, {"q": 1.0})


def test_holm_is_monotone_and_capped():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    adj = stats.holm({"a": 0.5, "b": 0.4, "c": 0.001})
    assert adj == pytest.approx({"c": 0.003, "b": 0.8, "a": 0.8})
    assert stats.holm({"x": 0.9, "y": 0.8}) == pytest.approx({"y": 1.0, "x": 1.0})
    assert stats.holm({}) == {}


def test_discordant_pairs_needed_matches_connor():
    # p_disc 0.2, ratio 2: pb 2/15, pc 1/15, so n = (1.96*sqrt(0.2) + 0.8416*sqrt(0.2 - 1/225))^2 / (1/225) = 350.8
    assert stats.discordant_pairs_needed(0.2, 2.0) == 351
    assert stats.discordant_pairs_needed(0.4, 2.0) < stats.discordant_pairs_needed(0.2, 2.0)
    assert stats.discordant_pairs_needed(0.2, 1.5) > stats.discordant_pairs_needed(0.2, 2.0)
    for p_disc, ratio in [(0.0, 2.0), (1.5, 2.0), (0.2, 1.0), (0.2, 0.0)]:
        with pytest.raises(ValueError):
            stats.discordant_pairs_needed(p_disc, ratio)
