"""Statistics for the canonical Jigsaw evaluation, fixed before any sealed run.

Conventions (pre-registered in internal/CANONICAL_RERUN_PLAN.md):

* The unit of analysis is the query. A query evaluated under several training
  seeds contributes one paired observation: its outcome is averaged over seeds
  first. Seed-to-seed spread is reported separately and never pooled into n.
* Rates carry Wilson 95% intervals. Paired differences carry percentile
  bootstrap intervals over queries and a sign-flip randomization p-value.
* A family of claims tested together (one table) is Holm-adjusted.
* "No worse than" claims use a pre-registered non-inferiority margin and are
  decided by the lower bound of the paired interval.

Only numpy is required. scipy is used when present for exact binomial tails.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial rate."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Rate:
    successes: int
    n: int
    rate: float
    lo: float
    hi: float

    def as_dict(self) -> dict:
        return asdict(self)


def rate(successes: int, n: int) -> Rate:
    lo, hi = wilson_interval(successes, n)
    return Rate(int(successes), int(n), successes / n if n else float("nan"), lo, hi)


# ---------------------------------------------------------------------------
# Exact tests on discordant pairs
# ---------------------------------------------------------------------------


def _log_binom_cdf_half(k: int, n: int) -> float:
    """log P(X <= k) for X ~ Binomial(n, 1/2), computed in log space."""
    if k < 0:
        return -math.inf
    if k >= n:
        return 0.0
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in range(k + 1)
    ]
    top = max(log_terms)
    return top + math.log(sum(math.exp(t - top) for t in log_terms))


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    try:  # scipy gives the same value with better speed for large n
        from scipy.stats import binomtest

        return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        return min(1.0, 2.0 * math.exp(_log_binom_cdf_half(min(b, c), n)))


# ---------------------------------------------------------------------------
# Paired comparisons with training-seed replicates
# ---------------------------------------------------------------------------


def seed_average(
    rows: Iterable[Mapping],
    *,
    unit: str = "query_uid",
    arm: str = "arm",
    replicate: str = "train_seed",
    value: str = "solved",
) -> dict[str, dict[str, float]]:
    """Average a per-(arm, replicate, unit) outcome over replicates.

    Returns {arm: {unit: mean outcome}}. Every (arm, unit) must appear under the
    same replicate set within an arm; a missing replicate raises, because a
    silently dropped seed biases the paired comparison.
    """
    buckets: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        a, u, r = str(row[arm]), str(row[unit]), str(row[replicate])
        per_unit = buckets.setdefault(a, {}).setdefault(u, {})
        if r in per_unit:
            raise ValueError(f"duplicate observation arm={a} unit={u} replicate={r}")
        per_unit[r] = float(row[value])
    out: dict[str, dict[str, float]] = {}
    for a, per_arm in buckets.items():
        replicate_sets = {frozenset(v) for v in per_arm.values()}
        if len(replicate_sets) != 1:
            raise ValueError(f"arm {a}: units observed under different replicate sets")
        out[a] = {u: float(np.mean(list(v.values()))) for u, v in per_arm.items()}
    return out


def aligned_pairs(a: Mapping[str, float], b: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align two {unit: value} maps; the unit sets must be identical."""
    if set(a) != set(b):
        missing = sorted(set(a) ^ set(b))[:5]
        raise ValueError(f"paired comparison over different unit sets, e.g. {missing}")
    units = sorted(a)
    return (
        np.array([a[u] for u in units], dtype=float),
        np.array([b[u] for u in units], dtype=float),
        units,
    )


@dataclass(frozen=True)
class PairedResult:
    n_units: int
    mean_a: float
    mean_b: float
    diff: float
    lo: float
    hi: float
    p_value: float
    wins: int
    losses: int
    ties: int

    def as_dict(self) -> dict:
        return asdict(self)


def sign_flip_pvalue(d: np.ndarray, n_perm: int = 100_000, seed: int = 0) -> float:
    """Two-sided randomization test of mean(d) = 0 by random sign flips."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    batch = 10_000
    while done < n_perm:
        m = min(batch, n_perm - done)
        signs = rng.integers(0, 2, size=(m, d.size), dtype=np.int8) * 2 - 1
        stats = np.abs((signs * d).mean(axis=1))
        hits += int((stats >= observed - 1e-12).sum())
        done += m
    return (hits + 1) / (n_perm + 1)


def paired_bootstrap_ci(
    d: np.ndarray,
    n_boot: int = 20_000,
    seed: int = 0,
    alpha: float = 0.05,
    stat: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = np.apply_along_axis(stat, 1, d[idx]) if stat is not np.mean else d[idx].mean(axis=1)
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def paired_compare(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    n_boot: int = 20_000,
    n_perm: int = 100_000,
    seed: int = 0,
) -> PairedResult:
    """Compare arm a against arm b on the same units (a minus b)."""
    xa, xb, _ = aligned_pairs(a, b)
    d = xa - xb
    lo, hi = paired_bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return PairedResult(
        n_units=int(d.size),
        mean_a=float(xa.mean()) if xa.size else float("nan"),
        mean_b=float(xb.mean()) if xb.size else float("nan"),
        diff=float(d.mean()) if d.size else float("nan"),
        lo=lo,
        hi=hi,
        p_value=sign_flip_pvalue(d, n_perm=n_perm, seed=seed + 1),
        wins=int((d > 0).sum()),
        losses=int((d < 0).sum()),
        ties=int((d == 0).sum()),
    )


def non_inferior(result: PairedResult, margin: float) -> bool:
    """a is no worse than b by more than margin (same units as the outcome)."""
    return bool(result.lo > -abs(margin))


def superior(result: PairedResult, alpha: float = 0.05) -> bool:
    return bool(result.lo > 0 and result.p_value < alpha)


# ---------------------------------------------------------------------------
# Hierarchical bootstrap: queries and training seeds both resampled
# ---------------------------------------------------------------------------


def hierarchical_bootstrap_diff(
    table_a: Mapping[str, Mapping[str, float]],
    table_b: Mapping[str, Mapping[str, float]],
    *,
    n_boot: int = 5_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Difference of mean outcomes when both units and seeds are random.

    table_x is {replicate: {unit: outcome}}. Units are resampled once per draw and
    shared by both arms (pairing); replicates are resampled independently per arm.
    Returns (point estimate, lo, hi).
    """
    units = sorted(next(iter(table_a.values())).keys())
    for tab in (table_a, table_b):
        for rep, per_unit in tab.items():
            if sorted(per_unit) != units:
                raise ValueError(f"replicate {rep} covers a different unit set")
    mat_a = np.array([[table_a[r][u] for u in units] for r in sorted(table_a)], dtype=float)
    mat_b = np.array([[table_b[r][u] for u in units] for r in sorted(table_b)], dtype=float)
    point = float(mat_a.mean() - mat_b.mean())
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    n_u = len(units)
    for i in range(n_boot):
        u_idx = rng.integers(0, n_u, size=n_u)
        ra = rng.integers(0, mat_a.shape[0], size=mat_a.shape[0])
        rb = rng.integers(0, mat_b.shape[0], size=mat_b.shape[0])
        diffs[i] = mat_a[np.ix_(ra, u_idx)].mean() - mat_b[np.ix_(rb, u_idx)].mean()
    return point, float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def quantile_with_ci(
    values: Sequence[float], q: float, n_boot: int = 5_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.quantile(x[idx], q, axis=1)
    return (
        float(np.quantile(x, q)),
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def paired_log_ratio(
    a: Mapping[str, float], b: Mapping[str, float], *, n_boot: int = 20_000, n_perm: int = 100_000, seed: int = 0
) -> dict:
    """Geometric-mean ratio a/b of paired positive costs (latency, bytes read)."""
    xa, xb, _ = aligned_pairs(a, b)
    if (xa <= 0).any() or (xb <= 0).any():
        raise ValueError("log-ratio needs strictly positive costs")
    d = np.log(xa) - np.log(xb)
    lo, hi = paired_bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return {
        "n_units": int(d.size),
        "gm_ratio": float(np.exp(d.mean())),
        "lo": float(np.exp(lo)),
        "hi": float(np.exp(hi)),
        "p_value": sign_flip_pvalue(d, n_perm=n_perm, seed=seed + 1),
        "median_ratio": float(np.exp(np.median(d))),
    }


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


# ---------------------------------------------------------------------------
# Power (used to size the sealed query sets before they are generated)
# ---------------------------------------------------------------------------


def discordant_pairs_needed(p_disc: float, ratio: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Units needed for a paired binary comparison (Connor 1987 approximation).

    p_disc is the expected share of discordant units; ratio = b/c among them.
    """
    if not 0 < p_disc <= 1 or ratio <= 0 or ratio == 1:
        raise ValueError("need 0 < p_disc <= 1 and ratio != 1")
    from statistics import NormalDist

    za = NormalDist().inv_cdf(1 - alpha / 2)
    zb = NormalDist().inv_cdf(power)
    pb = p_disc * ratio / (1 + ratio)
    pc = p_disc / (1 + ratio)
    diff = pb - pc
    n = (za * math.sqrt(p_disc) + zb * math.sqrt(p_disc - diff * diff)) ** 2 / (diff * diff)
    return int(math.ceil(n))


if __name__ == "__main__":  # small self-check, no data needed
    assert abs(exact_mcnemar(25, 8) - 0.004551384132355452) < 1e-9
    assert abs(exact_mcnemar(40, 24) - 0.05994118956699923) < 1e-9
    lo, hi = wilson_interval(1595, 1800)
    assert 0.87 < lo < hi < 0.90
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert abs(adj["a"] - 0.03) < 1e-12 and abs(adj["c"] - 0.06) < 1e-12 and abs(adj["b"] - 0.06) < 1e-12
    print("stats self-check OK")
