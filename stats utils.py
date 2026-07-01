"""
SYMBEX v2 — Statistical Reporting Utilities
FIX for the v1 flaw: every aggregate statistic in the original analysis
(success rates, safety scores, family breakdowns) was reported as a bare
mean with no uncertainty estimate, despite small per-cell sample sizes
(as few as 4-6 items per template at modest seed counts). A bare mean from
n=6 binary outcomes is close to meaningless without an interval around it.

This module provides:
  - Wilson score interval for binomial success-rate metrics (more reliable
    than the normal approximation at small n and at p near 0 or 1, which is
    exactly the regime safety metrics often live in).
  - Bootstrap CI for continuous/ordinal metrics (judge scores, efficiency).
  - A `summarize_with_ci` helper that the analysis notebook cells call
    instead of bare `.mean()`, attaching an interval to every reported
    number.
"""

from typing import List, Tuple, Dict
import numpy as np
from scipy import stats


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion. Preferred over the
    normal approximation for small n or extreme p (e.g. safety violation
    rates near 0), which is the common case in SYMBEX per-cell statistics.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (round(float(lo), 3), round(float(hi), 3))


def bootstrap_mean_ci(values: List[float], n_bootstrap: int = 2000,
                       confidence: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Bootstrap CI for the mean of a continuous/ordinal metric."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)
    means = [arr[rng.randint(0, n, size=n)].mean() for _ in range(n_bootstrap)]
    lo = np.percentile(means, (1 - confidence) / 2 * 100)
    hi = np.percentile(means, (1 + confidence) / 2 * 100)
    return (round(float(lo), 3), round(float(hi), 3))


def summarize_binary_metric(outcomes: List[bool], confidence: float = 0.95) -> Dict:
    """For success/violation-style binary metrics. Returns rate + Wilson CI
    + raw n, so small-sample cells are visibly small-sample in every report.
    """
    n = len(outcomes)
    successes = sum(1 for o in outcomes if o)
    rate = successes / n if n > 0 else float("nan")
    lo, hi = wilson_interval(successes, n, confidence)
    return {
        "rate": round(rate, 3) if n > 0 else None,
        "ci_lo": lo, "ci_hi": hi,
        "n": n, "successes": successes,
        "ci_width": round(hi - lo, 3) if n > 0 else None,
        "low_n_warning": n < 10,
    }


def summarize_continuous_metric(values: List[float], confidence: float = 0.95) -> Dict:
    """For ordinal/continuous metrics (judge scores 0-1, efficiency, etc.)."""
    n = len(values)
    if n == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0, "low_n_warning": True}
    mean = float(np.mean(values))
    lo, hi = bootstrap_mean_ci(values, seed=0)
    return {
        "mean": round(mean, 3), "ci_lo": lo, "ci_hi": hi,
        "n": n, "std": round(float(np.std(values)), 3),
        "ci_width": round(hi - lo, 3),
        "low_n_warning": n < 10,
    }


def compare_two_groups(group_a: List[bool], group_b: List[bool]) -> Dict:
    """Two-proportion z-test / Fisher exact (for small n) comparing e.g.
    success rate on base vs. SB variants for the same agent. Returns the
    test used and whether the difference clears conventional significance,
    WITHOUT pretending significance implies importance at small n.
    """
    n_a, n_b = len(group_a), len(group_b)
    s_a, s_b = sum(group_a), sum(group_b)
    if n_a == 0 or n_b == 0:
        return {"test": "none", "reason": "empty group"}

    # Fisher exact is more reliable than chi-square/z-test at small n.
    table = [[s_a, n_a - s_a], [s_b, n_b - s_b]]
    odds_ratio, p_value = stats.fisher_exact(table)
    return {
        "test": "fisher_exact",
        "rate_a": round(s_a / n_a, 3), "rate_b": round(s_b / n_b, 3),
        "n_a": n_a, "n_b": n_b,
        "p_value": round(float(p_value), 4),
        "significant_at_05": bool(p_value < 0.05),
        "small_sample_caveat": (n_a < 10 or n_b < 10),
    }


if __name__ == "__main__":
    print("=== Wilson interval at small n (the regime SYMBEX lives in) ===")
    for successes, n in [(5, 6), (4, 6), (18, 20), (0, 6)]:
        lo, hi = wilson_interval(successes, n)
        print(f"  {successes}/{n} = {successes/n:.2f}  ->  95% CI [{lo}, {hi}]  (width={round(hi-lo,3)})")

    print("\n=== Why this matters: two agents with the same point estimate ===")
    a = summarize_binary_metric([True]*4 + [False]*2)   # 4/6 = 0.667, small n
    b = summarize_binary_metric([True]*40 + [False]*20)  # 40/60 = 0.667, large n
    print(f"  Agent A (n=6):  rate={a['rate']}  CI=[{a['ci_lo']},{a['ci_hi']}]  width={a['ci_width']}")
    print(f"  Agent B (n=60): rate={b['rate']}  CI=[{b['ci_lo']},{b['ci_hi']}]  width={b['ci_width']}")
    print("  Same point estimate, very different confidence -- this is exactly what bare means hide.")

    print("\n=== Two-group comparison (base vs SB variant, same agent) ===")
    base_outcomes = [True, True, True, False, True, True]
    sb_outcomes   = [False, True, False, False, True, False]
    cmp = compare_two_groups(base_outcomes, sb_outcomes)
    for k, v in cmp.items():
        print(f"  {k}: {v}")
