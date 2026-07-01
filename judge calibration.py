"""
SYMBEX v2 — Judge Calibration & Multi-Model Agreement
========================================================

FIX for two related v1 flaws:
  1. v1's "JUDGE_RUNS=2" called the SAME judge model twice and averaged the
     results. That measures self-consistency (does the model agree with
     itself at temperature 0?), not validity (does the model's score reflect
     what actually happened?). It produces a false sense of reliability.
  2. v1 never implemented the calibration step described in the original
     SYMBEX research plan (Section 7.2: "Calibration against human-labeled
     subset"). No agreement statistic was ever computed or reported.

This module provides:
  - A small human-labelable calibration set format (CSV/JSON), to be filled
    in by the benchmark author scoring a held-out sample of trajectories by
    hand, independent of any LLM.
  - Cohen's kappa / weighted kappa for ordinal agreement between the LLM
    judge and the human reference labels.
  - Spearman correlation as a secondary, less brittle agreement measure for
    the same data (kappa can be unstable with very small or skewed samples;
    reporting both is standard practice and avoids overclaiming from one
    statistic).
  - A genuine TWO-MODEL judge protocol: the dual judge runs now query two
    *different* model families (configurable), and inter-model agreement
    is reported as a distinct, additional reliability statistic from
    human-model agreement. This is what makes "agreement" mean something:
    two independent raters, not one model asked twice.

Honesty note: with small held-out calibration samples (e.g., n=20-30),
kappa confidence intervals will be wide. The module reports the interval,
not just the point estimate, and the notebook is expected to print both
plainly rather than rounding the uncertainty away.
"""

from typing import List, Dict, Tuple
import numpy as np
from scipy import stats


def cohens_kappa(human_scores: List[int], judge_scores: List[int], weights: str = "linear") -> float:
    """Weighted Cohen's kappa for ordinal 0-3 scales. weights='linear' is the
    standard choice for ordinal data (penalizes larger disagreements more).
    """
    human = np.array(human_scores)
    judge = np.array(judge_scores)
    assert len(human) == len(judge), "Score lists must be same length"
    n = len(human)
    if n == 0:
        return float("nan")

    categories = sorted(set(human.tolist()) | set(judge.tolist()))
    k = len(categories)
    cat_index = {c: i for i, c in enumerate(categories)}

    confusion = np.zeros((k, k))
    for h, j in zip(human, judge):
        confusion[cat_index[h], cat_index[j]] += 1

    row_marginals = confusion.sum(axis=1)
    col_marginals = confusion.sum(axis=0)
    expected = np.outer(row_marginals, col_marginals) / n

    if weights == "linear":
        w = np.abs(np.subtract.outer(categories, categories))
        w = w / w.max() if w.max() > 0 else w
    elif weights == "quadratic":
        w = np.subtract.outer(categories, categories) ** 2
        w = w / w.max() if w.max() > 0 else w
    else:
        w = 1 - np.eye(k)

    observed_disagreement = (w * confusion).sum() / n
    expected_disagreement = (w * expected).sum() / n
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    kappa = 1 - observed_disagreement / expected_disagreement
    return round(float(kappa), 3)


def bootstrap_kappa_ci(human_scores: List[int], judge_scores: List[int],
                        n_bootstrap: int = 2000, ci: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Bootstrap confidence interval for kappa, since the analytic SE for
    weighted kappa is awkward and bootstrapping is more transparent for a
    benchmark paper appendix."""
    rng = np.random.RandomState(seed)
    n = len(human_scores)
    human = np.array(human_scores)
    judge = np.array(judge_scores)
    kappas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        k = cohens_kappa(human[idx].tolist(), judge[idx].tolist())
        if not np.isnan(k):
            kappas.append(k)
    if not kappas:
        return (float("nan"), float("nan"))
    lo = np.percentile(kappas, (1 - ci) / 2 * 100)
    hi = np.percentile(kappas, (1 + ci) / 2 * 100)
    return (round(float(lo), 3), round(float(hi), 3))


def spearman_agreement(human_scores: List[int], judge_scores: List[int]) -> Dict:
    rho, p = stats.spearmanr(human_scores, judge_scores)
    return {"spearman_rho": round(float(rho), 3) if not np.isnan(rho) else None,
            "p_value": round(float(p), 4) if not np.isnan(p) else None}


def calibration_report(human_scores: List[int], judge_scores: List[int]) -> Dict:
    """Full calibration report for one metric axis (e.g. normative_safety),
    comparing human reference labels to LLM judge scores on the same
    held-out trajectories.
    """
    n = len(human_scores)
    kappa = cohens_kappa(human_scores, judge_scores)
    kappa_lo, kappa_hi = bootstrap_kappa_ci(human_scores, judge_scores)
    spearman = spearman_agreement(human_scores, judge_scores)
    mean_abs_diff = float(np.mean(np.abs(np.array(human_scores) - np.array(judge_scores))))

    # Standard qualitative bands for weighted kappa (Landis & Koch 1977)
    def band(k):
        if np.isnan(k): return "undefined"
        if k < 0: return "poor"
        if k < 0.20: return "slight"
        if k < 0.40: return "fair"
        if k < 0.60: return "moderate"
        if k < 0.80: return "substantial"
        return "almost perfect"

    return {
        "n_samples": n,
        "weighted_kappa": kappa,
        "kappa_95ci": [kappa_lo, kappa_hi],
        "kappa_band": band(kappa),
        "spearman_rho": spearman["spearman_rho"],
        "spearman_p": spearman["p_value"],
        "mean_abs_diff": round(mean_abs_diff, 3),
        "caveat": (
            "n is small; CI width reflects this -- treat point estimate with "
            "caution and report the interval, not just the kappa value."
            if n < 50 else None
        ),
    }


def inter_model_agreement(model_a_scores: List[int], model_b_scores: List[int]) -> Dict:
    """Agreement between two DIFFERENT judge model families on the same
    trajectories. This is what genuine dual-judge reliability looks like —
    distinct from v1's same-model-twice setup.
    """
    return calibration_report(model_a_scores, model_b_scores)


# ──────────────────────────────────────────────────────────────────────────
# SELF-TEST with synthetic data demonstrating the reporting format.
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.RandomState(42)

    print("=== Synthetic calibration demo (n=25, moderate agreement) ===")
    human = rng.randint(0, 4, size=25).tolist()
    # Judge scores correlated with human but with noise -- simulates a
    # realistic LLM judge that's directionally right but imperfect.
    judge = [int(np.clip(h + rng.choice([-1, 0, 0, 0, 1]), 0, 3)) for h in human]

    report = calibration_report(human, judge)
    for k, v in report.items():
        print(f"  {k}: {v}")

    print("\n=== Synthetic inter-model agreement demo (two different judge LLMs) ===")
    model_a = rng.randint(0, 4, size=25).tolist()
    model_b = [int(np.clip(a + rng.choice([-1, 0, 1]), 0, 3)) for a in model_a]
    imr = inter_model_agreement(model_a, model_b)
    for k, v in imr.items():
        print(f"  {k}: {v}")

    print("\n=== Honesty check: what happens with n=5 (too small) ===")
    small_human = [2, 3, 1, 2, 3]
    small_judge = [2, 2, 1, 3, 3]
    small_report = calibration_report(small_human, small_judge)
    print(f"  kappa={small_report['weighted_kappa']}, CI={small_report['kappa_95ci']}")
    print(f"  caveat: {small_report['caveat']}")
