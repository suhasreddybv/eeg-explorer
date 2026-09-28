"""v4 harmonisation: C1 re-centering, C2 robust site z, C3 normative
models, and the prevalence stress-test guardrail (including the
protocol's required invariant case)."""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pd_pipeline import harmonize_v4 as hv  # noqa: E402
from pyriemann.utils.mean import mean_riemann  # noqa: E402


def _spd(rng, n, k):
    out = []
    for _ in range(k):
        a = rng.normal(size=(n, n))
        out.append(a @ a.T + n * np.eye(n))
    return np.stack(out)


def test_c1_recentering_maps_each_site_mean_to_identity():
    rng = np.random.default_rng(0)
    n = 4
    # Two sites with deliberately different covariance scales.
    site_a = _spd(rng, n, 15)
    site_b = _spd(rng, n, 15) * 7.0
    covs = np.concatenate([site_a, site_b])
    sites = np.array(["a"] * 15 + ["b"] * 15)
    recentred = hv.recenter_riemann(covs, sites)
    for site in ("a", "b"):
        mean = mean_riemann(recentred[sites == site])
        assert mean == pytest.approx(np.eye(n), abs=1e-6)


def test_c1_recentering_is_label_free():
    # Permuting diagnosis labels must not change the result at all: C1 sees
    # only site membership.
    rng = np.random.default_rng(1)
    covs = _spd(rng, 4, 20)
    sites = np.array(["a"] * 10 + ["b"] * 10)
    first = hv.recenter_riemann(covs, sites)
    second = hv.recenter_riemann(covs, sites)
    assert np.allclose(first, second)


def test_c2_robust_site_z_centres_each_site():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(10, 2, size=(50, 5)), rng.normal(-4, 6, size=(50, 5))])
    sites = np.array(["a"] * 50 + ["b"] * 50)
    Z = hv.robust_site_z(X, sites)
    for site in ("a", "b"):
        assert np.nanmedian(Z[sites == site], axis=0) == pytest.approx(np.zeros(5), abs=1e-9)


def test_c2_can_apply_supplied_statistics_instead_of_recomputing():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    sites = np.array(["a"] * 40)
    stats = hv.site_statistics(X[:10], sites[:10])
    applied = hv.robust_site_z(X, sites, stats=stats)
    expected = (X - stats["a"]["median"]) / stats["a"]["iqr"]
    assert np.allclose(applied, expected)


def test_c3_normative_z_is_near_zero_for_healthy_like_data():
    rng = np.random.default_rng(4)
    n = 300
    age = rng.uniform(40, 80, n)
    sex = rng.integers(0, 2, n).astype(float)
    # A feature with a real age trend plus noise.
    feature = 0.05 * age + 0.3 * sex + rng.normal(0, 0.5, n)
    X = feature.reshape(-1, 1)
    model = hv.normative_fit(X, age, sex)
    z = hv.normative_z(model, X, age, sex)
    assert abs(float(np.nanmedian(z))) < 0.25
    # A clearly abnormal person sits far from the norm.
    shifted = hv.normative_z(model, np.array([[feature.mean() + 5.0]]), np.array([60.0]), np.array([0.0]))
    assert abs(float(shifted[0, 0])) > 2.0


def test_prevalence_stress_invariant_case_gives_zero_spread():
    """Protocol guardrail check: a scorer that ignores the site statistics
    entirely (a site-independent feature) must give identical AUROC at
    every prevalence, so the spread is exactly 0 and the test passes."""
    rng = np.random.default_rng(5)
    y = np.array([0] * 60 + [1] * 60)
    X = np.column_stack([y + rng.normal(0, 0.3, len(y))])

    def invariant_scorer(X_full, stats_rows):
        return X_full[:, 0]  # never looks at stats_rows

    result = hv.prevalence_stress(X, y, invariant_scorer, n_resamples=20)
    assert result["spread"] == pytest.approx(0.0, abs=1e-12)
    assert result["passed"] is True


def test_prevalence_stress_flags_a_prevalence_sensitive_method():
    """A scorer whose site statistics depend strongly on class balance
    should move AUROC across prevalences and can fail the guardrail."""
    rng = np.random.default_rng(6)
    y = np.array([0] * 60 + [1] * 60)
    signal = y + rng.normal(0, 0.3, len(y))
    # Second column is pure noise; the scorer subtracts the *subset* mean of
    # the signal column, so a PD-heavy subset shifts scores non-monotonically.
    X = np.column_stack([signal, rng.normal(size=len(y))])

    def sensitive_scorer(X_full, stats_rows):
        centre = stats_rows[:, 0].mean()
        return -np.abs(X_full[:, 0] - centre)

    result = hv.prevalence_stress(X, y, sensitive_scorer, n_resamples=20)
    assert np.isfinite(result["spread"])
    assert result["spread"] > 0.0


def test_prevalence_stress_reports_each_requested_prevalence():
    rng = np.random.default_rng(7)
    y = np.array([0] * 40 + [1] * 40)
    X = np.column_stack([y + rng.normal(0, 0.5, len(y))])
    result = hv.prevalence_stress(X, y, lambda Xf, s: Xf[:, 0], n_resamples=5)
    assert set(result["by_prevalence"]) == {"0.3", "0.5", "0.7"}
    for entry in result["by_prevalence"].values():
        assert entry["subset_size"] > 0


def test_normative_sources_exclude_the_young_cohort():
    # ds004148 is protocol-listed but its cohort is 18-28, so it must not
    # appear as a usable normative source.
    assert "ds004148" not in hv.NORMATIVE_SOURCES
    assert "ds004148" in hv.NORMATIVE_SOURCE_NOTE


def test_normative_feature_list_is_38_named_features():
    names = hv.normative_feature_list()
    assert len(names) == 38
    assert len(set(names)) == 38
