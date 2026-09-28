"""Stage 6 execution: harmonisation diagnostics + the prevalence stress
test for C1 and C2 on every outer site.

Writes data/pd-runs/v4/harmonisation-report.json.

Two things to read carefully alongside the output:

1. Aggregation rule. docs/PD_PROTOCOL_V4.md defines the pass criterion per
   outer site but does not state how to aggregate across sites. This runner
   adopts the conservative reading -- a method must pass at EVERY outer site
   to enter the pre-declared pipeline -- because the pipeline is applied to
   all sites. Per-site results are reported either way.

2. The guardrail is label-aware on the test site. The frozen protocol
   prescribes computing AUROC on the held-out site, which necessarily uses
   that site's labels before the outer evaluation. The criterion is
   stability-only (spread across prevalences), so it cannot be used to pick
   whichever method scores highest -- but it is a real qualification of
   non-negotiable #4 and is flagged in docs/PD_V4_LOG.md rather than buried.

The scorer here is a simple pre-declared logistic regression, not the Stage 7
stacker: the guardrail decides pipeline membership *before* the models exist,
so it must not depend on them.
"""

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from pd_pipeline import features_v4 as fv  # noqa: E402
from pd_pipeline import harmonize_v4 as hv  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RUN = Path("data/pd-runs/v4")
FEAT = RUN / "features"
SEED = 20261001


def load_supervised():
    registry = json.loads((RUN / "registry.json").read_text())
    people = []
    for r in registry["records"]:
        if r["role"] != "supervised":
            continue
        f = FEAT / (r["person"].replace(":", "__") + ".npz")
        if not f.exists():
            continue
        with np.load(f) as d:
            entry = {
                "person": r["person"],
                "site": r["site"],
                "y": r["label"],
                "normative": fv.normative_features(d),
                "cov": d["b3_cov_mean"] if "b3_cov_mean" in d else None,
            }
        people.append(entry)
    return people, registry


def tangent_matrix(covs):
    identity = np.eye(covs.shape[-1])
    return np.stack([fv.tangent_vector(c.astype("float64"), identity) for c in covs])


def main():
    people, registry = load_supervised()
    X = np.array([p["normative"] for p in people])
    y = np.array([p["y"] for p in people])
    sites = np.array([p["site"] for p in people])
    has_cov = np.array([p["cov"] is not None for p in people])
    covs = np.stack([p["cov"] for p in people if p["cov"] is not None]).astype("float64")
    print(f"supervised people: {len(people)}, with covariance: {has_cov.sum()}")

    report = {"n_people": len(people), "by_site": {}}

    # --- descriptive: site-identity probe and per-site medians, before/after
    descriptive = hv.harmonisation_report(X, sites, run=RUN, write=False)
    print("site-identity probe:", descriptive["site_identity_probe_macro_auroc"])

    # --- prevalence stress test, per outer site, for C2 and C1
    stress = {"C2": {}, "C1": {}}
    for site in sorted(set(sites)):
        test_mask = sites == site
        train_mask = ~test_mask
        if len(set(y[test_mask])) < 2:
            continue

        # C2 on the 38-feature normative vector.
        train_stats = hv.site_statistics(X[train_mask], sites[train_mask])
        X_train_h = hv.robust_site_z(X[train_mask], sites[train_mask], stats=train_stats)
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")
        ).fit(np.nan_to_num(X_train_h), y[train_mask])

        def c2_scorer(X_full, stats_rows, _model=model, _site=site):
            site_stats = hv.site_statistics(stats_rows, np.array([_site] * len(stats_rows)))
            harmonised = hv.robust_site_z(X_full, np.array([_site] * len(X_full)), stats=site_stats)
            return _model.predict_proba(np.nan_to_num(harmonised))[:, 1]

        stress["C2"][site] = hv.prevalence_stress(
            X[test_mask], y[test_mask], c2_scorer, n_resamples=hv.STRESS_RESAMPLES, seed=SEED
        )
        print(
            f"  C2 {site}: spread={stress['C2'][site]['spread']:.4f} "
            f"passed={stress['C2'][site]['passed']}"
        )

        # C1 on the per-person Riemannian mean covariances.
        cov_sites = sites[has_cov]
        cov_y = y[has_cov]
        cov_test = cov_sites == site
        cov_train = ~cov_test
        if len(set(cov_y[cov_test])) < 2:
            continue
        train_recentred = hv.recenter_riemann(covs[cov_train], cov_sites[cov_train])
        T_train = tangent_matrix(train_recentred)
        cov_model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")
        ).fit(T_train, cov_y[cov_train])

        test_covs = covs[cov_test]

        def c1_scorer(X_full_idx, stats_idx, _model=cov_model, _covs=test_covs):
            from pyriemann.utils.mean import mean_riemann

            subset = _covs[stats_idx[:, 0].astype(int)]
            whitener = hv._matrix_power(mean_riemann(subset), -0.5)
            recentred = whitener @ _covs @ whitener
            return _model.predict_proba(tangent_matrix(recentred))[:, 1]

        index_matrix = np.arange(len(test_covs)).reshape(-1, 1).astype(float)
        stress["C1"][site] = hv.prevalence_stress(
            index_matrix, cov_y[cov_test], c1_scorer, n_resamples=25, seed=SEED
        )
        print(
            f"  C1 {site}: spread={stress['C1'][site]['spread']:.4f} "
            f"passed={stress['C1'][site]['passed']}"
        )

    report["descriptive"] = descriptive
    report["prevalence_stress"] = stress
    report["summary"] = {
        method: {
            "all_sites_passed": all(v["passed"] for v in per_site.values()) if per_site else None,
            "max_spread": max((v["spread"] for v in per_site.values()), default=None),
            "tolerance": hv.STRESS_TOLERANCE,
        }
        for method, per_site in stress.items()
    }
    (RUN / "harmonisation-report.json").write_text(json.dumps(report, indent=2))
    print("\nSUMMARY:", json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
