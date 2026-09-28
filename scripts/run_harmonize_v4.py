"""Stage 6 execution: harmonisation diagnostics + the prevalence stress
test for C1 and C2 on every outer site.

Writes data/pd-runs/v4/harmonisation-report.json.

Two things to read carefully alongside the output:

1. Aggregation rule. docs/PD_PROTOCOL_V4.md defines the pass criterion per
   outer site but does not state how to aggregate across sites. Reviewed
   decision (28 Sep 2026): MAJORITY of outer sites must pass, i.e. >=3 of 4.
   Note 2-of-4 is a tie, not a majority, so it fails. Per-site results are
   reported either way so the rule is auditable.

2. The guardrail is label-aware on the test site. The frozen protocol
   prescribes computing AUROC on the held-out site, which necessarily uses
   that site's labels before the outer evaluation -- a real qualification of
   non-negotiable #4. Reviewed decision: run BOTH the frozen label-aware
   criterion (primary) and a label-free variant (sensitivity check, verdict
   from random synthetic labellings only) and report whether they agree.

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
    stress_lf = {"C2": {}, "C1": {}}
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
        stress_lf["C2"][site] = hv.prevalence_stress_label_free(
            X[test_mask], y[test_mask], c2_scorer, n_resamples=hv.STRESS_RESAMPLES, seed=SEED
        )
        print(
            f"  C2 {site}: label-aware spread={stress['C2'][site]['spread']:.4f} "
            f"passed={stress['C2'][site]['passed']} | "
            f"label-free spread={stress_lf['C2'][site]['spread_mean']:.4f} "
            f"passed={stress_lf['C2'][site]['passed']}"
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
        stress_lf["C1"][site] = hv.prevalence_stress_label_free(
            index_matrix, cov_y[cov_test], c1_scorer, n_resamples=25, seed=SEED
        )
        print(
            f"  C1 {site}: label-aware spread={stress['C1'][site]['spread']:.4f} "
            f"passed={stress['C1'][site]['passed']} | "
            f"label-free spread={stress_lf['C1'][site]['spread_mean']:.4f} "
            f"passed={stress_lf['C1'][site]['passed']}"
        )

    report["descriptive"] = descriptive
    report["prevalence_stress_label_aware"] = stress
    report["prevalence_stress_label_free"] = stress_lf

    def verdict(per_site, key):
        """Majority rule (reviewed 28 Sep 2026): >= 3 of 4 outer sites must
        pass. 2 of 4 is a tie, not a majority, so it fails."""
        if not per_site:
            return None
        n_pass = sum(1 for v in per_site.values() if v["passed"])
        n_total = len(per_site)
        return {
            "sites_passed": n_pass,
            "sites_total": n_total,
            "majority_required": n_total // 2 + 1,
            "passed": n_pass >= n_total // 2 + 1,
            "max_spread": max(v[key] for v in per_site.values()),
            "per_site_spread": {s: v[key] for s, v in per_site.items()},
        }

    summary = {}
    for method in ("C1", "C2"):
        label_aware = verdict(stress[method], "spread")
        label_free = verdict(stress_lf[method], "spread_mean")
        summary[method] = {
            "label_aware": label_aware,
            "label_free": label_free,
            "criteria_agree": (
                None
                if (label_aware is None or label_free is None)
                else label_aware["passed"] == label_free["passed"]
            ),
            "tolerance": hv.STRESS_TOLERANCE,
        }
    summary["aggregation_rule"] = "majority of outer sites (>=3 of 4); 2 of 4 is a tie, not a majority"
    summary["pre_declared_transductive_methods"] = [
        m for m in ("C1", "C2") if summary[m]["label_aware"] and summary[m]["label_aware"]["passed"]
    ]
    report["summary"] = summary
    (RUN / "harmonisation-report.json").write_text(json.dumps(report, indent=2))
    print("\nSUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
