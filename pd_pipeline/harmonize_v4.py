"""v4 harmonisation.

C1 Riemannian re-centering, C2 robust per-site standardisation, C3
normative z-scores, and the prevalence stress test guardrail.

Regimes (docs/PD_PROTOCOL_V4.md S6):
- Inductive (I): no test-site data used; training-fold statistics only.
- Transductive (T): the test site's *unlabelled* people may set that
  site's statistics. C1/C2 are the transductive methods; both must pass
  the prevalence stress test before they may enter the pre-declared
  pipeline.
- C3 is allowed in both regimes, because its norms come from healthy
  auxiliary adults plus training-fold controls only -- never from
  test-fold people and never from anyone's diagnosis label.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .cohort_v3 import ARCHIVE
from .features_v4 import normative_feature_names

RUN = Path("data/pd-runs/v4")
MIN_NORMATIVE_AGE = 40
STRESS_PREVALENCES = (0.3, 0.5, 0.7)
STRESS_RESAMPLES = 200
STRESS_TOLERANCE = 0.03
SEED = 20261001

# Normative sources (protocol S6). Column names and categorical codings are
# taken from each dataset's own BIDS participants.json sidecar, never guessed:
#   ds004796 participants.json: sex Levels {"0": "male", "1": "female"}
#   ds004504 participants.json: Group Levels {A: Alzheimer, F: Frontotemporal,
#                                             C: Healthy}  -> controls are "C"
# ds004148 is listed in the protocol but contributes nothing: its cohort is
# aged 18-28, so no participant meets the >=40y healthy-adult rule. Recorded
# here rather than silently dropped.
NORMATIVE_SOURCES = {
    "ds005385": {"age": "age", "sex": "sex", "sex_map": {"M": 0.0, "F": 1.0}, "control_filter": None},
    "ds004796": {"age": "age", "sex": "sex", "sex_map": {"0": 0.0, "1": 1.0}, "control_filter": None},
    "ds004504": {
        "age": "Age",
        "sex": "Gender",
        "sex_map": {"M": 0.0, "F": 1.0},
        "control_filter": ("Group", "C"),
    },
}
NORMATIVE_SOURCE_NOTE = (
    "ds004148 is a protocol-listed normative source but contributes 0 people: "
    "its cohort is aged 18-28 and none reach the >=40y healthy-adult threshold."
)


def _tsv(path):
    return list(csv.DictReader(Path(path).read_text(encoding="utf-8-sig").splitlines(), delimiter="\t"))


def normative_source_people(registry, archive=ARCHIVE, min_age=MIN_NORMATIVE_AGE):
    """Healthy auxiliary adults >=40y from the protocol's normative sources,
    with age/sex. Handles the ds004504/ds006036 canonical merge: those 88
    people carry records from both datasets under one person key, so
    membership is tested against the person's whole dataset set rather than
    a single overwritten label."""
    person_datasets = {}
    for r in registry["records"]:
        if r["role"] == "auxiliary":
            person_datasets.setdefault(r["person"], set()).add(r["dataset"])
    out = {}
    for dataset, spec in NORMATIVE_SOURCES.items():
        rows = {
            r["participant_id"]: r
            for r in _tsv(archive / "provenance" / "openneuro" / dataset / "participants.tsv")
        }
        for person, datasets in person_datasets.items():
            if dataset not in datasets or person in out:
                continue
            row = rows.get(person.split(":", 1)[1])
            if row is None:
                continue
            if spec["control_filter"]:
                column, wanted = spec["control_filter"]
                if row.get(column) != wanted:
                    continue
            try:
                age = float(row[spec["age"]])
            except (TypeError, ValueError, KeyError):
                continue
            if age < min_age:
                continue
            sex = spec["sex_map"].get(str(row.get(spec["sex"], "")).strip().upper())
            if sex is None:
                sex = spec["sex_map"].get(str(row.get(spec["sex"], "")).strip())
            if sex is None:
                continue
            out[person] = {"age": age, "sex": sex, "source": dataset}
    return out


# ---------------------------------------------------------------------------
# C1: Riemannian re-centering
# ---------------------------------------------------------------------------


def _matrix_power(m, p):
    val, vec = np.linalg.eigh(m)
    return (vec * np.maximum(val, 1e-10) ** p) @ vec.T


def recenter_riemann(covs, site_ids):
    """Whiten every person's covariance by its own site's Riemannian mean,
    so each site's mean maps to the identity (Mellot 2023). Label-free:
    only site membership is used, never diagnosis."""
    from pyriemann.utils.mean import mean_riemann

    covs = np.asarray(covs, dtype="float64")
    site_ids = np.asarray(site_ids)
    out = np.empty_like(covs)
    for site in np.unique(site_ids):
        idx = site_ids == site
        whitener = _matrix_power(mean_riemann(covs[idx]), -0.5)
        out[idx] = whitener @ covs[idx] @ whitener
    return out


def rescale_dispersion(covs, site_ids):
    """Optional C1 second step: rescale each site's dispersion about the
    identity to unit mean squared Riemannian distance."""
    covs = np.asarray(covs, dtype="float64")
    site_ids = np.asarray(site_ids)
    out = np.empty_like(covs)
    for site in np.unique(site_ids):
        idx = site_ids == site
        logs = np.stack([_logm(c) for c in covs[idx]])
        dispersion = np.sqrt(np.mean(np.sum(logs**2, axis=(1, 2))))
        scale = 1.0 / max(dispersion, 1e-10)
        out[idx] = np.stack([_expm(log * scale) for log in logs])
    return out


def _logm(m):
    val, vec = np.linalg.eigh(m)
    return (vec * np.log(np.maximum(val, 1e-10))) @ vec.T


def _expm(m):
    val, vec = np.linalg.eigh(m)
    return (vec * np.exp(val)) @ vec.T


# ---------------------------------------------------------------------------
# C2: robust per-site standardisation
# ---------------------------------------------------------------------------


def site_statistics(X, site_ids):
    """Per-site median and IQR, the only quantities C2 transfers. Computed
    from whichever rows are passed in -- the prevalence stress test uses
    that to vary the class balance behind these statistics."""
    X = np.asarray(X, dtype="float64")
    site_ids = np.asarray(site_ids)
    stats = {}
    for site in np.unique(site_ids):
        rows = X[site_ids == site]
        q75, q25 = np.nanpercentile(rows, [75, 25], axis=0)
        stats[site] = {"median": np.nanmedian(rows, axis=0), "iqr": np.maximum(q75 - q25, 1e-9)}
    return stats


def robust_site_z(X, site_ids, stats=None):
    """C2: (x - site median) / site IQR. If `stats` is given, those site
    statistics are applied instead of being recomputed from X (used by the
    prevalence stress test and to apply training-site statistics)."""
    X = np.asarray(X, dtype="float64")
    site_ids = np.asarray(site_ids)
    stats = stats or site_statistics(X, site_ids)
    out = np.array(X, dtype="float64", copy=True)
    for site in np.unique(site_ids):
        idx = site_ids == site
        if site not in stats:
            continue
        out[idx] = (X[idx] - stats[site]["median"]) / stats[site]["iqr"]
    return out


# ---------------------------------------------------------------------------
# C3: normative models
# ---------------------------------------------------------------------------


def _design(age, sex):
    age = np.asarray(age, dtype="float64").reshape(-1, 1)
    sex = np.asarray(sex, dtype="float64").reshape(-1, 1)
    return age, sex


def normative_fit(X_healthy, age, sex, quantiles=(0.05, 0.5, 0.95)):
    """Spline quantile regression of each feature on age (natural cubic
    spline, 4 df, linear extrapolation beyond the boundary knots) plus sex,
    fit on healthy adults only. One QuantileRegressor per feature per
    quantile."""
    X_healthy = np.asarray(X_healthy, dtype="float64")
    age_col, sex_col = _design(age, sex)
    spline = SplineTransformer(n_knots=2, degree=3, extrapolation="linear").fit(age_col)
    design = np.hstack([spline.transform(age_col), sex_col])
    models = []
    for j in range(X_healthy.shape[1]):
        column = X_healthy[:, j]
        usable = np.isfinite(column)
        per_quantile = {}
        for q in quantiles:
            if usable.sum() < 10:
                per_quantile[q] = None
                continue
            model = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
            model.fit(design[usable], column[usable])
            per_quantile[q] = model
        models.append(per_quantile)
    return {"spline": spline, "models": models, "quantiles": quantiles, "n_healthy": len(X_healthy)}


def normative_z(model, X, age, sex):
    """z = (x - q50) / ((q95 - q5) / 3.29), the protocol's normal-equivalent
    scaling of a quantile spread."""
    X = np.asarray(X, dtype="float64")
    age_col, sex_col = _design(age, sex)
    design = np.hstack([model["spline"].transform(age_col), sex_col])
    out = np.full_like(X, np.nan, dtype="float64")
    for j, per_quantile in enumerate(model["models"]):
        if any(per_quantile.get(q) is None for q in (0.05, 0.5, 0.95)):
            continue
        q05 = per_quantile[0.05].predict(design)
        q50 = per_quantile[0.5].predict(design)
        q95 = per_quantile[0.95].predict(design)
        spread = np.maximum((q95 - q05) / 3.29, 1e-9)
        out[:, j] = (X[:, j] - q50) / spread
    return out


def normative_feature_list():
    return normative_feature_names()


# ---------------------------------------------------------------------------
# Prevalence stress test (C-guardrail)
# ---------------------------------------------------------------------------


def prevalence_stress(
    X,
    y,
    scorer,
    prevalences=STRESS_PREVALENCES,
    n_resamples=STRESS_RESAMPLES,
    tolerance=STRESS_TOLERANCE,
    seed=SEED,
):
    """Does a transductive method's site statistics depend on how many of
    the site's people happen to be PD?

    For each target PD share, draw stratified subsets of this site (without
    replacement, never exceeding the available class counts), estimate the
    site statistics from that subset only, re-score *all* of the site's
    people with them, and take the AUROC. Pass if the spread of mean AUROC
    across prevalences is <= tolerance.

    `scorer(X_full, stats_rows)` must return one score per row of X_full,
    having derived any site statistics from `stats_rows` alone.
    """
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pd_idx = np.flatnonzero(y == 1)
    control_idx = np.flatnonzero(y == 0)
    results = {}
    for prevalence in prevalences:
        # Largest stratified subset with this PD share that both classes support.
        n_total = min(
            int(len(pd_idx) / prevalence) if prevalence > 0 else 0,
            int(len(control_idx) / (1 - prevalence)) if prevalence < 1 else 0,
        )
        n_pd = int(round(n_total * prevalence))
        n_control = n_total - n_pd
        if n_pd < 2 or n_control < 2:
            results[prevalence] = {"mean_auroc": float("nan"), "n_resamples": 0, "subset_size": n_total}
            continue
        aurocs = []
        for _ in range(n_resamples):
            subset = np.concatenate(
                [
                    rng.choice(pd_idx, n_pd, replace=False),
                    rng.choice(control_idx, n_control, replace=False),
                ]
            )
            scores = scorer(X, X[subset])
            aurocs.append(roc_auc_score(y, scores))
        results[prevalence] = {
            "mean_auroc": float(np.mean(aurocs)),
            "sd_auroc": float(np.std(aurocs)),
            "n_resamples": n_resamples,
            "subset_size": int(n_total),
            "pd_share_actual": float(n_pd / n_total),
        }
    means = [r["mean_auroc"] for r in results.values() if np.isfinite(r["mean_auroc"])]
    spread = float(max(means) - min(means)) if len(means) > 1 else float("nan")
    return {
        "by_prevalence": {str(k): v for k, v in results.items()},
        "spread": spread,
        "tolerance": tolerance,
        "passed": bool(np.isfinite(spread) and spread <= tolerance),
    }


# ---------------------------------------------------------------------------
# Descriptive diagnostics (aggregate only, no participant-level output)
# ---------------------------------------------------------------------------


def site_identity_probe(X, site_ids, seed=SEED):
    """Multi-class logistic regression predicting *site* from features,
    scored by macro one-vs-rest AUROC on grouped-CV held-out people. Lower
    is better for transfer: it measures how much site identity a
    representation still carries."""
    X = np.asarray(X, dtype="float64")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sites = np.asarray(site_ids)
    codes, labels = np.unique(sites, return_inverse=True)
    if len(codes) < 2:
        return float("nan")
    scores = []
    splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
    for train, test in splitter.split(X, labels):
        if len(np.unique(labels[train])) < len(codes):
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
            X[train], labels[train]
        )
        probabilities = model.predict_proba(X[test])
        present = np.unique(labels[test])
        if len(present) < 2:
            continue
        scores.append(
            roc_auc_score(labels[test], probabilities, multi_class="ovr", average="macro", labels=model.classes_)
        )
    return float(np.mean(scores)) if scores else float("nan")


def per_site_medians(X, site_ids):
    """Aggregate per-site feature medians, for before/after harmonisation
    descriptive tables. No participant-level values."""
    X = np.asarray(X, dtype="float64")
    site_ids = np.asarray(site_ids)
    return {
        str(site): np.nanmedian(X[site_ids == site], axis=0).tolist()
        for site in np.unique(site_ids)
    }


def harmonisation_report(X, site_ids, run=RUN, write=True):
    """Descriptive before/after summary for C2 (the feature-space method):
    per-site medians and the site-identity probe, each side by side."""
    harmonised = robust_site_z(X, site_ids)
    report = {
        "site_identity_probe_macro_auroc": {
            "before": site_identity_probe(X, site_ids),
            "after_c2": site_identity_probe(harmonised, site_ids),
            "note": "Lower is better for cross-site transfer.",
        },
        "per_site_medians": {
            "before": per_site_medians(X, site_ids),
            "after_c2": per_site_medians(harmonised, site_ids),
        },
        "normative_sources": {
            "used": sorted(NORMATIVE_SOURCES),
            "note": NORMATIVE_SOURCE_NOTE,
        },
    }
    if write:
        (run / "harmonisation-report.json").write_text(json.dumps(report, indent=2))
    return report
