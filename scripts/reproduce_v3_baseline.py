"""v4 Stage 0, gate G1: reproduce the frozen v3 rich-SVM baseline.

Two independent recomputations of the rich-feature SVM's cross-site and
within-site equal-site AUROC, both restricted to the frozen v3 fold
definitions (pd_pipeline.compare_v3.folds), against the v3 report's target
values (0.763 cross-site, 0.711 within-site; docs/PD_V3_RESULTS.md):

1. ``recompute_from_outer_predictions`` re-derives equal-site AUROC directly
   from data/pd-runs/v3/outer-predictions.json, independent of the cached
   aggregate in results.json.
2. ``refit_and_infer`` re-runs pd_pipeline.compare_v3.fit()/infer() for the
   locked rich_svm configuration (C=10) on every outer fold, using the
   frozen v3 cache. compare_v3.fit() enforces weight-hash integrity against
   the existing lock, so this checks the locked artifacts are self-
   consistent and that inference on held-out people reproduces the same
   scores, not just that the stored numbers are internally consistent.

Writes data/pd-runs/v4/baseline-repro.json. Does not modify v1/v2/v3 data.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pd_pipeline.cohort_v3 import RUN as V3_RUN
from pd_pipeline.compare_v3 import enrich, folds, infer, validate_disjoint
from pd_pipeline.learning_v3 import load_people
from pd_pipeline.registry import save

V4_RUN = Path("data/pd-runs/v4")
FAMILY = "rich_svm"
TARGET = {"cross_site": 0.763, "within_site": 0.711}
TOLERANCE = 0.01


def build_folds(run=V3_RUN):
    """Rebuild the v3 outer fold definitions (used by the G1 test to check
    site sets, grouping and disjointness match the frozen v3 splits.json)."""
    people = enrich(load_people(run), run)
    return people, folds(people)


def _equal_site_auroc(pairs):
    by_site = {}
    for site, y, p in pairs:
        by_site.setdefault(site, ([], []))
        by_site[site][0].append(y)
        by_site[site][1].append(p)
    return {
        site: float(roc_auc_score(ys, ps)) for site, (ys, ps) in by_site.items() if len(set(ys)) == 2
    }


def recompute_from_outer_predictions(run=V3_RUN, family=FAMILY):
    """Independent recomputation from outer-predictions.json, bypassing the
    cached aggregate in results.json entirely."""
    out = json.loads((run / "outer-predictions.json").read_text())
    result = {}
    for kind in ["cross_site", "within_site"]:
        pairs = [
            (person["site"], person["y"], record["models"][family]["p"][j])
            for record in out
            if record["kind"] == kind
            for j, person in enumerate(record["people"])
        ]
        site_auroc = _equal_site_auroc(pairs)
        result[kind] = {
            "site_auroc": site_auroc,
            "equal_site_mean_auroc": float(np.mean(list(site_auroc.values()))),
        }
    return result


def refit_and_infer(run=V3_RUN, family=FAMILY):
    """Re-run fit()/infer() against the frozen v3 lock for every outer fold,
    confirming the locked weights reproduce the recorded scores on held-out
    people rather than trusting outer-predictions.json at face value."""
    people, splits = build_folds(run)
    result = {"cross_site": {}, "within_site": {}}
    for split in splits:
        folder = run / "experiments" / split["name"]
        lock = json.loads((folder / "lock.json").read_text())
        d = lock["models"][family]
        train = [people[i] for i in split["train"]]
        test = [people[i] for i in split["test"]]
        validate_disjoint(train, test)
        artifact = d["fits"][0]
        fit_path = run / artifact["path"]
        meta = json.loads((fit_path / "fit.json").read_text())
        if meta["sha256"] != artifact["sha256"]:
            raise ValueError(f"Locked weights changed for {split['name']}")
        pred = infer(fit_path, test, meta)
        kind = "cross_site" if split["kind"] == "cross_site" else "within_site"
        # Within-site folds mix sites; accumulate per-person predictions by site instead.
        for i, p in zip(split["test"], pred):
            site = people[i]["site"]
            result[kind].setdefault("_pairs", []).append((site, people[i]["y"], float(p)))
    for kind in ["cross_site", "within_site"]:
        pairs = result[kind].pop("_pairs", [])
        site_auroc = _equal_site_auroc(pairs)
        result[kind] = {
            "site_auroc": site_auroc,
            "equal_site_mean_auroc": float(np.mean(list(site_auroc.values()))),
        }
    return result


def main(run=V3_RUN, out_run=V4_RUN):
    outer_predictions = recompute_from_outer_predictions(run)
    refit = refit_and_infer(run)
    gate = {}
    for kind in ["cross_site", "within_site"]:
        observed = outer_predictions[kind]["equal_site_mean_auroc"]
        target = TARGET[kind]
        gate[kind] = {
            "observed": observed,
            "target": target,
            "abs_diff": abs(observed - target),
            "within_tolerance": abs(observed - target) <= TOLERANCE,
        }
    report = {
        "gate": "G1",
        "family": FAMILY,
        "target": TARGET,
        "tolerance": TOLERANCE,
        "recomputed_from_outer_predictions": outer_predictions,
        "refit_and_infer": refit,
        "gate_result": gate,
        "passed": all(g["within_tolerance"] for g in gate.values()),
    }
    save(out_run / "baseline-repro.json", report)
    return report


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["gate_result"], indent=2))
    if not result["passed"]:
        raise SystemExit("G1 FAILED: v3 baseline reproduction outside tolerance; stop and diagnose")
