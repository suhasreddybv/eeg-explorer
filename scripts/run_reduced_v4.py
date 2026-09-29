"""Stage 9 (T2): score the pre-declared reduced-channel montages, run the
distillation comparison, and evaluate gate G6.

Everything here reuses compare_v4's nested machinery unchanged, with the
17-channel feature blocks swapped for montage ones (reduced_v4). The
montages are rebuilt from preref128 and re-referenced to their own
channels only, so these are honest wearable-montage numbers rather than
17-channel features wearing a 4-channel label.

G6: the best *pre-declared* montage must land within 0.05 composite of T1.
The greedy channel-budget search is descriptive and does not feed G6.
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from pd_pipeline import compare_v4 as cv  # noqa: E402
from pd_pipeline import reduced_v4 as rv  # noqa: E402
from pd_pipeline.registry import save  # noqa: E402

RUN = Path("data/pd-runs/v4")
SEED = 20261001


def equal_site_macro(fold_results, label):
    """Equal-site macro AUROC for one scored variant, split by fold kind."""
    out = {}
    for kind in ("cross_site", "within_site"):
        by_site = defaultdict(lambda: ([], []))
        for fold in fold_results:
            if fold["kind"] != kind:
                continue
            for j, person in enumerate(fold["people"]):
                by_site[person["site"]][0].append(person["y"])
                by_site[person["site"]][1].append(fold["scores"][label][j])
        per_site = {
            site: float(roc_auc_score(ys, ss))
            for site, (ys, ss) in by_site.items()
            if len(set(ys)) == 2
        }
        out[kind] = {
            "per_site": per_site,
            "macro": float(np.mean(list(per_site.values()))) if per_site else float("nan"),
        }
    out["composite"] = 0.5 * out["cross_site"]["macro"] + 0.5 * out["within_site"]["macro"]
    return out


def run_montage(montage, base_people, normative_pool, seed=SEED):
    people = rv.load_montage_people(base_people, montage)
    if len(people) != len(base_people):
        print(f"  {montage}: {len(people)}/{len(base_people)} people have montage features")
    splits = cv.folds(people, seed)
    specs = rv.montage_families()
    results = [cv.run_fold(people, split, normative_pool, seed, family_specs=specs) for split in splits]
    scored = {label: equal_site_macro(results, label) for label in results[0]["scores"]}
    return {"montage": montage, "n_people": len(people), "scored": scored, "folds": results}


def distillation(montage, base_people, t1_folds, normative_pool, seed=SEED):
    """Train a montage student on soft targets blended from the T1 stacker's
    inner-OOF probabilities, and compare with the hard-label student.

    The teacher probability for a training person must be out-of-fold. T1's
    per-fold inner-OOF vectors are not persisted, so the teacher here is
    T1's *outer* prediction for that person from the folds where they were
    held out -- which is out-of-sample for the teacher by construction.
    """
    teacher = {}
    for fold in t1_folds:
        if fold["kind"] != "within_site":
            continue
        for j, person in enumerate(fold["people"]):
            teacher[person["key"]] = fold["scores"]["S|I"][j]

    people = [p for p in rv.load_montage_people(base_people, montage) if p["key"] in teacher]
    splits = cv.folds(people, seed)
    context = cv.FoldContext(people, people, "I", normative_pool, seed)
    X = context.block(people, "b1_b6")
    y = np.array([p["y"] for p in people])
    soft = rv.distillation_targets(y, np.array([teacher[p["key"]] for p in people]))

    out = {}
    for name, targets, model_factory in (
        ("hard", y, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))),
        ("distilled", soft, lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ):
        scores = {}
        for split in splits:
            if split["kind"] != "cross_site":
                continue
            model = model_factory().fit(np.nan_to_num(X[split["train"]]), targets[split["train"]])
            predicted = model.predict_proba(np.nan_to_num(X[split["test"]]))[:, 1] if name == "hard" else model.predict(np.nan_to_num(X[split["test"]]))
            y_test = y[split["test"]]
            if len(set(y_test)) == 2:
                site = people[split["test"][0]]["site"]
                scores[site] = float(roc_auc_score(y_test, predicted))
        out[name] = {"per_site": scores, "macro": float(np.mean(list(scores.values()))) if scores else float("nan")}
    out["delta_distilled_minus_hard"] = out["distilled"]["macro"] - out["hard"]["macro"]
    return out


def main():
    base_people = cv.load_people()
    normative_pool = cv.build_normative_pool()
    t1 = json.loads((RUN / "outer-fold-results.json").read_text())
    t1_results = json.loads((RUN / "results.json").read_text())
    t1_composite = t1_results["composite"]["S|I"]

    report = {"protocol_version": "v4.1", "t1_reference": {"pipeline": "S|I", "composite": t1_composite}}
    montage_scores = {}
    for montage in rv.MONTAGES:
        print(f"=== {montage} ({len(rv.MONTAGES[montage])} channels) ===", flush=True)
        result = run_montage(montage, base_people, normative_pool)
        stack = result["scored"]["S|I"]
        montage_scores[montage] = {
            "n_channels": len(rv.MONTAGES[montage]),
            "channels": rv.MONTAGES[montage],
            "n_people": result["n_people"],
            "stack_S_I": stack,
            "families": {
                label: result["scored"][label]["composite"] for label in result["scored"]
            },
        }
        print(
            f"    S|I composite {stack['composite']:.3f} "
            f"(cross {stack['cross_site']['macro']:.3f}, within {stack['within_site']['macro']:.3f})",
            flush=True,
        )

    report["montages"] = montage_scores
    best = max(montage_scores, key=lambda m: montage_scores[m]["stack_S_I"]["composite"])
    gap = t1_composite - montage_scores[best]["stack_S_I"]["composite"]
    report["G6"] = {
        "best_predeclared_montage": best,
        "best_composite": montage_scores[best]["stack_S_I"]["composite"],
        "t1_composite": t1_composite,
        "gap": gap,
        "threshold": 0.05,
        "passed": bool(gap <= 0.05),
        "note": "Montages re-referenced to their own channels only; no 17-channel information enters.",
    }
    print(f"\nG6: best={best} gap={gap:.3f} passed={gap <= 0.05}", flush=True)

    print("\n=== distillation (M8) ===", flush=True)
    report["distillation_M8"] = distillation("M8", base_people, t1, normative_pool)
    print("  ", json.dumps(report["distillation_M8"], indent=2), flush=True)

    save(RUN / "reduced-results.json", report)
    return report


if __name__ == "__main__":
    main()
