"""Development-only negative controls and descriptive, post-lock test slices."""

import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .cohort_v2 import load, SEED
from .experiments import spectral_features, metrics
from .registry import save


def permutation_check(run, count=100):
    dest = run / "permutation-check.json"
    if dest.exists():
        return json.loads(dest.read_text())
    data = [r for r in load(run) if r["split"] == "train"]
    x = np.stack([spectral_features(r["x"]) for r in data])
    y = np.array([r["label"] for r in data])
    rng = np.random.default_rng(SEED)

    def score(labels):
        folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(x, labels))
        scores = []
        for c in [0.01, 0.1, 1, 10]:
            values = []
            for a, b in folds:
                m = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, class_weight="balanced", max_iter=2000),
                ).fit(x[a], labels[a])
                values.append(roc_auc_score(labels[b], m.predict_proba(x[b])[:, 1]))
            scores.append(np.mean(values))
        return float(max(scores))

    observed = score(y)
    null = []
    for i in range(count):
        null.append(score(rng.permutation(y)))
        if (i + 1) % 20 == 0:
            print("permutations", i + 1, flush=True)
    result = {
        "training_participants": len(y),
        "permutations": count,
        "observed_selected_cv_auroc": observed,
        "null_mean": float(np.mean(null)),
        "null_95_interval": np.quantile(null, [0.025, 0.975]).tolist(),
        "monte_carlo_p": float((1 + sum(v >= observed for v in null)) / (count + 1)),
        "scope": "Spectral baseline only; repeats C selection and participant-fold construction per permutation. Does not certify neural implementations or rule out confounding.",
    }
    save(dest, result)
    return result


def subgroup_audit(run):
    predictions = json.loads((run / "participant-predictions.json").read_text())
    records = load(run, external=True)
    by_key = {r["participant_key"]: r for r in records}
    lock = json.loads((run / "all-models-lock.json").read_text())
    out = {}
    for endpoint, rows in predictions.items():
        groups = {}
        for label, filter_fn in [
            ("age_under_65", lambda r: float(r["age"]) < 65),
            ("age_65_plus", lambda r: float(r["age"]) >= 65),
            ("female", lambda r: r["sex"] in ["F", "Female"]),
            ("male", lambda r: r["sex"] in ["M", "Male"]),
        ]:
            chosen = []
            for row in rows:
                try:
                    keep = filter_fn(by_key[row["participant_key"]])
                except (ValueError, TypeError):
                    keep = False
                if keep:
                    chosen.append(row)
            y = np.array([r["label"] for r in chosen])
            counts = {str(c): int(sum(y == c)) for c in [0, 1]}
            if min(counts.values()) < 5:
                groups[label] = {
                    "class_counts": counts,
                    "not_estimated": "Fewer than five people in at least one class",
                }
                continue
            groups[label] = {
                "class_counts": counts,
                "models": {
                    n: metrics(
                        y,
                        [r["predictions"][n] for r in chosen],
                        lock["models"][n]["threshold"],
                        bootstrap=500,
                    )
                    for n in lock["models"]
                },
            }
        out[endpoint] = groups
    save(run / "subgroup-audit.json", out)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["permutation", "subgroups"])
    p.add_argument("--run", type=Path, default=Path("data/pd-runs/v2"))
    a = p.parse_args()
    (permutation_check if a.action == "permutation" else subgroup_audit)(a.run)
