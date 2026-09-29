"""Stage 10 (T3): cognition index, test-retest reliability, and a worked
Myna output record.

Both analyses here are exploratory (H6). They are reported with their
sample sizes attached because both are small: the cognition models run
within PD at two sites, and the ICC rests on the 22 Iowa participants who
have a second session and survive A2.
"""

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from scipy import stats  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pd_pipeline import compare_v4 as cv  # noqa: E402
from pd_pipeline import fusion_v4 as fu  # noqa: E402
from pd_pipeline import reduced_v4 as rv  # noqa: E402
from pd_pipeline.registry import save  # noqa: E402

RUN = Path("data/pd-runs/v4")
SEED = 20261001


def _clinical():
    return json.loads((RUN / "clinical.json").read_text())


def _numeric(value):
    try:
        out = float(value)
        return out if np.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def cognition_index(people, normative_pool, target="moca", montage=None):
    """Ridge regression of a clinical score on EEG features, within PD only.

    Two evaluations, both grouped by person: cross-site (train Iowa -> test
    Tel Aviv and vice versa) and grouped 5-fold. Spearman rho with a
    bootstrap CI, plus MAE.
    """
    clinical = _clinical()
    rows = []
    for person in people:
        if person["y"] != 1:  # within PD only
            continue
        record = clinical.get(person["key"])
        if not record:
            continue
        value = _numeric(record.get(target))
        if value is None:
            continue
        rows.append((person, value, record["site"]))
    if len(rows) < 30:
        return {"n": len(rows), "note": "too few people with this score to model"}

    selected = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    sites = np.array([r[2] for r in rows])

    if montage:
        montage_people = {p["key"]: p for p in rv.load_montage_people(selected, montage)}
        selected = [montage_people[p["key"]] for p in selected if p["key"] in montage_people]
        keep = [i for i, p in enumerate(rows) if p[0]["key"] in montage_people]
        y, sites = y[keep], sites[keep]

    context = cv.FoldContext(selected, selected, "I", normative_pool, SEED)
    X = np.nan_to_num(context.block(selected, "b1_b6"))

    def evaluate(train_idx, test_idx):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(X[train_idx], y[train_idx])
        return model.predict(X[test_idx])

    out = {"n": len(y), "target": target, "montage": montage or "T1_17ch",
           "by_site": {s: int((sites == s).sum()) for s in np.unique(sites)}}

    # Cross-site, both directions.
    cross = {}
    for held_out in np.unique(sites):
        test_idx = np.flatnonzero(sites == held_out)
        train_idx = np.flatnonzero(sites != held_out)
        if len(test_idx) < 10 or len(train_idx) < 10:
            continue
        predicted = evaluate(train_idx, test_idx)
        rho, p_value = stats.spearmanr(y[test_idx], predicted)
        cross[held_out] = {
            "n": int(len(test_idx)),
            "spearman_rho": float(rho),
            "p_value": float(p_value),
            "mae": float(np.mean(np.abs(y[test_idx] - predicted))),
        }
    out["cross_site"] = cross

    # Grouped 5-fold (each person appears once, so grouping is by person).
    groups = np.arange(len(y))
    predictions = np.full(len(y), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
        predictions[test_idx] = evaluate(train_idx, test_idx)
    rho, p_value = stats.spearmanr(y, predictions)
    rng = np.random.default_rng(SEED)
    boot = [
        stats.spearmanr(y[idx], predictions[idx]).statistic
        for idx in (rng.choice(len(y), len(y), replace=True) for _ in range(2000))
    ]
    out["grouped_cv"] = {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "rho_ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
        "mae": float(np.mean(np.abs(y - predictions))),
    }
    return out


def test_retest(people, normative_pool):
    """ICC(2,1) of the exported LLR across repeat Iowa sessions.

    A1 keeps one session per person, so repeat sessions are absent from the
    v4 registry and were not extracted in Stage 4. They are extracted here
    on demand (a few dozen recordings) rather than the analysis being
    skipped. Each person's two sessions are scored by a model trained on
    the other three sites only, so no session of theirs is ever in training.
    """
    from pd_pipeline import features_v4 as fv
    from pd_pipeline import qc_v4

    v3_registry = json.loads(Path("data/pd-runs/v3/registry.json").read_text())
    v4_records = json.loads((RUN / "registry.json").read_text())["records"]
    v4_ids = {r["id"] for r in v4_records}
    available = {p["key"]: p for p in people}

    repeats = {}
    for record in v3_registry["records"]:
        if record["site"] != "iowa" or record["person"] not in available:
            continue
        if record["id"] in v4_ids:
            continue
        repeats.setdefault(record["person"], []).append(record)

    thresholds = json.loads((RUN / "qc-thresholds.json").read_text())
    from pd_pipeline.cohort_v3 import ARCHIVE, safe_raw

    second_features = {}
    for person, records in repeats.items():
        record = sorted(records, key=lambda r: r["id"])[0]
        target = RUN / "windows" / (record["id"] + ".npz")
        if not target.exists():
            try:
                with safe_raw(ARCHIVE / record["path"]) as raw:
                    result, reason = qc_v4.extract(
                        raw, record["interval"], record["role"], thresholds
                    )
                if reason or not len(result["tokens"]):
                    continue
                np.savez(
                    target,
                    **{k: v for k, v in result.items() if k not in {"rejections", "attempted"}},
                )
            except Exception:
                continue
        try:
            second_features[person] = fv.person_features([record["id"]], run=RUN)
        except Exception:
            continue

    if len(second_features) < 5:
        return {
            "n_people_with_second_session": len(second_features),
            "status": "not run",
            "reason": "Too few repeat sessions produced usable windows.",
        }

    # Train on the three non-Iowa sites so neither session of an Iowa
    # person is ever in the model's training data.
    train_people = [p for p in people if p["site"] != "iowa"]
    context = cv.FoldContext(train_people, people, "I", normative_pool, SEED)
    from pd_pipeline.models_v4 import FAMILIES

    spec = FAMILIES["F0_rich_svm"]
    model = spec["build"]({"C": 1.0})
    model.fit(context.block(train_people, spec["block"]), [p["y"] for p in train_people])
    prevalence = float(np.mean([p["y"] for p in train_people]))

    paired, keys = [], []
    for person, features in second_features.items():
        first = available[person]
        second = dict(first)
        second["v3_rich"] = features["v3_rich"]
        p1 = model.predict_proba(context.block([first], spec["block"]))[0, 1]
        p2 = model.predict_proba(context.block([second], spec["block"]))[0, 1]
        paired.append(
            [
                fu.log_likelihood_ratio(p1, prevalence),
                fu.log_likelihood_ratio(p2, prevalence),
            ]
        )
        keys.append(person)

    paired = np.array(paired)
    icc = fu.icc_2_1(paired)
    rho = float(stats.spearmanr(paired[:, 0], paired[:, 1]).statistic)
    return {
        "n_people_with_second_session": len(paired),
        "status": "run",
        "icc_2_1_of_llr": icc,
        "spearman_between_sessions": rho,
        "meets_0.6_target": bool(np.isfinite(icc) and icc >= 0.6),
        "scorer": "F0 rich-SVM trained on the three non-Iowa sites; neither session of a scored person is in training.",
        "caveat": (
            f"n={len(paired)} people. An ICC on this few subjects has a wide confidence "
            "interval and should be read as indicative only."
        ),
    }


def main():
    people = cv.load_people()
    normative_pool = cv.build_normative_pool()
    prevalence = float(np.mean([p["y"] for p in people]))

    report = {"protocol_version": "v4.1", "training_prevalence": prevalence}

    print("schema ...", flush=True)
    fu.write_schema()
    report["schema_written"] = str(fu.SCHEMA_PATH)

    print("fusion over-counting demonstration ...", flush=True)
    report["naive_bayes_overcounting"] = fu.demonstrate_overcounting()

    print("cognition index: MoCA on T1 ...", flush=True)
    report["cognition_moca_T1"] = cognition_index(people, normative_pool, "moca")
    print("   ", json.dumps(report["cognition_moca_T1"].get("grouped_cv", {}), indent=2), flush=True)

    print("cognition index: MoCA on M8 ...", flush=True)
    report["cognition_moca_M8"] = cognition_index(people, normative_pool, "moca", montage="M8")

    print("cognition index: UPDRS-III on T1 ...", flush=True)
    report["cognition_updrs3_T1"] = cognition_index(people, normative_pool, "updrs3")

    print("test-retest ...", flush=True)
    report["test_retest"] = test_retest(people, normative_pool)

    # A worked example record, so the schema is demonstrated end to end.
    example = fu.build_output(
        model_id="example-not-a-fitted-model",
        montage="M8",
        regime="inductive",
        quality={"accepted_windows": 15, "acceptance_fraction": 0.83, "line_noise_ratio": 0.04},
        calibrated_probability=0.71,
        training_prevalence=prevalence,
        embedding=np.zeros(32),
    )
    fu.validate_output(example)
    report["example_record"] = example

    save(RUN / "fusion-results.json", report)
    print("\nDONE", flush=True)
    return report


if __name__ == "__main__":
    main()
