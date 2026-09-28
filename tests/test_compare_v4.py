"""v4 nested comparison: fold construction, the D3 objective, the
non-negative stacker, and the leakage boundary (no fitted transform may
see a held-out person)."""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pd_pipeline import compare_v4 as cv  # noqa: E402
from pd_pipeline.models_v4 import LeapdScorer, NonNegativeStacker, platt_calibrate  # noqa: E402


def _person(key, site, y, seed=0, with_cov=True):
    rng = np.random.default_rng(seed)
    cov = None
    if with_cov:
        a = rng.normal(size=(17, 17))
        cov = (a @ a.T + 17 * np.eye(17)).astype("float32")
    return {
        "key": key,
        "site": site,
        "y": y,
        "age": 60.0 + (seed % 20),
        "sex": float(seed % 2),
        "b1": rng.normal(size=684).astype("float32"),
        "b2": rng.normal(size=163).astype("float32"),
        "b4": rng.normal(size=408).astype("float32"),
        "b5": rng.normal(size=136).astype("float32"),
        "b6": rng.normal(size=205).astype("float32"),
        "v3_rich": rng.normal(size=1309).astype("float32"),
        "normative": rng.normal(size=38),
        "cov": cov,
    }


def _cohort(n_per_site=8):
    people = []
    seed = 0
    for site in ("a", "b", "c", "d"):
        for i in range(n_per_site):
            people.append(_person(f"{site}:{i}", site, i % 2, seed))
            seed += 1
    return people


def test_folds_are_participant_disjoint_and_site_pure():
    people = _cohort()
    splits = cv.folds(people)
    assert len([s for s in splits if s["kind"] == "cross_site"]) == 4
    assert len([s for s in splits if s["kind"] == "within_site"]) == 3
    for split in splits:
        assert set(split["train"]).isdisjoint(split["test"])
        cv.validate_disjoint(
            [people[i] for i in split["train"]], [people[i] for i in split["test"]]
        )
        if split["kind"] == "cross_site":
            test_sites = {people[i]["site"] for i in split["test"]}
            train_sites = {people[i]["site"] for i in split["train"]}
            assert len(test_sites) == 1 and test_sites.isdisjoint(train_sites)


def test_within_site_folds_cover_every_person_exactly_once():
    people = _cohort()
    splits = [s for s in cv.folds(people) if s["kind"] == "within_site"]
    covered = sorted(i for s in splits for i in s["test"])
    assert covered == list(range(len(people)))


def test_d3_objective_penalises_site_variability():
    steady = cv.d3_score(0.8, 0.8, [0.8, 0.8, 0.8])
    erratic = cv.d3_score(0.8, 0.8, [0.6, 0.8, 1.0])
    assert steady > erratic
    assert steady == pytest.approx(0.8)


def test_stacker_weights_are_non_negative():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    # One informative column, one deliberately anti-correlated column.
    good = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.05, len(y)), 0.01, 0.99)
    inverted = 1 - good
    P = np.column_stack([good, inverted])
    stacker = NonNegativeStacker().fit(P, y)
    assert (stacker.weights_ >= 0).all()
    assert stacker.predict_proba(P).shape == (len(y), 2)


def test_stacker_beats_chance_on_a_clearly_informative_input():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    y = np.array([0] * 50 + [1] * 50)
    P = np.column_stack(
        [np.clip(0.5 + 0.35 * (y - 0.5) + rng.normal(0, 0.05, len(y)), 0.01, 0.99)] * 2
    )
    stacker = NonNegativeStacker().fit(P, y)
    assert roc_auc_score(y, stacker.predict_proba(P)[:, 1]) > 0.9


def test_leapd_fits_hyperplanes_on_training_people_only():
    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(30, 408))
    y_train = np.array([0] * 15 + [1] * 15)
    scorer = LeapdScorer(n_components=2).fit(X_train, y_train)
    # Scoring unseen rows must work without refitting and must not mutate state.
    weights_before = [
        None if m[0] is None else m[0].components_.copy() for m in scorer.models_
    ]
    scores = scorer.transform(rng.normal(size=(7, 408)))
    assert scores.shape == (7, 17)
    for before, model in zip(weights_before, scorer.models_):
        if before is not None:
            assert np.allclose(before, model[0].components_)


def test_platt_calibration_is_monotone():
    rng = np.random.default_rng(3)
    y = np.array([0] * 40 + [1] * 40)
    scores = y + rng.normal(0, 0.3, len(y))
    model = platt_calibrate(scores, y)
    grid = np.linspace(-2, 3, 25).reshape(-1, 1)
    probabilities = model.predict_proba(grid)[:, 1]
    assert np.all(np.diff(probabilities) >= -1e-9)


def test_no_fitted_transform_sees_held_out_people():
    """Leakage boundary: poison every held-out person's features with NaN
    sentinels before fitting. If any fitted transform touched a test row,
    the fitted statistics would be corrupted and predictions would differ
    from a clean fit on the same training rows."""
    people = _cohort(n_per_site=6)
    normative_pool = [
        {"normative": np.random.default_rng(i).normal(size=38), "age": 55.0 + i % 20, "sex": float(i % 2)}
        for i in range(60)
    ]
    split = [s for s in cv.folds(people) if s["kind"] == "cross_site"][0]
    train_people = [people[i] for i in split["train"]]

    clean_context = cv.FoldContext(train_people, people, "I", normative_pool)
    clean_block = clean_context.block(train_people, "b2_c3")

    poisoned = [dict(p) for p in people]
    for i in split["test"]:
        poisoned[i] = dict(poisoned[i])
        poisoned[i]["normative"] = np.full(38, np.nan)
        poisoned[i]["b2"] = np.full(163, np.nan, dtype="float32")
    poisoned_train = [poisoned[i] for i in split["train"]]
    poisoned_context = cv.FoldContext(poisoned_train, poisoned, "I", normative_pool)
    poisoned_block = poisoned_context.block(poisoned_train, "b2_c3")

    # Training-row features must be identical: the normative model was fit
    # without ever reading a held-out person.
    assert np.allclose(clean_block, poisoned_block, equal_nan=True)


def test_transductive_regime_is_restricted_to_guardrail_survivors():
    # Stage 6 excluded C2, so only F3 may have a pre-declared T variant.
    assert set(cv.TRANSDUCTIVE_METHODS) == {"F3_riemann_logistic"}
    labels = {f"{family}|{regime}" for family, regime in cv.variants()}
    assert "F3_riemann_logistic|T" in labels
    assert "F1_relative_svm|T" not in labels


def test_evaluate_refuses_before_the_outer_lock_exists(tmp_path):
    with pytest.raises(RuntimeError, match="not locked"):
        cv.evaluate(run=tmp_path)
