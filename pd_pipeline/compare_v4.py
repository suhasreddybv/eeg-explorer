"""v4 nested comparison.

folds() (4 leave-one-site-out outer folds + 3 within-site participant
folds, adapted from compare_v3), the D3 selection objective, the level-0
families and stacker, the campaign/evaluate drivers, and the negative
controls.

Leakage boundaries, all enforced here rather than by convention:
- Every fitted transform lives inside a family Pipeline fit on training
  people, or is built by `_fold_context` from training people only.
- The C3 normative model is fit on healthy auxiliary adults plus the
  training fold's controls -- never a held-out person, never anyone's
  diagnosis beyond "is a training control".
- Regime I (inductive) never touches test-site rows. Regime T
  (transductive) may use the test site's *unlabelled* rows for site
  statistics only, and only via a method that passed the Stage 6
  prevalence guardrail (C1; C2 was excluded).
- The stacker trains on inner-OOF probabilities only.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md for the frozen rules this module implements.
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .cohort_v3 import ARCHIVE
from .features_v4 import normative_features, tangent_vector
from .harmonize_v4 import (
    normative_fit,
    normative_source_people,
    normative_z,
    recenter_riemann,
)
from .experiments import choose_threshold, metrics, wilson
from .models_v4 import FAMILIES, NonNegativeStacker
from .registry import freeze, save

RUN = Path("data/pd-runs/v4")
SEED = 20261001
# Stage 6 guardrail outcome: C1 passed at every outer site, C2 failed the
# majority rule. Only C1 may back a pre-declared transductive variant.
TRANSDUCTIVE_METHODS = {"F3_riemann_logistic": "C1"}


def _sex_to_number(value):
    return {"M": 0.0, "F": 1.0, "MALE": 0.0, "FEMALE": 1.0}.get(str(value).strip().upper(), np.nan)


def _age_to_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_people(run=RUN, role="supervised"):
    """Every person of `role` with cached features, as one record each."""
    registry = json.loads((run / "registry.json").read_text())
    features_dir = run / "features"
    people = []
    for record in registry["records"]:
        if record["role"] != role:
            continue
        path = features_dir / (record["person"].replace(":", "__") + ".npz")
        if not path.exists():
            continue
        with np.load(path) as data:
            entry = {
                "key": record["person"],
                "site": record["site"],
                "y": record["label"],
                "age": _age_to_number(record.get("age")),
                "sex": _sex_to_number(record.get("sex")),
                "b1": data["b1"],
                "b2": data["b2"],
                "b4": data["b4"],
                "b5": data["b5"],
                "b6": data["b6"],
                "v3_rich": data["v3_rich"],
                "normative": normative_features(data),
                "cov": data["b3_cov_mean"] if "b3_cov_mean" in data else None,
            }
        people.append(entry)
    seen = [p["key"] for p in people]
    if len(seen) != len(set(seen)):
        raise ValueError("Duplicate person in supervised set")
    return people


def validate_disjoint(train, test):
    if {p["key"] for p in train} & {p["key"] for p in test}:
        raise ValueError("Participant leakage")


def folds(people, seed=SEED):
    """4 leave-one-site-out outer folds plus 3 within-site participant
    folds stratified by site x diagnosis -- v3's construction with v4's
    seed."""
    sites = sorted({p["site"] for p in people})
    out = []
    for site in sites:
        test = [i for i, p in enumerate(people) if p["site"] == site]
        train = [i for i, p in enumerate(people) if p["site"] != site]
        out.append({"name": "site_" + site, "kind": "cross_site", "train": train, "test": test})
    strata = [p["site"] + ":" + str(p["y"]) for p in people]
    for i, (train, test) in enumerate(
        StratifiedKFold(3, shuffle=True, random_state=seed).split(strata, strata)
    ):
        out.append(
            {
                "name": "within_" + str(i),
                "kind": "within_site",
                "train": train.tolist(),
                "test": test.tolist(),
            }
        )
    return out


class FoldContext:
    """Everything fold-dependent: the C3 normative model, the C1 tangent
    reference, and the feature-block assembly that depends on them. Built
    from training people (plus label-free healthy auxiliary data) only."""

    def __init__(self, train_people, all_people, regime, normative_pool, seed=SEED):
        self.regime = regime
        self.seed = seed
        self._fit_normative(train_people, normative_pool)
        self._fit_tangent_reference(train_people, all_people, regime)

    def _fit_normative(self, train_people, normative_pool):
        X = [entry["normative"] for entry in normative_pool]
        age = [entry["age"] for entry in normative_pool]
        sex = [entry["sex"] for entry in normative_pool]
        for person in train_people:  # training-fold controls only
            if person["y"] == 0 and np.isfinite(person["age"]) and np.isfinite(person["sex"]):
                X.append(person["normative"])
                age.append(person["age"])
                sex.append(person["sex"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.normative_model = normative_fit(np.array(X), np.array(age), np.array(sex))

    def _fit_tangent_reference(self, train_people, all_people, regime):
        covs = [p["cov"] for p in train_people if p["cov"] is not None]
        if not covs:
            self.tangent_reference = None
            return
        from pyriemann.utils.mean import mean_riemann

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tangent_reference = mean_riemann(np.stack(covs).astype("float64"))
        self.recentred = None
        if regime == "T":
            # C1: per-site whitening using every person's covariance at that
            # site, including the test site's unlabelled people. Labels are
            # never used.
            with_cov = [p for p in all_people if p["cov"] is not None]
            recentred = recenter_riemann(
                np.stack([p["cov"] for p in with_cov]).astype("float64"),
                np.array([p["site"] for p in with_cov]),
            )
            self.recentred = {p["key"]: m for p, m in zip(with_cov, recentred)}

    def extend_transductive(self, people):
        """Transductive C1 for people the context was not fitted over -- a
        new site at scoring time.

        The whole point of C1 is that a site's statistics come from that
        site's own (unlabelled) recordings, so a locked site must be
        re-centred on itself, not looked up in the training map. Labels are
        never involved.
        """
        if self.regime != "T":
            return
        with_cov = [p for p in people if p.get("cov") is not None]
        missing = [p for p in with_cov if p["key"] not in (self.recentred or {})]
        if not missing:
            return
        recentred = recenter_riemann(
            np.stack([p["cov"] for p in missing]).astype("float64"),
            np.array([p["site"] for p in missing]),
        )
        if self.recentred is None:
            self.recentred = {}
        self.recentred.update({p["key"]: m for p, m in zip(missing, recentred)})

    def normative_z_for(self, people):
        X = np.array([p["normative"] for p in people])
        age = np.array([p["age"] for p in people])
        sex = np.array([p["sex"] for p in people])
        age = np.where(np.isfinite(age), age, np.nanmedian(age[np.isfinite(age)]) if np.isfinite(age).any() else 60.0)
        sex = np.where(np.isfinite(sex), sex, 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return normative_z(self.normative_model, X, age, sex)

    def block(self, people, name):
        """Assemble a family's input matrix for these people."""
        if name == "v3_rich":
            return np.stack([p["v3_rich"] for p in people])
        if name == "b1_b6":
            return np.stack([np.concatenate([p["b1"], p["b6"]]) for p in people])
        if name == "b2_c3":
            return np.hstack([np.stack([p["b2"] for p in people]), self.normative_z_for(people)])
        if name == "b4":
            return np.stack([p["b4"] for p in people])
        if name == "b5":
            return np.stack([p["b5"] for p in people])
        if name == "fm":
            # Track E: frozen foundation-model embedding (Stage 8). Present
            # only when fm_v4 has attached one; never part of the
            # pre-declared primary pipeline.
            return np.stack([p["fm"] for p in people])
        if name == "b2":
            # B-2 alone, no C3 z-scores. Used by the T2 montage variants:
            # the cached normative vector is a 17-channel quantity, so
            # feeding it to a montage model would leak electrodes the
            # montage does not have into a "wearable" result.
            return np.stack([p["b2"] for p in people])
        if name == "b3_tangent":
            # Dimension follows the covariances themselves (17 for T1, k for
            # a T2 montage), never a hardcoded 17.
            present = next((p["cov"] for p in people if p["cov"] is not None), None)
            width = 17 if present is None else present.shape[-1]
            identity = np.eye(width)
            empty = len(tangent_vector(identity, identity))
            rows = []
            for p in people:
                if p["cov"] is None:
                    rows.append(np.zeros(empty))
                    continue
                if self.regime == "T" and self.recentred is not None:
                    rows.append(tangent_vector(self.recentred[p["key"]], identity))
                else:
                    reference = (
                        self.tangent_reference
                        if self.tangent_reference is not None
                        and self.tangent_reference.shape[-1] == width
                        else identity
                    )
                    rows.append(tangent_vector(p["cov"].astype("float64"), reference))
            return np.stack(rows)
        if name == "compact":
            # F6: B-1 + B-2 + the C3 z-scores, capped at 250 features.
            z = self.normative_z_for(people)
            base = np.stack(
                [np.concatenate([p["b1"][-4:], p["b2"]]) for p in people]
            )
            combined = np.hstack([base, z])
            return combined[:, :250]
        raise ValueError(f"Unknown feature block {name}")


def context_factory():
    """Memoises FoldContext by (regime, training-people set).

    The C3 normative model and the C1/tangent reference depend only on
    which people are in the training fold, not on which family or
    hyper-parameter is being evaluated, so the same context is otherwise
    refit identically dozens of times. Caching changes nothing
    scientifically -- the key is the exact training set -- and cuts the
    campaign's runtime by roughly an order of magnitude.
    """
    cache = {}

    def get(train_people, all_people, regime, normative_pool, seed=SEED):
        key = (regime, frozenset(p["key"] for p in train_people))
        if key not in cache:
            cache[key] = FoldContext(train_people, all_people, regime, normative_pool, seed)
        return cache[key]

    return get


def d3_score(inner_cross, inner_within, site_aurocs):
    """Protocol D3: 0.5*cross + 0.5*within - 0.5*SD(per-site cross)."""
    return 0.5 * inner_cross + 0.5 * inner_within - 0.5 * float(np.std(site_aurocs))


def _equal_site_auroc(people, indices, scores):
    by_site = defaultdict(lambda: ([], []))
    for idx, score in zip(indices, scores):
        by_site[people[idx]["site"]][0].append(people[idx]["y"])
        by_site[people[idx]["site"]][1].append(score)
    per_site = {
        site: float(roc_auc_score(ys, ss))
        for site, (ys, ss) in by_site.items()
        if len(set(ys)) == 2
    }
    return per_site


def inner_evaluate(people, train_indices, family, params, regime, normative_pool, seed=SEED, get_context=None, family_specs=None):
    """Inner loop for one candidate: leave-one-site-out across the training
    sites (cross component) plus 3 stratified participant folds (within
    component), returning the D3 score and per-person OOF probabilities."""
    spec = (family_specs or FAMILIES)[family]
    get_context = get_context or context_factory()
    train_people = [people[i] for i in train_indices]
    sites = sorted({p["site"] for p in train_people})
    oof = np.full(len(train_indices), np.nan)

    cross_site_aurocs = []
    for site in sites:
        inner_train = [i for i, p in enumerate(train_people) if p["site"] != site]
        inner_test = [i for i, p in enumerate(train_people) if p["site"] == site]
        if len({train_people[i]["y"] for i in inner_test}) < 2 or not inner_train:
            continue
        context = get_context(
            [train_people[i] for i in inner_train], train_people, regime, normative_pool, seed
        )
        model = spec["build"](params)
        X_train = context.block([train_people[i] for i in inner_train], spec["block"])
        X_test = context.block([train_people[i] for i in inner_test], spec["block"])
        y_train = [train_people[i]["y"] for i in inner_train]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)[:, 1]
        oof[inner_test] = probabilities
        y_test = [train_people[i]["y"] for i in inner_test]
        cross_site_aurocs.append(float(roc_auc_score(y_test, probabilities)))

    within_aurocs = []
    strata = [p["site"] + ":" + str(p["y"]) for p in train_people]
    for inner_train, inner_test in StratifiedKFold(3, shuffle=True, random_state=seed).split(
        strata, strata
    ):
        context = get_context(
            [train_people[i] for i in inner_train], train_people, regime, normative_pool, seed
        )
        model = spec["build"](params)
        X_train = context.block([train_people[i] for i in inner_train], spec["block"])
        X_test = context.block([train_people[i] for i in inner_test], spec["block"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, [train_people[i]["y"] for i in inner_train])
            probabilities = model.predict_proba(X_test)[:, 1]
        per_site = _equal_site_auroc(train_people, inner_test, probabilities)
        if per_site:
            within_aurocs.append(float(np.mean(list(per_site.values()))))

    inner_cross = float(np.mean(cross_site_aurocs)) if cross_site_aurocs else 0.5
    inner_within = float(np.mean(within_aurocs)) if within_aurocs else 0.5
    return {
        "score": d3_score(inner_cross, inner_within, cross_site_aurocs or [0.0]),
        "inner_cross": inner_cross,
        "inner_within": inner_within,
        "site_aurocs": cross_site_aurocs,
        "oof": oof,
    }


def select_family(people, train_indices, family, regime, normative_pool, seed=SEED, get_context=None, family_specs=None):
    """Pick this family's hyper-parameters by the D3 objective, using inner
    folds of the training people only."""
    results = []
    family_specs = family_specs or FAMILIES
    for params in family_specs[family]["grid"]:
        evaluation = inner_evaluate(
            people, train_indices, family, params, regime, normative_pool, seed, get_context,
            family_specs,
        )
        results.append({"params": params, **evaluation})
    best = max(results, key=lambda r: r["score"])
    return best, [
        {k: v for k, v in r.items() if k != "oof"} for r in results
    ]


# ---------------------------------------------------------------------------
# Campaign: per-outer-fold selection, stacking, and the outer lock
# ---------------------------------------------------------------------------


def build_normative_pool(run=RUN, archive=ARCHIVE):
    """Healthy auxiliary adults >=40y with cached features. Label-free and
    fold-independent, so it is built once and reused."""
    registry = json.loads((run / "registry.json").read_text())
    demographics = normative_source_people(registry, archive)
    features_dir = run / "features"
    pool = []
    for person, meta in demographics.items():
        path = features_dir / (person.replace(":", "__") + ".npz")
        if not path.exists():
            continue
        with np.load(path) as data:
            pool.append(
                {"normative": normative_features(data), "age": meta["age"], "sex": meta["sex"]}
            )
    return pool


def variants(family_specs=None):
    """Every (family, regime) pair the frozen protocol allows. Regime T is
    offered only where a transductive method survived Stage 6's guardrail."""
    family_specs = family_specs or FAMILIES
    out = [(family, "I") for family in family_specs]
    out += [(family, "T") for family in family_specs if family in TRANSDUCTIVE_METHODS]
    return out


def run_fold(people, split, normative_pool, seed=SEED, family_specs=None):
    """One outer fold: select each family by D3 on the training people,
    stack their inner-OOF probabilities, refit level-0 on the full outer
    training set, and score the held-out people."""
    train_indices = split["train"]
    test_indices = split["test"]
    train_people = [people[i] for i in train_indices]
    test_people = [people[i] for i in test_indices]
    validate_disjoint(train_people, test_people)
    y_train = np.array([p["y"] for p in train_people])
    get_context = context_factory()
    family_specs = family_specs or FAMILIES

    selections = {}
    oof_columns = {}
    outer_scores = {}
    for family, regime in variants(family_specs):
        label = f"{family}|{regime}"
        best, all_candidates = select_family(
            people, train_indices, family, regime, normative_pool, seed, get_context, family_specs
        )
        selections[label] = {
            "params": best["params"],
            "d3_score": best["score"],
            "inner_cross": best["inner_cross"],
            "inner_within": best["inner_within"],
            "candidates": all_candidates,
        }
        oof = best["oof"]
        if np.isnan(oof).any():
            oof = np.where(np.isnan(oof), np.nanmean(oof) if np.isfinite(oof).any() else 0.5, oof)
        oof_columns[label] = oof

        # Refit on the whole outer-training set and score the held-out people.
        context = get_context(train_people, people, regime, normative_pool, seed)
        spec = family_specs[family]
        model = spec["build"](best["params"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(context.block(train_people, spec["block"]), y_train)
            outer_scores[label] = model.predict_proba(
                context.block(test_people, spec["block"])
            )[:, 1]

    # Pre-declared stacker S: inductive level-0 families only.
    inductive = [f"{family}|I" for family in family_specs]
    stacker = NonNegativeStacker(C=1.0).fit(
        np.column_stack([oof_columns[label] for label in inductive]), y_train
    )
    stacked_outer = stacker.predict_proba(
        np.column_stack([outer_scores[label] for label in inductive])
    )[:, 1]
    outer_scores["S|I"] = stacked_outer
    stacker_weights = dict(zip(inductive, stacker.weights_.tolist()))
    oof_columns["S|I"] = stacker.predict_proba(
        np.column_stack([oof_columns[label] for label in inductive])
    )[:, 1]

    # Secondary transductive stack: swap in the T variant where one exists.
    transductive = [
        f"{family}|T" if family in TRANSDUCTIVE_METHODS else f"{family}|I"
        for family in family_specs
    ]
    stacker_t = NonNegativeStacker(C=1.0).fit(
        np.column_stack([oof_columns[label] for label in transductive]), y_train
    )
    outer_scores["S|T"] = stacker_t.predict_proba(
        np.column_stack([outer_scores[label] for label in transductive])
    )[:, 1]
    oof_columns["S|T"] = stacker_t.predict_proba(
        np.column_stack([oof_columns[label] for label in transductive])
    )[:, 1]

    # D6: operating thresholds come from inner-OOF predictions only.
    thresholds = {
        label: choose_threshold(y_train, oof_columns[label]) for label in outer_scores
    }

    return {
        "name": split["name"],
        "kind": split["kind"],
        "selections": selections,
        "stacker_weights": {"S|I": stacker_weights, "S|T": dict(zip(transductive, stacker_t.weights_.tolist()))},
        "thresholds": thresholds,
        "people": [
            {"key": people[i]["key"], "site": people[i]["site"], "y": people[i]["y"]}
            for i in test_indices
        ],
        "scores": {label: np.asarray(values).tolist() for label, values in outer_scores.items()},
    }


def campaign(run=RUN, seed=SEED):
    """Fit every outer model, then freeze the outer lock. Outer predictions
    are written by evaluate(), strictly after the lock exists."""
    people = load_people(run)
    normative_pool = build_normative_pool(run)
    splits = folds(people, seed)
    freeze(
        run / "splits.json",
        {
            "people": [{"key": p["key"], "site": p["site"], "label": p["y"]} for p in people],
            "folds": splits,
            "seed": seed,
        },
    )
    results = []
    for split in splits:
        print(f"fold {split['name']} ...", flush=True)
        results.append(run_fold(people, split, normative_pool, seed))
        save(run / "campaign-progress.json", {"completed": len(results), "total": len(splits)})
    save(run / "outer-fold-results.json", results)
    freeze(
        run / "outer-models-lock.json",
        {
            "n_people": len(people),
            "n_folds": len(splits),
            "families": sorted(FAMILIES),
            "variants": [f"{f}|{r}" for f, r in variants()],
            "pre_declared_primary": "S|I",
            "pre_declared_secondary": "S|T",
            "transductive_methods": TRANSDUCTIVE_METHODS,
            "seed": seed,
            "selections": {r["name"]: {k: v["params"] for k, v in r["selections"].items()} for r in results},
        },
    )
    return results


# ---------------------------------------------------------------------------
# Evaluation: strictly after the outer lock exists
# ---------------------------------------------------------------------------


def _bootstrap_delta(y, scores, baseline, n=2000, seed=SEED):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    observed = float(roc_auc_score(y, scores) - roc_auc_score(y, baseline))
    draws = []
    for _ in range(n):
        idx = np.concatenate(
            [rng.choice(np.flatnonzero(y == label), (y == label).sum(), replace=True) for label in (0, 1)]
        )
        draws.append(roc_auc_score(y[idx], np.asarray(scores)[idx]) - roc_auc_score(y[idx], np.asarray(baseline)[idx]))
    return {
        "delta": observed,
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "interpretation": "Exploratory, unadjusted for multiplicity",
    }


def evaluate(run=RUN, seed=SEED):
    """Outer metrics, paired deltas and negative controls. Refuses to run
    before outer-models-lock.json exists, so no outer number can ever
    influence a model."""
    lock_path = run / "outer-models-lock.json"
    if not lock_path.exists():
        raise RuntimeError("Outer models are not locked; refusing to compute outer predictions")
    results = json.loads((run / "outer-fold-results.json").read_text())
    variant_names = sorted({name for fold in results for name in fold["scores"]})

    aggregate = {}
    for kind in ("cross_site", "within_site"):
        by_site = defaultdict(lambda: defaultdict(lambda: {"y": [], "p": [], "t": []}))
        for fold in results:
            if fold["kind"] != kind:
                continue
            for j, person in enumerate(fold["people"]):
                for name in variant_names:
                    bucket = by_site[person["site"]][name]
                    bucket["y"].append(person["y"])
                    bucket["p"].append(fold["scores"][name][j])
                    bucket["t"].append(fold["thresholds"][name])
        site_results = {}
        for site, per_variant in by_site.items():
            site_results[site] = {}
            baseline = np.array(per_variant["F0_rich_svm|I"]["p"])
            for name, bucket in per_variant.items():
                y = np.array(bucket["y"])
                p = np.array(bucket["p"])
                t = np.array(bucket["t"])
                if len(set(y)) < 2:
                    continue
                m = metrics(y, p, bootstrap=2000, seed=seed)
                classified = p >= t
                tn = int(((y == 0) & ~classified).sum())
                fp = int(((y == 0) & classified).sum())
                fn = int(((y == 1) & ~classified).sum())
                tp = int(((y == 1) & classified).sum())
                m.update(
                    threshold=None,
                    threshold_policy="Per-fold inner out-of-fold balanced-accuracy optimum",
                    confusion_matrix=[[tn, fp], [fn, tp]],
                    sensitivity=tp / max(tp + fn, 1),
                    specificity=tn / max(tn + fp, 1),
                    sensitivity_95ci_wilson=wilson(tp, max(tp + fn, 1)),
                    specificity_95ci_wilson=wilson(tn, max(tn + fp, 1)),
                    paired_delta_vs_F0=_bootstrap_delta(y, p, baseline, seed=seed),
                )
                site_results[site][name] = m
        aggregate[kind] = {
            "sites": site_results,
            "equal_site_macro_auroc": {
                name: float(np.mean([site_results[s][name]["auroc"] for s in site_results if name in site_results[s]]))
                for name in variant_names
            },
        }

    aggregate["composite"] = {
        name: 0.5 * aggregate["cross_site"]["equal_site_macro_auroc"][name]
        + 0.5 * aggregate["within_site"]["equal_site_macro_auroc"][name]
        for name in variant_names
    }
    aggregate["pre_declared"] = {
        "primary": "S|I",
        "secondary": "S|T",
        "note": "Every other variant is descriptive, not a pre-declared result.",
    }
    freeze(run / "outer-predictions.json", results)
    freeze(run / "results.json", aggregate)
    return aggregate


def negative_controls(run=RUN, seed=SEED, n_permutations=20):
    """A5 controls: permuted labels, a site-identity probe on each feature
    block, age/sex only, and recording-metadata only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .harmonize_v4 import site_identity_probe

    people = load_people(run)
    normative_pool = build_normative_pool(run)
    splits = folds(people, seed)
    rng = np.random.default_rng(seed)
    context = FoldContext(people, people, "I", normative_pool, seed)

    # (i) permutation null for the primary pipeline's strongest level-0 family
    permuted = []
    spec = FAMILIES["F0_rich_svm"]
    X_all = context.block(people, spec["block"])
    sites = np.array([p["site"] for p in people])
    y_all = np.array([p["y"] for p in people])
    for _ in range(n_permutations):
        y_permuted = y_all.copy()
        for site in np.unique(sites):  # permute within site
            idx = np.flatnonzero(sites == site)
            y_permuted[idx] = rng.permutation(y_permuted[idx])
        fold_scores = []
        for split in splits:
            if split["kind"] != "cross_site":
                continue
            model = spec["build"]({"C": 1.0})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_all[split["train"]], y_permuted[split["train"]])
                p = model.predict_proba(X_all[split["test"]])[:, 1]
            y_test = y_permuted[split["test"]]
            if len(set(y_test)) == 2:
                fold_scores.append(float(roc_auc_score(y_test, p)))
        permuted.append(float(np.mean(fold_scores)) if fold_scores else float("nan"))

    # (ii) site-identity probe per feature block
    probes = {}
    for block in ("v3_rich", "b1_b6", "b2_c3", "b3_tangent", "b4", "b5"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probes[block] = site_identity_probe(context.block(people, block), sites, seed)

    # (iii) age/sex only and (iv) recording metadata only
    def loso_auroc(X):
        scores = []
        for split in splits:
            if split["kind"] != "cross_site":
                continue
            y_test = y_all[split["test"]]
            if len(set(y_test)) < 2:
                continue
            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(np.nan_to_num(X[split["train"]]), y_all[split["train"]])
                p = model.predict_proba(np.nan_to_num(X[split["test"]]))[:, 1]
            scores.append(float(roc_auc_score(y_test, p)))
        return float(np.mean(scores)) if scores else float("nan")

    demographics = np.column_stack([[p["age"] for p in people], [p["sex"] for p in people]])
    report = {
        "permutation_null": {
            "n_permutations": n_permutations,
            "mean": float(np.nanmean(permuted)),
            "sd": float(np.nanstd(permuted)),
            "max": float(np.nanmax(permuted)),
            "values": permuted,
            "note": "Labels permuted within site; a real signal must sit far above this.",
        },
        "site_identity_probe_macro_auroc": probes,
        "age_sex_only_cross_site_auroc": loso_auroc(demographics),
        "note": "Recording-metadata control is covered by cohort_v4.confound_audit: A1 makes session count constant, so it carries no information by construction.",
    }
    save(run / "negative-controls.json", report)
    return report


# ---------------------------------------------------------------------------
# Stage 11: final all-data fits, then the final lock
# ---------------------------------------------------------------------------


def final_fit(run=RUN, seed=SEED, family_specs=None, regime="I", tag="P1", montage=None):
    """Fit the pre-named pipeline on ALL eligible supervised people, with
    hyper-parameters chosen by the same D3 inner selection. This produces
    the model that scores the locked site; it carries no test claim of its
    own."""
    import pickle

    people = load_people(run)
    if montage:
        # A montage pipeline must be FIT on montage features, not merely
        # scored with them. Fitting on 17-channel data and labelling the
        # result "M4T" would be exactly the leak Stage 9 exists to prevent.
        from .reduced_v4 import load_montage_people

        people = load_montage_people(people, montage, run)
        if not people:
            raise ValueError(f"No montage features cached for {montage}")
    normative_pool = build_normative_pool(run)
    family_specs = family_specs or FAMILIES
    all_indices = list(range(len(people)))
    get_context = context_factory()
    y = np.array([p["y"] for p in people])

    selections, oof = {}, {}
    for family in family_specs:
        best, _ = select_family(
            people, all_indices, family, regime, normative_pool, seed, get_context, family_specs
        )
        selections[family] = best["params"]
        column = best["oof"]
        oof[family] = np.where(np.isnan(column), np.nanmean(column), column)

    context = get_context(people, people, regime, normative_pool, seed)
    fitted = {}
    for family, params in selections.items():
        spec = family_specs[family]
        model = spec["build"](params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(context.block(people, spec["block"]), y)
        fitted[family] = model

    stacker = NonNegativeStacker(C=1.0).fit(
        np.column_stack([oof[f] for f in family_specs]), y
    )
    stacked_oof = stacker.predict_proba(np.column_stack([oof[f] for f in family_specs]))[:, 1]

    artefact = {
        "tag": tag,
        "regime": regime,
        "families": list(family_specs),
        "selections": selections,
        "stacker_weights": dict(zip(family_specs, stacker.weights_.tolist())),
        "threshold": choose_threshold(y, stacked_oof),
        "training_prevalence": float(y.mean()),
        "n_training_people": len(people),
        "montage": montage,
    }
    folder = run / "final-models"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{tag}.pkl").open("wb") as handle:
        pickle.dump(
            {"fitted": fitted, "stacker": stacker, "context": context, "meta": artefact}, handle
        )
    save(folder / f"{tag}.json", artefact)
    return artefact


def score_with_final(tag, people, run=RUN):
    """Apply a frozen final pipeline to new people (the locked site)."""
    import pickle

    with (run / "final-models" / f"{tag}.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    meta, context, stacker = bundle["meta"], bundle["context"], bundle["stacker"]
    from .models_v4 import FAMILIES as ALL

    from .reduced_v4 import montage_families

    specs = ALL if meta["tag"] != "P3" else montage_families()
    # A locked site is re-centred on its own recordings under regime T.
    context.extend_transductive(people)
    columns = []
    for family in meta["families"]:
        spec = specs[family]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            columns.append(bundle["fitted"][family].predict_proba(context.block(people, spec["block"]))[:, 1])
    return stacker.predict_proba(np.column_stack(columns))[:, 1], meta
