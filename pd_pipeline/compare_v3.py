"""Nested site/participant evaluation. Outer outcomes never drive candidate selection."""

import json
import pickle
import time
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from .cohort_v3 import RUN
from .learning_v3 import (
    SEED,
    SpectralEncoder,
    load_people,
    flatten,
    predict,
    device,
    supervised_fit,
    seed,
)
from .registry import freeze, save, digest
from .preprocess import file_hash
from .experiments import metrics, choose_threshold

CONFIGS = [
    {"family": family, "C": c}
    for family in ["spectral_lr", "rich_lr", "rich_svm", "ssl_hybrid_lr", "ssl_hybrid_svm"]
    for c in [0.1, 1.0, 10.0]
] + [
    {"family": family, "lr": lr}
    for family in ["scratch_neural", "ssl_neural"]
    for lr in [0.0003, 0.001]
]
FAMILIES = sorted({c["family"] for c in CONFIGS})


def validate_disjoint(train, valid):
    a = {p["key"] for p in train}
    b = {p["key"] for p in valid}
    if a & b:
        raise ValueError("Participant leakage")


def folds(people):
    sites = sorted({p["site"] for p in people})
    out = []
    for site in sites:
        test = [i for i, p in enumerate(people) if p["site"] == site]
        train = [i for i, p in enumerate(people) if p["site"] != site]
        out.append({"name": "site_" + site, "kind": "cross_site", "train": train, "test": test})
    strata = [p["site"] + ":" + str(p["y"]) for p in people]
    for i, (tr, te) in enumerate(
        StratifiedKFold(3, shuffle=True, random_state=SEED).split(strata, strata)
    ):
        out.append(
            {
                "name": "within_" + str(i),
                "kind": "within_site",
                "train": tr.tolist(),
                "test": te.tolist(),
            }
        )
    return out


def summary_features(p):
    basic = []
    rich = []
    freqs = np.arange(2, 36)
    for r in p["records"]:
        t = r["x"]
        power = 10**t
        bp = np.stack(
            [
                np.log10(np.maximum(power[:, :, (freqs >= lo) & (freqs < hi)].sum(-1), 1e-12))
                for lo, hi in [(2, 4), (4, 8), (8, 13), (13, 30)]
            ],
            axis=-1,
        )
        basic.append(bp.mean(0).ravel())
        rich.append(np.r_[t.mean(0).ravel(), t.std(0).ravel(), r["cov"].mean(0)])
    return np.mean(basic, 0), np.mean(rich, 0)


def enrich(people, run):
    checkpoint = torch.load(run / "ssl.pt", map_location="cpu", weights_only=True)
    model = SpectralEncoder().to(device())
    model.load_state_dict(checkpoint["model"])
    for p in people:
        p["spectral"], p["rich"] = summary_features(p)
        x, m, owner, _ = flatten([p])
        _, emb = predict(
            model, x, m, owner, [p], checkpoint["mean"].numpy(), checkpoint["std"].numpy(), device()
        )
        p["hybrid"] = np.r_[p["rich"], emb[0]]
    return people


def model_features(people, family):
    key = (
        "spectral"
        if family == "spectral_lr"
        else "hybrid"
        if family.startswith("ssl_hybrid")
        else "rich"
    )
    return np.stack([p[key] for p in people])


def fit(train, valid, config, folder, run, epochs=None, seed_value=SEED):
    folder.mkdir(parents=True, exist_ok=True)
    validate_disjoint(train, valid)
    identity = digest(
        {
            "train": [p["key"] for p in train],
            "valid": [p["key"] for p in valid],
            "config": config,
            "epochs": epochs,
            "seed": seed_value,
            "ssl": file_hash(run / "ssl.pt"),
            "policy": file_hash(run / "implementation-policy.json"),
            "registry": file_hash(run / "registry.json"),
        }
    )
    meta = folder / "fit.json"
    if meta.exists():
        d = json.loads(meta.read_text())
        if d["identity"] != identity or file_hash(folder / d["weights"]) != d["sha256"]:
            raise ValueError("Cached fit integrity mismatch")
        return d
    start = time.monotonic()
    family = config["family"]
    if family.endswith("_neural"):
        model, mean, std, n = supervised_fit(
            train, valid, config, run, epochs=epochs, seed_value=seed_value
        )
        torch.save(
            {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std),
            },
            folder / "weights.pt",
        )
        weight = "weights.pt"
    else:
        estimator = (
            SVC(
                C=config["C"],
                kernel="rbf",
                gamma="scale",
                probability=True,
                random_state=seed_value,
            )
            if family.endswith("_svm")
            else LogisticRegression(C=config["C"], max_iter=3000)
        )
        model = make_pipeline(StandardScaler(), estimator)
        counts = Counter((p["site"], p["y"]) for p in train)
        weights = np.array([1 / counts[(p["site"], p["y"])] for p in train])
        weights /= weights.mean()
        model.fit(
            model_features(train, family),
            [p["y"] for p in train],
            **{model.steps[-1][0] + "__sample_weight": weights},
        )
        with (folder / "weights.pkl").open("wb") as f:
            pickle.dump(model, f)
        n = 0
        weight = "weights.pkl"
    d = {
        "identity": identity,
        "config": config,
        "epochs": n,
        "weights": weight,
        "sha256": file_hash(folder / weight),
        "seconds": time.monotonic() - start,
    }
    if valid:
        pp = infer(folder, valid, d)
        d["validation_predictions"] = pp.tolist()
        d["validation_auroc"] = float(roc_auc_score([p["y"] for p in valid], pp))
    freeze(meta, d)
    return d


def infer(folder, people, meta=None):
    d = meta or json.loads((folder / "fit.json").read_text())
    if file_hash(folder / d["weights"]) != d["sha256"]:
        raise ValueError("Model weights changed")
    if d["config"]["family"].endswith("_neural"):
        cp = torch.load(folder / d["weights"], map_location="cpu", weights_only=True)
        model = SpectralEncoder().to(device())
        model.load_state_dict(cp["model"])
        x, m, owner, _ = flatten(people)
        pp, _ = predict(model, x, m, owner, people, cp["mean"].numpy(), cp["std"].numpy(), device())
        return pp
    with (folder / d["weights"]).open("rb") as f:
        model = pickle.load(f)
    return model.predict_proba(model_features(people, d["config"]["family"]))[:, 1]


def benchmark(run=RUN):
    people = enrich(load_people(run), run)
    sites = sorted({p["site"] for p in people})
    train = [p for p in people if p["site"] != sites[-1]]
    valid = [p for p in people if p["site"] == sites[-1]]
    start = time.monotonic()
    seed(SEED)
    supervised_fit(train, [], {"family": "ssl_neural", "lr": 0.001}, run, epochs=2)
    elapsed = time.monotonic() - start
    # Four site outer folds and three within-site folds; four neural candidates,
    # inner leave-one-site-out; two final neural families x three seeds per outer.
    inner_fits = sum(len({people[i]["site"] for i in f["train"]}) * 4 for f in folds(people))
    final_fits = len(folds(people)) * 2 * 3
    # Plus final deployment selection and fit, classical fits, reporting margin.
    max_neural_epochs = (inner_fits + final_fits + 16 + 3) * 40
    forecast = elapsed / 2 * max_neural_epochs * len(people) / len(train) * 1.5
    d = {
        "device": str(device()),
        "two_epochs_seconds": elapsed,
        "forecast_seconds": forecast,
        "inner_neural_fits": inner_fits,
        "outer_final_neural_fits": final_fits,
        "maximum_epochs": 40,
        "participants": len(people),
        "valid_site_participants": len(valid),
        "margin": 1.5,
    }
    save(run / "supervised-benchmark.json", d)
    print(d, flush=True)


def select_and_fit(people, folder, run):
    if (folder / "lock.json").exists():
        return json.loads((folder / "lock.json").read_text())
    inner_sites = sorted({p["site"] for p in people})
    results = []
    for config in CONFIGS:
        key = digest(config)[:12]
        all_p = np.zeros(len(people))
        epochs = []
        scores = []
        for site in inner_sites:
            idx = [i for i, p in enumerate(people) if p["site"] == site]
            tr = [p for p in people if p["site"] != site]
            va = [people[i] for i in idx]
            d = fit(tr, va, config, folder / "tuning" / key / site, run)
            all_p[idx] = d["validation_predictions"]
            scores.append(d["validation_auroc"])
            epochs.append(d["epochs"])
        results.append(
            {
                "config": config,
                "macro_auroc": float(np.mean(scores)),
                "site_aurocs": scores,
                "epochs": int(np.median(epochs)),
                "oof": all_p.tolist(),
            }
        )
        save(
            folder / "selection-progress.json",
            {"completed_candidates": len(results), "total": len(CONFIGS)},
        )
        print(folder.name, config, round(results[-1]["macro_auroc"], 3), flush=True)
    locked = {}
    y = [p["y"] for p in people]
    for family in FAMILIES:
        best = max(
            [r for r in results if r["config"]["family"] == family], key=lambda r: r["macro_auroc"]
        )
        paths = []
        seeds = [SEED, SEED + 1, SEED + 2] if family.endswith("_neural") else [SEED]
        for s in seeds:
            path = folder / "final" / family / str(s)
            d = fit(
                people, [], best["config"], path, run, epochs=best["epochs"] or None, seed_value=s
            )
            paths.append({"path": str(path.relative_to(run)), "sha256": d["sha256"]})
        locked[family] = {
            "config": best["config"],
            "inner_macro_auroc": best["macro_auroc"],
            "threshold": choose_threshold(y, best["oof"]),
            "fits": paths,
            "epochs": best["epochs"],
        }
    selected = max(locked, key=lambda f: locked[f]["inner_macro_auroc"])
    lock = {
        "models": locked,
        "selected_family": selected,
        "participant_keys": [p["key"] for p in people],
        "selection_results": results,
        "ssl_sha256": file_hash(run / "ssl.pt"),
    }
    freeze(folder / "lock.json", lock)
    return lock


def campaign(run=RUN):
    b = json.loads((run / "supervised-benchmark.json").read_text())
    if b["forecast_seconds"] > 96 * 3600:
        raise RuntimeError("USER_NOTICE_REQUIRED: supervised campaign forecast exceeds 96 hours")
    people = enrich(load_people(run), run)
    splits = folds(people)
    freeze(
        run / "splits.json",
        {
            "people": [{"key": p["key"], "site": p["site"], "label": p["y"]} for p in people],
            "folds": splits,
        },
    )
    for split in splits:
        select_and_fit(
            [people[i] for i in split["train"]], run / "experiments" / split["name"], run
        )
    # Freeze every outer fit before any outer inference or score reporting.
    freeze(
        run / "outer-models-lock.json",
        {s["name"]: file_hash(run / "experiments" / s["name"] / "lock.json") for s in splits},
    )


def evaluate(run=RUN):
    if (run / "results.json").exists():
        return
    locks = json.loads((run / "outer-models-lock.json").read_text())
    people = enrich(load_people(run), run)
    splits = json.loads((run / "splits.json").read_text())["folds"]
    out = []
    for s in splits:
        folder = run / "experiments" / s["name"]
        if file_hash(folder / "lock.json") != locks[s["name"]]:
            raise ValueError("Outer lock changed")
        lock = json.loads((folder / "lock.json").read_text())
        test = [people[i] for i in s["test"]]
        if {p["key"] for p in test} & set(lock["participant_keys"]):
            raise ValueError("Outer leakage")
        result = {}
        for family, d in lock["models"].items():
            pred = []
            for artifact in d["fits"]:
                path = run / artifact["path"]
                meta = json.loads((path / "fit.json").read_text())
                if meta["sha256"] != artifact["sha256"]:
                    raise ValueError("Wrong frozen fit")
                pred.append(infer(path, test, meta))
            pp = np.mean(pred, axis=0)
            result[family] = {"p": pp.tolist(), "threshold": d["threshold"]}
        result["selected_pipeline"] = result[lock["selected_family"]]
        out.append(
            {
                "name": s["name"],
                "kind": s["kind"],
                "people": [{"key": p["key"], "site": p["site"], "y": p["y"]} for p in test],
                "selected_family": lock["selected_family"],
                "models": result,
            }
        )
    freeze(run / "outer-predictions.json", out)
    aggregate = {}
    for kind in ["cross_site", "within_site"]:
        by_site = {}
        for site in sorted({p["site"] for p in people}):
            group = [
                (r, j)
                for r in out
                if r["kind"] == kind
                for j, p in enumerate(r["people"])
                if p["site"] == site
            ]
            names = FAMILIES + ["selected_pipeline"]
            site_results = {}
            keys = [r["people"][j]["key"] for r, j in group]
            if len(keys) != len(set(keys)):
                raise ValueError("Duplicate out-of-fold participant")
            y = np.array([r["people"][j]["y"] for r, j in group])
            for family in names:
                pp = np.array([r["models"][family]["p"][j] for r, j in group])
                tt = np.array([r["models"][family]["threshold"] for r, j in group])
                m = metrics(y, pp, bootstrap=2000)
                # Different folds have different frozen thresholds. Report aggregate confusion honestly.
                classified = pp >= tt
                tn = int(((y == 0) & ~classified).sum())
                fp = int(((y == 0) & classified).sum())
                fn = int(((y == 1) & ~classified).sum())
                tp = int(((y == 1) & classified).sum())
                m.update(
                    threshold=None,
                    threshold_policy="Per-fold inner out-of-fold selection",
                    confusion_matrix=[[tn, fp], [fn, tp]],
                    sensitivity=tp / (tp + fn),
                    specificity=tn / (tn + fp),
                    balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2,
                )
                from .experiments import wilson

                m.update(
                    sensitivity_95ci_wilson=wilson(tp, tp + fn),
                    specificity_95ci_wilson=wilson(tn, tn + fp),
                )
                baseline = np.array([r["models"]["spectral_lr"]["p"][j] for r, j in group])
                rng = np.random.default_rng(SEED)
                diff = []
                for _ in range(2000):
                    ids = np.concatenate(
                        [
                            rng.choice(np.flatnonzero(y == label), sum(y == label), replace=True)
                            for label in [0, 1]
                        ]
                    )
                    diff.append(
                        roc_auc_score(y[ids], pp[ids]) - roc_auc_score(y[ids], baseline[ids])
                    )
                m["paired_auroc_gain_vs_spectral"] = {
                    "gain": float(roc_auc_score(y, pp) - roc_auc_score(y, baseline)),
                    "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
                    "interpretation": "Exploratory, unadjusted multiple comparisons",
                }
                site_results[family] = m
            by_site[site] = site_results
        aggregate[kind] = {
            "sites": by_site,
            "macro_site_auroc": {
                f: float(np.mean([d[f]["auroc"] for d in by_site.values()]))
                for f in FAMILIES + ["selected_pipeline"]
            },
        }
    freeze(run / "results.json", aggregate)


def final_fit(run=RUN):
    # A model for future research inference, not a new test result on training people.
    people = enrich(load_people(run), run)
    select_and_fit(people, run / "deployment", run)


if __name__ == "__main__":
    import sys

    {"benchmark": benchmark, "campaign": campaign, "evaluate": evaluate, "final": final_fit}[
        sys.argv[1]
    ]()
