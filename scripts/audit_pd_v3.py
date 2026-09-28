"""Post-lock integrity review and descriptive diagnostics; never changes tested models."""

import json
import platform
import importlib.metadata
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    brier_score_loss,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pd_pipeline.cohort_v3 import RUN, ARCHIVE
from pd_pipeline.registry import digest, save
from pd_pipeline.preprocess import file_hash, receipts
from pd_pipeline.compare_v3 import CONFIGS, FAMILIES
from pd_pipeline.learning_v3 import SEED
from scripts.run_pd_v3 import policy


def auc_boot(y, p, rng, count=2000):
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    indices = np.c_[rng.choice(neg, (count, len(neg))), rng.choice(pos, (count, len(pos)))]
    scores = p[indices]
    ranks = rankdata(scores, axis=1)
    return (ranks[:, len(neg) :].sum(1) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    r = RUN

    def read(name):
        return json.loads((r / name).read_text())

    reg = read("registry.json")
    prep = read("preparation.json")
    split = read("splits.json")
    pred = read("outer-predictions.json")
    results = read("results.json")
    assert policy() == read("implementation-policy.json")
    assert reg["fingerprint"] == digest({k: v for k, v in reg.items() if k != "fingerprint"})
    assert prep["registry"] == reg["fingerprint"]
    refs = {x["id"]: x for x in reg["records"]}
    meta = {x["id"]: x for x in prep["records"]}
    assert set(refs) == set(meta) and len(refs) == len(prep["records"]) == 11211
    verified = receipts(ARCHIVE)
    checked = set()
    cached = 0
    for record in reg["records"]:
        for source in record["sources"]:
            p = ARCHIVE / source["path"]
            v = verified[source["path"]]
            assert p.stat().st_size == v["bytes"] == source["bytes"]
            assert p.stat().st_mtime_ns == v["mtime_ns"]
            assert source["sha256"] == v["sha256"]
            checked.add(source["path"])
        m = meta[record["id"]]
        assert m["identity"] == digest([reg["fingerprint"], record["sources"]])
        if m["status"] == "ready":
            p = r / "cache" / (record["id"] + ".npz")
            assert file_hash(p) == m["sha256"]
            with np.load(p) as a:
                assert len(a["tokens"]) == m["windows"] == len(a["starts"])
            cached += 1
        else:
            assert m["status"] == "excluded" and m["reason"]
    people = split["people"]
    keys = [p["key"] for p in people]
    assert len(keys) == len(set(keys)) == 440
    by_key = defaultdict(list)
    for x in reg["records"]:
        by_key[x["person"]].append(x)
    retained = {k for k, rr in by_key.items() if sum(meta[x["id"]]["windows"] for x in rr) >= 10}
    assert set(keys) == {k for k in retained if by_key[k][0]["role"] == "supervised"}
    ssl = read("ssl-lock.json")
    assert file_hash(r / "ssl.pt") == ssl["sha256"]
    a = set(ssl["training_people"])
    b = set(ssl["validation_people"])
    assert not a & b and not (a | b) & set(keys)
    assert a | b == {k for k in retained if by_key[k][0]["role"] == "auxiliary"}
    assert not any(
        x["site"] in {"iowa", "tel_aviv", "unm", "ucsd"} for k in a | b for x in by_key[k]
    )
    policy_hash = file_hash(r / "implementation-policy.json")
    registry_hash = file_hash(r / "registry.json")
    fit_count = 0
    fit_seconds = 0.0
    neural = 0
    outerlocks = read("outer-models-lock.json")
    plans = split["folds"] + [
        {"name": "deployment", "train": list(range(len(people))), "test": [], "kind": "final"}
    ]
    for f in plans:
        folder = r / "deployment" if f["name"] == "deployment" else r / "experiments" / f["name"]
        lock = json.loads((folder / "lock.json").read_text())
        train = [people[i] for i in f["train"]]
        train_keys = [p["key"] for p in train]
        assert lock["participant_keys"] == train_keys
        assert not set(train_keys) & {keys[i] for i in f["test"]}
        if f["kind"] == "cross_site":
            assert not {p["site"] for p in train} & {people[i]["site"] for i in f["test"]}
        if f["kind"] != "final":
            assert file_hash(folder / "lock.json") == outerlocks[f["name"]]
        assert len(lock["selection_results"]) == len(CONFIGS) == 19
        for item in lock["selection_results"]:
            config = item["config"]
            oof = np.zeros(len(train))
            scores = []
            epochs = []
            for site in sorted({p["site"] for p in train}):
                tr = [p["key"] for p in train if p["site"] != site]
                va = [p["key"] for p in train if p["site"] == site]
                path = folder / "tuning" / digest(config)[:12] / site
                d = json.loads((path / "fit.json").read_text())
                identity = digest(
                    {
                        "train": tr,
                        "valid": va,
                        "config": config,
                        "epochs": None,
                        "seed": SEED,
                        "ssl": ssl["sha256"],
                        "policy": policy_hash,
                        "registry": registry_hash,
                    }
                )
                assert d["identity"] == identity and file_hash(path / d["weights"]) == d["sha256"]
                idx = [j for j, p in enumerate(train) if p["site"] == site]
                oof[idx] = d["validation_predictions"]
                score = roc_auc_score([train[j]["label"] for j in idx], d["validation_predictions"])
                assert np.isclose(score, d["validation_auroc"])
                scores.append(score)
                epochs.append(d["epochs"])
                fit_count += 1
                fit_seconds += d["seconds"]
                neural += config["family"].endswith("_neural")
            np.testing.assert_allclose(oof, item["oof"])
            assert (
                np.isclose(np.mean(scores), item["macro_auroc"])
                and int(np.median(epochs)) == item["epochs"]
            )
        for family, m in lock["models"].items():
            best = max(
                (x for x in lock["selection_results"] if x["config"]["family"] == family),
                key=lambda x: x["macro_auroc"],
            )
            assert best["config"] == m["config"] and np.isclose(
                best["macro_auroc"], m["inner_macro_auroc"]
            )
            assert len(m["fits"]) == (3 if family.endswith("_neural") else 1)
            for art in m["fits"]:
                path = r / art["path"]
                d = json.loads((path / "fit.json").read_text())
                seed = int(path.name)
                identity = digest(
                    {
                        "train": train_keys,
                        "valid": [],
                        "config": m["config"],
                        "epochs": m["epochs"] or None,
                        "seed": seed,
                        "ssl": ssl["sha256"],
                        "policy": policy_hash,
                        "registry": registry_hash,
                    }
                )
                assert (
                    d["identity"] == identity
                    and file_hash(path / d["weights"]) == art["sha256"] == d["sha256"]
                )
                fit_count += 1
                fit_seconds += d["seconds"]
                neural += family.endswith("_neural")
        assert lock["selected_family"] == max(
            lock["models"], key=lambda x: lock["models"][x]["inner_macro_auroc"]
        )
    assert fit_count == 620 == len(list(r.glob("**/fit.json")))
    assert (
        max(p.stat().st_mtime for p in (r / "experiments").glob("**/fit.json"))
        <= (r / "outer-models-lock.json").stat().st_mtime
    )
    assert (r / "outer-models-lock.json").stat().st_mtime <= (
        r / "outer-predictions.json"
    ).stat().st_mtime
    # Match thresholds and independently recompute every reported outer point estimate.
    arrays = {}
    macro = {}
    subgroups = {}
    names = FAMILIES + ["selected_pipeline"]
    for kind in ["cross_site", "within_site"]:
        allkeys = [p["key"] for fold in pred if fold["kind"] == kind for p in fold["people"]]
        assert len(allkeys) == len(set(allkeys)) == 440
        boot = defaultdict(list)
        subgroups[kind] = {}
        for site in sorted({p["site"] for p in people}):
            rows = [
                (fold, j)
                for fold in pred
                if fold["kind"] == kind
                for j, p in enumerate(fold["people"])
                if p["site"] == site
            ]
            y = np.array([fold["people"][j]["y"] for fold, j in rows])
            ps = {}
            for family in names:
                pp = np.array([fold["models"][family]["p"][j] for fold, j in rows])
                tt = np.array([fold["models"][family]["threshold"] for fold, j in rows])
                for fold, j in rows:
                    lock = read("experiments/" + fold["name"] + "/lock.json")
                    actual = lock["selected_family"] if family == "selected_pipeline" else family
                    assert (
                        fold["models"][family]["threshold"] == lock["models"][actual]["threshold"]
                    )
                    if family == "selected_pipeline":
                        assert fold["models"][family] == fold["models"][actual]
                m = results[kind]["sites"][site][family]
                assert np.isclose(roc_auc_score(y, pp), m["auroc"])
                assert np.isclose(average_precision_score(y, pp), m["pr_auc_average_precision"])
                assert np.isclose(brier_score_loss(y, pp), m["brier"])
                cm = confusion_matrix(y, pp >= tt, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                assert np.isclose(tp / (tp + fn), m["sensitivity"])
                assert np.isclose(tn / (tn + fp), m["specificity"])
                assert np.isclose((tp / (tp + fn) + tn / (tn + fp)) / 2, m["balanced_accuracy"])
                assert (
                    confusion_matrix(y, pp >= tt, labels=[0, 1]).tolist() == m["confusion_matrix"]
                )
                ps[family] = pp
                boot[family].append(
                    auc_boot(y, pp, np.random.default_rng(SEED + int(digest(site)[:8], 16)))
                )
            arrays[kind, site] = (rows, y, ps)
            subgroups[kind][site] = {}
            for label, test in [
                ("female", lambda d: str(d["sex"]).lower() in ["f", "female"]),
                ("male", lambda d: str(d["sex"]).lower() in ["m", "male"]),
                ("age_below_65", lambda d: float(d["age"]) < 65),
                ("age_65_plus", lambda d: float(d["age"]) >= 65),
            ]:
                indices = []
                for j, (fold, k) in enumerate(rows):
                    record = by_key[fold["people"][k]["key"]][0]
                    try:
                        if test(record):
                            indices.append(j)
                    except (TypeError, ValueError):
                        pass
                yy = y[indices]
                counts = {str(v): int(sum(yy == v)) for v in [0, 1]}
                sg = {"counts": counts}
                if min(counts.values()) >= 5:
                    sg["auroc"] = {f: float(roc_auc_score(yy, pp[indices])) for f, pp in ps.items()}
                else:
                    sg["not_estimated"] = "Fewer than five retained people in either class"
                subgroups[kind][site][label] = sg
        mb = {f: np.mean(boot[f], axis=0) for f in names}
        macro[kind] = {
            "models": {
                f: {
                    "auroc": results[kind]["macro_site_auroc"][f],
                    "ci95": np.quantile(mb[f], [0.025, 0.975]).tolist(),
                }
                for f in names
            },
            "paired": {},
        }
        for f, g in [
            ("rich_svm", "spectral_lr"),
            ("ssl_hybrid_svm", "rich_svm"),
            ("ssl_hybrid_lr", "rich_lr"),
            ("ssl_neural", "scratch_neural"),
        ]:
            macro[kind]["paired"][f + "_minus_" + g] = {
                "difference": results[kind]["macro_site_auroc"][f]
                - results[kind]["macro_site_auroc"][g],
                "ci95": np.quantile(mb[f] - mb[g], [0.025, 0.975]).tolist(),
            }
    # Post-hoc confounding checks: fixed C=1; no selection or change to tested EEG models.
    confound = {}
    features = {"demographic": [], "recording_quality": []}
    profiles = defaultdict(dict)
    for p in people:
        rr = by_key[p["key"]]
        q = [meta[x["id"]] for x in rr]
        record = rr[0]
        try:
            age = float(record["age"])
        except (TypeError, ValueError):
            age = np.nan
        sex = {"m": 1, "male": 1, "f": 0, "female": 0}.get(str(record["sex"]).lower(), np.nan)
        features["demographic"].append([age, sex])
        accepted = sum(x["windows"] for x in q)
        attempted = sum(x.get("attempted", 0) for x in q)
        features["recording_quality"].append(
            [np.log1p(accepted), len(rr), accepted / max(attempted, 1)]
        )
    private_confound = {}
    for name, xx in features.items():
        xx = np.asarray(xx)
        private_confound[name] = {}
        for kind in ["cross_site", "within_site"]:
            pp = np.zeros(len(people))
            for f in split["folds"]:
                if f["kind"] != kind:
                    continue
                tr = f["train"]
                te = f["test"]
                model = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                    LogisticRegression(C=1, max_iter=2000, class_weight="balanced"),
                )
                model.fit(xx[tr], [people[i]["label"] for i in tr])
                pp[te] = model.predict_proba(xx[te])[:, 1]
            private_confound[name][kind] = pp.tolist()
            confound.setdefault(kind, {})[name] = {
                s: float(
                    roc_auc_score(
                        [p["label"] for p in people if p["site"] == s],
                        pp[[i for i, p in enumerate(people) if p["site"] == s]],
                    )
                )
                for s in sorted({p["site"] for p in people})
            }
    for site in sorted({p["site"] for p in people}):
        for label in [0, 1]:
            indices = [i for i, p in enumerate(people) if p["site"] == site and p["label"] == label]
            age = np.array(features["demographic"])[indices, 0]
            sex = np.array(features["demographic"])[indices, 1]
            assigned = sum(
                rr[0]["site"] == site and rr[0]["label"] == label for rr in by_key.values()
            )
            profiles[site][str(label)] = {
                "assigned": assigned,
                "retained": len(indices),
                "excluded": assigned - len(indices),
                "age_mean": float(np.nanmean(age)),
                "age_sd": float(np.nanstd(age, ddof=1)),
                "female": int(sum(sex == 0)),
                "male": int(sum(sex == 1)),
                "age_missing": int(np.isnan(age).sum()),
                "sex_missing": int(np.isnan(sex).sum()),
            }
    exclusions = defaultdict(Counter)
    for x in prep["records"]:
        if x["status"] == "excluded":
            exclusions[refs[x["id"]]["dataset"]].update(x.get("rejections", {}))
    audit = {
        "status": "passed",
        "planned_recordings": 11211,
        "cache_hashes_verified": cached,
        "source_receipts_and_stat_verified": len(checked),
        "source_method": "Sources read for preprocessing were rehashed then; final review reconciles all candidate source receipt SHA256, byte count and mtime, not a second full source rehash.",
        "verified_fits": fit_count,
        "neural_fits": neural,
        "classical_fits": fit_count - neural,
        "sum_fit_seconds": fit_seconds,
        "pd_people": 440,
        "aux_train_groups": len(a),
        "aux_validation_groups": len(b),
        "ssl_windows": read("ssl-benchmark.json"),
        "implementation_policy_matches": True,
        "all_point_estimates_reproduced": True,
        "outer_participant_and_site_separation": True,
        "auxiliary_separation": True,
        "software": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "scikit-learn", "torch", "mne"]
        },
        "python": platform.python_version(),
    }
    aggregate = {
        "audit": audit,
        "macro_uncertainty": macro,
        "demographic_profiles": dict(profiles),
        "subgroups": subgroups,
        "posthoc_confound_auroc": confound,
        "excluded_recording_window_reasons": dict(exclusions),
        "diagnostic_scope": "Post-lock descriptive analyses only. No model changes or retuning. Bootstrap conditions on these four sites and excludes training/selection uncertainty. Comparisons and subgroups are unadjusted and exploratory.",
    }
    save(r / "review-audit.json", aggregate)
    save(r / "posthoc-confound-predictions.json", private_confound)
    save(Path("docs/pd-v3-review.json"), aggregate)
    cards = {
        f: {
            "intended_use": "Exploratory PD-versus-control EEG research, not diagnosis or symptom severity",
            "test_scope": "Nested participant/site cross-validation on four previously examined sites",
            "cross_site": macro["cross_site"]["models"][f],
            "within_site": macro["within_site"]["models"][f],
            "limitations": [
                "Previously examined sites",
                "Engineering QC selection",
                "Multiple-session aggregation",
                "Age/sex and acquisition confounding",
                "No prospective clinical validation",
            ],
            "inputs": "Named scalp EEG electrodes; fixed spectral features; all accepted compatible resting duration; metadata not used in primary EEG model",
        }
        for f in names
    }
    final = read("deployment/lock.json")
    for f, card in cards.items():
        card["protocol"] = "PD_PROTOCOL_V3"
        card["retained_pd_control_participants"] = 440
        card["primary_selection_criterion"] = (
            "Equal-site mean AUROC in inner training-site validation"
        )
        card["direct_metadata_inputs"] = False
        card["recording_design_confound"] = (
            "ON/OFF repeated PD sessions versus one control visit at UCSD/UNM"
        )
        if f in final["models"]:
            card["final_configuration"] = final["models"][f]["config"]
            card["final_neural_epochs"] = final["models"][f]["epochs"] or None
            card["final_fit_seeds"] = (
                [SEED, SEED + 1, SEED + 2] if f.endswith("_neural") else [SEED]
            )
            card["chosen_by_final_all_data_selection"] = f == final["selected_family"]
        else:
            card["meaning"] = (
                "Inner-selected family may differ by outer fold; this evaluates the selection procedure"
            )
    save(Path("docs/pd-v3-model-cards.json"), cards)
    print(json.dumps({"audit": audit, "macro": macro}, indent=2))


if __name__ == "__main__":
    main()
