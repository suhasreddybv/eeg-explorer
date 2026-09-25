"""Participant-level fitting, metrics and measured local runtime gates."""

import json
import hashlib
import math
import platform
import time
from pathlib import Path

import numpy as np
import psutil
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .model import EEGNet, device
from .preprocess import file_hash
from .registry import digest, save, validate

MAX_STAGE_SECONDS = 96 * 3600


def runtime_gate(seconds):
    if not np.isfinite(seconds) or seconds < 0 or seconds > MAX_STAGE_SECONDS:
        raise ValueError(
            "Projected continuous runtime exceeds four days or is unknown. Notify the user before proceeding."
        )


def load_data(run, allow_partial=False):
    run = Path(run)
    registry = json.loads((run / "registry.json").read_text())
    validate(registry)
    report = json.loads((run / "preparation.json").read_text())
    config = json.loads((run / "preprocessing.json").read_text())
    if report.get("preprocessing_fingerprint") != digest(config):
        raise ValueError("Preprocessing fingerprint mismatch")
    if report["registry_fingerprint"] != registry["fingerprint"]:
        raise ValueError("Preparation belongs to another registry")
    pending = [r for r in report["records"] if r["status"] not in {"ready", "excluded_qc"}]
    expected = {
        r["recording_id"] for r in registry["records"] if r["split"] in {"train", "validation"}
    }
    if not allow_partial and (
        pending or {r["recording_id"] for r in report["records"]} != expected
    ):
        raise ValueError(
            "Downloads/preparation incomplete; partial data is allowed only for benchmarks"
        )
    by_id = {r["recording_id"]: r for r in registry["records"]}
    data, seen_hashes = [], {}
    for row in report["records"]:
        if row["status"] != "ready":
            continue
        reference = by_id[row["recording_id"]]
        if row.get("preprocessing_fingerprint") != digest(config):
            raise ValueError("Cached preprocessing fingerprint mismatch")
        if any(
            row[k] != reference[k] for k in ["participant_key", "split", "label", "dataset", "path"]
        ):
            raise ValueError("Cached identity differs from frozen registry")
        if row["split"] not in {"train", "validation"}:
            raise ValueError("Sealed test data encountered")
        path = run / "cache" / (row["recording_id"] + ".npz")
        if file_hash(path) != row["cache_sha256"]:
            raise ValueError("Cache fingerprint mismatch")
        with np.load(path, allow_pickle=False) as f:
            x = f["windows"]
        key = hashlib.sha256(x.tobytes()).hexdigest()
        if key in seen_hashes and seen_hashes[key] != row["participant_key"]:
            raise ValueError("Duplicate processed signal across participants")
        seen_hashes[key] = row["participant_key"]
        data.append({**row, "x": x})
    keys = [r["participant_key"] for r in data]
    if len(keys) != len(set(keys)):
        raise ValueError("Multiple sessions: freeze a primary session rule before fitting")
    return data, registry, report


def fit_scaler(records):
    if not records or any(r["split"] != "train" for r in records):
        raise ValueError("Scaling can only fit training participants")
    # Median of per-person robust channel scales gives each person equal weight.
    scales = []
    for r in records:
        x = r["x"].transpose(1, 0, 2).reshape(r["x"].shape[1], -1)
        scales.append(np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1) * 1.4826)
    return np.maximum(np.median(scales, axis=0), 1).astype("float32")


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(max(0, center - half)), float(min(1, center + half))]


def metrics(y, p, threshold=0.5, bootstrap=500, seed=42):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) != 2:
        raise ValueError("Both participant classes are required")
    prediction = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(bootstrap):
        ids = np.concatenate(
            [
                rng.choice(np.flatnonzero(y == label), size=(y == label).sum(), replace=True)
                for label in [0, 1]
            ]
        )
        aucs.append(roc_auc_score(y[ids], p[ids]))
    return {
        "participants": len(y),
        "pd_prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auroc_95ci": np.quantile(aucs, [0.025, 0.975]).tolist() if aucs else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "sensitivity_95ci_wilson": wilson(tp, tp + fn),
        "specificity_95ci_wilson": wilson(tn, tn + fp),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
    }


def choose_threshold(y, p):
    candidates = np.unique(np.r_[0, p, 1])
    return float(max(candidates, key=lambda t: (balanced_accuracy_score(y, p >= t), -abs(t - 0.5))))


def spectral_features(x):
    f, psd = signal.welch(x, fs=256, nperseg=512, axis=-1)
    bands = [(1, 4), (4, 8), (8, 13), (13, 30)]
    return (
        np.stack(
            [
                np.log10(
                    np.maximum(
                        np.trapezoid(
                            psd[..., (f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=-1
                        ),
                        1e-12,
                    )
                )
                for lo, hi in bands
            ],
            axis=-1,
        )
        .mean(axis=0)
        .ravel()
    )


def baseline(run):
    data, registry, report = load_data(run)
    train = [r for r in data if r["split"] == "train"]
    val = [r for r in data if r["split"] == "validation"]
    x, y = (
        np.stack([spectral_features(r["x"]) for r in train]),
        np.array([r["label"] for r in train]),
    )
    xv, yv = np.stack([spectral_features(r["x"]) for r in val]), np.array([r["label"] for r in val])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=registry["seed"]).split(x, y))
    scores = []
    for c in [0.01, 0.1, 1, 10]:
        auc = []
        for a, b in folds:
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
            )
            model.fit(x[a], y[a])
            auc.append(roc_auc_score(y[b], model.predict_proba(x[b])[:, 1]))
        scores.append({"C": c, "training_fold_auroc": float(np.mean(auc))})
    best = max(scores, key=lambda s: s["training_fold_auroc"])["C"]
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=best, class_weight="balanced", max_iter=2000)
    ).fit(x, y)
    p = model.predict_proba(xv)[:, 1]
    threshold = choose_threshold(yv, p)
    result = {
        "kind": "development_only_not_test_results",
        "registry_fingerprint": registry["fingerprint"],
        "training_participants": len(y),
        "configuration_search": scores,
        "validation": metrics(yv, p, threshold),
        "validation_prevalence_baseline": metrics(yv, np.full(len(yv), y.mean())),
        "warning": "Threshold selected on validation; validation classification scores are optimistic. Tests remain sealed.",
        "participants": [
            {
                "participant_key": r["participant_key"],
                "label": r["label"],
                "probability": float(prob),
            }
            for r, prob in zip(val, p)
        ],
        "coverage": report["counts"],
    }
    save(Path(run) / "baseline.json", result)
    scale, logistic = model.steps[0][1], model.steps[1][1]
    np.savez(
        Path(run) / "baseline-weights.npz",
        mean=scale.mean_,
        scale=scale.scale_,
        coef=logistic.coef_,
        intercept=logistic.intercept_,
        threshold=threshold,
    )
    return result


def loader(records, scale, training, seed, batch=32):
    x = np.concatenate([r["x"] / scale[None, :, None] for r in records])
    y = np.concatenate([np.full(len(r["x"]), r["label"], dtype="float32") for r in records])
    if training:
        class_counts = {k: sum(r["label"] == k for r in records) for k in [0, 1]}
        if min(class_counts.values()) < 2:
            raise ValueError("At least two training participants per class required")
        weights = np.concatenate(
            [np.full(len(r["x"]), 1 / (class_counts[r["label"]] * len(r["x"]))) for r in records]
        )
        sampler = WeightedRandomSampler(
            torch.tensor(weights),
            len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        sampler = None
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch,
        sampler=sampler,
        num_workers=0,
    )


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def epoch(model, batches, optimizer, dev, deadline=None):
    model.train()
    total, count = 0, 0
    for x, y in batches:
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError("Four-day watchdog stopped training")
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        model.constrain()
        total += float(loss.detach().cpu()) * len(y)
        count += len(y)
    sync(dev)
    return total / count


@torch.no_grad()
def predict(model, records, scale, dev):
    model.eval()
    result = []
    for r in records:
        x = torch.from_numpy(r["x"] / scale[None, :, None]).to(dev)
        result.append(float(model(x).sigmoid().mean().cpu()))
    return np.array(result)


def benchmark(run, device_name="auto", epochs=5):
    if not 2 <= epochs <= 10:
        raise ValueError("Benchmark must be 2–10 epochs")
    data, registry, _ = load_data(run, allow_partial=True)
    train = [r for r in data if r["split"] == "train"]
    # Benchmark only training participants, even when validation caches exist.
    counts = [sum(r["label"] == k for r in train) for k in [0, 1]]
    if min(counts) < 2:
        raise ValueError("Wait for at least two ready training participants per class")
    torch.set_num_threads(4)
    torch.manual_seed(registry["seed"])
    dev = device(device_name)
    scale = fit_scaler(train)
    model = EEGNet(len(scale)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    batches = loader(train, scale, True, registry["seed"])
    epoch(model, batches, opt, dev)  # warmup excluded
    elapsed = []
    for _ in range(epochs):
        start = time.monotonic()
        epoch(model, batches, opt, dev)
        # Include a training-only inference pass to approximate validation overhead.
        predict(model, train[: max(1, len(train) // 5)], scale, dev)
        sync(dev)
        elapsed.append(time.monotonic() - start)
    total_train = len({r["participant_key"] for r in registry["records"] if r["split"] == "train"})
    projected = float(np.median(elapsed)) * total_train / len(train) * 100 * 21 * 1.25
    result = {
        "device": str(dev),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "registry_fingerprint": registry["fingerprint"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_participants_measured": len(train),
        "windows_measured": sum(len(r["x"]) for r in train),
        "epoch_seconds": elapsed,
        "median_epoch_seconds": float(np.median(elapsed)),
        "projected_21_fits_100_epochs_seconds_with_25pct_buffer": projected,
        "runtime_gate_passed": projected <= MAX_STAGE_SECONDS,
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "note": "Measured real training windows; linear extrapolation, not a full experiment benchmark. No model saved; no holdout used.",
    }
    save(Path(run) / "benchmark.json", result)
    return result


def train(run, device_name="auto", max_epochs=100, seed=20260924):
    if not 1 <= max_epochs <= 100:
        raise ValueError("Epoch budget must be 1–100")
    run = Path(run)
    bench = json.loads((run / "benchmark.json").read_text())
    dev = device(device_name)
    if bench["device"] != str(dev):
        raise ValueError("Benchmark the chosen device before training")
    runtime_gate(bench["projected_21_fits_100_epochs_seconds_with_25pct_buffer"])
    data, registry, _ = load_data(run)
    if bench["registry_fingerprint"] != registry["fingerprint"]:
        raise ValueError("Benchmark registry mismatch")
    output = run / f"neural-seed-{seed}"
    output.mkdir(exist_ok=False)
    tr = [r for r in data if r["split"] == "train"]
    val = [r for r in data if r["split"] == "validation"]
    yv = np.array([r["label"] for r in val])
    if min(sum(yv == k) for k in [0, 1]) < 2:
        raise ValueError("Insufficient validation participants")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    scale = fit_scaler(tr)
    selection = json.loads((run / "tuning.json").read_text())
    if (
        selection.get("state") != "complete"
        or selection["registry_fingerprint"] != registry["fingerprint"]
    ):
        raise ValueError("Complete training-only tuning before final seed fits")
    config = selection["selected"]
    model = EEGNet(len(scale), dropout=config["dropout"]).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0001)
    batches = loader(tr, scale, True, seed)
    history, best, best_epoch = [], -math.inf, 0
    started = time.monotonic()
    for step in range(max_epochs):
        if time.monotonic() - started > MAX_STAGE_SECONDS - 60:
            raise RuntimeError("Four-day watchdog stopped training; latest checkpoint preserved")
        loss = epoch(model, batches, optimizer, dev, started + MAX_STAGE_SECONDS - 60)
        probability = predict(model, val, scale, dev)
        auc = float(roc_auc_score(yv, probability))
        history.append(
            {
                "epoch": step + 1,
                "loss": loss,
                "validation_auroc": auc,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        checkpoint = {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "epoch": step + 1,
            "registry_fingerprint": registry["fingerprint"],
            "scale": torch.from_numpy(scale),
            "seed": seed,
        }
        torch.save(checkpoint, output / "latest.pt")
        if auc > best:
            best, best_epoch = auc, step + 1
            torch.save(checkpoint, output / "best.pt")
        save(output / "history.json", history)
        print(f"Epoch {step + 1}: loss {loss:.4f}, validation AUROC {auc:.3f}", flush=True)
        if step + 1 - best_epoch >= 15:
            break
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    p = predict(model, val, scale, dev)
    threshold = choose_threshold(yv, p)
    result = {
        "kind": "development_only_tuned_configuration",
        "configuration": config,
        "registry_fingerprint": registry["fingerprint"],
        "device": str(dev),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs": len(history),
        "elapsed_seconds": time.monotonic() - started,
        "validation": metrics(yv, p, threshold),
        "participants": [
            {
                "participant_key": r["participant_key"],
                "label": r["label"],
                "probability": float(prob),
            }
            for r, prob in zip(val, p)
        ],
        "warning": "Validation used for early stopping/threshold; no unbiased test claim.",
    }
    save(output / "result.json", result)
    return result


def tune(run, device_name="auto", max_epochs=100):
    """Six preregistered configurations; participant CV entirely inside training."""
    if not 1 <= max_epochs <= 100:
        raise ValueError("Epoch budget must be 1–100")
    run = Path(run)
    bench = json.loads((run / "benchmark.json").read_text())
    dev = device(device_name)
    runtime_gate(bench["projected_21_fits_100_epochs_seconds_with_25pct_buffer"])
    data, registry, _ = load_data(run)
    if bench["device"] != str(dev) or bench["registry_fingerprint"] != registry["fingerprint"]:
        raise ValueError("Benchmark this registry and device first")
    tr = [r for r in data if r["split"] == "train"]
    labels = np.array([r["label"] for r in tr])
    if min(sum(labels == k) for k in [0, 1]) < 6:
        raise ValueError("Too few training participants for three-fold tuning")
    folds = list(
        StratifiedKFold(3, shuffle=True, random_state=registry["seed"]).split(labels, labels)
    )
    configs = [{"lr": lr, "dropout": drop} for lr in [0.0003, 0.001, 0.003] for drop in [0.25, 0.5]]
    result = {
        "state": "running",
        "registry_fingerprint": registry["fingerprint"],
        "device": str(dev),
        "configurations": configs,
        "fold_results": [],
        "max_epochs": max_epochs,
        "protocol": "3 participant folds, training split only; early stop patience 15; score each fold best AUROC",
    }
    destination = run / "tuning.json"
    if destination.exists():
        raise ValueError("Tuning output exists; preserve prior experiments and use a new run")
    started = time.monotonic()
    torch.set_num_threads(4)
    save(destination, result)
    for ci, cfg in enumerate(configs):
        for fi, (a, b) in enumerate(folds):
            torch.manual_seed(registry["seed"] + fi)
            training, validation = [tr[i] for i in a], [tr[i] for i in b]
            scale = fit_scaler(training)
            model = EEGNet(len(scale), dropout=cfg["dropout"]).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0001)
            batches = loader(training, scale, True, registry["seed"] + fi)
            best, best_epoch = -math.inf, 0
            for ep in range(max_epochs):
                epoch(model, batches, opt, dev, started + MAX_STAGE_SECONDS - 60)
                probability = predict(model, validation, scale, dev)
                auc = float(roc_auc_score(labels[b], probability))
                if auc > best:
                    best, best_epoch = auc, ep + 1
                if ep + 1 - best_epoch >= 15:
                    break
            result["fold_results"].append(
                {
                    "configuration_index": ci,
                    "fold": fi,
                    "best_auroc": best,
                    "best_epoch": best_epoch,
                    "epochs": ep + 1,
                    "training_people": [r["participant_key"] for r in training],
                    "validation_people": [r["participant_key"] for r in validation],
                }
            )
            result["elapsed_seconds"] = time.monotonic() - started
            save(destination, result)
            print(
                f"Config {ci + 1}/6 fold {fi + 1}/3: AUROC {best:.3f}, epochs {ep + 1}", flush=True
            )
    averages = [
        np.mean([r["best_auroc"] for r in result["fold_results"] if r["configuration_index"] == i])
        for i in range(6)
    ]
    result.update(
        state="complete", mean_fold_auroc=averages, selected=configs[int(np.argmax(averages))]
    )
    save(destination, result)
    return result


def campaign(run, device_name="auto", max_epochs=100):
    """Train-only search, three fixed seeds, frozen validation-selected ensemble."""
    from .registry import freeze

    run = Path(run)
    baseline(run)
    selection = tune(run, device_name, max_epochs)
    seeds = [20260924, 20260925, 20260926]
    fits = [train(run, device_name, max_epochs, seed) for seed in seeds]
    people = [r["participant_key"] for r in fits[0]["participants"]]
    for fit in fits:
        if [r["participant_key"] for r in fit["participants"]] != people:
            raise ValueError("Ensemble participant alignment mismatch")
    y = np.array([r["label"] for r in fits[0]["participants"]])
    p = np.mean([[r["probability"] for r in fit["participants"]] for fit in fits], axis=0)
    threshold = choose_threshold(y, p)
    baseline_file = run / "baseline.json"
    lock = {
        "state": "model_locked_tests_not_evaluated",
        "evaluation_policy_fingerprint": digest(
            json.loads((run / "evaluation-policy.json").read_text())
        ),
        "registry_fingerprint": selection["registry_fingerprint"],
        "preprocessing_fingerprint": digest(json.loads((run / "preprocessing.json").read_text())),
        "seeds": seeds,
        "configuration": selection["selected"],
        "threshold": threshold,
        "validation": metrics(y, p, threshold),
        "model_hashes": {
            f"neural-seed-{seed}/best.pt": file_hash(run / f"neural-seed-{seed}/best.pt")
            for seed in seeds
        },
        "baseline_sha256": file_hash(run / "baseline-weights.npz"),
        "baseline_report_sha256": file_hash(baseline_file),
        "warning": "Validation selected epochs and threshold; these are not test performance estimates.",
    }
    freeze(run / "model-lock.json", lock)
    return lock


def audit(run):
    """Development-only demographic baseline and explicit QC attrition audit."""
    from collections import Counter
    from sklearn.impute import SimpleImputer

    data, registry, report = load_data(run)
    identity = {r["participant_key"]: r for r in registry["records"]}
    train = [r for r in data if r["split"] == "train"]
    val = [r for r in data if r["split"] == "validation"]

    def features(rows):
        result = []
        for row in rows:
            meta = identity[row["participant_key"]]
            try:
                age = float(meta["age"])
            except (ValueError, TypeError):
                age = np.nan
            sex = {"M": 0.0, "F": 1.0, "MALE": 0.0, "FEMALE": 1.0}.get(
                str(meta["sex"]).upper(), np.nan
            )
            result.append([age, sex])
        return np.array(result)

    x, xv = features(train), features(val)
    y = np.array([r["label"] for r in train])
    yv = np.array([r["label"] for r in val])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=registry["seed"]).split(x, y))
    scores = []

    def estimator(c):
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=2000),
        )

    for c in [0.01, 0.1, 1, 10]:
        values = []
        for a, b in folds:
            model = estimator(c).fit(x[a], y[a])
            values.append(roc_auc_score(y[b], model.predict_proba(x[b])[:, 1]))
        scores.append({"C": c, "mean_training_cv_auroc": float(np.mean(values))})
    c = max(scores, key=lambda r: r["mean_training_cv_auroc"])["C"]
    model = estimator(c).fit(x, y)
    p = model.predict_proba(xv)[:, 1]
    result = {
        "scope": "Training/validation only; does not resolve all acquisition or artifact confounding",
        "registry_fingerprint": registry["fingerprint"],
        "demographic_cv": scores,
        "demographic_validation": metrics(yv, p, choose_threshold(yv, p)),
        "qc_counts": dict(
            Counter(f"{r['split']}:{r['label']}:{r['status']}" for r in report["records"])
        ),
        "age_summary_training_eligible": {
            str(label): {"n": int(sum(y == label)), "mean": float(np.nanmean(x[y == label, 0]))}
            for label in [0, 1]
        },
        "limitations": [
            "QC has not been reviewed by an EEG specialist.",
            "Different exclusion rates can create selection bias.",
            "Demographic validation has only 15 eligible participants in this pilot; a low score does not rule out confounding.",
        ],
    }
    save(Path(run) / "development-audit.json", result)
    return result
