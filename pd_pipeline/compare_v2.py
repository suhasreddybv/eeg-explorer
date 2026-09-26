"""Resumable, participant-separated model comparison with sealed test evaluation."""

import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from .cohort_v2 import load, CONFIG, SEED
from .models_v2 import FAMILIES, make, represent, periodic_features
from .experiments import (
    metrics,
    choose_threshold,
    spectral_features,
    fit_scaler,
    runtime_gate,
    sync,
)
from .model import device
from .registry import digest, freeze, save
from .preprocess import file_hash

SEARCH = [{"lr": lr, "dropout": drop} for lr in [0.0003, 0.001, 0.003] for drop in [0.25, 0.5]]
SEEDS = [20260926, 20260927, 20260928]
MAX_EPOCHS = 100
PATIENCE = 15


def setup(run):
    reg = json.loads((run / "registry.json").read_text())
    policy = {
        "version": 2,
        "registry": reg["fingerprint"],
        "preprocessing": digest(CONFIG),
        "neural_families": FAMILIES,
        "search": SEARCH,
        "seeds": SEEDS,
        "folds": 3,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "primary_external": "tel_aviv",
        "secondary_external": ["unm_on", "unm_off"],
        "internal": "Exploratory; cross-release overlap with pilot unresolved",
        "threshold": "validation maximum balanced accuracy, ties closest to 0.5",
        "aggregation": "mean window probability then equal-weight three-seed probability ensemble",
        "bootstrap": 500,
        "multiple_comparisons": "exploratory paired bootstrap; simultaneous Bonferroni intervals versus spectral baseline",
        "implementation_hashes": {
            n: file_hash(Path(__file__).parent / n)
            for n in ["models_v2.py", "cohort_v2.py", "compare_v2.py"]
        },
    }
    freeze(run / "comparison-policy.json", policy)
    return policy


def prepared(data, name):
    return [{**r, "x": represent(r["x"], name)} for r in data]


def scaling(train, name):
    if any(r["split"] != "train" for r in train):
        raise ValueError("Scaler leakage")
    if name not in ["wavelet_cnn", "transformer"]:
        return np.zeros((1, len(CONFIG["channels"]), 1), dtype="float32"), fit_scaler(train)[
            None, :, None
        ]
    axes = (0, 3) if name == "wavelet_cnn" else (0,)
    means = np.stack([r["x"].mean(axes, keepdims=True) for r in train])
    second = np.stack([(r["x"] ** 2).mean(axes, keepdims=True) for r in train])
    mean = means.mean(0)
    std = np.sqrt(np.maximum(second.mean(0) - mean**2, 1e-4))
    return mean.astype("float32"), std.astype("float32")


def batches(train, mean, std, seed):
    x = np.concatenate([(r["x"] - mean) / std for r in train])
    y = np.concatenate([np.full(len(r["x"]), r["label"], dtype="float32") for r in train])
    counts = {i: sum(r["label"] == i for r in train) for i in [0, 1]}
    weights = np.concatenate(
        [np.full(len(r["x"]), 1 / (counts[r["label"]] * len(r["x"]))) for r in train]
    )
    sampler = WeightedRandomSampler(
        torch.tensor(weights), len(weights), generator=torch.Generator().manual_seed(seed)
    )
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=32, sampler=sampler
    )


@torch.no_grad()
def predict(model, data, mean, std, dev):
    model.eval()
    out = []
    for r in data:
        x = torch.from_numpy((r["x"] - mean) / std).to(dev)
        out.append(float(model(x).sigmoid().mean().cpu()))
    return np.asarray(out)


def one_epoch(model, b, optimizer, dev, deadline):
    model.train()
    losses = []
    for x, y in b:
        if time.monotonic() > deadline:
            raise RuntimeError("96-hour stage watchdog")
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x.to(dev)), y.to(dev))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        model.constrain()
        losses.append(float(loss.detach().cpu()))
    sync(dev)
    return float(np.mean(losses))


def fit(train, val, name, config, seed, dev, path, deadline):
    path.mkdir(parents=True, exist_ok=True)
    identity = digest(
        [name, config, seed, [(r["recording_id"], r["cache_sha256"]) for r in train + val]]
    )
    if (path / "result.json").exists():
        out = json.loads((path / "result.json").read_text())
        if out["identity"] != identity or file_hash(path / "weights.pt") != out["weights_sha256"]:
            raise ValueError("Fit identity changed")
        return out
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, std = scaling(train, name)
    model = make(name, len(CONFIG["channels"]), config["dropout"]).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0001)
    b = batches(train, mean, std, seed)
    best = -1
    stale = 0
    history = []
    started = time.monotonic()
    yv = [r["label"] for r in val]
    for e in range(MAX_EPOCHS):
        loss = one_epoch(model, b, optimizer, dev, deadline)
        p = predict(model, val, mean, std, dev)
        score = float(roc_auc_score(yv, p))
        history.append({"epoch": e + 1, "loss": loss, "validation_auroc": score})
        if score > best + 1e-6:
            best = score
            stale = 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {"state": state, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)},
                path / "best.pt",
            )
        else:
            stale += 1
        save(
            path / "progress.json",
            {"epochs": history, "elapsed_seconds": time.monotonic() - started},
        )
        if stale >= PATIENCE:
            break
    checkpoint = torch.load(path / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state"])
    p = predict(model, val, mean, std, dev)
    torch.save(checkpoint, path / "weights.pt")
    out = {
        "identity": identity,
        "name": name,
        "config": config,
        "seed": seed,
        "best_auroc": best,
        "epochs": len(history),
        "elapsed_seconds": time.monotonic() - started,
        "parameters": sum(x.numel() for x in model.parameters()),
        "probabilities": p.tolist(),
        "weights_sha256": file_hash(path / "weights.pt"),
    }
    save(path / "result.json", out)
    print(
        "fit",
        name,
        path.name,
        "epochs",
        len(history),
        "seconds",
        round(out["elapsed_seconds"]),
        flush=True,
    )
    return out


def benchmark(run, dev):
    data = load(run)
    train = [r for r in data if r["split"] == "train"]
    out = {}
    start = time.monotonic()
    for name in FAMILIES:
        t = time.monotonic()
        rows = prepared(train, name)
        representation_seconds = time.monotonic() - t
        mean, std = scaling(rows, name)
        m = make(name, len(CONFIG["channels"])).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=0.001)
        b = batches(rows, mean, std, SEED)
        durations = []
        for _ in range(3):
            t = time.monotonic()
            one_epoch(m, b, opt, dev, time.monotonic() + 3600)
            predict(m, rows[: max(1, len(rows) // 3)], mean, std, dev)
            sync(dev)
            durations.append(time.monotonic() - t)
        seconds = float(max(durations[1:]))
        out[name] = {
            "epoch_seconds": seconds,
            "representation_seconds": representation_seconds,
            "forecast_seconds": 1.25 * (seconds * MAX_EPOCHS * 21 + representation_seconds * 2),
            "parameters": sum(p.numel() for p in m.parameters()),
        }
        print(name, out[name], flush=True)
    total = sum(x["forecast_seconds"] for x in out.values())
    runtime_gate(total)
    report = {
        "device": str(dev),
        "platform": platform.platform(),
        "models": out,
        "forecast_seconds": total,
        "elapsed_seconds": time.monotonic() - start,
        "policy_fingerprint": digest(json.loads((run / "comparison-policy.json").read_text())),
    }
    save(run / "comparison-benchmark.json", report)
    return report


def feature(r, name):
    if name == "spectral":
        return spectral_features(r["x"])
    if name == "periodic":
        return periodic_features(r["x"])

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    if name == "demographic":
        return np.array(
            [num(r["age"]), {"M": 1, "Male": 1, "F": 0, "Female": 0}.get(r["sex"], np.nan)]
        )
    if name == "artifact":
        return np.array(
            [len(r["rejected_windows"]), r["windows"], float(np.median(np.std(r["x"], axis=-1)))]
        )
    raise ValueError(name)


def classical(train, val, folds, name, path):
    import joblib

    path.mkdir(exist_ok=True, parents=True)
    if (path / "result.json").exists():
        existing = json.loads((path / "result.json").read_text())
        if file_hash(path / "weights.joblib") != existing["weights_sha256"]:
            raise ValueError("Classical model weights changed")
        return existing
    x = np.stack([feature(r, name) for r in train])
    y = np.array([r["label"] for r in train])
    xv = np.stack([feature(r, name) for r in val])
    scores = []
    for c in [0.01, 0.1, 1, 10]:
        values = []
        for a, b in folds:
            m = make_pipeline(
                SimpleImputer(add_indicator=True, keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(C=c, class_weight="balanced", max_iter=2000),
            ).fit(x[a], y[a])
            values.append(float(roc_auc_score(y[b], m.predict_proba(x[b])[:, 1])))
        scores.append({"C": c, "fold_auroc": values, "mean": float(np.mean(values))})
    best = max(scores, key=lambda r: r["mean"])
    m = make_pipeline(
        SimpleImputer(add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=best["C"], class_weight="balanced", max_iter=2000),
    ).fit(x, y)
    joblib.dump(m, path / "weights.joblib")
    pv = m.predict_proba(xv)[:, 1]
    threshold = choose_threshold([r["label"] for r in val], pv)
    result = {
        "name": name,
        "search": scores,
        "threshold": threshold,
        "validation": metrics([r["label"] for r in val], pv, threshold),
        "training": metrics(y, m.predict_proba(x)[:, 1], threshold),
        "weights_sha256": file_hash(path / "weights.joblib"),
    }
    save(path / "result.json", result)
    return result


def campaign(run, dev):
    policy = json.loads((run / "comparison-policy.json").read_text())
    for name, expected in policy["implementation_hashes"].items():
        if file_hash(Path(__file__).parent / name) != expected:
            raise ValueError("Implementation changed after policy lock")
    bench = json.loads((run / "comparison-benchmark.json").read_text())
    if bench["policy_fingerprint"] != digest(policy) or bench["device"] != str(dev):
        raise ValueError("Benchmark does not match policy/device")
    runtime_gate(bench["forecast_seconds"])
    deadline = time.monotonic() + 96 * 3600
    data = load(run)
    train = [r for r in data if r["split"] == "train"]
    val = [r for r in data if r["split"] == "validation"]
    y = np.array([r["label"] for r in train])
    yv = np.array([r["label"] for r in val])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    freeze(
        run / "training-folds.json",
        [
            {
                "train": [train[i]["participant_key"] for i in a],
                "validation": [train[i]["participant_key"] for i in b],
            }
            for a, b in folds
        ],
    )
    summary = {}
    (run / "models").mkdir(exist_ok=True)
    for name in ["spectral", "periodic", "demographic", "artifact"]:
        summary[name] = classical(train, val, folds, name, run / "models" / name)
    summary["majority"] = {
        "threshold": 0.5,
        "probability": float(y.mean()),
        "validation": metrics(yv, np.full(len(yv), y.mean())),
        "training": metrics(y, np.full(len(y), y.mean())),
    }
    for name in FAMILIES:
        root = run / "models" / name
        root.mkdir(exist_ok=True)
        tr = prepared(train, name)
        va = prepared(val, name)
        search = []
        for i, c in enumerate(SEARCH):
            values = []
            for k, (a, b) in enumerate(folds):
                res = fit(
                    [tr[j] for j in a],
                    [tr[j] for j in b],
                    name,
                    c,
                    SEED + k,
                    dev,
                    root / f"tune-{i}-{k}",
                    deadline,
                )
                values.append(res["best_auroc"])
            search.append({"config": c, "fold_auroc": values, "mean": float(np.mean(values))})
        best = max(search, key=lambda x: x["mean"])["config"]
        final = [fit(tr, va, name, best, s, dev, root / f"seed-{s}", deadline) for s in SEEDS]
        pv = np.mean([r["probabilities"] for r in final], 0)
        threshold = choose_threshold(yv, pv)
        ptr = []
        for s in SEEDS:
            checkpoint = torch.load(
                root / f"seed-{s}" / "weights.pt", weights_only=True, map_location="cpu"
            )
            m = make(name, len(CONFIG["channels"]), best["dropout"]).to(dev)
            m.load_state_dict(checkpoint["state"])
            ptr.append(predict(m, tr, checkpoint["mean"].numpy(), checkpoint["std"].numpy(), dev))
        summary[name] = {
            "search": search,
            "config": best,
            "seeds": final,
            "threshold": threshold,
            "validation": metrics(yv, pv, threshold),
            "training": metrics(y, np.mean(ptr, 0), threshold),
            "parameters": final[0]["parameters"],
        }
        save(root / "result.json", summary[name])
        save(run / "development-results.json", summary)
    save(run / "development-results.json", summary)
    lock = {
        "policy_fingerprint": digest(policy),
        "models": summary,
        "artifacts": {
            str(p.relative_to(run)): file_hash(p)
            for p in (run / "models").rglob("*")
            if p.name in ["weights.pt", "weights.joblib", "result.json"]
        },
    }
    lock["fingerprint"] = digest(lock)
    freeze(run / "all-models-lock.json", lock)


def evaluate(run, dev):
    import joblib

    lock = json.loads((run / "all-models-lock.json").read_text())
    if digest({k: v for k, v in lock.items() if k != "fingerprint"}) != lock["fingerprint"]:
        raise ValueError("Lock modified")
    if lock["policy_fingerprint"] != digest(
        json.loads((run / "comparison-policy.json").read_text())
    ):
        raise ValueError("Policy changed")
    for p, h in lock["artifacts"].items():
        if file_hash(run / p) != h:
            raise ValueError("Model artifact changed")
    if (run / "test-results.json").exists():
        return json.loads((run / "test-results.json").read_text())
    data = load(run, external=True)
    out = {}
    private = {}
    for endpoint in ["internal_test", "tel_aviv", "unm_on", "unm_off"]:
        rows = [
            r
            for r in data
            if r["endpoint"] == endpoint
            or (endpoint.startswith("unm_") and r["endpoint"] == "unm_control")
        ]
        if len({r["participant_key"] for r in rows}) != len(rows):
            raise ValueError("Duplicate primary participant")
        y = np.array([r["label"] for r in rows])
        preds = {}
        scores = {}
        if len(np.unique(y)) < 2:
            out[endpoint] = {"unavailable": "Both classes did not survive QC"}
            continue
        for name, record in lock["models"].items():
            if name == "majority":
                p = np.full(len(rows), record["probability"])
            elif name not in FAMILIES:
                m = joblib.load(run / "models" / name / "weights.joblib")
                p = m.predict_proba(np.stack([feature(r, name) for r in rows]))[:, 1]
            else:
                rr = prepared(rows, name)
                ps = []
                for seed in SEEDS:
                    checkpoint = torch.load(
                        run / "models" / name / f"seed-{seed}" / "weights.pt",
                        weights_only=True,
                        map_location="cpu",
                    )
                    m = make(name, len(CONFIG["channels"]), record["config"]["dropout"]).to(dev)
                    m.load_state_dict(checkpoint["state"])
                    ps.append(
                        predict(m, rr, checkpoint["mean"].numpy(), checkpoint["std"].numpy(), dev)
                    )
                p = np.mean(ps, 0)
            preds[name] = p
            scores[name] = metrics(y, p, record["threshold"])
        rng = np.random.default_rng(SEED)
        differences = {name: [] for name in preds if name != "spectral"}
        for _ in range(2000):
            idx = np.concatenate(
                [rng.choice(np.flatnonzero(y == k), size=sum(y == k), replace=True) for k in [0, 1]]
            )
            base = roc_auc_score(y[idx], preds["spectral"][idx])
            for name in differences:
                differences[name].append(roc_auc_score(y[idx], preds[name][idx]) - base)
        paired = {
            n: {
                "auroc_difference": float(scores[n]["auroc"] - scores["spectral"]["auroc"]),
                "paired_95ci": np.quantile(v, [0.025, 0.975]).tolist(),
                "bonferroni_95ci": np.quantile(
                    v, [0.025 / len(differences), 1 - 0.025 / len(differences)]
                ).tolist(),
            }
            for n, v in differences.items()
        }
        out[endpoint] = {"models": scores, "paired_vs_spectral": paired, "participants": len(rows)}
        private[endpoint] = [
            {
                "participant_key": r["participant_key"],
                "label": r["label"],
                "predictions": {n: float(p[i]) for n, p in preds.items()},
            }
            for i, r in enumerate(rows)
        ]
        save(run / "evaluation-progress.json", out)
    result = {"lock_fingerprint": lock["fingerprint"], "endpoints": out}
    save(run / "participant-predictions.json", private)
    freeze(run / "test-results.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["setup", "benchmark", "campaign", "evaluate"])
    p.add_argument("--run", type=Path, default=Path("data/pd-runs/v2"))
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    if a.action == "setup":
        setup(a.run)
    else:
        globals()[a.action](a.run, device(a.device))


if __name__ == "__main__":
    main()
