"""One-time held-out evaluation after a fingerprinted model lock."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from .experiments import metrics, predict, spectral_features
from .model import EEGNet, device
from .preprocess import DEFAULT, file_hash, read_raw, receipts, source_ready, windows_from_raw
from .registry import digest, freeze, save, validate

POLICY = {
    "internal_test": "All frozen Iowa internal-test participants, one recording each.",
    "ucsd_on": "Primary external comparison: controls ses-hc and PD ses-on; no OFF fallback.",
    "ucsd_off": "Secondary external medication-shift comparison: same controls and PD ses-off. Not independent of ON comparison.",
    "eye_condition_evidence": "UCSD eyes-open rest: George et al. 2013, doi:10.1016/j.nicl.2013.07.013; source dataset README cites this acquisition study. Iowa eyes-open: source README.",
    "scope": "Cross-site research stress test; medication for Iowa is not harmonized. No clinical screening claim.",
    "unm": "Reserved, not evaluated until eyes-open event intervals are verified.",
}


def select_records(registry, cohort):
    if cohort == "internal_test":
        return [r for r in registry["records"] if r["split"] == "internal_test"]
    if cohort not in {"ucsd_on", "ucsd_off"}:
        raise ValueError("Unreviewed external cohort")
    session = "ses-on" if cohort == "ucsd_on" else "ses-off"
    return [
        r
        for r in registry["records"]
        if r["dataset"] == "ds002778" and ("/ses-hc/" in r["path"] or f"/{session}/" in r["path"])
    ]


def evaluate(archive, run, cohort, device_name="auto"):
    run, archive = Path(run), Path(archive)
    lock = json.loads((run / "model-lock.json").read_text())
    registry = json.loads((run / "registry.json").read_text())
    validate(registry)
    config = json.loads((run / "preprocessing.json").read_text())
    policy = json.loads((run / "evaluation-policy.json").read_text())
    if policy != POLICY or config != DEFAULT:
        raise ValueError("Evaluation protocol changed after registration")
    if lock["registry_fingerprint"] != registry["fingerprint"] or lock[
        "preprocessing_fingerprint"
    ] != digest(config):
        raise ValueError("Model/data protocol mismatch")
    if lock.get("evaluation_policy_fingerprint") != digest(policy):
        raise ValueError("Evaluation policy was not locked before testing")
    for path, sha in lock["model_hashes"].items():
        if file_hash(run / path) != sha:
            raise ValueError("Model checkpoint changed after lock")
    if file_hash(run / "baseline-weights.npz") != lock["baseline_sha256"]:
        raise ValueError("Baseline changed after lock")
    destination = run / f"evaluation-{cohort}.json"
    if destination.exists():
        result = json.loads(destination.read_text())
        if result["model_lock_fingerprint"] != digest(lock):
            raise ValueError("Existing test result belongs to another model")
        return result
    rows = select_records(registry, cohort)
    if not rows or len({r["participant_key"] for r in rows}) != len(rows):
        raise ValueError("Primary session selection is missing or ambiguous")
    verified = receipts(archive)
    if any(not source_ready(archive, r, verified) for r in rows):
        raise ValueError("Complete all cohort downloads before opening test evaluation")
    freeze(
        run / f"test-opened-{cohort}.json",
        {"registry_fingerprint": registry["fingerprint"], "model_lock_fingerprint": digest(lock)},
    )
    test, qc = [], []
    for row in rows:
        source_hashes = {}
        for source in row["sources"]:
            if source["path"].endswith((".set", ".fdt", ".bdf", ".edf", ".vhdr", ".eeg", ".vmrk")):
                sha = file_hash(archive / source["path"])
                if sha != verified[source["path"]]["sha256"]:
                    raise ValueError("Held-out source fingerprint mismatch")
                source_hashes[source["path"]] = sha
        with read_raw(archive / row["path"]) as raw:
            x, starts, rejected = windows_from_raw(raw, config)
        eligible = len(x) >= config["min_windows"]
        qc.append(
            {
                "participant_key": row["participant_key"],
                "label": row["label"],
                "eligible": eligible,
                "accepted_windows": len(x),
                "starts_seconds": starts,
                "rejected_windows": rejected,
                "source_sha256": source_hashes,
            }
        )
        if eligible:
            test.append({**row, "x": x})
    result = {
        "cohort": cohort,
        "model_lock_fingerprint": digest(lock),
        "registry_fingerprint": registry["fingerprint"],
        "policy": policy,
        "participants_total": len(rows),
        "participants_eligible": len(test),
        "coverage": len(test) / len(rows),
        "qc": qc,
        "eligible_by_class": dict(Counter(str(r["label"]) for r in test)),
        "excluded_by_class": dict(Counter(str(r["label"]) for r in qc if not r["eligible"])),
        "note": "Participant-level research test. Exclusions may bias results; small cohorts give uncertain estimates. No refitting or threshold adjustment.",
    }
    if len({r["label"] for r in test}) < 2:
        result["status"] = "not_estimable_insufficient_class_coverage"
        save(destination, result)
        return result
    y = np.array([r["label"] for r in test])
    dev = device(device_name)
    predictions = []
    for path in lock["model_hashes"]:
        checkpoint = torch.load(run / path, map_location="cpu", weights_only=True)
        model = EEGNet(len(config["channels"]), dropout=lock["configuration"]["dropout"]).to(dev)
        model.load_state_dict(checkpoint["model"])
        predictions.append(predict(model, test, checkpoint["scale"].numpy(), dev))
    p = np.mean(predictions, axis=0)
    with np.load(run / "baseline-weights.npz", allow_pickle=False) as weights:
        features = np.stack([spectral_features(r["x"]) for r in test])
        baseline_p = expit(
            ((features - weights["mean"]) / weights["scale"]) @ weights["coef"].T
            + weights["intercept"]
        ).ravel()
        baseline_threshold = float(weights["threshold"])
    result.update(
        status="complete",
        neural=metrics(y, p, lock["threshold"]),
        baseline=metrics(y, baseline_p, baseline_threshold),
        predictions=[
            {
                "participant_key": r["participant_key"],
                "label": r["label"],
                "neural_probability": float(a),
                "baseline_probability": float(b),
            }
            for r, a, b in zip(test, p, baseline_p)
        ],
    )
    save(destination, result)
    return result
