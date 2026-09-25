import copy
import json

import mne
import numpy as np
import pytest

from pd_pipeline.registry import digest, freeze, validate
from pd_pipeline.preprocess import DEFAULT, prepare, windows_from_raw, file_hash


def registry(records):
    value = {"records": records, "seed": 1}
    value["fingerprint"] = digest(value)
    return value


def record(person="person1", split="train", path="signal.set", checksum="abc"):
    return {
        "participant_key": person,
        "dataset": "ds004584",
        "split": split,
        "path": path,
        "sources": [{"is_eeg_signal": True, "checksum": checksum, "hash_algorithm": "md5"}],
    }


def test_registry_rejects_identity_and_signal_leakage():
    with pytest.raises(ValueError, match="leakage"):
        validate(registry([record(), record(split="validation", path="other.set", checksum="def")]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate(
            registry([record(), record(person="person2", split="validation", path="other.set")])
        )
    with pytest.raises(ValueError, match="External"):
        validate(registry([{**record(), "dataset": "ds002778"}]))


def test_registry_tampering_and_immutable_split(tmp_path):
    value = registry([record()])
    freeze(tmp_path / "registry.json", value)
    freeze(tmp_path / "registry.json", value)
    changed = copy.deepcopy(value)
    changed["records"][0]["split"] = "validation"
    with pytest.raises(ValueError, match="fingerprint"):
        validate(changed)
    with pytest.raises(ValueError, match="immutable"):
        freeze(tmp_path / "registry.json", changed)


def test_preprocessing_physical_units_channels_and_disjoint_windows():
    rng = np.random.default_rng(1)
    t = np.arange(70 * 256) / 256
    data = np.array([20e-6 * np.sin(2 * np.pi * (8 + i / 2) * t) for i in range(18)])
    data += rng.normal(0, 0.3e-6, data.shape)
    raw = mne.io.RawArray(data, mne.create_info(DEFAULT["channels"], 256, "eeg"), verbose=False)
    x, starts, rejected = windows_from_raw(raw, DEFAULT)
    assert x.shape == (15, 18, 1024)
    assert np.diff(starts).min() == 4
    assert np.max(np.abs(x)) > 10  # microvolts, not volts
    assert np.allclose(x.mean(axis=1), 0, atol=1e-5)  # agreed-set average reference
    assert not rejected


def test_flat_channel_is_excluded_and_test_processing_is_sealed(tmp_path):
    raw = mne.io.RawArray(
        np.zeros((18, 70 * 256)), mne.create_info(DEFAULT["channels"], 256, "eeg"), verbose=False
    )
    x, _, rejected = windows_from_raw(raw, DEFAULT)
    assert len(x) == 0 and "flat_channel" in rejected[0]["reasons"]
    with pytest.raises(ValueError, match="sealed"):
        prepare(tmp_path, tmp_path, splits=("external",))


def test_missing_channel_rejected():
    raw = mne.io.RawArray(np.zeros((1, 1000)), mne.create_info(["Cz"], 256, "eeg"), verbose=False)
    with pytest.raises(ValueError, match="Missing"):
        windows_from_raw(raw, DEFAULT)


# Training extras are optional for the lightweight explorer installation.
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
from pd_pipeline.experiments import fit_scaler, runtime_gate, metrics, loader, load_data  # noqa: E402
from pd_pipeline.model import EEGNet  # noqa: E402


def test_scaler_rejects_holdouts_and_weights_people_equally():
    rng = np.random.default_rng(1)
    a = {"split": "train", "x": rng.normal(size=(2, 3, 100)).astype("float32")}
    b = {"split": "train", "x": rng.normal(size=(2, 3, 100)).astype("float32") * 5}
    expected = fit_scaler([a, b])
    assert np.allclose(fit_scaler([a, {**b, "x": np.tile(b["x"], (10, 1, 1))}]), expected)
    with pytest.raises(ValueError, match="training"):
        fit_scaler([{**a, "split": "validation"}])


def test_runtime_four_day_boundary():
    runtime_gate(96 * 3600)
    for value in [96 * 3600 + 1, float("nan"), float("inf"), -1]:
        with pytest.raises(ValueError, match="four days"):
            runtime_gate(value)


def test_neural_forward_backward_and_spatial_constraint():
    torch.set_num_threads(2)
    model = EEGNet()
    x = torch.randn(4, 18, 1024)
    logits = model(x)
    assert logits.shape == (4,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 1.0, 0.0, 1.0])
    )
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        model.features[2].weight.mul_(100)
    model.constrain()
    assert model.features[2].weight.flatten(1).norm(dim=1).max() <= 1.00001


def test_participant_metrics_and_sampling():
    result = metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], bootstrap=20)
    assert result["auroc"] == result["sensitivity"] == result["specificity"] == 1
    records = [
        {"x": np.zeros((n, 18, 1024), dtype="float32"), "label": y}
        for n, y in [(1, 0), (5, 0), (2, 1), (4, 1)]
    ]
    batches = loader(records, np.ones(18, dtype="float32"), True, 1)
    weights = batches.sampler.weights.numpy()
    assert np.isclose(weights[:1].sum(), weights[1:6].sum())
    assert np.isclose(weights[:6].sum(), weights[6:].sum())


def test_cache_identity_tampering_rejected(tmp_path):
    (tmp_path / "cache").mkdir()
    row = {**record(), "recording_id": "a", "label": 0, "status": "ready"}
    np.savez(tmp_path / "cache/a.npz", windows=np.ones((10, 18, 1024), dtype="float32"))
    row["preprocessing_fingerprint"] = digest(DEFAULT)
    (tmp_path / "preprocessing.json").write_text(json.dumps(DEFAULT))
    row["cache_sha256"] = file_hash(tmp_path / "cache/a.npz")
    reg = registry([row])
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    report = {
        "registry_fingerprint": reg["fingerprint"],
        "preprocessing_fingerprint": digest(DEFAULT),
        "records": [{**row, "label": 1}],
    }
    (tmp_path / "preparation.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity"):
        load_data(tmp_path)


def test_small_campaign_locks_models_without_opening_tests(tmp_path):
    from pd_pipeline.experiments import campaign
    from pd_pipeline.registry import save

    rng = np.random.default_rng(14)
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    for index in range(16):
        split = "train" if index < 12 else "validation"
        row = {
            **record(f"p{index}", split, f"p{index}.set", f"hash{index}"),
            "recording_id": str(index),
            "label": index % 2,
            "status": "ready",
            "preprocessing_fingerprint": digest(DEFAULT),
        }
        x = rng.normal(size=(10, 18, 1024)).astype("float32")
        np.savez(cache / f"{index}.npz", windows=x)
        row["cache_sha256"] = file_hash(cache / f"{index}.npz")
        rows.append(row)
    reg = registry(
        rows
        + [
            {
                **record("sealed", "internal_test", "sealed.set", "sealedhash"),
                "recording_id": "sealed",
                "label": 0,
            }
        ]
    )
    save(tmp_path / "registry.json", reg)
    save(tmp_path / "preprocessing.json", DEFAULT)
    save(
        tmp_path / "preparation.json",
        {
            "registry_fingerprint": reg["fingerprint"],
            "preprocessing_fingerprint": digest(DEFAULT),
            "records": rows,
            "counts": {"ready": 16},
        },
    )
    save(
        tmp_path / "benchmark.json",
        {
            "device": "cpu",
            "registry_fingerprint": reg["fingerprint"],
            "projected_21_fits_100_epochs_seconds_with_25pct_buffer": 10,
        },
    )
    from pd_pipeline.evaluate import POLICY

    save(tmp_path / "evaluation-policy.json", POLICY)
    result = campaign(tmp_path, "cpu", max_epochs=1)
    assert result["state"] == "model_locked_tests_not_evaluated"
    assert len(result["model_hashes"]) == 3
    tuning = json.loads((tmp_path / "tuning.json").read_text())
    assert len(tuning["fold_results"]) == 18
    for fold in tuning["fold_results"]:
        assert not set(fold["training_people"]) & set(fold["validation_people"])
        assert all(
            int(person[1:]) < 12 for person in fold["training_people"] + fold["validation_people"]
        )
    assert not (cache / "sealed.npz").exists()


def test_external_session_policy_never_doubles_participants():
    from pd_pipeline.evaluate import select_records

    rows = [
        {"dataset": "ds002778", "split": "external", "path": path}
        for path in ["sub-hc1/ses-hc/x", "sub-pd1/ses-on/x", "sub-pd1/ses-off/x"]
    ]
    assert len(select_records({"records": rows}, "ucsd_on")) == 2
    assert len(select_records({"records": rows}, "ucsd_off")) == 2
    with pytest.raises(ValueError, match="Unreviewed"):
        select_records({"records": rows}, "unm")


def test_small_sample_interval_does_not_claim_certainty():
    from pd_pipeline.experiments import wilson

    low, high = wilson(6, 6)
    assert low < 0.7 and high <= 1
