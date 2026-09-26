import numpy as np
import pytest
import torch
from pd_pipeline.cohort_v2 import eye_interval, validate
from pd_pipeline.registry import digest
from pd_pipeline.models_v2 import FAMILIES, make, represent
from pd_pipeline.compare_v2 import scaling, fit


def test_open_eye_interval_requires_one_contiguous_documented_segment():
    events = [
        {"onset": str(130 + i * 1.001), "trial_type": "Eyes Open: Every 1000 ms"} for i in range(60)
    ]
    assert eye_interval(events) == [130, 189.059]
    with pytest.raises(ValueError, match="Ambiguous"):
        eye_interval(events + [{**x, "onset": str(float(x["onset"]) + 200)} for x in events])
    with pytest.raises(ValueError):
        eye_interval(events[:20])


def test_cross_split_identity_rejected():
    records = [
        {"participant_key": "p", "split": s, "label": 1, "dataset": "ds008768"}
        for s in ["train", "internal_test"]
    ]
    reg = {"records": records}
    reg["fingerprint"] = digest(reg)
    with pytest.raises(ValueError, match="leakage"):
        validate(reg)


@pytest.mark.parametrize("name", FAMILIES)
def test_models_learn_and_preserve_batch_independence(name):
    torch.manual_seed(1)
    m = make(name)
    x = represent(np.random.default_rng(1).normal(size=(2, 17, 1024)).astype("float32"), name)
    p = m(torch.from_numpy(x))
    assert p.shape == (2,)
    p.sum().backward()
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in m.parameters())
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(torch.from_numpy(x))[0], m(torch.from_numpy(x[:1]))[0], atol=1e-5)


def test_scalers_refuse_holdouts_and_weight_people_equally():
    x = np.ones((2, 17, 5), dtype="float32")
    a = {"split": "train", "x": x}
    b = {"split": "train", "x": x * 3}
    mean, _ = scaling([a, b], "transformer")
    assert np.allclose(mean, 2)
    mean2, _ = scaling([a, {**b, "x": np.repeat(b["x"], 10, axis=0)}], "transformer")
    assert np.allclose(mean, mean2)
    with pytest.raises(ValueError):
        scaling([{**a, "split": "external"}], "transformer")


def test_completed_fit_resume_checks_identity_and_weights(tmp_path, monkeypatch):
    import pd_pipeline.compare_v2 as c

    monkeypatch.setattr(c, "MAX_EPOCHS", 1)
    rng = np.random.default_rng(5)
    records = [
        {
            "split": "train",
            "label": i % 2,
            "recording_id": str(i),
            "cache_sha256": str(i),
            "x": rng.normal(size=(2, 17, 1024)).astype("float32"),
        }
        for i in range(8)
    ]
    import time

    args = (
        records[:4],
        records[4:],
        "eegnet",
        {"lr": 0.001, "dropout": 0.5},
        1,
        torch.device("cpu"),
        tmp_path,
        time.monotonic() + 60,
    )
    first = fit(*args)
    assert fit(*args) == first
    (tmp_path / "weights.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        fit(*args)
