"""Scientific-boundary checks for expanded data and nested site evaluation."""

import numpy as np
import pytest
import torch
import mne
from pd_pipeline.cohort_v3 import channel_map, extract, CONFIG
from pd_pipeline.learning_v3 import SpectralEncoder, flatten, scaling
from pd_pipeline.compare_v3 import folds, validate_disjoint


def test_channel_aliases_are_explicit_and_ambiguous_channels_fail():
    assert channel_map(["EEG_FP1", "T3", "T6"]) == {"fp1": 0, "t7": 1, "p8": 2}
    with pytest.raises(ValueError):
        channel_map(["T3", "T7"])
    assert "fp1" not in channel_map(["A1"])


def test_full_duration_keeps_more_than_pilot_window_cap():
    t = np.arange(84 * 256) / 256
    x = np.stack([10e-6 * np.sin(2 * np.pi * (7 + i * 0.2) * t) for i in range(17)])
    raw = mne.io.RawArray(x, mne.create_info(CONFIG["channels"], 256, "eeg"), verbose=False)
    out, reason = extract(raw, None, "supervised")
    assert reason is None and len(out["tokens"]) == 20
    assert out["starts"][0] == 2 and out["starts"][-1] == 78
    assert out["cov"].shape == (20, 153) and np.isfinite(out["cov"]).all()
    cropped, _ = extract(raw, [8, 60], "supervised")
    assert cropped["starts"][0] == 10 and cropped["starts"][-1] + 4 <= 58


def test_missing_supervised_channels_are_not_zero_filled():
    raw = mne.io.RawArray(
        np.zeros((12, 2560)), mne.create_info(CONFIG["channels"][:12], 256, "eeg"), verbose=False
    )
    result, reason = extract(raw, None, "supervised")
    assert result is None and reason == "incompatible_montage"


def test_nested_outer_folds_separate_every_person_and_site():
    people = [
        {"key": f"{s}:{y}:{i}", "site": s, "y": y}
        for s in ["a", "b", "c", "d"]
        for y in [0, 1]
        for i in range(6)
    ]
    seen = {"cross_site": [], "within_site": []}
    for f in folds(people):
        train = [people[i] for i in f["train"]]
        test = [people[i] for i in f["test"]]
        validate_disjoint(train, test)
        if f["kind"] == "cross_site":
            assert not {p["site"] for p in train} & {p["site"] for p in test}
        seen[f["kind"]].extend(p["key"] for p in test)
    for keys in seen.values():
        assert len(keys) == len(set(keys)) == len(people)
    with pytest.raises(ValueError):
        validate_disjoint(people[:1], people[:1])


def test_auxiliary_scaling_does_not_let_long_sessions_dominate():
    def person(key, n, value):
        return {
            "key": key,
            "site": "a",
            "records": [
                {"x": np.full((n, 17, 34), value, dtype="float32"), "mask": np.ones(17, dtype=bool)}
            ],
        }

    x, m, _, w = flatten([person("short", 10, 1), person("long", 100, 3)])
    mean, std = scaling(x, m, w)
    np.testing.assert_allclose(mean, 2, atol=1e-5)
    np.testing.assert_allclose(std, 1, atol=1e-5)


def test_masked_channels_cannot_change_visible_encoder_predictions():
    torch.manual_seed(1)
    model = SpectralEncoder().eval()
    x = torch.randn(2, 17, 34)
    mask = torch.ones(2, 17, dtype=torch.bool)
    mask[:, 5:8] = False
    modified = x.clone()
    modified[:, 5:8] = 1000
    with torch.no_grad():
        a = model(x, mask)[0]
        b = model(modified, mask)[0]
    torch.testing.assert_close(a, b)
    model.train()
    logit, reconstruction, _ = model(x, mask, ~mask)
    (logit.square().mean() + reconstruction.square().mean()).backward()
    assert model.project.weight.grad is not None


def test_feature_summary_balances_sessions():
    from pd_pipeline.compare_v3 import summary_features

    records = [
        {"x": np.full((n, 17, 34), v, dtype="float32"), "cov": np.zeros((n, 153))}
        for n, v in [(10, 1), (100, 3)]
    ]
    _, rich = summary_features({"records": records})
    np.testing.assert_allclose(rich[: 17 * 34], 2)


@pytest.mark.parametrize("encoding", ["cp1252", "utf-8"])
@pytest.mark.parametrize("repair", [False, True])
def test_brainvision_header_preserves_encoding_and_original_bytes(
    tmp_path, monkeypatch, encoding, repair
):
    from contextlib import contextmanager
    from pd_pipeline import cohort_v3

    header = tmp_path / "record.vhdr"
    stem = "misspelling" if repair else "record"
    content = (
        f"Brain Vision Header\r\nCodepage={encoding}\r\n"
        f"DataFile={stem}.eeg\r\nMarkerFile={stem}.vmrk\r\n"
        "Ch1=Fp1,,0.1,µV\r\n"
    ).encode(encoding)
    header.write_bytes(content)
    (tmp_path / "record.eeg").write_bytes(b"samples-unchanged")
    (tmp_path / "record.vmrk").write_bytes(b"markers-unchanged")
    captured = []

    @contextmanager
    def fake_read(path):
        captured.append((path, path.read_bytes()))
        yield object()

    monkeypatch.setattr(cohort_v3, "read_raw", fake_read)
    with cohort_v3.safe_raw(header):
        pass
    assert header.read_bytes() == content
    assert (tmp_path / "record.eeg").read_bytes() == b"samples-unchanged"
    assert (tmp_path / "record.vmrk").read_bytes() == b"markers-unchanged"
    assert "Ch1=Fp1,,0.1,µV\r\n".encode(encoding) in captured[0][1]
    if repair:
        assert captured[0][0] != header
        assert b"misspelling" not in captured[0][1]
        assert not captured[0][0].exists()
    else:
        assert captured[0] == (header, content)
