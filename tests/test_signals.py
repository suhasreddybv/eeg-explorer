import numpy as np
import pytest

from eeg_explorer.signals import Recording, analyze, features, quality, read_csv, synthetic


def test_alpha_power_and_scale():
    fs = 256
    t = np.arange(fs * 4) / fs
    _, powers = features(10 * np.sin(2 * np.pi * 10 * t), fs)
    assert powers[2] / powers.sum() > 0.99
    assert powers[2] == pytest.approx(50, rel=0.01)


def test_demo_separates_known_changes_and_quality():
    result = analyze(synthetic())
    windows = result["windows"]
    reference = [w for w in windows if w["end"] <= 20]
    assert all(w["status"] == "baseline" for w in reference)
    changed = [w for w in windows if w["start"] >= 44 and w["end"] <= 66]
    assert all(w["status"] == "deviation" for w in changed)
    assert all("alpha" in w["reason"] for w in changed)
    flat = [w for w in windows if w["start"] >= 98 and w["end"] <= 106]
    assert all(any(q["channel"] == "O2" for q in w["quality"]) for w in flat)
    assert all(w["usable_channels"] == 3 for w in flat)


def test_bad_baseline_is_not_silently_used():
    rec = synthetic()
    rec.data[:, : int(20 * rec.sfreq)] = 0
    with pytest.raises(ValueError, match="too few unflagged"):
        analyze(rec)


def test_all_flat_window_has_no_score():
    rec = synthetic()
    rec.data[:, int(30 * rec.sfreq) : int(40 * rec.sfreq)] = 0
    result = analyze(rec)
    w = next(w for w in result["windows"] if w["start"] == 32)
    assert w["score"] is None
    assert w["status"] == "quality"


def test_baseline_does_not_use_later_windows():
    rec = synthetic()
    original = analyze(rec)
    rec.data[:, int(50 * rec.sfreq) :] *= 2
    altered = analyze(rec)
    # Offline filter has edge influence; distant earlier windows are stable.
    for a, b in zip(original["windows"][:12], altered["windows"][:12]):
        assert a["score"] == pytest.approx(b["score"], abs=0.01)


def test_quality_detects_mains_and_flat():
    fs = 256
    t = np.arange(4 * fs) / fs
    assert "60 Hz interference" in quality(10 * np.sin(2 * np.pi * 60 * t), fs)
    assert "flat / repeated samples" in quality(np.zeros(len(t)), fs)


def test_csv_scaling_and_invalid_data(tmp_path):
    path = tmp_path / "signal.csv"
    t = np.arange(24 * 100) / 100
    np.savetxt(
        path, (1e-5 * np.sin(2 * np.pi * 10 * t))[:, None], delimiter=",", header="O1", comments=""
    )
    rec = read_csv(path, 100, "V")
    assert np.max(rec.data) == pytest.approx(9.510565, rel=1e-5)
    assert rec.duration == 24
    path.write_text("time,O1\n0,1\n")
    with pytest.raises(ValueError, match="only EEG"):
        read_csv(path, 100, "uV")


def test_nonfinite_samples_rejected():
    from eeg_explorer.signals import validate

    rec = Recording(np.full((1, 2400), np.nan), 100, ["O1"], [])
    with pytest.raises(ValueError, match="non-finite"):
        validate(rec)


@pytest.mark.parametrize("interval", [(-1, 20), (0, 8), (100, 130)])
def test_invalid_baseline(interval):
    with pytest.raises(ValueError):
        analyze(synthetic(), *interval)
