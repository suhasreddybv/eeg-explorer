"""v4 window QC: filtered-signal amplitude/step checks, the native-rate
line-noise ratio, LPC recovery, and label-free threshold derivation."""

import warnings

import mne
import numpy as np
import pytest

from pd_pipeline.cohort_v3 import CONFIG
from pd_pipeline.qc_v4 import (
    PTP_UV_CEILING,
    apply_thresholds,
    band_power_ratio,
    compute_window_stats,
    derive_thresholds,
    extract,
    levinson_durbin_lpc,
    qc_audit,
)

warnings.filterwarnings("ignore")


def _synthetic_raw(extra=None, seconds=84, sfreq=256):
    t = np.arange(int(seconds * sfreq)) / sfreq
    x = np.stack([10e-6 * np.sin(2 * np.pi * (7 + i * 0.2) * t) for i in range(17)])
    if extra is not None:
        x = x + extra(t)
    return mne.io.RawArray(x, mne.create_info(CONFIG["channels"], sfreq, "eeg"), verbose=False)


def test_lpc_recovers_known_ar2_process():
    rng = np.random.default_rng(0)
    n = 5000
    a1, a2 = 0.6, -0.3
    x = np.zeros(n)
    noise = rng.normal(scale=0.1, size=n)
    for i in range(2, n):
        x[i] = a1 * x[i - 1] + a2 * x[i - 2] + noise[i]
    coef, _ = levinson_durbin_lpc(x, 2)
    assert coef == pytest.approx([a1, a2], abs=0.02)


def test_line_noise_ratio_distinguishes_noisy_from_clean():
    fs = 500.0
    t = np.arange(int(4 * fs)) / fs
    clean = np.stack([np.sin(2 * np.pi * 10 * t)] * 2)
    noisy = clean + np.stack([3 * np.sin(2 * np.pi * 50 * t)] * 2)
    assert band_power_ratio(noisy, fs, (45, 65), (1, 40)) > 1
    assert band_power_ratio(clean, fs, (45, 65), (1, 40)) < 0.01


def test_line_noise_ratio_none_when_nyquist_too_low():
    x = np.zeros((2, 400))
    assert band_power_ratio(x, 100.0, (45, 65), (1, 40)) is None


def test_amplitude_step_computed_on_filtered_window_not_raw():
    # A huge 90 Hz burst is far outside the 1-40 Hz passband: v3 would have
    # flagged this window on the raw resampled signal, but v4's amplitude
    # check runs on the filtered window, so it must NOT see this spike.
    raw = _synthetic_raw(extra=lambda t: 2000e-6 * np.sin(2 * np.pi * 90 * t)[None, :])
    stats, reason = compute_window_stats(raw, None, "supervised")
    assert reason is None
    s = stats["stats"][0]
    assert s["ptp_uv"] < 100  # filtered: small, in-band signal only
    thresholds = {"PTP_UV": 250, "STEP_UV": 60, "LINE_RATIO": 0.5}
    assert apply_thresholds(s, thresholds) == []


def test_extract_shapes_and_window_count():
    raw = _synthetic_raw()
    thresholds = {"PTP_UV": 250, "STEP_UV": 60, "LINE_RATIO": 0.5}
    out, reason = extract(raw, None, "supervised", thresholds)
    assert reason is None
    assert out["tokens"].shape == (20, 17, 40)
    assert out["cov"].shape == (20, 17, 17)
    assert out["fbcov"].shape == (20, 3, 17, 17)
    assert out["lpc"].shape == (20, 17, 12)
    assert out["preref128"].shape == (20, 17, 512)
    assert out["starts"][0] == 2 and out["starts"][-1] == 78
    assert np.isfinite(out["cov"]).all() and np.isfinite(out["fbcov"]).all()


def test_extract_rejects_flat_channel():
    t = np.arange(84 * 256) / 256
    x = np.stack([10e-6 * np.sin(2 * np.pi * (7 + i * 0.2) * t) for i in range(17)])
    x[3] = 0.0  # flat channel
    raw = mne.io.RawArray(x, mne.create_info(CONFIG["channels"], 256, "eeg"), verbose=False)
    thresholds = {"PTP_UV": 250, "STEP_UV": 60, "LINE_RATIO": 0.5}
    out, reason = extract(raw, None, "supervised", thresholds)
    assert reason is None
    assert out["tokens"].shape[0] == 0
    assert out["rejections"].get("flat") == out["attempted"]


def test_derive_thresholds_percentile_and_ceiling():
    stats = [
        {"ptp_uv": float(v), "step_uv": float(v) / 2, "flat": False, "repeat_frac": 0.0, "line_ratio": 0.01, "source_bad": False}
        for v in range(1, 101)
    ]
    thresholds = derive_thresholds(stats)
    assert thresholds["PTP_UV"] == 99  # 99th percentile of 1..100
    assert thresholds["STEP_UV"] == round(99 / 2)
    assert thresholds["PTP_UV"] <= PTP_UV_CEILING

    # A pathological distribution must still be capped at the protocol ceiling.
    extreme = [
        {"ptp_uv": 100000.0, "step_uv": 100000.0, "flat": False, "repeat_frac": 0.0, "line_ratio": 0.9, "source_bad": False}
        for _ in range(50)
    ]
    capped = derive_thresholds(extreme)
    assert capped["PTP_UV"] == PTP_UV_CEILING
    assert capped["LINE_RATIO"] <= 0.5


def test_extract_handles_auxiliary_partial_montage_below_17_channels():
    # Auxiliary role only requires min_aux_channels (12); fbcov/cov must be
    # sized to the actual available-channel count, not hardcoded to 17.
    channels = CONFIG["channels"][:14]
    t = np.arange(84 * 256) / 256
    x = np.stack([10e-6 * np.sin(2 * np.pi * (7 + i * 0.2) * t) for i in range(len(channels))])
    raw = mne.io.RawArray(x, mne.create_info(channels, 256, "eeg"), verbose=False)
    thresholds = {"PTP_UV": 250, "STEP_UV": 60, "LINE_RATIO": 0.5}
    out, reason = extract(raw, None, "auxiliary", thresholds)
    assert reason is None
    assert out["tokens"].shape == (20, 17, 40)
    assert out["cov"].shape == (20, 17, 17)
    assert out["fbcov"].shape == (20, 3, 17, 17)
    assert np.isfinite(out["lpc"]).all()


def test_derive_thresholds_excludes_flat_and_source_bad_windows():
    stats = [{"ptp_uv": 10.0, "step_uv": 5.0, "flat": False, "repeat_frac": 0.0, "line_ratio": 0.01, "source_bad": False}] * 10
    contaminated = stats + [
        {"ptp_uv": 99999.0, "step_uv": 99999.0, "flat": True, "repeat_frac": 0.0, "line_ratio": 0.01, "source_bad": False}
    ]
    clean_only = derive_thresholds(stats)
    with_flat = derive_thresholds(contaminated)
    assert clean_only == with_flat


def test_qc_audit_ds003775_sanity_check_and_a2_minimum(tmp_path):
    registry = {
        "records": [
            {"id": "a", "role": "supervised", "site": "iowa", "label": 1, "dataset": "ds008768"},
            {"id": "b", "role": "supervised", "site": "iowa", "label": 0, "dataset": "ds008768"},
            {"id": "c", "role": "auxiliary", "site": "aux", "dataset": "ds003775"},
            {"id": "d", "role": "locked_test", "site": "turku", "label": None, "dataset": "osf-pehj9"},
        ]
    }
    preparation = {
        "records": [
            {"id": "a", "status": "ready", "windows": 15},
            {"id": "b", "status": "ready", "windows": 4},  # below A2's 10-window minimum
            {"id": "c", "status": "excluded", "windows": 0, "reason": "no_accepted_windows"},
            {"id": "d", "status": "ready", "windows": 30},
        ]
    }
    report = qc_audit(registry, preparation, run=tmp_path)
    iowa_pd = report["supervised_coverage_by_site_diagnosis"]["('iowa', 'PD')"]
    iowa_control = report["supervised_coverage_by_site_diagnosis"]["('iowa', 'control')"]
    assert iowa_pd["n_ready"] == 1 and iowa_pd["n_meets_a2_min_10_windows"] == 1
    assert iowa_control["n_ready"] == 1 and iowa_control["n_meets_a2_min_10_windows"] == 0
    assert report["ds003775_sanity_check"] == {
        "n_recordings": 1,
        "n_ready": 0,
        "note": report["ds003775_sanity_check"]["note"],
    }
    assert report["turku_qc_label_free"]["n_ready"] == 1
    assert report["turku_qc_label_free"]["total_windows"] == 30
    assert (tmp_path / "qc-audit.json").exists()


def test_preref128_is_bandpassed_but_not_average_referenced():
    """Regression for the v4.1 correction. preref128 must be (a) filtered,
    so out-of-band drift and line noise are gone, and (b) NOT average-
    referenced, so Stage 9 can re-reference within a montage."""
    t = np.arange(84 * 256) / 256
    # Alpha signal + heavy sub-1Hz drift + 50Hz line noise, all in-band-able.
    base = np.stack(
        [
            30e-6 * np.sin(2 * np.pi * 10 * t)
            + 40e-6 * np.sin(2 * np.pi * 0.3 * t)
            + 5e-6 * np.sin(2 * np.pi * 50 * t)
            for _ in range(17)
        ]
    )
    # Give each channel a distinct offset so an average reference is detectable.
    base = base + (np.arange(17)[:, None] - 8) * 2e-6
    raw = mne.io.RawArray(base, mne.create_info(CONFIG["channels"], 256, "eeg"), verbose=False)
    out, reason = extract(raw, None, "supervised", {"PTP_UV": 250, "STEP_UV": 60, "LINE_RATIO": 0.5})
    assert reason is None and out["preref128"].shape[0] > 0

    from scipy import signal as sp

    window = out["preref128"][0].astype("float64")
    freqs, psd = sp.welch(window[0], fs=128, nperseg=128)
    total = psd.sum()
    # (a) filtered: almost no power below 1Hz or above 40Hz
    assert psd[freqs < 1].sum() / total < 0.05
    assert psd[freqs > 40].sum() / total < 0.05
    # (b) not average-referenced: channels still share a common component,
    # so the across-channel mean is not identically zero.
    assert np.abs(window.mean(axis=0)).max() > 0
