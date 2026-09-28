"""v4 feature families: the explicitly-required synthetic-signal
properties (1/f exponent recovery, B-1 gain invariance, Riemannian
re-centering maps the site mean to the identity, entropy of white noise >
entropy of a sinusoid), plus v3-rich dimensionality/definition parity."""

import json
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pd_pipeline import features_v4 as fv  # noqa: E402
from pyriemann.utils.mean import mean_riemann  # noqa: E402


def test_specparam_recovers_known_1_over_f_exponent():
    true_exponent, true_offset = 1.5, 2.0
    freqs = fv.FREQS.astype(float)
    power = 10**true_offset * freqs ** (-true_exponent)
    result = fv.specparam_channel(power)
    assert result["exponent"] == pytest.approx(true_exponent, abs=0.1)
    assert result["r_squared"] > 0.99


def test_specparam_recovers_exponent_with_added_alpha_peak():
    # A pure aperiodic fit is the easy case; add a realistic alpha peak on
    # top and confirm the exponent recovery still holds within tolerance.
    true_exponent, true_offset = 1.2, 1.8
    freqs = fv.FREQS.astype(float)
    aperiodic = 10**true_offset * freqs ** (-true_exponent)
    peak = 0.3 * aperiodic.mean() * np.exp(-((freqs - 10) ** 2) / (2 * 1.5**2))
    power = aperiodic + peak
    result = fv.specparam_channel(power)
    assert result["exponent"] == pytest.approx(true_exponent, abs=0.1)


def test_b1_relative_spectra_invariant_to_gain():
    rng = np.random.default_rng(0)
    tokens = np.log10(np.maximum(rng.random((15, 17, 40)) * 10 + 0.1, 1e-6)).astype("float32")
    mask = np.ones(17, dtype=bool)
    b1_a = fv.vectorize_relative_spectra(fv.relative_spectra(tokens, mask))
    tokens_gain = tokens + np.log10(3.0)  # x3 amplitude -> x9 power -> +log10(9)... use power directly
    tokens_gain = np.log10((10.0**tokens) * 9.0)  # (3x amplitude)^2 = 9x power
    b1_b = fv.vectorize_relative_spectra(fv.relative_spectra(tokens_gain.astype("float32"), mask))
    assert b1_a == pytest.approx(b1_b, abs=1e-4)


def test_riemannian_recentering_maps_site_mean_to_identity():
    rng = np.random.default_rng(0)
    n = 5
    mats = np.stack([a @ a.T + n * np.eye(n) for a in (rng.normal(size=(n, n)) for _ in range(20))])
    site_mean = mean_riemann(mats)
    ref_inv_sqrt = fv._matrix_power(site_mean, -0.5)
    whitened = np.stack([ref_inv_sqrt @ m @ ref_inv_sqrt for m in mats])
    recentered_mean = mean_riemann(whitened)
    assert recentered_mean == pytest.approx(np.eye(n), abs=1e-6)


def test_entropy_of_white_noise_exceeds_sinusoid():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=512)
    t = np.arange(512) / 128
    sine = np.sin(2 * np.pi * 10 * t)
    e_noise = fv.entropy_channel(noise)
    e_sine = fv.entropy_channel(sine)
    assert e_noise["sample"] > e_sine["sample"]
    assert e_noise["permutation"] > e_sine["permutation"]


def test_entropy_flat_channel_is_zero_not_nan():
    flat = np.zeros(512)
    e = fv.entropy_channel(flat)
    assert e == {"sample": 0.0, "permutation": 0.0, "higuchi": 1.0, "lziv": 0.0}


def test_entropy_features_survives_one_degenerate_window(monkeypatch):
    # A single window whose Higuchi FD comes back NaN (a real antropy edge
    # case observed on real Turku/Tel Aviv data) must not poison the whole
    # channel's mean/SD via NaN propagation.
    calls = {"n": 0}
    real = fv.antropy.higuchi_fd

    def flaky_higuchi(x, kmax):
        calls["n"] += 1
        if calls["n"] == 3:
            return float("nan")
        return real(x, kmax=kmax)

    monkeypatch.setattr(fv.antropy, "higuchi_fd", flaky_higuchi)
    rng = np.random.default_rng(0)
    preref128 = rng.normal(size=(5, 17, 128)).astype("float32") * 20
    mask = np.ones(17, dtype=bool)
    out = fv.entropy_features(preref128, mask)
    assert np.isfinite(out).all()


def test_build_all_aggregates_windows_across_a_persons_records_for_a2(tmp_path, monkeypatch):
    # A person with two auxiliary sessions, each individually below the
    # A2 minimum of 10 windows, but summing to >=10 together, must still
    # be counted eligible -- and a person whose sessions never reach 10
    # combined must still be excluded.
    registry = {
        "records": [
            {"id": "a", "person": "aux:person1", "role": "auxiliary"},
            {"id": "b", "person": "aux:person1", "role": "auxiliary"},
            {"id": "c", "person": "aux:person2", "role": "auxiliary"},
        ]
    }
    preparation = {
        "records": [
            {"id": "a", "status": "ready", "windows": 6},
            {"id": "b", "status": "ready", "windows": 5},
            {"id": "c", "status": "ready", "windows": 4},
        ]
    }
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    (tmp_path / "preparation.json").write_text(json.dumps(preparation))
    monkeypatch.setattr(fv, "cache_person_features", lambda *a, **k: None)
    n = fv.build_all(run=tmp_path, min_windows=10)
    assert n == 1  # only person1 (6+5=11 >= 10); person2 (4) stays excluded


def test_v3_rich_matches_v3_definition_and_dimension():
    from pd_pipeline.cohort_v3 import spectral_cov

    rng = np.random.default_rng(0)
    win = rng.normal(size=(17, 1024)) * 1e-5  # matches cohort_v3.extract's raw window scale
    v3_tokens, v3_cov_triu = spectral_cov(win)
    assert v3_tokens.shape == (17, 34)
    assert v3_cov_triu.shape == (153,)

    # Build a v4-shaped single window: tokens at 1-40Hz (40 bins) instead of
    # v3's 2-35Hz (34 bins); v4's cached cov is the shrunk covariance itself
    # (pre matrix-log), unlike v3's stored upper-triangle matrix-log.
    from scipy import signal as sp_signal

    f, psd = sp_signal.welch(win, fs=256, nperseg=256, noverlap=128, axis=-1)
    tokens40 = np.log10(np.maximum(psd[:, (f >= 1) & (f < 41)], 1e-12)).astype("float32")
    cov = np.cov(win)
    cov = 0.9 * cov + 0.1 * np.eye(len(cov)) * np.trace(cov) / len(cov)

    idx_2_35 = (fv.FREQS >= 2) & (fv.FREQS <= 35)
    assert np.allclose(tokens40[:, idx_2_35], v3_tokens, atol=1e-4)

    rich = fv.v3_rich(tokens40[None], cov[None].astype("float32"), np.ones(17, dtype=bool))
    assert rich.shape == (1309,)
    # mean/std over 1 window: std is 0, mean equals the single window's values.
    assert np.allclose(rich[:34], v3_tokens[0], atol=1e-3)
    assert np.allclose(rich[34 * 17 : 34 * 17 * 2], 0.0, atol=1e-3)
