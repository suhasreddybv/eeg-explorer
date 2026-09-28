"""v4 feature families.

B-1..B-6 (relative spectra, specparam, Riemannian, LPC/LEAPD, entropy,
robust distributions) plus the recomputed v3-rich baseline vector,
computed from the windows cached by qc_v4.extract().

B-7 (connectivity: wPLI/AEC region-pair blocks) is NOT yet implemented.
It is explicitly marked secondary/optional in docs/PD_PROTOCOL_V4.md and
depends only on preref128 (already cached), so it can be added later
without touching Stage 4's frozen windows. Family F8 and the T1 stacker's
optional F8 input are not available until it lands.

Every function here is pure: it maps one person's stacked window arrays to
a fixed-size vector, with no fitted state. Anything that must be fit
(scalers, PCA, tangent-space reference, LEAPD hyperplanes, normative
models) is fold-dependent and lives in models_v4.py / harmonize_v4.py
instead, computed only from training-fold people.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module implements.
"""

import json
import warnings
from pathlib import Path

import antropy
import numpy as np

from .cohort_v3 import CONFIG

RUN = Path("data/pd-runs/v4")
CHANNELS = CONFIG["channels"]
FREQS = np.arange(1, 41)  # tokens bin i <-> (i+1) Hz, per qc_v4.spectral_features
BANDS = {"theta": (4, 8), "pre_alpha": (5.5, 8), "alpha": (8, 13), "beta": (13, 30)}
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central": ["C3", "C4"],
    "temporal": ["T7", "T8", "P7", "P8"],
    "parietal": ["P3", "P4"],
    "occipital": ["O1", "O2"],
}
REGION_IDX = {name: [CHANNELS.index(c) for c in chans] for name, chans in REGIONS.items()}


# ---------------------------------------------------------------------------
# Loading and the A2 first-15-window rule
# ---------------------------------------------------------------------------


def load_record_windows(record_id, run=RUN):
    path = run / "windows" / (record_id + ".npz")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def stack_person(record_ids, run=RUN, first_n=15, all_windows=False):
    """Concatenate a person's ready-record windows, in (record order, then
    start-time order) -- the closest available proxy to time order when a
    person has more than one session (auxiliary only; supervised/locked_test
    have exactly one record after A1). Applies the first-15-window rule
    (A2) unless all_windows=True."""
    loaded = [load_record_windows(rid, run) for rid in sorted(record_ids)]
    keys = ["tokens", "cov", "fbcov", "lpc", "preref128"]
    stacked = {k: np.concatenate([r[k] for r in loaded], axis=0) for k in keys}
    mask = loaded[0]["mask"]
    if not all((r["mask"] == mask).all() for r in loaded):
        raise ValueError("Inconsistent channel mask across a person's own records")
    if not all_windows:
        stacked = {k: v[:first_n] for k, v in stacked.items()}
    stacked["mask"] = mask
    stacked["n_windows"] = stacked["tokens"].shape[0]
    return stacked


# ---------------------------------------------------------------------------
# B-1: relative spectra
# ---------------------------------------------------------------------------


def _band_power(power, lo, hi):
    idx = (FREQS >= lo) & (FREQS <= hi)
    return power[..., idx].sum(axis=-1)


def relative_spectra(tokens, mask):
    """log(PSD / sum PSD over 2-35Hz) per channel (mean over windows), plus
    alpha/theta ratio, individual alpha peak frequency, spectral centroid
    and 90% spectral edge (2-35Hz). Scale-invariant: every quantity is a
    ratio or a shape descriptor of the spectrum, never a raw magnitude."""
    power = 10.0**tokens  # (n_windows, 17, 40)
    idx_2_35 = (FREQS >= 2) & (FREQS <= 35)
    total = power[..., idx_2_35].sum(axis=-1, keepdims=True)
    relative = np.log10(np.maximum(power / np.maximum(total, 1e-30), 1e-30))
    relative_mean = relative.mean(axis=0)  # (17, 40)

    alpha = _band_power(power, *BANDS["alpha"]) + 1e-30
    theta = _band_power(power, *BANDS["theta"]) + 1e-30
    alpha_theta_ratio = float(np.mean(alpha / theta))

    mean_spectrum = power[:, mask].mean(axis=(0, 1))  # (40,) over available channels
    smoothed = np.convolve(mean_spectrum, np.ones(3) / 3, mode="same")
    alpha_window = (FREQS >= 7) & (FREQS <= 13)
    peak_freq = float(FREQS[alpha_window][np.argmax(smoothed[alpha_window])])

    band_2_35 = mean_spectrum[idx_2_35]
    freqs_2_35 = FREQS[idx_2_35]
    centroid = float(np.sum(freqs_2_35 * band_2_35) / np.sum(band_2_35))
    cumulative = np.cumsum(band_2_35) / np.sum(band_2_35)
    edge90 = float(freqs_2_35[np.searchsorted(cumulative, 0.9)])

    full_relative = np.zeros((17, 40), dtype="float32")
    full_relative[mask] = relative_mean[mask]
    return {
        "relative_spectrum": full_relative,
        "alpha_theta_ratio": alpha_theta_ratio,
        "alpha_peak_frequency": peak_freq,
        "spectral_centroid": centroid,
        "spectral_edge_90": edge90,
    }


def vectorize_relative_spectra(features):
    return np.concatenate(
        [
            features["relative_spectrum"].ravel(),
            [
                features["alpha_theta_ratio"],
                features["alpha_peak_frequency"],
                features["spectral_centroid"],
                features["spectral_edge_90"],
            ],
        ]
    ).astype("float32")


# ---------------------------------------------------------------------------
# B-2: specparam / fooof
# ---------------------------------------------------------------------------


def specparam_channel(mean_power_spectrum):
    """Fit specparam/fooof on one channel's mean PSD (2-40Hz). Returns
    exponent, offset, r_squared, and aperiodic-adjusted band powers; R^2 <
    0.8 is signalled via r_squared so the caller can impute within-fold."""
    from fooof import FOOOF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = FOOOF(
            aperiodic_mode="fixed",
            peak_width_limits=[1, 8],
            max_n_peaks=6,
            min_peak_height=0.05,
            verbose=False,
        )
        freq_range = (FREQS >= 2) & (FREQS <= 40)
        try:
            fm.fit(FREQS[freq_range], mean_power_spectrum[freq_range])
        except Exception:
            return {"exponent": np.nan, "offset": np.nan, "r_squared": 0.0, "adjusted": {b: np.nan for b in BANDS}, "peak_cf": np.nan, "peak_pw": np.nan}
    offset, exponent = fm.aperiodic_params_
    fit_freqs = FREQS[freq_range]
    aperiodic_fit = 10 ** (offset - exponent * np.log10(fit_freqs))
    flattened = np.maximum(mean_power_spectrum[freq_range] - aperiodic_fit, 0)
    adjusted = {
        name: float(flattened[(fit_freqs >= lo) & (fit_freqs <= hi)].mean())
        if ((fit_freqs >= lo) & (fit_freqs <= hi)).any()
        else float("nan")
        for name, (lo, hi) in BANDS.items()
    }
    peaks = fm.peak_params_
    alpha_peaks = peaks[(peaks[:, 0] >= 8) & (peaks[:, 0] <= 13)] if len(peaks) else np.zeros((0, 3))
    if len(alpha_peaks):
        best = alpha_peaks[np.argmax(alpha_peaks[:, 1])]
        peak_cf, peak_pw = float(best[0]), float(best[1])
    else:
        peak_cf, peak_pw = np.nan, np.nan
    return {
        "exponent": float(exponent),
        "offset": float(offset),
        "r_squared": float(fm.r_squared_),
        "adjusted": adjusted,
        "peak_cf": peak_cf,
        "peak_pw": peak_pw,
    }


def specparam_features(tokens, mask):
    """Per-channel specparam fit on each channel's mean PSD, plus regional
    averages (frontal/central/temporal/parietal/occipital)."""
    power = 10.0**tokens
    mean_power = power.mean(axis=0)  # (17, 40)
    per_channel = []
    for c in range(17):
        if mask[c]:
            per_channel.append(specparam_channel(mean_power[c]))
        else:
            per_channel.append(
                {"exponent": np.nan, "offset": np.nan, "r_squared": 0.0, "adjusted": {b: np.nan for b in BANDS}, "peak_cf": np.nan, "peak_pw": np.nan}
            )
    regional = {}
    for region, idxs in REGION_IDX.items():
        available = [i for i in idxs if mask[i]]
        if available:
            regional[region] = {
                "exponent": float(np.nanmean([per_channel[i]["exponent"] for i in available])),
                "alpha_adjusted": float(np.nanmean([per_channel[i]["adjusted"]["alpha"] for i in available])),
            }
        else:
            regional[region] = {"exponent": np.nan, "alpha_adjusted": np.nan}
    return {"per_channel": per_channel, "regional": regional}


def vectorize_specparam(features):
    parts = []
    for ch in features["per_channel"]:
        parts.extend([ch["exponent"], ch["offset"], ch["r_squared"], ch["peak_cf"], ch["peak_pw"]])
        parts.extend(ch["adjusted"][b] for b in BANDS)
    for region in REGIONS:
        parts.append(features["regional"][region]["exponent"])
        parts.append(features["regional"][region]["alpha_adjusted"])
    return np.array(parts, dtype="float32")


# ---------------------------------------------------------------------------
# B-3: Riemannian (label-free per-person mean; fold-dependent tangent
# mapping happens in harmonize_v4.py / models_v4.py, not here)
# ---------------------------------------------------------------------------


def riemannian_means(cov, fbcov, mask):
    """Per-person Riemannian mean of the broadband and band covariances
    over windows. Only defined for a full 17-channel montage (partial
    auxiliary montages return None -- B-3 is a supervised/Turku family)."""
    from pyriemann.utils.mean import mean_riemann

    if not mask.all():
        return None
    broadband = mean_riemann(cov.astype("float64"))
    bands = np.stack([mean_riemann(fbcov[:, i].astype("float64")) for i in range(fbcov.shape[1])])
    return {"cov_mean": broadband.astype("float32"), "fbcov_mean": bands.astype("float32")}


def tangent_vector(matrix, reference):
    """Tangent-space vector of an SPD matrix at a given SPD reference
    (identity for no re-centering), via matrix log after whitening."""
    ref_inv_sqrt = _matrix_power(reference, -0.5)
    whitened = ref_inv_sqrt @ matrix @ ref_inv_sqrt
    val, vec = np.linalg.eigh(whitened)
    log_whitened = (vec * np.log(np.maximum(val, 1e-10))) @ vec.T
    n = log_whitened.shape[0]
    iu = np.triu_indices(n)
    scale = np.sqrt(2) * np.ones(len(iu[0]))
    scale[iu[0] == iu[1]] = 1.0
    return (log_whitened[iu] * scale).astype("float32")


def _matrix_power(m, p):
    val, vec = np.linalg.eigh(m)
    return (vec * np.maximum(val, 1e-10) ** p) @ vec.T


# ---------------------------------------------------------------------------
# B-4: LPC / LEAPD (label-free part only; LEAPD hyperplane scoring is
# fold-dependent and lives in models_v4.py)
# ---------------------------------------------------------------------------


def lpc_summary(lpc, mask):
    mean = lpc.mean(axis=0)  # (17, 12)
    sd = lpc.std(axis=0)
    full_mean = np.zeros((17, 12), dtype="float32")
    full_sd = np.zeros((17, 12), dtype="float32")
    full_mean[mask] = mean[mask]
    full_sd[mask] = sd[mask]
    return np.concatenate([full_mean.ravel(), full_sd.ravel()]).astype("float32")


# ---------------------------------------------------------------------------
# B-5: entropy (from preref128, the only cached time-domain array)
# ---------------------------------------------------------------------------


def entropy_channel(x):
    """Per-window entropy metrics. antropy's estimators can occasionally
    return NaN or raise on a degenerate-but-not-exactly-flat window (e.g.
    amplifier near-saturation); each metric falls back independently to
    its flat-signal value rather than letting one bad window poison a
    whole channel's mean via NaN propagation."""
    x = x.astype("float64")
    sd = x.std()
    if sd < 1e-9:
        return {"sample": 0.0, "permutation": 0.0, "higuchi": 1.0, "lziv": 0.0}
    result = {}
    for key, fallback, fn in [
        ("sample", 0.0, lambda: antropy.sample_entropy(x, order=2, metric="chebyshev")),
        ("permutation", 0.0, lambda: antropy.perm_entropy(x, order=5, delay=1, normalize=True)),
        ("higuchi", 1.0, lambda: antropy.higuchi_fd(x, kmax=10)),
        ("lziv", 0.0, lambda: antropy.lziv_complexity(x > np.median(x), normalize=True)),
    ]:
        try:
            value = float(fn())
            result[key] = value if np.isfinite(value) else fallback
        except (ValueError, ZeroDivisionError, FloatingPointError):
            result[key] = fallback
    return result


def entropy_features(preref128, mask):
    """Per-channel sample/permutation entropy, Higuchi FD and normalized
    Lempel-Ziv complexity, mean+SD over windows, from the cached preref128
    (128Hz, pre-average-reference) time series."""
    n_windows = preref128.shape[0]
    per_channel = np.zeros((17, 4, 2), dtype="float32")  # [metric, mean/sd]
    for c in range(17):
        if not mask[c]:
            continue
        values = {"sample": [], "permutation": [], "higuchi": [], "lziv": []}
        for w in range(n_windows):
            e = entropy_channel(preref128[w, c])
            for k in values:
                values[k].append(e[k])
        for i, k in enumerate(["sample", "permutation", "higuchi", "lziv"]):
            per_channel[c, i, 0] = np.nanmean(values[k])
            per_channel[c, i, 1] = np.nanstd(values[k])
    return per_channel.reshape(17, 8).astype("float32")


# ---------------------------------------------------------------------------
# B-6: robust distributions
# ---------------------------------------------------------------------------


def robust_distributions(tokens, mask):
    """10/50/90th percentiles over windows of the B-1 band values, plus
    burst fraction (share of windows where theta or pre-alpha relative
    power exceeds the person's median x1.5)."""
    power = 10.0**tokens
    band_powers = {name: _band_power(power, lo, hi) for name, (lo, hi) in BANDS.items()}
    percentiles = {}
    for name, values in band_powers.items():
        p = np.percentile(values[:, mask], [10, 50, 90], axis=0)  # (3, n_available)
        full = np.zeros((3, 17), dtype="float32")
        full[:, mask] = p
        percentiles[name] = full
    theta_med = np.median(band_powers["theta"][:, mask], axis=0, keepdims=True)
    pre_alpha_med = np.median(band_powers["pre_alpha"][:, mask], axis=0, keepdims=True)
    burst = (
        (band_powers["theta"][:, mask] > 1.5 * theta_med)
        | (band_powers["pre_alpha"][:, mask] > 1.5 * pre_alpha_med)
    ).mean()
    return {"percentiles": percentiles, "burst_fraction": float(burst)}


def vectorize_robust_distributions(features):
    parts = [features["percentiles"][name].ravel() for name in BANDS]
    parts.append(np.array([features["burst_fraction"]], dtype="float32"))
    return np.concatenate(parts).astype("float32")


# ---------------------------------------------------------------------------
# v3-rich baseline, recomputed on v4 windows (same definition as
# compare_v3.summary_features / cohort_v3.spectral_cov)
# ---------------------------------------------------------------------------


def v3_rich(tokens, cov, mask):
    """Reproduces pd_pipeline.compare_v3's rich feature vector (1,309-dim
    at full montage) from v4-cached tokens/cov: v3's psd covers 2-35Hz (34
    bins), a strict subset of v4's 1-40Hz (40 bins); v3's cov field is the
    upper triangle of the *matrix log* of the shrunk covariance, while v4
    caches the shrunk covariance itself, so the log is applied here."""
    idx_2_35 = (FREQS >= 2) & (FREQS <= 35)
    t = tokens[:, :, idx_2_35]  # (n_windows, 17, 34), matches v3's freqs = arange(2, 36)
    triu = np.triu_indices(17)
    log_cov = []
    for c in cov:
        val, vec = np.linalg.eigh(c)
        lc = (vec * np.log(np.maximum(val, 1e-6))) @ vec.T
        log_cov.append(lc[triu])
    log_cov = np.asarray(log_cov, dtype="float32")
    return np.r_[t.mean(0).ravel(), t.std(0).ravel(), log_cov.mean(0)].astype("float32")


# ---------------------------------------------------------------------------
# C3 normative feature subset (the "38 regional features" the protocol
# defers to implementation -- itemised here so the list is documented)
# ---------------------------------------------------------------------------

# Cached-vector layouts, relied on below:
#   b1 = [relative_spectrum (17x40 = 680), alpha_theta_ratio, alpha_peak_frequency,
#         spectral_centroid, spectral_edge_90]                       -> 684
#   b2 = [per-channel 17 x (exponent, offset, r_squared, peak_cf, peak_pw,
#         adjusted[theta, pre_alpha, alpha, beta]) = 153,
#         then 5 regions x (exponent, alpha_adjusted) = 10]          -> 163
B2_FIELDS_PER_CHANNEL = 5 + len(BANDS)  # exponent, offset, r2, peak_cf, peak_pw + 4 bands


def normative_feature_names():
    """The 38 features the C3 normative model is fit on: 5 regions x 4
    relative band powers (20) + 5 regional specparam exponents (5) + 5
    regional offsets (5) + 5 regional alpha-adjusted powers (5) + 3 global
    spectral-shape descriptors (3)."""
    names = []
    for region in REGIONS:
        for band in BANDS:
            names.append(f"b1_relative_{band}_{region}")
    for region in REGIONS:
        names.append(f"b2_exponent_{region}")
    for region in REGIONS:
        names.append(f"b2_offset_{region}")
    for region in REGIONS:
        names.append(f"b2_alpha_adjusted_{region}")
    names += ["b1_alpha_peak_frequency", "b1_spectral_centroid", "b1_spectral_edge_90"]
    return names


def normative_features(cached):
    """Extract the 38 normative features from one person's cached arrays.
    `cached` is a loaded features npz (or dict) with 'b1' and 'b2'."""
    b1 = np.asarray(cached["b1"])
    b2 = np.asarray(cached["b2"])
    relative = b1[: 17 * 40].reshape(17, 40)
    per_channel = b2[: 17 * B2_FIELDS_PER_CHANNEL].reshape(17, B2_FIELDS_PER_CHANNEL)
    values = []
    for region, idxs in REGION_IDX.items():
        for band, (lo, hi) in BANDS.items():
            band_bins = (FREQS >= lo) & (FREQS <= hi)
            values.append(np.nanmean(relative[np.ix_(idxs, band_bins)]))
    for field_offset in (0, 1):  # exponent, offset
        for region, idxs in REGION_IDX.items():
            values.append(np.nanmean(per_channel[idxs, field_offset]))
    alpha_adjusted_col = 5 + list(BANDS).index("alpha")
    for region, idxs in REGION_IDX.items():
        values.append(np.nanmean(per_channel[idxs, alpha_adjusted_col]))
    values += [float(b1[681]), float(b1[682]), float(b1[683])]
    return np.array(values, dtype="float64")


# ---------------------------------------------------------------------------
# Person-level, label-free caching
# ---------------------------------------------------------------------------


def person_features(record_ids, run=RUN, all_windows=False):
    """All label-free per-person features (everything that does not need a
    fold-specific fit): B-1, B-2, B-4, B-5, B-6, v3-rich vectors and B-3's
    raw Riemannian means (tangent mapping deferred to model-fitting time).
    Cached as one .npz per person under run/features/."""
    stacked = stack_person(record_ids, run, all_windows=all_windows)
    mask = stacked["mask"]
    b1 = relative_spectra(stacked["tokens"], mask)
    b2 = specparam_features(stacked["tokens"], mask)
    b3 = riemannian_means(stacked["cov"], stacked["fbcov"], mask)
    b4 = lpc_summary(stacked["lpc"], mask)
    b5 = entropy_features(stacked["preref128"].astype("float32"), mask)
    b6 = robust_distributions(stacked["tokens"], mask)
    rich = v3_rich(stacked["tokens"], stacked["cov"], mask)
    out = {
        "b1": vectorize_relative_spectra(b1),
        "b2": vectorize_specparam(b2),
        "b4": b4,
        "b5": b5.ravel(),
        "b6": vectorize_robust_distributions(b6),
        "v3_rich": rich,
        "mask": mask,
        "n_windows": stacked["n_windows"],
    }
    if b3 is not None:
        out["b3_cov_mean"] = b3["cov_mean"]
        out["b3_fbcov_mean"] = b3["fbcov_mean"]
    return out


def cache_person_features(person_key, record_ids, run=RUN, all_windows=False):
    folder = run / "features"
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = person_key.replace(":", "__").replace("/", "__")
    target = folder / (safe_name + (".all.npz" if all_windows else ".npz"))
    if target.exists():
        return target
    features = person_features(record_ids, run, all_windows=all_windows)
    np.savez(target, **features)
    return target


def build_all(registry_path=None, preparation_path=None, run=RUN, min_windows=10):
    """Cache label-free per-person features for every A2-eligible person
    (supervised, auxiliary and locked_test alike). A2's >=10-window
    minimum is evaluated on a person's *total* accepted windows summed
    across all of their ready records (matching learning_v3.load_people),
    not per-record -- a person with several short auxiliary sessions that
    only sum to 10+ windows together is still eligible."""
    registry = json.loads((registry_path or run / "registry.json").read_text())
    preparation = json.loads((preparation_path or run / "preparation.json").read_text())
    status_by_id = {r["id"]: r for r in preparation["records"]}
    by_person = {}
    for rec in registry["records"]:
        status = status_by_id.get(rec["id"])
        if status and status["status"] == "ready":
            by_person.setdefault(rec["person"], []).append(rec["id"])
    eligible = {
        person: ids
        for person, ids in by_person.items()
        if sum(status_by_id[i]["windows"] for i in ids) >= min_windows
    }
    for person, ids in eligible.items():
        cache_person_features(person, ids, run)
    return len(eligible)


if __name__ == "__main__":
    n = build_all()
    print(f"cached features for {n} people", flush=True)
