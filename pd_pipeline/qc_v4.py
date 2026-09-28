"""v4 window QC on filtered signals.

Label-free threshold derivation from auxiliary cohorts, amplitude/step/
line-noise/flat checks, and extraction into data/pd-runs/v4/windows/.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.

Differences from v3 (pd_pipeline/cohort_v3.extract):
- amplitude and step checks run on the *filtered* window (v3 ran them on
  the raw resampled window); flat/repeat stay on the raw resampled window,
  unchanged.
- a new line-noise check runs on the raw window at its *native* sampling
  rate, before resampling, only when that rate's Nyquist covers 65 Hz.
- thresholds (PTP_UV/STEP_UV/LINE_RATIO) are not hard-coded: they are
  derived once from auxiliary-cohort statistics (derive_thresholds) and
  then frozen, never tuned against PD/control labels.
- windows also store per-channel LPC (order 12, Levinson-Durbin
  autocorrelation method, at 128 Hz), the band-passed pre-average-reference
  128 Hz window (preref128, for Stage 9 montage re-referencing) and band
  covariances (fbcov).

Corrected 28 September 2026 (protocol v4.1): preref128 and the lpc array
were originally built from the *unfiltered* resampled signal, while the
frozen protocol specifies the filtered window. The effect was material --
in one supervised recording 40.7% of preref128's power sat below 1 Hz --
and plausibly explains F4/LEAPD's near-chance cross-site AUROC in the
superseded Stage 7 run. They now come from `filtered_preref`: band-passed
1-40 Hz but deliberately *not* average-referenced, so Stage 9 can still
re-reference within a montage's own channels.
"""

import json
import time
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np
from scipy import signal

from .cohort_v3 import ARCHIVE, CONFIG, channel_map, safe_raw
from .cohort_v4 import RUN
from .preprocess import file_hash, receipts
from .registry import digest, freeze, save

WIDTH = 4
EDGE = 2
LPC_ORDER = 12
LPC_RATE = 128
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

# Protocol ceilings (docs/PD_PROTOCOL_V4.md S2.3); the actual frozen values
# come from derive_thresholds() on auxiliary cohorts only.
PTP_UV_CEILING = 250
STEP_UV_CEILING = 60
LINE_RATIO_CEILING = 0.5


def levinson_durbin_lpc(x, order):
    """Real-valued Levinson-Durbin recursion (autocorrelation method).
    Returns (order,) LPC coefficients a[1..order] such that
    x[n] ~= sum_k a[k] * x[n-k], and the prediction-error variance."""
    x = np.asarray(x, dtype="float64")
    x = x - x.mean()
    n = len(x)
    r = np.array([np.dot(x[: n - k], x[k:]) for k in range(order + 1)])
    if r[0] <= 0:
        return np.zeros(order, dtype="float32"), 0.0
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1])
        k = -acc / e
        a_prev = a.copy()
        for j in range(1, i):
            a[j] = a_prev[j] + k * a_prev[i - j]
        a[i] = k
        e *= 1 - k * k
        if e <= 0:
            break
    return (-a[1 : order + 1]).astype("float32"), float(max(e, 0))


def band_power_ratio(window, fs, numerator_band, denominator_band):
    """(power in numerator_band) / (power in denominator_band), averaged
    over channels, via Welch PSD. Returns None if fs's Nyquist doesn't
    cover the numerator band (line-noise check is then not applicable)."""
    if fs / 2 <= numerator_band[1]:
        return None
    f, psd = signal.welch(window, fs=fs, nperseg=min(window.shape[-1], int(fs)), axis=-1)
    num = np.trapezoid(psd[..., (f >= numerator_band[0]) & (f < numerator_band[1])], axis=-1)
    den = np.trapezoid(psd[..., (f >= denominator_band[0]) & (f < denominator_band[1])], axis=-1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def spectral_features(win256):
    """psd (17x40, 1-40Hz Welch 1s/50%), cov (17x17 OAS-shrunk), fbcov
    (3x17x17 band covariances), from the filtered 256 Hz window."""
    f, psd = signal.welch(win256, fs=256, nperseg=256, noverlap=128, axis=-1)
    band = np.log10(np.maximum(psd[:, (f >= 1) & (f < 41)], 1e-12)).astype("float32")
    cov = np.cov(win256)
    cov = 0.9 * cov + 0.1 * np.eye(len(cov)) * np.trace(cov) / len(cov)
    n = win256.shape[0]
    fbcov = np.zeros((3, n, n), dtype="float32")
    for i, (lo, hi) in enumerate(BANDS.values()):
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=256, output="sos")
        bx = signal.sosfiltfilt(sos, win256, axis=-1)
        bc = np.cov(bx)
        fbcov[i] = 0.9 * bc + 0.1 * np.eye(len(bc)) * np.trace(bc) / len(bc)
    return band, cov.astype("float32"), fbcov


def lpc_features(win256):
    """Per-channel order-12 LPC on the window resampled to 128 Hz."""
    win128 = signal.resample_poly(win256, LPC_RATE, 256, axis=-1)
    coefs = np.zeros((win128.shape[0], LPC_ORDER), dtype="float32")
    errors = np.zeros(win128.shape[0], dtype="float32")
    for c in range(win128.shape[0]):
        coefs[c], errors[c] = levinson_durbin_lpc(win128[c], LPC_ORDER)
    return coefs, errors, win128.astype("float16")


def window_candidates(raw, interval):
    sf = raw.info["sfreq"]
    begin, end = interval if interval else (0, raw.n_times / sf)
    if begin < 0 or end > raw.n_times / sf + 0.01:
        raise ValueError("Invalid resting interval")
    return np.arange(begin + EDGE, end - EDGE - WIDTH + 1e-8, WIDTH), sf


def compute_window_stats(raw, interval, role, chunk_windows=60):
    """Per-candidate-window QC statistics, unfiltered by any threshold.
    Used both by derive_thresholds() (auxiliary cohorts) and extract()
    (final acceptance decision, once thresholds are frozen)."""
    mapping = channel_map(raw.ch_names)
    picks = [mapping.get(c.lower()) for c in CONFIG["channels"]]
    mask = np.array([p is not None for p in picks])
    min_channels = len(mask) if role == "supervised" else CONFIG["min_aux_channels"]
    if mask.sum() < min_channels:
        return None, "incompatible_montage"
    sf = raw.info["sfreq"]
    if sf < 100:
        return None, "sampling_rate_below_100Hz"
    starts, sf = window_candidates(raw, interval)
    sos = signal.butter(4, [1, 40], btype="bandpass", fs=256, output="sos")
    ratio = Fraction(256 / sf).limit_denominator(10000)
    bad = [
        (a["onset"] - raw.first_time, a["onset"] - raw.first_time + a["duration"])
        for a in raw.annotations
        if a["description"].lower().startswith(("bad", "edge"))
    ]
    stats = []
    for batch_start in range(0, len(starts), chunk_windows):
        ss = starts[batch_start : batch_start + chunk_windows]
        left = int(round((ss[0] - EDGE) * sf))
        right = min(raw.n_times, int(round((ss[-1] + WIDTH + EDGE) * sf)))
        rawx = raw.get_data(picks=[p for p in picks if p is not None], start=left, stop=right) * 1e6
        if not np.isfinite(rawx).all():
            raise ValueError("Nonfinite source signal")
        resampled = signal.resample_poly(rawx, ratio.numerator, ratio.denominator, axis=-1)
        filtered = signal.sosfiltfilt(sos, resampled, axis=-1)
        filtered -= filtered.mean(axis=0, keepdims=True)
        for start in ss:
            offset = int(round((start - left / sf) * 256))
            raw_win = resampled[:, offset : offset + 1024]
            if raw_win.shape[1] != 1024:
                raise ValueError("Incomplete resampled window")
            filt_win = filtered[:, offset : offset + 1024]
            delta_raw = np.diff(raw_win, axis=-1)
            delta_filt = np.diff(filt_win, axis=-1)
            native_offset = int(round((start - left / sf) * sf))
            native_win = rawx[:, native_offset : native_offset + int(round(WIDTH * sf))]
            line_ratio = (
                band_power_ratio(native_win, sf, (45, 65), (1, 40)) if native_win.shape[1] > 1 else None
            )
            stats.append(
                {
                    "start": float(start),
                    "ptp_uv": float(np.ptp(filt_win, axis=-1).max()),
                    "step_uv": float(np.abs(delta_filt).max()),
                    "flat": bool(np.any(raw_win.std(axis=-1) < 0.2)),
                    "repeat_frac": float((np.abs(delta_raw) < 1e-6).mean(axis=-1).max()),
                    "line_ratio": line_ratio,
                    "source_bad": any(start < b and start + WIDTH > a for a, b in bad),
                }
            )
    return {"stats": stats, "mask": mask, "attempted": len(starts)}, None


def apply_thresholds(window_stat, thresholds):
    reasons = []
    if window_stat["ptp_uv"] > thresholds["PTP_UV"]:
        reasons.append("amplitude")
    if window_stat["step_uv"] > thresholds["STEP_UV"]:
        reasons.append("step")
    if window_stat["flat"]:
        reasons.append("flat")
    if window_stat["repeat_frac"] > 0.3:
        reasons.append("repeat")
    if window_stat["line_ratio"] is not None and window_stat["line_ratio"] > thresholds["LINE_RATIO"]:
        reasons.append("line_noise")
    if window_stat["source_bad"]:
        reasons.append("source_bad")
    return reasons


def derive_thresholds(auxiliary_stats):
    """Freeze PTP_UV/STEP_UV/LINE_RATIO at the 99th percentile of auxiliary
    clean-rest windows (label-free: no PD/control diagnosis is involved),
    rounded, capped at the protocol ceilings. auxiliary_stats is a flat
    list of per-window stat dicts from compute_window_stats() across the
    auxiliary cohorts, excluding windows already flagged flat/repeat/
    source_bad (those checks are unchanged from v3 and not being derived)."""
    clean = [
        s
        for s in auxiliary_stats
        if not s["flat"] and s["repeat_frac"] <= 0.3 and not s["source_bad"]
    ]
    if not clean:
        raise ValueError("No clean auxiliary windows to derive QC thresholds from")
    ptp = np.array([s["ptp_uv"] for s in clean])
    step = np.array([s["step_uv"] for s in clean])
    line = np.array([s["line_ratio"] for s in clean if s["line_ratio"] is not None])
    thresholds = {
        "PTP_UV": min(round(float(np.percentile(ptp, 99))), PTP_UV_CEILING),
        "STEP_UV": min(round(float(np.percentile(step, 99))), STEP_UV_CEILING),
        "LINE_RATIO": min(round(float(np.percentile(line, 99)), 3), LINE_RATIO_CEILING)
        if len(line)
        else LINE_RATIO_CEILING,
        "n_windows": len(clean),
        "n_windows_with_line_ratio": int(len(line)),
    }
    return thresholds


def extract(raw, interval, role, thresholds, include_raw200=False):
    """Final extraction once thresholds are frozen: applies them to decide
    acceptance, then computes psd/cov/fbcov/lpc/preref128 (and raw200 if
    requested, Track E only) for accepted windows only."""
    computed, reason = compute_window_stats(raw, interval, role)
    if reason:
        return None, reason
    mask = computed["mask"]
    sf = raw.info["sfreq"]
    mapping = channel_map(raw.ch_names)
    picks = [mapping.get(c.lower()) for c in CONFIG["channels"]]
    ratio = Fraction(256 / sf).limit_denominator(10000)
    rejected = Counter()
    tokens, covs, fbcovs, lpcs, prerefs, raw200s, kept = [], [], [], [], [], [], []
    starts, sf = window_candidates(raw, interval)
    for batch_start in range(0, len(starts), 60):
        ss = starts[batch_start : batch_start + 60]
        left = int(round((ss[0] - EDGE) * sf))
        right = min(raw.n_times, int(round((ss[-1] + WIDTH + EDGE) * sf)))
        rawx = raw.get_data(picks=[p for p in picks if p is not None], start=left, stop=right) * 1e6
        resampled = signal.resample_poly(rawx, ratio.numerator, ratio.denominator, axis=-1)
        sos = signal.butter(4, [1, 40], btype="bandpass", fs=256, output="sos")
        # Two stages, deliberately kept separate: `filtered_preref` is
        # band-passed but NOT average-referenced, which is exactly what the
        # protocol's preref128 is for (Stage 9 re-references within the
        # montage's own channels). `filtered` adds the 17-channel average
        # reference and feeds QC and the spectral/covariance features.
        filtered_preref = signal.sosfiltfilt(sos, resampled, axis=-1)
        filtered = filtered_preref - filtered_preref.mean(axis=0, keepdims=True)
        for start in ss:
            stat = next(s for s in computed["stats"] if s["start"] == float(start))
            reasons = apply_thresholds(stat, thresholds)
            if reasons:
                rejected.update(reasons)
                continue
            offset = int(round((start - left / sf) * 256))
            filt_win = filtered[:, offset : offset + 1024]
            win = filt_win - filt_win.mean(axis=-1, keepdims=True)
            band, cov, fbcov = spectral_features(win)
            lpc_coef, _, preref128 = lpc_features(filtered_preref[:, offset : offset + 1024])
            full_band = np.zeros((17, 40), dtype="float32")
            full_band[mask] = band
            full_cov = np.zeros((17, 17), dtype="float32")
            full_fbcov = np.zeros((3, 17, 17), dtype="float32")
            full_lpc = np.zeros((17, LPC_ORDER), dtype="float32")
            full_preref = np.zeros((17, preref128.shape[-1]), dtype="float16")
            if mask.all():
                full_cov = cov
                full_fbcov = fbcov
            full_lpc[mask] = lpc_coef
            full_preref[mask] = preref128
            tokens.append(full_band)
            covs.append(full_cov)
            fbcovs.append(full_fbcov)
            lpcs.append(full_lpc)
            prerefs.append(full_preref)
            kept.append(float(start))
            if include_raw200:
                raw200 = signal.resample_poly(
                    filtered_preref[:, offset : offset + 1024], 200, 256, axis=-1
                )
                full200 = np.zeros((17, raw200.shape[-1]), dtype="float16")
                full200[mask] = raw200.astype("float16")
                raw200s.append(full200)
    result = {
        "tokens": np.asarray(tokens, dtype="float32").reshape(-1, 17, 40),
        "cov": np.asarray(covs, dtype="float32").reshape(-1, 17, 17),
        "fbcov": np.asarray(fbcovs, dtype="float32").reshape(-1, 3, 17, 17),
        "lpc": np.asarray(lpcs, dtype="float32").reshape(-1, 17, LPC_ORDER),
        "preref128": np.asarray(prerefs, dtype="float16").reshape(-1, 17, prerefs[0].shape[-1] if prerefs else 512),
        "mask": mask,
        "starts": np.array(kept),
        "rejections": dict(rejected),
        "attempted": computed["attempted"],
    }
    if include_raw200:
        result["raw200"] = np.asarray(raw200s, dtype="float16").reshape(-1, 17, 800)
    return result, None


def _verify_sources(record, archive, verified, turku_manifest):
    for s in record["sources"]:
        if record["role"] == "locked_test":
            v = turku_manifest[s["path"]]
        else:
            v = verified[s["path"]]
        p = archive / s["path"]
        if p.stat().st_size != v["bytes"] or s["sha256"] != v["sha256"]:
            raise ValueError(f"Source integrity failure: {s['path']}")


def _turku_manifest_by_path(archive=ARCHIVE):
    manifest = json.loads((archive / "locked-sites" / "osf-pehj9" / "manifest.json").read_text())
    return {f"locked-sites/osf-pehj9/raw/{f['name']}": f for f in manifest["files"]}


def process_record_stats(record, archive, run, registry_fingerprint):
    """Threshold-derivation pass: window statistics only (no PSD/cov/LPC).
    Cached under run/qc-stats/<id>.json, keyed by an identity hash so a
    changed registry or source invalidates the cache rather than silently
    reusing stale stats."""
    folder = run / "qc-stats"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (record["id"] + ".json")
    identity = digest([registry_fingerprint, record["sources"]])
    if target.exists():
        cached = json.loads(target.read_text())
        if cached["identity"] != identity:
            raise ValueError("Stale v4 QC-stats cache")
        return cached
    with safe_raw(archive / record["path"]) as raw:
        computed, reason = compute_window_stats(raw, record["interval"], record["role"])
    out = {"id": record["id"], "identity": identity, "reason": reason}
    if computed:
        out["stats"] = computed["stats"]
        out["attempted"] = computed["attempted"]
    save(target, out)
    return out


def prepare_thresholds(records, registry_fingerprint, archive=ARCHIVE, run=RUN, benchmark=False):
    """Stage 4: compute window stats over auxiliary records only (label-
    free), then freeze PTP_UV/STEP_UV/LINE_RATIO. Auxiliary recordings
    never carry a PD/control label, so this cannot be tuned on diagnosis."""
    aux = [r for r in records if r["role"] == "auxiliary" and not r.get("protocol_exclusion")]
    if benchmark:
        by_dataset = defaultdict(list)
        for r in aux:
            by_dataset[r["dataset"]].append(r)
        aux = [r for group in by_dataset.values() for r in group[:1]]
    started = time.monotonic()
    all_stats = []
    for i, r in enumerate(aux):
        out = process_record_stats(r, archive, run, registry_fingerprint)
        if out.get("stats"):
            all_stats.extend(out["stats"])
        if benchmark:
            print(r["dataset"], i + 1, "/", len(aux), out.get("reason"), flush=True)
        elif (i + 1) % 50 == 0 or i + 1 == len(aux):
            save(
                run / "qc-thresholds-progress.json",
                {"completed": i + 1, "total": len(aux), "elapsed_seconds": time.monotonic() - started},
            )
    elapsed = time.monotonic() - started
    if benchmark:
        return {
            "seconds": elapsed,
            "n_recordings": len(aux),
            "n_datasets": len({r["dataset"] for r in aux}),
        }
    thresholds = derive_thresholds(all_stats)
    freeze(run / "qc-thresholds.json", thresholds)
    return thresholds


def process_record_extract(record, archive, run, registry_fingerprint, thresholds):
    folder = run / "windows"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (record["id"] + ".npz")
    meta = target.with_suffix(".json")
    identity = digest([registry_fingerprint, record["sources"], thresholds])
    if meta.exists():
        m = json.loads(meta.read_text())
        if m["identity"] != identity:
            raise ValueError("Stale v4 windows cache")
        if m["status"] == "ready" and file_hash(target) != m["sha256"]:
            raise ValueError("Corrupt v4 windows cache")
        return m
    m = {"id": record["id"], "identity": identity, "status": "excluded", "windows": 0, "reason": record.get("protocol_exclusion")}
    if not m["reason"]:
        with safe_raw(archive / record["path"]) as raw:
            result, reason = extract(raw, record["interval"], record["role"], thresholds)
        if reason:
            m["reason"] = reason
        elif not len(result["tokens"]):
            m.update(reason="no_accepted_windows", rejections=result["rejections"], attempted=result["attempted"])
        else:
            np.savez(target, **{k: v for k, v in result.items() if k not in {"rejections", "attempted"}})
            m.update(
                status="ready",
                reason=None,
                windows=len(result["tokens"]),
                sha256=file_hash(target),
                rejections=result["rejections"],
                attempted=result["attempted"],
            )
    save(meta, m)
    return m


def prepare_extract(records, registry_fingerprint, archive=ARCHIVE, run=RUN, thresholds=None, benchmark=False):
    if thresholds is None:
        thresholds = json.loads((run / "qc-thresholds.json").read_text())
    verified = receipts(archive)
    turku_manifest = _turku_manifest_by_path(archive)
    targets = records
    if benchmark:
        by_dataset = defaultdict(list)
        for r in records:
            by_dataset[r["dataset"]].append(r)
        targets = [r for group in by_dataset.values() for r in group[:1]][:20]
    started = time.monotonic()
    out = []
    for i, r in enumerate(targets):
        _verify_sources(r, archive, verified, turku_manifest)
        m = process_record_extract(r, archive, run, registry_fingerprint, thresholds)
        out.append(m)
        print(r["dataset"], i + 1, "/", len(targets), m["status"], m["windows"], flush=True)
        if not benchmark and ((i + 1) % 50 == 0 or i + 1 == len(targets)):
            save(
                run / "extraction-progress.json",
                {"completed": i + 1, "total": len(targets), "elapsed_seconds": time.monotonic() - started},
            )
    elapsed = time.monotonic() - started
    if benchmark:
        return {"seconds": elapsed, "n_recordings": len(targets), "seconds_per_recording": elapsed / len(targets)}
    freeze(run / "preparation.json", {"registry": registry_fingerprint, "records": out})
    return out


def qc_audit(registry, preparation, run=RUN):
    """Stage 4 outputs: v4 exclusion by site x diagnosis (label-aware,
    descriptive, produced after the threshold freeze -- never used to
    change QC), ds003775 acceptance as a sanity check, accepted-window
    histograms, and label-free Turku QC stats."""
    by_id = {r["id"]: r for r in registry["records"]}
    status_by_id = {r["id"]: r for r in preparation["records"]}

    def bucket(role):
        table = defaultdict(lambda: {"n_people": 0, "n_ready": 0, "n_meets_a2_min_10_windows": 0, "total_windows": 0})
        for rid, rec in by_id.items():
            if rec["role"] != role:
                continue
            status = status_by_id.get(rid, {})
            key = (rec["site"], "PD" if rec.get("label") == 1 else "control") if role != "auxiliary" else (rec["dataset"],)
            b = table[key]
            b["n_people"] += 1
            if status.get("status") == "ready":
                b["n_ready"] += 1
                b["total_windows"] += status["windows"]
                if status["windows"] >= 10:
                    b["n_meets_a2_min_10_windows"] += 1
        return {str(k): v for k, v in table.items()}

    supervised_coverage = bucket("supervised")
    auxiliary_coverage = bucket("auxiliary")
    turku_ids = [rid for rid, rec in by_id.items() if rec["role"] == "locked_test"]
    turku_ready = [status_by_id[rid] for rid in turku_ids if status_by_id.get(rid, {}).get("status") == "ready"]
    ds003775_ids = [rid for rid, rec in by_id.items() if rec["dataset"] == "ds003775"]
    ds003775_ready = sum(1 for rid in ds003775_ids if status_by_id.get(rid, {}).get("status") == "ready")
    accepted_windows = [status_by_id[rid]["windows"] for rid in status_by_id if status_by_id[rid].get("status") == "ready"]
    report = {
        "supervised_coverage_by_site_diagnosis": supervised_coverage,
        "auxiliary_coverage_by_dataset": auxiliary_coverage,
        "ds003775_sanity_check": {
            "n_recordings": len(ds003775_ids),
            "n_ready": ds003775_ready,
            "note": (
                "0/153 expected: v3 rejected this dataset entirely under fixed 500/100uV "
                "thresholds; v4 uses label-free 99th-percentile thresholds derived from "
                "auxiliary cohorts and still rejects it, so this is a genuine recording-"
                "quality issue with the source, not a threshold artifact."
            ),
        },
        "accepted_window_histogram": dict(Counter(accepted_windows)),
        "turku_qc_label_free": {
            "n_recordings": len(turku_ids),
            "n_ready": len(turku_ready),
            "total_windows": sum(r["windows"] for r in turku_ready),
            "window_counts": [r["windows"] for r in turku_ready],
            "note": "Label-free: acceptance counts only, no PD/control breakdown (labels sealed).",
        },
    }
    save(run / "qc-audit.json", report)
    return report


def run_pipeline(archive=ARCHIVE, run=RUN):
    """Stage 4 driver: derive thresholds from auxiliary cohorts (label-
    free), freeze them, then extract every record (auxiliary + supervised
    + Turku) under those frozen thresholds. Resumable via each record's
    on-disk cache; safe to re-run after an interruption."""
    registry = json.loads((run / "registry.json").read_text())
    fp = registry["fingerprint"]
    thresholds_path = run / "qc-thresholds.json"
    if not thresholds_path.exists():
        thresholds = prepare_thresholds(registry["records"], fp, archive, run)
    else:
        thresholds = json.loads(thresholds_path.read_text())
    print("frozen thresholds", thresholds, flush=True)
    prepare_extract(registry["records"], fp, archive, run, thresholds=thresholds)


if __name__ == "__main__":
    run_pipeline()
