"""Deterministic, inspectable signal features. All internal amplitudes are microvolts."""

from dataclasses import dataclass
from pathlib import Path
import csv

import mne
import numpy as np
from scipy import signal

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
WINDOW = 4.0
HOP = 2.0
MAX_VALUES = 12_000_000


@dataclass
class Recording:
    data: np.ndarray
    sfreq: float
    channels: list[str]
    events: list[dict]

    @property
    def duration(self):
        return self.data.shape[1] / self.sfreq


def validate(rec):
    if not 80 <= rec.sfreq <= 2048:
        raise ValueError("Sampling rate must be between 80 and 2048 Hz.")
    if rec.data.ndim != 2 or not 1 <= len(rec.channels) <= 128:
        raise ValueError("Choose a recording with 1–128 EEG channels.")
    if rec.data.shape[0] != len(rec.channels) or rec.data.size > MAX_VALUES:
        raise ValueError("Recording exceeds the 12-million-value limit. Import a shorter excerpt.")
    if rec.duration < 24:
        raise ValueError("At least 24 seconds are needed for a baseline and comparison.")
    if not np.isfinite(rec.data).all():
        raise ValueError(
            "Recording contains missing or non-finite values. Repair these before import."
        )
    return rec


def synthetic():
    rng = np.random.default_rng(42)
    fs = 256.0
    t = np.arange(int(120 * fs)) / fs
    data = []
    for i in range(4):
        alpha = np.where((t >= 40) & (t < 70), 24, 7)
        x = alpha * np.sin(2 * np.pi * (10 + i * 0.2) * t + i)
        x += 4 * np.sin(2 * np.pi * 6 * t) + rng.normal(0, 3, len(t))
        if i < 2:
            for center in (78, 82, 86):
                x += 240 * np.exp(-(((t - center) / 0.15) ** 2))
        if i == 3:
            x[(t >= 98) & (t < 106)] = 0
        data.append(x)
    return validate(
        Recording(
            np.array(data),
            fs,
            ["Fp1", "Fp2", "O1", "O2"],
            [
                {"start": 40, "duration": 30, "label": "Injected alpha increase (synthetic)"},
                {"start": 77, "duration": 10, "label": "Injected blink-like pulses (synthetic)"},
                {"start": 98, "duration": 8, "label": "Injected flat O2 channel (synthetic)"},
            ],
        )
    )


def read_edf(path: Path):
    raw = mne.io.read_raw_edf(path, preload=False, infer_types=True, verbose="ERROR")
    try:
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if not len(picks):
            raise ValueError("No EEG channels found in the EDF header.")
        if len(picks) > 128 or len(picks) * raw.n_times > MAX_VALUES:
            raise ValueError("EDF exceeds the 128-channel / 12-million-value limit.")
        rec = Recording(
            raw.get_data(picks=picks, units="uV"),
            float(raw.info["sfreq"]),
            [raw.ch_names[i] for i in picks],
            [
                {
                    "start": float(a["onset"]),
                    "duration": float(a["duration"]),
                    "label": str(a["description"]),
                }
                for a in raw.annotations
            ],
        )
        return validate(rec)
    finally:
        raw.close()


def read_csv(path: Path, sfreq: float, units: str):
    if units not in {"uV", "mV", "V"}:
        raise ValueError("Specify CSV amplitude units: uV, mV, or V.")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = [c.strip() for c in next(reader, [])]
        if not header or any(not c.strip() for c in header) or len(set(header)) != len(header):
            raise ValueError("CSV needs unique, nonempty channel names in its first row.")
        if any(c.strip().lower() in {"time", "timestamp", "time_s", "sample"} for c in header):
            raise ValueError(
                "CSV must contain only EEG channel columns. Remove time/index columns."
            )
        if len(header) > 128:
            raise ValueError("CSV supports up to 128 channels.")
        rows = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError("Every CSV row must have one number per channel.")
            rows.append([float(v) for v in row])
            if len(rows) * len(header) > MAX_VALUES:
                raise ValueError("CSV exceeds the 12-million-value limit.")
    values = np.asarray(rows, dtype=float).T * {"uV": 1, "mV": 1000, "V": 1e6}[units]
    return validate(Recording(values, sfreq, [c.strip() for c in header], []))


def quality(x, fs):
    """Conservative heuristic flags, not a validated quality classifier."""
    reasons = []
    span = float(np.ptp(x))
    if np.std(x) < 0.15 or np.mean(np.abs(np.diff(x)) < 1e-8) > 0.3:
        reasons.append("flat / repeated samples")
    if span > 200:
        reasons.append("large amplitude / possible movement")
    if np.max(np.abs(np.diff(x))) > 100:
        reasons.append("abrupt jump")
    freq, psd = signal.welch(x, fs=fs, nperseg=min(len(x), round(fs * 2)))
    total = psd[(freq >= 1) & (freq <= min(100, fs / 2))].sum()
    if total > 0:
        for hz in (50, 60):
            if hz + 1 < fs / 2 and psd[np.abs(freq - hz) <= 1].sum() / total > 0.25:
                reasons.append(f"{hz} Hz interference")
    return reasons


def clean(x, fs, mains=60):
    # Offline, zero-phase filtering. A live implementation must use causal filters.
    if mains and mains + 1 < fs / 2:
        b, a = signal.iirnotch(mains, 30, fs)
        x = signal.filtfilt(b, a, x, axis=-1)
    sos = signal.butter(4, [1, 30], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def features(x, fs):
    freq, psd = signal.welch(x, fs=fs, nperseg=min(len(x), round(2 * fs)))
    df = freq[1] - freq[0]
    powers = np.array([psd[(freq >= lo) & (freq < hi)].sum() * df for lo, hi in BANDS.values()])
    # Log power avoids amplitude-dependent scaling; floor prevents log(0).
    values = np.log10(np.maximum(powers, 1e-10))
    return values, powers


def analyze(rec, baseline_start=0.0, baseline_end=20.0, threshold=4.0, mains=60):
    if not 0 <= baseline_start < baseline_end <= rec.duration:
        raise ValueError("Baseline must lie inside the recording.")
    if baseline_end - baseline_start < 12:
        raise ValueError("Select at least 12 seconds for the baseline.")
    filtered = clean(rec.data, rec.sfreq, mains)
    starts = np.arange(0, rec.duration - WINDOW + 1e-8, HOP)
    values, powers, flags = [], [], []
    for start in starts:
        a, b = round(start * rec.sfreq), round((start + WINDOW) * rec.sfreq)
        channel_features = [features(x[a:b], rec.sfreq) for x in filtered]
        values.append([v[0] for v in channel_features])
        powers.append([v[1] for v in channel_features])
        flags.append([quality(x[a:b], rec.sfreq) for x in rec.data])
    values, powers = np.asarray(values), np.asarray(powers)
    in_base = (starts >= baseline_start) & (starts + WINDOW <= baseline_end)
    valid = np.array([[not flag for flag in window] for window in flags])
    medians, scales, counts = [], [], []
    for c in range(len(rec.channels)):
        reference = values[in_base & valid[:, c], c]
        counts.append(len(reference))
        if len(reference) < 5:
            medians.append(np.full(4, np.nan))
            scales.append(np.full(4, np.nan))
        else:
            center = np.median(reference, axis=0)
            medians.append(center)
            # 0.10 log10-power floor avoids huge scores on nearly invariant baselines.
            scales.append(np.maximum(1.4826 * np.median(np.abs(reference - center), axis=0), 0.10))
    medians, scales = np.array(medians), np.array(scales)
    available = np.array(counts) >= 5
    if not available.any():
        raise ValueError("Baseline has too few unflagged windows. Choose a cleaner interval.")
    z = np.abs((values - medians) / scales)
    windows = []
    for i, start in enumerate(starts):
        usable = valid[i] & available
        score, reason = None, "No channel with both a usable baseline and unflagged signal."
        if usable.any():
            candidates = np.where(usable[:, None], z[i], -1)
            c, band = np.unravel_index(np.argmax(candidates), candidates.shape)
            score = round(float(candidates[c, band]), 3)
            direction = "higher" if values[i, c, band] > medians[c, band] else "lower"
            reason = f"{rec.channels[c]} · {list(BANDS)[band]} power {direction} than baseline"
        quality_notes = [
            {"channel": rec.channels[c], "reasons": flag} for c, flag in enumerate(flags[i]) if flag
        ]
        status = (
            "quality"
            if not usable.any()
            else ("baseline" if in_base[i] else "deviation" if score >= threshold else "typical")
        )
        windows.append(
            {
                "start": float(start),
                "end": float(start + WINDOW),
                "score": score,
                "reason": reason,
                "status": status,
                "quality": quality_notes,
                "usable_channels": int(usable.sum()),
                "relative_power": (
                    powers[i] / np.maximum(powers[i].sum(axis=1, keepdims=True), 1e-10)
                ).tolist(),
            }
        )
    return {
        "method": "log-band-power / median-MAD v1",
        "window_seconds": WINDOW,
        "hop_seconds": HOP,
        "baseline": [baseline_start, baseline_end],
        "threshold": threshold,
        "mains_hz": mains,
        "bands": BANDS,
        "baseline_windows_per_channel": dict(zip(rec.channels, counts)),
        "windows": windows,
        "summary": {
            "windows": len(windows),
            "deviations": sum(w["status"] == "deviation" for w in windows),
            "quality_flagged": sum(bool(w["quality"]) for w in windows),
            "unscorable": sum(w["score"] is None for w in windows),
        },
    }
