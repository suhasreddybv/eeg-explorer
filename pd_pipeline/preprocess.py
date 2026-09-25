"""Versioned per-recording processing. No population statistics are fitted here."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import mne
import numpy as np
from scipy import signal

from .registry import digest, save, validate

# Exclude Pz: the Iowa reference must not be confused with a measured channel.
CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
]
DEFAULT = {
    "version": 1,
    "channels": CHANNELS,
    "sfreq": 256,
    "low_hz": 1,
    "high_hz": 40,
    "window_seconds": 4,
    "max_windows": 15,
    "min_windows": 10,
    "edge_seconds": 2,
    "max_input_seconds": 124,
    "peak_to_peak_uv": 500,
    "max_step_uv": 100,
    "min_std_uv": 0.2,
    "max_flat_fraction": 0.3,
    "qc_status": "Fixed engineering thresholds; require blinded QC review before final evaluation",
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def receipts(root):
    result = {}
    for line in (root / "verified-files.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
            result[row["path"]] = row
        except json.JSONDecodeError:
            continue
    return result


def source_ready(root, record, verified):
    needed = [
        i
        for i in record["sources"]
        if i["path"].endswith((".eeg", ".vhdr", ".vmrk", ".bdf", ".edf", ".set", ".fdt"))
    ]
    if not needed:
        return False
    for item in needed:
        receipt = verified.get(item["path"])
        path = root / item["path"]
        if (
            not receipt
            or not path.exists()
            or path.stat().st_size != receipt["bytes"]
            or path.stat().st_mtime_ns != receipt["mtime_ns"]
        ):
            return False
    return True


def read_raw(path):
    reader = {
        ".vhdr": mne.io.read_raw_brainvision,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
    }.get(path.suffix.lower())
    if reader is None:
        raise ValueError("Unsupported signal format")
    return reader(path, preload=False, verbose="ERROR")


def windows_from_raw(raw, config):
    normalized = {c.lower().strip().rstrip("."): c for c in raw.ch_names}
    if len(normalized) != len(raw.ch_names):
        raise ValueError("Ambiguous channel names")
    missing = [c for c in config["channels"] if c.lower() not in normalized]
    if missing:
        raise ValueError(f"Missing required channels: {missing}")
    raw.pick([normalized[c.lower()] for c in config["channels"]])
    raw.crop(tmax=min(raw.times[-1], config["max_input_seconds"]), include_tmax=False)
    if raw.info["sfreq"] < 2 * config["high_hz"]:
        raise ValueError("Sampling frequency too low")
    # Only one short recording is loaded; filtering never joins distinct recordings.
    raw.load_data()
    raw.resample(config["sfreq"], npad="auto", verbose="ERROR")
    x = raw.get_data() * 1e6
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite signal")
    sf = config["sfreq"]
    sos = signal.butter(
        4, [config["low_hz"], config["high_hz"]], btype="bandpass", fs=sf, output="sos"
    )
    filtered = signal.sosfiltfilt(sos, x, axis=-1)
    filtered -= filtered.mean(axis=0, keepdims=True)
    width = int(config["window_seconds"] * sf)
    edge = int(config["edge_seconds"] * sf)
    accepted, starts, rejected = [], [], []
    bad_intervals = [
        (a["onset"], a["onset"] + a["duration"])
        for a in raw.annotations
        if a["description"].lower().startswith(("bad", "edge"))
    ]
    for start in range(edge, x.shape[1] - edge - width + 1, width):
        segment = x[:, start : start + width]
        delta = np.diff(segment, axis=-1)
        reasons = []
        if np.any(np.ptp(segment, axis=-1) > config["peak_to_peak_uv"]):
            reasons.append("amplitude")
        if np.any(np.max(np.abs(delta), axis=-1) > config["max_step_uv"]):
            reasons.append("abrupt_step")
        if np.any(np.std(segment, axis=-1) < config["min_std_uv"]):
            reasons.append("flat_channel")
        if np.any((np.abs(delta) < 1e-6).mean(axis=-1) > config["max_flat_fraction"]):
            reasons.append("repeated_samples")
        if any(start / sf < b and (start + width) / sf > a for a, b in bad_intervals):
            reasons.append("source_bad_annotation")
        if reasons:
            rejected.append({"start_seconds": start / sf, "reasons": reasons})
        elif len(accepted) < config["max_windows"]:
            window = filtered[:, start : start + width].copy()
            window -= window.mean(axis=-1, keepdims=True)
            accepted.append(window.astype("float32"))
            starts.append(start / sf)
    array = (
        np.stack(accepted)
        if accepted
        else np.empty((0, len(config["channels"]), width), dtype="float32")
    )
    return array, starts, rejected


def prepare(archive, run, splits=("train", "validation"), limit=None):
    """External data stays sealed until a separate protocol review is supplied."""
    if not set(splits) <= {"train", "validation"}:
        raise ValueError("Test preparation is sealed in v1; complete QC/model lock first.")
    archive, run = Path(archive), Path(run)
    registry = json.loads((run / "registry.json").read_text())
    validate(registry)
    config = json.loads((run / "preprocessing.json").read_text())
    if config != DEFAULT:
        raise ValueError("Preprocessing changed: create a reviewed new protocol/version")
    verified = receipts(archive)
    cache = run / "cache"
    cache.mkdir(exist_ok=True)
    report = {
        "registry_fingerprint": registry["fingerprint"],
        "preprocessing_fingerprint": digest(config),
        "records": [],
        "elapsed_seconds": 0,
    }
    started = time.monotonic()
    count = 0
    for record in registry["records"]:
        if record["split"] not in splits:
            continue
        item = {
            k: record[k]
            for k in ["recording_id", "participant_key", "split", "label", "dataset", "path"]
        }
        path = cache / (record["recording_id"] + ".npz")
        meta = path.with_suffix(".json")
        if not source_ready(archive, record, verified):
            item["status"] = "pending_download"
        elif limit is not None and count >= limit:
            item["status"] = "deferred_limit"
        else:
            count += 1
            try:
                source_hashes = {
                    i["path"]: file_hash(archive / i["path"])
                    for i in record["sources"]
                    if i["path"].endswith(
                        (".eeg", ".vhdr", ".vmrk", ".bdf", ".edf", ".set", ".fdt")
                    )
                }
                for name, sha in source_hashes.items():
                    if sha != verified[name]["sha256"]:
                        raise ValueError("Source fingerprint differs from download receipt")
                identity = digest(
                    {"source": source_hashes, "config": config, "registry": registry["fingerprint"]}
                )
                if (
                    path.exists()
                    and meta.exists()
                    and json.loads(meta.read_text()).get("cache_identity") == identity
                    and file_hash(path) == json.loads(meta.read_text()).get("cache_sha256")
                ):
                    item = json.loads(meta.read_text())
                else:
                    with read_raw(archive / record["path"]) as raw:
                        x, starts, rejected = windows_from_raw(raw, config)
                    np.savez(path, windows=x, starts=np.asarray(starts))
                    item.update(
                        status="ready" if len(x) >= config["min_windows"] else "excluded_qc",
                        windows=len(x),
                        retained_seconds=len(x) * config["window_seconds"],
                        rejected_windows=rejected,
                        cache_identity=identity,
                        cache_sha256=file_hash(path),
                        source_sha256=source_hashes,
                        preprocessing_fingerprint=digest(config),
                    )
                    save(meta, item)
            except (ValueError, OSError, RuntimeError) as error:
                item.update(status="error", reason=str(error))
        report["records"].append(item)
        report["elapsed_seconds"] = time.monotonic() - started
        report["counts"] = dict(Counter(r["status"] for r in report["records"]))
        save(run / "preparation.json", report)
        print(record["subject"], item["status"], item.get("windows", ""), flush=True)
    return report
