"""All-duration, participant-grouped v3 features and an explicit archive usage ledger."""

import json
import re
import time
import tempfile
import shutil
from contextlib import contextmanager
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from .cohort_v2 import CONFIG as V2_CONFIG, audit_archive, eye_interval, tsv
from .preprocess import read_raw, file_hash, receipts
from .registry import digest, freeze, save

ARCHIVE = Path("/Volumes/Extreme Pro/EEG data")
RUN = Path("data/pd-runs/v3")
SITES = {"ds008768": "iowa", "ds007526": "tel_aviv", "ds003490": "unm", "ds002778": "ucsd"}
# Auxiliary cohorts never include anyone from a PD evaluation site, even without labels.
AUX = {
    "ds003775",
    "ds004148",
    "ds004504",
    "ds004796",
    "ds005385",
    "ds006036",
    "ds007864",
    "ds007865",
    "bigp3bci",
}
EXCLUDED = {
    "ds004574": "Iowa task release; unresolved overlap with expanded Iowa.",
    "ds004579": "Iowa task release; unresolved overlap with expanded Iowa.",
    "ds004580": "Iowa task release; unresolved overlap with expanded Iowa.",
    "ds004584": "Older Iowa resting release; replaced by expanded release.",
    "ds003506": "UNM task release overlaps evaluation participants; never used in auxiliary pretraining.",
    "ds003509": "UNM task release overlaps evaluation participants; never used in auxiliary pretraining.",
    "ds003194": "Older NeuroEPO release replaced by Phase II/III release to avoid unresolved overlap.",
    "ds003195": "Older placebo release replaced by Phase II/III release to avoid unresolved overlap.",
    "ds005048": "MATLAB v7.3 EEGLAB unsupported by installed reader; needs independently validated adapter.",
    "ds007987": "A1-D32 channel labels need an independently validated physical electrode mapping.",
}
CONFIG = {
    **V2_CONFIG,
    "version": 3,
    "max_windows": None,
    "max_input_seconds": None,
    "min_aux_channels": 12,
    "chunk_windows": 60,
    "representation": "log-PSD-2-35Hz-plus-logcov",
}
ALIASES = {"t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8"}


def channel_map(names):
    out = {}
    for i, name in enumerate(names):
        n = name.lower().strip().rstrip(".")
        if n.startswith("eeg_"):
            n = n[4:]
        n = ALIASES.get(n, n)
        if n in out:
            raise ValueError("Ambiguous channel alias")
        out[n] = i
    return out


def build(archive=ARCHIVE, run=RUN):
    run.mkdir(parents=True, exist_ok=True)
    plan, verified = audit_archive(archive, run)
    groups = defaultdict(list)
    for f in plan["files"]:
        groups[(f["dataset"], f.get("subject"))].append(f)
    metadata = {
        ds: {
            r["participant_id"]: r
            for r in tsv(archive / "provenance/openneuro" / ds / "participants.tsv")
        }
        for ds in SITES
    }
    records = []
    ledger = defaultdict(Counter)
    signals_seen = {}
    for f in plan["files"]:
        if not f["path"].endswith((".vhdr", ".set", ".edf", ".bdf")):
            continue
        ds = f["dataset"]
        ledger[ds]["recordings"] += 1
        if ds in EXCLUDED:
            ledger[ds]["excluded_dataset"] += 1
            continue
        if ds not in AUX | set(SITES):
            raise ValueError("Unclassified dataset " + ds)
        if ds in SITES and "task-rest" not in f["path"].lower():
            ledger[ds]["excluded_nonrest"] += 1
            continue
        row = metadata.get(ds, {}).get(f.get("subject"), {})
        interval = None
        reason = None
        label = None
        if ds in SITES:
            label = (
                1
                if f["diagnosis"] == "parkinsons"
                else 0
                if f["diagnosis"] == "healthy_control"
                else None
            )
            # Use the previously audited study group rather than assume manifest spelling.
            if ds != "ds002778":
                label = {"PD": 1, "Control": 0, "HC": 0, "CTL": 0}[
                    row.get("group", row.get("Group"))
                ]
            else:
                label = (
                    0
                    if f["subject"].startswith("sub-hc")
                    else 1
                    if f["subject"].startswith("sub-pd")
                    else None
                )
            if label is None:
                raise ValueError("Unknown PD/control label")
        stem = f["path"].rsplit("_eeg.", 1)[0]
        sources = [
            x
            for x in groups[(ds, f.get("subject"))]
            if x["path"].startswith(stem + "_")
            and x["path"].endswith((".eeg", ".vhdr", ".vmrk", ".set", ".fdt", ".edf", ".bdf"))
        ]
        if not sources:
            sources = [f]
        if ds == "ds003490":
            try:
                interval = eye_interval(tsv(archive / (stem + "_events.tsv")))
            except ValueError as e:
                reason = str(e)
        key = f["participant_key"]
        if ds == "ds006036":
            key = key.replace("ds006036:", "ds004504:")
        # NeuroEPO arm/releases are kept within one conservative pretraining group.
        if ds in {"ds007864", "ds007865"}:
            key = "neuroepo:unresolved-arm-identities"
        session = next((x for x in f["path"].split("/") if x.startswith("ses-")), "single")
        medication = "unknown"
        if ds == "ds003490":
            medication = row.get("sess1_Med" if session == "ses-01" else "sess2_Med", "unknown")
        if ds == "ds002778":
            medication = {"ses-on": "ON", "ses-off": "OFF", "ses-hc": "control"}.get(
                session, "unknown"
            )
        r = {
            "id": digest(f["path"])[:20],
            "path": f["path"],
            "dataset": ds,
            "site": SITES.get(ds, ds),
            "person": key,
            "role": "supervised" if ds in SITES else "auxiliary",
            "label": label,
            "interval": interval,
            "protocol_exclusion": reason,
            "session": session,
            "medication": medication,
            "age": row.get("age"),
            "sex": row.get("sex", row.get("gender")),
            "sources": [
                {
                    "path": x["path"],
                    "sha256": verified[x["path"]]["sha256"],
                    "bytes": verified[x["path"]]["bytes"],
                }
                for x in sources
            ],
        }
        duplicate = False
        for s in r["sources"]:
            if s["path"].endswith((".eeg", ".fdt", ".edf", ".bdf", ".set")):
                h = s["sha256"]
                if h in signals_seen:
                    old = signals_seen[h]
                    if old["person"] != key:
                        raise ValueError("Exact duplicate across participant identities")
                    duplicate = True
                signals_seen[h] = r
        if duplicate:
            ledger[ds]["excluded_exact_duplicate"] += 1
            continue
        ledger[ds][r["role"]] += 1
        records.append(r)
    reg = {
        "records": records,
        "config": CONFIG,
        "excluded_datasets": EXCLUDED,
        "ledger": dict(ledger),
        "metadata_hashes": {
            ds: file_hash(archive / "provenance/openneuro" / ds / "participants.tsv")
            for ds in SITES
        },
        "history": "All four PD sites were examined in v1/v2. All v3 estimates are exploratory.",
    }
    reg["fingerprint"] = digest(reg)
    freeze(run / "registry.json", reg)
    save(
        run / "usage-summary.json",
        {
            "datasets": dict(ledger),
            "exclusions": EXCLUDED,
            "candidate_people_by_site": {
                s: len({r["person"] for r in records if r["site"] == s})
                for s in sorted({r["site"] for r in records})
            },
        },
    )
    return reg


def spectral_cov(x):
    freq, psd = signal.welch(x, fs=256, nperseg=256, noverlap=128, axis=-1)
    tokens = np.log10(np.maximum(psd[:, (freq >= 2) & (freq <= 35)], 1e-12)).astype("float32")
    cov = np.cov(x)
    cov = 0.9 * cov + 0.1 * np.eye(len(cov)) * np.trace(cov) / len(cov)
    val, vec = np.linalg.eigh(cov)
    lc = (vec * np.log(np.maximum(val, 1e-6))) @ vec.T
    return tokens, lc[np.triu_indices(len(cov))].astype("float32")


def extract(raw, interval, role):
    mapping = channel_map(raw.ch_names)
    picks = [mapping.get(c.lower()) for c in CONFIG["channels"]]
    mask = np.array([p is not None for p in picks])
    if mask.sum() < (len(mask) if role == "supervised" else CONFIG["min_aux_channels"]):
        return None, "incompatible_montage"
    sf = raw.info["sfreq"]
    if sf < 100:
        return None, "sampling_rate_below_100Hz"
    begin, end = interval if interval else (0, raw.n_times / sf)
    if begin < 0 or end > raw.n_times / sf + 0.01:
        raise ValueError("Invalid resting interval")
    width = 4
    edge = 2
    starts = np.arange(begin + edge, end - edge - width + 1e-8, width)
    tokens = []
    covs = []
    kept = []
    rejected = Counter()
    bad = [
        (a["onset"] - raw.first_time, a["onset"] - raw.first_time + a["duration"])
        for a in raw.annotations
        if a["description"].lower().startswith(("bad", "edge"))
    ]
    sos = signal.butter(4, [1, 40], btype="bandpass", fs=256, output="sos")
    ratio = Fraction(256 / sf).limit_denominator(10000)
    for batch_start in range(0, len(starts), CONFIG["chunk_windows"]):
        ss = starts[batch_start : batch_start + CONFIG["chunk_windows"]]
        left = int(round((ss[0] - edge) * sf))
        right = min(raw.n_times, int(round((ss[-1] + width + edge) * sf)))
        rawx = raw.get_data(picks=[p for p in picks if p is not None], start=left, stop=right) * 1e6
        if not np.isfinite(rawx).all():
            raise ValueError("Nonfinite source signal")
        xx = signal.resample_poly(rawx, ratio.numerator, ratio.denominator, axis=-1)
        filtered = signal.sosfiltfilt(sos, xx, axis=-1)
        filtered -= filtered.mean(axis=0, keepdims=True)
        for start in ss:
            offset = int(round((start - left / sf) * 256))
            a = xx[:, offset : offset + 1024]
            if a.shape[1] != 1024:
                raise ValueError("Incomplete resampled window")
            reasons = []
            delta = np.diff(a, axis=-1)
            if np.any(np.ptp(a, axis=-1) > 500):
                reasons.append("amplitude")
            if np.any(np.abs(delta).max(axis=-1) > 100):
                reasons.append("step")
            if np.any(a.std(axis=-1) < 0.2):
                reasons.append("flat")
            if np.any((np.abs(delta) < 1e-6).mean(axis=-1) > 0.3):
                reasons.append("repeat")
            if any(start < b and start + 4 > a for a, b in bad):
                reasons.append("source_bad")
            if reasons:
                rejected.update(reasons)
                continue
            win = filtered[:, offset : offset + 1024]
            win -= win.mean(axis=-1, keepdims=True)
            tok, cv = spectral_cov(win)
            full = np.zeros((17, 34), dtype="float32")
            full[mask] = tok
            tokens.append(full)
            kept.append(float(start))
            if mask.all():
                covs.append(cv)
    return {
        "tokens": np.asarray(tokens, dtype="float32").reshape(-1, 17, 34),
        "cov": np.asarray(covs, dtype="float32").reshape(-1, 153),
        "mask": mask,
        "starts": np.array(kept),
        "rejections": dict(rejected),
        "attempted": len(starts),
    }, None


@contextmanager
def safe_raw(path):
    """Repair only an impossible EDF wall-clock second in a temporary copy."""
    if path.suffix.lower() == ".edf":
        with path.open("rb") as f:
            header = f.read(256)
        if header[182:184] == b"60":
            with tempfile.TemporaryDirectory(prefix="eeg-v3-edf-") as td:
                copy = Path(td) / path.name
                shutil.copyfile(path, copy)
                with copy.open("r+b") as f:
                    f.seek(182)
                    f.write(b"59")
                with read_raw(copy) as raw:
                    yield raw
            return
    if path.suffix.lower() == ".vhdr":
        header = path.read_text(encoding="utf-8-sig")
        references = re.findall(r"^(DataFile|MarkerFile)=(.+)$", header, re.MULTILINE)
        missing = any(not (path.parent / value.strip()).exists() for _, value in references)
        if missing:
            replacements = {
                "DataFile": path.with_suffix(".eeg"),
                "MarkerFile": path.with_suffix(".vmrk"),
            }
            if not all(p.exists() for p in replacements.values()):
                raise FileNotFoundError("Canonical BrainVision companions unavailable")
            with tempfile.TemporaryDirectory(prefix="eeg-v3-header-") as td:
                copy = Path(td) / path.name
                for key, value in replacements.items():
                    header = re.sub(
                        r"^" + key + r"=.+$",
                        lambda m: key + "=" + str(value),
                        header,
                        flags=re.MULTILINE,
                    )
                copy.write_text(header)
                with read_raw(copy) as raw:
                    yield raw
            return
    with read_raw(path) as raw:
        yield raw


def process_record(r, archive, run, reg, verified):
    cache = run / "cache"
    cache.mkdir(exist_ok=True)
    target = cache / (r["id"] + ".npz")
    meta = target.with_suffix(".json")
    identity = digest([reg["fingerprint"], r["sources"]])
    if meta.exists():
        m = json.loads(meta.read_text())
        if m["identity"] != identity:
            raise ValueError("Stale v3 cache")
        if m["status"] == "ready" and file_hash(target) != m["sha256"]:
            raise ValueError("Corrupt v3 cache")
        return m
    m = {
        "id": r["id"],
        "identity": identity,
        "status": "excluded",
        "windows": 0,
        "reason": r["protocol_exclusion"],
    }
    if not m["reason"]:
        for s in r["sources"]:
            p = archive / s["path"]
            v = verified[s["path"]]
            if p.stat().st_size != v["bytes"] or file_hash(p) != s["sha256"]:
                raise ValueError("Source integrity failure")
        with safe_raw(archive / r["path"]) as raw:
            result, reason = extract(raw, r["interval"], r["role"])
        if reason:
            m["reason"] = reason
        elif not len(result["tokens"]):
            m.update(
                reason="no_accepted_windows",
                rejections=result["rejections"],
                attempted=result["attempted"],
            )
        else:
            np.savez(
                target, **{k: v for k, v in result.items() if k not in {"rejections", "attempted"}}
            )
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


def prepare(archive=ARCHIVE, run=RUN, benchmark=False):
    reg = json.loads((run / "registry.json").read_text())
    verified = receipts(archive)
    records = reg["records"]
    started = time.monotonic()
    out = []
    if benchmark:
        selected = []
        for role in ["supervised", "auxiliary"]:
            ds = sorted({r["dataset"] for r in records if r["role"] == role})
            selected.extend(
                next(r for r in records if r["dataset"] == d and not r["protocol_exclusion"])
                for d in ds
            )
        records = selected
    for r in records:
        t = time.monotonic()
        m = process_record(r, archive, run, reg, verified)
        out.append(m)
        print(
            r["dataset"],
            len(out),
            "/",
            len(records),
            m["status"],
            m["windows"],
            round(time.monotonic() - t, 2),
            flush=True,
        )
        if not benchmark:
            save(
                run / "preparation-progress.json",
                {
                    "completed": len(out),
                    "total": len(records),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
    if benchmark:
        bytes_done = sum(sum(s["bytes"] for s in r["sources"]) for r in records)
        bytes_total = sum(sum(s["bytes"] for s in r["sources"]) for r in reg["records"])
        estimate = (
            (time.monotonic() - started)
            * max(bytes_total / bytes_done, len(reg["records"]) / len(records))
            * 2
        )
        save(
            run / "preparation-benchmark.json",
            {
                "seconds": time.monotonic() - started,
                "records": len(records),
                "forecast_seconds": estimate,
                "margin": 2,
            },
        )
        print("Forecast hours", estimate / 3600, flush=True)
    else:
        freeze(run / "preparation.json", {"registry": reg["fingerprint"], "records": out})
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["build", "benchmark", "prepare"])
    a = p.parse_args()
    if a.action == "build":
        build()
    else:
        prepare(benchmark=a.action == "benchmark")
