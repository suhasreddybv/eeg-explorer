"""Expanded, immutable cohort; old Iowa releases never enter this comparison."""

import csv
import json
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from .registry import digest, freeze, save
from .preprocess import DEFAULT, receipts, file_hash, read_raw, windows_from_raw

DATASETS = {"ds008768": "iowa_expanded", "ds007526": "tel_aviv", "ds003490": "unm"}
CONFIG = {**DEFAULT, "version": 2, "channels": [c for c in DEFAULT["channels"] if c != "Cz"]}
SEED = 20260926


def tsv(path):
    return list(
        csv.DictReader(Path(path).read_text(encoding="utf-8-sig").splitlines(), delimiter="\t")
    )


def audit_archive(archive, run):
    plan = json.loads((archive / "download-plan.json").read_text())
    status = json.loads((archive / "download-status.json").read_text())
    if status["state"] != "complete" or status.get("retry_pending") or status.get("failures"):
        raise ValueError("Archive not complete")
    verified = receipts(archive)
    mismatches = []
    for item in plan["files"]:
        p = archive / item["path"]
        v = verified.get(item["path"])
        s = p.stat() if p.exists() else None
        if (
            not v
            or not s
            or s.st_size != v["bytes"]
            or s.st_mtime_ns != v["mtime_ns"]
            or (item["bytes"] is not None and s.st_size != item["bytes"])
            or v.get("etag") != item.get("etag")
            or v.get("source_checksum") != item.get("checksum")
        ):
            mismatches.append(item["path"])
    report = {
        "planned_files": len(plan["files"]),
        "verified_bytes": sum(verified[x["path"]]["bytes"] for x in plan["files"]),
        "mismatches": mismatches,
        "method": "All receipt identities, sizes and modification times; selected EEG files rehashed during preprocessing",
        "plan_sha256": file_hash(archive / "download-plan.json"),
    }
    save(run / "archive-audit.json", report)
    if mismatches:
        raise ValueError(f"Archive has {len(mismatches)} mismatches")
    return plan, verified


def eye_interval(events):
    """Keep only a contiguous run of explicitly labelled 1-second open-eye triggers."""
    points = sorted(
        float(x["onset"]) for x in events if x["trial_type"] == "Eyes Open: Every 1000 ms"
    )
    if not points:
        raise ValueError("No explicit eyes-open triggers")
    runs = [[points[0]]]
    for p in points[1:]:
        if 0.8 <= p - runs[-1][-1] <= 1.2:
            runs[-1].append(p)
        else:
            runs.append([p])
    valid = [r for r in runs if len(r) >= 50]
    if len(valid) != 1:
        raise ValueError("Ambiguous eyes-open interval")
    # Stop at the last positive trigger, not beyond documented open-eye time.
    return [valid[0][0], valid[0][-1]]


def build(archive, run):
    plan, verified = audit_archive(archive, run)
    records = []
    people = {}
    for ds, site in DATASETS.items():
        rows = tsv(archive / "provenance/openneuro" / ds / "participants.tsv")
        for row in rows:
            subject = row["participant_id"]
            key = f"{ds}:{subject}"
            label = {"PD": 1, "Control": 0, "HC": 0, "CTL": 0}[row.get("group", row.get("Group"))]
            people[key] = {**row, "label": label, "site": site}
            files = [f for f in plan["files"] if f["dataset"] == ds and f.get("subject") == subject]
            signals = sorted(
                [
                    f
                    for f in files
                    if "task-rest" in f["path"].lower() and f["path"].endswith((".vhdr", ".set"))
                ],
                key=lambda f: f["path"],
            )
            if ds != "ds003490":
                signals = signals[:1]  # earliest numbered visit, never best-QC visit
            for f in signals:
                stem = f["path"].rsplit("_eeg.", 1)[0]
                sources = [x for x in files if x["path"].startswith(stem + "_")]
                session = next((x for x in f["path"].split("/") if x.startswith("ses-")), None)
                medication = (
                    row.get("sess1_Med" if session == "ses-01" else "sess2_Med", "unknown")
                    if ds == "ds003490"
                    else "unknown"
                )
                interval = None
                protocol_error = None
                if ds == "ds003490":
                    try:
                        interval = eye_interval(tsv(archive / (stem + "_events.tsv")))
                    except ValueError as error:
                        protocol_error = str(error)
                records.append(
                    {
                        "recording_id": hashlib.sha256(f["path"].encode()).hexdigest()[:20],
                        "participant_key": key,
                        "dataset": ds,
                        "site": site,
                        "subject": subject,
                        "label": label,
                        "path": f["path"],
                        "protocol_error": protocol_error,
                        "sources": sources,
                        "age": row["age"],
                        "sex": row["sex"],
                        "medication": medication,
                        "interval": interval,
                        "split": "external",
                        "endpoint": "tel_aviv"
                        if ds == "ds007526"
                        else "unm_" + medication.lower()
                        if label
                        else "unm_control",
                    }
                )
    splits = {}
    for label in [0, 1]:
        keys = sorted(
            [k for k, p in people.items() if p["site"] == "iowa_expanded" and p["label"] == label],
            key=lambda k: digest([SEED, k]),
        )
        hold = round(len(keys) * 0.15)
        for i, k in enumerate(keys):
            splits[k] = "internal_test" if i < hold else "validation" if i < 2 * hold else "train"
    for r in records:
        if r["participant_key"] in splits:
            r["split"] = splits[r["participant_key"]]
            r["endpoint"] = r["split"]
    # Exact signal duplicates with different people are never allowed, even in one split.
    hashes = {}
    for r in records:
        for s in r["sources"]:
            if s["path"].endswith((".eeg", ".fdt", ".set")):
                h = verified[s["path"]]["sha256"]
                if h in hashes and hashes[h] != r["participant_key"]:
                    raise ValueError("Duplicate signal identity across participants")
                hashes[h] = r["participant_key"]
    reg = {
        "version": 2,
        "seed": SEED,
        "records": records,
        "metadata_sha256": {
            ds: file_hash(archive / "provenance/openneuro" / ds / "participants.tsv")
            for ds in DATASETS
        },
        "excluded": {
            "older_iowa": "ds004584/574/579/580 excluded for unresolved cross-release overlap; expanded release replaces them",
            "ucsd": "Previously examined pilot test; excluded from v2",
            "unm_tasks": "ds003506/ds003509 share original IDs with resting study and use tasks; excluded",
            "pd_only": "No within-study controls, repeat treatment arms unresolved; excluded",
        },
        "interpretation": "Iowa internal test is exploratory because overlap with examined pilot is unresolved. Tel Aviv and UNM are newly examined external sites.",
    }
    reg["fingerprint"] = digest(reg)
    freeze(run / "registry.json", reg)
    freeze(run / "preprocessing.json", CONFIG)
    print(
        "Frozen participants",
        dict(Counter((r["split"], r["label"]) for r in records if r["dataset"] != "ds003490")),
        flush=True,
    )
    return reg


def validate(reg):
    if reg["fingerprint"] != digest({k: v for k, v in reg.items() if k != "fingerprint"}):
        raise ValueError("Registry changed")
    identities = {}
    for r in reg["records"]:
        if r["participant_key"] in identities and identities[r["participant_key"]] != (
            r["split"],
            r["label"],
        ):
            raise ValueError("Participant leakage")
        identities[r["participant_key"]] = (r["split"], r["label"])
        if r["dataset"] != "ds008768" and r["split"] != "external":
            raise ValueError("External contamination")


def prepare(archive, run, external=False):
    reg = json.loads((run / "registry.json").read_text())
    validate(reg)
    if external and not (run / "all-models-lock.json").exists():
        raise ValueError("All models must be locked before test processing")
    config = json.loads((run / "preprocessing.json").read_text())
    if config != CONFIG:
        raise ValueError("Preprocessing changed")
    verified = receipts(archive)
    (run / "cache").mkdir(exist_ok=True)
    out = []
    for r in reg["records"]:
        if (r["split"] in ["train", "validation"]) == external:
            continue
        if r.get("protocol_error"):
            out.append(
                {
                    **{k: v for k, v in r.items() if k != "sources"},
                    "status": "excluded_protocol",
                    "windows": 0,
                }
            )
            continue
        target = run / "cache" / (r["recording_id"] + ".npz")
        meta = target.with_suffix(".json")
        sigs = [
            s
            for s in r["sources"]
            if s["path"].endswith((".eeg", ".vhdr", ".vmrk", ".set", ".fdt"))
        ]
        identity = digest(
            [reg["fingerprint"], CONFIG, [(s["path"], verified[s["path"]]["sha256"]) for s in sigs]]
        )
        if target.exists() and meta.exists():
            m = json.loads(meta.read_text())
            if m["cache_identity"] == identity and m["cache_sha256"] == file_hash(target):
                out.append(m)
                continue
        for s in sigs:
            if file_hash(archive / s["path"]) != verified[s["path"]]["sha256"]:
                raise ValueError("Source content changed")
        with read_raw(archive / r["path"]) as raw:
            if r["interval"]:
                raw.crop(tmin=r["interval"][0], tmax=r["interval"][1], include_tmax=False)
                # Current events contain no BAD annotations; resetting offset is explicit.
                raw.set_annotations(None)
            x, starts, rejected = windows_from_raw(raw, CONFIG)
        np.savez(target, windows=x, starts=starts)
        m = {k: v for k, v in r.items() if k != "sources"}
        m.update(
            status="ready" if len(x) >= CONFIG["min_windows"] else "excluded_qc",
            windows=len(x),
            rejected_windows=rejected,
            cache_identity=identity,
            cache_sha256=file_hash(target),
        )
        save(meta, m)
        out.append(m)
        print(r["site"], r["subject"], m["status"], len(x), flush=True)
        save(
            run / ("test-preparation.json" if external else "preparation.json"),
            {"records": out, "registry_fingerprint": reg["fingerprint"]},
        )
    save(
        run / ("test-preparation.json" if external else "preparation.json"),
        {"records": out, "registry_fingerprint": reg["fingerprint"]},
    )
    return out


def load(run, external=False):
    reg = json.loads((run / "registry.json").read_text())
    validate(reg)
    report = json.loads(
        (run / ("test-preparation.json" if external else "preparation.json")).read_text()
    )
    if report["registry_fingerprint"] != reg["fingerprint"]:
        raise ValueError("Wrong registry")
    expected = {
        r["recording_id"]
        for r in reg["records"]
        if (r["split"] in ["train", "validation"]) != external
    }
    if {r["recording_id"] for r in report["records"]} != expected:
        raise ValueError("Preparation incomplete")
    data = []
    references = {r["recording_id"]: r for r in reg["records"]}
    seen = {}
    for r in report["records"]:
        reference = references[r["recording_id"]]
        if any(
            r[k] != reference[k]
            for k in ["participant_key", "label", "split", "path", "dataset", "endpoint"]
        ):
            raise ValueError("Cached participant identity changed")
        if r["status"] not in {"ready", "excluded_qc", "excluded_protocol"}:
            raise ValueError("Unresolved preprocessing record")
        if r["status"] != "ready":
            continue
        p = run / "cache" / (r["recording_id"] + ".npz")
        if file_hash(p) != r["cache_sha256"]:
            raise ValueError("Cache changed")
        with np.load(p, allow_pickle=False) as cache:
            x = cache["windows"]
        key = hashlib.sha256(x.tobytes()).hexdigest()
        if key in seen and seen[key] != r["participant_key"]:
            raise ValueError("Duplicate processed signal across participants")
        seen[key] = r["participant_key"]
        data.append({**r, "x": x})
    return data


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["build", "prepare", "test-prepare"])
    p.add_argument("--archive", type=Path, default=Path("/Volumes/Extreme Pro/EEG data"))
    p.add_argument("--run", type=Path, default=Path("data/pd-runs/v2"))
    a = p.parse_args()
    if a.action == "build":
        build(a.archive, a.run)
    else:
        prepare(a.archive, a.run, a.action == "test-prepare")
