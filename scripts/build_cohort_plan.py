"""Build a reviewed, file-level EEG archive plan. No waveform downloads.

OpenNeuro inventories must have been fetched with catalog_openneuro.py. ALS
labels are read from a representative EDF header for each bigP3BCI participant.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import quote

import httpx

from scripts.download_physionet import parse_checksums

HEALTHY = {"ds003775", "ds004796", "ds005385", "ds004148", "ds007987"}
PD_ONLY = {"ds003194", "ds003195", "ds007864", "ds007865"}
MIXED = {
    "ds002778",
    "ds003490",
    "ds003506",
    "ds003509",
    "ds004574",
    "ds004579",
    "ds004580",
    "ds004584",
    "ds007526",
    "ds008768",
    "ds004504",
    "ds006036",
    "ds005048",
}
GROUPS = {
    "PD": "parkinsons",
    "CTL": "healthy_control",
    "Control": "healthy_control",
    "HC": "healthy_control",
    "A": "alzheimers",
    "F": "frontotemporal_dementia",
    "C": "healthy_control",
    "Normal": "healthy_control",
    "Mild AD": "alzheimers",
    "Moderate AD": "alzheimers",
}


def participants(folder, dataset):
    path = folder / "participants.tsv"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines(), delimiter="\t"))
    result = {}
    for row in rows:
        subject = row["participant_id"]
        if dataset in HEALTHY:
            diagnosis, evidence = (
                "healthy_control",
                "Dataset README / source study describes healthy participants.",
            )
        elif dataset in PD_ONLY:
            diagnosis, evidence = (
                "parkinsons",
                "NeuroEPO PD trial cohort; placebo is treatment assignment, not healthy control.",
            )
        elif dataset == "ds002778":
            diagnosis = (
                "healthy_control"
                if subject.startswith("sub-hc")
                else "parkinsons"
                if subject.startswith("sub-pd")
                else "unclassified"
            )
            evidence = "Source participant identifier hc/pd and dataset README."
        else:
            field = next((k for k in row if k.lower() == "group"), None)
            label = row.get(field, "")
            diagnosis, evidence = (
                GROUPS.get(label, "unclassified"),
                f"participants.tsv {field}={label}",
            )
        family = dataset
        original = subject
        if dataset in {"ds003490", "ds003506", "ds003509"}:
            family, original = "unm-pd-2015", row.get("Original_ID", subject)
        elif dataset in {"ds004574", "ds004579", "ds004580", "ds004584"}:
            family, original = "iowa-pd-2017-2021", row.get("EEG", subject)
        elif dataset in {"ds004504", "ds006036"}:
            family = "ahepa-dementia-88"
        result[subject] = {
            "diagnosis": diagnosis,
            "label_evidence": evidence,
            "participant_key": f"{family}:{original}",
        }
    return result


def openneuro_items(folder, dataset):
    inventory = json.loads((folder / "inventory.json").read_text())
    description = json.loads((folder / "dataset_description.json").read_text())
    if description.get("License") != "CC0":
        raise ValueError(f"Review license before including {dataset}")
    people = participants(folder, dataset)
    items = []
    for obj in inventory["objects"]:
        relative = obj["key"].split("/", 1)[1]
        parts = relative.split("/")
        if any(p in {"derivatives", "sourcedata", ".git", ".datalad"} for p in parts):
            continue
        subject = next((p for p in parts if p.startswith("sub-")), None)
        is_eeg = "eeg" in parts
        if subject:
            person = people.get(subject, {"diagnosis": "unclassified"})
            if person["diagnosis"] == "unclassified":
                continue
            if not is_eeg and not relative.endswith((".tsv", ".json")):
                continue
        else:
            person = {}
            if "/" in relative and not (
                parts[0] == "code"
                and relative.endswith((".xls", ".xlsx", ".csv", ".tsv", ".json", ".txt", ".md"))
            ):
                continue
            if "/" not in relative and (
                relative.startswith(".")
                or not (
                    relative.endswith((".json", ".tsv", ".txt", ".md"))
                    or relative in {"README", "CHANGES", "LICENSE"}
                )
            ):
                continue
        binary_eeg = is_eeg and relative.lower().endswith(
            (".eeg", ".set", ".fdt", ".bdf", ".edf", ".fif")
        )
        group = (
            "healthy_control"
            if person.get("diagnosis") == "healthy_control"
            else "neurodegenerative"
            if person
            else "metadata"
        )
        items.append(
            {
                "url": "https://s3.amazonaws.com/openneuro.org/" + quote(obj["key"], safe="/"),
                "path": f"datasets/openneuro/{dataset}/{relative}",
                "provider": "openneuro",
                "dataset": dataset,
                "bytes": obj["size"],
                "etag": obj["etag"],
                "modified": obj["modified"],
                "hash_algorithm": "md5" if re.fullmatch(r"[0-9a-f]{32}", obj["etag"]) else None,
                "checksum": obj["etag"] if re.fullmatch(r"[0-9a-f]{32}", obj["etag"]) else None,
                "cohort": group if binary_eeg else "metadata",
                "is_eeg_signal": binary_eeg,
                "subject": subject,
                **person,
            }
        )
    return items, {
        "id": dataset,
        "provider": "openneuro",
        "title": description["Name"],
        "source": inventory["source"],
        "license": description["License"],
        "dataset_doi": description.get("DatasetDOI"),
        "listed_at": inventory["listed_at"],
        "note": "S3 object snapshot pinned by ETag; raw BIDS EEG and associated metadata only. No MRI or derivatives.",
    }


def bigp3_items(checksum_path, cache_path):
    checks = parse_checksums(checksum_path.read_text())
    representative = {}
    for _, name in checks:
        if name.endswith(".edf"):
            key = "/".join(name.split("/")[:3])
            representative.setdefault(key, name)
    labels = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    base = "https://physionet.org/files/bigp3bci/1.0.0/"
    with httpx.Client(timeout=60, follow_redirects=True) as client:

        def identify(pair):
            key, relative = pair
            if key in labels:
                return key, labels[key]
            # Streaming closes after the header even if a server ignores Range.
            with client.stream(
                "GET", base + quote(relative, safe="/"), headers={"Range": "bytes=0-255"}
            ) as response:
                response.raise_for_status()
                header = next(response.iter_bytes(256))
            patient = header[8:88].decode("ascii", errors="replace")
            label = (
                "healthy_control"
                if "NonALS" in patient
                else "als"
                if re.search(r"ALS_(?:\d+|X)", patient)
                else "unclassified"
            )
            return key, label

        with ThreadPoolExecutor(max_workers=4) as pool:
            labels.update(dict(pool.map(identify, representative.items())))
    cache_path.write_text(json.dumps(labels, indent=2))
    items = []
    for checksum, name in checks:
        signal = name.endswith(".edf")
        subject = "/".join(name.split("/")[:3]) if signal else None
        diagnosis = labels.get(subject) if signal else None
        if signal and diagnosis == "unclassified":
            continue
        items.append(
            {
                "url": base + quote(name, safe="/"),
                "path": "datasets/physionet/bigp3bci/1.0.0/" + name,
                "provider": "physionet",
                "dataset": "bigp3bci",
                "bytes": None,
                "hash_algorithm": "sha256",
                "checksum": checksum,
                "etag": None,
                "cohort": (
                    "healthy_control" if diagnosis == "healthy_control" else "neurodegenerative"
                )
                if signal
                else "metadata",
                "diagnosis": diagnosis,
                "is_eeg_signal": signal,
                "subject": subject,
                "participant_key": "bigp3bci:" + subject if subject else None,
                "label_evidence": "Representative participant EDF header ALS/NonALS field; source describes able-bodied and ALS groups.",
            }
        )
    print("bigP3BCI participant labels:", dict(Counter(labels.values())), flush=True)
    return items, {
        "id": "bigp3bci",
        "provider": "physionet",
        "title": "bigP3BCI P300 speller EEG",
        "source": "https://physionet.org/content/bigp3bci/1.0.0/",
        "license": "CC BY 4.0",
        "estimated_total_bytes": 44_600_000_000,
        "note": "ALS/NonALS labels from EDF headers. Do not infer neurologically normal EEG from a NonALS label alone.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openneuro", required=True, type=Path)
    parser.add_argument("--bigp3-checksums", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    items, datasets = [], []
    for ds in sorted(HEALTHY | PD_ONLY | MIXED):
        rows, meta = openneuro_items(args.openneuro / ds, ds)
        items.extend(rows)
        datasets.append(meta)
    rows, meta = bigp3_items(args.bigp3_checksums, args.output.parent / "bigp3-cohort-labels.json")
    items.extend(rows)
    datasets.append(meta)
    # Keep all matched controls; trim standalone healthy studies only by whole participant.
    # Reserve the entire bigP3BCI size against the control target conservatively.
    cap = 300_000_000_000 - 44_600_000_000
    healthy_total = sum(i["bytes"] or 0 for i in items if i["cohort"] == "healthy_control")
    omitted = []
    if healthy_total > cap:
        candidates = defaultdict(int)
        for i in items:
            if i["dataset"] == "ds004148" and i["cohort"] == "healthy_control":
                candidates[i["subject"]] += i["bytes"]
        for subject in sorted(candidates, reverse=True):
            if healthy_total <= cap:
                break
            healthy_total -= candidates[subject]
            omitted.append(subject)
        items = [i for i in items if not (i["dataset"] == "ds004148" and i["subject"] in omitted)]
    counts, sizes = Counter(), Counter()
    for item in items:
        counts[item["cohort"]] += 1
        sizes[item["cohort"]] += item["bytes"] or 0
    plan = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_bytes_per_cohort": 300_000_000_000,
        "scope": "Human scalp EEG. Source-described healthy/control participants and diagnosed Parkinson's, Alzheimer's, FTD, ALS. No epilepsy, post-arrest coma, MRI/MEG, duplicate derivatives, or unconfirmed MCI/risk cases counted as disease.",
        "datasets": datasets,
        "files": items,
        "known_bytes_by_cohort": dict(sizes),
        "file_counts_by_cohort": dict(counts),
        "bigp3_bytes_not_yet_split": 44_600_000_000,
        "healthy_budget_subset": {
            "dataset": "ds004148",
            "omitted_subjects": omitted,
            "rule": "Whole participants in descending ID order omitted to stay conservatively below 300 GB with full bigP3BCI allowance; not a statistical sample.",
        },
        "overlap_notes": [
            "ds004504 and ds006036 contain the same 88 people.",
            "New Mexico task datasets share Original_ID values; Iowa task datasets share EEG/ID values.",
            "NeuroEPO releases and other Parkinson's cohorts may overlap; reconcile before participant-level evaluation.",
            "ds007020 omitted because resting recordings may overlap the Iowa source cohort.",
            "No participant independence is implied by dataset count or raw byte count.",
        ],
    }
    args.output.write_text(json.dumps(plan, indent=2))
    print("Known EEG/metadata GB:", {k: round(v / 1e9, 3) for k, v in sizes.items()}, flush=True)
    print(
        "Additional bigP3BCI total: up to 44.6 GB; disease/control bytes measured during download.",
        flush=True,
    )
    print(f"Plan: {len(items)} files, {len(datasets)} datasets → {args.output}", flush=True)


if __name__ == "__main__":
    main()
