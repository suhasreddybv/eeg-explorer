"""Freeze participant identities and splits before reading signal windows."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SITES = {"ds004584": "iowa", "ds002778": "ucsd", "ds003490": "unm"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(
                "Manifest is immutable; use a new run directory for a changed protocol."
            )
    else:
        save(path, value)


def validate(registry):
    if registry["fingerprint"] != digest({k: v for k, v in registry.items() if k != "fingerprint"}):
        raise ValueError("Registry fingerprint mismatch")
    identities, files, hashes = {}, {}, {}
    for row in registry["records"]:
        split = row["split"]
        if row["dataset"] != "ds004584" and split != "external":
            raise ValueError("External-site contamination")
        for mapping, key in [(identities, row["participant_key"]), (files, row["path"])]:
            if key in mapping and mapping[key] != split:
                raise ValueError("Participant or recording leakage between splits")
            mapping[key] = split
        for source in row["sources"]:
            if source.get("is_eeg_signal") and source.get("checksum"):
                key = (source.get("hash_algorithm"), source["checksum"])
                if key in hashes and hashes[key] != split:
                    raise ValueError("Duplicate signal across splits")
                hashes[key] = split


def build(archive, seed=20260924):
    archive = Path(archive)
    plan = json.loads((archive / "download-plan.json").read_text())
    selected = [i for i in plan["files"] if i["dataset"] in SITES]
    people, demographics = {}, {}
    for ds in SITES:
        source = archive / "provenance" / "openneuro" / ds / "participants.tsv"
        rows = list(csv.DictReader(source.read_text().splitlines(), delimiter="\t"))
        for row in rows:
            demographics[(ds, row["participant_id"])] = row
    for item in selected:
        if not item.get("subject"):
            continue
        key = item["participant_key"]
        label = {"parkinsons": 1, "healthy_control": 0}.get(item["diagnosis"])
        if label is None:
            raise ValueError("Unreviewed diagnosis")
        person = (item["dataset"], item["subject"], label)
        if key in people and people[key] != person:
            raise ValueError("Ambiguous identity mapping")
        people[key] = person
    split = {key: "external" for key in people}
    for label in [0, 1]:
        keys = [k for k, (ds, _, y) in people.items() if ds == "ds004584" and y == label]
        keys.sort(key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
        n_hold = round(len(keys) * 0.15)
        if n_hold < 2:
            raise ValueError("Too few participants for stratified holdouts")
        for index, key in enumerate(keys):
            split[key] = (
                "internal_test"
                if index < n_hold
                else "validation"
                if index < 2 * n_hold
                else "train"
            )
    records = []
    for item in selected:
        # Only the planned resting acquisitions, never oddball/task periods.
        if (
            not item["path"].endswith((".vhdr", ".bdf", ".edf", ".set"))
            or "task-rest" not in item["path"].lower()
        ):
            continue
        ds, subject, label = people[item["participant_key"]]
        prefix = item["path"].rsplit("_eeg.", 1)[0]
        sources = [i for i in selected if i["path"].startswith(prefix + "_")]
        demo = demographics[(ds, subject)]
        records.append(
            {
                "recording_id": hashlib.sha256(item["path"].encode()).hexdigest()[:20],
                "path": item["path"],
                "dataset": ds,
                "site": SITES[ds],
                "participant_key": item["participant_key"],
                "subject": subject,
                "label": label,
                "label_evidence": item["label_evidence"],
                "split": split[item["participant_key"]],
                "eye_condition": "eyes_open" if ds == "ds004584" else "requires_protocol_review",
                "age": demo.get("AGE", demo.get("age", "n/a")),
                "sex": demo.get("GENDER", demo.get("gender", demo.get("sex", "n/a"))),
                "medication": "unknown; audit source session metadata",
                "sources": sources,
            }
        )
    registry = {
        "schema_version": 1,
        "protocol": "pd-rest-local-v1",
        "seed": seed,
        "split_policy": "Iowa stratified 70/15/15 by canonical participant; UCSD and UNM fully reserved",
        "excluded_datasets": {
            "ds008768": "Unresolved overlap with Iowa; excluded from all fitting/evaluation"
        },
        "source_metadata_hashes": {
            ds: hashlib.sha256(
                (archive / "provenance" / "openneuro" / ds / "participants.tsv").read_bytes()
            ).hexdigest()
            for ds in SITES
        },
        "records": sorted(records, key=lambda r: r["path"]),
    }
    registry["fingerprint"] = digest(registry)
    validate(registry)
    return registry


def summary(registry):
    groups = defaultdict(set)
    for r in registry["records"]:
        groups[(r["split"], r["label"])].add(r["participant_key"])
    return {
        f"{split}:{'PD' if label else 'control'}": len(keys)
        for (split, label), keys in sorted(groups.items())
    }
