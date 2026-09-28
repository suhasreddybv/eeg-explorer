"""v4 participant registry.

One session per person (A1), Turku locked-site records with sealed labels,
and the label-aware confound audit.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code
or writes into data/pd-runs/v3/. See docs/PD_PROTOCOL_V4.md (frozen before
any supervised fit) for the exact rules this module implements.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cohort_v2 import tsv
from .cohort_v3 import ARCHIVE
from .cohort_v3 import build as build_v3
from .registry import digest, freeze, save

RUN = Path("data/pd-runs/v4")
LOCKED_SITE_ROOT = ARCHIVE / "locked-sites" / "osf-pehj9"
# ds008768 subject folders hold sub-N_sessions.tsv with days_since_first_session.
IOWA_DATASET = "ds008768"
UCSD_DATASET = "ds002778"
UNM_DATASET = "ds003490"
TEL_AVIV_DATASET = "ds007526"


def _iowa_session_days(archive, subject):
    path = archive / "datasets" / "openneuro" / IOWA_DATASET / subject / f"{subject}_sessions.tsv"
    rows = tsv(path)
    return {r["session_id"]: float(r["days_since_first_session"]) for r in rows}


def select_a1_session(records, archive=ARCHIVE):
    """A1: exactly one session per person, for the four supervised sites
    only. Auxiliary records are untouched -- v4 keeps v3's all-duration
    rule there. Returns (filtered_records, {person: chosen_session})."""
    by_person = defaultdict(list)
    keep_ids = set()
    for r in records:
        if r["role"] == "supervised":
            by_person[r["person"]].append(r)
        else:
            keep_ids.add(r["id"])

    notes = {}
    for person, recs in by_person.items():
        dataset = recs[0]["dataset"]
        label = recs[0]["label"]
        if dataset == UCSD_DATASET:
            wanted = "ses-on" if label == 1 else "ses-hc"
            chosen = [r for r in recs if r["session"] == wanted]
        elif dataset == UNM_DATASET:
            chosen = [r for r in recs if r["medication"] == "ON"] if label == 1 else recs
        elif dataset == IOWA_DATASET:
            subject = person.split(":", 1)[1]
            days = _iowa_session_days(archive, subject)
            available = {r["session"]: r for r in recs}
            candidates = [s for s in days if s in available]
            if not candidates:
                raise ValueError(f"No _sessions.tsv session matches an available recording for {person}")
            earliest = min(candidates, key=lambda s: days[s])
            chosen = [available[earliest]]
        elif dataset == TEL_AVIV_DATASET:
            chosen = recs  # already a single recording per person (no ses- subfolders)
        else:
            raise ValueError(f"Unhandled supervised dataset {dataset} in A1 session selection")
        if len(chosen) != 1:
            raise ValueError(
                f"A1 session selection did not resolve to exactly one recording for {person} "
                f"({dataset}, label={label}): {len(chosen)} candidates -- stop and ask rather "
                "than guess."
            )
        keep_ids.add(chosen[0]["id"])
        notes[person] = chosen[0]["session"]
    return [r for r in records if r["id"] in keep_ids], notes


def turku_locked_records(archive=ARCHIVE):
    """Turku (OSF pehj9) records with label=None -- labels stay sealed
    (locked_site_v4.load_labels()) until both v4 model locks exist."""
    manifest_path = archive / "locked-sites" / "osf-pehj9" / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Turku manifest.json missing; run scripts.locked_site_v4 download first")
    manifest = json.loads(manifest_path.read_text())
    records = []
    for f in manifest["files"]:
        if f["role"] != "raw":
            continue
        stem = f["name"].rsplit("_eyesOpen.set", 1)[0]
        rel_path = f"locked-sites/osf-pehj9/raw/{f['name']}"
        records.append(
            {
                "id": digest(rel_path)[:20],
                "path": rel_path,
                "dataset": "osf-pehj9",
                "site": "turku",
                "person": f"osf-pehj9:{stem}",
                "role": "locked_test",
                "label": None,
                "interval": None,
                "protocol_exclusion": None,
                "session": "single",
                "medication": "unknown",
                "age": None,
                "sex": None,
                "sources": [{"path": rel_path, "sha256": f["sha256"], "bytes": f["bytes"]}],
            }
        )
    if len(records) != 40:
        raise ValueError(f"Expected 40 Turku locked_test records, built {len(records)}")
    return records


def build(archive=ARCHIVE, run=RUN):
    run.mkdir(parents=True, exist_ok=True)
    # Reuse cohort_v3.build's logic exactly, but write its intermediate
    # artifacts to a v4-owned scratch path -- never into data/pd-runs/v3/.
    v3_reg = build_v3(archive=archive, run=run / "_cohort_v3_reuse")
    records, session_notes = select_a1_session(v3_reg["records"], archive)
    records = records + turku_locked_records(archive)
    reg = {
        "records": records,
        "config": v3_reg["config"],
        "excluded_datasets": v3_reg["excluded_datasets"],
        "a1_session_selection": session_notes,
        "locked_sites": {
            "turku": {
                "dataset": "osf-pehj9",
                "n_records": sum(1 for r in records if r["role"] == "locked_test"),
                "labels_sealed": True,
            }
        },
        "excluded_overlap_candidates": {
            "ds008875": "Likely overlap with ds008768/ds004584 (Iowa group); not downloaded."
        },
        "source_v3_style_fingerprint": v3_reg["fingerprint"],
        "history": (
            "v4: A1 one-session-per-person applied to the four v3 supervised sites "
            "(cohort_v3.build reused unmodified for extraction/labelling); Turku (OSF "
            "pehj9) added as a label-sealed locked_test site; auxiliary sources "
            "unchanged from v3."
        ),
    }
    reg["fingerprint"] = digest(reg)
    freeze(run / "registry.json", reg)
    save(
        run / "usage-summary.json",
        {
            "supervised_people_by_site": {
                s: len({r["person"] for r in records if r["site"] == s})
                for s in sorted({r["site"] for r in records if r["role"] == "supervised"})
            },
            "locked_test_people": len({r["person"] for r in records if r["role"] == "locked_test"}),
            "auxiliary_records": sum(1 for r in records if r["role"] == "auxiliary"),
        },
    )
    return reg


def clinical_data(records, archive=ARCHIVE, run=RUN):
    """Iowa (from the chosen session's _sessions.tsv row) and Tel Aviv
    (from participants.tsv, which already carries moca/updrs/ledd) clinical
    fields, for the person kept by A1. Stored separately from registry.json
    -- used only in Stage 10 (cognition/reliability) and descriptive
    tables, never for model fitting or QC."""
    by_person = {}
    for r in records:
        if r["role"] == "supervised" and r["dataset"] in {IOWA_DATASET, TEL_AVIV_DATASET}:
            by_person[r["person"]] = r
    tel_aviv_rows = {
        row["participant_id"]: row
        for row in tsv(archive / "provenance" / "openneuro" / TEL_AVIV_DATASET / "participants.tsv")
    }
    clinical = {}
    for person, r in by_person.items():
        subject = person.split(":", 1)[1]
        if r["dataset"] == IOWA_DATASET:
            rows = {
                row["session_id"]: row
                for row in tsv(archive / "datasets" / "openneuro" / IOWA_DATASET / subject / f"{subject}_sessions.tsv")
            }
            row = rows.get(r["session"], {})
            clinical[person] = {
                "site": "iowa",
                "age_at_visit": row.get("age_at_visit"),
                "moca": row.get("moca"),
                "updrs3": row.get("updrs_part_iii"),
                "disease_duration": row.get("disease_duration"),
                "ledd": row.get("levodopa_equivalent_daily_dose"),
            }
        elif r["dataset"] == TEL_AVIV_DATASET:
            row = tel_aviv_rows.get(subject, {})
            clinical[person] = {
                "site": "tel_aviv",
                "moca": row.get("moca"),
                "updrs3": row.get("updrs_part_iii"),
                "updrs_total": row.get("updrs_total"),
                "disease_duration": row.get("disease_duration"),
                "ledd": row.get("ledd"),
            }
    save(run / "clinical.json", clinical)
    return clinical


def _normalize_sex(value):
    return {"M": 0.0, "F": 1.0, "MALE": 0.0, "FEMALE": 1.0}.get(
        str(value).strip().upper(), np.nan
    )


def _normalize_age(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def demographic_confound_baseline(records):
    """Leave-one-site-out logistic regression on age/sex only (no window
    features exist yet -- this runs before Stage 4 extraction). Mirrors
    experiments.audit()'s demographic baseline, adapted to v4's one-
    session-per-person people."""
    supervised = [r for r in records if r["role"] == "supervised"]
    by_person = {r["person"]: r for r in supervised}
    people = list(by_person.values())
    sites = sorted({p["site"] for p in people})
    result = {}
    for site in sites:
        train = [p for p in people if p["site"] != site]
        test = [p for p in people if p["site"] == site]
        y_train = np.array([p["label"] for p in train])
        y_test = np.array([p["label"] for p in test])
        if len(set(y_test)) != 2:
            result[site] = {"auroc": None, "note": "single-class test site"}
            continue
        x_train = np.array([[_normalize_age(p["age"]), _normalize_sex(p["sex"])] for p in train])
        x_test = np.array([[_normalize_age(p["age"]), _normalize_sex(p["sex"])] for p in test])
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ).fit(x_train, y_train)
        pred = model.predict_proba(x_test)[:, 1]
        result[site] = {"auroc": float(roc_auc_score(y_test, pred)), "n_test": len(test)}
    return result


def confound_audit(records, run=RUN):
    """Stage 3 §2.2: per site x diagnosis, session count (must now be 1 for
    everyone), age, sex; the demographic AUROC re-run; and a structural
    check that session count can no longer discriminate diagnosis (it is
    now constant by construction, unlike v3 where it was the confound that
    drove AUROC 1.0 at UCSD/UNM). Accepted-window count and acceptance
    fraction require Stage 4 QC extraction and are appended there."""
    supervised = [r for r in records if r["role"] == "supervised"]
    by_person = {}
    session_counts = Counter()
    for r in supervised:
        by_person.setdefault(r["person"], r)
        session_counts[r["person"]] += 1
    coverage = defaultdict(lambda: {"n": 0, "ages": [], "sexes": Counter(), "session_counts": Counter()})
    for person, r in by_person.items():
        key = (r["site"], "PD" if r["label"] == 1 else "control")
        bucket = coverage[key]
        bucket["n"] += 1
        age = _normalize_age(r["age"])
        if not np.isnan(age):
            bucket["ages"].append(age)
        bucket["sexes"][str(r["sex"])] += 1
        bucket["session_counts"][session_counts[person]] += 1
    table = []
    for (site, diagnosis), bucket in sorted(coverage.items()):
        table.append(
            {
                "site": site,
                "diagnosis": diagnosis,
                "n": bucket["n"],
                "age_mean": float(np.mean(bucket["ages"])) if bucket["ages"] else None,
                "age_sd": float(np.std(bucket["ages"])) if bucket["ages"] else None,
                "sex_counts": dict(bucket["sexes"]),
                "distinct_session_counts_present": sorted(bucket["session_counts"]),
            }
        )
    all_session_counts = set(session_counts.values())
    demographic = demographic_confound_baseline(records)
    report = {
        "coverage": table,
        "session_count_confound": {
            "distinct_values_across_all_supervised_people": sorted(all_session_counts),
            "structurally_eliminated": all_session_counts == {1},
            "note": (
                "v3's UCSD/New Mexico session-count confound (metadata AUROC 1.000; "
                "PD had repeated medication sessions, controls one) is eliminated by "
                "construction under A1: every supervised person now has exactly one "
                "session, so session count carries zero information and cannot achieve "
                "any AUROC above chance."
            ),
        },
        "demographic_auroc_by_site": demographic,
        "pending_stage_4": (
            "accepted-window count and acceptance-fraction confound baselines require "
            "QC-extracted windows and are appended to this report in Stage 4."
        ),
    }
    if all_session_counts != {1}:
        raise ValueError(
            f"A1 did not eliminate the session-count confound: distinct counts "
            f"{sorted(all_session_counts)} -- stop and ask rather than proceed."
        )
    save(run / "confound-audit.json", report)
    return report


if __name__ == "__main__":
    registry = build()
    clinical_data(registry["records"])
    confound_audit(registry["records"])
