"""Turku (OSF pehj9) locked-site handling: download | seal | evaluate.

download: page the OSF API, fetch the 40 raw eyes-open .set files plus
InfoAboutParticipants.xlsx and the preprocessing .m file into
/Volumes/Extreme Pro/EEG data/locked-sites/osf-pehj9/raw/, with a
manifest.json of sha256/bytes/OSF version. Resumes partial files and
verifies byte counts before accepting a download as complete.

seal: read InfoAboutParticipants.xlsx once, inside this process, without
printing it. Write labels (participant number -> PD/control, plus age/sex
if present) to locked-sites/osf-pehj9/sealed/labels.json, and record its
sha256 in data/pd-runs/v4/locked-site-seal.json. load_labels() must raise
unless both outer-models-lock.json and final-models-lock.json exist in
data/pd-runs/v4/.

evaluate: the one-shot Turku evaluation (Stage 11). Refuses a second run
without --i-understand-this-is-a-repeat, which also stamps results as
non-confirmatory. Not implemented until Stage 11.

overlap_audit: confirms no Turku raw-file sha256 equals any hash in the
main archive's verified-files.jsonl, and records that ds008875 (the Iowa
Visser/Narayanan/Uc release) is excluded from v4 because of likely overlap
with ds008768/ds004584 -- its object inventory is checked via the
OpenNeuro API only, never downloaded.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md for the exact rules this module must implement.

Only downloads the *raw* eyesOpen.set files (never *_PREP_preprocessed.set,
which carries the Turku group's own pipeline shift), plus
InfoAboutParticipants.xlsx and the preprocessing .m file for provenance.
"""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ARCHIVE = Path("/Volumes/Extreme Pro/EEG data")
LOCKED_ROOT = ARCHIVE / "locked-sites" / "osf-pehj9"
RAW_DIR = LOCKED_ROOT / "raw"
SEALED_DIR = LOCKED_ROOT / "sealed"
V4_RUN = Path("data/pd-runs/v4")

OSF_NODE = "pehj9"
OSF_API = "https://api.osf.io/v2"
DATA_FOLDER_URL = f"{OSF_API}/nodes/{OSF_NODE}/files/osfstorage/5ead625a62d4ab004d6cec1c/"
ROOT_URL = f"{OSF_API}/nodes/{OSF_NODE}/files/osfstorage/"
LICENCE_NOTE = (
    "OSF project pehj9 (Railo et al., University of Turku) shows no licence. "
    "Internal, non-commercial research evaluation only; no redistribution; "
    "contact Dr Henry Railo (University of Turku) before any commercial or "
    "publication use."
)


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "eeg-explorer-v4-locked-site/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _list_folder(url):
    items = []
    while url:
        d = _get_json(url)
        items.extend(d["data"])
        url = d["links"].get("next")
    return items


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


EXPECTED_RAW = {"eyesOpen": 40, "eyesClosed": 44}


def discover_targets(condition="eyesOpen"):
    """The raw <NN>_<condition>.set files (never *_PREP_preprocessed.set),
    plus InfoAboutParticipants.xlsx and the preprocessing .m file, each
    with the OSF-reported size/sha256/download link/version.

    eyesOpen is the locked-site test condition (40 files). eyesClosed (44
    files, same participants) is used only for the protocol v4.2 condition-
    shift diagnostic; it is never a fresh test."""
    root_items = {i["attributes"]["name"]: i for i in _list_folder(ROOT_URL)}
    data_items = _list_folder(DATA_FOLDER_URL)
    raw = sorted(
        (i for i in data_items if i["attributes"]["name"].endswith(f"_{condition}.set")),
        key=lambda i: i["attributes"]["name"],
    )
    if len(raw) != EXPECTED_RAW[condition]:
        raise ValueError(
            f"Expected {EXPECTED_RAW[condition]} raw {condition}.set files, found {len(raw)}; "
            "stop and ask before proceeding (source layout may have changed)."
        )
    targets = []
    for item in raw:
        targets.append(("raw", item))
    for name in ["InfoAboutParticipants.xlsx", "EEG_PD_restingstate_preprocess.m"]:
        if name not in root_items:
            raise ValueError(f"Expected root file {name} missing; stop and ask.")
        targets.append(("provenance", root_items[name]))
    return targets


def _download_one(item, dest):
    a = item["attributes"]
    expected_bytes = a["size"]
    expected_sha256 = a["extra"]["hashes"].get("sha256")
    content_url = item["links"]["download"]
    if dest.exists() and dest.stat().st_size == expected_bytes:
        if expected_sha256 and _sha256_file(dest) == expected_sha256:
            return {"status": "already_verified", "bytes": expected_bytes, "sha256": expected_sha256}
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(content_url, headers={"User-Agent": "eeg-explorer-v4-locked-site/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    got_bytes = tmp.stat().st_size
    if got_bytes != expected_bytes:
        raise ValueError(f"{dest.name}: byte count mismatch ({got_bytes} != {expected_bytes})")
    got_sha256 = _sha256_file(tmp)
    if expected_sha256 and got_sha256 != expected_sha256:
        raise ValueError(f"{dest.name}: sha256 mismatch against OSF-reported hash")
    tmp.replace(dest)
    return {"status": "downloaded", "bytes": got_bytes, "sha256": got_sha256}


def download(condition="eyesOpen"):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets = discover_targets(condition)
    manifest = {"osf_node": OSF_NODE, "condition": condition, "licence_note": LICENCE_NOTE, "files": []}
    for role, item in targets:
        a = item["attributes"]
        dest = RAW_DIR / a["name"]
        started = time.time()
        result = _download_one(item, dest)
        manifest["files"].append(
            {
                "name": a["name"],
                "role": role,
                "bytes": result["bytes"],
                "sha256": result["sha256"],
                "osf_file_id": item["id"],
                "osf_current_version": item["attributes"]["current_version"],
                "status": result["status"],
                "download_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            }
        )
        print(a["name"], result["status"], result["bytes"], flush=True)
    manifest["total_bytes"] = sum(f["bytes"] for f in manifest["files"])
    manifest["total_files"] = len(manifest["files"])
    # The eyes-open manifest backs the sealed one-shot and must never be
    # overwritten by the diagnostic condition's download.
    manifest_path = LOCKED_ROOT / ("manifest.json" if condition == "eyesOpen" else f"manifest-{condition}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest written: {manifest_path}", flush=True)
    return manifest


def find_column(header, *names):
    for name in names:
        for i, h in enumerate(header):
            if h.lower() == name.lower():
                return i
    return None


def resolve_pd_group_coding(data_rows, group_col, ledd_col, duration_col):
    """The 'PD group' column is 0/1 with the mapping undocumented anywhere in
    the source files. Resolve it from internal consistency instead of
    guessing: LEDD (a Parkinson's medication dose) and PD duration are
    clinical fields with no meaning for a healthy control, so whichever
    group value has them populated is PD. Raises rather than guesses if
    that invariant does not hold cleanly."""
    group_values = {r[group_col] for r in data_rows if r[group_col] is not None}
    if group_values != {0, 1}:
        raise ValueError(
            f"Unexpected 'PD group' values {group_values}; stop and ask rather than guess "
            "the label mapping."
        )
    ledd_by_group = {
        g: sum(1 for r in data_rows if r[group_col] == g and ledd_col is not None and r[ledd_col] not in (None, ""))
        for g in group_values
    }
    duration_by_group = {
        g: sum(
            1
            for r in data_rows
            if r[group_col] == g and duration_col is not None and r[duration_col] not in (None, "")
        )
        for g in group_values
    }
    pd_group_value = max(ledd_by_group, key=ledd_by_group.get)
    control_group_value = next(g for g in group_values if g != pd_group_value)
    if (
        ledd_by_group[pd_group_value] == 0
        or ledd_by_group[control_group_value] != 0
        or duration_by_group[control_group_value] != 0
    ):
        raise ValueError(
            "Could not resolve 'PD group' 0/1 coding from LEDD/PD-duration population "
            f"(LEDD={ledd_by_group}, duration={duration_by_group}); stop and ask rather "
            "than guess the label mapping."
        )
    return pd_group_value, control_group_value, ledd_by_group, duration_by_group


def seal():
    """Read InfoAboutParticipants.xlsx once and write sealed labels. Never
    prints participant-level content."""
    import openpyxl

    manifest = json.loads((LOCKED_ROOT / "manifest.json").read_text())
    xlsx_path = RAW_DIR / "InfoAboutParticipants.xlsx"
    if not xlsx_path.exists():
        raise ValueError("InfoAboutParticipants.xlsx not downloaded; run download() first")
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in rows[0]]

    id_col = find_column(header, "id", "Subject", "Participant", "Number", "Subject ID", "subject_id")
    group_col = find_column(header, "PD group", "Group", "Diagnosis", "group")
    age_col = find_column(header, "age", "Age")
    sex_col = find_column(header, "sex", "Sex", "Gender", "gender")
    ledd_col = find_column(header, "LEDD")
    duration_col = find_column(header, "PD duration")
    if id_col is None or group_col is None:
        raise ValueError(
            "Could not identify participant-id/diagnosis columns in InfoAboutParticipants.xlsx; "
            "stop and ask before guessing the schema."
        )
    data_rows = [r for r in rows[1:] if r[id_col] is not None]
    pd_group_value, control_group_value, ledd_by_group, duration_by_group = resolve_pd_group_coding(
        data_rows, group_col, ledd_col, duration_col
    )
    labels = {}
    for row in data_rows:
        pid = str(row[id_col]).strip()
        label = 1 if row[group_col] == pd_group_value else 0
        entry = {"label": label}
        if age_col is not None and row[age_col] is not None:
            entry["age"] = row[age_col]
        if sex_col is not None and row[sex_col] is not None:
            entry["sex"] = row[sex_col]
        labels[pid] = entry
    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    sealed_path = SEALED_DIR / "labels.json"
    sealed_path.write_text(
        json.dumps(
            {
                "source_sha256": next(
                    f["sha256"] for f in manifest["files"] if f["name"] == "InfoAboutParticipants.xlsx"
                ),
                "n_labels": len(labels),
                "labels": labels,
            },
            indent=2,
        )
    )
    seal_record = {
        "sealed_labels_sha256": _sha256_file(sealed_path),
        "n_labels": len(labels),
        "pd_group_coding": {
            "column": "PD group",
            "pd_group_value": pd_group_value,
            "control_group_value": control_group_value,
            "resolved_via": (
                "internal consistency, not documented anywhere in the source files: LEDD "
                "and PD duration (clinical fields with no meaning for a healthy control) "
                "are populated only for PD group value "
                f"{pd_group_value} (n={ledd_by_group[pd_group_value]} with LEDD, "
                f"{duration_by_group[pd_group_value]} with PD duration), never for "
                f"{control_group_value}"
            ),
        },
        "note": "Labels sealed; load_labels() refuses without both outer/final model locks.",
    }
    V4_RUN.mkdir(parents=True, exist_ok=True)
    (V4_RUN / "locked-site-seal.json").write_text(json.dumps(seal_record, indent=2))
    print(f"sealed {len(labels)} labels -> {sealed_path}", flush=True)
    return seal_record


def load_labels():
    if not (V4_RUN / "outer-models-lock.json").exists():
        raise RuntimeError("Turku labels sealed until outer-models-lock.json exists (Stage 7)")
    if not (V4_RUN / "final-models-lock.json").exists():
        raise RuntimeError("Turku labels sealed until final-models-lock.json exists (Stage 11)")
    return json.loads((SEALED_DIR / "labels.json").read_text())


def verify_format():
    """Stage 1a format check: mne.io.read_raw_eeglab must open every raw
    file, the channel names must map to the 17 v3 channels, and sampling
    rate/duration must be sane. Label-free; stops rather than guesses if
    any of the 17 channels is missing or ambiguous."""
    import mne
    import warnings

    from pd_pipeline.cohort_v3 import CONFIG, channel_map

    warnings.filterwarnings("ignore")
    # exFAT/macOS AppleDouble sidecar files (._*) pollute globs on this drive.
    files = sorted(f for f in RAW_DIR.glob("*_eyesOpen.set") if not f.name.startswith("._"))
    if len(files) != 40:
        raise ValueError(f"Expected 40 raw files on disk, found {len(files)}")
    sfreqs, durations, failures = set(), [], []
    for f in files:
        try:
            raw = mne.io.read_raw_eeglab(str(f), preload=False, verbose=False)
        except Exception as e:
            failures.append({"file": f.name, "error": str(e)})
            continue
        sfreqs.add(raw.info["sfreq"])
        durations.append(raw.n_times / raw.info["sfreq"])
        mapping = channel_map(raw.ch_names)
        missing = [c for c in CONFIG["channels"] if c.lower() not in mapping]
        if missing:
            failures.append({"file": f.name, "error": f"missing channels {missing}"})
    report = {
        "n_files_checked": len(files),
        "n_failures": len(failures),
        "failures": failures,
        "sampling_rates_hz": sorted(sfreqs),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "mean": float(sum(durations) / len(durations)) if durations else None,
            "max": max(durations) if durations else None,
        },
        "all_17_v3_channels_present_no_alias_needed": len(failures) == 0,
        "note": "Raw unprocessed .set; amplitude/QC not evaluated here (Stage 4).",
    }
    V4_RUN.mkdir(parents=True, exist_ok=True)
    (V4_RUN / "turku-format-verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise ValueError(
            f"{len(failures)} Turku file(s) failed format verification; stop and ask "
            "rather than guessing channel positions."
        )
    return report


def evaluate(repeat=False):
    """The one-shot locked-site evaluation.

    Verifies both locks and the seal hash, opens the sealed labels, scores
    P1/P2/P3, and writes locked-site-results.json. Refuses a second run
    unless `repeat=True`, which also stamps the output as non-confirmatory.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    from pd_pipeline import compare_v4 as cv
    from pd_pipeline.experiments import wilson
    from pd_pipeline.registry import save

    results_path = V4_RUN / "locked-site-results.json"
    if results_path.exists() and not repeat:
        raise RuntimeError(
            "Turku has already been evaluated. This is a one-shot test: re-running makes "
            "the result non-confirmatory. Pass --i-understand-this-is-a-repeat to proceed."
        )
    for required in ("outer-models-lock.json", "final-models-lock.json"):
        if not (V4_RUN / required).exists():
            raise RuntimeError(f"{required} missing; refusing to open the locked site")
    seal = json.loads((V4_RUN / "locked-site-seal.json").read_text())
    sealed_path = SEALED_DIR / "labels.json"
    if _sha256_file(sealed_path) != seal["sealed_labels_sha256"]:
        raise RuntimeError("Sealed labels file has changed since it was sealed")

    labels = load_labels()["labels"]
    # Turku filenames are zero-padded ("02") while the spreadsheet keys the
    # same participant unpadded ("2"). Matching on the raw string silently
    # dropped every single-digit participant, so normalise both sides.
    def _normalise(key):
        key = str(key).strip()
        return str(int(key)) if key.isdigit() else key

    labels = {_normalise(k): v for k, v in labels.items()}
    people = cv.load_people(role="locked_test")
    unmatched = []
    scored = []
    for person in people:
        subject = _normalise(person["key"].split(":", 1)[1])
        entry = labels.get(subject)
        if entry is None:
            unmatched.append(person["key"])
            continue
        person = dict(person)
        person["y"] = int(entry["label"])
        person["age"] = float(entry.get("age")) if entry.get("age") is not None else float("nan")
        sex = str(entry.get("sex", "")).strip().upper()
        person["sex"] = {"M": 0.0, "F": 1.0, "1": 1.0, "0": 0.0}.get(sex, float("nan"))
        scored.append(person)

    y = np.array([p["y"] for p in scored])
    report = {
        "protocol_version": "v4.1",
        "confirmatory": not repeat,
        "n_scored": len(scored),
        "n_source_recordings": 40,
        "coverage": {
            "scored": len(scored),
            "pd": int((y == 1).sum()),
            "control": int((y == 0).sum()),
            "excluded_before_labels": 40 - len(scored),
            "note": (
                "QC exclusions were determined label-free, before any label was opened: "
                "13 recordings produced no accepted window and 3 fell below A2's 10-window "
                "minimum."
            ),
        },
        "pipelines": {},
        "unmatched_after_qc": unmatched,
    }
    if len(set(y)) < 2:
        report["error"] = "Only one diagnosis class survived QC; AUROC is undefined."
        save(results_path, report)
        return report

    from pd_pipeline.reduced_v4 import load_montage_people

    rng = np.random.default_rng(20261001)
    for tag in ("P1", "P2", "P3"):
        target = load_montage_people(scored, "M4T") if tag == "P3" else scored
        if len(target) != len(scored):
            report["pipelines"][tag] = {"error": "missing montage features"}
            continue
        probabilities, meta = cv.score_with_final(tag, target)
        y_t = np.array([p["y"] for p in target])
        auroc = float(roc_auc_score(y_t, probabilities))
        boot = []
        for _ in range(2000):
            idx = np.concatenate(
                [rng.choice(np.flatnonzero(y_t == c), (y_t == c).sum(), replace=True) for c in (0, 1)]
            )
            boot.append(roc_auc_score(y_t[idx], probabilities[idx]))
        threshold = meta["threshold"]
        predicted = probabilities >= threshold
        tp = int(((y_t == 1) & predicted).sum())
        fn = int(((y_t == 1) & ~predicted).sum())
        tn = int(((y_t == 0) & ~predicted).sum())
        fp = int(((y_t == 0) & predicted).sum())
        report["pipelines"][tag] = {
            "pipeline": meta["tag"] + " / " + meta["regime"],
            "auroc": auroc,
            "auroc_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "pr_auc": float(average_precision_score(y_t, probabilities)),
            "brier": float(brier_score_loss(y_t, probabilities)),
            "threshold": threshold,
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "sensitivity_95ci_wilson": wilson(tp, max(tp + fn, 1)),
            "specificity_95ci_wilson": wilson(tn, max(tn + fp, 1)),
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }

    ages = np.array([p["age"] for p in scored], dtype="float64")
    sexes = np.array([p["sex"] for p in scored], dtype="float64")
    report["balance"] = {
        "age_mean_pd": float(np.nanmean(ages[y == 1])) if np.isfinite(ages[y == 1]).any() else None,
        "age_mean_control": float(np.nanmean(ages[y == 0])) if np.isfinite(ages[y == 0]).any() else None,
        "female_fraction_pd": float(np.nanmean(sexes[y == 1])) if np.isfinite(sexes[y == 1]).any() else None,
        "female_fraction_control": float(np.nanmean(sexes[y == 0])) if np.isfinite(sexes[y == 0]).any() else None,
    }
    save(results_path, report)
    return report


DS008875 = {
    "dataset": "ds008875",
    "name": "Resting-state EEG in Parkinson's disease and controls",
    "authors": ["Simin J. Visser", "Nandakumar S. Narayanan", "Ergun Y. Uc"],
    "group": "Iowa (same group as ds008768/ds004584)",
    "excluded_reason": (
        "Likely participant overlap with ds008768/ds004584 (same Iowa group). "
        "Not downloaded into the main archive; not used as a v4 test site. May be "
        "considered for v5 *training* only after a signal-hash and metadata overlap "
        "audit against ds008768."
    ),
}


def overlap_audit():
    """Stage 1b/1c: confirm Turku was never used by v1-v3, and record the
    ds008875 exclusion. Writes data/pd-runs/v4/overlap-audit.json."""
    manifest = json.loads((LOCKED_ROOT / "manifest.json").read_text())
    turku_hashes = {f["sha256"] for f in manifest["files"] if f["role"] == "raw"}
    archive_hashes = set()
    verified_path = ARCHIVE / "verified-files.jsonl"
    with verified_path.open() as f:
        for line in f:
            row = json.loads(line)
            h = row.get("sha256") or row.get("source_checksum")
            if h:
                archive_hashes.add(h)
    collisions = sorted(turku_hashes & archive_hashes)
    ds008875_inventory = None
    try:
        req = urllib.request.Request(
            "https://openneuro.org/crn/graphql",
            data=json.dumps(
                {
                    "query": (
                        'query { dataset(id: "ds008875") { id, latestSnapshot '
                        "{ tag, size, description { Name Authors } } } }"
                    )
                }
            ).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "eeg-explorer-v4/0.1"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            ds008875_inventory = json.load(r)["data"]["dataset"]
    except Exception as e:
        ds008875_inventory = {"error": f"inventory listing unavailable: {e}"}
    report = {
        "turku_never_used_by_v1_v3": {
            "n_turku_raw_files": len(turku_hashes),
            "n_archive_hashes_checked": len(archive_hashes),
            "collisions": collisions,
            "passed": len(collisions) == 0,
        },
        "ds008875_excluded": {**DS008875, "openneuro_inventory": ds008875_inventory},
    }
    V4_RUN.mkdir(parents=True, exist_ok=True)
    (V4_RUN / "overlap-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["turku_never_used_by_v1_v3"], indent=2), flush=True)
    if not report["turku_never_used_by_v1_v3"]["passed"]:
        raise ValueError("Turku file hash collides with the main archive; stop and diagnose")
    return report


if __name__ == "__main__":
    {
        "download": lambda: download("eyesClosed" if "--eyes-closed" in sys.argv else "eyesOpen"),
        "seal": seal,
        "evaluate": lambda: evaluate(repeat="--i-understand-this-is-a-repeat" in sys.argv),
        "overlap_audit": overlap_audit,
        "verify_format": verify_format,
    }[sys.argv[1]]()
