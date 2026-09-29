"""BrainLat (Synapse syn51549340): second locked site, protocol v4.2.

download | seal | verify_format | evaluate

Access: BrainLat is CC-BY-4.0 with zero Synapse access requirements, but
downloading needs an authenticated Synapse account. This script never
takes a token as an argument or reads one from chat: synapseclient picks
it up from the standard ~/.synapseConfig file or the SYNAPSE_AUTH_TOKEN
environment variable, and the download command refuses to start without
one so nothing is ever attempted anonymously and silently fails halfway.

Label sealing mirrors Turku. Diagnosis is given by Synapse folder
membership (3_PD vs 5_HC); `download` records it straight to the sealed
file and `load_labels` refuses until both model locks exist. Subject IDs
repeat across BrainLat's sites, so every person key is site-qualified:
brainlat:<site>:<subject>.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

ARCHIVE = Path("/Volumes/Extreme Pro/EEG data")
LOCKED_ROOT = ARCHIVE / "locked-sites" / "synapse-brainlat"
RAW_DIR = LOCKED_ROOT / "raw"
SEALED_DIR = LOCKED_ROOT / "sealed"
V4_RUN = Path("data/pd-runs/v4")

PROJECT = "syn51549340"
EEG_FOLDER = "syn53972960"
GROUP_FOLDERS = {"3_PD": ("syn53600453", 1), "5_HC": ("syn53222486", 0)}
SITES = ("AR", "CL")  # Colombia (UV) contributes no PD EEG and is excluded
LICENCE_NOTE = (
    "BrainLat (Prado et al., Sci Data 2023), Creative Commons Attribution 4.0. "
    "Zero Synapse access requirements; account required to download. Research use only."
)


def person_key(site, subject):
    """Site-qualified key. BrainLat reuses subject numbers across sites
    (sub-40001 exists under both PD/AR and PD/CL), so the bare subject id
    is not an identity."""
    if site not in SITES:
        raise ValueError(f"Unknown BrainLat site {site!r}")
    return f"brainlat:{site}:{subject}"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _client():
    import synapseclient

    if not (os.environ.get("SYNAPSE_AUTH_TOKEN") or (Path.home() / ".synapseConfig").exists()):
        raise RuntimeError(
            "No Synapse credentials found. Put a personal access token (scopes: view, "
            "download) in ~/.synapseConfig or the SYNAPSE_AUTH_TOKEN environment "
            "variable. Never paste it into chat."
        )
    syn = synapseclient.Synapse(silent=True)
    syn.login(authToken=os.environ.get("SYNAPSE_AUTH_TOKEN"))
    return syn


def discover(syn):
    """Every subject folder under PD/{AR,CL} and HC/{AR,CL}, with its label.
    Returns [(group, site, subject, folder_synid, label)]."""
    out = []
    for group, (folder_id, label) in GROUP_FOLDERS.items():
        for site_entity in syn.getChildren(folder_id, includeTypes=["folder"]):
            site = site_entity["name"]
            if site not in SITES:
                continue
            for subject in syn.getChildren(site_entity["id"], includeTypes=["folder"]):
                out.append((group, site, subject["name"], subject["id"], label))
    return out


def download():
    """Download each subject's eeg/ folder and seal labels in the same pass.
    Nothing is printed per subject beyond the key and byte count."""
    syn = _client()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    subjects = discover(syn)
    keys = [person_key(site, subject) for _, site, subject, _, _ in subjects]
    if len(keys) != len(set(keys)):
        raise ValueError("Site-qualified keys collide; refusing to proceed")

    manifest = {"project": PROJECT, "licence_note": LICENCE_NOTE, "subjects": []}
    labels = {}
    for group, site, subject, folder_id, label in subjects:
        key = person_key(site, subject)
        dest = RAW_DIR / site / subject
        dest.mkdir(parents=True, exist_ok=True)
        files = []
        for entity in syn.getChildren(folder_id, includeTypes=["folder", "file"]):
            if entity["name"] != "eeg":
                continue
            for f in syn.getChildren(entity["id"], includeTypes=["file"]):
                local = syn.get(f["id"], downloadLocation=str(dest), ifcollision="keep.local")
                path = Path(local.path)
                files.append(
                    {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path), "synid": f["id"]}
                )
        manifest["subjects"].append({"key": key, "site": site, "subject": subject, "files": files})
        labels[key] = {"label": label, "site": site}
        print(key, sum(f["bytes"] for f in files), flush=True)

    (LOCKED_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sealed = SEALED_DIR / "labels.json"
    sealed.write_text(json.dumps({"n_labels": len(labels), "labels": labels}, indent=2))
    V4_RUN.mkdir(parents=True, exist_ok=True)
    (V4_RUN / "brainlat-seal.json").write_text(
        json.dumps(
            {
                "sealed_labels_sha256": _sha256_file(sealed),
                "n_labels": len(labels),
                "n_pd": sum(1 for v in labels.values() if v["label"] == 1),
                "n_hc": sum(1 for v in labels.values() if v["label"] == 0),
                "note": "Labels sealed at download from folder membership; load_labels refuses without both model locks.",
            },
            indent=2,
        )
    )
    print(f"sealed {len(labels)} labels", flush=True)
    return manifest


def load_labels():
    for required in ("outer-models-lock.json", "final-models-lock.json"):
        if not (V4_RUN / required).exists():
            raise RuntimeError(f"BrainLat labels sealed until {required} exists")
    return json.loads((SEALED_DIR / "labels.json").read_text())


# ---------------------------------------------------------------------------
# Pre-download inspection: does preprocessing track diagnosis?
# ---------------------------------------------------------------------------
# A first look at the folder listing showed PD files named *_reject.set
# (artifact-rejected by the BrainLat team) and HC files without the suffix.
# If PD and HC recordings were processed differently, a classifier could
# score on the processing signature instead of the disease -- the same leak
# the Turku PREP files were refused over. `inspect` settles this on four
# files before the full download and seal: it counts naming classes by
# group and site, then compares label-free processing signatures between
# one flagged and one plain file per group. The report never records which
# subject is in which group.

EEG_SUFFIXES = (".set", ".bdf", ".edf", ".vhdr")
INSPECTION_DIR = LOCKED_ROOT / "inspection"  # outside raw/: never enters the manifest


def listing(syn):
    """EEG file names per subject, no downloads. [{group, site, subject, folder_id, files}]"""
    rows = []
    for group, site, subject, folder_id, _label in discover(syn):
        names = []
        for entity in syn.getChildren(folder_id, includeTypes=["folder"]):
            if entity["name"] == "eeg":
                names = sorted(f["name"] for f in syn.getChildren(entity["id"], includeTypes=["file"]))
        rows.append({"group": group, "site": site, "subject": subject, "folder_id": folder_id, "files": names})
    return rows


def _eeg_name(row):
    hits = [n for n in row["files"] if n.lower().endswith(EEG_SUFFIXES)]
    return hits[0] if len(hits) == 1 else None


def naming_summary(rows):
    """Counts of preprocessing-flagged vs plain EEG file names by group and
    site, plus subjects with no or ambiguous EEG files. Pure; no labels
    beyond the group aggregate."""
    out = {}
    for row in rows:
        cell = out.setdefault(f"{row['group']}/{row['site']}", {"flagged": 0, "plain": 0, "none_or_ambiguous": 0})
        name = _eeg_name(row)
        if name is None:
            cell["none_or_ambiguous"] += 1
        elif PREPROCESSING_FLAG.search(name):
            cell["flagged"] += 1
        else:
            cell["plain"] += 1
    return out


def pick_inspection_sample(rows):
    """First flagged and first plain subject per group (any site), in sorted
    order so the choice is reproducible and not ours to make."""
    sample = []
    for group in GROUP_FOLDERS:
        for cls in ("flagged", "plain"):
            for row in sorted((r for r in rows if r["group"] == group), key=lambda r: (r["site"], r["subject"])):
                name = _eeg_name(row)
                if name is None:
                    continue
                if (cls == "flagged") == bool(PREPROCESSING_FLAG.search(name)):
                    sample.append({**row, "eeg_file": name, "naming_class": cls})
                    break
    return sample


def signature(path):
    """Label-free processing fingerprint of one EEG file. Everything here is
    something an artifact-rejection or filtering step would change and a
    diagnosis would not: boundary events, EEGLAB history/ICA fields, and
    PSD ratios at the high-pass, low-pass and mains edges."""
    import warnings

    import mne

    warnings.filterwarnings("ignore")
    try:
        data = mne.io.read_raw(str(path), preload=True, verbose=False)
        sig = {
            "structure": "continuous",
            "duration_s": round(data.n_times / data.info["sfreq"], 1),
            "n_boundary_events": int(sum("boundary" in d.lower() for d in data.annotations.description)),
        }
    except TypeError as e:
        # An epoched .set file is itself a processing signature: the raw
        # recording was segmented and (in EEGLAB's usual pipeline) rejected
        # epoch-wise. Record the structure rather than fail.
        if "trials" not in str(e):
            raise
        data = mne.io.read_epochs_eeglab(str(path), verbose=False)
        sig = {
            "structure": "epoched",
            "n_epochs": len(data),
            "epoch_len_s": round(data.times[-1] - data.times[0], 3),
            "duration_s": round(len(data) * (data.times[-1] - data.times[0]), 1),
        }
    sig.update(
        sfreq=float(data.info["sfreq"],),
        n_channels=len(data.ch_names),
        highpass_hz=float(data.info["highpass"]),
        lowpass_hz=float(data.info["lowpass"]),
    )
    if path.suffix.lower() == ".set":
        sig.update(_eeglab_fields(path))
    picks = mne.pick_types(data.info, eeg=True)
    psd = data.compute_psd(picks=picks, fmin=0.1, fmax=min(100.0, data.info["sfreq"] / 2 - 1), verbose=False)
    p, f = psd.get_data().reshape(-1, len(psd.freqs)).mean(axis=0), psd.freqs

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(p[m].mean()) if m.any() else float("nan")

    sig["psd_ratio_0p1_0p5_over_1_4"] = band(0.1, 0.5) / band(1, 4)  # ~0 if high-pass filtered
    sig["psd_ratio_60_90_over_20_30"] = band(60, 90) / band(20, 30)  # ~0 if low-pass filtered
    sig["psd_ratio_mains_50"] = band(49, 51) / band(45, 55)  # 1/5 ≈ 0.2 if notch-free and no mains noise
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sig.items()}


def _eeglab_fields(path):
    """EEGLAB keeps a processing history string and ICA/reject structs in
    the .set file; MNE drops them. Read them directly when the file is a
    classic MAT; v7.3 (HDF5) files are recorded as unreadable, not guessed."""
    try:
        from scipy.io import loadmat

        m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        eeg = m.get("EEG", m)
        get = (lambda k: getattr(eeg, k, None)) if not isinstance(eeg, dict) else eeg.get
        hist = get("history")
        hist = str(hist) if hist is not None else ""
        ica = get("icaweights")
        return {
            "eeglab_history_lines": hist.count("pop_") + hist.count("eeg_"),
            "eeglab_history_has_reject": bool(__import__("re").search(r"reject|clean_|pop_select", hist, __import__("re").I)),
            "eeglab_history_has_filter": "eegfilt" in hist,
            "eeglab_has_ica": ica is not None and getattr(ica, "size", 0) > 0,
        }
    except Exception as e:  # noqa: BLE001 -- recorded so the report says why
        return {"eeglab_fields": f"unreadable: {type(e).__name__}"}


def inspect():
    syn = _client()
    rows = listing(syn)
    summary = naming_summary(rows)
    sample = pick_inspection_sample(rows)
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    signatures = {}
    for s in sample:
        dest = INSPECTION_DIR / s["group"] / s["naming_class"]
        dest.mkdir(parents=True, exist_ok=True)
        eeg_id = None
        for entity in syn.getChildren(s["folder_id"], includeTypes=["folder"]):
            if entity["name"] == "eeg":
                eeg_id = entity["id"]
        for f in syn.getChildren(eeg_id, includeTypes=["file"]):
            # The .fdt companion holds the samples for a .set header; fetch both.
            if f["name"] == s["eeg_file"] or f["name"].lower().endswith(".fdt"):
                syn.get(f["id"], downloadLocation=str(dest), ifcollision="keep.local")
        signatures[f"{s['group']}/{s['site']}/{s['naming_class']}"] = signature(dest / s["eeg_file"])
    report = {
        "purpose": "Does preprocessing track diagnosis? Naming classes by group/site and label-free processing signatures of one flagged + one plain file per group.",
        "naming_by_group_site": summary,
        "n_subjects": len(rows),
        "signatures": signatures,
        "note": "Subject identities of the sampled files are deliberately not recorded.",
    }
    V4_RUN.mkdir(parents=True, exist_ok=True)
    (V4_RUN / "brainlat-inspection.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report




# ---------------------------------------------------------------------------
# Label-free format check, extraction, and the guarded one-shot
# ---------------------------------------------------------------------------

EC_STYLE_RUN = V4_RUN / "brainlat"  # separate cache; sealed Turku artefacts untouched
BIOSEMI_LABEL = __import__("re").compile(r"^[A-D]\d{1,2}$")


PREPROCESSING_FLAG = __import__("re").compile(r"reject|prep|clean|ica|filt|proc", __import__("re").I)


def _eeg_files():
    """One EEG file per subject. BrainLat's EEG folder is NOT BIDS-named
    (e.g. s40001_CH_PD_reject.set, s6_sub-100012_rs_eeg.set), so match any
    .set/.bdf/.edf/.vhdr and refuse ambiguity rather than pick silently."""
    out = []
    for site in SITES:
        for subject_dir in sorted((RAW_DIR / site).glob("sub-*")):
            hits = sorted(h for h in subject_dir.rglob("*") if h.suffix.lower() in (".set", ".bdf", ".edf", ".vhdr"))
            if len(hits) > 1:
                raise ValueError(f"{subject_dir}: {len(hits)} candidate EEG files; refusing to guess which is the recording")
            if hits:
                out.append((site, subject_dir.name, hits[0]))
    return out


def preprocessing_flags():
    """Label-free survey: which files carry names suggesting the BrainLat
    team already artifact-rejected/preprocessed them, by site. If PD and HC
    files were processed differently, diagnosis could leak through
    processing artefacts and no result here would be interpretable."""
    out = {}
    for site, subject, path in _eeg_files():
        out[person_key(site, subject)] = {"file": path.name, "flagged": bool(PREPROCESSING_FLAG.search(path.name))}
    return out


def verify_format():
    """Label-free. Opens every EEG file, checks all 17 v3 channels map
    through channel_map, records sampling rate and duration.

    Biosemi 128-channel caps are natively labelled A1..D32, not 10-20. If
    that is what the BIDS export kept, this function STOPS rather than
    guess: the protocol forbids electrode-position guesses, and the right
    fix is to derive positions from the dataset's own electrodes.tsv
    coordinates -- a separate, reviewed step."""
    import warnings

    import mne

    from pd_pipeline.cohort_v3 import CONFIG, channel_map

    warnings.filterwarnings("ignore")
    flags = preprocessing_flags()
    report = {"n_files": 0, "failures": [], "biosemi_abcd_labels": [], "sfreq": set(), "durations": [],
              "preprocessing_flagged_files": sum(1 for v in flags.values() if v["flagged"]),
              "preprocessing_flags_by_file": flags}
    for site, subject, path in _eeg_files():
        report["n_files"] += 1
        try:
            raw = mne.io.read_raw(str(path), preload=False, verbose=False)
        except Exception as e:  # noqa: BLE001 -- recorded, never swallowed
            report["failures"].append({"key": person_key(site, subject), "error": str(e)[:200]})
            continue
        names = raw.ch_names
        if sum(bool(BIOSEMI_LABEL.match(n)) for n in names) > len(names) / 2:
            report["biosemi_abcd_labels"].append(person_key(site, subject))
            continue
        mapping = channel_map(names)
        missing = [c for c in CONFIG["channels"] if c.lower() not in mapping]
        if missing:
            report["failures"].append({"key": person_key(site, subject), "error": f"missing {missing}"})
        report["sfreq"].add(float(raw.info["sfreq"]))
        report["durations"].append(raw.n_times / raw.info["sfreq"])
    report["sfreq"] = sorted(report["sfreq"])
    d = report.pop("durations")
    report["duration_seconds"] = {"min": min(d), "max": max(d), "mean": sum(d) / len(d)} if d else None
    V4_RUN.mkdir(parents=True, exist_ok=True)
    (V4_RUN / "brainlat-format-verification.json").write_text(json.dumps(report, indent=2))
    if report["biosemi_abcd_labels"]:
        raise RuntimeError(
            f"{len(report['biosemi_abcd_labels'])} files carry Biosemi A1..D32 labels, not 10-20 names. "
            "STOP AND ASK: positions must be derived from electrodes.tsv coordinates, never guessed."
        )
    if report["failures"]:
        raise RuntimeError(f"{len(report['failures'])} BrainLat files failed format verification; stop and ask.")
    return report


def _records():
    from pd_pipeline.registry import digest

    manifest = json.loads((LOCKED_ROOT / "manifest.json").read_text())
    by_key = {s["key"]: s for s in manifest["subjects"]}
    records = []
    for site, subject, path in _eeg_files():
        key = person_key(site, subject)
        rel = str(path.relative_to(ARCHIVE))
        entry = next((f for f in by_key.get(key, {}).get("files", []) if f["name"] == path.name), None)
        if entry is None:
            raise ValueError(f"{rel} not in the download manifest; refusing to extract unverified data")
        records.append(
            {
                "id": digest(rel)[:20], "path": rel, "dataset": "brainlat", "site": "brainlat_" + site,
                "person": key, "key": key,
                # QC as a supervised-grade site: all 17 channels required, not the auxiliary minimum of 12.
                "role": "supervised", "label": None, "interval": None, "protocol_exclusion": None,
                "sources": [{"path": rel, "sha256": entry["sha256"], "bytes": entry["bytes"]}],
            }
        )
    return records


def extract_and_featurise():
    """Windows under the frozen thresholds, then 17-channel and M4T
    features, into the brainlat cache. Labels are never read here."""
    import numpy as np

    from pd_pipeline import features_v4 as fv
    from pd_pipeline import reduced_v4 as rv
    from pd_pipeline.qc_v4 import process_record_extract

    thresholds = json.loads((V4_RUN / "qc-thresholds.json").read_text())
    fingerprint = json.loads((V4_RUN / "registry.json").read_text())["fingerprint"]
    EC_STYLE_RUN.mkdir(parents=True, exist_ok=True)
    coverage, ready = {}, {}
    for record in _records():
        p = ARCHIVE / record["path"]
        if p.stat().st_size != record["sources"][0]["bytes"] or _sha256_file(p) != record["sources"][0]["sha256"]:
            raise ValueError(f"Source integrity failure: {record['path']}")
        meta = process_record_extract(record, ARCHIVE, EC_STYLE_RUN, fingerprint, thresholds)
        status = meta["status"] if meta["status"] != "ready" or meta["windows"] >= 10 else "below_a2_minimum"
        coverage[record["key"]] = {"status": status, "windows": meta["windows"], "reason": meta.get("reason")}
        if status != "ready":
            continue
        feats = fv.person_features([record["id"]], run=EC_STYLE_RUN)
        with np.load(EC_STYLE_RUN / "windows" / (record["id"] + ".npz")) as d:
            montage = rv.montage_person_features(d["preref128"][:15], "M4T")
        np.savez(EC_STYLE_RUN / (record["key"].replace(":", "__") + ".npz"), **feats)
        np.savez(EC_STYLE_RUN / (record["key"].replace(":", "__") + ".M4T.npz"), **montage)
        ready[record["key"]] = record["site"]
    (V4_RUN / "brainlat-coverage-labelfree.json").write_text(json.dumps(coverage, indent=2))
    return ready


def _people(labels):
    import numpy as np

    from pd_pipeline.features_v4 import normative_features

    people, montage_people = [], []
    for path in sorted(EC_STYLE_RUN.glob("brainlat__*[!T].npz")):
        if path.name.endswith(".M4T.npz"):
            continue
        key = path.stem.replace("__", ":")
        lab = labels.get(key)
        if lab is None:
            continue
        with np.load(path) as f, np.load(path.with_suffix(".M4T.npz")) as m:
            f, m = {k: f[k] for k in f.files}, {k: m[k] for k in m.files}
        base = {"key": key, "site": "brainlat_" + lab["site"], "y": int(lab["label"]),
                "age": float("nan"), "sex": float("nan")}
        people.append({**base, **{k: f[k] for k in ("b1", "b2", "b4", "b5", "b6", "v3_rich")},
                       "normative": normative_features(f), "cov": f.get("b3_cov_mean")})
        montage_people.append({**base, **{k: m[k] for k in ("b1", "b2", "b4", "b5", "b6")},
                               "cov": m["cov"], "normative": normative_features(f)})
    return people, montage_people


def evaluate(repeat=False):
    """The one-shot for the second locked site. Same four guards as Turku:
    both locks, seal hash, repeat refusal, label-free coverage first."""
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    from pd_pipeline import compare_v4 as cv
    from pd_pipeline.experiments import wilson

    results_path = V4_RUN / "brainlat-results.json"
    if results_path.exists() and not repeat:
        raise RuntimeError("BrainLat has already been evaluated (one-shot). Pass --i-understand-this-is-a-repeat.")
    for required in ("outer-models-lock.json", "final-models-lock.json"):
        if not (V4_RUN / required).exists():
            raise RuntimeError(f"{required} missing; refusing to open the locked site")
    seal = json.loads((V4_RUN / "brainlat-seal.json").read_text())
    if _sha256_file(SEALED_DIR / "labels.json") != seal["sealed_labels_sha256"]:
        raise RuntimeError("Sealed labels file has changed since it was sealed")
    coverage = json.loads((V4_RUN / "brainlat-coverage-labelfree.json").read_text())

    labels = load_labels()["labels"]
    people, montage_people = _people(labels)
    y = np.array([p["y"] for p in people])
    report = {
        "protocol_version": "v4.2", "confirmatory": not repeat, "n_scored": len(people),
        "coverage": {"scored": len(people), "pd": int((y == 1).sum()), "control": int((y == 0).sum()),
                     "excluded_by_status": {s: sum(1 for c in coverage.values() if c["status"] == s) for s in set(c["status"] for c in coverage.values())},
                     "note": "QC/A2 decided label-free before labels were opened; exclusions are cross-tabulated by diagnosis only here."},
        "pipelines": {},
    }
    if len(set(y)) < 2:
        report["error"] = "Only one class survived QC"
        results_path.write_text(json.dumps(report, indent=2))
        return report
    rng = np.random.default_rng(SEED)
    for tag, target in (("P1", people), ("P2", people), ("P3", montage_people)):
        prob, meta = cv.score_with_final(tag, target)
        yt = np.array([p["y"] for p in target])
        boot = [roc_auc_score(yt[i], prob[i]) for i in (np.concatenate([rng.choice(np.flatnonzero(yt == c), (yt == c).sum(), replace=True) for c in (0, 1)]) for _ in range(2000))]
        pred = prob >= meta["threshold"]
        tp, fn = int(((yt == 1) & pred).sum()), int(((yt == 1) & ~pred).sum())
        tn, fp = int(((yt == 0) & ~pred).sum()), int(((yt == 0) & pred).sum())
        report["pipelines"][tag] = {
            "pipeline": f"{meta['tag']} / {meta['regime']}" + (" / M4T" if tag == "P3" else ""),
            "auroc": float(roc_auc_score(yt, prob)), "auroc_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "pr_auc": float(average_precision_score(yt, prob)), "brier": float(brier_score_loss(yt, prob)),
            "threshold": meta["threshold"], "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "sensitivity_95ci_wilson": wilson(tp, max(tp + fn, 1)), "specificity_95ci_wilson": wilson(tn, max(tn + fp, 1)),
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }
    report["condition_caveat"] = "BrainLat is eyes-closed; training and Turku are eyes-open. Interpret only through the Turku-EC control verdict (turku-eyesclosed-control.json)."
    results_path.write_text(json.dumps(report, indent=2))
    return report


SEED = 20261001
COMMANDS = {
    "inspect": inspect,  # run first: four files, label-free, decides whether download is worth doing
    "download": download,
    "verify_format": verify_format,
    "extract": extract_and_featurise,
    "evaluate": lambda: evaluate(repeat="--i-understand-this-is-a-repeat" in sys.argv),
}


if __name__ == "__main__":
    COMMANDS[sys.argv[1]]()
