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


if __name__ == "__main__":
    {"download": download}[sys.argv[1]]()
