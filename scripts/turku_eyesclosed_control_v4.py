"""Turku eyes-closed condition-shift control (protocol v4.2 §14.2).

Scores the 39 Turku eyes-CLOSED recordings on OSF — a subset of the same
people already tested eyes-open — with the identical frozen P1/P2/P3, then applies the
pre-registered decision rule to the EO−EC gap. This is a diagnostic on
labels that are already unsealed. It is NOT a new test: the output is
stamped non-confirmatory and never feeds any model.

Why it exists: BrainLat, the only remaining external site, is eyes-closed
against eyes-open training. Without this control a null BrainLat result
could not be attributed to site shift versus condition shift.
"""

import json
import sys
from pathlib import Path

import numpy as np

ARCHIVE = Path("/Volumes/Extreme Pro/EEG data")
LOCKED_ROOT = ARCHIVE / "locked-sites" / "osf-pehj9"
V4_RUN = Path("data/pd-runs/v4")
EC_RUN = V4_RUN / "ec-control"  # separate cache: never touches the sealed EO artefacts
SEED = 20261001

# §14.2 thresholds, fixed before any number existed.
SMALL_GAP = 0.05
CONFOUNDED_GAP = 0.10
EXPECTED_EC = 39  # distinct eyes-closed files on OSF (see locked_site_v4.EXPECTED_RAW)


def normalise_id(value):
    """'02' and '2' are the same Turku participant. This is the bug that
    cost the eyes-open one-shot its first run; never match on raw strings."""
    value = str(value).strip()
    return str(int(value)) if value.isdigit() else value


def verdict(delta):
    """delta = AUROC(EO) − AUROC(EC) for P1 on the same people.

    A *negative* delta (EC no worse than EO) is not a condition penalty and
    is classed with the small-gap case."""
    if delta > CONFOUNDED_GAP:
        return "confounded_by_condition"
    if delta > SMALL_GAP:
        return "partially_confounded"
    return "site_generalisation_interpretable"


def ec_records(manifest_path=LOCKED_ROOT / "manifest-eyesClosed.json"):
    from pd_pipeline.registry import digest

    manifest = json.loads(manifest_path.read_text())
    records = []
    for f in manifest["files"]:
        if f["role"] != "raw" or not f["name"].endswith("_eyesClosed.set"):
            continue
        stem = f["name"].rsplit("_eyesClosed.set", 1)[0]
        rel = f"locked-sites/osf-pehj9/raw/{f['name']}"
        records.append(
            {
                "id": digest(rel)[:20],
                "path": rel,
                "dataset": "osf-pehj9",
                "site": "turku",
                "person": f"osf-pehj9-ec:{stem}",
                "subject": normalise_id(stem),
                "role": "locked_test_ec",
                "label": None,
                "interval": None,
                "protocol_exclusion": None,
                "sources": [{"path": rel, "sha256": f["sha256"], "bytes": f["bytes"]}],
            }
        )
    # OSF holds 39 distinct eyes-closed files, not the 44 the paper implies;
    # the first listing double-counted one file and hid the shortfall.
    if len(records) != EXPECTED_EC or len({r["subject"] for r in records}) != EXPECTED_EC:
        raise ValueError(f"Expected {EXPECTED_EC} distinct eyes-closed records, built {len(records)}")
    return records


def _verify(record):
    manifest = json.loads((LOCKED_ROOT / "manifest-eyesClosed.json").read_text())
    by_name = {f["name"]: f for f in manifest["files"]}
    for s in record["sources"]:
        f = by_name[Path(s["path"]).name]
        p = ARCHIVE / s["path"]
        if p.stat().st_size != f["bytes"] or s["sha256"] != f["sha256"]:
            raise ValueError(f"Source integrity failure: {s['path']}")


def extract_and_featurise():
    """Windows under the frozen thresholds, then per-person 17-channel and
    M4T features. Uses EC_RUN so nothing under the eyes-open caches moves."""
    from pd_pipeline import features_v4 as fv
    from pd_pipeline import reduced_v4 as rv
    from pd_pipeline.qc_v4 import process_record_extract

    thresholds = json.loads((V4_RUN / "qc-thresholds.json").read_text())
    fingerprint = json.loads((V4_RUN / "registry.json").read_text())["fingerprint"]
    EC_RUN.mkdir(parents=True, exist_ok=True)
    out = {}
    for record in ec_records():
        _verify(record)
        meta = process_record_extract(record, ARCHIVE, EC_RUN, fingerprint, thresholds)
        if meta["status"] != "ready" or meta["windows"] < 10:  # A2, unchanged
            # A2 failures must not keep the QC status "ready": _people keys on it.
            status = meta["status"] if meta["status"] != "ready" else "below_a2_minimum"
            out[record["subject"]] = {"status": status, "windows": meta["windows"], "reason": meta.get("reason")}
            continue
        feats = fv.person_features([record["id"]], run=EC_RUN)
        with np.load(EC_RUN / "windows" / (record["id"] + ".npz")) as d:
            montage = rv.montage_person_features(d["preref128"][:15], "M4T")
        out[record["subject"]] = {"status": "ready", "windows": meta["windows"], "features": feats, "montage": montage}
    return out


def _people(entries, labels):
    from pd_pipeline.features_v4 import normative_features

    people, montage_people = [], []
    for subject, e in entries.items():
        if e["status"] != "ready" or "features" not in e or subject not in labels:
            continue
        lab = labels[subject]
        f, m = e["features"], e["montage"]
        sex = str(lab.get("sex", "")).strip().upper()
        base = {
            "key": f"osf-pehj9:{subject}",
            "site": "turku",
            "y": int(lab["label"]),
            "age": float(lab["age"]) if lab.get("age") is not None else float("nan"),
            "sex": {"M": 0.0, "F": 1.0, "1": 1.0, "0": 0.0}.get(sex, float("nan")),
        }
        people.append({**base, **{k: f[k] for k in ("b1", "b2", "b4", "b5", "b6", "v3_rich")}, "normative": normative_features(f), "cov": f.get("b3_cov_mean")})
        montage_people.append({**base, **{k: m[k] for k in ("b1", "b2", "b4", "b5", "b6")}, "cov": m["cov"], "normative": normative_features(f)})
    return people, montage_people


def _auroc_ci(y, p, rng, n=2000):
    from sklearn.metrics import roc_auc_score

    y, p = np.asarray(y), np.asarray(p)
    boot = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum(), replace=True) for c in (0, 1)])
        boot.append(roc_auc_score(y[idx], p[idx]))
    return float(roc_auc_score(y, p)), [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def run():
    from pd_pipeline import compare_v4 as cv
    from pd_pipeline.reduced_v4 import load_montage_people
    from scripts.locked_site_v4 import load_labels

    raw_labels = load_labels()["labels"]  # both locks exist; labels already unsealed by the EO one-shot
    labels = {normalise_id(k): v for k, v in raw_labels.items()}

    ec_people, ec_montage = _people(extract_and_featurise(), labels)

    # Eyes-open scores for the same people, recomputed with the frozen models
    # (deterministic; equals the corrected n=24 run). Not a new evaluation.
    eo_base = []
    for p in cv.load_people(role="locked_test"):
        subject = normalise_id(p["key"].split(":", 1)[1])
        if subject in labels:
            lab = labels[subject]
            sex = str(lab.get("sex", "")).strip().upper()
            eo_base.append({**p, "key": f"osf-pehj9:{subject}", "y": int(lab["label"]),
                            "age": float(lab["age"]) if lab.get("age") is not None else float("nan"),
                            "sex": {"M": 0.0, "F": 1.0, "1": 1.0, "0": 0.0}.get(sex, float("nan"))})
    eo_montage = load_montage_people(eo_base, "M4T")

    rng = np.random.default_rng(SEED)
    report = {"protocol_version": "v4.2", "confirmatory": False, "diagnostic": "Turku eyes-closed condition-shift control (§14.2)",
              "n_ec_scored": len(ec_people), "n_eo_scored": len(eo_base), "pipelines": {}}
    for tag, eo_set, ec_set in (("P1", eo_base, ec_people), ("P2", eo_base, ec_people), ("P3", eo_montage, ec_montage)):
        p_eo, _ = cv.score_with_final(tag, eo_set)
        p_ec, _ = cv.score_with_final(tag, ec_set)
        eo = {p["key"]: (p["y"], s) for p, s in zip(eo_set, p_eo)}
        ec = {p["key"]: (p["y"], s) for p, s in zip(ec_set, p_ec)}
        overlap = sorted(set(eo) & set(ec))
        y = [eo[k][0] for k in overlap]
        entry = {"n_overlap": len(overlap), "n_ec_all": len(ec)}
        if len(set(y)) == 2:
            a_eo, ci_eo = _auroc_ci(y, [eo[k][1] for k in overlap], rng)
            a_ec, ci_ec = _auroc_ci(y, [ec[k][1] for k in overlap], rng)
            entry.update(eo_auroc_overlap=a_eo, eo_ci95=ci_eo, ec_auroc_overlap=a_ec, ec_ci95=ci_ec, delta_eo_minus_ec=a_eo - a_ec)
        y_all = [v[0] for v in ec.values()]
        if len(set(y_all)) == 2:
            a, ci = _auroc_ci(y_all, [v[1] for v in ec.values()], rng)
            entry.update(ec_auroc_all=a, ec_ci95_all=ci)
        report["pipelines"][tag] = entry

    delta = report["pipelines"]["P1"].get("delta_eo_minus_ec")
    report["decision_rule"] = {
        "delta_P1": delta, "small_gap": SMALL_GAP, "confounded_gap": CONFOUNDED_GAP,
        "verdict": verdict(delta) if delta is not None else "undefined",
        "caveat": "Turku-EO is itself near chance (0.545), so a small delta can mean 'both near 0.5 for different reasons'; the rule was fixed before the numbers existed and is applied as written.",
    }
    (V4_RUN / "turku-eyesclosed-control.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "pipelines"}, indent=2))
    for tag, e in r["pipelines"].items():
        print(tag, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}))
    sys.exit(0)
