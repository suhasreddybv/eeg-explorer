"""Generates docs/pd-v4-results.json and docs/pd-v4-model-cards.json.

Aggregate reporting only: nothing per person leaves data/pd-runs/v4/.
Refuses to run until outer-models-lock.json, final-models-lock.json and
locked-site-results.json exist. docs/PD_V4_RESULTS.md and the PDF are
rendered from the JSON by scripts/report_pd_v4_pdf.py.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md for the rules this module reports against.
"""

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/pd-runs/v4"
DOCS = ROOT / "docs"

VARIANTS = [
    "F0_rich_svm|I", "F1_relative_svm|I", "F2_specparam_enet|I", "F3_riemann_logistic|I",
    "F3_riemann_logistic|T", "F4_leapd|I", "F5_entropy_logistic|I", "F6_boosting|I", "S|I", "S|T",
]
NAMES = {
    "F0_rich_svm|I": "F0 v3 rich-feature SVM",
    "F1_relative_svm|I": "F1 relative-spectrum SVM",
    "F2_specparam_enet|I": "F2 specparam elastic-net",
    "F3_riemann_logistic|I": "F3 Riemannian tangent logistic",
    "F3_riemann_logistic|T": "F3 Riemannian tangent logistic (transductive)",
    "F4_leapd|I": "F4 LEAPD / LPC hyperplanes",
    "F5_entropy_logistic|I": "F5 entropy logistic",
    "F6_boosting|I": "F6 gradient boosting",
    "S|I": "S non-negative stack (P1, inductive)",
    "S|T": "S non-negative stack (P2, transductive)",
}
SITES = {"iowa": "Iowa", "tel_aviv": "Tel Aviv", "ucsd": "UC San Diego", "unm": "New Mexico"}


def load(name, required=True):
    path = RUN / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{name} missing under {RUN}")
        return None
    return json.loads(path.read_text())


def sha256_of(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def software():
    out = {"python": platform.python_version()}
    for m in ("numpy", "scipy", "sklearn", "mne", "pyriemann", "fooof", "antropy", "lightgbm", "torch", "braindecode"):
        try:
            out[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001 -- absent modules are reported as absent
            out[m] = "absent"
    return out


def site_metrics(block):
    """Per-site metrics without any person-level content."""
    keep = ("participants", "pd_prevalence", "auroc", "auroc_95ci", "balanced_accuracy", "sensitivity",
            "specificity", "sensitivity_95ci_wilson", "specificity_95ci_wilson", "pr_auc_average_precision",
            "brier", "confusion_matrix", "threshold_policy", "paired_delta_vs_F0")
    return {k: block[k] for k in keep if k in block}


def main():
    for required in ("outer-models-lock.json", "final-models-lock.json", "locked-site-results.json"):
        if not (RUN / required).exists():
            raise RuntimeError(f"Refusing to report: {required} does not exist")

    results = load("results.json")
    gates = load("gates.json")
    reduced = load("reduced-results.json")
    fm = load("fm-results.json")
    fusion = load("fusion-results.json")
    negative = load("negative-controls.json")
    harmon = load("harmonisation-report.json")
    qc_audit = load("qc-audit.json")
    confound = load("confound-audit.json")
    turku24 = load("locked-site-results.json")
    turku20 = load("locked-site-results-run1-n20.json")
    brainlat = load("brainlat-results.json", required=False)
    ec = load("turku-eyesclosed-control.json", required=False)
    final_lock = load("final-models-lock.json")
    outer_lock = load("outer-models-lock.json")
    benchmark = load("supervised-benchmark.json")
    thresholds = load("qc-thresholds.json")
    policy = load("implementation-policy.json")
    turku_format = load("turku-format-verification.json", required=False)
    brainlat_format = load("brainlat-format-verification.json", required=False)
    if brainlat_format:
        # Per-subject file names are a participant table; only the count is aggregate.
        brainlat_format = {k: v for k, v in brainlat_format.items() if k != "preprocessing_flags_by_file"}
    brainlat_inspection = load("brainlat-inspection.json", required=False)
    brainlat_coverage = load("brainlat-coverage-labelfree.json", required=False)
    outer_folds = load("outer-fold-results.json")

    # Outer-fold selections and stacker weights are aggregate; people/scores are not.
    selections = {
        f["name"]: {"kind": f["kind"], "selections": f.get("selections"), "stacker_weights": f.get("stacker_weights")}
        for f in outer_folds
    }

    internal = {}
    for kind in ("cross_site", "within_site"):
        internal[kind] = {
            "equal_site_macro_auroc": results[kind]["equal_site_macro_auroc"],
            "sites": {
                site: {v: site_metrics(results[kind]["sites"][site][v]) for v in VARIANTS if v in results[kind]["sites"][site]}
                for site in results[kind]["sites"]
            },
        }

    coverage = {}
    for key, c in qc_audit["supervised_coverage_by_site_diagnosis"].items():
        site, dx = eval(key)  # noqa: S307 -- keys are "('site', 'dx')" tuples written by our own code
        coverage.setdefault(site, {})[dx] = c
    demographics = {}
    for row in confound["coverage"]:
        # Sites code sex as F/M, f/m or Female/Male; key on the first letter.
        sex = {}
        for k, v in row["sex_counts"].items():
            sex[str(k)[:1].upper()] = sex.get(str(k)[:1].upper(), 0) + v
        demographics.setdefault(row["site"], {})[row["diagnosis"]] = {
            "assigned": row["n"], "age_mean": row["age_mean"], "age_sd": row["age_sd"],
            "female": sex.get("F", 0), "male": sex.get("M", 0),
        }

    def strip_people(report):
        if report is None:
            return None
        return {k: v for k, v in report.items() if k not in ("people", "scores", "predictions")}

    aggregate = {
        "protocol": "PD_PROTOCOL_V4 (v4.2 with 14.4/14.5 amendments)",
        "protocol_sha256": sha256_of(DOCS / "PD_PROTOCOL_V4.md"),
        "implementation_policy_scope": policy.get("scope"),
        "software": software(),
        "hardware": "Apple M4, 16 GB unified memory; CPU/MPS only; no cloud",
        "qc_thresholds": thresholds,
        "cohort": {
            "n_assigned": 537, "n_ready": sum(c["n_ready"] for s in coverage.values() for c in s.values()),
            "n_eligible_a2": sum(c["n_meets_a2_min_10_windows"] for s in coverage.values() for c in s.values()),
            "n_training_people": final_lock["n_training_people"],
            "coverage_by_site_diagnosis": coverage, "demographics_assigned": demographics,
            "session_count_confound": confound.get("session_count_confound"),
            "demographic_auroc_by_site": confound.get("demographic_auroc_by_site"),
        },
        "internal": internal,
        "composite": results["composite"],
        "pre_declared": results["pre_declared"],
        "gates": gates,
        "selections_by_outer_fold": selections,
        "harmonisation": harmon["summary"],
        "negative_controls": negative,
        "reduced_montages": {
            "t1_reference": reduced["t1_reference"], "G6": reduced["G6"],
            "montages": {m: {k: v for k, v in d.items() if k in ("n_channels", "channels", "n_people", "stack_S_I", "families")} for m, d in reduced["montages"].items()},
            "distillation_M8": reduced["distillation_M8"],
        },
        "foundation_models": {k: fm[k] for k in ("n_people", "licences", "probe", "controls", "stack", "site_identity_probe", "gate", "embedding_dimension", "caveat")},
        "fusion": {k: v for k, v in fusion.items() if k != "example_record"},
        "final_models_lock": final_lock,
        "outer_models_lock": {k: v for k, v in outer_lock.items() if k != "fits"},
        "compute": {"supervised_forecast_hours": benchmark["forecast_hours"], "gate_96h_passed": benchmark["gate_96h_passed"],
                    "measured_seconds_per_fold": benchmark["measured_seconds_per_fold"], "n_folds": benchmark["n_folds"]},
        "external": {
            "turku": {
                "confirmatory_n20": strip_people(turku20), "corrected_n24_non_confirmatory": strip_people(turku24),
                "format_verification": turku_format, "qc_label_free": qc_audit.get("turku_qc_label_free"),
            },
            "turku_eyes_closed_control": ec,
            "brainlat": {
                "results": strip_people(brainlat), "format_verification": brainlat_format,
                "inspection": brainlat_inspection,
                "coverage_status_counts": (
                    {s: sum(1 for c in brainlat_coverage.values() if c["status"] == s) for s in {c["status"] for c in brainlat_coverage.values()}}
                    if brainlat_coverage else None
                ),
            },
        },
    }
    (DOCS / "pd-v4-results.json").write_text(json.dumps(aggregate, indent=2, default=str))

    cards = {}
    limitations_common = [
        "Four training sites previously examined in v1-v3; internal estimates are exploratory",
        "Site identity is recoverable from every representation (probe AUROC 0.81-0.97)",
        "Engineering QC without blinded clinical validation; label-free thresholds sat at their ceilings",
        "Turku one-shot (raw, eyes-open, n=20/24): H5 failed; CI spans chance",
        "BrainLat one-shot (ICA-cleaned, eyes-closed, 16 PD / 13 HC): H5b failed on CI; ranking 0.70-0.74 but locked threshold does not transfer",
        "No prospective clinical validation; not a diagnostic, progression or symptom-tracking product",
    ]
    for tag, variant in (("P1", "S|I"), ("P2", "S|T")):
        lock = final_lock[tag]
        cards[tag] = {
            "name": NAMES[variant], "pipeline": variant, "regime": "inductive" if variant.endswith("|I") else "transductive (C1 re-centering on the test site's own covariances)",
            "intended_use": "Exploratory PD-versus-control resting EEG research on 17 named scalp channels; not diagnosis or symptom severity",
            "inputs": "17 named 10-20 channels, 256 Hz, 1-40 Hz, average reference; first 15 accepted 4 s windows per person (A2); families B-1..B-6 plus v3-rich; C3 normative z-scores; no metadata inputs",
            "test_scope": "Nested leave-one-site-out (4 sites) plus 3 within-site participant folds; two one-shot external sites",
            "cross_site_macro_auroc": internal["cross_site"]["equal_site_macro_auroc"][variant],
            "within_site_macro_auroc": internal["within_site"]["equal_site_macro_auroc"][variant],
            "composite": results["composite"][variant],
            "per_site_cross_auroc": {s: internal["cross_site"]["sites"][s][variant]["auroc"] for s in SITES},
            "locked_site_turku": {
                "confirmatory_n20": turku20["pipelines"][tag], "corrected_n24_non_confirmatory": turku24["pipelines"][tag],
                "gate_H5_ci_lower_above_0.60": False,
            },
            "locked_site_brainlat": ({**brainlat["pipelines"][tag], "n_scored": brainlat["n_scored"], "gate_H5b_ci_lower_above_0.60": False} if brainlat else None),
            "eyes_closed_control": (ec["pipelines"][tag] if ec else None),
            "threshold": lock["threshold"], "training_prevalence": final_lock["P1"]["training_prevalence"],
            "n_training_people": final_lock["n_training_people"], "weights_sha256": lock["sha256"],
            "limitations": limitations_common, "protocol": "PD_PROTOCOL_V4", "protocol_sha256": aggregate["protocol_sha256"],
        }
    m4t = reduced["montages"]["M4T"]
    cards["P3"] = {
        "name": "P3 M4T around-ear montage stack", "pipeline": "M4T montage stack (T7, T8, P7, P8), inductive",
        "intended_use": "Exploratory around-ear feasibility on a montage subset of full caps; NOT a claim about any around-ear device (no public around-ear PD dataset exists)",
        "inputs": "4 channels re-referenced within the montage from preref128; families B-1, B-2, B-4, B-5, B-6 and Riemannian covariance on the 4 channels; C3 normative features",
        "selection_note": final_lock["P3"]["selection_note"],
        "test_scope": cards["P1"]["test_scope"],
        "cross_site_macro_auroc": m4t["stack_S_I"]["cross_site"]["macro"], "within_site_macro_auroc": m4t["stack_S_I"]["within_site"]["macro"],
        "composite": m4t["stack_S_I"]["composite"], "per_site_cross_auroc": m4t["stack_S_I"]["cross_site"]["per_site"],
        "locked_site_turku": {"confirmatory_n20": turku20["pipelines"]["P3"], "corrected_n24_non_confirmatory": turku24["pipelines"]["P3"]},
        "locked_site_brainlat": ({**brainlat["pipelines"]["P3"], "n_scored": brainlat["n_scored"], "channels": "spline-interpolated from 128 Biosemi electrodes"} if brainlat else None),
        "eyes_closed_control": (ec["pipelines"]["P3"] if ec else None),
        "threshold": final_lock["P3"]["threshold"], "n_training_people": final_lock["n_training_people"], "weights_sha256": final_lock["P3"]["sha256"],
        "limitations": limitations_common + ["Eyes-closed Turku: P3 fell to 0.30 [0.06, 0.60] on n=18; the around-ear montage is condition-sensitive"],
        "protocol": "PD_PROTOCOL_V4", "protocol_sha256": aggregate["protocol_sha256"],
    }
    (DOCS / "pd-v4-model-cards.json").write_text(json.dumps(cards, indent=2, default=str))
    print("wrote docs/pd-v4-results.json and docs/pd-v4-model-cards.json")
    return aggregate


if __name__ == "__main__":
    main()
    sys.exit(0)
