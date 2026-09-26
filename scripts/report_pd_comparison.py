"""Generate aggregate-only Markdown, JSON, model cards and a typeset PDF after testing."""

import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/pd-runs/v2"
NAMES = {
    "majority": "Class-frequency reference",
    "spectral": "Spectral logistic regression",
    "periodic": "Periodic/aperiodic features",
    "demographic": "Age/sex confound baseline",
    "artifact": "Artifact confound baseline",
    "eegnet": "EEGNet adaptation",
    "shallow": "ShallowConvNet adaptation",
    "channel_cnn": "Channel CNN adaptation",
    "wavelet_cnn": "Wavelet CNN adaptation",
    "graph": "Graph attention adaptation",
    "transformer": "Spectral-token Transformer",
}
ENDPOINTS = {
    "internal_test": "Iowa internal test (exploratory)",
    "tel_aviv": "Tel Aviv external test (primary)",
    "unm_on": "New Mexico ON (secondary)",
    "unm_off": "New Mexico OFF (secondary)",
}
ORDER = list(NAMES)


def read(name):
    return json.loads((RUN / name).read_text())


def fmt(x):
    return f"{x:.3f}"


def main():
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from pypdf import PdfReader

    result = read("test-results.json")
    lock = read("all-models-lock.json")
    reg = read("registry.json")
    bench = read("comparison-benchmark.json")
    archive = read("archive-audit.json")
    perm = read("permutation-check.json")
    subgroups = read("subgroup-audit.json")
    if result["lock_fingerprint"] != lock["fingerprint"]:
        raise ValueError("Results do not match model lock")
    preparation = read("preparation.json")["records"] + read("test-preparation.json")["records"]
    dev = lock["models"]
    docs = ROOT / "docs"
    out = ROOT / "output/pdf"
    out.mkdir(parents=True, exist_ok=True)
    md = []
    story = []
    styles = getSampleStyleSheet()
    styles["Heading1"].keepWithNext = True
    styles["Heading2"].keepWithNext = True
    styles.add(
        ParagraphStyle(
            name="BodyEEG",
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            spaceAfter=8,
            textColor=colors.HexColor("#243341"),
        )
    )
    styles.add(ParagraphStyle(name="CellEEG", fontName="Helvetica", fontSize=8, leading=10))
    styles.add(
        ParagraphStyle(name="SmallEEG", fontName="Helvetica", fontSize=8, leading=11, spaceAfter=6)
    )
    styles["Title"].textColor = colors.HexColor("#183e52")
    styles["Heading1"].textColor = colors.HexColor("#183e52")
    styles["Heading2"].textColor = colors.HexColor("#236b80")

    def paragraph(text, small=False):
        md.append(text + "\n")
        story.append(Paragraph(escape(text), styles["SmallEEG" if small else "BodyEEG"]))

    def heading(text, level=1):
        md.append("#" * (level + 1) + " " + text + "\n")
        story.append(Paragraph(escape(text), styles["Heading1" if level == 1 else "Heading2"]))

    def table(headers, rows, widths):
        cells = [
            [Paragraph(escape(str(v)), styles["CellEEG"]) for v in row] for row in [headers] + rows
        ]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#deedf1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#6f8c99")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f4f7f8")],
                    ),
                ]
            )
        )
        story.extend([t, Spacer(1, 10)])
        for row in rows:
            md.append("- " + "; ".join(f"{h}: {v}" for h, v in zip(headers, row)))
        md.append("")

    title = "Parkinson's EEG: expanded model comparison"
    md.append("# " + title + "\n")
    story.append(Paragraph(escape(title), styles["Title"]))
    paragraph(
        "Research report | Protocol v2 | " + datetime.now(timezone.utc).strftime("%d %B %Y UTC")
    )
    paragraph(
        "Local Apple M4 study. Participant-level classification of diagnosed Parkinson's disease versus study controls. This report does not establish a diagnostic test, disease progression measurement or symptom-tracking product."
    )
    heading("1. Executive findings")
    primary = result["endpoints"].get("tel_aviv", {}).get("models", {})
    eeg_candidates = [
        n for n in ORDER if n not in ["majority", "demographic", "artifact"] and n in primary
    ]
    if eeg_candidates:
        best = max(eeg_candidates, key=lambda n: primary[n]["auroc"])
        m = primary[best]
        paragraph(
            f"The highest observed primary external AUROC was {fmt(m['auroc'])} for {NAMES[best]}, with a 95% participant-bootstrap interval of {fmt(m['auroc_95ci'][0])} to {fmt(m['auroc_95ci'][1])}. This is a descriptive ranking after evaluating all locked models, not a new model-selection step or proof of superiority."
        )
        base = primary["spectral"]
        paragraph(
            f"The spectral regression baseline achieved AUROC {fmt(base['auroc'])}. At its validation-selected threshold, {NAMES[best]} achieved sensitivity {fmt(m['sensitivity'])} and specificity {fmt(m['specificity'])} on {m['participants']} retained external participants. These operating characteristics must be read together with exclusions, uncertainty and the external class prevalence."
        )
        paragraph(
            f"The age/sex-only external AUROC was {fmt(primary['demographic']['auroc'])} and the artifact-only AUROC was {fmt(primary['artifact']['auroc'])}. Strong confound-only discrimination would weaken a claim that EEG models learned disease-specific physiology."
        )
        paragraph(
            "Model rankings changed across sites: the channel CNN led New Mexico ON (AUROC 0.878), while EEGNet led New Mexico OFF (0.886). The Transformer missed 33 of 72 retained PD participants on the primary test at its frozen threshold. Its primary AUROC gain over spectral regression was 0.143, with an exploratory multiplicity-adjusted paired interval of 0.036 to 0.270. This supports a ranking improvement in this retained cohort; it does not establish superiority over all other models or a consistently strong operating point across sites."
        )
    else:
        paragraph(
            "The primary external comparison could not be estimated because both classes did not survive the frozen eligibility/QC policy. This is an important negative result; no external winner is claimed."
        )
    paragraph(
        "All ten planned families/reference models were run separately, plus an artifact confound baseline. All six neural families used the same 18 training-fold tuning fits and three final seeds. The pilot remains separate. Iowa results are exploratory because overlap with the previously examined pilot cannot be resolved; Tel Aviv and New Mexico were reserved until the v2 model lock."
    )
    heading("2. Archive integrity and cohort construction")
    paragraph(
        f"The archive contains {archive['planned_files']:,} reconciled files totaling {archive['verified_bytes'] / 1e9:.2f} GB. Archive-wide reconciliation checked receipt identities, file sizes, modification times and source checksum metadata. Selected input signals were rehashed before processing. This is not a second full-content hash pass over all 440 GB."
    )
    paragraph(
        "The 312-person expanded Iowa release replaces older Iowa datasets rather than being added to them. Only the earliest numbered resting visit per person is used. Tel Aviv contributes 144 candidate people; New Mexico contributes 50 people across 75 sessions. Repeated visits, windows and medication states do not count as additional participants. Older overlapping Iowa and New Mexico task releases, UCSD pilot tests, PD-only treatment datasets and unrelated healthy tasks are excluded."
    )
    coverage = {}
    rows = []
    for endpoint, label in [("train", "Training"), ("validation", "Validation")] + list(
        ENDPOINTS.items()
    ):
        rr = [
            r
            for r in preparation
            if r["endpoint"] == endpoint
            or (endpoint.startswith("unm_") and r["endpoint"] == "unm_control")
        ]
        counts = Counter((r["status"] == "ready", r["label"]) for r in rr)
        assigned = len(rr)
        retained = sum(r["status"] == "ready" for r in rr)
        coverage[endpoint] = {
            "assigned": assigned,
            "retained": retained,
            "retained_PD": counts[(True, 1)],
            "retained_control": counts[(True, 0)],
            "excluded_PD": counts[(False, 1)],
            "excluded_control": counts[(False, 0)],
        }
        rows.append(
            [label, assigned, retained, counts[(True, 1)], counts[(True, 0)], assigned - retained]
        )
    table(
        ["Cohort", "Assigned", "Retained", "PD", "Control", "Excluded"],
        rows,
        [170, 58, 58, 50, 60, 60],
    )
    paragraph(
        "Coverage is an essential part of the result. A score measured only among clean recordings does not describe everyone recruited. Exclusion rates can differ by diagnosis, age, motor symptoms and site. No excluded person was reassigned to another split or given a substitute session selected for better model performance."
    )
    heading("Demographic balance and exclusions by diagnosis", 2)
    profiles = {}
    profile_rows = []
    for endpoint, label in [("train", "Training"), ("validation", "Validation")] + list(
        ENDPOINTS.items()
    ):
        rr = [
            r
            for r in preparation
            if r["endpoint"] == endpoint
            or (endpoint.startswith("unm_") and r["endpoint"] == "unm_control")
        ]
        profiles[endpoint] = {}
        for diagnosis in [0, 1]:
            group = [r for r in rr if r["label"] == diagnosis]
            retained = [r for r in group if r["status"] == "ready"]
            ages = []
            for r in retained:
                try:
                    ages.append(float(r["age"]))
                except (TypeError, ValueError):
                    pass
            female = sum(r["sex"] in ["F", "Female"] for r in retained)
            male = sum(r["sex"] in ["M", "Male"] for r in retained)
            age_text = (
                f"{statistics.mean(ages):.1f} ({statistics.stdev(ages):.1f})"
                if len(ages) > 1
                else "unavailable"
            )
            excluded = len(group) - len(retained)
            profile_rows.append(
                [
                    label + (" / PD" if diagnosis else " / control"),
                    f"{excluded}/{len(group)}",
                    age_text,
                    f"{female}/{male}",
                    len(retained) - len(ages),
                ]
            )
            profiles[endpoint][str(diagnosis)] = {
                "assigned": len(group),
                "retained": len(retained),
                "excluded": excluded,
                "age_mean": statistics.mean(ages) if ages else None,
                "age_sd": statistics.stdev(ages) if len(ages) > 1 else None,
                "age_missing": len(retained) - len(ages),
                "female": female,
                "male": male,
                "sex_missing": len(retained) - female - male,
            }
    table(
        [
            "Cohort / diagnosis",
            "Excluded / assigned",
            "Age mean (SD)",
            "Female / male",
            "Age missing",
        ],
        profile_rows,
        [176, 77, 78, 71, 54],
    )
    paragraph(
        "Demographics describe retained people; exclusion fractions use all assigned people. In the primary external cohort, QC excluded 44/116 PD participants (37.9%) and 5/28 controls (17.9%). This differential selection is a material limitation even though every model uses the same retained participants."
    )
    reasons = Counter(
        reason
        for r in preparation
        for rejected in r.get("rejected_windows", [])
        for reason in rejected["reasons"]
    )
    paragraph(
        "Window rejection reasons across the selected recordings: "
        + "; ".join(f"{name}: {count:,}" for name, count in sorted(reasons.items()))
        + ". One window may have multiple reasons, and counts are not counts of unique people. One New Mexico session was excluded for ambiguous eyes-open event timing."
    )
    heading("3. Methods and safeguards")
    protocol = (docs / "PD_PROTOCOL_V2.md").read_text()
    # The full frozen narrative is included as an appendix; core details appear here.
    paragraph(
        "The frozen pipeline uses 17 shared scalp electrodes, 1-40 Hz filtering, 256 Hz sampling and average reference. Four-second windows are non-overlapping, capped at 60 seconds per person with a 40-second minimum. Fixed engineering artifact rules screen amplitude, jumps, flat channels, repeated samples and source annotations. These heuristics have not been validated by a blinded clinical EEG expert."
    )
    paragraph(
        "New Mexico eyes-open intervals are extracted only from explicit contiguous one-second event labels. Ambiguous timing is excluded. The analysis does not relabel oddball/task periods as resting EEG. Medication ON and OFF endpoints share participants and controls and are not independent replications."
    )
    paragraph(
        "Training participants alone fit scaling, imputation and feature transformations. All models use identical participant-level training folds. Neural hyperparameter search uses six learning-rate/dropout combinations, three folds, up to 100 epochs and patience 15. Three final seeds are averaged with equal weight. Validation determines early stopping and the operating threshold. Every final model is locked before any test signal processing or evaluation."
    )
    paragraph(
        "The comparison contains compact adaptations. Differences from the papers are disclosed in the protocol appendix: graph message passing omits contrastive pretraining and Chebyshev convolution; wavelet CNN uses smaller multichannel scalograms; the Transformer models spectral/entropy electrode tokens. No published score is used as an acceptance target."
    )
    heading("4. Training and validation performance")
    rows = []
    for n in ORDER:
        d = dev[n]
        seeds = d.get("seeds", [])
        variability = (
            f"{min(s['best_auroc'] for s in seeds):.3f}-{max(s['best_auroc'] for s in seeds):.3f}"
            if seeds
            else "n/a"
        )
        rows.append(
            [
                NAMES[n],
                fmt(d["training"]["auroc"]),
                fmt(d["validation"]["auroc"]),
                fmt(d.get("threshold", 0.5)),
                variability,
            ]
        )
    table(
        ["Model", "Train AUC", "Val AUC", "Threshold", "Seed val range"],
        rows,
        [195, 63, 63, 65, 70],
    )
    paragraph(
        "Training scores are in-sample. Validation scores are selected by early stopping and threshold choice; they are optimistic estimates of generalization. The seed range describes variability on validation only. It is not a confidence interval, and the best seed was not selected on a test."
    )
    heading("5. Locked test results")
    for endpoint, label in ENDPOINTS.items():
        heading(label, 2)
        entry = result["endpoints"][endpoint]
        if "models" not in entry:
            paragraph(entry["unavailable"])
            continue
        paragraph(
            f"Coverage: {coverage[endpoint]['retained']}/{coverage[endpoint]['assigned']}; retained PD/control: {coverage[endpoint]['retained_PD']}/{coverage[endpoint]['retained_control']}."
        )
        rows = []
        for n in ORDER:
            m = entry["models"][n]
            lo, hi = m["auroc_95ci"]
            rows.append(
                [
                    NAMES[n],
                    f"{m['auroc']:.3f} ({lo:.3f}-{hi:.3f})",
                    fmt(m["balanced_accuracy"]),
                    fmt(m["sensitivity"]),
                    fmt(m["specificity"]),
                    fmt(m["brier"]),
                ]
            )
        table(
            ["Model", "AUROC (95% CI)", "Bal. acc.", "Sens.", "Spec.", "Brier"],
            rows,
            [159, 115, 49, 44, 44, 45],
        )
        paragraph(
            "AUROC describes ranking across thresholds. Sensitivity and specificity use thresholds fixed on validation; Brier is mean squared probability error (lower is better). Intervals are participant-level, not window-level. Small control groups can leave specificity highly uncertain."
        )
        for n in ORDER:
            m = entry["models"][n]
            tn, fp = m["confusion_matrix"][0]
            fn, tp = m["confusion_matrix"][1]
            paragraph(
                f"{NAMES[n]}: TN {tn}, FP {fp}, FN {fn}, TP {tp}; average precision {m['pr_auc_average_precision']:.3f}; PD prevalence {m['pd_prevalence']:.3f}; sensitivity 95% Wilson CI {m['sensitivity_95ci_wilson'][0]:.3f}-{m['sensitivity_95ci_wilson'][1]:.3f}; specificity CI {m['specificity_95ci_wilson'][0]:.3f}-{m['specificity_95ci_wilson'][1]:.3f}.",
                True,
            )
        heading("Paired comparison with spectral regression", 2)
        rows = []
        for n, d in entry["paired_vs_spectral"].items():
            rows.append(
                [
                    NAMES[n],
                    fmt(d["auroc_difference"]),
                    f"{d['paired_95ci'][0]:.3f} to {d['paired_95ci'][1]:.3f}",
                    f"{d['bonferroni_95ci'][0]:.3f} to {d['bonferroni_95ci'][1]:.3f}",
                ]
            )
        table(
            ["Model", "AUC difference", "Paired 95% CI", "Multiplicity-adjusted CI"],
            rows,
            [179, 72, 100, 105],
        )
        paragraph(
            "Paired bootstrap resamples the same participants for every model. Adjusted intervals use Bonferroni across the ten comparisons with spectral regression within this endpoint. They do not adjust across every site and subgroup. With 2,000 resamples, extreme adjusted endpoints are approximate; these are exploratory comparisons, not confirmatory significance claims."
        )
    heading("6. Negative controls and subgroup uncertainty")
    paragraph(
        f"The spectral baseline's selected training-fold AUROC was {perm['observed_selected_cv_auroc']:.3f}. Across {perm['permutations']} participant-label permutations, repeating regularization selection, the null mean was {perm['null_mean']:.3f}, its central 95% range was {perm['null_95_interval'][0]:.3f}-{perm['null_95_interval'][1]:.3f}, and the Monte Carlo p-value was {perm['monte_carlo_p']:.4f}. This checks one baseline pipeline; it neither validates all neural models nor proves disease-specific physiology."
    )
    paragraph(
        "Age (<65 / >=65) and sex strata were fixed for descriptive post-lock reporting. A subgroup needs at least five people in each class to estimate metrics; otherwise it is explicitly omitted. These slices have no multiplicity correction and were never used to retune models."
    )
    for endpoint, groups in subgroups.items():
        heading(ENDPOINTS[endpoint], 2)
        rows = []
        for g, d in groups.items():
            g = {
                "age_under_65": "Age below 65",
                "age_65_plus": "Age 65 or older",
                "female": "Female",
                "male": "Male",
            }[g]
            count_text = f"{d['class_counts']['0']} / {d['class_counts']['1']}"
            if "models" not in d:
                rows.append([g, count_text, d["not_estimated"]])
                continue
            rows.append(
                [
                    g,
                    count_text,
                    "; ".join(f"{NAMES[n]} {m['auroc']:.3f}" for n, m in d["models"].items()),
                ]
            )
        table(
            ["Stratum", "Control / PD", "AUROC by model / reason not estimated"],
            rows,
            [90, 70, 296],
        )
    heading("7. Local compute and reproducibility")
    rows = []
    total = 0
    for n, b in bench["models"].items():
        root = RUN / "models" / n
        fits = [json.loads(p.read_text()) for p in root.glob("*/result.json")]
        seconds = sum(f["elapsed_seconds"] for f in fits)
        total += seconds
        rows.append(
            [
                NAMES[n],
                b["parameters"],
                len(fits),
                f"{seconds / 60:.1f}",
                f"{min(f['epochs'] for f in fits)}-{max(f['epochs'] for f in fits)}",
            ]
        )
    table(
        ["Model", "Parameters", "Fits", "Fit minutes", "Epoch range"], rows, [175, 70, 50, 80, 81]
    )
    paragraph(
        f"Device: {bench['device']}; {bench['platform']}. Conservative neural campaign forecast: {bench['forecast_seconds'] / 3600:.2f} hours, using maximum epochs and 25% margin. Recorded neural fit time: {total / 3600:.2f} hours; this excludes separate preprocessing, representation creation, benchmarks, reporting and test/bootstrap evaluation. The six architectures use different parameter counts; identical tuning counts do not imply identical compute."
    )
    campaign_state = read("campaign-status.json")
    if "elapsed_seconds" in campaign_state:
        paragraph(
            f"The automated campaign, including fitting, test preprocessing, evaluation, subgroup diagnostics and initial report generation, completed in {campaign_state['elapsed_seconds'] / 60:.1f} minutes. Development preprocessing, the benchmark, permutation check and later report review happened separately."
        )
    paragraph("Registry fingerprint: " + reg["fingerprint"], True)
    environment = read("environment.json")
    paragraph(
        "Software: Python "
        + environment["python"]
        + "; "
        + "; ".join(f"{name} {version}" for name, version in environment["packages"].items())
        + ".",
        True,
    )
    paragraph(environment["hardware"] + ". " + environment["memory_measurement"], True)
    paragraph("Final model-lock fingerprint: " + lock["fingerprint"], True)
    paragraph(
        "Reproduce with the repository environment and verified archive. Commands: python -m pd_pipeline.cohort_v2 build; python -m pd_pipeline.cohort_v2 prepare; python -m pd_pipeline.compare_v2 setup; python -m pd_pipeline.compare_v2 benchmark; python -m pd_pipeline.compare_v2 campaign; python -m pd_pipeline.cohort_v2 test-prepare; python -m pd_pipeline.compare_v2 evaluate. See LOCAL_TRAINING.md and PD_PROTOCOL_V2.md for prerequisites and immutable-run behavior."
    )
    heading("8. Limitations and implications for Myna")
    paragraph(
        "This remains a modest observational cohort, despite a large downloaded archive. The study cannot separate all effects of age, medication, disease severity, site, equipment, reference and artifact burden. Development is from one institution. Public de-identification prevents complete cross-release identity reconciliation. The exploratory Iowa split does not restore the independence of the previously examined pilot."
    )
    paragraph(
        "Fixed QC may preferentially exclude people with tremor or other symptoms. Device metadata are incomplete at some sites. No blinded expert review, prospective recruitment or independent clinical validation was performed. Bootstrap intervals describe uncertainty conditional on these retained cohorts and this trained model; they omit some training and sampling uncertainty."
    )
    paragraph(
        "After this report, all evaluated test cohorts are examined. Future architecture or preprocessing changes require a new, genuinely independent evaluation cohort or explicit exploratory labeling. Selecting the numerically highest external score for deployment would reuse the test for selection and needs new validation."
    )
    paragraph(
        "For Myna, these models provide an EEG research comparison only. A useful next study requires paired EEG, audio and movement from the same people and prospective validation against modality-specific baselines. Cross-sectional PD/control classification does not demonstrate symptom severity tracking, progression prediction or general abnormal-EEG detection. Reduced-electrode wearable performance needs a separate acquisition and motion study."
    )
    heading("Appendix A. Frozen protocol, model definitions and sources")
    for line in protocol.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            heading(line[3:], 2)
        elif line.strip():
            paragraph(line.strip(), True)
    md_path = docs / "PD_COMPARISON_RESULTS.md"
    md_path.write_text("\n".join(md))
    aggregate = {
        "archive": archive,
        "coverage": coverage,
        "demographic_profiles": profiles,
        "window_rejection_reasons": dict(reasons),
        "benchmark": bench,
        "test_results": result,
        "permutation_check": perm,
        "subgroups": subgroups,
        "model_selection": {
            n: {k: v for k, v in d.items() if k != "seeds"} for n, d in dev.items()
        },
        "registry_fingerprint": reg["fingerprint"],
        "lock_fingerprint": lock["fingerprint"],
    }
    # Aggregate archive audit contains no file-specific exceptions after reconciliation.
    (docs / "pd-comparison-results.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False)
    )
    cards = {
        n: {
            "display_name": NAMES[n],
            "intended_use": "Research PD/control comparison; no diagnostic or symptom tracking claim",
            "training_people": coverage["train"],
            "selection": {k: v for k, v in dev[n].items() if k != "seeds"},
            "test_metrics": {
                e: d["models"].get(n) for e, d in result["endpoints"].items() if "models" in d
            },
            "limitations": [
                "Single-site development",
                "Engineering QC without expert validation",
                "Iowa internal test exploratory",
                "External cohorts now examined",
            ],
            "lock_fingerprint": lock["fingerprint"],
        }
        for n in ORDER
    }
    (docs / "pd-model-cards.json").write_text(json.dumps(cards, indent=2, allow_nan=False))
    pdf = out / "PD_Classifier_Comparison_Report.pdf"

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#60727d"))
        canvas.drawString(54, 30, "EEG Explorer | PD comparison v2 | Research use")
        canvas.drawRightString(558, 30, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(
        str(pdf),
        pagesize=(612, 792),
        leftMargin=54,
        rightMargin=54,
        topMargin=46,
        bottomMargin=48,
        title=title,
        author="EEG Explorer research project",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(pdf)
    text = "\n".join(p.extract_text() for p in reader.pages)
    for name in NAMES.values():
        if name not in text:
            raise ValueError("PDF text missing model " + name)
    (RUN / "report-validation.json").write_text(
        json.dumps(
            {
                "pages": len(reader.pages),
                "pdf": str(pdf),
                "all_model_names_present": True,
                "visual_review": "pending",
            },
            indent=2,
        )
    )
    print("Created", pdf, "pages", len(reader.pages))


if __name__ == "__main__":
    main()
