"""Typeset the reviewed v3 aggregates without changing any tested model or prediction."""

import json
import hashlib
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/pd-runs/v3"
NAMES = {
    "spectral_lr": "Spectral regression",
    "rich_lr": "Rich-feature regression",
    "rich_svm": "Rich-feature SVM",
    "ssl_hybrid_lr": "Pretrained hybrid regression",
    "ssl_hybrid_svm": "Pretrained hybrid SVM",
    "scratch_neural": "Neural, from scratch",
    "ssl_neural": "Neural, pretrained",
    "selected_pipeline": "Inner-selected pipeline",
}
ORDER = [
    "spectral_lr",
    "rich_lr",
    "rich_svm",
    "ssl_hybrid_lr",
    "ssl_hybrid_svm",
    "scratch_neural",
    "ssl_neural",
    "selected_pipeline",
]
SITES = {"iowa": "Iowa", "tel_aviv": "Tel Aviv", "ucsd": "UC San Diego", "unm": "New Mexico"}


def main():
    def read(p):
        return json.loads(p.read_text())

    a = read(ROOT / "docs/pd-v3-results.json")
    v = read(ROOT / "docs/pd-v3-review.json")
    results = a["results"]
    state = read(RUN / "campaign-status.json")
    ssl = read(RUN / "ssl-lock.json")
    deployment = read(RUN / "deployment/lock.json")
    story = []
    md = []
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            spaceAfter=8,
            textColor=colors.HexColor("#263547"),
        )
    )
    styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=8, leading=10))
    styles.add(
        ParagraphStyle(
            name="SmallReport", fontName="Helvetica", fontSize=8, leading=10.5, spaceAfter=7
        )
    )
    styles["Heading1"].fontSize = 18
    styles["Heading1"].leading = 22
    styles["Heading1"].textColor = colors.HexColor("#183b56")
    styles["Heading1"].spaceAfter = 14
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 15
    styles["Heading2"].spaceAfter = 9
    styles["Heading1"].keepWithNext = True
    styles["Heading2"].keepWithNext = True

    def para(text, small=False):
        story.append(Paragraph(escape(text), styles["SmallReport" if small else "ReportBody"]))
        md.extend([text, ""])

    def title(text):
        if story:
            story.append(PageBreak())
        story.append(Paragraph(escape(text), styles["Heading1"]))
        md.extend(["## " + text, ""])

    def heading(text):
        story.append(Paragraph(escape(text), styles["Heading2"]))
        md.extend(["### " + text, ""])

    def table(headers, rows, widths):
        grid = [
            [Paragraph(escape(str(c)), styles["Cell"]) for c in row] for row in [headers] + rows
        ]
        t = Table(grid, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#638298")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f5f8fa")],
                    ),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([t, Spacer(1, 10)])
        for row in rows:
            md.append("- " + "; ".join(f"{h}: {c}" for h, c in zip(headers, row)))
        md.append("")

    def interval(m):
        return f"{m['auroc']:.3f} ({m['ci95'][0]:.3f}-{m['ci95'][1]:.3f})"

    def detailed(site, kind):
        heading(
            SITES[site] + f" | {results[kind]['sites'][site]['rich_svm']['participants']} people"
        )
        rows = []
        for f in ORDER:
            m = results[kind]["sites"][site][f]
            ci = m["auroc_95ci"]
            rows.append(
                [
                    NAMES[f],
                    f"{m['auroc']:.3f}",
                    f"{ci[0]:.3f}-{ci[1]:.3f}",
                    f"{m['sensitivity']:.3f}",
                    f"{m['specificity']:.3f}",
                    f"{m['pr_auc_average_precision']:.3f}",
                    f"{m['brier']:.3f}",
                ]
            )
        table(
            ["Model", "AUROC", "95% CI", "Sens.", "Spec.", "PR-AUC", "Brier"],
            rows,
            [156, 43, 79, 44, 44, 49, 47],
        )

    title("Parkinson's EEG: expanded-data comparison")
    para("Protocol v3 | Completed 27 September 2026 | Reviewed 28 September 2026 UTC")
    para(
        "Research question: does broader EEG pretraining and full-duration, multi-site PD data improve discrimination between diagnosed Parkinson's disease and study controls?"
    )
    heading("The result")
    para(
        "There is no demonstrated cross-site benefit from self-supervised pretraining in this experiment. The rich-feature SVM had the highest observed equal-site mean AUROC, 0.763 (95% participant-bootstrap interval 0.704-0.820). The pretrained hybrid SVM was essentially tied at 0.761. The pretrained neural model scored 0.685 versus 0.699 when trained from scratch."
    )
    para(
        "The full procedure that selected a family using inner validation scored 0.690 cross-site (95% interval 0.611-0.758). This is the appropriate result for the selection procedure. Selecting the observed best family after viewing outer results introduces selection optimism; 0.763 is a descriptive family result, not a new unbiased best-model claim."
    )
    table(
        ["Completed work", "Scope"],
        [
            ["Archive processing", "11,211 candidate recordings; 7,988 with accepted windows"],
            [
                "PD comparison",
                "440 eligible participants at four sites; all compatible duration and repeated resting sessions",
            ],
            [
                "Auxiliary pretraining",
                "321,285 accepted windows, 850 training groups and 110 validation groups",
            ],
            [
                "Training",
                "620 planned fits: 160 neural and 460 classical; one pretraining run; 14 post-hoc diagnostic fits",
            ],
            [
                "Evaluation",
                "Four held-out-site folds and three separate within-site participant folds",
            ],
        ],
        [130, 382],
    )
    heading("Interpretation comes first")
    para(
        "All four PD sites were previously examined in preliminary studies; v3 is exploratory. Recording metadata alone perfectly separated diagnoses at UCSD and New Mexico, where the study design gave PD participants repeated medication sessions and controls one visit. EEG models did not directly receive session count, but measurement and acquisition confounding remain serious concerns."
    )
    para(
        "This result supports continued research. It does not establish a diagnostic test, disease progression score, wearable monitor performance or symptom-tracking product. The earlier v2 report remains unchanged."
    )

    title("1. Evaluation design and model selection")
    para(
        "Primary: hold out one entire site, train on the other three, and repeat for all four sites. Within each outer training set, hold out its sites in turn to choose configurations by equal-site mean AUROC. Every person, visit, medication session and window remains within the same fold."
    )
    para(
        "Secondary: three participant folds stratified jointly by site and diagnosis. Each test person is unseen, while the training data represent the same institutions. Inner site validation still selects settings. Results are reported per site and as an equal-site mean, separately from primary transfer results."
    )
    para(
        "All supervised test people were excluded from all auxiliary pretraining, including overlapping releases. Fixed within-recording feature extraction was cached globally, but population scalers, calibration, classical fits and supervised learning used only the relevant training data."
    )
    table(
        ["Stage", "Frozen rule"],
        [
            [
                "Candidate search",
                "19 settings across seven families: three C values for five classical families and two learning rates for two neural families",
            ],
            [
                "Neural tuning",
                "Maximum 40 epochs, patience seven, inner validation only; one fixed tuning seed",
            ],
            [
                "Final neural fits",
                "Median inner best epoch; three fixed seeds averaged without choosing a winning seed",
            ],
            [
                "Threshold",
                "Inner out-of-fold balanced-accuracy optimum; no outer threshold adjustment",
            ],
            [
                "Outer lock",
                "Every outer fit completed and locked before outer predictions were generated",
            ],
            [
                "Final all-data fits",
                "Separate inner-site selection on all 440 people, then final research fits; no new test claim",
            ],
        ],
        [129, 383],
    )
    para(
        "The inner-selected family differed across outer folds. This variability is part of the model-selection procedure, not a reason to replace its results with the best family seen on each test."
    )
    table(
        ["Outer evaluation", "Family selected without its test"],
        [
            [name.replace("_", " "), NAMES[d["selected_family"]]]
            for name, d in a["selection"].items()
        ],
        [160, 352],
    )
    para(
        "Bootstrap intervals resample participants within the four fixed sites. They do not quantify uncertainty from having only four sites, model refitting, hyperparameter selection or earlier exposure to these datasets. Paired and subgroup comparisons are exploratory and unadjusted for multiplicity.",
        True,
    )

    title("2. Did we use the whole downloaded archive?")
    para(
        "No. Every dataset was accounted for, but not every recording is an eligible PD example or a compatible pretraining input. Unlike v2, v3 has no 60-second cap and includes repeated compatible resting sessions. It uses fixed spectral representations of accepted signal windows, not the complete raw waveform as neural input."
    )
    rows = []
    for ds, d in a["archive_usage"].items():
        role = (
            "PD rest" if d.get("supervised") else "Auxiliary" if d.get("auxiliary") else "Excluded"
        )
        rows.append([ds, d["recordings"], d.get("supervised", d.get("auxiliary", 0)), role])
    table(["Dataset", "Raw recordings", "Candidate inputs", "Role"], rows, [160, 105, 115, 132])
    para(
        "Ten sources were excluded for overlapping/replaced releases, an unsupported MATLAB v7.3 format, or unresolved physical electrode mapping. PD walking/task acquisitions were excluded. Two exact repeated Iowa signals were retained once. The ledger and complete exclusion reasons accompany the repository report.",
        True,
    )
    para(
        "Of 11,211 candidates, 3,223 produced no input: 2,855 incompatible-montage recordings, 367 recordings with no accepted window and one ambiguous New Mexico eye interval. The 7,988 recordings with windows are not 7,988 people; some groups still fail the minimum 40 seconds.",
        True,
    )

    title("3. Cohort coverage and demographic balance")
    rows = []
    for site, groups in v["demographic_profiles"].items():
        for label, d in groups.items():
            rows.append(
                [
                    SITES[site],
                    "PD" if label == "1" else "Control",
                    f"{d['retained']}/{d['assigned']}",
                    f"{100 * d['excluded'] / d['assigned']:.1f}%",
                    f"{d['age_mean']:.1f} ({d['age_sd']:.1f})",
                    f"{d['female']}/{d['male']}",
                ]
            )
    table(
        ["Site", "Diagnosis", "Retained / assigned", "Excluded", "Age mean (SD)", "Female / male"],
        rows,
        [85, 62, 103, 64, 106, 82],
    )
    para(
        "The four source cohorts contain 537 assigned people; 440 remain after fixed QC and the minimum accepted-duration rule. Age and sex describe retained people. All retained demographic fields used here were available. Tel Aviv excluded 37/116 PD participants (31.9%) and 5/28 controls (17.9%), creating differential selection."
    )
    heading("Auxiliary coverage")
    rows = []
    for key, c in a["coverage"].items():
        if "auxiliary" in key:
            rows.append(
                [
                    key.split(" /")[0],
                    f"{c['retained_groups']}/{c['assigned_groups']}",
                    f"{c['accepted_windows'] * 4 / 3600:.2f}",
                ]
            )
    table(
        ["Canonical source group", "Retained / assigned groups", "Accepted hours"],
        rows,
        [235, 147, 130],
    )
    para(
        "The complementary ds004504/ds006036 data share person IDs and are grouped together. The newer NeuroEPO treatment/placebo arms are conservatively grouped as one unresolved identity group, so their count is not a count of patients. Public de-identification prevents proof of absolute cross-study uniqueness."
    )
    para(
        "All 153 ds003775 recordings failed the unchanged amplitude/step rules: 9,027 windows were rejected for both reasons. That source supplied no pretraining examples. This cohort-wide rejection merits an independent unit/reference/QC review before any future protocol, not post-result threshold relaxation."
    )

    title("4. Signal processing, representation and new model")
    para(
        "Seventeen common named scalp channels; 256 Hz; 1-40 Hz fourth-order zero-phase filtering; average reference over observed channels; four-second non-overlapping windows with bounded chunks and two-second margins. Supervised inputs require all 17 channels; auxiliary inputs require at least 12 with explicit missing-channel masks. Legacy T3/T4/T5/T6 names map to T7/T8/P7/P8. Electrode positions are never guessed."
    )
    para(
        "Keep every accepted window across documented compatible resting duration, including repeated sessions; require at least ten accepted windows per person. Window QC checks peak-to-peak amplitude, abrupt steps, flat channels, repeated samples and source bad annotations. These are engineering heuristics without blinded clinical EEG validation."
    )
    table(
        ["Representation", "Definition"],
        [
            [
                "Spectral baseline",
                "Four per-electrode log-bandpower summaries: 2-4, 4-8, 8-13 and 13-30 Hz; 68 person-level inputs",
            ],
            [
                "Rich features",
                "Means and variability of 34 log-PSD bins per electrode, plus 153 log-covariance terms after 10% isotropic shrinkage; 1,309 inputs",
            ],
            [
                "Pretrained hybrid",
                "Rich features plus 96 pooled embedding mean/variability features; 1,405 inputs",
            ],
            [
                "Neural encoder",
                "48-dimensional electrode tokens; learned positions; two four-head Transformer layers; 96-wide feed-forward layers; masked spectral decoder and binary head; 42,275 parameters",
            ],
        ],
        [125, 387],
    )
    para(
        "Per-person summaries weight sessions equally; all windows contribute within a session. Neural loss weights sites, diagnoses, people and sessions to limit domination by large cohorts or long recordings. Missing auxiliary channels are masked from attention keys, pooling and reconstruction targets."
    )
    heading("What the new model is, and is not")
    para(
        "Pretraining reconstructs masked electrode spectra, motivated by masked EEG representation learning. It is a compact spectral encoder built for this Mac, not a reproduction of BENDR or LaBraM and not a large raw-waveform foundation model. The pretrained hybrid uses frozen embeddings; the pretrained neural classifier fine-tunes the encoder with a fresh binary head."
    )
    para(
        "C = 0.1/1/10 for regression and RBF SVM; SVM gamma = scale. Neural learning rates = 0.0003/0.001; batch 128; AdamW weight decay 0.01; gradient clipping 5; final seeds 20260927-29. No settings were chosen from outer performance.",
        True,
    )

    title("5. Primary and secondary aggregate results")
    table(
        ["Family / procedure", "Cross-site AUROC (95% CI)", "Within-site AUROC (95% CI)"],
        [
            [
                NAMES[f],
                interval(v["macro_uncertainty"]["cross_site"]["models"][f]),
                interval(v["macro_uncertainty"]["within_site"]["models"][f]),
            ]
            for f in ORDER
        ],
        [190, 161, 161],
    )
    para(
        "These are equal-site means, so the small New Mexico and UCSD cohorts carry the same weight as Iowa. Per-site results and sample sizes follow. The inner-selected pipeline is the predeclared family-selection procedure; the observed best family is a separate descriptive comparison."
    )
    heading("Did pretraining help? Paired differences")
    contrasts = [
        ("rich_svm_minus_spectral_lr", "Rich SVM minus spectral regression"),
        ("ssl_hybrid_svm_minus_rich_svm", "Pretrained hybrid SVM minus rich SVM"),
        ("ssl_hybrid_lr_minus_rich_lr", "Pretrained hybrid LR minus rich LR"),
        ("ssl_neural_minus_scratch_neural", "Pretrained neural minus scratch neural"),
    ]

    def difftext(d):
        return f"{d['difference']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f})"

    table(
        ["Paired contrast", "Cross-site delta (95% CI)", "Within-site delta (95% CI)"],
        [
            [
                label,
                difftext(v["macro_uncertainty"]["cross_site"]["paired"][key]),
                difftext(v["macro_uncertainty"]["within_site"]["paired"][key]),
            ]
            for key, label in contrasts
        ],
        [190, 161, 161],
    )
    para(
        "Rich spectral/covariance features with an SVM outperformed the four-band regression baseline in this comparison. Adding pretrained embeddings did not provide a demonstrated cross-site gain. The pretrained neural model did not outperform its scratch counterpart; the paired interval includes both harm and benefit. These findings do not prove that all self-supervised EEG approaches are ineffective."
    )

    for kind, heading_text in [
        ("cross_site", "Primary cross-site"),
        ("within_site", "Separate within-site"),
    ]:
        for index, sites in enumerate([["iowa", "tel_aviv"], ["ucsd", "unm"]]):
            title(
                f"{heading_text}: " + ("Iowa and Tel Aviv" if index == 0 else "UCSD and New Mexico")
            )
            for site in sites:
                detailed(site, kind)
            para(
                "Sensitivity and specificity use training-selected thresholds; within-site thresholds can differ across folds. PR-AUC is average precision and depends on PD prevalence. Brier score assesses probability error. High AUROC alone does not imply a useful operating threshold.",
                True,
            )
            para(
                "UCSD and New Mexico aggregate repeated PD medication sessions, while controls typically have one session. The metadata confound findings must accompany these scores. All participants are counted once per evaluation design.",
                True,
            )

    title("6. Pretraining, compute and final research models")
    para(
        f"Auxiliary pretraining used 284,219 training windows and 37,066 validation windows, equivalent to 315.8 and 41.2 accepted recording hours. A deterministic group split gave 850 training groups and 110 validation groups. None belonged to the PD evaluation cohorts. Twenty epochs completed in {ssl['seconds'] / 60:.1f} minutes on the Mac GPU."
    )
    para(
        "The best held-out reconstruction loss was approximately 0.348, down from 0.453 at epoch one. Reconstruction improved, but downstream cross-site PD discrimination did not show a pretraining advantage. Improving the auxiliary task does not establish improved disease classification."
    )
    table(
        ["Compute measure", "Observed / forecast"],
        [
            ["Mac", "Apple M4, 16 GB unified memory; PyTorch MPS"],
            ["Pretraining forecast", "4.00 hours, maximum-epoch estimate with margin"],
            ["Supervised campaign forecast", "15.16 hours, maximum-epoch estimate with margin"],
            [
                "Completed fitting",
                f"160 neural and 460 classical fits; summed fit duration {v['audit']['sum_fit_seconds'] / 60:.1f} minutes",
            ],
            [
                "Resumed independent campaign",
                f"{state['elapsed_seconds'] / 60:.1f} minutes from final preprocessing restart through aggregate report",
            ],
            [
                "Timing exclusions",
                "Earlier interrupted preparation, initial benchmarks, final audit and PDF review; no measured peak total GPU/unified-memory figure",
            ],
        ],
        [185, 327],
    )
    heading("Final all-data selection")
    para(
        "After nested evaluation, the prescribed training-only selection on all 440 eligible people chose the pretrained hybrid SVM (C=10). Its inner-site selection AUROC was 0.780; this is a tuning statistic, not a fresh external test score. All seven final family fits are retained locally. The chosen research model is not promoted as clinically superior."
    )
    table(
        ["Final family", "Chosen setting", "Neural epochs"],
        [
            [
                NAMES[f],
                str(m["config"].get("C", m["config"].get("lr"))),
                m["epochs"] if m["epochs"] else "n/a",
            ]
            for f, m in deployment["models"].items()
        ],
        [248, 144, 120],
    )
    para(
        "All original models, predictions and scientific records remain frozen. The final report adds interpretation and descriptive diagnostics without retuning or selecting a new outer-test winner.",
        True,
    )

    title("7. Confounding checks and exclusions")
    para(
        "Post-lock descriptive baselines used fixed C=1 logistic regression and the same outer splits. The demographic baseline used age/sex; the recording-quality baseline used accepted-window count, assigned session count and acceptance fraction. Imputation and scaling were fitted inside each training fold. These were diagnostic additions, not prespecified competing EEG families."
    )
    rows = []
    for site in SITES:
        rows.append(
            [SITES[site]]
            + [
                f"{v['posthoc_confound_auroc'][kind][f][site]:.3f}"
                for kind, f in [
                    ("cross_site", "demographic"),
                    ("cross_site", "recording_quality"),
                    ("within_site", "demographic"),
                    ("within_site", "recording_quality"),
                ]
            ]
        )
    table(
        [
            "Site",
            "Cross: demographics",
            "Cross: recording metadata",
            "Within: demographics",
            "Within: recording metadata",
        ],
        rows,
        [88, 103, 115, 103, 103],
    )
    para(
        "The metadata baseline achieves AUROC 1.000 at both UCSD and New Mexico. Their acquisition schedules encode diagnosis: PD participants were invited for medication ON/OFF visits, controls for one. The EEG models never directly receive session count, but averaging unequal acquisition histories can affect measurement reliability and medication mixtures. Therefore these sites cannot establish disease-specific learning from EEG alone."
    )
    para(
        "Age/sex-only discrimination was much weaker. This does not rule out residual demographic or equipment effects; a weak simple confound baseline is not proof that the EEG model uses PD-specific physiology."
    )
    heading("Additional material limitations")
    para(
        "The artifact rules rejected whole sources or disproportionately excluded PD participants at some sites. Missing montages removed 2,855 bigP3BCI recordings. The newer NeuroEPO arms use an intentionally conservative combined identity group. Auxiliary pretraining mixes resting, task, eyes-open and eyes-closed data, while supervised inputs use compatible eyes-open rest. These shifts can reduce transfer."
    )
    para(
        "A future confirmatory protocol should use a genuinely unexamined site, prespecify one comparable medication/session rule and a fixed recording-duration analysis, and obtain blinded expert QC review. Those changes require a new protocol and cannot be used to revise the present test results."
    )

    title("8. Descriptive subgroup review")
    para(
        "The table shows the selected procedure only. All seven families and exact stratum counts are retained in the aggregate review JSON. Estimates require at least five retained people of each diagnosis. These are post-lock, unadjusted descriptions with substantial small-sample uncertainty; they were not used to alter models."
    )
    rows = []
    for site in SITES:
        for group, label in [
            ("female", "Female"),
            ("male", "Male"),
            ("age_below_65", "Age <65"),
            ("age_65_plus", "Age >=65"),
        ]:
            x = v["subgroups"]["cross_site"][site][group]
            z = v["subgroups"]["within_site"][site][group]
            rows.append(
                [
                    SITES[site],
                    label,
                    f"{x['counts']['0']} / {x['counts']['1']}",
                    f"{x['auroc']['selected_pipeline']:.3f}" if "auroc" in x else "Not estimated",
                    f"{z['auroc']['selected_pipeline']:.3f}" if "auroc" in z else "Not estimated",
                ]
            )
    table(
        ["Site", "Stratum", "Control / PD", "Cross-site AUROC", "Within-site AUROC"],
        rows,
        [93, 94, 94, 115, 116],
    )
    para(
        "A subgroup point estimate is not a reliable fairness assessment in cohorts this small. Diagnoses, age, sex, medication schedules, acquisition sites and QC retention are entangled. No claims about protected-group equivalence or clinical safety follow from these slices."
    )

    title("9. Verification, reproducibility and next decision")
    au = v["audit"]
    table(
        ["Verified item", "Evidence"],
        [
            [
                "Registry and preprocessing",
                "11,211 planned records reconciled; all 7,988 cache hashes checked",
            ],
            [
                "Sources",
                f"{au['source_receipts_and_stat_verified']:,} selected source files reconciled with receipt SHA-256, byte count and modification time; processed inputs rehashed during preparation",
            ],
            [
                "All completed fits",
                "620 fit identities and weight hashes verified; inner validation AUC recalculated; outer selection and thresholds checked",
            ],
            [
                "Leakage boundaries",
                "Participant/site separation, auxiliary exclusion and one out-of-fold observation per person checked",
            ],
            [
                "Evaluation",
                "All outer AUROC, PR-AUC, Brier, sensitivity/specificity, balanced accuracy and confusion counts independently reproduced",
            ],
            [
                "Frozen records",
                "Implementation policy matches; all outer fits predate the outer lock, which predates saved predictions",
            ],
        ],
        [142, 370],
    )
    para(
        "Recovery fixed legacy BrainVision header encoding, misspelled companion references and one invalid EDF wall-clock second through byte-preserving/temporary-copy handling. Original downloads, numeric samples, completed caches and QC rules were preserved. Unexpected failures halted work instead of silently dropping files."
    )
    para(
        "Python "
        + au["python"]
        + "; "
        + "; ".join(k + " " + val for k, val in au["software"].items())
        + "."
    )
    heading("Decision supported by this experiment")
    para(
        "Keep the richer feature-based SVMs as strong research comparators. Do not claim that the new pretrained neural model is better: that hypothesis was not supported. Do not replace the model-selection procedure's 0.690 cross-site result with the retrospectively highest 0.763 family score. No direct numerical v2-to-v3 improvement claim is justified because training populations, recording duration, medication aggregation and eligibility changed."
    )
    para(
        "For Myna, the current output remains a cross-sectional research classifier. Symptom tracking needs repeated measurements aligned with clinical or patient-reported symptoms, and multimodal validation requires EEG/audio/movement from the same people. None of those outcomes is established here."
    )
    para(
        "Repository artifacts: frozen protocol, results Markdown/JSON, review diagnostics and model cards. Raw EEG, identities, predictions, checkpoints and fitted weights stay local. V1/v2 and the preliminary PDF are preserved. No clinical deployment or publication claim is made."
    )

    title("10. Research references and dataset provenance")
    para(
        "This is a focused research synthesis, not a systematic review. Architectural inspiration does not imply reproduction of a published method or performance score."
    )
    refs = [
        (
            "BENDR: self-supervised EEG representations",
            "Kostas et al. (2021). https://arxiv.org/abs/2101.12037",
        ),
        (
            "LaBraM: masked EEG modeling",
            "Jiang et al. (ICLR 2024). https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf",
        ),
        (
            "Generalizable EEG classification of PD",
            "Informatics in Medicine Unlocked (2023). https://doi.org/10.1016/j.imu.2023.101352",
        ),
        ("EEGNet", "Lawhern et al. (2018). https://arxiv.org/abs/1611.08024"),
        (
            "EEG convolutional decoding",
            "Schirrmeister et al. (2017). https://arxiv.org/abs/1703.05051",
        ),
        (
            "Periodic / aperiodic spectral parameterization",
            "Donoghue et al. (2020). https://doi.org/10.1038/s41593-020-00744-x",
        ),
    ]
    for label, ref in refs:
        heading(label)
        para(ref, True)
    heading("Supervised data")
    for ds in ["ds008768", "ds007526", "ds003490", "ds002778"]:
        para("https://openneuro.org/datasets/" + ds, True)
    heading("Auxiliary data and licenses")
    para(
        "OpenNeuro: ds003775, ds004148, ds004504, ds004796, ds005385, ds006036, ds007864 and ds007865; individual source pages and snapshot/license metadata are listed in docs/cohort-sources.json. PhysioNet bigP3BCI: https://physionet.org/content/bigp3bci/1.0.0/ . Source data licenses are separate from software licensing.",
        True,
    )
    para(
        "Companion documents: PD_PROTOCOL_V3.md; PD_CLASSIFIER_RESEARCH_PLAN.md; PD_PROTOCOL_V2.md; PD_Classifier_Comparison_Report.pdf (preserved preliminary report).",
        True,
    )

    pdf = ROOT / "output/pdf/PD_Classifier_V3_Report.pdf"

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#ccd9e1"))
        canvas.line(50, 39, 562, 39)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536b7d"))
        canvas.drawString(50, 26, "EEG Explorer | Protocol v3 | Exploratory research")
        canvas.drawRightString(562, 26, str(doc.page))

    SimpleDocTemplate(
        str(pdf),
        pagesize=(612, 792),
        rightMargin=50,
        leftMargin=50,
        topMargin=43,
        bottomMargin=51,
        title="PD EEG Expanded Comparison - Protocol v3",
        author="EEG Explorer research project",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "docs/PD_V3_RESULTS.md").write_text(
        "# Expanded PD EEG comparison: reviewed report\n\n" + "\n".join(md)
    )
    n = len(PdfReader(pdf).pages)
    (RUN / "report-validation.json").write_text(
        json.dumps(
            {
                "pages": n,
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "visual_review": "pending",
            },
            indent=2,
        )
    )
    print("PDF generated:", n, "pages")


if __name__ == "__main__":
    main()
