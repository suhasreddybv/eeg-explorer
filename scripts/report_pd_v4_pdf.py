"""Renders output/pdf/PD_Classifier_V4_Report.pdf and docs/PD_V4_RESULTS.md
from docs/pd-v4-results.json and docs/pd-v4-model-cards.json.

Style mirrors report_pd_v3_pdf.py. Every page must be rendered to PNG and
visually inspected before delivery (see --png).

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md for the rules this module reports against.
"""

import hashlib
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/pd-runs/v4"
NAMES = {
    "F0_rich_svm|I": "F0 v3 rich-feature SVM",
    "F1_relative_svm|I": "F1 relative-spectrum SVM",
    "F2_specparam_enet|I": "F2 specparam elastic-net",
    "F3_riemann_logistic|I": "F3 Riemannian tangent logistic",
    "F3_riemann_logistic|T": "F3 Riemannian tangent (transductive)",
    "F4_leapd|I": "F4 LEAPD / LPC",
    "F5_entropy_logistic|I": "F5 entropy logistic",
    "F6_boosting|I": "F6 gradient boosting",
    "S|I": "S stack, inductive (P1)",
    "S|T": "S stack, transductive (P2)",
}
ORDER = list(NAMES)
SITES = {"iowa": "Iowa", "tel_aviv": "Tel Aviv", "ucsd": "UC San Diego", "unm": "New Mexico"}


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def ci(m, key="auroc_ci95", alt="auroc_95ci"):
    c = m.get(key) or m.get(alt)
    return f"{c[0]:.3f}-{c[1]:.3f}" if c else "n/a"


def main():
    a = json.loads((ROOT / "docs/pd-v4-results.json").read_text())
    cards = json.loads((ROOT / "docs/pd-v4-model-cards.json").read_text())
    ext = a["external"]
    t20, t24 = ext["turku"]["confirmatory_n20"], ext["turku"]["corrected_n24_non_confirmatory"]
    ec, bl = ext["turku_eyes_closed_control"], ext["brainlat"]["results"]
    gates = a["gates"]
    story, md = [], []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=8, textColor=colors.HexColor("#263547")))
    styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="SmallReport", fontName="Helvetica", fontSize=8, leading=10.5, spaceAfter=7))
    for h, size, lead, after in (("Heading1", 18, 22, 14), ("Heading2", 12, 15, 9)):
        styles[h].fontSize, styles[h].leading, styles[h].spaceAfter, styles[h].keepWithNext = size, lead, after, True
    styles["Heading1"].textColor = colors.HexColor("#183b56")

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
        grid = [[Paragraph(escape(str(c)), styles["Cell"]) for c in row] for row in [headers] + rows]
        t = Table(grid, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf1")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#638298")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([t, Spacer(1, 10)])
        md.append("| " + " | ".join(headers) + " |")
        md.append("|" + "---|" * len(headers))
        for row in rows:
            md.append("| " + " | ".join(str(c) for c in row) + " |")
        md.append("")

    def detailed(site, kind):
        block = a["internal"][kind]["sites"][site]
        heading(f"{SITES[site]} | {block['S|I']['participants']} people")
        rows = []
        for v in ORDER:
            m = block[v]
            rows.append([NAMES[v], f3(m["auroc"]), ci(m), f3(m["sensitivity"]), f3(m["specificity"]), f3(m["pr_auc_average_precision"]), f3(m["brier"])])
        table(["Model", "AUROC", "95% CI", "Sens.", "Spec.", "PR-AUC", "Brier"], rows, [166, 40, 76, 42, 42, 46, 44])

    P1c, P1w = a["internal"]["cross_site"]["equal_site_macro_auroc"]["S|I"], a["internal"]["within_site"]["equal_site_macro_auroc"]["S|I"]

    # ------------------------------------------------------------------ cover
    title("Parkinson's EEG v4: a pre-registered pipeline and two external sites")
    para("Protocol v4 (frozen 28 September 2026; corrections v4.1, amendments v4.2 / 14.4 / 14.5) | Completed 29 September 2026")
    para("Research question: can a nested, pre-registered, multi-representation EEG pipeline exceed the v3 baseline internally (H1) and, decisively, generalise to a site it has never seen (H5)?")
    heading("The result")
    para(f"Internally the pre-declared primary pipeline P1 (a non-negative stack of six feature families with inductive harmonisation) reached an equal-site cross-site AUROC of {P1c:.3f} and within-site {P1w:.3f} (composite {a['composite']['S|I']:.3f}), exceeding the v3 pre-declared pipeline on both components (H1 met, with the caveat that cohort, QC and features all changed).")
    para(f"Externally it did not hold. On Turku, the first genuinely unexamined site (raw, eyes-open), the confirmatory one-shot gave AUROC {t20['pipelines']['P1']['auroc']:.3f} (95% CI {ci(t20['pipelines']['P1'])}, n=20); the corrected non-confirmatory re-run gave {t24['pipelines']['P1']['auroc']:.3f} ({ci(t24['pipelines']['P1'])}, n=24). H5 required the lower bound above 0.60 and failed decisively. On BrainLat (ICA-cleaned, eyes-closed, 16 PD / 13 HC) the one-shot gave {bl['pipelines']['P1']['auroc']:.3f} ({ci(bl['pipelines']['P1'])}); H5b also failed on its interval, although ranking survived and only the locked threshold did not transfer.")
    para("Every pre-registered secondary hypothesis was unsupported: new feature families did not beat the v3 features (H2), transductive harmonisation added 0.007 (H3), the foundation-model gate failed on all three conditions, and test-retest reliability of the output was 0.44 against a 0.6 target. The generalisation claim that would justify a wearable product is not supported by this work.")
    table(["Completed work", "Scope"], [
        ["Cohort", "537 assigned people at four sites; one session per person (A1); 324 eligible after label-free QC and A2"],
        ["Features", "Six new families plus v3-rich; three harmonisation regimes with guardrails; four reduced montages; frozen CBraMod embeddings"],
        ["Evaluation", "Four held-out-site folds and three within-site folds; D3 selection objective inside every fold; outer and final locks"],
        ["External", "Turku one-shot (n=20 confirmatory, n=24 corrected), Turku eyes-closed control (n=23), BrainLat one-shot (n=29)"],
        ["Negative", "Permutation null, site-identity probes, age/sex baselines, prevalence stress tests, FM random-init and label-permutation controls"],
    ], [110, 402])
    heading("Interpretation comes first")
    para("Internal cross-site folds share acquisition era, montage conventions and preprocessing lineage in ways an external Finnish clinical recording does not. The site-identity probe (AUROC 0.81-0.97 for every representation) warned before the shot that site dominates these features; Turku confirmed it. BrainLat, where the site's own team had already ICA-cleaned the data, is the first hint that the failure is partly a raw-versus-cleaned data effect - a hypothesis for v5, not a finding.")
    para("This work supports continued research and does not establish a diagnostic test, a progression score, an around-ear device result or a symptom-tracking product. The v1-v3 reports are unchanged.")

    # ------------------------------------------------------------ 1. design
    title("1. Pre-registration, evaluation design and locks")
    para("The protocol was frozen and hashed before any supervised fit; every pipeline module is hashed into the implementation policy and a re-freeze after a source change raises. Corrections after the freeze (v4.1 filtered preref128/LPC; v4.2 second locked site and condition-shift control; 14.4 BrainLat cohort; 14.5 Copenhagen) are dated in the protocol and each states what it could and could not have influenced.")
    table(["Stage", "Frozen rule"], [
        ["Primary evaluation", "Hold out one site, train on three, repeat for all four; three within-site participant folds; every person in exactly one fold"],
        ["Selection", "Inside each outer fold only: D3 = 0.5 x cross + 0.5 x within - 0.5 x SD across sites, on inner leave-one-site-out"],
        ["Threshold", "Per-fold inner out-of-fold balanced-accuracy optimum; never adjusted on a test set"],
        ["Locks", "outer-models-lock before outer predictions; final all-data fits then final-models-lock; locked-site labels sealed until both exist"],
        ["One-shot", "Each external site scored once; a repeat needs an explicit override flag and is labelled non-confirmatory"],
        ["Gates", "G1 baseline reproduction (|delta| < 0.01), G2 features add >= 0.02, G3 transductive gain >= 0.02 and stress test, G4 report as-is, G5 delivery, G6 montage gap <= 0.05"],
    ], [110, 402])
    heading("Families selected inside each outer fold")
    def short(variant):
        return variant.split("_")[0] + ("T" if variant.endswith("|T") else "")

    def params(p):
        return ",".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in p.items())

    rows = []
    for name, d in a["selections_by_outer_fold"].items():
        sel = d.get("selections") or {}
        settings = "; ".join(f"{short(v)} {params(s['params'])}" for v, s in sel.items())
        weights = " | ".join(
            f"{'P1' if tag == 'S|I' else 'P2'}: " + ", ".join(f"{short(v)} {w:.2f}" for v, w in ws.items() if w > 0)
            for tag, ws in (d.get("stacker_weights") or {}).items()
        )
        rows.append([name.replace("_", " "), settings or "n/a", weights or "n/a"])
    table(["Outer fold", "Inner-selected settings (D3 optimum per family)", "Non-negative stack weights"], rows, [70, 262, 180])
    para("Family codes: F0 v3-rich SVM, F1 relative-spectrum SVM, F2 specparam elastic-net, F3 Riemannian logistic (F3T transductive), F4 LEAPD, F5 entropy logistic, F6 boosting. Weights are the stacker's non-negative coefficients over calibrated family probabilities; a zero weight means the family was dropped in that fold.", True)
    para("Bootstrap intervals resample participants within the fixed sites. They do not quantify uncertainty from having four training sites, from refitting, or from earlier exposure to these datasets in v1-v3. Paired deltas are exploratory and unadjusted for multiplicity.", True)

    # ------------------------------------------------------------ 2. cohort
    title("2. Cohort, session rule and label-free QC")
    para("A1: one session per person, resolved per site from documented metadata (UCSD ses-on / ses-hc; New Mexico the ON-medication session; Iowa the earliest resting session; Tel Aviv the single recording). This makes session count constant and structurally removes the v3 recording-metadata confound that gave AUROC 1.000 at UCSD and New Mexico. A2: the first 15 accepted 4 s windows, minimum 10.")
    para(f"QC v4 thresholds were derived label-free on auxiliary cohorts and all sat at their ceilings (peak-to-peak {a['qc_thresholds']['PTP_UV']} uV, step {a['qc_thresholds']['STEP_UV']} uV, line-noise ratio {a['qc_thresholds']['LINE_RATIO']}); {a['qc_thresholds']['n_windows']:,} windows informed them. Of 537 assigned people, {a['cohort']['n_ready']} produced accepted windows and {a['cohort']['n_eligible_a2']} met A2. The drop from v3's 440 is expected: A1 removes the backup session v3 had when a first recording failed QC.")
    rows = []
    for site in SITES:
        for dx in ("PD", "control"):
            c = a["cohort"]["coverage_by_site_diagnosis"][site][dx]
            d = a["cohort"]["demographics_assigned"][site][dx]
            rows.append([SITES[site], dx, f"{c['n_meets_a2_min_10_windows']}/{c['n_people']}", f"{100 * (1 - c['n_meets_a2_min_10_windows'] / c['n_people']):.1f}%", f"{d['age_mean']:.1f} ({d['age_sd']:.1f})", f"{d['female']}/{d['male']}"])
    table(["Site", "Diagnosis", "Eligible / assigned", "Excluded", "Age mean (SD), assigned", "F / M, assigned"], rows, [80, 60, 100, 60, 122, 90])
    para("Exclusion is differential at Tel Aviv (PD 56%, controls 36%) and Iowa (PD 37%, controls 39%): the label-free rules remove more PD recordings where PD recordings are noisier. That selection travels with every internal number. Age and sex are those of the assigned cohort; demographics of the retained subset were not separately tabulated.", True)
    heading("Confound audit after A1")
    rows = [[SITES[s], f3(v["auroc"]), v["n_test"]] for s, v in (a["cohort"].get("demographic_auroc_by_site") or {}).items() if s in SITES]
    if rows:
        table(["Site", "Age/sex-only leave-one-site-out AUROC", "People"], rows, [160, 262, 90])
    para(f"Session count is {{1}} for every supervised person, so the recording-metadata baseline carries no information by construction. Age/sex-only cross-site AUROC is {a['negative_controls']['age_sex_only_cross_site_auroc']:.3f} overall.")

    # ------------------------------------------------------------ 3. methods
    title("3. Representations, harmonisation and models")
    para("Seventeen named scalp channels, 256 Hz, 1-40 Hz zero-phase filtering, average reference over the 17 channels for the spectral/covariance families; a band-passed but un-referenced 128 Hz copy (preref128) is kept for the LPC family and for montage re-referencing. Electrode positions are never guessed.")
    table(["Family", "Representation"], [
        ["B-1", "Relative band spectra per channel (684 inputs)"],
        ["B-2", "specparam aperiodic exponent/offset and adjusted band powers (163)"],
        ["B-3", "Riemannian tangent-space covariance (153) with C1 re-centering"],
        ["B-4", "Order-12 LPC coefficients via Levinson-Durbin (408), LEAPD-style hyperplane scores"],
        ["B-5", "Entropy and complexity measures (136)"],
        ["B-6", "Robust amplitude distribution summaries (205)"],
        ["v3-rich", "The v3 log-PSD/log-covariance vector (1,309), the F0 baseline"],
        ["C3", "Normative spline quantile z-scores of 38 regional features fitted on 467 healthy people aged 40+"],
    ], [70, 442])
    heading("Harmonisation guardrails")
    h = a["harmonisation"]
    rows = []
    for c in ("C1", "C2"):
        for mode in ("label_aware", "label_free"):
            g = h[c][mode]
            rows.append([c, mode.replace("_", "-"), f"{g['sites_passed']}/{g['sites_total']}", f"{g['max_spread']:.4f}", "pass" if g["passed"] else "fail"])
    table(["Regime", "Criterion", "Sites within tolerance", "Max AUROC spread across prevalence", "Verdict (>= 3 of 4)"], rows, [55, 80, 110, 160, 107])
    para("C1 (Riemannian re-centering) passed at all four sites under both criteria and is the only transductive method allowed into the model grid. C2 (robust per-site z) failed the label-aware criterion at UCSD and New Mexico and was excluded; its label-free companion disagreed, which is itself reported. C3 is inductive and always on.")
    heading("Model grid")
    para("F0 v3-rich SVM (the baseline); F1 relative-spectrum SVM; F2 specparam elastic-net; F3 Riemannian tangent logistic (inductive, and transductive with C1); F4 LEAPD/LPC; F5 entropy logistic; F6 gradient boosting; S a non-negative stacker over the family probabilities with Platt calibration. P1 = S inductive; P2 = S transductive; P3 = the S stack on the M4T around-ear montage, fixed by reviewed decision, not by score.")

    # ------------------------------------------------------------ 4. internal
    title("4. Internal results: equal-site means and gates")
    rows = [[NAMES[v], f3(a["internal"]["cross_site"]["equal_site_macro_auroc"][v]), f3(a["internal"]["within_site"]["equal_site_macro_auroc"][v]), f3(a["composite"][v])] for v in ORDER]
    table(["Family / pipeline", "Cross-site AUROC", "Within-site AUROC", "Composite"], rows, [220, 98, 98, 96])
    para("Composite is 0.5 x cross-site + 0.5 x within-site. Only P1 (S|I) and P2 (S|T) are pre-declared results; every other row is descriptive. F6 boosting has the highest cross-site mean but was not pre-declared and is not promoted.")
    g2, g3, g4, h1 = gates["G2_b_features_add_over_v3_rich"], gates["G3_transductive_gain_and_stress_test"], gates["G4_primary_outer_composite"], gates["H1_vs_v3_predeclared"]
    table(["Gate / hypothesis", "Requirement", "Observed", "Verdict"], [
        ["G1", "Reproduce v3 within 0.01", "cross 0.7627 vs 0.763; within 0.7113 vs 0.711", "met"],
        ["G2 / H2", "Best new family >= F0 + 0.02 (inner)", f"{g2['best_b_family']}: +{g2['delta']:.4f}", "not met"],
        ["G3 / H3", "Transductive gain >= 0.02 and stress test", f"+{g3['transductive_cross_site_gain']:.3f}; C1 pass, C2 fail", "not met"],
        ["G4", "Report the primary composite as is", f"P1 cross {g4['cross_site']:.3f}, within {g4['within_site']:.3f}", "reported"],
        ["H1", ">= 0.75 on each component vs v3's 0.690 / 0.693", f"{h1['v4_1']['cross']:.3f} / {h1['v4_1']['within']:.3f} (+{h1['delta_composite']:.3f} composite)", "met, with cohort caveat"],
        ["G6", "Best pre-declared montage within 0.05 of T1", f"M5S {a['reduced_montages']['G6']['best_composite']:.3f} vs T1 {a['reduced_montages']['G6']['t1_composite']:.3f}", "met"],
    ], [70, 150, 200, 92])
    para(f"The v4.1 correction (band-passing preref128 and the LPC input) changed only F4 and F5; the hypothesis that the unfiltered input explained F4's poor result was wrong (F4 moved {gates['correction_effect']['F4_leapd']['old']:.3f} to {gates['correction_effect']['F4_leapd']['new']:.3f}); F5 improved from {gates['correction_effect']['F5_entropy']['old']:.3f} to {gates['correction_effect']['F5_entropy']['new']:.3f}. No gate verdict changed.", True)

    for kind, heading_text in (("cross_site", "Primary cross-site"), ("within_site", "Separate within-site")):
        for index, sites in enumerate((["iowa", "tel_aviv"], ["ucsd", "unm"])):
            title(f"{heading_text}: " + ("Iowa and Tel Aviv" if index == 0 else "UCSD and New Mexico"))
            for site in sites:
                detailed(site, kind)
            para("Sensitivity and specificity use the per-fold inner threshold. PR-AUC depends on PD prevalence. High AUROC alone does not imply a usable operating threshold; the external sites show why.", True)

    # ------------------------------------------------------------ 5. negative
    title("5. Negative controls and site identity")
    n = a["negative_controls"]
    para(f"Permutation null (labels permuted within site, {n['permutation_null']['n_permutations']} draws): mean composite {n['permutation_null']['mean']:.3f}, SD {n['permutation_null']['sd']:.3f}, max {n['permutation_null']['max']:.3f}. P1's 0.770 sits far above it: the internal signal is real, whatever it is.")
    rows = [[k, f3(v)] for k, v in n["site_identity_probe_macro_auroc"].items()]
    table(["Representation", "Site-identity probe AUROC (leave-one-site-out, which site?)"], rows, [160, 352])
    para("Every representation carries the recording site almost perfectly. Harmonisation reduced but did not remove it (B-2 with C3 is lowest at 0.83). This is the mechanism by which an internally strong pipeline can fail on a new site, and it was documented before Turku was opened.")

    # ------------------------------------------------------------ 6. montages
    title("6. T2 reduced-channel montages")
    rows = []
    for mname, d in a["reduced_montages"]["montages"].items():
        s = d["stack_S_I"]
        rows.append([mname, ", ".join(d["channels"]), f3(s["cross_site"]["macro"]), f3(s["within_site"]["macro"]), f3(s["composite"])])
    table(["Montage", "Channels", "Cross-site", "Within-site", "Composite"], rows, [60, 190, 80, 80, 102])
    para(f"Montages are re-referenced within their own channels only; no 17-channel information enters. G6 passed: the best pre-declared montage (M5S, {a['reduced_montages']['G6']['best_composite']:.3f}) is within 0.05 of T1. P3 was fixed to M4T (T7, T8, P7, P8) as the around-ear form factor of interest by reviewed decision, explicitly not on score. Distilling the 17-channel stack into M8 did not help (delta {a['reduced_montages']['distillation_M8']['delta_distilled_minus_hard']:+.3f}).")
    para("These are montage subsets of full research caps, not around-ear device recordings. No public around-ear PD dataset exists (confirmed by the 29 September sweep), so no around-ear device claim can be made from any number here.")

    # ------------------------------------------------------------ 7. FM
    title("7. Track E: frozen foundation-model embeddings")
    fmr = a["foundation_models"]
    para(f"Licences were checked before any weights were downloaded: CBraMod (BSD-3-Clause) was used; LaBraM was skipped because its temporal embedding does not match the protocol's 4 s window; REVE was skipped under the unclear-licence rule. Frozen CBraMod embeddings are {fmr['embedding_dimension']:,}-dimensional for {fmr['n_people']} people.")
    d = fmr["gate"]["detail"]
    table(["FM gate condition", "Observed", "Verdict"], [
        ["Beats negative controls", f"probe {d['probe_composite']:.3f} vs random-init {d['random_init_composite']:.3f}, permuted {d['permuted_composite']:.3f}", "fail"],
        ["Adds >= 0.01 to the stack", f"{d['stack_composite']:.3f} to {d['stack_with_fm_composite']:.3f} ({d['delta_to_stack']:+.3f})", "fail"],
        ["Site probe no worse than v3-rich", f"{d['probe_site_identity_auroc']:.3f} vs {d['v3_rich_site_identity_auroc']:.3f}", "fail"],
    ], [150, 262, 100])
    para(fmr["caveat"], True)

    # ------------------------------------------------------------ 8. fusion
    title("8. T3: fusion-ready output, cognition and reliability")
    fu = a["fusion"]
    para(f"The T3 output is a prior-free log-likelihood ratio, logit(p) - logit(training prevalence {fu['training_prevalence']:.3f}), written to the Myna schema (docs/myna_eeg_output_v1.schema.json) with a guarded naive-Bayes combiner. Adding a near-duplicate modality inflated the combined evidence from {fu['naive_bayes_overcounting']['mean_abs_llr_single']:.2f} to {fu['naive_bayes_overcounting']['mean_abs_llr_with_near_duplicate']:.2f}, almost as much as a genuinely independent one ({fu['naive_bayes_overcounting']['mean_abs_llr_with_independent']:.2f}): that gap is the over-counting the guard exists for.")
    rows = []
    for key, label in (("cognition_moca_T1", "MoCA, 17 channels"), ("cognition_moca_M8", "MoCA, M8"), ("cognition_updrs3_T1", "UPDRS-III, 17 channels")):
        c = fu[key]["grouped_cv"]
        rows.append([label, fu[key]["n"], f"{c['spearman_rho']:.3f} ({c['rho_ci95'][0]:.2f} to {c['rho_ci95'][1]:.2f})", f"{c['mae']:.2f}"])
    table(["Target (PD only, exploratory)", "n", "Grouped-CV Spearman rho (95% CI)", "MAE"], rows, [170, 40, 200, 102])
    tr = fu["test_retest"]
    para(f"Test-retest: {tr['n_people_with_second_session']} Iowa people with a second session, scored by a model trained on the other three sites. ICC(2,1) of the LLR is {tr['icc_2_1_of_llr']:.2f} (Spearman {tr['spearman_between_sessions']:.2f}) against a pre-declared 0.6 target: not met. An output that does not agree with itself across sessions cannot track symptoms.")

    # ------------------------------------------------------------ 9. external
    title("9. External sites: Turku, the eyes-closed control, BrainLat")
    heading("Turku (OSF pehj9): raw, eyes-open, 66-channel clinical cap")
    para("Forty recordings; labels sealed at download and resolved from medication fields, never guessed. Label-free QC excluded 16 before any label was opened (13 with no accepted window, 3 below A2). Two bugs were caught before the shot was spent (transductive re-centering on a new site; P3 fitting on 17-channel data while labelled M4T) and one after: zero-padded file names versus unpadded spreadsheet IDs silently dropped every single-digit participant, 4 of 24. The corrected re-run is reported as non-confirmatory.")
    rows = []
    for tag in ("P1", "P2", "P3"):
        m20, m24 = t20["pipelines"][tag], t24["pipelines"][tag]
        rows.append([tag + (" (primary)" if tag == "P1" else ""), f"{m20['auroc']:.3f} ({ci(m20)})", f"{m24['auroc']:.3f} ({ci(m24)})", f"{m24['sensitivity']:.2f} / {m24['specificity']:.2f}"])
    table(["Pipeline", "Confirmatory n=20", "Corrected n=24 (non-confirmatory)", "Sens / spec at locked threshold (n=24)"], rows, [90, 130, 160, 132])
    para(f"H5 required P1's lower bound above 0.60; observed {t20['pipelines']['P1']['auroc_ci95'][0]:.3f} (n=20) and {t24['pipelines']['P1']['auroc_ci95'][0]:.3f} (n=24). Both intervals span chance. Age (PD {t24['balance']['age_mean_pd']:.1f}, control {t24['balance']['age_mean_control']:.1f}) and sex are balanced. This is the most important result in v4 and it contradicts every internal number.")
    heading("Turku eyes-closed condition-shift control (protocol 14.2, non-confirmatory)")
    if ec:
        rows = []
        for tag, e in ec["pipelines"].items():
            rows.append([tag, e["n_overlap"], f3(e.get("eo_auroc_overlap")), f3(e.get("ec_auroc_overlap")), f"{e.get('delta_eo_minus_ec', float('nan')):+.3f}"])
        table(["Pipeline", "Overlap n", "Eyes-open AUROC", "Eyes-closed AUROC", "Delta EO - EC"], rows, [80, 70, 110, 110, 142])
        para(f"Pre-registered rule on P1: |delta| <= 0.05 means a BrainLat result is interpretable as site generalisation. Observed delta {ec['decision_rule']['delta_P1']:+.3f}: verdict '{ec['decision_rule']['verdict']}'. Honest reading: both conditions sit at chance on Turku, so the control establishes that condition is not the reason, nothing more. P3 is the exception (delta +0.20 on n=18): the around-ear montage is condition-sensitive.")
    heading("BrainLat (Synapse): ICA-cleaned, eyes-closed, 128-electrode Biosemi, Chile")
    if bl:
        insp = ext["brainlat"]["inspection"]["naming_by_group_site"] if ext["brainlat"].get("inspection") else {}
        para(f"A label-free inspection before the seal found that files named *_reject.set are a different pipeline (epoched, no ICA, no low-pass) present only under PD (PD/AR {insp.get('3_PD/AR', {}).get('flagged', '?')}/{insp.get('3_PD/AR', {}).get('flagged', 0) + insp.get('3_PD/AR', {}).get('plain', 0)}, PD/CL {insp.get('3_PD/CL', {}).get('flagged', '?')}/{insp.get('3_PD/CL', {}).get('flagged', 0) + insp.get('3_PD/CL', {}).get('plain', 0)}, HC 0/46), that no raw data exists, and that the cap is Biosemi A1..D32. Protocol 14.4 fixed the cohort to plain-named Chile files with 17 channels spline-interpolated from each file's own electrode coordinates, before any label was sealed. Fourteen of 27 Chile controls turned out to be header-only on Synapse; 16 PD and 13 HC were scored.")
        rows = []
        for tag, m in bl["pipelines"].items():
            rows.append([tag + (" (primary)" if tag == "P1" else ""), f"{m['auroc']:.3f} ({ci(m)})", f3(m["pr_auc"]), f3(m["brier"]), f"{m['sensitivity']:.2f} / {m['specificity']:.2f}"])
        table(["Pipeline", "AUROC (95% CI)", "PR-AUC", "Brier", "Sens / spec at locked threshold"], rows, [90, 150, 60, 60, 152])
        para("H5b fails as written (P1 lower bound 0.500). But ranking is at 0.70-0.74 with P2/P3 intervals excluding chance, while specificity is 1.00 and sensitivity 0.25-0.31: BrainLat's cleaned signals score 'healthier' across the board and the locked threshold misses most PD. Discrimination survived a shift that the threshold did not; P2's harmonisation did no better on it than P1. With 13 controls no interval could have cleared 0.60 unless AUROC were about 0.85.")
        para("Caveats: eyes-closed against eyes-open training; ICA-cleaned and low-passed against raw; the 13 excluded PD were the artifact-heavier recordings, so the PD sample is selected toward cleaner EEG; interpolated rather than physical electrodes; n=29.", True)
    heading("Two-site summary and the third site")
    para("On the external site with raw eyes-open data the pipeline is at chance; on the one with pre-cleaned eyes-closed data it ranks at about 0.70 with a threshold that no longer transfers. Neither meets its gate. A repository sweep on 29 September found no further downloadable PD-versus-control resting EEG site and no public around-ear PD data; the one request-only candidate (Copenhagen, 40 PD OFF-medication / 26 HC, raw, eyes-open, 128 channels) is pre-registered as protocol 14.5 and would test, not rescue, the raw-versus-cleaned hypothesis.")

    # ------------------------------------------------------------ 10. corrections
    title("10. Bugs, corrections and what each could have changed")
    table(["Found", "Defect", "Effect and handling"], [
        ["Stage 7", "preref128 and the LPC input were built from the unfiltered signal", "Only F4/F5 changed; no gate changed; protocol v4.1 dated correction"],
        ["Stage 11, before the shot", "Transductive P2 looked up a new site in the training re-centering map", "Would have crashed; extend_transductive re-centres a locked site on itself; tested"],
        ["Stage 11, before the shot", "P3 was fitting 17-channel features while labelled M4T", "Caught by a dimension mismatch; final_fit now takes a montage; tested"],
        ["Stage 11, after the shot", "Zero-padded file IDs versus unpadded spreadsheet IDs", "4 of 24 Turku people silently dropped; corrected re-run reported as non-confirmatory"],
        ["v4.2", "OSF pagination duplicated one eyes-closed file six times and hid five", "Listing de-duplicated by file id; 39 files, not 44; protocol text corrected"],
        ["14.4", "Standard-montage sphere centre 16 mm behind the Biosemi origin", "Targets were tilted about 10 degrees; positions aligned by fitted sphere centres; synthetic-field test"],
        ["14.4", "Biosemi C3/C4 labels collide with 10-20 names", "Sources prefixed before interpolation; tested"],
        ["14.4", "PD headers reference pre-rename .fdt names; 14 HC header-only; macOS ._ shadows", "Resolved in scratch directories, originals untouched; header-only recorded, not scored"],
    ], [92, 200, 220])

    # ------------------------------------------------------------ 11. verification
    title("11. Verification, compute and the decision")
    sw = a["software"]
    table(["Verified item", "Evidence"], [
        ["Baseline reproduction", "v3 numbers reproduced two ways within 0.001 before any v4 fit"],
        ["Archive", "41,746 planned files reconciled against receipts with zero mismatches; Turku and BrainLat hashes verified against source-reported values"],
        ["Leakage", "Tests poison held-out features with NaN and assert fitted transforms are unchanged; participant/site disjointness asserted per fold"],
        ["Locks", "Outer lock precedes outer predictions; final lock precedes each one-shot; sealed-label hashes checked before every evaluation"],
        ["Compute", f"Supervised campaign forecast {a['compute']['supervised_forecast_hours']:.2f} h against a 96 h gate ({a['compute']['measured_seconds_per_fold']:.0f} s per fold); local Mac only"],
        ["Tests", "176 passing; ruff clean; no raw EEG, identities, predictions, weights or local paths in Git"],
    ], [120, 392])
    para("Python " + sw["python"] + "; " + "; ".join(f"{k} {v}" for k, v in sw.items() if k != "python") + ". " + a["hardware"] + ".", True)
    heading("Model cards (docs/pd-v4-model-cards.json)")
    table(["Card", "Pipeline", "Composite", "Turku n=20 / n=24", "BrainLat", "Threshold"], [
        [tag, c["pipeline"][:44], f3(c["composite"]),
         f"{c['locked_site_turku']['confirmatory_n20']['auroc']:.3f} / {c['locked_site_turku']['corrected_n24_non_confirmatory']['auroc']:.3f}",
         f3(c["locked_site_brainlat"]["auroc"]) if c.get("locked_site_brainlat") else "n/a", f3(c["threshold"])]
        for tag, c in cards.items()
    ], [40, 190, 62, 100, 60, 60])
    heading("Decision supported by this experiment")
    para("Do not build a product on this classifier. The internal gains over v3 are real but the pipeline does not generalise to the one raw external site it was tested on, its output is not test-retest reliable, and none of the pre-registered secondary hypotheses held. For Myna the honest deliverable is the fusion-ready LLR schema with a guarded combiner, plus the negative evidence.")
    para("What v5 should test, in this order: whether the Turku failure is a raw-versus-cleaned data effect (Copenhagen, pre-registered 14.5; or apply an identical cleaning pipeline to every site before features); site-invariant training with the site-identity probe as an explicit objective; and a reliability gate (ICC >= 0.6) before any external shot.")
    para("Repository artefacts: frozen protocol with dated amendments, stage log, results JSON, model cards for P1/P2/P3, this report. Raw EEG, labels, predictions, checkpoints and weights stay local. No clinical deployment or publication claim is made.")

    # ------------------------------------------------------------ 12. references
    title("12. References and data provenance")
    for label, ref in (
        ("Riemannian re-centering for cross-site EEG", "Zanini et al. (2018). https://doi.org/10.1109/TBME.2017.2742541"),
        ("Periodic / aperiodic spectral parameterization", "Donoghue et al. (2020). https://doi.org/10.1038/s41593-020-00744-x"),
        ("LEAPD: linear predictive coding for PD EEG", "Anjum et al. (2020). https://doi.org/10.1016/j.parkreldis.2020.08.001"),
        ("CBraMod", "Wang et al. (ICLR 2025). https://arxiv.org/abs/2412.07236"),
        ("Turku eyes-open / eyes-closed PD EEG", "Railo et al., OSF pehj9. https://osf.io/pehj9/ (no licence shown; research use only, contact the authors before other use)"),
        ("BrainLat", "Prado et al. (2023) Sci Data. Synapse syn51549340, CC-BY-4.0"),
    ):
        heading(label)
        para(ref, True)
    heading("Supervised data")
    for ds in ("ds008768", "ds007526", "ds003490", "ds002778"):
        para("https://openneuro.org/datasets/" + ds, True)
    para("Companion documents: PD_PROTOCOL_V4.md; PD_V4_LOG.md; pd-v4-results.json; pd-v4-model-cards.json; PD_V3_RESULTS.md.", True)

    pdf = ROOT / "output/pdf/PD_Classifier_V4_Report.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#ccd9e1"))
        canvas.line(50, 39, 562, 39)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536b7d"))
        canvas.drawString(50, 26, "EEG Explorer | Protocol v4 | Pre-registered research; not a clinical claim")
        canvas.drawRightString(562, 26, str(doc.page))

    SimpleDocTemplate(str(pdf), pagesize=(612, 792), rightMargin=50, leftMargin=50, topMargin=43, bottomMargin=51,
                      title="PD EEG v4 - pre-registered pipeline and external sites", author="EEG Explorer research project").build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "docs/PD_V4_RESULTS.md").write_text("# PD EEG v4: pre-registered pipeline and two external sites\n\n" + "\n".join(md))
    n = len(PdfReader(pdf).pages)
    (RUN / "report-validation.json").write_text(json.dumps({"pages": n, "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "visual_review": "pending"}, indent=2))
    print("PDF generated:", n, "pages ->", pdf)
    if "--png" in sys.argv:
        import pypdfium2 as pdfium

        out = Path(sys.argv[sys.argv.index("--png") + 1])
        out.mkdir(parents=True, exist_ok=True)
        doc = pdfium.PdfDocument(str(pdf))
        for i in range(len(doc)):
            doc[i].render(scale=1.3).to_pil().save(out / f"page-{i + 1:02d}.png")
        print("rendered", len(doc), "pages to", out)
    return n


if __name__ == "__main__":
    main()
