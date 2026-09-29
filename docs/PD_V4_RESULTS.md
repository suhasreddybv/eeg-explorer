# PD EEG v4: pre-registered pipeline and two external sites

## Parkinson's EEG v4: a pre-registered pipeline and two external sites

Protocol v4 (frozen 28 September 2026; corrections v4.1, amendments v4.2 / 14.4 / 14.5) | Completed 29 September 2026

Research question: can a nested, pre-registered, multi-representation EEG pipeline exceed the v3 baseline internally (H1) and, decisively, generalise to a site it has never seen (H5)?

### The result

Internally the pre-declared primary pipeline P1 (a non-negative stack of six feature families with inductive harmonisation) reached an equal-site cross-site AUROC of 0.765 and within-site 0.774 (composite 0.770), exceeding the v3 pre-declared pipeline on both components (H1 met, with the caveat that cohort, QC and features all changed).

Externally it did not hold. On Turku, the first genuinely unexamined site (raw, eyes-open), the confirmatory one-shot gave AUROC 0.570 (95% CI 0.280-0.810, n=20); the corrected non-confirmatory re-run gave 0.545 (0.301-0.776, n=24). H5 required the lower bound above 0.60 and failed decisively. On BrainLat (ICA-cleaned, eyes-closed, 16 PD / 13 HC) the one-shot gave 0.697 (0.500-0.880); H5b also failed on its interval, although ranking survived and only the locked threshold did not transfer.

Every pre-registered secondary hypothesis was unsupported: new feature families did not beat the v3 features (H2), transductive harmonisation added 0.007 (H3), the foundation-model gate failed on all three conditions, and test-retest reliability of the output was 0.44 against a 0.6 target. The generalisation claim that would justify a wearable product is not supported by this work.

| Completed work | Scope |
|---|---|
| Cohort | 537 assigned people at four sites; one session per person (A1); 324 eligible after label-free QC and A2 |
| Features | Six new families plus v3-rich; three harmonisation regimes with guardrails; four reduced montages; frozen CBraMod embeddings |
| Evaluation | Four held-out-site folds and three within-site folds; D3 selection objective inside every fold; outer and final locks |
| External | Turku one-shot (n=20 confirmatory, n=24 corrected), Turku eyes-closed control (n=23), BrainLat one-shot (n=29) |
| Negative | Permutation null, site-identity probes, age/sex baselines, prevalence stress tests, FM random-init and label-permutation controls |

### Interpretation comes first

Internal cross-site folds share acquisition era, montage conventions and preprocessing lineage in ways an external Finnish clinical recording does not. The site-identity probe (AUROC 0.81-0.97 for every representation) warned before the shot that site dominates these features; Turku confirmed it. BrainLat, where the site's own team had already ICA-cleaned the data, is the first hint that the failure is partly a raw-versus-cleaned data effect - a hypothesis for v5, not a finding.

This work supports continued research and does not establish a diagnostic test, a progression score, an around-ear device result or a symptom-tracking product. The v1-v3 reports are unchanged.

## 1. Pre-registration, evaluation design and locks

The protocol was frozen and hashed before any supervised fit; every pipeline module is hashed into the implementation policy and a re-freeze after a source change raises. Corrections after the freeze (v4.1 filtered preref128/LPC; v4.2 second locked site and condition-shift control; 14.4 BrainLat cohort; 14.5 Copenhagen) are dated in the protocol and each states what it could and could not have influenced.

| Stage | Frozen rule |
|---|---|
| Primary evaluation | Hold out one site, train on three, repeat for all four; three within-site participant folds; every person in exactly one fold |
| Selection | Inside each outer fold only: D3 = 0.5 x cross + 0.5 x within - 0.5 x SD across sites, on inner leave-one-site-out |
| Threshold | Per-fold inner out-of-fold balanced-accuracy optimum; never adjusted on a test set |
| Locks | outer-models-lock before outer predictions; final all-data fits then final-models-lock; locked-site labels sealed until both exist |
| One-shot | Each external site scored once; a repeat needs an explicit override flag and is labelled non-confirmatory |
| Gates | G1 baseline reproduction (|delta| < 0.01), G2 features add >= 0.02, G3 transductive gain >= 0.02 and stress test, G4 report as-is, G5 delivery, G6 montage gap <= 0.05 |

### Families selected inside each outer fold

| Outer fold | Inner-selected settings (D3 optimum per family) | Non-negative stack weights |
|---|---|---|
| site iowa | F0 C=1; F1 C=1; F2 C=0.1,l1_ratio=0.8; F3 C=1; F4 k=3; F5 C=0.1; F6 max_depth=3; F3T C=0.1 | P1: F0 0.34, F2 0.39, F3 0.04, F5 0.16, F6 0.46 | P2: F0 0.28, F2 0.38, F3T 0.14, F5 0.19, F6 0.42 |
| site tel aviv | F0 C=10; F1 C=10; F2 C=0.1,l1_ratio=0.8; F3 C=0.1; F4 k=3; F5 C=0.1; F6 max_depth=2; F3T C=0.1 | P1: F0 0.73, F2 0.63, F3 0.09, F5 0.17, F6 0.15 | P2: F0 0.69, F2 0.64, F3T 0.11, F5 0.16, F6 0.16 |
| site ucsd | F0 C=10; F1 C=1; F2 C=0.1,l1_ratio=0.8; F3 C=0.1; F4 k=3; F5 C=0.1; F6 max_depth=3; F3T C=0.1 | P1: F0 0.33, F1 0.15, F3 0.05, F5 0.10, F6 0.25 | P2: F0 0.32, F1 0.15, F3T 0.14, F5 0.10, F6 0.20 |
| site unm | F0 C=1; F1 C=1; F2 C=0.1,l1_ratio=0.8; F3 C=0.1; F4 k=3; F5 C=0.1; F6 max_depth=3; F3T C=0.1 | P1: F0 0.33, F1 0.46, F3 0.05, F5 0.13, F6 0.10 | P2: F0 0.30, F1 0.45, F3T 0.12, F5 0.12, F6 0.11 |
| within 0 | F0 C=1; F1 C=1; F2 C=0.1,l1_ratio=0.8; F3 C=0.1; F4 k=3; F5 C=0.1; F6 max_depth=2; F3T C=1 | P1: F0 0.29, F1 0.61, F3 0.15, F4 0.15, F5 0.26 | P2: F0 0.41, F1 0.58, F3T 0.06, F4 0.13, F5 0.25 |
| within 1 | F0 C=1; F1 C=10; F2 C=0.1,l1_ratio=0.8; F3 C=0.1; F4 k=3; F5 C=0.1; F6 max_depth=2; F3T C=0.1 | P1: F0 0.20, F1 0.47, F2 0.00, F3 0.14, F5 0.06, F6 0.52 | P2: F0 0.17, F1 0.46, F2 0.01, F3T 0.18, F5 0.06, F6 0.52 |
| within 2 | F0 C=1; F1 C=1; F2 C=0.1,l1_ratio=0.8; F3 C=0.1; F4 k=3; F5 C=1; F6 max_depth=3; F3T C=0.1 | P1: F0 0.15, F2 0.33, F3 0.18, F5 0.04, F6 0.69 | P2: F0 0.12, F2 0.32, F3T 0.20, F5 0.05, F6 0.76 |

Family codes: F0 v3-rich SVM, F1 relative-spectrum SVM, F2 specparam elastic-net, F3 Riemannian logistic (F3T transductive), F4 LEAPD, F5 entropy logistic, F6 boosting. Weights are the stacker's non-negative coefficients over calibrated family probabilities; a zero weight means the family was dropped in that fold.

Bootstrap intervals resample participants within the fixed sites. They do not quantify uncertainty from having four training sites, from refitting, or from earlier exposure to these datasets in v1-v3. Paired deltas are exploratory and unadjusted for multiplicity.

## 2. Cohort, session rule and label-free QC

A1: one session per person, resolved per site from documented metadata (UCSD ses-on / ses-hc; New Mexico the ON-medication session; Iowa the earliest resting session; Tel Aviv the single recording). This makes session count constant and structurally removes the v3 recording-metadata confound that gave AUROC 1.000 at UCSD and New Mexico. A2: the first 15 accepted 4 s windows, minimum 10.

QC v4 thresholds were derived label-free on auxiliary cohorts and all sat at their ceilings (peak-to-peak 250 uV, step 60 uV, line-noise ratio 0.5); 389,702 windows informed them. Of 537 assigned people, 379 produced accepted windows and 324 met A2. The drop from v3's 440 is expected: A1 removes the backup session v3 had when a first recording failed QC.

| Site | Diagnosis | Eligible / assigned | Excluded | Age mean (SD), assigned | F / M, assigned |
|---|---|---|---|---|---|
| Iowa | PD | 128/202 | 36.6% | 68.4 (7.9) | 63/139 |
| Iowa | control | 67/110 | 39.1% | 69.2 (10.2) | 57/53 |
| Tel Aviv | PD | 51/116 | 56.0% | 66.1 (9.4) | 44/72 |
| Tel Aviv | control | 18/28 | 35.7% | 60.8 (7.3) | 15/13 |
| UC San Diego | PD | 13/15 | 13.3% | 63.3 (7.9) | 8/7 |
| UC San Diego | control | 15/16 | 6.2% | 63.5 (9.4) | 9/7 |
| New Mexico | PD | 17/25 | 32.0% | 69.7 (8.6) | 9/16 |
| New Mexico | control | 15/25 | 40.0% | 69.3 (9.4) | 9/16 |

Exclusion is differential at Tel Aviv (PD 56%, controls 36%) and Iowa (PD 37%, controls 39%): the label-free rules remove more PD recordings where PD recordings are noisier. That selection travels with every internal number. Age and sex are those of the assigned cohort; demographics of the retained subset were not separately tabulated.

### Confound audit after A1

| Site | Age/sex-only leave-one-site-out AUROC | People |
|---|---|---|
| Iowa | 0.539 | 312 |
| Tel Aviv | 0.486 | 144 |
| UC San Diego | 0.496 | 31 |
| New Mexico | 0.510 | 50 |

Session count is {1} for every supervised person, so the recording-metadata baseline carries no information by construction. Age/sex-only cross-site AUROC is 0.541 overall.

## 3. Representations, harmonisation and models

Seventeen named scalp channels, 256 Hz, 1-40 Hz zero-phase filtering, average reference over the 17 channels for the spectral/covariance families; a band-passed but un-referenced 128 Hz copy (preref128) is kept for the LPC family and for montage re-referencing. Electrode positions are never guessed.

| Family | Representation |
|---|---|
| B-1 | Relative band spectra per channel (684 inputs) |
| B-2 | specparam aperiodic exponent/offset and adjusted band powers (163) |
| B-3 | Riemannian tangent-space covariance (153) with C1 re-centering |
| B-4 | Order-12 LPC coefficients via Levinson-Durbin (408), LEAPD-style hyperplane scores |
| B-5 | Entropy and complexity measures (136) |
| B-6 | Robust amplitude distribution summaries (205) |
| v3-rich | The v3 log-PSD/log-covariance vector (1,309), the F0 baseline |
| C3 | Normative spline quantile z-scores of 38 regional features fitted on 467 healthy people aged 40+ |

### Harmonisation guardrails

| Regime | Criterion | Sites within tolerance | Max AUROC spread across prevalence | Verdict (>= 3 of 4) |
|---|---|---|---|---|
| C1 | label-aware | 4/4 | 0.0039 | pass |
| C1 | label-free | 4/4 | 0.0059 | pass |
| C2 | label-aware | 2/4 | 0.0444 | fail |
| C2 | label-free | 3/4 | 0.0355 | pass |

C1 (Riemannian re-centering) passed at all four sites under both criteria and is the only transductive method allowed into the model grid. C2 (robust per-site z) failed the label-aware criterion at UCSD and New Mexico and was excluded; its label-free companion disagreed, which is itself reported. C3 is inductive and always on.

### Model grid

F0 v3-rich SVM (the baseline); F1 relative-spectrum SVM; F2 specparam elastic-net; F3 Riemannian tangent logistic (inductive, and transductive with C1); F4 LEAPD/LPC; F5 entropy logistic; F6 gradient boosting; S a non-negative stacker over the family probabilities with Platt calibration. P1 = S inductive; P2 = S transductive; P3 = the S stack on the M4T around-ear montage, fixed by reviewed decision, not by score.

## 4. Internal results: equal-site means and gates

| Family / pipeline | Cross-site AUROC | Within-site AUROC | Composite |
|---|---|---|---|
| F0 v3 rich-feature SVM | 0.758 | 0.785 | 0.771 |
| F1 relative-spectrum SVM | 0.677 | 0.714 | 0.696 |
| F2 specparam elastic-net | 0.774 | 0.801 | 0.788 |
| F3 Riemannian tangent logistic | 0.718 | 0.748 | 0.733 |
| F3 Riemannian tangent (transductive) | 0.737 | 0.724 | 0.730 |
| F4 LEAPD / LPC | 0.590 | 0.549 | 0.570 |
| F5 entropy logistic | 0.678 | 0.690 | 0.684 |
| F6 gradient boosting | 0.783 | 0.738 | 0.760 |
| S stack, inductive (P1) | 0.765 | 0.774 | 0.770 |
| S stack, transductive (P2) | 0.773 | 0.777 | 0.775 |

Composite is 0.5 x cross-site + 0.5 x within-site. Only P1 (S|I) and P2 (S|T) are pre-declared results; every other row is descriptive. F6 boosting has the highest cross-site mean but was not pre-declared and is not promoted.

| Gate / hypothesis | Requirement | Observed | Verdict |
|---|---|---|---|
| G1 | Reproduce v3 within 0.01 | cross 0.7627 vs 0.763; within 0.7113 vs 0.711 | met |
| G2 / H2 | Best new family >= F0 + 0.02 (inner) | F2_specparam_enet|I: +0.0068 | not met |
| G3 / H3 | Transductive gain >= 0.02 and stress test | +0.007; C1 pass, C2 fail | not met |
| G4 | Report the primary composite as is | P1 cross 0.765, within 0.774 | reported |
| H1 | >= 0.75 on each component vs v3's 0.690 / 0.693 | 0.765 / 0.774 (+0.078 composite) | met, with cohort caveat |
| G6 | Best pre-declared montage within 0.05 of T1 | M5S 0.775 vs T1 0.770 | met |

The v4.1 correction (band-passing preref128 and the LPC input) changed only F4 and F5; the hypothesis that the unfiltered input explained F4's poor result was wrong (F4 moved 0.580 to 0.570); F5 improved from 0.635 to 0.684. No gate verdict changed.

## Primary cross-site: Iowa and Tel Aviv

### Iowa | 195 people

| Model | AUROC | 95% CI | Sens. | Spec. | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| F0 v3 rich-feature SVM | 0.713 | 0.637-0.784 | 0.602 | 0.716 | 0.836 | 0.224 |
| F1 relative-spectrum SVM | 0.709 | 0.632-0.782 | 0.680 | 0.657 | 0.821 | 0.214 |
| F2 specparam elastic-net | 0.699 | 0.618-0.776 | 0.547 | 0.776 | 0.792 | 0.236 |
| F3 Riemannian tangent logistic | 0.624 | 0.538-0.705 | 0.242 | 0.836 | 0.756 | 0.409 |
| F3 Riemannian tangent (transductive) | 0.657 | 0.575-0.737 | 0.438 | 0.687 | 0.785 | 0.264 |
| F4 LEAPD / LPC | 0.571 | 0.486-0.655 | 0.508 | 0.567 | 0.721 | 0.295 |
| F5 entropy logistic | 0.692 | 0.616-0.767 | 0.594 | 0.672 | 0.810 | 0.248 |
| F6 gradient boosting | 0.727 | 0.652-0.796 | 0.617 | 0.672 | 0.844 | 0.215 |
| S stack, inductive (P1) | 0.749 | 0.677-0.818 | 0.602 | 0.791 | 0.837 | 0.259 |
| S stack, transductive (P2) | 0.749 | 0.680-0.818 | 0.547 | 0.821 | 0.836 | 0.246 |

### Tel Aviv | 69 people

| Model | AUROC | 95% CI | Sens. | Spec. | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| F0 v3 rich-feature SVM | 0.691 | 0.542-0.820 | 1.000 | 0.167 | 0.866 | 0.179 |
| F1 relative-spectrum SVM | 0.550 | 0.405-0.695 | 0.569 | 0.389 | 0.810 | 0.233 |
| F2 specparam elastic-net | 0.650 | 0.480-0.802 | 0.922 | 0.222 | 0.783 | 0.191 |
| F3 Riemannian tangent logistic | 0.577 | 0.413-0.743 | 0.706 | 0.278 | 0.783 | 0.263 |
| F3 Riemannian tangent (transductive) | 0.590 | 0.423-0.752 | 0.529 | 0.611 | 0.786 | 0.328 |
| F4 LEAPD / LPC | 0.601 | 0.445-0.751 | 0.922 | 0.167 | 0.808 | 0.210 |
| F5 entropy logistic | 0.534 | 0.357-0.708 | 0.922 | 0.056 | 0.741 | 0.233 |
| F6 gradient boosting | 0.709 | 0.564-0.836 | 0.882 | 0.222 | 0.877 | 0.173 |
| S stack, inductive (P1) | 0.655 | 0.497-0.801 | 0.961 | 0.056 | 0.816 | 0.210 |
| S stack, transductive (P2) | 0.650 | 0.490-0.794 | 0.941 | 0.167 | 0.819 | 0.206 |

Sensitivity and specificity use the per-fold inner threshold. PR-AUC depends on PD prevalence. High AUROC alone does not imply a usable operating threshold; the external sites show why.

## Primary cross-site: UCSD and New Mexico

### UC San Diego | 28 people

| Model | AUROC | 95% CI | Sens. | Spec. | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| F0 v3 rich-feature SVM | 0.749 | 0.549-0.923 | 0.692 | 0.667 | 0.745 | 0.208 |
| F1 relative-spectrum SVM | 0.682 | 0.472-0.862 | 0.769 | 0.467 | 0.695 | 0.243 |
| F2 specparam elastic-net | 0.856 | 0.687-0.985 | 0.692 | 0.933 | 0.858 | 0.179 |
| F3 Riemannian tangent logistic | 0.851 | 0.687-0.974 | 0.846 | 0.733 | 0.867 | 0.164 |
| F3 Riemannian tangent (transductive) | 0.846 | 0.682-0.969 | 0.846 | 0.800 | 0.861 | 0.174 |
| F4 LEAPD / LPC | 0.585 | 0.359-0.795 | 0.231 | 0.733 | 0.524 | 0.269 |
| F5 entropy logistic | 0.677 | 0.462-0.862 | 0.308 | 0.867 | 0.663 | 0.240 |
| F6 gradient boosting | 0.790 | 0.605-0.938 | 0.846 | 0.733 | 0.772 | 0.221 |
| S stack, inductive (P1) | 0.779 | 0.590-0.938 | 0.615 | 0.800 | 0.770 | 0.196 |
| S stack, transductive (P2) | 0.805 | 0.621-0.954 | 0.462 | 0.867 | 0.800 | 0.189 |

### New Mexico | 32 people

| Model | AUROC | 95% CI | Sens. | Spec. | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| F0 v3 rich-feature SVM | 0.878 | 0.745-0.973 | 0.882 | 0.667 | 0.906 | 0.165 |
| F1 relative-spectrum SVM | 0.769 | 0.584-0.922 | 0.765 | 0.667 | 0.744 | 0.205 |
| F2 specparam elastic-net | 0.890 | 0.757-0.980 | 0.824 | 0.667 | 0.920 | 0.158 |
| F3 Riemannian tangent logistic | 0.820 | 0.655-0.953 | 0.647 | 0.800 | 0.825 | 0.183 |
| F3 Riemannian tangent (transductive) | 0.855 | 0.706-0.973 | 0.882 | 0.667 | 0.858 | 0.154 |
| F4 LEAPD / LPC | 0.604 | 0.404-0.792 | 0.588 | 0.667 | 0.632 | 0.251 |
| F5 entropy logistic | 0.808 | 0.635-0.953 | 0.412 | 0.933 | 0.842 | 0.168 |
| F6 gradient boosting | 0.906 | 0.780-0.996 | 0.824 | 0.933 | 0.940 | 0.157 |
| S stack, inductive (P1) | 0.878 | 0.737-0.988 | 0.882 | 0.467 | 0.887 | 0.162 |
| S stack, transductive (P2) | 0.886 | 0.757-0.988 | 0.882 | 0.467 | 0.905 | 0.155 |

Sensitivity and specificity use the per-fold inner threshold. PR-AUC depends on PD prevalence. High AUROC alone does not imply a usable operating threshold; the external sites show why.

## Separate within-site: Iowa and Tel Aviv

### Iowa | 195 people

| Model | AUROC | 95% CI | Sens. | Spec. | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| F0 v3 rich-feature SVM | 0.751 | 0.680-0.823 | 0.609 | 0.612 | 0.853 | 0.188 |
| F1 relative-spectrum SVM | 0.732 | 0.661-0.801 | 0.711 | 0.567 | 0.858 | 0.192 |
| F2 specparam elastic-net | 0.729 | 0.657-0.800 | 0.602 | 0.761 | 0.849 | 0.213 |
| F3 Riemannian tangent logistic | 0.653 | 0.569-0.740 | 0.617 | 0.612 | 0.758 | 0.260 |
| F3 Riemannian tangent (transductive) | 0.632 | 0.546-0.717 | 0.477 | 0.716 | 0.746 | 0.296 |
| F4 LEAPD / LPC | 0.624 | 0.537-0.706 | 0.820 | 0.254 | 0.765 | 0.250 |
| F5 entropy logistic | 0.679 | 0.598-0.760 | 0.578 | 0.642 | 0.804 | 0.246 |
| F6 gradient boosting | 0.707 | 0.634-0.777 | 0.516 | 0.806 | 0.849 | 0.202 |
| S stack, inductive (P1) | 0.749 | 0.680-0.816 | 0.680 | 0.687 | 0.861 | 0.198 |
| S stack, transductive (P2) | 0.749 | 0.681-0.816 | 0.578 | 0.716 | 0.859 | 0.205 |

### Tel Aviv | 69 people

| Model | AUROC | 95% CI | Sens. | Spec. | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| F0 v3 rich-feature SVM | 0.729 | 0.588-0.854 | 0.902 | 0.333 | 0.885 | 0.168 |
| F1 relative-spectrum SVM | 0.752 | 0.615-0.874 | 0.804 | 0.444 | 0.893 | 0.164 |
| F2 specparam elastic-net | 0.768 | 0.640-0.873 | 0.882 | 0.333 | 0.912 | 0.170 |
| F3 Riemannian tangent logistic | 0.688 | 0.533-0.829 | 0.824 | 0.444 | 0.858 | 0.213 |
| F3 Riemannian tangent (transductive) | 0.682 | 0.532-0.817 | 0.471 | 0.667 | 0.866 | 0.286 |
| F4 LEAPD / LPC | 0.475 | 0.324-0.626 | 0.725 | 0.167 | 0.720 | 0.313 |
| F5 entropy logistic | 0.682 | 0.556-0.813 | 0.784 | 0.444 | 0.878 | 0.217 |
| F6 gradient boosting | 0.658 | 0.514-0.799 | 0.843 | 0.333 | 0.848 | 0.184 |
| S stack, inductive (P1) | 0.734 | 0.596-0.852 | 0.863 | 0.333 | 0.888 | 0.171 |
| S stack, transductive (P2) | 0.742 | 0.608-0.860 | 0.824 | 0.389 | 0.894 | 0.174 |

Sensitivity and specificity use the per-fold inner threshold. PR-AUC depends on PD prevalence. High AUROC alone does not imply a usable operating threshold; the external sites show why.

## Separate within-site: UCSD and New Mexico

### UC San Diego | 28 people

| Model | AUROC | 95% CI | Sens. | Spec. | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| F0 v3 rich-feature SVM | 0.815 | 0.636-0.949 | 0.615 | 0.733 | 0.819 | 0.188 |
| F1 relative-spectrum SVM | 0.595 | 0.369-0.800 | 0.538 | 0.667 | 0.597 | 0.270 |
| F2 specparam elastic-net | 0.810 | 0.610-0.969 | 0.615 | 0.867 | 0.818 | 0.192 |
| F3 Riemannian tangent logistic | 0.826 | 0.651-0.959 | 0.769 | 0.667 | 0.846 | 0.176 |
| F3 Riemannian tangent (transductive) | 0.805 | 0.621-0.944 | 0.615 | 0.867 | 0.814 | 0.215 |
| F4 LEAPD / LPC | 0.564 | 0.338-0.795 | 0.769 | 0.267 | 0.546 | 0.279 |
| F5 entropy logistic | 0.600 | 0.379-0.810 | 0.385 | 0.800 | 0.590 | 0.291 |
| F6 gradient boosting | 0.759 | 0.544-0.933 | 0.692 | 0.867 | 0.760 | 0.211 |
| S stack, inductive (P1) | 0.749 | 0.538-0.933 | 0.692 | 0.733 | 0.690 | 0.199 |
| S stack, transductive (P2) | 0.749 | 0.538-0.928 | 0.692 | 0.800 | 0.701 | 0.195 |

### New Mexico | 32 people

| Model | AUROC | 95% CI | Sens. | Spec. | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| F0 v3 rich-feature SVM | 0.843 | 0.682-0.965 | 0.706 | 0.867 | 0.895 | 0.171 |
| F1 relative-spectrum SVM | 0.776 | 0.608-0.925 | 0.824 | 0.533 | 0.827 | 0.207 |
| F2 specparam elastic-net | 0.898 | 0.780-0.980 | 0.706 | 0.800 | 0.917 | 0.158 |
| F3 Riemannian tangent logistic | 0.824 | 0.655-0.961 | 0.706 | 0.800 | 0.827 | 0.184 |
| F3 Riemannian tangent (transductive) | 0.776 | 0.588-0.941 | 0.588 | 0.800 | 0.772 | 0.202 |
| F4 LEAPD / LPC | 0.533 | 0.325-0.741 | 0.706 | 0.200 | 0.622 | 0.274 |
| F5 entropy logistic | 0.800 | 0.631-0.941 | 0.706 | 0.800 | 0.842 | 0.178 |
| F6 gradient boosting | 0.827 | 0.674-0.945 | 0.588 | 0.867 | 0.877 | 0.191 |
| S stack, inductive (P1) | 0.863 | 0.714-0.973 | 0.765 | 0.733 | 0.905 | 0.167 |
| S stack, transductive (P2) | 0.867 | 0.718-0.976 | 0.706 | 0.867 | 0.911 | 0.163 |

Sensitivity and specificity use the per-fold inner threshold. PR-AUC depends on PD prevalence. High AUROC alone does not imply a usable operating threshold; the external sites show why.

## 5. Negative controls and site identity

Permutation null (labels permuted within site, 20 draws): mean composite 0.504, SD 0.051, max 0.580. P1's 0.770 sits far above it: the internal signal is real, whatever it is.

| Representation | Site-identity probe AUROC (leave-one-site-out, which site?) |
|---|---|
| v3_rich | 0.948 |
| b1_b6 | 0.922 |
| b2_c3 | 0.834 |
| b3_tangent | 0.964 |
| b4 | 0.945 |
| b5 | 0.813 |

Every representation carries the recording site almost perfectly. Harmonisation reduced but did not remove it (B-2 with C3 is lowest at 0.83). This is the mechanism by which an internally strong pipeline can fail on a new site, and it was documented before Turku was opened.

## 6. T2 reduced-channel montages

| Montage | Channels | Cross-site | Within-site | Composite |
|---|---|---|---|---|
| M4F | Fp1, Fp2, F7, F8 | 0.561 | 0.574 | 0.567 |
| M4T | T7, T8, P7, P8 | 0.769 | 0.776 | 0.772 |
| M5S | F8, F4, T7, O1, O2 | 0.777 | 0.772 | 0.775 |
| M8 | Fp1, Fp2, F7, F8, T7, T8, O1, O2 | 0.761 | 0.767 | 0.764 |

Montages are re-referenced within their own channels only; no 17-channel information enters. G6 passed: the best pre-declared montage (M5S, 0.775) is within 0.05 of T1. P3 was fixed to M4T (T7, T8, P7, P8) as the around-ear form factor of interest by reviewed decision, explicitly not on score. Distilling the 17-channel stack into M8 did not help (delta -0.042).

These are montage subsets of full research caps, not around-ear device recordings. No public around-ear PD dataset exists (confirmed by the 29 September sweep), so no around-ear device claim can be made from any number here.

## 7. Track E: frozen foundation-model embeddings

Licences were checked before any weights were downloaded: CBraMod (BSD-3-Clause) was used; LaBraM was skipped because its temporal embedding does not match the protocol's 4 s window; REVE was skipped under the unclear-licence rule. Frozen CBraMod embeddings are 27,200-dimensional for 324 people.

| FM gate condition | Observed | Verdict |
|---|---|---|
| Beats negative controls | probe 0.641 vs random-init 0.696, permuted 0.434 | fail |
| Adds >= 0.01 to the stack | 0.770 to 0.697 (-0.073) | fail |
| Site probe no worse than v3-rich | 0.969 vs 0.948 | fail |

The pooled embedding is 27,200-dimensional for 324 people, so the linear probe is in a severe p>>n regime even at C=0.1. A weak probe result cannot be cleanly separated from that.

## 8. T3: fusion-ready output, cognition and reliability

The T3 output is a prior-free log-likelihood ratio, logit(p) - logit(training prevalence 0.645), written to the Myna schema (docs/myna_eeg_output_v1.schema.json) with a guarded naive-Bayes combiner. Adding a near-duplicate modality inflated the combined evidence from 1.18 to 2.31, almost as much as a genuinely independent one (2.07): that gap is the over-counting the guard exists for.

| Target (PD only, exploratory) | n | Grouped-CV Spearman rho (95% CI) | MAE |
|---|---|---|---|
| MoCA, 17 channels | 136 | 0.061 (-0.11 to 0.23) | 4.75 |
| MoCA, M8 | 136 | 0.253 (0.09 to 0.41) | 3.69 |
| UPDRS-III, 17 channels | 135 | 0.307 (0.14 to 0.45) | 9.49 |

Test-retest: 18 Iowa people with a second session, scored by a model trained on the other three sites. ICC(2,1) of the LLR is 0.44 (Spearman 0.48) against a pre-declared 0.6 target: not met. An output that does not agree with itself across sessions cannot track symptoms.

## 9. External sites: Turku, the eyes-closed control, BrainLat

### Turku (OSF pehj9): raw, eyes-open, 66-channel clinical cap

Forty recordings; labels sealed at download and resolved from medication fields, never guessed. Label-free QC excluded 16 before any label was opened (13 with no accepted window, 3 below A2). Two bugs were caught before the shot was spent (transductive re-centering on a new site; P3 fitting on 17-channel data while labelled M4T) and one after: zero-padded file names versus unpadded spreadsheet IDs silently dropped every single-digit participant, 4 of 24. The corrected re-run is reported as non-confirmatory.

| Pipeline | Confirmatory n=20 | Corrected n=24 (non-confirmatory) | Sens / spec at locked threshold (n=24) |
|---|---|---|---|
| P1 (primary) | 0.570 (0.280-0.810) | 0.545 (0.301-0.776) | 0.27 / 0.77 |
| P2 | 0.600 (0.330-0.850) | 0.552 (0.301-0.769) | 0.18 / 0.85 |
| P3 | 0.600 (0.330-0.840) | 0.587 (0.336-0.804) | 0.27 / 0.85 |

H5 required P1's lower bound above 0.60; observed 0.280 (n=20) and 0.301 (n=24). Both intervals span chance. Age (PD 68.9, control 68.0) and sex are balanced. This is the most important result in v4 and it contradicts every internal number.

### Turku eyes-closed condition-shift control (protocol 14.2, non-confirmatory)

| Pipeline | Overlap n | Eyes-open AUROC | Eyes-closed AUROC | Delta EO - EC |
|---|---|---|---|---|
| P1 | 22 | 0.463 | 0.488 | -0.025 |
| P2 | 22 | 0.471 | 0.504 | -0.033 |
| P3 | 18 | 0.500 | 0.300 | +0.200 |

Pre-registered rule on P1: |delta| <= 0.05 means a BrainLat result is interpretable as site generalisation. Observed delta -0.025: verdict 'site_generalisation_interpretable'. Honest reading: both conditions sit at chance on Turku, so the control establishes that condition is not the reason, nothing more. P3 is the exception (delta +0.20 on n=18): the around-ear montage is condition-sensitive.

### BrainLat (Synapse): ICA-cleaned, eyes-closed, 128-electrode Biosemi, Chile

A label-free inspection before the seal found that files named *_reject.set are a different pipeline (epoched, no ICA, no low-pass) present only under PD (PD/AR 7/7, PD/CL 6/22, HC 0/46), that no raw data exists, and that the cap is Biosemi A1..D32. Protocol 14.4 fixed the cohort to plain-named Chile files with 17 channels spline-interpolated from each file's own electrode coordinates, before any label was sealed. Fourteen of 27 Chile controls turned out to be header-only on Synapse; 16 PD and 13 HC were scored.

| Pipeline | AUROC (95% CI) | PR-AUC | Brier | Sens / spec at locked threshold |
|---|---|---|---|---|
| P1 (primary) | 0.697 (0.500-0.880) | 0.809 | 0.220 | 0.31 / 1.00 |
| P2 | 0.736 (0.519-0.899) | 0.833 | 0.207 | 0.25 / 1.00 |
| P3 | 0.726 (0.519-0.899) | 0.777 | 0.224 | 0.25 / 1.00 |

H5b fails as written (P1 lower bound 0.500). But ranking is at 0.70-0.74 with P2/P3 intervals excluding chance, while specificity is 1.00 and sensitivity 0.25-0.31: BrainLat's cleaned signals score 'healthier' across the board and the locked threshold misses most PD. Discrimination survived a shift that the threshold did not; P2's harmonisation did no better on it than P1. With 13 controls no interval could have cleared 0.60 unless AUROC were about 0.85.

Caveats: eyes-closed against eyes-open training; ICA-cleaned and low-passed against raw; the 13 excluded PD were the artifact-heavier recordings, so the PD sample is selected toward cleaner EEG; interpolated rather than physical electrodes; n=29.

### Two-site summary and the third site

On the external site with raw eyes-open data the pipeline is at chance; on the one with pre-cleaned eyes-closed data it ranks at about 0.70 with a threshold that no longer transfers. Neither meets its gate. A repository sweep on 29 September found no further downloadable PD-versus-control resting EEG site and no public around-ear PD data; the one request-only candidate (Copenhagen, 40 PD OFF-medication / 26 HC, raw, eyes-open, 128 channels) is pre-registered as protocol 14.5 and would test, not rescue, the raw-versus-cleaned hypothesis.

## 10. Bugs, corrections and what each could have changed

| Found | Defect | Effect and handling |
|---|---|---|
| Stage 7 | preref128 and the LPC input were built from the unfiltered signal | Only F4/F5 changed; no gate changed; protocol v4.1 dated correction |
| Stage 11, before the shot | Transductive P2 looked up a new site in the training re-centering map | Would have crashed; extend_transductive re-centres a locked site on itself; tested |
| Stage 11, before the shot | P3 was fitting 17-channel features while labelled M4T | Caught by a dimension mismatch; final_fit now takes a montage; tested |
| Stage 11, after the shot | Zero-padded file IDs versus unpadded spreadsheet IDs | 4 of 24 Turku people silently dropped; corrected re-run reported as non-confirmatory |
| v4.2 | OSF pagination duplicated one eyes-closed file six times and hid five | Listing de-duplicated by file id; 39 files, not 44; protocol text corrected |
| 14.4 | Standard-montage sphere centre 16 mm behind the Biosemi origin | Targets were tilted about 10 degrees; positions aligned by fitted sphere centres; synthetic-field test |
| 14.4 | Biosemi C3/C4 labels collide with 10-20 names | Sources prefixed before interpolation; tested |
| 14.4 | PD headers reference pre-rename .fdt names; 14 HC header-only; macOS ._ shadows | Resolved in scratch directories, originals untouched; header-only recorded, not scored |

## 11. Verification, compute and the decision

| Verified item | Evidence |
|---|---|
| Baseline reproduction | v3 numbers reproduced two ways within 0.001 before any v4 fit |
| Archive | 41,746 planned files reconciled against receipts with zero mismatches; Turku and BrainLat hashes verified against source-reported values |
| Leakage | Tests poison held-out features with NaN and assert fitted transforms are unchanged; participant/site disjointness asserted per fold |
| Locks | Outer lock precedes outer predictions; final lock precedes each one-shot; sealed-label hashes checked before every evaluation |
| Compute | Supervised campaign forecast 0.13 h against a 96 h gate (54 s per fold); local Mac only |
| Tests | 176 passing; ruff clean; no raw EEG, identities, predictions, weights or local paths in Git |

Python 3.12.14; numpy 2.5.3; scipy 1.18.1; sklearn 1.9.1; mne 1.13.2; pyriemann 0.12; fooof 1.1.1; antropy 0.2.2; lightgbm 4.7.0; torch 2.14.0; braindecode 1.8.1. Apple M4, 16 GB unified memory; CPU/MPS only; no cloud.

### Model cards (docs/pd-v4-model-cards.json)

| Card | Pipeline | Composite | Turku n=20 / n=24 | BrainLat | Threshold |
|---|---|---|---|---|---|
| P1 | S|I | 0.770 | 0.570 / 0.545 | 0.697 | 0.547 |
| P2 | S|T | 0.775 | 0.600 / 0.552 | 0.736 | 0.597 |
| P3 | M4T montage stack (T7, T8, P7, P8), inductiv | 0.772 | 0.600 / 0.587 | 0.726 | 0.558 |

### Decision supported by this experiment

Do not build a product on this classifier. The internal gains over v3 are real but the pipeline does not generalise to the one raw external site it was tested on, its output is not test-retest reliable, and none of the pre-registered secondary hypotheses held. For Myna the honest deliverable is the fusion-ready LLR schema with a guarded combiner, plus the negative evidence.

What v5 should test, in this order: whether the Turku failure is a raw-versus-cleaned data effect (Copenhagen, pre-registered 14.5; or apply an identical cleaning pipeline to every site before features); site-invariant training with the site-identity probe as an explicit objective; and a reliability gate (ICC >= 0.6) before any external shot.

Repository artefacts: frozen protocol with dated amendments, stage log, results JSON, model cards for P1/P2/P3, this report. Raw EEG, labels, predictions, checkpoints and weights stay local. No clinical deployment or publication claim is made.

## 12. References and data provenance

### Riemannian re-centering for cross-site EEG

Zanini et al. (2018). https://doi.org/10.1109/TBME.2017.2742541

### Periodic / aperiodic spectral parameterization

Donoghue et al. (2020). https://doi.org/10.1038/s41593-020-00744-x

### LEAPD: linear predictive coding for PD EEG

Anjum et al. (2020). https://doi.org/10.1016/j.parkreldis.2020.08.001

### CBraMod

Wang et al. (ICLR 2025). https://arxiv.org/abs/2412.07236

### Turku eyes-open / eyes-closed PD EEG

Railo et al., OSF pehj9. https://osf.io/pehj9/ (no licence shown; research use only, contact the authors before other use)

### BrainLat

Prado et al. (2023) Sci Data. Synapse syn51549340, CC-BY-4.0

### Supervised data

https://openneuro.org/datasets/ds008768

https://openneuro.org/datasets/ds007526

https://openneuro.org/datasets/ds003490

https://openneuro.org/datasets/ds002778

Companion documents: PD_PROTOCOL_V4.md; PD_V4_LOG.md; pd-v4-results.json; pd-v4-model-cards.json; PD_V3_RESULTS.md.
