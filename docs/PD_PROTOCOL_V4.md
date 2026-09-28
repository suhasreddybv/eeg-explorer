# PD classifier v4: Myna-ready proof of concept

Approved scope: raise both cross-site and within-site AUROC, weighted equally, over the v3 pre-declared pipeline. Deliver three tiers: T1 (17-channel montage), T2 (reduced-channel wearable montages), T3 (a fusion-ready EEG output for Myna). Evaluate the final pre-named pipelines once on a new, locked external site (Turku, Finland). Local Mac only (Apple M4, 16 GB, PyTorch MPS or CPU); no cloud, no CUDA. Every continuous training stage is benchmarked and gated at 96 hours before launch. This document, and every `pd_pipeline/*_v4.py` module, is hashed into an implementation policy (`scripts/run_pd_v4.py`) before any supervised model is fitted. v1/v2/v3 code, docs, run directories and PDFs are read-only; v4 imports from v3, never edits it.

## 1. Questions and hypotheses

- **H1 (primary, composite).** The pre-declared v4 T1 pipeline improves on v3's pre-declared pipeline (0.690 cross-site / 0.693 within-site; reproduced as gate G1, `data/pd-runs/v4/baseline-repro.json`).
  - Primary endpoint: **composite = 0.5 × cross-site equal-site AUROC + 0.5 × within-site equal-site AUROC**.
  - Report both components separately. Target (not a promise): at least 0.75 on each.
- **H2.** Aperiodic/relative/LPC feature families (B-1, B-2, B-4) add at least 0.02 inner-CV composite over the rich-SVM.
- **H3.** Label-free harmonisation (C1/C2) adds at least 0.02 cross-site in the transductive regime, and passes the prevalence stress test.
- **H4.** At least one predeclared montage with ≤8 channels is within 0.05 composite of T1.
- **H5.** The locked Turku AUROC of the primary T1 pipeline has a 95% CI lower bound above 0.60. Reported either way.
- **H6 (exploratory).** Within PD, EEG predicts MoCA (Spearman ρ) and UPDRS-III, and the EEG score has test-retest ICC ≥ 0.6 on repeat Iowa sessions.

## 2. Sites and archive accounting

Same 4 supervised sites as v3 (`cohort_v3.SITES`: ds008768 Iowa, ds007526 Tel Aviv, ds003490 UNM, ds002778 UCSD), the same auxiliary set (`cohort_v3.AUX`), the same `cohort_v3.EXCLUDED`. Turku (OSF `pehj9`) is added as `locked_test` — see §8.

**ds008875** (Visser/Narayanan/Uc, "Resting-state EEG in Parkinson's disease and controls", Iowa group, 156 PD / 78 control / 18 DBS, ~10 GB) is **excluded** from v4: likely participant overlap with ds008768/ds004584 (same Iowa group). Not downloaded into the archive; confirmed via the OpenNeuro API only (`data/pd-runs/v4/overlap-audit.json`). May be considered for v5 *training* only, after a signal-hash and metadata overlap audit against ds008768.

## 3. Cohort rules (`cohort_v4.py`)

**A1: one session per person.**

| Site | Rule |
|---|---|
| UCSD (ds002778) | PD use `ses-on`; controls `ses-hc` |
| New Mexico (ds003490) | PD use the session whose `sessN_Med` is ON; controls their single session |
| Iowa (ds008768) | Earliest resting session by `days since first session` from `_sessions.tsv` |
| Tel Aviv (ds007526) | The single resting recording |

Medication state recorded as reported (Iowa/Tel Aviv: `unknown/assumed-usual-medication`). OFF-only is a pre-declared sensitivity analysis (UCSD/UNM PD only). New Mexico stays restricted to `cohort_v2.eye_interval()`.

**A2: fixed duration.** Primary: the first 15 QC-accepted 4-s windows in time order (60 s). People with fewer than 10 accepted windows are excluded (unchanged from v3). Sensitivity analysis: all accepted windows.

**Confound audit** (label-aware, model-free, run before fitting): per site × diagnosis, tabulate session count (now must be 1 for everyone), accepted-window count, age, sex. Re-run the v3 metadata-only baseline under v4 rules. Expected result: metadata AUROC is no longer 1.0 at UCSD/UNM (v3's session-count confound, §7 of `PD_V3_RESULTS.md`). If it still is, stop and ask.

## 4. QC v4 (`qc_v4.py`, A3)

Processing order per recording: (1) pick the 17 channels; (2) resample to 256 Hz; (3) 1–40 Hz 4th-order zero-phase Butterworth filter; (4) average-reference over the 17 channels; (5) cut 4-s windows with 2-s margins (as v3).

| Check | Computed on | Rule |
|---|---|---|
| Amplitude | filtered window | peak-to-peak per channel ≤ `PTP_UV` |
| Step | filtered window | max \|diff\| per channel ≤ `STEP_UV` |
| Flat / repeat | raw resampled window | std ≥ 0.2 µV; repeated-sample fraction ≤ 0.3 (unchanged from v3) |
| Line noise | raw window, native rate | 45–65 Hz power / 1–40 Hz power ≤ `LINE_RATIO` |
| Source bad annotations | – | unchanged |

**Threshold derivation (label-free):** compute each QC statistic's distribution on the auxiliary cohorts only, including ds003775 (which v3's fixed thresholds rejected entirely — §3 of `PD_V3_RESULTS.md`). Set `PTP_UV`, `STEP_UV`, `LINE_RATIO` at the 99th percentile of auxiliary clean-rest windows, rounded, ceilings `PTP_UV ≤ 250`, `STEP_UV ≤ 60`, `LINE_RATIO ≤ 0.5`. Do not tune QC on PD/control separation, ever.

> **Numeric values (Stage 4 addendum, frozen 28 September 2026):** derived from 389,702 clean windows across 10,590 auxiliary recordings, before any supervised-site or Turku recording was processed:
> `PTP_UV = 250`, `STEP_UV = 60`, `LINE_RATIO = 0.5` — all three at the protocol ceiling (the raw 99th-percentile values exceeded the ceiling). See `data/pd-runs/v4/qc-thresholds.json`.

**Report:** exclusion rate by site × diagnosis under v3 QC vs v4 QC (label-aware, descriptive, produced after the freeze); ds003775 acceptance under v4 QC.

## 5. Representations (`features_v4.py`)

Per-window: `psd` (Welch, 1-s segments, 50% overlap, 1–40 Hz, 1 Hz resolution, 17×40; 2–35 Hz subset retained for v3-compatible families), `cov` (17×17 broadband covariance, fixed 10% isotropic shrinkage toward the identity — matching v3's convention (`PD_PROTOCOL_V3.md`), not data-driven OAS shrinkage; implemented in `qc_v4.spectral_features`, corrected here to match Stage 4's actual implementation before any supervised fit), `fbcov` (θ 4–8 / α 8–13 / β 13–30 Hz band covariances, same fixed shrinkage), `lpc` (order-12 LPC per channel at 128 Hz, Levinson-Durbin autocorrelation method, plus prediction-error variance), `ent` (sample entropy m=2 r=0.2·SD, permutation entropy order 5 delay 1, Higuchi FD kmax 10, Lempel-Ziv complexity), `raw200` (Track E only, 200 Hz float16 17×800), `preref128` (band-passed 1-40 Hz at 128 Hz, before average referencing, for Stage 9 montage re-referencing).

> **Correction, protocol v4.1 (28 September 2026).** The Stage 4 implementation originally built `preref128` and the per-window `lpc` array from the *unfiltered* resampled signal, contradicting the filtered-window rule above. The deviation was material (up to 40.7% of a supervised recording's `preref128` power sat below 1 Hz) and plausibly explains F4/LEAPD's near-chance cross-site AUROC in the superseded run. Both now derive from a band-passed, deliberately un-referenced signal. The fix was determined solely by this document's text, not by any outcome; it was found while preparing Stage 9, after Stage 7's outer results had been seen, so the superseded run is preserved under `data/pd-runs/v4/superseded-v4.0/` and Stages 4-7 were re-run. Everything reported from the re-run is labelled protocol v4.1.

**QC thresholds (frozen in Stage 4, label-free, from 389,702 clean windows across 10,590 auxiliary recordings):** `PTP_UV = 250`, `STEP_UV = 60`, `LINE_RATIO = 0.5` (all three at the protocol's ceiling — the raw 99th-percentile values were higher). See `data/pd-runs/v4/qc-thresholds.json` and `qc-audit.json`.

Per-person (15-window primary):

- **B-1 relative spectra:** log(PSD / Σ PSD, 2–35 Hz) per channel, mean over windows; alpha/theta ratio; individual alpha peak frequency (max 7–13 Hz of smoothed mean spectrum); spectral centroid 2–35 Hz; 90% spectral edge.
- **B-2 specparam/fooof** per channel's mean PSD, fit range 2–40 Hz, `aperiodic_mode='fixed'`, `peak_width_limits=[1,8]`, `max_n_peaks=6`, `min_peak_height=0.05`. Exponent, offset, R² (R²<0.8 → missing, imputed within training fold); aperiodic-adjusted power (θ, pre-alpha 5.5–8, α, β); α peak CF/PW. Regional averages: frontal (Fp1 Fp2 F7 F3 Fz F4 F8), central (C3 C4), temporal (T7 T8 P7 P8), parietal (P3 P4), occipital (O1 O2).
- **B-3 Riemannian:** per-person Riemannian mean of `cov`/`fbcov` over windows; tangent-space map at the training-fold reference (inductive) or after C1 re-centering (transductive).
- **B-4 LPC/LEAPD:** mean/SD of LPC coefficients per channel; LEAPD score via per-channel PCA hyperplanes (k=3) fit on PD/control training people, all inside training folds.
- **B-5 entropy:** mean/SD of `ent` per channel.
- **B-6 robust distributions:** 10th/50th/90th percentiles over windows of B-1 band values and specparam-free band powers; burst fraction (share of windows where θ or pre-alpha relative power exceeds the person's median × 1.5).
- **B-7 connectivity (secondary):** wPLI and orthogonalised AEC in θ/α/β, averaged into 5-region × 15-pair blocks (90 features).
- **v3-rich:** the exact v3 1,309-feature vector, recomputed on v4 windows, as the baseline to beat (gate G1 already reproduces this on v3 windows).

Every per-person transform that fits parameters (scalers, PCA, tangent reference, imputation, hyperplanes, normative model) is fit on training people only, inside each fold. Fixed within-person transforms may be cached globally. Leakage tests (Stage 5) assert this with sentinel NaNs for held-out people.

## 6. Harmonisation (`harmonize_v4.py`)

- **Inductive (I):** no test-site data used; training-fold scalers only.
- **Transductive (T):** test site's unlabelled people may set site statistics.
  - **C1** Riemannian re-centering: per-site whitening by `pyriemann.utils.mean.mean_riemann`, tangent map at identity; optional dispersion rescale (Mellot 2023).
  - **C2** robust per-site standardisation: `(x − site median) / site IQR`.
- **C3 normative z-scores** (both regimes): 38 regional features (B-1 band values, B-2 exponent/offset/adjusted powers, alpha peak frequency). Spline quantile regression (5th/50th/95th percentile) on age (natural spline, 4 df) + sex, fit on healthy adults ≥40y from ds005385, ds004148, ds004796, ds004504 controls, through v4 QC/windows. `z = (x − q50) / ((q95 − q5) / 3.29)`. Normative-source sites harmonised with C2, anchored on training-site controls.
- **Prevalence stress test (guardrail):** per outer test site and transductive method, recompute site statistics on stratified resamples (PD share 0.3/0.5/0.7, 200 resamples each, without replacement), re-score, compute AUROC on the full site. **Pass if max − min of mean AUROC ≤ 0.03.** A failing method is reported but excluded from the pre-declared pipeline.

## 7. Model families and frozen grid (`models_v4.py`, `compare_v4.py`)

| ID | Family | Input | Grid |
|---|---|---|---|
| F0 | v3 rich-SVM (baseline) | v3-rich | C∈{1,10}; γ=scale |
| F1 | Relative-spectra SVM | B-1+B-6 | C∈{1,10} |
| F2 | specparam logistic (elastic net) | B-2 (+C3 z) | C∈{0.1,1}; l1_ratio∈{0.2,0.8} |
| F3 | Riemannian tangent logistic | B-3 | C∈{0.1,1} |
| F4 | LPC/LEAPD | B-4 | PCA k∈{2,3}; LR C=1 |
| F5 | Entropy logistic | B-5 | C∈{0.1,1} |
| F6 | Gradient boosting | B-1+B-2+B-4 score+C3 z (≤250 features) | HistGradientBoosting: depth∈{2,3}, lr 0.05, 300 iters, early-stop inner |
| F7 | Attention-MIL over window features | per-window B-1+LPC(+mask) | lr∈{3e-4,1e-3}; 2 heads; hidden 32; dropout 0.3; ≤60 epochs, patience 8 |
| F8 | Connectivity logistic (secondary) | B-7 | C∈{0.1,1} |
| S | **Stacked ensemble (pre-declared primary)** | inner-OOF probabilities of F0–F6 (+F7, +FM if gated) | non-negative LR, C=1, no intercept tuning |

Each of F1–F6 runs under regime I, and under T (C1 for F3, C2 otherwise) when the stress test passes.

**Selection objective (D3, inner loops only):**

`score = 0.5·AUROC_inner_cross + 0.5·AUROC_inner_within − 0.5·SD_inner_sites`

where `AUROC_inner_cross` is the equal-site mean over inner leave-one-site-out folds, `AUROC_inner_within` the equal-site mean over 3 inner participant folds stratified by site×diagnosis, `SD_inner_sites` the SD of per-site inner cross-site AUROCs.

**Stacking:** each level-0 family picks hyperparameters by D3; inner OOF predictions (one per person) train the stacker; outer predictions come from level-0 models refit on the full outer-training set, combined by the frozen stacker. No outer information used anywhere.

**Calibration (D6):** Platt scaling on inner OOF predictions. Thresholds from inner-OOF balanced-accuracy optimum (as v3).

**Seeds:** neural F7 uses 3 seeds (20261001–03), averaged. Classical fits deterministic.

**The pre-declared primary T1 pipeline is `S` under regime I. `S` under regime T is the pre-declared secondary.** Every other number is descriptive.

## 8. Locked site: Turku (OSF `pehj9`)

Source: `https://osf.io/pehj9/` (Railo et al., University of Turku). 40 raw eyes-open `.set` files (never `*_PREP_preprocessed.set`, which carries Turku's own pipeline shift), plus `InfoAboutParticipants.xlsx` and the preprocessing `.m` file for provenance. Downloaded, byte/sha256-verified against OSF, and format-verified (all 17 v3 channels present, 500 Hz, 119–179 s eyes-open rest) — see `data/pd-runs/v4/turku-format-verification.json`.

**Licence.** OSF shows no licence. Internal, non-commercial research evaluation only; no redistribution; contact Dr Henry Railo (University of Turku) before any commercial or publication use.

**Labels sealed.** The source `PD group` column is 0/1 with no documented key. Resolved from internal consistency, not guessed: `LEDD` and `PD duration` (clinical fields meaningless for a healthy control) are populated only for `PD group == 1` (20/20 with LEDD, 16/20 with duration), never for `== 0` — so `1 = PD`, `0 = control`, a clean 20/20 split matching the 40 eyes-open recordings exactly. This rule is implemented and unit-tested in `locked_site_v4.resolve_pd_group_coding` (including a synthetic flipped-coding case and a genuinely-ambiguous case that must raise rather than guess). Labels written to `locked-sites/osf-pehj9/sealed/labels.json`; `locked_site_v4.load_labels()` raises unless both `outer-models-lock.json` and `final-models-lock.json` exist.

**Overlap:** confirmed none of the 40 Turku raw-file sha256 hashes match any of the 19,888 unique hashes in the main archive's `verified-files.jsonl` — Turku was never used by v1–v3.

## 9. Reduced-channel montages (T2, `reduced_v4.py`)

| ID | Channels | Proxy |
|---|---|---|
| M4F | Fp1 Fp2 F7 F8 | forehead headband |
| M4T | T7 T8 P7 P8 | around-ear / temporal |
| M5S | F8 F4 T7 O1 O2 | spread 5-ch (Suuronen 2023) |
| M8 | Fp1 Fp2 F7 F8 T7 T8 O1 O2 | 8-ch compact cap |

**Critical:** montages are re-referenced from the stored `preref128` windows with an average reference over the montage channels only — the v3 17-channel average reference would leak information from absent channels. Features per montage: B-1, B-2, B-4, B-5, B-3 (montage covariance). Families: F1, F2, F3, F4, F5, stack `S_k`, both regimes.

**Distillation:** student logistic regression on montage features, soft targets = 0.5·hard label + 0.5·T1 stacker inner-OOF probability (training people only).

**Nested channel-budget search** (inner loops only): greedy forward selection k=2…8 maximising D3; outer evaluation reports selected sets per fold and the AUROC-vs-k curve with CIs. Channel-dropout augmentation (F7 only): mask 20–60% of channels during training.

**G6:** the best *predeclared* montage (not the searched one) must be within 0.05 composite of T1.

## 10. T3: fusion-ready outputs (`fusion_v4.py`)

Output schema `docs/myna_eeg_output_v1.schema.json`, per session: `pd_probability`, `pd_log_likelihood_ratio = logit(calibrated p) − logit(training prevalence)` (prior-free for fusion), `cognition_index` (exploratory), a 32-float `embedding` (PCA of stacked-feature space, fit on training people), `quality`, `research_use_only: true`.

`combine_llr`: naive-Bayes sum under conditional independence, **flagged unvalidated**. `StackingFusion`: logistic regression over modality LLRs for paired data. Unit tests on synthetic correlated/independent modalities must show naive-Bayes over-counts when modalities are correlated.

**Cognition index (H6):** ridge regression of MoCA on T1/M8 features within PD at Iowa/Tel Aviv; cross-site (Iowa↔Tel Aviv) and grouped 5-fold CV; Spearman ρ with bootstrap CI, MAE. UPDRS-III reported z-scored within site (scales differ, never pooled raw).

**Test-retest reliability:** for Iowa people with ≥2 sessions, score with within-site-fold T1/M8 predictions (excluding that person), compute ICC(2,1) of `pd_log_likelihood_ratio` and the cognition index.

## 11. Evaluation

**Primary outer:** 4 leave-one-site-out folds. **Co-primary outer:** 3 participant folds stratified by site×diagnosis (v3's fold-construction logic, new seed 20261001). Composite = mean of the two equal-site means.

**Metrics:** AUROC (2,000 participant-bootstrap resamples within site), PR-AUC, Brier, sensitivity/specificity with Wilson intervals at inner-OOF thresholds, confusion counts, coverage. Paired deltas vs F0 (bootstrap), exploratory/unadjusted, stated as such.

**Negative controls (A5):** (i) S refit on permuted labels within site (20 permutations, null distribution); (ii) site-identity probe (4-class logistic on each representation, macro-AUROC, lower is better); (iii) age/sex-only; (iv) recording metadata only; (v) for FM inputs, a random-init encoder of the same architecture.

**Lock order:** (1) fit all outer models → (2) `outer-models-lock.json` → (3) generate outer predictions → (4) fit final all-data models (inner selection on all ~440 people) → (5) `final-models-lock.json` → (6) run the Turku one-shot evaluation.

## 12. Gates

- **G1:** v3 rich-SVM reproduction within ±0.01 AUROC (cross-site 0.763, within-site 0.711). **Passed** — see Stage 0, `data/pd-runs/v4/baseline-repro.json` (0.7627 / 0.7113, |Δ|≈0.0003 both).
- **G2:** inner-CV composite of the B-feature stack ≥0.02 above F0. If not met, S still runs; report that features did not help.
- **G3:** transductive gain and the prevalence stress test (§6).
- **G4:** S(I) outer composite, reported as is.
- **G5:** delivery-ready checks (Stage 11): tests/ruff green, no data in Git, results/model-cards/PDF produced.
- **G6:** best predeclared montage within 0.05 composite of T1 (§9).

No outer-fold or locked-site result may change a model, threshold, feature, QC rule or selection rule after this document is frozen. Any required change is a new protocol version, and everything fit after it is labelled exploratory.

## 13. Non-negotiables (carried from the execution plan)

Local Mac only; 96-hour gate before every continuous stage; freeze-before-fit; no outer-test tuning; locked site sealed until both model locks exist; participant grouping everywhere (ds004504=ds006036, NeuroEPO arms one group, exact-duplicate check); never commit raw EEG/participant tables/predictions/identities/checkpoints/weights/receipts/local paths; v1–v3 read-only; honest reporting (pre-declared pipeline first, cross-site/within-site separate, inductive/transductive separate, coverage + negative controls + CIs always shown, negative results reported plainly).

**Stop-and-ask triggers:** SSD not mounted; `download-status.json` not `complete`; a licence or terms question on new data/weights; any leakage check fails; v3-baseline reproduction off by >0.01 AUROC; any forecast >96h; any need to deviate from this frozen protocol.

---

*Research use only. Nothing in v4 establishes a diagnostic device, progression measure or wearable performance claim.*
