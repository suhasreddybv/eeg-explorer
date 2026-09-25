# Parkinson’s EEG classifier: research and starting protocol

**Approved starting protocol · 24 September 2026 · Local Mac only.**

## Recommendation

Start with a compact EEGNet-style neural network for **diagnosed Parkinson’s disease (PD) versus study controls**, alongside a regularized spectral-feature baseline. Use resting-state EEG, one consistent eye condition, participant-level separation, and an untouched external study. Choose the simpler baseline if the neural network does not improve generalization. We do not need the entire 600 GB target archive to begin.

The first result will be a reproducible research comparison, not a diagnostic product or a symptom severity score. Later, expand to the specified neurodegenerative diagnoses, then test whether EEG adds information to Myna’s audio and movement measurements using paired participant data.

**Start on the existing M4 Mac.** The user approved local implementation. Notify them before proceeding with any single continuous stage estimated to exceed four days.

## Agreed scope and research question

- First label: control = 0, diagnosed PD = 1. Medication state is session metadata, not a separate disease.
- Later disease-positive means one of the included neurodegenerative diagnoses; it does not mean any abnormal EEG or any neurological disease.
- Primary question: can a model distinguish previously unseen people with PD from matched study controls, and retain useful discrimination on a separate acquisition site?
- Secondary questions: does performance persist across medication states, age/sex strata, and reduced channel counts? Does it exceed inexpensive feature and demographic baselines?
- Out of scope for this phase: identifying prodromal PD, predicting disease in the general population, measuring progression, medical recommendations, and real-time wearable diagnosis.

A clinical diagnosis supplies a recording’s label. It does not establish that each four-second window contains a PD-specific physiological pattern. Patient-level labels make this a weakly supervised problem.

## Research reviewed and what it changes

This is a focused starting review, not a systematic review or meta-analysis. Searches on 24 September 2026 covered PD EEG classification, subject-wise leakage, external validation, EEGNet, neural spectral features, and current hardware pricing. Publisher papers, author manuscripts/code, official dataset descriptions, and vendor documentation were preferred. Reported scores below describe the original studies and are not forecasts for our model.

### 1. External validation is the strongest starting precedent

*Generalizable electroencephalographic classification of Parkinson’s disease using deep learning*, **Informatics in Medicine Unlocked (2023)**, used a channel-wise CNN with subject-wise tuning and an external cohort. The published abstract reports patient-level accuracy of **80.4% internally and 82.8% externally**. This directly supports our participant-level and external-test design. The paper’s small cohorts and protocol differ from ours; these figures are not acceptance targets. The earlier 2022 preprint reports different results, so we use the final publication’s figures. [Published paper](https://doi.org/10.1016/j.imu.2023.101352); [author code](https://github.com/diamandis-lab/CNN_EEG_PD).

### 2. Very high within-dataset scores need careful interpretation

Shaban and Amara, *Resting-state electroencephalography based deep-learning for the detection of Parkinson’s disease*, **PLOS ONE (2022)**, proposed wavelet images and a CNN and reported results up to **99.9%**. Its evaluation defines counts over images; the reviewed methods do not establish an independent-site test. This is an architectural reference, not evidence that our unseen-person/site performance should approach 100%. We will explicitly publish split units and participant counts rather than compare incomparable headline accuracies. [Paper](https://doi.org/10.1371/journal.pone.0263159).

### 3. EEGNet is a suitable small-data architecture, not a proven PD detector

Lawhern et al., *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*, **Journal of Neural Engineering (2018)**, uses temporal, depthwise spatial, and separable convolutions. Its evidence concerns BCI paradigms, not PD diagnosis. Its compact structure motivates our starting model and limits unnecessary compute. We will use a pinned implementation and validate its behavior on our task. [Author manuscript / publication record](https://arxiv.org/abs/1611.08024); [Braindecode EEGNet implementation documentation](https://braindecode.org/1.4/generated/braindecode.models.EEGNet.html).

### 4. A convolutional feature baseline is useful

Schirrmeister et al., *Deep learning with convolutional neural networks for EEG decoding and visualization*, **Human Brain Mapping (2017)**, provides EEG-specific convolutional architectures and analysis of learned features. It supports a small ShallowConvNet as a later comparator, but does not establish PD performance. Start with one neural architecture before broadening the search. [Paper](https://doi.org/10.1002/hbm.23730).

### 5. More elaborate models do not guarantee stronger generalization

*Parkinson’s Disease Detection from Resting State EEG using Multi-Head Graph Structure Learning with Gradient Weighted Graph Attention Explanations*, **2024 author preprint**, reports **69.40% accuracy** with subject-wise leave-one-out evaluation on UCSD data. It motivates later spatial-connectivity comparisons, not an initial graph network. [Manuscript](https://arxiv.org/abs/2408.00906).

Lee et al., *Transformer-Based Classification of Parkinson’s Disease from EEG Using BIDS-Formatted OpenNeuro Datasets*, **AAAI Symposium Series (2026)**, evaluates grouped subject-wise cross-validation for control/PD-ON/PD-OFF classification across three datasets. The abstract reports high but variable within-dataset F1 scores; grouped validation is valuable, but it is not equivalent to an untouched external-site evaluation. Its medication-state task also differs from our binary task. We will defer transformers until the simpler models establish a trustworthy baseline. [Proceedings paper](https://doi.org/10.1609/aaaiss.v8i1.42519).

### 6. Physiology guides hypotheses, not one-band diagnoses

Jackson et al., *Characteristics of Waveform Shape in Parkinson’s Disease Detected with Scalp Electroencephalography*, **eNeuro (2019)**, motivates investigating waveform shape as well as power. We will treat such features as secondary hypotheses rather than fixed disease signatures. [Paper](https://doi.org/10.1523/ENEURO.0151-19.2019).

Donoghue et al., *Parameterizing neural power spectra into periodic and aperiodic components*, **Nature Neuroscience (2020)**, shows why broad spectral changes and oscillatory peaks should be distinguished. It motivates an optional aperiodic/peak feature baseline; it is not a PD classifier study. [Paper](https://doi.org/10.1038/s41593-020-00744-x).

**Design implication:** prioritize cohort comparability, leakage prevention, simple baselines, and external testing over architecture complexity. These are our proposed design choices, not a claim that any paper validates the future product.

## Cohort selection before window generation

Use diagnosed PD and controls collected under the same study protocol. Do not use unrelated young healthy, motor-imagery, walking, or sleep collections to enlarge the negative class. Otherwise the classifier may learn age, equipment, task or study identity.

Provisional allocations, to be frozen after metadata/QC review and before fitting:

1. **Development cohort: Iowa resting EEG, ds004584.** Source README describes **100 PD and 49 controls**, 64-channel BrainVision, eyes-open rest for two minutes. Create a stratified participant split of approximately **70% train / 15% validation / 15% internal test**. Before exclusions, an illustrative allocation is 70/15/15 PD and 35/7/7 controls: 105/22/22 participants. Actual split counts will reflect QC and identity resolution. [Dataset](https://openneuro.org/datasets/ds004584).
2. **External candidate: UCSD ds002778.** Participant metadata contains **15 PD and 16 controls**, with repeated PD medication sessions. Reserve the entire cohort. Confirm eye condition, electrode mapping, reference and medication comparability before declaring the external endpoint. If those cannot be aligned, describe it as a protocol-shift stress test rather than a matched external replication. [Dataset](https://openneuro.org/datasets/ds002778).
3. **Second external candidate: New Mexico ds003490.** Source README describes **25 PD and 25 matched controls** with one-minute eyes-open/eyes-closed rest segments and task recordings; PD have repeated ON/OFF sessions. Only correctly identified resting segments qualify. Reserve this cohort initially; include it as a second external check once event timing is verified. Do not count an entire oddball recording as resting EEG. [Dataset](https://openneuro.org/datasets/ds003490).
4. **Expansion candidate: ds008768.** Source metadata describes **312 unique participants / 358 recordings**, eyes-open rest, multiple contributing studies, and Iowa authorship. Its relationship to older Iowa releases remains unresolved. Do not combine it with ds004584 or call it an independent external site until provenance/overlap is reconciled. It can replace the older development cohort in a separately versioned protocol, but must not be added after inspecting test results. [Dataset](https://openneuro.org/datasets/ds008768).

These are source-level counts, not verified eligible counts. The archive’s source README/participant metadata were inspected; complete signal QC and exact cross-release identity matching have not been performed. The source catalogue is [cohort-sources.json](cohort-sources.json).

### Identity and confounding checks

Build a subject registry linking dataset IDs, original participant IDs, study/site, sessions, medication state, age, sex, clinical labels and acquisition details. Use source identifiers, recording fingerprints, acquisition provenance, and available metadata to identify duplicate/re-exported recordings. Exact file hashes alone cannot detect all re-exports. Shared lab origin indicates an overlap risk, not proof of duplicated people. If overlap cannot be ruled out, keep the releases in the same evaluation group or exclude the ambiguous release.

Every session, medication state, epoch and derived representation of one person inherits the same split. No fitting on the external data, including unsupervised pretraining or normalization statistics. Metadata-only compatibility checks on external data are allowed; model selection based on its labels/scores is not.

Record age/sex imbalance, cognition, disease severity, medication state, reference, montage, sampling frequency and retained duration by cohort. Demographic matching/reweighting rules are fixed using development data; apply them as declared sensitivity analyses rather than selectively changing the test cohort. Unknown medication state remains unknown. OFF medication is not equivalent to untreated/de novo disease.

## Proposed preprocessing

1. Read BDF, EDF, BrainVision and EEGLAB with MNE; preserve raw files and original units. Validate linked header/data files and reject inconsistent headers. This extends the app’s current EDF/CSV import support; the app’s existing anomaly score is not a training feature by default.
2. Select documented awake resting segments in one eye condition, provisionally eyes open. Exclude tasks and auxiliary audio, acceleration, ECG and trigger channels from model input. Their availability must not reveal the class.
3. Harmonize a fixed electrode list and channel order, provisionally **16–32 shared scalp channels**, after a label-blind montage audit. Freeze the list before training. Do not assume a reference electrode is a recorded channel or infer a montage from array position.
4. Apply a fixed **1–40 Hz** bandpass and anti-alias resampling to **256 Hz**, with padding and edge trimming. Avoid filtering across recording discontinuities or between participants. Suppress line noise only as needed using a fixed rule; do not select a filter using test performance. Downsample high-rate recordings early without loading the complete archive into memory.
5. Re-reference consistently over the agreed channel set if technically valid. Record missing/bad channels. For v1, prefer excluding incompatible recordings to inserting artificial zeros. If interpolation is added, its montage and maximum missing-channel rule must be fixed and evaluated as a separate sensitivity analysis.
6. Use label-blind artifact/QC rules for saturation, flatlines, extreme amplitude, abrupt jumps and EOG/EMG contamination. Set thresholds on training data only; log retained/rejected duration by class and site. Do not copy the current explorer’s heuristic thresholds without validation. Manual QC should be blinded to diagnosis where feasible.
7. Use **non-overlapping four-second windows** for the first experiment. Cap the primary comparison at **60 seconds of accepted rest per participant**, with at least **40 seconds** required. This accommodates shorter New Mexico rest segments and prevents long recordings dominating. These are proposed thresholds; confirm feasibility without looking at classifier outcomes. Report exclusions and test coverage.
8. Convert to consistent physical units, remove each window’s channel mean, then scale with robust per-channel parameters fitted only on training participants. Do not independently z-score every window in v1, which would remove amplitude information. Fixed processing of a new recording is permitted; population-level parameters never use validation/test participants.

QC and processing are versioned, reproducible transformations. Cache compact float32 arrays and lazy-load minibatches from the SSD. Record the software versions, preprocessing configuration and raw-file fingerprints. Fit no ICA, feature selection, scaler, imputer, augmentation policy or batch-normalization statistics jointly across evaluation splits.

## Starting models and training budget

**Baseline A:** class-frequency/majority predictions. **Baseline B:** regularized logistic regression on participant-level log bandpower summaries (delta, theta, alpha, beta; fixed channel/region summaries), with all scaling and regularization selected on development data. **Confounding check:** age/sex-only baseline when available. Missingness indicators require the same audit as features.

**Neural model:** EEGNet-style temporal convolution → depthwise spatial convolution → separable convolution → pooling/dropout → binary output. Input is channels × 1,024 samples. Starting values: F1=8, depth multiplier=2, F2=16, temporal kernel=128 samples, dropout=0.5, AdamW, learning rate 0.001, weight decay 0.0001, batch size 32. Train at most 100 epochs with early stopping after 15 epochs without improvement in participant-level validation AUROC. These are proposed defaults, not established optimal values. Count actual parameters when implemented.

Use participant-balanced sampling, then class balancing at participant level. Each participant contributes approximately equal weight regardless of number of windows/sessions. Do not simultaneously apply multiple imbalance corrections without an explicit experiment. Start without augmentation; consider modest amplitude/noise perturbations only in training and only after verifying physiological plausibility.

Limit initial neural tuning to **six configurations**, evaluated using **three participant-grouped folds within the training split**. After choosing a configuration, train **three seeds** on the training split with the separate validation set for early stopping. Lock a single prespecified probability-average ensemble, or a single fixed-seed model if resources require, before either test set is opened. Do not pick the best seed on test performance.

Average window probabilities to one session score. For primary participant evaluation select one eligible session by a predefined rule (earliest eligible visit, medication state documented); use all grouped medication sessions only in a separately reported robustness analysis. Do not present dozens of windows as dozens of independent patients.

## Validation and test protocol

The internal test measures transfer to new people under the development study’s conditions. The external test measures transfer across people and acquisition context. They answer different questions and stay separate in reporting.

- Write and hash an immutable split manifest before generating model-ready windows. Assert zero overlap in canonical participant IDs and duplicate recording identities across splits. Windowing never creates a new split unit.
- Tune architectures/features only within training folds. Use the validation set for early stopping, threshold selection and optional calibration. A 22-person validation set is small: default to no learned calibration initially; any later calibration needs grouped out-of-fold predictions or more validation participants.
- Primary metric: **participant-level AUROC**, with a stratified participant-bootstrap 95% interval. Also report balanced accuracy, sensitivity, specificity, confusion matrix, PR-AUC with class prevalence, Brier score and coverage (fraction of participants receiving a result).
- Choose the operating threshold on validation predictions using a prespecified rule, initially maximum balanced accuracy. Freeze it before testing. Include a sensitivity-at-target-specificity analysis only with its uncertainty and validation-selected threshold; do not promise the same specificity externally.
- Treat window scores as descriptive diagnostics, not independent observations for confidence intervals. Compare models with paired participant-level uncertainty estimates. Small external cohorts will produce wide intervals.
- Report each external study separately, including medication-state and age/sex slices when sample sizes permit. Do not hide poor study transfer in a pooled average.
- Check whether site, artifact burden, recording length, or available channels can explain predictions. Include a training-only subject-label permutation sanity check. Any adjusted protocol after viewing test scores requires a new model/protocol version and an honest declaration that those test sets have become exploratory.

There is no justified promise of 90% or 99% accuracy. A useful first outcome may be discovering that a spectral baseline matches the network or that apparent performance fails externally. A credible result needs above-chance discrimination with uncertainty reported, an advantage or practical reason over the simple baseline, acceptable coverage, and no obvious leakage/confounding explanation. It is not a clinical deployment gate.

## Local compute: what the existing Mac can do

Inspected hardware: **MacBook Pro, Apple M4, 10 CPU cores, 10 GPU cores, 16 GB unified memory**. PyTorch supports Apple GPU execution through MPS; compatibility and speed must be checked for the pinned model/operators. CPU fallback is possible but must be reported in timing. [Official MPS documentation](https://docs.pytorch.org/docs/main/notes/mps.html).

This compact workload should fit if preprocessing is streamed and batches are bounded. Shared RAM also serves macOS, the app and GPU; the entire archive must never be preloaded. For illustration, 200 participants × 60 seconds × 32 channels × 256 samples/second × four bytes is only **0.393 GB** of float32 signal before metadata/copies. Large raw archive size is not model memory demand.

**Planning estimates, not measured benchmarks:**

- Initial metadata reconciliation, adapters, split checks and QC implementation: approximately **3–5 working days**.
- Cached preprocessing for a 150–300 participant pilot: allow **1–6 machine-hours**, potentially longer for ICA, unusual formats, high sampling rates or SSD contention.
- One compact-model fit: allow **0.5–3 hours** locally until benchmarked.
- Eighteen tuning fits plus three final seeds: approximately **10.5–63 training-hours** at that assumed rate; allow **roughly 1–4 days of machine time** including preprocessing and reruns. This assumes sustained running, not normal intermittent laptop use.
- Baselines, pipeline integration, initial training and error analysis: another **4–7 working days** after data audit; locked evaluations, documentation and reproducibility: another **3–5 working days**. Allow **2–4 calendar weeks** for a carefully reviewed first research result, with **1–3 additional weeks** if cohort identities/protocols need resolution. This is an engineering estimate, not a clinical-validation schedule.

Timing basis: 105 training participants × 15 four-second windows = 1,575 training windows per epoch. At an **assumed, unmeasured** 20–100 training windows/second, 100 epochs take 26–131 minutes before validation and I/O, motivating the 0.5–3-hour allowance. These throughput numbers are sensitivity assumptions, not claims about M4 performance. More data, overlap, larger models and CPU fallback can increase the estimate materially.

Before committing to the full experiment matrix, time 5–10 warm epochs on the real cached training subset, including validation/I/O. Record epochs/second, windows/second, peak memory, device and fallback behavior. Forecast total time from the measured full-epoch duration × remaining epochs × fits, with a 25% buffer. This benchmark is future implementation work; none has been run for this document.

## Approved compute policy

Use this local Mac only. Before launching any uninterrupted stage projected to exceed 96 hours (four days), stop and notify the user with the measured estimate. Cloud cost planning is excluded from the active project. Checkpoints and a runtime watchdog enforce this bound.

## Later expansion and Myna integration

Once PD evaluation is credible, define a new protocol for control versus the included disease set and a conditional disease subtype model. Split people and sites once for the entire hierarchy. Train the subtype model on ground-truth disease cases; report both subtype performance among true disease cases and end-to-end performance including cases rejected by the first stage. Compare against a flat control/PD/AD/FTD/ALS model because a hard two-stage gate can compound errors.

Disease class must not equal dataset/site. AD/FTD and ALS datasets may use different eye conditions, tasks, age distributions and hardware. Matched controls at each site do not alone eliminate disease-specific site confounding; credible multiclass testing needs disease coverage across sites or independently validated class-specific tasks. Retain an insufficient-quality/out-of-scope outcome; low model confidence does not establish that an unknown disease has been detected.

For Myna, initially export a versioned EEG research score, quality summary, recording protocol, model version, and participant/session linkage. Do not concatenate unrelated people’s public EEG with Myna audio or movement records. Multimodal benefit requires paired modalities from the same participants, grouped evaluation, and comparison of audio-only, movement-only, EEG-only and combined models.

Symptom tracking is a separate longitudinal study: repeated recordings with time-aligned clinical or patient-reported symptom labels, medication timing and test–retest reliability. Cross-sectional PD classification cannot show progression or sensitivity to symptom change. A future unobtrusive sensor also needs its own electrode/channel and motion validation; success on a stationary 32/64-channel cap does not transfer automatically.

## Concrete implementation deliverables after approval

1. A reviewed subject registry, cohort card, immutable split manifest and exclusions report.
2. Format adapters and versioned preprocessing with a small manually inspected QC set.
3. Unit checks for no participant/recording leakage, train-only statistics, reproducible window provenance and consistent units/channel order.
4. Spectral/demographic baselines and one EEGNet-style model, with local CPU/MPS capability and timing checks.
5. A bounded experiment configuration, participant-level evaluation and uncertainty report, then locked internal/external results.
6. A model card recording intended use, data overlap limitations, subgroup results, failure cases, resource use and reproducibility instructions. Only then consider adding research predictions to the local application.

**Review decision:** approved for local implementation. Runtime estimates above remain provisional until measured.
