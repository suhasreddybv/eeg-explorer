# Expanded PD EEG comparison: reviewed report

## Parkinson's EEG: expanded-data comparison

Protocol v3 | Completed 27 September 2026 | Reviewed 28 September 2026 UTC

Research question: does broader EEG pretraining and full-duration, multi-site PD data improve discrimination between diagnosed Parkinson's disease and study controls?

### The result

There is no demonstrated cross-site benefit from self-supervised pretraining in this experiment. The rich-feature SVM had the highest observed equal-site mean AUROC, 0.763 (95% participant-bootstrap interval 0.704-0.820). The pretrained hybrid SVM was essentially tied at 0.761. The pretrained neural model scored 0.685 versus 0.699 when trained from scratch.

The full procedure that selected a family using inner validation scored 0.690 cross-site (95% interval 0.611-0.758). This is the appropriate result for the selection procedure. Selecting the observed best family after viewing outer results introduces selection optimism; 0.763 is a descriptive family result, not a new unbiased best-model claim.

- Completed work: Archive processing; Scope: 11,211 candidate recordings; 7,988 with accepted windows
- Completed work: PD comparison; Scope: 440 eligible participants at four sites; all compatible duration and repeated resting sessions
- Completed work: Auxiliary pretraining; Scope: 321,285 accepted windows, 850 training groups and 110 validation groups
- Completed work: Training; Scope: 620 planned fits: 160 neural and 460 classical; one pretraining run; 14 post-hoc diagnostic fits
- Completed work: Evaluation; Scope: Four held-out-site folds and three separate within-site participant folds

### Interpretation comes first

All four PD sites were previously examined in preliminary studies; v3 is exploratory. Recording metadata alone perfectly separated diagnoses at UCSD and New Mexico, where the study design gave PD participants repeated medication sessions and controls one visit. EEG models did not directly receive session count, but measurement and acquisition confounding remain serious concerns.

This result supports continued research. It does not establish a diagnostic test, disease progression score, wearable monitor performance or symptom-tracking product. The earlier v2 report remains unchanged.

## 1. Evaluation design and model selection

Primary: hold out one entire site, train on the other three, and repeat for all four sites. Within each outer training set, hold out its sites in turn to choose configurations by equal-site mean AUROC. Every person, visit, medication session and window remains within the same fold.

Secondary: three participant folds stratified jointly by site and diagnosis. Each test person is unseen, while the training data represent the same institutions. Inner site validation still selects settings. Results are reported per site and as an equal-site mean, separately from primary transfer results.

All supervised test people were excluded from all auxiliary pretraining, including overlapping releases. Fixed within-recording feature extraction was cached globally, but population scalers, calibration, classical fits and supervised learning used only the relevant training data.

- Stage: Candidate search; Frozen rule: 19 settings across seven families: three C values for five classical families and two learning rates for two neural families
- Stage: Neural tuning; Frozen rule: Maximum 40 epochs, patience seven, inner validation only; one fixed tuning seed
- Stage: Final neural fits; Frozen rule: Median inner best epoch; three fixed seeds averaged without choosing a winning seed
- Stage: Threshold; Frozen rule: Inner out-of-fold balanced-accuracy optimum; no outer threshold adjustment
- Stage: Outer lock; Frozen rule: Every outer fit completed and locked before outer predictions were generated
- Stage: Final all-data fits; Frozen rule: Separate inner-site selection on all 440 people, then final research fits; no new test claim

The inner-selected family differed across outer folds. This variability is part of the model-selection procedure, not a reason to replace its results with the best family seen on each test.

- Outer evaluation: site iowa; Family selected without its test: Neural, pretrained
- Outer evaluation: site tel aviv; Family selected without its test: Rich-feature SVM
- Outer evaluation: site ucsd; Family selected without its test: Neural, pretrained
- Outer evaluation: site unm; Family selected without its test: Neural, from scratch
- Outer evaluation: within 0; Family selected without its test: Pretrained hybrid SVM
- Outer evaluation: within 1; Family selected without its test: Rich-feature SVM
- Outer evaluation: within 2; Family selected without its test: Neural, from scratch

Bootstrap intervals resample participants within the four fixed sites. They do not quantify uncertainty from having only four sites, model refitting, hyperparameter selection or earlier exposure to these datasets. Paired and subgroup comparisons are exploratory and unadjusted for multiplicity.

## 2. Did we use the whole downloaded archive?

No. Every dataset was accounted for, but not every recording is an eligible PD example or a compatible pretraining input. Unlike v2, v3 has no 60-second cap and includes repeated compatible resting sessions. It uses fixed spectral representations of accepted signal windows, not the complete raw waveform as neural input.

- Dataset: ds002778; Raw recordings: 46; Candidate inputs: 46; Role: PD rest
- Dataset: ds003194; Raw recordings: 29; Candidate inputs: 0; Role: Excluded
- Dataset: ds003195; Raw recordings: 20; Candidate inputs: 0; Role: Excluded
- Dataset: ds003490; Raw recordings: 75; Candidate inputs: 75; Role: PD rest
- Dataset: ds003506; Raw recordings: 84; Candidate inputs: 0; Role: Excluded
- Dataset: ds003509; Raw recordings: 84; Candidate inputs: 0; Role: Excluded
- Dataset: ds003775; Raw recordings: 153; Candidate inputs: 153; Role: Auxiliary
- Dataset: ds004148; Raw recordings: 195; Candidate inputs: 195; Role: Auxiliary
- Dataset: ds004504; Raw recordings: 88; Candidate inputs: 88; Role: Auxiliary
- Dataset: ds004574; Raw recordings: 146; Candidate inputs: 0; Role: Excluded
- Dataset: ds004579; Raw recordings: 139; Candidate inputs: 0; Role: Excluded
- Dataset: ds004580; Raw recordings: 147; Candidate inputs: 0; Role: Excluded
- Dataset: ds004584; Raw recordings: 149; Candidate inputs: 0; Role: Excluded
- Dataset: ds004796; Raw recordings: 235; Candidate inputs: 235; Role: Auxiliary
- Dataset: ds005048; Raw recordings: 27; Candidate inputs: 0; Role: Excluded
- Dataset: ds005385; Raw recordings: 3264; Candidate inputs: 3264; Role: Auxiliary
- Dataset: ds006036; Raw recordings: 88; Candidate inputs: 88; Role: Auxiliary
- Dataset: ds007526; Raw recordings: 277; Candidate inputs: 144; Role: PD rest
- Dataset: ds007864; Raw recordings: 84; Candidate inputs: 84; Role: Auxiliary
- Dataset: ds007865; Raw recordings: 47; Candidate inputs: 47; Role: Auxiliary
- Dataset: ds007987; Raw recordings: 172; Candidate inputs: 0; Role: Excluded
- Dataset: ds008768; Raw recordings: 358; Candidate inputs: 356; Role: PD rest
- Dataset: bigp3bci; Raw recordings: 6436; Candidate inputs: 6436; Role: Auxiliary

Ten sources were excluded for overlapping/replaced releases, an unsupported MATLAB v7.3 format, or unresolved physical electrode mapping. PD walking/task acquisitions were excluded. Two exact repeated Iowa signals were retained once. The ledger and complete exclusion reasons accompany the repository report.

Of 11,211 candidates, 3,223 produced no input: 2,855 incompatible-montage recordings, 367 recordings with no accepted window and one ambiguous New Mexico eye interval. The 7,988 recordings with windows are not 7,988 people; some groups still fail the minimum 40 seconds.

## 3. Cohort coverage and demographic balance

- Site: Iowa; Diagnosis: Control; Retained / assigned: 97/110; Excluded: 11.8%; Age mean (SD): 69.4 (10.6); Female / male: 50/47
- Site: Iowa; Diagnosis: PD; Retained / assigned: 171/202; Excluded: 15.3%; Age mean (SD): 68.7 (8.0); Female / male: 50/121
- Site: Tel Aviv; Diagnosis: Control; Retained / assigned: 23/28; Excluded: 17.9%; Age mean (SD): 60.0 (7.9); Female / male: 11/12
- Site: Tel Aviv; Diagnosis: PD; Retained / assigned: 79/116; Excluded: 31.9%; Age mean (SD): 65.5 (9.6); Female / male: 16/63
- Site: UC San Diego; Diagnosis: Control; Retained / assigned: 15/16; Excluded: 6.2%; Age mean (SD): 62.8 (9.6); Female / male: 8/7
- Site: UC San Diego; Diagnosis: PD; Retained / assigned: 15/15; Excluded: 0.0%; Age mean (SD): 63.3 (8.2); Female / male: 8/7
- Site: New Mexico; Diagnosis: Control; Retained / assigned: 18/25; Excluded: 28.0%; Age mean (SD): 68.2 (9.9); Female / male: 6/12
- Site: New Mexico; Diagnosis: PD; Retained / assigned: 22/25; Excluded: 12.0%; Age mean (SD): 70.0 (7.8); Female / male: 8/14

The four source cohorts contain 537 assigned people; 440 remain after fixed QC and the minimum accepted-duration rule. Age and sex describe retained people. All retained demographic fields used here were available. Tel Aviv excluded 37/116 PD participants (31.9%) and 5/28 controls (17.9%), creating differential selection.

### Auxiliary coverage

- Canonical source group: ds003775; Retained / assigned groups: 0/111; Accepted hours: 0.00
- Canonical source group: ds004148; Retained / assigned groups: 13/13; Accepted hours: 10.80
- Canonical source group: ds004504; Retained / assigned groups: 87/88; Accepted hours: 19.24
- Canonical source group: ds004796; Retained / assigned groups: 79/79; Accepted hours: 30.73
- Canonical source group: ds005385; Retained / assigned groups: 604/608; Accepted hours: 131.05
- Canonical source group: ds007864; Retained / assigned groups: 1/1; Accepted hours: 25.86
- Canonical source group: bigp3bci; Retained / assigned groups: 176/308; Accepted hours: 139.32

The complementary ds004504/ds006036 data share person IDs and are grouped together. The newer NeuroEPO treatment/placebo arms are conservatively grouped as one unresolved identity group, so their count is not a count of patients. Public de-identification prevents proof of absolute cross-study uniqueness.

All 153 ds003775 recordings failed the unchanged amplitude/step rules: 9,027 windows were rejected for both reasons. That source supplied no pretraining examples. This cohort-wide rejection merits an independent unit/reference/QC review before any future protocol, not post-result threshold relaxation.

## 4. Signal processing, representation and new model

Seventeen common named scalp channels; 256 Hz; 1-40 Hz fourth-order zero-phase filtering; average reference over observed channels; four-second non-overlapping windows with bounded chunks and two-second margins. Supervised inputs require all 17 channels; auxiliary inputs require at least 12 with explicit missing-channel masks. Legacy T3/T4/T5/T6 names map to T7/T8/P7/P8. Electrode positions are never guessed.

Keep every accepted window across documented compatible resting duration, including repeated sessions; require at least ten accepted windows per person. Window QC checks peak-to-peak amplitude, abrupt steps, flat channels, repeated samples and source bad annotations. These are engineering heuristics without blinded clinical EEG validation.

- Representation: Spectral baseline; Definition: Four per-electrode log-bandpower summaries: 2-4, 4-8, 8-13 and 13-30 Hz; 68 person-level inputs
- Representation: Rich features; Definition: Means and variability of 34 log-PSD bins per electrode, plus 153 log-covariance terms after 10% isotropic shrinkage; 1,309 inputs
- Representation: Pretrained hybrid; Definition: Rich features plus 96 pooled embedding mean/variability features; 1,405 inputs
- Representation: Neural encoder; Definition: 48-dimensional electrode tokens; learned positions; two four-head Transformer layers; 96-wide feed-forward layers; masked spectral decoder and binary head; 42,275 parameters

Per-person summaries weight sessions equally; all windows contribute within a session. Neural loss weights sites, diagnoses, people and sessions to limit domination by large cohorts or long recordings. Missing auxiliary channels are masked from attention keys, pooling and reconstruction targets.

### What the new model is, and is not

Pretraining reconstructs masked electrode spectra, motivated by masked EEG representation learning. It is a compact spectral encoder built for this Mac, not a reproduction of BENDR or LaBraM and not a large raw-waveform foundation model. The pretrained hybrid uses frozen embeddings; the pretrained neural classifier fine-tunes the encoder with a fresh binary head.

C = 0.1/1/10 for regression and RBF SVM; SVM gamma = scale. Neural learning rates = 0.0003/0.001; batch 128; AdamW weight decay 0.01; gradient clipping 5; final seeds 20260927-29. No settings were chosen from outer performance.

## 5. Primary and secondary aggregate results

- Family / procedure: Spectral regression; Cross-site AUROC (95% CI): 0.559 (0.479-0.635); Within-site AUROC (95% CI): 0.641 (0.562-0.714)
- Family / procedure: Rich-feature regression; Cross-site AUROC (95% CI): 0.710 (0.636-0.778); Within-site AUROC (95% CI): 0.664 (0.588-0.736)
- Family / procedure: Rich-feature SVM; Cross-site AUROC (95% CI): 0.763 (0.704-0.820); Within-site AUROC (95% CI): 0.711 (0.642-0.779)
- Family / procedure: Pretrained hybrid regression; Cross-site AUROC (95% CI): 0.720 (0.645-0.787); Within-site AUROC (95% CI): 0.658 (0.582-0.731)
- Family / procedure: Pretrained hybrid SVM; Cross-site AUROC (95% CI): 0.761 (0.700-0.819); Within-site AUROC (95% CI): 0.715 (0.646-0.784)
- Family / procedure: Neural, from scratch; Cross-site AUROC (95% CI): 0.699 (0.624-0.769); Within-site AUROC (95% CI): 0.642 (0.563-0.720)
- Family / procedure: Neural, pretrained; Cross-site AUROC (95% CI): 0.685 (0.605-0.756); Within-site AUROC (95% CI): 0.677 (0.600-0.747)
- Family / procedure: Inner-selected pipeline; Cross-site AUROC (95% CI): 0.690 (0.611-0.758); Within-site AUROC (95% CI): 0.693 (0.616-0.768)

These are equal-site means, so the small New Mexico and UCSD cohorts carry the same weight as Iowa. Per-site results and sample sizes follow. The inner-selected pipeline is the predeclared family-selection procedure; the observed best family is a separate descriptive comparison.

### Did pretraining help? Paired differences

- Paired contrast: Rich SVM minus spectral regression; Cross-site delta (95% CI): +0.203 (+0.130 to +0.281); Within-site delta (95% CI): +0.070 (-0.007 to +0.153)
- Paired contrast: Pretrained hybrid SVM minus rich SVM; Cross-site delta (95% CI): -0.001 (-0.018 to +0.014); Within-site delta (95% CI): +0.004 (-0.024 to +0.033)
- Paired contrast: Pretrained hybrid LR minus rich LR; Cross-site delta (95% CI): +0.011 (-0.007 to +0.030); Within-site delta (95% CI): -0.006 (-0.033 to +0.022)
- Paired contrast: Pretrained neural minus scratch neural; Cross-site delta (95% CI): -0.014 (-0.063 to +0.034); Within-site delta (95% CI): +0.035 (-0.014 to +0.089)

Rich spectral/covariance features with an SVM outperformed the four-band regression baseline in this comparison. Adding pretrained embeddings did not provide a demonstrated cross-site gain. The pretrained neural model did not outperform its scratch counterpart; the paired interval includes both harm and benefit. These findings do not prove that all self-supervised EEG approaches are ineffective.

## Primary cross-site: Iowa and Tel Aviv

### Iowa | 268 people

- Model: Spectral regression; AUROC: 0.605; 95% CI: 0.531-0.671; Sens.: 0.380; Spec.: 0.742; PR-AUC: 0.732; Brier: 0.408
- Model: Rich-feature regression; AUROC: 0.676; 95% CI: 0.606-0.739; Sens.: 0.462; Spec.: 0.794; PR-AUC: 0.782; Brier: 0.286
- Model: Rich-feature SVM; AUROC: 0.742; 95% CI: 0.676-0.801; Sens.: 0.474; Spec.: 0.835; PR-AUC: 0.847; Brier: 0.208
- Model: Pretrained hybrid regression; AUROC: 0.679; 95% CI: 0.608-0.742; Sens.: 0.199; Spec.: 0.907; PR-AUC: 0.778; Brier: 0.314
- Model: Pretrained hybrid SVM; AUROC: 0.746; 95% CI: 0.682-0.804; Sens.: 0.485; Spec.: 0.814; PR-AUC: 0.850; Brier: 0.206
- Model: Neural, from scratch; AUROC: 0.679; 95% CI: 0.611-0.740; Sens.: 0.456; Spec.: 0.825; PR-AUC: 0.802; Brier: 0.240
- Model: Neural, pretrained; AUROC: 0.709; 95% CI: 0.643-0.771; Sens.: 0.251; Spec.: 0.948; PR-AUC: 0.822; Brier: 0.269
- Model: Inner-selected pipeline; AUROC: 0.709; 95% CI: 0.643-0.771; Sens.: 0.251; Spec.: 0.948; PR-AUC: 0.822; Brier: 0.269

### Tel Aviv | 102 people

- Model: Spectral regression; AUROC: 0.395; 95% CI: 0.272-0.510; Sens.: 0.658; Spec.: 0.130; PR-AUC: 0.749; Brier: 0.388
- Model: Rich-feature regression; AUROC: 0.616; 95% CI: 0.471-0.755; Sens.: 0.949; Spec.: 0.217; PR-AUC: 0.813; Brier: 0.197
- Model: Rich-feature SVM; AUROC: 0.690; 95% CI: 0.582-0.795; Sens.: 0.823; Spec.: 0.304; PR-AUC: 0.898; Brier: 0.170
- Model: Pretrained hybrid regression; AUROC: 0.627; 95% CI: 0.479-0.767; Sens.: 0.987; Spec.: 0.174; PR-AUC: 0.807; Brier: 0.194
- Model: Pretrained hybrid SVM; AUROC: 0.698; 95% CI: 0.587-0.801; Sens.: 0.785; Spec.: 0.348; PR-AUC: 0.902; Brier: 0.169
- Model: Neural, from scratch; AUROC: 0.789; 95% CI: 0.687-0.878; Sens.: 0.937; Spec.: 0.391; PR-AUC: 0.935; Brier: 0.147
- Model: Neural, pretrained; AUROC: 0.691; 95% CI: 0.576-0.802; Sens.: 0.975; Spec.: 0.174; PR-AUC: 0.895; Brier: 0.178
- Model: Inner-selected pipeline; AUROC: 0.690; 95% CI: 0.582-0.795; Sens.: 0.823; Spec.: 0.304; PR-AUC: 0.898; Brier: 0.170

Sensitivity and specificity use training-selected thresholds; within-site thresholds can differ across folds. PR-AUC is average precision and depends on PD prevalence. Brier score assesses probability error. High AUROC alone does not imply a useful operating threshold.

UCSD and New Mexico aggregate repeated PD medication sessions, while controls typically have one session. The metadata confound findings must accompany these scores. All participants are counted once per evaluation design.

## Primary cross-site: UCSD and New Mexico

### UC San Diego | 30 people

- Model: Spectral regression; AUROC: 0.529; 95% CI: 0.311-0.733; Sens.: 0.600; Spec.: 0.400; PR-AUC: 0.551; Brier: 0.271
- Model: Rich-feature regression; AUROC: 0.769; 95% CI: 0.569-0.929; Sens.: 0.467; Spec.: 0.933; PR-AUC: 0.836; Brier: 0.202
- Model: Rich-feature SVM; AUROC: 0.831; 95% CI: 0.658-0.960; Sens.: 0.667; Spec.: 0.800; PR-AUC: 0.863; Brier: 0.194
- Model: Pretrained hybrid regression; AUROC: 0.809; 95% CI: 0.618-0.956; Sens.: 0.400; Spec.: 0.933; PR-AUC: 0.861; Brier: 0.185
- Model: Pretrained hybrid SVM; AUROC: 0.822; 95% CI: 0.649-0.951; Sens.: 0.667; Spec.: 0.800; PR-AUC: 0.860; Brier: 0.195
- Model: Neural, from scratch; AUROC: 0.627; 95% CI: 0.418-0.827; Sens.: 0.267; Spec.: 0.800; PR-AUC: 0.657; Brier: 0.242
- Model: Neural, pretrained; AUROC: 0.658; 95% CI: 0.453-0.844; Sens.: 0.133; Spec.: 0.933; PR-AUC: 0.653; Brier: 0.229
- Model: Inner-selected pipeline; AUROC: 0.658; 95% CI: 0.453-0.844; Sens.: 0.133; Spec.: 0.933; PR-AUC: 0.653; Brier: 0.229

### New Mexico | 40 people

- Model: Spectral regression; AUROC: 0.710; 95% CI: 0.543-0.866; Sens.: 0.591; Spec.: 0.667; PR-AUC: 0.722; Brier: 0.214
- Model: Rich-feature regression; AUROC: 0.778; 95% CI: 0.621-0.907; Sens.: 0.636; Spec.: 0.722; PR-AUC: 0.784; Brier: 0.217
- Model: Rich-feature SVM; AUROC: 0.788; 95% CI: 0.639-0.909; Sens.: 0.545; Spec.: 0.778; PR-AUC: 0.836; Brier: 0.199
- Model: Pretrained hybrid regression; AUROC: 0.765; 95% CI: 0.606-0.896; Sens.: 0.682; Spec.: 0.667; PR-AUC: 0.783; Brier: 0.219
- Model: Pretrained hybrid SVM; AUROC: 0.779; 95% CI: 0.629-0.903; Sens.: 0.682; Spec.: 0.722; PR-AUC: 0.829; Brier: 0.199
- Model: Neural, from scratch; AUROC: 0.702; 95% CI: 0.533-0.856; Sens.: 0.727; Spec.: 0.667; PR-AUC: 0.725; Brier: 0.227
- Model: Neural, pretrained; AUROC: 0.684; 95% CI: 0.513-0.841; Sens.: 0.273; Spec.: 0.889; PR-AUC: 0.754; Brier: 0.230
- Model: Inner-selected pipeline; AUROC: 0.702; 95% CI: 0.533-0.856; Sens.: 0.727; Spec.: 0.667; PR-AUC: 0.725; Brier: 0.227

Sensitivity and specificity use training-selected thresholds; within-site thresholds can differ across folds. PR-AUC is average precision and depends on PD prevalence. Brier score assesses probability error. High AUROC alone does not imply a useful operating threshold.

UCSD and New Mexico aggregate repeated PD medication sessions, while controls typically have one session. The metadata confound findings must accompany these scores. All participants are counted once per evaluation design.

## Separate within-site: Iowa and Tel Aviv

### Iowa | 268 people

- Model: Spectral regression; AUROC: 0.634; 95% CI: 0.564-0.698; Sens.: 0.573; Spec.: 0.588; PR-AUC: 0.769; Brier: 0.267
- Model: Rich-feature regression; AUROC: 0.695; 95% CI: 0.629-0.762; Sens.: 0.503; Spec.: 0.732; PR-AUC: 0.781; Brier: 0.259
- Model: Rich-feature SVM; AUROC: 0.718; 95% CI: 0.656-0.778; Sens.: 0.602; Spec.: 0.732; PR-AUC: 0.841; Brier: 0.200
- Model: Pretrained hybrid regression; AUROC: 0.692; 95% CI: 0.624-0.760; Sens.: 0.602; Spec.: 0.629; PR-AUC: 0.766; Brier: 0.266
- Model: Pretrained hybrid SVM; AUROC: 0.718; 95% CI: 0.655-0.777; Sens.: 0.667; Spec.: 0.680; PR-AUC: 0.837; Brier: 0.201
- Model: Neural, from scratch; AUROC: 0.667; 95% CI: 0.598-0.735; Sens.: 0.532; Spec.: 0.732; PR-AUC: 0.806; Brier: 0.232
- Model: Neural, pretrained; AUROC: 0.670; 95% CI: 0.605-0.736; Sens.: 0.468; Spec.: 0.784; PR-AUC: 0.801; Brier: 0.238
- Model: Inner-selected pipeline; AUROC: 0.715; 95% CI: 0.651-0.775; Sens.: 0.632; Spec.: 0.680; PR-AUC: 0.842; Brier: 0.205

### Tel Aviv | 102 people

- Model: Spectral regression; AUROC: 0.594; 95% CI: 0.463-0.725; Sens.: 0.823; Spec.: 0.261; PR-AUC: 0.827; Brier: 0.222
- Model: Rich-feature regression; AUROC: 0.659; 95% CI: 0.529-0.786; Sens.: 0.696; Spec.: 0.522; PR-AUC: 0.854; Brier: 0.195
- Model: Rich-feature SVM; AUROC: 0.739; 95% CI: 0.621-0.840; Sens.: 0.823; Spec.: 0.522; PR-AUC: 0.910; Brier: 0.153
- Model: Pretrained hybrid regression; AUROC: 0.662; 95% CI: 0.538-0.784; Sens.: 0.722; Spec.: 0.478; PR-AUC: 0.865; Brier: 0.197
- Model: Pretrained hybrid SVM; AUROC: 0.747; 95% CI: 0.624-0.853; Sens.: 0.873; Spec.: 0.522; PR-AUC: 0.912; Brier: 0.152
- Model: Neural, from scratch; AUROC: 0.692; 95% CI: 0.580-0.799; Sens.: 0.785; Spec.: 0.391; PR-AUC: 0.886; Brier: 0.176
- Model: Neural, pretrained; AUROC: 0.782; 95% CI: 0.684-0.869; Sens.: 0.785; Spec.: 0.522; PR-AUC: 0.932; Brier: 0.150
- Model: Inner-selected pipeline; AUROC: 0.711; 95% CI: 0.594-0.822; Sens.: 0.835; Spec.: 0.435; PR-AUC: 0.890; Brier: 0.162

Sensitivity and specificity use training-selected thresholds; within-site thresholds can differ across folds. PR-AUC is average precision and depends on PD prevalence. Brier score assesses probability error. High AUROC alone does not imply a useful operating threshold.

UCSD and New Mexico aggregate repeated PD medication sessions, while controls typically have one session. The metadata confound findings must accompany these scores. All participants are counted once per evaluation design.

## Separate within-site: UCSD and New Mexico

### UC San Diego | 30 people

- Model: Spectral regression; AUROC: 0.760; 95% CI: 0.569-0.933; Sens.: 0.800; Spec.: 0.600; PR-AUC: 0.706; Brier: 0.208
- Model: Rich-feature regression; AUROC: 0.649; 95% CI: 0.444-0.827; Sens.: 0.467; Spec.: 0.733; PR-AUC: 0.668; Brier: 0.297
- Model: Rich-feature SVM; AUROC: 0.782; 95% CI: 0.600-0.933; Sens.: 0.533; Spec.: 0.800; PR-AUC: 0.816; Brier: 0.207
- Model: Pretrained hybrid regression; AUROC: 0.636; 95% CI: 0.422-0.818; Sens.: 0.533; Spec.: 0.667; PR-AUC: 0.691; Brier: 0.300
- Model: Pretrained hybrid SVM; AUROC: 0.773; 95% CI: 0.587-0.929; Sens.: 0.533; Spec.: 0.800; PR-AUC: 0.798; Brier: 0.200
- Model: Neural, from scratch; AUROC: 0.627; 95% CI: 0.418-0.813; Sens.: 0.533; Spec.: 0.667; PR-AUC: 0.660; Brier: 0.248
- Model: Neural, pretrained; AUROC: 0.631; 95% CI: 0.427-0.818; Sens.: 0.600; Spec.: 0.533; PR-AUC: 0.686; Brier: 0.246
- Model: Inner-selected pipeline; AUROC: 0.729; 95% CI: 0.529-0.907; Sens.: 0.600; Spec.: 0.733; PR-AUC: 0.720; Brier: 0.216

### New Mexico | 40 people

- Model: Spectral regression; AUROC: 0.576; 95% CI: 0.384-0.755; Sens.: 0.727; Spec.: 0.333; PR-AUC: 0.667; Brier: 0.269
- Model: Rich-feature regression; AUROC: 0.654; 95% CI: 0.462-0.818; Sens.: 0.455; Spec.: 0.611; PR-AUC: 0.676; Brier: 0.284
- Model: Rich-feature SVM; AUROC: 0.606; 95% CI: 0.419-0.780; Sens.: 0.636; Spec.: 0.556; PR-AUC: 0.678; Brier: 0.250
- Model: Pretrained hybrid regression; AUROC: 0.644; 95% CI: 0.457-0.801; Sens.: 0.500; Spec.: 0.611; PR-AUC: 0.716; Brier: 0.316
- Model: Pretrained hybrid SVM; AUROC: 0.621; 95% CI: 0.442-0.788; Sens.: 0.591; Spec.: 0.611; PR-AUC: 0.697; Brier: 0.241
- Model: Neural, from scratch; AUROC: 0.581; 95% CI: 0.396-0.760; Sens.: 0.500; Spec.: 0.667; PR-AUC: 0.693; Brier: 0.261
- Model: Neural, pretrained; AUROC: 0.624; 95% CI: 0.442-0.783; Sens.: 0.455; Spec.: 0.556; PR-AUC: 0.723; Brier: 0.243
- Model: Inner-selected pipeline; AUROC: 0.616; 95% CI: 0.437-0.793; Sens.: 0.591; Spec.: 0.611; PR-AUC: 0.716; Brier: 0.259

Sensitivity and specificity use training-selected thresholds; within-site thresholds can differ across folds. PR-AUC is average precision and depends on PD prevalence. Brier score assesses probability error. High AUROC alone does not imply a useful operating threshold.

UCSD and New Mexico aggregate repeated PD medication sessions, while controls typically have one session. The metadata confound findings must accompany these scores. All participants are counted once per evaluation design.

## 6. Pretraining, compute and final research models

Auxiliary pretraining used 284,219 training windows and 37,066 validation windows, equivalent to 315.8 and 41.2 accepted recording hours. A deterministic group split gave 850 training groups and 110 validation groups. None belonged to the PD evaluation cohorts. Twenty epochs completed in 26.4 minutes on the Mac GPU.

The best held-out reconstruction loss was approximately 0.348, down from 0.453 at epoch one. Reconstruction improved, but downstream cross-site PD discrimination did not show a pretraining advantage. Improving the auxiliary task does not establish improved disease classification.

- Compute measure: Mac; Observed / forecast: Apple M4, 16 GB unified memory; PyTorch MPS
- Compute measure: Pretraining forecast; Observed / forecast: 4.00 hours, maximum-epoch estimate with margin
- Compute measure: Supervised campaign forecast; Observed / forecast: 15.16 hours, maximum-epoch estimate with margin
- Compute measure: Completed fitting; Observed / forecast: 160 neural and 460 classical fits; summed fit duration 72.2 minutes
- Compute measure: Resumed independent campaign; Observed / forecast: 132.7 minutes from final preprocessing restart through aggregate report
- Compute measure: Timing exclusions; Observed / forecast: Earlier interrupted preparation, initial benchmarks, final audit and PDF review; no measured peak total GPU/unified-memory figure

### Final all-data selection

After nested evaluation, the prescribed training-only selection on all 440 eligible people chose the pretrained hybrid SVM (C=10). Its inner-site selection AUROC was 0.780; this is a tuning statistic, not a fresh external test score. All seven final family fits are retained locally. The chosen research model is not promoted as clinically superior.

- Final family: Rich-feature regression; Chosen setting: 0.1; Neural epochs: n/a
- Final family: Rich-feature SVM; Chosen setting: 10.0; Neural epochs: n/a
- Final family: Neural, from scratch; Chosen setting: 0.001; Neural epochs: 6
- Final family: Spectral regression; Chosen setting: 0.1; Neural epochs: n/a
- Final family: Pretrained hybrid regression; Chosen setting: 0.1; Neural epochs: n/a
- Final family: Pretrained hybrid SVM; Chosen setting: 10.0; Neural epochs: n/a
- Final family: Neural, pretrained; Chosen setting: 0.0003; Neural epochs: 2

All original models, predictions and scientific records remain frozen. The final report adds interpretation and descriptive diagnostics without retuning or selecting a new outer-test winner.

## 7. Confounding checks and exclusions

Post-lock descriptive baselines used fixed C=1 logistic regression and the same outer splits. The demographic baseline used age/sex; the recording-quality baseline used accepted-window count, assigned session count and acceptance fraction. Imputation and scaling were fitted inside each training fold. These were diagnostic additions, not prespecified competing EEG families.

- Site: Iowa; Cross: demographics: 0.568; Cross: recording metadata: 0.491; Within: demographics: 0.600; Within: recording metadata: 0.495
- Site: Tel Aviv; Cross: demographics: 0.547; Cross: recording metadata: 0.466; Within: demographics: 0.582; Within: recording metadata: 0.427
- Site: UC San Diego; Cross: demographics: 0.462; Cross: recording metadata: 1.000; Within: demographics: 0.342; Within: recording metadata: 1.000
- Site: New Mexico; Cross: demographics: 0.492; Cross: recording metadata: 1.000; Within: demographics: 0.434; Within: recording metadata: 1.000

The metadata baseline achieves AUROC 1.000 at both UCSD and New Mexico. Their acquisition schedules encode diagnosis: PD participants were invited for medication ON/OFF visits, controls for one. The EEG models never directly receive session count, but averaging unequal acquisition histories can affect measurement reliability and medication mixtures. Therefore these sites cannot establish disease-specific learning from EEG alone.

Age/sex-only discrimination was much weaker. This does not rule out residual demographic or equipment effects; a weak simple confound baseline is not proof that the EEG model uses PD-specific physiology.

### Additional material limitations

The artifact rules rejected whole sources or disproportionately excluded PD participants at some sites. Missing montages removed 2,855 bigP3BCI recordings. The newer NeuroEPO arms use an intentionally conservative combined identity group. Auxiliary pretraining mixes resting, task, eyes-open and eyes-closed data, while supervised inputs use compatible eyes-open rest. These shifts can reduce transfer.

A future confirmatory protocol should use a genuinely unexamined site, prespecify one comparable medication/session rule and a fixed recording-duration analysis, and obtain blinded expert QC review. Those changes require a new protocol and cannot be used to revise the present test results.

## 8. Descriptive subgroup review

The table shows the selected procedure only. All seven families and exact stratum counts are retained in the aggregate review JSON. Estimates require at least five retained people of each diagnosis. These are post-lock, unadjusted descriptions with substantial small-sample uncertainty; they were not used to alter models.

- Site: Iowa; Stratum: Female; Control / PD: 50 / 50; Cross-site AUROC: 0.642; Within-site AUROC: 0.632
- Site: Iowa; Stratum: Male; Control / PD: 47 / 121; Cross-site AUROC: 0.709; Within-site AUROC: 0.738
- Site: Iowa; Stratum: Age <65; Control / PD: 25 / 43; Cross-site AUROC: 0.751; Within-site AUROC: 0.681
- Site: Iowa; Stratum: Age >=65; Control / PD: 72 / 128; Cross-site AUROC: 0.698; Within-site AUROC: 0.729
- Site: Tel Aviv; Stratum: Female; Control / PD: 11 / 16; Cross-site AUROC: 0.676; Within-site AUROC: 0.699
- Site: Tel Aviv; Stratum: Male; Control / PD: 12 / 63; Cross-site AUROC: 0.681; Within-site AUROC: 0.676
- Site: Tel Aviv; Stratum: Age <65; Control / PD: 16 / 28; Cross-site AUROC: 0.536; Within-site AUROC: 0.670
- Site: Tel Aviv; Stratum: Age >=65; Control / PD: 7 / 51; Cross-site AUROC: 0.868; Within-site AUROC: 0.692
- Site: UC San Diego; Stratum: Female; Control / PD: 8 / 8; Cross-site AUROC: 0.531; Within-site AUROC: 0.719
- Site: UC San Diego; Stratum: Male; Control / PD: 7 / 7; Cross-site AUROC: 0.816; Within-site AUROC: 0.714
- Site: UC San Diego; Stratum: Age <65; Control / PD: 9 / 8; Cross-site AUROC: 0.597; Within-site AUROC: 0.653
- Site: UC San Diego; Stratum: Age >=65; Control / PD: 6 / 7; Cross-site AUROC: 0.690; Within-site AUROC: 0.833
- Site: New Mexico; Stratum: Female; Control / PD: 6 / 8; Cross-site AUROC: 0.521; Within-site AUROC: 0.667
- Site: New Mexico; Stratum: Male; Control / PD: 12 / 14; Cross-site AUROC: 0.750; Within-site AUROC: 0.589
- Site: New Mexico; Stratum: Age <65; Control / PD: 6 / 4; Cross-site AUROC: Not estimated; Within-site AUROC: Not estimated
- Site: New Mexico; Stratum: Age >=65; Control / PD: 12 / 18; Cross-site AUROC: 0.713; Within-site AUROC: 0.574

A subgroup point estimate is not a reliable fairness assessment in cohorts this small. Diagnoses, age, sex, medication schedules, acquisition sites and QC retention are entangled. No claims about protected-group equivalence or clinical safety follow from these slices.

## 9. Verification, reproducibility and next decision

- Verified item: Registry and preprocessing; Evidence: 11,211 planned records reconciled; all 7,988 cache hashes checked
- Verified item: Sources; Evidence: 12,858 selected source files reconciled with receipt SHA-256, byte count and modification time; processed inputs rehashed during preparation
- Verified item: All completed fits; Evidence: 620 fit identities and weight hashes verified; inner validation AUC recalculated; outer selection and thresholds checked
- Verified item: Leakage boundaries; Evidence: Participant/site separation, auxiliary exclusion and one out-of-fold observation per person checked
- Verified item: Evaluation; Evidence: All outer AUROC, PR-AUC, Brier, sensitivity/specificity, balanced accuracy and confusion counts independently reproduced
- Verified item: Frozen records; Evidence: Implementation policy matches; all outer fits predate the outer lock, which predates saved predictions

Recovery fixed legacy BrainVision header encoding, misspelled companion references and one invalid EDF wall-clock second through byte-preserving/temporary-copy handling. Original downloads, numeric samples, completed caches and QC rules were preserved. Unexpected failures halted work instead of silently dropping files.

Python 3.12.14; numpy 2.5.3; scipy 1.18.1; scikit-learn 1.9.1; torch 2.14.0; mne 1.13.2.

### Decision supported by this experiment

Keep the richer feature-based SVMs as strong research comparators. Do not claim that the new pretrained neural model is better: that hypothesis was not supported. Do not replace the model-selection procedure's 0.690 cross-site result with the retrospectively highest 0.763 family score. No direct numerical v2-to-v3 improvement claim is justified because training populations, recording duration, medication aggregation and eligibility changed.

For Myna, the current output remains a cross-sectional research classifier. Symptom tracking needs repeated measurements aligned with clinical or patient-reported symptoms, and multimodal validation requires EEG/audio/movement from the same people. None of those outcomes is established here.

Repository artifacts: frozen protocol, results Markdown/JSON, review diagnostics and model cards. Raw EEG, identities, predictions, checkpoints and fitted weights stay local. V1/v2 and the preliminary PDF are preserved. No clinical deployment or publication claim is made.

## 10. Research references and dataset provenance

This is a focused research synthesis, not a systematic review. Architectural inspiration does not imply reproduction of a published method or performance score.

### BENDR: self-supervised EEG representations

Kostas et al. (2021). https://arxiv.org/abs/2101.12037

### LaBraM: masked EEG modeling

Jiang et al. (ICLR 2024). https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf

### Generalizable EEG classification of PD

Informatics in Medicine Unlocked (2023). https://doi.org/10.1016/j.imu.2023.101352

### EEGNet

Lawhern et al. (2018). https://arxiv.org/abs/1611.08024

### EEG convolutional decoding

Schirrmeister et al. (2017). https://arxiv.org/abs/1703.05051

### Periodic / aperiodic spectral parameterization

Donoghue et al. (2020). https://doi.org/10.1038/s41593-020-00744-x

### Supervised data

https://openneuro.org/datasets/ds008768

https://openneuro.org/datasets/ds007526

https://openneuro.org/datasets/ds003490

https://openneuro.org/datasets/ds002778

### Auxiliary data and licenses

OpenNeuro: ds003775, ds004148, ds004504, ds004796, ds005385, ds006036, ds007864 and ds007865; individual source pages and snapshot/license metadata are listed in docs/cohort-sources.json. PhysioNet bigP3BCI: https://physionet.org/content/bigp3bci/1.0.0/ . Source data licenses are separate from software licensing.

Companion documents: PD_PROTOCOL_V3.md; PD_CLASSIFIER_RESEARCH_PLAN.md; PD_PROTOCOL_V2.md; PD_Classifier_Comparison_Report.pdf (preserved preliminary report).
