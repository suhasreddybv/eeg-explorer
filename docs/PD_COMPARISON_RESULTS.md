# Parkinson's EEG: expanded model comparison

Research report | Protocol v2 | 26 September 2026 UTC

Local Apple M4 study. Participant-level classification of diagnosed Parkinson's disease versus study controls. This report does not establish a diagnostic test, disease progression measurement or symptom-tracking product.

## 1. Executive findings

The highest observed primary external AUROC was 0.786 for Spectral-token Transformer, with a 95% participant-bootstrap interval of 0.673 to 0.886. This is a descriptive ranking after evaluating all locked models, not a new model-selection step or proof of superiority.

The spectral regression baseline achieved AUROC 0.643. At its validation-selected threshold, Spectral-token Transformer achieved sensitivity 0.542 and specificity 0.870 on 95 retained external participants. These operating characteristics must be read together with exclusions, uncertainty and the external class prevalence.

The age/sex-only external AUROC was 0.556 and the artifact-only AUROC was 0.548. Strong confound-only discrimination would weaken a claim that EEG models learned disease-specific physiology.

Model rankings changed across sites: the channel CNN led New Mexico ON (AUROC 0.878), while EEGNet led New Mexico OFF (0.886). The Transformer missed 33 of 72 retained PD participants on the primary test at its frozen threshold. Its primary AUROC gain over spectral regression was 0.143, with an exploratory multiplicity-adjusted paired interval of 0.036 to 0.270. This supports a ranking improvement in this retained cohort; it does not establish superiority over all other models or a consistently strong operating point across sites.

All ten planned families/reference models were run separately, plus an artifact confound baseline. All six neural families used the same 18 training-fold tuning fits and three final seeds. The pilot remains separate. Iowa results are exploratory because overlap with the previously examined pilot cannot be resolved; Tel Aviv and New Mexico were reserved until the v2 model lock.

## 2. Archive integrity and cohort construction

The archive contains 41,746 reconciled files totaling 440.59 GB. Archive-wide reconciliation checked receipt identities, file sizes, modification times and source checksum metadata. Selected input signals were rehashed before processing. This is not a second full-content hash pass over all 440 GB.

The 312-person expanded Iowa release replaces older Iowa datasets rather than being added to them. Only the earliest numbered resting visit per person is used. Tel Aviv contributes 144 candidate people; New Mexico contributes 50 people across 75 sessions. Repeated visits, windows and medication states do not count as additional participants. Older overlapping Iowa and New Mexico task releases, UCSD pilot tests, PD-only treatment datasets and unrelated healthy tasks are excluded.

- Cohort: Training; Assigned: 220; Retained: 177; PD: 113; Control: 64; Excluded: 43
- Cohort: Validation; Assigned: 46; Retained: 39; PD: 27; Control: 12; Excluded: 7
- Cohort: Iowa internal test (exploratory); Assigned: 46; Retained: 35; PD: 23; Control: 12; Excluded: 11
- Cohort: Tel Aviv external test (primary); Assigned: 144; Retained: 95; PD: 72; Control: 23; Excluded: 49
- Cohort: New Mexico ON (secondary); Assigned: 50; Retained: 33; PD: 15; Control: 18; Excluded: 17
- Cohort: New Mexico OFF (secondary); Assigned: 50; Retained: 38; PD: 20; Control: 18; Excluded: 12

Coverage is an essential part of the result. A score measured only among clean recordings does not describe everyone recruited. Exclusion rates can differ by diagnosis, age, motor symptoms and site. No excluded person was reassigned to another split or given a substitute session selected for better model performance.

### Demographic balance and exclusions by diagnosis

- Cohort / diagnosis: Training / control; Excluded / assigned: 14/78; Age mean (SD): 68.7 (11.2); Female / male: 35/29; Age missing: 0
- Cohort / diagnosis: Training / PD; Excluded / assigned: 29/142; Age mean (SD): 69.0 (8.2); Female / male: 31/82; Age missing: 0
- Cohort / diagnosis: Validation / control; Excluded / assigned: 4/16; Age mean (SD): 74.7 (8.9); Female / male: 5/7; Age missing: 0
- Cohort / diagnosis: Validation / PD; Excluded / assigned: 3/30; Age mean (SD): 66.6 (7.7); Female / male: 7/20; Age missing: 0
- Cohort / diagnosis: Iowa internal test (exploratory) / control; Excluded / assigned: 4/16; Age mean (SD): 66.6 (9.9); Female / male: 8/4; Age missing: 0
- Cohort / diagnosis: Iowa internal test (exploratory) / PD; Excluded / assigned: 7/30; Age mean (SD): 68.3 (7.4); Female / male: 9/14; Age missing: 0
- Cohort / diagnosis: Tel Aviv external test (primary) / control; Excluded / assigned: 5/28; Age mean (SD): 60.0 (7.9); Female / male: 11/12; Age missing: 0
- Cohort / diagnosis: Tel Aviv external test (primary) / PD; Excluded / assigned: 44/116; Age mean (SD): 65.2 (9.7); Female / male: 14/58; Age missing: 0
- Cohort / diagnosis: New Mexico ON (secondary) / control; Excluded / assigned: 7/25; Age mean (SD): 68.2 (9.9); Female / male: 6/12; Age missing: 0
- Cohort / diagnosis: New Mexico ON (secondary) / PD; Excluded / assigned: 10/25; Age mean (SD): 71.1 (5.1); Female / male: 5/10; Age missing: 0
- Cohort / diagnosis: New Mexico OFF (secondary) / control; Excluded / assigned: 7/25; Age mean (SD): 68.2 (9.9); Female / male: 6/12; Age missing: 0
- Cohort / diagnosis: New Mexico OFF (secondary) / PD; Excluded / assigned: 5/25; Age mean (SD): 70.0 (8.2); Female / male: 8/12; Age missing: 0

Demographics describe retained people; exclusion fractions use all assigned people. In the primary external cohort, QC excluded 44/116 PD participants (37.9%) and 5/28 controls (17.9%). This differential selection is a material limitation even though every model uses the same retained participants.

Window rejection reasons across the selected recordings: abrupt_step: 3,982; amplitude: 2,361; flat_channel: 60; repeated_samples: 60. One window may have multiple reasons, and counts are not counts of unique people. One New Mexico session was excluded for ambiguous eyes-open event timing.

## 3. Methods and safeguards

The frozen pipeline uses 17 shared scalp electrodes, 1-40 Hz filtering, 256 Hz sampling and average reference. Four-second windows are non-overlapping, capped at 60 seconds per person with a 40-second minimum. Fixed engineering artifact rules screen amplitude, jumps, flat channels, repeated samples and source annotations. These heuristics have not been validated by a blinded clinical EEG expert.

New Mexico eyes-open intervals are extracted only from explicit contiguous one-second event labels. Ambiguous timing is excluded. The analysis does not relabel oddball/task periods as resting EEG. Medication ON and OFF endpoints share participants and controls and are not independent replications.

Training participants alone fit scaling, imputation and feature transformations. All models use identical participant-level training folds. Neural hyperparameter search uses six learning-rate/dropout combinations, three folds, up to 100 epochs and patience 15. Three final seeds are averaged with equal weight. Validation determines early stopping and the operating threshold. Every final model is locked before any test signal processing or evaluation.

The comparison contains compact adaptations. Differences from the papers are disclosed in the protocol appendix: graph message passing omits contrastive pretraining and Chebyshev convolution; wavelet CNN uses smaller multichannel scalograms; the Transformer models spectral/entropy electrode tokens. No published score is used as an acceptance target.

## 4. Training and validation performance

- Model: Class-frequency reference; Train AUC: 0.500; Val AUC: 0.500; Threshold: 0.500; Seed val range: n/a
- Model: Spectral logistic regression; Train AUC: 0.782; Val AUC: 0.565; Threshold: 0.546; Seed val range: n/a
- Model: Periodic/aperiodic features; Train AUC: 0.833; Val AUC: 0.741; Threshold: 0.525; Seed val range: n/a
- Model: Age/sex confound baseline; Train AUC: 0.674; Val AUC: 0.735; Threshold: 0.605; Seed val range: n/a
- Model: Artifact confound baseline; Train AUC: 0.558; Val AUC: 0.355; Threshold: 0.473; Seed val range: n/a
- Model: EEGNet adaptation; Train AUC: 0.976; Val AUC: 0.833; Threshold: 0.495; Seed val range: 0.781-0.818
- Model: ShallowConvNet adaptation; Train AUC: 0.972; Val AUC: 0.694; Threshold: 0.626; Seed val range: 0.673-0.707
- Model: Channel CNN adaptation; Train AUC: 1.000; Val AUC: 0.843; Threshold: 0.832; Seed val range: 0.830-0.852
- Model: Wavelet CNN adaptation; Train AUC: 1.000; Val AUC: 0.673; Threshold: 0.867; Seed val range: 0.630-0.682
- Model: Graph attention adaptation; Train AUC: 0.777; Val AUC: 0.728; Threshold: 0.442; Seed val range: 0.713-0.775
- Model: Spectral-token Transformer; Train AUC: 0.843; Val AUC: 0.722; Threshold: 0.708; Seed val range: 0.691-0.738

Training scores are in-sample. Validation scores are selected by early stopping and threshold choice; they are optimistic estimates of generalization. The seed range describes variability on validation only. It is not a confidence interval, and the best seed was not selected on a test.

## 5. Locked test results

### Iowa internal test (exploratory)

Coverage: 35/46; retained PD/control: 23/12.

- Model: Class-frequency reference; AUROC (95% CI): 0.500 (0.500-0.500); Bal. acc.: 0.500; Sens.: 1.000; Spec.: 0.000; Brier: 0.226
- Model: Spectral logistic regression; AUROC (95% CI): 0.750 (0.585-0.909); Bal. acc.: 0.638; Sens.: 0.609; Spec.: 0.667; Brier: 0.193
- Model: Periodic/aperiodic features; AUROC (95% CI): 0.783 (0.630-0.917); Bal. acc.: 0.639; Sens.: 0.696; Spec.: 0.583; Brier: 0.180
- Model: Age/sex confound baseline; AUROC (95% CI): 0.621 (0.424-0.816); Bal. acc.: 0.594; Sens.: 0.522; Spec.: 0.667; Brier: 0.243
- Model: Artifact confound baseline; AUROC (95% CI): 0.627 (0.429-0.806); Bal. acc.: 0.621; Sens.: 0.826; Spec.: 0.417; Brier: 0.234
- Model: EEGNet adaptation; AUROC (95% CI): 0.848 (0.710-0.960); Bal. acc.: 0.705; Sens.: 0.826; Spec.: 0.583; Brier: 0.153
- Model: ShallowConvNet adaptation; AUROC (95% CI): 0.822 (0.665-0.946); Bal. acc.: 0.683; Sens.: 0.783; Spec.: 0.583; Brier: 0.167
- Model: Channel CNN adaptation; AUROC (95% CI): 0.844 (0.725-0.953); Bal. acc.: 0.786; Sens.: 0.739; Spec.: 0.833; Brier: 0.174
- Model: Wavelet CNN adaptation; AUROC (95% CI): 0.667 (0.493-0.841); Bal. acc.: 0.594; Sens.: 0.522; Spec.: 0.667; Brier: 0.263
- Model: Graph attention adaptation; AUROC (95% CI): 0.819 (0.668-0.946); Bal. acc.: 0.745; Sens.: 0.739; Spec.: 0.750; Brier: 0.177
- Model: Spectral-token Transformer; AUROC (95% CI): 0.812 (0.652-0.948); Bal. acc.: 0.717; Sens.: 0.435; Spec.: 1.000; Brier: 0.173

AUROC describes ranking across thresholds. Sensitivity and specificity use thresholds fixed on validation; Brier is mean squared probability error (lower is better). Intervals are participant-level, not window-level. Small control groups can leave specificity highly uncertain.

Class-frequency reference: TN 0, FP 12, FN 0, TP 23; average precision 0.657; PD prevalence 0.657; sensitivity 95% Wilson CI 0.857-1.000; specificity CI 0.000-0.242.

Spectral logistic regression: TN 8, FP 4, FN 9, TP 14; average precision 0.847; PD prevalence 0.657; sensitivity 95% Wilson CI 0.408-0.778; specificity CI 0.391-0.862.

Periodic/aperiodic features: TN 7, FP 5, FN 7, TP 16; average precision 0.886; PD prevalence 0.657; sensitivity 95% Wilson CI 0.491-0.844; specificity CI 0.320-0.807.

Age/sex confound baseline: TN 8, FP 4, FN 11, TP 12; average precision 0.742; PD prevalence 0.657; sensitivity 95% Wilson CI 0.330-0.708; specificity CI 0.391-0.862.

Artifact confound baseline: TN 5, FP 7, FN 4, TP 19; average precision 0.759; PD prevalence 0.657; sensitivity 95% Wilson CI 0.629-0.930; specificity CI 0.193-0.680.

EEGNet adaptation: TN 7, FP 5, FN 4, TP 19; average precision 0.929; PD prevalence 0.657; sensitivity 95% Wilson CI 0.629-0.930; specificity CI 0.320-0.807.

ShallowConvNet adaptation: TN 7, FP 5, FN 5, TP 18; average precision 0.922; PD prevalence 0.657; sensitivity 95% Wilson CI 0.581-0.903; specificity CI 0.320-0.807.

Channel CNN adaptation: TN 10, FP 2, FN 6, TP 17; average precision 0.930; PD prevalence 0.657; sensitivity 95% Wilson CI 0.535-0.875; specificity CI 0.552-0.953.

Wavelet CNN adaptation: TN 8, FP 4, FN 11, TP 12; average precision 0.819; PD prevalence 0.657; sensitivity 95% Wilson CI 0.330-0.708; specificity CI 0.391-0.862.

Graph attention adaptation: TN 9, FP 3, FN 6, TP 17; average precision 0.923; PD prevalence 0.657; sensitivity 95% Wilson CI 0.535-0.875; specificity CI 0.468-0.911.

Spectral-token Transformer: TN 12, FP 0, FN 13, TP 10; average precision 0.905; PD prevalence 0.657; sensitivity 95% Wilson CI 0.256-0.632; specificity CI 0.758-1.000.

### Paired comparison with spectral regression

- Model: Periodic/aperiodic features; AUC difference: 0.033; Paired 95% CI: -0.083 to 0.156; Multiplicity-adjusted CI: -0.134 to 0.228
- Model: Age/sex confound baseline; AUC difference: -0.129; Paired 95% CI: -0.388 to 0.127; Multiplicity-adjusted CI: -0.475 to 0.236
- Model: Artifact confound baseline; AUC difference: -0.123; Paired 95% CI: -0.308 to 0.054; Multiplicity-adjusted CI: -0.391 to 0.138
- Model: Class-frequency reference; AUC difference: -0.250; Paired 95% CI: -0.409 to -0.065; Multiplicity-adjusted CI: -0.446 to 0.015
- Model: EEGNet adaptation; AUC difference: 0.098; Paired 95% CI: -0.040 to 0.261; Multiplicity-adjusted CI: -0.101 to 0.326
- Model: ShallowConvNet adaptation; AUC difference: 0.072; Paired 95% CI: -0.058 to 0.221; Multiplicity-adjusted CI: -0.116 to 0.279
- Model: Channel CNN adaptation; AUC difference: 0.094; Paired 95% CI: -0.069 to 0.268; Multiplicity-adjusted CI: -0.127 to 0.355
- Model: Wavelet CNN adaptation; AUC difference: -0.083; Paired 95% CI: -0.236 to 0.043; Multiplicity-adjusted CI: -0.293 to 0.112
- Model: Graph attention adaptation; AUC difference: 0.069; Paired 95% CI: -0.080 to 0.214; Multiplicity-adjusted CI: -0.138 to 0.297
- Model: Spectral-token Transformer; AUC difference: 0.062; Paired 95% CI: -0.051 to 0.181; Multiplicity-adjusted CI: -0.098 to 0.228

Paired bootstrap resamples the same participants for every model. Adjusted intervals use Bonferroni across the ten comparisons with spectral regression within this endpoint. They do not adjust across every site and subgroup. With 2,000 resamples, extreme adjusted endpoints are approximate; these are exploratory comparisons, not confirmatory significance claims.

### Tel Aviv external test (primary)

Coverage: 95/144; retained PD/control: 72/23.

- Model: Class-frequency reference; AUROC (95% CI): 0.500 (0.500-0.500); Bal. acc.: 0.500; Sens.: 1.000; Spec.: 0.000; Brier: 0.198
- Model: Spectral logistic regression; AUROC (95% CI): 0.643 (0.512-0.764); Bal. acc.: 0.590; Sens.: 0.875; Spec.: 0.304; Brier: 0.176
- Model: Periodic/aperiodic features; AUROC (95% CI): 0.757 (0.667-0.835); Bal. acc.: 0.642; Sens.: 0.806; Spec.: 0.478; Brier: 0.178
- Model: Age/sex confound baseline; AUROC (95% CI): 0.556 (0.405-0.699); Bal. acc.: 0.579; Sens.: 0.681; Spec.: 0.478; Brier: 0.216
- Model: Artifact confound baseline; AUROC (95% CI): 0.548 (0.414-0.681); Bal. acc.: 0.553; Sens.: 0.889; Spec.: 0.217; Brier: 0.236
- Model: EEGNet adaptation; AUROC (95% CI): 0.723 (0.600-0.835); Bal. acc.: 0.642; Sens.: 0.806; Spec.: 0.478; Brier: 0.186
- Model: ShallowConvNet adaptation; AUROC (95% CI): 0.679 (0.547-0.795); Bal. acc.: 0.580; Sens.: 0.639; Spec.: 0.522; Brier: 0.192
- Model: Channel CNN adaptation; AUROC (95% CI): 0.740 (0.623-0.842); Bal. acc.: 0.685; Sens.: 0.500; Spec.: 0.870; Brier: 0.174
- Model: Wavelet CNN adaptation; AUROC (95% CI): 0.707 (0.577-0.819); Bal. acc.: 0.642; Sens.: 0.458; Spec.: 0.826; Brier: 0.193
- Model: Graph attention adaptation; AUROC (95% CI): 0.693 (0.566-0.797); Bal. acc.: 0.631; Sens.: 0.958; Spec.: 0.304; Brier: 0.178
- Model: Spectral-token Transformer; AUROC (95% CI): 0.786 (0.673-0.886); Bal. acc.: 0.706; Sens.: 0.542; Spec.: 0.870; Brier: 0.161

AUROC describes ranking across thresholds. Sensitivity and specificity use thresholds fixed on validation; Brier is mean squared probability error (lower is better). Intervals are participant-level, not window-level. Small control groups can leave specificity highly uncertain.

Class-frequency reference: TN 0, FP 23, FN 0, TP 72; average precision 0.758; PD prevalence 0.758; sensitivity 95% Wilson CI 0.949-1.000; specificity CI 0.000-0.143.

Spectral logistic regression: TN 7, FP 16, FN 9, TP 63; average precision 0.851; PD prevalence 0.758; sensitivity 95% Wilson CI 0.779-0.933; specificity CI 0.156-0.509.

Periodic/aperiodic features: TN 11, FP 12, FN 14, TP 58; average precision 0.920; PD prevalence 0.758; sensitivity 95% Wilson CI 0.700-0.880; specificity CI 0.292-0.670.

Age/sex confound baseline: TN 11, FP 12, FN 23, TP 49; average precision 0.780; PD prevalence 0.758; sensitivity 95% Wilson CI 0.566-0.777; specificity CI 0.292-0.670.

Artifact confound baseline: TN 5, FP 18, FN 8, TP 64; average precision 0.787; PD prevalence 0.758; sensitivity 95% Wilson CI 0.796-0.943; specificity CI 0.097-0.419.

EEGNet adaptation: TN 11, FP 12, FN 14, TP 58; average precision 0.893; PD prevalence 0.758; sensitivity 95% Wilson CI 0.700-0.880; specificity CI 0.292-0.670.

ShallowConvNet adaptation: TN 12, FP 11, FN 26, TP 46; average precision 0.875; PD prevalence 0.758; sensitivity 95% Wilson CI 0.524-0.740; specificity CI 0.330-0.708.

Channel CNN adaptation: TN 20, FP 3, FN 36, TP 36; average precision 0.896; PD prevalence 0.758; sensitivity 95% Wilson CI 0.387-0.613; specificity CI 0.679-0.955.

Wavelet CNN adaptation: TN 19, FP 4, FN 39, TP 33; average precision 0.883; PD prevalence 0.758; sensitivity 95% Wilson CI 0.348-0.573; specificity CI 0.629-0.930.

Graph attention adaptation: TN 7, FP 16, FN 3, TP 69; average precision 0.879; PD prevalence 0.758; sensitivity 95% Wilson CI 0.885-0.986; specificity CI 0.156-0.509.

Spectral-token Transformer: TN 20, FP 3, FN 33, TP 39; average precision 0.917; PD prevalence 0.758; sensitivity 95% Wilson CI 0.427-0.652; specificity CI 0.679-0.955.

### Paired comparison with spectral regression

- Model: Periodic/aperiodic features; AUC difference: 0.115; Paired 95% CI: 0.021 to 0.217; Multiplicity-adjusted CI: -0.018 to 0.263
- Model: Age/sex confound baseline; AUC difference: -0.087; Paired 95% CI: -0.277 to 0.121; Multiplicity-adjusted CI: -0.353 to 0.198
- Model: Artifact confound baseline; AUC difference: -0.095; Paired 95% CI: -0.205 to 0.004; Multiplicity-adjusted CI: -0.264 to 0.042
- Model: Class-frequency reference; AUC difference: -0.143; Paired 95% CI: -0.268 to -0.010; Multiplicity-adjusted CI: -0.329 to 0.045
- Model: EEGNet adaptation; AUC difference: 0.080; Paired 95% CI: -0.020 to 0.184; Multiplicity-adjusted CI: -0.056 to 0.242
- Model: ShallowConvNet adaptation; AUC difference: 0.037; Paired 95% CI: -0.039 to 0.107; Multiplicity-adjusted CI: -0.082 to 0.141
- Model: Channel CNN adaptation; AUC difference: 0.098; Paired 95% CI: -0.007 to 0.204; Multiplicity-adjusted CI: -0.043 to 0.273
- Model: Wavelet CNN adaptation; AUC difference: 0.064; Paired 95% CI: -0.065 to 0.203; Multiplicity-adjusted CI: -0.114 to 0.269
- Model: Graph attention adaptation; AUC difference: 0.050; Paired 95% CI: -0.011 to 0.115; Multiplicity-adjusted CI: -0.042 to 0.136
- Model: Spectral-token Transformer; AUC difference: 0.143; Paired 95% CI: 0.062 to 0.237; Multiplicity-adjusted CI: 0.036 to 0.270

Paired bootstrap resamples the same participants for every model. Adjusted intervals use Bonferroni across the ten comparisons with spectral regression within this endpoint. They do not adjust across every site and subgroup. With 2,000 resamples, extreme adjusted endpoints are approximate; these are exploratory comparisons, not confirmatory significance claims.

### New Mexico ON (secondary)

Coverage: 33/50; retained PD/control: 15/18.

- Model: Class-frequency reference; AUROC (95% CI): 0.500 (0.500-0.500); Bal. acc.: 0.500; Sens.: 1.000; Spec.: 0.000; Brier: 0.282
- Model: Spectral logistic regression; AUROC (95% CI): 0.737 (0.552-0.889); Bal. acc.: 0.667; Sens.: 0.667; Spec.: 0.667; Brier: 0.213
- Model: Periodic/aperiodic features; AUROC (95% CI): 0.715 (0.519-0.887); Bal. acc.: 0.667; Sens.: 0.667; Spec.: 0.667; Brier: 0.210
- Model: Age/sex confound baseline; AUROC (95% CI): 0.509 (0.304-0.721); Bal. acc.: 0.544; Sens.: 0.533; Spec.: 0.556; Brier: 0.273
- Model: Artifact confound baseline; AUROC (95% CI): 0.537 (0.326-0.726); Bal. acc.: 0.572; Sens.: 0.867; Spec.: 0.278; Brier: 0.252
- Model: EEGNet adaptation; AUROC (95% CI): 0.867 (0.713-0.978); Bal. acc.: 0.878; Sens.: 0.867; Spec.: 0.889; Brier: 0.150
- Model: ShallowConvNet adaptation; AUROC (95% CI): 0.859 (0.694-0.976); Bal. acc.: 0.711; Sens.: 0.533; Spec.: 0.889; Brier: 0.156
- Model: Channel CNN adaptation; AUROC (95% CI): 0.878 (0.744-0.989); Bal. acc.: 0.772; Sens.: 0.600; Spec.: 0.944; Brier: 0.130
- Model: Wavelet CNN adaptation; AUROC (95% CI): 0.726 (0.522-0.885); Bal. acc.: 0.633; Sens.: 0.267; Spec.: 1.000; Brier: 0.238
- Model: Graph attention adaptation; AUROC (95% CI): 0.822 (0.659-0.943); Bal. acc.: 0.722; Sens.: 0.667; Spec.: 0.778; Brier: 0.182
- Model: Spectral-token Transformer; AUROC (95% CI): 0.741 (0.548-0.887); Bal. acc.: 0.633; Sens.: 0.267; Spec.: 1.000; Brier: 0.199

AUROC describes ranking across thresholds. Sensitivity and specificity use thresholds fixed on validation; Brier is mean squared probability error (lower is better). Intervals are participant-level, not window-level. Small control groups can leave specificity highly uncertain.

Class-frequency reference: TN 0, FP 18, FN 0, TP 15; average precision 0.455; PD prevalence 0.455; sensitivity 95% Wilson CI 0.796-1.000; specificity CI 0.000-0.176.

Spectral logistic regression: TN 12, FP 6, FN 5, TP 10; average precision 0.761; PD prevalence 0.455; sensitivity 95% Wilson CI 0.417-0.848; specificity CI 0.437-0.837.

Periodic/aperiodic features: TN 12, FP 6, FN 5, TP 10; average precision 0.762; PD prevalence 0.455; sensitivity 95% Wilson CI 0.417-0.848; specificity CI 0.437-0.837.

Age/sex confound baseline: TN 10, FP 8, FN 7, TP 8; average precision 0.509; PD prevalence 0.455; sensitivity 95% Wilson CI 0.301-0.752; specificity CI 0.337-0.754.

Artifact confound baseline: TN 5, FP 13, FN 2, TP 13; average precision 0.484; PD prevalence 0.455; sensitivity 95% Wilson CI 0.621-0.963; specificity CI 0.125-0.509.

EEGNet adaptation: TN 16, FP 2, FN 2, TP 13; average precision 0.856; PD prevalence 0.455; sensitivity 95% Wilson CI 0.621-0.963; specificity CI 0.672-0.969.

ShallowConvNet adaptation: TN 16, FP 2, FN 7, TP 8; average precision 0.873; PD prevalence 0.455; sensitivity 95% Wilson CI 0.301-0.752; specificity CI 0.672-0.969.

Channel CNN adaptation: TN 17, FP 1, FN 6, TP 9; average precision 0.825; PD prevalence 0.455; sensitivity 95% Wilson CI 0.357-0.802; specificity CI 0.742-0.990.

Wavelet CNN adaptation: TN 18, FP 0, FN 11, TP 4; average precision 0.756; PD prevalence 0.455; sensitivity 95% Wilson CI 0.109-0.520; specificity CI 0.824-1.000.

Graph attention adaptation: TN 14, FP 4, FN 5, TP 10; average precision 0.849; PD prevalence 0.455; sensitivity 95% Wilson CI 0.417-0.848; specificity CI 0.548-0.910.

Spectral-token Transformer: TN 18, FP 0, FN 11, TP 4; average precision 0.776; PD prevalence 0.455; sensitivity 95% Wilson CI 0.109-0.520; specificity CI 0.824-1.000.

### Paired comparison with spectral regression

- Model: Periodic/aperiodic features; AUC difference: -0.022; Paired 95% CI: -0.122 to 0.070; Multiplicity-adjusted CI: -0.174 to 0.122
- Model: Age/sex confound baseline; AUC difference: -0.228; Paired 95% CI: -0.480 to 0.039; Multiplicity-adjusted CI: -0.546 to 0.120
- Model: Artifact confound baseline; AUC difference: -0.200; Paired 95% CI: -0.441 to 0.048; Multiplicity-adjusted CI: -0.519 to 0.148
- Model: Class-frequency reference; AUC difference: -0.237; Paired 95% CI: -0.385 to -0.048; Multiplicity-adjusted CI: -0.430 to 0.074
- Model: EEGNet adaptation; AUC difference: 0.130; Paired 95% CI: -0.037 to 0.326; Multiplicity-adjusted CI: -0.111 to 0.419
- Model: ShallowConvNet adaptation; AUC difference: 0.122; Paired 95% CI: -0.019 to 0.281; Multiplicity-adjusted CI: -0.089 to 0.378
- Model: Channel CNN adaptation; AUC difference: 0.141; Paired 95% CI: -0.030 to 0.326; Multiplicity-adjusted CI: -0.119 to 0.426
- Model: Wavelet CNN adaptation; AUC difference: -0.011; Paired 95% CI: -0.193 to 0.156; Multiplicity-adjusted CI: -0.274 to 0.263
- Model: Graph attention adaptation; AUC difference: 0.085; Paired 95% CI: -0.011 to 0.196; Multiplicity-adjusted CI: -0.052 to 0.248
- Model: Spectral-token Transformer; AUC difference: 0.004; Paired 95% CI: -0.093 to 0.100; Multiplicity-adjusted CI: -0.137 to 0.148

Paired bootstrap resamples the same participants for every model. Adjusted intervals use Bonferroni across the ten comparisons with spectral regression within this endpoint. They do not adjust across every site and subgroup. With 2,000 resamples, extreme adjusted endpoints are approximate; these are exploratory comparisons, not confirmatory significance claims.

### New Mexico OFF (secondary)

Coverage: 38/50; retained PD/control: 20/18.

- Model: Class-frequency reference; AUROC (95% CI): 0.500 (0.500-0.500); Bal. acc.: 0.500; Sens.: 1.000; Spec.: 0.000; Brier: 0.262
- Model: Spectral logistic regression; AUROC (95% CI): 0.728 (0.554-0.871); Bal. acc.: 0.708; Sens.: 0.750; Spec.: 0.667; Brier: 0.213
- Model: Periodic/aperiodic features; AUROC (95% CI): 0.739 (0.556-0.892); Bal. acc.: 0.658; Sens.: 0.650; Spec.: 0.667; Brier: 0.209
- Model: Age/sex confound baseline; AUROC (95% CI): 0.465 (0.293-0.659); Bal. acc.: 0.503; Sens.: 0.450; Spec.: 0.556; Brier: 0.279
- Model: Artifact confound baseline; AUROC (95% CI): 0.558 (0.349-0.736); Bal. acc.: 0.539; Sens.: 0.800; Spec.: 0.278; Brier: 0.245
- Model: EEGNet adaptation; AUROC (95% CI): 0.886 (0.767-0.982); Bal. acc.: 0.794; Sens.: 0.700; Spec.: 0.889; Brier: 0.160
- Model: ShallowConvNet adaptation; AUROC (95% CI): 0.825 (0.682-0.947); Bal. acc.: 0.719; Sens.: 0.550; Spec.: 0.889; Brier: 0.171
- Model: Channel CNN adaptation; AUROC (95% CI): 0.825 (0.687-0.949); Bal. acc.: 0.697; Sens.: 0.450; Spec.: 0.944; Brier: 0.166
- Model: Wavelet CNN adaptation; AUROC (95% CI): 0.736 (0.583-0.881); Bal. acc.: 0.625; Sens.: 0.250; Spec.: 1.000; Brier: 0.218
- Model: Graph attention adaptation; AUROC (95% CI): 0.767 (0.605-0.907); Bal. acc.: 0.664; Sens.: 0.550; Spec.: 0.778; Brier: 0.209
- Model: Spectral-token Transformer; AUROC (95% CI): 0.736 (0.574-0.888); Bal. acc.: 0.575; Sens.: 0.150; Spec.: 1.000; Brier: 0.203

AUROC describes ranking across thresholds. Sensitivity and specificity use thresholds fixed on validation; Brier is mean squared probability error (lower is better). Intervals are participant-level, not window-level. Small control groups can leave specificity highly uncertain.

Class-frequency reference: TN 0, FP 18, FN 0, TP 20; average precision 0.526; PD prevalence 0.526; sensitivity 95% Wilson CI 0.839-1.000; specificity CI 0.000-0.176.

Spectral logistic regression: TN 12, FP 6, FN 5, TP 15; average precision 0.785; PD prevalence 0.526; sensitivity 95% Wilson CI 0.531-0.888; specificity CI 0.437-0.837.

Periodic/aperiodic features: TN 12, FP 6, FN 7, TP 13; average precision 0.759; PD prevalence 0.526; sensitivity 95% Wilson CI 0.433-0.819; specificity CI 0.437-0.837.

Age/sex confound baseline: TN 10, FP 8, FN 11, TP 9; average precision 0.541; PD prevalence 0.526; sensitivity 95% Wilson CI 0.258-0.658; specificity CI 0.337-0.754.

Artifact confound baseline: TN 5, FP 13, FN 4, TP 16; average precision 0.626; PD prevalence 0.526; sensitivity 95% Wilson CI 0.584-0.919; specificity CI 0.125-0.509.

EEGNet adaptation: TN 16, FP 2, FN 6, TP 14; average precision 0.881; PD prevalence 0.526; sensitivity 95% Wilson CI 0.481-0.855; specificity CI 0.672-0.969.

ShallowConvNet adaptation: TN 16, FP 2, FN 9, TP 11; average precision 0.871; PD prevalence 0.526; sensitivity 95% Wilson CI 0.342-0.742; specificity CI 0.672-0.969.

Channel CNN adaptation: TN 17, FP 1, FN 11, TP 9; average precision 0.828; PD prevalence 0.526; sensitivity 95% Wilson CI 0.258-0.658; specificity CI 0.742-0.990.

Wavelet CNN adaptation: TN 18, FP 0, FN 15, TP 5; average precision 0.779; PD prevalence 0.526; sensitivity 95% Wilson CI 0.112-0.469; specificity CI 0.824-1.000.

Graph attention adaptation: TN 14, FP 4, FN 9, TP 11; average precision 0.833; PD prevalence 0.526; sensitivity 95% Wilson CI 0.342-0.742; specificity CI 0.548-0.910.

Spectral-token Transformer: TN 18, FP 0, FN 17, TP 3; average precision 0.793; PD prevalence 0.526; sensitivity 95% Wilson CI 0.052-0.360; specificity CI 0.824-1.000.

### Paired comparison with spectral regression

- Model: Periodic/aperiodic features; AUC difference: 0.011; Paired 95% CI: -0.111 to 0.128; Multiplicity-adjusted CI: -0.192 to 0.172
- Model: Age/sex confound baseline; AUC difference: -0.263; Paired 95% CI: -0.512 to -0.022; Multiplicity-adjusted CI: -0.597 to 0.092
- Model: Artifact confound baseline; AUC difference: -0.169; Paired 95% CI: -0.403 to 0.056; Multiplicity-adjusted CI: -0.494 to 0.139
- Model: Class-frequency reference; AUC difference: -0.228; Paired 95% CI: -0.378 to -0.064; Multiplicity-adjusted CI: -0.417 to 0.011
- Model: EEGNet adaptation; AUC difference: 0.158; Paired 95% CI: -0.025 to 0.342; Multiplicity-adjusted CI: -0.100 to 0.419
- Model: ShallowConvNet adaptation; AUC difference: 0.097; Paired 95% CI: -0.028 to 0.231; Multiplicity-adjusted CI: -0.078 to 0.286
- Model: Channel CNN adaptation; AUC difference: 0.097; Paired 95% CI: -0.053 to 0.242; Multiplicity-adjusted CI: -0.119 to 0.308
- Model: Wavelet CNN adaptation; AUC difference: 0.008; Paired 95% CI: -0.164 to 0.172; Multiplicity-adjusted CI: -0.239 to 0.244
- Model: Graph attention adaptation; AUC difference: 0.039; Paired 95% CI: -0.075 to 0.145; Multiplicity-adjusted CI: -0.131 to 0.211
- Model: Spectral-token Transformer; AUC difference: 0.008; Paired 95% CI: -0.086 to 0.100; Multiplicity-adjusted CI: -0.136 to 0.150

Paired bootstrap resamples the same participants for every model. Adjusted intervals use Bonferroni across the ten comparisons with spectral regression within this endpoint. They do not adjust across every site and subgroup. With 2,000 resamples, extreme adjusted endpoints are approximate; these are exploratory comparisons, not confirmatory significance claims.

## 6. Negative controls and subgroup uncertainty

The spectral baseline's selected training-fold AUROC was 0.705. Across 100 participant-label permutations, repeating regularization selection, the null mean was 0.539, its central 95% range was 0.421-0.635, and the Monte Carlo p-value was 0.0099. This checks one baseline pipeline; it neither validates all neural models nor proves disease-specific physiology.

Age (<65 / >=65) and sex strata were fixed for descriptive post-lock reporting. A subgroup needs at least five people in each class to estimate metrics; otherwise it is explicitly omitted. These slices have no multiplicity correction and were never used to retune models.

### Iowa internal test (exploratory)

- Stratum: Age below 65; Control / PD: 4 / 6; AUROC by model / reason not estimated: Fewer than five people in at least one class
- Stratum: Age 65 or older; Control / PD: 8 / 17; AUROC by model / reason not estimated: Spectral logistic regression 0.743; Periodic/aperiodic features 0.809; Age/sex confound baseline 0.717; Artifact confound baseline 0.699; Class-frequency reference 0.500; EEGNet adaptation 0.809; ShallowConvNet adaptation 0.794; Channel CNN adaptation 0.875; Wavelet CNN adaptation 0.610; Graph attention adaptation 0.846; Spectral-token Transformer 0.868
- Stratum: Female; Control / PD: 8 / 9; AUROC by model / reason not estimated: Spectral logistic regression 0.750; Periodic/aperiodic features 0.722; Age/sex confound baseline 0.493; Artifact confound baseline 0.708; Class-frequency reference 0.500; EEGNet adaptation 0.778; ShallowConvNet adaptation 0.708; Channel CNN adaptation 0.750; Wavelet CNN adaptation 0.472; Graph attention adaptation 0.792; Spectral-token Transformer 0.792
- Stratum: Male; Control / PD: 4 / 14; AUROC by model / reason not estimated: Fewer than five people in at least one class

### Tel Aviv external test (primary)

- Stratum: Age below 65; Control / PD: 16 / 26; AUROC by model / reason not estimated: Spectral logistic regression 0.591; Periodic/aperiodic features 0.649; Age/sex confound baseline 0.684; Artifact confound baseline 0.502; Class-frequency reference 0.500; EEGNet adaptation 0.550; ShallowConvNet adaptation 0.572; Channel CNN adaptation 0.661; Wavelet CNN adaptation 0.659; Graph attention adaptation 0.587; Spectral-token Transformer 0.697
- Stratum: Age 65 or older; Control / PD: 7 / 46; AUROC by model / reason not estimated: Spectral logistic regression 0.748; Periodic/aperiodic features 0.829; Age/sex confound baseline 0.441; Artifact confound baseline 0.683; Class-frequency reference 0.500; EEGNet adaptation 0.901; ShallowConvNet adaptation 0.801; Channel CNN adaptation 0.823; Wavelet CNN adaptation 0.767; Graph attention adaptation 0.780; Spectral-token Transformer 0.835
- Stratum: Female; Control / PD: 11 / 14; AUROC by model / reason not estimated: Spectral logistic regression 0.656; Periodic/aperiodic features 0.766; Age/sex confound baseline 0.302; Artifact confound baseline 0.545; Class-frequency reference 0.500; EEGNet adaptation 0.597; ShallowConvNet adaptation 0.662; Channel CNN adaptation 0.643; Wavelet CNN adaptation 0.708; Graph attention adaptation 0.649; Spectral-token Transformer 0.721
- Stratum: Male; Control / PD: 12 / 58; AUROC by model / reason not estimated: Spectral logistic regression 0.635; Periodic/aperiodic features 0.746; Age/sex confound baseline 0.339; Artifact confound baseline 0.555; Class-frequency reference 0.500; EEGNet adaptation 0.786; ShallowConvNet adaptation 0.675; Channel CNN adaptation 0.805; Wavelet CNN adaptation 0.703; Graph attention adaptation 0.687; Spectral-token Transformer 0.815

### New Mexico ON (secondary)

- Stratum: Age below 65; Control / PD: 6 / 2; AUROC by model / reason not estimated: Fewer than five people in at least one class
- Stratum: Age 65 or older; Control / PD: 12 / 13; AUROC by model / reason not estimated: Spectral logistic regression 0.840; Periodic/aperiodic features 0.782; Age/sex confound baseline 0.583; Artifact confound baseline 0.519; Class-frequency reference 0.500; EEGNet adaptation 0.942; ShallowConvNet adaptation 0.942; Channel CNN adaptation 0.994; Wavelet CNN adaptation 0.827; Graph attention adaptation 0.904; Spectral-token Transformer 0.833
- Stratum: Female; Control / PD: 6 / 5; AUROC by model / reason not estimated: Spectral logistic regression 0.500; Periodic/aperiodic features 0.600; Age/sex confound baseline 0.250; Artifact confound baseline 0.433; Class-frequency reference 0.500; EEGNet adaptation 0.800; ShallowConvNet adaptation 0.733; Channel CNN adaptation 0.800; Wavelet CNN adaptation 0.433; Graph attention adaptation 0.833; Spectral-token Transformer 0.500
- Stratum: Male; Control / PD: 12 / 10; AUROC by model / reason not estimated: Spectral logistic regression 0.825; Periodic/aperiodic features 0.758; Age/sex confound baseline 0.583; Artifact confound baseline 0.608; Class-frequency reference 0.500; EEGNet adaptation 0.883; ShallowConvNet adaptation 0.925; Channel CNN adaptation 0.883; Wavelet CNN adaptation 0.850; Graph attention adaptation 0.833; Spectral-token Transformer 0.825

### New Mexico OFF (secondary)

- Stratum: Age below 65; Control / PD: 6 / 4; AUROC by model / reason not estimated: Fewer than five people in at least one class
- Stratum: Age 65 or older; Control / PD: 12 / 16; AUROC by model / reason not estimated: Spectral logistic regression 0.760; Periodic/aperiodic features 0.734; Age/sex confound baseline 0.500; Artifact confound baseline 0.557; Class-frequency reference 0.500; EEGNet adaptation 0.911; ShallowConvNet adaptation 0.859; Channel CNN adaptation 0.880; Wavelet CNN adaptation 0.807; Graph attention adaptation 0.792; Spectral-token Transformer 0.750
- Stratum: Female; Control / PD: 6 / 8; AUROC by model / reason not estimated: Spectral logistic regression 0.562; Periodic/aperiodic features 0.604; Age/sex confound baseline 0.302; Artifact confound baseline 0.500; Class-frequency reference 0.500; EEGNet adaptation 0.938; ShallowConvNet adaptation 0.729; Channel CNN adaptation 0.792; Wavelet CNN adaptation 0.688; Graph attention adaptation 0.708; Spectral-token Transformer 0.583
- Stratum: Male; Control / PD: 12 / 12; AUROC by model / reason not estimated: Spectral logistic regression 0.819; Periodic/aperiodic features 0.806; Age/sex confound baseline 0.562; Artifact confound baseline 0.597; Class-frequency reference 0.500; EEGNet adaptation 0.868; ShallowConvNet adaptation 0.847; Channel CNN adaptation 0.840; Wavelet CNN adaptation 0.757; Graph attention adaptation 0.840; Spectral-token Transformer 0.806

## 7. Local compute and reproducibility

- Model: EEGNet adaptation; Parameters: 1905; Fits: 21; Fit minutes: 17.6; Epoch range: 17-50
- Model: ShallowConvNet adaptation; Parameters: 7381; Fits: 21; Fit minutes: 11.0; Epoch range: 17-38
- Model: Channel CNN adaptation; Parameters: 52764; Fits: 21; Fit minutes: 11.5; Epoch range: 16-43
- Model: Wavelet CNN adaptation; Parameters: 25665; Fits: 21; Fit minutes: 5.5; Epoch range: 16-39
- Model: Graph attention adaptation; Parameters: 13473; Fits: 21; Fit minutes: 15.6; Epoch range: 17-52
- Model: Spectral-token Transformer; Parameters: 17857; Fits: 21; Fit minutes: 9.0; Epoch range: 16-48

Device: mps; macOS-15.7.9-arm64-arm-64bit. Conservative neural campaign forecast: 9.94 hours, using maximum epochs and 25% margin. Recorded neural fit time: 1.17 hours; this excludes separate preprocessing, representation creation, benchmarks, reporting and test/bootstrap evaluation. The six architectures use different parameter counts; identical tuning counts do not imply identical compute.

The automated campaign, including fitting, test preprocessing, evaluation, subgroup diagnostics and initial report generation, completed in 78.3 minutes. Development preprocessing, the benchmark, permutation check and later report review happened separately.

Registry fingerprint: 9aab6b52b55cb2bc36f335fb41e73869e01d60bceec0a0cda1ca648eee91cb5c

Software: Python 3.12.14; torch 2.14.0; numpy 2.5.3; scipy 1.18.1; mne 1.13.2; scikit-learn 1.9.1; psutil 7.2.2; httpx 0.28.1.

Apple M4, 10 CPU cores, 10 GPU cores, 16 GB unified memory. Peak unified/GPU memory was not instrumented; bounded per-recording preprocessing and minibatches were used.

Final model-lock fingerprint: d0d50c7f684da8eb2e281b167e3b449f52b1cfadd4f72ed5545d3c786089a7a4

Reproduce with the repository environment and verified archive. Commands: python -m pd_pipeline.cohort_v2 build; python -m pd_pipeline.cohort_v2 prepare; python -m pd_pipeline.compare_v2 setup; python -m pd_pipeline.compare_v2 benchmark; python -m pd_pipeline.compare_v2 campaign; python -m pd_pipeline.cohort_v2 test-prepare; python -m pd_pipeline.compare_v2 evaluate. See LOCAL_TRAINING.md and PD_PROTOCOL_V2.md for prerequisites and immutable-run behavior.

## 8. Limitations and implications for Myna

This remains a modest observational cohort, despite a large downloaded archive. The study cannot separate all effects of age, medication, disease severity, site, equipment, reference and artifact burden. Development is from one institution. Public de-identification prevents complete cross-release identity reconciliation. The exploratory Iowa split does not restore the independence of the previously examined pilot.

Fixed QC may preferentially exclude people with tremor or other symptoms. Device metadata are incomplete at some sites. No blinded expert review, prospective recruitment or independent clinical validation was performed. Bootstrap intervals describe uncertainty conditional on these retained cohorts and this trained model; they omit some training and sampling uncertainty.

After this report, all evaluated test cohorts are examined. Future architecture or preprocessing changes require a new, genuinely independent evaluation cohort or explicit exploratory labeling. Selecting the numerically highest external score for deployment would reuse the test for selection and needs new validation.

For Myna, these models provide an EEG research comparison only. A useful next study requires paired EEG, audio and movement from the same people and prospective validation against modality-specific baselines. Cross-sectional PD/control classification does not demonstrate symptom severity tracking, progression prediction or general abnormal-EEG detection. Reduced-electrode wearable performance needs a separate acquisition and motion study.

## Appendix A. Frozen protocol, model definitions and sources

Implementation started 26 September 2026, after all 41,746 planned archive files completed. One timestamp discrepancy was resolved by checking the entire affected file against its source MD5 and stored SHA-256; content was unchanged. All archive paths were reconciled against receipts, sizes, timestamps and source identities. Selected model-input EEG files are rehashed in full before preprocessing.

### Cohort and interpretation

The primary development source is ds008768: 312 unique Iowa participants (202 PD, 110 controls), with the earliest numbered session selected before QC. Its 358 visits are not independent people. The older Iowa releases are excluded entirely because a reliable cross-release participant map is unavailable. This replacement increases development data without claiming 312 additional people beyond the pilot.

A deterministic class-stratified 70/15/15 split assigns 220 people to training, 46 to validation and 46 to an internal test. Existing outcome knowledge from the older Iowa pilot may overlap this release; therefore the internal test is explicitly exploratory. Before training, fixed QC retained 177 training participants (113 PD, 64 controls) and 39 validation participants (27 PD, 12 controls). Excluded people stay excluded and are not reassigned.

Tel Aviv ds007526 supplies a new external cohort of 144 people (116 PD, 28 controls); only sitting, eyes-open resting recordings qualify. This is the primary external endpoint. New Mexico ds003490 supplies a separate external cohort of 50 people (25 PD, 25 controls), with ON and OFF analyses sharing controls. Its original IDs also occur in ds003506/ds003509; those task datasets are excluded. PD-only treatment datasets and unrelated healthy tasks do not enlarge the binary comparison. UCSD is omitted because its pilot outcomes have already been examined.

The distinct institutions, published acquisition provenance and exact source fingerprints support separation of the external sites. Public de-identification prevents absolute proof that no individual ever attended two institutions. No raw signal duplicates with different participant identities were found among included source files. Within-release participant identifiers define identity; metadata and source hashes are bound into the registry.

### Signal processing and QC

Use 17 common electrodes: Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, C4, T8, P7, P3, P4, P8, O1, O2. Cz is absent from Tel Aviv; Pz is a reference in Iowa. This channel decision uses source metadata, not external model scores.

Use four-second non-overlapping windows at 256 Hz, 1-40 Hz filtering, common average reference and per-window channel demeaning. Keep at most 15 accepted windows (60 seconds), requiring at least 10 (40 seconds). Inspect at most the first 124 seconds of the selected rest acquisition. Engineering QC rejects peak-to-peak amplitude above 500 microvolts, steps above 100 microvolts, channel standard deviation below 0.2 microvolts, excessive repeated samples, and bad source annotations. These rules are inherited from the pilot and have not received blinded clinical expert validation.

New Mexico windows come only from the explicitly labelled eyes-open one-second trigger sequence. Require exactly one contiguous run of at least 50 triggers, with intervals between 0.8 and 1.2 seconds; crop from its first to its last trigger, then apply the same two-second edge trimming and QC. One session with ambiguous event timing is excluded by protocol, without consulting its model score. Never substitute the following oddball task or an eyes-closed interval. Eye-interval checks use event metadata; test signals remain sealed until every model is locked.

Each person has equal sampling weight within class. Raw-signal neural models use robust per-channel scales fitted only on training people. Wavelet and spectral-token models use per-feature statistics with equal weight per training person. Every tuning fold refits its own scaler. No test-side population normalization, calibration or pretraining is permitted.

### Models and explicit adaptations

These are separate architecture-family experiments, not exact reproductions or replications of published scores.

- Majority/class-frequency reference: constant training PD prevalence, threshold 0.5.

- Spectral logistic regression: participant-mean log delta/theta/alpha/beta power per electrode; training-fold regularization search.

- Periodic/aperiodic feature logistic regression: log-log spectral slope and offset from 2-35 Hz, lower-residual refit, positive band residuals and peak summaries. This transparent approximation is inspired by Donoghue et al.; it is not the specparam/FOOOF algorithm and does not perform Gaussian peak fitting.

- Demographic logistic regression: age and sex, with training-only missing-value imputation and indicators. A confounding check, not an EEG model.

- Artifact logistic regression: accepted/rejected window counts and signal variability. An additional confounding check.

- EEGNet adaptation: temporal, depthwise spatial and separable convolutions, global pooling; 1,905 parameters.

- ShallowConvNet adaptation: 20 temporal/spatial filters, square, average pooling and log activation; 7,381 parameters. Compact filter count and global readout differ from the original.

- Channel CNN adaptation: author-code-inspired four temporal convolution blocks with 21/42/42/64 outputs and kernels 20/10/10/5; smaller global-pooling classifier, shared preprocessing and binary logits; 52,764 parameters. The original author code mixes electrode input channels; this is not a depthwise independent-channel implementation.

- Wavelet CNN adaptation: fixed Morlet scalograms at 12 log-spaced frequencies (2-35 Hz), 64 temporal bins and 17 electrode input planes; three convolution blocks, 25,665 parameters. This differs from the paper's 128x128 individual-channel images and 20-layer network.

- Graph adaptation: shared temporal electrode encoder, four learned adjacency heads and one-/two-step message passing; 13,473 parameters. Omits the cited paper's structured long convolution, contrastive pretraining, Chebyshev operator and explanation procedure. Learned edges are predictive weights, not evidence of physiological connectivity.

- Transformer adaptation: four bandpower features plus normalized spectral entropy at each electrode token, two 32-dimensional four-head encoder layers and electrode-position embeddings; 17,857 parameters. Binary per-window classification with person-level aggregation differs from the paper's medication-state classification and representation sequence.

### Model selection and test lock

Each neural family receives the same six settings: learning rate 0.0003/0.001/0.003 crossed with dropout 0.25/0.5. Use the same three stratified participant folds within training, at most 100 epochs and patience 15. Training-fold results choose the configuration. Three fixed final seeds (20260926-28) then train using the separate validation split for early stopping. Average their window-mean probabilities; choose the validation threshold maximizing balanced accuracy, ties closest to 0.5. Never choose a winning seed on a test.

Logistic models search C = 0.01/0.1/1/10 using the same folds. Models with different optimization algorithms have different costs; report them. The validation classification metrics are optimistic because that set chooses thresholds and stopping. Internal CV scores also serve model selection and are not independent test estimates.

Freeze the policy, registry, preprocessing, training folds and hashes of all final model artifacts before test preparation. All models receive the same retained people per endpoint. Report endpoints separately with participant-level AUROC and 95% bootstrap intervals, balanced accuracy, sensitivity/specificity with Wilson intervals, average precision, prevalence, Brier score, confusion counts and coverage. Compare each model with spectral regression through paired participant bootstrap AUROC differences, including Bonferroni-adjusted intervals. These small-sample comparisons remain exploratory; no automatic winner is declared.

A measured MPS benchmark forecasts all 126 neural fits with maximum epochs and a 25% margin before launch. Each campaign also has a 96-hour watchdog. Completed fits resume only when identities and weight hashes match. Interrupted unfinished fits restart under the same frozen configuration; progress and best checkpoints remain available. Raw data, caches, weights and participant predictions remain outside Git.

### Primary references

1. Lawhern et al., EEGNet (2018): https://arxiv.org/abs/1611.08024

2. Schirrmeister et al., EEG convolutional decoding (2017): https://arxiv.org/abs/1703.05051

3. Generalizable EEG classification of PD (2023): https://doi.org/10.1016/j.imu.2023.101352 ; author implementation: https://github.com/diamandis-lab/CNN_EEG_PD

4. Shaban and Amara, wavelet CNN (2022): https://doi.org/10.1371/journal.pone.0263159 . The methods explicitly describe separating training and validation images by patient ID; its image-level metrics still do not establish external-site performance.

5. Neves et al., multi-head graph structure learning (2024 preprint): https://arxiv.org/html/2408.00906v1

6. Lee et al., spectral/entropy Transformer (2026): https://ojs.aaai.org/index.php/AAAI-SS/article/view/42519

7. Donoghue et al., spectral parameterization (2020): https://doi.org/10.1038/s41593-020-00744-x

8. Source datasets: https://openneuro.org/datasets/ds008768 ; https://openneuro.org/datasets/ds007526 ; https://openneuro.org/datasets/ds003490
