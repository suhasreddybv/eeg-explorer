# PD classifier v3: expanded-data exploratory development

Approved scope: Parkinson's disease versus controls; cross-site AUROC is primary, within-site performance is reported separately; use compatible auxiliary EEG for self-supervised pretraining. Local Mac only. Preserve v1/v2 as preliminary results.

## Evaluation and prior exposure

All four available PD/control sites (Iowa, Tel Aviv, New Mexico and UCSD) have already had outcomes examined in v1/v2. V3 therefore provides **exploratory nested cross-validation**, not a fresh confirmatory test. A new unexamined institution is still needed for confirmation. No promised AUROC or repeated search until a desired test score is reached.

Primary evaluation: four outer leave-one-site-out folds. Within each outer training set, leave each remaining site out in turn for configuration selection. Choose the configuration by equal-site mean AUROC, not pooled cross-site discrimination. Select each family separately and also evaluate the complete inner-selected model-selection pipeline. Report each outer site and the equal-site macro AUROC; do not substitute pooled accuracy.

Secondary evaluation: three participant-grouped folds stratified jointly by site and diagnosis. Each fold contains unseen people from the same sites represented in training. Inner site validation still selects configurations. Combine out-of-fold predictions once per person, and report within-site results separately from primary results.

All sessions and windows of a person remain in the same outer/inner fold. Fit scalers and classical estimators on inner training people only. Auxiliary pretraining excludes **all** participants from all four PD evaluation sites, including their overlapping task releases, even when that site's people are allowed in supervised training. No test-side calibration, normalization, threshold selection or unsupervised adaptation. Fixed signal-only feature computation can be cached for every recording before training; it fits no population parameters.

Lock every outer model before computing any outer predictions. Thresholds come from inner out-of-fold predictions; final neural epoch counts use the median inner best epoch, so outer tests never determine stopping. Three final neural seeds are averaged; no best-seed selection. Classical models use one deterministic fit. All aggregate performance refers to held-out people, never training fits. A final research model is subsequently fitted using all eligible PD/control people, with separate inner-site selection; it has no new test result.

## Archive accounting

Every downloaded dataset is assigned a role or explicit exclusion in the frozen v3 registry and usage ledger. Four PD/control sources supply documented eyes-open resting recordings, including repeated visits and both medication sessions. No arbitrary 60-second cap: process all documented compatible resting duration. New Mexico remains restricted to the explicitly timed eyes-open interval, never its subsequent task.

Auxiliary sources: ds003775, ds004148, ds004504, ds004796, ds005385, ds006036, ds007864, ds007865 and bigP3BCI. These can include eyes-closed or task EEG because they supply an unlabeled reconstruction task, not PD-negative training labels. This mismatch may harm transfer and is tested by a from-scratch comparator. Keep all compatible recording durations. Canonicalize the shared participants in ds004504/ds006036. Treat the unresolved NeuroEPO arm identities as one conservative auxiliary group rather than claim independent participants. Auxiliary subject counts are grouping counts, not proof of cross-study unique people.

Exclude older Iowa releases, UNM task releases and older NeuroEPO releases because of unresolved overlaps/replacement. Exclude ds005048 until its MATLAB v7.3 format has a validated reader, and ds007987 until its A1-D32 electrodes have a validated physical mapping. Within the four supervised sources, walking/task acquisitions are excluded. Exact duplicate recordings within a person are kept once; different identities sharing the same signal hash stop the audit. Source hashes and metadata are frozen; de-identification means absolute cross-study identity independence cannot be proven.

## Full-duration processing

Use the same 17 named scalp electrodes as v2. Explicit legacy mappings T3/T4/T5/T6 -> T7/T8/P7/P8 and an EEG_ prefix removal are allowed; no electrode-position guesses. Supervised recordings require all 17 channels. Auxiliary recordings require at least 12; missing channels are masked throughout attention and reconstruction and are never reconstruction targets.

Process four-second, non-overlapping windows in bounded chunks with two seconds of padding on each side; resample to 256 Hz, 1-40 Hz fourth-order zero-phase Butterworth filtering and average reference over observed electrodes. Remove channel means. Keep all accepted windows and their time provenance. Retain engineering QC thresholds from v2: peak-to-peak <=500 uV, step <=100 uV, channel SD >=0.2 uV, repeated-sample fraction <=0.3, no bad source annotation. These are not clinically validated artifact rules. A person needs at least ten accepted windows across sessions. QC is never changed based on model performance. Report exclusion rates by site/diagnosis and accepted duration.

An EDF header with an impossible wall-clock second of 60 is read through a temporary copy with that field set to 59. The original checksum-verified file and all sample values remain unchanged. Absolute wall-clock timestamps are not classifier inputs. A BrainVision header whose misspelled companion references are missing can point to checksum-verified companions sharing the exact canonical recording stem, through a temporary header; sample data remain unchanged. Unexpected reader or integrity failures halt the stage; they are not silently treated as QC exclusions.

Each window yields 34 one-Hz log-PSD bins from 2-35 Hz per electrode (Welch, one-second segments, 50% overlap). Complete montages additionally yield the upper triangle of a matrix-log covariance after fixed 10% isotropic shrinkage. These descriptors are fixed within-window transformations. Per-person feature vectors average session summaries equally, with all accepted windows contributing within each session.

## New model and comparison matrix

A compact masked spectral encoder projects each electrode's spectrum to 48 dimensions; adds learned electrode positions; applies two four-head Transformer encoder layers (96-dimensional feed-forward layer, dropout 0.2); reconstructs hidden electrode spectra and pools available electrodes for a binary head. This is a custom compact adaptation, not an implementation of BENDR or LaBraM and not a raw-waveform foundation model.

Auxiliary pretraining: deterministic 90/10 canonical-group split, training-only weighted scaling, 40% random channel masking, weighted mean-square reconstruction, AdamW at 0.001 with weight decay 0.01, batch 128, maximum 20 epochs, patience five on held-out auxiliary reconstruction. Equal study, person and session contributions reduce duration/site domination. Diagnosis labels are not used. Auxiliary validation groups never contribute to gradient fitting.

Compare seven supervised families on shared participants/folds:

1. Four-band spectral logistic regression (2-4, 4-8, 8-13, 13-30 Hz).
2. Rich spectral mean/variability and log-covariance logistic regression.
3. The same rich descriptors with an RBF SVM.
4. Rich descriptors plus frozen pretrained embedding mean/variability, logistic regression.
5. The same hybrid descriptors with an RBF SVM.
6. The spectral neural encoder trained from scratch.
7. The pretrained spectral neural encoder fine-tuned for PD, with a fresh binary head.

Classical C grid: 0.1, 1, 10. SVM gamma is fixed to scale; probability calibration is internal to the training set, one example per person. StandardScaler is fit within each training fold. Neural learning rates: 0.0003 and 0.001; batch 128; AdamW decay 0.01; gradient clipping 5; maximum 40 epochs; inner stopping patience seven. Equal site/class/person/session influence in supervised loss. Final seed ensemble: 20260927, 20260928, 20260929. Inner tuning uses the first seed. This finite matrix is frozen before fitting.

Per-site uncertainty uses 2,000 stratified participant bootstrap resamples, with paired AUROC differences against spectral regression. Pairwise intervals are exploratory and unadjusted; report this explicitly. Also report class counts, PR-AUC, Brier score, sensitivity/specificity with Wilson intervals, confusion counts and coverage. Different within-site folds have different training-selected thresholds. Subgroup and confound follow-up must remain descriptive after test opening and cannot change tested models. V2 and v3 have different training populations, recording durations and eligibility: a higher v3 number alone does not isolate an architecture benefit.

## Research basis

- Kostas et al., BENDR (2021), self-supervised EEG representation learning: https://arxiv.org/abs/2101.12037 . Motivation for unlabeled pretraining, not evidence of PD improvement.
- Jiang et al., LaBraM (ICLR 2024), masked EEG modeling: https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf . Motivation for channel masking and transferable representations; our reconstruction target and scale differ.
- External-validation PD study and architecture references remain in PD_CLASSIFIER_RESEARCH_PLAN.md and PD_PROTOCOL_V2.md.

## Execution and delivery

Benchmark preprocessing and both training campaigns on this Mac before launch. Stop and notify before any continuous stage forecast exceeds 96 hours. A separate macOS job runs stages without Codex, with a process lock, persistent logs, content-checked completed fits and a 96-hour stage watchdog. Code/protocol hashes freeze the implementation before training. Failures remain visible and require diagnosis; completed fits are preserved.

Publish code and aggregate documentation only. Keep identities, predictions, source receipts and weights local. Produce a new report that includes negative findings and explicitly answers whether pretraining or any new model improved the relevant held-out AUROC. Final PDF delivery requires visual inspection. Success means a correctly executed comparison; higher AUROC is a hypothesis to test, not a guaranteed outcome.
