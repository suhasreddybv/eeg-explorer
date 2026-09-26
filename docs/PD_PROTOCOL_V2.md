# Expanded PD comparison: frozen protocol v2

Implementation started 26 September 2026, after all 41,746 planned archive files completed. One timestamp discrepancy was resolved by checking the entire affected file against its source MD5 and stored SHA-256; content was unchanged. All archive paths were reconciled against receipts, sizes, timestamps and source identities. Selected model-input EEG files are rehashed in full before preprocessing.

## Cohort and interpretation

The primary development source is ds008768: 312 unique Iowa participants (202 PD, 110 controls), with the earliest numbered session selected before QC. Its 358 visits are not independent people. The older Iowa releases are excluded entirely because a reliable cross-release participant map is unavailable. This replacement increases development data without claiming 312 additional people beyond the pilot.

A deterministic class-stratified 70/15/15 split assigns 220 people to training, 46 to validation and 46 to an internal test. Existing outcome knowledge from the older Iowa pilot may overlap this release; therefore the internal test is explicitly exploratory. Before training, fixed QC retained 177 training participants (113 PD, 64 controls) and 39 validation participants (27 PD, 12 controls). Excluded people stay excluded and are not reassigned.

Tel Aviv ds007526 supplies a new external cohort of 144 people (116 PD, 28 controls); only sitting, eyes-open resting recordings qualify. This is the primary external endpoint. New Mexico ds003490 supplies a separate external cohort of 50 people (25 PD, 25 controls), with ON and OFF analyses sharing controls. Its original IDs also occur in ds003506/ds003509; those task datasets are excluded. PD-only treatment datasets and unrelated healthy tasks do not enlarge the binary comparison. UCSD is omitted because its pilot outcomes have already been examined.

The distinct institutions, published acquisition provenance and exact source fingerprints support separation of the external sites. Public de-identification prevents absolute proof that no individual ever attended two institutions. No raw signal duplicates with different participant identities were found among included source files. Within-release participant identifiers define identity; metadata and source hashes are bound into the registry.

## Signal processing and QC

Use 17 common electrodes: Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, C4, T8, P7, P3, P4, P8, O1, O2. Cz is absent from Tel Aviv; Pz is a reference in Iowa. This channel decision uses source metadata, not external model scores.

Use four-second non-overlapping windows at 256 Hz, 1-40 Hz filtering, common average reference and per-window channel demeaning. Keep at most 15 accepted windows (60 seconds), requiring at least 10 (40 seconds). Inspect at most the first 124 seconds of the selected rest acquisition. Engineering QC rejects peak-to-peak amplitude above 500 microvolts, steps above 100 microvolts, channel standard deviation below 0.2 microvolts, excessive repeated samples, and bad source annotations. These rules are inherited from the pilot and have not received blinded clinical expert validation.

New Mexico windows come only from the explicitly labelled eyes-open one-second trigger sequence. Require exactly one contiguous run of at least 50 triggers, with intervals between 0.8 and 1.2 seconds; crop from its first to its last trigger, then apply the same two-second edge trimming and QC. One session with ambiguous event timing is excluded by protocol, without consulting its model score. Never substitute the following oddball task or an eyes-closed interval. Eye-interval checks use event metadata; test signals remain sealed until every model is locked.

Each person has equal sampling weight within class. Raw-signal neural models use robust per-channel scales fitted only on training people. Wavelet and spectral-token models use per-feature statistics with equal weight per training person. Every tuning fold refits its own scaler. No test-side population normalization, calibration or pretraining is permitted.

## Models and explicit adaptations

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

## Model selection and test lock

Each neural family receives the same six settings: learning rate 0.0003/0.001/0.003 crossed with dropout 0.25/0.5. Use the same three stratified participant folds within training, at most 100 epochs and patience 15. Training-fold results choose the configuration. Three fixed final seeds (20260926-28) then train using the separate validation split for early stopping. Average their window-mean probabilities; choose the validation threshold maximizing balanced accuracy, ties closest to 0.5. Never choose a winning seed on a test.

Logistic models search C = 0.01/0.1/1/10 using the same folds. Models with different optimization algorithms have different costs; report them. The validation classification metrics are optimistic because that set chooses thresholds and stopping. Internal CV scores also serve model selection and are not independent test estimates.

Freeze the policy, registry, preprocessing, training folds and hashes of all final model artifacts before test preparation. All models receive the same retained people per endpoint. Report endpoints separately with participant-level AUROC and 95% bootstrap intervals, balanced accuracy, sensitivity/specificity with Wilson intervals, average precision, prevalence, Brier score, confusion counts and coverage. Compare each model with spectral regression through paired participant bootstrap AUROC differences, including Bonferroni-adjusted intervals. These small-sample comparisons remain exploratory; no automatic winner is declared.

A measured MPS benchmark forecasts all 126 neural fits with maximum epochs and a 25% margin before launch. Each campaign also has a 96-hour watchdog. Completed fits resume only when identities and weight hashes match. Interrupted unfinished fits restart under the same frozen configuration; progress and best checkpoints remain available. Raw data, caches, weights and participant predictions remain outside Git.

## Primary references

1. Lawhern et al., EEGNet (2018): https://arxiv.org/abs/1611.08024
2. Schirrmeister et al., EEG convolutional decoding (2017): https://arxiv.org/abs/1703.05051
3. Generalizable EEG classification of PD (2023): https://doi.org/10.1016/j.imu.2023.101352 ; author implementation: https://github.com/diamandis-lab/CNN_EEG_PD
4. Shaban and Amara, wavelet CNN (2022): https://doi.org/10.1371/journal.pone.0263159 . The methods explicitly describe separating training and validation images by patient ID; its image-level metrics still do not establish external-site performance.
5. Neves et al., multi-head graph structure learning (2024 preprint): https://arxiv.org/html/2408.00906v1
6. Lee et al., spectral/entropy Transformer (2026): https://ojs.aaai.org/index.php/AAAI-SS/article/view/42519
7. Donoghue et al., spectral parameterization (2020): https://doi.org/10.1038/s41593-020-00744-x
8. Source datasets: https://openneuro.org/datasets/ds008768 ; https://openneuro.org/datasets/ds007526 ; https://openneuro.org/datasets/ds003490
