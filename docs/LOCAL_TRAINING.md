# Local Parkinson’s EEG training

The approved policy is **local Mac only**. Before launching any continuous stage projected to exceed **96 hours**, stop and notify the user. The training commands require a real-data benchmark and enforce the four-day forecast gate; training loops also have a wall-clock watchdog. A runtime forecast is not permission for a longer run.

## What is implemented

`pd_pipeline` is separate from the recording explorer and does not turn the explorer’s anomaly scores into diagnostic labels. It provides an immutable subject/recording registry, verified-file preprocessing, training-only scaling, spectral logistic regression, an EEGNet-style network, a real-data CPU/MPS benchmark, six-configuration/three-fold training-only tuning, three seeded final fits, a fingerprinted model lock and held-out evaluation.

The network has 1,921 parameters with the current 18-channel settings. It uses temporal, depthwise spatial and separable convolutions with bounded spatial filters and global average pooling. This is an EEGNet-style adaptation, not a claim of exact reproduction of the published architecture or pretrained clinical performance.

The frozen Iowa split initially contains 105 training participants (70 PD / 35 controls), 22 validation participants (15 / 7), and 22 internal-test participants (15 / 7). UCSD and New Mexico are reserved external sites. All sessions from the same person keep one split; source identity and duplicate checks reject contradictions. The newer Iowa release remains excluded because overlap has not been resolved.

Fixed initial QC retained 80 training participants (50 PD / 30 controls) and 15 validation participants (9 / 6). The remaining 32 development participants were excluded by the engineering QC rules, not by model performance. Differential exclusion rates are a limitation that must accompany results. Four-second windows are non-overlapping, capped at 15 per recording, with at least ten accepted windows required. No discarded participant is moved to another split.

Eighteen common electrodes are used, excluding Iowa’s Pz reference: Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, P4, P8, O1, O2. Fixed 1–40 Hz filtering, 256 Hz resampling, agreed-channel average reference, window demeaning and training-only robust amplitude scaling are applied. Artifact rules currently reject large peak-to-peak amplitude, abrupt steps, flat channels, repeated samples and marked bad intervals. They do not comprehensively remove all eye/muscle artifacts; the pilot is not clinically validated.

Source acquisition descriptions do not always match exported file formats. The Iowa release inspected here uses EEGLAB `.set` plus `.fdt`. The pipeline uses actual file headers and complete verified companions, not the README’s acquisition-system name, to select a reader.

## Commands

Install optional training dependencies from the project directory:

```sh
uv sync --extra training --python 3.12
```

Create a new run and freeze the registry before preprocessing:

```sh
.venv/bin/python -m pd_pipeline registry \
  --archive '/Volumes/Extreme Pro/EEG data' --run data/pd-runs/v1
.venv/bin/python -m pd_pipeline prepare \
  --archive '/Volumes/Extreme Pro/EEG data' --run data/pd-runs/v1
```

Preprocessing can be rerun as downloads complete. It reuses caches only when source/configuration/registry fingerprints and cached bytes agree. No completed download receipt means the recording stays pending. Changing the split or processing rules requires a new protocol/run rather than overwriting the existing registry. By default only train/validation signals are processed; test access is a separate command.

Benchmark and run the experiment:

```sh
.venv/bin/python -m pd_pipeline benchmark --run data/pd-runs/v1 --device mps --epochs 5
.venv/bin/python -m pd_pipeline campaign --run data/pd-runs/v1 --device mps
```

Use `--device cpu` only after benchmarking CPU separately. There is no cloud/CUDA path. On a Mac, a restrictive command sandbox may hide MPS even when the hardware supports it; the successful benchmark must identify the actual selected device. Do not silently label a CPU result as GPU training.

The campaign runs the spectral baseline, then six combinations of learning rate (0.0003, 0.001, 0.003) and dropout (0.25, 0.5), each in three participant folds drawn only from the training split. Each fold is capped at 100 epochs with patience 15. Three fixed seeds fit the selected configuration on training data, using the separate validation set for early stopping. The final ensemble averages their probabilities. Validation chooses its operating threshold. No test result chooses a seed, model, threshold or preprocessing setting.

Run directories preserve logs, histories, best/latest neural checkpoints, baseline weights, tuning results and a final model lock. Re-running an existing campaign is deliberately refused rather than overwriting a scientific record. Download/preprocessing resume is supported; automated mid-campaign training resume is not yet implemented. Checkpoints remain available for inspection or an explicitly documented recovery. Keep the Mac powered and awake for a campaign.

## Held-out evaluation

Only after the model lock exists:

```sh
.venv/bin/python -m pd_pipeline evaluate \
  --archive '/Volumes/Extreme Pro/EEG data' --run data/pd-runs/v1 \
  --cohort internal_test --device mps
.venv/bin/python -m pd_pipeline evaluate \
  --archive '/Volumes/Extreme Pro/EEG data' --run data/pd-runs/v1 \
  --cohort ucsd_on --device mps
.venv/bin/python -m pd_pipeline evaluate \
  --archive '/Volumes/Extreme Pro/EEG data' --run data/pd-runs/v1 \
  --cohort ucsd_off --device mps
```

The evaluation policy is saved before holdouts are opened and bound into the model lock. UCSD’s source study describes eyes-open rest: [George et al. (2013)](https://doi.org/10.1016/j.nicl.2013.07.013). Medication ON is the primary external comparison; OFF is a secondary medication-shift analysis. This explicit session rule replaces an ambiguous “earliest visit” choice because ON/OFF filename order does not establish acquisition chronology. No fallback between states is allowed. Iowa medication state is not harmonized, so these remain cross-site research stress tests. The two external comparisons share controls and PD participants.

New Mexico remains reserved but is not automatically evaluated: its mixed eyes-open/closed resting periods require verified event-interval extraction. A whole recording must not be treated as eyes-open rest. This limitation is visible rather than silently mixing conditions.

Tests apply the fixed processing, learned scaling and frozen thresholds without refitting. Reports show participant-level AUROC, bootstrap intervals, balanced accuracy, sensitivity/specificity with Wilson intervals, average precision, Brier score, exclusions, class counts, source fingerprints and individual predictions. An AUROC bootstrap interval can be degenerate in a very small perfectly separated sample; that does not establish certainty. If QC removes a whole class, evaluation reports that performance cannot be estimated rather than manufacturing a score.

Results are memoized under the model-lock fingerprint. Once test outcomes have been inspected, further tuning makes that cohort exploratory for the next protocol. A different run directory alone does not restore a genuinely unseen test set.

## Measured Mac benchmark

On this M4/16 GB Mac, a five-epoch real-data MPS benchmark with 70 available training participants and 1,027 windows measured a median **0.997 seconds per epoch**, including a small training-only inference pass. Linear extrapolation to all 105 originally assigned training participants, 21 fits × 100 epochs and a 25% buffer was **3,926 seconds (about 65 minutes)**. Peak observed process RSS at reporting was about 409 MB; this is not a measure of all GPU/unified-memory allocation.

This supersedes the initial rough multi-day allowance for this compact experiment, not for arbitrary future models. Actual epochs depend on early stopping, and simultaneous downloads/apps affect elapsed time. Benchmark output is local in `data/pd-runs/v1/benchmark.json`.

## App and provenance

The **Parkinson’s research** panel reads aggregate progress through the loopback API. It does not initiate training or diagnose an uploaded recording. The configured run path is stored locally in `data/pd-training-path.txt`. Patient-level metadata, predictions, caches and checkpoints remain under ignored `data/`; only code and aggregate documentation belong in Git.

The current model is an initial research pilot. Remaining scientific work includes blinded expert artifact review, stronger demographic/acquisition confound analysis, verified New Mexico event extraction, protocol sensitivity studies, and longitudinal/paired-modality data for symptom tracking. These are not implied by completing a classification run.

## First completed result

The first campaign and locked internal/UCSD tests are complete. See [PD_PILOT_RESULTS.md](PD_PILOT_RESULTS.md) for coverage, uncertainty, false positives, timing, and the implications for future test-set reuse. Run `python -m pd_pipeline audit --run data/pd-runs/v1` for the development-only demographic/QC audit. The untouched New Mexico cohort remains reserved.

## Expanded comparison v2 (26 September 2026)

The completed archive has been reconciled. The expanded comparison is implemented in `cohort_v2`, `models_v2`, `compare_v2` and `diagnostics_v2`; the exact scientific protocol and architecture adaptations are in [PD_PROTOCOL_V2.md](PD_PROTOCOL_V2.md). The pilot remains unchanged.

The development pool now has 220 assigned training people and 46 validation people from the expanded Iowa release. Fixed QC retained 177 and 39 respectively. Another 46 Iowa people are an exploratory internal holdout; Tel Aviv (144 candidates) and New Mexico (50 candidates, repeated medication sessions) are reserved external endpoints. Locked evaluation retained 35 Iowa participants, 95 Tel Aviv participants, 33 New Mexico ON participants and 38 New Mexico OFF participants. The ON/OFF endpoints share people. See [the completed results](PD_COMPARISON_RESULTS.md).

The final MPS benchmark forecast approximately 9.94 hours for all six neural families at the full epoch limits, including a 25% margin. This is a conservative training forecast, not a promise of completion at a particular time. Early stopping, interruptions, preprocessing and report generation change actual elapsed time.

The local macOS job `local.eeg-explorer.comparison.v2` runs the lock-protected `scripts.run_pd_comparison` driver. It executes training, test preprocessing only after the all-model lock, test evaluation, subgroup analysis and aggregate/PDF reporting. No Codex calls occur inside the job. Keep the Mac powered, awake and connected to the SSD. Completion stops the job; a failure is recorded in `data/pd-runs/v2/campaign-status.json` and stage logs rather than silently dropping a model. The job can resume completed fits after an explicitly reviewed recovery, but never refits after a completed all-model lock. Do not start a second campaign manually while the job is running.

Preparation and reproduction commands, from the project root:

```sh
.venv/bin/python -m pd_pipeline.cohort_v2 build
.venv/bin/python -m pd_pipeline.cohort_v2 prepare
.venv/bin/python -m pd_pipeline.compare_v2 setup
.venv/bin/python -m pd_pipeline.compare_v2 benchmark --device mps
.venv/bin/python -m pd_pipeline.diagnostics_v2 permutation
.venv/bin/python -m pd_pipeline.compare_v2 campaign --device mps
.venv/bin/python -m pd_pipeline.cohort_v2 test-prepare
.venv/bin/python -m pd_pipeline.compare_v2 evaluate --device mps
.venv/bin/python -m pd_pipeline.diagnostics_v2 subgroups
```

The report generator `scripts/report_pd_comparison.py` uses a separate document runtime with `reportlab` and `pypdf`. It refuses to report until the locked test results and diagnostics exist. It writes aggregate Markdown/JSON and model cards under `docs/`, and `output/pdf/PD_Classifier_Comparison_Report.pdf`. The final 17-page PDF passed rendered-page visual review on 26 September 2026. Local run artifacts include identities and predictions and must remain ignored by Git. The completed local campaign took 78.3 minutes, including its recorded training-to-report stages; the separate permutation check, initial preparation and final editorial review are outside this elapsed time. All 126 neural fits and locked model artifacts are preserved. See [the verification record](PD_COMPARISON_VERIFICATION.md).

## Expanded-data development v3 (27 September 2026)

V1/v2 are preserved as preliminary studies. The user approved PD-versus-control development using compatible auxiliary EEG pretraining, with cross-site AUROC primary and within-site results separate. The frozen design is [PD_PROTOCOL_V3.md](PD_PROTOCOL_V3.md). It uses all compatible recording duration and repeated sessions, with canonical participants kept together. Every archived dataset has an explicit role or exclusion. There is no fresh confirmatory site remaining among the four previously examined PD sites.

The independent Mac job `local.eeg-explorer.comparison.v3` runs preparation, auxiliary pretraining, supervised benchmarks, nested model comparison, outer evaluation, final research fitting and aggregate reporting. It does not call Codex. Status and errors are in ignored `data/pd-runs/v3/campaign-status.json`; completed fits and data caches resume with hash checks. Each training stage is benchmarked before launch, and a forecast above 96 hours stops for user notification. Completed data preparation and benchmarks are not a claim of a trained or improved model.

The v3 usage ledger initially assigns 11,211 candidate recordings across four supervised and nine auxiliary sources; the other ten sources and non-rest PD recordings have explicit exclusions. Candidate counts precede montage/QC checks and do not equal independent people. The initial representative-record preprocessing benchmark forecast 1.15 hours with a 2x margin. Training estimates are recorded separately after data preparation.

All 63 local tests and Ruff passed before initial service installation. Final artifacts require an independent review and a new visually inspected PDF; the v2 PDF is preserved.
