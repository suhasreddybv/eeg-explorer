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
