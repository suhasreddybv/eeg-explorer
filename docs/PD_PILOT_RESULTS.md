# First local PD classifier pilot

**25 September 2026 · Research prototype; not ready for diagnostic use.**

The small EEGNet-style ensemble transferred better than the spectral baseline to the reserved UCSD cohort, but the validation-selected operating threshold produced too many false positives externally. This is a useful first experiment, not evidence of clinical readiness or symptom-tracking validity.

## Run and selection

Run: local `data/pd-runs/v1`. Registry fingerprint: `f43b68051e273867e679533eb51cff11444e947058fdc623dbbab087e4c0fe94`.

The initial Iowa assignment was 105 training / 22 validation / 22 internal-test participants. Fixed signal-quality rules retained 80 training (50 PD, 30 controls), 15 validation (9 PD, 6 controls), and 20 internal-test (14 PD, 6 controls). Excluded people were not reassigned. Development exclusions affected 26/85 PD participants and 6/42 controls, which can introduce selection bias.

Six predefined configurations were evaluated with three participant folds strictly within the training split. The selected learning rate was 0.0003 and dropout 0.25; its mean training-fold best AUROC was 0.798. Three fixed seeds were fitted with validation early stopping, then their window-averaged probabilities were averaged. The frozen ensemble threshold was **0.3552**. No test predictions selected configuration, seed, threshold, scaling or preprocessing.

Training used the M4 GPU. Tuning took 543 seconds; the three final fits took approximately 29, 35 and 21 seconds. Total neural fitting was **about 628 seconds (10.5 minutes)**, excluding source downloads, preprocessing, benchmark, baseline and report generation. The earlier conservative 65-minute forecast assumed every fit reached 100 epochs; early stopping substantially shortened this run.

Model checkpoints, source fingerprints, participant predictions, training histories, evaluation policy and immutable split records remain local, outside Git. The source protocol and evaluation code are documented in [LOCAL_TRAINING.md](LOCAL_TRAINING.md).

## Held-out internal test: Iowa

Twenty of 22 participants passed QC, a coverage of 90.9%.

- Neural AUROC: **0.976**, participant-bootstrap 95% interval **0.899–1.000**.
- Spectral baseline AUROC: **0.917**, interval **0.756–1.000**.
- Neural sensitivity: **12/14 = 85.7%**; specificity: **6/6 = 100%**.
- Neural balanced accuracy: **92.9%**. The specificity Wilson interval is only **61.0–100%**, reflecting the six eligible controls.
- Neural Brier score: **0.189**, versus **0.164** for the baseline. Better ranking did not imply better probability accuracy.

This tests new people under the development study’s conditions. It does not establish transfer to a new device or site.

## Primary external test: UCSD, medication ON

Twenty-eight of 31 participants passed QC, including 14 PD and 14 controls; coverage was 90.3%.

- Neural AUROC: **0.867**, participant-bootstrap 95% interval **0.704–0.969**.
- Spectral baseline AUROC: **0.673**, interval **0.444–0.845**.
- Neural sensitivity: **12/14 = 85.7%**, Wilson interval **60.1–96.0%**.
- Neural specificity: **8/14 = 57.1%**, Wilson interval **32.6–78.6%**.
- Neural balanced accuracy: **71.4%**; confusion counts: 8 true negatives, 6 false positives, 2 false negatives, 12 true positives.

The ranking signal is encouraging, but six false positives among 14 eligible controls makes the frozen threshold unsuitable for screening claims. We did not retune the threshold on this test cohort. The small sample, selection by QC, acquisition differences and incompletely harmonized medication status limit interpretation.

## Secondary external test: UCSD, medication OFF

Twenty-nine of 31 participants passed QC, including 15 PD and 14 controls; coverage was 93.5%.

- Neural AUROC: **0.814**, interval **0.621–0.941**.
- Spectral baseline AUROC: **0.581**, interval **0.359–0.800**.
- Neural sensitivity: **11/15 = 73.3%**; specificity: **8/14 = 57.1%**.
- Neural balanced accuracy: **65.2%**; confusion counts: 8 true negatives, 6 false positives, 4 false negatives, 11 true positives.

ON and OFF analyses share participants and controls. They are two medication-condition analyses of one external cohort, not two independent replications. New Mexico remains reserved until its eyes-open event intervals are verified.

## What remains uncertain

The age/sex-only development baseline reached validation AUROC **0.880** on the same small 15-person validation set. This does not prove that the EEG model uses demographics, but it makes demographic/acquisition confounding an active concern. The neural and baseline AUROC intervals overlap; no statistically established superiority is claimed.

Quality rules are engineering heuristics, not expert-reviewed artifact annotations. Differential rejection may select easier or systematically different patients. Reported performance applies to the people receiving predictions; exclusions do not count as correct classifications. Possible EEG/EMG contamination and study-specific preprocessing remain potential explanations for some signal.

We have now inspected the Iowa internal and UCSD external test outcomes. Future tuning against those outcomes must treat these cohorts as exploratory and seek a fresh locked evaluation cohort. Creating a new run directory does not reset that history.

## Next scientific steps

Prioritize a blinded EEG-quality review, age/sex and acquisition sensitivity analyses, and the untouched New Mexico event-based evaluation. Preserve the current frozen result before changing preprocessing or thresholds. More architecture complexity is not the immediate need.

The app’s Parkinson’s research panel displays aggregate test results. It does not diagnose an uploaded recording. Progression and symptom tracking will require separate longitudinal labels and validation; Myna multimodal gains will require paired EEG/audio/movement data from the same participants.
