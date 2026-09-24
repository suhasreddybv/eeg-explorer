# Signal processing and interpretation

## Data representation

Signals are stored as channels × samples in microvolts. Channel order and sampling rate are preserved. EDF input is interpreted using [MNE's EDF reader](https://mne.tools/stable/generated/mne.io.read_raw_edf.html). There is no montage inference, source localization, or automatic reference change.

The deterministic synthetic walkthrough uses seed 42 and deliberately changes its signal at known times. It exercises the pipeline; success on this signal is not evidence of biological or clinical validity.

## Quality heuristics

For each channel in every 4-second raw window:

- Standard deviation below 0.15 µV, or more than 30% of adjacent differences smaller than 1e-8 µV: flat/repeated samples.
- Peak-to-peak amplitude above 200 µV: large amplitude / possible movement.
- Adjacent change above 100 µV: abrupt jump.
- Welch power in ±1 Hz around 50 or 60 Hz above 25% of 1–min(100, Nyquist) Hz power, when that frequency is observable: mains interference.

These thresholds are provisional. Large physiological signals can be flagged, and subtle motion, eye movement, muscle activity, saturation, bridging, and poor contact can be missed. The checks are not a validated artifact-removal system. A quality flag does not identify the underlying cause. ADC rail clipping is not specifically classified.

Any flagged channel is excluded from that window's deviation score. Other eligible channels remain usable. Counts report windows containing at least one flagged channel, not the proportion of a recording proven unusable.

## Filtering and spectral estimation

- Optional 50 or 60 Hz notch (Q=30), only when safely below Nyquist.
- Fourth-order Butterworth 1–30 Hz bandpass, implemented as second-order sections.
- Zero-phase forward/backward filtering; this uses future samples and is offline only.
- Welch PSD: 2-second segments, default 50% overlap, constant detrending, SciPy's default periodic Hann window, density scaling.
- Band integration: sum PSD bins × frequency resolution. Nonoverlapping half-open ranges: delta [1,4), theta [4,8), alpha [8,13), beta [13,30) Hz.
- Relative power: a band's absolute power divided by the sum of these four band powers. Gamma is not analyzed.

The filter response attenuates frequencies near 1 and 30 Hz. View filtering uses five seconds of context at either side where available; results very near edges can differ slightly from full-recording filtering used for analysis. Waveforms use a min/max envelope when downsampling for display. Vertical scale is automatic per visible channel and is labeled in µV; compare numerical scales before comparing heights. The spectrum uses the visible interval and selected raw/filtered view, while analysis features always use the fixed processing pipeline.

[SciPy Welch documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)

## Reference comparison

Analysis windows last 4 seconds and advance by 2 seconds. Incomplete final windows are omitted. A reference interval must be at least 12 seconds and lie within the recording. Only windows fully inside it are eligible for reference estimation. Each channel requires five unflagged reference windows; adjacent windows overlap and are not independent.

For each channel and band:

1. Transform absolute power with log10(max(power, 1e-10)).
2. Estimate the reference median.
3. Estimate scale as max(1.4826 × MAD, 0.10), where MAD is the median absolute deviation from that median. The 0.10 log-power floor prevents unstable scores for almost constant reference features.
4. Compute absolute distance from the median divided by that scale.

The window score is the maximum over available channel/band pairs. The explanation identifies the pair and direction. Default threshold = 4. Scores are **not calibrated probabilities or ordinary Gaussian z-scores**. More channels mean more opportunities for a high maximum; thresholds are not comparable across montages without calibration. A selected reference is a within-recording comparison, not a certified healthy baseline.

Reference windows are labeled as reference rather than counted as deviations. If no channel is eligible in a window, its score is null. If no channel has a usable reference, analysis fails explicitly. Windows overlapping a reference boundary can share samples with it; this is exploratory within-recording analysis, not a held-out performance evaluation.

## Meaning and validation limits

Frequency bands are signal descriptors. An increase in a band does not uniquely identify attention, mood, cognitive status, medication response, or disease. Context and recording quality can change the same features.

No sensitivity, specificity, diagnostic accuracy, or false-alert rate has been established. The public sample is an import/integration check, not a disease benchmark. There is no training/test split because there is no trained model or performance claim in this release. A future learned model must split by participant/session, preserve a held-out cohort, report false alerts per hour and quality coverage, and prevent overlapping-window leakage.

The application is not suitable for diagnosing neurodegenerative disease, deciding treatment, seizure alerts, or safety-critical monitoring. Future symptom studies should define one measurable endpoint, pair recordings with appropriate symptom assessments, and test incremental value over other modalities.
