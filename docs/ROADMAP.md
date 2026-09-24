# Roadmap

## 1. Explore recordings locally — current prototype

Import and replay signals, inspect quality and spectral changes, select a reference interval, save observations, and export reproducible feature summaries. Keep every inference traceable to a time window, channel, and method version.

## 2. Repeatable personal recording sessions

- Define a brief eyes-open / eyes-closed / rest protocol.
- Record device, electrodes, reference, sampling rate, signal quality, and contextual factors consistently.
- Add side-by-side session comparisons only when montage, reference, and protocol are compatible.
- Validate reproducibility, recording failure rates, and day-to-day variability before personal trend claims.
- Support explicit reference selection from a prior session rather than implicitly treating all prior data as normal.

## 3. Wearable acquisition

- Start with an existing isolated, battery-powered EEG acquisition system.
- Add a device adapter and timestamped streaming interface.
- Implement causal filtering, disconnection handling, dropped-packet detection, and quality monitoring.
- Test reduced electrode coverage before deciding on a headband or around-ear design.
- Verify offline/live feature agreement within specified tolerances.

## 4. Multimodal symptom research

Preserve an integration path for audio and movement using an explicit session key. A shared key groups recordings; it does not synchronize their clocks. Real alignment will require a shared timing protocol, timestamp precision, clock-drift handling, and modality provenance.

Select one symptom endpoint and population before building a prediction model. Compare audio/movement baselines with and without EEG, use participant-level held-out evaluation, and measure whether EEG adds useful information. Disease detection and longitudinal symptom tracking require separate studies and claims.

## Before public release

- Choose a source-code license and contributor policy.
- Review tracked files and Git history for recordings, exports, identifiers, notes, and secrets.
- Recheck dataset licensing and attribution.
- Run tests from a clean clone and document known limits.
- Add installation screenshots and an issue template using only synthetic or permitted public examples.
- Keep exploratory descriptions separate from clinical claims.
