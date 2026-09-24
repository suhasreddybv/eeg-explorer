# EEG Explorer

A local-first application for looking closely at EEG recordings: replay signals, inspect frequency patterns, flag changes relative to a chosen reference interval, and record observations.

**Research prototype.** A deviation is a change in signal features, not a diagnosis, symptom measurement, or probability of disease. This application has not undergone clinical validation.

## Run locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv sync --frozen --python 3.12
uv run uvicorn eeg_explorer.app:app --host 127.0.0.1 --port 8765
```

Open **http://127.0.0.1:8765**. Keep the terminal running; use Control-C to stop.

On macOS, after the initial dependency setup, `start.command` launches the same server. Double-click it in Finder or run `./start.command` from this directory. This is a local web application, not a packaged native desktop app.

## First session

1. Choose **Synthetic walkthrough**. It creates a deterministic 120-second, four-channel signal with an alpha-power increase, blink-like pulses, and a flat channel. This is generated data, not a person's EEG.
2. Click the timeline around **00:44**. The insight panel identifies increased alpha power; the waveform and spectrum update to that interval.
3. Inspect **01:38** to see a flat O2 channel flagged separately from pattern changes.
4. Add an interval observation or save session context in **Notes & context**.
5. Choose **PhysioNet recording** for a real, public eyes-open recording. This button downloads S001R01 from the official dataset. An internet connection is needed only for this download and initial dependency setup.
6. Import a local EDF/EDF+ or CSV recording when ready. Files and observations remain on this computer.

## Included in v0.1

- Local library with EDF/EDF+ and CSV import; units normalized to microvolts.
- Explicitly labeled synthetic demo and on-demand public PhysioNet sample.
- Waveform replay, raw/filtered toggle, channel selection, min/max display envelopes, and Welch spectra.
- Four-second analysis windows, two-second steps, adjustable reference interval and deviation threshold.
- Raw-signal quality heuristics and channel-specific exclusion from anomaly scoring.
- Persistent annotations and session context, including an optional shared session identifier for future multimodal work.
- JSON export with recording provenance, context, annotations, analysis settings, and per-window results. Raw signals are not included in this export.
- Offline processing with no analytics, external scripts, account, or cloud upload.

There is no trained disease model, live-device connection, audio or movement analysis, cross-session statistical comparison, or clinical alerting in this release.

## Import contract

**EDF/EDF+:** Sampling rate, units, channel labels, and source annotations are read using MNE. EEG channels are selected from MNE's inferred channel types. Inspect the channel labels and source metadata; EDF files with missing channel-type information can include non-EEG signals. Original referencing is retained. No automatic rereferencing or resampling is applied by the pipeline; MNE may internally upsample mixed-rate EDF channels. Use uniformly sampled EEG inputs for interpretable comparisons.

**CSV:** First row: unique EEG channel names. Remaining rows: numeric samples, one column per channel. No timestamp or index columns. Specify the sampling rate and amplitude unit (µV, mV, or V). Missing/non-finite values are rejected, not silently filled.

Limits: 64 MB per upload, 1–128 channels, 80–2048 Hz, at least 24 seconds, and at most 12 million scalar samples. Use shorter excerpts for larger recordings. At most eight channels are displayed simultaneously; analysis uses all eligible imported channels.

## Analysis method

Full details and limits are in [docs/METHOD.md](docs/METHOD.md).

Raw-signal checks flag flat/repeated samples, large amplitudes, abrupt jumps, and strong 50/60 Hz power. Analysis uses an optional notch filter and a 1–30 Hz bandpass, both applied offline with zero phase. Welch estimates produce absolute delta, theta, alpha, and beta band power.

Each channel/band's log power is compared with its median and median absolute deviation in the selected reference interval. The displayed score is the maximum scaled deviation across eligible channel/band pairs. The default threshold of 4 is a heuristic, not a validated operating point. Channels lacking five unflagged baseline windows are excluded; an entirely unusable baseline returns an actionable error.

## Local data and privacy

The default `data/` directory holds compressed normalized signals and JSON metadata/notes. `EEG_EXPLORER_DATA` can point to another local directory. Data is not encrypted by the app; it relies on your OS account and disk protections. Only one application process should use a data directory at a time.

Raw EDF headers are not retained, but imported filenames, channel names, source event text, and notes can contain identifying information. Review these before sharing an export. The application's data directory, EEG files, exports, environment files, and development caches are excluded from Git. Do not force-add them.

The server binds to `127.0.0.1`. It is intended for one trusted local user and has no login system. Do not expose it to a network. Mutating API requests require a custom header, cross-origin mutations are rejected, and Host validation helps prevent DNS rebinding. These are local-use safeguards, not a deployment security model.

## Development and tests

```sh
uv run ruff check .
uv run pytest -q
```

Tests cover known spectral power, injected changes, artifact gating, unusable baselines, CSV unit conversion, EDF round-trip units/annotations, local persistence, malformed inputs, and local-request protection. CI runs the same checks with the committed `uv.lock`.

The backend uses FastAPI, NumPy, SciPy, and MNE. The interface is plain HTML/CSS/JavaScript with Canvas rendering; no Node build step or third-party browser runtime is required.

## Data attribution

The optional sample is **S001R01**, the eyes-open baseline in the **EEG Motor Movement/Imagery Dataset v1.0.0**, contributed by Gerwin Schalk and colleagues to PhysioNet. It contains 64 EEG channels sampled at 160 Hz. Dataset terms: **Open Data Commons Attribution License v1.0**. This repository does not bundle that dataset.

- [Dataset, documentation, license, and citation](https://physionet.org/content/eegmmidb/1.0.0/)
- Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. *BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.* IEEE Transactions on Biomedical Engineering. 2004;51(6):1034–1043. DOI: [10.1109/TBME.2004.827072](https://doi.org/10.1109/TBME.2004.827072).
- Schalk G. *EEG Motor Movement/Imagery Dataset (version 1.0.0).* PhysioNet. 2009. DOI: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P).
- Pollard T et al. *PhysioNet as a global platform for biomedical research.* Nature Health. 2026;1:792–795. DOI: [10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).

Downloaded source bytes and normalized arrays receive SHA-256 fingerprints for provenance. The source-file fingerprint is recorded at download time, not compared against a bundled authoritative checksum. TLS and the official source URL protect transport; the fingerprint helps identify the exact imported file later.

## Roadmap and publication

See [docs/ROADMAP.md](docs/ROADMAP.md) for the progression toward repeatable personal sessions, wearable acquisition, and symptom-specific multimodal studies.

This repository starts private. Before making it public, choose a source-code license, review the full Git history for private data, confirm dataset attribution, and update validation results. No open-source license is granted by this prototype yet; public visibility and open-source licensing are separate decisions.

## External research archive

The **Data archive** button shows live progress for a configured external archive. The resumable downloader preserves source layouts, participant/cohort provenance and integrity receipts while keeping raw data outside Git. See [archive scope, operation and limitations](docs/DATA_ARCHIVE.md) and the [reviewed source catalogue](docs/cohort-sources.json). The current collection falls short of the 300 GB diagnosed-neurodegenerative target; unrelated neurological conditions are not used to fill it.

## Parkinson’s classifier research plan

The [research-backed PD classifier proposal](docs/PD_CLASSIFIER_RESEARCH_PLAN.md) describes the initial neural model, participant-separated train/validation/test split, independent-study evaluation, local runtime estimates, cloud costs and later Myna integration. It is a decision draft; no classifier has been trained.
