# Public EEG archive

The reviewed September 24, 2026 plan contains 23 public datasets from OpenNeuro and PhysioNet. The target is approximately 300 GB each of healthy/control EEG and diagnosed neurodegenerative EEG. This is a collection target, not a claim that sufficient eligible data exists.

Known raw EEG totals are 255.14 GB healthy/control and 140.79 GB neurodegenerative, plus up to 44.6 GB from bigP3BCI with healthy/control and ALS recordings. The actual bigP3BCI split is measured during transfer. The reviewed plan does not reach 300 GB of neurodegenerative data. GB means decimal bytes / 1,000,000,000.

## Cohort rules

Source descriptions and participant metadata determine labels. Included diagnoses are Parkinson's disease, Alzheimer's disease, frontotemporal dementia, and ALS. Epilepsy, post-cardiac-arrest coma, MCI without confirmed dementia, and asymptomatic genetic risk are not counted as diagnosed neurodegeneration. Controls follow each study's eligibility criteria; the label does not establish universal clinical normality.

Matched controls are retained. A deterministic whole-participant subset of ds004148 keeps the control budget below 300 GB while conservatively reserving the entire bigP3BCI size. It is not a representative statistical sample. Its omitted participant IDs and selection rule are recorded in the SSD manifest.

The manifest retains dataset, original path, diagnosis, label evidence, participant key, source URL, expected size, and available integrity evidence. Some studies share participants: ds004504/ds006036 are the same 88 people; New Mexico and Iowa tasks share source participant identifiers; NeuroEPO releases may overlap. Never split these overlapping recordings randomly between training and evaluation. Byte counts and dataset counts are not independent participant counts.

## Storage and provenance

Raw files are stored once under `datasets/{provider}/{dataset}/`, preserving associated headers and sidecars. The `cohorts/` JSONL indexes point to those files without duplicating signals. `download-plan.json` is the complete frozen transfer plan. `verified-files.jsonl` records successful files and SHA-256 fingerprints. `download-status.json` and `download.log` expose current progress. Source inventories are retained under `provenance/`.

OpenNeuro S3 objects are pinned with ETags and conditional requests. Single-part ETags are verified as MD5; multipart objects use ETag and size validation plus a locally computed SHA-256 receipt. PhysioNet files are verified against its SHA256SUMS. An OpenNeuro dataset DOI identifies a published dataset version but does not imply the current S3 snapshot is identical to that release. Preserve the actual object inventory for reproducibility.

Source metadata and licenses accompany the archive; software licensing is separate. No raw research data, participant tables, local archive paths, or download receipts are committed to GitHub. Aggregate sources and attribution are in [cohort-sources.json](cohort-sources.json).

## Running and stopping

From the project directory, with the external drive mounted:

```sh
.venv/bin/python -m scripts.start_archive \
  --root '/Volumes/Extreme Pro/EEG data' \
  --plan '/Volumes/Extreme Pro/EEG data/download-plan.json'
```

The detached worker resumes partial files, retries transient failures, checks free space, and avoids duplicate simultaneous workers. On macOS it prevents idle sleep while running; closing the lid, shutting down, losing power, or disconnecting the drive can still interrupt it. Re-run the command to resume after an interruption or retry failed files.

To pause safely, create an empty file called `PAUSE` in the archive folder. Wait for `download-status.json` to report `paused` before disconnecting the drive. To resume, remove only that empty `PAUSE` marker; the installed service checks again within a minute. Without the service, run the command above. The worker also stops if free space falls below 20 GiB. The installed macOS LaunchAgent resumes eligible work after login and checks every minute when no worker is running. It does not depend on Codex being open or having usage remaining.

The app's **Data archive** panel reads progress from the path in local `data/archive-path.txt` and aggregate metadata in `data/archive-summary.json`. It is read-only and does not restart downloads. Its timestamp is the last worker update; a stale timestamp can indicate an interrupted worker.

The signal explorer currently imports bounded EDF/CSV files. The archive also contains BDF, BrainVision and EEGLAB formats and large recordings. Download completion does not mean all files can be opened directly in this first application version. Format adapters, streaming excerpts, harmonization, and participant-level evaluation are subsequent pipeline work.

## Rebuilding inventories

`scripts/catalog_openneuro.py` lists public raw objects and participant metadata; `scripts/build_cohort_plan.py` applies the explicit reviewed dataset/cohort mappings. `scripts/catalog_physionet.py` inventories PhysioNet resources, while `scripts/download_physionet.py` is a checksum-based utility. Only the reviewed cohort manifest should be supplied to `scripts/archive_download.py`; a broad EEG catalogue is not a clinically eligible cohort plan.

## Autonomous service and retry recovery

The service is installed as `~/Library/LaunchAgents/local.eeg-explorer.archive.plist`. It uses the local project interpreter, not Codex, an AI API or a scheduled Codex task. Keep the project/virtual environment in place. It runs when the user is logged in, requires the SSD and network, and cannot transfer while the Mac is shut down or the lid causes sleep. `caffeinate -i` prevents idle sleep while a worker is active.

A September 25 audit found 1,280 failures, all reporting DNS name-resolution errors. Current DNS and a byte-range request to a previously failed source succeeded. The old status was preserved under `provenance/pre-retry-upgrade-status.json`. This supports a past network/DNS interruption; it does not establish why DNS was unavailable.

The upgraded worker imports those failures, prioritizes retries, and retains a persistent `retry-state.json` plus append-only `retry-history.jsonl`. It retries until files pass integrity checks or the archive is paused. DNS/connection failures back off the affected host, up to five minutes, while other hosts can progress. Rate limits honor numeric Retry-After. Missing/denied/changed sources are retained with an hourly retry and visible error rather than treated as completed. Failures are removed only after verified success. Completion requires every manifest entry to be verified, with an empty retry queue.

Install on another Mac from the project directory with:

```sh
.venv/bin/python -m scripts.install_archive_service --root '/Volumes/Extreme Pro/EEG data'
```

Service logs are under local `data/archive-service/`; download progress and retry history stay on the SSD. The service waits for the mounted drive rather than recreating its mount directory on the internal disk. A `PAUSE` file remains the safe stop control.

### Source-balanced scheduling (25 September 2026)

The three concurrent transfer slots are shared across eligible sources, preferring the source with fewer active transfers and using dispatch counts to break ties. A slow source cannot fill every newly available slot while another eligible source waits. A source on cooldown does not block healthy sources. The global concurrency limit, persistent retries, resumable partial files, source checksums/ETags and verification receipts remain enforced.

`download-status.json` now includes `received_bytes_by_host`, counting payload bytes received per source during the current worker run. Measure differences over a stable interval from the same run to estimate throughput; counters reset on restart and include bytes later rejected by integrity checks. Estimate completion separately for each source using remaining manifest bytes and receipts. PhysioNet entries with unknown lengths require an approximate source budget, so its ETA is provisional. Short measurements include metadata traffic and do not guarantee sustained throughput.
