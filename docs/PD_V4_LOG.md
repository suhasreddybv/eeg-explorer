# PD v4 log

Aggregate, no-participant-data stage log. Full artifacts stay under ignored `data/pd-runs/v4/`.

## Stage 0: orientation and baseline reproduction (28 September 2026)

- SSD mounted; archive audit (`cohort_v2.audit_archive`) reconciled all 41,746 planned files against receipts with zero mismatches.
- `download-status.json` reports `"state": "complete"`.
- Branch `v4-poc` created; v4 module/script scaffolding added (`pd_pipeline/*_v4.py`, `scripts/{run_pd_v4,report_pd_v4,report_pd_v4_pdf,locked_site_v4}.py`), docstrings only.
- **Gate G1: passed.** Reproduced the frozen v3 rich-feature SVM (C=10) baseline two independent ways — (a) recomputing equal-site AUROC directly from `data/pd-runs/v3/outer-predictions.json`, bypassing the cached `results.json` aggregate, and (b) re-running `compare_v3.fit()`/`infer()` against the locked weights for every outer fold. Both reproduce the v3 report's numbers within noise:
  - cross-site: 0.7627 observed vs 0.763 target (|Δ| = 0.0003)
  - within-site: 0.7113 observed vs 0.711 target (|Δ| = 0.0003)
  - Both well inside the ±0.01 tolerance. See `data/pd-runs/v4/baseline-repro.json` (ignored).
- Added `tests/test_v4_baseline.py`: fold site sets/grouping/disjointness match the frozen v3 `splits.json` exactly.
- `uv run ruff check .` and `uv run pytest -q` both green (72 passed).
- **Next:** Stage 1 (Turku/OSF `pehj9` locked-site acquisition and overlap audits). This is the first stage touching an external network source and a new no-licence data source (OSF shows no licence for the Turku data) — flagging per protocol before starting.

## Stage 1: locked-site acquisition and overlap audits (28 September 2026)

- **1a. Turku (OSF `pehj9`) download.** Fetched exactly the 40 raw `NN_eyesOpen.set` files (never `*_PREP_preprocessed.set`) plus `InfoAboutParticipants.xlsx` and the preprocessing `.m` file, matching the plan's expected counts exactly (159 total objects in the OSF `data/` folder; 40 raw eyes-open files, 1.619 GB). Every file's byte count and sha256 verified against OSF's reported values before being accepted; `manifest.json` written under `/Volumes/Extreme Pro/EEG data/locked-sites/osf-pehj9/`.
  - **Licence note for Suhas:** OSF project `pehj9` shows no licence. Recorded per protocol: internal, non-commercial research evaluation only; no redistribution; contact Dr Henry Railo (University of Turku) before any commercial or publication use.
- **Format verification (label-free):** `mne.io.read_raw_eeglab` opened all 40 files; all 17 v3 channels present with no alias needed; sampling rate 500 Hz for every file; duration 119–179 s (mean 152 s, ≈2–3 min eyes-open rest). See `data/pd-runs/v4/turku-format-verification.json`.
- **Label sealing:** `InfoAboutParticipants.xlsx` has an undocumented `PD group` column coded 0/1, with no key anywhere in the source files or the preprocessing script for which value means PD. Resolved without guessing: `LEDD` (a Parkinson's-medication dose) and `PD duration` — fields meaningless for a healthy control — are populated only for `PD group == 1` (20/20 with LEDD, 16/20 with duration) and never for `== 0`, so `1 = PD`, `0 = control` (20/20 split, matching the 40 eyes-open files exactly). This resolution rule is now code (`locked_site_v4.resolve_pd_group_coding`, unit-tested including the case where the numeric coding is flipped and the case where it's genuinely ambiguous and must raise) rather than a one-off read. Labels sealed to `locked-sites/osf-pehj9/sealed/labels.json`; `load_labels()` verified to refuse until both `outer-models-lock.json` and `final-models-lock.json` exist.
- **1b. ds008875 (Iowa, Visser/Narayanan/Uc)** confirmed to exist via the OpenNeuro API (156 PD / 78 control / 18 DBS, ~10 GB) but not downloaded; excluded from v4 as a likely overlap with ds008768/ds004584, recorded in `overlap-audit.json`.
- **1c. Turku-to-archive overlap check:** none of the 40 Turku raw-file sha256 hashes match any of the 19,888 unique hashes in the main archive's `verified-files.jsonl` — Turku was never used by v1–v3.
- Added `tests/test_locked_site_v4.py` (7 tests): the label loader's two-lock refusal, and the PD-group coding resolution logic (including a flipped-coding case and an ambiguous-input case that must raise, not guess).
- Added the `v4` optional dependency extra (pyriemann, fooof — specparam 2.0 isn't released stable yet, antropy, lightgbm, mne-connectivity, openpyxl, braindecode); `uv.lock` refreshed; `uv run ruff check .` and `uv run pytest -q` both green (79 passed).
- **Next:** Stage 2 (protocol v4 freeze — `docs/PD_PROTOCOL_V4.md` — must be written and hashed before any supervised model is fitted).

## Stage 2: protocol v4 freeze (28 September 2026)

- Wrote `docs/PD_PROTOCOL_V4.md`, modelled on `PD_PROTOCOL_V3.md`: hypotheses (H1–H6), cohort/session/duration rules (A1/A2), QC v4 rule (numeric thresholds pending Stage 4's label-free derivation — the rule is frozen now, values appended before any supervised QC report), feature families B-1…B-7 and v3-rich, harmonisation regimes C1–C3 and the prevalence stress test, the F0–F8/S model grid and D3 selection objective, the Turku locked-site licence/label-sealing summary from Stage 1, the T2 montage table and re-referencing rule, the T3 output schema/fusion approach, evaluation design, and all six gates (G1–G6) with G1 already resolved.
- Implemented `scripts/run_pd_v4.py:policy()`/`freeze_policy()`, hashing the protocol doc and every `pd_pipeline/*_v4.py` module (plus the reused v3 modules) into `data/pd-runs/v4/implementation-policy.json`, mirroring `run_pd_v3.py`. Frozen; a re-freeze after any source change now raises rather than silently overwriting.
- Added `tests/test_run_pd_v4.py`: freeze is idempotent, and refuses when the hashed source changes after freezing.
- `uv run ruff check .` and `uv run pytest -q` both green (81 passed).
- **Next:** Stage 3 (cohort v4 registry — `cohort_v4.build()`, session selection, Turku `locked_test` records, confound audit). This is the first stage that starts implementing real pipeline logic against the frozen protocol rather than scaffolding/documentation.

## Stage 3: cohort v4 registry (28 September 2026)

- `cohort_v4.build()` reuses `cohort_v3.build` **unmodified** (`cohort_v4.build_v3 is cohort_v3.build`, asserted in a test) to get archive extraction/labelling/dedup, redirecting its intermediate artifacts to a v4-owned scratch path so `data/pd-runs/v3/` is never touched (verified: v3's `registry.json`/`usage-summary.json` mtimes still 27 Sep, unchanged after the run).
- **A1 one-session-per-person** implemented per site (UCSD: PD→`ses-on`/control→`ses-hc`; New Mexico: PD→the session with `sess{1,2}_Med == "ON"`/control→their single session; Iowa: earliest by `days_since_first_session` from each subject's `_sessions.tsv`; Tel Aviv: already one recording, no `ses-` subfolders). Raises rather than guesses whenever a person doesn't resolve to exactly one recording.
- **Result: 537 supervised people, one record each** — matches v3's independently-reported "537 assigned people" exactly, and every site×diagnosis cell (Iowa 110/202, Tel Aviv 28/116, UCSD 16/15, UNM 25/25) matches v3's reported assigned counts exactly. Strong cross-check that A1 didn't drop or duplicate anyone.
- Added 40 Turku `locked_test` records (label/age/sex all `None` — genuinely absent from the registry, not just hidden) from the Stage 1 manifest.
- **Confound audit** (`data/pd-runs/v4/confound-audit.json`): session count is now `{1}` for every supervised person (structurally eliminates v3's UCSD/UNM session-count confound, which previously gave metadata AUROC 1.000, by construction rather than by a model choice). Age/sex-only leave-one-site-out AUROC is now near chance everywhere (Iowa 0.539, Tel Aviv 0.486, UCSD 0.496, UNM 0.510) — the expected result per protocol §2.2. Accepted-window/acceptance-fraction components are deferred to a Stage 4 addendum since they require QC-extracted windows that don't exist yet.
- Added `clinical.json` (Iowa: age-at-visit/MoCA/UPDRS-III/disease-duration/LEDD from the chosen session's `_sessions.tsv`; Tel Aviv: same fields directly from `participants.tsv`) — 456 entries, kept separate from the registry, used only in Stage 10.
- Added `tests/test_cohort_v4.py` (10 tests): per-site A1 rules, the ambiguous-case raise, Turku label-absence, and clinical-data parsing.
- `uv run ruff check .` and `uv run pytest -q` both green (91 passed).
- **Next:** Stage 4 (QC v4 — label-free threshold derivation on auxiliary cohorts, then full-archive re-extraction). This is a full-archive compute stage requiring its own benchmark and 96-hour forecast gate before launch.

## Stage 4: QC v4 and full-archive re-extraction (28 September 2026)

- Implemented `qc_v4.py`: amplitude/step checks now run on the **filtered** window (v3 ran them on the raw resampled window — verified with a regression test injecting a huge 90 Hz burst that a filtered check correctly ignores but a raw check would not); flat/repeat unchanged on the raw resampled window; a new line-noise check (45–65 Hz / 1–40 Hz power ratio) runs on the native-rate raw signal before resampling, returning "not applicable" when Nyquist can't cover 65 Hz. Added per-channel LPC (order 12, Levinson–Durbin autocorrelation — recovers a synthetic AR(2) process to within 0.02), band covariances (fbcov), and `preref128` storage for Stage 9.
- **Caught and fixed a real bug via testing**, not in production: `spectral_features()` hardcoded 17×17 covariance arrays, which crashed on auxiliary recordings with fewer than 17 available channels (auxiliary only requires a minimum of 12). Fixed and covered by a regression test (14-channel partial montage) before any real extraction ran.
- **Benchmark** (20 files across edf/vhdr/set/bdf + Turku's set format): ~0.31s/file for the stats-only pass, ~0.65s/file for full extraction. Forecast ~5.9h total (well under the 96h gate) — actual full run finished in well under half that.
- **Threshold derivation** (label-free, auxiliary cohorts only, 389,702 clean windows across 10,590 recordings, ~21 min): all three thresholds hit the protocol's ceiling caps — `PTP_UV=250`, `STEP_UV=60`, `LINE_RATIO=0.5` — meaning the raw 99th-percentile values were even higher than the ceilings. Frozen to `qc-thresholds.json` before any supervised or Turku recording was processed.
- **Full extraction** (11,167 records, ~59 min actual): 7,893 ready / 3,274 excluded overall.
  - Supervised: 379/537 people ready (12,564 total windows); 324/537 meet the A2 ≥10-window minimum (compare to v3's 440 — lower, and expected: A1's one-session-per-person rule removes the "backup session" v3 had if a person's first recording failed QC).
  - **Turku: 27/40 raw recordings usable (818 windows)** under thresholds derived entirely from auxiliary cohorts, never seeing Turku or PD/control labels — the locked-site evaluation will have real people to score, not a near-empty test set.
  - **ds003775: 0/153 ready**, exactly matching v3's finding under completely different (data-driven, not hardcoded) thresholds — a genuine recording-quality issue with that source, not a threshold artifact.
- Two self-inflicted process hiccups during this stage, both caught immediately and not indicative of a real cache/integrity bug: stale-identity exceptions from my own placeholder-threshold smoke tests polluting the resumable cache before the real run. Cleared and relaunched cleanly each time; the cache's refuse-rather-than-silently-overwrite behavior (mirroring v3) worked exactly as intended.
- Added `tests/test_qc_v4.py` (10 tests: LPC recovery, line-noise discrimination, the filtered-vs-raw amplitude behavioral difference, partial-montage shape regression, threshold percentile/ceiling logic, and `qc_audit`'s A2-minimum/ds003775/Turku reporting).
- `uv run ruff check .` and `uv run pytest -q` both green (101 passed).
- **Next:** Stage 5 (feature families B-1…B-7 and the recomputed v3-rich baseline, on the now-cached v4 windows).
