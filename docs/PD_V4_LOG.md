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
