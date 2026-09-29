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

## Stage 5: feature families (28 September 2026)

- Implemented `features_v4.py`: B-1 (relative spectra), B-2 (specparam/fooof — `specparam` 2.0 isn't released stable, using `fooof` per the protocol's documented fallback), B-3 (per-person Riemannian means; fold-dependent tangent-space mapping deferred to Stage 6/7 since it needs a training-fold reference), B-4 (LPC mean/SD; LEAPD hyperplane scoring deferred similarly), B-5 (entropy from cached `preref128`), B-6 (robust distributions/burst fraction), and the v3-rich baseline recomputed on v4 windows.
- **B-7 (connectivity) is not implemented yet** — explicitly noted in the module docstring rather than silently skipped; it's marked secondary/optional in the protocol and only needs `preref128` (already cached), so it can be added later without touching Stage 4's frozen windows.
- Verified every explicitly-required synthetic-signal property: specparam recovers a known 1/f exponent exactly on pure aperiodic input and within 0.1 with an added alpha peak; B-1 is exactly gain-invariant (×3 amplitude → ×9 power → identical vector, since every B-1 quantity is a ratio or spectral-shape descriptor); Riemannian re-centering maps a synthetic site mean to the identity to numerical precision; white-noise entropy exceeds sinusoid entropy on both sample and permutation entropy.
- **Caught and fixed two real bugs via testing, before either touched the real cached data:**
  1. `specparam_channel`'s aperiodic-adjusted band-power computation indexed a 40-bin frequency mask into a 39-bin fitted-range array (off-by-one from `FREQS>=2` vs the full `FREQS`), crashing on the very first real person tested.
  2. `entropy_features` used plain `np.mean`/`np.std`, so a single degenerate window where `antropy.higuchi_fd` returns NaN (a real edge case, hit on 1% of a 100-person sample of real data) poisoned that whole channel's entropy estimate. Fixed with per-metric fallback-on-NaN/exception plus `nanmean`/`nanstd` as defense in depth; verified 0/1312 people have any entropy NaN after the fix (was ~13 estimated before).
- Also fixed a real (if here inconsequential) eligibility-aggregation bug in `build_all`: A2's ≥10-window minimum must be evaluated on a person's *total* windows across all their ready records, not per-record — the original code could silently drop a person's shorter qualifying sessions. Confirmed the real archive's eligible-people count (1312: 324 supervised, 964 auxiliary, 24 Turku) was unaffected here, but the fix matters for correctness regardless.
- Cached label-free per-person features for all 1,312 A2-eligible people (~4 min real compute; Riemannian mean computation, not specparam, was the dominant per-person cost at ~1.24s for full-montage people).
- Added `tests/test_features_v4.py` (9 tests, all passing).
- No leakage test was added in this stage: `features_v4.py` deliberately contains zero fitted transforms (everything fold-dependent — scalers, tangent-space reference, LEAPD hyperplanes, normative models — is deferred to `harmonize_v4.py`/`models_v4.py`), so there is nothing to leakage-test yet; that test lands with those modules in Stage 6/7.
- `uv run ruff check .` and `uv run pytest -q` both green (110 passed).
- **Next:** Stage 6 (harmonisation: C1 Riemannian re-centering, C2 robust per-site standardisation, C3 normative z-scores, and the prevalence stress test).

## Stage 6: harmonisation and the prevalence guardrail (28 September 2026)

- Implemented `harmonize_v4.py`: C1 `recenter_riemann`/`rescale_dispersion`, C2 `site_statistics`/`robust_site_z`, C3 `normative_fit`/`normative_z` (spline quantile regression, 5th/50th/95th percentile, natural cubic spline on age with 4 df plus sex), `prevalence_stress`, and the descriptive `site_identity_probe`/`per_site_medians`.
- **Defined and documented the C3 "38 regional features"** the protocol deferred to implementation: 5 regions × 4 relative band powers (20) + 5 regional specparam exponents (5) + 5 regional offsets (5) + 5 regional alpha-adjusted powers (5) + 3 global spectral-shape descriptors (3) = 38. Names are enumerated by `features_v4.normative_feature_names()`.
- **Resolved two source-coding ambiguities from official BIDS sidecars rather than guessing:** ds004796 codes sex as `0=male, 1=female` (from its `participants.json` `Levels`), and ds004504's `Group` is `A=Alzheimer, F=Frontotemporal, C=Healthy`, so only the 29 `C` participants enter the normative pool.
- **ds004148 is a protocol-listed normative source that contributes zero people:** its entire cohort is aged 18–28, so none meet the ≥40y healthy-adult rule. Recorded in `NORMATIVE_SOURCE_NOTE` and tested, rather than silently dropped. Final normative pool: **467 healthy adults ≥40y** with cached features (360 ds005385 + 79 ds004796 + 29 ds004504 controls), before per-fold training controls are added.
- Caught a bug in my own diagnostic query that would have silently dropped all 29 ds004504 controls: those 88 people are canonically merged with ds006036 under one person key, so a `person → dataset` dict overwrites the label. `normative_source_people()` tests membership against the person's whole dataset set instead.
- **C3 sanity check on real data:** control median z = −0.04 (≈0 against a healthy normative model, as expected) vs PD median z = +0.22. Descriptive only, but the right direction.
- **Prevalence stress test, run on all four outer sites for both transductive methods:**

  | Method | Iowa | Tel Aviv | UCSD | UNM | Verdict |
  |---|---|---|---|---|---|
  | C1 (Riemannian re-centering) | 0.0019 | 0.0020 | 0.0039 | 0.0038 | **passes everywhere** |
  | C2 (robust per-site z) | 0.0067 | 0.0119 | **0.0444** | **0.0314** | **fails at UCSD and UNM** |

  Tolerance is 0.03. C2 fails at exactly the two smallest sites (UCSD n=31, UNM n=50), where a per-site median/IQR estimated from a subset is noisy and moves with class balance — precisely the failure mode the guardrail exists to detect. **Per the frozen protocol, C2 is therefore excluded from the pre-declared transductive pipeline**, so `S(T)` rests on C1 (family F3); the C2-based T-variants of F1/F2/F4/F5/F6 are reported descriptively but are not pre-declared. This decision was made by a pre-declared stability rule, not by looking at which method scored better.
### Reviewed decisions (28 September 2026)

Two interpretation gaps were raised for review before Stage 7 locks anything, and both were resolved by Suhas:

1. **Aggregation rule → majority of outer sites (≥3 of 4).** The protocol defines the per-site criterion but never how to aggregate. Note that 2 of 4 is a *tie, not a majority*, so C2's 2/4 still fails; the exclusion is unchanged from the conservative reading, just reached by a different route.
2. **Label-aware guardrail → run both and compare.** The frozen protocol computes AUROC on the held-out site, which necessarily reads that site's labels before the outer evaluation — a real qualification of non-negotiable #4. A label-free companion (`prevalence_stress_label_free`) was added: same resampling scheme, but the verdict comes from many *random synthetic labellings* rather than the true diagnosis, yielding a number in the same AUROC units and comparable to the same 0.03 tolerance. The frozen label-aware criterion remains primary and governs pipeline membership; the label-free result is a sensitivity check.

**Both criteria, side by side (spread; tolerance 0.03):**

| | Iowa | Tel Aviv | UCSD | UNM | Sites passing | Verdict |
|---|---|---|---|---|---|---|
| C1 label-aware (primary) | 0.0019 | 0.0020 | 0.0039 | 0.0038 | 4/4 | passes |
| C1 label-free (check) | 0.0009 | 0.0027 | 0.0046 | 0.0059 | 4/4 | passes |
| C2 label-aware (primary) | 0.0067 | 0.0119 | **0.0444** | **0.0314** | 2/4 | **fails** |
| C2 label-free (check) | 0.0056 | 0.0122 | **0.0355** | 0.0237 | 3/4 | passes |

- **C1 is robust: both criteria agree it passes at every site.**
- **C2's exclusion is NOT robust — the two criteria disagree.** Both agree UCSD fails; the disagreement is entirely about UNM, which sits just above tolerance under the label-aware criterion (0.0314) and just below it under the label-free one (0.0237). C2 is a genuinely borderline method, not a clear failure.
- **Decision applied:** the frozen (label-aware) criterion governs, so **C2 is excluded from the pre-declared transductive pipeline** and `S(T)` rests on C1 (family F3). The C2-based T-variants of F1/F2/F4/F5/F6 are reported descriptively only. The borderline nature of this call must be stated in the final report, because a defensible alternative criterion would have retained C2.
- Descriptive finding: the site-identity probe barely moves under C2 (macro-AUROC 0.774 → 0.775), i.e. C2 removes first-order per-site location/scale but substantial site information survives in higher-order structure.
- Added `tests/test_harmonize_v4.py` (10 tests) including the protocol's required invariant case — a site-independent scorer gives spread exactly 0 — plus a positive control that a prevalence-sensitive scorer is flagged.
- `uv run ruff check .` and `uv run pytest -q` both green (120 passed).
- **Next:** Stage 7 (T1 nested comparison, stacking, negative controls; gates G2–G4) — the first stage that fits supervised models, so the implementation policy must be re-frozen and the outer lock respected.

## Stage 7: T1 nested comparison, stacking, negative controls (28 September 2026)

- Implemented `models_v4.py` (F0–F6 as sklearn Pipelines, the `LeapdScorer`, the non-negative stacker `S`, Platt calibration) and `compare_v4.py` (person loading, `folds`, the D3 objective, `FoldContext`, `campaign`, `evaluate`, `negative_controls`).
- **Not implemented, and recorded rather than passed over:** F7 (attention-MIL) and F8 (connectivity logistic, blocked on B-7 from Stage 5). The stacker therefore combines F0–F6. Since the protocol writes S's inputs as "F0–F6 (+F7, +FM if gated)", F0–F6 is the core it specifies, but the absence must be stated in the final report.
- **Leakage boundaries enforced, not assumed.** Every fitted transform lives inside a training-fold Pipeline or in a `FoldContext` built from training people only. A sentinel test (`test_no_fitted_transform_sees_held_out_people`) poisons every held-out person's features with NaN and asserts the training-row design matrix is bit-identical, proving the C3 normative fit never read a test person.
- **Performance optimisation, verified neutral:** `FoldContext` is memoised by (regime, training-people set) — the C3 normative model and tangent reference depend only on the training set, not the family or hyper-parameter, so they were being refit identically dozens of times. D3 scores verified bit-identical with and without the cache; runtime fell ~5× (89.2s → 17.9s for a seven-family pass).
- **Benchmark and gate:** 53.7 s/fold measured on one cross-site and one within-site fold → 7 folds × 1.25 margin ≈ **0.13 h forecast**, far under the 96-hour gate.
- **Lock order respected:** all outer models fitted → `outer-models-lock.json` frozen → only then `evaluate()` generated outer predictions. `evaluate()` raises if the lock is absent (tested).

### Results — pre-declared pipeline first

**Primary (`S|I`, pre-declared): cross-site 0.759, within-site 0.779, composite 0.769.**
Secondary (`S|T`): cross-site 0.765, within-site 0.782, composite 0.774.

| Variant | Cross-site | Within-site | Composite |
|---|---|---|---|
| F2 specparam elastic-net (I) | 0.774 | 0.801 | 0.788 |
| **S\|T** (pre-declared secondary) | 0.765 | 0.782 | 0.774 |
| F0 rich-SVM (I) — v3 baseline recomputed | 0.758 | 0.785 | 0.771 |
| **S\|I** (pre-declared primary) | **0.759** | **0.779** | **0.769** |
| F6 boosting (I) | 0.783 | 0.738 | 0.760 |
| F3 Riemannian (I) | 0.718 | 0.748 | 0.733 |
| F3 Riemannian (T) | 0.737 | 0.724 | 0.730 |
| F1 relative-spectra SVM (I) | 0.677 | 0.714 | 0.696 |
| F5 entropy (I) | 0.607 | 0.663 | 0.635 |
| F4 LEAPD (I) | 0.520 | 0.640 | 0.580 |

F2's 0.788 is the highest observed composite, but it is a **post-hoc observation, not a pre-declared result** — exactly the distinction v3 drew between its 0.763 best family and its 0.690 selection procedure. The honest headline is S|I = 0.769.

### Gates

- **H1 / G4 — met.** S|I beats v3's pre-declared pipeline on both components (cross 0.759 vs 0.690, within 0.779 vs 0.693; composite 0.769 vs 0.692, **+0.078**) and clears the "at least 0.75 on each" target. **Caveat that must travel with this number:** v3 and v4 differ in cohort (440 vs 324 eligible people, a direct consequence of A1's one-session rule), QC thresholds and features, so this is not a clean architecture-only comparison.
- **G2 — NOT met. H2 unsupported.** Best B-only family (F2) inner-CV composite 0.7574 vs F0 (recomputed v3-rich) 0.7506 = **+0.0068**, well short of the required +0.02. The new aperiodic/relative/LPC families did **not** add materially over the v3-rich baseline on inner CV. Operationalised as best-B-family vs F0 because the gate has no pipeline consequence (S runs either way) and the margin is nowhere near the threshold; the full inner table is in `gates.json`.
- **G3 — NOT met on gain. H3 unsupported.** S|T improves cross-site by only +0.006 (<0.02). The stress-test component did pass for C1. Note F3 itself was *worse* transductively (0.730 vs 0.733 composite), so the small stack-level gain is not a clean C1 effect.

### Negative controls (A5)

- **Permutation null** (labels permuted within site, 20 permutations): mean 0.504, SD 0.051, max 0.580. The real cross-site 0.759 sits roughly 5 SD above — the signal is not an artifact of the fitting procedure.
- **Age/sex only:** 0.541 cross-site — weak, consistent with v3.
- **Site-identity probe** (macro-AUROC, lower is better): v3-rich 0.948, B-1+B-6 0.922, **B-2+C3 0.834**, B-3 tangent 0.964, B-4 0.945, B-5 0.961. Every representation still carries substantial site information; this is a material limitation for any cross-site claim. Coherently, the block that leaks least site identity (B-2+C3) is also the best cross-site transferer (F2).
- **Recording metadata:** covered structurally — A1 makes session count constant, so it carries zero information by construction (Stage 3).

- Added `tests/test_compare_v4.py` (10 tests). `uv run ruff check .` and `uv run pytest -q` green (132 passed).
- **Next:** Stage 8 (Track E foundation models, optional/gated) or Stage 9 (T2 reduced-channel montages).

## Protocol v4.1 correction and Stage 4-7 re-run (28 September 2026)

- **Bug found while preparing Stage 9.** Stage 4 built `preref128` and the per-window `lpc` array from the *unfiltered* resampled signal, contradicting the frozen protocol's filtered-window rule. The effect was material: in one UCSD recording **40.7% of `preref128` power sat below 1 Hz**. Since Stage 9's montages are built entirely from `preref128`, this had to be fixed before T2 could mean anything.
- **A tempting shortcut was tested and rejected.** Filtering the stored 128 Hz array on the fly would have avoided a re-run, but it only approximates the protocol: correlation 0.956 with the correct path, yet **29.5% off in RMS**. Using the array raw was far worse (correlation 0.189). So the array was rebuilt properly.
- **Fix:** both arrays now derive from `filtered_preref` — band-passed 1–40 Hz but deliberately *not* average-referenced, which is precisely what `preref128` exists for. Verified in the rebuilt data: UCSD sub-1 Hz power **40.7% → 11.8%**, in-band 51.4% → 87.7%, and the across-channel mean remains non-zero (still un-referenced). `raw200` corrected the same way. Regression-tested.
- **Provenance.** The fix follows solely from the protocol text, not from any outcome — but it was found *after* Stage 7's outer results had been seen, so the prior run is preserved under `data/pd-runs/v4/superseded-v4.0/` and everything from the re-run is labelled protocol v4.1.

### What changed, and what didn't

| Variant | v4.1 composite | superseded | Δ |
|---|---|---|---|
| F5 entropy | 0.684 | 0.635 | **+0.049** |
| F4 LEAPD | 0.570 | 0.580 | **−0.010** |
| F0 / F1 / F2 / F3 / F6 | unchanged | | +0.000 |
| **S\|I (primary)** | **0.770** (cross 0.765, within 0.774) | 0.769 | +0.000 |

- **A stated hypothesis was refuted.** The expectation was that the unfiltered-LPC bug explained F4/LEAPD's near-chance cross-site result. On corrected data F4 got *slightly worse*. LEAPD genuinely underperforms on this cohort; the bug was not the explanation. F5/entropy is what actually benefited.
- **The selectivity is itself a check.** Only F4 and F5 moved — exactly the two families consuming the corrected arrays. F0/F1/F2/F3/F6 are identical to three decimals because they derive from the `tokens`/`cov` arrays the fix did not touch.
- **No gate changed.** G2 still not met (+0.0068, identical — both F2 and F0 use untouched arrays; H2 unsupported). G3 still not met (+0.007; H3 unsupported). G4/H1 still met: S|I composite 0.770, +0.078 over v3's pre-declared 0.6915, clearing the ≥0.75 target on both components. The correction cost ~2 hours of recompute and changed no conclusion — which is now demonstrated rather than assumed.


## Stage 9: T2 reduced-channel montages (28 September 2026, protocol v4.1)

Montages are rebuilt from `preref128` and re-referenced to **their own channels only**. This is the crux of T2: v4's 17-channel features use an average reference over all 17 electrodes, so reusing them for a 4-channel montage would encode electrodes the device does not have. A test asserts montage output is unchanged when every non-montage channel is scrambled.

| Montage | Ch | Electrodes | Cross-site | Within-site | Composite | vs T1 (0.770) |
|---|---|---|---|---|---|---|
| **M4F** forehead headband | 4 | Fp1 Fp2 F7 F8 | 0.561 | 0.574 | **0.567** | −0.202 |
| **M4T** around-ear | 4 | T7 T8 P7 P8 | 0.769 | 0.776 | **0.772** | +0.003 |
| **M5S** spread (Suuronen 2023) | 5 | F8 F4 T7 O1 O2 | 0.777 | 0.772 | **0.775** | +0.005 |
| **M8** compact cap | 8 | Fp1 Fp2 F7 F8 T7 T8 O1 O2 | 0.761 | 0.767 | **0.764** | −0.005 |

- **G6: PASSED.** Best pre-declared montage (M5S, 0.775) sits within 0.005 of T1 — far inside the 0.05 allowance.
- **The finding that matters: placement dominates channel count.** Four around-ear electrodes (M4T) match a 17-channel cap. Four *forehead* electrodes (M4F) collapse to near-chance at every site — Iowa 0.540, Tel Aviv 0.519, UCSD 0.585, UNM 0.600. Since the forehead headband is the most common consumer form factor, this is a direct negative signal for that design and the single most actionable result in v4 so far.
- **M4T/M5S do not beat T1.** The +0.003/+0.005 margins are well inside participant-bootstrap noise. The defensible claim is that they are statistically indistinguishable from the full montage, not superior. Reading these as "fewer channels are better" would be wrong.
- **Distillation (M8): inconclusive, not a negative result.** The distilled student scored 0.042 below the hard-label student, but the comparison is confounded by two implementation weaknesses that are recorded rather than hidden: (i) the teacher used T1's *outer* within-site predictions because the protocol's per-fold inner-OOF vectors were not persisted by the Stage 7 campaign — still out-of-sample for the teacher, but not what the protocol specifies; and (ii) the distilled student is a Ridge regressor on soft targets while the hard-label student is a logistic regression, so model class and target type differ simultaneously. A clean test needs persisted inner-OOF teachers and a single model class trained on both target types.
- **Scope decisions recorded in `reduced_v4.montage_families()`:** F0 and F6 are excluded from T2 (both are 17-channel-only constructions, so neither is a wearable claim), and montage F2 uses B-2 alone rather than B-2+C3, because the C3 z-scores come from a 17-channel normative model and scoring a 4-channel montage against them would reintroduce absent electrodes. Refitting normative models per montage over the 467-person auxiliary pool was out of scope for this run.
- **Not run:** the greedy channel-budget search (k=2…8). It is explicitly descriptive — G6 uses only the pre-declared montages — and an exhaustive search over arbitrary subsets requires montage features recomputed per candidate subset for all 324 people.
- `uv run ruff check .` and `uv run pytest -q` green (141 passed).

## Stage 8: Track E foundation models (28 September 2026, protocol v4.1)

**Licences were checked before any weights were downloaded**, per the protocol's stop-and-ask rule. Two of three candidates were dropped:

| Model | Licence | Outcome |
|---|---|---|
| CBraMod `braindecode/cbramod-pretrained` | BSD-3-Clause, ungated | **used** (4,951,202 params; accepts 17×800 @ 200 Hz directly) |
| LaBraM `braindecode/labram-pretrained` | BSD-3-Clause | **skipped (technical)** — the checkpoint's `temporal_embedding` is `[1,16,200]` (16 s windows) against braindecode's `[1,5,200]` for a 4 s window, unresolved by changing `n_times`. v4's 4 s window is fixed by the frozen protocol, so the window was not reshaped to suit the checkpoint. |
| REVE `brain-bzh/reve-base` | declared only as **"other"**; companion `reve-positions` declares none | **skipped (licence)** — the protocol's unclear-licence rule |

### The FM gate fails on all three conditions

| Condition | Result |
|---|---|
| Beats both negative controls | **NO** — probe 0.641 vs **random-init 0.696**, permuted 0.434 |
| Adds ≥0.01 composite to the stack | **NO** — S+FM 0.697 vs S(I) 0.770, i.e. **−0.073** |
| Site-identity probe no worse than v3-rich | **NO** — FM 0.969 vs v3-rich 0.948 |

- **The pretrained weights were beaten by the same architecture with random weights** (0.696 vs 0.641). Both clear the permuted-label floor (0.434), so there is real signal — but none of it is attributable to CBraMod's pretraining here. Because the dimensionality, probe, folds and pooling are identical between the two arms, that specific comparison is internally fair.
- **Adding the FM actively degraded the stack** by 0.073, and its embedding carries *more* site identity than v3-rich — the opposite of what a transferable representation should do.
- **`S+FM` is therefore excluded.** `S(I)` remains the pre-declared primary, as it would have regardless: Track E can only ever produce a secondary pipeline.

**Two caveats that qualify this as a negative result about *this run*, not about CBraMod:**
1. **Channel positions were not verified against CBraMod's expected layout.** Sampling rate (200 Hz) and unit scaling (÷100 µV) follow the documentation, but the 17 channels were passed in v3's order without confirming the model's expected montage/positional encoding. That is a real deviation from the protocol's "follow each model's documented input preprocessing … channel naming/positions" and is a plausible cause of the poor showing.
2. **The pooled embedding is 27,200-dimensional for 324 people** — a severe p≫n regime even at C=0.1. This applies equally to both arms, so it does not explain random-init beating pretrained, but it does cap how much any probe here could achieve.

## Stage 10: T3 fusion outputs, cognition, reliability (28 September 2026, protocol v4.1)

- **Exported quantity is a prior-free log-likelihood ratio**, not a probability: `logit(calibrated p) − logit(training prevalence)`. A calibrated probability bakes in this cohort's ~64.5% PD rate, an artefact of how these research cohorts were assembled that has nothing to do with whoever Myna eventually scores.
- **The naive-Bayes combiner is guarded in code, not footnoted.** `combine_llr` refuses to run without `acknowledge_unvalidated=True`, because EEG, speech and movement in PD are all downstream of one disease process and are plainly not conditionally independent. `demonstrate_overcounting()` shows the failure rather than asserting it: a **near-duplicate** modality inflates combined evidence to 2.31, *more* than a genuinely **independent** one at 2.07, despite carrying almost no new information. `StackingFusion` is the alternative once paired data exist.
- `docs/myna_eeg_output_v1.schema.json` written, with structural validation and a `research_use_only` constant that cannot be set false.

### Cognition index (H6, exploratory, within PD only)

| Target | Features | Grouped-CV ρ (95% CI) | Cross-site ρ: Iowa | Cross-site ρ: Tel Aviv |
|---|---|---|---|---|
| MoCA | T1 (17 ch) | +0.061 (−0.114, 0.230) | +0.259 (p=0.016) | +0.063 (ns) |
| **MoCA** | **M8** | **+0.253 (0.086, 0.406)** | **+0.378 (p=0.0003)** | +0.212 (ns) |
| UPDRS-III | T1 (17 ch) | +0.307 (0.137, 0.453) | +0.012 (ns) | −0.084 (ns) |

- **The UPDRS-III row is a trap and must not be reported as a finding.** Its grouped-CV ρ of +0.307 looks the strongest in the table, but cross-site transfer is *null in both directions* (+0.012, −0.084). Grouped CV pools both sites, so a model that learns site-level offsets in UPDRS-III distribution scores well without any clinical signal. The cross-site evaluation is the honest one, and it says there is nothing here.
- **MoCA on M8 is the only credible signal**: positive in both evaluation types, and — notably — stronger than the full 17-channel montage. Tel Aviv remains underpowered (n=50, ns).

### Test-retest reliability (H6)

- **ICC(2,1) of the exported LLR = 0.436** across 18 Iowa participants with a usable second session (between-session Spearman 0.484). Repeat sessions are absent from the v4 registry by construction (A1 keeps one per person), so they were extracted on demand; each person's two sessions were scored by a model trained on the three non-Iowa sites, so neither session was ever in training.
- **This misses H6's ≥0.6 target.** At n=18 the interval is wide, but 0.436 is moderate at best, and it is the single most important number for any Myna *tracking* claim: session-to-session variability is substantial relative to between-person differences. A cross-sectional classifier is better supported by this work than a progression monitor.
- `uv run ruff check .` and `uv run pytest -q` green (153 passed).
