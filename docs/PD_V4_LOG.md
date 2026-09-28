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
