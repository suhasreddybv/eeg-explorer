"""Lock-protected, stage-resumable v4 driver.

Hashes docs/PD_PROTOCOL_V4.md and every pd_pipeline/*_v4.py module into an
implementation policy before any supervised fit, mirroring
scripts/run_pd_v3.py. Stage-resumable via data/pd-runs/v4/campaign-status.json.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.
"""
