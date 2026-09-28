"""Generates docs/PD_V4_RESULTS.md and docs/pd-v4-results.json.

Refuses to run until outer-models-lock.json, final-models-lock.json and
locked-site-results.json exist in data/pd-runs/v4/.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.
"""
