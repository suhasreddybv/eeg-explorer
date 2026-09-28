"""v4 window QC on filtered signals.

Label-free threshold derivation from auxiliary cohorts, amplitude/step/
line-noise/flat checks, and extraction into data/pd-runs/v4/windows/.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.
"""
