"""v4 model families.

F0..F8 as sklearn-style estimators (fit(people, y) / predict_proba), the
non-negative-logistic stacker S, and Platt calibration on inner OOF
predictions.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.
"""
