"""v4 nested comparison.

folds() (4 leave-one-site-out outer folds + 3 within-site participant
folds, adapted from compare_v3), the D3 selection objective, campaign/
evaluate/final_fit, and negative controls (permutation, site-identity
probe, demographic/metadata baselines).

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.
"""
