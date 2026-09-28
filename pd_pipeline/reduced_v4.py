"""v4 reduced-channel montages (T2).

Re-referencing from preref128 within montage channels only, distillation
from the T1 stacker's inner-OOF probabilities, channel-dropout
augmentation, and the nested greedy channel-budget search.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.
"""
