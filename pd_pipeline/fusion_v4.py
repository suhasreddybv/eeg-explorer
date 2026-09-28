"""v4 T3 fusion-ready outputs.

The Myna EEG output schema writer (docs/myna_eeg_output_v1.schema.json),
the naive-Bayes LLR combiner (flagged unvalidated), and StackingFusion for
paired multimodal data.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.
"""
