"""v4 Track E foundation-model probes (optional, gated).

CBraMod/LaBraM/REVE frozen embeddings, linear probes inside the nested
folds, the FM gate, and gated CBraMod fine-tuning. Only runs if Stage 7
finished and the 96-hour forecast allows it.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.
"""
