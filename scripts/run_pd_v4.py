"""Lock-protected, stage-resumable v4 driver.

Hashes docs/PD_PROTOCOL_V4.md and every pd_pipeline/*_v4.py module into an
implementation policy before any supervised fit, mirroring
scripts/run_pd_v3.py. Stage-resumable via data/pd-runs/v4/campaign-status.json.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.

The full stage-orchestration campaign (Stages 3-11) is built out
incrementally as pd_pipeline/{cohort,qc,features,harmonize,models,compare,
reduced,fm,fusion}_v4.py stop being docstring stubs; SOURCES below already
covers every v4 module so the implementation policy captures the current
state at each freeze() call.
"""

import json
from pathlib import Path

from pd_pipeline.preprocess import file_hash
from pd_pipeline.registry import freeze

RUN = Path("data/pd-runs/v4")

SOURCES = [
    "docs/PD_PROTOCOL_V4.md",
    "pd_pipeline/cohort_v4.py",
    "pd_pipeline/qc_v4.py",
    "pd_pipeline/features_v4.py",
    "pd_pipeline/harmonize_v4.py",
    "pd_pipeline/models_v4.py",
    "pd_pipeline/compare_v4.py",
    "pd_pipeline/reduced_v4.py",
    "pd_pipeline/fm_v4.py",
    "pd_pipeline/fusion_v4.py",
    "scripts/run_pd_v4.py",
    "scripts/locked_site_v4.py",
    # Reused v3 modules the v4 pipeline imports from rather than copies.
    "pd_pipeline/cohort_v3.py",
    "pd_pipeline/learning_v3.py",
    "pd_pipeline/compare_v3.py",
    "pd_pipeline/experiments.py",
    "pd_pipeline/registry.py",
    "pd_pipeline/preprocess.py",
]


def policy():
    return {
        "code_and_protocol": {p: file_hash(Path(p)) for p in SOURCES},
        "max_stage_seconds": 96 * 3600,
        "scope": (
            "PD versus controls; T1/T2/T3 tiers; cross-site and within-site AUROC "
            "weighted equally; nested inner selection, locked outer evaluation; "
            "Turku (OSF pehj9) one-shot locked-site evaluation"
        ),
    }


def freeze_policy():
    """Stage 2: freeze the implementation policy once docs/PD_PROTOCOL_V4.md
    exists, before any supervised model is fitted. Stage 7+ campaign code
    re-checks policy() == the frozen file before every stage, as
    scripts/run_pd_v3.py does, and refuses to proceed if source changed."""
    RUN.mkdir(parents=True, exist_ok=True)
    current = policy()
    path = RUN / "implementation-policy.json"
    if path.exists():
        frozen = json.loads(path.read_text())
        if frozen != current:
            raise ValueError(
                "Implementation policy already frozen and differs from current source; "
                "this is a protocol deviation and needs a new protocol version, not a "
                "silent re-freeze."
            )
        return frozen
    freeze(path, current)
    return current


if __name__ == "__main__":
    print(json.dumps(freeze_policy(), indent=2))
