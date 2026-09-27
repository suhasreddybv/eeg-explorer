"""Independent local v3 campaign; bounded stages and immutable implementation policy."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from pd_pipeline.registry import freeze, save
from pd_pipeline.preprocess import file_hash
from pd_pipeline.cohort_v3 import RUN

SOURCES = [
    "pd_pipeline/cohort_v3.py",
    "pd_pipeline/learning_v3.py",
    "pd_pipeline/compare_v3.py",
    "pd_pipeline/experiments.py",
    "pd_pipeline/registry.py",
    "pd_pipeline/preprocess.py",
    "scripts/run_pd_v3.py",
    "docs/PD_PROTOCOL_V3.md",
]


def policy():
    return {
        "code_and_protocol": {p: file_hash(Path(p)) for p in SOURCES},
        "registry_sha256": file_hash(RUN / "registry.json"),
        "max_stage_seconds": 96 * 3600,
        "scope": "PD versus controls; nested cross-site primary, within-site secondary; exploratory",
    }


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / "delivery-ready.json").exists():
            return
        stages = [
            ("preparation", "pd_pipeline.cohort_v3", ["prepare"], "preparation.json", None),
            ("ssl_benchmark", "pd_pipeline.learning_v3", ["benchmark"], "ssl-benchmark.json", None),
            (
                "ssl_pretraining",
                "pd_pipeline.learning_v3",
                [],
                "ssl-lock.json",
                "ssl-benchmark.json",
            ),
            (
                "supervised_benchmark",
                "pd_pipeline.compare_v3",
                ["benchmark"],
                "supervised-benchmark.json",
                None,
            ),
            (
                "supervised_training",
                "pd_pipeline.compare_v3",
                ["campaign"],
                "outer-models-lock.json",
                "supervised-benchmark.json",
            ),
            ("evaluation", "pd_pipeline.compare_v3", ["evaluate"], "results.json", None),
            (
                "final_research_models",
                "pd_pipeline.compare_v3",
                ["final"],
                "deployment/lock.json",
                "supervised-benchmark.json",
            ),
            ("aggregate_report", "scripts.report_pd_v3", [], "aggregate-report-ready.json", None),
        ]
        start = time.monotonic()
        state = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_stages": [],
            "status": "running",
        }
        try:
            for name, module, args, done, bench in stages:
                if (RUN / done).exists():
                    state["completed_stages"].append(name)
                    continue
                if name == "preparation":
                    bench = "preparation-benchmark.json"
                if bench and json.loads((RUN / bench).read_text())["forecast_seconds"] > 96 * 3600:
                    raise RuntimeError(
                        "USER_NOTICE_REQUIRED: " + name + " forecast exceeds 96 hours"
                    )
                if name != "preparation":
                    freeze(RUN / "implementation-policy.json", policy())
                if name != "preparation" and policy() != json.loads(
                    (RUN / "implementation-policy.json").read_text()
                ):
                    raise ValueError("Implementation changed during frozen campaign")
                state.update(
                    stage=name, status="running", updated_at=datetime.now(timezone.utc).isoformat()
                )
                save(RUN / "campaign-status.json", state)
                with (RUN / (name + ".log")).open("a") as log:
                    subprocess.run(
                        [sys.executable, "-u", "-m", module, *args],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=96 * 3600,
                    )
                if not (RUN / done).exists():
                    raise RuntimeError("Stage did not produce expected completion artifact")
                state["completed_stages"].append(name)
            state.update(
                status="complete_awaiting_review", elapsed_seconds=time.monotonic() - start
            )
            save(RUN / "delivery-ready.json", state)
        except Exception as e:
            state.update(
                status="needs_attention", error=str(e), elapsed_seconds=time.monotonic() - start
            )
            raise
        finally:
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            save(RUN / "campaign-status.json", state)


if __name__ == "__main__":
    main()
