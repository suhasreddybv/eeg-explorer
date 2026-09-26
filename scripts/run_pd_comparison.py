"""One local, lock-protected campaign; logs failures and never opens tests early."""

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from pd_pipeline.registry import save


def main():
    run = Path("data/pd-runs/v2")
    run.mkdir(parents=True, exist_ok=True)
    with (run / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "delivery-ready.json").exists():
            return
        started = time.monotonic()
        stages = [
            ("training", [sys.executable, "-u", "-m", "pd_pipeline.compare_v2", "campaign"]),
            (
                "test_preprocessing",
                [sys.executable, "-u", "-m", "pd_pipeline.cohort_v2", "test-prepare"],
            ),
            ("test_evaluation", [sys.executable, "-u", "-m", "pd_pipeline.compare_v2", "evaluate"]),
            (
                "subgroup_audit",
                [sys.executable, "-u", "-m", "pd_pipeline.diagnostics_v2", "subgroups"],
            ),
            ("report", [os.environ["EEG_REPORT_PYTHON"], "scripts/report_pd_comparison.py"]),
        ]
        state = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "completed_stages": [],
        }
        try:
            for name, command in stages:
                if name == "training" and (run / "all-models-lock.json").exists():
                    state["completed_stages"].append(name)
                    continue
                state.update(
                    stage=name, status="running", updated_at=datetime.now(timezone.utc).isoformat()
                )
                save(run / "campaign-status.json", state)
                with (run / (name + ".log")).open("a") as log:
                    remaining = 96 * 3600 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise RuntimeError("96-hour continuous stage bound reached")
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining
                    )
                state["completed_stages"].append(name)
            state.update(
                status="complete_awaiting_visual_review",
                elapsed_seconds=time.monotonic() - started,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            save(run / "campaign-status.json", state)
            save(run / "delivery-ready.json", state)
        except Exception as e:
            state.update(
                status="failed", error=str(e), updated_at=datetime.now(timezone.utc).isoformat()
            )
            save(run / "campaign-status.json", state)
            raise


if __name__ == "__main__":
    main()
