"""Install the approved, one-shot local PD comparison as a macOS user job."""

import os
from pathlib import Path
import plistlib
import subprocess
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    run = project / "data/pd-runs/v2"
    label = "local.eeg-explorer.comparison.v2"
    report_python = Path(
        "/Users/suhas/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
    )
    if not report_python.exists():
        raise FileNotFoundError("Bundled PDF runtime missing")
    plist = {
        "Label": label,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-i",
            sys.executable,
            "-u",
            "-m",
            "scripts.run_pd_comparison",
        ],
        "WorkingDirectory": str(project),
        "RunAtLoad": True,
        "ProcessType": "Background",
        "EnvironmentVariables": {"EEG_REPORT_PYTHON": str(report_python)},
        "StandardOutPath": str(run / "service.log"),
        "StandardErrorPath": str(run / "service-error.log"),
    }
    destination = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    if destination.exists() and plistlib.loads(destination.read_bytes()) != plist:
        raise ValueError("Existing job configuration differs")
    destination.parent.mkdir(exist_ok=True, parents=True)
    destination.write_bytes(plistlib.dumps(plist))
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(destination)], check=True)
    print("Installed", label)


if __name__ == "__main__":
    main()
