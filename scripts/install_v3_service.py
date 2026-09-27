"""Install a one-shot local v3 job, independent of Codex and usage limits."""

import os
from pathlib import Path
import plistlib
import subprocess
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    run = project / "data/pd-runs/v3"
    run.mkdir(parents=True, exist_ok=True)
    label = "local.eeg-explorer.comparison.v3"
    plist = {
        "Label": label,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-i",
            sys.executable,
            "-u",
            "-m",
            "scripts.run_pd_v3",
        ],
        "WorkingDirectory": str(project),
        "RunAtLoad": True,
        "ProcessType": "Background",
        "StandardOutPath": str(run / "service.log"),
        "StandardErrorPath": str(run / "service-error.log"),
    }
    destination = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    if destination.exists() and plistlib.loads(destination.read_bytes()) != plist:
        raise ValueError("Different service already exists")
    destination.write_bytes(plistlib.dumps(plist))
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(destination)], check=True)
    print("Installed", label)


if __name__ == "__main__":
    main()
