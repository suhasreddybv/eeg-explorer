"""Install this user's autonomous EEG archive LaunchAgent on macOS."""

import argparse
import os
from pathlib import Path
import plistlib
import subprocess
import sys

LABEL = "local.eeg-explorer.archive"


def install(root):
    project = Path(__file__).resolve().parent.parent
    folder = Path.home() / "Library" / "LaunchAgents"
    folder.mkdir(parents=True, exist_ok=True)
    logs = project / "data" / "archive-service"
    logs.mkdir(parents=True, exist_ok=True)
    target = folder / f"{LABEL}.plist"
    value = {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "scripts.archive_service", "--root", str(root)],
        "WorkingDirectory": str(project),
        "RunAtLoad": True,
        "StartInterval": 60,
        "ThrottleInterval": 60,
        "ProcessType": "Background",
        "StandardOutPath": str(logs / "service.log"),
        "StandardErrorPath": str(logs / "service-error.log"),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
    }
    if target.exists() and plistlib.loads(target.read_bytes()) != value:
        raise ValueError("Existing service differs; review it before replacement")
    target.write_bytes(plistlib.dumps(value))
    domain = f"gui/{os.getuid()}"
    active = subprocess.run(["launchctl", "print", f"{domain}/{LABEL}"], capture_output=True)
    if active.returncode != 0:
        subprocess.run(["launchctl", "bootstrap", domain, str(target)], check=True)
    print(
        f"Installed {target}. launchd starts the worker without Codex and resumes after login/mount availability."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("This installer is for macOS only")
    install(args.root)
