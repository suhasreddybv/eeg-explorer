"""macOS service entry point: no Codex, model calls, credentials or cloud jobs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    # Never recreate an absent mount point on the internal drive.
    if not os.path.ismount(root.parent) or not root.is_dir() or (root / "PAUSE").exists():
        return
    plan = root / "download-plan.json"
    if not plan.exists():
        return
    try:
        status = json.loads((root / "download-status.json").read_text())
        if (
            status.get("state") == "complete"
            and status.get("verified_files") == status.get("total_files")
            and not status.get("failures")
        ):
            return
    except (OSError, ValueError):
        pass
    command = [
        sys.executable,
        "-m",
        "scripts.archive_download",
        "--root",
        str(root),
        "--plan",
        str(plan),
    ]
    if sys.platform == "darwin":
        command = ["/usr/bin/caffeinate", "-i", *command]
    with (root / "download.log").open("a") as log:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
