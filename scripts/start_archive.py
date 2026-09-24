"""Start the archive worker detached from the terminal; writes only to its archive."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def launch(root: Path, plan: Path):
    if not root.parent.is_dir():
        raise ValueError("Reconnect the destination drive first.")
    root.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "scripts.archive_download",
        "--root",
        str(root),
        "--plan",
        str(plan),
    ]
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        command = ["caffeinate", "-i", *command]
    with (root / "download.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    print(f"Archive worker started (PID {process.pid}). Status: {root / 'download-status.json'}")
    return process.pid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    launch(args.root, args.plan)
