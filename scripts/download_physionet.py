"""Resumable, checksum-verified downloads from reviewed open PhysioNet datasets.

Usage: python scripts/download_physionet.py --plan PLAN.json --destination /path/to/archive
Only reviewed entries with download=true and access=open are downloaded.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading
import time
from urllib.parse import quote

import httpx


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe path in checksum manifest: {name!r}")
    return p


def parse_checksums(text):
    files = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+[* ]?(.+)", line)
        if not match:
            raise ValueError("Unexpected SHA256SUMS.txt format")
        checksum, name = match.groups()
        safe_relative(name)
        files.append((checksum.lower(), name))
    if not files:
        raise ValueError("Empty checksum manifest")
    return files


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, content):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(content, indent=2))
    tmp.replace(path)


class Downloader:
    def __init__(self, root, plan, reserve_gib=20, workers=2):
        self.root = root
        self.plan = plan
        self.reserve = reserve_gib * 1024**3
        self.workers = workers
        self.lock = threading.Lock()
        self.last_write = 0
        self.status = {
            "started_at": utc(),
            "updated_at": utc(),
            "state": "preparing",
            "verified_files": 0,
            "verified_bytes": 0,
            "received_bytes_this_run": 0,
            "total_files": 0,
            "failures": [],
            "datasets": [],
            "active_files": {},
        }
        self.client = httpx.Client(
            timeout=httpx.Timeout(120, connect=30),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=workers + 2),
            headers={"User-Agent": "EEG-Explorer-Archive/0.1 (public-data downloader)"},
        )

    def checkpoint(self, force=False):
        with self.lock:
            if force or time.monotonic() - self.last_write > 3:
                self.status["updated_at"] = utc()
                atomic_json(self.root / "download-status.json", self.status)
                self.last_write = time.monotonic()

    def check_disk(self):
        if not self.root.exists() or shutil.disk_usage(self.root).free < self.reserve:
            raise RuntimeError(
                "Download paused: external destination unavailable or free-space reserve reached."
            )

    def fetch_file(self, base, directory, checksum, relative):
        safe_relative(relative)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) == checksum:
            return target.stat().st_size
        partial = target.with_name(target.name + ".part")
        key = str(target.relative_to(self.root))
        last_error = None
        for attempt in range(5):
            self.check_disk()
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                # A previous interruption may have happened after the final byte.
                if offset and sha256(partial) == checksum:
                    partial.replace(target)
                    return target.stat().st_size
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with self.client.stream(
                    "GET", base + quote(relative, safe="/"), headers=headers
                ) as response:
                    if response.status_code == 416:
                        # Keep suspect partials for inspection; a fresh attempt starts at byte zero.
                        partial.replace(
                            partial.with_name(partial.name + f".invalid-{int(time.time())}")
                        )
                        raise ValueError("Server rejected resume offset; retrying from the start.")
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError("Server returned an unexpected resume range.")
                    with partial.open("ab" if append else "wb") as f:
                        for chunk in response.iter_bytes(1024 * 1024):
                            self.check_disk()
                            f.write(chunk)
                            with self.lock:
                                self.status["received_bytes_this_run"] += len(chunk)
                                self.status["active_files"][key] = f.tell()
                            self.checkpoint()
                        f.flush()
                        os.fsync(f.fileno())
                if sha256(partial) != checksum:
                    partial.replace(
                        partial.with_name(partial.name + f".invalid-{int(time.time())}")
                    )
                    raise ValueError("SHA-256 mismatch; preserving the invalid file and retrying.")
                partial.replace(target)
                with self.lock:
                    self.status["active_files"].pop(key, None)
                return target.stat().st_size
            except (httpx.HTTPError, OSError, ValueError) as e:
                last_error = e
                if attempt < 4:
                    time.sleep(min(60, 2 ** (attempt + 1)))
        raise RuntimeError(f"{relative}: {last_error}")

    def run(self):
        selected = [d for d in self.plan["datasets"] if d.get("download") and d["access"] == "open"]
        print(f"Preparing {len(selected)} open datasets", flush=True)
        work = []
        # Small collections first, so useful data is available while large corpora transfer.
        selected.sort(key=lambda d: d.get("estimated_bytes") or float("inf"))
        for dataset in selected:
            slug, version = dataset["slug"], dataset["version"]
            if not re.fullmatch(r"[a-zA-Z0-9_.-]+", slug) or not re.fullmatch(
                r"[a-zA-Z0-9_.-]+", version
            ):
                raise ValueError("Invalid dataset identifier")
            base = f"https://physionet.org/files/{slug}/{version}/"
            directory = self.root / "datasets" / slug / version
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "SOURCE.json", dataset)
            record = {
                "slug": slug,
                "version": version,
                "state": "pending",
                "verified_files": 0,
                "verified_bytes": 0,
            }
            try:
                response = self.client.get(base + "SHA256SUMS.txt")
                response.raise_for_status()
                checks = parse_checksums(response.text)
                (directory / "SHA256SUMS.txt").write_text(response.text)
                record["total_files"] = len(checks)
                work.extend((base, directory, check, name, record) for check, name in checks)
            except (httpx.HTTPError, ValueError) as e:
                record["state"] = "blocked"
                record["error"] = str(e)
                self.status["failures"].append({"dataset": slug, "error": str(e)})
            self.status["datasets"].append(record)
            self.checkpoint(force=True)
        self.status["total_files"] = len(work)
        self.status["state"] = "downloading"
        self.checkpoint(force=True)
        print(f"Downloading/verifying {len(work)} files with {self.workers} workers", flush=True)
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self.fetch_file, *item[:4]): item for item in work}
                for future in as_completed(futures):
                    *_, name, record = futures[future]
                    try:
                        size = future.result()
                        with self.lock:
                            self.status["verified_files"] += 1
                            self.status["verified_bytes"] += size
                            record["verified_files"] += 1
                            record["verified_bytes"] += size
                            record["state"] = (
                                "complete"
                                if record["verified_files"] == record["total_files"]
                                else "downloading"
                            )
                        if record["state"] == "complete":
                            print(
                                f"Complete: {record['slug']} ({record['verified_files']} files)",
                                flush=True,
                            )
                    except Exception as e:
                        with self.lock:
                            record["state"] = "incomplete"
                            self.status["failures"].append(
                                {"dataset": record["slug"], "file": name, "error": str(e)}
                            )
                        print(f"Failed: {record['slug']}/{name}: {e}", flush=True)
                    self.checkpoint(force=True)
            self.status["state"] = "complete" if not self.status["failures"] else "incomplete"
        finally:
            self.checkpoint(force=True)
            self.client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--reserve-gib", type=int, default=20)
    args = parser.parse_args()
    # Require the parent to already exist: never recreate a missing mount point.
    if not args.destination.parent.is_dir():
        raise SystemExit("Destination parent is not mounted. Reconnect the drive first.")
    args.destination.mkdir(exist_ok=True)
    plan = json.loads(args.plan.read_text())
    with (args.destination / ".download.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another archive download is already running.")
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        atomic_json(args.destination / "download-plan.json", plan)
        Downloader(args.destination, plan, args.reserve_gib, args.workers).run()


if __name__ == "__main__":
    main()
