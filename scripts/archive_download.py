"""Download a reviewed cohort manifest with resume, source integrity checks and receipts."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import urlparse

import httpx

from scripts.download_physionet import atomic_json, safe_relative


def now():
    return datetime.now(timezone.utc).isoformat()


class Archive:
    def __init__(self, root, plan, workers=3, reserve_gib=20):
        self.root, self.plan = root, plan
        self.workers, self.reserve = workers, reserve_gib * 1024**3
        self.lock = threading.RLock()
        self.last_checkpoint = 0
        self.stopped = threading.Event()
        self.status = {
            "started_at": now(),
            "updated_at": now(),
            "state": "downloading",
            "total_files": len(plan["files"]),
            "verified_files": 0,
            "verified_bytes": 0,
            "received_bytes_this_run": 0,
            "cohort_bytes": {},
            "cohort_files": {},
            "active_files": {},
            "failures": [],
        }
        self.receipts = {}
        receipt_path = root / "verified-files.jsonl"
        if receipt_path.exists():
            for line in receipt_path.read_text().splitlines():
                try:
                    item = json.loads(line)
                    self.receipts[item["path"]] = item
                except json.JSONDecodeError:
                    pass
        self.journal = receipt_path.open("a", buffering=1)
        self.client = httpx.Client(
            timeout=httpx.Timeout(120, connect=30),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=workers),
            headers={"Accept-Encoding": "identity", "User-Agent": "EEG-Explorer-Archive/0.1"},
        )

    def checkpoint(self, force=False):
        with self.lock:
            if force or time.monotonic() - self.last_checkpoint > 3:
                self.status["updated_at"] = now()
                atomic_json(self.root / "download-status.json", self.status)
                self.last_checkpoint = time.monotonic()

    def guard(self):
        if self.stopped.is_set():
            raise RuntimeError("Download paused")
        if (self.root / "PAUSE").exists() or not self.root.exists():
            self.stopped.set()
            raise RuntimeError("Download paused or drive disconnected")
        if shutil.disk_usage(self.root).free < self.reserve:
            self.stopped.set()
            raise RuntimeError("20 GiB free-space reserve reached")

    def verify(self, path, item):
        if item["bytes"] is not None and path.stat().st_size != item["bytes"]:
            return None
        sha = hashlib.sha256()
        source = hashlib.new(item["hash_algorithm"]) if item.get("hash_algorithm") else None
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                sha.update(chunk)
                if source:
                    source.update(chunk)
        if source and source.hexdigest() != item["checksum"]:
            return None
        return sha.hexdigest()

    def complete(self, item, target, sha):
        receipt = {
            "path": item["path"],
            "sha256": sha,
            "bytes": target.stat().st_size,
            "mtime_ns": target.stat().st_mtime_ns,
            "etag": item.get("etag"),
            "source_checksum": item.get("checksum"),
            "cohort": item["cohort"],
            "diagnosis": item.get("diagnosis"),
            "verified_at": now(),
        }
        with self.lock:
            self.journal.write(json.dumps(receipt) + "\n")
            self.receipts[item["path"]] = receipt
        return receipt

    def download(self, item):
        self.guard()
        safe_relative(item["path"])
        url = urlparse(item["url"])
        if url.scheme != "https" or url.hostname not in {"physionet.org", "s3.amazonaws.com"}:
            raise ValueError("Unreviewed download host")
        target = self.root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = self.receipts.get(item["path"])
        if target.exists():
            stat = target.stat()
            if (
                receipt
                and stat.st_size == receipt["bytes"]
                and stat.st_mtime_ns == receipt["mtime_ns"]
                and receipt.get("etag") == item.get("etag")
                and receipt.get("source_checksum") == item.get("checksum")
            ):
                return receipt
            # A multipart ETag is not a content checksum: without a matching receipt
            # fetch it again with If-Match rather than trust an unrelated local file.
            sha = self.verify(target, item) if item.get("checksum") else None
            if sha:
                return self.complete(item, target, sha)
        partial = target.with_name(target.name + ".part")
        for attempt in range(5):
            self.guard()
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset and item.get("checksum"):
                    sha = self.verify(partial, item)
                    if sha:
                        partial.replace(target)
                        return self.complete(item, target, sha)
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                if item.get("etag"):
                    headers["If-Match"] = '"' + item["etag"] + '"'
                with self.client.stream("GET", item["url"], headers=headers) as response:
                    if response.status_code == 416:
                        partial.replace(
                            partial.with_name(partial.name + f".invalid-{int(time.time())}")
                        )
                        raise ValueError("Invalid resume offset; retrying from zero")
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append and not response.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"
                    ):
                        raise ValueError("Unexpected resume Content-Range")
                    if (
                        item.get("etag")
                        and response.headers.get("ETag", "").strip('"') != item["etag"]
                    ):
                        raise ValueError("Source object changed since inventory")
                    with partial.open("ab" if append else "wb") as f:
                        for chunk in response.iter_bytes(1024 * 1024):
                            self.guard()
                            f.write(chunk)
                            with self.lock:
                                self.status["received_bytes_this_run"] += len(chunk)
                                self.status["active_files"][item["path"]] = f.tell()
                            self.checkpoint()
                        f.flush()
                        os.fsync(f.fileno())
                sha = self.verify(partial, item)
                if not sha:
                    partial.replace(
                        partial.with_name(partial.name + f".invalid-{int(time.time())}")
                    )
                    raise ValueError("Source checksum or size mismatch")
                partial.replace(target)
                return self.complete(item, target, sha)
            except (OSError, ValueError, httpx.HTTPError) as error:
                if attempt == 4:
                    raise RuntimeError(str(error)) from error
                time.sleep(min(30, 2**attempt))

    def run(self):
        # First fetch metadata, then smaller diagnostic/control recordings, then
        # large standalone healthy datasets. This produces useful coverage early.
        files = sorted(
            self.plan["files"],
            key=lambda i: (
                i["dataset"] not in {"ds002778", "ds004504", "ds006036"},
                i["dataset"] in {"ds004148", "ds004796", "ds005385", "ds007987"},
                i["dataset"],
                i["cohort"] != "metadata",
                i["path"],
            ),
        )
        print(f"Starting {len(files)} files → {self.root}", flush=True)
        self.checkpoint(force=True)
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self.download, item): item for item in files}
                for future in as_completed(futures):
                    item = futures[future]
                    if self.stopped.is_set():
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        receipt = future.result()
                        with self.lock:
                            self.status["verified_files"] += 1
                            self.status["verified_bytes"] += receipt["bytes"]
                            group = item["cohort"]
                            self.status["cohort_bytes"][group] = (
                                self.status["cohort_bytes"].get(group, 0) + receipt["bytes"]
                            )
                            self.status["cohort_files"][group] = (
                                self.status["cohort_files"].get(group, 0) + 1
                            )
                            self.status["active_files"].pop(item["path"], None)
                        if self.status["verified_files"] % 100 == 0:
                            print(
                                f"Verified {self.status['verified_files']}/{len(files)} files · {self.status['verified_bytes'] / 1e9:.2f} GB",
                                flush=True,
                            )
                    except Exception as error:
                        with self.lock:
                            self.status["failures"].append(
                                {"path": item["path"], "error": str(error)}
                            )
                        print(f"Failed {item['path']}: {error}", flush=True)
                    self.checkpoint()
            self.status["state"] = (
                "paused"
                if self.stopped.is_set()
                else "incomplete"
                if self.status["failures"]
                else "complete"
            )
            print(f"Archive state: {self.status['state']}", flush=True)
        finally:
            self.journal.close()
            self.client.close()
            self.checkpoint(force=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    args = parser.parse_args()
    if not args.root.parent.is_dir():
        raise SystemExit("Reconnect the external drive before starting.")
    args.root.mkdir(exist_ok=True)
    if (args.root / "PAUSE").exists():
        raise SystemExit("Archive is paused. Remove its PAUSE file to resume.")
    plan = json.loads(args.plan.read_text())
    with (args.root / ".download.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An archive worker is already running.")
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        atomic_json(args.root / "download-plan.json", plan)
        Archive(args.root, plan, workers=args.workers).run()


if __name__ == "__main__":
    main()
