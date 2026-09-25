"""Download a reviewed cohort manifest with resume, source integrity checks and receipts."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import heapq
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
        for attempt in range(1):
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
                if attempt == 0:
                    raise RuntimeError(str(error)) from error
                time.sleep(min(30, 2**attempt))

    def run(self):
        """Bounded in-flight work; persist failures and retry until verified or paused."""
        retry_path = self.root / "retry-state.json"
        try:
            retry = json.loads(retry_path.read_text())
        except (OSError, ValueError):
            try:
                old = json.loads((self.root / "download-status.json").read_text())
            except (OSError, ValueError):
                old = {}
            retry = {
                r["path"]: {"path": r["path"], "error": r["error"], "attempts": 0, "next_retry": 0}
                for r in old.get("failures", [])
            }
        ordered = sorted(
            self.plan["files"],
            key=lambda i: (
                i["path"] not in retry,
                i["dataset"] != "ds004584",
                i["dataset"] not in {"ds002778", "ds004504", "ds006036"},
                i["dataset"] in {"ds004148", "ds004796", "ds005385", "ds007987"},
                i["dataset"],
                i["cohort"] != "metadata",
                i["path"],
            ),
        )
        queues, hosts = {}, {}
        recovered = 0
        counted = set()

        def success(item, receipt):
            nonlocal recovered
            if item["path"] in counted:
                return
            counted.add(item["path"])
            if item["path"] in retry:
                recovered += 1
                retry.pop(item["path"], None)
            self.status["verified_files"] += 1
            self.status["verified_bytes"] += receipt["bytes"]
            group = item["cohort"]
            self.status["cohort_bytes"][group] = (
                self.status["cohort_bytes"].get(group, 0) + receipt["bytes"]
            )
            self.status["cohort_files"][group] = self.status["cohort_files"].get(group, 0) + 1

        # Fast resume of already verified objects, without thousands of network jobs.
        for rank, item in enumerate(ordered):
            safe_relative(item["path"])
            target = self.root / item["path"]
            receipt = self.receipts.get(item["path"])
            try:
                stat = target.stat()
            except OSError:
                stat = None
            if (
                receipt
                and stat
                and stat.st_size == receipt["bytes"]
                and stat.st_mtime_ns == receipt["mtime_ns"]
                and receipt.get("etag") == item.get("etag")
                and receipt.get("source_checksum") == item.get("checksum")
            ):
                success(item, receipt)
                continue
            host = urlparse(item["url"]).hostname
            queues.setdefault(host, [])
            hosts.setdefault(host, {"failures": 0, "until": 0})
            heapq.heappush(
                queues[host], (retry.get(item["path"], {}).get("next_retry", 0), rank, item)
            )

        def persist():
            self.status["failures"] = list(retry.values())
            self.status["retry_pending"] = len(retry)
            self.status["recovered_files_this_run"] = recovered
            self.status["host_cooldowns"] = {
                h: v["until"] for h, v in hosts.items() if v["until"] > time.time()
            }
            atomic_json(retry_path, retry)
            self.checkpoint(force=True)

        print(
            f"Resumed {len(counted)} verified files; {len(retry)} failures retained for retry",
            flush=True,
        )
        persist()
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                active = {}
                while active or any(queues.values()):
                    try:
                        self.guard()
                    except RuntimeError:
                        for future in active:
                            future.cancel()
                        break
                    for host, queue in queues.items():
                        while (
                            queue
                            and len(active) < self.workers
                            and max(queue[0][0], hosts[host]["until"]) <= time.time()
                        ):
                            _, rank, item = heapq.heappop(queue)
                            active[pool.submit(self.download, item)] = (host, rank, item)
                    if not active:
                        self.status["state"] = "waiting_to_retry"
                        persist()
                        self.stopped.wait(3)
                        continue
                    self.status["state"] = "downloading"
                    done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        host, rank, item = active.pop(future)
                        try:
                            receipt = future.result()
                            success(item, receipt)
                            hosts[host] = {"failures": 0, "until": 0}
                            if self.status["verified_files"] % 100 == 0:
                                print(
                                    f"Verified {self.status['verified_files']}/{len(ordered)} files; recovered {recovered}; pending retry {len(retry)}",
                                    flush=True,
                                )
                        except Exception as error:
                            if not self.stopped.is_set():
                                attempts = retry.get(item["path"], {}).get("attempts", 0) + 1
                                delay, cooldown = retry_delay(attempts, error)
                                next_retry = time.time() + delay
                                retry[item["path"]] = {
                                    "path": item["path"],
                                    "error": str(error),
                                    "attempts": attempts,
                                    "next_retry": next_retry,
                                    "last_failure_at": now(),
                                }
                                with (self.root / "retry-history.jsonl").open("a") as history:
                                    history.write(json.dumps(retry[item["path"]]) + "\n")
                                heapq.heappush(queues[host], (next_retry, rank, item))
                                if cooldown:
                                    hosts[host]["failures"] += 1
                                    hosts[host]["until"] = time.time() + min(
                                        300, 15 * 2 ** min(hosts[host]["failures"] - 1, 5)
                                    )
                                print(
                                    f"Retry retained {item['path']} in {delay}s: {error}",
                                    flush=True,
                                )
                        finally:
                            with self.lock:
                                self.status["active_files"].pop(item["path"], None)
                        persist()
                    self.checkpoint()
            self.status["state"] = "paused" if self.stopped.is_set() else "complete"
            if not self.stopped.is_set() and (retry or len(counted) != len(ordered)):
                raise RuntimeError("Completion invariant failed: unverified files remain")
        finally:
            self.journal.close()
            self.client.close()
            self.status["active_files"] = {}
            persist()


def retry_delay(attempt, error):
    cause = error
    while getattr(cause, "__cause__", None):
        cause = cause.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        code = cause.response.status_code
        if code in {400, 401, 403, 404, 410, 412}:
            return 3600, False  # Still retained; source/access changes need attention.
        if code == 429:
            try:
                return max(
                    60, min(3600, int(cause.response.headers.get("Retry-After", "60")))
                ), True
            except ValueError:
                return 60, True
    network = isinstance(cause, (httpx.TransportError, OSError)) or "nodename" in str(error)
    return min(300, 15 * 2 ** min(attempt - 1, 5)), network


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
