import hashlib

import httpx
import pytest

from scripts.download_physionet import Downloader, parse_checksums, safe_relative
from scripts.archive_download import Archive


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", "a/../../x", "a\\b", ""])
def test_unsafe_manifest_paths_rejected(path):
    with pytest.raises(ValueError):
        safe_relative(path)


def test_checksum_manifest():
    digest = "a" * 64
    assert parse_checksums(f"{digest}  subject/test.edf\n") == [(digest, "subject/test.edf")]
    with pytest.raises(ValueError):
        parse_checksums("not a checksum file")


def test_resume_verifies_complete_file(tmp_path):
    data = b"example-recording" * 100
    digest = hashlib.sha256(data).hexdigest()
    target = tmp_path / "signal.edf"
    partial = tmp_path / "signal.edf.part"
    partial.write_bytes(data[:100])
    downloader = Downloader(tmp_path, {"datasets": []}, reserve_gib=0)

    def respond(request):
        assert request.headers["range"] == "bytes=100-"
        return httpx.Response(
            206,
            headers={"Content-Range": f"bytes 100-{len(data) - 1}/{len(data)}"},
            content=data[100:],
        )

    downloader.client.close()
    downloader.client = httpx.Client(transport=httpx.MockTransport(respond))
    assert downloader.fetch_file(
        "https://physionet.org/files/test/1/", tmp_path, digest, "signal.edf"
    ) == len(data)
    assert target.read_bytes() == data
    assert not partial.exists()
    # Re-running validates the existing file without making another request.
    assert downloader.fetch_file(
        "https://physionet.org/files/test/1/", tmp_path, digest, "signal.edf"
    ) == len(data)
    downloader.client.close()


def test_server_ignoring_range_restarts_safely(tmp_path):
    data = b"complete file"
    (tmp_path / "file.part").write_bytes(b"incomplete")
    downloader = Downloader(tmp_path, {"datasets": []}, reserve_gib=0)
    downloader.client.close()

    downloader.client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))
    )
    downloader.fetch_file(
        "https://physionet.org/files/test/1/", tmp_path, hashlib.sha256(data).hexdigest(), "file"
    )
    assert (tmp_path / "file").read_bytes() == data
    downloader.client.close()


def test_s3_archive_checks_etag_md5_and_records_cohort(tmp_path):
    data = b"public EEG test fixture"
    digest = hashlib.md5(data).hexdigest()
    item = {
        "url": "https://s3.amazonaws.com/openneuro.org/ds-test/sub-1/eeg/signal.eeg",
        "path": "datasets/openneuro/ds-test/sub-1/eeg/signal.eeg",
        "bytes": len(data),
        "etag": digest,
        "hash_algorithm": "md5",
        "checksum": digest,
        "cohort": "neurodegenerative",
        "diagnosis": "parkinsons",
    }
    archive = Archive(tmp_path, {"files": [item]}, reserve_gib=0)
    archive.client.close()

    def respond(request):
        assert request.headers["if-match"] == f'"{digest}"'
        return httpx.Response(200, headers={"ETag": f'"{digest}"'}, content=data)

    archive.client = httpx.Client(transport=httpx.MockTransport(respond))
    receipt = archive.download(item)
    assert receipt["sha256"] == hashlib.sha256(data).hexdigest()
    assert receipt["cohort"] == "neurodegenerative"
    assert archive.download(item) == receipt
    archive.journal.close()
    archive.client.close()


def test_archive_pause_stops_before_network(tmp_path):
    archive = Archive(tmp_path, {"files": []}, reserve_gib=0)
    (tmp_path / "PAUSE").touch()
    with pytest.raises(RuntimeError, match="paused"):
        archive.guard()
    archive.journal.close()
    archive.client.close()


def test_persistent_retry_recovers_failed_file_without_manual_restart(tmp_path, monkeypatch):
    import json
    import scripts.archive_download as module

    data = b"recovered EEG bytes"
    digest = hashlib.md5(data).hexdigest()
    item = {
        "url": "https://s3.amazonaws.com/openneuro.org/test.eeg",
        "path": "datasets/test.eeg",
        "bytes": len(data),
        "etag": digest,
        "hash_algorithm": "md5",
        "checksum": digest,
        "cohort": "healthy_control",
        "dataset": "ds-test",
    }
    archive = Archive(tmp_path, {"files": [item]}, workers=1, reserve_gib=0)
    archive.client.close()
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("DNS unavailable", request=request)
        return httpx.Response(200, headers={"ETag": f'"{digest}"'}, content=data)

    archive.client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(module, "retry_delay", lambda *_: (0, False))
    archive.run()
    status = json.loads((tmp_path / "download-status.json").read_text())
    assert len(calls) == 2
    assert status["state"] == "complete" and status["verified_files"] == 1
    assert status["recovered_files_this_run"] == 1 and not status["failures"]
    assert json.loads((tmp_path / "retry-state.json").read_text()) == {}
    assert (tmp_path / "retry-history.jsonl").exists()


def test_old_failures_are_retried_and_cleared(tmp_path):
    import json

    data = b"old failure"
    digest = hashlib.md5(data).hexdigest()
    item = {
        "url": "https://s3.amazonaws.com/openneuro.org/test.eeg",
        "path": "datasets/test.eeg",
        "bytes": len(data),
        "etag": digest,
        "hash_algorithm": "md5",
        "checksum": digest,
        "cohort": "healthy_control",
        "dataset": "ds-test",
    }
    (tmp_path / "download-status.json").write_text(
        json.dumps({"failures": [{"path": item["path"], "error": "DNS"}]})
    )
    archive = Archive(tmp_path, {"files": [item]}, workers=1, reserve_gib=0)
    archive.client.close()
    archive.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"ETag": f'"{digest}"'}, content=data)
        )
    )
    archive.run()
    status = json.loads((tmp_path / "download-status.json").read_text())
    assert status["recovered_files_this_run"] == 1 and not status["failures"]


def test_retry_backoff_and_source_errors_are_not_dropped():
    from scripts.archive_download import retry_delay

    assert retry_delay(20, httpx.ConnectError("DNS")) == (300, True)
    request = httpx.Request("GET", "https://physionet.org/missing")
    response = httpx.Response(404, request=request)
    assert retry_delay(1, httpx.HTTPStatusError("missing", request=request, response=response)) == (
        3600,
        False,
    )


def test_dns_outage_cools_host_before_more_files_are_attempted(tmp_path):
    import json

    items = [
        {
            "url": f"https://physionet.org/files/{i}.edf",
            "path": f"data/{i}.edf",
            "bytes": 10,
            "cohort": "healthy_control",
            "dataset": "test",
        }
        for i in range(5)
    ]
    archive = Archive(tmp_path, {"files": items}, workers=1, reserve_gib=0)
    archive.client.close()
    calls = []

    def respond(request):
        calls.append(request)
        raise httpx.ConnectError("DNS unavailable", request=request)

    archive.client = httpx.Client(transport=httpx.MockTransport(respond))
    original = archive.guard

    def guard():
        if archive.status.get("host_cooldowns"):
            archive.stopped.set()
        original()

    archive.guard = guard
    archive.run()
    status = json.loads((tmp_path / "download-status.json").read_text())
    assert len(calls) == 1 and status["state"] == "paused"
    assert status["retry_pending"] == 1 and status["host_cooldowns"]
