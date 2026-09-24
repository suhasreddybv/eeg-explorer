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
