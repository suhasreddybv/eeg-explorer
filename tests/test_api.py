import importlib
import io
import json

from fastapi.testclient import TestClient
import mne
import numpy as np
import pytest

module = importlib.import_module("eeg_explorer.app")
HEADERS = {"X-EEG-Local": "1"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DATA", tmp_path)
    module.recording.cache_clear()
    with TestClient(module.app, headers=HEADERS) as c:
        yield c
    module.recording.cache_clear()


def test_complete_local_workflow(client):
    demo = client.post("/api/demo").json()
    rid = demo["id"]
    assert client.post("/api/demo").json()["id"] == rid
    assert len(client.get("/api/recordings").json()) == 1
    analysis = client.post(f"/api/recordings/{rid}/analysis", json={})
    assert analysis.status_code == 200
    assert analysis.json()["summary"]["deviations"] > 0
    assert analysis.json()["recording_sha256"] == demo["sha256"]
    segment = client.get(f"/api/recordings/{rid}/segment?start=39&channels=0,2").json()
    assert segment["channels"] == ["Fp1", "O1"]
    assert len(segment["times"]) <= 1000
    assert np.isfinite(segment["psd"]).all()
    added = client.post(
        f"/api/recordings/{rid}/annotations", json={"start": 40, "end": 50, "label": "Eyes closed"}
    ).json()
    assert added["annotations"][0]["label"] == "Eyes closed"
    context = client.put(
        f"/api/recordings/{rid}/context", json={"activity": "Rest", "session_key": "visit-001"}
    )
    assert context.status_code == 200
    # Data is persisted independently of the in-memory recording cache.
    module.recording.cache_clear()
    persisted = json.loads((module.DATA / f"{rid}.json").read_text())
    assert persisted["context"]["session_key"] == "visit-001"
    aid = added["annotations"][0]["id"]
    assert not client.delete(f"/api/recordings/{rid}/annotations/{aid}").json()["annotations"]


def test_local_request_protection(client):
    assert client.post("/api/demo", headers={"X-EEG-Local": "0"}).status_code == 403
    assert client.post("/api/demo", headers={"Origin": "https://example.com"}).status_code == 403
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/data/").status_code == 404
    assert "frame-ancestors 'none'" in client.get("/").headers["Content-Security-Policy"]


def test_invalid_requests(client):
    rid = client.post("/api/demo").json()["id"]
    assert client.get(f"/api/recordings/{rid}/segment?channels=-1").status_code == 400
    assert client.get(f"/api/recordings/{rid}/segment?start=nan").status_code == 400
    assert (
        client.post(f"/api/recordings/{rid}/analysis", json={"baseline_end": 500}).status_code
        == 400
    )
    assert (
        client.post(
            f"/api/recordings/{rid}/annotations", json={"start": 80, "end": 70, "label": "bad"}
        ).status_code
        == 400
    )
    assert client.post("/api/recordings/not-a-real-id/analysis", json={}).status_code == 404
    assert client.post("/api/import", files={"file": ("bad.edf", b"not an EDF")}).status_code == 400


def test_csv_import_units(client):
    text = io.StringIO()
    samples = 10 * np.sin(2 * np.pi * 10 * np.arange(256 * 24) / 256)
    np.savetxt(text, samples[:, None], header="O1", comments="")
    response = client.post(
        "/api/import",
        files={"file": ("eyes-open.csv", text.getvalue())},
        data={"sfreq": 256, "units": "uV"},
    )
    assert response.status_code == 200
    rec = response.json()
    assert rec["channels"] == ["O1"]
    assert rec["duration"] == 24
    assert (
        client.post(f"/api/recordings/{rec['id']}/analysis", json={"baseline_end": 12}).status_code
        == 200
    )


def test_edf_import_preserves_units_and_events(client, tmp_path):
    t = np.arange(256 * 24) / 256
    raw = mne.io.RawArray(
        (10e-6 * np.sin(2 * np.pi * 10 * t))[None, :],
        mne.create_info(["O1"], 256, "eeg"),
        verbose="ERROR",
    )
    raw.set_annotations(mne.Annotations([10], [3], ["eyes closed"]))
    path = tmp_path / "test.edf"
    raw.export(path, fmt="edf", verbose="ERROR")
    response = client.post("/api/import", files={"file": ("test.edf", path.read_bytes())})
    assert response.status_code == 200
    rec = response.json()
    assert rec["events"][0]["label"] == "eyes closed"
    assert rec["sfreq"] == 256
    stored = module.recording(rec["id"])
    assert np.max(stored.data) == pytest.approx(10, abs=0.01)


def test_archive_status_and_recording_library(client, tmp_path):
    assert client.get("/api/archive/status").json()["available"] is False
    archive = tmp_path / "archive"
    archive.mkdir()
    (tmp_path / "archive-path.txt").write_text(str(archive))
    (tmp_path / "archive-summary.json").write_text('{"datasets": 23}')
    (archive / "download-status.json").write_text('{"state": "downloading", "verified_bytes": 100}')
    status = client.get("/api/archive/status").json()
    assert status["status"]["verified_bytes"] == 100
    assert client.get("/api/recordings").json() == []
    (archive / "download-status.json").unlink()
    assert client.get("/api/archive/status").json()["available"] is False


def test_research_status_exposes_aggregate_progress_only(client, tmp_path):
    assert client.get("/api/research/status").json()["available"] is False
    run = tmp_path / "research"
    run.mkdir()
    (tmp_path / "pd-training-path.txt").write_text(str(run))
    (run / "registry.json").write_text('{"records": [{"participant_key": "private-subject"}]}')
    (run / "preparation.json").write_text(
        '{"records": [{"split": "train", "status": "ready", "participant_key": "private-subject"}]}'
    )
    result = client.get("/api/research/status")
    assert result.json()["counts"]["train:ready"] == 1
    assert "private-subject" not in result.text
    assert "test_results" in result.json()
