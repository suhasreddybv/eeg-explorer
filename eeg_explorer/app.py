"""Loopback-only API and local recording store."""

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import httpx
import numpy as np
from scipy import signal

from .signals import Recording, analyze, clean, read_csv, read_edf, synthetic

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("EEG_EXPLORER_DATA", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)
LOCK = RLock()
SOURCE = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf"
SOURCE_PAGE = "https://physionet.org/content/eegmmidb/1.0.0/"
MAX_UPLOAD = 64 * 1024 * 1024
app = FastAPI(title="EEG Explorer", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def local_requests(request: Request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        if request.headers.get("x-eeg-local") != "1":
            return JSONResponse(
                {"detail": "Use the local EEG Explorer application."}, status_code=403
            )
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse(
                {"detail": "Cross-origin requests are not allowed."}, status_code=403
            )
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'"
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def atomic_json(path, value):
    temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def metadata(rid):
    if len(rid) != 24 or any(c not in "0123456789abcdef" for c in rid):
        raise HTTPException(404, "Recording not found.")
    path = DATA / f"{rid}.json"
    if not path.exists():
        raise HTTPException(404, "Recording not found.")
    return json.loads(path.read_text())


@lru_cache(maxsize=2)
def recording(rid):
    meta = metadata(rid)
    with np.load(DATA / f"{rid}.npz", allow_pickle=False) as f:
        return Recording(f["data"], meta["sfreq"], meta["channels"], meta["events"])


def save_recording(rec, name, source, provenance=None):
    digest = hashlib.sha256(
        rec.data.tobytes() + str(rec.sfreq).encode() + json.dumps(rec.channels).encode()
    ).hexdigest()
    rid = digest[:24]
    with LOCK:
        if (DATA / f"{rid}.json").exists():
            return metadata(rid)
        meta = {
            "id": rid,
            "name": name,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sha256": digest,
            "sfreq": rec.sfreq,
            "channels": rec.channels,
            "duration": rec.duration,
            "events": rec.events,
            "provenance": provenance or {},
            "annotations": [],
            "context": {
                "activity": "",
                "sleep": "",
                "notes": "",
                "electrode_setup": "",
                "session_key": "",
            },
        }
        np.savez_compressed(DATA / f"{rid}.npz", data=rec.data)
        atomic_json(DATA / f"{rid}.json", meta)
    return meta


@app.get("/api/recordings")
def list_recordings():
    with LOCK:
        return sorted(
            [json.loads(p.read_text()) for p in DATA.glob("*.json")],
            key=lambda x: x["created_at"],
            reverse=True,
        )


@app.post("/api/demo")
def demo():
    return save_recording(synthetic(), "A first look at EEG", "synthetic", {"seed": 42})


@app.get("/api/recordings/{rid}")
def get_recording(rid: str):
    return metadata(rid)


@app.post("/api/sample")
def sample():
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.get(SOURCE)
            response.raise_for_status()
        if len(response.content) > MAX_UPLOAD:
            raise ValueError("Sample exceeds the import size limit.")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.edf"
            path.write_bytes(response.content)
            rec = read_edf(path)
        return save_recording(
            rec,
            "PhysioNet · S001 · Eyes open",
            "public",
            {
                "url": SOURCE,
                "dataset": SOURCE_PAGE,
                "subject": "S001",
                "run": "R01",
                "task": "Eyes-open baseline",
                "license": "Open Data Commons Attribution 1.0",
                "file_sha256": hashlib.sha256(response.content).hexdigest(),
                "note": "Source EDF downloaded on request; normalized signals stored locally.",
            },
        )
    except (httpx.HTTPError, OSError, ValueError) as e:
        raise HTTPException(400, f"Could not load PhysioNet sample: {e}") from e


@app.post("/api/import")
def import_file(file: UploadFile = File(...), sfreq: float = Form(256), units: str = Form("uV")):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".edf", ".csv"}:
        raise HTTPException(400, "Import an EDF/EDF+ or CSV file.")
    try:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"recording{suffix}"
            size = 0
            digest = hashlib.sha256()
            with path.open("wb") as f:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise ValueError("File is larger than 64 MB. Import a shorter excerpt.")
                    digest.update(chunk)
                    f.write(chunk)
            rec = read_edf(path) if suffix == ".edf" else read_csv(path, sfreq, units)
        return save_recording(
            rec,
            Path(file.filename).name,
            "imported",
            {
                "file_sha256": digest.hexdigest(),
                "format": suffix,
                "csv_units": units if suffix == ".csv" else None,
            },
        )
    except (ValueError, OSError, RuntimeError, UnicodeError) as e:
        raise HTTPException(400, f"Could not import recording: {e}") from e
    finally:
        file.file.close()


class AnalysisOptions(BaseModel):
    baseline_start: float = Field(default=0, ge=0, allow_inf_nan=False)
    baseline_end: float = Field(default=20, gt=0, allow_inf_nan=False)
    threshold: float = Field(default=4, ge=2, le=20, allow_inf_nan=False)
    mains: int = 60


@app.post("/api/recordings/{rid}/analysis")
def run_analysis(rid: str, options: AnalysisOptions):
    if options.mains not in (0, 50, 60):
        raise HTTPException(400, "Mains must be 0, 50 or 60 Hz.")
    try:
        result = analyze(recording(rid), **options.model_dump())
        result["recording_id"] = rid
        result["recording_sha256"] = metadata(rid)["sha256"]
        result["app_version"] = "0.1.0"
        return result
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/recordings/{rid}/segment")
def segment(
    rid: str,
    start: float = 0,
    seconds: float = 10,
    channels: str = "0,1,2,3",
    filtered: bool = True,
    mains: int = 60,
):
    rec = recording(rid)
    try:
        picks = list(dict.fromkeys(int(x) for x in channels.split(",")))
        if not 1 <= len(picks) <= 8 or any(p < 0 or p >= len(rec.channels) for p in picks):
            raise ValueError("Select 1–8 valid channels.")
        if not np.isfinite(start) or not np.isfinite(seconds) or not 0 <= start < rec.duration:
            raise ValueError("Start must be inside the recording.")
        if not 1 <= seconds <= 30 or mains not in (0, 50, 60):
            raise ValueError("Invalid display window or mains setting.")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    a, b = int(start * rec.sfreq), min(rec.data.shape[1], int((start + seconds) * rec.sfreq))
    # Include context for display filtering; do not filter just the visible edge.
    if filtered:
        left, right = max(0, a - int(5 * rec.sfreq)), min(rec.data.shape[1], b + int(5 * rec.sfreq))
        y = clean(rec.data[picks, left:right], rec.sfreq, mains)[:, a - left : b - left]
    else:
        y = rec.data[picks, a:b]
    # Min/max envelope preserves peaks without aliasing caused by naive sample skipping.
    step = max(1, int(np.ceil(y.shape[1] / 1000)))
    times, low, high = [], [], []
    for offset in range(0, y.shape[1], step):
        block = y[:, offset : offset + step]
        times.append((a + offset) / rec.sfreq)
        low.append(block.min(axis=1))
        high.append(block.max(axis=1))
    freqs, psd = signal.welch(y, fs=rec.sfreq, nperseg=min(y.shape[1], int(2 * rec.sfreq)))
    mask = (freqs >= 1) & (freqs <= 40)
    return {
        "times": times,
        "low": np.array(low).T.tolist(),
        "high": np.array(high).T.tolist(),
        "channels": [rec.channels[p] for p in picks],
        "frequencies": freqs[mask].tolist(),
        "psd": psd[:, mask].tolist(),
        "start": a / rec.sfreq,
        "end": b / rec.sfreq,
    }


class Annotation(BaseModel):
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(ge=0, allow_inf_nan=False)
    label: str = Field(min_length=1, max_length=500)


@app.post("/api/recordings/{rid}/annotations")
def add_annotation(rid: str, note: Annotation):
    with LOCK:
        meta = metadata(rid)
        if not note.label.strip() or not note.start <= note.end <= meta["duration"]:
            raise HTTPException(400, "Provide a note and a valid interval within the recording.")
        meta["annotations"].append({"id": uuid.uuid4().hex, **note.model_dump()})
        atomic_json(DATA / f"{rid}.json", meta)
    return meta


@app.delete("/api/recordings/{rid}/annotations/{aid}")
def remove_annotation(rid: str, aid: str):
    with LOCK:
        meta = metadata(rid)
        meta["annotations"] = [a for a in meta["annotations"] if a["id"] != aid]
        atomic_json(DATA / f"{rid}.json", meta)
    return meta


class Context(BaseModel):
    activity: str = Field(default="", max_length=200)
    sleep: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=2000)
    electrode_setup: str = Field(default="", max_length=500)
    session_key: str = Field(default="", max_length=200)


@app.put("/api/recordings/{rid}/context")
def save_context(rid: str, context: Context):
    with LOCK:
        meta = metadata(rid)
        meta["context"] = context.model_dump()
        atomic_json(DATA / f"{rid}.json", meta)
    return meta


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
