"""FastAPI service: web UI + JSON API for HazardLens."""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from . import vision as V
from .actions import ActionStore
from .agent import make_llm_planner, result_payload, run_inspection

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
log = logging.getLogger("hazardlens.app")

ROOT = Path(__file__).resolve().parent
SAMPLES = Path(os.getenv("SAMPLES_DIR", ROOT.parent / "data" / "samples"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
MAX_SIDE = 4096
APPROVER_TOKEN = os.getenv("APPROVER_TOKEN", "")  # if set, required to approve/reject

app = FastAPI(title="HazardLens", version=__version__,
              description="Agentic trip & slip hazard inspection with OpenCV 5")
store = ActionStore()
RUNS: dict[str, dict] = {}  # recent runs (bounded) for the UI; traces also go to S3 when configured


def _decode(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(413, f"image must be 1 byte to {MAX_BYTES} bytes")
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)  # re-decode strips EXIF/GPS
    if img is None:
        raise HTTPException(415, "unsupported or corrupt image (use JPEG/PNG)")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE or min(h, w) < 64:
        raise HTTPException(422, "image side must be between 64 and 4096 px")
    if w > 1280:
        img = cv.resize(img, (1280, int(h * 1280 / w)), interpolation=cv.INTER_AREA)
    return img


def _run(img: np.ndarray, location: str, planner: str) -> dict:
    s = run_inspection(img, store, location=re.sub(r"[^\w .,'-]", "", location)[:120], planner=planner)
    payload = result_payload(s)
    payload["trace_uri"] = store.persist_trace(payload)
    RUNS[s.run_id] = {k: v for k, v in payload.items() if k != "annotated_jpeg_b64"}
    while len(RUNS) > 200:
        RUNS.pop(next(iter(RUNS)))
    log.info(json.dumps({"event": "run", "run_id": s.run_id, "outcome": s.outcome, "steps": len(s.trace),
                         "planners": s.planner_used}))
    return payload


@app.get("/healthz")
def healthz():
    return {"status": "ok", "opencv": V.opencv_version(), "version": __version__,
            "llm_configured": make_llm_planner() is not None}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/samples")
def samples():
    labels = json.loads((SAMPLES / "labels.json").read_text()) if (SAMPLES / "labels.json").exists() else []
    return [{"name": l["name"], "kind": l["kind"], "expected": l["expected_action"]} for l in labels]


@app.get("/api/samples/{name}")
def sample_image(name: str):
    if not re.fullmatch(r"scene_\d{3}\.jpg", name) or not (SAMPLES / name).exists():
        raise HTTPException(404)
    return FileResponse(SAMPLES / name, media_type="image/jpeg")


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...), location: str = Form(""), planner: str = Form("auto")):
    if planner not in ("auto", "rule", "llm"):
        raise HTTPException(422, "planner must be auto|rule|llm")
    return _run(_decode(await file.read()), location, planner)


@app.post("/api/inspect_sample/{name}")
def inspect_sample(name: str, planner: str = "auto"):
    if not re.fullmatch(r"scene_\d{3}\.jpg", name) or not (SAMPLES / name).exists():
        raise HTTPException(404)
    return _run(cv.imread(str(SAMPLES / name)), f"demo {name}", planner)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404)
    return RUNS[run_id]


@app.get("/api/approvals")
def approvals():
    return {"approvals": list(store.approvals.values())[-50:], "work_orders": list(store.work_orders.values())[-50:],
            "notifications": store.notifications[-50:]}


class Decision(BaseModel):
    approve: bool
    reviewer: str = Field("anonymous", max_length=60)


@app.post("/api/approvals/{approval_id}")
def decide(approval_id: str, d: Decision, x_approver_token: str = Header("")):
    if APPROVER_TOKEN and not hmac.compare_digest(x_approver_token, APPROVER_TOKEN):
        raise HTTPException(401, "approver token required")
    try:
        return store.decide(approval_id, d.approve, d.reviewer)
    except KeyError:
        raise HTTPException(404)


@app.exception_handler(Exception)
async def unhandled(_, exc):  # never leak stack traces to clients
    log.exception("unhandled error")
    return JSONResponse({"detail": "internal error"}, status_code=500)
