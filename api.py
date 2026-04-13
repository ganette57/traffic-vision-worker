from __future__ import annotations

from typing import Dict, List, Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

from counter import COCO_CLASS_NAME_TO_ID, ROUND_MANAGER, RoundSpec


class LinePayload(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class StartRoundPayload(BaseModel):
    roundId: str = Field(min_length=1)
    streamUrl: str = Field(min_length=1)
    cameraId: Optional[str] = None
    sourceType: Literal["local_video", "remote_stream"] = "local_video"
    durationSec: int = Field(default=60, ge=1, le=3600)
    line: LinePayload
    classes: List[str] = Field(default_factory=lambda: ["car", "bus", "truck", "motorcycle"])
    tracker: str = "bytetrack"


class RoundStatusResponse(BaseModel):
    roundId: str
    status: Literal["running", "ended", "stopped"]
    currentCount: int
    startedAt: int
    endsAt: int
    sourceOpened: bool
    lastFrameAt: Optional[int] = None
    detectionsLastFrame: int = 0
    frameWidth: Optional[int] = None
    frameHeight: Optional[int] = None
    countingLineX: Optional[int] = None
    countingLineY: Optional[int] = None
    lastCountedTrackId: Optional[int] = None
    lastCrossingDirection: Optional[str] = None
    lastDecisionTrackId: Optional[int] = None
    lastDecisionReason: Optional[str] = None
    lastDecisionCounted: Optional[bool] = None
    lastTrackDeltaX: Optional[float] = None
    lastTrackSamples: Optional[int] = None


class StopRoundPayload(BaseModel):
    reason: str = "manual_stop"


class StopRoundCompatPayload(BaseModel):
    roundId: Optional[str] = None
    reason: str = "manual_stop"


class StopRoundResponse(BaseModel):
    roundId: str
    finalCount: int


app = FastAPI(title="Traffic Vision Worker", version="0.1.0")


def _build_placeholder_jpeg(round_id: str) -> bytes:
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    frame[:, :] = (24, 24, 24)
    cv2.putText(frame, "Waiting for camera...", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2)
    cv2.putText(frame, f"roundId: {round_id[:64]}", (30, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 2)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        return encoded.tobytes()
    # Last-resort valid JPEG SOI/EOI markers.
    return b"\xff\xd8\xff\xd9"


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"ok": "true"}


@app.post("/rounds/start", response_model=RoundStatusResponse)
def start_round(payload: StartRoundPayload) -> Dict[str, object]:
    classes = [c.strip().lower() for c in payload.classes if str(c).strip()]
    class_ids = [COCO_CLASS_NAME_TO_ID[c] for c in classes if c in COCO_CLASS_NAME_TO_ID]
    if not class_ids:
        raise HTTPException(status_code=400, detail="No supported classes provided.")

    spec = RoundSpec(
        round_id=payload.roundId.strip(),
        stream_url=payload.streamUrl.strip(),
        camera_id=(payload.cameraId or "").strip() or None,
        source_type=payload.sourceType,
        duration_sec=int(payload.durationSec),
        line={
            "x1": float(payload.line.x1),
            "y1": float(payload.line.y1),
            "x2": float(payload.line.x2),
            "y2": float(payload.line.y2),
        },
        classes=classes,
        class_ids=class_ids,
        tracker=(payload.tracker or "bytetrack").strip() or "bytetrack",
    )
    return ROUND_MANAGER.start_round(spec)


@app.get("/rounds/{round_id}/status", response_model=RoundStatusResponse)
def round_status(round_id: str) -> Dict[str, object]:
    status = ROUND_MANAGER.get_status(round_id.strip())
    if not status:
        raise HTTPException(status_code=404, detail="Round not found.")
    return status


@app.get("/rounds/{round_id}/frame.jpg")
def round_frame(round_id: str) -> Response:
    normalized_round_id = round_id.strip()
    status = ROUND_MANAGER.get_status(normalized_round_id)
    if not status:
        raise HTTPException(status_code=404, detail="Round not found.")

    frame_jpeg = ROUND_MANAGER.get_debug_frame_jpeg(normalized_round_id)
    if not frame_jpeg:
        frame_jpeg = _build_placeholder_jpeg(normalized_round_id)
    return Response(
        content=frame_jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


def _stop_round_by_id(round_id: str, reason: str) -> Dict[str, object]:
    out = ROUND_MANAGER.stop_round(round_id.strip(), reason=(reason or "manual_stop"))
    if not out:
        raise HTTPException(status_code=404, detail="Round not found.")
    return out


@app.post("/rounds/{round_id}/stop", response_model=StopRoundResponse)
def stop_round(round_id: str, payload: StopRoundPayload) -> Dict[str, object]:
    return _stop_round_by_id(round_id=round_id, reason=(payload.reason or "manual_stop"))


@app.post("/rounds/stop", response_model=StopRoundResponse)
def stop_round_compat(
    payload: StopRoundCompatPayload,
    round_id: Optional[str] = Query(default=None, alias="roundId"),
) -> Dict[str, object]:
    resolved_round_id = ((payload.roundId or round_id or "").strip())
    if not resolved_round_id:
        raise HTTPException(status_code=400, detail="roundId is required.")
    return _stop_round_by_id(round_id=resolved_round_id, reason=(payload.reason or "manual_stop"))
