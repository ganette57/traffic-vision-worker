from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, Query
from fastapi.responses import Response

from config import configure_logging, get_settings
from counter import RoundManager, utc_now

SETTINGS = get_settings()
configure_logging(SETTINGS.log_level)
LOGGER = logging.getLogger("traffic_vision.api")
ROUND_MANAGER = RoundManager(SETTINGS)

app = FastAPI(title=SETTINGS.app_name)


def _extract_round_id(
    payload: dict[str, Any],
    fallback: str | None = None,
    *,
    allow_generate: bool = True,
) -> str | None:
    for key in ("roundId", "round_id", "id", "marketRoundId"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(fallback, str) and fallback.strip():
        return fallback
    if allow_generate:
        return f"round-{uuid4().hex[:12]}"
    return None


def _extract_source(payload: dict[str, Any]) -> str | int | None:
    for key in ("source", "cameraUrl", "streamUrl", "videoUrl", "rtspUrl", "camera"):
        value = payload.get(key)
        if isinstance(value, (str, int)):
            if isinstance(value, str) and not value.strip():
                continue
            return value
    return None


def _parse_utc_datetime(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw.astimezone(timezone.utc)
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if isinstance(raw, str):
        candidate = raw.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = f"{candidate[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _extract_ends_at(payload: dict[str, Any]) -> datetime | None:
    for key in ("endsAt", "endAt", "expiresAt"):
        parsed = _parse_utc_datetime(payload.get(key))
        if parsed is not None:
            return parsed
    for key in ("durationSec", "durationSeconds", "duration"):
        duration = payload.get(key)
        if isinstance(duration, (int, float)):
            seconds = int(duration)
            if seconds > 0:
                return utc_now() + timedelta(seconds=seconds)
    return ROUND_MANAGER.ensure_round_ends_at(None)


def _status_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "roundId": snapshot.get("roundId"),
        "running": bool(snapshot.get("running")),
        "currentCount": int(snapshot.get("currentCount") or 0),
        "startedAt": snapshot.get("startedAt"),
        "endsAt": snapshot.get("endsAt"),
        "cameraAvailable": bool(snapshot.get("cameraAvailable")),
        "lastError": snapshot.get("lastError"),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SETTINGS.app_name,
        "runningRounds": ROUND_MANAGER.running_rounds(),
    }


@app.post("/rounds/start")
def start_round(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    round_id = _extract_round_id(payload, allow_generate=True)
    source = _extract_source(payload)
    ends_at = _extract_ends_at(payload)
    snapshot = ROUND_MANAGER.start_round(
        round_id=round_id,
        source=source,
        ends_at=ends_at,
        metadata=payload,
    )
    LOGGER.info("API round start roundId=%s", round_id)
    return {"status": "started", **_status_payload(snapshot)}


@app.post("/rounds/stop")
def stop_round(
    payload: dict[str, Any] = Body(default_factory=dict),
    round_id_query: str | None = Query(default=None, alias="roundId"),
) -> dict[str, Any]:
    round_id = _extract_round_id(payload, fallback=round_id_query, allow_generate=False)
    if not round_id:
        rounds = ROUND_MANAGER.get_all_rounds()
        round_id = rounds[0]["roundId"] if rounds else None
    if not round_id:
        return {
            "status": "stopped",
            "roundId": None,
            "running": False,
            "currentCount": 0,
            "startedAt": None,
            "endsAt": None,
            "cameraAvailable": False,
            "lastError": "round not found",
        }
    snapshot = ROUND_MANAGER.stop_round(round_id)
    if snapshot is None:
        return {
            "status": "stopped",
            "roundId": round_id,
            "running": False,
            "currentCount": 0,
            "startedAt": None,
            "endsAt": None,
            "cameraAvailable": False,
            "lastError": "round not found",
        }
    LOGGER.info("API round stop roundId=%s", round_id)
    return {"status": "stopped", **_status_payload(snapshot)}


@app.get("/rounds/status")
def round_status(round_id: str | None = Query(default=None, alias="roundId")) -> dict[str, Any]:
    if round_id:
        snapshot = ROUND_MANAGER.get_round(round_id)
        if snapshot is None:
            return {
                "roundId": round_id,
                "running": False,
                "currentCount": 0,
                "startedAt": None,
                "endsAt": None,
                "cameraAvailable": False,
                "lastError": "round not found",
            }
        return _status_payload(snapshot)
    rounds = ROUND_MANAGER.get_all_rounds()
    return {
        "rounds": [_status_payload(item) for item in rounds],
        "runningRounds": sum(1 for item in rounds if item.get("running")),
    }


@app.get("/frame.jpg")
def frame(round_id: str | None = Query(default=None, alias="roundId")) -> Response:
    data = ROUND_MANAGER.get_frame(round_id)
    return Response(content=data, media_type="image/jpeg")


# Legacy aliases kept for compatibility.
@app.post("/start")
def legacy_start(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return start_round(payload)


@app.post("/stop")
def legacy_stop(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return stop_round(payload=payload, round_id_query=None)


@app.post("/count")
def legacy_count(round_id: str | None = Query(default=None, alias="roundId")) -> dict[str, Any]:
    status = round_status(round_id=round_id)
    if "currentCount" in status:
        return {"count": int(status["currentCount"])}
    rounds = status.get("rounds", [])
    if not rounds:
        return {"count": 0}
    return {"count": int(rounds[0].get("currentCount") or 0)}
