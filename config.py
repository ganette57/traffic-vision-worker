from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    app_name: str = "traffic-vision-worker"
    log_level: str = "INFO"
    camera_source: str | None = None
    reconnect_interval_seconds: float = 1.5
    count_interval_seconds: float = 0.2
    default_round_duration_seconds: int = 0
    fallback_width: int = 960
    fallback_height: int = 540
    jpeg_quality: int = 80
    min_motion_area: int = 600
    enable_yolo: bool = False
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence: float = 0.25


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("TVW_APP_NAME", "traffic-vision-worker"),
        log_level=os.getenv("TVW_LOG_LEVEL", "INFO").upper(),
        camera_source=os.getenv("TVW_CAMERA_SOURCE"),
        reconnect_interval_seconds=_read_float("TVW_RECONNECT_INTERVAL_SECONDS", 1.5),
        count_interval_seconds=_read_float("TVW_COUNT_INTERVAL_SECONDS", 0.2),
        default_round_duration_seconds=_read_int("TVW_DEFAULT_ROUND_DURATION_SECONDS", 0),
        fallback_width=_read_int("TVW_FALLBACK_WIDTH", 960),
        fallback_height=_read_int("TVW_FALLBACK_HEIGHT", 540),
        jpeg_quality=_read_int("TVW_JPEG_QUALITY", 80),
        min_motion_area=_read_int("TVW_MIN_MOTION_AREA", 600),
        enable_yolo=_read_bool("TVW_ENABLE_YOLO", False),
        yolo_model_path=os.getenv("TVW_YOLO_MODEL_PATH", "yolov8n.pt"),
        yolo_confidence=_read_float("TVW_YOLO_CONFIDENCE", 0.25),
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
