from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import cv2
import numpy as np

from config import Settings
from yt_resolver import resolve_stream_source

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

LOGGER = logging.getLogger("traffic_vision.counter")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso8601(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("could not encode frame to JPEG")
    return data.tobytes()


def build_placeholder_jpeg(
    settings: Settings,
    title: str = "Waiting for camera...",
    detail: str | None = None,
) -> bytes:
    canvas = np.zeros((settings.fallback_height, settings.fallback_width, 3), dtype=np.uint8)
    canvas[:, :] = (20, 20, 20)
    ts = utc_now().strftime("%Y-%m-%d %H:%M:%SZ")
    cv2.putText(canvas, title, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 170, 255), 3)
    cv2.putText(canvas, ts, (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    if detail:
        cv2.putText(canvas, detail[:80], (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    return encode_jpeg(canvas, settings.jpeg_quality)


class VisionCounter:
    """
    Counter engine that can use YOLO (if enabled) and gracefully falls back to motion counting.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=32)
        self._yolo_model = None
        if settings.enable_yolo and YOLO is not None:
            try:
                self._yolo_model = YOLO(settings.yolo_model_path)
                LOGGER.info("YOLO model loaded from %s", settings.yolo_model_path)
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("YOLO initialization failed; motion fallback enabled: %s", exc)
                self._yolo_model = None
        elif settings.enable_yolo:
            LOGGER.warning("TVW_ENABLE_YOLO enabled but ultralytics import failed; motion fallback enabled")

    def count(self, frame: np.ndarray) -> int:
        if self._yolo_model is not None:
            count = self._count_with_yolo(frame)
            if count is not None:
                return count
        return self._count_with_motion(frame)

    def _count_with_yolo(self, frame: np.ndarray) -> int | None:
        try:
            results = self._yolo_model.predict(
                frame,
                verbose=False,
                conf=self.settings.yolo_confidence,
                classes=[0],  # person class
            )
            if not results:
                return 0
            boxes = results[0].boxes
            return int(len(boxes) if boxes is not None else 0)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("YOLO inference failed; using motion fallback: %s", exc)
            return None

    def _count_with_motion(self, frame: np.ndarray) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = self._bg_subtractor.apply(gray)
        _, thresh = cv2.threshold(fgmask, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = sum(1 for c in contours if cv2.contourArea(c) >= self.settings.min_motion_area)
        return max(0, min(count, 200))


@dataclass(slots=True)
class RoundInit:
    round_id: str
    source: str | int | None
    started_at: datetime
    ends_at: datetime | None
    metadata: dict[str, Any]


class RoundRuntime:
    def __init__(self, settings: Settings, init: RoundInit) -> None:
        self.settings = settings
        self.round_id = init.round_id
        self.source = init.source
        self.started_at = init.started_at
        self.ends_at = init.ends_at
        self.metadata = init.metadata
        self.running = False
        self.current_count = 0
        self.camera_available = False
        self.last_error: str | None = None
        self.updated_at = self.started_at
        self._counter = VisionCounter(settings)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_frame_jpeg = build_placeholder_jpeg(settings)

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self.running = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name=f"round-{self.round_id}", daemon=True)
            self._thread.start()
        LOGGER.info("Round start roundId=%s source=%s endsAt=%s", self.round_id, self.source, self.ends_at)

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.5)
        with self._lock:
            self.running = False
            self.updated_at = utc_now()
        LOGGER.info("Round stop roundId=%s currentCount=%s", self.round_id, self.current_count)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "roundId": self.round_id,
                "running": self.running,
                "currentCount": int(self.current_count),
                "startedAt": to_iso8601(self.started_at),
                "endsAt": to_iso8601(self.ends_at),
                "cameraAvailable": self.camera_available,
                "lastError": self.last_error,
                "updatedAt": to_iso8601(self.updated_at),
            }

    def frame_jpeg(self) -> bytes:
        with self._lock:
            return self._last_frame_jpeg

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            while not self._stop_event.is_set():
                now = utc_now()
                if self.ends_at is not None and now >= self.ends_at:
                    break

                if cap is None or not cap.isOpened():
                    cap = self._open_capture()
                    if cap is None or not cap.isOpened():
                        self._set_unavailable("camera unavailable")
                        time.sleep(self.settings.reconnect_interval_seconds)
                        continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    self._set_unavailable("failed to read frame")
                    try:
                        cap.release()
                    finally:
                        cap = None
                    time.sleep(self.settings.reconnect_interval_seconds)
                    continue

                count = self._counter.count(frame)
                annotated = self._annotate_frame(frame, count)
                self._set_frame(annotated, count)
                time.sleep(self.settings.count_interval_seconds)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Round runtime error roundId=%s err=%s", self.round_id, exc)
            self._set_unavailable(str(exc))
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:  # pragma: no cover
                    pass
            with self._lock:
                self.running = False
                self.updated_at = utc_now()
            LOGGER.info("Round loop finished roundId=%s finalCount=%s", self.round_id, self.current_count)

    def _resolve_source(self) -> str | int | None:
        source = self.source if self.source is not None else self.settings.camera_source
        if source is None:
            return None
        if isinstance(source, str):
            trimmed = source.strip()
            if trimmed.isdigit():
                return int(trimmed)
            return resolve_stream_source(trimmed)
        return source

    def _open_capture(self) -> cv2.VideoCapture | None:
        source = self._resolve_source()
        if source is None:
            return None
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:  # pragma: no cover
                pass
            return None
        return cap

    def _set_unavailable(self, error_message: str) -> None:
        placeholder = build_placeholder_jpeg(
            self.settings,
            title="Waiting for camera...",
            detail=f"roundId={self.round_id}",
        )
        with self._lock:
            self.camera_available = False
            self.last_error = error_message
            self.updated_at = utc_now()
            self._last_frame_jpeg = placeholder
        LOGGER.error("Round error roundId=%s err=%s", self.round_id, error_message)

    def _set_frame(self, frame: np.ndarray, count: int) -> None:
        jpeg = encode_jpeg(frame, self.settings.jpeg_quality)
        previous = self.current_count
        with self._lock:
            self.camera_available = True
            self.last_error = None
            self.current_count = count
            self.updated_at = utc_now()
            self._last_frame_jpeg = jpeg
        if count != previous:
            LOGGER.info("Round count update roundId=%s count=%s", self.round_id, count)

    def _annotate_frame(self, frame: np.ndarray, count: int) -> np.ndarray:
        out = frame.copy()
        cv2.putText(
            out,
            f"roundId: {self.round_id}",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            out,
            f"count: {count}",
            (16, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
        )
        return out


class RoundManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._rounds: dict[str, RoundRuntime] = {}
        self._lock = threading.Lock()

    def start_round(
        self,
        *,
        round_id: str,
        source: str | int | None,
        ends_at: datetime | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        init = RoundInit(
            round_id=round_id,
            source=source,
            started_at=utc_now(),
            ends_at=ends_at,
            metadata=metadata,
        )
        runtime = RoundRuntime(self.settings, init)

        existing: RoundRuntime | None = None
        with self._lock:
            existing = self._rounds.get(round_id)
            self._rounds[round_id] = runtime
        if existing is not None:
            existing.stop()
        runtime.start()
        return runtime.snapshot()

    def stop_round(self, round_id: str) -> dict[str, Any] | None:
        with self._lock:
            runtime = self._rounds.get(round_id)
        if runtime is None:
            return None
        runtime.stop()
        return runtime.snapshot()

    def get_round(self, round_id: str) -> dict[str, Any] | None:
        with self._lock:
            runtime = self._rounds.get(round_id)
        if runtime is None:
            return None
        return runtime.snapshot()

    def get_all_rounds(self) -> list[dict[str, Any]]:
        with self._lock:
            rounds = list(self._rounds.values())
        snapshots = [runtime.snapshot() for runtime in rounds]
        snapshots.sort(key=lambda item: item["startedAt"] or "", reverse=True)
        return snapshots

    def get_frame(self, round_id: str | None = None) -> bytes:
        runtime: RoundRuntime | None = None
        with self._lock:
            if round_id:
                runtime = self._rounds.get(round_id)
            elif self._rounds:
                runtime = next(iter(self._rounds.values()))
        if runtime is None:
            return build_placeholder_jpeg(self.settings, title="No round running")
        return runtime.frame_jpeg()

    def running_rounds(self) -> int:
        with self._lock:
            rounds = list(self._rounds.values())
        return sum(1 for runtime in rounds if runtime.snapshot().get("running"))

    def ensure_round_ends_at(self, duration_seconds: int | None) -> datetime | None:
        if duration_seconds is None:
            duration_seconds = self.settings.default_round_duration_seconds
        if duration_seconds <= 0:
            return None
        return utc_now() + timedelta(seconds=duration_seconds)
