from __future__ import annotations

from dataclasses import dataclass, field
import os
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import SETTINGS

try:
    import resource
except Exception:
    resource = None


COCO_CLASS_NAME_TO_ID = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}

COCO_CLASS_ID_TO_NAME = {v: k for k, v in COCO_CLASS_NAME_TO_ID.items()}
REMOTE_STREAM_LINE_MARGIN_PX = 5.0
REMOTE_STREAM_MIN_SAMPLES = 2
REMOTE_STREAM_MIN_MOTION_PX = 8.0
REMOTE_STREAM_MIN_BBOX_AREA_PX = 0.0
DEBUG_CROSSING = os.environ.get("TRAFFIC_DEBUG_CROSSING", "1") == "1"
REMOTE_STREAM_HISTORY_SIZE = 10
REMOTE_STREAM_ROI_X_MIN_RATIO = 0.18
REMOTE_STREAM_ROI_X_MAX_RATIO = 0.95
REMOTE_STREAM_ROI_Y_MIN_RATIO = 0.24
REMOTE_STREAM_ROI_Y_MAX_RATIO = 0.90
HIGHWAY_STREAM_SIGNATURES = tuple(
    s.strip().lower()
    for s in os.environ.get(
        "TRAFFIC_HIGHWAY_STREAM_SIGNATURES",
        "highway,autoroute,motorway,freeway,trafficcam",
    ).split(",")
    if s.strip()
)
HIGHWAY_REMOTE_STREAM_LINE_X_RATIO = 0.72
HIGHWAY_REMOTE_STREAM_ROI_X_MIN_RATIO = 0.46
HIGHWAY_REMOTE_STREAM_ROI_X_MAX_RATIO = 0.94
HIGHWAY_REMOTE_STREAM_ROI_Y_MIN_RATIO = 0.58
HIGHWAY_REMOTE_STREAM_ROI_Y_MAX_RATIO = 0.92
HIGHWAY_REMOTE_STREAM_MIN_SAMPLES = 4
HIGHWAY_REMOTE_STREAM_MIN_MOTION_PX = 18.0
HIGHWAY_REMOTE_STREAM_MIN_BBOX_AREA_PX = 2200.0
CAM1_REMOTE_STREAM_LINE_Y_RATIO = 0.43
CAM2_REMOTE_STREAM_LINE_Y_RATIO = 0.48
CAM2_REMOTE_STREAM_SEGMENT_X1_RATIO = 0.57
CAM2_REMOTE_STREAM_SEGMENT_Y1_RATIO = 0.52
CAM2_REMOTE_STREAM_SEGMENT_X2_RATIO = 0.75
CAM2_REMOTE_STREAM_SEGMENT_Y2_RATIO = 0.47
IOWA_REMOTE_STREAM_LINE_Y_RATIO = 0.54
IOWA_REMOTE_STREAM_MIN_SAMPLES = 3
IOWA_REMOTE_STREAM_MIN_MOTION_PX = 7.0
IOWA_REMOTE_STREAM_MIN_BBOX_AREA_PX = 1200.0
IOWA_REMOTE_STREAM_ROI_X_MIN_RATIO = 0.55
IOWA_REMOTE_STREAM_ROI_X_MAX_RATIO = 0.98
IOWA_REMOTE_STREAM_ROI_Y_MIN_RATIO = 0.38
IOWA_REMOTE_STREAM_ROI_Y_MAX_RATIO = 0.96
IOWA_REMOTE_STREAM_SEGMENT_X1_RATIO = 0.47
IOWA_REMOTE_STREAM_SEGMENT_X2_RATIO = 0.74
MARYLAND_REMOTE_STREAM_LINE_Y_RATIO = 0.60
MARYLAND_REMOTE_STREAM_MIN_SAMPLES = 2
MARYLAND_REMOTE_STREAM_MIN_MOTION_PX = 6.0
MARYLAND_REMOTE_STREAM_MIN_BBOX_AREA_PX = 700.0
MARYLAND_REMOTE_STREAM_ROI_X_MIN_RATIO = 0.52
MARYLAND_REMOTE_STREAM_ROI_X_MAX_RATIO = 0.99
MARYLAND_REMOTE_STREAM_ROI_Y_MIN_RATIO = 0.35
MARYLAND_REMOTE_STREAM_ROI_Y_MAX_RATIO = 0.98
MARYLAND_REMOTE_STREAM_SEGMENT_X1_RATIO = 0.47
MARYLAND_REMOTE_STREAM_SEGMENT_X2_RATIO = 0.70

CAM1_STREAM_SIGNATURE = "wf05-24af-4d42-c307-aa51_nj"
CAM2_STREAM_SIGNATURE = "wf05-24af-4d24-2558-f999_nj"
CAM3_STREAM_SIGNATURE = "wf05-24b0-46ee-2155-1a86_nj"
IOWA_STREAM_SIGNATURE = "iowadotsfs2.us-east-1.skyvdn.com/rtplive/dmtv05lb"
MARYLAND_STREAM_SIGNATURE = "strmr5.sha.maryland.gov/rtplive/0900adbd00ee00e30051fa36c4235c0a"
LAS_VEGAS_STREAM_URL = "https://videos-3.earthcam.com/fecnetwork/42116.flv/chunklist_w554170088.m3u8?t=UGa97G27%2BOZQYx%2FZGv8bHLblEJHHKvee7g9yK8V46vLMi6ZfhNViHDRNTwaj6Uqq&td=202604051541"
LAS_VEGAS_STREAM_SIGNATURE = "videos-3.earthcam.com/fecnetwork/42116.flv"
SUPPORTED_PRODUCTION_CAMERA_IDS = {"cam2", "cam3", "iowa"}

CAMERA_REMOTE_STREAM_PROFILES: Dict[str, Dict[str, object]] = {
    "cam1": {
        "line_x1_ratio": 0.0,
        "line_y1_ratio": CAM1_REMOTE_STREAM_LINE_Y_RATIO,
        "line_x2_ratio": 1.0,
        "line_y2_ratio": CAM1_REMOTE_STREAM_LINE_Y_RATIO,
        "roi_x_min_ratio": 0.0,
        "roi_x_max_ratio": 1.0,
        "roi_y_min_ratio": 0.0,
        "roi_y_max_ratio": 1.0,
        "min_samples": 2,
        "min_motion_px": 5.0,
        "min_bbox_area_px": 300.0,
        "line_margin_px": 20.0,
    },
    "cam2": {
        "line_x1_ratio": 0.08,
        "line_y1_ratio": 0.70,
        "line_x2_ratio": 0.20,
        "line_y2_ratio": 0.62,
        "roi_x_min_ratio": 0.00,
        "roi_x_max_ratio": 1.00,
        "roi_y_min_ratio": 0.18,
        "roi_y_max_ratio": 0.90,
        "min_samples": 2,
        "min_motion_px": 5.0,
        "min_bbox_area_px": 250.0,
        "line_margin_px": 20.0,
    },
    "cam3": {
        "line_x1_ratio": 0.10,
        "line_y1_ratio": 0.665,
        "line_x2_ratio": 0.52,
        "line_y2_ratio": 0.665,
        "roi_x_min_ratio": 0.00,
        "roi_x_max_ratio": 1.00,
        "roi_y_min_ratio": 0.28,
        "roi_y_max_ratio": 1.00,
        "min_samples": 2,
        "min_motion_px": 5.0,
        "min_bbox_area_px": 220.0,
        "line_margin_px": 20.0,
    },
    "cam5": {
        "line_x1_ratio": 0.02,
        "line_y1_ratio": 0.56,
        "line_x2_ratio": 0.99,
        "line_y2_ratio": 0.56,
        "roi_x_min_ratio": 0.00,
        "roi_x_max_ratio": 1.00,
        "roi_y_min_ratio": 0.24,
        "roi_y_max_ratio": 1.00,
        "min_samples": 2,
        "min_motion_px": 5.0,
        "min_bbox_area_px": 220.0,
        "line_margin_px": 22.0,
    },
    "las_vegas": {
        "line_x1_ratio": 0.10,
        "line_y1_ratio": 0.60,
        "line_x2_ratio": 0.90,
        "line_y2_ratio": 0.60,
        "roi_x_min_ratio": 0.05,
        "roi_x_max_ratio": 0.95,
        "roi_y_min_ratio": 0.30,
        "roi_y_max_ratio": 0.90,
        "min_samples": 2,
        "min_motion_px": 5.0,
        "min_bbox_area_px": 300.0,
        "line_margin_px": 12.0,
    },
    "iowa": {
        "line_x1_ratio": IOWA_REMOTE_STREAM_SEGMENT_X1_RATIO,
        "line_y1_ratio": IOWA_REMOTE_STREAM_LINE_Y_RATIO,
        "line_x2_ratio": IOWA_REMOTE_STREAM_SEGMENT_X2_RATIO,
        "line_y2_ratio": IOWA_REMOTE_STREAM_LINE_Y_RATIO,
        "roi_x_min_ratio": 0.45,
        "roi_x_max_ratio": 1.0,
        "roi_y_min_ratio": 0.30,
        "roi_y_max_ratio": 1.0,
        "min_samples": 2,
        "min_motion_px": 5.0,
        "min_bbox_area_px": 200.0,
        "line_margin_px": 20.0,
    },
    "maryland": {
        "line_x1_ratio": MARYLAND_REMOTE_STREAM_SEGMENT_X1_RATIO,
        "line_y1_ratio": MARYLAND_REMOTE_STREAM_LINE_Y_RATIO,
        "line_x2_ratio": MARYLAND_REMOTE_STREAM_SEGMENT_X2_RATIO,
        "line_y2_ratio": MARYLAND_REMOTE_STREAM_LINE_Y_RATIO,
        "roi_x_min_ratio": MARYLAND_REMOTE_STREAM_ROI_X_MIN_RATIO,
        "roi_x_max_ratio": MARYLAND_REMOTE_STREAM_ROI_X_MAX_RATIO,
        "roi_y_min_ratio": MARYLAND_REMOTE_STREAM_ROI_Y_MIN_RATIO,
        "roi_y_max_ratio": MARYLAND_REMOTE_STREAM_ROI_Y_MAX_RATIO,
        "min_samples": 2,
        "min_motion_px": 6.0,
        "min_bbox_area_px": 300.0,
        "line_margin_px": 12.0,
    },
}


def _line_signed_distance(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> float:
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    line_len = max(1e-6, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    return cross / line_len


def _clamp_int(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _is_highway_remote_profile(stream_url: str, round_id: str) -> bool:
    forced = os.environ.get("TRAFFIC_REMOTE_PROFILE", "").strip().lower()
    if forced == "highway":
        return True
    source = f"{stream_url} {round_id}".lower()
    return any(signature in source for signature in HIGHWAY_STREAM_SIGNATURES)


def _normalize_camera_id(value: Optional[str]) -> str:
    camera_id = str(value or "").strip().lower()
    if camera_id in CAMERA_REMOTE_STREAM_PROFILES:
        return camera_id
    return ""


def _camera_id_from_stream(stream_url: str) -> str:
    source = str(stream_url or "").strip().lower()
    if not source:
        return ""
    if CAM1_STREAM_SIGNATURE in source:
        return "cam1"
    if CAM2_STREAM_SIGNATURE in source:
        return "cam2"
    if CAM3_STREAM_SIGNATURE in source:
        return "cam3"
    if IOWA_STREAM_SIGNATURE in source:
        return "iowa"
    if MARYLAND_STREAM_SIGNATURE in source:
        return "maryland"
    if LAS_VEGAS_STREAM_SIGNATURE in source:
        return "las_vegas"
    return ""


def _resolve_remote_profile(camera_id: Optional[str], stream_url: str, round_id: str) -> Dict[str, object]:
    normalized_camera_id = _normalize_camera_id(camera_id)
    if not normalized_camera_id:
        normalized_camera_id = _camera_id_from_stream(stream_url)

    if normalized_camera_id and normalized_camera_id in CAMERA_REMOTE_STREAM_PROFILES:
        profile = dict(CAMERA_REMOTE_STREAM_PROFILES[normalized_camera_id])
        profile["profile_id"] = normalized_camera_id
        profile["profile_source"] = "camera_id_or_stream_signature"
        return profile

    if _is_highway_remote_profile(stream_url, round_id):
        return {
            "profile_id": "highway_fallback",
            "profile_source": "highway_signature",
            "line_x1_ratio": HIGHWAY_REMOTE_STREAM_LINE_X_RATIO,
            "line_y1_ratio": 0.0,
            "line_x2_ratio": HIGHWAY_REMOTE_STREAM_LINE_X_RATIO,
            "line_y2_ratio": 1.0,
            "roi_x_min_ratio": HIGHWAY_REMOTE_STREAM_ROI_X_MIN_RATIO,
            "roi_x_max_ratio": HIGHWAY_REMOTE_STREAM_ROI_X_MAX_RATIO,
            "roi_y_min_ratio": HIGHWAY_REMOTE_STREAM_ROI_Y_MIN_RATIO,
            "roi_y_max_ratio": HIGHWAY_REMOTE_STREAM_ROI_Y_MAX_RATIO,
            "min_samples": HIGHWAY_REMOTE_STREAM_MIN_SAMPLES,
            "min_motion_px": HIGHWAY_REMOTE_STREAM_MIN_MOTION_PX,
            "min_bbox_area_px": HIGHWAY_REMOTE_STREAM_MIN_BBOX_AREA_PX,
            "line_margin_px": REMOTE_STREAM_LINE_MARGIN_PX,
        }

    return {
        "profile_id": "default_remote",
        "profile_source": "default",
        "line_x1_ratio": 0.50,
        "line_y1_ratio": 0.0,
        "line_x2_ratio": 0.50,
        "line_y2_ratio": 1.0,
        "roi_x_min_ratio": REMOTE_STREAM_ROI_X_MIN_RATIO,
        "roi_x_max_ratio": REMOTE_STREAM_ROI_X_MAX_RATIO,
        "roi_y_min_ratio": REMOTE_STREAM_ROI_Y_MIN_RATIO,
        "roi_y_max_ratio": REMOTE_STREAM_ROI_Y_MAX_RATIO,
        "min_samples": REMOTE_STREAM_MIN_SAMPLES,
        "min_motion_px": REMOTE_STREAM_MIN_MOTION_PX,
        "min_bbox_area_px": REMOTE_STREAM_MIN_BBOX_AREA_PX,
        "line_margin_px": REMOTE_STREAM_LINE_MARGIN_PX,
    }


def _is_supported_production_camera(camera_id: str) -> bool:
    return str(camera_id or "").strip().lower() in SUPPORTED_PRODUCTION_CAMERA_IDS


@dataclass
class FFmpegFrameReader:
    source_url: str
    max_width: int
    process: Optional[subprocess.Popen] = None
    frame_width: int = 0
    frame_height: int = 0
    frame_size_bytes: int = 0

    def _probe_source_size(self) -> Tuple[int, int]:
        if not shutil.which("ffprobe"):
            raise RuntimeError("ffprobe not found on PATH")
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            self.source_url,
        ]
        completed = subprocess.run(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip()
            raise RuntimeError(stderr_text or "ffprobe failed to read source stream")
        size_token = ""
        for line in (completed.stdout or "").splitlines():
            line = line.strip()
            if line:
                size_token = line
                break
        if "x" not in size_token:
            raise RuntimeError("ffprobe did not return a video size")
        width_token, height_token = size_token.lower().split("x", 1)
        width = int(width_token)
        height = int(height_token)
        if width <= 0 or height <= 0:
            raise RuntimeError("invalid source video size from ffprobe")
        return width, height

    def _read_process_error(self) -> str:
        if self.process is None or self.process.stderr is None:
            return ""
        try:
            data = self.process.stderr.read()
        except Exception:
            return ""
        if not data:
            return ""
        return data.decode("utf-8", errors="ignore").strip()

    def open(self) -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found on PATH")
        source_width, source_height = self._probe_source_size()
        target_width = max(64, int(self.max_width))
        output_width = int(target_width)
        output_height = max(1, int(round((float(source_height) * float(output_width)) / float(source_width))))
        self.frame_width = int(output_width)
        self.frame_height = int(output_height)
        self.frame_size_bytes = int(self.frame_width * self.frame_height * 3)
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_delay_max",
            "5",
            "-i",
            self.source_url,
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"scale={self.frame_width}:{self.frame_height}",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self.process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=max(10**6, self.frame_size_bytes * 2),
        )
        time.sleep(0.15)
        if self.process.poll() is not None:
            error_text = self._read_process_error() or "ffmpeg exited before streaming frames"
            self.close()
            raise RuntimeError(error_text)

    def _read_exact(self, size: int) -> Optional[bytes]:
        if self.process is None or self.process.stdout is None:
            return None
        data = bytearray()
        while len(data) < size:
            chunk = self.process.stdout.read(size - len(data))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) != size:
            return None
        return bytes(data)

    def read_frame(self) -> Optional[object]:
        if self.frame_size_bytes <= 0 or self.frame_width <= 0 or self.frame_height <= 0:
            return None
        raw_bytes = self._read_exact(self.frame_size_bytes)
        if raw_bytes is None:
            return None
        frame = np.frombuffer(raw_bytes, dtype=np.uint8)
        expected = self.frame_width * self.frame_height * 3
        if frame.size != expected:
            return None
        return frame.reshape((self.frame_height, self.frame_width, 3))

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdout is not None:
                self.process.stdout.close()
        except Exception:
            pass
        try:
            if self.process.stderr is not None:
                self.process.stderr.close()
        except Exception:
            pass
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
                except Exception:
                    pass
        self.process = None


@dataclass
class RoundSpec:
    round_id: str
    stream_url: str
    camera_id: Optional[str]
    source_type: str
    duration_sec: int
    line: Dict[str, float]
    classes: List[str]
    class_ids: List[int]
    tracker: str


@dataclass
class RoundRuntime:
    spec: RoundSpec
    started_at: float
    ends_at: float
    status: str = "running"
    current_count: int = 0
    stop_reason: Optional[str] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    counted_track_ids: Set[int] = field(default_factory=set)
    last_side_by_track: Dict[int, int] = field(default_factory=dict)
    source_opened: bool = False
    source_url: Optional[str] = None
    last_frame_at: Optional[float] = None
    detections_last_frame: int = 0
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    counting_line_x: Optional[int] = None
    counting_line_y: Optional[int] = None
    counting_direction: Optional[str] = None
    counting_zone_half_width: Optional[int] = None
    counting_roi_x1: Optional[int] = None
    counting_roi_y1: Optional[int] = None
    counting_roi_x2: Optional[int] = None
    counting_roi_y2: Optional[int] = None
    last_counted_track_id: Optional[int] = None
    last_crossing_direction: Optional[str] = None
    track_point_history: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    track_last_seen_frame_by_id: Dict[int, int] = field(default_factory=dict)
    track_in_line_band_by_id: Dict[int, bool] = field(default_factory=dict)
    track_samples_by_id: Dict[int, int] = field(default_factory=dict)
    simple_counted_key_last_frame: Dict[str, int] = field(default_factory=dict)
    last_track_samples: Optional[int] = None
    last_reject_reason: Optional[str] = None
    last_debug_frame_jpeg: Optional[bytes] = None
    last_debug_frame_log_at: Optional[float] = None
    latest_inference_frame: Optional[object] = None
    latest_inference_frame_idx: int = 0
    latest_async_debug_detections: List[Dict[str, object]] = field(default_factory=list)
    previous_gray_frame: Optional[object] = None
    motion_counted_buckets_last_frame: Dict[str, int] = field(default_factory=dict)
    motion_last_count_frame: Optional[int] = None
    latest_motion_debug_detections: List[Dict[str, object]] = field(default_factory=list)
    inference_thread: Optional[threading.Thread] = None
    inference_stop_event: threading.Event = field(default_factory=threading.Event)
    inference_lock: threading.Lock = field(default_factory=threading.Lock)
    inference_busy: bool = False
    last_inference_at: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return {
                "roundId": self.spec.round_id,
                "status": self.status,
                "currentCount": int(self.current_count),
                "startedAt": int(self.started_at),
                "endsAt": int(self.ends_at),
                "sourceOpened": bool(self.source_opened),
                "lastFrameAt": int(self.last_frame_at) if self.last_frame_at else None,
                "detectionsLastFrame": int(self.detections_last_frame),
                "frameWidth": int(self.frame_width) if self.frame_width is not None else None,
                "frameHeight": int(self.frame_height) if self.frame_height is not None else None,
                "countingLineX": int(self.counting_line_x) if self.counting_line_x is not None else None,
                "countingLineY": int(self.counting_line_y) if self.counting_line_y is not None else None,
                "countingDirection": self.counting_direction,
                "countingZoneHalfWidth": int(self.counting_zone_half_width)
                if self.counting_zone_half_width is not None
                else None,
                "countingRoi": (
                    {
                        "x1": int(self.counting_roi_x1),
                        "y1": int(self.counting_roi_y1),
                        "x2": int(self.counting_roi_x2),
                        "y2": int(self.counting_roi_y2),
                    }
                    if self.counting_roi_x1 is not None
                    and self.counting_roi_y1 is not None
                    and self.counting_roi_x2 is not None
                    and self.counting_roi_y2 is not None
                    else None
                ),
                "lastCountedTrackId": int(self.last_counted_track_id)
                if self.last_counted_track_id is not None
                else None,
                "lastCrossingDirection": self.last_crossing_direction,
                "lastTrackSamples": int(self.last_track_samples)
                if self.last_track_samples is not None
                else None,
                "lastRejectReason": self.last_reject_reason,
            }


class TrafficRoundManager:
    def __init__(self) -> None:
        self._rounds: Dict[str, RoundRuntime] = {}
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._model: Optional[YOLO] = None
        print(
            "[traffic-vision-worker] inference mode",
            {"useTracking": SETTINGS.use_tracking, "modelName": SETTINGS.model_name},
        )

    def _get_model(self) -> Optional[YOLO]:
        with self._model_lock:
            if SETTINGS.disable_inference or SETTINGS.motion_line_count:
                return None
            if self._model is None:
                self._model = YOLO(SETTINGS.model_name)
            return self._model

    def start_round(self, spec: RoundSpec) -> Dict[str, object]:
        now = time.time()
        runtime = RoundRuntime(
            spec=spec,
            started_at=now,
            ends_at=now + max(1, int(spec.duration_sec)),
        )
        threads_to_join: List[threading.Thread] = []

        with self._lock:
            previous = self._rounds.get(spec.round_id)
            if previous and previous.status == "running":
                return previous.snapshot()

            if not SETTINGS.allow_concurrent_rounds:
                for previous_round_id, running_runtime in self._rounds.items():
                    with running_runtime.lock:
                        if running_runtime.status != "running":
                            continue
                        running_runtime.status = "stopped"
                        if running_runtime.stop_reason is None:
                            running_runtime.stop_reason = "replaced_by_new_round"
                    running_runtime.stop_event.set()
                    print(
                        "[traffic-vision-worker] stopping previous running round before new start",
                        {"previousRoundId": previous_round_id, "newRoundId": spec.round_id},
                    )
                    if running_runtime.thread and running_runtime.thread.is_alive():
                        threads_to_join.append(running_runtime.thread)
            elif previous and previous.thread and previous.thread.is_alive():
                previous.stop_event.set()

            thread = threading.Thread(
                target=self._run_round,
                args=(runtime,),
                daemon=True,
                name=f"traffic-round-{spec.round_id}",
            )
            runtime.thread = thread
            self._rounds[spec.round_id] = runtime
            thread.start()

        for previous_thread in threads_to_join:
            previous_thread.join(timeout=0.5)

        print(
            "[traffic-vision-worker] worker round started",
            {
                "roundId": spec.round_id,
                "streamUrl": spec.stream_url,
                "cameraId": spec.camera_id,
                "sourceType": spec.source_type,
                "durationSec": spec.duration_sec,
                "classes": spec.classes,
                "tracker": spec.tracker,
            },
        )
        print(
            "[Traffic] starting round",
            {
                "roundId": spec.round_id,
                "sourceType": spec.source_type,
                "url": spec.stream_url,
            },
        )
        return runtime.snapshot()

    def get_status(self, round_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            runtime = self._rounds.get(round_id)
        if not runtime:
            return None
        return runtime.snapshot()

    def get_debug_frame_jpeg(self, round_id: str) -> Optional[bytes]:
        lookup_key = str(round_id).strip()
        with self._lock:
            runtime = self._rounds.get(lookup_key)
        if not runtime:
            print(
                "[traffic-vision-worker] debug frame lookup miss",
                {"roundId": lookup_key, "foundRound": False, "hasBytes": False},
            )
            return None
        with runtime.lock:
            frame_bytes = runtime.last_debug_frame_jpeg
            has_bytes = frame_bytes is not None
            size = len(frame_bytes) if frame_bytes is not None else 0
            print(
                "[traffic-vision-worker] debug frame lookup",
                {"roundId": lookup_key, "foundRound": True, "hasBytes": has_bytes, "bytes": size},
            )
            if frame_bytes is None:
                return None
            return bytes(frame_bytes)

    def stop_round(self, round_id: str, reason: str = "manual_stop") -> Optional[Dict[str, object]]:
        with self._lock:
            runtime = self._rounds.get(round_id)
        if not runtime:
            return None

        print(
            "[Traffic][STOP_ROUND]",
            {
                "roundId": round_id,
                "reason": reason,
            },
        )

        with runtime.lock:
            if runtime.status != "running":
                return {"roundId": round_id, "finalCount": int(runtime.current_count)}
            runtime.status = "stopped"
            if runtime.stop_reason is None:
                runtime.stop_reason = reason

        runtime.stop_event.set()
        runtime.inference_stop_event.set()
        if runtime.inference_thread and runtime.inference_thread.is_alive():
            runtime.inference_thread.join(timeout=0.5)
        runtime.inference_thread = None
        if runtime.thread and runtime.thread.is_alive():
            runtime.thread.join(timeout=1.0)

        with runtime.lock:
            final_count = int(runtime.current_count)

        return {"roundId": round_id, "finalCount": final_count}

    def _get_track_point(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x_min, y_min, x_max, y_max = bbox
        return (float(x_min) + float(x_max)) / 2.0, (float(y_min) + float(y_max)) / 2.0

    def _synthetic_track_id(self, class_id: int, bbox: Tuple[float, float, float, float]) -> int:
        x_min, y_min, x_max, y_max = bbox
        center_x = (float(x_min) + float(x_max)) / 2.0
        center_y = (float(y_min) + float(y_max)) / 2.0
        area = max(1.0, (float(x_max) - float(x_min)) * (float(y_max) - float(y_min)))
        bucket_x = int(center_x // 64.0)
        bucket_y = int(center_y // 64.0)
        bucket_s = int((area ** 0.5) // 24.0)
        return (abs(hash((int(class_id), bucket_x, bucket_y, bucket_s))) % 2_000_000_000) + 1

    def _is_track_in_roi(
        self,
        point_x: float,
        point_y: float,
        roi_x1: Optional[int],
        roi_y1: Optional[int],
        roi_x2: Optional[int],
        roi_y2: Optional[int],
    ) -> bool:
        if roi_x1 is None or roi_y1 is None or roi_x2 is None or roi_y2 is None:
            return True
        return (
            float(roi_x1) <= point_x <= float(roi_x2)
            and float(roi_y1) <= point_y <= float(roi_y2)
        )

    def _get_track_motion(self, history: List[Tuple[float, float]]) -> float:
        if len(history) < 2:
            return 0.0
        x0, y0 = history[0]
        x1, y1 = history[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    def _update_track_history(
        self,
        runtime: RoundRuntime,
        track_id: int,
        point_x: float,
        point_y: float,
        frame_idx: int,
    ) -> Tuple[int, float]:
        history = runtime.track_point_history.get(track_id)
        if history is None:
            history = []
            runtime.track_point_history[track_id] = history
        history.append((float(point_x), float(point_y)))
        if len(history) > REMOTE_STREAM_HISTORY_SIZE:
            history.pop(0)
        runtime.track_last_seen_frame_by_id[track_id] = int(frame_idx)
        samples = len(history)
        motion = self._get_track_motion(history)
        runtime.track_samples_by_id[track_id] = samples
        runtime.last_track_samples = samples
        return samples, motion

    def _get_effective_side(
        self,
        point_x: float,
        point_y: float,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        line_margin_px: float,
    ) -> int:
        distance = _line_signed_distance(point_x, point_y, line_x1, line_y1, line_x2, line_y2)
        if abs(distance) <= max(0.0, float(line_margin_px)):
            return 0
        return 1 if distance > 0.0 else -1

    def _is_line_touch(
        self,
        bbox: Tuple[float, float, float, float],
        point_x: float,
        point_y: float,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        line_margin_px: float,
    ) -> bool:
        x_min, y_min, x_max, y_max = bbox
        margin = max(0.0, float(line_margin_px))
        if abs(line_y1 - line_y2) <= 1.0 and abs(line_x1 - line_x2) > 1.0:
            line_y = (float(line_y1) + float(line_y2)) / 2.0
            return (float(y_min) <= line_y <= float(y_max)) or (abs(float(point_y) - line_y) <= margin)
        if abs(line_x1 - line_x2) <= 1.0 and abs(line_y1 - line_y2) > 1.0:
            line_x = (float(line_x1) + float(line_x2)) / 2.0
            return (float(x_min) <= line_x <= float(x_max)) or (abs(float(point_x) - line_x) <= margin)
        distance = _line_signed_distance(float(point_x), float(point_y), line_x1, line_y1, line_x2, line_y2)
        return abs(distance) <= margin

    def _maybe_count_simple_line_touch(
        self,
        runtime: RoundRuntime,
        track_id: int,
        class_id: int,
        bbox: Tuple[float, float, float, float],
        point_x: float,
        point_y: float,
        in_roi: bool,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        line_margin_px: float,
        frame_idx: int,
        cooldown_frames: int,
    ) -> Tuple[int, bool]:
        side = self._get_effective_side(
            point_x,
            point_y,
            line_x1,
            line_y1,
            line_x2,
            line_y2,
            line_margin_px,
        )
        if not in_roi:
            return side, False

        line_y = (float(line_y1) + float(line_y2)) / 2.0
        distance = abs(float(point_y) - float(line_y))
        if distance > float(line_margin_px):
            return side, False
        print(
            "[Traffic][LINE_TOUCH]",
            {
                "roundId": runtime.spec.round_id,
                "frame": int(frame_idx),
                "centerX": int(point_x),
                "centerY": int(point_y),
                "lineY": int(round(line_y)),
                "distance": float(distance),
            },
        )

        simple_key = f"{int(class_id)}|{int(float(point_x) // 80.0)}"
        current_count: Optional[int] = None
        with runtime.lock:
            if runtime.status != "running" or runtime.stop_event.is_set():
                return side, True
            expired_keys = [
                key
                for key, last_seen_frame in runtime.simple_counted_key_last_frame.items()
                if (int(frame_idx) - int(last_seen_frame)) > int(cooldown_frames)
            ]
            for key in expired_keys:
                runtime.simple_counted_key_last_frame.pop(key, None)
            if simple_key in runtime.simple_counted_key_last_frame:
                return side, True
            runtime.simple_counted_key_last_frame[simple_key] = int(frame_idx)
            runtime.current_count += 1
            runtime.last_counted_track_id = int(track_id)
            runtime.last_crossing_direction = f"simple_line_touch_side_{int(side)}"
            runtime.last_reject_reason = None
            current_count = int(runtime.current_count)

        print("[Traffic][COUNT_PLUS_ONE]", {"roundId": runtime.spec.round_id, "currentCount": current_count})
        return side, True

    def _count_yolo_line_touch(
        self,
        runtime: RoundRuntime,
        frame_idx: int,
        track_id: int,
        cls_id: int,
        bbox: Tuple[float, float, float, float],
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
    ) -> bool:
        x_min, y_min, x_max, y_max = bbox
        is_vertical_line = abs(float(line_x1) - float(line_x2)) <= 1.0 and abs(float(line_y1) - float(line_y2)) > 1.0
        if is_vertical_line:
            line_x = (float(line_x1) + float(line_x2)) / 2.0
            touches = float(x_min) <= line_x <= float(x_max)
            line_axis_label = "lineX"
            line_axis_value = int(round(line_x))
        else:
            line_y = (float(line_y1) + float(line_y2)) / 2.0
            touches = float(y_min) <= line_y <= float(y_max)
            line_axis_label = "lineY"
            line_axis_value = int(round(line_y))

        if not touches:
            return False

        current_count: Optional[int] = None
        with runtime.lock:
            if runtime.status != "running" or runtime.stop_event.is_set():
                return False
            if int(track_id) in runtime.counted_track_ids:
                return False
            runtime.counted_track_ids.add(int(track_id))
            runtime.current_count += 1
            runtime.last_counted_track_id = int(track_id)
            runtime.last_crossing_direction = "yolo_line_touch"
            runtime.last_reject_reason = None
            current_count = int(runtime.current_count)

        print(
            "[Traffic][YOLO_LINE_TOUCH]",
            {
                "roundId": runtime.spec.round_id,
                "frame": int(frame_idx),
                "trackId": int(track_id),
                "classId": int(cls_id),
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                line_axis_label: line_axis_value,
            },
        )
        print(
            "[Traffic][COUNT_PLUS_ONE]",
            {
                "roundId": runtime.spec.round_id,
                "frame": int(frame_idx),
                "trackId": int(track_id),
                "currentCount": current_count,
            },
        )
        return True

    def _commit_count(
        self,
        runtime: RoundRuntime,
        track_id: int,
        side: int,
    ) -> int:
        runtime.counted_track_ids.add(track_id)
        runtime.current_count += 1
        runtime.last_counted_track_id = track_id
        runtime.last_crossing_direction = f"touch_line_side_{side}"
        runtime.last_reject_reason = None
        runtime.last_side_by_track[track_id] = side
        runtime.track_in_line_band_by_id[track_id] = True
        return int(runtime.current_count)

    def _cleanup_stale_tracks(
        self,
        runtime: RoundRuntime,
        frame_idx: int,
        max_idle_frames: int = 90,
    ) -> None:
        with runtime.lock:
            stale_ids = [
                track_id
                for track_id, last_seen in runtime.track_last_seen_frame_by_id.items()
                if (frame_idx - int(last_seen)) > max_idle_frames
            ]
            for track_id in stale_ids:
                runtime.track_last_seen_frame_by_id.pop(track_id, None)
                runtime.track_point_history.pop(track_id, None)
                runtime.last_side_by_track.pop(track_id, None)
                runtime.track_in_line_band_by_id.pop(track_id, None)
                runtime.track_samples_by_id.pop(track_id, None)

    def _maybe_count_track(
        self,
        runtime: RoundRuntime,
        track_id: int,
        class_id: int,
        bbox: Tuple[float, float, float, float],
        bbox_area: float,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        roi_x1: Optional[int],
        roi_y1: Optional[int],
        roi_x2: Optional[int],
        roi_y2: Optional[int],
        min_samples: int,
        min_motion_px: float,
        min_bbox_area_px: float,
        line_margin_px: float,
        frame_idx: int,
    ) -> int:
        point_x, point_y = self._get_track_point(bbox)
        side = self._get_effective_side(
            point_x,
            point_y,
            line_x1,
            line_y1,
            line_x2,
            line_y2,
            line_margin_px,
        )
        in_roi = self._is_track_in_roi(point_x, point_y, roi_x1, roi_y1, roi_x2, roi_y2)
        should_log_reject = False
        reject_reason = ""
        current_count: Optional[int] = None
        prev_side_for_log: Optional[int] = None
        samples_for_log: Optional[int] = None
        movement_for_log: Optional[float] = None
        distance_for_log: Optional[float] = None

        with runtime.lock:
            if runtime.status != "running" or runtime.stop_event.is_set():
                return side

            if class_id not in runtime.spec.class_ids:
                runtime.last_reject_reason = f"T{track_id} reject=class class={class_id}"
                return side

            if track_id in runtime.counted_track_ids:
                return side

            samples, movement = self._update_track_history(runtime, track_id, point_x, point_y, frame_idx)
            samples_for_log = samples
            movement_for_log = movement
            prev_side = runtime.last_side_by_track.get(track_id)
            prev_side_for_log = prev_side
            distance = _line_signed_distance(point_x, point_y, line_x1, line_y1, line_x2, line_y2)
            distance_for_log = distance
            in_line_band = abs(distance) <= max(0.0, float(line_margin_px))
            was_in_line_band = runtime.track_in_line_band_by_id.get(track_id, False)
            runtime.track_in_line_band_by_id[track_id] = in_line_band

            def set_reject(reason: str) -> None:
                nonlocal should_log_reject, reject_reason
                reject_reason = reason
                runtime.last_reject_reason = reason
                should_log_reject = True

            if bbox_area < min_bbox_area_px:
                set_reject(
                    f"T{track_id} reject=min_area area={bbox_area:.0f} min={min_bbox_area_px:.0f}"
                )
            elif not in_roi:
                set_reject(f"T{track_id} reject=roi")
            elif samples < min_samples:
                set_reject(f"T{track_id} reject=min_samples s={samples} min={min_samples}")
            elif movement < min_motion_px:
                set_reject(f"T{track_id} reject=min_motion m={movement:.1f} min={min_motion_px:.1f}")
            elif not in_line_band:
                runtime.last_side_by_track[track_id] = side
                set_reject(
                    f"T{track_id} reject=off_line dist={distance:.1f} band={line_margin_px:.1f}"
                )
            elif was_in_line_band:
                runtime.last_side_by_track[track_id] = side
                set_reject(f"T{track_id} reject=already_in_band")
            else:
                current_count = self._commit_count(runtime, track_id, int(side))

        if current_count is not None:
            if DEBUG_CROSSING:
                print(
                    f"[CROSSING_DEBUG] COUNT T{track_id} class={COCO_CLASS_ID_TO_NAME.get(class_id, str(class_id))} "
                    f"samples={samples_for_log} movement={movement_for_log:.1f} "
                    f"dist={distance_for_log:.1f} band={line_margin_px:.1f} "
                    f"prev_side={prev_side_for_log} curr_side={side} count={current_count}"
                )
            print(
                "[traffic-vision-worker] object counted with trackId",
                {
                    "roundId": runtime.spec.round_id,
                    "trackId": track_id,
                    "class": COCO_CLASS_ID_TO_NAME.get(class_id, str(class_id)),
                    "currentCount": current_count,
                },
            )
            return side

        if DEBUG_CROSSING and should_log_reject:
            print(
                f"[CROSSING_DEBUG] {reject_reason} "
                f"samples={samples_for_log} movement={movement_for_log:.1f} "
                f"dist={distance_for_log:.1f} band={line_margin_px:.1f} "
                f"prev_side={prev_side_for_log} curr_side={side}"
            )
        return side

    def _update_debug_frame(
        self,
        runtime: RoundRuntime,
        frame,
        detections: List[Dict[str, object]],
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
    ) -> None:
        if frame is None:
            return

        annotated = frame.copy()
        x1 = int(float(line_x1))
        y1 = int(float(line_y1))
        x2 = int(float(line_x2))
        y2 = int(float(line_y2))

        # Keep the counting line always visible with a strong green stroke.
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

        with runtime.lock:
            current_count = int(runtime.current_count)

        count_text = f"COUNT {current_count}"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(count_text, font_face, font_scale, font_thickness)
        x_text = 12
        y_text = 16 + text_h
        cv2.rectangle(
            annotated,
            (x_text - 8, y_text - text_h - 8),
            (x_text + text_w + 8, y_text + baseline + 8),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            count_text,
            (x_text, y_text),
            font_face,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return

        with runtime.lock:
            runtime.last_debug_frame_jpeg = encoded.tobytes()
            now = time.time()
            should_log = False
            if runtime.last_debug_frame_log_at is None or (now - runtime.last_debug_frame_log_at) >= 5.0:
                should_log = True
                runtime.last_debug_frame_log_at = now
            if should_log:
                print(
                    "[traffic-vision-worker] debug frame stored",
                    {
                        "roundId": runtime.spec.round_id,
                        "bytes": len(runtime.last_debug_frame_jpeg),
                    },
                )

    def _get_async_line_geometry(self, runtime: RoundRuntime, frame_width: int, frame_height: int) -> Tuple[float, float, float, float]:
        with runtime.lock:
            line_x = runtime.counting_line_x
            line_y = runtime.counting_line_y
        if line_x is not None and line_y is None:
            clamped_x = _clamp_int(int(line_x), 0, max(0, frame_width - 1))
            return float(clamped_x), 0.0, float(clamped_x), float(max(0, frame_height - 1))
        if line_y is not None and line_x is None:
            clamped_y = _clamp_int(int(line_y), 0, max(0, frame_height - 1))
            return 0.0, float(clamped_y), float(max(0, frame_width - 1)), float(clamped_y)
        fallback_x = float(max(0, frame_width - 1) // 2)
        return fallback_x, 0.0, fallback_x, float(max(0, frame_height - 1))

    def _run_async_inference(
        self,
        runtime: RoundRuntime,
        model: YOLO,
        line_margin_px: float,
        cooldown_frames: int,
    ) -> None:
        while not runtime.stop_event.is_set() and not runtime.inference_stop_event.is_set():
            frame_for_inference = None
            frame_idx = 0
            with runtime.inference_lock:
                if (not runtime.inference_busy) and runtime.latest_inference_frame is not None:
                    frame_for_inference = runtime.latest_inference_frame
                    frame_idx = int(runtime.latest_inference_frame_idx)
                    runtime.latest_inference_frame = None
                    runtime.inference_busy = True
            if frame_for_inference is None:
                time.sleep(0.01)
                continue

            print(
                "[traffic-vision-worker] async inference start",
                {"roundId": runtime.spec.round_id, "frame": int(frame_idx)},
            )
            started_at = time.time()
            try:
                results = model.predict(
                    frame_for_inference,
                    classes=runtime.spec.class_ids,
                    conf=SETTINGS.conf_threshold,
                    iou=SETTINGS.iou_threshold,
                    verbose=False,
                )
            except Exception as inference_error:
                print(
                    "[traffic-vision-worker] async inference error",
                    {
                        "roundId": runtime.spec.round_id,
                        "frame": int(frame_idx),
                        "error": str(inference_error),
                    },
                )
                with runtime.inference_lock:
                    runtime.inference_busy = False
                time.sleep(0.01)
                continue

            line_x1, line_y1, line_x2, line_y2 = self._get_async_line_geometry(
                runtime,
                int(frame_for_inference.shape[1]),
                int(frame_for_inference.shape[0]),
            )
            with runtime.lock:
                roi_x1 = runtime.counting_roi_x1
                roi_y1 = runtime.counting_roi_y1
                roi_x2 = runtime.counting_roi_x2
                roi_y2 = runtime.counting_roi_y2

            debug_detections: List[Dict[str, object]] = []
            detections_len = 0
            touched_any = False
            if results:
                first = results[0]
                boxes = getattr(first, "boxes", None)
                if boxes is not None and boxes.cls is not None and boxes.xyxy is not None:
                    class_ids = boxes.cls.int().cpu().tolist()
                    bboxes = boxes.xyxy.cpu().tolist()
                    track_ids = [
                        self._synthetic_track_id(int(class_id), tuple(map(float, bbox)))
                        for class_id, bbox in zip(class_ids, bboxes)
                    ]
                    detections_len = len(track_ids)
                    for track_id_raw, class_id_raw, bbox in zip(track_ids, class_ids, bboxes):
                        track_id = int(track_id_raw)
                        class_id = int(class_id_raw)
                        if class_id not in runtime.spec.class_ids:
                            continue
                        x_min, y_min, x_max, y_max = bbox
                        bbox_tuple = (float(x_min), float(y_min), float(x_max), float(y_max))
                        center_x, center_y = self._get_track_point(bbox_tuple)
                        in_roi = self._is_track_in_roi(center_x, center_y, roi_x1, roi_y1, roi_x2, roi_y2)
                        side, touched = self._maybe_count_simple_line_touch(
                            runtime,
                            track_id,
                            class_id,
                            bbox_tuple,
                            center_x,
                            center_y,
                            in_roi,
                            line_x1,
                            line_y1,
                            line_x2,
                            line_y2,
                            line_margin_px,
                            frame_idx,
                            cooldown_frames,
                        )
                        if touched:
                            touched_any = True
                        debug_detections.append(
                            {
                                "track_id": track_id,
                                "class_id": class_id,
                                "bbox": (float(x_min), float(y_min), float(x_max), float(y_max)),
                                "point_x": int(center_x),
                                "point_y": int(center_y),
                                "side": side,
                                "in_roi": in_roi,
                            }
                        )

            with runtime.lock:
                runtime.detections_last_frame = int(detections_len)
                runtime.latest_async_debug_detections = list(debug_detections)
                current_count = int(runtime.current_count)
            print(
                "[Traffic][DETECTIONS]",
                {"roundId": runtime.spec.round_id, "frame": int(frame_idx), "detections": int(detections_len)},
            )
            if detections_len > 0 and not touched_any:
                print(
                    "[Traffic][NO_TOUCH]",
                    {"roundId": runtime.spec.round_id, "frame": int(frame_idx), "detections": int(detections_len)},
                )

            self._update_debug_frame(
                runtime,
                frame_for_inference,
                debug_detections,
                line_x1,
                line_y1,
                line_x2,
                line_y2,
            )

            elapsed_ms = int((time.time() - started_at) * 1000.0)
            print(
                "[traffic-vision-worker] async inference done",
                {
                    "roundId": runtime.spec.round_id,
                    "frame": int(frame_idx),
                    "detections": int(detections_len),
                    "currentCount": current_count,
                    "elapsedMs": elapsed_ms,
                },
            )
            with runtime.inference_lock:
                runtime.inference_busy = False

    def _run_motion_line_count(
        self,
        runtime: RoundRuntime,
        frame,
        frame_idx: int,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        roi_x1: Optional[int],
        roi_y1: Optional[int],
        roi_x2: Optional[int],
        roi_y2: Optional[int],
    ) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        frame_height = int(gray.shape[0])
        line_y = int(round((float(line_y1) + float(line_y2)) / 2.0))
        line_y = _clamp_int(line_y, 0, max(0, frame_height - 1))
        band_height = int(SETTINGS.motion_band_height_px)
        roi_top = int(roi_y1) if roi_y1 is not None else 0
        roi_bottom = int(roi_y2) if roi_y2 is not None else max(0, frame_height - 1)
        roi_top = _clamp_int(roi_top, 0, max(0, frame_height - 1))
        roi_bottom = _clamp_int(roi_bottom, 0, max(0, frame_height - 1))
        if roi_bottom < roi_top:
            roi_top, roi_bottom = roi_bottom, roi_top
        band_top = max(roi_top, line_y - band_height)
        band_bottom = min(roi_bottom, line_y + band_height)
        if band_bottom <= band_top:
            band_bottom = min(max(0, frame_height - 1), band_top + 1)

        with runtime.lock:
            previous_gray = runtime.previous_gray_frame
            runtime.previous_gray_frame = gray
        if previous_gray is None:
            with runtime.lock:
                runtime.latest_motion_debug_detections = []
                runtime.detections_last_frame = 0
            return

        if previous_gray.shape != gray.shape:
            with runtime.lock:
                runtime.latest_motion_debug_detections = []
                runtime.detections_last_frame = 0
            return

        previous_band = previous_gray[band_top : band_bottom + 1, :]
        current_band = gray[band_top : band_bottom + 1, :]
        if previous_band.size == 0 or current_band.size == 0:
            with runtime.lock:
                runtime.latest_motion_debug_detections = []
                runtime.detections_last_frame = 0
            return

        delta = cv2.absdiff(previous_band, current_band)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(SETTINGS.motion_min_area)
        max_area = float(SETTINGS.motion_max_area)
        min_width = int(SETTINGS.motion_min_width)
        min_height = int(SETTINGS.motion_min_height)
        max_height = int(SETTINGS.motion_max_height)
        min_aspect = float(SETTINGS.motion_min_aspect)
        max_aspect = float(SETTINGS.motion_max_aspect)
        cooldown = int(SETTINGS.motion_cooldown_frames)
        raw_contours = len(contours)
        accepted = 0
        rejected = 0
        reject_logs = 0
        max_reject_logs = 3
        debug_detections: List[Dict[str, object]] = []
        line_margin_px = float(SETTINGS.motion_line_margin_px)
        frame_counted = False

        resolved_camera_id = _normalize_camera_id(runtime.spec.camera_id)
        if not resolved_camera_id:
            resolved_camera_id = _camera_id_from_stream(runtime.source_url or runtime.spec.stream_url)
        is_cam2 = resolved_camera_id == "cam2"

        with runtime.lock:
            expired = [
                key
                for key, seen_frame in runtime.motion_counted_buckets_last_frame.items()
                if (int(frame_idx) - int(seen_frame)) > cooldown
            ]
            for key in expired:
                runtime.motion_counted_buckets_last_frame.pop(key, None)

        def reject_blob(
            reason: str,
            x: int,
            y: int,
            w: int,
            h: int,
            area: float,
            center_x: float,
            center_y: float,
        ) -> None:
            nonlocal rejected, reject_logs
            rejected += 1
            if reject_logs >= max_reject_logs:
                return
            reject_logs += 1
            print(
                "[Traffic][MOTION_REJECT]",
                {
                    "roundId": runtime.spec.round_id,
                    "frame": int(frame_idx),
                    "reason": reason,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": float(area),
                    "centerX": int(center_x),
                    "centerY": int(center_y),
                    "lineY": int(line_y),
                },
            )

        for contour in contours:
            area = float(cv2.contourArea(contour))
            x_band, y_band, w, h = cv2.boundingRect(contour)
            x = int(x_band)
            y = int(y_band + band_top)
            center_x = float(x) + (float(w) / 2.0)
            center_y = float(y) + (float(h) / 2.0)

            if area < min_area:
                reject_blob("area_too_small", x, y, w, h, area, center_x, center_y)
                continue
            if area > max_area:
                reject_blob("area_too_large", x, y, w, h, area, center_x, center_y)
                continue
            if w < min_width:
                reject_blob("width_too_small", x, y, w, h, area, center_x, center_y)
                continue
            if h < min_height:
                reject_blob("height_too_small", x, y, w, h, area, center_x, center_y)
                continue
            if h > max_height:
                reject_blob("height_too_large", x, y, w, h, area, center_x, center_y)
                continue
            aspect = float(w) / float(max(1, h))
            if aspect < min_aspect or aspect > max_aspect:
                reject_blob("aspect_out_of_range", x, y, w, h, area, center_x, center_y)
                continue
            if is_cam2:
                fill_ratio = float(area) / float(max(1, w * h))
                if fill_ratio < 0.18:
                    reject_blob("cam2_fill_ratio_low", x, y, w, h, area, center_x, center_y)
                    continue

            touches_line_zone = (int(y) <= int(line_y + line_margin_px)) and (
                int(y + h) >= int(line_y - line_margin_px)
            )
            if not touches_line_zone:
                reject_blob("line_band_no_overlap", x, y, w, h, area, center_x, center_y)
                continue

            center_line_distance = abs(float(center_y) - float(line_y))
            if center_line_distance > float(line_margin_px):
                reject_blob("center_too_far_from_line", x, y, w, h, area, center_x, center_y)
                continue

            in_roi = True
            if roi_x1 is not None and center_x < float(roi_x1):
                reject_blob("center_x_left_of_roi", x, y, w, h, area, center_x, center_y)
                continue
            if roi_x2 is not None and center_x > float(roi_x2):
                reject_blob("center_x_right_of_roi", x, y, w, h, area, center_x, center_y)
                continue

            side = self._get_effective_side(
                center_x,
                center_y,
                line_x1,
                line_y1,
                line_x2,
                line_y2,
                line_margin_px,
            )
            print(
                "[Traffic][MOTION_TOUCH]",
                {
                    "roundId": runtime.spec.round_id,
                    "frame": int(frame_idx),
                    "centerX": int(center_x),
                    "centerY": int(center_y),
                    "lineY": int(line_y),
                    "area": area,
                    "bbox": [int(x), int(y), int(w), int(h)],
                },
            )

            bucket_key = str(int(center_x // 30))
            counted = False
            cooldown_rejected = False
            with runtime.lock:
                global_last = runtime.motion_last_count_frame
                global_cooldown_active = (
                    global_last is not None and (int(frame_idx) - int(global_last)) < cooldown
                )
                if frame_counted or global_cooldown_active:
                    cooldown_rejected = True
                else:
                    last_seen = runtime.motion_counted_buckets_last_frame.get(bucket_key)
                    bucket_cooldown_active = (
                        last_seen is not None and (int(frame_idx) - int(last_seen)) < cooldown
                    )
                    if bucket_cooldown_active:
                        cooldown_rejected = True
                    else:
                        runtime.motion_counted_buckets_last_frame[bucket_key] = int(frame_idx)
                        runtime.motion_last_count_frame = int(frame_idx)
                        runtime.current_count += 1
                        runtime.last_counted_track_id = int(center_x)
                        runtime.last_crossing_direction = "motion_line_touch"
                        runtime.last_reject_reason = None
                        current_count = int(runtime.current_count)
                        counted = True
                        print(
                            "[Traffic][COUNT_PLUS_ONE]",
                            {
                                "roundId": runtime.spec.round_id,
                                "frame": int(frame_idx),
                                "bbox": [int(x), int(y), int(w), int(h)],
                                "center": [int(center_x), int(center_y)],
                                "cooldownFrames": int(cooldown),
                                "currentCount": current_count,
                            },
                        )

            if counted:
                frame_counted = True
                accepted += 1
                debug_detections.append(
                    {
                        "track_id": int(center_x),
                        "class_id": 2,
                        "bbox": (float(x), float(y), float(x + w), float(y + h)),
                        "point_x": int(center_x),
                        "point_y": int(center_y),
                        "side": int(side),
                        "in_roi": bool(in_roi),
                        "counted": True,
                        "cooldown": False,
                    }
                )
                continue

            if cooldown_rejected:
                reject_blob("cooldown_active", x, y, w, h, area, center_x, center_y)
            debug_detections.append(
                {
                    "track_id": int(center_x),
                    "class_id": 2,
                    "bbox": (float(x), float(y), float(x + w), float(y + h)),
                    "point_x": int(center_x),
                    "point_y": int(center_y),
                    "side": int(side),
                    "in_roi": bool(in_roi),
                    "counted": False,
                    "cooldown": bool(cooldown_rejected),
                }
            )

        print(
            "[Traffic][MOTION_BLOBS]",
            {
                "roundId": runtime.spec.round_id,
                "frame": int(frame_idx),
                "rawContours": int(raw_contours),
                "accepted": int(accepted),
                "rejected": int(rejected),
            },
        )
        with runtime.lock:
            runtime.latest_motion_debug_detections = debug_detections
            runtime.detections_last_frame = int(accepted)

    def _run_yolo_simple_count(
        self,
        runtime: RoundRuntime,
        model: YOLO,
        frame,
        frame_idx: int,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
    ) -> None:
        try:
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=runtime.spec.class_ids,
                imgsz=int(SETTINGS.max_process_width),
                conf=float(SETTINGS.conf_threshold),
                iou=float(SETTINGS.iou_threshold),
                verbose=False,
            )
        except Exception as inference_error:
            print(
                "[traffic-vision-worker] yolo simple track error",
                {
                    "roundId": runtime.spec.round_id,
                    "frame": int(frame_idx),
                    "error": str(inference_error),
                },
            )
            return

        debug_detections: List[Dict[str, object]] = []
        detections_len = 0
        if results:
            first = results[0]
            boxes = getattr(first, "boxes", None)
            if boxes is not None and boxes.xyxy is not None:
                xyxy_list = boxes.xyxy.cpu().tolist()
                cls_list = (
                    boxes.cls.int().cpu().tolist()
                    if boxes.cls is not None
                    else [0] * len(xyxy_list)
                )
                if boxes.id is not None:
                    track_id_list = boxes.id.int().cpu().tolist()
                else:
                    track_id_list = [
                        self._synthetic_track_id(int(c), tuple(map(float, b)))
                        for c, b in zip(cls_list, xyxy_list)
                    ]
                detections_len = len(xyxy_list)
                for tid_raw, cid_raw, bbox_raw in zip(track_id_list, cls_list, xyxy_list):
                    track_id = int(tid_raw)
                    class_id = int(cid_raw)
                    if class_id not in runtime.spec.class_ids:
                        continue
                    x_min = float(bbox_raw[0])
                    y_min = float(bbox_raw[1])
                    x_max = float(bbox_raw[2])
                    y_max = float(bbox_raw[3])
                    bbox_tuple = (x_min, y_min, x_max, y_max)
                    counted = self._count_yolo_line_touch(
                        runtime,
                        int(frame_idx),
                        track_id,
                        class_id,
                        bbox_tuple,
                        line_x1,
                        line_y1,
                        line_x2,
                        line_y2,
                    )
                    debug_detections.append(
                        {
                            "track_id": track_id,
                            "class_id": class_id,
                            "bbox": bbox_tuple,
                            "point_x": int((x_min + x_max) / 2.0),
                            "point_y": int((y_min + y_max) / 2.0),
                            "side": 0,
                            "in_roi": True,
                            "counted": bool(counted),
                            "cooldown": False,
                        }
                    )

        with runtime.lock:
            runtime.detections_last_frame = int(detections_len)
            runtime.latest_async_debug_detections = list(debug_detections)
        print(
            "[Traffic][DETECTIONS]",
            {
                "roundId": runtime.spec.round_id,
                "frame": int(frame_idx),
                "detections": int(detections_len),
            },
        )

    def _run_round(self, runtime: RoundRuntime) -> None:
        model: Optional[YOLO] = None
        source_candidates: List[str] = []
        default_source = str(runtime.spec.stream_url or "").strip()
        debug_source = str(SETTINGS.debug_video_file or "").strip()
        source_type = str(runtime.spec.source_type or "").strip().lower() or "local_video"
        remote_line_x1_ratio = 0.50
        remote_line_y1_ratio = 0.0
        remote_line_x2_ratio = 0.50
        remote_line_y2_ratio = 1.0
        remote_roi_x_min_ratio = REMOTE_STREAM_ROI_X_MIN_RATIO
        remote_roi_x_max_ratio = REMOTE_STREAM_ROI_X_MAX_RATIO
        remote_roi_y_min_ratio = REMOTE_STREAM_ROI_Y_MIN_RATIO
        remote_roi_y_max_ratio = REMOTE_STREAM_ROI_Y_MAX_RATIO
        remote_min_samples = REMOTE_STREAM_MIN_SAMPLES
        remote_min_motion_px = REMOTE_STREAM_MIN_MOTION_PX
        remote_min_bbox_area_px = REMOTE_STREAM_MIN_BBOX_AREA_PX
        remote_line_margin_px = REMOTE_STREAM_LINE_MARGIN_PX
        remote_profile_id = "default"
        remote_profile_source = "default"
        resolved_camera_id = ""

        if source_type == "remote_stream":
            profile = _resolve_remote_profile(runtime.spec.camera_id, default_source, runtime.spec.round_id)
            remote_profile_id = str(profile.get("profile_id", "default_remote"))
            remote_profile_source = str(profile.get("profile_source", "default"))
            remote_line_x1_ratio = float(profile.get("line_x1_ratio", remote_line_x1_ratio))
            remote_line_y1_ratio = float(profile.get("line_y1_ratio", remote_line_y1_ratio))
            remote_line_x2_ratio = float(profile.get("line_x2_ratio", remote_line_x2_ratio))
            remote_line_y2_ratio = float(profile.get("line_y2_ratio", remote_line_y2_ratio))
            remote_roi_x_min_ratio = float(profile.get("roi_x_min_ratio", remote_roi_x_min_ratio))
            remote_roi_x_max_ratio = float(profile.get("roi_x_max_ratio", remote_roi_x_max_ratio))
            remote_roi_y_min_ratio = float(profile.get("roi_y_min_ratio", remote_roi_y_min_ratio))
            remote_roi_y_max_ratio = float(profile.get("roi_y_max_ratio", remote_roi_y_max_ratio))
            remote_min_samples = int(profile.get("min_samples", remote_min_samples))
            remote_min_motion_px = float(profile.get("min_motion_px", remote_min_motion_px))
            remote_min_bbox_area_px = float(profile.get("min_bbox_area_px", remote_min_bbox_area_px))
            remote_line_margin_px = float(profile.get("line_margin_px", remote_line_margin_px))
            print(
                "[Traffic] remote profile enabled",
                {
                    "roundId": runtime.spec.round_id,
                    "cameraId": runtime.spec.camera_id,
                    "profileId": remote_profile_id,
                    "profileSource": remote_profile_source,
                    "line": [
                        remote_line_x1_ratio,
                        remote_line_y1_ratio,
                        remote_line_x2_ratio,
                        remote_line_y2_ratio,
                    ],
                    "roi": [
                        remote_roi_x_min_ratio,
                        remote_roi_y_min_ratio,
                        remote_roi_x_max_ratio,
                        remote_roi_y_max_ratio,
                    ],
                    "minSamples": remote_min_samples,
                    "minMotionPx": remote_min_motion_px,
                    "minBboxAreaPx": remote_min_bbox_area_px,
                    "lineMarginPx": remote_line_margin_px,
                },
            )
            resolved_camera_id = _normalize_camera_id(runtime.spec.camera_id) or _camera_id_from_stream(
                default_source
            )
            if resolved_camera_id and not _is_supported_production_camera(resolved_camera_id):
                with runtime.lock:
                    runtime.status = "stopped"
                    runtime.stop_reason = "unsupported_camera"
                    runtime.source_opened = False
                    final_count = int(runtime.current_count)
                    runtime.counted_track_ids.clear()
                    runtime.last_side_by_track.clear()
                    runtime.track_point_history.clear()
                    runtime.track_last_seen_frame_by_id.clear()
                    runtime.track_in_line_band_by_id.clear()
                    runtime.track_samples_by_id.clear()
                    runtime.simple_counted_key_last_frame.clear()
                    runtime.latest_inference_frame = None
                    runtime.latest_inference_frame_idx = 0
                    runtime.latest_async_debug_detections.clear()
                    runtime.previous_gray_frame = None
                    runtime.motion_counted_buckets_last_frame.clear()
                    runtime.latest_motion_debug_detections.clear()
                    runtime.inference_busy = False
                    runtime.last_inference_at = None
                    runtime.inference_thread = None
                print(
                    "[traffic-vision-worker] worker round stopped",
                    {
                        "roundId": runtime.spec.round_id,
                        "reason": "unsupported_camera",
                        "cameraId": resolved_camera_id,
                        "finalCount": final_count,
                    },
                )
                print(
                    "[traffic-vision-worker] round runtime memory cleaned",
                    {"roundId": runtime.spec.round_id, "finalCount": final_count},
                )
                return
        if source_type == "remote_stream":
            if resolved_camera_id == "las_vegas":
                source_candidates.append(LAS_VEGAS_STREAM_URL)
            if default_source and default_source != LAS_VEGAS_STREAM_URL:
                source_candidates.append(default_source)
        else:
            if SETTINGS.debug_use_local_file:
                if debug_source:
                    source_candidates.append(debug_source)
                if default_source and default_source != debug_source:
                    source_candidates.append(default_source)
            else:
                if default_source:
                    source_candidates.append(default_source)
                if debug_source and debug_source != default_source:
                    source_candidates.append(debug_source)

        cap: Optional[cv2.VideoCapture] = None
        ffmpeg_reader: Optional[FFmpegFrameReader] = None
        use_ffmpeg_reader = False
        ffmpeg_fallback_attempted = False
        active_source: Optional[str] = None
        for source in source_candidates:
            candidate = str(source).strip()
            if not candidate:
                continue

            maybe_file = os.path.exists(candidate)
            print(
                "[traffic-vision-worker] source opened attempt",
                {
                    "roundId": runtime.spec.round_id,
                    "source": candidate,
                    "isLocalFile": maybe_file,
                },
            )

            if source_type == "remote_stream":
                try:
                    reader = FFmpegFrameReader(candidate, int(SETTINGS.max_process_width))
                    reader.open()
                    ffmpeg_reader = reader
                    active_source = candidate
                    use_ffmpeg_reader = True
                    print(
                        "[traffic-vision-worker] ffmpeg stream opened",
                        {"roundId": runtime.spec.round_id, "source": candidate},
                    )
                    break
                except Exception as ffmpeg_error:
                    print(
                        "[traffic-vision-worker] ffmpeg stream open failed, falling back to cv2.VideoCapture",
                        {
                            "roundId": runtime.spec.round_id,
                            "source": candidate,
                            "error": str(ffmpeg_error),
                        },
                    )
                    trial = cv2.VideoCapture(candidate)
                    if trial.isOpened():
                        cap = trial
                        active_source = candidate
                        use_ffmpeg_reader = False
                        break
                    trial.release()
            else:
                trial = cv2.VideoCapture(candidate)
                if trial.isOpened():
                    cap = trial
                    active_source = candidate
                    break
                trial.release()

            print(
                "[traffic-vision-worker] source opened failed",
                {
                    "roundId": runtime.spec.round_id,
                    "source": candidate,
                },
            )

        if (ffmpeg_reader is None and cap is None) or active_source is None:
            with runtime.lock:
                runtime.status = "stopped"
                runtime.stop_reason = "stream_open_failed"
                runtime.source_opened = False
                final_count = int(runtime.current_count)
                runtime.counted_track_ids.clear()
                runtime.last_side_by_track.clear()
                runtime.track_point_history.clear()
                runtime.track_last_seen_frame_by_id.clear()
                runtime.track_in_line_band_by_id.clear()
                runtime.track_samples_by_id.clear()
                runtime.simple_counted_key_last_frame.clear()
                runtime.latest_inference_frame = None
                runtime.latest_inference_frame_idx = 0
                runtime.latest_async_debug_detections.clear()
                runtime.previous_gray_frame = None
                runtime.motion_counted_buckets_last_frame.clear()
                runtime.latest_motion_debug_detections.clear()
                runtime.inference_busy = False
                runtime.last_inference_at = None
                runtime.inference_thread = None
            print(
                "[traffic-vision-worker] worker round stopped",
                {
                    "roundId": runtime.spec.round_id,
                    "reason": "stream_open_failed",
                    "finalCount": final_count,
                },
            )
            print(
                "[traffic-vision-worker] round runtime memory cleaned",
                {"roundId": runtime.spec.round_id, "finalCount": final_count},
            )
            return

        with runtime.lock:
            runtime.source_opened = True
            runtime.source_url = active_source
        print(
            "[traffic-vision-worker] source opened ok",
            {
                "roundId": runtime.spec.round_id,
                "source": active_source,
            },
        )
        if SETTINGS.motion_line_count:
            print(
                "[traffic-vision-worker] motion line count enabled (opencv-only mode)",
                {"roundId": runtime.spec.round_id},
            )
        elif SETTINGS.disable_inference:
            print(
                "[traffic-vision-worker] inference disabled (live-only mode)",
                {"roundId": runtime.spec.round_id},
            )
        else:
            try:
                model = self._get_model()
            except Exception as model_error:
                model = None
                print(
                    "[traffic-vision-worker] model load failed, continuing camera-only",
                    {"roundId": runtime.spec.round_id, "error": str(model_error)},
                )
            if model is not None:
                print(
                    "[traffic-vision-worker] yolo simple line touch mode enabled",
                    {
                        "roundId": runtime.spec.round_id,
                        "inferenceFps": float(SETTINGS.inference_fps),
                        "imgsz": int(SETTINGS.max_process_width),
                    },
                )

        frame_idx = 0
        processed_frame_idx = 0
        frame_read_failures = 0
        last_process_at = 0.0
        last_debug_frame_store_at = 0.0
        process_interval = 1.0 / SETTINGS.process_fps
        debug_frame_interval = 1.0 / SETTINGS.debug_frame_fps
        max_process_width = int(SETTINGS.max_process_width)
        line_x1 = float(runtime.spec.line["x1"])
        line_y1 = float(runtime.spec.line["y1"])
        line_x2 = float(runtime.spec.line["x2"])
        line_y2 = float(runtime.spec.line["y2"])
        roi_x1: Optional[int] = None
        roi_y1: Optional[int] = None
        roi_x2: Optional[int] = None
        roi_y2: Optional[int] = None
        line_metrics_logged = False
        use_motion_line_count = bool(SETTINGS.motion_line_count)
        use_yolo_simple = (
            (not SETTINGS.disable_inference)
            and (not use_motion_line_count)
            and (model is not None)
        )
        inference_interval = 1.0 / max(0.1, float(SETTINGS.inference_fps))
        last_yolo_inference_at = 0.0
        if use_motion_line_count:
            print(
                "[traffic-vision-worker] motion mode settings",
                {
                    "roundId": runtime.spec.round_id,
                    "cooldownFrames": int(SETTINGS.motion_cooldown_frames),
                    "processFps": float(SETTINGS.process_fps),
                    "maxProcessWidth": int(SETTINGS.max_process_width),
                    "minArea": float(SETTINGS.motion_min_area),
                    "lineMargin": float(SETTINGS.motion_line_margin_px),
                },
            )

        try:
            while not runtime.stop_event.is_set():
                now = time.time()
                if now >= runtime.ends_at:
                    print(
                        "[traffic-vision-worker] traffic round reached end_time",
                        {"roundId": runtime.spec.round_id, "endsAt": int(runtime.ends_at)},
                    )
                    break

                if use_ffmpeg_reader and ffmpeg_reader is not None:
                    frame = ffmpeg_reader.read_frame()
                    ok = frame is not None
                    if ok and (
                        frame_idx <= 3
                        or (
                            SETTINGS.frame_log_interval > 0
                            and frame_idx % SETTINGS.frame_log_interval == 0
                        )
                    ):
                        print(
                            "[traffic-vision-worker] ffmpeg frame read success",
                            {"roundId": runtime.spec.round_id, "frame": int(frame_idx + 1)},
                        )
                else:
                    if cap is None:
                        ok = False
                        frame = None
                    else:
                        ok, frame = cap.read()
                if not ok or frame is None:
                    frame_read_failures += 1
                    if use_ffmpeg_reader:
                        print(
                            "[traffic-vision-worker] ffmpeg frame read failed",
                            {
                                "roundId": runtime.spec.round_id,
                                "failures": frame_read_failures,
                            },
                        )
                        if not ffmpeg_fallback_attempted and active_source:
                            ffmpeg_fallback_attempted = True
                            print(
                                "[traffic-vision-worker] ffmpeg pipe failed, attempting cv2.VideoCapture fallback",
                                {"roundId": runtime.spec.round_id, "source": active_source},
                            )
                            fallback = cv2.VideoCapture(active_source)
                            if fallback.isOpened():
                                cap = fallback
                                use_ffmpeg_reader = False
                                if ffmpeg_reader is not None:
                                    ffmpeg_reader.close()
                                    ffmpeg_reader = None
                                frame_read_failures = 0
                                print(
                                    "[traffic-vision-worker] ffmpeg fallback to cv2.VideoCapture enabled",
                                    {"roundId": runtime.spec.round_id, "source": active_source},
                                )
                                continue
                            fallback.release()
                            print(
                                "[traffic-vision-worker] ffmpeg fallback to cv2.VideoCapture failed",
                                {"roundId": runtime.spec.round_id, "source": active_source},
                            )
                    if frame_read_failures <= 3 or (
                        SETTINGS.frame_log_interval > 0
                        and frame_read_failures % SETTINGS.frame_log_interval == 0
                    ):
                        print(
                            "[traffic-vision-worker] frame read failed",
                            {
                                "roundId": runtime.spec.round_id,
                                "failures": frame_read_failures,
                            },
                        )
                    time.sleep(0.03)
                    continue

                frame_idx += 1
                raw_frame_bytes: Optional[bytes] = None
                ok_raw, encoded_raw = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
                )
                if ok_raw:
                    raw_frame_bytes = encoded_raw.tobytes()
                with runtime.lock:
                    runtime.last_frame_at = now
                if raw_frame_bytes is not None and frame_idx <= 3:
                    print(
                        "[traffic-vision-worker] raw live frame stored",
                        {
                            "roundId": runtime.spec.round_id,
                            "frame": int(frame_idx),
                            "bytes": len(raw_frame_bytes),
                        },
                    )

                frame_height, frame_width = frame.shape[:2]
                processed_frame = frame
                if (
                    frame_width > 0
                    and max_process_width > 0
                    and frame_width > max_process_width
                ):
                    scale = float(max_process_width) / float(frame_width)
                    resized_height = max(1, int(round(float(frame_height) * scale)))
                    processed_frame = cv2.resize(
                        frame,
                        (int(max_process_width), int(resized_height)),
                        interpolation=cv2.INTER_AREA,
                    )
                    frame_height, frame_width = processed_frame.shape[:2]

                now = time.time()
                should_process = (now - last_process_at) >= process_interval
                should_store_debug = (now - last_debug_frame_store_at) >= debug_frame_interval
                if SETTINGS.disable_inference:
                    should_process = True
                    should_store_debug = True
                if not should_process and not should_store_debug:
                    time.sleep(0.005)
                    continue

                if should_store_debug:
                    last_debug_frame_store_at = now

                if frame_width > 0 and frame_height > 0:
                    if source_type == "remote_stream":
                        line_x1 = float(
                            _clamp_int(int(round(frame_width * remote_line_x1_ratio)), 0, frame_width - 1)
                        )
                        line_y1 = float(
                            _clamp_int(int(round(frame_height * remote_line_y1_ratio)), 0, frame_height - 1)
                        )
                        line_x2 = float(
                            _clamp_int(int(round(frame_width * remote_line_x2_ratio)), 0, frame_width - 1)
                        )
                        line_y2 = float(
                            _clamp_int(int(round(frame_height * remote_line_y2_ratio)), 0, frame_height - 1)
                        )
                        if line_x1 == line_x2 and line_y1 == line_y2:
                            line_x2 = float(min(frame_width - 1, int(line_x1) + 1))

                        roi_x1 = _clamp_int(
                            int(round(frame_width * remote_roi_x_min_ratio)),
                            0,
                            frame_width - 1,
                        )
                        roi_x2 = _clamp_int(int(round(frame_width * remote_roi_x_max_ratio)), 0, frame_width - 1)
                        if roi_x2 <= roi_x1:
                            roi_x2 = min(frame_width - 1, roi_x1 + 1)
                        roi_y1 = _clamp_int(
                            int(round(frame_height * remote_roi_y_min_ratio)),
                            0,
                            frame_height - 1,
                        )
                        roi_y2 = _clamp_int(int(round(frame_height * remote_roi_y_max_ratio)), 0, frame_height - 1)
                        if roi_y2 <= roi_y1:
                            roi_y2 = min(frame_height - 1, roi_y1 + 1)
                    else:
                        line_x1 = float(_clamp_int(int(round(line_x1)), 0, frame_width - 1))
                        line_x2 = float(_clamp_int(int(round(line_x2)), 0, frame_width - 1))
                        line_y1 = float(_clamp_int(int(round(line_y1)), 0, frame_height - 1))
                        line_y2 = float(_clamp_int(int(round(line_y2)), 0, frame_height - 1))
                        if line_x1 == line_x2 and line_y1 == line_y2:
                            line_x2 = float(min(frame_width - 1, int(line_x1) + 1))
                        roi_x1 = None
                        roi_y1 = None
                        roi_x2 = None
                        roi_y2 = None

                    if abs(line_x1 - line_x2) <= 1.0 and abs(line_y1 - line_y2) > 1.0:
                        effective_line_x = int(round((line_x1 + line_x2) / 2.0))
                        effective_line_y = None
                    elif abs(line_y1 - line_y2) <= 1.0 and abs(line_x1 - line_x2) > 1.0:
                        effective_line_x = None
                        effective_line_y = int(round((line_y1 + line_y2) / 2.0))
                    else:
                        effective_line_x = None
                        effective_line_y = None

                    with runtime.lock:
                        runtime.frame_width = int(frame_width)
                        runtime.frame_height = int(frame_height)
                        runtime.counting_line_x = (
                            int(effective_line_x) if effective_line_x is not None else None
                        )
                        runtime.counting_line_y = (
                            int(effective_line_y) if effective_line_y is not None else None
                        )
                        runtime.counting_direction = "both"
                        runtime.counting_zone_half_width = int(
                            remote_line_margin_px if source_type == "remote_stream" else REMOTE_STREAM_LINE_MARGIN_PX
                        )
                        runtime.counting_roi_x1 = int(roi_x1) if roi_x1 is not None else None
                        runtime.counting_roi_y1 = int(roi_y1) if roi_y1 is not None else None
                        runtime.counting_roi_x2 = int(roi_x2) if roi_x2 is not None else None
                        runtime.counting_roi_y2 = int(roi_y2) if roi_y2 is not None else None

                    if not line_metrics_logged:
                        print(
                            "[Traffic] counting line calibrated",
                            {
                                "roundId": runtime.spec.round_id,
                                "sourceType": source_type,
                                "frameWidth": int(frame_width),
                                "frameHeight": int(frame_height),
                                "effectiveCountingLineX": int(effective_line_x)
                                if effective_line_x is not None
                                else None,
                                "effectiveCountingLineY": int(effective_line_y)
                                if effective_line_y is not None
                                else None,
                                "countingRoi": (
                                    {
                                        "x1": int(roi_x1),
                                        "y1": int(roi_y1),
                                        "x2": int(roi_x2),
                                        "y2": int(roi_y2),
                                    }
                                    if roi_x1 is not None
                                    and roi_y1 is not None
                                    and roi_x2 is not None
                                    and roi_y2 is not None
                                    else None
                                ),
                            },
                        )
                        line_metrics_logged = True

                    with runtime.lock:
                        pre_inference_debug_detections = list(runtime.latest_async_debug_detections)
                    self._update_debug_frame(
                        runtime,
                        processed_frame,
                        pre_inference_debug_detections,
                        line_x1,
                        line_y1,
                        line_x2,
                        line_y2,
                    )
                    if frame_idx <= 3:
                        with runtime.lock:
                            bytes_present = runtime.last_debug_frame_jpeg is not None
                        print(
                            "[traffic-vision-worker] raw debug frame stored before inference",
                            {
                                "roundId": runtime.spec.round_id,
                                "frame": int(frame_idx),
                                "bytesPresent": bool(bytes_present),
                            },
                        )

                if not should_process:
                    if should_store_debug:
                        with runtime.lock:
                            if use_motion_line_count:
                                debug_detections_for_frame = list(runtime.latest_motion_debug_detections)
                            else:
                                debug_detections_for_frame = list(runtime.latest_async_debug_detections)
                        self._update_debug_frame(
                            runtime,
                            processed_frame,
                            debug_detections_for_frame,
                            line_x1,
                            line_y1,
                            line_x2,
                            line_y2,
                        )
                    continue

                processed_frame_idx += 1
                last_process_at = now
                should_log_tick = processed_frame_idx <= 3 or (
                    SETTINGS.frame_log_interval > 0
                    and processed_frame_idx % SETTINGS.frame_log_interval == 0
                )
                if should_log_tick:
                    print(
                        "[traffic-vision-worker] frame loop tick",
                        {"roundId": runtime.spec.round_id, "frame": processed_frame_idx},
                    )

                if processed_frame_idx <= 3 or (
                    SETTINGS.frame_log_interval > 0
                    and processed_frame_idx % SETTINGS.frame_log_interval == 0
                ):
                    print(
                        "[traffic-vision-worker] frame read success",
                        {"roundId": runtime.spec.round_id, "frame": processed_frame_idx},
                    )

                if (
                    SETTINGS.frame_log_interval > 0
                    and processed_frame_idx % SETTINGS.frame_log_interval == 0
                ):
                    print(
                        "[traffic-vision-worker] frame processing active",
                        {"roundId": runtime.spec.round_id, "frame": processed_frame_idx},
                    )

                if use_motion_line_count:
                    self._run_motion_line_count(
                        runtime,
                        processed_frame,
                        processed_frame_idx,
                        line_x1,
                        line_y1,
                        line_x2,
                        line_y2,
                        roi_x1,
                        roi_y1,
                        roi_x2,
                        roi_y2,
                    )

                if use_yolo_simple:
                    now_yolo = time.time()
                    if (now_yolo - last_yolo_inference_at) >= inference_interval:
                        last_yolo_inference_at = now_yolo
                        if should_log_tick:
                            print(
                                "[traffic-vision-worker] yolo simple inference",
                                {
                                    "roundId": runtime.spec.round_id,
                                    "frame": processed_frame_idx,
                                },
                            )
                        self._run_yolo_simple_count(
                            runtime,
                            model,
                            processed_frame,
                            processed_frame_idx,
                            line_x1,
                            line_y1,
                            line_x2,
                            line_y2,
                        )
                        with runtime.lock:
                            post_inference_debug_detections = list(runtime.latest_async_debug_detections)
                        self._update_debug_frame(
                            runtime,
                            processed_frame,
                            post_inference_debug_detections,
                            line_x1,
                            line_y1,
                            line_x2,
                            line_y2,
                        )
                elif not use_motion_line_count:
                    with runtime.lock:
                        runtime.detections_last_frame = 0
                with runtime.lock:
                    if use_motion_line_count:
                        debug_detections_for_frame = list(runtime.latest_motion_debug_detections)
                    else:
                        debug_detections_for_frame = list(runtime.latest_async_debug_detections)

                self._update_debug_frame(
                    runtime,
                    processed_frame,
                    debug_detections_for_frame,
                    line_x1,
                    line_y1,
                    line_x2,
                    line_y2,
                )
        finally:
            runtime.inference_stop_event.set()
            if runtime.inference_thread and runtime.inference_thread.is_alive():
                runtime.inference_thread.join(timeout=0.5)
            if ffmpeg_reader is not None:
                ffmpeg_reader.close()
            if cap is not None:
                cap.release()
            with runtime.lock:
                if runtime.status == "running":
                    runtime.status = "stopped" if runtime.stop_event.is_set() else "ended"
                if runtime.stop_reason is None:
                    runtime.stop_reason = "manual_stop" if runtime.stop_event.is_set() else "end_time_reached"
                final_count = int(runtime.current_count)
                runtime.counted_track_ids.clear()
                runtime.last_side_by_track.clear()
                runtime.track_point_history.clear()
                runtime.track_last_seen_frame_by_id.clear()
                runtime.track_in_line_band_by_id.clear()
                runtime.track_samples_by_id.clear()
                runtime.simple_counted_key_last_frame.clear()
                runtime.latest_inference_frame = None
                runtime.latest_inference_frame_idx = 0
                runtime.latest_async_debug_detections.clear()
                runtime.previous_gray_frame = None
                runtime.motion_counted_buckets_last_frame.clear()
                runtime.latest_motion_debug_detections.clear()
                runtime.inference_busy = False
                runtime.last_inference_at = None
                runtime.inference_thread = None

            print(
                "[traffic-vision-worker] worker round stopped",
                {
                    "roundId": runtime.spec.round_id,
                    "reason": runtime.stop_reason,
                    "finalCount": final_count,
                },
            )
            print(
                "[traffic-vision-worker] final frozen count returned",
                {"roundId": runtime.spec.round_id, "finalCount": final_count},
            )
            print(
                "[traffic-vision-worker] round runtime memory cleaned",
                {"roundId": runtime.spec.round_id, "finalCount": final_count},
            )


ROUND_MANAGER = TrafficRoundManager()
