from dataclasses import dataclass
import os


@dataclass(frozen=True)
class WorkerSettings:
    host: str = os.getenv("TRAFFIC_VISION_HOST", "0.0.0.0")
    port: int = int(os.getenv("TRAFFIC_VISION_PORT", "8090"))
    model_name: str = os.getenv("TRAFFIC_VISION_MODEL", "yolov8n.pt")
    conf_threshold: float = float(os.getenv("TRAFFIC_VISION_CONF", "0.20"))
    iou_threshold: float = float(os.getenv("TRAFFIC_VISION_IOU", "0.45"))
    frame_log_interval: int = int(os.getenv("TRAFFIC_VISION_FRAME_LOG_INTERVAL", "120"))
    debug_video_file: str = os.getenv("TRAFFIC_VISION_DEBUG_VIDEO_FILE", "/tmp/traffic_sample.mp4")
    debug_use_local_file: bool = os.getenv("TRAFFIC_VISION_DEBUG_USE_LOCAL_FILE", "0") == "1"
    allow_concurrent_rounds: bool = os.getenv("TRAFFIC_ALLOW_CONCURRENT_ROUNDS", "0") == "1"
    process_fps: float = max(0.1, float(os.getenv("TRAFFIC_PROCESS_FPS", "2")))
    debug_frame_fps: float = max(0.1, float(os.getenv("TRAFFIC_DEBUG_FRAME_FPS", "2")))
    max_process_width: int = max(64, int(os.getenv("TRAFFIC_MAX_PROCESS_WIDTH", "960")))


SETTINGS = WorkerSettings()
