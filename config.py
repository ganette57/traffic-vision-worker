from dataclasses import dataclass
import os


@dataclass(frozen=True)
class WorkerSettings:
    host: str = os.getenv("TRAFFIC_VISION_HOST", "0.0.0.0")
    port: int = int(os.getenv("TRAFFIC_VISION_PORT", "8090"))
    model_name: str = os.getenv("TRAFFIC_VISION_MODEL", "yolov8n.pt")
    conf_threshold: float = float(
        os.getenv("TRAFFIC_CONF_THRESHOLD", os.getenv("TRAFFIC_VISION_CONF", "0.25"))
    )
    iou_threshold: float = float(os.getenv("TRAFFIC_VISION_IOU", "0.45"))
    frame_log_interval: int = int(os.getenv("TRAFFIC_VISION_FRAME_LOG_INTERVAL", "120"))
    debug_video_file: str = os.getenv("TRAFFIC_VISION_DEBUG_VIDEO_FILE", "/tmp/traffic_sample.mp4")
    debug_use_local_file: bool = os.getenv("TRAFFIC_VISION_DEBUG_USE_LOCAL_FILE", "0") == "1"
    allow_concurrent_rounds: bool = os.getenv("TRAFFIC_ALLOW_CONCURRENT_ROUNDS", "0") == "1"
    disable_inference: bool = os.getenv("TRAFFIC_DISABLE_INFERENCE", "0") == "1"
    motion_line_count: bool = os.getenv("TRAFFIC_MOTION_LINE_COUNT", "0") == "1"
    motion_min_area: float = max(1.0, float(os.getenv("TRAFFIC_MOTION_MIN_AREA", "250")))
    motion_max_area: float = max(1.0, float(os.getenv("TRAFFIC_MOTION_MAX_AREA", "80000")))
    motion_line_margin_px: float = max(1.0, float(os.getenv("TRAFFIC_MOTION_LINE_MARGIN_PX", "30")))
    motion_cooldown_frames: int = max(1, int(os.getenv("TRAFFIC_MOTION_COOLDOWN_FRAMES", "25")))
    motion_band_height_px: int = max(1, int(os.getenv("TRAFFIC_MOTION_BAND_HEIGHT_PX", "70")))
    motion_min_width: int = max(1, int(os.getenv("TRAFFIC_MOTION_MIN_WIDTH", "8")))
    motion_min_height: int = max(1, int(os.getenv("TRAFFIC_MOTION_MIN_HEIGHT", "6")))
    motion_max_height: int = max(1, int(os.getenv("TRAFFIC_MOTION_MAX_HEIGHT", "220")))
    motion_min_aspect: float = max(0.01, float(os.getenv("TRAFFIC_MOTION_MIN_ASPECT", "0.15")))
    motion_max_aspect: float = max(0.01, float(os.getenv("TRAFFIC_MOTION_MAX_ASPECT", "10.0")))
    async_inference: bool = os.getenv("TRAFFIC_ASYNC_INFERENCE", "1") == "1"
    inference_fps: float = max(0.1, float(os.getenv("TRAFFIC_INFERENCE_FPS", "2")))
    use_tracking: bool = os.getenv("TRAFFIC_USE_TRACKING", "1") == "1"
    simple_line_touch_count: bool = os.getenv("TRAFFIC_SIMPLE_LINE_TOUCH_COUNT", "1") == "1"
    simple_count_cooldown_frames: int = max(
        1, int(os.getenv("TRAFFIC_SIMPLE_COUNT_COOLDOWN_FRAMES", "90"))
    )
    line_margin_px: float = max(1.0, float(os.getenv("TRAFFIC_LINE_MARGIN_PX", "35")))
    process_fps: float = max(0.1, float(os.getenv("TRAFFIC_PROCESS_FPS", "2")))
    debug_frame_fps: float = max(0.1, float(os.getenv("TRAFFIC_DEBUG_FRAME_FPS", "4")))
    max_process_width: int = max(64, int(os.getenv("TRAFFIC_MAX_PROCESS_WIDTH", "640")))


SETTINGS = WorkerSettings()
