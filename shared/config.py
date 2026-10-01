"""Central runtime configuration for the CET CCET pipeline."""

NUM_STREAMS = 4
BATCH_SIZE = 8
TARGET_FPS = 100
ADAPTIVE_BATCH = True

QUEUE_MAX_SIZE = 20
QUEUE_TIMEOUT = 0.1

CUDA_DEVICE = 0
USE_TENSORRT = False
USE_FP16 = False

VIDEO_PATHS = [
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
]

TARGET_CAPTURE_FPS = 30

STREAM_WEIGHTS = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

BATCH_SIZE_MIN = 4
BATCH_SIZE_MAX = 16

QUEUE_DEPTH_HIGH = 1
QUEUE_DEPTH_LOW = 0

FAIRNESS_CV_THRESHOLD = 0.10
LATENCY_TARGET_MS = 300
DISPLAY_UPDATE_INTERVAL = 1.0

LOG_DIR = "logs/"
CHECKPOINT_DIR = "checkpoints/"
OUTPUT_DIR = "output/"

MODEL = "yolo11l"
LATENCY_TARGET = 300


def validate() -> None:
    """Fail fast on configuration combinations the pipeline cannot handle."""
    if NUM_STREAMS <= 0:
        raise ValueError("NUM_STREAMS must be greater than zero")
    if len(VIDEO_PATHS) < NUM_STREAMS:
        raise ValueError(
            f"VIDEO_PATHS must contain at least {NUM_STREAMS} sources; "
            f"found {len(VIDEO_PATHS)}"
        )
    if QUEUE_MAX_SIZE <= 0:
        raise ValueError("QUEUE_MAX_SIZE must be greater than zero")
    if TARGET_CAPTURE_FPS <= 0:
        raise ValueError("TARGET_CAPTURE_FPS must be greater than zero")
    if not BATCH_SIZE_MIN <= BATCH_SIZE <= BATCH_SIZE_MAX:
        raise ValueError(
            f"BATCH_SIZE must be between {BATCH_SIZE_MIN} and {BATCH_SIZE_MAX}"
        )
    if QUEUE_DEPTH_LOW < 0 or QUEUE_DEPTH_HIGH < QUEUE_DEPTH_LOW:
        raise ValueError("Queue depth thresholds are invalid")
    if set(STREAM_WEIGHTS) != set(range(NUM_STREAMS)):
        raise ValueError("STREAM_WEIGHTS must define exactly one weight per stream")
    if any(weight < 0 for weight in STREAM_WEIGHTS.values()):
        raise ValueError("STREAM_WEIGHTS cannot contain negative values")
