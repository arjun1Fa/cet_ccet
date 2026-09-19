# /shared/config.py

NUM_STREAMS = 4
BATCH_SIZE = 8
TARGET_FPS = 100
ADAPTIVE_BATCH = True

QUEUE_MAX_SIZE = 20
QUEUE_TIMEOUT = 0.1

CUDA_DEVICE = 0
USE_TENSORRT = False
USE_FP16 = False

# Google Cloud URLs — streamed directly by the A40's 10Gbps datacenter connection
VIDEO_PATHS = [
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4',
]

TARGET_CAPTURE_FPS = 30

STREAM_WEIGHTS = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
}

BATCH_SIZE_MIN = 4
BATCH_SIZE_MAX = 16

QUEUE_DEPTH_HIGH = 1
QUEUE_DEPTH_LOW = 0

FAIRNESS_CV_THRESHOLD = 0.10
LATENCY_TARGET_MS = 300
DISPLAY_UPDATE_INTERVAL = 1.0

LOG_DIR = 'logs/'
CHECKPOINT_DIR = 'checkpoints/'
OUTPUT_DIR = 'output/'

MODEL = "yolo11l"
LATENCY_TARGET = 300