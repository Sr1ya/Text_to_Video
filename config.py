import os
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "indices")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # Directory for generated videos

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "videos"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "texts"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "audio"), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Index directory: {INDEX_DIR}")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# Fixed index paths
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")
CHUNK_SIZE = 512
MODEL_NAME = "openai/clip-vit-large-patch14"

AUDIO_MODEL = "openai/whisper-small"
AUDIO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Video processing parameters
VIDEO_FRAME_INTERVAL = 10  # Process video in 10-second intervals
VIDEO_SIMILARITY_THRESHOLD = 0.85  # Reduced from 0.95 to allow more frames
VIDEO_AUDIO_PADDING = 5  # 5 seconds padding on each side to match 10-second frame interval
VIDEO_MAX_FRAMES = 1000  # Maximum number of frames to process per video
VIDEO_FRAME_SIZE = (640, 360)  # Target frame size for processing
VIDEO_BATCH_SIZE = 10  # Number of frames to process in each batch

# Video output settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
VIDEO_CODEC = 'libx264'
VIDEO_PRESET = 'medium'
VIDEO_CRF = 23  # Constant Rate Factor (18-28 is good, lower is better quality)

# Subtitle settings
SUBTITLE_FONT = 'Arial'
SUBTITLE_FONT_SIZE = 48
SUBTITLE_COLOR = 'white'
SUBTITLE_OUTLINE_COLOR = 'black'
SUBTITLE_OUTLINE_WIDTH = 2
SUBTITLE_PADDING = 20  # Padding from bottom of video
SUBTITLE_MAX_CHARS_PER_LINE = 42  # Maximum characters per line for subtitles