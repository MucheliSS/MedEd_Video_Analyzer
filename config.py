"""Configuration for the Video Analyzer app."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
UPLOADS_DIR = PROJECT_DIR / "uploads"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── LM Studio (OpenAI-compatible API) ─────────────────────────────────
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"  # placeholder; LM Studio doesn't enforce keys
DEFAULT_MODEL = "gemma-4-e4b-it"  # smaller/faster model; overridden by sidebar

# ── Frame extraction ──────────────────────────────────────────────────
FRAME_INTERVAL_SEC = 3          # extract a frame every N seconds
MAX_FRAMES = 60                 # cap to avoid overwhelming the VLM
SCENE_CHANGE_THRESHOLD = 30.0   # histogram diff threshold for scene changes
FRAME_QUALITY = 70              # JPEG quality (lower = smaller images = fewer tokens)

# ── Analysis ──────────────────────────────────────────────────────────
MAX_FRAMES_PER_BATCH = 2        # frames sent per VLM call (keep under context limit)
CHARACTER_ID_FRAMES = 4         # frames sampled for character identification
WHISPER_MODEL = "base"          # tiny | base | small | medium | large
