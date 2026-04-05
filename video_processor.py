"""Extract frames and audio from simulation videos."""

import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    interval_sec: float = config.FRAME_INTERVAL_SEC,
    max_frames: int = config.MAX_FRAMES,
    scene_change_threshold: float = config.SCENE_CHANGE_THRESHOLD,
) -> list[Path]:
    """
    Extract key frames from a video using interval sampling + scene-change
    detection.  Returns list of saved JPEG paths sorted by timestamp.
    """
    video_path = Path(video_path)
    out_dir = config.UPLOADS_DIR / video_path.stem / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    interval_frames = int(fps * interval_sec)

    logger.info(
        "Video: %.1f sec, %.0f fps, %d total frames, sampling every %d frames",
        duration_sec, fps, total_frames, interval_frames,
    )

    saved: list[Path] = []
    prev_hist = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_interval = (frame_idx % interval_frames == 0)
        is_scene_change = False

        # lightweight scene-change detection via histogram diff
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            if diff > scene_change_threshold:
                is_scene_change = True
        prev_hist = hist

        if is_interval or is_scene_change:
            timestamp = frame_idx / fps
            out_path = out_dir / f"frame_{frame_idx:06d}_t{timestamp:.1f}s.jpg"
            cv2.imwrite(
                str(out_path), frame,
                [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_QUALITY],
            )
            saved.append(out_path)

            if len(saved) >= max_frames:
                logger.info("Reached max frames (%d), stopping extraction.", max_frames)
                break

        frame_idx += 1

    cap.release()
    logger.info("Extracted %d frames from %s", len(saved), video_path.name)
    return saved


def get_video_metadata(video_path: str | Path) -> dict:
    """Return basic metadata about the video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 1),
        "resolution": f"{width}x{height}",
    }


def extract_audio(video_path: str | Path) -> Path | None:
    """Extract audio track as WAV using ffmpeg. Returns None if no audio."""
    video_path = Path(video_path)
    out_dir = config.UPLOADS_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "audio.wav"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(audio_path),
            ],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            logger.warning("ffmpeg exited with code %d: %s", result.returncode, stderr[-500:])
            return None
        if not audio_path.exists():
            logger.warning("ffmpeg ran but audio file was not created.")
            return None
        file_size = audio_path.stat().st_size
        logger.info("Audio file size: %d bytes", file_size)
        if file_size < 100:
            logger.info("Audio track too small (%d bytes), likely silent.", file_size)
            return None
        return audio_path
    except FileNotFoundError:
        logger.warning("ffmpeg not found. Install it: winget install Gyan.FFmpeg")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out after 300s.")
        return None
    except Exception as e:
        logger.warning("Audio extraction failed: %s", e)
        return None


def transcribe_audio(audio_path: str | Path) -> str:
    """Transcribe audio using local Whisper model."""
    try:
        import whisper
    except ImportError:
        logger.warning("Whisper not installed — skipping transcription.")
        return ""

    model = whisper.load_model(config.WHISPER_MODEL)
    result = model.transcribe(str(audio_path), language="en")
    return result.get("text", "")
