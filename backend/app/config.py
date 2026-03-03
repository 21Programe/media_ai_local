from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _get_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name, str(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default

@dataclass(frozen=True)
class Settings:
    device: str = os.getenv("DEVICE", "cpu").strip().lower()
    low_vram: bool = _get_bool("LOW_VRAM", True)

    image_model: str = os.getenv("IMAGE_MODEL", "runwayml/stable-diffusion-v1-5").strip()
    video_model: str = os.getenv("VIDEO_MODEL", "stabilityai/stable-video-diffusion-img2vid-xt").strip()

    data_dir: str = os.getenv("DATA_DIR", "./data").strip()
    sqlite_path: str = os.getenv("SQLITE_PATH", "./data/app.db").strip()
    faiss_dir: str = os.getenv("FAISS_DIR", "./data/faiss").strip()

    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()

    max_image_size: int = _get_int("MAX_IMAGE_SIZE", 1024)
    max_video_frames: int = _get_int("MAX_VIDEO_FRAMES", 25)
    max_video_fps: int = _get_int("MAX_VIDEO_FPS", 8)

settings = Settings()