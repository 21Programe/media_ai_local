from __future__ import annotations

import os
import time
from typing import Optional
from PIL import Image

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def ts_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def save_image(img: Image.Image, out_path: str) -> None:
    img.save(out_path, format="PNG", optimize=True)

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))