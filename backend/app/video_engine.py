from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import List

from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np
import imageio

from .config import settings
from .utils import clamp

@dataclass
class VideoResult:
    mp4_path: str
    frames: int
    fps: int

class VideoEngine:
    """
    Gera vídeo curto a partir de 1 imagem (image -> video) usando SVD.
    """
    def __init__(self):
        self.device = "cuda" if (settings.device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model_id = settings.video_model
        self.pipe = None

    def _load(self):
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

        if self.device == "cuda":
            self.pipe = self.pipe.to("cuda")
            if settings.low_vram:
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()
        else:
            self.pipe.enable_attention_slicing()

    def ensure_loaded(self):
        if self.pipe is None:
            self._load()

    def image_to_video(
        self,
        init_image: Image.Image,
        out_mp4_path: str,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        frames: int = 25,
        fps: int = 8,
        seed: int = 0,
    ) -> VideoResult:
        self.ensure_loaded()

        frames = clamp(frames, 8, settings.max_video_frames)
        fps = clamp(fps, 4, settings.max_video_fps)

        # SVD costuma preferir 576x1024 ou 1024x576 (depende do modelo). Vamos ajustar suave:
        img = init_image.convert("RGB")
        img = img.resize((1024, 576))

        g = torch.Generator(device=self.device)
        if seed and seed > 0:
            g.manual_seed(seed)
        else:
            seed = torch.seed() % (2**31 - 1)
            g.manual_seed(seed)

        out = self.pipe(
            image=img,
            num_frames=frames,
            motion_bucket_id=int(motion_bucket_id),
            noise_aug_strength=float(noise_aug_strength),
            generator=g,
        )

        # out.frames: list[list[PIL]] dependendo da versão
        frames_list = out.frames[0] if isinstance(out.frames, list) and len(out.frames) > 0 else out.frames

        # salvar mp4
        writer = imageio.get_writer(out_mp4_path, fps=fps, codec="libx264", quality=8)
        try:
            for fr in frames_list:
                arr = np.array(fr.convert("RGB"))
                writer.append_data(arr)
        finally:
            writer.close()

        return VideoResult(mp4_path=out_mp4_path, frames=frames, fps=fps)