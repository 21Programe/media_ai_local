from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image

from .config import settings
from .utils import clamp

@dataclass
class ImageResult:
    image: Image.Image
    seed: int

class ImageEngine:
    def __init__(self):
        self.device = "cuda" if (settings.device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model_id = settings.image_model

        self.pipe_txt = None
        self.pipe_img = None

    def _is_sdxl(self) -> bool:
        return "xl" in self.model_id.lower()

    def _load(self):
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self._is_sdxl():
            self.pipe_txt = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
            self.pipe_img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        else:
            self.pipe_txt = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
            self.pipe_img = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )

        if self.device == "cuda":
            self.pipe_txt = self.pipe_txt.to("cuda")
            self.pipe_img = self.pipe_img.to("cuda")

            if settings.low_vram:
                self.pipe_txt.enable_attention_slicing()
                self.pipe_img.enable_attention_slicing()
                self.pipe_txt.enable_vae_slicing()
                self.pipe_img.enable_vae_slicing()
        else:
            # CPU: reduzir memória
            self.pipe_txt.enable_attention_slicing()
            self.pipe_img.enable_attention_slicing()

    def ensure_loaded(self):
        if self.pipe_txt is None or self.pipe_img is None:
            self._load()

    def txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 768,
        steps: int = 30,
        cfg: float = 7.0,
        seed: int = 0,
    ) -> ImageResult:
        self.ensure_loaded()

        maxs = settings.max_image_size
        width = clamp(width, 256, maxs)
        height = clamp(height, 256, maxs)

        g = torch.Generator(device=self.device)
        if seed and seed > 0:
            g.manual_seed(seed)
        else:
            seed = torch.seed() % (2**31 - 1)
            g.manual_seed(seed)

        out = self.pipe_txt(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=clamp(steps, 5, 80),
            guidance_scale=float(cfg),
            generator=g,
        )
        return ImageResult(image=out.images[0], seed=int(seed))

    def img2img(
        self,
        prompt: str,
        init_image: Image.Image,
        negative_prompt: str = "",
        strength: float = 0.55,
        steps: int = 30,
        cfg: float = 7.0,
        seed: int = 0,
    ) -> ImageResult:
        self.ensure_loaded()

        g = torch.Generator(device=self.device)
        if seed and seed > 0:
            g.manual_seed(seed)
        else:
            seed = torch.seed() % (2**31 - 1)
            g.manual_seed(seed)

        out = self.pipe_img(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=init_image,
            strength=float(max(0.05, min(0.95, strength))),
            num_inference_steps=clamp(steps, 5, 80),
            guidance_scale=float(cfg),
            generator=g,
        )
        return ImageResult(image=out.images[0], seed=int(seed))