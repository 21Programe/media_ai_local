from __future__ import annotations

from sqlalchemy import String, Text, DateTime, func, Integer
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base

class Generation(Base):
    __tablename__ = "generations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(50), default="image")  # image, img2img, video
    prompt: Mapped[str] = mapped_column(Text, default="")
    negative_prompt: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(300), default="")
    seed: Mapped[int] = mapped_column(Integer, default=0)
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    output_path: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())