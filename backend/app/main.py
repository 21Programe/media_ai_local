from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, desc
from sqlalchemy.orm import Session
from PIL import Image
import io

from .config import settings
from .utils import ensure_dir, ts_id, save_image, clamp
from .safety import check_prompt
from .db import Base, engine, get_db
from .models_db import Generation
from .memory_faiss import FaissMemory
from .image_engine import ImageEngine
from .video_engine import VideoEngine

app = FastAPI(
    title="Media AI Local",
    description="Geração local de imagem/vídeo com memória semântica.",
    version="1.0.0",
)

ensure_dir(settings.data_dir)
ensure_dir(settings.data_dir / "outputs" / "images")
ensure_dir(settings.data_dir / "outputs" / "videos")
ensure_dir(settings.faiss_dir)

Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent / "templates")
)
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
    name="static",
)

img_engine = ImageEngine()
vid_engine = VideoEngine()
memory = FaissMemory(str(settings.faiss_dir), settings.embed_model)


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": settings.device,
        "low_vram": settings.low_vram,
        "image_model": settings.image_model,
        "video_model": settings.video_model,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _save_generation(
    db: Session,
    kind: str,
    prompt: str,
    negative: str,
    model: str,
    seed: int,
    params: Dict[str, Any],
    out_path: Path,
) -> int:
    generation = Generation(
        kind=kind,
        prompt=prompt,
        negative_prompt=negative,
        model=model,
        seed=int(seed),
        params_json=json.dumps(params, ensure_ascii=False),
        output_path=str(out_path),
    )
    db.add(generation)
    db.commit()
    db.refresh(generation)
    memory.add(gen_id=generation.id, text=f"{kind} | {prompt}")
    return generation.id


def _validate_upload_size(raw: bytes) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload excede o limite de {settings.max_upload_mb} MB.",
        )


def _safe_data_file(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    root = settings.data_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Acesso negado.") from exc

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")

    return candidate


@app.post("/generate/image")
def generate_image(payload: Dict[str, Any], db: Session = Depends(get_db)):
    prompt = (payload.get("prompt") or "").strip()
    negative = (payload.get("negative_prompt") or "").strip()

    ok, msg = check_prompt(prompt)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    width = clamp(int(payload.get("width") or 768), 256, settings.max_image_size)
    height = clamp(int(payload.get("height") or 768), 256, settings.max_image_size)
    steps = clamp(int(payload.get("steps") or 30), 5, 80)
    cfg = float(payload.get("cfg") or 7.0)
    seed = int(payload.get("seed") or 0)

    result = img_engine.txt2img(prompt, negative, width, height, steps, cfg, seed)

    out_path = settings.data_dir / "outputs" / "images" / f"img_{ts_id()}_{result.seed}.png"
    save_image(result.image, out_path)

    generation_id = _save_generation(
        db=db,
        kind="image",
        prompt=prompt,
        negative=negative,
        model=settings.image_model,
        seed=result.seed,
        params={"width": width, "height": height, "steps": steps, "cfg": cfg},
        out_path=out_path,
    )

    return {
        "ok": True,
        "id": generation_id,
        "seed": result.seed,
        "url": f"/file?path={out_path}",
    }


@app.post("/generate/img2img")
async def generate_img2img(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.55),
    steps: int = Form(30),
    cfg: float = Form(7.0),
    seed: int = Form(0),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    prompt = prompt.strip()
    ok, msg = check_prompt(prompt)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    raw = await image.read()
    _validate_upload_size(raw)

    try:
        init = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Imagem inválida.") from exc

    result = img_engine.img2img(
        prompt,
        init,
        negative_prompt,
        strength,
        steps,
        cfg,
        seed,
    )

    out_path = settings.data_dir / "outputs" / "images" / f"img2img_{ts_id()}_{result.seed}.png"
    save_image(result.image, out_path)

    generation_id = _save_generation(
        db=db,
        kind="img2img",
        prompt=prompt,
        negative=negative_prompt,
        model=settings.image_model,
        seed=result.seed,
        params={"strength": float(strength), "steps": int(steps), "cfg": float(cfg)},
        out_path=out_path,
    )

    return {
        "ok": True,
        "id": generation_id,
        "seed": result.seed,
        "url": f"/file?path={out_path}",
    }


@app.post("/generate/video_from_image")
async def generate_video_from_image(
    image: UploadFile = File(...),
    frames: int = Form(25),
    fps: int = Form(8),
    motion_bucket_id: int = Form(127),
    noise_aug_strength: float = Form(0.02),
    seed: int = Form(0),
    db: Session = Depends(get_db),
):
    raw = await image.read()
    _validate_upload_size(raw)

    try:
        init = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Imagem inválida.") from exc

    frames = clamp(int(frames), 8, settings.max_video_frames)
    fps = clamp(int(fps), 4, settings.max_video_fps)

    out_path = settings.data_dir / "outputs" / "videos" / f"vid_{ts_id()}_{seed or 'auto'}.mp4"

    result = vid_engine.image_to_video(
        init_image=init,
        out_mp4_path=str(out_path),
        motion_bucket_id=int(motion_bucket_id),
        noise_aug_strength=float(noise_aug_strength),
        frames=frames,
        fps=fps,
        seed=int(seed),
    )

    generation_id = _save_generation(
        db=db,
        kind="video",
        prompt="(video from image)",
        negative="",
        model=settings.video_model,
        seed=int(seed),
        params={
            "frames": frames,
            "fps": fps,
            "motion_bucket_id": motion_bucket_id,
            "noise_aug_strength": noise_aug_strength,
        },
        out_path=out_path,
    )

    return {
        "ok": True,
        "id": generation_id,
        "url": f"/file?path={out_path}",
        "frames": result.frames,
        "fps": result.fps,
    }


@app.get("/history")
def history(limit: int = 50, db: Session = Depends(get_db)):
    limit = clamp(int(limit), 1, 200)
    rows = db.execute(
        select(Generation).order_by(desc(Generation.id)).limit(limit)
    ).scalars().all()

    return {
        "ok": True,
        "items": [
            {
                "id": row.id,
                "kind": row.kind,
                "prompt": row.prompt,
                "model": row.model,
                "seed": row.seed,
                "params": json.loads(row.params_json or "{}"),
                "url": f"/file?path={row.output_path}",
                "created_at": str(row.created_at),
            }
            for row in rows
        ],
    }


@app.get("/search")
def search(q: str, k: int = 10, db: Session = Depends(get_db)):
    k = clamp(int(k), 1, 50)
    hits = memory.search(q, k=k)
    if not hits:
        return {"ok": True, "results": []}

    ids = [hit.gen_id for hit in hits]
    rows = db.execute(
        select(Generation).where(Generation.id.in_(ids))
    ).scalars().all()
    by_id = {row.id: row for row in rows}

    results = []
    for hit in hits:
        row = by_id.get(hit.gen_id)
        if not row:
            continue

        results.append(
            {
                "id": row.id,
                "score": hit.score,
                "kind": row.kind,
                "prompt": row.prompt,
                "model": row.model,
                "seed": row.seed,
                "url": f"/file?path={row.output_path}",
            }
        )

    return {"ok": True, "results": results}


@app.get("/file")
def get_file(path: str):
    return FileResponse(_safe_data_file(path))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return HTMLResponse(status_code=204)
