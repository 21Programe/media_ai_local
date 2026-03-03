from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy.orm import Session
from sqlalchemy import select, desc

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

app = FastAPI(title="Media AI Local")

# pastas
ensure_dir(settings.data_dir)
ensure_dir(os.path.join(settings.data_dir, "outputs", "images"))
ensure_dir(os.path.join(settings.data_dir, "outputs", "videos"))
ensure_dir(settings.faiss_dir)

# DB
Base.metadata.create_all(bind=engine)

# UI
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Engines
img_engine = ImageEngine()
vid_engine = VideoEngine()
memory = FaissMemory(settings.faiss_dir, settings.embed_model)

@app.get("/health")
def health():
    return {"ok": True, "device": settings.device, "low_vram": settings.low_vram}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def _save_generation(db: Session, kind: str, prompt: str, negative: str, model: str, seed: int, params: Dict[str, Any], out_path: str) -> int:
    g = Generation(
        kind=kind,
        prompt=prompt,
        negative_prompt=negative,
        model=model,
        seed=int(seed),
        params_json=json.dumps(params, ensure_ascii=False),
        output_path=out_path,
    )
    db.add(g)
    db.commit()
    db.refresh(g)
    # indexa o prompt
    memory.add(gen_id=g.id, text=f"{kind} | {prompt}")
    return g.id

@app.post("/generate/image")
def generate_image(payload: Dict[str, Any], db: Session = Depends(get_db)):
    prompt = (payload.get("prompt") or "").strip()
    negative = (payload.get("negative_prompt") or "").strip()

    ok, msg = check_prompt(prompt)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    width = int(payload.get("width") or 768)
    height = int(payload.get("height") or 768)
    steps = int(payload.get("steps") or 30)
    cfg = float(payload.get("cfg") or 7.0)
    seed = int(payload.get("seed") or 0)

    res = img_engine.txt2img(prompt, negative, width, height, steps, cfg, seed)

    out_name = f"img_{ts_id()}_{res.seed}.png"
    out_path = os.path.join(settings.data_dir, "outputs", "images", out_name)
    save_image(res.image, out_path)

    gen_id = _save_generation(
        db=db,
        kind="image",
        prompt=prompt,
        negative=negative,
        model=settings.image_model,
        seed=res.seed,
        params={"width": width, "height": height, "steps": steps, "cfg": cfg},
        out_path=out_path,
    )

    return {"ok": True, "id": gen_id, "seed": res.seed, "path": out_path, "url": f"/file?path={out_path}"}

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
    try:
        init = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    res = img_engine.img2img(prompt, init, negative_prompt, strength, steps, cfg, seed)

    out_name = f"img2img_{ts_id()}_{res.seed}.png"
    out_path = os.path.join(settings.data_dir, "outputs", "images", out_name)
    save_image(res.image, out_path)

    gen_id = _save_generation(
        db=db,
        kind="img2img",
        prompt=prompt,
        negative=negative_prompt,
        model=settings.image_model,
        seed=res.seed,
        params={"strength": float(strength), "steps": int(steps), "cfg": float(cfg)},
        out_path=out_path,
    )

    return {"ok": True, "id": gen_id, "seed": res.seed, "path": out_path, "url": f"/file?path={out_path}"}

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
    try:
        init = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    frames = clamp(int(frames), 8, settings.max_video_frames)
    fps = clamp(int(fps), 4, settings.max_video_fps)

    out_name = f"vid_{ts_id()}_{seed if seed else 'auto'}.mp4"
    out_path = os.path.join(settings.data_dir, "outputs", "videos", out_name)

    res = vid_engine.image_to_video(
        init_image=init,
        out_mp4_path=out_path,
        motion_bucket_id=int(motion_bucket_id),
        noise_aug_strength=float(noise_aug_strength),
        frames=frames,
        fps=fps,
        seed=int(seed),
    )

    gen_id = _save_generation(
        db=db,
        kind="video",
        prompt="(video from image)",
        negative="",
        model=settings.video_model,
        seed=int(seed),
        params={"frames": frames, "fps": fps, "motion_bucket_id": motion_bucket_id, "noise_aug_strength": noise_aug_strength},
        out_path=out_path,
    )

    return {"ok": True, "id": gen_id, "path": out_path, "url": f"/file?path={out_path}", "frames": res.frames, "fps": res.fps}

@app.get("/history")
def history(limit: int = 50, db: Session = Depends(get_db)):
    limit = clamp(int(limit), 1, 200)
    rows = db.execute(select(Generation).order_by(desc(Generation.id)).limit(limit)).scalars().all()
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "kind": r.kind,
            "prompt": r.prompt,
            "model": r.model,
            "seed": r.seed,
            "params": json.loads(r.params_json or "{}"),
            "path": r.output_path,
            "url": f"/file?path={r.output_path}",
            "created_at": str(r.created_at),
        })
    return {"ok": True, "items": out}

@app.get("/search")
def search(q: str, k: int = 10, db: Session = Depends(get_db)):
    k = clamp(int(k), 1, 50)
    hits = memory.search(q, k=k)
    if not hits:
        return {"ok": True, "results": []}

    ids = [h.gen_id for h in hits]
    rows = db.execute(select(Generation).where(Generation.id.in_(ids))).scalars().all()
    mp = {r.id: r for r in rows}

    results = []
    for h in hits:
        r = mp.get(h.gen_id)
        if not r:
            continue
        results.append({
            "id": r.id,
            "score": h.score,
            "kind": r.kind,
            "prompt": r.prompt,
            "model": r.model,
            "seed": r.seed,
            "url": f"/file?path={r.output_path}",
            "path": r.output_path,
        })
    return {"ok": True, "results": results}

@app.get("/file")
def get_file(path: str):
    # segurança: só permitir dentro da pasta data_dir
    path = os.path.abspath(path)
    root = os.path.abspath(settings.data_dir)
    if not path.startswith(root):
        raise HTTPException(status_code=403, detail="Acesso negado.")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")
    # FastAPI StaticFiles não cobre caminhos dinâmicos; vamos devolver como HTML simples com link
    # (O frontend usa <img src="/file?path=..."> com PNG funciona em muitos casos, mas aqui mantemos simples.)
    return HTMLResponse(f"<a href='file:///{path}'>Abrir no PC</a><pre>{path}</pre>")