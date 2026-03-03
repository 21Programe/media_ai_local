# Media AI Local (Imagem + Vídeo) — FastAPI + HTML + SQLite + FAISS

Gera:
- Imagens (txt2img, img2img) com Diffusers
- Vídeo curto (image -> video) usando Stable Video Diffusion (SVD)

Memória:
- SQLite: histórico, presets
- FAISS: busca semântica em prompts

## Requisitos
- Windows 10/11
- Python 3.10+ (recomendado 3.11)
- (Opcional) GPU NVIDIA com CUDA para velocidade

## Instalação rápida
1) Rode:
   setup_windows.bat

2) Copie o .env:
   copy .env.example .env

3) Inicie:
   run_local.bat

Abra:
http://127.0.0.1:8000

## Modelos (você escolhe)
Por padrão, o projeto tenta usar SDXL para imagens (pesado).
Se seu PC for fraco, troque para SD 1.5:

No .env:
IMAGE_MODEL=runwayml/stable-diffusion-v1-5

Para vídeo (image->video), use:
VIDEO_MODEL=stabilityai/stable-video-diffusion-img2vid-xt

Observação:
- Modelos baixam na primeira execução (cache).
- Você precisa de espaço em disco.

## Endpoints
- POST /generate/image
- POST /generate/img2img
- POST /generate/video_from_image
- GET  /history
- GET  /search?q=...&k=10
- GET  /health

## Aviso ético
- Sem deepfake de pessoas reais sem consentimento.
- Sem dados sensíveis (CPF, documentos, rastreamento).