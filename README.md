# 🎨 Media AI Local

> Aplicação local de IA generativa para **imagem, img2img e vídeo**, construída com FastAPI, Diffusers, SQLite e FAISS.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Diffusers](https://img.shields.io/badge/Hugging%20Face-Diffusers-FFD21E)
![FAISS](https://img.shields.io/badge/Memory-FAISS-00A98F)
![SQLite](https://img.shields.io/badge/Storage-SQLite-003B57)

## O que demonstra

Este projeto foi organizado como um case de **IA aplicada**:

- geração de imagens com txt2img;
- transformação de imagens com img2img;
- geração de vídeo curto a partir de imagem;
- histórico persistente em SQLite;
- busca semântica de prompts com FAISS + Sentence Transformers;
- API FastAPI com documentação automática;
- configuração por ambiente;
- controles de limite para uploads, resolução e frames;
- filtro básico para prompts com dados sensíveis e usos indevidos.

## Arquitetura

```text
Browser / Client
       │
       ▼
    FastAPI
       │
 ┌─────┼───────────────┐
 ▼     ▼               ▼
Image  Video         Safety
Engine Engine        Layer
 │      │
 └──┬───┘
    ▼
Generated Media
    │
    ├── SQLite  → histórico/metadados
    └── FAISS   → busca semântica
```

## Estrutura

```text
media_ai_local/
├── backend/app/
│   ├── main.py          # API e endpoints
│   ├── config.py        # configuração
│   ├── image_engine.py  # txt2img / img2img
│   ├── video_engine.py  # image -> video
│   ├── memory_faiss.py  # memória semântica
│   ├── safety.py        # filtros de entrada
│   ├── db.py            # SQLite
│   ├── models_db.py     # modelo de persistência
│   └── templates/       # interface web
├── data/                # dados gerados localmente, fora do Git
├── .env.example
├── requirements.txt
└── run_local.bat
```

## Instalação no Windows

### 1. Criar ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configurar

```powershell
Copy-Item .env.example .env
```

Para GPU NVIDIA:

```text
DEVICE=cuda
LOW_VRAM=true
```

Para CPU, use:

```text
DEVICE=cpu
```

### 3. Iniciar

```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Abra:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

Também é possível usar `run_local.bat`.

## API

| Endpoint | Função |
|---|---|
| `GET /health` | saúde/configuração |
| `POST /generate/image` | txt2img |
| `POST /generate/img2img` | img2img |
| `POST /generate/video_from_image` | imagem → vídeo |
| `GET /history` | histórico |
| `GET /search?q=...` | busca semântica |
| `GET /file?path=...` | entrega segura de arquivos gerados |

## Segurança

O projeto não deve ser tratado como plataforma pública de geração sem uma camada adicional de autenticação/autorização.

A implementação atual inclui:

- segregação de configurações em `.env`;
- limite de tamanho de upload;
- validação de caminhos com `Path.relative_to()`;
- filtro de prompts sensíveis;
- armazenamento de artefatos gerados fora do controle de versão.

> O filtro de segurança é heurístico e não substitui controles de segurança de uma aplicação de produção.

## Dados e modelos

Modelos da Hugging Face e arquivos gerados podem ocupar muito espaço. Eles são deliberadamente mantidos fora do Git.

O repositório público contém **código e configuração reproduzível**, não a biblioteca de modelos nem o histórico local de gerações.

## Roadmap

- [x] API FastAPI
- [x] txt2img
- [x] img2img
- [x] image → video
- [x] memória SQLite
- [x] busca semântica FAISS
- [x] configuração por ambiente
- [x] proteção do endpoint de arquivos
- [ ] testes automatizados da API
- [ ] autenticação
- [ ] observabilidade
- [ ] demonstração visual/release

## 👨‍💻 Autor

**21Programe**
