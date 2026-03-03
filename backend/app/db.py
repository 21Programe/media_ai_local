from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import settings
from .utils import ensure_dir

# SQLite local
ensure_dir(os.path.dirname(settings.sqlite_path) or ".")

engine = create_engine(f"sqlite:///{settings.sqlite_path}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()