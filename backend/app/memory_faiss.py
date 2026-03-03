from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

@dataclass
class MemoryHit:
    gen_id: int
    score: float

class FaissMemory:
    """
    Indexa PROMPTS (texto) para buscar gerações parecidas.
    Guarda:
      - index.faiss
      - meta.json (pos -> gen_id)
    """
    def __init__(self, faiss_dir: str, model_name: str):
        self.faiss_dir = faiss_dir
        os.makedirs(self.faiss_dir, exist_ok=True)

        self.index_path = os.path.join(self.faiss_dir, "index.faiss")
        self.meta_path = os.path.join(self.faiss_dir, "meta.json")

        self.model = SentenceTransformer(model_name)

        self.doc_ids: List[int] = []
        self.index = None
        self.dim = None

        self._load()

    def _load(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.doc_ids = json.load(f)
        else:
            self.doc_ids = []

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.dim = self.index.d
        else:
            self.index = None
            self.dim = None

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)

    def embed(self, texts: List[str]) -> np.ndarray:
        vec = self.model.encode(texts, normalize_embeddings=True)
        if isinstance(vec, list):
            vec = np.array(vec, dtype="float32")
        return vec.astype("float32")

    def add(self, gen_id: int, text: str) -> None:
        v = self.embed([text])
        if self.index is None:
            self.dim = v.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(v)
        self.doc_ids.append(int(gen_id))
        self._save()

    def search(self, query: str, k: int = 10) -> List[MemoryHit]:
        if self.index is None or not self.doc_ids:
            return []
        qv = self.embed([query])
        scores, idxs = self.index.search(qv, k)
        hits: List[MemoryHit] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx == -1:
                continue
            hits.append(MemoryHit(gen_id=self.doc_ids[idx], score=float(score)))
        return hits