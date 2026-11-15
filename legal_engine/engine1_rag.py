import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel


class LegalChunk(BaseModel):
    chunk_id: str
    act_title: str
    section: str
    text: str
    effective_date: str  # YYYY-MM-DD
    expiry_date: Optional[str] = None


class TemporalLegalRAG:
    def __init__(self, dense_model_name: str = "BAAI/bge-m3", encoder: Any = None):
        """
        encoder: optional object with .encode(texts, normalize_embeddings=..., convert_to_numpy=...).
        Pass a fake one in tests so the BGE-M3 model (~2 GB) is not downloaded.
        """
        self.chunks: List[LegalChunk] = []
        self.dense_embeddings: Optional[np.ndarray] = None
        self.dense_model_name = dense_model_name
        self._model = encoder

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.dense_model_name)
        return self._model

    def load_corpus(self, documents: List[Dict[str, Any]]):
        self.chunks = [LegalChunk(**d) for d in documents]
        if not self.chunks:
            self.dense_embeddings = None
            return
        texts = [f"Section: {c.section}. Content: {c.text}" for c in self.chunks]
        self.dense_embeddings = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    @staticmethod
    def _is_valid_temporally(chunk: LegalChunk, target_date: str) -> bool:
        target = datetime.strptime(target_date, "%Y-%m-%d")
        effective = datetime.strptime(chunk.effective_date, "%Y-%m-%d")
        if target < effective:
            return False
        if chunk.expiry_date:
            expiry = datetime.strptime(chunk.expiry_date, "%Y-%m-%d")
            if target > expiry:
                return False
        return True

    def retrieve(
        self, query: str, target_date: str = "2026-07-01", top_k: int = 3
    ) -> List[LegalChunk]:
        if not self.chunks or self.dense_embeddings is None:
            return []

        valid_indices = [
            i for i, c in enumerate(self.chunks)
            if self._is_valid_temporally(c, target_date)
        ]
        if not valid_indices:
            return []

        # Dense scoring (always 1-D, even with a single valid chunk)
        q_emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        dense_scores = self.dense_embeddings[valid_indices] @ q_emb

        # Lexical scoring
        q_terms = set(re.findall(r"\w+", query.lower()))
        lexical_scores = np.array([
            len(q_terms & set(re.findall(r"\w+", self.chunks[i].text.lower())))
            / (len(q_terms) + 1e-8)
            for i in valid_indices
        ])

        # Reciprocal Rank Fusion
        d_ranks = np.argsort(np.argsort(-dense_scores))
        l_ranks = np.argsort(np.argsort(-lexical_scores))
        rrf = 1.0 / (60 + d_ranks) + 1.0 / (60 + l_ranks)

        top_local = np.argsort(-rrf)[:top_k]
        return [self.chunks[valid_indices[i]] for i in top_local]