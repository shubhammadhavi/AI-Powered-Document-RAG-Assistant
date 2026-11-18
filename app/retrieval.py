"""Chunk storage + vector search, kept deliberately small.

The index is a JSONL file of chunks plus (for dense embeddings) a .npy matrix.
No Chroma, no SQLite, no server - so there is nothing to install that can break,
and nothing to pay for.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config


@dataclass
class Chunk:
    text: str
    source: str  # original file name, not a path on disk
    page: int = 1
    chunk_index: int = 0

    def label(self) -> str:
        return f"{self.source} (p.{self.page})" if self.page else self.source


@dataclass
class Hit:
    chunk: Chunk
    score: float


# --------------------------------------------------------------------------- store


class DocumentStore:
    """Chunks on disk, in one JSONL file."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = Path(directory or config.INDEX_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "chunks.jsonl"
        self.chunks: list[Chunk] = []
        self.load()

    def load(self) -> None:
        self.chunks = []
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.chunks.append(Chunk(**json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue  # skip a corrupted line rather than refusing to start

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    # mutations -------------------------------------------------------------
    def add(self, chunks: list[Chunk]) -> int:
        self.chunks.extend(chunks)
        self.save()
        return len(chunks)

    def remove_source(self, source: str) -> int:
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.source != source]
        removed = before - len(self.chunks)
        if removed:
            self.save()
        return removed

    def clear(self) -> int:
        removed = len(self.chunks)
        self.chunks = []
        self.save()
        return removed

    # reads -----------------------------------------------------------------
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        return dict(sorted(counts.items()))

    def __len__(self) -> int:
        return len(self.chunks)


# ----------------------------------------------------------------- retrievers


class TfidfRetriever:
    """Keyword/statistical retrieval with scikit-learn. No model download, no API."""

    name = "tfidf"
    description = "scikit-learn TF-IDF (local, free, no download)"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer_cls = TfidfVectorizer
        self._vectorizer = None
        self._matrix = None

    def build(self, texts: list[str], index_dir: Path) -> None:  # noqa: ARG002
        if not texts:
            self._vectorizer, self._matrix = None, None
            return
        self._vectorizer = self._vectorizer_cls(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._vectorizer is None or self._matrix is None:
            return []
        vector = self._vectorizer.transform([query])
        scores = (self._matrix @ vector.T).toarray().ravel()
        return _top_k(scores, k)


class SentenceTransformerRetriever:
    """Local embedding model (free, offline after the first download)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.ST_MODEL
        self.model = SentenceTransformer(self.model_name)
        self._matrix: np.ndarray | None = None

    @property
    def description(self) -> str:
        return f"local embeddings via {self.model_name}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def build(self, texts: list[str], index_dir: Path) -> None:
        cache = Path(index_dir) / "vectors.npy"
        if not texts:
            self._matrix = None
            cache.unlink(missing_ok=True)
            return
        if cache.exists():
            try:
                cached = np.load(cache)
                if cached.shape[0] == len(texts):
                    self._matrix = cached
                    return
            except Exception:
                pass
        self._matrix = self._encode(texts)
        np.save(cache, self._matrix)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._matrix is None or not len(self._matrix):
            return []
        vector = self._encode([query])[0]
        scores = self._matrix @ vector
        return _top_k(scores, k)


def _top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    k = max(1, min(int(k), len(scores)))
    order = np.argpartition(-scores, k - 1)[:k]
    order = order[np.argsort(-scores[order])]
    return [(int(i), float(scores[i])) for i in order]


def make_retriever(kind: str | None = None):
    """Pick a retriever. 'auto' prefers local embeddings but never hard-fails."""
    kind = (kind or config.EMBEDDINGS).lower()

    if kind in {"st", "sentence-transformers", "sentence_transformers"}:
        return SentenceTransformerRetriever()
    if kind == "tfidf":
        return TfidfRetriever()

    # auto
    try:
        return SentenceTransformerRetriever()
    except Exception:
        return TfidfRetriever()
