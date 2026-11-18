"""The RAG engine: ingest documents, retrieve chunks, ground an answer in them."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from . import config
from .chunking import split_text
from .llm import Answer, ExtractiveProvider, LLMError, get_provider
from .loaders import load_file
from .retrieval import Chunk, DocumentStore, Hit, make_retriever

SYSTEM_PROMPT = (
    "You are a technical documentation assistant. Answer the question using only "
    "the context below. If the context does not contain the answer, say plainly "
    "that the documentation does not cover it. Never invent details, and cite the "
    "source labels (for example [handbook.pdf p.3]) next to the facts you use."
)


@dataclass
class QueryResult:
    answer: str
    sources: list[str]
    provider: str
    model: str
    hits: list[Hit]

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "provider": self.provider,
            "model": self.model,
            "chunks": [
                {
                    "source": hit.chunk.source,
                    "page": hit.chunk.page,
                    "score": round(hit.score, 4),
                    "preview": hit.chunk.text[:300],
                }
                for hit in self.hits
            ],
        }


class RAGEngine:
    def __init__(self, store: DocumentStore | None = None, retriever=None, provider=None) -> None:
        config.ensure_dirs()
        self.store = store or DocumentStore()
        self.retriever = retriever or make_retriever()
        self.provider = provider or get_provider()
        self._lock = threading.Lock()
        self._dirty = True

    # ----------------------------------------------------------------- ingest
    def ingest_file(self, path: str | Path, source_name: str | None = None) -> dict:
        path = Path(path)
        source = source_name or path.name
        pages = load_file(path)

        chunks: list[Chunk] = []
        for page in pages:
            for index, text in enumerate(
                split_text(page.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            ):
                chunks.append(
                    Chunk(text=text, source=source, page=page.page, chunk_index=index)
                )

        if not chunks:
            return {"source": source, "chunks_indexed": 0, "replaced": 0, "pages": len(pages)}

        with self._lock:
            replaced = self.store.remove_source(source)  # re-ingest replaces, never duplicates
            self.store.add(chunks)
            self._dirty = True

        return {
            "source": source,
            "pages": len(pages),
            "chunks_indexed": len(chunks),
            "replaced": replaced,
            "total_chunks": len(self.store),
        }

    # -------------------------------------------------------------- retrieval
    def _reindex(self) -> None:
        if not self._dirty:
            return
        with self._lock:
            # the dense cache is keyed by row count, so drop it whenever chunks change
            cache = Path(config.INDEX_DIR) / "vectors.npy"
            cache.unlink(missing_ok=True)
            self.retriever.build(self.store.texts(), Path(config.INDEX_DIR))
            self._dirty = False

    def search(self, question: str, top_k: int | None = None) -> list[Hit]:
        if not len(self.store):
            return []
        self._reindex()
        k = top_k or config.TOP_K
        results = self.retriever.search(question, k)
        return [
            Hit(chunk=self.store.chunks[index], score=score)
            for index, score in results
            if score >= config.MIN_SCORE
        ]

    # ------------------------------------------------------------------- ask
    def ask(self, question: str, top_k: int | None = None) -> QueryResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("Ask an actual question.")

        if not len(self.store):
            return QueryResult(
                answer="No documents have been indexed yet. Upload a PDF, TXT or MD file first.",
                sources=[], provider=self.provider.name, model="-", hits=[],
            )

        hits = self.search(question, top_k)
        if not hits:
            return QueryResult(
                answer="Nothing in the indexed documents matches that question.",
                sources=[], provider=self.provider.name, model="-", hits=[],
            )

        context = "\n\n---\n\n".join(
            f"[{hit.chunk.label()}]\n{hit.chunk.text}" for hit in hits
        )
        user_prompt = f"Question: {question}\n\nContext:\n{context}"

        try:
            answer = self.provider.complete(SYSTEM_PROMPT, user_prompt)
        except LLMError as exc:
            # a dead provider should never cost you the retrieval work
            fallback = ExtractiveProvider().complete(SYSTEM_PROMPT, user_prompt)
            answer = Answer(
                text=f"{fallback.text}\n\n(Provider '{self.provider.name}' failed: {exc})",
                provider=f"{self.provider.name}->extractive",
                model=fallback.model,
            )

        sources: list[str] = []
        for hit in hits:
            label = hit.chunk.label()
            if label not in sources:
                sources.append(label)

        return QueryResult(
            answer=answer.text,
            sources=sources,
            provider=answer.provider,
            model=answer.model,
            hits=hits,
        )

    # ---------------------------------------------------------------- admin
    def documents(self) -> dict[str, int]:
        return self.store.sources()

    def delete(self, source: str) -> int:
        with self._lock:
            removed = self.store.remove_source(source)
            self._dirty = True
        for suffix in ("", ".tmp"):
            candidate = Path(config.UPLOAD_DIR) / f"{source}{suffix}"
            if candidate.is_file():
                candidate.unlink(missing_ok=True)
        return removed

    def reset(self) -> int:
        with self._lock:
            removed = self.store.clear()
            (Path(config.INDEX_DIR) / "vectors.npy").unlink(missing_ok=True)
            self._dirty = True
        return removed

    def status(self) -> dict:
        return {
            "documents": len(self.store.sources()),
            "chunks": len(self.store),
            "retriever": self.retriever.name,
            "retriever_detail": getattr(self.retriever, "description", ""),
            "llm_provider": self.provider.name,
            "llm_model": getattr(self.provider, "model", "") or "auto",
            "paid_services_used": "none",
            "data_dir": str(config.DATA_DIR),
        }


_engine: RAGEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> RAGEngine:
    """Lazy singleton, so importing the app never touches the network or a model."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = RAGEngine()
        return _engine
