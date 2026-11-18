"""FastAPI layer for the Document RAG Assistant.

Run it:  uvicorn main:app --reload
Then open http://127.0.0.1:8000 in a browser.

Nothing here needs an API key. With no key configured the assistant answers by
quoting the matching passages; add a free Gemini or Groq key (or run Ollama) and
it writes proper answers instead. See README.md.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app import config
from app.engine import get_engine
from app.loaders import EmptyDocument, UnsupportedFile

app = FastAPI(
    title="AI-Powered Document RAG Assistant",
    description="Ask questions about your own PDFs, TXT and Markdown files. Runs free.",
    version="2.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ \-()]+")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How do I rotate the API keys?"])
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    provider: str
    model: str
    chunks: list[dict]


def _safe_filename(name: str | None) -> str:
    """Never trust an upload's filename: no paths, no surprises."""
    candidate = Path(name or "").name.strip()
    candidate = _SAFE_NAME.sub("_", candidate)
    if not candidate or candidate.startswith("."):
        raise HTTPException(status_code=400, detail="That file needs a normal file name.")
    if Path(candidate).suffix.lower() not in config.ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT and MD files are supported.",
        )
    return candidate


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        return HTMLResponse(
            "<h1>Document RAG Assistant</h1><p>UI file missing. "
            'API docs are at <a href="/docs">/docs</a>.</p>'
        )
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict:
    return get_engine().status()


@app.post("/ingest")
def ingest_document(file: UploadFile = File(...)) -> dict:
    filename = _safe_filename(file.filename)
    config.ensure_dirs()
    destination = Path(config.UPLOAD_DIR) / filename

    limit = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    written = 0
    with destination.open("wb") as buffer:
        while True:
            block = file.file.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            if written > limit:
                buffer.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File is larger than the {config.MAX_UPLOAD_MB:g} MB limit.",
                )
            buffer.write(block)

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="That file is empty.")

    try:
        result = get_engine().ingest_file(destination, source_name=filename)
    except (UnsupportedFile, EmptyDocument) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "success", **result}


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    try:
        result = get_engine().ask(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return QueryResponse(**result.as_dict())


@app.get("/documents")
def list_documents() -> dict:
    documents = get_engine().documents()
    return {
        "count": len(documents),
        "documents": [{"source": name, "chunks": n} for name, n in documents.items()],
    }


@app.delete("/documents/{source}")
def delete_document(source: str) -> dict:
    removed = get_engine().delete(Path(source).name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No indexed document named {source!r}.")
    return {"status": "deleted", "source": source, "chunks_removed": removed}


@app.post("/reset")
def reset_index() -> dict:
    return {"status": "cleared", "chunks_removed": get_engine().reset()}


@app.exception_handler(Exception)
def unhandled_error(_request, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
