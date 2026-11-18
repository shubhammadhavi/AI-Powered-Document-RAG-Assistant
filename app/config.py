"""Configuration. Everything has a working default, so the app runs with no .env at all."""

from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load a .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env("RAG_DATA_DIR") or (BASE_DIR / "data")).expanduser()
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

# Retrieval ---------------------------------------------------------------
# "auto"  -> sentence-transformers if installed, else tfidf
# "tfidf" -> scikit-learn TF-IDF, zero downloads, zero cost
# "st"    -> local sentence-transformers model (free, runs offline after first download)
EMBEDDINGS = _env("RAG_EMBEDDINGS", "auto").lower()
ST_MODEL = _env("RAG_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = _int_env("RAG_CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int_env("RAG_CHUNK_OVERLAP", 200)
TOP_K = _int_env("RAG_TOP_K", 4)
MIN_SCORE = float(_env("RAG_MIN_SCORE", "0.02") or 0.02)

# Generation --------------------------------------------------------------
# "auto" picks the first provider that is actually usable:
#   Gemini free tier -> Groq free tier -> local Ollama -> extractive (no model at all)
LLM_PROVIDER = _env("RAG_LLM", "auto").lower()

GOOGLE_API_KEY = _env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY")
GEMINI_MODEL = _env("GEMINI_MODEL")  # blank = auto-discover a free Flash model

GROQ_API_KEY = _env("GROQ_API_KEY")
GROQ_MODEL = _env("GROQ_MODEL")  # blank = auto-discover a small free model

OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.2")

REQUEST_TIMEOUT = float(_env("RAG_TIMEOUT", "60") or 60)
MAX_UPLOAD_MB = float(_env("RAG_MAX_UPLOAD_MB", "25") or 25)

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
