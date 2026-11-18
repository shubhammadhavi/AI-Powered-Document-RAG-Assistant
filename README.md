# AI-Powered Document RAG Assistant

Ask questions about your own PDFs, text and Markdown files and get answers that are
grounded in those documents, with the source page cited.

**It runs at zero cost.** Retrieval happens locally on your machine. Answer writing
uses a free-tier model if you add a key, and if you add nothing at all the assistant
still works — it quotes the passages that answer the question instead of writing prose.

---

## Quickstart

```bash
git clone https://github.com/shubhammadhavi/AI-Powered-Document-RAG-Assistant.git
cd AI-Powered-Document-RAG-Assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload
```

Open <http://127.0.0.1:8000> — drop in a PDF, ask a question. No API key needed to get
that far, and nothing is sent anywhere.

Prefer the terminal?

```bash
python cli.py add docs/handbook.pdf
python cli.py ask "when do deploys happen?"
python cli.py status
```

---

## Getting written answers (still free)

Retrieval always runs locally. For the answer itself, pick whichever of these suits you —
all three are free, and the first two need no credit card.

| Option | Cost | Setup |
|---|---|---|
| **Nothing** (default) | free | Works out of the box. Quotes the best-matching sentences instead of writing an answer. |
| **Google AI Studio** (recommended) | free tier | Get a key at <https://aistudio.google.com/apikey>, put it in `.env` as `GOOGLE_API_KEY` |
| **Groq** | free tier | Get a key at <https://console.groq.com/keys>, put it in `.env` as `GROQ_API_KEY` |
| **Ollama** | free, offline | `ollama pull llama3.2`, then leave it running. Needs ~4 GB RAM. |

```bash
cp .env.example .env     # then paste a key into it, if you have one
```

Model names on free tiers get deprecated regularly, so **this app does not hardcode one**:
it asks the provider which models it currently serves and picks a free one (newest Flash
for Gemini, smallest open model for Groq), caching the choice for a week. If a model
disappears mid-flight, the 404 triggers a re-discovery and the request is retried once.
Pin a specific model with `GEMINI_MODEL=` / `GROQ_MODEL=` in `.env` if you'd rather.

### Better retrieval, also free

The default retriever is scikit-learn TF-IDF: nothing to download, instant start, and
good enough for keyword-ish questions. For semantic matching ("declined payment" finding
"card was rejected"), install local embeddings — free, and offline after the first
download:

```bash
pip install -r requirements-optional.txt   # sentence-transformers (~2 GB with torch)
echo "RAG_EMBEDDINGS=st" >> .env
```

---

## How it works

```
file ─▶ loader ─▶ chunker ─▶ index (JSONL + optional .npy)
 (pypdf)        (recursive, 1000 chars, 200 overlap)
                                   │
question ──────────────────────────┴─▶ top-k by cosine / TF-IDF
                                           │
                                           ▼
                        strict grounded prompt ─▶ free model ─▶ answer + cited pages
                                           └──▶ no model? quote the best sentences
```

| File | What it does |
|---|---|
| `app/loaders.py` | PDF/TXT/MD → page-level text, with encoding and encrypted-PDF handling |
| `app/chunking.py` | Recursive character splitter with overlap |
| `app/retrieval.py` | Chunk store (one JSONL file) + TF-IDF and embedding retrievers |
| `app/llm.py` | Gemini / Groq / Ollama / extractive, model auto-discovery, retries |
| `app/engine.py` | Ingest, retrieve, prompt, cite |
| `main.py` | FastAPI endpoints + the browser UI |
| `cli.py` | Same engine from the terminal |

The index is just `data/index/chunks.jsonl` (plus `vectors.npy` when embeddings are on).
No vector database, no Docker, no SQLite version requirements — deleting the `data/`
folder is a full reset.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Browser UI |
| `GET` | `/status` | Which retriever and model are active |
| `POST` | `/ingest` | Upload and index a file (multipart `file`) |
| `POST` | `/query` | `{"question": "...", "top_k": 4}` → answer, sources, retrieved passages |
| `GET` | `/documents` | Indexed documents and chunk counts |
| `DELETE` | `/documents/{name}` | Remove one document from the index |
| `POST` | `/reset` | Clear the index |
| `GET` | `/docs` | Interactive OpenAPI docs |

```bash
curl -F "file=@handbook.pdf" http://127.0.0.1:8000/ingest
curl -X POST http://127.0.0.1:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question":"what is the rollback procedure?"}'
```

---

## Configuration

All optional; defaults work. See `.env.example` for the full list.

| Variable | Default | Notes |
|---|---|---|
| `RAG_LLM` | `auto` | `auto` / `gemini` / `groq` / `ollama` / `extractive` |
| `RAG_EMBEDDINGS` | `auto` | `auto` / `tfidf` / `st` |
| `RAG_CHUNK_SIZE` | `1000` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_TOP_K` | `4` | Passages retrieved per question |
| `RAG_MIN_SCORE` | `0.02` | Below this, the app says it found nothing rather than guessing |
| `RAG_MAX_UPLOAD_MB` | `25` | Upload size limit |
| `RAG_DATA_DIR` | `./data` | Where uploads and the index live |

---

## Tests

```bash
python tests/test_core.py     # or: pytest -q
```

13 tests, no network needed. They cover chunking edge cases, loader encoding failures,
empty/scanned PDFs, re-ingest deduplication, index persistence across restarts, a
two-page PDF round trip, model discovery and ranking, the Gemini and Groq request and
response paths (stubbed HTTP), the reasoning-model retry, and the rate-limit fallback
that keeps your sources when a provider refuses.

---

## Deploying for free

The app is a single process with local files, so any free tier works: Hugging Face
Spaces, Fly.io, Render, or a Raspberry Pi on your desk. Two things to remember:

- Containers on free tiers have **ephemeral disks** — mount a volume for `data/`, or
  expect the index to reset on redeploy.
- Set the provider key as a secret/environment variable rather than committing `.env`.

```bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

---

## Notes and limits

- Scanned PDFs have no text layer, so they are rejected with a clear message; OCR them
  first (for example with `ocrmypdf`) and re-upload.
- Free tiers rate-limit. On a 429 the app tells you to wait or switch provider, and the
  retrieved passages are still returned, so a quota limit never loses your work.
- Answers are constrained to the retrieved context and asked to cite page labels. That
  reduces hallucination; it does not eliminate it.

## License

MIT
