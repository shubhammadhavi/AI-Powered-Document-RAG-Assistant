#!/usr/bin/env python3
"""Tests for the pieces that matter, runnable with plain python or pytest:

    python tests/test_core.py
    pytest -q
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.chunking import split_text  # noqa: E402
from app.llm import ExtractiveProvider, GeminiProvider, GroqProvider, extract_answer  # noqa: E402
from app.loaders import EmptyDocument, UnsupportedFile, load_file  # noqa: E402

SAMPLE = """# Payments service

The billing API returns a 402 status code when the customer's card is declined.
Retries are capped at three attempts per hour per customer.

## Key rotation

API keys are rotated every 90 days by the platform team. A rotation invalidates
the previous key after a 24 hour grace period.

## Storage

Invoices are archived to cold storage after 18 months and cannot be edited
afterwards.
"""


def test_chunking_respects_size_and_keeps_text():
    chunks = split_text(SAMPLE * 6, chunk_size=300, chunk_overlap=60)
    assert chunks, "expected chunks"
    assert max(len(c) for c in chunks) <= 300
    assert all(c.strip() for c in chunks)
    assert "402" in " ".join(chunks)


def test_chunking_edge_cases():
    assert split_text("") == []
    assert split_text("   \n ") == []
    assert split_text("short text", 500, 100) == ["short text"]
    assert len(split_text("x" * 900, 200, 50)) >= 4  # no separators anywhere


def test_loader_reads_markdown_and_rejects_junk():
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "payments.md"
        md.write_text(SAMPLE, encoding="utf-8")
        pages = load_file(md)
        assert len(pages) == 1 and "402" in pages[0].text

        weird = Path(tmp) / "weird.txt"
        weird.write_bytes("caf\xe9 latte costs \xa33".encode("cp1252"))
        assert "caf" in load_file(weird)[0].text  # no UnicodeDecodeError

        empty = Path(tmp) / "empty.txt"
        empty.write_text("   ", encoding="utf-8")
        try:
            load_file(empty)
            raise AssertionError("expected EmptyDocument")
        except EmptyDocument:
            pass

        bad = Path(tmp) / "thing.docx"
        bad.write_text("x", encoding="utf-8")
        try:
            load_file(bad)
            raise AssertionError("expected UnsupportedFile")
        except UnsupportedFile:
            pass


def _engine(tmp: str):
    """A fresh engine pointed at a throwaway data directory."""
    config.DATA_DIR = Path(tmp)
    config.UPLOAD_DIR = Path(tmp) / "uploads"
    config.INDEX_DIR = Path(tmp) / "index"
    config.ensure_dirs()
    from app.engine import RAGEngine
    from app.retrieval import DocumentStore, TfidfRetriever

    return RAGEngine(
        store=DocumentStore(config.INDEX_DIR),
        retriever=TfidfRetriever(),
        provider=ExtractiveProvider(),
    )


def test_ingest_query_and_dedupe():
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(tmp)
        source = Path(tmp) / "payments.md"
        source.write_text(SAMPLE, encoding="utf-8")

        first = engine.ingest_file(source)
        assert first["chunks_indexed"] > 0 and first["replaced"] == 0

        again = engine.ingest_file(source)  # re-ingest must replace, not duplicate
        assert again["replaced"] == first["chunks_indexed"]
        assert len(engine.store) == again["chunks_indexed"]

        result = engine.ask("what happens when a card is declined?")
        assert "402" in result.answer
        assert result.sources and result.sources[0].startswith("payments.md")
        assert all("/" not in s and "\\" not in s for s in result.sources)  # no disk paths leak

        off_topic = engine.ask("what is the airspeed velocity of a swallow")
        assert "match" in off_topic.answer.lower() or "does not" in off_topic.answer.lower()

        assert engine.documents() == {"payments.md": again["chunks_indexed"]}
        assert engine.delete("payments.md") == again["chunks_indexed"]
        assert engine.ask("anything").sources == []


def test_persistence_across_restarts():
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(tmp)
        source = Path(tmp) / "payments.md"
        source.write_text(SAMPLE, encoding="utf-8")
        engine.ingest_file(source)
        count = len(engine.store)

        reopened = _engine(tmp)  # simulates stopping and restarting the server
        assert len(reopened.store) == count
        assert "90 days" in reopened.ask("how often are API keys rotated?").answer


def test_pdf_roundtrip():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        print("  (skipped: reportlab not installed)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "handbook.pdf"
        pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
        pdf.drawString(72, 760, "Page one: the deployment window is Tuesday 02:00 UTC.")
        pdf.showPage()
        pdf.drawString(72, 760, "Page two: rollbacks are performed with the deploy CLI.")
        pdf.save()

        engine = _engine(tmp)
        result = engine.ingest_file(pdf_path)
        assert result["pages"] == 2, result

        answer = engine.ask("when is the deployment window?")
        assert "Tuesday" in answer.answer
        assert any("p.1" in s for s in answer.sources)


def test_extractive_answer_is_grounded():
    text = extract_answer("how are keys rotated?", SAMPLE)
    assert "90 days" in text
    assert extract_answer("x", "")  # never crashes on empty context


def test_model_ranking_prefers_free_models():
    gemini = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview"]
    assert max(gemini, key=GeminiProvider._rank) == "gemini-3.5-flash-lite"
    groq = ["llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
    assert max(groq, key=GroqProvider._rank) == "openai/gpt-oss-20b"


def test_provider_falls_back_without_keys():
    config.GOOGLE_API_KEY = ""
    config.GROQ_API_KEY = ""
    config.LLM_PROVIDER = "auto"
    config.OLLAMA_HOST = "http://127.0.0.1:1"  # nothing listening
    from app.llm import get_provider

    assert get_provider().name == "extractive"


def _stub_http(responses):
    """Replace the provider HTTP call with canned responses; records the calls."""
    from app import llm

    calls = []

    def fake(url, *, method="GET", payload=None, headers=None, timeout=None):  # noqa: ARG001
        calls.append({"url": url, "method": method, "payload": payload})
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    llm._request = fake
    return calls


def test_gemini_path_with_stubbed_http():
    from app import config as cfg
    from app import llm

    original = llm._request
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg.INDEX_DIR = Path(tmp)
            llm.CACHE = llm._ModelCache()
            cfg.GOOGLE_API_KEY = "test-key"
            cfg.GEMINI_MODEL = ""
            calls = _stub_http([
                {"models": [
                    {"name": "models/gemini-3.1-pro-preview",
                     "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/gemini-3.5-flash-lite",
                     "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/gemini-embedding-2",
                     "supportedGenerationMethods": ["embedContent"]},
                ]},
                {"candidates": [{"content": {"parts": [{"text": "Keys rotate every 90 days."}]}}]},
            ])
            provider = llm.GeminiProvider()
            answer = provider.complete("sys", "Question: how often?\n\nContext:\nrotated every 90 days")
            assert answer.text == "Keys rotate every 90 days."
            assert answer.model == "gemini-3.5-flash-lite", answer.model
            assert "generateContent" in calls[1]["url"]
            assert calls[1]["payload"]["generationConfig"]["temperature"] == 0.0
            # the model choice is cached, so a second question lists no models again
            calls2 = _stub_http([
                {"candidates": [{"content": {"parts": [{"text": "cached"}]}}]},
            ])
            assert llm.GeminiProvider().complete("s", "u").text == "cached"
            assert len(calls2) == 1
    finally:
        llm._request = original


def test_gemini_retries_when_thinking_eats_the_budget():
    from app import config as cfg
    from app import llm

    original = llm._request
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg.INDEX_DIR = Path(tmp)
            llm.CACHE = llm._ModelCache()
            cfg.GOOGLE_API_KEY = "k"
            cfg.GEMINI_MODEL = "gemini-3.5-flash"
            calls = _stub_http([
                {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]},
                {"candidates": [{"content": {"parts": [{"text": "second try works"}]}}]},
            ])
            assert llm.GeminiProvider().complete("s", "u").text == "second try works"
            budgets = [c["payload"]["generationConfig"]["maxOutputTokens"] for c in calls]
            assert budgets[1] > budgets[0], budgets
    finally:
        llm._request = original
        cfg.GEMINI_MODEL = ""


def test_groq_path_with_stubbed_http():
    from app import config as cfg
    from app import llm

    original = llm._request
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg.INDEX_DIR = Path(tmp)
            llm.CACHE = llm._ModelCache()
            cfg.GROQ_API_KEY = "gsk_test"
            cfg.GROQ_MODEL = ""
            calls = _stub_http([
                {"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "openai/gpt-oss-20b"},
                          {"id": "whisper-large-v3"}]},
                {"choices": [{"message": {"content": "402 means the card was declined."}}]},
            ])
            answer = llm.GroqProvider().complete("sys", "user")
            assert "402" in answer.text
            assert answer.model == "openai/gpt-oss-20b", answer.model
            assert calls[1]["payload"]["model"] == "openai/gpt-oss-20b"
    finally:
        llm._request = original
        cfg.GROQ_API_KEY = ""


def test_provider_failure_still_returns_sources():
    """A quota error must never cost the user the retrieval work."""
    from app.llm import LLMError

    class Broken:
        name = "gemini"
        model = "x"

        def complete(self, system, user):  # noqa: ARG002
            raise LLMError("Free-tier rate limit reached.")

    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(tmp)
        engine.provider = Broken()
        source = Path(tmp) / "payments.md"
        source.write_text(SAMPLE, encoding="utf-8")
        engine.ingest_file(source)

        result = engine.ask("what happens when a card is declined?")
        assert "402" in result.answer                    # extractive fallback answered
        assert "rate limit" in result.answer.lower()     # and said why it fell back
        assert result.sources                            # sources survived
        assert result.provider == "gemini->extractive"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
