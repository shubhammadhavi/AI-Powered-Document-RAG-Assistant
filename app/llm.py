"""Answer generation across free providers.

Order of preference when RAG_LLM=auto:
  1. Gemini    - Google AI Studio free tier (no card needed)
  2. Groq      - free tier (no card needed)
  3. Ollama    - a model running locally on your own machine
  4. extractive - no model at all: quote the best-matching sentences

Model *names* on free tiers get deprecated often, so instead of hardcoding one we
ask each provider which models it currently serves and pick a sensible free one.
Only the standard library is used for HTTP, so there is no SDK to install or pin.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config


class LLMError(RuntimeError):
    """Raised when a provider is reachable but refuses to answer."""


@dataclass
class Answer:
    text: str
    provider: str
    model: str


# ------------------------------------------------------------------ http helpers


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)

    try:
        with urllib.request.urlopen(request, timeout=timeout or config.REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:600]
        if exc.code == 429:
            raise LLMError(
                "Free-tier rate limit reached. Wait a minute and retry, or set a "
                "different provider in .env (RAG_LLM=groq / ollama / extractive)."
            ) from exc
        if exc.code in (401, 403):
            raise LLMError(f"API key rejected ({exc.code}). Check the key in .env. {body}") from exc
        raise LLMError(f"HTTP {exc.code} from provider: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Could not reach provider: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError("Provider timed out.") from exc


class _ModelCache:
    """Remember the discovered model so we do not list models on every question."""

    TTL = 7 * 24 * 3600

    def __init__(self) -> None:
        self.path = Path(config.INDEX_DIR) / "model_cache.json"

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get(self, provider: str) -> str | None:
        entry = self._read().get(provider)
        if not entry:
            return None
        if time.time() - entry.get("at", 0) > self.TTL:
            return None
        return entry.get("model")

    def set(self, provider: str, model: str) -> None:
        data = self._read()
        data[provider] = {"model": model, "at": time.time()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def drop(self, provider: str) -> None:
        data = self._read()
        if data.pop(provider, None) is not None:
            try:
                self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass


CACHE = _ModelCache()

_VERSION = re.compile(r"(\d+(?:\.\d+)?)")


def _version_score(name: str) -> float:
    match = _VERSION.search(name)
    return float(match.group(1)) if match else 0.0


# --------------------------------------------------------------------- providers


class GeminiProvider:
    name = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta"
    SKIP = (
        "embedding", "embed", "aqa", "tts", "image", "imagen", "veo", "lyria",
        "audio", "live", "robotics", "computer-use", "vision",
    )

    def __init__(self) -> None:
        self.key = config.GOOGLE_API_KEY
        self.model = config.GEMINI_MODEL or ""

    def available(self) -> bool:
        return bool(self.key)

    def _candidates(self) -> list[str]:
        data = _request(f"{self.BASE}/models?key={self.key}&pageSize=200")
        names = []
        for model in data.get("models", []):
            methods = model.get("supportedGenerationMethods") or model.get(
                "supportedActions", []
            )
            if methods and "generateContent" not in methods:
                continue
            name = (model.get("name") or "").removeprefix("models/")
            if not name or any(bad in name.lower() for bad in self.SKIP):
                continue
            names.append(name)
        return names

    @staticmethod
    def _rank(name: str) -> float:
        """Higher is better: newest Flash generation first, Pro and preview last."""
        low = name.lower()
        score = 8 * _version_score(low)   # newer generation wins
        if "flash" in low:
            score += 100
        if "lite" in low:
            score += 6                    # within a generation, Lite has roomier quota
        if "gemma" in low:
            score += 10
        if "pro" in low:
            score -= 60                   # Pro is generally not on the free tier
        if "preview" in low or "-exp" in low:
            score -= 40                   # preview models get much tighter limits
        if low.endswith("latest"):
            score += 3
        return score

    def resolve_model(self, refresh: bool = False) -> str:
        if self.model:
            return self.model
        if not refresh:
            cached = CACHE.get(self.name)
            if cached:
                return cached
        candidates = self._candidates()
        if not candidates:
            raise LLMError("Gemini returned no usable text models for this key.")
        best = max(candidates, key=self._rank)
        CACHE.set(self.name, best)
        return best

    def complete(
        self, system: str, user: str, _retried: bool = False, max_tokens: int = 2048
    ) -> Answer:
        model = self.resolve_model(refresh=_retried)
        url = f"{self.BASE}/models/{model}:generateContent?key={self.key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens},
        }
        try:
            data = _request(url, method="POST", payload=payload)
        except LLMError as exc:
            # a deprecated/renamed model shows up as a 404: rediscover once
            if not _retried and "404" in str(exc) and not config.GEMINI_MODEL:
                CACHE.drop(self.name)
                return self.complete(system, user, _retried=True)
            raise

        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMError(f"Gemini returned no answer ({reason}).")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        if text:
            return Answer(text=text, provider=self.name, model=model)

        # Reasoning models can spend the whole budget thinking and return no text.
        finish = candidate.get("finishReason", "")
        if finish == "MAX_TOKENS" and max_tokens < 8192:
            return self.complete(system, user, _retried=True, max_tokens=max_tokens * 4)
        raise LLMError(f"Gemini returned an empty answer (finishReason={finish or 'unknown'}).")


class GroqProvider:
    name = "groq"
    BASE = "https://api.groq.com/openai/v1"
    SKIP = ("whisper", "tts", "guard", "embed", "vision", "ocr", "moderation")
    PREFER = ("oss-20b", "8b", "instant", "lite", "mini", "scout", "oss", "llama", "qwen", "gemma")

    def __init__(self) -> None:
        self.key = config.GROQ_API_KEY
        self.model = config.GROQ_MODEL or ""

    def available(self) -> bool:
        return bool(self.key)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"}

    def _candidates(self) -> list[str]:
        data = _request(f"{self.BASE}/models", headers=self._headers())
        return [
            item["id"]
            for item in data.get("data", [])
            if item.get("id") and not any(bad in item["id"].lower() for bad in self.SKIP)
        ]

    @classmethod
    def _rank(cls, name: str) -> tuple:
        low = name.lower()
        score = 0.0
        for index, token in enumerate(cls.PREFER):
            if token in low:
                score = len(cls.PREFER) - index
                break
        if "70b" in low or "120b" in low or "maverick" in low:
            score -= 2  # bigger models have the smallest free daily quotas
        return (score, _version_score(low))

    def resolve_model(self, refresh: bool = False) -> str:
        if self.model:
            return self.model
        if not refresh:
            cached = CACHE.get(self.name)
            if cached:
                return cached
        candidates = self._candidates()
        if not candidates:
            raise LLMError("Groq returned no usable chat models for this key.")
        best = max(candidates, key=self._rank)
        CACHE.set(self.name, best)
        return best

    def complete(self, system: str, user: str, _retried: bool = False, cap: bool = True) -> Answer:
        model = self.resolve_model(refresh=_retried)
        payload = {
            "model": model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if cap:
            payload["max_completion_tokens"] = 2048
        try:
            data = _request(
                f"{self.BASE}/chat/completions",
                method="POST",
                payload=payload,
                headers=self._headers(),
            )
        except LLMError as exc:
            message = str(exc).lower()
            if cap and "max_completion_tokens" in message:
                return self.complete(system, user, _retried=_retried, cap=False)
            if not _retried and ("404" in message or "decommissioned" in message):
                if not config.GROQ_MODEL:
                    CACHE.drop(self.name)
                    return self.complete(system, user, _retried=True)
            raise

        choices = data.get("choices") or []
        text = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
        if not text:
            raise LLMError("Groq returned an empty answer.")
        return Answer(text=text, provider=self.name, model=model)


class OllamaProvider:
    """A model running on your own machine. Free forever, no key, works offline."""

    name = "ollama"

    def __init__(self) -> None:
        self.host = config.OLLAMA_HOST.rstrip("/")
        self.model = config.OLLAMA_MODEL

    def _tags(self) -> list[str]:
        data = _request(f"{self.host}/api/tags", timeout=2.0)
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    def available(self) -> bool:
        try:
            return bool(self._tags())
        except LLMError:
            return False

    def resolve_model(self) -> str:
        tags = self._tags()
        for tag in tags:
            if tag == self.model or tag.startswith(f"{self.model}:"):
                return tag
        return tags[0]

    def complete(self, system: str, user: str) -> Answer:
        model = self.resolve_model()
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0.0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = _request(f"{self.host}/api/chat", method="POST", payload=payload)
        text = ((data.get("message") or {}).get("content") or "").strip()
        if not text:
            raise LLMError("Ollama returned an empty answer.")
        return Answer(text=text, provider=self.name, model=model)


class ExtractiveProvider:
    """No LLM. Quotes the sentences from the retrieved text that best match the
    question. Always available, always free, never invents anything."""

    name = "extractive"
    model = "none"

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str) -> Answer:  # noqa: ARG002
        question, context = _split_prompt(user)
        return Answer(text=extract_answer(question, context), provider=self.name, model="none")


def _split_prompt(user: str) -> tuple[str, str]:
    question, _, context = user.partition("\n\nContext:\n")
    return question.removeprefix("Question: ").strip(), context.strip()


_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def extract_answer(question: str, context: str, max_sentences: int = 3) -> str:
    """Rank context sentences against the question and quote the best ones."""
    sentences = [s.strip() for s in _SENTENCE.split(context) if len(s.strip()) > 30]
    if not sentences:
        return context[:600].strip() or "The documentation does not contain an answer."

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        matrix = vectorizer.fit_transform(sentences)
        scores = (matrix @ vectorizer.transform([question]).T).toarray().ravel()
        ranked = sorted(range(len(sentences)), key=lambda i: -scores[i])
        best = scores[ranked[0]] if ranked else 0.0
        # keep only sentences close to the best match, so weak chunks add no noise
        picked = [
            i for i in ranked[:max_sentences] if scores[i] > 0 and scores[i] >= 0.45 * best
        ]
    except Exception:
        picked = list(range(min(max_sentences, len(sentences))))

    if not picked:
        return (
            "The retrieved documentation does not appear to answer that question. "
            "Try rephrasing it with words that appear in the document."
        )

    picked.sort()  # keep the document's own order
    quoted = " ".join(sentences[i] for i in picked)
    return (
        f"{quoted}\n\n(No language model is configured, so this is quoted straight "
        "from the document. Add a free GOOGLE_API_KEY or GROQ_API_KEY to .env for "
        "a written answer.)"
    )


# ---------------------------------------------------------------------- factory


PROVIDERS = {
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "groq": GroqProvider,
    "ollama": OllamaProvider,
    "extractive": ExtractiveProvider,
    "none": ExtractiveProvider,
}


def get_provider(kind: str | None = None):
    """Return the provider to use. Never raises: falls back to extractive."""
    kind = (kind or config.LLM_PROVIDER).lower()

    if kind != "auto":
        provider_cls = PROVIDERS.get(kind)
        if provider_cls is None:
            raise LLMError(
                f"Unknown RAG_LLM={kind!r}. Use one of: "
                "auto, gemini, groq, ollama, extractive."
            )
        return provider_cls()

    for provider_cls in (GeminiProvider, GroqProvider, OllamaProvider):
        try:
            provider = provider_cls()
            if provider.available():
                return provider
        except Exception:
            continue
    return ExtractiveProvider()
