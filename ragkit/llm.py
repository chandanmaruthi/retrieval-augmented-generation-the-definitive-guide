"""
The optional generative step.

Most of this repository runs without an LLM. Retrieval, chunking, indexing,
fusion, graph traversal and the retrieval metrics are all deterministic local
computation. An LLM is needed for exactly two things:

1. Writing the final answer from retrieved context.
2. The strategies that *use* a model during indexing -- generating questions
   for question-anchored chunking, summarising neighbours, judging
   faithfulness.

Everything here caches to disk. These calls cost money and the examples are
meant to be run repeatedly while reading, so a second run of the same script
should be free and instant.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Any

from . import config

CACHE_PATH = config.CACHE_DIR / "llm.db"


class JSONTruncated(RuntimeError):
    """A JSON response was cut off by max_tokens before it was complete.

    Its own class because the fix is specific and the symptom is misleading:
    a truncated response fails to parse, which looks like the model produced
    bad JSON when in fact it produced good JSON and was interrupted.

    Callers that extract over many passages should catch this and skip the
    passage rather than abort the run -- one long passage should not end an
    extraction over a thousand of them.
    """

    def __init__(self, message: str, partial: str = "") -> None:
        super().__init__(message)
        self.partial = partial


class LLMUnavailable(RuntimeError):
    """Raised when a generative step is required but no provider is configured.

    Carries the remediation text so callers can print something useful rather
    than a stack trace.
    """

    def __init__(self, purpose: str) -> None:
        super().__init__(f"No LLM configured. Needed for: {purpose}")
        self.purpose = purpose


# Set RAGKIT_STUB_LLM=1 to make every call return canned text instead of
# reaching the API. This exists so `make smoke` can execute the real code
# paths of the LLM-dependent examples in CI, where there is no key.
#
# It is a testing facility, not a way to read the examples. The canned replies
# are deterministic filler, so the *output* is meaningless even though the
# *mechanics* are exercised faithfully. A reader without a key is better served
# by each script's no-key explanation, which shows the real prompt.
STUB = config.get("RAGKIT_STUB_LLM") == "1"


def available() -> bool:
    """True when a completion call would actually succeed."""
    if STUB:
        return True
    if not config.has_openai():
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def _stub_response(prompt: str, as_json: bool) -> str:
    """Deterministic filler, shaped like what the real call would return.

    The shape has to be right or the stub tests nothing: a caller that parses
    ``anchors`` and receives only ``questions`` will exercise its error path
    and report success. So when the prompt contains a numbered sentence list,
    we read the highest number out of it and return ranges that actually tile
    it, the way a compliant model would.
    """
    if not as_json:
        return "[stub] A one-sentence summary of the passage supplied in the prompt."

    payload: dict[str, Any] = {
        "questions": [
            "[stub] What happened in this passage?",
            "[stub] Who was present?",
            "[stub] Where did it take place?",
        ],
        "summary": "[stub] A summary of the passage.",
        "entities": [{"name": "Sherlock Holmes", "kind": "person"}],
        "relations": [
            {"subject": "Sherlock Holmes", "predicate": "lives_with", "object": "John Watson"}
        ],
    }

    numbered = re.findall(r"^\s*(\d+)\.\s+\S", prompt, flags=re.MULTILINE)
    if numbered:
        total = max(int(n) for n in numbered)
        step = max(1, total // 3)
        anchors = []
        start = 1
        while start <= total:
            end = min(total, start + step - 1)
            anchors.append(
                {"question": f"[stub] What happens in sentences {start}-{end}?",
                 "start": start, "end": end}
            )
            start = end + 1
        payload["anchors"] = anchors

    return json.dumps(payload)


def why_unavailable() -> str:
    """A specific reason, so the reader knows which of the two things to fix."""
    try:
        import openai  # noqa: F401
    except ImportError:
        return "the openai package is not installed (pip install -r requirements-openai.txt)"
    if not config.has_openai():
        return "OPENAI_API_KEY is not set (copy .env.example to .env and add it)"
    return ""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class _Cache:
    def __init__(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(CACHE_PATH)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS completions ("
            "  key TEXT PRIMARY KEY, model TEXT, response TEXT)"
        )
        self._db.commit()

    @staticmethod
    def key(model: str, system: str | None, prompt: str, temperature: float) -> str:
        # Temperature is part of the key: a cached temperature-0 answer must
        # not be served to a caller that asked for sampling.
        payload = json.dumps([model, system, prompt, temperature], sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        row = self._db.execute(
            "SELECT response FROM completions WHERE key = ?", [key]
        ).fetchone()
        return row[0] if row else None

    def put(self, key: str, model: str, response: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO completions (key, model, response) VALUES (?, ?, ?)",
            [key, model, response],
        )
        self._db.commit()


_cache: _Cache | None = None


def _get_cache() -> _Cache:
    global _cache
    if _cache is None:
        _cache = _Cache()
    return _cache


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def complete(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    purpose: str = "generation",
    use_cache: bool = True,
) -> str:
    """One completion. Cached on (model, system, prompt, temperature)."""
    if not available():
        raise LLMUnavailable(purpose)
    if STUB:
        return _stub_response(prompt, as_json=False)

    model = model or config.OPENAI_CHAT_MODEL
    cache = _get_cache()
    key = cache.key(model, system, prompt, temperature)

    if use_cache:
        hit = cache.get(key)
        if hit is not None:
            return hit

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (response.choices[0].message.content or "").strip()

    if use_cache:
        cache.put(key, model, text)
    return text


def complete_json(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    purpose: str = "structured generation",
) -> Any:
    """A completion constrained to valid JSON.

    Used wherever a chunking strategy needs a *list* back -- generated
    questions, extracted entities, neighbour summaries. Free-text parsing of
    "1. ... 2. ..." works until the model decides to add a preamble, which it
    eventually will.
    """
    if not available():
        raise LLMUnavailable(purpose)
    if STUB:
        return json.loads(_stub_response(prompt, as_json=True))

    model = model or config.OPENAI_CHAT_MODEL
    cache = _get_cache()
    key = cache.key(model + ":json", system, prompt, temperature)

    hit = cache.get(key)
    if hit is not None:
        return json.loads(hit)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    choice = response.choices[0]
    text = (choice.message.content or "{}").strip()

    # JSON mode guarantees valid JSON only if the model is allowed to finish.
    # Hit the token ceiling and you get a syntactically broken fragment --
    # the object is cut mid-key and json.loads fails on something that looks
    # like a model defect but is really a budget the caller set too low.
    if choice.finish_reason == "length":
        raise JSONTruncated(
            f"Response hit the {max_tokens}-token limit and the JSON is "
            f"incomplete (purpose: {purpose}). Raise max_tokens, or ask for "
            f"fewer items per call.",
            partial=text,
        )

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Model returned invalid JSON (purpose: {purpose}): {text[:200]}"
        ) from error

    cache.put(key, model, text)
    return parsed


def cached_count() -> int:
    """How many responses are already on disk -- printed by the examples."""
    try:
        return _get_cache()._db.execute(
            "SELECT count(*) FROM completions"
        ).fetchone()[0]
    except sqlite3.Error:
        return 0
