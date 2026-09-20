"""
Embedding providers.

Three of them, in descending order of quality:

    local   sentence-transformers all-MiniLM-L6-v2   384-dim, free, offline
    openai  text-embedding-3-small                  1536-dim, costs money
    hash    deterministic token hashing              384-dim, no dependencies

The hash embedder exists so that the *database* lessons do not depend on a
2GB PyTorch install. It produces the same 384 dimensions as MiniLM, so the
schema, the HNSW index, the hybrid-fusion SQL and the row-level-security
policies all exercise identically under it. What it does not produce is
meaning: two paraphrases of the same sentence land nowhere near each other.

That trade is fine for "does this SQL work" and useless for "is this chunking
strategy better". So every run prints which embedder it used, loudly, and the
examples whose *conclusions* depend on real semantics refuse to draw them from
hash vectors.

All embedders return L2-normalised float32. That matters downstream: once
vectors are unit length, cosine similarity is just a dot product, and
pgvector's inner-product operator becomes interchangeable with its cosine one.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import struct
import threading
import warnings
from typing import Protocol, Sequence

import numpy as np

from . import config

CACHE_PATH = config.CACHE_DIR / "embeddings.db"


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    name: str
    dim: int
    semantic: bool  # False for the hash fallback

    def encode(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving zero rows alone."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------------------------
# Hash embedder — no dependencies
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Signed token hashing, sometimes called the hashing trick.

    Each token is hashed to a dimension and a sign, and we sum those into a
    bag-of-words vector. Two documents sharing vocabulary end up close; two
    documents sharing *meaning* do not, because nothing here knows that
    "adder" and "snake" are related.

    This is a real technique with real history (it predates dense embeddings
    and is still used for very high-cardinality features), which is why it is
    worth seeing rather than being a fake stub.
    """

    name = "hash-384"
    dim = 384
    semantic = False

    def encode(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                token = token.strip(".,;:!?\"'()[]{}“”‘’")
                if not token:
                    continue
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = struct.unpack("<Q", digest)[0]
                # Low bits pick the dimension, one more bit picks the sign, so
                # unrelated tokens cancel instead of always accumulating.
                index = value % self.dim
                sign = 1.0 if (value >> 40) & 1 else -1.0
                out[row, index] += sign
        return _normalise(out)


# ---------------------------------------------------------------------------
# Local embedder — sentence-transformers
# ---------------------------------------------------------------------------


class LocalEmbedder:
    """all-MiniLM-L6-v2 run locally on CPU.

    384 dimensions, about 90MB of weights, no network after the first load and
    no per-call cost. This is the default and the one the repo's published
    numbers were produced with.
    """

    name = "all-MiniLM-L6-v2"
    dim = 384
    semantic = True

    def __init__(self) -> None:
        # Keep the libraries' console noise out of example output. These are
        # advisory notices about tokenizer forking and Hub rate limits;
        # neither is actionable here, and both otherwise land in the middle of
        # a results table.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        # Imported lazily so that `import ragkit.embed` stays cheap for the
        # examples that never embed anything.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sentence_transformers import SentenceTransformer

            # Silenced *after* the import: huggingface_hub sets an explicit
            # level on its own child loggers when it loads, so quieting the
            # parent beforehand has no effect on them.
            for name in (
                "huggingface_hub",
                "huggingface_hub.utils._http",
                "sentence_transformers",
                "transformers",
            ):
                logging.getLogger(name).setLevel(logging.ERROR)

            # The weight-loading progress bar is a tqdm instance owned by
            # transformers, not a logger, so the level changes above do not
            # reach it.
            try:
                from transformers.utils import logging as hf_logging

                hf_logging.disable_progress_bar()
            except Exception:  # noqa: BLE001 - cosmetic only, never fatal
                pass

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # A single torch model is not safe to call from several threads at
        # once. On Apple's MPS backend concurrent calls do not raise -- they
        # segfault the interpreter, which is a spectacularly unhelpful way to
        # find out. Example 28 runs retrieval agents in a thread pool, so the
        # lock is what keeps that from being a crash report.
        #
        # It costs nothing in practice: embedding one query is sub-millisecond,
        # and the parallelism worth having there is in the database round
        # trips, not in the encoder.
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        with self._lock:
            vectors = self._model.encode(
                list(texts),
                batch_size=64,
                show_progress_bar=show_progress and len(texts) > 256,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        return _normalise(vectors)


# ---------------------------------------------------------------------------
# OpenAI embedder
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """text-embedding-3-small — 1536 dimensions, billed per token.

    Note the dimension change. It is not a detail: a pgvector column is fixed
    width, so switching to this provider means writing into a different table
    and building a different index. The schema handles that explicitly rather
    than hoping nobody notices.
    """

    name = "text-embedding-3-small"
    dim = 1536
    semantic = True

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def encode(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        vectors: list[list[float]] = []
        # The endpoint caps how much it will take per call, and the cap is in
        # tokens rather than items, so we keep batches modest.
        for start in range(0, len(texts), 128):
            batch = [t.replace("\n", " ") for t in texts[start : start + 128]]
            response = self._client.embeddings.create(
                model=self.name, input=batch
            )
            vectors.extend(item.embedding for item in response.data)
            if show_progress and len(texts) > 256:
                print(f"    embedded {min(start + 128, len(texts))}/{len(texts)}")
        return _normalise(np.array(vectors, dtype=np.float32))


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Disk cache so thirteen chunking strategies do not re-embed one book.

    Keyed on ``(model, sha256(text))``. The model name is part of the key for
    a reason that bites people: cache on the text alone and a run that
    switches provider silently reads back the previous model's vectors, with
    the wrong dimensionality or — worse — the right dimensionality and the
    wrong meaning.
    """

    def __init__(self, path=CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False plus an explicit lock, because example 28
        # runs retrieval agents in a thread pool and they all embed. SQLite
        # refuses cross-thread use of a connection by default, and the error
        # surfaces inside a worker where it is easy to mistake for an agent
        # bug rather than a cache bug.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "  model TEXT NOT NULL, key TEXT NOT NULL, dim INTEGER NOT NULL,"
            "  vec BLOB NOT NULL, PRIMARY KEY (model, key))"
        )
        self._db.commit()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: Sequence[str]) -> dict[int, np.ndarray]:
        found: dict[int, np.ndarray] = {}
        keys = [self._key(t) for t in texts]
        # Chunked IN clauses: SQLite's default variable limit is 999.
        for start in range(0, len(keys), 500):
            window = keys[start : start + 500]
            placeholders = ",".join("?" * len(window))
            with self._lock:
                rows = self._db.execute(
                    f"SELECT key, dim, vec FROM vectors WHERE model = ? AND key IN ({placeholders})",
                    [model, *window],
                ).fetchall()
            by_key = {k: (d, v) for k, d, v in rows}
            for offset, key in enumerate(window):
                if key in by_key:
                    dim, blob = by_key[key]
                    found[start + offset] = np.frombuffer(blob, dtype=np.float32, count=dim)
        return found

    def put_many(self, model: str, texts: Sequence[str], vectors: np.ndarray) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO vectors (model, key, dim, vec) VALUES (?, ?, ?, ?)",
                [
                    (model, self._key(text), int(vectors.shape[1]), vectors[i].tobytes())
                    for i, text in enumerate(texts)
                ],
            )
            self._db.commit()


class CachedEmbedder:
    """Wraps an embedder so repeated text is embedded once, ever."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache = EmbeddingCache()
        self.name = inner.name
        self.dim = inner.dim
        self.semantic = inner.semantic

    def encode(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        hits = self._cache.get_many(self.name, texts)
        missing = [i for i in range(len(texts)) if i not in hits]

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for index, vector in hits.items():
            out[index] = vector

        if missing:
            if show_progress and len(missing) > 256:
                print(f"    embedding {len(missing):,} new / {len(texts):,} total ({self.name})")
            fresh = self._inner.encode([texts[i] for i in missing], show_progress=show_progress)
            self._cache.put_many(self.name, [texts[i] for i in missing], fresh)
            for slot, index in enumerate(missing):
                out[index] = fresh[slot]

        return out


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def get_embedder(provider: str | None = None, cached: bool = True) -> Embedder:
    """Build the embedder named by ``provider`` or EMBEDDING_PROVIDER.

    ``"auto"`` prefers the local model and falls back to hashing, so a reader
    who has installed nothing optional still gets a working pipeline.
    """
    provider = (provider or config.EMBEDDING_PROVIDER or "auto").lower()

    if provider == "openai":
        if not config.has_openai():
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        embedder: Embedder = OpenAIEmbedder()
    elif provider == "hash":
        embedder = HashEmbedder()
    elif provider == "local":
        embedder = LocalEmbedder()  # let the ImportError surface; the user asked for this
    elif provider == "auto":
        try:
            embedder = LocalEmbedder()
        except ImportError:
            embedder = HashEmbedder()
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER {provider!r}")

    return CachedEmbedder(embedder) if cached else embedder


def describe(embedder: Embedder) -> str:
    """One line naming the embedder, so no run is ambiguous about its quality."""
    if not embedder.semantic:
        return (
            f"embedder: {embedder.name} ({embedder.dim}d) "
            "-- FALLBACK, not semantic; install -r requirements-local.txt for real quality"
        )
    return f"embedder: {embedder.name} ({embedder.dim}d)"
