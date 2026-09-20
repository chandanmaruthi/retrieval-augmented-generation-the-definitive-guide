"""
Vector stores.

Two implementations behind one interface:

    NumpyStore      brute-force cosine over an in-memory matrix
    PgVectorStore   Postgres + pgvector (see ragkit/pg.py)

The NumPy one is not a toy. At this corpus's scale — roughly 900 to 5,000
chunks depending on strategy — an exact scan over a 5,000x384 matrix is a
single BLAS call taking well under a millisecond, and it returns perfect
recall. An approximate index would be *slower* and *less accurate* here.
Example 16 measures exactly that and says so, because a repo that reaches for
HNSW at 900 rows is teaching cargo cult.

Postgres earns its place later, when the lesson stops being "find the nearest
vector" and becomes "fuse two retrievers", "filter by ACL inside the index",
or "traverse a graph" — things a matrix cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .corpus import Chunk


@dataclass
class Hit:
    """One retrieved chunk plus why it was retrieved."""

    chunk: Chunk
    score: float
    rank: int
    # Per-retriever contributions, so hybrid examples can show their working:
    # {"vector": 0.71, "lexical": 0.04, "vector_rank": 3, "lexical_rank": 11}
    components: dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def location(self) -> str:
        return self.chunk.location


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray, strategy: str) -> int: ...
    def search(self, query_vector: np.ndarray, k: int = 5, strategy: str | None = None) -> list[Hit]: ...
    def count(self, strategy: str | None = None) -> int: ...


class NumpyStore:
    """Exact nearest-neighbour search over a normalised matrix.

    Because every embedder in this repo returns unit-length vectors, cosine
    similarity reduces to a dot product, and the whole search is::

        scores = matrix @ query
        top    = np.argpartition(-scores, k)[:k]

    That is the entire algorithm. Seeing it in four lines is the point; once
    it is hidden behind a client library, the fact that "vector search" is a
    matrix multiply followed by a partial sort stops being obvious.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray, strategy: str | None = None) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} chunks but {len(vectors)} vectors -- these must line up"
            )
        self._chunks.extend(chunks)
        self._vectors.extend(np.asarray(v, dtype=np.float32) for v in vectors)
        self._matrix = None  # invalidate; rebuilt lazily on the next search
        return len(chunks)

    def _stack(self) -> np.ndarray:
        if self._matrix is None:
            if not self._vectors:
                raise RuntimeError("Nothing has been added to this store yet.")
            self._matrix = np.vstack(self._vectors)
        return self._matrix

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        strategy: str | None = None,
    ) -> list[Hit]:
        matrix = self._stack()
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)

        # Vectors are unit length, so this dot product *is* cosine similarity.
        scores = matrix @ query

        if strategy is not None:
            # Mask rather than slice, so the indices still address self._chunks.
            mask = np.array([c.strategy == strategy for c in self._chunks])
            scores = np.where(mask, scores, -np.inf)

        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []

        # argpartition finds the top k without sorting all n — O(n) rather than
        # O(n log n). We then sort just those k.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            Hit(
                chunk=self._chunks[int(index)],
                score=float(scores[int(index)]),
                rank=rank + 1,
                components={"vector": float(scores[int(index)])},
            )
            for rank, index in enumerate(top)
        ]

    def count(self, strategy: str | None = None) -> int:
        if strategy is None:
            return len(self._chunks)
        return sum(1 for c in self._chunks if c.strategy == strategy)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks
