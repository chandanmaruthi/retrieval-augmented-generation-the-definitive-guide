"""
Postgres and pgvector.

Everything here goes through psycopg and raw SQL. No ORM, because in the
examples that need a database the SQL *is* the lesson -- the hybrid fusion
query, the row-level-security policies and the recursive graph walk are the
content, and a client library that hid them would defeat the purpose.

The target is the Postgres in docker-compose.yml. Any Postgres 15+ with
pgvector installed will do; the only thing that changes is DATABASE_URL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

import numpy as np

from . import config
from .corpus import Chunk
from .store import Hit

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


class DatabaseUnavailable(RuntimeError):
    """DATABASE_URL is unset, unreachable, or the schema is missing.

    Carries a specific remedy, because these three failures need three
    different responses and a generic "could not connect" helps nobody.
    """

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.remedy = remedy


def available() -> bool:
    if not config.DATABASE_URL:
        return False
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return True


def why_unavailable() -> str:
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return "psycopg is not installed (pip install -r requirements-db.txt)"
    if not config.DATABASE_URL:
        return "DATABASE_URL is not set"
    return ""


def connect(autocommit: bool = True, application_name: str = "ragkit"):
    """Open a connection to whatever DATABASE_URL points at."""
    if not config.DATABASE_URL:
        raise DatabaseUnavailable(
            "DATABASE_URL is not set",
            "docker compose up -d && python scripts/init_db.py",
        )

    import psycopg

    url = config.DATABASE_URL
    _check_url(url)

    try:
        return psycopg.connect(
            url,
            autocommit=autocommit,
            connect_timeout=10,
            application_name=application_name,
        )
    except psycopg.OperationalError as error:
        raise DatabaseUnavailable(str(error).strip(), _diagnose(str(error))) from error


def _check_url(url: str) -> None:
    """Reject a connection string that is malformed in a familiar way.

    The specific trap: a username or password containing an unescaped '@'.
    A URL has exactly one userinfo section, delimited by the LAST '@', so

        postgresql://me@gmail.com:secret@db.example.com:5432/postgres

    parses with host 'gmail.com' and a password of 'secret@db.example.com'.
    psycopg then reports that it cannot resolve 'gmail.com', which sends you
    looking at DNS instead of at the credential you pasted.

    Checking here turns ten minutes of confusion into one line of output.
    """
    parsed = urlsplit(url)
    userinfo = url.split("//", 1)[-1].rsplit("@", 1)[0] if "@" in url else ""

    if userinfo.count("@") >= 1:
        # Different parsers split this differently -- psycopg and urlsplit do
        # not agree on which '@' delimits the host -- so do not claim what the
        # host "really" is. The ambiguity itself is the bug.
        raise DatabaseUnavailable(
            "DATABASE_URL is ambiguous: the username or password contains an "
            "unescaped '@', so the host cannot be determined reliably.",
            "Percent-encode it: @ becomes %40, & becomes %26, # becomes %23.\n"
            "An unencoded # is the worst of these -- it starts a URL fragment,\n"
            "so everything after it is silently discarded.",
        )

    if not parsed.hostname:
        raise DatabaseUnavailable(
            "DATABASE_URL has no host.",
            "Expected: postgresql://USER:PASSWORD@HOST:5432/DBNAME",
        )


def _diagnose(message: str) -> str:
    """Turn a psycopg connection error into something actionable."""
    lowered = message.lower()
    if "connection refused" in lowered:
        return (
            "Nothing is listening on that port.\n"
            "  docker compose up -d && docker compose ps"
        )
    if "password authentication failed" in lowered or "auth" in lowered:
        return "Credentials rejected. Check the user and password in DATABASE_URL."
    if "does not exist" in lowered:
        return "The database named in DATABASE_URL does not exist."
    return "Check DATABASE_URL in your .env"


def check_schema(conn) -> None:
    """Fail clearly when connected to a database that was never initialised."""
    row = conn.execute("SELECT to_regclass('public.chunks')").fetchone()
    if row is None or row[0] is None:
        raise DatabaseUnavailable(
            "Connected, but the schema is missing",
            "python scripts/init_db.py",
        )


def server_info(conn) -> dict[str, Any]:
    version = conn.execute("SHOW server_version").fetchone()[0]
    vector = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    host = urlsplit(config.DATABASE_URL or "").hostname or ""
    return {
        "postgres": version,
        "pgvector": vector[0] if vector else None,
        "host": host,
    }


def apply_sql_files(conn, names: Sequence[str]) -> list[str]:
    """Run the numbered files in sql/ in order."""
    applied = []
    for name in names:
        path = SQL_DIR / name
        if not path.exists():
            continue
        conn.execute(path.read_text(encoding="utf-8"))
        applied.append(name)
    return applied


def query(name: str) -> str:
    """Load a named query from sql/queries/ verbatim.

    Kept in .sql files rather than Python strings so they can be read, run in
    psql, and explained with EXPLAIN ANALYZE without being reassembled.
    """
    return (SQL_DIR / "queries" / f"{name}.sql").read_text(encoding="utf-8")


def vec_literal(vector: np.ndarray) -> str:
    """Format a numpy vector as a pgvector literal.

    str(numpy_array) is not valid input: it inserts newlines, elides long
    arrays with '...', and uses spaces rather than commas. That elision is
    especially nasty -- it produces a *syntactically plausible* string that
    silently truncates a 1536-dimension vector.
    """
    return "[" + ",".join(f"{float(x):.7g}" for x in np.asarray(vector).ravel()) + "]"


def set_principal(conn, groups: Sequence[str]) -> None:
    """Set the caller's ACL groups for row-level security.

    Note the shape. This does NOT work::

        conn.execute("SET app.groups = %s", [groups])

    SET does not accept bind parameters -- it is a utility statement, not a
    query, and psycopg cannot substitute into it. set_config() is a regular
    function call and takes parameters normally.
    """
    conn.execute("SELECT set_config('app.groups', %s, false)", [",".join(groups)])


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PgVectorStore:
    """The same interface as NumpyStore, backed by Postgres.

    Everything NumpyStore can do, plus the things it cannot: lexical search,
    hybrid fusion, ACL enforcement inside the index, and graph traversal.
    """

    def __init__(self, conn, dim: int = 384, model: str = "all-MiniLM-L6-v2") -> None:
        self.conn = conn
        self.dim = dim
        self.model = model
        self.table = f"embeddings_{dim}"

    # --- writing ---------------------------------------------------------

    def upsert_document(self, slug: str, title: str, **fields) -> int:
        row = self.conn.execute(
            """
            INSERT INTO documents (slug, title, author, source, sha256, n_words)
            VALUES (%(slug)s, %(title)s, %(author)s, %(source)s, %(sha256)s, %(n_words)s)
            ON CONFLICT (slug) DO UPDATE
                SET title = EXCLUDED.title, n_words = EXCLUDED.n_words
            RETURNING id
            """,
            {
                "slug": slug,
                "title": title,
                "author": fields.get("author"),
                "source": fields.get("source"),
                "sha256": fields.get("sha256"),
                "n_words": fields.get("n_words"),
            },
        ).fetchone()
        return int(row[0])

    def clear(self, strategy: str) -> int:
        row = self.conn.execute(
            "DELETE FROM chunks WHERE strategy = %s RETURNING 1", [strategy]
        ).fetchall()
        return len(row)

    def add(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        strategy: str,
        document_id: int,
    ) -> int:
        """Insert chunks and their vectors.

        Uses COPY for the vectors. executemany over a few thousand rows of 384
        floats each is slow enough to notice and 1536 floats is slow enough to
        be annoying; COPY streams them in one pass.
        """
        import hashlib

        chunk_ids: list[int] = []
        with self.conn.cursor() as cur:
            for ordinal, chunk in enumerate(chunks):
                content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                cur.execute(
                    """
                    INSERT INTO chunks
                        (document_id, strategy, ordinal, body, embed_body,
                         parent_body, story, section, acl, content_hash, meta)
                    VALUES
                        (%(document_id)s, %(strategy)s, %(ordinal)s, %(body)s,
                         %(embed_body)s, %(parent_body)s, %(story)s, %(section)s,
                         %(acl)s, %(content_hash)s, %(meta)s)
                    ON CONFLICT (document_id, strategy, ordinal) DO UPDATE
                        SET body = EXCLUDED.body,
                            embed_body = EXCLUDED.embed_body,
                            parent_body = EXCLUDED.parent_body,
                            updated_at = now()
                    RETURNING id
                    """,
                    {
                        "document_id": document_id,
                        "strategy": strategy,
                        "ordinal": ordinal,
                        "body": chunk.text,
                        "embed_body": chunk.retrieval_text,
                        "parent_body": chunk.parent_text,
                        "story": chunk.story,
                        "section": chunk.section,
                        "acl": chunk.meta.get("acl", ["public"]),
                        "content_hash": content_hash,
                        "meta": _json(chunk.meta),
                    },
                )
                chunk_ids.append(int(cur.fetchone()[0]))

            # Replace this strategy's vectors for the affected chunks, then
            # COPY the new ones in.
            cur.execute(
                f"DELETE FROM {self.table} WHERE chunk_id = ANY(%s) AND model = %s",
                [chunk_ids, self.model],
            )
            with cur.copy(
                f"COPY {self.table} (chunk_id, model, strategy, vec) FROM STDIN"
            ) as copy:
                for chunk_id, vector in zip(chunk_ids, vectors):
                    copy.write_row([chunk_id, self.model, strategy, vec_literal(vector)])

        return len(chunk_ids)

    # --- reading ---------------------------------------------------------

    def count(self, strategy: str | None = None) -> int:
        if strategy:
            row = self.conn.execute(
                "SELECT count(*) FROM chunks WHERE strategy = %s", [strategy]
            ).fetchone()
        else:
            row = self.conn.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0])

    def strategies(self) -> list[tuple[str, int]]:
        return [
            (name, int(count))
            for name, count in self.conn.execute(
                "SELECT strategy, count(*) FROM chunks GROUP BY strategy ORDER BY 1"
            ).fetchall()
        ]

    def search(
        self, query_vector: np.ndarray, k: int = 5, strategy: str | None = None
    ) -> list[Hit]:
        """Dense search. Note the bare `<=>` in the ORDER BY -- see the SQL."""
        rows = self.conn.execute(
            f"""
            SELECT c.id, c.body, c.parent_body, c.story, c.section, c.strategy,
                   e.vec <=> %(qvec)s::vector AS distance
            FROM   {self.table} e
            JOIN   chunks c ON c.id = e.chunk_id
            WHERE  e.model = %(model)s
              AND  (%(strategy)s::text IS NULL OR e.strategy = %(strategy)s)
            ORDER  BY e.vec <=> %(qvec)s::vector
            LIMIT  %(k)s
            """,
            {
                "qvec": vec_literal(query_vector),
                "model": self.model,
                "strategy": strategy,
                "k": k,
            },
        ).fetchall()

        return [
            _to_hit(row[:6], score=1.0 - float(row[6]), rank=rank,
                    components={"vector": 1.0 - float(row[6])})
            for rank, row in enumerate(rows, start=1)
        ]

    def search_lexical(
        self,
        query_text: str,
        k: int = 5,
        strategy: str | None = None,
        match: str = "all",
    ) -> list[Hit]:
        """Full-text search.

        Called "lexical" rather than "BM25" throughout this repo, because
        Postgres ts_rank is not BM25: there is no IDF term and ts_rank_cd is a
        cover-density measure. Example 18 implements real BM25 alongside it
        and shows where the two rankings disagree.

        ``match`` chooses how the query terms are combined, and the default is
        a trap worth knowing about:

        * ``"all"``  -- websearch_to_tsquery, which joins bare terms with AND.
          A chunk must contain *every* word. Precise, and a recall cliff: one
          uncommon word in a natural-language question and the arm returns
          nothing at all.
        * ``"any"``  -- terms joined with OR. Far higher recall, which is what
          a candidate-generation stage actually wants. Used by the lexical
          agent in example 28 for exactly that reason.
        """
        if match == "any":
            # plainto_tsquery also ANDs, so the OR query is built explicitly.
            # to_tsquery needs pre-normalised input, hence the strip.
            terms = [
                t.strip(".,;:!?\"'()[]{}").lower()
                for t in query_text.split()
                if len(t.strip(".,;:!?\"'()[]{}")) > 2
            ]
            if not terms:
                return []
            tsquery_sql = "to_tsquery('english', %(q)s)"
            query_param = " | ".join(terms)
        else:
            tsquery_sql = "websearch_to_tsquery('english', %(q)s)"
            query_param = query_text

        rows = self.conn.execute(
            f"""
            SELECT c.id, c.body, c.parent_body, c.story, c.section, c.strategy,
                   ts_rank_cd(c.tsv, {tsquery_sql}, 32) AS score
            FROM   chunks c
            WHERE  (%(strategy)s::text IS NULL OR c.strategy = %(strategy)s)
              AND  c.tsv @@ {tsquery_sql}
            ORDER  BY score DESC
            LIMIT  %(k)s
            """,
            {"q": query_param, "strategy": strategy, "k": k},
        ).fetchall()

        return [
            _to_hit(row[:6], score=float(row[6]), rank=rank,
                    components={"lexical": float(row[6])})
            for rank, row in enumerate(rows, start=1)
        ]

    def search_hybrid(
        self,
        query_text: str,
        query_vector: np.ndarray,
        k: int = 5,
        fanout: int = 50,
        mode: str = "rrf",
        w_vec: float = 0.5,
        w_txt: float = 0.5,
        rrf_k: int = 60,
        strategy: str | None = None,
    ) -> list[Hit]:
        """Run one of the two fusion queries from sql/queries/."""
        params = {
            "qvec": vec_literal(query_vector),
            "qtext": query_text,
            "strategy": strategy,
            "model": self.model,
            "fanout": fanout,
            "k": k,
            "w_vec": w_vec,
            "w_txt": w_txt,
        }
        if mode == "rrf":
            params["rrf_k"] = rrf_k

        rows = self.conn.execute(
            query("hybrid_rrf" if mode == "rrf" else "hybrid_linear"), params
        ).fetchall()

        hits = []
        for rank, row in enumerate(rows, start=1):
            mapping = dict(zip([d.name for d in self.conn.cursor().description or []], row))
            hits.append(
                _to_hit(
                    (row[0], row[3], row[4], row[1], row[2], strategy or ""),
                    score=float(row[-1]),
                    rank=rank,
                    components=_components(row, mode),
                )
            )
        return hits


def _components(row: tuple, mode: str) -> dict[str, float]:
    """Pull the per-arm diagnostics out of a fusion result row."""
    if mode == "rrf":
        # chunk_id, story, section, body, parent_body, vec_score, txt_score,
        # vec_rank, txt_rank, score
        return {
            "vector": float(row[5] or 0.0),
            "lexical": float(row[6] or 0.0),
            "vector_rank": float(row[7]) if row[7] is not None else 0.0,
            "lexical_rank": float(row[8]) if row[8] is not None else 0.0,
        }
    # linear adds the normalised columns
    return {
        "vector": float(row[5] or 0.0),
        "lexical": float(row[6] or 0.0),
        "vector_norm": float(row[7] or 0.0),
        "lexical_norm": float(row[8] or 0.0),
        "vector_rank": float(row[9]) if row[9] is not None else 0.0,
        "lexical_rank": float(row[10]) if row[10] is not None else 0.0,
    }


def _to_hit(row: tuple, score: float, rank: int, components: dict) -> Hit:
    chunk_id, body, parent_body, story, section, strategy = row
    return Hit(
        chunk=Chunk(
            text=body,
            strategy=strategy or "",
            story=story or "",
            index=int(chunk_id),
            section=section,
            parent_text=parent_body,
        ),
        score=score,
        rank=rank,
        components=components,
    )


def _json(value) -> str:
    import json

    return json.dumps(value, default=str)
