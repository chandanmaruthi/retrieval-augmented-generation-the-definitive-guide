#!/usr/bin/env python3
"""
29. Streaming RAG (incremental updates, freshness, and recency)
Book: Chapter 16, Streaming RAG

WHAT THIS SHOWS: What changes when the corpus is not a fixed book.

Four mechanisms, all of them local computation against Postgres:

    content-hash upserts   re-ingesting unchanged text costs nothing
    a recency prior        score = sim - lambda * age_hours   (the book's formula)
    TTL and watermarks     expiry, and knowing how stale the index is
    deletion propagation   a removed source must leave the index

The recency prior is one of only two explicit formulas in the whole book, so
it is implemented literally and its lambda is swept so you can see what the
parameter actually does.

HOW THIS SCRIPT PROCEEDS
    1. Ingest a feed                     everything is new
    2. Ingest the SAME feed again        everything is unchanged, 0 embed calls
    3. Ingest one edited document        new hash, new vector, old one replaced
    4. Sweep the recency prior's lambda  <-- THE FORMULA, one of two in the book
    5. TTL, freshness watermark, deletion

Step 4 is the interesting one: at lambda = 0 the most relevant document wins;
at lambda = 0.01 it is demoted to last with a negative score.


WHAT CHANGED SINCE EVERY PREVIOUS EXAMPLE
    The corpus stopped being a fixed book. That single change makes four
    things matter that did not before: whether re-ingesting is cheap, whether
    old documents expire, how stale the index is, and whether a deleted
    document really left.


REQUIRES: Postgres
RUN: python examples/29_streaming_rag.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import get_embedder

# Simulated live documents arriving after the book was indexed, as
# (source_id, text, hours_ago).
#
# bulletin-4 is the interesting one: it is the BEST match for the query by
# pure similarity and also the oldest still-live document. Without a recency
# term it wins; with one it loses. That is the whole mechanism, and it needs a
# document that is genuinely both relevant and stale to be visible at all.
INCOMING = [
    ("bulletin-1", "Inspector Lestrade reports a new burglary at a Kensington jeweller.", 0.5),
    ("bulletin-2", "A fresh sighting of the escaped convict was reported near Dartmoor.", 6),
    ("bulletin-3", "Scotland Yard has closed the Kensington jeweller case.", 0.1),
    ("bulletin-4", "A burglary was reported to the police in Whitechapel last season.", 96),
    ("bulletin-5", "An archived notice of a theft, long past its retention window.", 720),
]

QUERY = "recent burglary reported to the police"


def ensure_stream_table(conn) -> None:
    """A small table standing in for a live feed.

    Separate from `chunks` on purpose: streaming sources usually have a
    different lifecycle, a different retention policy and a different owner
    from the document corpus, and mixing them makes TTL a per-row decision.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_items (
            id           bigserial PRIMARY KEY,
            source_id    text NOT NULL UNIQUE,
            body         text NOT NULL,
            content_hash char(64) NOT NULL,
            vec          vector(384),
            observed_at  timestamptz NOT NULL,
            ingested_at  timestamptz NOT NULL DEFAULT now(),
            expires_at   timestamptz,
            deleted      boolean NOT NULL DEFAULT false
        )
        """
    )


def upsert(conn, embedder, source_id: str, body: str, age_hours: float, ttl_hours: float = 168):
    """Insert or update one streamed item.

    Returns one of "inserted", "updated" or "unchanged".

    The content hash is what makes this cheap. Feeds re-send the same records
    constantly, and embedding is the expensive step -- so the hash is checked
    *before* the embedder is called, not after. Getting that order wrong means
    paying full price for a no-op.
    """
    content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

    existing = conn.execute(
        "SELECT content_hash FROM stream_items WHERE source_id = %s", [source_id]
    ).fetchone()

    # ---- THE KEY ORDERING ------------------------------------------
    # The hash is compared BEFORE the embedder is called. Feeds resend
    # unchanged records constantly and embedding is the expensive step,
    # so this ordering is the entire cost saving. Reverse it -- embed,
    # then compare -- and you pay full price for every no-op.
    if existing and existing[0] == content_hash:
        return "unchanged"

    observed = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    vector = embedder.encode([body], show_progress=False)[0]

    conn.execute(
        """
        INSERT INTO stream_items
            (source_id, body, content_hash, vec, observed_at, expires_at)
        VALUES (%(sid)s, %(body)s, %(hash)s, %(vec)s::vector, %(observed)s,
                %(observed)s + %(ttl)s * interval '1 hour')
        ON CONFLICT (source_id) DO UPDATE SET
            body = EXCLUDED.body,
            content_hash = EXCLUDED.content_hash,
            vec = EXCLUDED.vec,
            observed_at = EXCLUDED.observed_at,
            expires_at = EXCLUDED.expires_at,
            ingested_at = now(),
            deleted = false
        """,
        {
            "sid": source_id, "body": body, "hash": content_hash,
            "vec": pg.vec_literal(vector), "observed": observed, "ttl": ttl_hours,
        },
    )
    return "updated" if existing else "inserted"


def search_with_recency(conn, query_vector, lambda_: float, k: int = 5):
    """Dense search with the book's recency prior applied in SQL.

        score = similarity - lambda * age_hours

    Two things worth noting. The age is computed from `observed_at`, not
    `ingested_at` -- when an event happened matters, when we heard about it
    does not. And the arithmetic happens in the database, so the ranking and
    the filtering stay in one query rather than fetching rows to re-sort in
    Python.
    """
    return conn.execute(
        """
        SELECT source_id,
               body,
               1 - (vec <=> %(qvec)s::vector)                        AS similarity,
               EXTRACT(EPOCH FROM (now() - observed_at)) / 3600.0    AS age_hours,
               (1 - (vec <=> %(qvec)s::vector))
                   - %(lambda)s * (EXTRACT(EPOCH FROM (now() - observed_at)) / 3600.0)
                                                                     AS score
        FROM   stream_items
        WHERE  NOT deleted
          AND  (expires_at IS NULL OR expires_at > now())
        ORDER  BY score DESC
        LIMIT  %(k)s
        """,
        {"qvec": pg.vec_literal(query_vector), "lambda": lambda_, "k": k},
    ).fetchall()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-", dest="lam", type=float, default=0.002)
    args = parser.parse_args()

    display.banner(
        "29. Streaming RAG",
        "Chapter 16, Streaming RAG",
        "Incremental ingestion, content-hash deduplication, TTL expiry, and "
        "the book's recency prior applied in SQL.",
    )

    if not pg.available():
        display.missing_database("Incremental upserts and TTL need a database.")
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    embedder = get_embedder()
    ensure_stream_table(conn)
    conn.execute("TRUNCATE stream_items")

    # --- first pass -------------------------------------------------------
    display.heading("Ingest: first pass")
    results = []
    for source_id, body, age in INCOMING:
        started = time.perf_counter()
        outcome = upsert(conn, embedder, source_id, body, age)
        results.append([source_id, outcome, f"{age:.1f}h ago",
                        f"{(time.perf_counter()-started)*1000:.0f} ms"])
    display.table(["source", "outcome", "observed", "time"], results)

    # --- second pass, nothing changed ------------------------------------
    display.heading("Ingest: the same feed again, unchanged")
    results = []
    for source_id, body, age in INCOMING:
        started = time.perf_counter()
        outcome = upsert(conn, embedder, source_id, body, age)
        results.append([source_id, outcome, f"{(time.perf_counter()-started)*1000:.0f} ms"])
    display.table(["source", "outcome", "time"], results)
    print("\n  Every row unchanged, and no embedding call was made. The hash is")
    print("  checked before the embedder, which is the whole saving -- feeds")
    print("  resend constantly and embedding is the expensive step.")

    # --- an edit ----------------------------------------------------------
    display.heading("Ingest: one document edited")
    outcome = upsert(
        conn, embedder, "bulletin-1",
        "Inspector Lestrade reports the Kensington jeweller burglary is now solved.",
        0.2,
    )
    print(f"  bulletin-1: {outcome}")
    print("  New text, new hash, new vector. The old vector is replaced rather")
    print("  than accumulated -- otherwise the index fills with stale versions")
    print("  of the same document, all still retrievable.")

    # --- recency sweep ----------------------------------------------------
    display.heading("The recency prior:  score = similarity - lambda * age_hours")
    print(f"  query: {QUERY!r}\n")
    query_vector = embedder.encode([QUERY], show_progress=False)[0]

    for lam in (0.0, 0.0005, 0.002, 0.01):
        rows = search_with_recency(conn, query_vector, lam, k=5)
        label = "lambda = 0 (pure similarity)" if lam == 0 else f"lambda = {lam}"
        print(f"  {label}")
        display.table(
            ["source", "similarity", "age (h)", "score"],
            [[r[0], f"{float(r[2]):.3f}", f"{float(r[3]):.1f}", f"{float(r[4]):.3f}"]
             for r in rows],
        )
        print()

    print("  As lambda rises, age overwhelms relevance. At 0.01 a document one")
    print("  day old loses 0.24 of score, which is more than the entire spread")
    print("  of similarity in this set -- so the ranking becomes purely")
    print("  chronological and the retriever has stopped retrieving.")
    print("\n  That is the risk with a linear penalty: it is unbounded. A decay")
    print("  of the form  similarity * exp(-age/tau)  keeps the ordering")
    print("  relevance-first while still preferring fresh documents.")

    # --- TTL and watermark ------------------------------------------------
    display.heading("TTL and the freshness watermark")
    conn.execute(
        "UPDATE stream_items SET expires_at = now() - interval '1 hour' "
        "WHERE source_id = 'bulletin-5'"
    )
    live = conn.execute(
        "SELECT count(*) FROM stream_items WHERE NOT deleted "
        "AND (expires_at IS NULL OR expires_at > now())"
    ).fetchone()[0]
    total = conn.execute("SELECT count(*) FROM stream_items").fetchone()[0]
    display.kv("rows in table", total)
    display.kv("live (not expired)", live)

    watermark = conn.execute(
        "SELECT max(observed_at), min(observed_at) FROM stream_items WHERE NOT deleted"
    ).fetchone()
    display.kv("newest observation", str(watermark[0])[:19])
    display.kv("oldest observation", str(watermark[1])[:19])
    print("\n  The watermark answers 'how fresh is this index?' -- the question")
    print("  a caller needs answered before trusting an answer about anything")
    print("  time-sensitive. The book suggests requiring at least one context")
    print("  updated within a freshness window for safety-critical queries.")

    # --- deletion ---------------------------------------------------------
    display.heading("Deletion propagation")
    conn.execute("UPDATE stream_items SET deleted = true WHERE source_id = 'bulletin-2'")
    after = conn.execute(
        "SELECT count(*) FROM stream_items WHERE NOT deleted "
        "AND (expires_at IS NULL OR expires_at > now())"
    ).fetchone()[0]
    display.kv("retrievable after delete", after)
    print("\n  A soft delete keeps the row for audit while removing it from")
    print("  retrieval. For a right-to-be-forgotten request that is not enough")
    print("  -- example 35 does the hard delete, and the vector must go too.")

    display.notice(
        "The content hash is checked before embedding, not after. That "
        "ordering is the entire cost saving on a feed that re-sends unchanged "
        "records.",
        "Age is measured from when the event was observed, not when it was "
        "ingested. A backfill of last month's data should not arrive looking "
        "brand new.",
        "The book's linear recency prior is unbounded: a large enough lambda "
        "turns retrieval into a reverse-chronological list. Exponential decay "
        "keeps relevance in charge.",
        "Updating replaces the vector rather than adding one. Accumulate "
        "versions and every edit leaves a retrievable ghost of the old text.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
