#!/usr/bin/env python3
"""
Create the schema in whatever DATABASE_URL points at.

Expects the Postgres from docker-compose.yml, or any Postgres 15+ with
pgvector available.

    python scripts/init_db.py           # extensions, tables, indexes, RLS
    python scripts/init_db.py --ann     # also build the HNSW index
    python scripts/init_db.py --reset   # drop everything first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg

TABLES = [
    "query_log", "feedback", "memory", "edges", "entities",
    "embeddings_1536", "embeddings_384", "chunks", "documents",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop all tables first")
    parser.add_argument(
        "--ann",
        action="store_true",
        help="also build the HNSW index (unnecessary at this corpus size; see sql/002)",
    )
    args = parser.parse_args()

    if not pg.available():
        display.missing_database(
            f"Cannot initialise the schema: {pg.why_unavailable()}."
        )
        return 1

    try:
        conn = pg.connect()
    except pg.DatabaseUnavailable as error:
        print(f"\nCould not connect: {error}")
        if error.remedy:
            print(f"\n{error.remedy}\n")
        return 1

    info = pg.server_info(conn)
    print(f"\nConnected to {info['host']}")
    print(f"  Postgres {info['postgres']}")

    if args.reset:
        print("\nDropping existing tables...")
        for table in TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        print(f"  dropped {len(TABLES)}")

    print("\nApplying schema...")
    for name in ["001_schema.sql", "002_indexes.sql", "003_rls.sql"]:
        try:
            pg.apply_sql_files(conn, [name])
            print(f"  {name}")
        except Exception as error:  # noqa: BLE001
            print(f"  {name} FAILED: {error}")
            return 1

    info = pg.server_info(conn)
    print(f"\n  pgvector {info['pgvector']}")

    if args.ann:
        # Built on request only. At this corpus's size an exact scan beats it
        # on both latency and recall -- example 20 measures that.
        print("\nBuilding HNSW index (this is slower than the scan it replaces)...")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS emb384_hnsw ON embeddings_384 "
            "USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
        print("  emb384_hnsw")

    tables = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"
    ).fetchall()
    print(f"\n{len(tables)} tables ready:")
    print("  " + ", ".join(t[0] for t in tables))

    print("\nNext:  python scripts/index_corpus.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
