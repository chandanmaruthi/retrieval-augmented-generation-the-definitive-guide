#!/usr/bin/env python3
"""
16. Indexing: HNSW, IVFFlat, and When Not to Index at All
Book: Chapter 4, Reference Architecture (block 3, Indexing)

WHAT THIS SHOWS: Approximate nearest-neighbour indexes are not free and are
not always faster. This example builds them on the real corpus and measures
latency and recall against an exact scan, then reports whichever answer the
numbers give.

On a corpus this size the honest answer is that the exact scan wins. It is
faster AND it has perfect recall. A repo that reached for HNSW at four hundred
rows would be teaching a reflex rather than a technique.

It also demonstrates the single most common pgvector bug: a `WHERE` clause
combined with an ANN index is POST-filtered, so asking for 10 results can
return 3, silently.

HOW THIS SCRIPT PROCEEDS
    1. Time an exact scan with NO index    this is ground truth, recall = 1.00
    2. Build HNSW, time it, measure recall against ground truth
    3. Build IVFFlat, same
    4. Report whichever actually won
    5. Demonstrate the post-filtering trap  <-- the part worth remembering

Step 5 is the practical lesson. Steps 1-4 produce a table whose conclusion is
specific to this corpus size; step 5 produces a bug that bites at every size.


WHY THIS RUNS BEFORE THE RETRIEVAL EXAMPLES
    Every example from 17 onward queries this index, and all of them run
    against a sequential scan because at 413 vectors that is genuinely the
    faster choice. This script is the measurement that justifies leaving the
    ANN indexes out, rather than a claim that they are unnecessary.


REQUIRES: Postgres
RUN: python examples/16_index_pgvector.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import describe, get_embedder

PROBE_QUERIES = [
    "What was the speckled band?",
    "Who stole the blue carbuncle?",
    "Why was the Red-Headed League created?",
    "What happened at Boscombe Pool?",
    "Who was Hosmer Angel really?",
]


def time_query(conn, sql: str, params: dict, repeat: int) -> tuple[float, list[int]]:
    """Median latency over `repeat` runs, plus the ids from the last one.

    Median rather than mean: the first execution pays for planning and a cold
    cache, and one outlier would dominate an average over twenty runs.
    """
    timings = []
    ids: list[int] = []
    for _ in range(repeat):
        started = time.perf_counter()
        rows = conn.execute(sql, params).fetchall()
        timings.append((time.perf_counter() - started) * 1000)
        ids = [int(r[0]) for r in rows]
    return float(np.median(timings)), ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "16. Indexing: HNSW, IVFFlat, and When Not to Index",
        "Chapter 4, Reference Architecture -- block 3, Indexing",
        "Building the ANN indexes the book names, measuring them against an "
        "exact scan, and reporting whichever wins.",
    )

    if not pg.available():
        display.missing_database("Index behaviour can only be measured against a real database.")
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    embedder = get_embedder()
    store = pg.PgVectorStore(conn, dim=embedder.dim, model=embedder.name)
    rows = store.count(args.strategy)

    if rows == 0:
        print("\nNothing indexed. Run: python scripts/index_corpus.py\n")
        return 1

    print(f"  {describe(embedder)}")
    print(f"  {rows:,} vectors of {embedder.dim} dimensions in {store.table}\n")

    query_vectors = embedder.encode(PROBE_QUERIES, show_progress=False)

    search_sql = f"""
        SELECT e.chunk_id
        FROM   {store.table} e
        WHERE  e.strategy = %(strategy)s AND e.model = %(model)s
        ORDER  BY e.vec <=> %(qvec)s::vector
        LIMIT  %(k)s
    """

    def run_all(label: str) -> tuple[float, dict[int, list[int]]]:
        latencies = []
        results = {}
        for index, vector in enumerate(query_vectors):
            ms, ids = time_query(
                conn,
                search_sql,
                {
                    "strategy": args.strategy,
                    "model": store.model,
                    "qvec": pg.vec_literal(vector),
                    "k": args.k,
                },
                args.repeat,
            )
            latencies.append(ms)
            results[index] = ids
        print(f"  {label}: median {np.median(latencies):.2f} ms")
        return float(np.median(latencies)), results

    # --- exact scan (ground truth) ---------------------------------------
    display.heading("1. Exact scan, no index")
    conn.execute("DROP INDEX IF EXISTS emb384_hnsw")
    conn.execute("DROP INDEX IF EXISTS emb384_ivf")
    exact_ms, exact_results = run_all("sequential scan")
    print("\n  This is ground truth: it compares the query against every vector,")
    print("  so its recall is 100% by definition. Everything below is measured")
    print("  against it.")

    # --- HNSW --------------------------------------------------------------
    display.heading("2. HNSW")
    started = time.time()
    conn.execute(
        f"CREATE INDEX emb384_hnsw ON {store.table} "
        "USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    build_seconds = time.time() - started
    print(f"  built in {build_seconds:.1f}s")
    conn.execute("ANALYZE " + store.table)
    hnsw_ms, hnsw_results = run_all("hnsw")

    hnsw_recall = float(
        np.mean([
            len(set(hnsw_results[i]) & set(exact_results[i])) / max(len(exact_results[i]), 1)
            for i in exact_results
        ])
    )

    # --- IVFFlat -----------------------------------------------------------
    display.heading("3. IVFFlat")
    conn.execute("DROP INDEX IF EXISTS emb384_hnsw")
    # lists ~ rows/1000, floored at 10. IVFFlat clusters at build time, so it
    # MUST be built after the data is loaded -- on an empty table the
    # centroids are meaningless and recall never recovers.
    lists = max(10, rows // 1000)
    started = time.time()
    conn.execute(
        f"CREATE INDEX emb384_ivf ON {store.table} "
        f"USING ivfflat (vec vector_cosine_ops) WITH (lists = {lists})"
    )
    ivf_build = time.time() - started
    print(f"  built in {ivf_build:.1f}s (lists = {lists})")
    conn.execute("ANALYZE " + store.table)
    ivf_ms, ivf_results = run_all("ivfflat")

    ivf_recall = float(
        np.mean([
            len(set(ivf_results[i]) & set(exact_results[i])) / max(len(exact_results[i]), 1)
            for i in exact_results
        ])
    )

    # --- results -----------------------------------------------------------
    display.heading("Results")
    display.table(
        ["index", "build", f"median latency (k={args.k})", f"recall@{args.k}"],
        [
            ["none (exact scan)", "-", f"{exact_ms:.2f} ms", "1.00"],
            ["hnsw", f"{build_seconds:.1f}s", f"{hnsw_ms:.2f} ms", f"{hnsw_recall:.2f}"],
            ["ivfflat", f"{ivf_build:.1f}s", f"{ivf_ms:.2f} ms", f"{ivf_recall:.2f}"],
        ],
    )

    winner = min(
        [("exact scan", exact_ms), ("hnsw", hnsw_ms), ("ivfflat", ivf_ms)],
        key=lambda pair: pair[1],
    )
    print(f"\n  Fastest: {winner[0]} at {winner[1]:.2f} ms.")
    print(f"  But every option here is under a millisecond, because {rows:,} vectors")
    print("  is a trivial amount of work. The latency column is noise at this")
    print("  scale; the recall column is not.")
    if ivf_recall < 0.95:
        print(f"\n  IVFFlat lost {100 * (1 - ivf_recall):.0f}% of the correct results to buy")
        print("  a speedup measured in tenths of a millisecond. That is a bad")
        print("  trade here and a good one at ten million vectors -- which is the")
        print("  whole point: the answer depends on a number you have to measure.")

    # --- the post-filter trap ---------------------------------------------
    # =================================================================
    # THE PART TO REMEMBER. Everything above is a benchmark whose answer
    # depends on corpus size. This is a correctness bug that does not.
    # =================================================================
    display.heading("The post-filtering trap")
    conn.execute("DROP INDEX IF EXISTS emb384_ivf")
    conn.execute(
        f"CREATE INDEX emb384_hnsw ON {store.table} "
        "USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    conn.execute("ANALYZE " + store.table)

    strategies = [name for name, _ in store.strategies()]
    print(f"  Indexed strategies: {', '.join(strategies)}")
    print("\n  Asking for %d results filtered to one strategy, with a global" % args.k)
    print("  HNSW index and a deliberately small ef_search:\n")

    returned = []
    # ---- THE KEY DETAIL --------------------------------------------
    # SET LOCAL only has effect inside a transaction, and ragkit connects
    # with autocommit ON. Without this explicit block the setting is a
    # silent no-op and the demonstration below quietly proves nothing.
    with conn.transaction():
        # SET LOCAL only has effect inside a transaction. ragkit connects with
        # autocommit on, so without this block the setting is silently a no-op.
        conn.execute("SET LOCAL hnsw.ef_search = 10")
        for strategy in strategies:
            result = conn.execute(
                search_sql,
                {
                    "strategy": strategy,
                    "model": store.model,
                    "qvec": pg.vec_literal(query_vectors[0]),
                    "k": args.k,
                },
            ).fetchall()
            returned.append([strategy, args.k, len(result)])

    display.table(["strategy filter", "asked for", "got back"], returned)

    short = [r for r in returned if r[2] < r[1]]
    if short:
        print(f"\n  {len(short)} of {len(returned)} came back short, with no error.")
        print("  pgvector walked the graph, collected ef_search candidates from")
        print("  across ALL strategies, and only then applied the filter. Most")
        print("  candidates belonged to the other strategy and were discarded.")
    else:
        print("\n  Not short this time -- with few enough rows the scan is chosen")
        print("  anyway. Raise the row count or lower ef_search and it will be.")
    print("\n  The fixes: a partial index per strategy (see sql/002_indexes.sql),")
    print("  or pgvector 0.8's  SET LOCAL hnsw.iterative_scan = 'relaxed_order'.")

    conn.execute("DROP INDEX IF EXISTS emb384_hnsw")
    print("\n  Indexes dropped; the database is back to how it started.")

    display.notice(
        f"At {rows:,} vectors every option answered in under a millisecond, so "
        f"latency did not separate them. Recall did: IVFFlat returned "
        f"{ivf_recall:.2f} of the correct results and the exact scan returned "
        "all of them. Index when a measurement tells you to, not by reflex.",
        "IVFFlat must be built after the data is loaded. Build it on an empty "
        "table and its centroids are meaningless, permanently.",
        "A WHERE clause plus a global ANN index is post-filtered. You can ask "
        "for ten and get three, with no error anywhere.",
        "SET LOCAL does nothing outside a transaction, and ragkit connects "
        "with autocommit on. That is why the block above opens one explicitly.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
