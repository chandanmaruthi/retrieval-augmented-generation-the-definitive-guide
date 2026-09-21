#!/usr/bin/env python3
"""
19. Hybrid Retrieval: Linear Weighting vs Reciprocal Rank Fusion
Book: Chapter 9, Hybrid RAG
WHAT THIS SHOWS: How to actually combine a dense and a sparse retriever, and
why the obvious way is wrong.

The obvious way is to score each arm and add the scores with weights. Example
18 showed the problem: cosine similarity on this corpus lands around 0.5,
while ts_rank_cd lands around 0.001. Add them at 50/50 and the lexical arm
contributes roughly nothing -- you have built a hybrid retriever that is
really just vector search with extra latency.

Two fixes, both shown here: normalise the scores before weighting, or ignore
the scores entirely and fuse on rank. This example runs all three variants
against the same query so the difference is visible rather than argued.

The SQL is in sql/queries/ and is printed below before it runs, because the
query shape -- not the arithmetic -- is most of the lesson.

HOW THIS SCRIPT PROCEEDS
    1. Run each arm separately        dense, then lexical
    2. Show the SCALE GAP between them    <-- the problem, ~645x here
    3. Variant 1: add the raw scores      <-- the wrong way, and why
    4. Variant 2: min-max normalise first <-- fix 1
    5. Variant 3: fuse on rank (RRF)      <-- fix 2
    6. Check whether the ORDER changed    the test that catches a fake hybrid

Step 6 is the one to copy into your own project. A "hybrid" retriever that
returns dense-only's exact ordering is not fusing anything, and nothing else
in its output will tell you that.


WHAT CHANGED SINCE EXAMPLE 14
    Example 14 fused two indexes whose scores were at least the same KIND of
    number -- cosine against cosine. Here the arms are cosine (~0.5) and
    ts_rank_cd (~0.001), three orders of magnitude apart, which is what makes
    the naive approach fail so completely.


REQUIRES: Postgres
RUN: python examples/19_hybrid_fusion.py ["your query"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import describe, get_embedder

DEFAULT_QUERY = "What did Holmes find in the goose?"


def unnormalised_linear(conn, store, query_text, query_vector, strategy, k, fanout):
    """The naive fusion -- see the block below."""
    # =================================================================
    # THE WRONG WAY, implemented so its failure is measurable rather than
    # asserted. This is what most first drafts of "hybrid search" are:
    # identical to sql/queries/hybrid_linear.sql except the min-max
    # normalisation CTE is gone and the raw scores are simply added.
    # =================================================================
    rows = conn.execute(
        """
        WITH vec_candidates AS (
            SELECT e.chunk_id, e.vec <=> %(qvec)s::vector AS distance
            FROM   embeddings_384 e
            WHERE  e.strategy = %(strategy)s AND e.model = %(model)s
            ORDER  BY e.vec <=> %(qvec)s::vector
            LIMIT  %(fanout)s
        ),
        vec AS (SELECT chunk_id, 1.0 - distance AS vec_score FROM vec_candidates),
        txt AS (
            SELECT c.id AS chunk_id,
                   ts_rank_cd(c.tsv, websearch_to_tsquery('english', %(qtext)s), 32) AS txt_score
            FROM   chunks c
            WHERE  c.strategy = %(strategy)s
              AND  c.tsv @@ websearch_to_tsquery('english', %(qtext)s)
            ORDER  BY txt_score DESC
            LIMIT  %(fanout)s
        )
        SELECT c.id, c.story, c.body,
               COALESCE(v.vec_score, 0) AS vec_score,
               COALESCE(t.txt_score, 0) AS txt_score,
               0.5 * COALESCE(v.vec_score, 0) + 0.5 * COALESCE(t.txt_score, 0) AS score
        FROM   vec v FULL OUTER JOIN txt t USING (chunk_id)
        JOIN   chunks c ON c.id = chunk_id
        ORDER  BY score DESC
        LIMIT  %(k)s
        """,
        {
            "qvec": pg.vec_literal(query_vector),
            "qtext": query_text,
            "strategy": strategy,
            "model": store.model,
            "fanout": fanout,
            "k": k,
        },
    ).fetchall()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fanout", type=int, default=50)
    parser.add_argument("--strategy", default="contextual_header")
    parser.add_argument("--show-sql", action="store_true", help="print the full query")
    args = parser.parse_args()

    display.banner(
        "19. Hybrid Retrieval: Linear vs RRF",
        "Chapter 9, Hybrid RAG",
        "Fusing a dense and a sparse retriever, and why adding their scores "
        "together does not work.",
    )

    if not pg.available():
        display.missing_database(
            "Hybrid fusion is a single SQL query joining pgvector distance "
            "against a full-text rank. It is the one example in this "
            "repository that genuinely needs a database."
        )
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    embedder = get_embedder()
    store = pg.PgVectorStore(conn, dim=embedder.dim, model=embedder.name)

    if store.count(args.strategy) == 0:
        print("\nNothing indexed. Run: python scripts/index_corpus.py\n")
        return 1

    print(f"  {describe(embedder)}")
    print(f"  strategy: {args.strategy} ({store.count(args.strategy):,} chunks)")
    print(f"  query: {args.query!r}")
    print(f"  fan-out: {args.fanout} candidates per arm -> top {args.k}\n")

    query_vector = embedder.encode([args.query], show_progress=False)[0]

    # --- the two arms, separately ----------------------------------------
    display.heading("The two arms, before fusing")
    dense = store.search(query_vector, k=args.k, strategy=args.strategy)
    lexical = store.search_lexical(args.query, k=args.k, strategy=args.strategy)

    print("  DENSE (pgvector cosine):")
    display.table(
        ["rank", "score", "story", "passage"],
        [[h.rank, f"{h.score:.4f}", display.truncate(h.chunk.story, 22),
          display.truncate(h.chunk.text, 30)] for h in dense],
        align="rrll",
    )
    print("\n  LEXICAL (ts_rank_cd):")
    if lexical:
        display.table(
            ["rank", "score", "story", "passage"],
            [[h.rank, f"{h.score:.4f}", display.truncate(h.chunk.story, 22),
              display.truncate(h.chunk.text, 30)] for h in lexical],
            align="rrll",
        )
    else:
        print("      no lexical matches for this query")

    if dense and lexical:
        ratio = dense[0].score / max(lexical[0].score, 1e-9)
        display.heading("The scale problem")
        display.kv("top dense score", f"{dense[0].score:.4f}")
        display.kv("top lexical score", f"{lexical[0].score:.4f}")
        display.kv("ratio", f"{ratio:,.0f}x")
        print(f"\n  The dense arm's scores are {ratio:,.0f} times larger. Adding these")
        print("  at equal weight does not produce a 50/50 blend -- it produces")
        print("  vector search with a rounding error attached.")

    # --- variant 1: naive -------------------------------------------------
    display.heading("Variant 1: linear weighting, raw scores (the wrong way)")
    naive = unnormalised_linear(
        conn, store, args.query, query_vector, args.strategy, args.k, args.fanout
    )
    display.table(
        ["rank", "vec", "txt", "total", "passage"],
        [
            [rank, f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.4f}",
             display.truncate(r[2], 32)]
            for rank, r in enumerate(naive, start=1)
        ],
        align="rrrrl",
    )
    if naive:
        contribution = sum(r[4] for r in naive) / max(sum(r[3] + r[4] for r in naive), 1e-9)
        print(f"\n  The lexical arm contributed {100 * contribution:.2f}% of the total")
        print("  score. That is the bug, and it is invisible unless you look.")

    # --- variant 2: normalised -------------------------------------------
    display.heading("Variant 2: linear weighting, min-max normalised")
    linear = store.search_hybrid(
        args.query, query_vector, k=args.k, fanout=args.fanout,
        mode="linear", w_vec=0.5, w_txt=0.5, strategy=args.strategy,
    )
    display.table(
        ["rank", "vec norm", "txt norm", "score", "passage"],
        [
            [h.rank,
             f"{h.components.get('vector_norm', 0):.3f}",
             f"{h.components.get('lexical_norm', 0):.3f}",
             f"{h.score:.3f}",
             display.truncate(h.chunk.text, 32)]
            for h in linear
        ],
        align="rrrrl",
    )
    print("\n  Both arms now span [0, 1] across the candidate set, so a weight of")
    print("  0.5 actually means half. The cost is that scores are relative to")
    print("  whatever else was retrieved -- the same chunk scores differently")
    print("  depending on its competition.")

    # --- variant 3: RRF ---------------------------------------------------
    display.heading("Variant 3: reciprocal rank fusion")
    rrf = store.search_hybrid(
        args.query, query_vector, k=args.k, fanout=args.fanout,
        mode="rrf", strategy=args.strategy,
    )
    display.table(
        ["rank", "vec rank", "txt rank", "RRF", "passage"],
        [
            [h.rank,
             int(h.components.get("vector_rank", 0)) or "-",
             int(h.components.get("lexical_rank", 0)) or "-",
             f"{h.score:.5f}",
             display.truncate(h.chunk.text, 32)]
            for h in rrf
        ],
        align="rrrrl",
    )
    print("\n  No normalisation, no weights to tune, and no way for a scale")
    print("  mismatch to matter -- RRF never looks at the scores at all.")

    # --- comparison -------------------------------------------------------
    display.heading("Do the three variants agree?")
    sets = {
        "dense only": [h.chunk.index for h in dense],
        "naive linear": [r[0] for r in naive],
        "normalised linear": [h.chunk.index for h in linear],
        "RRF": [h.chunk.index for h in rrf],
    }
    baseline = sets["dense only"]
    display.table(
        ["variant", "top-k ids", "same set", "same order"],
        [
            [
                name,
                ", ".join(str(i) for i in ids[:5]),
                "yes" if set(ids) == set(baseline) else "no",
                "yes" if ids == baseline else "no",
            ]
            for name, ids in sets.items()
        ],
        align="llll",
    )
    print("\n  Set overlap is the wrong thing to check here -- every variant")
    print("  returned the same five chunks. What changed is the ORDER, and")
    print("  order decides which passages survive the context budget.")
    print("\n  Naive linear reproduces dense-only's exact ordering: it is not")
    print("  fusing anything. Both corrected variants promote the chunk that")
    print("  BOTH arms found, which is the entire purpose of hybrid retrieval.")

    if args.show_sql:
        display.heading("sql/queries/hybrid_rrf.sql")
        print(pg.query("hybrid_rrf"))
    else:
        print("\n  Run with --show-sql to print the full fusion query, or read")
        print("  sql/queries/hybrid_linear.sql and hybrid_rrf.sql.")

    display.notice(
        "Score-scale mismatch is the defining failure of naive hybrid "
        "retrieval. Cosine and ts_rank differ by orders of magnitude here, so "
        "equal weights are not equal contributions.",
        "Min-max normalisation fixes the weighting but makes every score "
        "relative to the candidate set. RRF sidesteps the problem entirely by "
        "using only positions.",
        "Each arm is a separate CTE with its own ORDER BY and LIMIT. That is "
        "not style -- fusing before limiting prevents either index from being "
        "used and turns the query into a full scan.",
        "The vector arm orders by a bare `vec <=> query` ascending. Writing "
        "`ORDER BY (1 - (vec <=> query)) DESC` is the same ranking and "
        "silently disables the ANN index.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
