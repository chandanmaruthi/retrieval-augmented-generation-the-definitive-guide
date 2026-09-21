#!/usr/bin/env python3
"""
24. Hierarchical RAG (coarse-to-fine retrieval)
Book: Chapter 12, Hierarchical RAG
WHAT THIS SHOWS: Retrieve in levels instead of searching everything at once.

    Topic  ->  Document  ->  Section  ->  Passage

The book names that hierarchy explicitly, and this corpus has it for real: a
book of twelve stories, several with numbered parts, each made of paragraphs.
So the levels are the document's own, not clusters invented by k-means.

Two things this buys. First, each level can use a different, cheaper
retriever, because narrowing from twelve stories to one does not need a
cross-encoder. Second -- and more interesting -- it changes what "relevant"
means: a passage can be a weak match on its own while sitting inside
overwhelmingly the right story.

HOW THIS SCRIPT PROCEEDS
    1. Flat retrieval over everything     the baseline to beat
    2. Level 1: score each STORY by aggregating its chunks  <-- the design choice
    3. Keep the top 2 stories
    4. Level 2: retrieve only within those
    5. Show which passages this surfaced that flat retrieval missed

Step 2 is where the thinking is. MAX finds the story containing the single
best passage; MEAN finds the story most consistently about the topic. Both
are printed so the difference is visible.


WHAT THIS CHANGES ABOUT "RELEVANT"
    Flat retrieval ranks passages and has no notion of a document, so
    passages from unrelated stories interleave freely.

    Restricting to the right document first promotes passages that were
    outranked globally -- a weak match inside the right story often beats a
    strong match inside the wrong one. The cost is a hard failure mode: pick
    the wrong story at level 1 and no level-2 quality recovers.


REQUIRES: Postgres
RUN: python examples/24_hierarchical_rag.py ["your question"]
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

DEFAULT_QUESTION = "How did the thief get into the room at night?"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--stories", type=int, default=2, help="stories to keep at level 1")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "24. Hierarchical RAG",
        "Chapter 12, Hierarchical RAG",
        f"Narrow to {args.stories} stories first, then search only inside "
        "them. Coarse to fine, using the document's own structure.",
    )

    if not pg.available():
        display.missing_database("Both levels retrieve from the indexed corpus.")
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
    print(f"  question: {args.question!r}\n")

    query_vector = embedder.encode([args.question], show_progress=False)[0]

    # --- flat baseline ----------------------------------------------------
    display.heading("Baseline: flat retrieval over everything")
    started = time.perf_counter()
    flat = store.search(query_vector, k=args.k, strategy=args.strategy)
    flat_ms = (time.perf_counter() - started) * 1000
    display.table(
        ["rank", "score", "story", "passage"],
        [[h.rank, f"{h.score:.3f}", display.truncate(h.chunk.story, 26),
          display.truncate(h.chunk.text, 28)] for h in flat],
        align="rrll",
    )
    stories_hit = len({h.chunk.story for h in flat})
    print(f"\n  {stories_hit} different stories in the top {args.k}. Flat retrieval has")
    print("  no notion of a document -- it ranks passages, and passages from")
    print("  unrelated stories can interleave freely.")

    # --- level 1: which story? -------------------------------------------
    display.heading("Level 1: which story is this question about?")
    started = time.perf_counter()

    # Score each story by aggregating over its chunks. The aggregate matters:
    # MAX finds the story containing the single best passage, MEAN finds the
    # story that is most consistently about this topic. MEAN of the top few is
    # a middle ground -- it resists both a single lucky chunk and dilution by
    # a long story's irrelevant remainder.
    rows = conn.execute(
        f"""
        SELECT c.story,
               count(*)                                         AS n_chunks,
               max(1 - (e.vec <=> %(qvec)s::vector))            AS best,
               avg(1 - (e.vec <=> %(qvec)s::vector))            AS mean
        FROM   {store.table} e
        JOIN   chunks c ON c.id = e.chunk_id
        WHERE  e.strategy = %(strategy)s AND e.model = %(model)s
        GROUP  BY c.story
        ORDER  BY best DESC
        """,
        {
            "qvec": pg.vec_literal(query_vector),
            "strategy": args.strategy,
            "model": store.model,
        },
    ).fetchall()
    level1_ms = (time.perf_counter() - started) * 1000

    display.table(
        ["story", "chunks", "best chunk", "mean"],
        [[display.truncate(r[0], 36), r[1], f"{float(r[2]):.3f}", f"{float(r[3]):.3f}"]
         for r in rows[:6]],
    )
    print(f"\n  {level1_ms:.1f} ms to rank all {len(rows)} stories.")

    # --- level 1 narrows the search space -- book:retrieval-hierarchical
    chosen = [r[0] for r in rows[: args.stories]]
    # ------------------------------------------------------------- /book
    print(f"\n  Keeping: {', '.join(display.truncate(s, 34) for s in chosen)}")

    # --- level 2: within those stories ------------------------------------
    display.heading(f"Level 2: passages within those {args.stories} stories")
    started = time.perf_counter()
    rows = conn.execute(
        f"""
        SELECT c.id, c.body, c.parent_body, c.story, c.section,
               1 - (e.vec <=> %(qvec)s::vector) AS score
        FROM   {store.table} e
        JOIN   chunks c ON c.id = e.chunk_id
        WHERE  e.strategy = %(strategy)s
          AND  e.model = %(model)s
          AND  c.story = ANY(%(stories)s)
        ORDER  BY e.vec <=> %(qvec)s::vector
        LIMIT  %(k)s
        """,
        {
            "qvec": pg.vec_literal(query_vector),
            "strategy": args.strategy,
            "model": store.model,
            "stories": chosen,
            "k": args.k,
        },
    ).fetchall()
    level2_ms = (time.perf_counter() - started) * 1000

    display.table(
        ["rank", "score", "story", "passage"],
        [[rank, f"{float(r[5]):.3f}", display.truncate(r[3], 26),
          display.truncate(r[1], 28)]
         for rank, r in enumerate(rows, start=1)],
        align="rrll",
    )

    # --- comparison -------------------------------------------------------
    display.heading("What changed")
    flat_ids = [h.chunk.index for h in flat]
    hier_ids = [int(r[0]) for r in rows]
    new = [i for i in hier_ids if i not in flat_ids]

    display.table(
        ["", "stories in top-k", "time", "chunk ids"],
        [
            ["flat", stories_hit, f"{flat_ms:.1f} ms",
             ", ".join(str(i) for i in flat_ids)],
            ["hierarchical", len({r[3] for r in rows}),
             f"{level1_ms + level2_ms:.1f} ms",
             ", ".join(str(i) for i in hier_ids)],
        ],
        align="llll",
    )
    print(f"\n  {len(new)} passage(s) in the hierarchical result never appeared in")
    print("  the flat top-k. They were outranked globally by chunks from other")
    print("  stories, but they are the best available evidence inside the story")
    print("  this question is actually about.")

    print("\n  Note the timing: the two-level version is not faster here. At 413")
    print("  chunks there is nothing to save. The win at this scale is")
    print("  precision, not latency -- and at ten million chunks it would be")
    print("  both.")

    display.notice(
        "The levels are the document's own -- book, story, part, paragraph. "
        "No clustering step, and the level names mean something a human can "
        "check.",
        "Aggregating chunk scores into a document score is a real design "
        "choice. MAX chases the single best passage; MEAN rewards consistent "
        "topicality and penalises long documents.",
        "Restricting to the right document promotes passages that flat "
        "retrieval ranked too low to show. A weak match inside the right story "
        "often beats a strong match inside the wrong one.",
        "The cost is a hard failure mode: if level 1 picks the wrong story, no "
        "amount of level-2 quality recovers. Keeping 2-3 documents rather than "
        "1 is cheap insurance.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
