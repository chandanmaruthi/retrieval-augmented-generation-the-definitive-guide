#!/usr/bin/env python3
"""
14. Dual-Index Question-Referenced Chunking
Book: Chapter 5, Chunking Strategies -- section 3.13
WHAT THIS SHOWS: Keep two indexes instead of one compromise.

    question index   short generated questions -> chunk id
    canonical index  the passages themselves

Search both, map every question hit back to its chunk, merge, deduplicate,
rerank. A query phrased as a question finds the question index; a query
phrased as a keyword or a quotation finds the passage index. Neither has to
be degraded to serve the other.

This is the production-shaped version of example 11, and the last of the
book's thirteen chunking strategies.

HOW THIS SCRIPT PROCEEDS
    1. Build the canonical chunk index      the passages themselves
    2. Build a question index               pointers, not copies
    3. Search BOTH for one query
    4. Collapse question hits to one per chunk   <-- easy to forget, see below
    5. Fuse the two rankings by reciprocal rank  <-- THE TECHNIQUE
    6. Show which results only one index found

Step 5 needs no model and no database -- fuse() is pure arithmetic over two
ranked lists, and it is worth reading even if you skip everything else here.


WHAT CHANGED SINCE EXAMPLE 11
    Example 11 replaced the passage index with a question index. This keeps
    both, because they fail on opposite queries: a question-phrased query
    matches the question index, and a keyword or quotation matches the
    passages. Neither has to be degraded to serve the other.

    This is also the first appearance of reciprocal rank fusion, which example
    19 then applies to a much wider scale mismatch.


REQUIRES: OPENAI_API_KEY for the question index (falls back to explaining)
RUN: python examples/14_chunk_dual_index.py [--limit 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

from ragkit import config, display, llm
from ragkit.corpus import Book, Chunk, load_book
from ragkit.embed import describe, get_embedder

STRATEGY = "dual_index"

# Reciprocal-rank-fusion constant. 60 is the value from the original paper
# (Cormack et al., 2009) and it is what everyone uses. Smaller trusts rank 1
# more; larger flattens the two lists toward a round-robin merge.
RRF_K = 60


def build_dual_index(book: Book, limit: int | None = None) -> tuple[list[Chunk], list[dict]]:
    """Return the canonical chunks and the question entries pointing at them."""
    derived = import_module("11_chunk_question_derived")
    chunks = import_module("05_chunk_heading").chunk_by_heading(book)
    if limit:
        chunks = chunks[:limit]

    for chunk in chunks:
        chunk.strategy = STRATEGY

    questions: list[dict] = []
    for position, chunk in enumerate(chunks):
        print(f"    generating questions {position + 1}/{len(chunks)}", end="\r", flush=True)
        for question in derived.generate_questions(chunk.text):
            # A question entry is a pointer, not a copy. Storing the chunk
            # text here would double the storage and, worse, let the two
            # indexes drift apart when a document is updated.
            questions.append({"question": question, "chunk_id": position})
    print(" " * 50, end="\r")
    print()

    return chunks, questions


def fuse(
    passage_ranking: list[tuple[int, float]],
    question_ranking: list[tuple[int, float]],
    k: int = 5,
) -> list[dict]:
    """Reciprocal rank fusion over the two indexes.

    Scores are never consulted -- only positions -- which is why the two arms
    need no normalisation despite measuring different things.
    """
    # ---- THE EASY MISTAKE ------------------------------------------
    # Collapse the question side to its BEST rank per chunk before
    # fusing. Skip this and a chunk with four matching questions
    # accumulates four RRF contributions and wins on multiplicity rather
    # than on relevance -- which looks like a great result and is not.
    # --- collapse the question arm to one entry -- book:chunk-dual-index
    best_question_rank: dict[int, int] = {}
    for rank, (chunk_id, _) in enumerate(question_ranking, start=1):
        if chunk_id not in best_question_rank:
            best_question_rank[chunk_id] = rank
    # ------------------------------------------------------------- /book

    passage_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(passage_ranking, start=1)}

    scored: dict[int, dict] = {}
    for chunk_id in set(passage_rank) | set(best_question_rank):
        p_rank = passage_rank.get(chunk_id)
        q_rank = best_question_rank.get(chunk_id)
        # ---- THE KEY LINES -----------------------------------------
        # Reciprocal rank fusion. Note what is absent: the scores. Only
        # POSITIONS are used, which is why the two arms need no
        # normalisation despite being measured on different things.
        #
        # A chunk missing from an arm contributes exactly 0 for it.
        # --- ...continued ------------------------------ book:chunk-dual-index
        score = 0.0
        if p_rank:
            score += 1.0 / (RRF_K + p_rank)
        if q_rank:
            score += 1.0 / (RRF_K + q_rank)
        # ------------------------------------------------------------- /book
        scored[chunk_id] = {
            "chunk_id": chunk_id,
            "score": score,
            "passage_rank": p_rank,
            "question_rank": q_rank,
        }

    return sorted(scored.values(), key=lambda row: -row["score"])[:k]


def explain_without_key(book: Book) -> None:
    chunks = import_module("05_chunk_heading").chunk_by_heading(book)
    display.para(
        "Without a key the question index cannot be built. The structure is "
        "the lesson, and it is independent of what generated the questions:"
    )
    print()
    print("      QUESTION INDEX                    CANONICAL INDEX")
    print("      ---------------------------       ---------------------------")
    print("      'Why was the League a hoax?'  ->  [chunk 41: the passage]")
    print("      'Who hired Jabez Wilson?'     ->  [chunk 41]")
    print("      'What was in the cellar?'     ->  [chunk 42: the passage]")
    print()
    display.para(
        "A query searches both. Question hits are mapped to their chunk ids, "
        "collapsed to one entry per chunk, and fused with the passage hits by "
        "reciprocal rank. The fuse() function in this file does that part and "
        "needs no model at all -- it is pure arithmetic over two ranked lists."
    )
    print()
    display.kv("canonical index", f"{len(chunks):,} vectors")
    display.kv("question index", f"~{len(chunks) * 4:,} vectors")
    display.kv("stored text", "once -- question entries are pointers")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--query", default="Why was the Red-Headed League invented?")
    args = parser.parse_args()

    display.banner(
        "14. Dual-Index Question-Referenced Chunking",
        "Chapter 5, Chunking Strategies -- section 3.13",
        "Two indexes, one corpus: generated questions and the passages "
        "themselves, fused at query time by reciprocal rank.",
    )

    book = load_book()

    if not llm.available():
        print(f"  No LLM configured: {llm.why_unavailable()}\n")
        explain_without_key(book)
        display.notice(
            "Two indexes beat one compromise. Question-phrased queries match "
            "the question index; keyword and quotation queries match the "
            "passage index.",
            "RRF fuses by rank, not score. Cosine against a 12-token question "
            "and cosine against a 300-token passage are not comparable "
            "numbers, and averaging them would be meaningless.",
            "Question entries store a chunk id, not a copy of the text. Two "
            "indexes, one source of truth.",
            "This completes the book's thirteen chunking strategies. Example "
            "15 scores all of them against the same questions.",
        )
        return 0

    if llm.STUB:
        display.stub_warning()
    else:
        print(f"  model: {config.OPENAI_CHAT_MODEL}")
    print()

    chunks, questions = build_dual_index(book, args.limit)
    embedder = get_embedder()
    print(f"  {describe(embedder)}")

    display.kv("canonical index", f"{len(chunks):,} vectors")
    display.kv("question index", f"{len(questions):,} vectors")

    # Embed both indexes and the query.
    passage_vectors = embedder.encode([c.text for c in chunks], show_progress=False)
    question_vectors = embedder.encode([q["question"] for q in questions], show_progress=False)
    query_vector = embedder.encode([args.query], show_progress=False)[0]

    passage_scores = passage_vectors @ query_vector
    question_scores = question_vectors @ query_vector

    passage_ranking = [
        (int(i), float(passage_scores[i])) for i in np.argsort(-passage_scores)[:20]
    ]
    question_ranking = [
        (questions[int(i)]["chunk_id"], float(question_scores[i]))
        for i in np.argsort(-question_scores)[:20]
    ]

    display.heading(f"Query: {args.query!r}")

    print("  Top 5 from the CANONICAL index (passage similarity):")
    display.table(
        ["rank", "chunk", "score", "text"],
        [
            [rank, cid, f"{score:.3f}", display.truncate(chunks[cid].text, 40)]
            for rank, (cid, score) in enumerate(passage_ranking[:5], start=1)
        ],
        align="rrrl",
    )

    print("\n  Top 5 from the QUESTION index:")
    seen: set[int] = set()
    rows = []
    for rank, (cid, score) in enumerate(question_ranking, start=1):
        if cid in seen:
            continue
        seen.add(cid)
        question = next(q["question"] for q in questions if q["chunk_id"] == cid)
        rows.append([rank, cid, f"{score:.3f}", display.truncate(question, 40)])
        if len(rows) == 5:
            break
    display.table(["rank", "chunk", "score", "matched question"], rows, align="rrrl")

    fused = fuse(passage_ranking, question_ranking, k=5)
    print("\n  FUSED (reciprocal rank, k=%d):" % RRF_K)
    display.table(
        ["rank", "chunk", "RRF", "passage rank", "question rank", "text"],
        [
            [
                rank,
                row["chunk_id"],
                f"{row['score']:.4f}",
                row["passage_rank"] or "-",
                row["question_rank"] or "-",
                display.truncate(chunks[row["chunk_id"]].text, 30),
            ]
            for rank, row in enumerate(fused, start=1)
        ],
        align="rrrrrl",
    )

    only_one = [r for r in fused if not (r["passage_rank"] and r["question_rank"])]
    print(f"\n  {len(only_one)} of the top {len(fused)} were found by only one index.")
    print("  Those are the results a single-index system would have missed.")

    display.sample_chunks(chunks, n=4, start=0)

    display.notice(
        f"{len(questions):,} question vectors and {len(chunks):,} passage "
        "vectors over the same text, with the text stored once.",
        "RRF fuses on rank because the two similarity scores are not on a "
        "comparable scale. The same reasoning drives the hybrid fusion in "
        "example 19, where the mismatch is even wider.",
        "The question side is collapsed to one entry per chunk before fusing. "
        "Skip that and a chunk wins by having four matching questions rather "
        "than by being the right answer.",
        "That completes the thirteen chunking strategies. Example 15 runs all "
        "of them against the same gold questions and reports which ones "
        "actually retrieve better.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
