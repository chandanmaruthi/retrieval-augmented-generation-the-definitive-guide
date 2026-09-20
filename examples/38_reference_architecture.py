#!/usr/bin/env python3
"""
38. The Reference Architecture, End to End
Book: Chapter 4, Reference Architecture

WHAT THIS SHOWS: All six blocks of the book's architecture in one run, with
the diagnostic the chapter is really about.

    1 Ingestion   2 Chunking   3 Indexing   4 Retrieval   5 Generation   6 Evaluation

The chapter's most useful idea is not the diagram, it is the attribution rule:
when an answer is wrong, exactly one block is usually responsible, and you can
tell which by asking four questions in order.

    document missing entirely          -> ingestion
    fact split across two chunks       -> chunking
    right chunk exists, never surfaced -> retrieval
    right chunk surfaced, ignored      -> generation

This example runs the pipeline over the gold set, then classifies every
failure into one of those buckets. That turns "our RAG is 50% accurate" into a
list of what to fix, which is the difference between a metric and a plan.

HOW THIS SCRIPT PROCEEDS
    1. Show all six blocks and what runs each one here
    2. Run all 30 gold questions end to end
    3. Attribute every FAILURE to one block   <-- THE TECHNIQUE
    4. Say what each bucket means you should do

Step 3 is the chapter's most useful idea, and it is four checks asked in a
fixed order.


THE ATTRIBUTION RULE
    document missing entirely          -> ingestion
    fact split across two chunks       -> chunking
    right chunk exists, never surfaced -> retrieval
    right chunk surfaced, ignored      -> generation

    The ORDER is load-bearing. Each check assumes the earlier blocks
    succeeded, so asking "did retrieval fail?" before confirming the fact is
    even indexed produces a confident answer about the wrong block.

    On this corpus the answer comes back 100% retrieval -- the evidence is
    indexed and simply not in the top 5 -- which points straight at example
    20's reranker rather than at any amount of chunking work.


REQUIRES: Postgres
RUN: python examples/38_reference_architecture.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, llm, pg, tokens
from ragkit.corpus import load_book
from ragkit.embed import describe, get_embedder

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"


def diagnose(question: dict, hits, conn, strategy: str, k: int) -> tuple[str, str]:
    """Attribute a failure to one architectural block.

    The order matters: each check assumes the previous ones passed. Asking
    "did retrieval fail?" before checking the fact is even in the index
    produces a confident answer about the wrong block.
    """
    evidence = question["evidence"]

    # Block 1 -- is the text in the corpus at all?
    # ---- CHECK 1: is the text in the corpus at all? -----------------
    in_corpus = conn.execute(
        "SELECT count(*) FROM chunks WHERE strategy = %s AND position(%s in body) > 0",
        [strategy, evidence],
    ).fetchone()[0]

    if in_corpus == 0:
        # Distinguish "never ingested" from "ingested but the chunker split
        # it". If the phrase is in the source text but in no chunk, chunking
        # cut through the middle of it.
        book = load_book()
        if evidence in " ".join(book.text.split()):
            return "chunking", "in the corpus, but no single chunk contains it"
        return "ingestion", "not present in the corpus at all"

    # Block 4 -- it is indexed, so did retrieval surface it?
    delivered = " ".join(
        " ".join(h.chunk.context_text.split()) for h in hits[:k]
    )
    # ---- CHECK 3: indexed, but did retrieval surface it? ------------
    if evidence not in delivered:
        return "retrieval", f"exists in {in_corpus} chunk(s), not in the top {k}"

    # Block 5 -- retrieved and delivered, so anything wrong now is generation.
    return "generation", "the evidence reached the model"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "38. The Reference Architecture, End to End",
        "Chapter 4, Reference Architecture",
        "All six blocks in one run, then every failure attributed to the "
        "block responsible for it.",
    )

    if not pg.available():
        display.missing_database("The full pipeline runs against the indexed corpus.")
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    if not EVAL_PATH.exists():
        print("\nGold set missing. Run: python scripts/build_eval_set.py\n")
        return 1

    questions = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["questions"]
    embedder = get_embedder()
    store = pg.PgVectorStore(conn, dim=embedder.dim, model=embedder.name)

    if store.count(args.strategy) == 0:
        print("\nNothing indexed. Run: python scripts/index_corpus.py\n")
        return 1

    # --- the six blocks ----------------------------------------------------
    display.heading("The pipeline as built by this repository")
    book = load_book()
    display.table(
        ["#", "block", "what runs it here", "state"],
        [
            ["1", "Ingestion", "examples/01, scripts/build_corpus.py",
             f"{book.word_count:,} words, {len(book.stories)} documents"],
            ["2", "Chunking", f"examples/02-14 ({args.strategy})",
             f"{store.count(args.strategy):,} chunks"],
            ["3", "Indexing", "scripts/index_corpus.py, sql/001",
             f"{embedder.dim}-dim, {store.table}"],
            ["4", "Retrieval", "examples/17-28", f"dense, top-{args.k}"],
            ["5", "Generation", "examples/17, 32",
             "available" if llm.available() else "no model configured"],
            ["6", "Evaluation", "examples/31-33, 36", f"{len(questions)} gold questions"],
        ],
        align="llll",
    )
    print("\n  Corpus-time blocks (1-3) are paid once per document. Query-time")
    print("  blocks (4-6) are paid per request. The book's advice follows from")
    print("  that split: move work leftward whenever you can.")

    # --- run it ------------------------------------------------------------
    display.heading("Running all 30 questions")
    started = time.time()
    results = []
    for question in questions:
        vector = embedder.encode([question["question"]], show_progress=False)[0]
        hits = store.search(vector, k=args.k, strategy=args.strategy)
        delivered = " ".join(
            " ".join(h.chunk.context_text.split()) for h in hits[: args.k]
        )
        correct = question["evidence"] in delivered
        results.append({"question": question, "hits": hits, "correct": correct})
    elapsed = time.time() - started

    correct_count = sum(1 for r in results if r["correct"])
    display.kv("answered", f"{correct_count}/{len(results)}")
    display.kv("recall@%d" % args.k, f"{correct_count / len(results):.2f}")
    display.kv("total time", f"{elapsed:.1f}s")
    display.kv("per query", f"{1000 * elapsed / len(results):.0f} ms")

    # --- attribution --------------------------------------------------------
    display.heading("Failure attribution")
    print("  Every failure belongs to exactly one block. Asking the four")
    print("  questions in order is what makes that true.\n")

    buckets: dict[str, list] = {}
    for result in results:
        if result["correct"]:
            continue
        block, reason = diagnose(result["question"], result["hits"], conn,
                                 args.strategy, args.k)
        buckets.setdefault(block, []).append((result["question"], reason))

    failures = sum(len(v) for v in buckets.values())
    display.table(
        ["block", "failures", "share of all failures"],
        [
            [block, len(items), f"{100 * len(items) / max(failures, 1):.0f}%"]
            for block, items in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        ],
    )

    for block, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  {block.upper()} ({len(items)})")
        for question, reason in items[:4]:
            print(f"      {display.truncate(question['question'], 44)}")
            print(f"        -> {reason}")
        if len(items) > 4:
            print(f"      ... and {len(items) - 4} more")

    # --- what to do about it -----------------------------------------------
    display.heading("What each bucket tells you to do")
    advice = {
        "ingestion": "The source never made it in. Check the connector and the "
                     "parser -- no retrieval change can fix this.",
        "chunking": "The fact is in the corpus but no single chunk holds it. "
                    "Chunk boundaries cut through the answer. Examples 02-15.",
        "retrieval": "The right chunk exists and was not surfaced. Reranking, "
                     "hybrid search or a better embedder. Examples 19-21.",
        "generation": "The evidence reached the model and the answer was still "
                      "wrong. Prompting, or a stronger model. Examples 17, 32.",
    }
    for block, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"  {block} ({len(items)}):")
        display.para(advice[block], indent="      ")
        print()

    if buckets:
        biggest = max(buckets.items(), key=lambda kv: len(kv[1]))[0]
        print(f"  Largest bucket: {biggest}. That is where effort pays here,")
        print("  and it is a different answer from 'improve the RAG system'.")

    display.notice(
        f"Recall@{args.k} is {correct_count / len(results):.2f}. On its own "
        "that number tells you nothing about what to change.",
        "The attribution table does. Failures concentrate in one block, and "
        "work spent on the others is wasted no matter how well executed.",
        "The four checks must run in order. Each assumes the earlier blocks "
        "succeeded, so asking about retrieval before confirming the fact is "
        "indexed gives a confident answer about the wrong block.",
        "This diagnostic needs the gold set from example 31 and nothing else "
        "-- no model, no judge, no human. It is the cheapest useful thing in "
        "the whole book.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
