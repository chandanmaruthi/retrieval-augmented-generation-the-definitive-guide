#!/usr/bin/env python3
"""
15. Comparing the Chunking Strategies
Book: Chapter 5, Chunking Strategies
WHAT THIS SHOWS: All of the book's chunking strategies, scored against the
same thirty gold questions, on the same corpus, with the same embedder.

The book claims that "chunking quality is a stronger predictor of RAG accuracy
than embedding model choice". This example is where that claim either survives
contact with a measurement or does not.

HOW SCORING WORKS: each gold question carries a literal phrase from the
corpus. A retrieval counts as a hit when the text the model would actually
receive contains that phrase. Scoring on the *delivered context* rather than
the matched chunk is what makes strategies like parent-child and
sentence-window comparable at all -- they deliberately retrieve something
smaller than they return.

HOW THIS SCRIPT PROCEEDS
    1. Load the 30 gold questions
    2. For each strategy, run ITS OWN CODE from examples/   <-- see below
    3. Embed, index in NumPy, retrieve top-k
    4. Score at fixed k          -- the obvious comparison
    5. Score at a fixed CONTEXT BUDGET  <-- THE IMPORTANT ONE
    6. Report how the ranking changed between 4 and 5

Step 2 loads each example script by filename and calls its function, so this
table always measures the code you just read. Change example 04 and this
output changes.


WHY THERE ARE TWO TABLES
    Step 4 is how retrieval strategies are usually compared, and on this
    corpus it gives the wrong answer. A strategy returning 1,000-token parents
    delivers eight times as much text at k=5 as one returning 120-token
    paragraphs, so it scores higher partly for being better and partly just
    for being bigger.

    Step 5 gives every strategy the same number of context tokens. The
    ranking changes substantially -- parent_child goes from first to last --
    and that reversal is the most useful thing in this file.


REQUIRES: -r requirements-local.txt
RUN: python examples/15_compare_chunking.py [--k 5] [--include-llm]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

from ragkit import display, llm, tokens
from ragkit.corpus import Chunk, load_book
from ragkit.embed import describe, get_embedder
from ragkit.store import NumpyStore

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"

# Strategies that need nothing but text and an embedder.
FREE_STRATEGIES = [
    ("fixed_token", "02_chunk_fixed_token", "chunk_fixed_token", "3.1"),
    ("sentence", "03_chunk_sentence", "chunk_sentences", "3.2"),
    ("paragraph", "04_chunk_paragraph", "chunk_paragraphs", "3.3"),
    ("heading", "05_chunk_heading", "chunk_by_heading", "3.4"),
    ("semantic", "06_chunk_semantic", None, "3.5"),  # different signature
    ("sentence_window", "07_chunk_sentence_window", "chunk_sentence_window", "3.6"),
    ("parent_child", "08_chunk_parent_child", None, "3.7"),  # returns a tuple
    ("contextual_header", "09_chunk_contextual_header", "chunk_with_headers", "3.8"),
]

# Strategies that spend API calls at index time.
LLM_STRATEGIES = [
    ("context_buffered", "10_chunk_context_buffered", "3.9"),
    ("question_derived", "11_chunk_question_derived", "3.10"),
    ("question_anchored", "12_chunk_question_anchored", "3.11"),
    ("qa_context_buffered", "13_chunk_qa_context_buffered", "3.12"),
]


def load_questions() -> list[dict]:
    if not EVAL_PATH.exists():
        raise SystemExit(
            f"Gold questions missing at {EVAL_PATH}.\n"
            "Run:  python scripts/build_eval_set.py"
        )
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))["questions"]


def build_strategy(name: str, module_name: str, function_name: str | None, book, embedder):
    """Run one strategy's own code, exactly as the example script defines it.

    Loaded by file rather than reimplemented here, so this comparison is
    guaranteed to be measuring the code the reader just read. If example 04
    changes, this table changes with it.
    """
    module = import_module(module_name)

    if name == "semantic":
        chunks, _ = module.chunk_semantic(book, embedder)
        return chunks
    if name == "parent_child":
        chunks, _ = module.chunk_parent_child(book)
        return chunks
    return getattr(module, function_name)(book)


def evaluate_budget(
    chunks: list[Chunk], questions: list[dict], embedder, budget: int
) -> dict:
    """Score under a fixed context budget instead of a fixed k.

    Every strategy gets the same number of context tokens, taking chunks in
    rank order until the budget is spent -- which is also what a real system
    does, since the constraint is the model's context window rather than a
    round number of chunks. See WHY THERE ARE TWO TABLES at the top of this
    file.
    """
    vectors = embedder.encode([c.embed_text for c in chunks], show_progress=False)
    store = NumpyStore()
    store.add(chunks, vectors)

    query_vectors = embedder.encode([q["question"] for q in questions], show_progress=False)

    hits = 0
    used_tokens = []
    used_chunks = []

    for question, query_vector in zip(questions, query_vectors):
        # Ask for plenty of candidates, then spend the budget on them.
        results = store.search(query_vector, k=40)

        delivered_parts: list[str] = []
        spent = 0
        seen: set[str] = set()

        for result in results:
            context = result.chunk.context_text
            # Parent-child and sentence-window return overlapping context, so
            # the same parent can arrive several times. Charging the budget
            # twice for text the model sees once would understate them.
            key = context[:200]
            if key in seen:
                continue
            cost = tokens.count_tokens(context)
            if spent + cost > budget:
                continue
            seen.add(key)
            delivered_parts.append(context)
            spent += cost

        # ---- THE SCORING RULE --------------------------------------
        # Scored on the text the model would actually RECEIVE, not on the
        # chunk that matched. Without this, parent-child and
        # sentence-window chunking would be penalised for deliberately
        # retrieving something smaller than they return.
        delivered = " ".join(" ".join(delivered_parts).split())
        if question["evidence"] in delivered:
            hits += 1
        used_tokens.append(spent)
        used_chunks.append(len(delivered_parts))

    return {
        "recall": hits / len(questions),
        "tokens": float(np.mean(used_tokens)),
        "chunks_used": float(np.mean(used_chunks)),
    }


def evaluate(
    chunks: list[Chunk], questions: list[dict], embedder, k: int
) -> dict:
    """Index the chunks, run every question, and score the delivered context."""
    # What gets embedded is the strategy's choice -- a bare sentence, a
    # header-prefixed passage, a generated question. Chunk.embed_text encodes
    # that decision, so honouring it here is what keeps the comparison fair.
    # ---- THE FAIRNESS RULE -----------------------------------------
    # Embed `embed_text`, not `text`. Each strategy chose what it wants
    # indexed -- a bare sentence, a header-prefixed passage, a generated
    # question -- and honouring that choice is what makes this a
    # comparison of strategies rather than of one thing done eight ways.
    started = time.time()
    vectors = embedder.encode([c.embed_text for c in chunks], show_progress=False)
    embed_seconds = time.time() - started

    store = NumpyStore()
    store.add(chunks, vectors)

    query_vectors = embedder.encode([q["question"] for q in questions], show_progress=False)

    hits_at_k = 0
    reciprocal_ranks = []
    context_tokens = []

    for question, query_vector in zip(questions, query_vectors):
        results = store.search(query_vector, k=k)
        evidence = question["evidence"]

        rank_of_hit = None
        for result in results:
            # context_text is what the generator would receive: the parent for
            # parent-child, the window for sentence-window, the chunk itself
            # otherwise. Normalise whitespace so a phrase spanning a line
            # break still matches.
            delivered = " ".join(result.chunk.context_text.split())
            if evidence in delivered:
                rank_of_hit = result.rank
                break

        if rank_of_hit:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank_of_hit)
        else:
            reciprocal_ranks.append(0.0)

        context_tokens.append(
            sum(tokens.count_tokens(r.chunk.context_text) for r in results)
        )

    return {
        "chunks": len(chunks),
        "recall_at_k": hits_at_k / len(questions),
        "mrr": float(np.mean(reciprocal_ranks)),
        "context_tokens": float(np.mean(context_tokens)),
        "embed_seconds": embed_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="how many chunks to retrieve")
    parser.add_argument(
        "--budget",
        type=int,
        default=2000,
        help="context tokens each strategy may spend in the fair comparison",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="also score the strategies that spend API calls (slow, costs money)",
    )
    args = parser.parse_args()

    display.banner(
        "15. Comparing the Chunking Strategies",
        "Chapter 5, Chunking Strategies",
        f"Every strategy, the same 30 gold questions, the same embedder. "
        f"Scored on whether the text delivered at top-{args.k} contains the "
        "answer.",
    )

    try:
        embedder = get_embedder()
    except ImportError:
        display.missing_dependency(
            "sentence-transformers",
            "requirements-local.txt",
            "Comparing retrieval quality needs real embeddings. With the hash "
            "fallback every strategy would score badly and identically, which "
            "would tell you nothing.",
        )
        return 0

    print(f"  {describe(embedder)}")
    if not embedder.semantic:
        display.para(
            "The hash embedder cannot match a question to a passage that "
            "answers it in different words, so these numbers would compare "
            "vocabulary overlap, not retrieval. Install -r "
            "requirements-local.txt before drawing conclusions."
        )
        return 0

    questions = load_questions()
    print(f"  {len(questions)} gold questions across 12 stories")
    print(f"  retrieving top-{args.k}\n")

    book = load_book()

    strategies = list(FREE_STRATEGIES)
    if args.include_llm:
        if llm.available():
            strategies += [(n, m, None, s) for n, m, s in LLM_STRATEGIES]
        else:
            print(f"  skipping LLM strategies: {llm.why_unavailable()}\n")

    rows = []
    for name, module_name, function_name, section in strategies:
        print(f"  running {name}...", end=" ", flush=True)
        try:
            chunks = build_strategy(name, module_name, function_name, book, embedder)
            result = evaluate(chunks, questions, embedder, args.k)
            result["budget"] = evaluate_budget(chunks, questions, embedder, args.budget)
        except Exception as error:  # noqa: BLE001 - one bad strategy must not
            print(f"FAILED ({type(error).__name__}: {error})")  # kill the table
            continue
        print(f"{result['chunks']:,} chunks, recall {result['recall_at_k']:.2f}")
        rows.append((name, section, result))

    if not rows:
        print("\n  No strategy completed. Nothing to compare.")
        return 1

    rows.sort(key=lambda row: (-row[2]["recall_at_k"], -row[2]["mrr"]))

    display.heading(f"Results (top-{args.k}, {len(questions)} questions)")
    display.table(
        ["strategy", "book", "chunks", f"recall@{args.k}", "MRR", "ctx tokens"],
        [
            [
                name,
                f"§{section}",
                f"{result['chunks']:,}",
                f"{result['recall_at_k']:.2f}",
                f"{result['mrr']:.3f}",
                f"{result['context_tokens']:.0f}",
            ]
            for name, section, result in rows
        ],
    )
    print(f"\n  recall@{args.k}  how often the answer was in the delivered context")
    print( "  MRR         1/rank of the first correct chunk, averaged")
    print( "  ctx tokens  what the generator is asked to read per query")

    spread = rows[0][2]["context_tokens"] / rows[-1][2]["context_tokens"]
    print(f"\n  Before reading that ranking: the top strategy delivers "
          f"{spread:.1f}x more")
    print("  text per query than the bottom one. Some of its advantage is just")
    print("  volume. The next table removes that.")

    # -----------------------------------------------------------------
    # The fair comparison
    # -----------------------------------------------------------------
    budget_rows = sorted(rows, key=lambda row: -row[2]["budget"]["recall"])

    display.heading(f"Same question, equal context budget ({args.budget} tokens)")
    print("  Every strategy may now spend the same number of context tokens,")
    print("  taking chunks in rank order until the budget runs out. This is")
    print("  the comparison that isolates chunking quality from chunk size.\n")
    display.table(
        ["strategy", "book", f"recall@{args.budget}tok", "tokens used", "chunks used"],
        [
            [
                name,
                f"§{section}",
                f"{result['budget']['recall']:.2f}",
                f"{result['budget']['tokens']:.0f}",
                f"{result['budget']['chunks_used']:.1f}",
            ]
            for name, section, result in budget_rows
        ],
    )

    display.heading("How the ranking changed")
    top_k_order = [name for name, _, _ in rows]
    budget_order = [name for name, _, _ in budget_rows]
    moved = [
        (name, top_k_order.index(name) + 1, budget_order.index(name) + 1)
        for name in top_k_order
        if top_k_order.index(name) != budget_order.index(name)
    ]
    if moved:
        display.table(
            ["strategy", f"rank at top-{args.k}", "rank at equal budget", "move"],
            [
                [name, before, after, f"{before - after:+d}"]
                for name, before, after in moved
            ],
        )
    else:
        print("  No strategy changed position.")

    best_k = rows[0]
    best_budget = budget_rows[0]

    display.notice(
        f"At top-{args.k}, {best_k[0]} wins with recall "
        f"{best_k[2]['recall_at_k']:.2f}. At an equal context budget, "
        f"{best_budget[0]} wins with {best_budget[2]['budget']['recall']:.2f}. "
        "If those differ, the first table was partly measuring chunk size.",
        f"The top-k winner delivered {spread:.1f}x more text per query than "
        "the bottom one. Comparing retrieval strategies at fixed k without "
        "checking that is one of the easiest ways to draw a confident wrong "
        "conclusion.",
        "Every number came from running the example scripts' own functions, "
        "not a reimplementation. Change example 04 and this table changes.",
        "Thirty questions on one 104,000-word book of Victorian detective "
        "fiction. These rankings are evidence about this corpus, not a general "
        "law -- which is exactly why the book tells you to measure your own.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
