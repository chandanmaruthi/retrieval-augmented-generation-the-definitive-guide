#!/usr/bin/env python3
"""
17. Baseline RAG Pipeline
Book: Chapter 7, Baseline RAG Pipeline

WHAT THIS SHOWS: The whole pipeline, end to end, with nothing clever in it.

    query -> embed -> nearest neighbours -> assemble context -> generate

Every strategy in the rest of the book is a modification of one of those five
steps. This is the thing they are all trying to beat, so it is worth seeing
plainly, and worth measuring, because a surprising number of elaborations turn
out not to beat it.

HOW THIS SCRIPT PROCEEDS
    1. Embed the query
    2. Retrieve the nearest k
    3. Assemble a context      <-- dedup + budget + citations happen HERE
    4. Generate                <-- note the abstain instruction in SYSTEM
    5. Report where the time went

Five steps. Every later chapter in the book modifies exactly one of them, so
it is worth knowing which is which before reading any of them.


HOW THE LATER EXAMPLES CHANGE THIS
    step 1   example 22 rewrites the query first; example 30 adds memory
    step 2   examples 18-21, 24-26 replace or augment the retriever
    2 and 3  example 20 inserts a reranking stage between them
    2-4      examples 27, 28 wrap the whole thing in a loop
    step 3   example 35 enforces access control BEFORE it, not after


REQUIRES: Postgres. OPENAI_API_KEY optional -- without it you get the
retrieval and the assembled prompt, just not the generated answer.
RUN: python examples/17_baseline_rag.py ["your question"]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import config, display, llm, pg, tokens
from ragkit.embed import describe, get_embedder

DEFAULT_QUESTION = "What was the speckled band?"

# The abstain instruction is the most important line in this prompt. Without
# it the model will answer from its own knowledge of Sherlock Holmes when
# retrieval fails, and a confident wrong answer that looks grounded is worse
# than an admission of ignorance.
SYSTEM = """You answer questions using only the numbered context passages provided.

Rules:
- Use only the passages. Do not use anything you know about these stories.
- Cite the passages you used as [1], [2] and so on.
- If the passages do not contain the answer, say exactly:
  "The retrieved passages do not answer this question."
"""

PROMPT = """Context passages:

{context}

Question: {question}

Answer using only the passages above, with citations."""


def assemble_context(hits, budget: int = 2000) -> tuple[str, list]:
    """Turn ranked chunks into a numbered context block.

    Three decisions live in this function, and the book's reference
    architecture calls all three out as part of block 5 rather than retrieval:

    * **deduplication** -- parent-child and sentence-window strategies return
      overlapping context, so the same passage can arrive twice.
    * **budget** -- the context window is finite; spend it in rank order.
    * **citation binding** -- each passage gets a number and a source, so the
      answer can point at something checkable.
    """
    # =================================================================
    # THREE DECISIONS, all of which the book's reference architecture
    # puts in the GENERATION block rather than in retrieval:
    #   dedup    -- parent-child strategies return overlapping context
    #   budget   -- the window is finite; spend it in rank order
    #   citation -- bind numbers to sources while the mapping still exists
    # =================================================================
    parts: list[str] = []
    used = []
    spent = 0
    seen: set[str] = set()

    for hit in hits:
        text = hit.chunk.context_text
        key = text[:200]
        if key in seen:
            continue
        cost = tokens.count_tokens(text)
        if spent + cost > budget:
            continue
        seen.add(key)
        used.append(hit)
        parts.append(f"[{len(used)}] ({hit.chunk.location}) {text}")
        spent += cost

    return "\n\n".join(parts), used


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "17. Baseline RAG Pipeline",
        "Chapter 7, Baseline RAG Pipeline",
        "Embed, retrieve the nearest k, assemble a context, generate. The "
        "simplest thing that works, and the baseline everything else is "
        "measured against.",
    )

    if not pg.available():
        display.missing_database(
            "The baseline pipeline retrieves from the pgvector index built by "
            "scripts/index_corpus.py."
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
        print(f"\nNothing indexed for strategy {args.strategy!r}.")
        print(f"Run:  python scripts/index_corpus.py --strategy {args.strategy}\n")
        return 1

    print(f"  {describe(embedder)}")
    print(f"  strategy: {args.strategy} ({store.count(args.strategy):,} chunks)")
    print(f"  question: {args.question!r}\n")

    # --- step 1: embed the query ----------------------------------------
    display.heading("1. Embed the query")
    started = time.time()
    query_vector = embedder.encode([args.question], show_progress=False)[0]
    embed_ms = (time.time() - started) * 1000
    display.kv("vector", f"{len(query_vector)} dimensions")
    display.kv("time", f"{embed_ms:.0f} ms")

    # --- step 2: retrieve -------------------------------------------------
    display.heading("2. Retrieve the nearest neighbours")
    started = time.time()
    hits = store.search(query_vector, k=args.k, strategy=args.strategy)
    search_ms = (time.time() - started) * 1000
    display.kv("time", f"{search_ms:.0f} ms")
    print()
    display.table(
        ["rank", "score", "story", "passage"],
        [
            [h.rank, f"{h.score:.3f}", display.truncate(h.chunk.story, 26),
             display.truncate(h.chunk.text, 34)]
            for h in hits
        ],
        align="rrll",
    )

    # --- step 3: assemble -------------------------------------------------
    display.heading("3. Assemble the context")
    context, used = assemble_context(hits)
    display.kv("passages used", f"{len(used)} of {len(hits)} retrieved")
    display.kv("context tokens", f"{tokens.count_tokens(context):,}")
    if len(used) < len(hits):
        print("\n  Some were dropped as duplicates or over budget.")

    # --- step 4: generate -------------------------------------------------
    display.heading("4. Generate")

    if not llm.available():
        print(f"  No LLM configured: {llm.why_unavailable()}")
        print("  Retrieval above is real. Here is the prompt it produced:\n")
        display.prompt_block("SYSTEM", SYSTEM)
        display.prompt_block(
            "USER",
            PROMPT.format(
                context="\n\n".join(
                    f"[{i+1}] ({h.chunk.location}) {display.truncate(h.chunk.context_text, 220)}"
                    for i, h in enumerate(used)
                ),
                question=args.question,
            ),
        )
    else:
        if llm.STUB:
            display.stub_warning()
        started = time.time()
        answer = llm.complete(
            PROMPT.format(context=context, question=args.question),
            system=SYSTEM,
            purpose="the baseline RAG answer",
        )
        generate_ms = (time.time() - started) * 1000
        print()
        display.para(answer, indent="  ")
        print()
        display.kv("time", f"{generate_ms:.0f} ms")
        print("\n  Sources:")
        for index, hit in enumerate(used, start=1):
            print(f"      [{index}] {hit.chunk.location}")

    display.heading("Where the time went")
    display.table(
        ["step", "ms"],
        [["embed query", f"{embed_ms:.0f}"], ["vector search", f"{search_ms:.0f}"]],
    )
    print("\n  Search is over a sequential scan -- there is no ANN index on this")
    print(f"  table. At {store.count(args.strategy):,} rows that is the faster choice,")
    print("  and example 16 measures it rather than assuming.")

    display.notice(
        "Five steps, and every later chapter changes one of them. Hybrid "
        "retrieval changes step 2, reranking inserts a step between 2 and 3, "
        "agentic RAG loops 2-4, and memory rewrites step 1.",
        "The abstain instruction in the system prompt is doing real work. "
        "Without it, a model that knows these stories will answer from memory "
        "when retrieval fails, and you will not be able to tell.",
        "Deduplication and budgeting happen in step 3, not step 2. The book's "
        "reference architecture puts them in the generation block for a "
        "reason: they are about what the model reads, not about what ranks.",
        "Citations are bound at assembly time, when the mapping from number "
        "to source still exists. Trying to recover it from the answer text "
        "afterwards does not work.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
