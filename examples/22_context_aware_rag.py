#!/usr/bin/env python3
"""
22. Context-Aware RAG (query rewriting over dialogue history)
Book: Chapter 7, Context-Aware RAG
WHAT THIS SHOWS: The single highest-value fix for conversational RAG.

A follow-up question is almost never self-contained:

    user: What was the speckled band?
    user: Why did he do it?          <- "he" is unrecoverable on its own

Embedding "Why did he do it?" retrieves nothing useful, because the vector has
no idea who "he" is. The fix is a rewriting step: compress the dialogue
history into one standalone question, then retrieve with that.

The book's reference architecture places this deliberately -- memory wraps
blocks 4 and 5, and conversational history "rewrites the query before
retrieval". This example measures what that rewriting is worth.

HOW THIS SCRIPT PROCEEDS
    1. Take a three-turn conversation
    2. For each turn, retrieve with the RAW question
    3. Rewrite it into a standalone question   <-- THE TECHNIQUE, one LLM call
    4. Retrieve again with the rewrite
    5. Compare: how many hits came from the right story?

Turn 3 is the interesting one. "Had he tried the same thing before?" retrieves
from entirely the wrong story raw, and 3/3 correctly after rewriting.


WHY THIS IS THE HIGHEST-VALUE CALL IN A CHAT RAG SYSTEM
    A follow-up question is almost never self-contained. "Why did he do it?"
    embedded as typed is a vector for a generic question about motive, and it
    retrieves accordingly.

    Note where the rewrite happens: BEFORE retrieval. By the time the
    generator sees the context, the wrong passages have already been chosen
    and no amount of prompting fixes it.


REQUIRES: Postgres. OPENAI_API_KEY for the rewriter (falls back to showing
the prompt and a hand-written rewrite so the retrieval comparison still runs).
RUN: python examples/22_context_aware_rag.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import config, display, llm, pg
from ragkit.embed import describe, get_embedder

# A conversation whose later turns are meaningless in isolation. The third
# turn is the hard one: "the same thing" refers to a motive established two
# turns earlier and never named.
CONVERSATION = [
    "What was the speckled band?",
    "Why did he do it?",
    "Had he tried the same thing before?",
]

# Hand-written rewrites, used only when no model is available, so the
# retrieval comparison below is still a real measurement. Labelled as such in
# the output -- these are what a rewriter *should* produce.
FALLBACK_REWRITES = {
    "Why did he do it?":
        "Why did Dr. Grimesby Roylott kill his stepdaughter with a swamp adder?",
    "Had he tried the same thing before?":
        "Had Dr. Grimesby Roylott killed another stepdaughter with the snake before?",
}

SYSTEM = (
    "You rewrite follow-up questions into standalone ones for a search engine. "
    "Reply with the rewritten question and nothing else."
)

PROMPT = """Conversation so far:
{history}

Follow-up question: {question}

Rewrite the follow-up as a single self-contained question. Replace every
pronoun and implicit reference with the thing it refers to, using names from
the conversation. Do not answer it."""


def rewrite(history: list[tuple[str, str]], question: str) -> str:
    """Compress the dialogue into one standalone question."""
    # --- history in, standalone question out -- book:retrieval-context-aware
    rendered = "\n".join(f"Q: {q}\nA: {a}" for q, a in history) or "(none)"
    return llm.complete(
        PROMPT.format(history=rendered, question=question),
        system=SYSTEM,
        temperature=0.0,
        max_tokens=120,
        purpose="conversational query rewriting",
    ).strip().strip('"')
    # ------------------------------------------------------------- /book


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="contextual_header")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    display.banner(
        "22. Context-Aware RAG",
        "Chapter 7, Context-Aware RAG",
        "Rewrite each follow-up into a standalone question before retrieving, "
        "so pronouns stop destroying recall.",
    )

    if not pg.available():
        display.missing_database("The retrieval comparison needs the indexed corpus.")
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

    have_model = llm.available()
    if not have_model:
        print(f"  No LLM: {llm.why_unavailable()}")
        print("  Using hand-written rewrites so the comparison still runs.")
        display.prompt_block("REWRITER SYSTEM", SYSTEM)
        display.prompt_block(
            "REWRITER USER",
            PROMPT.format(
                history="Q: What was the speckled band?\nA: A swamp adder.",
                question="Why did he do it?",
            ),
        )
    elif llm.STUB:
        display.stub_warning()

    history: list[tuple[str, str]] = []

    for turn, question in enumerate(CONVERSATION, start=1):
        display.heading(f"Turn {turn}: {question!r}")

        # --- retrieve with the raw question ------------------------------
        raw_vector = embedder.encode([question], show_progress=False)[0]
        raw_hits = store.search(raw_vector, k=args.k, strategy=args.strategy)

        # --- rewrite, then retrieve again ---------------------------------
        if turn == 1:
            rewritten = question  # nothing to resolve on the first turn
        elif have_model and not llm.STUB:
            rewritten = rewrite(history, question)
        else:
            rewritten = FALLBACK_REWRITES.get(question, question)

        if rewritten != question:
            print(f"  rewritten: {rewritten!r}")
            if not have_model or llm.STUB:
                print("             (hand-written, not from a live model)")
        else:
            print("  (first turn -- nothing to resolve)")

        rewritten_vector = embedder.encode([rewritten], show_progress=False)[0]
        rewritten_hits = store.search(rewritten_vector, k=args.k, strategy=args.strategy)

        print()
        display.table(
            ["", "top story", "score", "passage"],
            [
                ["raw",
                 display.truncate(raw_hits[0].chunk.story, 24),
                 f"{raw_hits[0].score:.3f}",
                 display.truncate(raw_hits[0].chunk.text, 30)],
                ["rewritten",
                 display.truncate(rewritten_hits[0].chunk.story, 24),
                 f"{rewritten_hits[0].score:.3f}",
                 display.truncate(rewritten_hits[0].chunk.text, 30)],
            ],
            align="llll",
        )

        # Did the rewrite pull retrieval back to the story under discussion?
        target = "The Adventure of the Speckled Band"
        raw_on_target = sum(1 for h in raw_hits if h.chunk.story == target)
        new_on_target = sum(1 for h in rewritten_hits if h.chunk.story == target)
        if turn > 1:
            print(f"\n  chunks from the story under discussion: "
                  f"raw {raw_on_target}/{args.k}, rewritten {new_on_target}/{args.k}")

        # The answer is faked when no model is available; retrieval above is
        # real either way, and retrieval is what this example is about.
        history.append((rewritten, "(answer omitted)"))

    display.notice(
        "A follow-up question embedded as typed carries no reference for its "
        "pronouns. 'Why did he do it?' is a vector for a generic question "
        "about motive, and it retrieves accordingly.",
        "Rewriting happens before retrieval, not after. By the time the "
        "generator sees the context, the wrong passages have already been "
        "chosen.",
        "The rewriter is a small, cheap, low-temperature call. It is usually "
        "the highest-value LLM call in a conversational RAG system, and the "
        "one most often left out.",
        "This is also where conversational systems leak: the rewriter sees the "
        "whole history, so anything sensitive mentioned earlier can end up in "
        "a query sent to a search backend.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
