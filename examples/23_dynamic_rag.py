#!/usr/bin/env python3
"""
23. Dynamic RAG (adaptive depth and confidence-driven iteration)
Book: Chapter 9, Dynamic RAG

WHAT THIS SHOWS: Stop using the same k for every question.

The book's numbers: a simple factual lookup needs 2-3 passages; a complex
multi-hop question needs 10-20, or several rounds of retrieval. Serving both
with a fixed k=5 means paying for ten passages on easy questions and starving
hard ones.

Three mechanisms here, and only the last needs a model:

    1. a complexity classifier that picks k from the query's own shape
    2. a retrieval-confidence check on the score distribution
    3. self-verification and re-retrieval when confidence is low

Mechanisms 1 and 2 are pure local computation. That matters: the cheapest
adaptivity is available without any extra model call at all.

HOW THIS SCRIPT PROCEEDS
    1. Classify each query, pick k from its shape   <-- no model call
    2. Retrieve at that k
    3. Score retrieval confidence from the SCORES   <-- no model call
    4. Show what low confidence catches: an unanswerable query
    5. Only then, the LLM-dependent self-verification loop

Steps 1-3 are pure arithmetic. That ordering is deliberate -- most of the
adaptivity in this chapter is available before you spend anything.


THE FINDING WORTH NOTING
    In step 4, a question with NO answer in the corpus gets a top score of
    0.601 -- as high as the genuinely answerable ones. What separates it is
    the MARGIN: its results are all equally mediocre.

    That is why the confidence function below weights margin as heavily as
    top score. A retriever cannot return nothing; it returns its k nearest
    neighbours whether or not any of them are relevant.


REQUIRES: Postgres. OPENAI_API_KEY only for step 3.
RUN: python examples/23_dynamic_rag.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, llm, pg
from ragkit.embed import describe, get_embedder

QUESTIONS = [
    "What was the speckled band?",
    "Who stole the blue carbuncle?",
    "How did Holmes connect the hat, the goose and the stolen gem, and why did he let the thief go?",
    "Compare how Holmes deduced the truth in the Red-Headed League and the Copper Beeches.",
]

# Words that signal a question needing several pieces of evidence rather than
# one lookup.
MULTI_HOP_MARKERS = {
    "compare", "contrast", "both", "difference", "differences", "versus",
    "relationship", "connect", "connection", "why", "how", "explain",
    "and then", "after", "before", "sequence",
}


def classify(question: str) -> tuple[str, int]:
    """Pick a retrieval depth from the query's surface features.

    A deliberately simple rule-based classifier. The book lists three options
    -- reinforcement learning, meta-learning, or "rule-based heuristics (query
    length, named entity count)" -- and this is the third, because it costs
    nothing, is debuggable, and captures most of the benefit.

    Returns (label, k).
    """
    lowered = question.lower()
    words = len(question.split())

    # Capitalised words that are not sentence-initial: a crude proper-noun
    # count. More named entities usually means more places to look.
    entities = len(re.findall(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]+", question))

    markers = sum(1 for marker in MULTI_HOP_MARKERS if marker in lowered)
    clauses = lowered.count(" and ") + lowered.count(",")

    # ---- THE KEY LINE ------------------------------------------------
    # Four cheap surface features added together. The book offers three
    # options for this decision -- reinforcement learning, meta-learning,
    # or "rule-based heuristics (query length, named entity count)" --
    # and this is the third, because it costs nothing and is debuggable.
    score = markers + clauses + (words > 15) + (entities > 2)

    if score >= 3:
        return "complex / multi-hop", 15
    if score >= 1:
        return "moderate", 8
    return "simple factual", 3


def retrieval_confidence(hits) -> tuple[float, str]:
    """Judge retrieval quality from the score distribution alone.

    Two signals, neither needing a model:

    * **top score** -- if nothing scored well, nothing matched.
    * **margin** between the best and the rest -- a clear winner suggests one
      passage really answers this. A flat distribution means the retriever
      found many things that are equally mediocre, which is what "the answer
      is not in the corpus" looks like from the inside.
    """
    if not hits:
        return 0.0, "nothing retrieved"

    scores = np.array([h.score for h in hits])
    top = float(scores[0])
    margin = float(scores[0] - scores[1:].mean()) if len(scores) > 1 else 0.0

    # ---- THE KEY LINE ------------------------------------------------
    # Margin is weighted as heavily as the top score, and that is what
    # makes this work. A high top score with a flat tail means the
    # retriever found many equally-mediocre things -- which is exactly
    # what "the answer is not in this corpus" looks like from inside.
    confidence = min(1.0, max(0.0, top)) * (0.5 + min(0.5, margin * 3))

    if confidence >= 0.35:
        reason = f"top {top:.2f}, clear margin {margin:+.2f}"
    elif top < 0.35:
        reason = f"top score only {top:.2f} -- nothing matched well"
    else:
        reason = f"flat distribution, margin only {margin:+.2f}"
    return confidence, reason


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="contextual_header")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    display.banner(
        "23. Dynamic RAG",
        "Chapter 9, Dynamic RAG",
        "Choose the retrieval depth per query, then check whether what came "
        "back is good enough before trusting it.",
    )

    if not pg.available():
        display.missing_database("Adaptive retrieval still needs something to retrieve from.")
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

    print(f"  {describe(embedder)}\n")

    display.heading("1. Adaptive retrieval depth")
    print("  The classifier reads the query's shape -- length, named entities,")
    print("  multi-hop markers -- and picks k. No model call.\n")

    rows = []
    for question in QUESTIONS:
        label, k = classify(question)
        rows.append([display.truncate(question, 46), label, k])
    display.table(["question", "classified as", "k"], rows)

    fixed_cost = len(QUESTIONS) * 5
    adaptive_cost = sum(classify(q)[1] for q in QUESTIONS)
    print(f"\n  Fixed k=5 across all four: {fixed_cost} passages retrieved.")
    print(f"  Adaptive: {adaptive_cost}. The easy questions got cheaper and the")
    print("  hard ones got more evidence -- which is the point, not the total.")

    display.heading("2. Retrieval confidence, before generating anything")
    print("  Judged from the score distribution alone. A flat distribution")
    print("  means many mediocre matches, which is what a missing answer looks")
    print("  like from inside the retriever.\n")

    results = []
    for question in QUESTIONS:
        label, k = classify(question)
        vector = embedder.encode([question], show_progress=False)[0]
        hits = store.search(vector, k=k, strategy=args.strategy)
        confidence, reason = retrieval_confidence(hits)
        results.append((question, label, k, hits, confidence, reason))

    display.table(
        ["question", "k", "top score", "confidence", "verdict"],
        [
            [
                display.truncate(q, 34),
                k,
                f"{hits[0].score:.3f}",
                f"{confidence:.2f}",
                "proceed" if confidence >= args.threshold else "RE-RETRIEVE",
            ]
            for q, label, k, hits, confidence, reason in results
        ],
    )

    # --- an out-of-corpus question ---------------------------------------
    display.heading("3. What low confidence is supposed to catch")
    unanswerable = "What did Holmes say about quantum mechanics?"
    label, k = classify(unanswerable)
    vector = embedder.encode([unanswerable], show_progress=False)[0]
    hits = store.search(vector, k=k, strategy=args.strategy)
    confidence, reason = retrieval_confidence(hits)

    print(f"  query: {unanswerable!r}")
    print(f"  This has no answer in the corpus at all.\n")
    display.kv("top score", f"{hits[0].score:.3f}")
    display.kv("confidence", f"{confidence:.2f}")
    display.kv("reason", reason)
    display.kv("verdict", "proceed" if confidence >= args.threshold else "RE-RETRIEVE / abstain")
    print("\n  The retriever always returns its k nearest neighbours -- it has no")
    print("  way to return nothing. Something always comes back, ranked, and it")
    print("  looks exactly like a real result until you inspect the scores.")

    # --- step 4: the LLM-dependent part ----------------------------------
    display.heading("4. Self-verification and re-retrieval")
    if not llm.available():
        print(f"  No LLM: {llm.why_unavailable()}")
        print("\n  Steps 1-3 above are complete and needed no model. The remaining")
        print("  loop would be:")
        print("      - generate an answer from the retrieved context")
        print("      - ask the model whether the context actually supports it")
        print("      - if not, reformulate the query toward the gap and retrieve")
        print("        again with a larger k")
        print("      - stop after N rounds or when confidence clears the threshold")
        print("\n  Worth noting how much adaptivity was available before this")
        print("  point. The classifier and the confidence check are arithmetic.")
    else:
        if llm.STUB:
            display.stub_warning()
        low = [r for r in results if r[4] < args.threshold]
        print(f"  {len(low)} of {len(results)} queries fell below the threshold.")
        for question, label, k, hits, confidence, reason in low:
            context = "\n\n".join(h.chunk.context_text for h in hits[:5])
            verdict = llm.complete(
                f"Context:\n{context}\n\nQuestion: {question}\n\n"
                "Does the context contain enough information to answer this "
                "question? Reply with YES or NO and one short sentence.",
                system="You assess whether retrieved context supports a question.",
                purpose="self-verification",
            )
            print(f"\n  {question!r}")
            display.para(verdict, indent="      ")

    display.notice(
        "The classifier and the confidence check are pure arithmetic over the "
        "query string and the score distribution. Most of the adaptivity in "
        "this chapter costs nothing.",
        "A retriever cannot return nothing. It returns its k nearest "
        "neighbours whether or not any of them are relevant, so a question "
        "with no answer in the corpus still produces a confident-looking "
        "ranked list.",
        "Score margin is the more useful of the two confidence signals. A high "
        "top score with a flat tail often means a generically popular chunk, "
        "not a specific answer.",
        "Thresholds like the 0.35 used here are corpus- and model-specific. "
        "Calibrate against a gold set -- example 31 is where that machinery "
        "lives -- rather than copying a number out of a book.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
