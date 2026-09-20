#!/usr/bin/env python3
"""
30. Memory-Augmented RAG
Book: Chapter 17, Memory-Augmented RAG

WHAT THIS SHOWS: Two memories, not one.

    short-term buffer    the last few turns, verbatim, always included
    long-term store      durable facts, retrieved by relevance when useful

The distinction matters because they fail differently. A buffer that is too
long wastes context on small talk; a long-term store that is always included
is not memory, it is just a bigger prompt. The controller decides what gets
written and what gets recalled, and those two decisions are where the design
lives.

The write policy here is confidence-and-novelty based, as the book describes:
a fact is stored when it looks durable AND is not already known. The recall
policy blends semantic similarity with recency, which is the same tension as
example 29's recency prior in a different costume.

HOW THIS SCRIPT PROCEEDS
    1. Walk a conversation turn by turn
    2. For each turn, decide: store or discard?   <-- THE WRITE GATE
    3. Check novelty against what is already known  <-- semantic, not string
    4. Recall: blend similarity with recency
    5. Show the three sources that reach the prompt

Steps 2 and 3 are the design. A memory that stores everything is a transcript,
and the short-term buffer already is one.


WHAT CHANGED SINCE EXAMPLE 22
    Example 22 used conversation history to rewrite the query, then threw it
    away. This keeps some of it, and the interesting question becomes WHICH
    parts are worth keeping across sessions.

    Two memories, not one, because they fail differently: a buffer that is too
    long wastes context on small talk, while a long-term store that is always
    included is not memory, just a bigger prompt.


REQUIRES: Postgres
RUN: python examples/30_memory_rag.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import get_embedder

SESSION = "demo-session"

# A conversation where earlier turns matter much later.
TURNS = [
    ("user", "I'm researching Victorian poisons for a novel.", True),
    ("user", "Hello again.", False),
    ("user", "What was the speckled band?", False),
    ("user", "I prefer answers that cite the original text.", True),
    ("user", "Thanks, that's helpful.", False),
    ("user", "Anything else like that in the stories?", False),
]

# Cues that a statement is a durable preference or fact rather than a passing
# question. Crude on purpose: the book describes a "lightweight neural
# controller", and this is the honest cheap version of the same decision.
DURABLE_CUES = [
    "i'm ", "i am ", "i prefer", "i like", "i want", "my name",
    "i work", "i'm researching", "remember", "always", "never",
]

BUFFER_TURNS = 3

# Write gates. A turn is stored only if it clears both.
CONFIDENCE_THRESHOLD = 0.30
# Novelty is 1 - (similarity to the nearest stored memory), so 0.35 means
# "reject anything more than 0.65 similar to something already known". That is
# deliberately strict: a near-duplicate memory is worse than a missing one,
# because duplicates retrieve together and crowd the recall budget.
NOVELTY_THRESHOLD = 0.35


def looks_durable(text: str) -> tuple[bool, float]:
    """Decide whether a turn is worth remembering, and how strongly.

    Returns (should_store, confidence). The book's rule is that facts are
    written "when confidence exceeds a threshold" -- the point being that
    storing everything is not memory, it is a transcript, and a transcript is
    what the short-term buffer already is.
    """
    lowered = text.lower()
    hits = sum(1 for cue in DURABLE_CUES if cue in lowered)
    is_question = text.strip().endswith("?")

    confidence = min(1.0, 0.35 * hits)
    if is_question:
        # Questions are almost never durable facts about the user. They belong
        # in the buffer and then should be forgotten.
        confidence *= 0.2

    return confidence >= CONFIDENCE_THRESHOLD, confidence


def novelty(conn, embedder, text: str) -> float:
    """1.0 if nothing like this is stored, 0.0 if it is already known.

    Without this check a user who repeats a preference three times gets three
    near-identical memories, all of which then retrieve together and crowd out
    everything else. Deduplicating on meaning rather than on string equality
    is the only version that works.
    """
    vector = embedder.encode([text], show_progress=False)[0]
    row = conn.execute(
        """
        SELECT max(1 - (vec_384 <=> %(vec)s::vector))
        FROM   memory
        WHERE  session_id = %(session)s AND kind = 'fact' AND vec_384 IS NOT NULL
        """,
        {"vec": pg.vec_literal(vector), "session": SESSION},
    ).fetchone()
    # ---- THE KEY IDEA ------------------------------------------------
    # Novelty is measured SEMANTICALLY. A user restating a preference in
    # different words is not a new fact, and string-level deduplication
    # cannot see that. Without this check the same fact accumulates and
    # then retrieves several times, crowding out everything else.
    best = float(row[0]) if row and row[0] is not None else 0.0
    return 1.0 - best


def remember(conn, embedder, text: str, importance: float) -> None:
    vector = embedder.encode([text], show_progress=False)[0]
    conn.execute(
        """
        INSERT INTO memory (session_id, kind, role, content, importance, vec_384, model)
        VALUES (%(session)s, 'fact', 'user', %(content)s, %(importance)s,
                %(vec)s::vector, %(model)s)
        """,
        {
            "session": SESSION, "content": text, "importance": importance,
            "vec": pg.vec_literal(vector), "model": embedder.name,
        },
    )


def recall(conn, embedder, query: str, k: int = 3, recency_weight: float = 0.3):
    """Fetch long-term memories relevant to the current turn.

    Score blends similarity, recency and stored importance. Pure similarity
    would keep surfacing an old fact that happens to match wording; pure
    recency would just be the buffer again.
    """
    vector = embedder.encode([query], show_progress=False)[0]
    return conn.execute(
        """
        SELECT content,
               1 - (vec_384 <=> %(vec)s::vector) AS similarity,
               importance,
               use_count,
               EXTRACT(EPOCH FROM (now() - created_at)) / 60.0 AS age_minutes,
               (1 - %(rw)s) * (1 - (vec_384 <=> %(vec)s::vector))
                 + %(rw)s * exp(-EXTRACT(EPOCH FROM (now() - created_at)) / 3600.0)
                 AS score
        FROM   memory
        WHERE  session_id = %(session)s AND kind = 'fact' AND vec_384 IS NOT NULL
        ORDER  BY score DESC
        LIMIT  %(k)s
        """,
        {
            "vec": pg.vec_literal(vector), "session": SESSION,
            "rw": recency_weight, "k": k,
        },
    ).fetchall()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=int, default=BUFFER_TURNS)
    args = parser.parse_args()

    display.banner(
        "30. Memory-Augmented RAG",
        "Chapter 17, Memory-Augmented RAG",
        "A short-term buffer for the conversation and a long-term store for "
        "durable facts, with a controller deciding what crosses between them.",
    )

    if not pg.available():
        display.missing_database("The long-term store is the `memory` table.")
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    embedder = get_embedder()
    conn.execute("DELETE FROM memory WHERE session_id = %s", [SESSION])

    # --- the write decision ----------------------------------------------
    display.heading("The controller: what gets written to long-term memory?")
    rows = []
    for role, text, expected in TURNS:
        durable, confidence = looks_durable(text)
        new = novelty(conn, embedder, text) if durable else 1.0
        stored = durable and new > NOVELTY_THRESHOLD
        if stored:
            remember(conn, embedder, text, confidence)
        rows.append([
            display.truncate(text, 42),
            f"{confidence:.2f}",
            f"{new:.2f}" if durable else "-",
            "stored" if stored else "buffer only",
        ])
    display.table(
        ["turn", "confidence", "novelty", "outcome"], rows, align="lrrl"
    )

    stored_count = conn.execute(
        "SELECT count(*) FROM memory WHERE session_id = %s", [SESSION]
    ).fetchone()[0]
    print(f"\n  {stored_count} of {len(TURNS)} turns became long-term memories.")
    print("  Questions and pleasantries stay in the buffer and are forgotten")
    print("  when it rolls over. Storing every turn would make the long-term")
    print("  store a slower copy of the transcript.")

    # --- novelty ----------------------------------------------------------
    display.heading("Novelty: the user repeats themselves")
    repeat = "I prefer that you quote directly from the original text."
    durable, confidence = looks_durable(repeat)
    new = novelty(conn, embedder, repeat)
    print(f"  new turn: {repeat!r}")
    display.kv("confidence", f"{confidence:.2f}")
    display.kv("novelty vs stored", f"{new:.2f}")

    # Name the gate that actually decided, rather than reporting one outcome
    # for two different reasons -- the whole point of showing both numbers.
    if not durable:
        verdict = f"skipped: confidence {confidence:.2f} below {CONFIDENCE_THRESHOLD}"
    elif new <= NOVELTY_THRESHOLD:
        verdict = f"skipped: novelty {new:.2f} below {NOVELTY_THRESHOLD} -- already known"
    else:
        verdict = "stored"
    display.kv("outcome", verdict)
    print("\n  Semantically this is the same preference already on file, phrased")
    print("  differently. String equality would not catch it; vector similarity")
    print("  does. Without this check the same fact accumulates and then")
    print("  retrieves several times, crowding out everything else.")

    # --- recall -----------------------------------------------------------
    display.heading("Recall: what memory is relevant to the current turn?")
    current = "Anything else like that in the stories?"
    print(f"  current turn: {current!r}\n")

    for weight, label in ((0.0, "similarity only"), (0.3, "blended"), (0.9, "recency-dominated")):
        results = recall(conn, embedder, current, k=3, recency_weight=weight)
        print(f"  recency weight = {weight} ({label})")
        display.table(
            ["memory", "similarity", "score"],
            [[display.truncate(r[0], 44), f"{float(r[1]):.3f}", f"{float(r[5]):.3f}"]
             for r in results],
            align="lrr",
        )
        print()

    print("  At weight 0 the ordering is pure relevance; at 0.9 it is almost")
    print("  pure recency and the long-term store has become the buffer with")
    print("  extra steps. The useful settings are in between.")

    # --- what the prompt actually gets ------------------------------------
    display.heading("What the model receives")
    buffer = [t[1] for t in TURNS][-args.buffer:]
    memories = recall(conn, embedder, current, k=2, recency_weight=0.3)

    print(f"  SHORT-TERM BUFFER (last {args.buffer} turns, verbatim):")
    for text in buffer:
        print(f"      - {display.truncate(text, 66)}")
    print("\n  LONG-TERM MEMORY (retrieved, not always present):")
    for row in memories:
        print(f"      - {display.truncate(row[0], 66)}")
    print("\n  RETRIEVED DOCUMENTS: (from the corpus, as in example 17)")
    print("\n  Three distinct sources, assembled into one prompt. Collapsing")
    print("  them into a single 'context' is how systems end up citing the")
    print("  user's own preferences as though they were evidence.")

    display.notice(
        "The buffer is verbatim and bounded; the long-term store is retrieved "
        "and unbounded. Treating them the same wastes context on one and "
        "loses information from the other.",
        "The write decision is a filter, not a log. If everything is stored, "
        "recall degrades because the store fills with questions and "
        "pleasantries that match everything weakly.",
        "Novelty is checked semantically. A user restating a preference in new "
        "words is not a new fact, and string-level deduplication will not see "
        "that.",
        "Memory is the highest-risk surface in a RAG system for privacy. It "
        "persists what a user said across sessions by design -- so retention, "
        "expiry and deletion belong here, using the machinery in example 35.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
