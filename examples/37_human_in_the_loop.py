#!/usr/bin/env python3
"""
37. Human-in-the-Loop RAG
Book: Chapter 24, Human-in-the-Loop RAG Systems

WHAT THIS SHOWS: How to spend a limited amount of human attention well.

Expert review is the most accurate signal available and the most expensive.
The design question is never "should we collect feedback" but "which hundred
of these ten thousand queries should a person look at".

The answer is uncertainty sampling: review the queries the system is least
sure about, because a query it handled confidently and correctly teaches
nothing, and one it got wrong while confident is where the damage is.

Then the loop closes: ratings become retrieval training pairs, corrections
become generation training targets, and both feed back into examples 34 and 21.

HOW THIS SCRIPT PROCEEDS
    1. Score every query for uncertainty    <-- THE TECHNIQUE, needs no model
    2. CHECK whether uncertainty predicts failure   <-- the honesty step
    3. Build a review queue from the most uncertain
    4. Compare its yield against random sampling
    5. Record feedback, and turn it into training data

Step 2 is the one most write-ups skip. If the flagged half fails at the same
rate as the rest, you have built a random sampler with extra steps -- and on
this small gold set the separation is genuinely weak, which the script says.


WHAT THIS FEEDS
    Human attention is the scarce resource, so the question is never "should
    we collect feedback" but "which hundred of these ten thousand queries
    should a person look at".

    What comes back splits into two different datasets: relevance ratings
    train the retriever (example 34), rewritten answers train the generator.
    Mixing them produces something that trains neither. And a human-confirmed
    negative is strictly better than a mined one -- example 34 has to filter
    out false negatives, whereas a reviewer has already done that.


REQUIRES: Postgres
RUN: python examples/37_human_in_the_loop.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import get_embedder

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"

REVIEW_BUDGET = 8


def uncertainty(hits) -> tuple[float, str]:
    """How unsure is the retriever about this query?

    Two components, both free:

    * **low top score** -- nothing matched well.
    * **low margin / high entropy** -- several results scored alike, so the
      ranking is close to arbitrary. The book calls this "high retrieval
      entropy" and it is the more informative of the two: a confident wrong
      answer looks fine on top score alone.

    Returns (uncertainty in 0..1, the dominant reason).
    """
    if not hits:
        return 1.0, "nothing retrieved"

    scores = np.array([h.score for h in hits], dtype=np.float64)
    top = float(scores[0])
    margin = float(scores[0] - scores[1]) if len(scores) > 1 else 1.0

    # Softmax over the scores, then normalised entropy. A flat distribution
    # means the retriever had no real preference between its results.
    exponentials = np.exp((scores - scores.max()) * 12)
    probabilities = exponentials / exponentials.sum()
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    max_entropy = float(np.log(len(scores)))
    normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    score_uncertainty = 1.0 - min(1.0, top / 0.65)
    combined = 0.5 * score_uncertainty + 0.5 * normalised_entropy

    if score_uncertainty > normalised_entropy:
        reason = f"low top score ({top:.2f})"
    else:
        reason = f"flat ranking (margin {margin:.3f})"
    return float(min(1.0, combined)), reason


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=REVIEW_BUDGET)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "37. Human-in-the-Loop RAG",
        "Chapter 24, Human-in-the-Loop RAG Systems",
        f"Choosing which {args.budget} queries deserve expert review, and "
        "turning what comes back into training data.",
    )

    if not pg.available():
        display.missing_database("Feedback is stored in the feedback table.")
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

    conn.execute("TRUNCATE feedback")

    # --- score every query -------------------------------------------------
    display.heading("Scoring every query for uncertainty")
    rows = []
    for question in questions:
        vector = embedder.encode([question["question"]], show_progress=False)[0]
        hits = store.search(vector, k=args.k, strategy=args.strategy)
        score, reason = uncertainty(hits)

        # We know the truth here because this is a gold set. In production you
        # would not -- which is exactly why uncertainty has to be estimated
        # from the score distribution rather than from correctness.
        context = " ".join(" ".join(h.chunk.context_text.split()) for h in hits)
        correct = question["evidence"] in context

        rows.append({
            "question": question,
            "hits": hits,
            "uncertainty": score,
            "reason": reason,
            "correct": correct,
        })

    rows.sort(key=lambda r: -r["uncertainty"])

    # --- does uncertainty predict failure? --------------------------------
    display.heading("Is uncertainty actually predictive?")
    top_half = rows[: len(rows) // 2]
    bottom_half = rows[len(rows) // 2:]
    display.table(
        ["group", "queries", "actually wrong"],
        [
            ["most uncertain half", len(top_half),
             f"{sum(1 for r in top_half if not r['correct'])} "
             f"({100*sum(1 for r in top_half if not r['correct'])/len(top_half):.0f}%)"],
            ["least uncertain half", len(bottom_half),
             f"{sum(1 for r in bottom_half if not r['correct'])} "
             f"({100*sum(1 for r in bottom_half if not r['correct'])/len(bottom_half):.0f}%)"],
        ],
        align="lrr",
    )
    top_rate = sum(1 for r in top_half if not r["correct"]) / len(top_half)
    bottom_rate = sum(1 for r in bottom_half if not r["correct"]) / len(bottom_half)
    separation = top_rate - bottom_rate

    print(f"\n  Separation: {separation:+.0%}.")
    if separation < 0.15:
        print("  That is weak. On thirty questions it is also well within noise,")
        print("  so this run is not evidence that uncertainty sampling works")
        print("  here -- it is evidence that you need a few hundred queries")
        print("  before this comparison means anything.")
        print("\n  Reporting it anyway is the point. A tutorial that only showed")
        print("  the flattering number would teach you to skip the check.")
    else:
        print("  The uncertain half fails materially more often, so the sampler")
        print("  is finding real failures rather than sampling at random.")

    # --- what gets reviewed -------------------------------------------------
    display.heading(f"The review queue (budget: {args.budget})")
    queue = rows[: args.budget]
    display.table(
        ["question", "uncertainty", "why flagged"],
        [[display.truncate(r["question"]["question"], 40),
          f"{r['uncertainty']:.2f}", r["reason"]] for r in queue],
        align="lrl",
    )

    random.seed(3)
    random_sample = random.sample(rows, args.budget)
    display.table(
        ["sampling method", "wrong answers caught", "of budget"],
        [
            ["uncertainty", sum(1 for r in queue if not r["correct"]), args.budget],
            ["random", sum(1 for r in random_sample if not r["correct"]), args.budget],
        ],
        align="lrr",
    )
    print("\n  Same amount of human time, different yield. That ratio is the")
    print("  argument for active sampling -- but read it against the weak")
    print("  separation above before believing it. Two numbers from a set of")
    print("  thirty are a hypothesis, not a result.")

    # --- recording feedback -------------------------------------------------
    display.heading("Recording what the reviewer said")
    print("  Simulated here from the gold set. In production these rows come")
    print("  from a review UI.\n")

    for row in queue:
        question = row["question"]
        for hit in row["hits"][: args.k]:
            context = " ".join(hit.chunk.context_text.split())
            relevant = question["evidence"] in context
            conn.execute(
                """
                INSERT INTO feedback
                    (query, chunk_id, rank, rating, verdict, correction, reviewer)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    question["question"],
                    hit.chunk.index,
                    hit.rank,
                    5 if relevant else 2,
                    1 if relevant else -1,
                    question["answer"] if (relevant and hit.rank > 1) else None,
                    "sme-1",
                ],
            )

    counts = conn.execute(
        "SELECT verdict, count(*) FROM feedback GROUP BY verdict ORDER BY verdict"
    ).fetchall()
    display.table(
        ["verdict", "rows"],
        [[{-1: "not relevant", 0: "unsure", 1: "relevant"}.get(r[0], r[0]), r[1]]
         for r in counts],
    )

    # --- closing the loop ---------------------------------------------------
    display.heading("Closing the loop: feedback becomes training data")
    pairs = conn.execute(
        """
        SELECT f.query,
               count(*) FILTER (WHERE f.verdict = 1)  AS positives,
               count(*) FILTER (WHERE f.verdict = -1) AS negatives
        FROM   feedback f
        GROUP  BY f.query
        HAVING count(*) FILTER (WHERE f.verdict = 1) > 0
           AND count(*) FILTER (WHERE f.verdict = -1) > 0
        """
    ).fetchall()

    display.kv("queries with both signals", len(pairs))
    print("\n  A query with a positive AND a negative is a contrastive training")
    print("  pair, ready for example 34. And these negatives are better than")
    print("  the mined ones there: a human confirmed they were retrieved and")
    print("  wrong, so there is no false-negative risk at all.")

    corrections = conn.execute(
        "SELECT count(*) FROM feedback WHERE correction IS NOT NULL"
    ).fetchone()[0]
    display.kv("corrections recorded", corrections)
    print("\n  Corrections are generation training targets rather than")
    print("  retrieval ones -- a different dataset, a different model, and")
    print("  worth keeping separate from the start.")

    display.notice(
        "Human attention is the scarce resource. Uncertainty sampling spends "
        "it where the system is least sure, which is where it is most often "
        "wrong.",
        "Check that uncertainty actually predicts failure on your data -- this "
        "run separates the two halves by only a few points, which on thirty "
        "questions is noise. If the flagged half fails at the same rate as the "
        "rest, you have built a random sampler with extra steps.",
        "Feedback yields two different datasets. Relevance ratings train the "
        "retriever; rewritten answers train the generator. Mixing them "
        "produces something that trains neither.",
        "Human-confirmed negatives are strictly better than mined ones. "
        "Example 34 has to filter out false negatives; a reviewer has already "
        "done that here.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
