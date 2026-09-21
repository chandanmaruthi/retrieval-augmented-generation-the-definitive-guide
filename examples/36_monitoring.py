#!/usr/bin/env python3
"""
36. Real-Time Evaluation and Monitoring
Book: Chapter 23, Real-Time Evaluation & Monitoring
WHAT THIS SHOWS: What to log per query, and what the logs can tell you that
an offline evaluation cannot.

Offline evaluation answers "is this system good on our gold set". Monitoring
answers "is it still good, on the traffic it is actually getting". Those come
apart in production constantly: the corpus drifts, users ask new things, and
a retriever tuned on last quarter's questions degrades quietly.

The book's one numeric threshold appears here -- alert when the hallucination
rate exceeds 15% -- along with the cheaper signals that need no judge model:
latency, retrieval score distributions, and the abstention rate.

HOW THIS SCRIPT PROCEEDS
    1. Simulate realistic traffic       75% answerable, 25% not  <-- the mix matters
    2. Log every query                  latency, grounding, tokens
    3. Read the dashboard               p95, not mean
    4. Check the book's 15% alert threshold
    5. Compare offline vs live measurement   <-- THE POINT
    6. Score distributions as a drift signal

Step 5 is why monitoring exists as a separate discipline from evaluation.


WHAT OFFLINE EVALUATION CANNOT SEE
    A gold set contains only ANSWERABLE questions. Real traffic does not, and
    a query with no answer in the corpus still produces a confident ranked
    list, because a retriever cannot return nothing.

    That failure mode dominates real traffic and is completely invisible to
    example 31's metrics -- and it is not a gap you can close by writing more
    gold questions.


REQUIRES: Postgres
RUN: python examples/36_monitoring.py [--queries 60]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import get_embedder

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"

HALLUCINATION_ALERT = 0.15  # the book's threshold

# Queries with no answer in this corpus. A real traffic mix always contains
# these, and they are what the monitoring is for -- a system that answers them
# confidently is hallucinating and offline evaluation will never show it,
# because the gold set only contains answerable questions.
OUT_OF_CORPUS = [
    "What did Holmes say about quantum mechanics?",
    "How much does a train ticket to Paris cost today?",
    "What is the capital of Brazil?",
    "Who won the 1966 World Cup?",
    "What are the side effects of aspirin?",
]


def log_query(conn, **fields) -> None:
    conn.execute(
        """
        INSERT INTO query_log
            (query, strategy, retriever, k, latency_ms, retrieved_ids,
             grounded, hallucinated, prompt_tokens, answer_tokens)
        VALUES (%(query)s, %(strategy)s, %(retriever)s, %(k)s, %(latency_ms)s,
                %(retrieved_ids)s, %(grounded)s, %(hallucinated)s,
                %(prompt_tokens)s, %(answer_tokens)s)
        """,
        fields,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=60)
    parser.add_argument("--strategy", default="contextual_header")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    display.banner(
        "36. Real-Time Evaluation and Monitoring",
        "Chapter 23, Real-Time Evaluation & Monitoring",
        "Instrumenting a live pipeline, then reading the logs for the signals "
        "an offline gold set cannot give you.",
    )

    if not pg.available():
        display.missing_database("Query logs are written to the query_log table.")
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

    conn.execute("TRUNCATE query_log")

    # --- simulate traffic -------------------------------------------------
    display.heading(f"Simulating {args.queries} queries")
    random.seed(7)

    # A realistic mix: mostly answerable, with a steady trickle that is not.
    # The proportion matters -- this is the number that decides whether the
    # alert threshold below ever fires.
    traffic = []
    for _ in range(args.queries):
        if random.random() < 0.25:
            traffic.append((random.choice(OUT_OF_CORPUS), None))
        else:
            question = random.choice(questions)
            traffic.append((question["question"], question["evidence"]))

    print(f"  {sum(1 for _, e in traffic if e)} answerable, "
          f"{sum(1 for _, e in traffic if not e)} out-of-corpus\n")

    for query, evidence in traffic:
        started = time.perf_counter()
        vector = embedder.encode([query], show_progress=False)[0]
        hits = store.search(vector, k=args.k, strategy=args.strategy)
        latency_ms = int((time.perf_counter() - started) * 1000)

        context = " ".join(" ".join(h.chunk.context_text.split()) for h in hits)

        # Grounded: the evidence for this query is actually in the context.
        grounded = bool(evidence and evidence in context)

        # Hallucinated: we would have answered confidently when the context
        # does not support an answer. Approximated by the retrieval score --
        # a high top score with no real evidence is precisely the dangerous
        # case, because nothing downstream will flag it.
        top = hits[0].score if hits else 0.0
        hallucinated = (not grounded) and top > 0.45

        log_query(
            conn,
            query=query,
            strategy=args.strategy,
            retriever="dense",
            k=args.k,
            latency_ms=latency_ms,
            retrieved_ids=[h.chunk.index for h in hits],
            grounded=grounded,
            hallucinated=hallucinated,
            prompt_tokens=sum(len(h.chunk.context_text.split()) for h in hits),
            answer_tokens=random.randint(40, 160),
        )

    # --- the dashboard ----------------------------------------------------
    display.heading("Dashboard")
    row = conn.execute(
        """
        SELECT count(*),
               avg(latency_ms), percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms),
               percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),
               avg(CASE WHEN grounded THEN 1.0 ELSE 0.0 END),
               avg(CASE WHEN hallucinated THEN 1.0 ELSE 0.0 END),
               avg(prompt_tokens), avg(answer_tokens)
        FROM   query_log
        """
    ).fetchone()

    display.table(
        ["metric", "value"],
        [
            ["queries", row[0]],
            ["latency mean", f"{float(row[1]):.0f} ms"],
            ["latency p50", f"{float(row[2]):.0f} ms"],
            ["latency p95", f"{float(row[3]):.0f} ms"],
            ["grounding rate", f"{float(row[4]):.1%}"],
            ["hallucination rate", f"{float(row[5]):.1%}"],
            ["mean prompt tokens", f"{float(row[6]):.0f}"],
            ["mean answer tokens", f"{float(row[7]):.0f}"],
        ],
    )

    print("\n  p95 rather than mean. A mean latency hides the tail, and the")
    print("  tail is what users actually complain about.")

    # --- the alert --------------------------------------------------------
    display.heading(f"Alert threshold: hallucination rate > {HALLUCINATION_ALERT:.0%}")
    rate = float(row[5])
    if rate > HALLUCINATION_ALERT:
        print(f"  FIRING: {rate:.1%} exceeds {HALLUCINATION_ALERT:.0%}")
        print("\n  The book's suggested response is to investigate or roll back.")
        print("  Here the cause is visible in the traffic mix: a quarter of the")
        print("  queries have no answer in this corpus, and the pipeline answers")
        print("  them anyway because retrieval always returns its top k.")
    else:
        print(f"  OK: {rate:.1%} is within {HALLUCINATION_ALERT:.0%}")

    # --- what offline eval would have said --------------------------------
    display.heading("Why offline evaluation would have missed this")
    answerable = conn.execute(
        "SELECT avg(CASE WHEN grounded THEN 1.0 ELSE 0.0 END) FROM query_log "
        "WHERE query IN (SELECT unnest(%s::text[]))",
        [[q["question"] for q in questions]],
    ).fetchone()[0]

    display.table(
        ["measured on", "grounding rate"],
        [
            ["the gold set only (offline)", f"{float(answerable or 0):.1%}"],
            ["live traffic (all queries)", f"{float(row[4]):.1%}"],
        ],
        align="lr",
    )
    print("\n  The gold set contains only answerable questions, so offline")
    print("  evaluation cannot see the failure mode that dominates real")
    print("  traffic. That gap is the argument for monitoring, and it is not a")
    print("  gap you can close by adding more gold questions.")

    # --- score distribution as a drift signal ----------------------------
    display.heading("Drift signal: the score distribution")
    scores_answerable, scores_other = [], []
    for query, evidence in traffic:
        vector = embedder.encode([query], show_progress=False)[0]
        hits = store.search(vector, k=1, strategy=args.strategy)
        (scores_answerable if evidence else scores_other).append(hits[0].score)

    display.table(
        ["query type", "n", "mean top score", "p10", "p90"],
        [
            ["answerable", len(scores_answerable),
             f"{np.mean(scores_answerable):.3f}",
             f"{np.percentile(scores_answerable, 10):.3f}",
             f"{np.percentile(scores_answerable, 90):.3f}"],
            ["out-of-corpus", len(scores_other),
             f"{np.mean(scores_other):.3f}",
             f"{np.percentile(scores_other, 10):.3f}",
             f"{np.percentile(scores_other, 90):.3f}"],
        ],
    )
    overlap = np.mean(scores_other) / max(np.mean(scores_answerable), 1e-9)
    print(f"\n  Out-of-corpus queries score {overlap:.0%} as high as real ones.")
    print("  The distributions overlap heavily, which is why a single score")
    print("  threshold cannot separate them -- and why the margin-based")
    print("  confidence in example 23 works better than an absolute cutoff.")
    print("\n  Tracked over time, a shift in this distribution is the earliest")
    print("  available signal that the corpus and the traffic have drifted")
    print("  apart. It needs no labels and no judge model.")

    display.notice(
        f"The hallucination rate here is {rate:.1%} against the book's "
        f"{HALLUCINATION_ALERT:.0%} threshold -- driven entirely by queries "
        "with no answer in the corpus.",
        "Offline evaluation measures a gold set of answerable questions. "
        "Production traffic is not that, and the difference is where most "
        "real failures live.",
        "Latency belongs at p95, not mean. Grounding and hallucination rates "
        "belong per segment, not aggregated -- an average across query types "
        "hides which type is failing.",
        "Score distributions are a free drift signal. No labels, no judge "
        "model, and a shift shows up before any accuracy metric moves.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
