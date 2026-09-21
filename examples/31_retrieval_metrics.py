#!/usr/bin/env python3
"""
31. Retrieval Metrics
Book: Chapter 19, Evaluation Metrics
WHAT THIS SHOWS: Recall@k, Precision@k, MRR and NDCG, implemented from their
definitions and run over the thirty gold questions.

Written out rather than imported, for two reasons. They are about thirty lines
in total, and -- more importantly -- the book's whole argument in this chapter
is that retrieval and generation must be measured *separately*. That argument
is much easier to believe once you have seen that measuring retrieval needs no
model, no API key and no judgement: it is arithmetic over a ranked list and a
set of known-correct answers.

The metrics disagree with each other here, which is the useful part. A change
that improves Recall@5 can leave MRR flat, and knowing which you care about
is a product decision, not a technical one.

HOW THIS SCRIPT PROCEEDS
    1. Run all 30 gold questions
    2. Mark each retrieved chunk relevant or not
    3. Compute Recall@k, Precision@k, MRR, NDCG   <-- all four written out
    4. Show WHERE the first correct result landed
    5. List the questions that failed             <-- the most useful output
    6. Re-run at k = 1, 3, 5, 10, 20

Step 5 matters more than steps 3 and 4. "Recall@5 is 0.50" is a status
report; "these fifteen questions failed" is a work item.


WHY THIS NEEDS NO MODEL
    The book's argument in this chapter is that retrieval and generation must
    be measured SEPARATELY. That is much easier to believe once you have seen
    that measuring retrieval needs no model, no API key and no human -- it is
    arithmetic over a ranked list and a set of known-correct answers.

    Step 6 shows the signature to look for: recall rising with k while MRR
    stays flat means the retriever finds the right passage but ranks it
    poorly, which points at reranking rather than at better embeddings.


REQUIRES: Postgres
RUN: python examples/31_retrieval_metrics.py [--k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import describe, get_embedder

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------


def recall_at_k(relevant: list[bool]) -> float:
    """Did any correct result appear in the top k?

    With one correct answer per question this is binary, and averaged over a
    question set it becomes "what fraction of questions did we find an answer
    for". That is usually the number a product cares about most: it is the
    ceiling on how often the system can possibly be right.
    """
    return 1.0 if any(relevant) else 0.0


def precision_at_k(relevant: list[bool]) -> float:
    """What fraction of the returned results were correct?

    Low precision is not automatically bad in RAG. If the generator reads all
    k passages, one correct passage among five is enough -- the other four
    cost context budget, not accuracy. Precision matters when k is small or
    the context window is tight.
    """
    return sum(relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(relevant: list[bool]) -> float:
    """1 / (rank of the first correct result), or 0 if there is none.

    Averaged over questions this is MRR. It is the metric that notices
    reranking: moving the right answer from position 4 to position 1 leaves
    Recall@5 unchanged and takes MRR from 0.25 to 1.0.
    """
    for index, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevant: list[bool]) -> float:
    """Normalised discounted cumulative gain.

    DCG sums the gains, discounting each by log2(rank + 1) so later positions
    count for less. Normalising against the ideal ordering puts it on [0, 1]
    and makes it comparable across questions.

    With binary relevance and a single correct answer, NDCG and reciprocal
    rank measure much the same thing with a gentler discount. NDCG earns its
    keep with graded relevance -- "this passage is perfect, that one is
    partly useful" -- which this gold set does not have, and the table below
    says so rather than implying more rigour than exists.
    """
    gains = [1.0 if r else 0.0 for r in relevant]
    dcg = sum(g / np.log2(i + 1) for i, g in enumerate(gains, start=1))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "31. Retrieval Metrics",
        "Chapter 19, Evaluation Metrics",
        "Recall@k, Precision@k, MRR and NDCG over thirty gold questions -- "
        "all of it arithmetic, none of it needing a model.",
    )

    if not pg.available():
        display.missing_database("Metrics need something to retrieve from.")
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

    print(f"  {describe(embedder)}")
    print(f"  {len(questions)} gold questions, strategy {args.strategy!r}, k={args.k}\n")

    # Note: retrieval here runs as the table owner, so row-level security is
    # not filtering anything. Evaluating through a restricted identity would
    # measure the policy as well as the retriever, and those are different
    # questions.
    per_question = []
    for question in questions:
        vector = embedder.encode([question["question"]], show_progress=False)[0]
        hits = store.search(vector, k=args.k, strategy=args.strategy)

        # A hit is relevant when the text the model would receive contains the
        # gold evidence phrase. Whitespace is normalised because chunkers join
        # lines and the phrase may straddle an original line break.
        relevant = [
            question["evidence"] in " ".join(h.chunk.context_text.split())
            for h in hits
        ]
        per_question.append(
            {
                "id": question["id"],
                "question": question["question"],
                "story": question["story"],
                "relevant": relevant,
                "recall": recall_at_k(relevant),
                "precision": precision_at_k(relevant),
                "rr": reciprocal_rank(relevant),
                "ndcg": ndcg_at_k(relevant),
                "first_hit": next((i for i, r in enumerate(relevant, 1) if r), None),
            }
        )

    # --- headline ---------------------------------------------------------
    display.heading(f"Aggregate over {len(questions)} questions")
    display.table(
        ["metric", "value", "what it answers"],
        [
            [f"Recall@{args.k}", f"{np.mean([q['recall'] for q in per_question]):.3f}",
             "was an answer found at all?"],
            [f"Precision@{args.k}", f"{np.mean([q['precision'] for q in per_question]):.3f}",
             "how much of what we returned was useful?"],
            ["MRR", f"{np.mean([q['rr'] for q in per_question]):.3f}",
             "how high up was the first correct one?"],
            [f"NDCG@{args.k}", f"{np.mean([q['ndcg'] for q in per_question]):.3f}",
             "rank quality, discounted by position"],
        ],
        align="lll",
    )

    # --- where the answers landed ----------------------------------------
    display.heading("Rank of the first correct result")
    positions = [q["first_hit"] for q in per_question]
    counts = [[f"rank {r}", sum(1 for p in positions if p == r)] for r in range(1, args.k + 1)]
    counts.append(["not found", sum(1 for p in positions if p is None)])
    display.table(["position", "questions"], counts)

    found = [p for p in positions if p]
    if found:
        print(f"\n  Of the {len(found)} questions answered, {sum(1 for p in found if p == 1)}")
        print("  were answered at rank 1. That gap between Recall and MRR is")
        print("  exactly what a reranker exists to close -- see example 20.")

    # --- what failed -------------------------------------------------------
    display.heading("Questions with no correct result in the top-k")
    failures = [q for q in per_question if q["first_hit"] is None]
    if failures:
        display.table(
            ["id", "question", "expected story"],
            [[q["id"], display.truncate(q["question"], 42),
              display.truncate(q["story"], 24)] for q in failures],
            align="lll",
        )
        print(f"\n  {len(failures)} of {len(questions)} failed. This list is the most")
        print("  useful output here -- an aggregate score tells you how you are")
        print("  doing, and this tells you what to fix.")
    else:
        print("  None.")

    # --- how k changes the picture ---------------------------------------
    display.heading("The same system at different k")
    rows = []
    for k in (1, 3, 5, 10, 20):
        recalls, rrs = [], []
        for question in questions:
            vector = embedder.encode([question["question"]], show_progress=False)[0]
            hits = store.search(vector, k=k, strategy=args.strategy)
            relevant = [
                question["evidence"] in " ".join(h.chunk.context_text.split())
                for h in hits
            ]
            recalls.append(recall_at_k(relevant))
            rrs.append(reciprocal_rank(relevant))
        rows.append([k, f"{np.mean(recalls):.3f}", f"{np.mean(rrs):.3f}"])

    display.table(["k", "Recall@k", "MRR"], rows)
    print("\n  Recall rises with k and MRR barely moves. That is the signature")
    print("  of a retriever that finds the right passage but ranks it poorly,")
    print("  and it points at reranking rather than at better embeddings.")

    display.notice(
        "None of this needed a model, an API key or a human. Retrieval quality "
        "is measurable on its own, which is why the book insists on separating "
        "it from generation quality.",
        "Recall and MRR answer different questions. Recall asks whether the "
        "evidence was available; MRR asks whether it was prominent. A system "
        "can improve on one and not the other.",
        "The failure list matters more than the averages. 'Recall@5 is 0.63' "
        "is a status report; 'these eleven questions failed' is a work item.",
        "NDCG is reported for completeness, but with one correct answer per "
        "question and no graded relevance it adds little over MRR here. It "
        "earns its place when passages are partly relevant.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
