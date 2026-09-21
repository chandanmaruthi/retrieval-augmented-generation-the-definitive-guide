#!/usr/bin/env python3
"""
34. Fine-Tuning Data Preparation (hard negatives)
Book: Chapter 21, Domain-Specific Fine-Tuning
WHAT THIS SHOWS: How to build the training set for a retriever, and why the
negatives matter far more than the positives.

Contrastive training needs triples: (query, positive passage, negative
passage). Positives are easy -- they come from the gold set. The negatives are
where the work is, and where most retriever fine-tuning goes wrong.

    random negatives   a passage from another story. Trivially separable, so
                       the model learns nothing and loss drops beautifully.
    hard negatives     a passage the current retriever ranks highly and that
                       is nonetheless wrong. This is what teaches it.

This script mines hard negatives from the live index, checks them for false
negatives, and writes a JSONL training file. It does not run training -- that
needs a GPU and a different repository -- but everything up to that point is
here and is the part that decides whether training helps.

HOW THIS SCRIPT PROCEEDS
    1. For each gold question, find its positive passage
    2. Mine HARD negatives from the live index   <-- THE TECHNIQUE
    3. Filter out false negatives                <-- not optional, see below
    4. Add a random negative for contrast
    5. Measure the margin: positive vs hard vs random

Step 5 is the diagnostic. If hard negatives score ABOVE the positives, the
retriever is actively preferring wrong passages -- which is both the reason to
fine-tune and the measurement that says so.


WHY THE NEGATIVES ARE THE TRAINING SET
    Positives are easy; they come from the gold set. The negatives are where
    the work is, and where most retriever fine-tuning goes wrong.

    A random negative is a passage from another story -- trivially separable,
    so the model learns nothing while the loss curve looks beautiful. A hard
    negative is one the CURRENT retriever ranks highly and that is
    nonetheless wrong. That narrow margin is the entire training signal.

    Step 3 is mandatory: a passage that ranks highly and IS correct looks
    exactly like a hard negative from the outside, and training on it teaches
    the model to push away a right answer.


REQUIRES: Postgres
RUN: python examples/34_finetuning_data_prep.py [--out train.jsonl]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import describe, get_embedder

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"

# Mine negatives from ranks 2..N. Rank 1 is skipped when it is the positive,
# and very low ranks are too easy to be useful.
NEGATIVE_POOL = 20
NEGATIVES_PER_QUERY = 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "34. Fine-Tuning Data Preparation",
        "Chapter 21, Domain-Specific Fine-Tuning",
        "Building contrastive training triples, and mining the hard negatives "
        "that make them worth training on.",
    )

    if not pg.available():
        display.missing_database("Hard negatives are mined from the live index.")
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
    print(f"  {len(questions)} gold questions\n")

    all_chunks = conn.execute(
        "SELECT id, body, story FROM chunks WHERE strategy = %s", [args.strategy]
    ).fetchall()
    by_id = {int(r[0]): (r[1], r[2]) for r in all_chunks}

    triples = []
    stats = {"with_positive": 0, "no_positive": 0, "false_negatives_caught": 0}
    random.seed(0)

    for question in questions:
        vector = embedder.encode([question["question"]], show_progress=False)[0]
        hits = store.search(vector, k=NEGATIVE_POOL, strategy=args.strategy)

        evidence = question["evidence"]

        positives = [
            h for h in hits if evidence in " ".join(h.chunk.context_text.split())
        ]
        if not positives:
            # The current retriever cannot find this answer at all. It still
            # belongs in training -- these are exactly the cases fine-tuning
            # should fix -- so the positive is fetched by evidence instead.
            match = next(
                (cid for cid, (body, _) in by_id.items()
                 if evidence in " ".join(body.split())),
                None,
            )
            if match is None:
                stats["no_positive"] += 1
                continue
            positive_text = by_id[match][0]
            positive_id = match
        else:
            positive_text = positives[0].chunk.text
            positive_id = positives[0].chunk.index
            stats["with_positive"] += 1

        # Hard negatives: highly ranked AND wrong.
        hard = []
        for hit in hits:
            if hit.chunk.index == positive_id:
                continue
            text = " ".join(hit.chunk.context_text.split())
            if evidence in text:
                # Ranked highly and actually correct -- a false negative.
                # Training on this teaches the model to push away a right
                # answer, which is worse than not training at all.
                stats["false_negatives_caught"] += 1
                continue
            hard.append(hit)
            if len(hard) == NEGATIVES_PER_QUERY:
                break

        # One random negative for contrast in the report below.
        random_negative = random.choice(
            [c for c in all_chunks if int(c[0]) != positive_id]
        )

        triples.append(
            {
                "query": question["question"],
                "positive": positive_text,
                "hard_negatives": [h.chunk.text for h in hard],
                "hard_negative_ranks": [h.rank for h in hard],
                "random_negative": random_negative[1],
                "story": question["story"],
            }
        )

    # --- what was built ---------------------------------------------------
    display.heading("Triples built")
    display.kv("training triples", len(triples))
    display.kv("positive found by retriever", stats["with_positive"])
    display.kv("positive recovered by evidence", len(triples) - stats["with_positive"])
    display.kv("false negatives filtered out", stats["false_negatives_caught"])
    display.kv("questions dropped", stats["no_positive"])

    print("\n  The false-negative filter is not optional. A passage that ranks")
    print("  highly and IS correct looks exactly like a hard negative from the")
    print("  outside. Train on it and you teach the retriever to push away a")
    print("  right answer.")

    # --- hard vs random ---------------------------------------------------
    display.heading("Hard negatives vs random negatives")
    example = next(t for t in triples if t["hard_negatives"])
    print(f"  query: {example['query']!r}\n")
    print("  POSITIVE:")
    display.preview(example["positive"], limit=180)
    print(f"\n  HARD NEGATIVE (ranked #{example['hard_negative_ranks'][0]} by the current retriever):")
    display.preview(example["hard_negatives"][0], limit=180)
    print("\n  RANDOM NEGATIVE:")
    display.preview(example["random_negative"], limit=180)

    print("\n  Read those three. The hard negative is on-topic, same register,")
    print("  often the same story -- and does not answer the question. The")
    print("  random one is obviously unrelated.")

    # --- why it matters ----------------------------------------------------
    display.heading("Why random negatives teach nothing")
    positives_v = embedder.encode([t["positive"] for t in triples], show_progress=False)
    queries_v = embedder.encode([t["query"] for t in triples], show_progress=False)
    hard_v = embedder.encode(
        [t["hard_negatives"][0] if t["hard_negatives"] else t["random_negative"]
         for t in triples],
        show_progress=False,
    )
    random_v = embedder.encode([t["random_negative"] for t in triples], show_progress=False)

    import numpy as np

    pos_sim = float(np.mean(np.sum(queries_v * positives_v, axis=1)))
    hard_sim = float(np.mean(np.sum(queries_v * hard_v, axis=1)))
    rand_sim = float(np.mean(np.sum(queries_v * random_v, axis=1)))

    display.table(
        ["passage type", "mean similarity to query", "margin vs positive"],
        [
            ["positive", f"{pos_sim:.3f}", "-"],
            ["hard negative", f"{hard_sim:.3f}", f"{pos_sim - hard_sim:+.3f}"],
            ["random negative", f"{rand_sim:.3f}", f"{pos_sim - rand_sim:+.3f}"],
        ],
    )
    print(f"\n  Random negatives are already {pos_sim - rand_sim:.3f} below the")
    print("  positives, so a loss computed against them is near zero before")
    print("  training starts. There is nothing left to learn from them.")

    if hard_sim > pos_sim:
        print(f"\n  The hard negatives score {hard_sim - pos_sim:.3f} ABOVE the")
        print("  positives. The current retriever ranks these wrong passages")
        print("  higher than the right ones -- which is not a quirk of the")
        print("  mining, it is the direct cause of the recall measured in")
        print("  example 31. This is the case fine-tuning exists to fix, and")
        print("  the sign of that margin is how you know it is worth doing.")
    else:
        print(f"\n  The hard negatives sit only {pos_sim - hard_sim:.3f} below the")
        print("  positives. That narrow margin is the entire training signal.")

    # --- output -----------------------------------------------------------
    if args.out:
        path = Path(args.out)
        with path.open("w", encoding="utf-8") as handle:
            for triple in triples:
                handle.write(json.dumps({
                    "query": triple["query"],
                    "positive": triple["positive"],
                    "negatives": triple["hard_negatives"],
                }, ensure_ascii=False) + "\n")
        print(f"\n  Wrote {len(triples)} triples to {path}")

    display.heading("What training would look like")
    print("  This file stops here deliberately -- training needs a GPU and a")
    print("  different toolchain. The shape, for reference:\n")
    print("      MultipleNegativesRankingLoss over (query, positive, negatives)")
    print("      LoRA adapters rather than full fine-tuning, for a set this small")
    print("      held-out split by STORY, not by question")
    print("\n  That last point is the one that bites. Split by question and")
    print("  passages from the same story appear in both train and test, so the")
    print("  model is evaluated on text it was trained on and the numbers lie.")

    display.notice(
        "Negatives are the training set. Positives just say what right looks "
        "like; negatives say what wrong-but-plausible looks like, and that is "
        "the boundary the model has to learn.",
        f"Random negatives are already separated by {pos_sim - rand_sim:.3f}. "
        "Training against them produces a satisfying loss curve and no "
        "improvement in retrieval.",
        f"Hard negatives here score {hard_sim:.3f} against the positives' "
        f"{pos_sim:.3f}. When that gap is the wrong way round, the retriever "
        "is actively preferring wrong passages -- which is both the reason to "
        "fine-tune and the measurement that says so.",
        f"{stats['false_negatives_caught']} correct passages were caught "
        "masquerading as hard negatives. Mining negatives from your own "
        "retriever without this check actively damages the model.",
        "Split held-out data by document, not by question. Same-document "
        "leakage is the most common way a fine-tuned retriever reports gains "
        "it does not have.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
