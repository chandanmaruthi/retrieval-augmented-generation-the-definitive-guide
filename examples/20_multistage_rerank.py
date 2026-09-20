#!/usr/bin/env python3
"""
20. Multi-Stage Retrieval
Book: Chapter 11, Multi-Stage Retrieval

WHAT THIS SHOWS: Cast wide, then narrow.

    stage 1  cheap retrieval, high recall     -> N candidates (N = 100..1000)
    stage 2  expensive reranking, high precision -> top k

The asymmetry that makes this work: a bi-encoder embeds the query and the
document separately, so documents can be embedded once, in advance. A
cross-encoder reads the query and the document *together*, which is far more
accurate and far too slow to run over a whole corpus. Running it over a
hundred candidates is affordable; running it over four hundred thousand is
not.

Also implements MMR, which trades a little relevance for diversity -- because
five chunks that all say the same thing are worth less than four good ones and
something different.

HOW THIS SCRIPT PROCEEDS
    1. Stage 1: retrieve N candidates cheaply     bi-encoder, N = 50 here
    2. Stage 2a: rerank them with a cross-encoder <-- THE TECHNIQUE
    3. Show which results were PROMOTED from deep in the list
    4. Project what stage 2 would cost corpus-wide  <-- why it is two stages
    5. Stage 2b: MMR diversification              a different stage-2 job

Step 3 is the evidence: on this corpus the reranker pulls results from ranks
43, 38 and 29 into the top 5. Those are answers a single-stage retriever would
never show, no matter how good its embeddings.


THE ASYMMETRY THIS EXPLOITS
    A bi-encoder embeds query and document SEPARATELY, so documents can be
    embedded once, in advance. A cross-encoder reads them TOGETHER, which is
    far more accurate and cannot be precomputed at all.

    Step 4 measures that: cross-encoding all 413 chunks takes ~2.9s per query
    against 3.7ms for the bi-encoder. Running it over 50 candidates is
    affordable; running it over a corpus is not. Everything about the
    two-stage design follows from that one fact.


REQUIRES: Postgres. Cross-encoder needs -r requirements-local.txt.
RUN: python examples/20_multistage_rerank.py ["your question"]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import describe, get_embedder

DEFAULT_QUESTION = "Why did Dr. Roylott want to harm his stepdaughters?"

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_cross_encoder():
    """A local reranker -- no API key, about 90MB of weights."""
    import logging
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sentence_transformers import CrossEncoder

        for name in ("sentence_transformers", "transformers", "huggingface_hub"):
            logging.getLogger(name).setLevel(logging.ERROR)
        return CrossEncoder(CROSS_ENCODER)


def mmr(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    k: int,
    lambda_: float = 0.6,
) -> list[int]:
    """Maximal Marginal Relevance -- greedy selection balancing relevance
    against redundancy. The book names it as a stage-2 diversification step
    because near-duplicate chunks are common in real corpora, and guaranteed
    with overlapping chunking strategies.
    """
    # ---- THE KEY IDEA ----------------------------------------------
    # Two matrices, and the second is what plain top-k never computes:
    # how similar the candidates are TO EACH OTHER. Top-k picks the five
    # best chunks independently, which on an overlapping corpus is often
    # the same chunk five times.
    relevance = candidate_vectors @ query_vector
    similarity = candidate_vectors @ candidate_vectors.T

    selected: list[int] = [int(np.argmax(relevance))]
    while len(selected) < min(k, len(candidate_vectors)):
        best_index, best_score = None, -np.inf
        for index in range(len(candidate_vectors)):
            if index in selected:
                continue
            # The MMR objective: reward relevance, penalise resemblance
            # to what is already chosen. lambda_ = 1.0 reduces exactly to
            # plain top-k.
            redundancy = max(similarity[index][j] for j in selected)
            score = lambda_ * relevance[index] - (1 - lambda_) * redundancy
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            break
        selected.append(best_index)

    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--fanout", type=int, default=50, help="stage-1 candidates")
    parser.add_argument("--k", type=int, default=5, help="final results")
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "20. Multi-Stage Retrieval",
        "Chapter 11, Multi-Stage Retrieval",
        f"Retrieve {args.fanout} candidates cheaply, then rerank them "
        f"expensively down to {args.k}.",
    )

    if not pg.available():
        display.missing_database("Stage 1 retrieves from the pgvector index.")
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
    print(f"  question: {args.question!r}\n")

    query_vector = embedder.encode([args.question], show_progress=False)[0]

    # --- stage 1 ----------------------------------------------------------
    display.heading(f"Stage 1: candidate generation (N = {args.fanout})")
    started = time.perf_counter()
    candidates = store.search(query_vector, k=args.fanout, strategy=args.strategy)
    stage1_ms = (time.perf_counter() - started) * 1000
    display.kv("candidates", len(candidates))
    display.kv("time", f"{stage1_ms:.1f} ms")
    print("\n  Top 5 by bi-encoder similarity:")
    display.table(
        ["rank", "score", "passage"],
        [[h.rank, f"{h.score:.3f}", display.truncate(h.chunk.text, 52)]
         for h in candidates[:5]],
        align="rrl",
    )

    # --- stage 2: cross-encoder ------------------------------------------
    display.heading("Stage 2a: cross-encoder reranking")
    try:
        reranker = load_cross_encoder()
    except ImportError:
        display.missing_dependency(
            "sentence-transformers", "requirements-local.txt",
            "Stage 1 ran and is shown above. The cross-encoder needs the "
            "local model stack.",
        )
        reranker = None

    if reranker is not None:
        print(f"  model: {CROSS_ENCODER}")
        pairs = [(args.question, h.chunk.text) for h in candidates]
        started = time.perf_counter()
        scores = reranker.predict(pairs, show_progress_bar=False)
        stage2_ms = (time.perf_counter() - started) * 1000
        display.kv("pairs scored", len(pairs))
        display.kv("time", f"{stage2_ms:.1f} ms")
        display.kv("per pair", f"{stage2_ms / len(pairs):.1f} ms")

        order = np.argsort(-np.asarray(scores))
        print("\n  Top 5 after reranking:")
        display.table(
            ["new", "was", "cross-enc", "bi-enc", "passage"],
            [
                [
                    new_rank,
                    candidates[int(i)].rank,
                    f"{scores[int(i)]:.2f}",
                    f"{candidates[int(i)].score:.3f}",
                    display.truncate(candidates[int(i)].chunk.text, 38),
                ]
                for new_rank, i in enumerate(order[: args.k], start=1)
            ],
            align="rrrrl",
        )

        moved = sum(
            1 for new_rank, i in enumerate(order[: args.k], start=1)
            if candidates[int(i)].rank != new_rank
        )
        print(f"\n  {moved} of the top {args.k} changed position.")
        promoted = [
            (candidates[int(i)].rank, new_rank)
            for new_rank, i in enumerate(order[: args.k], start=1)
            if candidates[int(i)].rank > args.k
        ]
        if promoted:
            print(f"  {len(promoted)} came from outside the original top-{args.k} "
                  f"(was rank {', '.join(str(p[0]) for p in promoted)}).")
            print("  Those are results a single-stage retriever would never have")
            print("  shown, no matter how good its embeddings were.")

        # --- cost projection ---------------------------------------------
        display.heading("Why this is two stages and not one")
        per_pair = stage2_ms / len(pairs)
        total = store.count(args.strategy)
        print(f"  Cross-encoding all {total:,} chunks would take about")
        print(f"  {per_pair * total / 1000:.1f}s per query, against {stage1_ms:.1f} ms")
        print(f"  for the bi-encoder over the same set.")
        print(f"\n  A corpus 1,000x this size would take "
              f"{per_pair * total * 1000 / 1000 / 60:.0f} minutes per query.")
        print("  That is the asymmetry the two-stage design exists to exploit.")

    # --- stage 2b: MMR ----------------------------------------------------
    display.heading("Stage 2b: MMR diversification")
    candidate_vectors = embedder.encode(
        [h.chunk.text for h in candidates], show_progress=False
    )
    plain = list(range(args.k))
    diverse = mmr(query_vector, candidate_vectors, k=args.k, lambda_=0.6)

    def mean_pairwise(indices: list[int]) -> float:
        if len(indices) < 2:
            return 0.0
        vectors = candidate_vectors[indices]
        similarity = vectors @ vectors.T
        upper = similarity[np.triu_indices(len(indices), k=1)]
        return float(np.mean(upper))

    display.table(
        ["selection", "mean pairwise similarity", "chunks"],
        [
            ["plain top-k", f"{mean_pairwise(plain):.3f}",
             ", ".join(str(candidates[i].chunk.index) for i in plain)],
            ["MMR (lambda=0.6)", f"{mean_pairwise(diverse):.3f}",
             ", ".join(str(candidates[i].chunk.index) for i in diverse)],
        ],
        align="lll",
    )
    print("\n  Lower pairwise similarity means the selected chunks repeat each")
    print("  other less. With a fixed context budget, redundancy is wasted")
    print("  budget -- the model gains nothing from reading the same scene twice.")

    display.notice(
        "Stage 1 optimises recall, stage 2 optimises precision. Asking one "
        "retriever to do both is what forces the usual bad compromise.",
        "A cross-encoder reads query and document together, which is why it is "
        "both more accurate and too slow to run corpus-wide. The fan-out is "
        "what makes it affordable.",
        "Reranking can promote a chunk from outside the original top-k. That "
        "is the whole value: stage 1 only has to get the answer into the "
        "candidate set, not to the top of it.",
        "MMR optimises the set rather than each item. Top-k picks the five "
        "best chunks independently, which on an overlapping corpus is often "
        "the same chunk five times.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
