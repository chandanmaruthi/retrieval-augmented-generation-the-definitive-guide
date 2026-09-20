#!/usr/bin/env python3
"""
21. Late Interaction (ColBERT-style max-sim)
Book: Chapter 11, Multi-Stage Retrieval

WHAT THIS SHOWS: The middle ground between a bi-encoder and a cross-encoder.

    bi-encoder      one vector per document. Fast, precomputable, lossy --
                    a 300-token passage is averaged into 384 numbers.
    cross-encoder   query and document read together. Accurate, and nothing
                    can be precomputed.
    late interaction  one vector per *token*, precomputable, and the
                    query-document comparison happens at query time.

The scoring function is MaxSim: for each query token, find the document token
it matches best, and sum those maxima. A document scores well when every part
of the query finds something, which is a meaningfully different question from
"is the average of this document near the average of this query".

This implementation uses MiniLM token embeddings rather than real ColBERT
weights, so it demonstrates the mechanism rather than reproducing ColBERT's
published quality. That distinction is stated in the output too.

HOW THIS SCRIPT PROCEEDS
    1. Stage 1: bi-encoder candidates        as in example 20
    2. Compute PER-TOKEN vectors for each    <-- the thing normally thrown away
    3. Score with MaxSim                     <-- THE TECHNIQUE, 3 lines
    4. Show which query token matched where  <-- the interpretability payoff
    5. Compute the storage multiplier

Step 4 prints a table no bi-encoder can produce: for each word of the query,
the word in the passage that matched it and how strongly.


WHERE THIS SITS BETWEEN 20's TWO OPTIONS
    bi-encoder       one vector per document. Precomputable, and lossy -- a
                     300-token passage averaged into 384 numbers.
    cross-encoder    query and document read together. Accurate, nothing
                     precomputable.
    late interaction one vector per TOKEN, still precomputable, with the
                     comparison deferred to query time. Hence "late".


REQUIRES: Postgres and -r requirements-local.txt
RUN: python examples/21_late_interaction.py ["your question"]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import get_embedder

DEFAULT_QUESTION = "What did the engineer lose to the hydraulic press?"


def token_embeddings(model, texts: list[str]) -> list[np.ndarray]:
    """Per-token vectors, L2-normalised, padding removed.

    A normal sentence embedder pools these into one vector and throws the rest
    away. Late interaction is what you get by keeping them.
    """
    import torch

    tokenizer = model.tokenizer
    transformer = model[0].auto_model

    # sentence-transformers places the model on the best available device --
    # MPS on Apple silicon, CUDA elsewhere. Tokenizer output is always on the
    # CPU, and mixing the two raises "Passed CPU tensor to MPS op", so the
    # inputs have to be moved explicitly.
    device = next(transformer.parameters()).device

    out: list[np.ndarray] = []
    for start in range(0, len(texts), 16):
        batch = texts[start : start + 16]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            hidden = transformer(**encoded).last_hidden_state

        hidden = torch.nn.functional.normalize(hidden, p=2, dim=2)

        for row in range(len(batch)):
            # Drop padding positions: their vectors are arbitrary, and a
            # MaxSim over them would let a short document score on noise.
            mask = encoded["attention_mask"][row].bool()
            out.append(hidden[row][mask].cpu().numpy().astype(np.float32))

    return out


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """ColBERT's scoring function.

    For every query token, take its best match anywhere in the document, then
    sum. The sum (rather than the mean) is deliberate in ColBERT; we divide by
    the query length here only so scores from different queries are
    comparable in the table below.
    """
    # ---- THE KEY LINES ---------------------------------------------
    # For every query token, take its best match ANYWHERE in the document
    # (max over axis 1), then sum those maxima.
    #
    # The question this asks is "did every part of the query find
    # something?" -- which is a different question from "are these two
    # averaged vectors close?", and they disagree most on long passages.
    similarity = query_tokens @ doc_tokens.T   # (query_len, doc_len)
    return float(similarity.max(axis=1).sum() / len(query_tokens))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--fanout", type=int, default=30)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "21. Late Interaction (ColBERT-style MaxSim)",
        "Chapter 11, Multi-Stage Retrieval",
        "One vector per token instead of one per document, compared at query "
        "time with MaxSim.",
    )

    if not pg.available():
        display.missing_database("Candidates come from the pgvector index.")
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    try:
        embedder = get_embedder()
        if not embedder.semantic:
            raise ImportError
        model = embedder._inner._model  # the underlying SentenceTransformer
    except (ImportError, AttributeError):
        display.missing_dependency(
            "sentence-transformers", "requirements-local.txt",
            "Late interaction needs per-token embeddings from a transformer.",
        )
        return 0

    store = pg.PgVectorStore(conn, dim=embedder.dim, model=embedder.name)
    if store.count(args.strategy) == 0:
        print("\nNothing indexed. Run: python scripts/index_corpus.py\n")
        return 1

    print("  NOTE: this uses MiniLM's token embeddings, not trained ColBERT")
    print("  weights. The mechanism is faithful; the quality is not ColBERT's.")
    print(f"\n  question: {args.question!r}\n")

    query_vector = embedder.encode([args.question], show_progress=False)[0]
    candidates = store.search(query_vector, k=args.fanout, strategy=args.strategy)

    display.heading(f"Stage 1: {args.fanout} candidates from the bi-encoder")
    display.table(
        ["rank", "cosine", "passage"],
        [[h.rank, f"{h.score:.3f}", display.truncate(h.chunk.text, 52)]
         for h in candidates[:5]],
        align="rrl",
    )

    # --- token embeddings -------------------------------------------------
    display.heading("Stage 2: MaxSim over token embeddings")
    started = time.perf_counter()
    query_tokens = token_embeddings(model, [args.question])[0]
    doc_tokens = token_embeddings(model, [h.chunk.text for h in candidates])
    embed_ms = (time.perf_counter() - started) * 1000

    display.kv("query tokens", len(query_tokens))
    display.kv("mean document tokens", f"{np.mean([len(d) for d in doc_tokens]):.0f}")
    display.kv("vectors held for these 30 docs", f"{sum(len(d) for d in doc_tokens):,}")
    display.kv("time", f"{embed_ms:.0f} ms")

    started = time.perf_counter()
    scores = np.array([maxsim(query_tokens, d) for d in doc_tokens])
    score_ms = (time.perf_counter() - started) * 1000
    display.kv("scoring time", f"{score_ms:.1f} ms")

    order = np.argsort(-scores)
    print("\n  Reranked by MaxSim:")
    display.table(
        ["new", "was", "maxsim", "cosine", "passage"],
        [
            [
                rank,
                candidates[int(i)].rank,
                f"{scores[int(i)]:.3f}",
                f"{candidates[int(i)].score:.3f}",
                display.truncate(candidates[int(i)].chunk.text, 36),
            ]
            for rank, i in enumerate(order[: args.k], start=1)
        ],
        align="rrrrl",
    )

    moved = sum(
        1 for rank, i in enumerate(order[: args.k], start=1)
        if candidates[int(i)].rank != rank
    )
    print(f"\n  {moved} of the top {args.k} changed position.")

    # --- what MaxSim actually matched -------------------------------------
    display.heading("Which query token matched where")
    best = int(order[0])
    tokenizer = model.tokenizer
    query_pieces = tokenizer.convert_ids_to_tokens(
        tokenizer(args.question, truncation=True, max_length=256)["input_ids"]
    )
    doc_pieces = tokenizer.convert_ids_to_tokens(
        tokenizer(candidates[best].chunk.text, truncation=True, max_length=256)["input_ids"]
    )
    similarity = query_tokens @ doc_tokens[best].T

    print("  For the top-ranked passage, each query token and its best match:\n")
    rows = []
    for index, piece in enumerate(query_pieces[: len(query_tokens)]):
        if piece in ("[CLS]", "[SEP]"):
            continue
        match = int(similarity[index].argmax())
        rows.append([piece, doc_pieces[match] if match < len(doc_pieces) else "?",
                     f"{similarity[index][match]:.3f}"])
    display.table(["query token", "best match in passage", "similarity"], rows[:12],
                  align="lll")
    print("\n  This is the interpretability a single pooled vector cannot give")
    print("  you. A bi-encoder produces one number; here you can see which")
    print("  word carried the match and which query terms found nothing.")

    display.heading("The cost")
    total = store.count(args.strategy)
    per_doc = np.mean([len(d) for d in doc_tokens])
    print(f"  Storing token vectors for all {total:,} chunks would need about")
    print(f"  {total * per_doc:,.0f} vectors instead of {total:,} --")
    print(f"  roughly {per_doc:.0f}x the index size.")
    print("\n  That is why late interaction is normally a stage-2 reranker over")
    print("  a few dozen candidates, exactly as used here. Real ColBERT reduces")
    print("  the multiplier with dimension reduction and quantisation, but the")
    print("  shape of the trade does not change.")

    display.notice(
        "MaxSim asks whether every part of the query found something, rather "
        "than whether two averaged vectors are close. Those are different "
        "questions, and they disagree most on long passages.",
        f"Token vectors cost about {per_doc:.0f}x the storage of pooled ones, "
        "which is why this sits in stage 2 rather than replacing the index.",
        "Padding tokens must be masked out before scoring. Leave them in and "
        "short documents score on meaningless positions.",
        "Unlike a cross-encoder, document token vectors can be precomputed -- "
        "only the comparison is deferred. That is the 'late' in late "
        "interaction.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
