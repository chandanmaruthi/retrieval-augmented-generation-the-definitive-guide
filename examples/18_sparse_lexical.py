#!/usr/bin/env python3
"""
18. Sparse Lexical Retrieval (and why it is not BM25)
Book: Chapter 9, Hybrid RAG
WHAT THIS SHOWS: The other half of hybrid retrieval. Dense vectors match
meaning; sparse lexical matching matches words. Each finds things the other
cannot.

It also corrects something the book -- and most RAG writing -- glosses over.
Postgres full-text search is *not* BM25. `ts_rank_cd` is a cover-density
measure with no inverse-document-frequency term at all, so it does not know
that "Holmes" is worthless as a discriminator in this corpus and "carbuncle"
is gold. This example implements real BM25 in about twenty-five lines of
NumPy, runs both over the same chunks, and shows where they disagree.

HOW THIS SCRIPT PROCEEDS
    1. Run Postgres full-text search        ts_rank_cd
    2. Run real Okapi BM25 over the same chunks   <-- written out, 25 lines
    3. Compare the two rankings
    4. Print the IDF of each query term     <-- the thing ts_rank lacks

Step 4 is the payoff. It shows in one small table exactly what Postgres
full-text search does not know.


WHY THIS EXISTS SEPARATELY FROM EXAMPLE 19
    Almost every RAG tutorial calls the Postgres arm of a hybrid retriever
    "BM25". It is not. ts_rank_cd measures cover density -- how tightly the
    query terms cluster -- and has no inverse-document-frequency term at all,
    so it does not know that "Holmes" is worthless as a discriminator in this
    corpus while "carbuncle" is gold.

    Getting that wrong in a design document changes which passages reach the
    model, so the difference is measured here rather than asserted.


REQUIRES: Postgres
RUN: python examples/18_sparse_lexical.py ["your query"]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg

DEFAULT_QUERY = "blue carbuncle goose"

# Okapi BM25's usual constants. k1 controls how fast term frequency saturates
# -- above it, a fifth occurrence of a word adds almost nothing. b controls
# length normalisation: 1.0 fully penalises long documents, 0.0 ignores length.
K1 = 1.5
B = 0.75

_WORD = re.compile(r"[a-z0-9']+")

# A small stopword list. BM25's IDF term already discounts common words
# heavily, so this matters less than it would for ts_rank -- but "the"
# appearing 8,000 times still wastes work.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "is", "was",
    "it", "that", "this", "for", "on", "with", "as", "at", "by", "from",
    "he", "she", "i", "you", "his", "her", "my", "we", "they", "had", "have",
}


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class BM25:
    """Okapi BM25, written out rather than imported.

    Worth reading once, because the whole difference from ts_rank is the
    `idf` line. BM25 asks how *surprising* a term is across the corpus;
    Postgres ts_rank does not ask at all.
    """

    def __init__(self, documents: list[str]) -> None:
        self.tokenized = [tokenize(d) for d in documents]
        self.n = len(documents)
        self.lengths = np.array([len(t) for t in self.tokenized], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if self.n else 0.0

        # Document frequency: how many documents contain each term.
        self.df: Counter = Counter()
        for tokens in self.tokenized:
            self.df.update(set(tokens))

        self.term_frequency = [Counter(t) for t in self.tokenized]

    def idf(self, term: str) -> float:
        """Inverse document frequency, the part ts_rank has no equivalent of.

        A term in every document scores near zero; a term in one document
        scores high. The +0.5 terms are BM25's smoothing, which keeps the
        logarithm finite for a term that appears nowhere.
        """
        # ---- THE KEY LINE ------------------------------------------
        # This single expression is the entire difference from Postgres
        # ts_rank. A term appearing in every document scores near zero; a
        # term in one document scores high. Postgres full-text search has
        # no equivalent of it.
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)

        for term in terms:
            if term not in self.df:
                continue
            idf = self.idf(term)
            for index, frequency in enumerate(self.term_frequency):
                tf = frequency.get(term, 0)
                if not tf:
                    continue
                # Saturating term frequency, normalised by document length.
                norm = 1 - B + B * (self.lengths[index] / self.avg_length)
                scores[index] += idf * (tf * (K1 + 1)) / (tf + K1 * norm)

        return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "18. Sparse Lexical Retrieval",
        "Chapter 9, Hybrid RAG",
        "Word matching rather than meaning matching -- and the difference "
        "between what Postgres gives you and what BM25 actually is.",
    )

    if not pg.available():
        display.missing_database(
            "This example compares Postgres full-text search against a local "
            "BM25 implementation over the same indexed chunks."
        )
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    store = pg.PgVectorStore(conn)
    if store.count(args.strategy) == 0:
        print(f"\nNothing indexed. Run: python scripts/index_corpus.py\n")
        return 1

    print(f"  strategy: {args.strategy} ({store.count(args.strategy):,} chunks)")
    print(f"  query: {args.query!r}\n")

    # --- Postgres full-text ---------------------------------------------
    display.heading("Postgres full-text search (ts_rank_cd)")
    ts_hits = store.search_lexical(args.query, k=args.k, strategy=args.strategy)
    display.table(
        ["rank", "ts_rank", "story", "passage"],
        [
            [h.rank, f"{h.score:.4f}", display.truncate(h.chunk.story, 24),
             display.truncate(h.chunk.text, 34)]
            for h in ts_hits
        ],
        align="rrll",
    )
    if len(ts_hits) < args.k:
        print(f"\n  Only {len(ts_hits)} result(s), not {args.k}. websearch_to_tsquery")
        print("  joins bare terms with AND, so a chunk must contain every word to")
        print("  match at all. That is a recall cliff: one unusual word in the")
        print("  query and the lexical arm returns nothing. Quoting the query as")
        print("  'blue OR carbuncle OR goose' changes this.")
    print("\n  Note the magnitude of those scores. ts_rank_cd with normalisation")
    print("  flag 32 is bounded to (0, 1) but lands around 0.001 -- which matters")
    print("  enormously when example 19 tries to add it to a cosine similarity")
    print("  that lands around 0.5.")

    # --- real BM25 --------------------------------------------------------
    display.heading("Okapi BM25 over the same chunks")
    rows = conn.execute(
        "SELECT id, body, story FROM chunks WHERE strategy = %s ORDER BY ordinal",
        [args.strategy],
    ).fetchall()
    bodies = [r[1] for r in rows]

    bm25 = BM25(bodies)
    scores = bm25.score(args.query)
    order = np.argsort(-scores)[: args.k]

    display.table(
        ["rank", "bm25", "story", "passage"],
        [
            [rank, f"{scores[i]:.2f}", display.truncate(rows[i][2], 24),
             display.truncate(rows[i][1], 34)]
            for rank, i in enumerate(order, start=1)
        ],
        align="rrll",
    )

    # --- where they disagree ---------------------------------------------
    display.heading("Where the two rankings disagree")
    ts_ids = [h.chunk.index for h in ts_hits]
    bm_ids = [rows[i][0] for i in order]
    overlap = len(set(ts_ids) & set(bm_ids))
    display.kv("shared in top-%d" % args.k, f"{overlap} of {args.k}")

    print("\n  IDF per query term -- what BM25 knows and ts_rank does not:\n")
    display.table(
        ["term", "chunks containing it", "IDF"],
        [
            [term, bm25.df.get(term, 0), f"{bm25.idf(term):.2f}"]
            for term in tokenize(args.query)
        ],
    )
    print("\n  A term in almost every chunk gets a low IDF and is nearly ignored.")
    print("  A rare term dominates the score. That weighting is the single")
    print("  biggest difference between these two rankers, and Postgres")
    print("  full-text search simply does not have it.")

    display.notice(
        "Postgres full-text search is not BM25. There is no IDF term, and "
        "ts_rank_cd measures cover density -- how tightly the query terms "
        "cluster -- rather than term informativeness.",
        f"The two rankings shared {overlap} of {args.k} results here. Calling "
        "the Postgres arm 'BM25' in a design document would be wrong in a way "
        "that changes which passages reach the model.",
        "This still earns its place in a hybrid system. Lexical matching finds "
        "exact names, quotations and rare terms that a 384-dimension embedding "
        "smooths away -- which is precisely what example 19 fuses.",
        "If you need real BM25 inside Postgres, extensions like pg_search "
        "provide it. Using ts_rank and knowing what it is beats using it and "
        "believing it is something else.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
