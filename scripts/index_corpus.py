#!/usr/bin/env python3
"""
Chunk, embed and load the corpus into Postgres.

Runs a chunking strategy's own code from examples/, so what lands in the
database is exactly what the example script you just read produces.

    python scripts/index_corpus.py                      # the default strategy
    python scripts/index_corpus.py --strategy sentence  # a different one
    python scripts/index_corpus.py --all                # every free strategy
    python scripts/index_corpus.py --list               # what is loaded now
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from ragkit import display, pg
from ragkit.corpus import TEXT_PATH, load_book
from ragkit.embed import describe, get_embedder

# name -> (module, function, book section). None means a custom call shape.
STRATEGIES = {
    "fixed_token": ("02_chunk_fixed_token", "chunk_fixed_token", "3.1"),
    "sentence": ("03_chunk_sentence", "chunk_sentences", "3.2"),
    "paragraph": ("04_chunk_paragraph", "chunk_paragraphs", "3.3"),
    "heading": ("05_chunk_heading", "chunk_by_heading", "3.4"),
    "semantic": ("06_chunk_semantic", None, "3.5"),
    "sentence_window": ("07_chunk_sentence_window", "chunk_sentence_window", "3.6"),
    "parent_child": ("08_chunk_parent_child", None, "3.7"),
    "contextual_header": ("09_chunk_contextual_header", "chunk_with_headers", "3.8"),
}

# The examples that need one strategy present use this one. It won the
# equal-budget comparison in example 15.
DEFAULT_STRATEGY = "contextual_header"


def build(name: str, book, embedder):
    module_name, function_name, _ = STRATEGIES[name]
    module = import_module(module_name)
    if name == "semantic":
        chunks, _ = module.chunk_semantic(book, embedder)
        return chunks
    if name == "parent_child":
        chunks, _ = module.chunk_parent_child(book)
        return chunks
    return getattr(module, function_name)(book)


def assign_acl(chunk) -> list[str]:
    """Give each story a synthetic access label, for example 35.

    The corpus is public domain, so these are invented -- but they have to be
    attached at index time rather than bolted on later, because that is the
    whole point the privacy chapter makes. Four of the twelve stories are
    restricted to a group; the rest stay public.
    """
    restricted = {
        "The Adventure of the Speckled Band": "medical",
        "The Adventure of the Beryl Coronet": "finance",
        "A Case of Identity": "finance",
        "The Boscombe Valley Mystery": "legal",
    }
    group = restricted.get(chunk.story)
    return [group] if group else ["public"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=list(STRATEGIES))
    parser.add_argument("--all", action="store_true", help="index every free strategy")
    parser.add_argument("--list", action="store_true", help="show what is loaded")
    args = parser.parse_args()

    if not pg.available():
        display.missing_database(f"Cannot index: {pg.why_unavailable()}.")
        return 1

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}")
        if error.remedy:
            print(f"\n{error.remedy}\n")
        return 1

    store = pg.PgVectorStore(conn)

    if args.list:
        loaded = store.strategies()
        if not loaded:
            print("\nNothing indexed yet. Run without --list.\n")
            return 0
        print()
        display.table(
            ["strategy", "chunks"], [[name, f"{count:,}"] for name, count in loaded]
        )
        print()
        return 0

    try:
        embedder = get_embedder()
    except ImportError:
        display.missing_dependency(
            "sentence-transformers", "requirements-local.txt",
            "Indexing needs an embedder.",
        )
        return 1

    print(f"\n{describe(embedder)}")
    if embedder.dim != store.dim:
        # The schema has one table per vector space; using the wrong one would
        # either error on insert or, worse, succeed against a different model's
        # rows. Fail loudly instead.
        store = pg.PgVectorStore(conn, dim=embedder.dim, model=embedder.name)
        print(f"  writing to {store.table}")

    book = load_book()
    document_id = store.upsert_document(
        slug="sherlock-holmes",
        title=book.title,
        author=book.author,
        source=str(TEXT_PATH.name),
        sha256=hashlib.sha256(TEXT_PATH.read_bytes()).hexdigest(),
        n_words=book.word_count,
    )

    names = list(STRATEGIES) if args.all else [args.strategy]

    for name in names:
        print(f"\n{name}")
        started = time.time()

        chunks = build(name, book, embedder)
        for chunk in chunks:
            chunk.meta["acl"] = assign_acl(chunk)

        vectors = embedder.encode([c.embed_text for c in chunks], show_progress=True)

        store.clear(name)
        written = store.add(chunks, vectors, strategy=name, document_id=document_id)

        restricted = sum(1 for c in chunks if c.meta["acl"] != ["public"])
        print(f"  {written:,} chunks in {time.time() - started:.1f}s "
              f"({restricted:,} access-restricted)")

    print()
    display.table(
        ["strategy", "chunks"], [[n, f"{c:,}"] for n, c in store.strategies()]
    )
    print("\nReady. Try:  python examples/17_baseline_rag.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
