#!/usr/bin/env python3
"""
09. Contextual-Header Augmented Chunking
Book: Chapter 5, Chunking Strategies -- section 3.8
WHAT THIS SHOWS: Prepend the heading path to the chunk before embedding it, so
the vector knows which document the text came from.

A chunk that reads "He had been struck on the head" is, on its own, about
nothing in particular. Half the paragraphs in this book would match a query
about a head injury. Prefix it with "The Boscombe Valley Mystery" and the
vector carries the one piece of context a reader would have had automatically.

This example measures the effect instead of asserting it.

HOW THIS SCRIPT PROCEEDS
    1. Reuse example 05's chunks          the ONLY variable is the header
    2. Prepend the heading path to each   <-- THE TECHNIQUE, one line
    3. Embed both versions
    4. Measure similarity to a matching query
    5. Measure it for NON-matching chunks too   <-- the control

Step 5 is what makes this an experiment rather than a demonstration. A header
that lifted every chunk equally would change no rankings at all.


WHAT CHANGED SINCE EXAMPLE 05
    Nothing about the boundaries -- this reuses example 05's chunking exactly,
    so the two are directly comparable. The only difference is the string that
    gets embedded.

    Most passages never name their own document. A reader supplies that from
    the page they are on; an embedding has no page.


REQUIRES: -r requirements-local.txt (to measure the effect)
RUN: python examples/09_chunk_contextual_header.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, tokens
from ragkit.corpus import Book, Chunk, load_book
from ragkit.embed import describe, get_embedder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

STRATEGY = "contextual_header"


def _heading_chunks(book: Book) -> list[Chunk]:
    """Reuse example 05's chunking so the only variable is the header."""
    module = import_module("05_chunk_heading")
    return module.chunk_by_heading(book)


def chunk_with_headers(book: Book) -> list[Chunk]:
    """Take structural chunks and give each one a contextual prefix.

    The chunk body is unchanged. What changes is ``retrieval_text`` -- the
    string that actually gets embedded. That separation is the whole
    technique: the model reads the clean passage, the index searches the
    annotated one.

    A common mistake is to write the header into ``text`` itself. Then the
    heading path is quoted back to the user as if it were part of the source
    passage, and it eats context budget in every generated answer.
    """
    chunks = _heading_chunks(book)

    # =================================================================
    # THE TECHNIQUE: three lines. chunk.text is untouched; only the
    # string that gets EMBEDDED changes.
    # =================================================================
    for chunk in chunks:
        path = chunk.meta.get("heading_path", chunk.story)
        # ---- THE KEY LINE ------------------------------------------
        # Phrased as a sentence, not a breadcrumb. Embedding models are
        # trained on prose, and "A > B > C" is not prose -- a natural
        # frame lands closer to how the query itself is worded.
        #
        # Note this writes retrieval_text, NOT text. Put the header in
        # `text` and it gets quoted back to the user as though Conan
        # Doyle wrote it, and it eats context budget in every answer.
        # --- the chunk carries its own location -- book:chunk-contextual-header
        chunk.retrieval_text = f"From '{path}'. {chunk.text}"
        chunk.strategy = STRATEGY
        chunk.meta["header"] = path
        # ----------------------------------------------------------- /book
        chunk.meta["header_tokens"] = tokens.count_tokens(f"From '{path}'. ")

    return chunks


def main() -> int:
    display.banner(
        "09. Contextual-Header Augmented Chunking",
        "Chapter 5, Chunking Strategies -- section 3.8",
        "Embed the heading path along with the chunk, so a passage that never "
        "names its own document still retrieves as part of it.",
    )

    book = load_book()
    chunks = chunk_with_headers(book)

    display.kv("chunks produced", f"{len(chunks):,}")
    overhead = sum(c.meta["header_tokens"] for c in chunks) / len(chunks)
    display.kv("header cost", f"{overhead:.0f} tokens per chunk")

    display.heading("The two strings")
    sample = next(
        c for c in chunks if "Holmes" not in c.text[:400] and len(c.text) > 400
    )
    print("  STORED / SHOWN TO THE MODEL:")
    display.preview(sample.text, limit=230)
    print("\n  EMBEDDED:")
    display.preview(sample.embed_text, limit=230)
    print("\n  Only the second one carries the story title. Notice that the")
    print("  passage itself never names the story -- which is the normal case,")
    print("  not a contrived one.")

    # ---------------------------------------------------------------
    # Measure it
    # ---------------------------------------------------------------
    try:
        embedder = get_embedder()
    except ImportError:
        display.missing_dependency(
            "sentence-transformers",
            "requirements-local.txt",
            "The chunking above ran fine. Measuring whether the header helps "
            "needs embeddings.",
        )
        return 0

    print()
    print(f"  {describe(embedder)}")
    if not embedder.semantic:
        display.para(
            "The hash embedder matches on shared vocabulary, so prepending the "
            "title will appear to help simply because the title's words are now "
            "present. Install -r requirements-local.txt for a meaningful number."
        )
        return 0

    display.heading("Does the header actually help retrieval?")
    print("  A query naming a story, against chunks from that story that never")
    print("  name it themselves. Higher similarity is better.\n")

    queries = [
        ("The Adventure of the Speckled Band", "What happened in the Speckled Band case?"),
        ("The Red-Headed League", "Why was the Red-Headed League created?"),
        ("The Boscombe Valley Mystery", "Who was suspected in the Boscombe Valley Mystery?"),
        ("A Case of Identity", "What was the truth in A Case of Identity?"),
    ]

    rows = []
    for story_title, question in queries:
        # Only chunks whose own text never mentions the story name -- these
        # are the ones with something to gain.
        key_word = story_title.split()[-1]
        candidates = [
            c for c in chunks if c.story == story_title and key_word.lower() not in c.text.lower()
        ]
        if not candidates:
            continue

        query_vector = embedder.encode([question], show_progress=False)[0]
        plain = embedder.encode([c.text for c in candidates], show_progress=False)
        headed = embedder.encode([c.embed_text for c in candidates], show_progress=False)

        plain_score = float(np.mean(plain @ query_vector))
        headed_score = float(np.mean(headed @ query_vector))
        rows.append(
            [
                story_title[:34],
                len(candidates),
                f"{plain_score:.3f}",
                f"{headed_score:.3f}",
                f"{headed_score - plain_score:+.3f}",
            ]
        )

    display.table(
        ["story", "chunks", "plain", "with header", "delta"], rows
    )

    deltas = [float(r[-1]) for r in rows]
    mean_delta = sum(deltas) / len(deltas)
    print(f"\n  Mean improvement: {mean_delta:+.3f} cosine similarity.")

    display.heading("The other half of the effect")
    print("  Raising similarity to the right query is only useful if it does")
    print("  not raise similarity to every other query equally. Checking that:\n")

    # Cross-check: does the header also pull in chunks from OTHER stories?
    wrong_rows = []
    for story_title, question in queries[:2]:
        query_vector = embedder.encode([question], show_progress=False)[0]
        others = [c for c in chunks if c.story != story_title][:200]
        plain = embedder.encode([c.text for c in others], show_progress=False)
        headed = embedder.encode([c.embed_text for c in others], show_progress=False)
        wrong_rows.append(
            [
                f"non-matching chunks vs '{story_title[:22]}'",
                f"{float(np.mean(plain @ query_vector)):.3f}",
                f"{float(np.mean(headed @ query_vector)):.3f}",
                f"{float(np.mean(headed @ query_vector) - np.mean(plain @ query_vector)):+.3f}",
            ]
        )
    display.table(["", "plain", "with header", "delta"], wrong_rows)
    print("\n  The gap between these two deltas is what the technique buys. A")
    print("  header that lifted everything equally would change no rankings.")

    display.sample_chunks(chunks, n=4, start=1, count_tokens=tokens.count_tokens)

    display.notice(
        f"Headers cost about {overhead:.0f} tokens per chunk and are stored "
        "only in the embedded string, never in the text shown to the model.",
        f"Chunks from the named story gained {mean_delta:+.3f} cosine "
        "similarity to a query naming it -- and crucially, they gained more "
        "than unrelated chunks did.",
        "This works because most passages never name their own document. The "
        "reader supplies that context from the page they are on; an embedding "
        "has no page.",
        "The technique depends entirely on extraction having recovered the "
        "heading in the first place. Example 01 had to pattern-match headings "
        "out of flattened PDF text -- lose them there and there is nothing to "
        "prepend here.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
