#!/usr/bin/env python3
"""
02. Fixed-Length Token Chunking
Book: Chapter 6, Chunking Strategies -- section 3.1

WHAT THIS SHOWS: The simplest strategy, and the baseline every other one is
measured against. Cut the text every N tokens, with M tokens of overlap so a
fact sitting on a boundary appears whole in at least one chunk.

It is fast, predictable, and completely blind to meaning -- it will cut a
sentence in half mid-clause, and it does so here, visibly.

HOW THIS SCRIPT PROCEEDS
    1. Load the book                      ragkit/corpus.py
    2. Tokenise each story                ragkit/tokens.py, real tokenizer if present
    3. Slide a fixed window across it     <-- THE TECHNIQUE, 10 lines
    4. Step forward by size minus overlap <-- the overlap rule
    5. Count how many chunks end cleanly  the number that motivates example 03

Note step 1 chunks per STORY, not across the whole book. Even the simplest
strategy respects the document boundary -- without that it is not a baseline,
it is confetti.


WHY START HERE
    This is the strategy everything else is measured against. It needs no
    model, no sentence splitter and no structure, so it always works and it is
    always fast. Example 15 shows what the more elaborate strategies actually
    buy over it, which on this corpus is less than you would expect.


REQUIRES: nothing
RUN: python examples/02_chunk_fixed_token.py [--size 256] [--overlap 32]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, tokens
from ragkit.corpus import Book, Chunk, load_book

STRATEGY = "fixed_token"

# The book specifies sizes in tokens "using the target model's tokenizer".
# 256/32 is a reasonable default for a 384-dim sentence embedder, whose
# effective input window is 256 word-pieces -- past that, MiniLM truncates and
# the tail of the chunk simply is not in the vector.
DEFAULT_SIZE = 256
DEFAULT_OVERLAP = 32


def chunk_fixed_token(
    book: Book, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Slide a fixed window across each story.

    We chunk per story rather than across the whole book, so that no chunk
    ever straddles two unrelated adventures. That single decision -- respect
    the document boundary even when the strategy says "fixed size" -- is the
    difference between this being usable and being a bag of confetti.

    The step is ``size - overlap``. Getting that wrong in the other direction
    (stepping by ``size`` and then *adding* overlap) is a common bug that
    quietly drops text between chunks.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size, or the window never advances")

    # `size` and `overlap` are budgets in *tokens*, but we slice the units that
    # tokens.encode() returns. Those are the same thing under a real tokenizer
    # and not the same thing under the fallback, so convert explicitly.
    window_units = tokens.units_for_tokens(size)
    overlap_units = tokens.units_for_tokens(overlap)
    step = window_units - overlap_units

    chunks: list[Chunk] = []

    # =================================================================
    # THE TECHNIQUE. A window, a step, and no opinion whatsoever about
    # what the text says. That indifference is the point: it is why
    # this is cheap and why it cuts sentences in half.
    # =================================================================

    for story in book.stories:
        token_ids = tokens.encode(story.text)
        index = 0
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + window_units]
            if not window:
                break
            # Skip a final sliver that is entirely contained in the previous
            # chunk's overlap -- it carries no new text.
            if start > 0 and len(window) <= overlap_units:
                break
            # ---- THE KEY LINE --------------------------------------
            # The boundary is wherever `start + window_units` happens to
            # land. No sentence, paragraph or heading is consulted, which
            # is the whole character of this strategy -- and the reason
            # only ~12% of these chunks end on a sentence boundary.
            chunks.append(
                Chunk(
                    text=tokens.decode(window),
                    strategy=STRATEGY,
                    story=story.title,
                    index=index,
                    meta={"size": size, "overlap": overlap, "token_start": start},
                )
            )
            index += 1

    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()

    display.banner(
        "02. Fixed-Length Token Chunking",
        "Chapter 6, Chunking Strategies -- section 3.1",
        f"Cut every {args.size} tokens with {args.overlap} tokens of overlap. "
        "No regard for sentences, paragraphs or meaning.",
    )
    print(f"  {tokens.tokenizer_note()}\n")

    book = load_book()
    chunks = chunk_fixed_token(book, args.size, args.overlap)

    sizes = [tokens.count_tokens(c.text) for c in chunks]
    display.kv("chunks produced", f"{len(chunks):,}")
    display.kv("mean tokens", f"{sum(sizes) / len(sizes):.0f}")
    display.kv("min / max tokens", f"{min(sizes)} / {max(sizes)}")

    display.heading("Two consecutive chunks, showing the overlap")
    first, second = chunks[3], chunks[4]
    print(f"\n  chunk {first.index} ends:")
    print(f"      ...{display.truncate(first.text[-260:], 200)}")
    print(f"\n  chunk {second.index} begins:")
    print(f"      {display.truncate(second.text[:260], 200)}...")
    print(
        f"\n  The repeated text is the {args.overlap}-token overlap. Without it, a "
        f"fact\n  straddling that boundary would be split across two chunks and "
        "complete\n  in neither."
    )

    display.heading("Where the boundaries actually fall")
    # A chunk that neither starts nor ends on a sentence boundary is the
    # characteristic failure of this strategy, so let us count them rather
    # than assert it.
    clean_end = sum(1 for c in chunks if c.text.rstrip().endswith((".", "!", "?", '"', "”")))
    print(f"  chunks ending on a sentence boundary: {clean_end:,} of {len(chunks):,} "
          f"({100 * clean_end / len(chunks):.0f}%)")
    print("\n  A chunk cut mid-sentence:")
    for chunk in chunks:
        stripped = chunk.text.rstrip()
        if not stripped.endswith((".", "!", "?", '"', "”")):
            print(f"      ...{display.truncate(chunk.text[-150:], 140)}")
            break

    display.sample_chunks(chunks, n=4, start=3, count_tokens=tokens.count_tokens)

    display.notice(
        f"{len(chunks):,} chunks from {book.word_count:,} words, produced in "
        "well under a second with no model and no API call. When the corpus is "
        "millions of documents, that cost difference is the whole argument for "
        "this strategy.",
        f"Only {100 * clean_end / len(chunks):.0f}% of chunks end on a sentence "
        "boundary. The rest end mid-clause, and the embedding of a half-sentence "
        "is a vector for something nobody said.",
        "Overlap is the mitigation, not a fix: it costs storage and duplicates "
        "retrieval hits, and it only helps when the split fact happens to fit "
        "inside the overlap window.",
        "Everything from example 03 onward is an attempt to buy better "
        "boundaries. Example 15 measures whether any of them are worth the "
        "extra cost on this corpus.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
