#!/usr/bin/env python3
"""
07. Sentence-Window Context Chunking
Book: Chapter 6, Chunking Strategies -- section 3.6

WHAT THIS SHOWS: Decouple what you *search* from what you *read*. Index one
sentence at a time, so the embedding is about exactly one thing. Then, when a
sentence is retrieved, expand it to the K sentences either side before handing
it to the model.

The insight is that these two jobs want opposite sizes. Small units retrieve
precisely; large units answer completely. Most strategies compromise on a
middle size that is mediocre at both. This one refuses to compromise.

HOW THIS SCRIPT PROCEEDS
    1. Flatten each section to sentences
    2. For each sentence, take a window of its neighbours
    3. Store the sentence as the search key     <-- THE TECHNIQUE, half 1
    4. Store the window as the answer text      <-- THE TECHNIQUE, half 2
    5. Compare the two sizes, and the storage cost

The whole strategy is steps 3 and 4 writing to two different fields of one
Chunk. There is no clever algorithm here at all.


WHAT CHANGED SINCE EXAMPLE 06
    Every strategy up to now produced ONE piece of text per chunk, used for
    both searching and answering. This is the first to decouple them, and the
    reason is that those two jobs want opposite sizes: small units retrieve
    precisely, large units answer completely.

    Watch for Chunk.retrieval_text and Chunk.parent_text below. That pair of
    fields is what examples 08, 09, 11 and 13 all exploit in different ways.


REQUIRES: nothing
RUN: python examples/07_chunk_sentence_window.py [--window 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, tokens
from ragkit.corpus import Book, Chunk, load_book, split_sentences

STRATEGY = "sentence_window"

DEFAULT_WINDOW = 3


def chunk_sentence_window(book: Book, window: int = DEFAULT_WINDOW) -> list[Chunk]:
    """One chunk per sentence, each carrying its neighbourhood as context.

    The ``Chunk`` type has two text fields for exactly this pattern:

    * ``retrieval_text`` -- the bare sentence. This is what gets embedded.
    * ``parent_text``    -- the window. This is what the generator receives.

    The window is computed once, at index time, rather than looked up at query
    time. Both work; precomputing trades storage for latency, and storage is
    usually the cheaper of the two.
    """
    chunks: list[Chunk] = []

    for story in book.stories:
        for section in story.sections:
            # Flatten the section to a sentence list first, so a window can
            # cross a paragraph boundary but never a section boundary.
            sentences: list[str] = []
            for paragraph in section.paragraphs:
                sentences.extend(split_sentences(paragraph))

            for position, sentence in enumerate(sentences):
                start = max(0, position - window)
                end = min(len(sentences), position + window + 1)
                context = " ".join(sentences[start:end])

                # ---- THE KEY LINES -------------------------------------
                # One chunk, two texts. retrieval_text is what gets
                # embedded; parent_text is what the generator reads. Every
                # other strategy so far set only one of these.
                chunks.append(
                    Chunk(
                        text=sentence,
                        strategy=STRATEGY,
                        story=story.title,
                        section=section.number,
                        index=position,
                        # Embed the sentence alone...
                        retrieval_text=sentence,
                        # ...but generate from the window around it.
                        parent_text=context,
                        meta={
                            "window": window,
                            "window_start": start,
                            "window_end": end,
                            "truncated": start == 0 or end == len(sentences),
                        },
                    )
                )

    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = parser.parse_args()

    display.banner(
        "07. Sentence-Window Context Chunking",
        "Chapter 6, Chunking Strategies -- section 3.6",
        f"Index single sentences for precision; expand to +/-{args.window} "
        "sentences for the answer.",
    )
    print(f"  {tokens.tokenizer_note()}\n")

    book = load_book()
    chunks = chunk_sentence_window(book, args.window)

    retrieval_sizes = [tokens.count_tokens(c.embed_text) for c in chunks]
    context_sizes = [tokens.count_tokens(c.context_text) for c in chunks]

    display.kv("chunks produced", f"{len(chunks):,}")
    display.table(
        ["", "mean tokens", "min", "max"],
        [
            [
                "what gets embedded",
                f"{sum(retrieval_sizes)/len(retrieval_sizes):.0f}",
                min(retrieval_sizes),
                max(retrieval_sizes),
            ],
            [
                "what the model reads",
                f"{sum(context_sizes)/len(context_sizes):.0f}",
                min(context_sizes),
                max(context_sizes),
            ],
        ],
    )
    ratio = (sum(context_sizes) / len(context_sizes)) / (
        sum(retrieval_sizes) / len(retrieval_sizes)
    )
    print(f"\n  The model sees about {ratio:.1f}x more text than was searched.")

    display.heading("One chunk, both halves")
    sample = next(c for c in chunks if len(c.text.split()) > 12 and not c.meta["truncated"])
    print("  EMBEDDED (the retrieval key) --")
    display.preview(sample.embed_text, limit=240)
    print("\n  SUPPLIED TO THE MODEL (the window) --")
    display.preview(sample.context_text, limit=420)
    print(f"\n  Same chunk. The vector is built from the first string only.")

    display.heading("The cost")
    display.kv("vectors to store", f"{len(chunks):,}")
    print(f"      vs {413:,} for heading chunking in example 05")
    duplication = sum(context_sizes) / sum(tokens.count_tokens(p) for s in book.stories for p in s.paragraphs)
    display.kv("context stored vs corpus size", f"{duplication:.1f}x")
    print(f"\n  Each sentence appears in {2*args.window + 1} windows, so the stored")
    print("  context is several times the size of the book. That is the trade:")
    print("  storage is cheap, a wrong retrieval is not.")

    display.sample_chunks(chunks, n=4, start=10, count_tokens=tokens.count_tokens)

    display.notice(
        "A sentence embedding is about one thing. A 400-token chunk embedding "
        "is an average of everything in it, and averages match everything "
        "weakly -- the 'topic dilution' the book names as a core failure mode.",
        f"Retrieval searches {sum(retrieval_sizes)/len(retrieval_sizes):.0f} "
        f"tokens on average but answers from "
        f"{sum(context_sizes)/len(context_sizes):.0f}. Neither job is "
        "compromised for the other.",
        f"The price is {len(chunks):,} vectors instead of 413, and "
        f"{duplication:.1f}x the corpus stored as overlapping context.",
        "This is the same retrieve-small / generate-large idea as parent-child "
        "chunking in example 08. The difference is where the large unit comes "
        "from: a fixed window of neighbours here, the document's own structure "
        "there.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
