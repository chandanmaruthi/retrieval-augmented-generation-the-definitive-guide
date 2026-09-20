#!/usr/bin/env python3
"""
03. Sentence-Boundary Chunking
Book: Chapter 6, Chunking Strategies -- section 3.2

WHAT THIS SHOWS: The smallest possible improvement on fixed-length chunking,
and one of the highest-value. Accumulate whole sentences until adding the next
one would exceed the token budget, then start a new chunk.

Chunks come out slightly uneven in size. In exchange, every chunk is made of
complete sentences, against 12% of fixed-length chunks ending cleanly.


HOW THIS SCRIPT PROCEEDS
    1. Load the book                      ragkit/corpus.py, shared by all examples
    2. Split each story into sentences    ragkit/corpus.py split_sentences()
    3. Pack sentences up to a budget      <-- THE TECHNIQUE, in chunk_sentences()
    4. Carry one sentence into the next   <-- the overlap rule
    5. Measure the cost: size variance    the price paid for intact sentences

Steps 1, 2 and 5 are scaffolding. Steps 3 and 4 are the strategy, and they are
about fifteen lines.


WHAT CHANGED SINCE EXAMPLE 02
    Example 02 cut every N tokens and did not care where the cut landed. This
    one asks the same question -- "is the chunk full yet?" -- but is only
    allowed to answer it at a sentence boundary. That single constraint is the
    whole difference, and it costs a sentence splitter and some wasted budget
    at the end of each chunk.


REQUIRES: nothing
RUN: python examples/03_chunk_sentence.py [--max-tokens 256]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, tokens
from ragkit.corpus import Book, Chunk, load_book, split_sentences

STRATEGY = "sentence"

DEFAULT_MAX_TOKENS = 256
# One sentence of overlap. Cheaper than the token-window overlap in example 02
# and better targeted: the sentence is the unit a fact tends to live in.
DEFAULT_OVERLAP_SENTENCES = 1


def chunk_sentences(
    book: Book,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[Chunk]:
    """Pack whole sentences up to a token budget.

    Two details that matter more than they look:

    1. A single sentence longer than the budget still has to become a chunk.
       Dropping it loses text; splitting it re-introduces exactly the problem
       we came here to avoid. We emit it oversized and count it, because a
       strategy that silently discards input is worse than one that admits
       it is over budget.

    2. Overlap is carried as whole sentences, so the repeated text is always
       readable on its own.
    """
    chunks: list[Chunk] = []
    oversized = 0

    for story in book.stories:
        sentences: list[str] = []
        for paragraph in story.paragraphs:
            sentences.extend(split_sentences(paragraph))

        buffer: list[str] = []
        budget = 0
        index = 0

        def emit(sentence_list: list[str], idx: int) -> Chunk:
            return Chunk(
                text=" ".join(sentence_list),
                strategy=STRATEGY,
                story=story.title,
                index=idx,
                meta={"max_tokens": max_tokens, "n_sentences": len(sentence_list)},
            )

        # =================================================================
        # THE TECHNIQUE. Everything above this point is setup; everything
        # below is bookkeeping. These fifteen lines are example 03.
        # =================================================================
        for sentence in sentences:
            cost = tokens.count_tokens(sentence)

            # Edge case first: a single sentence bigger than the whole budget.
            # It cannot be packed, and splitting it would re-create exactly
            # the problem this strategy exists to avoid, so it is emitted
            # oversized and counted.
            if cost > max_tokens and not buffer:
                chunks.append(emit([sentence], index))
                index += 1
                oversized += 1
                continue

            # ---- THE KEY LINE ------------------------------------------
            # Close the chunk BEFORE adding the sentence that would overflow
            # it, rather than after. That ordering is the entire difference
            # from example 02: the boundary is forced to land between two
            # sentences instead of wherever the token count ran out.
            #
            # Get this backwards -- append first, then check -- and every
            # chunk overshoots its budget by one sentence, which on a long
            # sentence means the tail is silently truncated by the embedder.
            if budget + cost > max_tokens and buffer:
                chunks.append(emit(buffer, index))
                index += 1

                # ---- THE OVERLAP RULE ----------------------------------
                # Start the next chunk holding the last sentence(s) of this
                # one. Cheaper and better targeted than example 02's token
                # window: a sentence is the unit a fact tends to live in, so
                # the repeated text is always readable on its own.
                buffer = buffer[-overlap_sentences:] if overlap_sentences else []
                budget = sum(tokens.count_tokens(s) for s in buffer)

            buffer.append(sentence)
            budget += cost
        # ================= end of the technique ==========================

        if buffer:
            chunks.append(emit(buffer, index))

    if oversized:
        print(f"  note: {oversized} sentence(s) exceeded the budget on their own")

    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-sentences", type=int, default=DEFAULT_OVERLAP_SENTENCES)
    args = parser.parse_args()

    display.banner(
        "03. Sentence-Boundary Chunking",
        "Chapter 6, Chunking Strategies -- section 3.2",
        f"Accumulate whole sentences up to {args.max_tokens} tokens, never "
        "cutting one in half.",
    )
    print(f"  {tokens.tokenizer_note()}\n")

    book = load_book()
    chunks = chunk_sentences(book, args.max_tokens, args.overlap_sentences)

    sizes = [tokens.count_tokens(c.text) for c in chunks]
    counts = [c.meta["n_sentences"] for c in chunks]
    display.kv("chunks produced", f"{len(chunks):,}")
    display.kv("mean tokens", f"{sum(sizes) / len(sizes):.0f}")
    display.kv("min / max tokens", f"{min(sizes)} / {max(sizes)}")
    display.kv("mean sentences per chunk", f"{sum(counts) / len(counts):.1f}")

    display.heading("Size variance is the cost")
    # Fixed-length chunking produces near-identical sizes; this does not. The
    # spread is what you trade for intact sentences.
    mean = sum(sizes) / len(sizes)
    spread = (sum((s - mean) ** 2 for s in sizes) / len(sizes)) ** 0.5
    display.kv("standard deviation", f"{spread:.0f} tokens")
    under_half = sum(1 for s in sizes if s < args.max_tokens / 2)
    display.kv("chunks under half budget", f"{under_half:,} ({100*under_half/len(chunks):.0f}%)")
    print("\n  Short chunks appear where a long sentence would not fit and the")
    print("  chunk had to close early. That wasted budget is the real cost here.")

    display.heading("Every chunk ends on a sentence boundary")
    print("  By construction: chunks are built by appending whole sentences, so")
    print("  this is 100% and cannot be otherwise. What is worth measuring is")
    print("  how those sentences end.")
    print()
    # Note the closing single quote in this set. Conan Doyle nests dialogue --
    # a character quoting someone else inside their own speech -- so sentences
    # legitimately end on U+2019. Leave it out and 19 perfectly good chunks
    # look like splitter failures.
    terminal = (".", "!", "?", '"', "”", "’", "'")
    full_stop = sum(1 for c in chunks if c.text.rstrip().endswith(terminal))
    other = len(chunks) - full_stop
    display.kv("ending in . ! ? or a closing quote", f"{full_stop:,} ({100*full_stop/len(chunks):.0f}%)")
    display.kv("ending on anything else", f"{other:,}")
    print("\n  The remainder end on a colon, where a sentence introduces a quoted")
    print("  passage that the splitter treated as new. That is a genuine limit of")
    print("  a rule-based splitter, and it is the honest number rather than a")
    print("  rounded-up one.")
    print("\n  A representative chunk:")
    display.preview(chunks[5].text, limit=300)

    display.sample_chunks(chunks, n=4, start=2, count_tokens=tokens.count_tokens)

    display.notice(
        "Every chunk is whole sentences, against 12% of fixed-length chunks "
        "ending cleanly in example 02. That is the entire change, and it costs "
        "nothing but a sentence splitter.",
        f"The price is variance: {spread:.0f} tokens of standard deviation, and "
        f"{100*under_half/len(chunks):.0f}% of chunks using less than half their "
        "budget. Under-filled chunks mean more vectors and more storage for the "
        "same corpus.",
        "The sentence splitter is now load-bearing. Conan Doyle writes 'Mr.', "
        "'Dr.' and 'St.' constantly, and a splitter that breaks on those would "
        "shred the chunks -- see the abbreviation guard in ragkit/corpus.py.",
        "This strategy still ignores paragraphs. A chunk here can begin "
        "mid-scene and end in the middle of a different conversation, which is "
        "what example 04 addresses.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
