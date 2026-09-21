#!/usr/bin/env python3
"""
04. Paragraph-Boundary Chunking
Book: Chapter 5, Chunking Strategies -- section 3.3
WHAT THIS SHOWS: Chunk on the boundaries the author already chose. A paragraph
is a unit of thought that someone deliberately delimited, which makes it a
better guess at "one idea" than any window we could compute.

The complication is size. Paragraphs in this corpus range from three words to
over four hundred, so the strategy needs a policy for both ends: merge the
runts, split the monsters.

HOW THIS SCRIPT PROCEEDS
    1. Load the book                      ragkit/corpus.py
    2. Measure the paragraph size spread  shows WHY a size policy is needed
    3. For each paragraph, pick a case:   <-- THE TECHNIQUE, three branches
         too small  -> merge forward
         too large  -> split by sentence
         just right -> emit as is
    4. Report how often each case fired

The three-way branch in step 3 is the strategy. Everything else is measurement.


WHAT CHANGED SINCE EXAMPLE 03
    Example 03 packed sentences to fill a budget, ignoring where the author
    put paragraph breaks. This one treats those breaks as the primary
    boundary -- a paragraph is a unit of thought somebody deliberately
    delimited, which is a better guess at "one idea" than any window.

    The complication is that real paragraphs are wildly uneven. Two thirds of
    this corpus's are under 48 tokens, so the merge branch does most of the
    work, and without it you would index thousands of vectors for the word
    "Indeed!".


REQUIRES: nothing
RUN: python examples/04_chunk_paragraph.py [--max-tokens 384] [--min-tokens 48]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, tokens
from ragkit.corpus import Book, Chunk, load_book, split_sentences

STRATEGY = "paragraph"

DEFAULT_MAX_TOKENS = 384
DEFAULT_MIN_TOKENS = 48


def chunk_paragraphs(
    book: Book,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
) -> list[Chunk]:
    """One chunk per paragraph, with merging and splitting at the extremes.

    The three cases:

    * **Too small** -- merge with the following paragraph(s) until the chunk
      clears ``min_tokens``. Dialogue-heavy fiction is full of two-word
      paragraphs ('"Indeed!"') and embedding those individually produces
      vectors that match everything and mean nothing.

    * **Too large** -- fall back to sentence packing within that paragraph.
      Note we do *not* fall back to fixed-token chunking: having come this far
      to respect boundaries, it would be odd to abandon them at the last step.

    * **Just right** -- emit as is.
    """
    chunks: list[Chunk] = []
    stats = {"merged": 0, "split": 0, "asis": 0}

    for story in book.stories:
        # Track which section each paragraph came from, so the chunk keeps its
        # provenance even though the strategy itself ignores sections.
        located: list[tuple[str | None, str]] = []
        for section in story.sections:
            for paragraph in section.paragraphs:
                located.append((section.number, paragraph))

        index = 0
        position = 0
        while position < len(located):
            section_number, paragraph = located[position]
            size = tokens.count_tokens(paragraph)

            # =========================================================
            # THE TECHNIQUE: three cases, and the decision of which one
            # applies is the whole strategy.
            # =========================================================

            # ---- CASE 1: too small -> merge forward -----------------
            # A two-word paragraph ('"Indeed!"') embedded alone produces a
            # vector that is weakly similar to everything and specific to
            # nothing -- the "topic dilution" failure in reverse.
            # --- too small: merge forward ------------- book:chunk-paragraph
            if size < min_tokens:
                merged = [paragraph]
                total = size
                position += 1
                while position < len(located) and total < min_tokens:
            # ------------------------------------------------------- /book
                    _, following = located[position]
                    following_size = tokens.count_tokens(following)
                    if total + following_size > max_tokens:
                        break
                    merged.append(following)
                    total += following_size
                    position += 1
                if len(merged) > 1:
                    stats["merged"] += 1
                else:
                    stats["asis"] += 1
                chunks.append(
                    Chunk(
                        text="\n\n".join(merged),
                        strategy=STRATEGY,
                        story=story.title,
                        section=section_number,
                        index=index,
                        meta={"n_paragraphs": len(merged), "case": "merged"},
                    )
                )
                index += 1
                continue

            # ---- CASE 2: too large -> split by sentence -------------
            # Note the fallback is example 03's sentence packing, NOT
            # example 02's fixed window. Having come this far to respect
            # boundaries, abandoning them at the last step would undo it.
            # --- ...too big: split by sentence -------- book:chunk-paragraph
            if size > max_tokens:
                buffer: list[str] = []
                budget = 0
                for sentence in split_sentences(paragraph):
            # ------------------------------------------------------- /book
                    cost = tokens.count_tokens(sentence)
                    if budget + cost > max_tokens and buffer:
                        chunks.append(
                            Chunk(
                                text=" ".join(buffer),
                                strategy=STRATEGY,
                                story=story.title,
                                section=section_number,
                                index=index,
                                meta={"n_paragraphs": 1, "case": "split"},
                            )
                        )
                        index += 1
                        buffer, budget = [], 0
                    buffer.append(sentence)
                    budget += cost
                if buffer:
                    chunks.append(
                        Chunk(
                            text=" ".join(buffer),
                            strategy=STRATEGY,
                            story=story.title,
                            section=section_number,
                            index=index,
                            meta={"n_paragraphs": 1, "case": "split"},
                        )
                    )
                    index += 1
                stats["split"] += 1
                position += 1
                continue

            # ---- CASE 3: just right -> emit unchanged ---------------
            # The only case where the author's boundary survives exactly.
            chunks.append(
                Chunk(
                    text=paragraph,
                    strategy=STRATEGY,
                    story=story.title,
                    section=section_number,
                    index=index,
                    meta={"n_paragraphs": 1, "case": "asis"},
                )
            )
            index += 1
            stats["asis"] += 1
            position += 1

    print(
        f"  paragraphs kept as-is: {stats['asis']:,}   "
        f"merged: {stats['merged']:,}   split: {stats['split']:,}\n"
    )
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    args = parser.parse_args()

    display.banner(
        "04. Paragraph-Boundary Chunking",
        "Chapter 5, Chunking Strategies -- section 3.3",
        "Use the author's own boundaries, then fix up the paragraphs that are "
        "too short or too long to stand alone.",
    )
    print(f"  {tokens.tokenizer_note()}\n")

    book = load_book()

    # First, the distribution that forces the merge/split policy to exist.
    display.heading("Why a size policy is unavoidable")
    raw = [tokens.count_tokens(p) for s in book.stories for p in s.paragraphs]
    raw.sort()
    display.kv("paragraphs in the book", f"{len(raw):,}")
    display.kv("shortest", f"{raw[0]} tokens")
    display.kv("median", f"{raw[len(raw)//2]} tokens")
    display.kv("95th percentile", f"{raw[int(len(raw)*0.95)]} tokens")
    display.kv("longest", f"{raw[-1]} tokens")
    tiny = sum(1 for r in raw if r < args.min_tokens)
    huge = sum(1 for r in raw if r > args.max_tokens)
    display.kv(f"under {args.min_tokens} tokens", f"{tiny:,} ({100*tiny/len(raw):.0f}%)")
    display.kv(f"over {args.max_tokens} tokens", f"{huge:,} ({100*huge/len(raw):.0f}%)")
    print(f"\n  Two thirds of this book's paragraphs are shorter than {args.min_tokens}")
    print("  tokens -- it is dialogue-heavy fiction, and a paragraph is often one")
    print("  spoken line. Embedded alone those are near-useless, which is what")
    print("  the merge rule is for.")

    print()
    chunks = chunk_paragraphs(book, args.max_tokens, args.min_tokens)

    sizes = [tokens.count_tokens(c.text) for c in chunks]
    display.kv("chunks produced", f"{len(chunks):,}")
    display.kv("mean tokens", f"{sum(sizes)/len(sizes):.0f}")
    display.kv("min / max tokens", f"{min(sizes)} / {max(sizes)}")

    display.heading("A merged chunk")
    for chunk in chunks:
        if chunk.meta["n_paragraphs"] > 2:
            print(f"  {chunk.meta['n_paragraphs']} short paragraphs became one chunk:")
            for line in chunk.text.split("\n\n"):
                print(f"      | {display.truncate(line, 70)}")
            break

    display.sample_chunks(chunks, n=4, start=4, count_tokens=tokens.count_tokens)

    display.notice(
        "Paragraph boundaries are free structure -- the author already decided "
        "where one idea stops. No model is needed to find them, unlike the "
        "semantic boundaries in example 06.",
        f"{100*tiny/len(raw):.0f}% of paragraphs fall below the minimum, so "
        "merging is the common path, not an edge case. Splitting fires only "
        f"{huge} times in the whole book. On a corpus of technical manuals "
        "those proportions invert, which is why both rules have to exist.",
        "Merging joins paragraphs that are adjacent but not necessarily "
        "related. A question and an unrelated answer can land in one vector. "
        "This is the specific failure that semantic chunking targets.",
        "Chunks now carry their section number. The strategy does not use it, "
        "but retrieval will -- see examples 09 and 24.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
