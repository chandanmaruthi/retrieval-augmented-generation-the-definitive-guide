#!/usr/bin/env python3
"""
05. Heading-Boundary Chunking
Book: Chapter 6, Chunking Strategies -- section 3.4

WHAT THIS SHOWS: Chunk on the document's own outline. Never let a chunk cross
a heading, and keep the heading path as metadata so every chunk knows where it
came from.

This is the strategy that most rewards good extraction. It only works if
example 01 successfully recovered the structure -- headings the extractor
flattened into body text are boundaries this strategy cannot see.

HOW THIS SCRIPT PROCEEDS
    1. Load the book                      ragkit/corpus.py
    2. Walk story by story, section by section
    3. Pack paragraphs WITHIN a section   <-- THE TECHNIQUE: never cross a heading
    4. Record the heading path as metadata
    5. Verify the boundary held           checked, not assumed

Step 3 is a nested loop that simply never accumulates across a section. That
omission is the entire strategy.


WHAT CHANGED SINCE EXAMPLE 04
    Example 04 respected paragraph breaks but had no idea which section a
    paragraph belonged to, so a chunk could span the end of Part I and the
    start of Part II -- two scenes days apart, fused into one vector. This
    one makes the document outline a hard boundary.

    It is also the strategy most dependent on example 01. The headings it
    splits on were recovered by pattern-matching flattened PDF text, because
    font size and weight did not survive extraction. Lose them upstream and
    this boundary silently ceases to exist.


REQUIRES: nothing
RUN: python examples/05_chunk_heading.py [--max-tokens 384]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, tokens
from ragkit.corpus import Book, Chunk, load_book, split_sentences

STRATEGY = "heading"

DEFAULT_MAX_TOKENS = 384


def chunk_by_heading(book: Book, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    """Split within sections, never across them.

    Two headings exist in this corpus: the story title and, in a few stories,
    an inner numbered part. Both are hard boundaries.

    The heading path is stored in ``meta`` rather than glued onto the text.
    That separation matters: example 09 shows what happens when you *do* put
    the heading into the embedded string, and it is a different strategy with
    different results. Keeping them apart here means the two can be compared.
    """
    chunks: list[Chunk] = []

    for story in book.stories:
        for section in story.sections:
            # The heading path, most general first -- the same shape you would
            # build from nested <h1>/<h2>/<h3> tags or a DOCX outline level.
            path = [story.title]
            if section.number:
                path.append(f"Part {section.number}")
            heading_path = " > ".join(path)

            # =========================================================
            # THE TECHNIQUE. Note what is NOT here: any path by which
            # `buffer` survives from one section to the next. It is
            # declared inside the section loop, so crossing a heading is
            # not forbidden by a check -- it is structurally impossible.
            # =========================================================
            index = 0
            buffer: list[str] = []
            budget = 0

            def emit(body: list[str], idx: int) -> Chunk:
                return Chunk(
                    text="\n\n".join(body),
                    strategy=STRATEGY,
                    story=story.title,
                    section=section.number,
                    index=idx,
                    meta={
                        "heading_path": heading_path,
                        "depth": len(path),
                        "n_paragraphs": len(body),
                    },
                )

            for paragraph in section.paragraphs:
                cost = tokens.count_tokens(paragraph)

                # A single paragraph over budget still cannot escape its
                # section, so it is packed by sentence *within* the section.
                if cost > max_tokens:
                    if buffer:
                        chunks.append(emit(buffer, index))
                        index += 1
                        buffer, budget = [], 0
                    inner: list[str] = []
                    inner_budget = 0
                    for sentence in split_sentences(paragraph):
                        sentence_cost = tokens.count_tokens(sentence)
                        if inner_budget + sentence_cost > max_tokens and inner:
                            chunks.append(emit([" ".join(inner)], index))
                            index += 1
                            inner, inner_budget = [], 0
                        inner.append(sentence)
                        inner_budget += sentence_cost
                    if inner:
                        chunks.append(emit([" ".join(inner)], index))
                        index += 1
                    continue

                if budget + cost > max_tokens and buffer:
                    chunks.append(emit(buffer, index))
                    index += 1
                    buffer, budget = [], 0

                buffer.append(paragraph)
                budget += cost

            if buffer:
                chunks.append(emit(buffer, index))

    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    display.banner(
        "05. Heading-Boundary Chunking",
        "Chapter 6, Chunking Strategies -- section 3.4",
        "Respect the document outline. No chunk crosses a heading, and every "
        "chunk carries the heading path it came from.",
    )
    print(f"  {tokens.tokenizer_note()}\n")

    book = load_book()
    chunks = chunk_by_heading(book, args.max_tokens)

    sizes = [tokens.count_tokens(c.text) for c in chunks]
    display.kv("chunks produced", f"{len(chunks):,}")
    display.kv("mean tokens", f"{sum(sizes)/len(sizes):.0f}")
    display.kv("min / max tokens", f"{min(sizes)} / {max(sizes)}")

    display.heading("The outline this corpus actually has")
    sections = [(s.title, sec.number) for s in book.stories for sec in s.sections]
    nested = sum(1 for _, number in sections if number)
    display.kv("sections total", len(sections))
    display.kv("inside a numbered part", nested)
    display.kv("stories with inner parts", len({t for t, n in sections if n}))
    print("\n  Only two of the twelve stories are subdivided. That is a thin")
    print("  outline, and it caps what this strategy can do here -- on a")
    print("  technical manual with six heading levels it would be the strongest")
    print("  strategy in the book.")

    display.heading("Chunks carry their heading path")
    display.table(
        ["heading path", "chunk", "tokens"],
        [
            [c.meta["heading_path"], c.index, tokens.count_tokens(c.text)]
            for c in chunks[:8]
        ],
    )

    display.heading("The boundary guarantee, verified")
    # Assert rather than assume: every chunk's text must appear inside the
    # single section it claims to come from. If any chunk had accumulated
    # across a heading, its text would not be found in either section.
    by_section = {
        (story.title, section.number): " ".join(section.text.split())
        for story in book.stories
        for section in story.sections
    }
    contained = 0
    for chunk in chunks:
        body = " ".join(chunk.text.split())
        if body in by_section.get((chunk.story, chunk.section), ""):
            contained += 1
    display.kv(
        "chunks inside one section",
        f"{contained:,} of {len(chunks):,} ({100*contained/len(chunks):.0f}%)",
    )
    print("\n  Checked, not assumed: each chunk's text is searched for in the")
    print("  section it claims. A chunk that had accumulated across a heading")
    print("  would appear in neither and fail this test.")
    print("\n  The strategies in examples 02-04 make no such promise. They chunk")
    print("  within a story, but nothing stops a chunk spanning the boundary")
    print("  between Part I and Part II of 'A Scandal in Bohemia' -- two scenes")
    print("  separated by days, fused into one vector.")

    display.sample_chunks(chunks, n=4, start=1, count_tokens=tokens.count_tokens)

    display.notice(
        "This strategy is only as good as extraction. The headings it splits "
        "on were recovered by pattern-matching in example 01 -- font size and "
        "weight were lost when the PDF was parsed. Flatten a heading into body "
        "text upstream and this boundary silently disappears.",
        f"Only {len({t for t, n in sections if n})} of 12 stories have inner "
        "parts, so most chunks here sit directly under a story title. The "
        "shallower the outline, the less this strategy has to work with.",
        "The heading path is metadata, not text. Example 09 prepends it to the "
        "embedded string instead and measures what that changes -- these are "
        "genuinely two strategies, and the book lists them separately for that "
        "reason.",
        "Heading paths give citations something to say. 'The Red-Headed League "
        "> Part II' is a reference a reader can check; 'chunk 417' is not.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
