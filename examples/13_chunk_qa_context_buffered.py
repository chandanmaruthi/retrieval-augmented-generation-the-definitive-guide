#!/usr/bin/env python3
"""
13. Question-Anchored, Context-Buffered Chunking
Book: Chapter 6, Chunking Strategies -- section 3.12

WHAT THIS SHOWS: The book's canonical chunk. Everything from sections 3.8 to
3.11 assembled into one unit:

    heading path
    + summary of what came before
    + the passage itself (bounded by an anchor question)
    + summary of what comes after
    + the questions it answers

Each part is already justified on its own. The point of this example is the
assembly -- what order they go in, which parts are embedded, which parts the
model reads, and what the whole thing costs.

HOW THIS SCRIPT PROCEEDS
    1. Run example 12  -> anchor questions place the boundaries
    2. Add example 09  -> the heading path
    3. Run example 11  -> the questions each chunk answers
    4. Run example 10  -> summaries of the neighbours
    5. Assemble all five parts in order   <-- THE TECHNIQUE is the ASSEMBLY
    6. Report the inflation and the call count

Note this file implements no new technique. It imports four earlier examples
and decides what order their outputs go in. That ordering is the content.


WHAT CHANGED SINCE EXAMPLE 12
    Nothing, and everything. Each of the four parts is already justified on
    its own; what is new is that they compose -- and they compose only
    because each writes a DIFFERENT field of Chunk. Boundaries come from the
    anchors, the embedded string gains context, and the stored text is never
    touched by any of them.

    This is also the most expensive strategy in the book by a wide margin:
    two calls per chunk plus one per section, paid again on every reindex.


REQUIRES: OPENAI_API_KEY (falls back to showing the assembly)
RUN: python examples/13_chunk_qa_context_buffered.py [--limit 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

from ragkit import config, display, llm, tokens
from ragkit.corpus import Book, Chunk, load_book

STRATEGY = "qa_context_buffered"


def assemble(chunk: Chunk) -> str:
    """Build the canonical retrieval string from a chunk's parts.

    Order is deliberate:

    * The heading path leads, because it is the most general context and
      embedding models weight early tokens slightly more heavily.
    * Summaries are bracketed and labelled, so they are visibly not the
      passage. A model reading this string should never confuse borrowed
      context for source text.
    * Questions go last. They are retrieval keys, and putting them first would
      let four short questions dominate the vector for a 300-token passage.
    """
    # =================================================================
    # THE TECHNIQUE: order. Every part below is already justified by an
    # earlier example; the only decision this file makes is sequence.
    #
    #   heading first   -- most general context, and embedding models
    #                      weight early tokens slightly more heavily
    #   summaries next  -- bracketed and labelled, so they are visibly
    #                      not the passage
    #   passage         -- the only part stored as chunk.text
    #   questions last  -- retrieval keys; first, four short questions
    #                      would dominate the vector for a 300-token body
    # =================================================================
    parts: list[str] = []

    header = chunk.meta.get("header")
    if header:
        parts.append(f"From '{header}'.")

    if chunk.meta.get("pre_summary"):
        parts.append(f"[Previously: {chunk.meta['pre_summary']}]")

    parts.append(chunk.text)

    if chunk.meta.get("post_summary"):
        parts.append(f"[Next: {chunk.meta['post_summary']}]")

    questions = chunk.meta.get("questions") or []
    if questions:
        parts.append("Answers: " + " ".join(questions))

    return "\n\n".join(parts)


def chunk_canonical(book: Book, limit: int = 3) -> list[Chunk]:
    """Compose the four strategies into one chunk type.

    Built by running the earlier examples rather than reimplementing them, so
    there is exactly one copy of each technique in this repository and this
    file is only responsible for the assembly.
    """
    anchored = import_module("12_chunk_question_anchored")
    derived = import_module("11_chunk_question_derived")
    buffered = import_module("10_chunk_context_buffered")

    # 3.11 -- anchor questions place the boundaries.
    chunks = anchored.chunk_question_anchored(book, limit=limit)

    # 3.8 -- the heading path.
    for chunk in chunks:
        path = chunk.story + (f" > Part {chunk.section}" if chunk.section else "")
        chunk.meta["header"] = path

    # 3.10 -- the questions this chunk answers.
    for position, chunk in enumerate(chunks):
        print(f"    questions {position + 1}/{len(chunks)}", end="\r", flush=True)
        chunk.meta["questions"] = derived.generate_questions(chunk.text)

    # 3.9 -- summaries of the neighbours, within the same section.
    summaries = []
    for position, chunk in enumerate(chunks):
        print(f"    summaries {position + 1}/{len(chunks)}", end="\r", flush=True)
        summaries.append(buffered.summarise(chunk.text))

    for position, chunk in enumerate(chunks):
        same = lambda other: other.story == chunk.story and other.section == chunk.section
        chunk.meta["pre_summary"] = (
            summaries[position - 1] if position > 0 and same(chunks[position - 1]) else ""
        )
        chunk.meta["post_summary"] = (
            summaries[position + 1]
            if position + 1 < len(chunks) and same(chunks[position + 1])
            else ""
        )
        chunk.strategy = STRATEGY
        chunk.retrieval_text = assemble(chunk)

    print(" " * 40, end="\r")
    print()
    return chunks


def show_assembly() -> None:
    """Diagram the canonical chunk without needing a model."""
    display.para(
        "Without a key this example cannot generate the summaries or "
        "questions. The assembly is the lesson here, though, and it is "
        "structural rather than generated:"
    )
    print()
    for label, role, source in [
        ("From 'The Speckled Band'.", "embedded", "example 09, heading path"),
        ("[Previously: ...]", "embedded", "example 10, neighbour summary"),
        ("The passage itself.", "embedded + read", "example 12, anchor boundaries"),
        ("[Next: ...]", "embedded", "example 10, neighbour summary"),
        ("Answers: Q1 Q2 Q3", "embedded", "example 11, derived questions"),
    ]:
        print(f"      {label:<28}  {role:<16}  {source}")
    print()
    display.para(
        "Only the middle line is stored as the chunk's text. Everything else "
        "exists to make the vector findable and is discarded before the "
        "passage reaches the model -- which is why the summaries and questions "
        "can never be quoted back as though they were Conan Doyle."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="sections to process")
    args = parser.parse_args()

    display.banner(
        "13. Question-Anchored, Context-Buffered Chunking",
        "Chapter 6, Chunking Strategies -- section 3.12",
        "The book's canonical chunk: heading, pre-summary, passage, "
        "post-summary and questions, assembled into one retrieval unit.",
    )

    book = load_book()

    if not llm.available():
        print(f"  No LLM configured: {llm.why_unavailable()}\n")
        show_assembly()
        display.notice(
            "This chunk costs two model calls per chunk plus one per section "
            "to build. It is the most expensive strategy in the book by a wide "
            "margin.",
            "Four techniques compose because each one modifies a different "
            "field. Boundaries come from the anchors, the embedded string "
            "gains context, and the stored text is never touched.",
            "The embedded string can easily end up twice the length of the "
            "passage. Everything added competes with the passage for influence "
            "over one fixed-size vector -- more context is not monotonically "
            "better.",
            "Whether any of this beats example 04's paragraph chunking on a "
            "given corpus is an empirical question. Example 15 answers it for "
            "this one.",
        )
        return 0

    if llm.STUB:
        display.stub_warning()
    else:
        print(f"  model: {config.OPENAI_CHAT_MODEL}")
        print(f"  cached responses on disk: {llm.cached_count():,}")
    print()

    chunks = chunk_canonical(book, args.limit)

    display.kv("chunks produced", f"{len(chunks):,}")
    passage = sum(tokens.count_tokens(c.text) for c in chunks) / len(chunks)
    embedded = sum(tokens.count_tokens(c.embed_text) for c in chunks) / len(chunks)
    display.kv("mean passage tokens", f"{passage:.0f}")
    display.kv("mean embedded tokens", f"{embedded:.0f}")
    display.kv("inflation", f"{embedded / passage:.2f}x")

    display.heading("One canonical chunk, part by part")
    sample = next(
        (c for c in chunks if c.meta.get("pre_summary") and c.meta.get("questions")),
        chunks[0],
    )
    for part in assemble(sample).split("\n\n"):
        marker = "  " if part == sample.text else "+ "
        display.preview(f"{marker}{display.truncate(part, 180)}", limit=200, indent="      ")
    print("\n  Lines marked '+' are additions. The unmarked line is the only")
    print("  part stored as the chunk's text.")

    display.heading("What it cost")
    display.kv("model calls", f"~{args.limit + 2 * len(chunks):,}")
    print(f"      {args.limit} for anchors, {len(chunks)} for questions, {len(chunks)} for summaries")

    display.sample_chunks(chunks, n=4, start=0, count_tokens=tokens.count_tokens)

    display.notice(
        f"The embedded string is {embedded / passage:.2f}x the passage it "
        "represents. Every added token competes for influence over one "
        "fixed-size vector.",
        "Four strategies compose cleanly because each writes a different "
        "field. That is a property of how Chunk is designed, not a property "
        "of the techniques.",
        "Cost scales with chunks, not corpus size, and it is paid again on "
        "every reindex -- including the one you do after changing your "
        "embedding model.",
        "Example 14 splits this apart again into two indexes, which recovers "
        "most of the benefit at a fraction of the query-time cost.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
