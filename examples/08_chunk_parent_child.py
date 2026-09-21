#!/usr/bin/env python3
"""
08. Parent-Child Hierarchical Chunking
Book: Chapter 5, Chunking Strategies -- section 3.7
WHAT THIS SHOWS: Build two layers. Small child chunks (300-500 tokens) are
what you search; large parent chunks (1,000-1,500 tokens) are what you read.
A hit on any child returns its whole parent.

Same retrieve-small/read-large principle as the sentence windows in example
07, but the large unit is a real passage rather than an arbitrary span of
neighbours -- so it starts and ends somewhere sensible.

HOW THIS SCRIPT PROCEEDS
    1. Pack paragraphs into PARENTS first   <-- order matters, see below
    2. Subdivide each parent into CHILDREN  <-- THE TECHNIQUE
    3. Index only the children
    4. Attach the parent as the answer text
    5. Show the fan-out and the storage saved

Step 1 before step 2 is a deliberate choice, not an implementation detail.
Building children first and grouping them afterwards lets parent boundaries
fall wherever the child packing happened to land -- mid-scene.


WHAT CHANGED SINCE EXAMPLE 07
    Same retrieve-small / answer-large idea, better large unit. Example 07's
    window is however many sentences happened to sit nearby, so it can start
    and end anywhere. A parent here is a real passage with real boundaries.

    The cost is a new obligation: two children of one parent can both rank in
    the top-k and both expand to identical text, so retrieval MUST deduplicate
    after expansion or the model reads the same passage twice.


REQUIRES: nothing
RUN: python examples/08_chunk_parent_child.py [--child 400] [--parent 1200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, tokens
from ragkit.corpus import Book, Chunk, load_book, split_sentences

STRATEGY = "parent_child"

# The book's numbers: children 300-500 tokens, parents 1,000-1,500.
DEFAULT_CHILD_TOKENS = 400
DEFAULT_PARENT_TOKENS = 1200


def chunk_parent_child(
    book: Book,
    child_tokens: int = DEFAULT_CHILD_TOKENS,
    parent_tokens: int = DEFAULT_PARENT_TOKENS,
) -> tuple[list[Chunk], list[str]]:
    """Build parents first, then subdivide each into children.

    Direction matters. Building children first and grouping them afterwards
    lets a parent's boundaries fall wherever the child packing happened to
    land. Building parents first means both layers respect section boundaries,
    and every child has exactly one parent by construction.

    Only the children are returned as retrievable chunks -- the parents ride
    along in ``parent_text``. Indexing both layers would mean the same
    sentence competes with itself for a slot in the top-k.
    """
    chunks: list[Chunk] = []
    parents: list[str] = []

    for story in book.stories:
        for section in story.sections:
            # =========================================================
            # THE TECHNIQUE: build the hierarchy top-down, in two layers.
            # =========================================================

            # ---- LAYER 1: parents ------------------------------------
            parent_bodies: list[list[str]] = []
            buffer: list[str] = []
            budget = 0

            for paragraph in section.paragraphs:
                cost = tokens.count_tokens(paragraph)
                if budget + cost > parent_tokens and buffer:
                    parent_bodies.append(buffer)
                    buffer, budget = [], 0
                buffer.append(paragraph)
                budget += cost
            if buffer:
                parent_bodies.append(buffer)

            # ---- LAYER 2: children, WITHIN each parent ---------------
            # The inner loop runs over one parent's sentences only, so a
            # child cannot straddle two parents. Same structural trick as
            # example 05: the boundary is guaranteed by the loop nesting,
            # not by a check that could be forgotten.
            # --- children are built inside a parent ---- book:chunk-parent-child
            for parent_index, body in enumerate(parent_bodies):
                parent_text = "\n\n".join(body)
            # ------------------------------------------------------- /book
                parent_id = len(parents)
                parents.append(parent_text)

                # Children are packed by sentence within the parent, so a
                # child never crosses a parent boundary.
                sentences: list[str] = []
                for paragraph in body:
                    sentences.extend(split_sentences(paragraph))

                child_buffer: list[str] = []
                child_budget = 0
                child_index = 0

                def emit(sentence_list: list[str], idx: int) -> Chunk:
                    return Chunk(
                        text=" ".join(sentence_list),
                        strategy=STRATEGY,
                        story=story.title,
                        section=section.number,
                        index=idx,
                        # --- ...continued ---------- book:chunk-parent-child
                        # Searched on the child...
                        retrieval_text=" ".join(sentence_list),
                        # ...but answered from the parent.
                        parent_text=parent_text,
                        # --------------------------------------------- /book
                        meta={
                            "parent_id": parent_id,
                            "parent_index": parent_index,
                            "parent_tokens": tokens.count_tokens(parent_text),
                        },
                    )

                for sentence in sentences:
                    cost = tokens.count_tokens(sentence)
                    if child_budget + cost > child_tokens and child_buffer:
                        chunks.append(emit(child_buffer, child_index))
                        child_index += 1
                        child_buffer, child_budget = [], 0
                    child_buffer.append(sentence)
                    child_budget += cost

                if child_buffer:
                    chunks.append(emit(child_buffer, child_index))

    return chunks, parents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=int, default=DEFAULT_CHILD_TOKENS)
    parser.add_argument("--parent", type=int, default=DEFAULT_PARENT_TOKENS)
    args = parser.parse_args()

    display.banner(
        "08. Parent-Child Hierarchical Chunking",
        "Chapter 5, Chunking Strategies -- section 3.7",
        f"Search {args.child}-token children; answer from {args.parent}-token "
        "parents. Two layers, one index.",
    )
    print(f"  {tokens.tokenizer_note()}\n")

    book = load_book()
    children, parents = chunk_parent_child(book, args.child, args.parent)

    child_sizes = [tokens.count_tokens(c.text) for c in children]
    parent_sizes = [tokens.count_tokens(p) for p in parents]

    display.table(
        ["layer", "count", "mean tokens", "min", "max"],
        [
            [
                "children (indexed)",
                f"{len(children):,}",
                f"{sum(child_sizes)/len(child_sizes):.0f}",
                min(child_sizes),
                max(child_sizes),
            ],
            [
                "parents (returned)",
                f"{len(parents):,}",
                f"{sum(parent_sizes)/len(parent_sizes):.0f}",
                min(parent_sizes),
                max(parent_sizes),
            ],
        ],
    )
    fanout = len(children) / len(parents)
    print(f"\n  {fanout:.1f} children per parent on average.")

    display.heading("One child and the parent it returns")
    # Pick a child that is NOT the first in its parent, so the expansion is
    # visible on both sides rather than only at the end.
    sample = next(
        c
        for c in children
        if c.index > 0 and tokens.count_tokens(c.parent_text) > args.parent * 0.7
    )
    print(f"  CHILD ({tokens.count_tokens(sample.text)} tokens) -- this is what gets embedded:")
    display.preview(sample.text, limit=280)
    print(
        f"\n  PARENT ({tokens.count_tokens(sample.parent_text)} tokens) -- this is "
        "what the model reads."
    )
    print("  The child sits in the middle of it; note the text recovered on each side:")
    position = sample.parent_text.find(sample.text[:60])
    print(f"      {display.truncate(sample.parent_text[:200], 190)}")
    print(f"      [...{position} characters of preceding context the child did not have...]")
    print(f"      {display.truncate(sample.parent_text[-200:], 190)}")

    display.heading("Why only children are indexed")
    print(f"  Indexing both layers would put {len(children) + len(parents):,} vectors")
    print("  in the store, and every parent would compete with its own children")
    print("  for a place in the top-k. A query matching one passage would fill")
    print("  the results with overlapping copies of that passage and crowd out")
    print("  the second-best answer entirely.")

    display.heading("Deduplication is now mandatory")
    # Two children of the same parent can both rank highly, and each expands
    # to the identical parent text. Without a dedupe step the model receives
    # the same passage twice and the context budget is spent on a copy.
    print("  Two children of one parent can both land in the top-k, and both")
    print("  expand to the same text. Retrieval code must deduplicate on")
    print("  parent_id after expansion, or the model reads one passage twice")
    print("  and the context budget pays for it. Example 17 does this.")

    display.sample_chunks(children, n=4, start=1, count_tokens=tokens.count_tokens)

    display.notice(
        f"Search happens over {sum(child_sizes)/len(child_sizes):.0f}-token units "
        f"and answers come from {sum(parent_sizes)/len(parent_sizes):.0f}-token "
        "units. Precision and completeness stop competing for one size setting.",
        "Parents are built before children, so both layers respect section "
        "boundaries. Build children first and group them afterwards, and "
        "parents end up starting mid-scene.",
        f"Only {len(children):,} vectors are stored, against {len(children) + len(parents):,} "
        "if both layers were indexed. The parents are storage, not search space.",
        "This is the same idea as example 07's sentence windows, with a better "
        "large unit: a parent has real boundaries, while a window is however "
        "many sentences happened to sit nearby.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
