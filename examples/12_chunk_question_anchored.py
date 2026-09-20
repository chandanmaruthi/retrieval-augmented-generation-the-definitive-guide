#!/usr/bin/env python3
"""
12. Question-Anchored Chunking
Book: Chapter 6, Chunking Strategies -- section 3.11

WHAT THIS SHOWS: Questions decide where the boundaries go.

Example 11 generated questions *from* chunks that already existed. This
inverts it: ask what questions a section answers first, then cut the section
so that each chunk fully answers at least one of them.

The test a chunk has to pass is self-containment -- can this chunk, alone,
answer its question? If not, the boundary moves. That is a different and
stronger requirement than "is this chunk about one topic", which is all
semantic chunking in example 06 can check.

HOW THIS SCRIPT PROCEEDS
    1. Flatten a section to a numbered sentence list
    2. Ask the model which questions it answers, and over which sentences
    3. Repair the ranges                  <-- coverage is VERIFIED, not trusted
    4. Cut the section at those ranges    <-- THE TECHNIQUE
    5. Report how many chunks came from a real anchor vs a gap-filler

Step 3 is the one to read closely. A model that skips sentences 11-14 would
silently drop them from the index, and nothing downstream would ever notice.


WHAT CHANGED SINCE EXAMPLE 11
    Example 11 generated questions FROM chunks that already existed. This
    inverts the order: the questions come first, and they decide where the
    boundaries go.

    That makes the test for a good boundary much stronger. Semantic chunking
    (example 06) can only ask "is this chunk about one topic". This asks "can
    this chunk, alone, answer its question" -- which is what retrieval
    actually needs.


REQUIRES: OPENAI_API_KEY (falls back to showing the prompt)
RUN: python examples/12_chunk_question_anchored.py [--limit 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragkit import config, display, llm, tokens
from ragkit.corpus import Book, Chunk, load_book, split_sentences

STRATEGY = "question_anchored"

# The book's range: 3-8 core questions per section.
MIN_QUESTIONS, MAX_QUESTIONS = 3, 8

ANCHOR_SYSTEM = (
    "You identify the core questions a passage of fiction answers. "
    "You reply with JSON."
)

ANCHOR_PROMPT = """Below is a section of a Sherlock Holmes story, given as a numbered
list of sentences.

Identify between {lo} and {hi} core questions this section answers. For each
one, give the range of sentence numbers that together answer it completely --
a reader seeing only those sentences should not need anything else.

Ranges may not overlap, and together they should cover the section.

Reply as JSON:
{{"anchors": [{{"question": "...", "start": 1, "end": 12}}, ...]}}

SENTENCES:
{numbered}"""


def find_anchors(sentences: list[str]) -> list[dict]:
    """Ask the model where the answerable units are."""
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    reply = llm.complete_json(
        ANCHOR_PROMPT.format(lo=MIN_QUESTIONS, hi=MAX_QUESTIONS, numbered=numbered),
        system=ANCHOR_SYSTEM,
        temperature=0.0,
        max_tokens=900,
        purpose="anchor questions for question-anchored chunking",
    )

    anchors = []
    for item in reply.get("anchors", []):
        if not isinstance(item, dict):
            continue
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        # Clamp to the real sentence range. Models routinely return indices
        # past the end of the list, and an unchecked slice silently produces
        # an empty chunk.
        start = max(1, min(start, len(sentences)))
        end = max(start, min(end, len(sentences)))
        anchors.append(
            {"question": str(item.get("question", "")).strip(), "start": start, "end": end}
        )

    return sorted(anchors, key=lambda a: a["start"])


def repair_coverage(anchors: list[dict], n_sentences: int) -> list[dict]:
    """Close gaps and overlaps the model left behind."""
    # =================================================================
    # VERIFYING MODEL OUTPUT. The prompt asks for non-overlapping ranges
    # that tile the section. The model mostly complies. "Mostly" is the
    # problem: text falling in a gap is never indexed, and that is a
    # silent data-loss bug no test catches unless you look for it.
    #
    # The general shape of working with model output: ask clearly, then
    # verify structurally.
    # =================================================================
    if not anchors:
        return [{"question": "", "start": 1, "end": n_sentences}]

    repaired: list[dict] = []
    cursor = 1
    for anchor in anchors:
        start = max(cursor, anchor["start"])
        if anchor["start"] > cursor:
            # Gap: extend this anchor backwards to swallow the orphaned text
            # rather than dropping it.
            start = cursor
        end = max(start, anchor["end"])
        if end < cursor:
            continue  # fully consumed by a previous range
        repaired.append({**anchor, "start": start, "end": end})
        cursor = end + 1

    if cursor <= n_sentences:
        repaired.append({"question": "", "start": cursor, "end": n_sentences})

    return repaired


def chunk_question_anchored(book: Book, limit: int | None = None) -> list[Chunk]:
    """Cut each section at the boundaries its anchor questions imply."""
    sections = [
        (story.title, section.number, section)
        for story in book.stories
        for section in story.sections
    ]
    if limit:
        sections = sections[:limit]

    chunks: list[Chunk] = []
    for position, (story_title, section_number, section) in enumerate(sections):
        print(f"    anchoring section {position + 1}/{len(sections)}", end="\r", flush=True)

        sentences: list[str] = []
        for paragraph in section.paragraphs:
            sentences.extend(split_sentences(paragraph))

        anchors = repair_coverage(find_anchors(sentences), len(sentences))

        # ---- THE KEY LINE ------------------------------------------
        # The chunk boundary is a sentence range the MODEL chose, because
        # that range is what its question needs in order to be answerable.
        # Not a token count, not a similarity threshold, not a paragraph.
        for index, anchor in enumerate(anchors):
            body = " ".join(sentences[anchor["start"] - 1 : anchor["end"]])
            if not body.strip():
                continue
            chunks.append(
                Chunk(
                    text=body,
                    strategy=STRATEGY,
                    story=story_title,
                    section=section_number,
                    index=index,
                    # The anchor question is stored as metadata AND used as an
                    # additional retrieval key -- the book keeps both.
                    retrieval_text=f"{anchor['question']} {body}".strip(),
                    meta={
                        "anchor_question": anchor["question"],
                        "sentence_range": [anchor["start"], anchor["end"]],
                        "n_sentences": anchor["end"] - anchor["start"] + 1,
                    },
                )
            )
    print(" " * 50, end="\r")
    print()

    return chunks


def explain_without_key(book: Book) -> None:
    section = book.stories[0].sections[0]
    sentences: list[str] = []
    for paragraph in section.paragraphs[:6]:
        sentences.extend(split_sentences(paragraph))

    display.para(
        "Without a key this example cannot place boundaries. Here is the real "
        "prompt for the opening of 'A Scandal in Bohemia', truncated to the "
        "first few sentences so it fits on screen."
    )
    numbered = "\n".join(f"{i + 1}. {display.truncate(s, 90)}" for i, s in enumerate(sentences[:10]))
    display.prompt_block("SYSTEM", ANCHOR_SYSTEM)
    display.prompt_block(
        "USER",
        ANCHOR_PROMPT.format(lo=MIN_QUESTIONS, hi=MAX_QUESTIONS, numbered=numbered),
    )
    display.illustrative(
        '{"anchors": [',
        '  {"question": "What was Holmes\'s attitude to Irene Adler?",',
        '   "start": 1, "end": 4},',
        '  {"question": "Why had Watson seen little of Holmes?",',
        '   "start": 5, "end": 10}',
        ']}',
    )
    print()
    display.para(
        "Sentences 1-4 become one chunk and 5-10 another. The boundary is at "
        "sentence 4 not because the topic shifted by some cosine threshold, "
        "but because that is where the first question stops being answered."
    )
    print()
    display.para(
        "Note what repair_coverage() in this file does with that reply: it "
        "checks the ranges actually tile the section. A model that skips "
        "sentences 11-14 would silently drop them from the index, and nothing "
        "downstream would ever report it."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4, help="sections to process")
    args = parser.parse_args()

    display.banner(
        "12. Question-Anchored Chunking",
        "Chapter 6, Chunking Strategies -- section 3.11",
        "Ask what questions a section answers, then cut it so each chunk "
        "answers one of them completely.",
    )

    book = load_book()

    if not llm.available():
        print(f"  No LLM configured: {llm.why_unavailable()}\n")
        explain_without_key(book)
        display.notice(
            "The boundary test here is self-containment -- can this chunk "
            "answer its question alone? Semantic chunking in example 06 can "
            "only test topical similarity, which is weaker.",
            "Questions are chosen before boundaries. That is the inversion "
            "against example 11, where boundaries came first and questions "
            "were derived from whatever the chunker happened to produce.",
            "Model output is repaired structurally, not trusted. Uncovered "
            "sentences would vanish from the index without any error.",
            "Cost is one call per section rather than per chunk, which makes "
            "this cheaper than example 11 on a corpus with large sections.",
        )
        return 0

    if llm.STUB:
        display.stub_warning()
    else:
        print(f"  model: {config.OPENAI_CHAT_MODEL}")
        print(f"  cached responses on disk: {llm.cached_count():,}")
    print()

    chunks = chunk_question_anchored(book, args.limit)
    sizes = [tokens.count_tokens(c.text) for c in chunks]

    display.kv("chunks produced", f"{len(chunks):,}")
    display.kv("mean tokens", f"{sum(sizes)/len(sizes):.0f}")
    display.kv("min / max tokens", f"{min(sizes)} / {max(sizes)}")

    display.heading("The anchors that placed the boundaries")
    display.table(
        ["anchor question", "sentences", "tokens"],
        [
            [
                display.truncate(c.meta["anchor_question"] or "(coverage filler)", 46),
                f"{c.meta['sentence_range'][0]}-{c.meta['sentence_range'][1]}",
                tokens.count_tokens(c.text),
            ]
            for c in chunks[:10]
        ],
    )

    filler = sum(1 for c in chunks if not c.meta["anchor_question"])
    display.heading("Coverage repair")
    display.kv("chunks from a model anchor", f"{len(chunks) - filler:,}")
    display.kv("chunks added to close gaps", f"{filler:,}")
    print("\n  Filler chunks are text the model's ranges did not cover. They are")
    print("  indexed anyway -- unreachable text is worse than unlabelled text.")

    display.sample_chunks(chunks, n=4, start=0, count_tokens=tokens.count_tokens)

    display.notice(
        f"{len(chunks):,} chunks whose boundaries were chosen by what they "
        "answer, not by size or similarity.",
        f"{filler:,} chunks had to be synthesised to cover gaps the model left. "
        "Verifying coverage is not optional when a model is choosing ranges.",
        "Each chunk carries its anchor question as both metadata and part of "
        "its retrieval key, so it benefits from the question-matching effect "
        "of example 11 as well.",
        "Example 13 composes this with the context buffers of example 10 into "
        "the book's full canonical chunk.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
