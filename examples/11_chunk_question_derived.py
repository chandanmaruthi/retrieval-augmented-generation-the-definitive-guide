#!/usr/bin/env python3
"""
11. Question-Derived Chunking
Book: Chapter 6, Chunking Strategies -- section 3.10

WHAT THIS SHOWS: Stop embedding the answer and start embedding the question.

For each chunk, have a model write the questions that chunk answers, then use
those questions as the retrieval key. The chunk text is still what the
generator reads -- but what sits in the vector index is a question.

The reason this helps is a mismatch nobody designed on purpose: users type
questions, corpora contain statements, and a question and its answer are often
not similar strings. "Why was the Red-Headed League invented?" shares almost
no vocabulary with the paragraph that explains it.

HOW THIS SCRIPT PROCEEDS
    1. Start from example 05's chunks
    2. Ask a model what questions each one answers   one call per chunk
    3. Index each QUESTION as its own entry     <-- THE TECHNIQUE, the inversion
    4. Point every entry back at the same passage
    5. Show the index multiplier and the size difference

Step 3 is the whole idea: what sits in the vector index is no longer the
document. It is a question the document answers.


WHAT CHANGED SINCE EXAMPLE 10
    Example 10 added context to the chunk before embedding it. This replaces
    the embedded text entirely.

    The problem being solved is a vocabulary mismatch nobody designed on
    purpose: users type questions, corpora contain statements, and a question
    rarely shares wording with its own answer. "Why was the Red-Headed League
    invented?" has almost nothing in common with the paragraph explaining it --
    but it has a great deal in common with another question.


REQUIRES: OPENAI_API_KEY (falls back to showing the prompt)
RUN: python examples/11_chunk_question_derived.py [--limit 20]
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

STRATEGY = "question_derived"

QUESTIONS_PER_CHUNK = 4

SYSTEM = (
    "You generate retrieval questions for a search index over fiction. "
    "You only ask questions the passage itself answers. "
    "You reply with JSON."
)

PROMPT = """Read the passage and write {n} questions that it answers.

Rules:
- Every question must be answerable from this passage alone.
- Write the questions a reader would actually type, not exam questions.
- Name people and places explicitly; do not write "he" or "there", because
  the question will be read without the passage next to it.
- Vary the form: some factual, some about motive or consequence.

Reply as JSON: {{"questions": ["...", "..."]}}

PASSAGE:
{passage}"""


def generate_questions(passage: str, n: int = QUESTIONS_PER_CHUNK) -> list[str]:
    """Ask the model what this passage answers."""
    reply = llm.complete_json(
        PROMPT.format(n=n, passage=passage),
        system=SYSTEM,
        temperature=0.0,
        max_tokens=400,
        purpose="question generation for question-derived chunking",
    )
    questions = reply.get("questions", [])
    # Defend against the model returning a bare string or nested junk. This
    # costs three lines and saves an indexing run that dies at chunk 900.
    return [q.strip() for q in questions if isinstance(q, str) and q.strip()][:n]


def chunk_question_derived(book: Book, limit: int | None = None) -> list[Chunk]:
    """Index the questions; return the chunk.

    One chunk becomes several index entries, one per generated question. They
    all point back at the same passage, so retrieval has to deduplicate on
    chunk identity before building the context -- otherwise a passage with
    four matching questions occupies four of your five context slots.
    """
    base = import_module("05_chunk_heading").chunk_by_heading(book)
    if limit:
        base = base[:limit]

    entries: list[Chunk] = []
    for position, chunk in enumerate(base):
        print(f"    generating questions {position + 1}/{len(base)}", end="\r", flush=True)
        questions = generate_questions(chunk.text)

        # ---- THE TECHNIQUE -----------------------------------------
        # One chunk becomes SEVERAL index entries -- one per question --
        # all carrying the same passage as `text` but a different question
        # as `retrieval_text`.
        #
        # The consequence to remember: several of these can match a single
        # query, so retrieval MUST deduplicate on chunk_key afterwards or
        # one passage fills the entire top-k with itself.
        for question_index, question in enumerate(questions):
            entries.append(
                Chunk(
                    # The passage is still what the generator reads.
                    text=chunk.text,
                    strategy=STRATEGY,
                    story=chunk.story,
                    section=chunk.section,
                    index=chunk.index,
                    # The question is what gets embedded.
                    retrieval_text=question,
                    meta={
                        "source_chunk": chunk.index,
                        "chunk_key": f"{chunk.story}#{chunk.section}#{chunk.index}",
                        "question": question,
                        "question_index": question_index,
                        "n_questions": len(questions),
                    },
                )
            )
    print(" " * 50, end="\r")
    print()

    return entries


def explain_without_key(book: Book) -> None:
    base = import_module("05_chunk_heading").chunk_by_heading(book)
    sample = base[40]

    display.para(
        "Without a key this example cannot generate questions. The prompt is "
        "the technique, so here it is on a real chunk, with the shape of the "
        "reply and what would be done with it."
    )
    display.prompt_block("SYSTEM", SYSTEM)
    display.prompt_block(
        "USER", PROMPT.format(n=QUESTIONS_PER_CHUNK, passage=display.truncate(sample.text, 550))
    )
    display.illustrative(
        '{"questions": [',
        '  "What did Holmes deduce from the condition of Watson\'s boots?",',
        '  "How did Holmes know Watson had been out in bad weather?",',
        '  "Why did Holmes conclude Mary Jane was careless?",',
        '  "What is the difference between seeing and observing?"',
        ']}',
    )
    print()
    display.para(
        "Each of those four questions would be embedded and indexed as a "
        "separate entry pointing back at this one passage. A user asking "
        "anything close to one of them matches a question-to-question "
        "similarity, which is a far easier match than question-to-prose."
    )
    print()
    display.kv("index entries", f"{len(base) * QUESTIONS_PER_CHUNK:,} for {len(base):,} chunks")
    display.kv("cost", f"one call per chunk -- {len(base):,} calls")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    display.banner(
        "11. Question-Derived Chunking",
        "Chapter 6, Chunking Strategies -- section 3.10",
        "Generate the questions each chunk answers, and index those instead of "
        "the chunk. Match question to question, not question to prose.",
    )

    book = load_book()

    if not llm.available():
        print(f"  No LLM configured: {llm.why_unavailable()}\n")
        explain_without_key(book)
        display.notice(
            "The retrieval problem this solves is a vocabulary mismatch: users "
            "write questions, documents contain statements, and the two rarely "
            "share wording even when one answers the other.",
            "Indexing questions multiplies the index. Four questions per chunk "
            "is four times the vectors, and several of them can match one "
            "query -- so retrieval must deduplicate back to the chunk.",
            "The generated questions are only as good as the model's reading. "
            "A question about a fact the passage does not contain creates a "
            "confident match to a passage that cannot answer it.",
            "Example 14 keeps both indexes -- questions and passages -- and "
            "searches them together, which is the version worth running in "
            "production.",
        )
        return 0

    if llm.STUB:
        display.stub_warning()
    else:
        print(f"  model: {config.OPENAI_CHAT_MODEL}")
        print(f"  cached responses on disk: {llm.cached_count():,}")
    print()

    entries = chunk_question_derived(book, args.limit)
    unique_chunks = len({e.meta["chunk_key"] for e in entries})

    display.kv("source chunks", f"{unique_chunks:,}")
    display.kv("index entries", f"{len(entries):,}")
    display.kv("multiplier", f"{len(entries)/unique_chunks:.1f}x")

    display.heading("One passage, several retrieval keys")
    key = entries[0].meta["chunk_key"]
    group = [e for e in entries if e.meta["chunk_key"] == key]
    print("  PASSAGE (what the generator reads):")
    display.preview(group[0].text, limit=240)
    print("\n  EMBEDDED AS (what the index searches):")
    for entry in group:
        print(f"      - {entry.embed_text}")

    display.heading("Size comparison")
    question_tokens = sum(tokens.count_tokens(e.embed_text) for e in entries) / len(entries)
    passage_tokens = sum(tokens.count_tokens(e.text) for e in entries) / len(entries)
    display.table(
        ["", "mean tokens"],
        [["embedded (a question)", f"{question_tokens:.0f}"],
         ["read by the model (the passage)", f"{passage_tokens:.0f}"]],
    )

    display.sample_chunks(entries, n=4, start=0, count_tokens=tokens.count_tokens)

    display.notice(
        f"{len(entries):,} index entries for {unique_chunks:,} passages -- "
        f"{len(entries)/unique_chunks:.1f}x the vectors, all pointing at the "
        "same text.",
        "Retrieval must deduplicate on chunk identity. Without it, one passage "
        "whose questions all match will fill the entire top-k with itself.",
        f"The embedded strings average {question_tokens:.0f} tokens against "
        f"{passage_tokens:.0f} for the passages. Short, specific keys are "
        "precisely what dense retrieval is good at.",
        "This strategy moves work from query time to index time, which the "
        "book's reference architecture calls the cheaper win -- but here the "
        "index-time work is paid in API calls, not CPU.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
