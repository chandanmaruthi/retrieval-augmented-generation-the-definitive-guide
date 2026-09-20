#!/usr/bin/env python3
"""
06. Semantic-Boundary Chunking
Book: Chapter 6, Chunking Strategies -- section 3.5

WHAT THIS SHOWS: Let the embedding model decide where the topic changes.
Embed every sentence, measure the similarity between each adjacent pair, and
cut where that similarity falls off a cliff.

This is the first strategy that costs something to compute -- it embeds the
entire corpus before it can produce a single chunk. Whether the boundaries it
finds are worth that is exactly what example 15 measures.

HOW THIS SCRIPT PROCEEDS
    1. Flatten every section to sentences
    2. Embed all 6,665 of them            the expensive step, done once
    3. Score each gap by cosine similarity <-- THE TECHNIQUE: adjacent pairs
    4. Pick a threshold by percentile     <-- the part people get wrong
    5. Cut at low-similarity gaps, subject to min/max size
    6. Report WHY each chunk ended        model's choice, or the size limit?

Step 6 is the honest check. If the size limits placed most of the boundaries,
you paid to embed the corpus and got example 03's output back.


WHAT CHANGED SINCE EXAMPLE 05
    Every strategy so far used structure the author left behind -- sentences,
    paragraphs, headings. This is the first that has to COMPUTE its boundaries,
    and it pays for them: the whole corpus is embedded before a single chunk
    exists, and that bill arrives again on every reindex.


REQUIRES: -r requirements-local.txt (for real embeddings)
RUN: python examples/06_chunk_semantic.py [--percentile 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, tokens
from ragkit.corpus import Book, Chunk, load_book, split_sentences
from ragkit.embed import describe, get_embedder

STRATEGY = "semantic"

# Cut at the Nth percentile of similarity drops rather than at a fixed cosine
# value. The absolute numbers a model produces are not comparable across
# models or corpora -- MiniLM's "unrelated" is around 0.1 while another
# encoder's might be 0.4 -- so a hard-coded 0.5 threshold tuned on one setup
# silently produces one enormous chunk or ten thousand tiny ones on another.
# A percentile adapts, and it also gives you direct control over chunk count.
DEFAULT_PERCENTILE = 20
DEFAULT_MIN_TOKENS = 64
DEFAULT_MAX_TOKENS = 384


def chunk_semantic(
    book: Book,
    embedder,
    percentile: float = DEFAULT_PERCENTILE,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[list[Chunk], dict]:
    """Cut where consecutive sentences stop being about the same thing.

    The algorithm in three steps:

    1. Embed every sentence in a section.
    2. Score each gap by the cosine similarity of the sentences either side.
       Low similarity means the topic moved.
    3. Cut at the lowest-scoring gaps, subject to size limits.

    The size limits are not decoration. Left alone, similarity-based cutting
    produces a long tail of one-sentence chunks wherever the prose changes
    subject twice in a row, and occasionally a 2,000-token chunk across a
    stretch of uniform description. The book calls for min/max constraints for
    this reason and they do most of the work of making the output usable.
    """
    chunks: list[Chunk] = []
    all_gaps: list[float] = []

    # Embed the whole corpus once, section by section, so the cache is warm
    # and the per-section loop below is pure arithmetic.
    sections: list[tuple[str, str | None, list[str]]] = []
    for story in book.stories:
        for section in story.sections:
            sentences: list[str] = []
            for paragraph in section.paragraphs:
                sentences.extend(split_sentences(paragraph))
            if sentences:
                sections.append((story.title, section.number, sentences))

    flat = [s for _, _, sentences in sections for s in sentences]
    print(f"  embedding {len(flat):,} sentences...")
    vectors = embedder.encode(flat, show_progress=True)

    # =================================================================
    # THE TECHNIQUE, part 1: score every gap between adjacent sentences.
    # Vectors are unit length, so the dot product IS cosine similarity --
    # block[:-1] * block[1:] pairs each sentence with its successor.
    # =================================================================
    offset = 0
    per_section_gaps: list[np.ndarray] = []
    for _, _, sentences in sections:
        block = vectors[offset : offset + len(sentences)]
        gaps = (
            np.sum(block[:-1] * block[1:], axis=1)
            if len(sentences) > 1
            else np.array([])
        )
        per_section_gaps.append(gaps)
        all_gaps.extend(gaps.tolist())
        offset += len(sentences)

    # One global threshold, computed across the whole corpus. Per-section
    # thresholds would make every section split at its own weakest seam, even
    # one that is entirely about a single subject.
    #
    # ---- THE KEY DECISION ------------------------------------------
    # A PERCENTILE, not a fixed cosine value. This is the line that
    # decides whether the strategy survives a change of embedding model.
    #
    # Absolute thresholds do not travel: MiniLM's "unrelated" is around
    # 0.1 while another encoder's might be 0.4, so a hard-coded 0.5 tuned
    # on one setup yields either one enormous chunk or ten thousand tiny
    # ones on the next. A percentile adapts, and it also gives direct
    # control over how many chunks you end up with.
    threshold = float(np.percentile(all_gaps, percentile)) if all_gaps else 0.0

    for (story_title, section_number, sentences), gaps in zip(sections, per_section_gaps):
        buffer: list[str] = []
        budget = 0
        index = 0

        def emit(body: list[str], idx: int, reason: str) -> Chunk:
            return Chunk(
                text=" ".join(body),
                strategy=STRATEGY,
                story=story_title,
                section=section_number,
                index=idx,
                meta={"n_sentences": len(body), "boundary": reason,
                      "threshold": round(threshold, 4)},
            )

        for position, sentence in enumerate(sentences):
            buffer.append(sentence)
            budget += tokens.count_tokens(sentence)

            if position >= len(gaps):
                break  # last sentence: closed after the loop

            similarity = float(gaps[position])
            # Cut when the topic moved AND the chunk is big enough to stand
            # alone, or when we have simply run out of budget.
            # ---- THE CUT ---------------------------------------
            # Two conditions, and the second matters as much as the
            # first: a topic shift is only allowed to end a chunk that
            # is already big enough to stand alone. Without the size
            # floor, dialogue produces a long tail of one-line chunks
            # wherever the subject changes twice in a row.
            topic_shift = similarity < threshold and budget >= min_tokens
            over_budget = budget >= max_tokens

            if topic_shift or over_budget:
                chunks.append(
                    emit(buffer, index, "topic_shift" if topic_shift else "max_tokens")
                )
                index += 1
                buffer, budget = [], 0

        if buffer:
            chunks.append(emit(buffer, index, "section_end"))

    reasons = {"topic_shift": 0, "max_tokens": 0, "section_end": 0}
    for chunk in chunks:
        reasons[chunk.meta["boundary"]] += 1

    return chunks, {
        "threshold": threshold,
        "gaps": np.array(all_gaps),
        "reasons": reasons,
        "n_sentences": len(flat),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    display.banner(
        "06. Semantic-Boundary Chunking",
        "Chapter 6, Chunking Strategies -- section 3.5",
        "Embed every sentence, then cut where adjacent sentences stop "
        "resembling each other.",
    )

    try:
        embedder = get_embedder()
    except ImportError:
        display.missing_dependency(
            "sentence-transformers",
            "requirements-local.txt",
            "This strategy finds boundaries by comparing sentence embeddings, "
            "so unlike examples 02-05 it cannot run on text alone.",
        )
        return 0

    print(f"  {describe(embedder)}")
    if not embedder.semantic:
        print()
        display.para(
            "The hash embedder has no notion of meaning, so the 'semantic' "
            "boundaries below are really vocabulary-overlap boundaries. The "
            "code runs and the mechanics are visible, but do not read the "
            "chunk quality as evidence for or against this strategy. Install "
            "-r requirements-local.txt for that."
        )
    print(f"  {tokens.tokenizer_note()}\n")

    book = load_book()
    chunks, info = chunk_semantic(
        book, embedder, args.percentile, args.min_tokens, args.max_tokens
    )

    sizes = [tokens.count_tokens(c.text) for c in chunks]
    print()
    display.kv("chunks produced", f"{len(chunks):,}")
    display.kv("mean tokens", f"{sum(sizes)/len(sizes):.0f}")
    display.kv("min / max tokens", f"{min(sizes)} / {max(sizes)}")

    display.heading("The similarity distribution the threshold came from")
    gaps = info["gaps"]
    for label, value in [
        ("5th percentile", np.percentile(gaps, 5)),
        (f"{args.percentile:.0f}th percentile (cut here)", info["threshold"]),
        ("median", np.median(gaps)),
        ("95th percentile", np.percentile(gaps, 95)),
    ]:
        display.kv(label, f"{value:.3f}")
    print("\n  These are cosine similarities between neighbouring sentences.")
    print("  Note how low the median is: consecutive sentences in narrative")
    print("  prose are much less alike than intuition suggests, which is why a")
    print("  hand-picked absolute threshold is so hard to get right.")

    display.heading("Why each chunk ended")
    display.table(
        ["reason", "chunks", "share"],
        [
            [reason, f"{count:,}", f"{100*count/len(chunks):.0f}%"]
            for reason, count in sorted(
                info["reasons"].items(), key=lambda kv: -kv[1]
            )
        ],
    )
    shift = info["reasons"]["topic_shift"]
    forced = len(chunks) - shift
    print(f"\n  {100*shift/len(chunks):.0f}% of boundaries came from the similarity")
    print(f"  signal; the size limits forced only {forced}. That ratio is the number")
    print("  to watch. If the limits were placing most of the cuts, you would be")
    print("  paying to embed the corpus and getting example 03's boundaries back.")

    display.heading("A boundary the model chose")
    for position, chunk in enumerate(chunks[:-1]):
        if chunk.meta["boundary"] == "topic_shift" and len(chunk.text.split()) > 30:
            print("  end of one chunk:")
            print(f"      ...{display.truncate(chunk.text[-180:], 170)}")
            print("\n  start of the next:")
            print(f"      {display.truncate(chunks[position + 1].text[:180], 170)}...")
            break

    display.sample_chunks(chunks, n=4, start=2, count_tokens=tokens.count_tokens)

    display.notice(
        f"This strategy embedded {info['n_sentences']:,} sentences before "
        "producing a single chunk. Examples 02-05 produced theirs from text "
        "alone. That cost is paid once at index time -- and paid again every "
        "time the corpus changes.",
        f"A percentile threshold ({args.percentile:.0f}th here, cosine "
        f"{info['threshold']:.3f}) travels between models and corpora; a "
        "hard-coded cosine value does not. Swap the embedder and a fixed 0.5 "
        "threshold produces either one chunk or ten thousand.",
        f"{100*shift/len(chunks):.0f}% of boundaries came from the similarity "
        f"signal rather than the size limits, so the model really is placing "
        "these cuts. Whether they are *better* cuts is a separate question, and "
        "example 15 is where it gets answered with numbers.",
        "Adjacent-sentence similarity is a noisy signal in dialogue. '\"Indeed!\" "
        "said he.' resembles nothing on either side, so the model sees a topic "
        "shift where a human sees a conversation continuing.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
