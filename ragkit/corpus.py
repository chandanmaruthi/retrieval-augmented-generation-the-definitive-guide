"""
Loading and structuring the Sherlock Holmes corpus.

Every example starts here. The corpus has four nested levels:

    Book  ->  Story  ->  Section  ->  Paragraph  ->  Sentence

which is the same hierarchy the book's Chapter 13 calls
``Topic -> Doc -> Section -> Sentence``. The hierarchical, parent-child and
heading-boundary chunking strategies all rely on those levels being real
rather than invented, which is why this corpus was chosen.

Nothing here implements a retrieval technique. This module exists so that
thirty-eight example scripts do not each re-parse a book.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_PATH = DATA_DIR / "sherlock_holmes.txt"
PDF_PATH = DATA_DIR / "sherlock_holmes.pdf"

_STORY_HEADING = re.compile(r"^([IVXL]+)\.\s+([A-Z][A-Z' ’.\-]+)$")
_SECTION_HEADING = re.compile(r"^([IVXL]+)\.$")


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """A numbered division within a story. Most stories have exactly one."""

    number: str | None
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


@dataclass
class Story:
    """One of the twelve adventures — the natural 'document' in this corpus."""

    number: str
    title: str
    sections: list[Section] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)

    @property
    def paragraphs(self) -> list[str]:
        return [p for s in self.sections for p in s.paragraphs]

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Book:
    """The whole corpus."""

    title: str
    author: str
    stories: list[Story] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.stories)

    @property
    def word_count(self) -> int:
        return sum(s.word_count for s in self.stories)

    def story(self, needle: str) -> Story:
        """Find one story by a case-insensitive substring of its title."""
        needle = needle.lower()
        for candidate in self.stories:
            if needle in candidate.title.lower():
                return candidate
        raise KeyError(f"No story matching {needle!r}")


@dataclass
class Chunk:
    """A retrievable unit produced by one of the chunking strategies.

    Every strategy in examples 02-14 emits these, so the comparison harness in
    example 15 and the indexer in scripts/index_corpus.py can treat all
    thirteen identically.

    ``text`` is what gets embedded. ``retrieval_text`` is what gets embedded
    *instead*, when a strategy deliberately indexes something other than the
    chunk body — contextual headers prepend a heading path, question-derived
    chunking indexes generated questions. Keeping them separate is what makes
    those strategies work.
    """

    text: str
    strategy: str
    story: str
    index: int
    section: str | None = None
    retrieval_text: str | None = None
    parent_text: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """The string this chunk should be embedded as."""
        return self.retrieval_text if self.retrieval_text is not None else self.text

    @property
    def context_text(self) -> str:
        """The string that should be handed to the generator once retrieved.

        For parent-child strategies this is the larger parent passage, which
        is the whole point: retrieve on a precise child, generate from a
        complete parent.
        """
        return self.parent_text if self.parent_text is not None else self.text

    @property
    def location(self) -> str:
        """Human-readable provenance, used in citations and debug output."""
        if self.section:
            return f"{self.story} §{self.section}"
        return self.story


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_book(text: str) -> Book:
    """Parse the canonical plain-text corpus into its four levels."""
    lines = text.split("\n")
    book = Book(title="The Adventures of Sherlock Holmes", author="Arthur Conan Doyle")

    current_section: Section | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer and current_section is not None:
            paragraph = re.sub(r"\s+", " ", " ".join(buffer)).strip()
            if paragraph:
                current_section.paragraphs.append(paragraph)
        buffer = []

    for raw in lines:
        line = raw.strip()

        heading = _STORY_HEADING.match(line)
        if heading:
            flush()
            number, title = heading.groups()
            book.stories.append(Story(number=number, title=_titlecase(title.strip())))
            current_section = Section(number=None)
            book.stories[-1].sections.append(current_section)
            continue

        section = _SECTION_HEADING.match(line)
        if section and book.stories:
            flush()
            current_section = Section(number=section.group(1))
            book.stories[-1].sections.append(current_section)
            continue

        if not line:
            flush()
        else:
            buffer.append(line)

    flush()

    for story in book.stories:
        story.sections = [s for s in story.sections if s.paragraphs]

    return book


_TITLE_MINOR = {
    "a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of",
    "on", "or", "the", "to", "with",
}


def _titlecase(text: str) -> str:
    words = text.lower().split()
    out = []
    for i, word in enumerate(words):
        if word in _TITLE_MINOR and 0 < i < len(words) - 1:
            out.append(word)
        else:
            out.append("-".join(p[:1].upper() + p[1:] for p in word.split("-")))
    return " ".join(out)


@lru_cache(maxsize=1)
def load_book() -> Book:
    """Load the corpus from the committed plain text.

    Cached, because thirteen chunking strategies in one comparison run should
    parse the book once. Use :func:`load_book_from_pdf` when you specifically
    want to demonstrate PDF extraction.
    """
    if not TEXT_PATH.exists():
        raise SystemExit(
            f"Corpus not found at {TEXT_PATH}.\n"
            "Run:  python scripts/build_corpus.py --download"
        )
    return parse_book(TEXT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Abbreviations that end in a period without ending a sentence. Conan Doyle's
# prose is thick with "Mr.", "Dr." and "St.", so a naive split on /\. / cuts
# sentences in the wrong place several hundred times across the book.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "co", "inc", "ltd", "jr", "sr",
    "vs", "etc", "e.g", "i.e", "prof", "rev", "hon", "capt", "col",
    "gen", "lt", "sgt", "no", "vol", "p.m", "a.m",
}

# Split on whitespace that follows terminal punctuation and precedes something
# that looks like a new sentence. The two lookbehind branches are spelled out
# separately because Python's `re` only accepts fixed-width lookbehind, so the
# closing quote cannot be written as an optional `?`.
_SENTENCE_BOUNDARY = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'”’]))\s+(?=[\"'“‘(]?[A-Z0-9])"
)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences.

    A deliberately small rule-based splitter rather than an NLP dependency:
    sentence segmentation is an input to several chunking strategies, not the
    subject of any of them, and a 20-line splitter that you can read beats a
    model you cannot.
    """
    candidates = _SENTENCE_BOUNDARY.split(text)

    sentences: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        # Re-join a fragment that was split after an abbreviation.
        if sentences:
            previous = sentences[-1]
            last_word = previous.rstrip(".").split()[-1].lower() if previous.split() else ""
            if last_word in _ABBREVIATIONS:
                sentences[-1] = f"{previous} {candidate}"
                continue
        sentences.append(candidate)

    return sentences


def iter_paragraphs(book: Book):
    """Yield ``(story_title, section_number, paragraph_index, text)`` across the book."""
    for story in book.stories:
        index = 0
        for section in story.sections:
            for paragraph in section.paragraphs:
                yield story.title, section.number, index, paragraph
                index += 1
