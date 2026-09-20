#!/usr/bin/env python3
"""
Build the corpus that every example in this repository runs against.

    Project Gutenberg #1661  ->  data/sherlock_holmes.txt  ->  data/sherlock_holmes.pdf

Both outputs are committed to the repo, so you do not need to run this script
to use the examples. It is here for provenance: it shows exactly how the
corpus was produced, and lets you rebuild it from source if you want to.

The PDF is deliberately *not* a scan or a stylised ebook. It is a plain,
well-structured, ~300-page book PDF, because example 01 has to extract text
back out of it and we want that extraction to be honest work rather than a
rigged demo.

    python scripts/build_corpus.py              # rebuild from the cached text
    python scripts/build_corpus.py --download   # re-fetch from Project Gutenberg
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TEXT_PATH = DATA_DIR / "sherlock_holmes.txt"
PDF_PATH = DATA_DIR / "sherlock_holmes.pdf"

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"

# Project Gutenberg wraps every work in a licence header and footer. Stripping
# them is the first real task in any ingestion pipeline: boilerplate that is
# structurally part of the file but not part of the document.
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"

# Story headings look like "I. A SCANDAL IN BOHEMIA" — a Roman numeral, a dot,
# then the title in capitals. Inner divisions are a bare Roman numeral: "II.".
STORY_HEADING_RE = re.compile(r"^([IVXL]+)\.\s+([A-Z][A-Z' ’.\-]+)$")
SECTION_HEADING_RE = re.compile(r"^([IVXL]+)\.$")

# Words that stay lowercase inside a title unless they open or close it.
TITLE_MINOR_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of",
    "on", "or", "the", "to", "with",
}


def titlecase(text: str) -> str:
    """Convert "THE ADVENTURE OF THE ENGINEER'S THUMB" to book title case.

    str.title() is no good here: it capitalises the letter after an
    apostrophe, giving "Engineer'S", and it capitalises articles that belong
    in lowercase.
    """
    words = text.lower().split()
    out: list[str] = []
    for index, word in enumerate(words):
        is_edge = index == 0 or index == len(words) - 1
        if word in TITLE_MINOR_WORDS and not is_edge:
            out.append(word)
        else:
            # Capitalise each hyphen-separated part ("Red-Headed") but leave
            # anything after an apostrophe alone ("Engineer's").
            out.append("-".join(part[:1].upper() + part[1:] for part in word.split("-")))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """A numbered division inside a story. Some stories have none."""

    number: str | None
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class Story:
    """One of the twelve adventures."""

    number: str
    title: str
    sections: list[Section] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(p.split()) for s in self.sections for p in s.paragraphs)


# ---------------------------------------------------------------------------
# Fetch and clean
# ---------------------------------------------------------------------------


def download_source() -> str:
    """Fetch the raw ebook from Project Gutenberg."""
    print(f"  downloading {GUTENBERG_URL}")
    with urllib.request.urlopen(GUTENBERG_URL, timeout=60) as response:
        return response.read().decode("utf-8")


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Drop everything outside the START/END markers.

    We locate the markers rather than slicing at fixed line numbers, because
    Gutenberg re-issues its files and the header length drifts between
    revisions.
    """
    start = raw.find(START_MARKER)
    end = raw.find(END_MARKER)
    if start == -1 or end == -1:
        raise SystemExit(
            "Could not find the Project Gutenberg START/END markers. "
            "The upstream file format may have changed."
        )
    # Advance past the marker's own line.
    start = raw.index("\n", start) + 1
    return raw[start:end]


def normalise(text: str) -> str:
    """Fold the text into a consistent encoding.

    Gutenberg's text uses typographic punctuation and marks italics with
    underscores. We keep the curly quotes (they are part of the prose) but
    normalise the Unicode form and remove the italic markers, which are a
    plain-text convention rather than content.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # _word_ is Gutenberg's italic convention; the emphasis is not recoverable
    # in plain text, so drop the markers rather than leak underscores into the
    # embeddings.
    text = re.sub(r"_([^_\n]+)_", r"\1", text)
    return text


def drop_table_of_contents(lines: list[str]) -> list[str]:
    """Remove the CONTENTS listing that precedes the first story.

    The listing repeats all twelve titles in title case. Left in, it becomes a
    chunk that matches every query about any story while containing no actual
    information — a textbook retrieval decoy.
    """
    for index, line in enumerate(lines):
        if line.strip().upper() == "CONTENTS":
            # Skip forward to the first real story heading.
            for offset in range(index + 1, len(lines)):
                if STORY_HEADING_RE.match(lines[offset].strip()):
                    return lines[offset:]
    return lines


# ---------------------------------------------------------------------------
# Parse into stories, sections and paragraphs
# ---------------------------------------------------------------------------


def parse_stories(text: str) -> list[Story]:
    """Split the cleaned text into twelve stories of numbered sections."""
    lines = drop_table_of_contents(text.split("\n"))

    stories: list[Story] = []
    current_section: Section | None = None
    buffer: list[str] = []

    def flush_paragraph() -> None:
        """Join the wrapped lines we have accumulated into one paragraph."""
        nonlocal buffer
        if buffer and current_section is not None:
            paragraph = " ".join(line.strip() for line in buffer).strip()
            paragraph = re.sub(r"\s+", " ", paragraph)
            if paragraph:
                current_section.paragraphs.append(paragraph)
        buffer = []

    for raw_line in lines:
        line = raw_line.strip()

        story_match = STORY_HEADING_RE.match(line)
        if story_match:
            flush_paragraph()
            number, title = story_match.groups()
            stories.append(Story(number=number, title=titlecase(title.strip())))
            # A story may open with prose before any numbered section, so start
            # an unnumbered one to catch it.
            current_section = Section(number=None)
            stories[-1].sections.append(current_section)
            continue

        section_match = SECTION_HEADING_RE.match(line)
        if section_match and stories:
            flush_paragraph()
            current_section = Section(number=section_match.group(1))
            stories[-1].sections.append(current_section)
            continue

        if not line:
            flush_paragraph()
        else:
            buffer.append(line)

    flush_paragraph()

    # Drop the leading unnumbered section where the story went straight into a
    # numbered one and left it empty.
    for story in stories:
        story.sections = [s for s in story.sections if s.paragraphs]

    return stories


# ---------------------------------------------------------------------------
# Emit the cleaned text
# ---------------------------------------------------------------------------


def render_text(stories: list[Story]) -> str:
    """Write the canonical plain-text corpus.

    One blank line between paragraphs, two before a heading. The examples that
    skip the PDF read this file, so its layout is a contract: paragraph
    boundaries must survive a `split("\\n\\n")`.
    """
    parts: list[str] = ["THE ADVENTURES OF SHERLOCK HOLMES", "by Arthur Conan Doyle", ""]
    for story in stories:
        parts.append("")
        parts.append(f"{story.number}. {story.title.upper()}")
        parts.append("")
        for section in story.sections:
            if section.number:
                parts.append(f"{section.number}.")
                parts.append("")
            for paragraph in section.paragraphs:
                parts.append(paragraph)
                parts.append("")
    return "\n".join(parts).strip() + "\n"


# ---------------------------------------------------------------------------
# Render the PDF
# ---------------------------------------------------------------------------


def render_pdf(stories: list[Story], path: Path) -> None:
    """Typeset the stories as a plain book PDF.

    We use ReportLab's base-14 Times faces so the file has no font
    dependencies and pypdf can extract the text cleanly in example 01.
    """
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
    )

    body = ParagraphStyle(
        "Body",
        fontName="Times-Roman",
        fontSize=11,
        leading=15.5,
        alignment=TA_JUSTIFY,
        firstLineIndent=18,
        spaceAfter=0,
    )
    # The first paragraph after a heading is not indented — standard book
    # typography, and it gives the extractor a visual cue to work with.
    body_first = ParagraphStyle("BodyFirst", parent=body, firstLineIndent=0)
    story_title = ParagraphStyle(
        "StoryTitle",
        fontName="Times-Bold",
        fontSize=17,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=26,
    )
    section_title = ParagraphStyle(
        "SectionTitle",
        fontName="Times-Bold",
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        spaceBefore=18,
        spaceAfter=14,
    )
    cover_title = ParagraphStyle(
        "CoverTitle",
        fontName="Times-Bold",
        fontSize=26,
        leading=32,
        alignment=TA_CENTER,
    )
    cover_author = ParagraphStyle(
        "CoverAuthor",
        fontName="Times-Roman",
        fontSize=14,
        leading=20,
        alignment=TA_CENTER,
    )
    toc_entry = ParagraphStyle(
        "TOCEntry", fontName="Times-Roman", fontSize=11, leading=20
    )

    def draw_page_number(canvas, doc) -> None:
        """Running footer. Page numbers are a classic extraction nuisance —
        example 01 has to strip them back out, which is the point."""
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(LETTER[0] / 2.0, 0.62 * inch, str(doc.page))
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=1.25 * inch,
        rightMargin=1.25 * inch,
        topMargin=1.0 * inch,
        bottomMargin=1.0 * inch,
        title="The Adventures of Sherlock Holmes",
        author="Arthur Conan Doyle",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="body",
    )
    doc.addPageTemplates(
        [PageTemplate(id="book", frames=[frame], onPage=draw_page_number)]
    )

    flow: list = []

    # Cover
    flow.append(Spacer(1, 2.4 * inch))
    flow.append(Paragraph("The Adventures of<br/>Sherlock Holmes", cover_title))
    flow.append(Spacer(1, 0.4 * inch))
    flow.append(Paragraph("Arthur Conan Doyle", cover_author))
    flow.append(PageBreak())

    # Contents
    flow.append(Paragraph("Contents", story_title))
    for story in stories:
        flow.append(Paragraph(f"{story.number}. &nbsp; {story.title}", toc_entry))
    flow.append(PageBreak())

    # Stories
    for story in stories:
        flow.append(Paragraph(f"{story.number}. {story.title}", story_title))
        for section_index, section in enumerate(story.sections):
            if section.number:
                flow.append(Paragraph(section.number + ".", section_title))
            for para_index, paragraph in enumerate(section.paragraphs):
                opens_section = para_index == 0
                style = body_first if opens_section else body
                # ReportLab parses a mini-HTML dialect, so literal ampersands
                # and angle brackets in the prose have to be escaped.
                safe = (
                    paragraph.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                flow.append(Paragraph(safe, style))
        flow.append(PageBreak())

    doc.build(flow)


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="re-fetch the source text from Project Gutenberg",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    print("Building the Sherlock Holmes corpus")

    if args.download or not TEXT_PATH.exists():
        raw = download_source()
        text = normalise(strip_gutenberg_boilerplate(raw))
        stories = parse_stories(text)
        TEXT_PATH.write_text(render_text(stories), encoding="utf-8")
        print(f"  wrote {TEXT_PATH.relative_to(REPO_ROOT)}")
    else:
        print(f"  using cached {TEXT_PATH.relative_to(REPO_ROOT)} (--download to refresh)")
        stories = parse_stories(TEXT_PATH.read_text(encoding="utf-8"))

    if len(stories) != 12:
        raise SystemExit(
            f"Expected 12 stories, parsed {len(stories)}. "
            "The source layout has probably changed."
        )

    total_words = sum(s.word_count for s in stories)
    print(f"  parsed {len(stories)} stories, {total_words:,} words")
    for story in stories:
        sections = len(story.sections)
        print(f"    {story.number:>4}. {story.title:<38} {story.word_count:>6,} words, {sections} section(s)")

    render_pdf(stories, PDF_PATH)
    size_mb = PDF_PATH.stat().st_size / 1_000_000
    print(f"  wrote {PDF_PATH.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
