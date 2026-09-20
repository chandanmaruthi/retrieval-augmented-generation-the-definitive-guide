#!/usr/bin/env python3
"""
01. Data Extraction
Book: Chapter 5, Data Extraction

WHAT THIS SHOWS: Everything that has to happen between "we have a PDF" and
"we have clean text with structure and metadata". Page numbers have to go,
wrapped lines have to be rejoined into paragraphs, paragraphs broken across a
page boundary have to be stitched, and the document's structure has to be
recovered from typography.

This is the least glamorous chapter in the book and the one that decides how
well everything downstream works. A chunking strategy cannot respect a
paragraph boundary that extraction destroyed.

HOW THIS SCRIPT PROCEEDS
    1. Raw extraction from the PDF        pypdf, one string per page
    2. Strip page numbers                 running furniture, not content
    3. Rebuild paragraphs from lines      <-- THE HARD PART, uses the indent cue
    4. Normalise encoding                 ligatures, soft hyphens, Unicode form
    5. Recover structure                  headings, by shape, since fonts are gone
    6. Enrich and deduplicate             metadata + ACLs, attached HERE
    7. Measure fidelity                   against the known-good source text

Step 3 is where the real work is: a PDF has no paragraphs, only glyphs at
coordinates, so every boundary below was reconstructed rather than read.


WHY THIS COMES FIRST
    Every later example inherits whatever this one produces. A heading lost
    here is a boundary example 05 cannot split on; a paragraph broken here is
    a fact example 15 will score as a chunking failure. Extraction is the
    least glamorous chapter in the book and the one that caps everything else.


REQUIRES: nothing
RUN: python examples/01_extract_pdf.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pypdf import PdfReader

from ragkit import display
from ragkit.corpus import PDF_PATH, TEXT_PATH


# ---------------------------------------------------------------------------
# Stage 1 — raw extraction
# ---------------------------------------------------------------------------


def extract_pages(path: Path) -> list[str]:
    """Pull raw text out of the PDF, one string per page.

    Note what we are NOT doing: asking pypdf for paragraphs. A PDF has no
    concept of a paragraph. It has glyphs at coordinates. Everything past this
    line is us reconstructing meaning that the format threw away.
    """
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


# ---------------------------------------------------------------------------
# Stage 2 — strip running page furniture
# ---------------------------------------------------------------------------

PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")


def strip_page_furniture(pages: list[str]) -> tuple[list[list[str]], int]:
    """Remove page numbers and other repeating headers/footers.

    Our PDF puts a bare page number on its own line. Left in, it becomes a
    token inside whatever chunk happens to span that page boundary --
    "...binding you 7 to absolute secrecy..." -- which is noise in the
    embedding and, worse, noise in the text the model is asked to quote from.
    """
    cleaned: list[list[str]] = []
    removed = 0
    for page in pages:
        keep: list[str] = []
        for line in page.split("\n"):
            if PAGE_NUMBER.match(line):
                removed += 1
                continue
            keep.append(line)
        cleaned.append(keep)
    return cleaned, removed


# ---------------------------------------------------------------------------
# Stage 3 — rebuild paragraphs from wrapped lines
# ---------------------------------------------------------------------------


def rebuild_paragraphs(pages: list[list[str]]) -> tuple[list[str], int]:
    """Join wrapped lines back into paragraphs, across page breaks.

    The layout signal we exploit: this book is typeset with a first-line
    indent, and that indent survives extraction as a leading space. So a line
    starting with whitespace opens a paragraph, and a line starting at column
    zero continues the previous one.

    That is a real reading-order cue rather than a guess, and it is exactly
    the kind of format-specific knowledge the book means by "layout
    reconstruction". A two-column academic paper would need a completely
    different rule -- which is the point: extraction is per-format work, not a
    library call.
    """
    paragraphs: list[str] = []
    buffer: list[str] = []
    stitched_across_pages = 0

    for page_index, lines in enumerate(pages):
        for line_index, line in enumerate(lines):
            if not line.strip():
                continue

            # Headings are centred, so they carry no first-line indent and the
            # indent rule alone would glue them onto the previous paragraph --
            # which silently destroys the entire document structure. We give
            # them their own test.
            #
            # In a production extractor this is where you would use font size
            # and weight from the PDF's text state. pypdf's plain extract_text
            # discards that, so we fall back to shape, which is the usual
            # trade: cheaper parser, more pattern matching afterwards.
            looks_like_heading = bool(
                STORY_HEADING.match(line.strip()) or SECTION_HEADING.match(line.strip())
            )

            opens_paragraph = line.startswith((" ", "\t")) or looks_like_heading

            if opens_paragraph and buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            elif not opens_paragraph and buffer and line_index == 0 and page_index > 0:
                # A continuation line that is the first line of a page: the
                # paragraph runs across the page break. Count it, because this
                # is the case naive per-page extraction silently truncates.
                stitched_across_pages += 1

            if looks_like_heading:
                # Stands alone. The paragraph that follows a heading is also
                # set without an indent, so without this the heading and its
                # opening paragraph would fuse into one.
                paragraphs.append(line.strip())
                continue

            buffer.append(line.strip())

    if buffer:
        paragraphs.append(" ".join(buffer))

    return [re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip()], stitched_across_pages


# ---------------------------------------------------------------------------
# Stage 4 — normalise
# ---------------------------------------------------------------------------

LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
}
SOFT_HYPHEN = "­"


def normalise(paragraphs: list[str]) -> tuple[list[str], dict[str, int]]:
    """Fold the text into one consistent encoding.

    Counted rather than asserted. This particular PDF was typeset from clean
    Unicode, so it has no ligatures and no soft hyphens, and the counts below
    come back zero -- which is the honest result, and worth seeing. Run the
    same code over a PDF produced by LaTeX or a scanner and the counts will
    not be zero: "fi" arrives as a single glyph U+FB01 that no tokenizer
    matches, and hyphenated line breaks leave soft hyphens mid-word so that
    "invest-igation" never matches a query for "investigation".
    """
    counts = {"ligatures": 0, "soft_hyphens": 0, "recomposed": 0}
    out: list[str] = []

    for paragraph in paragraphs:
        for ligature, replacement in LIGATURES.items():
            if ligature in paragraph:
                counts["ligatures"] += paragraph.count(ligature)
                paragraph = paragraph.replace(ligature, replacement)

        if SOFT_HYPHEN in paragraph:
            counts["soft_hyphens"] += paragraph.count(SOFT_HYPHEN)
            paragraph = paragraph.replace(SOFT_HYPHEN, "")

        # A hyphen followed by a line break is a typesetting artefact, not a
        # real hyphen. Our line-joining already collapsed the break to a
        # space, so the tell is "word- word".
        paragraph = re.sub(r"(\w)-\s+(\w)", r"\1\2", paragraph)

        composed = unicodedata.normalize("NFC", paragraph)
        if composed != paragraph:
            counts["recomposed"] += 1
        out.append(composed)

    return out, counts


# ---------------------------------------------------------------------------
# Stage 5 — recover structure
# ---------------------------------------------------------------------------

STORY_HEADING = re.compile(r"^([IVXL]+)\.\s+([A-Z][A-Za-z' ’\-]+)$")
SECTION_HEADING = re.compile(r"^([IVXL]+)\.$")


def recover_structure(paragraphs: list[str]) -> list[dict]:
    """Attach every paragraph to the story and section it belongs to.

    Headings are recognised by shape, because that is all extraction left us:
    the PDF knew these lines were 17pt bold and centred, and pypdf returned
    them as plain strings identical to body text. Font information is exactly
    the sort of structure that gets lost, and recovering it by pattern is
    normal extraction work.

    The result is the section path that examples 05 and 09 need -- you cannot
    do heading-boundary chunking without knowing where the headings were.
    """
    records: list[dict] = []
    story_number = story_title = None
    section = None

    for position, paragraph in enumerate(paragraphs):
        heading = STORY_HEADING.match(paragraph)
        if heading:
            # The contents page lists all twelve titles in exactly the same
            # shape as the real headings, so pattern matching alone would
            # "find" each story twice and attribute nothing to the first
            # twelve. The tell is that a contents entry is followed
            # immediately by another entry, with no prose in between.
            following = paragraphs[position + 1] if position + 1 < len(paragraphs) else ""
            if STORY_HEADING.match(following):
                continue

            story_number, story_title = heading.groups()
            section = None
            continue

        marker = SECTION_HEADING.match(paragraph)
        if marker and story_title:
            section = marker.group(1)
            continue

        if story_title is None:
            continue  # cover and contents pages

        records.append(
            {
                "story_number": story_number,
                "story": story_title,
                "section": section,
                "text": paragraph,
            }
        )

    return records


# ---------------------------------------------------------------------------
# Stage 6 — enrich and deduplicate
# ---------------------------------------------------------------------------


def enrich(records: list[dict], source: Path) -> list[dict]:
    """Attach the metadata the rest of the pipeline will need.

    The book is emphatic that this happens at extraction time, not later --
    in particular access-control identifiers, because a system that retrieves
    first and filters afterwards has already put the document into the
    model's context. We carry an `acl` field from here on for that reason;
    example 35 is where it starts doing work.
    """
    for index, record in enumerate(records):
        record["source"] = source.name
        record["index"] = index
        record["n_chars"] = len(record["text"])
        record["content_hash"] = hashlib.sha256(
            record["text"].encode("utf-8")
        ).hexdigest()
        record["acl"] = ["public"]
    return records


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """Drop repeated paragraphs, keyed on a hash of their content.

    In a real corpus this catches boilerplate: the confidentiality notice on
    every contract, the header repeated in every exported ticket. Here it
    mostly catches short repeated lines of dialogue -- and that is a useful
    warning. Deduplicating on exact content will happily delete a legitimately
    repeated sentence, so production systems dedupe on a document or section
    hash, not a paragraph one.
    """
    seen: set[str] = set()
    kept: list[dict] = []
    dropped = 0
    for record in records:
        if record["content_hash"] in seen:
            dropped += 1
            continue
        seen.add(record["content_hash"])
        kept.append(record)
    return kept, dropped


# ---------------------------------------------------------------------------


def main() -> int:
    display.banner(
        "01. Data Extraction",
        "Chapter 5, Data Extraction",
        "Turning a 189-page PDF into clean, structured, attributed text -- and "
        "measuring how much of the original survived the round trip.",
    )

    if not PDF_PATH.exists():
        print(f"Corpus PDF missing. Run:  python scripts/build_corpus.py --download")
        return 1

    # --- Stage 1 ---------------------------------------------------------
    display.heading("Stage 1: raw extraction")
    pages = extract_pages(PDF_PATH)
    raw_chars = sum(len(p) for p in pages)
    display.kv("pages", len(pages))
    display.kv("characters", f"{raw_chars:,}")
    print("\n  What pypdf actually returns for page 6 (first 5 lines):")
    for line in pages[5].split("\n")[:5]:
        print(f"      {line[:72]!r}")
    print("\n  Note line 1: a bare page number, and lines that stop mid-sentence.")

    # --- Stage 2 ---------------------------------------------------------
    display.heading("Stage 2: strip running page furniture")
    page_lines, removed = strip_page_furniture(pages)
    display.kv("page numbers removed", removed)

    # --- Stage 3 ---------------------------------------------------------
    display.heading("Stage 3: rebuild paragraphs from wrapped lines")
    paragraphs, stitched = rebuild_paragraphs(page_lines)
    display.kv("paragraphs recovered", f"{len(paragraphs):,}")
    display.kv("stitched across a page break", stitched)
    print("\n  A long paragraph reassembled from wrapped lines:")
    for paragraph in paragraphs:
        if len(paragraph) > 400 and "Contents" not in paragraph:
            display.preview(paragraph, limit=210)
            break

    # --- Stage 4 ---------------------------------------------------------
    display.heading("Stage 4: normalise encoding")
    paragraphs, counts = normalise(paragraphs)
    for key, value in counts.items():
        display.kv(key, value)
    print("\n  Zeroes here are the honest answer for a digitally typeset PDF.")
    print("  Run this stage over a scanned or LaTeX-produced document and they")
    print("  will not be zero -- see the comment in normalise().")

    # --- Stage 5 ---------------------------------------------------------
    display.heading("Stage 5: recover structure")
    records = recover_structure(paragraphs)
    stories = {r["story"] for r in records}
    with_section = sum(1 for r in records if r["section"])
    display.kv("stories detected", f"{len(stories)} (expected 12)")
    display.kv("paragraphs attributed", f"{len(records):,}")
    display.kv("inside a numbered section", f"{with_section:,}")

    # --- Stage 6 ---------------------------------------------------------
    display.heading("Stage 6: enrich and deduplicate")
    records = enrich(records, PDF_PATH)
    records, dropped = deduplicate(records)
    display.kv("duplicate paragraphs dropped", dropped)
    display.kv("final records", f"{len(records):,}")
    print("\n  One finished record:")
    sample = records[12]
    for key in ("story", "section", "source", "acl", "n_chars"):
        print(f"      {key:<14} {sample[key]}")
    print(f"      {'content_hash':<14} {sample['content_hash'][:16]}...")
    print(f"      {'text':<14} {display.truncate(sample['text'], 60)}")

    # --- Fidelity --------------------------------------------------------
    # We have the original text committed alongside the PDF, so extraction
    # quality is measurable rather than a matter of faith.
    display.heading("Fidelity against the original text")
    original = TEXT_PATH.read_text(encoding="utf-8")
    original_words = len(original.split())
    extracted_words = sum(len(r["text"].split()) for r in records)
    retention = 100.0 * extracted_words / original_words
    display.kv("words in source text", f"{original_words:,}")
    display.kv("words after extraction", f"{extracted_words:,}")
    display.kv("retention", f"{retention:.2f}%")

    display.notice(
        "A PDF stores glyphs at coordinates, not paragraphs. Every paragraph "
        "boundary in the output above was reconstructed from a typographic "
        "cue -- here, the first-line indent surviving as a leading space.",
        "Page numbers are the clearest case of structure that is in the file "
        "but not in the document. Left in, they land inside chunks and get "
        "quoted back as if they were content.",
        f"{stitched} paragraphs ran across a page break. Extract page-by-page "
        "without stitching and each of those becomes two half-paragraphs, "
        "which is a retrieval failure the chunker gets blamed for.",
        "Access-control identifiers are attached here, at extraction, not at "
        "query time. The book is firm on this: filtering after retrieval "
        "means the document already reached the model.",
        f"Retention is {retention:.1f}% against the known original -- the sort "
        "of number worth tracking per source, since a drop is how you find out "
        "a parser broke on a new document format.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
