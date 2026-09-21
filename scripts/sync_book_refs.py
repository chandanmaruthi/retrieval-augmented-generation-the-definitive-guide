#!/usr/bin/env python3
"""
Keep this repo's references to book chapters correct.

    python scripts/sync_book_refs.py --check    # CI: fail if anything is stale
    python scripts/sync_book_refs.py --write    # rewrite the numbers

Every example opens with a line naming the chapter it belongs to:

    Book: Chapter 10, Hybrid RAG

Chapter *numbers* move whenever the book gains a chapter; chapter *titles* do
not. So the title is the key, the number is generated, and nobody maintains
38 of these by hand. The same goes for the README's chapter tables, which are
emitted between markers rather than written.

The map comes from docs/book_chapters.json, which the book repo publishes
with its export_chapters.py. If a title here has no match there, that is a
real error and not something to paper over -- either the chapter was renamed
in the book, or this example cites a chapter that does not exist.

Stdlib only, so CI needs no dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "docs" / "book_chapters.json"
README = ROOT / "README.md"
EXAMPLES = ROOT / "examples"

# "Book: Chapter 10, Hybrid RAG" and the chunking variant that adds a section:
# "Book: Chapter 6, Chunking Strategies -- section 3.1"
HEADER_RE = re.compile(
    r"^(?P<prefix>Book:\s+Chapter\s+)(?P<num>\d+)(?P<sep>,\s*)(?P<title>[^\n]+?)"
    r"(?P<suffix>(?:\s+--\s+section\s+[\d.]+)?)\s*$",
    re.MULTILINE,
)

# The same reference a second time, inside display.banner():
#
#     display.banner(
#         "29. Streaming RAG",
#         "Chapter 16, Streaming RAG",      <-- this one
#
# It is the copy a reader actually SEES, because it prints when the example
# runs. Updating only the docstring header leaves the two disagreeing, which
# is worse than leaving both stale — one of them is then provably wrong.
BANNER_RE = re.compile(
    r'(?P<prefix>"Chapter\s+)(?P<num>\d+)(?P<sep>,\s*)(?P<title>[^"\n]+?)'
    r'(?P<suffix>(?:\s+--\s+section\s+[\d.]+)?)"',
)

BEGIN = "<!-- book-index:begin -->"
END = "<!-- book-index:end -->"

# The book's structure, also emitted rather than written. It went stale
# within one edition last time: parts were renumbered and the README kept
# advertising a "Part I — About" that no longer existed.
PARTS_BEGIN = "<!-- book-parts:begin -->"
PARTS_END = "<!-- book-parts:end -->"

# Titles that differ between the two repos, usually because the example header
# was written before the chapter was renamed. Kept explicit rather than fuzzy
# matching, so a genuine mismatch still fails loudly.
ALIASES = {
    "Human-in-the-Loop RAG Systems": "Human-in-the-Loop RAG",
    "Reference Architecture (block 3, Indexing)": "Reference Architecture",
}


def load_map():
    if not MAP.exists():
        sys.exit(
            "No %s.\nThe book repo publishes it:\n"
            "  python3 export_chapters.py --repo %s" % (MAP, ROOT))
    data = json.loads(MAP.read_text(encoding="utf-8"))
    by_title = {}
    for chapter in data["chapters"]:
        by_title[chapter["title"]] = chapter
    return data, by_title


def resolve(title, by_title):
    """Chapter record for a title as an example writes it, or None."""
    title = title.strip()
    if title in by_title:
        return by_title[title]
    if title in ALIASES and ALIASES[title] in by_title:
        return by_title[ALIASES[title]]
    # "Reference Architecture (block 3, Indexing)" -> strip the parenthetical
    bare = re.sub(r"\s*\(.*\)\s*$", "", title)
    if bare in by_title:
        return by_title[bare]
    return None


# ----------------------------------------------------------------------
# Example headers
# ----------------------------------------------------------------------

def scan_examples(by_title):
    """Return (path, current_number, wanted_number, title, new_text)."""
    results = []
    for path in sorted(EXAMPLES.glob("[0-9][0-9]_*.py")):
        text = path.read_text(encoding="utf-8")
        match = HEADER_RE.search(text)
        if not match:
            results.append((path, None, None, None, None))
            continue

        title = match.group("title").strip()
        chapter = resolve(title, by_title)
        if chapter is None:
            results.append((path, int(match.group("num")), None, title, None))
            continue

        wanted = chapter["number"]
        current = int(match.group("num"))

        updated = text
        reasons = []
        changed = current != wanted
        if changed:
            reasons.append("docstring header says Chapter %d, book says "
                           "Chapter %d" % (current, wanted))
            replacement = "%s%d%s%s%s" % (
                match.group("prefix"), wanted, match.group("sep"),
                match.group("title"), match.group("suffix"))
            updated = text[:match.start()] + replacement + text[match.end():]

        # Then the banner copies, resolved independently — a file may name a
        # different chapter in its banner than in its header, and each one
        # should follow the chapter its own title points at.
        def fix_banner(m):
            banner_chapter = resolve(m.group("title"), by_title)
            if banner_chapter is None:
                return m.group(0)
            return '%s%d%s%s%s"' % (
                m.group("prefix"), banner_chapter["number"], m.group("sep"),
                m.group("title"), m.group("suffix"))

        after_banner = BANNER_RE.sub(fix_banner, updated)
        if after_banner != updated:
            changed = True
            updated = after_banner
            reasons.append("the banner printed when the example runs names a "
                           "different chapter")

        results.append((path, current, wanted, title,
                        updated if changed else None,
                        "; ".join(reasons)))
    return results


# ----------------------------------------------------------------------
# README
# ----------------------------------------------------------------------

def render_chapter_index(data, by_number_examples):
    """The 'Index by book chapter' table, emitted from the map."""
    lines = [
        "| Book chapter | Run this | Needs |",
        "|---|---|---|",
    ]
    for chapter in data["chapters"]:
        entries = by_number_examples.get(chapter["number"])
        if not entries:
            continue
        scripts = " · ".join(
            "[`%s`](examples/%s)" % (name, name) for name, _ in sorted(entries))
        needs = " ".join(sorted({n for _, n in entries if n}))
        lines.append("| **%d** — %s | %s | %s |"
                     % (chapter["number"], chapter["title"], scripts,
                        needs or "○"))
    return "\n".join(lines)


def example_needs(path):
    """The legend symbol for what an example requires, from its REQUIRES line."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^REQUIRES:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return "○"
    req = match.group(1).lower()
    symbols = []
    if "database_url" in req or "postgres" in req or "requirements-db" in req:
        symbols.append("▣")
    # The examples name the requirements file, not the package: the header
    # reads "REQUIRES: -r requirements-local.txt (for real embeddings)".
    if ("requirements-local" in req or "sentence-transformers" in req
            or "sentence_transformers" in req):
        symbols.append("◐")
    if "openai_api_key" in req or "requirements-openai" in req:
        symbols.append("★")
    return " ".join(symbols) if symbols else "○"


def build_readme_block(data, by_title):
    by_number = {}
    for path in sorted(EXAMPLES.glob("[0-9][0-9]_*.py")):
        text = path.read_text(encoding="utf-8")
        match = HEADER_RE.search(text)
        if not match:
            continue
        chapter = resolve(match.group("title"), by_title)
        if chapter is None:
            continue
        by_number.setdefault(chapter["number"], set()).add(
            (path.name, example_needs(path)))
    return render_chapter_index(data, by_number)


def build_parts_block(data):
    """The book's part/chapter structure, as a table."""
    by_part = {}
    for chapter in data["chapters"]:
        by_part.setdefault(
            (chapter["part"], chapter["part_title"]), []).append(chapter)

    total = len(data["chapters"])
    lines = [
        "**%d chapters in %d parts.**" % (total, len(by_part)),
        "",
        "| Part | Chapters |",
        "|---|---|",
    ]
    for (number, title), chapters in by_part.items():
        listed = " · ".join("%d %s" % (c["number"], c["title"])
                            for c in chapters)
        lines.append("| **%s — %s** | %s |" % (number, title, listed))
    return "\n".join(lines)


def _replace_region(text, begin, end, wanted, label, write, path):
    """Swap the content between two markers. Returns a problem list."""
    if begin not in text or end not in text:
        return text, ["%s has no %s / %s markers — the %s is still "
                      "hand-maintained and will go stale."
                      % (path.name, begin, end, label)]
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    block = "\n" + wanted + "\n"
    if text[start:stop] == block:
        return text, []
    if write:
        return text[:start] + block + text[stop:], []
    return text, ["%s %s is stale" % (path.name, label)]


def sync_readme(data, by_title, write):
    if not README.exists():
        return []
    text = README.read_text(encoding="utf-8")
    problems = []

    text, found = _replace_region(
        text, BEGIN, END, build_readme_block(data, by_title),
        "chapter table", write, README)
    problems.extend(found)

    text, found = _replace_region(
        text, PARTS_BEGIN, PARTS_END, build_parts_block(data),
        "parts table", write, README)
    problems.extend(found)

    if write and not problems:
        README.write_text(text, encoding="utf-8")
    return problems


# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()

    data, by_title = load_map()
    results = scan_examples(by_title)

    problems = []
    changed = []

    for path, current, wanted, title, new_text, reason in results:
        rel = path.relative_to(ROOT)
        if title is None:
            problems.append("  %s: no 'Book: Chapter N, Title' header" % rel)
        elif wanted is None:
            problems.append(
                "  %s: cites %r, which is not a chapter in the book.\n"
                "      Either the chapter was renamed, or this header is wrong."
                % (rel, title))
        elif new_text is not None:
            if args.write:
                path.write_text(new_text, encoding="utf-8")
                changed.append("  %s: %s" % (rel, reason))
            else:
                problems.append("  %s: %s" % (rel, reason))

    problems.extend(sync_readme(data, by_title, args.write))

    if problems:
        print("\nReferences to the book are out of date:\n", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        print("\nFix with:  python scripts/sync_book_refs.py --write\n",
              file=sys.stderr)
        return 1

    if args.write:
        if changed:
            print("updated %d file(s):" % len(changed))
            print("\n".join(changed))
        else:
            print("nothing to change")
        return 0

    print("ok  %d example reference(s) and the README match the book"
          % len([r for r in results if r[3]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
