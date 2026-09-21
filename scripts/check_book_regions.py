#!/usr/bin/env python3
"""
Check that the regions printed in the book still say what the book says.

    python scripts/check_book_regions.py

Some lines in this repo are quoted verbatim in *RAG: The Definitive Guide*.
They are marked with `book:<tag>` comments and recorded, with hashes, in
docs/book_snippets.lock.json.

This runs in CI so that a change to quoted code fails the pull request that
causes it, rather than months later when someone rebuilds the book. It needs
no dependencies and takes about a second.

If a failure here is intentional -- you improved the code, and the book should
follow -- that is fine. Say so in the PR; the book repo regenerates its
manifest with:

    python3 verify_snippets.py --refresh --repo <this repo>

The column budget is not recomputed here: it comes from the lock file, because
deriving it needs the book's fonts and page geometry, which this repo does not
have and should not grow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import book_regions as br           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "docs" / "book_snippets.lock.json"


def main() -> int:
    if not LOCK.exists():
        print("No %s — nothing is quoted in the book yet." % LOCK.name)
        return 0

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected = lock.get("snippets", {})
    budget = lock.get("column_budget", 76)

    try:
        regions, files = br.parse_tree(ROOT)
    except br.RegionError as exc:
        print("Marker error:\n  %s" % exc, file=sys.stderr)
        return 1

    failures = []

    for tag, want in sorted(expected.items()):
        region = regions.get(tag)
        if region is None:
            failures.append(
                "  book:%s has been removed from %s, but the book still "
                "prints it." % (tag, want["path"]))
            continue
        if region.path != want["path"]:
            failures.append(
                "  book:%s moved from %s to %s — the book's link still points "
                "at the old path." % (tag, want["path"], region.path))
            continue
        if region.sha256() != want["sha256"]:
            failures.append(
                "  book:%s CHANGED in %s.\n"
                "     The book prints these lines verbatim. If the change is\n"
                "     intentional, the book's manifest needs regenerating.\n"
                "     Now reads:\n%s"
                % (tag, region.path,
                   "\n".join("       " + l for l in region.text().split("\n"))))
            continue

        parts = [{"start": s, "end": e} for s, e in region.parts]
        if parts != want.get("parts"):
            failures.append(
                "  book:%s still matches, but has moved within %s "
                "(%s -> %s). The printed line anchor is now wrong."
                % (tag, region.path,
                   _span(want.get("parts", [])), _span(parts)))
            continue

        for offset, line in enumerate(region.text().split("\n")):
            if len(line) > budget:
                failures.append(
                    "  book:%s line %d is %d columns (book prints at most %d):\n"
                    "       %s" % (tag, offset + 1, len(line), budget, line))

    if failures:
        print("\nQuoted-in-the-book regions are out of sync:\n", file=sys.stderr)
        print("\n\n".join(failures), file=sys.stderr)
        print("\nSee CONTRIBUTING.md, 'Code that appears in the book'.\n",
              file=sys.stderr)
        return 1

    print("ok  %d region(s) match what the book prints" % len(expected))
    return 0


def _span(parts):
    if not parts:
        return "?"
    return "L%d-L%d" % (parts[0]["start"], parts[-1]["end"])


if __name__ == "__main__":
    sys.exit(main())
