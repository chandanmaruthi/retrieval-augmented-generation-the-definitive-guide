#!/usr/bin/env python3
"""
Extract the regions of this repo that are printed in the book.

A region is a contiguous run of real source, delimited by comment markers:

    # --- fuse on rank, not on score ---------------------- book:ch14-rrf
    fused = defaultdict(float)
    # -------------------------------------------------------------- /book

The opening line is the section banner these files already use; the tag rides
along as the right-hand filler where the dashes were. Net cost over the status
quo is one line per snippet, and the tag doubles as something the printed book
can tell a reader to search for -- a content anchor that survives the file
moving, which no line number does.

Elisions are expressed as SEVERAL regions sharing one tag, never as typed
dots. The gap between them is measured here and rendered as

    # ... 14 lines omitted ...

so the count cannot drift from the source. Whole lines only: a mid-line "..."
reads as real code and is not.

This module is the single implementation, living next to the files it parses.
Both repos call it -- the book repo loads it by path -- so the book and CI can
never disagree about what a region contains.

Stdlib only, and no f-strings in the parsing hot path that would break on the
book repo's Python 3.9.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# `book:tag` as the last thing on a comment line. The tag charset is
# deliberately narrow so a tag is safe as a JSON key and a URL fragment.
OPEN_RE = re.compile(r"(?:#|--|//)\s*.*?\bbook:([a-z0-9][a-z0-9._-]*)\s*$")

# The closing rule. Accepts the banner form and a bare comment.
CLOSE_RE = re.compile(r"(?:#|--|//)\s*-*\s*/book\s*-*\s*$")

MAX_LINES = 15      # a printed snippet longer than this stops being a snippet
MAX_PARTS = 3       # more than two elisions and the reader has lost the thread


class RegionError(ValueError):
    """A marker problem in a source file. Always fatal -- never skipped."""


# The elision note is written in the comment syntax of the language it sits
# in, so a snippet pasted into an editor is still syntactically that language
# -- visibly incomplete rather than subtly broken.
COMMENT_PREFIX = {".py": "#", ".sql": "--"}


class Region:
    """One tag's worth of source: one or more parts, in file order."""

    def __init__(self, tag, path):
        self.tag = tag
        self.path = path
        self.parts = []          # list of (start_line, end_line), 1-based, inclusive
        self.raw = []            # list of list-of-str, parallel to parts

    @property
    def comment(self):
        for suffix, prefix in COMMENT_PREFIX.items():
            if self.path.endswith(suffix):
                return prefix
        return "#"

    # -- geometry ---------------------------------------------------------

    @property
    def anchor(self):
        """The GitHub line anchor: the full span the snippet is drawn from."""
        first = self.parts[0][0]
        last = self.parts[-1][1]
        return "L%d-L%d" % (first, last) if last != first else "L%d" % first

    def gaps(self):
        """Real source lines skipped between consecutive parts."""
        out = []
        for (prev, nxt) in zip(self.parts, self.parts[1:]):
            # Between the previous part's last content line and the next
            # part's first, minus the two marker lines, which are scaffolding
            # rather than code the reader is missing.
            out.append(nxt[0] - prev[1] - 1 - 2)
        return out

    # -- text -------------------------------------------------------------

    def text(self):
        """
        The printable snippet: parts joined by a measured elision note,
        with the common indent removed once across the whole thing.
        """
        flat = []
        for lines in self.raw:
            flat.extend(lines)
        indent = _common_indent(flat)

        out = []
        gaps = self.gaps()
        for i, lines in enumerate(self.raw):
            for line in lines:
                out.append(line[indent:] if line.strip() else "")
            if i < len(gaps):
                n = gaps[i]
                plural = "line" if n == 1 else "lines"
                out.append("%s ... %d %s omitted ..." % (self.comment, n, plural))
        return "\n".join(out)

    def dedent(self):
        flat = []
        for lines in self.raw:
            flat.extend(lines)
        return _common_indent(flat)

    def sha256(self):
        return hashlib.sha256(self.text().encode("utf-8")).hexdigest()

    def max_cols(self):
        return max((len(l) for l in self.text().split("\n")), default=0)

    def to_entry(self, file_sha):
        """The manifest record. `lines` is an array so drift shows as a diff."""
        return {
            "path": self.path,
            "parts": [{"start": s, "end": e} for s, e in self.parts],
            "anchor": self.anchor,
            "dedent": self.dedent(),
            "omitted": self.gaps(),
            "lines": self.text().split("\n"),
            "max_cols": self.max_cols(),
            "sha256": self.sha256(),
            "file_sha256": file_sha,
        }


def _common_indent(lines):
    """Leading spaces shared by every non-blank line."""
    widths = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    return min(widths) if widths else 0


def parse_file(path, rel=None):
    """
    Extract every region in one file.

    Returns {tag: Region}. Raises RegionError on an unterminated region, a
    stray close, or a nested open -- loudly, because the alternative is a
    silently truncated snippet in a printed book.
    """
    path = Path(path)
    rel = rel or path.name
    text = path.read_text(encoding="utf-8")
    file_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    regions = {}
    open_tag = None
    start = None
    buf = []

    for no, line in enumerate(text.split("\n"), start=1):
        close = CLOSE_RE.search(line)
        if close and open_tag:
            regions.setdefault(open_tag, Region(open_tag, rel))
            regions[open_tag].parts.append((start, no - 1))
            regions[open_tag].raw.append(buf)
            open_tag, start, buf = None, None, []
            continue
        if close and not open_tag:
            raise RegionError("%s:%d: /book with no open marker" % (rel, no))

        opened = OPEN_RE.search(line)
        if opened:
            if open_tag:
                raise RegionError(
                    "%s:%d: book:%s opened inside book:%s -- regions cannot nest"
                    % (rel, no, opened.group(1), open_tag))
            open_tag = opened.group(1)
            start = no + 1
            buf = []
            continue

        if open_tag:
            buf.append(line)

    if open_tag:
        raise RegionError(
            "%s: book:%s is never closed" % (rel, open_tag))

    for tag, region in regions.items():
        if not any(l.strip() for part in region.raw for l in part):
            raise RegionError("%s: book:%s is empty" % (rel, tag))
        if len(region.parts) > MAX_PARTS:
            raise RegionError(
                "%s: book:%s has %d parts (max %d) -- quote less, or quote a "
                "different place" % (rel, tag, len(region.parts), MAX_PARTS))
        n = len(region.text().split("\n"))
        if n > MAX_LINES:
            raise RegionError(
                "%s: book:%s is %d printed lines (max %d)"
                % (rel, tag, n, MAX_LINES))

    return regions, file_sha


# The tooling that implements this format documents it, and the parser is
# line-based, so it cannot tell a marker in a docstring from a real one. These
# files describe the format rather than contributing to it.
SELF_EXCLUDE = {
    "scripts/book_regions.py",
    "scripts/check_book_regions.py",
}


def parse_tree(root, subdirs=("examples", "scripts", "sql", "ragkit")):
    """
    Every region in the repo, keyed by tag.

    Raises RegionError if one tag is claimed by two different files -- the tag
    is what the book cites, so it has to be unique across the repo.
    """
    root = Path(root)
    found = {}
    files = {}
    for sub in subdirs:
        base = root / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in (".py", ".sql"):
                continue
            rel = str(path.relative_to(root))
            if rel in SELF_EXCLUDE:
                continue
            regions, file_sha = parse_file(path, rel)
            for tag, region in regions.items():
                if tag in found:
                    raise RegionError(
                        "tag book:%s claimed by both %s and %s"
                        % (tag, found[tag].path, rel))
                found[tag] = region
                files[rel] = file_sha
    return found, files
