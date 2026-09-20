"""
Terminal output shared by all thirty-eight example scripts.

The examples are meant to be *read* as much as run, so their output follows
one shape: a banner saying which chapter the technique comes from, the work
itself, then a short "what to notice" block connecting the output back to the
book's claim. Centralising that here keeps the formatting out of the example
scripts, where it would compete with the technique for the reader's attention.
"""

from __future__ import annotations

import shutil
import textwrap

WIDTH = min(shutil.get_terminal_size((88, 24)).columns, 88)


def banner(title: str, chapter: str, what: str) -> None:
    """Open a script: what this is, where it comes from in the book."""
    print()
    print("=" * WIDTH)
    print(title)
    print(f"Book: {chapter}")
    print("=" * WIDTH)
    for line in textwrap.wrap(what, WIDTH):
        print(line)
    print()


def rule(label: str = "") -> None:
    if label:
        print(f"\n--- {label} " + "-" * max(0, WIDTH - len(label) - 6))
    else:
        print("-" * WIDTH)


def heading(text: str) -> None:
    print(f"\n{text}")
    print("-" * len(text))


def para(text: str, indent: str = "") -> None:
    for line in textwrap.wrap(text, WIDTH - len(indent)):
        print(indent + line)


def kv(key: str, value, width: int = 26) -> None:
    print(f"  {key:<{width}} {value}")


def truncate(text: str, limit: int = 160) -> str:
    """Single-line preview of a passage."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def preview(text: str, limit: int = 160, indent: str = "      ") -> None:
    print(indent + truncate(text, limit))


def table(headers: list[str], rows: list[list], align: str | None = None) -> None:
    """Print a plain aligned table.

    ``align`` is a string of 'l'/'r' per column; the default left-aligns the
    first column and right-aligns the rest, which is what every table in this
    repo wants.
    """
    cells = [[str(c) for c in row] for row in rows]
    columns = len(headers)
    if align is None:
        align = "l" + "r" * (columns - 1)

    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row[:columns]):
            widths[i] = max(widths[i], len(cell))

    def render(values: list[str]) -> str:
        out = []
        for i, value in enumerate(values[:columns]):
            out.append(value.ljust(widths[i]) if align[i] == "l" else value.rjust(widths[i]))
        return "  ".join(out)

    print("  " + render(headers))
    print("  " + "  ".join("-" * w for w in widths))
    for row in cells:
        print("  " + render(row))


def sample_chunks(chunks, n: int = 4, start: int = 0, count_tokens=None) -> None:
    """Print n consecutive chunks so the reader can see what the strategy did.

    Every chunking example ends with this, in the same shape, because the
    point of thirteen strategies is comparison -- and comparison is much
    easier when the output looks identical apart from the chunks themselves.

    Consecutive, not scattered: the interesting thing about a chunking
    strategy is where one chunk STOPS and the next STARTS, and you cannot see
    that from four chunks picked at random. Each one shows its opening and its
    closing words, so the boundary between them is visible on the page.
    """
    window = chunks[start : start + n]
    if not window:
        return

    heading(f"{len(window)} consecutive chunks, as this strategy produced them")

    for position, chunk in enumerate(window):
        flat = " ".join(chunk.text.split())
        size = f"{count_tokens(chunk.text)} tokens" if count_tokens else f"{len(flat)} chars"

        label = f"[{start + position}]"
        where = f"  {chunk.location}" if getattr(chunk, "location", "") else ""
        print(f"\n  {label} {size}{where}")

        # Head and tail, so the boundary is visible. A chunk short enough to
        # print whole is printed whole rather than faked into two halves.
        if len(flat) <= 300:
            for line in textwrap.wrap(flat, WIDTH - 8):
                print(f"        {line}")
        else:
            for line in textwrap.wrap(flat[:150].rstrip(), WIDTH - 8):
                print(f"        {line}")
            print(f"        {'.' * 12}")
            for line in textwrap.wrap("..." + flat[-150:].lstrip(), WIDTH - 8):
                print(f"        {line}")

        # Only shown when the strategy actually set them, so the reader sees
        # at a glance which strategies index or return something other than
        # the chunk body.
        if getattr(chunk, "retrieval_text", None) and chunk.retrieval_text != chunk.text:
            print(f"        EMBEDDED AS: {truncate(chunk.retrieval_text, WIDTH - 22)}")
        if getattr(chunk, "parent_text", None) and chunk.parent_text != chunk.text:
            extra = (count_tokens(chunk.parent_text) if count_tokens else 0)
            suffix = f" ({extra} tokens)" if extra else ""
            print(f"        RETURNS PARENT{suffix}: {truncate(chunk.parent_text, WIDTH - 30)}")

    print("\n  Read down the right-hand edge of one chunk and the left edge of")
    print("  the next: that gap is the boundary this strategy chose.")


def notice(*lines: str) -> None:
    """Close a script by tying its output back to the book."""
    print()
    print("=" * WIDTH)
    print("WHAT TO NOTICE")
    print("=" * WIDTH)
    for line in lines:
        for wrapped in textwrap.wrap(line, WIDTH - 2, initial_indent="* ", subsequent_indent="  "):
            print(wrapped)
    print()


def missing_dependency(package: str, requirements_file: str, purpose: str) -> None:
    """Explain an absent optional dependency instead of raising a traceback.

    Called by scripts that need torch or the OpenAI client. A reader who has
    only run ``pip install -r requirements.txt`` should be told exactly what
    to type, not shown an ImportError.
    """
    print()
    print("=" * WIDTH)
    print(f"This example needs {package}, which is not installed.")
    print("=" * WIDTH)
    para(purpose)
    print()
    print(f"  pip install -r {requirements_file}")
    print()


def missing_api_key(purpose: str) -> None:
    """Explain an absent OPENAI_API_KEY without failing."""
    print()
    print("=" * WIDTH)
    print("This example needs OPENAI_API_KEY, which is not set.")
    print("=" * WIDTH)
    para(purpose)
    print()
    print("  cp .env.example .env     # then add your key")
    print()


def prompt_block(label: str, text: str) -> None:
    """Show a prompt exactly as it would be sent to the model.

    The examples that need an API key print this when none is set. The prompt
    is most of what there is to learn about an LLM-assisted chunking strategy
    -- the rest is a loop and a JSON parse -- so a reader without a key still
    gets the substance.
    """
    print(f"\n  {label}")
    print("  " + "─" * (WIDTH - 4))
    for raw_line in text.strip().split("\n"):
        if not raw_line.strip():
            print("  │")
            continue
        for line in textwrap.wrap(raw_line, WIDTH - 6) or [""]:
            print(f"  │ {line}")
    print("  " + "─" * (WIDTH - 4))


def stub_warning() -> None:
    """Flag that RAGKIT_STUB_LLM is on, so no output below is a real result."""
    print()
    print("!" * WIDTH)
    print("RAGKIT_STUB_LLM=1 -- model calls are returning canned filler.")
    print("The mechanics below are real; the generated text is meaningless.")
    print("!" * WIDTH)


def illustrative(*lines: str) -> None:
    """Print sample output that was NOT produced by a live model.

    Labelled unmistakably. A reader without an API key should never be left
    wondering whether what they are looking at came from a real call.
    """
    print("\n  ILLUSTRATIVE OUTPUT -- hand-written, not from a live model call:")
    for line in lines:
        print(f"      {line}")


def missing_database(purpose: str) -> None:
    """Explain an absent DATABASE_URL without failing."""
    print()
    print("=" * WIDTH)
    print("This example needs Postgres, and DATABASE_URL is not set.")
    print("=" * WIDTH)
    para(purpose)
    print()
    print("  docker compose up -d")
    print("  python scripts/init_db.py")
    print()
