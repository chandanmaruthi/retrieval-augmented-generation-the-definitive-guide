"""
Configuration, read from the environment and an optional .env file.

A deliberately tiny .env reader rather than a python-dotenv dependency: it is
fifteen lines, and keeping requirements.txt short is a feature of this repo.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache"


def _load_dotenv() -> None:
    """Populate os.environ from .env, without overriding real env vars.

    Within the file, the LAST assignment of a key wins. That matches what
    people expect when they paste a corrected value at the bottom of the file
    rather than editing the line above -- and the opposite rule produces the
    worst possible symptom, where an edit is saved, is visibly present in the
    file, and silently does nothing.
    """
    path = REPO_ROOT / ".env"
    if not path.exists():
        return

    from_file: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if value:
            from_file[key] = value  # later lines overwrite earlier ones

    for key, value in from_file.items():
        # A real environment variable still wins over the file, so that
        # `DATABASE_URL=... python script.py` works for one-off overrides.
        if key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value or default


DATABASE_URL = get("DATABASE_URL")
EMBEDDING_PROVIDER = get("EMBEDDING_PROVIDER", "local")
OPENAI_API_KEY = get("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = get("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def has_database() -> bool:
    return bool(DATABASE_URL)


def has_openai() -> bool:
    return bool(OPENAI_API_KEY)
