"""
Token counting.

The book's fixed-length strategy (§3.1) specifies chunk sizes "in tokens,
using the target model's tokenizer", and that qualifier matters: a chunk sized
with the wrong tokenizer will overflow or underfill the real context window.

So we use the real thing when it is installed, and a documented approximation
when it is not, rather than silently pretending word counts are token counts.
"""

from __future__ import annotations

import re

# Average tokens per whitespace-delimited word for English prose under a
# BPE vocabulary. Measured against tiktoken's cl100k_base on this corpus:
# 104,346 words -> ~138,000 tokens.
_TOKENS_PER_WORD = 1.32

_WORD = re.compile(r"\S+")


def _load_tiktoken():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Not installed, or installed but unable to fetch its BPE files
        # (they download on first use). Either way, fall back.
        return None


_ENCODING = _load_tiktoken()
USING_REAL_TOKENIZER = _ENCODING is not None


def count_tokens(text: str) -> int:
    """Number of tokens in ``text``."""
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, round(len(_WORD.findall(text)) * _TOKENS_PER_WORD))


def encode(text: str) -> list[int | str]:
    """Split ``text`` into token units.

    Returns real token ids under tiktoken, or word strings under the
    approximation. Callers only ever count these and slice them, never
    interpret them, so the two are interchangeable for our purposes.
    """
    if _ENCODING is not None:
        return _ENCODING.encode(text)
    return _WORD.findall(text)


def decode(tokens: list[int | str]) -> str:
    """Inverse of :func:`encode`."""
    if _ENCODING is not None:
        return _ENCODING.decode(tokens)
    return " ".join(tokens)


def units_for_tokens(n_tokens: int) -> int:
    """How many :func:`encode` units add up to roughly ``n_tokens`` tokens.

    Under tiktoken a unit *is* a token, so this is the identity. Under the
    fallback a unit is a word, and a word is worth more than a token, so
    asking for 256 units would hand back about 338 tokens.

    Without this conversion a strategy configured for a 256-token budget
    quietly overruns the embedding model's window, and the tail of every
    chunk is truncated before it is ever embedded.
    """
    if _ENCODING is not None:
        return n_tokens
    return max(1, round(n_tokens / _TOKENS_PER_WORD))


def tokenizer_note() -> str:
    """One line for a script to print, so its numbers are never ambiguous."""
    if USING_REAL_TOKENIZER:
        return "token counts: tiktoken cl100k_base (exact)"
    return (
        f"token counts: approximated at {_TOKENS_PER_WORD} tokens/word "
        "(pip install tiktoken for exact counts)"
    )
