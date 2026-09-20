"""
ragkit — the shared plumbing behind the examples in this repository.

This package deliberately contains no retrieval techniques. It loads the
corpus, embeds text, talks to Postgres and formats terminal output; every
method the book teaches is implemented in the open, in examples/, where you
can read it.

The rule: if you would have to open ragkit/ to understand how a technique
works, the technique is in the wrong place.
"""

__all__ = ["config", "corpus", "display", "embed", "store", "tokens"]
