# Contributing

This repo is a teaching artifact attached to a book, so the bar is different from a
library: **a change is good if it makes a technique easier to understand, and bad if it
makes the repo cleverer.**

Three kinds of contribution are especially welcome, in descending order of how much we
want them.

---

## 1. Results from your own corpus

The most useful thing you can send. Every number in the README comes from 30 questions
about one book of Victorian detective fiction, and the README says so. Whether the
chunking ranking survives contact with technical documentation, support tickets or legal
contracts is an open question.

Open an issue using the **Results on my corpus** template with:

- what the corpus is (roughly — no need to share it)
- the output of `python examples/15_compare_chunking.py`
- your embedder and, if relevant, your model

A result that contradicts the tables is more interesting than one that confirms them.

**Fair warning:** there is no clean seam for this yet. `ragkit/corpus.py` hardcodes the
corpus path and a Roman-numeral heading grammar, and `data/eval_questions.json` is
Holmes-specific, so today this means editing `corpus.py` and writing your own gold set
with `scripts/build_eval_set.py` as the template. Making that a supported path is the next
planned change; if you are attempting it, say so in an issue and we will coordinate.

## 2. A new strategy

One new file in `examples/`, numbered after the last one. Follow the shape every other
example uses — it is the whole convention:

```python
"""
39. Short title of the technique

Book: Chapter N, Chapter Title

WHAT THIS SHOWS
    One paragraph. What the reader will understand after running it.

HOW THIS SCRIPT PROCEEDS
    1. ...
    2. ...

REQUIRES: DATABASE_URL          (or: nothing / OPENAI_API_KEY / sentence-transformers)
RUN:      python examples/39_your_thing.py
"""
```

and end with a `WHAT TO NOTICE` block that ties the printed output back to a claim.

**The house rule:** `ragkit/` holds plumbing — corpus loading, embedding, Postgres,
table formatting. It holds **no retrieval techniques**. If a reader has to open `ragkit/`
to understand how your method works, the method is in the wrong file. Duplicating twenty
lines across two examples is correct here; factoring them into a shared helper is not.

Other expectations:

- **Degrade, never crash.** If the example needs something the reader has not installed,
  print what to install and `return` — do not raise. `scripts/smoke_test.py` enforces this;
  the phrases it recognises are in `SKIP_MARKERS`.
- **Take arguments.** `argparse` with a positional query and `--k` where it makes sense,
  matching the neighbours.
- **Print real numbers.** Measured output beats a claim in a comment.
- Run `python scripts/smoke_test.py --only 39` before opening the PR.

## 3. Breaking a claim

If a number here is wrong, or an example crashes on your machine, open an issue with the
output. The Known Limits section exists because these results are model-, embedder- and
corpus-specific — a correction is a contribution, not a complaint.

---

## Code that appears in the book

Some regions of this repo are printed verbatim in *RAG: The Definitive Guide*. They are
marked in place:

```sql
-- --- both arms, keyed by rank rather than score ------------ book:ch14-rrf
fused AS (
    SELECT chunk_id, v.vec_score, t.ts_score, v.vec_rank, t.txt_rank
    FROM   vec v FULL OUTER JOIN txt t USING (chunk_id)
)
-- ------------------------------------------------------------------ /book
```

Two consequences if you edit code inside a marked region:

- **Lines are capped at 76 columns.** That is the printed measure, and a book cannot
  scroll sideways. `make book-check` tells you which line and by how much.
- **Changing the code changes the book.** CI fails the PR, on purpose. That is not a veto
  — if the code should change, change it and say so in the PR. The book's manifest is
  regenerated from your branch; it is a handoff, not an argument.

Run `make book-check` before opening a PR that touches a marked file. It needs no
dependencies.

Moving a marked region within its file is fine; the tooling regenerates the printed line
anchor. Deleting a marker is not — the book would still be quoting it.

## Running the tests

There is no unit test suite. The smoke test is the test:

```bash
python scripts/smoke_test.py --no-db      # as a reader with core deps only
python scripts/smoke_test.py              # everything you have configured
python scripts/smoke_test.py --stub-llm --fail-on-skip   # what CI runs
```

`--fail-on-skip` turns clean declines into failures. Use it only when you have a database
and an indexed corpus, since otherwise the skips are correct behaviour.

## Style

Nothing enforced, nothing to install. Match the file you are editing: 4-space indent,
`from __future__ import annotations` at the top, comments that explain *why* rather than
restating the line below. Prose in comments is a feature of this repo, not clutter.
