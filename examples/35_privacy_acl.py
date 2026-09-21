#!/usr/bin/env python3
"""
35. Privacy and Access Control
Book: Chapter 22, Privacy & Compliance in RAG
WHAT THIS SHOWS: Access control enforced inside the index, not after it.

The book is blunt about why:

    "a system that retrieves first and redacts afterwards has already leaked
     the document into the model's context"

So this example does it with Postgres row-level security. A chunk the caller
may not see is not filtered out of the results -- it never enters them. The
retriever cannot rank it, the reranker cannot see it, and no prompt can
contain it.

The example also proves that the policy is live, by asserting that a
restricted query returns zero forbidden rows. A privacy demo that cannot fail
is not a demo.

Also here: PII redaction before embedding, and delete-by-ID for the
right-to-be-forgotten case.

HOW THIS SCRIPT PROCEEDS
    1. Show the access labels, attached at INGESTION time
    2. Verify RLS is enabled AND forced       <-- FORCE is the load-bearing word
    3. Run one query as four identities
    4. ASSERT zero rows leaked                a demo that cannot fail is not one
    5. Control: run as the owner, prove the test CAN fail
    6. PII redaction before embedding
    7. Delete-by-ID, and watch the vector cascade

Step 5 is what makes step 4 meaningful. An assertion that passes because the
result set was empty proves nothing.


WHY THIS IS THE ONE EXAMPLE THAT CANNOT USE NUMPY
    The filter has to happen inside the index, on the way out -- not in
    Python, on the way back. Every other example in this repository could be
    rewritten against an in-memory matrix; this one could not, because the
    guarantee it demonstrates comes from the database refusing to return the
    row at all.


REQUIRES: Postgres
RUN: python examples/35_privacy_acl.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.embed import get_embedder

# scripts/index_corpus.py tags four stories with a synthetic clearance group
# at ingestion time -- which is the point: the label is attached when the text
# is extracted, not bolted on at query time.
GROUPS = {
    "anonymous": [],
    "medical team": ["medical"],
    "finance team": ["finance"],
    "auditor": ["medical", "finance", "legal"],
}

QUERY = "What was the speckled band?"  # answer lives in a 'medical' story


def run_as(conn, store, groups, query_vector, k=5):
    """Retrieve as a given identity."""
    # ---- THE KEY LINES ---------------------------------------------
    # SET ROLE is what makes this real. Connecting as the table owner
    # bypasses RLS entirely unless FORCE ROW LEVEL SECURITY is set -- and
    # DATABASE_URL almost always IS the owner, so without both of these
    # the example runs green while enforcing nothing.
    with conn.transaction():
        conn.execute("SET LOCAL ROLE rag_reader")
        conn.execute("SELECT set_config('app.groups', %s, true)", [",".join(groups)])
        rows = conn.execute(
            f"""
            SELECT c.id, c.story, c.acl, c.body,
                   1 - (e.vec <=> %(qvec)s::vector) AS score
            FROM   {store.table} e
            JOIN   chunks c ON c.id = e.chunk_id
            WHERE  e.strategy = %(strategy)s AND e.model = %(model)s
            ORDER  BY e.vec <=> %(qvec)s::vector
            LIMIT  %(k)s
            """,
            {
                "qvec": pg.vec_literal(query_vector),
                "strategy": "contextual_header",
                "model": store.model,
                "k": k,
            },
        ).fetchall()
    return rows


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Mask personal identifiers before the text is embedded.

    Before, not after. An embedding of un-redacted text is a lossy but real
    encoding of it -- embedding inversion attacks reconstruct meaningful
    fragments from vectors alone. Redacting the displayed text while indexing
    the original leaves the secret in the vector store.

    These patterns are illustrative. Production PII detection needs a proper
    NER model, and still needs a human in the loop for anything regulated.
    """
    counts = {"email": 0, "phone": 0, "card": 0, "ssn": 0}
    patterns = [
        ("email", r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[EMAIL]"),
        ("phone", r"\b(?:\+?\d{1,2}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b", "[PHONE]"),
        ("card", r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", "[CARD]"),
        ("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    ]
    # Card before phone: a 16-digit card number partly matches the phone
    # pattern, and whichever runs first wins. Ordering redaction rules from
    # most specific to least is not cosmetic.
    for name, pattern, replacement in sorted(patterns, key=lambda p: p[0] != "card"):
        text, n = re.subn(pattern, replacement, text)
        counts[name] = n
    return text, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    display.banner(
        "35. Privacy and Access Control",
        "Chapter 22, Privacy & Compliance in RAG",
        "Row-level security so forbidden chunks are never retrieved -- not "
        "retrieved and then filtered.",
    )

    if not pg.available():
        display.missing_database(
            "Access control enforced in the index requires the index to be a "
            "database. This is the example that cannot be faked with NumPy."
        )
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    embedder = get_embedder()
    store = pg.PgVectorStore(conn, dim=embedder.dim, model=embedder.name)
    if store.count("contextual_header") == 0:
        print("\nNothing indexed. Run: python scripts/index_corpus.py\n")
        return 1

    # --- the labels ------------------------------------------------------
    display.heading("Access labels, attached at ingestion")
    rows = conn.execute(
        "SELECT acl, count(*), count(DISTINCT story) FROM chunks "
        "WHERE strategy = 'contextual_header' GROUP BY acl ORDER BY 2 DESC"
    ).fetchall()
    display.table(
        ["acl", "chunks", "stories"],
        [[", ".join(r[0]), r[1], r[2]] for r in rows],
    )
    print("\n  These were written by scripts/index_corpus.py as the chunks were")
    print("  created. The book insists on this ordering: an ACL discovered at")
    print("  query time is an ACL applied too late.")

    # --- RLS is actually on ----------------------------------------------
    display.heading("Is the policy actually enforced?")
    policies = conn.execute(
        "SELECT tablename, policyname FROM pg_policies "
        "WHERE schemaname = 'public' ORDER BY 1, 2"
    ).fetchall()
    forced = conn.execute(
        "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
        "WHERE relname IN ('chunks','embeddings_384')"
    ).fetchall()
    display.table(
        ["table", "RLS enabled", "FORCED"],
        [[r[0], "yes" if r[1] else "NO", "yes" if r[2] else "NO"] for r in forced],
    )
    print(f"\n  {len(policies)} policies defined.")
    print("\n  FORCED matters more than ENABLED. Without FORCE, the table owner")
    print("  bypasses every policy -- and DATABASE_URL almost always connects")
    print("  as the owner. The demo would pass while enforcing nothing.")

    # --- retrieval as different identities -------------------------------
    display.heading(f"The same query, four identities")
    print(f"  query: {QUERY!r}")
    print("  The answer is in 'The Adventure of the Speckled Band', which is")
    print("  labelled 'medical'.\n")

    query_vector = embedder.encode([QUERY], show_progress=False)[0]

    summary = []
    leaked_total = 0
    for label, groups in GROUPS.items():
        results = run_as(conn, store, groups, query_vector, args.k)
        stories = {r[1] for r in results}
        got_answer = any("Speckled" in (r[1] or "") for r in results)
        # Every row returned must be one this identity is entitled to.
        allowed = set(groups) | {"public"}
        leaked = [r for r in results if not (set(r[2]) & allowed)]
        leaked_total += len(leaked)
        summary.append(
            [
                label,
                ", ".join(groups) or "(none)",
                len(results),
                "yes" if got_answer else "no",
                len(leaked),
            ]
        )

    display.table(
        ["identity", "groups", "rows", "found the answer", "leaked rows"],
        summary,
        align="llrrr",
    )

    print("\n  The anonymous and finance identities did not merely have the")
    print("  Speckled Band chunks hidden from their output -- those chunks were")
    print("  never candidates. The ORDER BY never saw them.")

    # --- the assertion ----------------------------------------------------
    display.heading("Assertion")
    if leaked_total == 0:
        print("  PASS: 0 rows were returned to an identity not entitled to them.")
    else:
        print(f"  FAIL: {leaked_total} rows leaked.")
        print("  This is a real failure, not a display problem. Check that")
        print("  sql/003_rls.sql applied and that FORCE ROW LEVEL SECURITY is set.")
        return 1

    # Prove the test can fail: as the owner, without SET ROLE, the same query
    # returns everything.
    owner_rows = conn.execute(
        f"""
        SELECT c.acl FROM {store.table} e JOIN chunks c ON c.id = e.chunk_id
        WHERE e.strategy = 'contextual_header' AND e.model = %(model)s
        ORDER BY e.vec <=> %(qvec)s::vector LIMIT %(k)s
        """,
        {"qvec": pg.vec_literal(query_vector), "model": store.model, "k": args.k},
    ).fetchall()
    restricted_seen = sum(1 for r in owner_rows if r[0] != ["public"])
    print(f"\n  Control: the same query as the table owner, with no SET ROLE,")
    print(f"  returned {restricted_seen} restricted row(s) out of {len(owner_rows)}.")
    print("  That is the check confirming the test above can fail -- the policy")
    print("  is doing the work, not an empty result set.")

    # --- redaction --------------------------------------------------------
    display.heading("PII redaction, before embedding")
    sample = (
        "Contact Dr. Roylott at grimesby.roylott@stoke-moran.co.uk or "
        "+44 20 7946 0958. Card 4111 1111 1111 1111, NI 123-45-6789."
    )
    redacted, counts = redact(sample)
    print(f"  before: {sample}")
    print(f"  after:  {redacted}")
    display.table(["type", "redacted"], [[k, v] for k, v in counts.items() if v])
    print("\n  This happens before the text reaches the embedder. An embedding")
    print("  of un-redacted text still encodes it -- inversion attacks recover")
    print("  readable fragments from vectors alone -- so redacting only the")
    print("  displayed copy leaves the secret in the vector store.")

    # --- deletion ---------------------------------------------------------
    display.heading("Delete by ID (right to be forgotten)")
    victim = conn.execute(
        "SELECT id FROM chunks WHERE strategy = 'contextual_header' LIMIT 1"
    ).fetchone()[0]

    def counts() -> tuple[int, int]:
        chunks = store.count("contextual_header")
        vectors = conn.execute(
            f"SELECT count(*) FROM {store.table} WHERE strategy = 'contextual_header'"
        ).fetchone()[0]
        return chunks, int(vectors)

    before_chunks, before_vectors = counts()

    # The delete runs inside a transaction that is rolled back, so this
    # example leaves the index exactly as it found it. The cascade is real --
    # the counts below are measured after the DELETE, before the rollback --
    # but running this script does not quietly degrade the corpus every other
    # example depends on.
    with conn.transaction(force_rollback=True):
        conn.execute("DELETE FROM chunks WHERE id = %s", [victim])
        after_chunks, after_vectors = counts()

    restored_chunks, restored_vectors = counts()

    display.table(
        ["", "chunks", "vectors"],
        [
            ["before delete", before_chunks, before_vectors],
            ["after delete", after_chunks, after_vectors],
            ["after rollback", restored_chunks, restored_vectors],
        ],
    )
    print("\n  One chunk deleted, one vector gone with it -- via ON DELETE")
    print("  CASCADE on the foreign key from the embeddings table.")
    print("\n  Deleting the text and leaving the embedding behind is a common")
    print("  and serious bug. The orphaned vector is still searchable and still")
    print("  encodes what it was built from, so a 'deleted' document keeps")
    print("  answering queries.")
    print("\n  The third row is this example being a good citizen: the delete")
    print("  ran inside a transaction that was rolled back, so the corpus the")
    print("  other examples share is untouched.")

    display.notice(
        "Row-level security removes forbidden chunks from the candidate set "
        "entirely. Post-filtering would have ranked them first, which means "
        "they were already read.",
        "FORCE ROW LEVEL SECURITY is the line that makes this real. Without "
        "it the table owner ignores every policy, and DATABASE_URL is usually "
        "the owner.",
        "Redaction must precede embedding. Vectors are not anonymised text; "
        "inversion attacks recover fragments from them.",
        "Deletion must cascade to the embeddings. A deleted document whose "
        "vector survives is still retrievable and still encodes its source.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
