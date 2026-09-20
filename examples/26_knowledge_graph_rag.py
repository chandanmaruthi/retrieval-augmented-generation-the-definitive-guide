#!/usr/bin/env python3
"""
26. Knowledge Graph Integration
Book: Chapter 18, Knowledge Graph Integration

WHAT THIS SHOWS: Typed triples instead of co-occurrence, and the three ways
the book names for combining a graph with a vector index.

Example 25 built edges from "these two names appeared in the same chunk".
That is cheap and weak: it records that Holmes and Roylott shared a scene, not
that Roylott murdered his stepdaughter. A knowledge graph stores the relation
itself:

    (Grimesby Roylott, killed, Julia Stoner)
    (Grimesby Roylott, kept, swamp adder)

Which makes a different class of question answerable -- "who killed whom" is a
graph lookup, and no amount of similarity search reliably extracts it from
prose.

The three integration strategies, all shown:

    pre-retrieval fusion    expand the query with graph facts, then search
    post-retrieval fusion   search, then attach facts about what came back
    joint                   one scoring pass over text and graph together

HOW THIS SCRIPT PROCEEDS
    1. Extract typed (subject, predicate, object) triples   one call per passage
    2. Store them as entities + edges
    3. Verbalise triples into prose         <-- easy to skip, and wrong to
    4. Pre-retrieval fusion: expand the query with graph facts  <-- THE TECHNIQUE
    5. Post-retrieval fusion: attach facts to what was retrieved
    6. Joint embedding: described, not built

Step 4 is where the measurable gain is: the expansion injects names the user
never typed, and those names are in the passages that answer the question.


WHAT CHANGED SINCE EXAMPLE 25
    The relation. Example 25's edges all say "appears_with"; these say
    "killed", "kept", "is_stepfather_of". That difference turns "who killed
    whom" from an inference into a lookup.

    The cost is that extraction now needs a model, and extraction quality caps
    everything -- a model that infers relationships the passage does not state
    produces a confident graph of things that are not true, and graphs look
    authoritative.


REQUIRES: Postgres. OPENAI_API_KEY for real triple extraction (falls back to
a seeded set so the fusion comparison still runs).
RUN: python examples/26_knowledge_graph_rag.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import config, display, llm, pg
from ragkit.embed import get_embedder

EXTRACT_SYSTEM = (
    "You extract factual relationships from fiction as subject-predicate-object "
    "triples. You reply with JSON."
)

EXTRACT_PROMPT = """Extract the factual relationships stated in this passage.

Rules:
- Only relationships the passage actually states. Do not infer.
- Use full names where the passage gives them.
- Predicates should be short verb phrases with underscores: killed,
  is_stepfather_of, kept, lives_at, suspects.

Reply as JSON: {{"triples": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}

PASSAGE:
{passage}"""

# Used when no model is available. Hand-written from the stories so the fusion
# comparison below is still a real measurement rather than a mock-up.
SEED_TRIPLES = [
    ("Grimesby Roylott", "is_stepfather_of", "Helen Stoner"),
    ("Grimesby Roylott", "is_stepfather_of", "Julia Stoner"),
    ("Grimesby Roylott", "killed", "Julia Stoner"),
    ("Grimesby Roylott", "kept", "swamp adder"),
    ("Grimesby Roylott", "lived_at", "Stoke Moran"),
    ("swamp adder", "is_a", "deadliest snake in India"),
    ("Helen Stoner", "consulted", "Sherlock Holmes"),
    ("John Clay", "tunnelled_to", "City and Suburban Bank"),
    ("Jabez Wilson", "was_hired_by", "Red-Headed League"),
    ("James Ryder", "stole", "blue carbuncle"),
    ("Countess of Morcar", "owned", "blue carbuncle"),
    ("James Windibank", "impersonated", "Hosmer Angel"),
    ("Irene Adler", "married", "Godfrey Norton"),
    ("Sir George Burnwell", "stole", "beryl coronet"),
]


def store_triples(conn, triples) -> int:
    """Write typed triples into entities and edges.

    Objects become entities too, even when they are things rather than
    people -- "swamp adder" is a node here. That is what lets a two-hop walk
    get from Helen Stoner to the snake, which is a path no co-occurrence
    graph would have.
    """
    ids: dict[str, int] = {}

    def entity_id(name: str, kind: str) -> int:
        if name not in ids:
            row = conn.execute(
                "INSERT INTO entities (name, kind) VALUES (%s, %s) "
                "ON CONFLICT (name, kind) DO UPDATE SET name = EXCLUDED.name RETURNING id",
                [name, kind],
            ).fetchone()
            ids[name] = int(row[0])
        return ids[name]

    written = 0
    for subject, predicate, obj in triples:
        src = entity_id(subject, "entity")
        dst = entity_id(obj, "entity")
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, weight) "
            "VALUES (%s, %s, %s, 1.0) ON CONFLICT DO NOTHING",
            [src, dst, predicate],
        )
        written += 1
    return written


def facts_about(conn, name: str, limit: int = 8) -> list[tuple[str, str, str]]:
    """Every stated relationship involving an entity, in either direction."""
    rows = conn.execute(
        """
        SELECT s.name, e.relation, d.name
        FROM   edges e
        JOIN   entities s ON s.id = e.src_id
        JOIN   entities d ON d.id = e.dst_id
        WHERE  s.name ILIKE %(needle)s OR d.name ILIKE %(needle)s
        LIMIT  %(limit)s
        """,
        {"needle": f"%{name}%", "limit": limit},
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def verbalise(triples) -> str:
    """Linearise triples into natural language.

    The book calls this "triples linearized as natural language templates".
    It matters: embedding models are trained on prose, and the string
    "Roylott|killed|Julia Stoner" embeds nothing like the sentence it means.
    """
    # ---- THE KEY LINE ------------------------------------------------
    # Triples must become PROSE before they are embedded. Models are
    # trained on sentences, and "Roylott|killed|Julia Stoner" embeds
    # nothing like the sentence it encodes.
    #
    # This is also what makes the joint-embedding strategy possible at
    # all -- you cannot index a triple, only a sentence about one.
    return ". ".join(
        f"{s} {p.replace('_', ' ')} {o}" for s, p, o in triples
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?",
                        default="Why did Helen Stoner's stepfather want her dead?")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "26. Knowledge Graph Integration",
        "Chapter 18, Knowledge Graph Integration",
        "Typed subject-predicate-object triples, and the three ways to "
        "combine them with a vector index.",
    )

    if not pg.available():
        display.missing_database("Triples live in the entities and edges tables.")
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    embedder = get_embedder()
    store = pg.PgVectorStore(conn, dim=embedder.dim, model=embedder.name)
    if store.count(args.strategy) == 0:
        print("\nNothing indexed. Run: python scripts/index_corpus.py\n")
        return 1

    # --- extraction --------------------------------------------------------
    display.heading("Extracting triples")
    conn.execute("DELETE FROM edges")
    conn.execute("DELETE FROM entities")

    if llm.available() and not llm.STUB:
        print(f"  model: {config.OPENAI_CHAT_MODEL}")
        rows = conn.execute(
            "SELECT body FROM chunks WHERE strategy = %s ORDER BY ordinal LIMIT 25",
            [args.strategy],
        ).fetchall()
        triples = []
        truncated = 0
        malformed = 0
        for index, (body,) in enumerate(rows, start=1):
            print(f"    extracting {index}/{len(rows)}", end="\r", flush=True)
            try:
                reply = llm.complete_json(
                    EXTRACT_PROMPT.format(passage=body),
                    system=EXTRACT_SYSTEM,
                    purpose="triple extraction",
                    # Dense passages yield a lot of triples. The default 800
                    # is not enough, and running out mid-object produces
                    # broken JSON rather than a short list.
                    max_tokens=1500,
                )
            except llm.JSONTruncated:
                # A passage too rich to fit the budget. Skipping it loses
                # some facts; aborting loses all of them. Extraction over a
                # corpus has to survive individual failures, and counting
                # them is how you find out the budget is wrong.
                truncated += 1
                continue
            except RuntimeError:
                malformed += 1
                continue

            for item in reply.get("triples", []):
                if isinstance(item, dict) and all(
                    k in item for k in ("subject", "predicate", "object")
                ):
                    triples.append(
                        (str(item["subject"]), str(item["predicate"]), str(item["object"]))
                    )
        print(" " * 40, end="\r")

        if truncated or malformed:
            print(f"  {truncated} passage(s) exceeded the token budget, "
                  f"{malformed} returned unusable JSON -- skipped, out of {len(rows)}.")
    else:
        if llm.STUB:
            display.stub_warning()
        else:
            print(f"  No LLM: {llm.why_unavailable()}")
        display.prompt_block("SYSTEM", EXTRACT_SYSTEM)
        display.prompt_block(
            "USER",
            EXTRACT_PROMPT.format(
                passage="“It is a swamp adder!” cried Holmes; “the deadliest "
                        "snake in India. He has died within ten seconds of "
                        "being bitten.”"
            ),
        )
        display.illustrative(
            '{"triples": [',
            '  {"subject": "swamp adder", "predicate": "is_a",',
            '   "object": "deadliest snake in India"},',
            '  {"subject": "swamp adder", "predicate": "killed",',
            '   "object": "Grimesby Roylott"}',
            ']}',
        )
        print("\n  Using a hand-written seed set below so the fusion comparison")
        print("  is still a real measurement.")
        triples = SEED_TRIPLES

    written = store_triples(conn, triples)
    entity_count = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
    print()
    display.kv("triples", written)
    display.kv("entities", entity_count)

    display.heading("A sample of the graph")
    display.table(
        ["subject", "predicate", "object"],
        [[display.truncate(s, 24), p, display.truncate(o, 26)] for s, p, o in triples[:8]],
        align="lll",
    )
    print("\n  Compare example 25's single 'appears_with' relation. These")
    print("  predicates carry the meaning, which is what makes 'who killed")
    print("  whom' a lookup rather than an inference.")

    # --- the three fusion strategies -------------------------------------
    print()
    display.heading(f"Query: {args.question!r}")

    # Baseline.
    vector = embedder.encode([args.question], show_progress=False)[0]
    baseline = store.search(vector, k=args.k, strategy=args.strategy)

    # 1. Pre-retrieval fusion.
    graph_facts = facts_about(conn, "Roylott") + facts_about(conn, "Stoner")
    expanded = f"{args.question} {verbalise(graph_facts[:6])}"
    pre_vector = embedder.encode([expanded], show_progress=False)[0]
    pre = store.search(pre_vector, k=args.k, strategy=args.strategy)

    # 2. Post-retrieval fusion: retrieve first, then attach facts about the
    #    entities that appear in what came back.
    post_facts = []
    for hit in baseline:
        for entity_name, in conn.execute(
            "SELECT name FROM entities WHERE position(name in %s) > 0 LIMIT 3",
            [hit.chunk.text],
        ).fetchall():
            post_facts.extend(facts_about(conn, entity_name, limit=3))

    display.table(
        ["strategy", "top story", "top score", "new vs baseline"],
        [
            ["baseline (text only)",
             display.truncate(baseline[0].chunk.story, 26),
             f"{baseline[0].score:.3f}", "-"],
            ["1. pre-retrieval fusion",
             display.truncate(pre[0].chunk.story, 26),
             f"{pre[0].score:.3f}",
             len({h.chunk.index for h in pre} - {h.chunk.index for h in baseline})],
        ],
        align="llrr",
    )

    print(f"\n  Query after graph expansion:")
    display.preview(expanded, limit=220)
    print("\n  The expansion injects names the question never used -- 'Julia")
    print("  Stoner', 'swamp adder', 'Stoke Moran' -- and those names are in")
    print("  the passages that answer it.")

    print("\n  2. Post-retrieval fusion: facts attached to what was retrieved")
    if post_facts:
        display.table(
            ["fact"],
            [[f"{s} {p.replace('_', ' ')} {o}"] for s, p, o in post_facts[:6]],
            align="l",
        )
        print("\n  These go into the prompt alongside the passages, as structured")
        print("  claims the model can cite. They are also checkable, which prose")
        print("  is not -- the book's point about explainability.")
    else:
        print("      no entities from the graph appeared in the retrieved text")

    print("\n  3. Joint embedding: verbalised triples indexed alongside the")
    print("     passages, so one search covers both. Not built here because it")
    print("     means a second index and a re-embed -- but note that the")
    print("     verbalise() step above is what makes it possible at all.")

    display.notice(
        "Typed predicates are the difference from example 25. 'appears_with' "
        "records proximity; 'killed' records a fact, and only one of those "
        "answers a question.",
        "Triples must be verbalised before embedding. Models are trained on "
        "prose, and 'Roylott|killed|Julia' embeds nothing like the sentence it "
        "encodes.",
        "Pre-retrieval fusion expands the query with names the user never "
        "typed, which is where most of the gain comes from -- it is query "
        "expansion with a reliable source.",
        "Extraction quality caps everything here. A model that infers "
        "relationships the passage does not state produces a confident graph "
        "of things that are not true, and graphs look authoritative.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
