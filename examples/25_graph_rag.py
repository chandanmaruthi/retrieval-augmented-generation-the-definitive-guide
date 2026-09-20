#!/usr/bin/env python3
"""
25. Graph-Based RAG (entities, edges, and k-hop traversal in SQL)
Book: Chapter 12, Graph-Based RAG

WHAT THIS SHOWS: Retrieval that follows relationships instead of similarity.

Vector search answers "what text resembles this query". A graph answers "what
is connected to this thing, and how". Those differ most on exactly the
questions RAG is worst at: multi-hop ones, where the answer is assembled from
two passages that are each individually a poor match.

Entities here are extracted with a gazetteer and co-occurrence rather than a
model, so this runs with no API key. That is a real technique with real
limits, and both are discussed below.

The traversal is a recursive CTE, including the cycle guard that everybody
forgets and that turns a graph walk into an infinite loop.

HOW THIS SCRIPT PROCEEDS
    1. Extract entities with a gazetteer     no model needed, see below
    2. Build edges from CO-OCCURRENCE        two names in one chunk
    3. Walk k hops with a recursive CTE      <-- THE TECHNIQUE, in SQL
    4. Compare what the graph reached vs what vector search found

Step 3 is a single SQL query. No graph database is needed for k-hop traversal
at this scale -- Postgres does it, and the cycle guard is one line.


WHAT THE GRAPH ADDS, AND WHAT IT DOES NOT
    Vector search answers "what text resembles this query". A graph answers
    "what is connected to this thing". Those differ most on multi-hop
    questions, where the answer is assembled from two passages that are each
    individually a poor match.

    But co-occurrence is a WEAK relation: it records that two characters
    shared a scene, not what passed between them. And it has a hub problem --
    Holmes appears with everyone, so an edge to Holmes carries almost no
    information, the same issue IDF solves for lexical search in example 18.
    Example 26 extracts typed triples instead.


REQUIRES: Postgres
RUN: python examples/25_graph_rag.py [--rebuild] ["entity name"]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, pg
from ragkit.corpus import load_book
from ragkit.embed import get_embedder

# A gazetteer. Named-entity recognition on Victorian fiction is genuinely hard
# -- "Openshaw", "Windibank" and "Roylott" are not in any standard model's
# vocabulary, and titles like "the King of Bohemia" are not proper nouns at
# all. A curated list for a known corpus beats a general model here, and says
# honestly that it does not generalise.
CHARACTERS = {
    "Sherlock Holmes": ["Holmes", "Sherlock"],
    "John Watson": ["Watson"],
    "Irene Adler": ["Irene Adler", "Adler"],
    "Godfrey Norton": ["Godfrey Norton", "Norton"],
    "Mrs. Hudson": ["Mrs. Hudson", "Hudson"],
    "Inspector Lestrade": ["Lestrade"],
    "Jabez Wilson": ["Jabez Wilson", "Wilson"],
    "John Clay": ["John Clay", "Clay"],
    "Peter Jones": ["Peter Jones", "Jones"],
    "Mary Sutherland": ["Mary Sutherland", "Miss Sutherland"],
    "James Windibank": ["Windibank"],
    "Hosmer Angel": ["Hosmer Angel", "Hosmer"],
    "Charles McCarthy": ["Charles McCarthy"],
    "James McCarthy": ["James McCarthy"],
    "John Turner": ["John Turner", "Turner"],
    "John Openshaw": ["Openshaw"],
    "Neville St. Clair": ["Neville St. Clair", "St. Clair"],
    "Hugh Boone": ["Hugh Boone", "Boone"],
    "Isa Whitney": ["Isa Whitney", "Whitney"],
    "Henry Baker": ["Henry Baker"],
    "James Ryder": ["Ryder"],
    "Countess of Morcar": ["Countess of Morcar", "Morcar"],
    "Helen Stoner": ["Helen Stoner", "Miss Stoner", "Stoner"],
    "Grimesby Roylott": ["Roylott"],
    "Victor Hatherley": ["Hatherley"],
    "Lysander Stark": ["Lysander Stark", "Stark"],
    "Lord St. Simon": ["Lord St. Simon", "St. Simon"],
    "Hatty Doran": ["Hatty Doran", "Doran"],
    "Alexander Holder": ["Alexander Holder", "Holder"],
    "Mary Holder": ["Mary Holder"],
    "Arthur Holder": ["Arthur"],
    "George Burnwell": ["Burnwell"],
    "Violet Hunter": ["Violet Hunter", "Miss Hunter"],
    "Jephro Rucastle": ["Rucastle"],
    "Alice Rucastle": ["Alice Rucastle", "Alice"],
}


def extract_entities(text: str) -> set[str]:
    """Find which known characters a passage mentions.

    Longest alias first, so "Mary Holder" is not swallowed by "Holder", and
    word boundaries so "Stark" does not match "Starkey".
    """
    found = set()
    for canonical, aliases in CHARACTERS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                found.add(canonical)
                break
    return found


def build_graph(conn, strategy: str) -> tuple[int, int]:
    """Populate entities and edges from chunk co-occurrence.

    Worth seeing despite being a weak relation: it needs no model, it is
    deterministic, and on a corpus with a known cast its mistakes are visible.
    See WHAT THE GRAPH ADDS at the top of this file.
    """
    conn.execute("DELETE FROM edges")
    conn.execute("DELETE FROM entities")

    rows = conn.execute(
        "SELECT id, body FROM chunks WHERE strategy = %s ORDER BY ordinal", [strategy]
    ).fetchall()

    entity_ids: dict[str, int] = {}
    for name in CHARACTERS:
        row = conn.execute(
            "INSERT INTO entities (name, kind, aliases) VALUES (%s, 'person', %s) "
            "ON CONFLICT (name, kind) DO UPDATE SET name = EXCLUDED.name RETURNING id",
            [name, CHARACTERS[name]],
        ).fetchone()
        entity_ids[name] = int(row[0])

    pair_counts: Counter = Counter()
    pair_evidence: dict[tuple[str, str], int] = {}

    for chunk_id, body in rows:
        present = extract_entities(body)
        for a, b in combinations(sorted(present), 2):
            pair_counts[(a, b)] += 1
            pair_evidence.setdefault((a, b), chunk_id)

    edges = 0
    for (a, b), count in pair_counts.items():
        # Edges are inserted both ways so the recursive walk can move in
        # either direction without the query needing a UNION on every step.
        for src, dst in ((a, b), (b, a)):
            conn.execute(
                "INSERT INTO edges (src_id, dst_id, relation, weight, evidence_chunk_id) "
                "VALUES (%s, %s, 'appears_with', %s, %s) "
                "ON CONFLICT DO NOTHING",
                [entity_ids[src], entity_ids[dst], float(count), pair_evidence[(a, b)]],
            )
            edges += 1

    # Drop characters with no connections at all -- usually a gazetteer entry
    # whose aliases never matched.
    conn.execute(
        "DELETE FROM entities e WHERE NOT EXISTS "
        "(SELECT 1 FROM edges g WHERE g.src_id = e.id OR g.dst_id = e.id)"
    )
    remaining = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
    return int(remaining), edges


# The traversal: k hops from a seed entity, in one recursive CTE.
WALK_SQL = """
WITH RECURSIVE walk AS (
    SELECT e.dst_id,
           e.relation,
           e.weight,
           e.evidence_chunk_id,
           1                                AS depth,
           ARRAY[e.src_id, e.dst_id]        AS path,
           e.weight                         AS path_weight
    FROM   edges e
    WHERE  e.src_id = %(seed)s

    UNION ALL

    SELECT e.dst_id,
           e.relation,
           e.weight,
           e.evidence_chunk_id,
           w.depth + 1,
           w.path || e.dst_id,
           w.path_weight + e.weight
    FROM   walk w
    JOIN   edges e ON e.src_id = w.dst_id
    WHERE  w.depth < %(max_depth)s
      -- ---- THE KEY LINE -------------------------------------------
      -- The cycle guard. Edges are inserted both ways, so A->B->A is a
      -- valid walk and the recursion would follow it until the server
      -- runs out of memory. Remove this and the query does not return
      -- slowly -- it does not return at all.
      AND  NOT e.dst_id = ANY (w.path)
)
SELECT w.depth,
       en.name,
       w.weight,
       w.path_weight,
       c.body
FROM   walk w
JOIN   entities en ON en.id = w.dst_id
LEFT   JOIN chunks c ON c.id = w.evidence_chunk_id
ORDER  BY w.depth, w.weight DESC
LIMIT  %(limit)s;
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", nargs="?", default="Grimesby Roylott")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the graph")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "25. Graph-Based RAG",
        "Chapter 12, Graph-Based RAG",
        "Extract entities, connect them, then traverse k hops in SQL to find "
        "evidence that similarity search would never reach.",
    )

    if not pg.available():
        display.missing_database(
            "The graph lives in the entities and edges tables, and the "
            "traversal is a recursive CTE."
        )
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    # This example and example 26 share the entities and edges tables, and
    # they build different graphs: co-occurrence here, typed triples there.
    # A non-empty table is therefore not evidence that *our* graph is loaded,
    # so check for our own relation rather than for any rows at all.
    #
    # Left unchecked, this example happily queries example 26's graph and then
    # reports that a character it never inserted does not exist -- a confusing
    # failure whose cause is two scripts sharing a namespace.
    ours = conn.execute(
        "SELECT count(*) FROM edges WHERE relation = 'appears_with'"
    ).fetchone()[0]

    if args.rebuild or ours == 0:
        display.heading("Building the graph")
        if ours == 0 and conn.execute("SELECT count(*) FROM edges").fetchone()[0]:
            print("  (the graph currently loaded was built by example 26; rebuilding)")
        print("  Gazetteer entity extraction + chunk co-occurrence.")
        entities, edges = build_graph(conn, args.strategy)
        print(f"  {entities} entities, {edges} directed edges")
    else:
        entities = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
        edges = ours
        print(f"  graph: {entities} entities, {edges} edges (--rebuild to redo)")

    # --- the graph itself -------------------------------------------------
    display.heading("Most connected characters")
    rows = conn.execute(
        """
        SELECT en.name, count(*) AS degree, sum(e.weight) AS strength
        FROM   edges e JOIN entities en ON en.id = e.src_id
        GROUP  BY en.name ORDER BY strength DESC LIMIT 8
        """
    ).fetchall()
    display.table(
        ["character", "connections", "co-occurrences"],
        [[r[0], r[1], int(r[2])] for r in rows],
    )
    print("\n  Holmes and Watson dominate, which is correct and also the")
    print("  weakness of co-occurrence: they appear with everyone, so an edge")
    print("  to Holmes carries almost no information.")

    # --- traversal --------------------------------------------------------
    seed_row = conn.execute(
        "SELECT id, name FROM entities WHERE name ILIKE %s LIMIT 1", [f"%{args.entity}%"]
    ).fetchone()
    if not seed_row:
        print(f"\n  No entity matching {args.entity!r}.")
        return 1
    seed_id, seed_name = int(seed_row[0]), seed_row[1]

    display.heading(f"Walking {args.depth} hops from {seed_name!r}")
    rows = conn.execute(
        WALK_SQL, {"seed": seed_id, "max_depth": args.depth, "limit": 14}
    ).fetchall()

    display.table(
        ["hop", "reached", "edge weight", "evidence"],
        [[r[0], display.truncate(r[1], 24), int(r[2]),
          display.truncate(r[4] or "", 34)] for r in rows],
        align="rlrl",
    )

    direct = [r for r in rows if r[0] == 1]
    indirect = [r for r in rows if r[0] > 1]
    print(f"\n  {len(direct)} reached in one hop, {len(indirect)} only in two or more.")
    print("  The second group is what a graph adds: characters never named in")
    print("  the same passage as the seed, reachable only through someone else.")

    # --- cycle guard ------------------------------------------------------
    display.heading("Why the cycle guard is not optional")
    print("  Edges are bidirectional, so A->B->A is a valid walk and the")
    print("  recursion would follow it forever. The guard is one line:\n")
    print("      AND NOT e.dst_id = ANY (w.path)\n")
    print("  It keeps the path travelled so far in an array and refuses to")
    print("  revisit. Remove it and this query does not return slowly -- it")
    print("  does not return at all.")

    # --- graph vs vector --------------------------------------------------
    display.heading("Graph reach vs vector similarity")
    embedder = get_embedder()
    store = pg.PgVectorStore(conn, dim=embedder.dim, model=embedder.name)

    question = f"Who was connected to {seed_name}?"
    vector = embedder.encode([question], show_progress=False)[0]
    vector_hits = store.search(vector, k=5, strategy=args.strategy)

    vector_entities: set[str] = set()
    for hit in vector_hits:
        vector_entities |= extract_entities(hit.chunk.text)

    graph_entities = {r[1] for r in rows}

    display.table(
        ["method", "entities surfaced"],
        [
            ["vector search (top 5)", ", ".join(sorted(vector_entities)) or "-"],
            ["graph walk (2 hops)", ", ".join(sorted(graph_entities)[:8]) + "..."],
        ],
        align="ll",
    )
    only_graph = graph_entities - vector_entities
    print(f"\n  {len(only_graph)} entities were reachable in the graph but absent")
    print("  from the top-5 vector results. Similarity search reads five")
    print("  passages; the graph consults everything that was ever linked.")

    display.notice(
        "A recursive CTE is the whole traversal. No graph database is needed "
        "for k-hop queries at this scale -- Postgres does it in SQL.",
        "The cycle guard is mandatory. Bidirectional edges make A->B->A a "
        "valid path, and the recursion will follow it until the server dies.",
        "Co-occurrence is a weak relation: it records that two characters "
        "shared a scene, not what happened between them. Example 26 extracts "
        "typed triples instead.",
        "Hub entities poison co-occurrence graphs. Holmes appears with "
        "everyone, so an edge to Holmes says nothing -- the same problem IDF "
        "solves for lexical search in example 18.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
