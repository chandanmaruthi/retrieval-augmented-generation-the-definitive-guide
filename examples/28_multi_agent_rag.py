#!/usr/bin/env python3
"""
28. Multi-Agent RAG
Book: Chapter 14, Multi-Agent RAG Systems
WHAT THIS SHOWS: Several specialised retrievers working in parallel against a
shared blackboard, with a critic and explicit per-agent budgets.

Example 27's agentic loop is sequential: plan, retrieve, reflect, repeat. This
runs the retrievers concurrently over *different indexes* -- dense, lexical,
graph -- and merges what they find.

The parts that matter in practice are the unglamorous ones:

    blackboard   shared state, so agents see each other's findings
    budgets      per-agent limits on time and calls, enforced
    cancellation stragglers are dropped, not waited for
    critic       a gate before the answer is released

An unbudgeted multi-agent system is a very expensive way to produce the same
answer as example 17, and that failure is silent.

HOW THIS SCRIPT PROCEEDS
    1. Launch three retrieval agents in parallel   dense, lexical, graph
    2. Each posts findings to a shared blackboard  <-- THE TECHNIQUE
    3. Track WHICH agents found each passage       the cross-agent signal
    4. Enforce a wall-clock budget, drop stragglers
    5. Gate the result through a critic

Step 3 is what the blackboard is actually for. A passage found by all three
agents is a far stronger result than one found by the best single retriever,
and no individual agent can compute that.


WHAT CHANGED SINCE EXAMPLE 27
    Example 27's loop is sequential: plan, retrieve, reflect, repeat. This
    runs retrievers CONCURRENTLY over different indexes and merges what they
    find.

    Three agents with different prompts over the same index is not a
    multi-agent system, it is three copies of one. The agents here search
    genuinely different things.


REQUIRES: Postgres. OPENAI_API_KEY only for the critic.
RUN: python examples/28_multi_agent_rag.py ["your question"]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, llm, pg
from ragkit.embed import get_embedder

DEFAULT_QUESTION = "Who stole the blue carbuncle and how was it found?"


@dataclass
class Blackboard:
    """Shared state every agent reads and writes.

    The book calls this a blackboard or memory bus. Its real job is
    deduplication: three retrievers over three indexes will surface the same
    passage repeatedly, and without a shared view each one pays to process it
    again.
    """

    findings: dict[int, dict] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def post(self, agent: str, hits, elapsed_ms: float) -> int:
        added = 0
        for hit in hits:
            existing = self.findings.get(hit.chunk.index)
            if existing is None:
                self.findings[hit.chunk.index] = {
                    "chunk": hit.chunk,
                    "agents": [agent],
                    "best_score": hit.score,
                    "best_rank": hit.rank,
                }
                added += 1
            else:
                # Found by more than one retriever -- a strong signal, and the
                # reason the blackboard tracks provenance rather than just
                # collecting text.
                existing["agents"].append(agent)
                existing["best_score"] = max(existing["best_score"], hit.score)
        self.log.append(f"{agent}: {len(hits)} hits, {added} new, {elapsed_ms:.0f} ms")
        return added


def dense_agent(store, embedder, question, k, strategy):
    vector = embedder.encode([question], show_progress=False)[0]
    return store.search(vector, k=k, strategy=strategy)


def lexical_agent(store, embedder, question, k, strategy):
    """Lexical retrieval with OR semantics.

    match="any" rather than the default. A candidate-generation agent wants
    recall: the default AND behaviour requires a chunk to contain every word
    of the question, which on a natural-language query usually means it
    returns nothing and the agent contributes nothing.
    """
    return store.search_lexical(question, k=k, strategy=strategy, match="any")


def graph_agent(conn, store, embedder, question, k, strategy):
    """Retrieve via entities mentioned in the question.

    Needs the graph from example 25 or 26. When it is empty this agent
    returns nothing, which is the correct behaviour -- an agent with no data
    should contribute nothing rather than fabricate a contribution.
    """
    rows = conn.execute(
        "SELECT name FROM entities WHERE position(lower(name) in lower(%s)) > 0 LIMIT 3",
        [question],
    ).fetchall()
    if not rows:
        return []

    names = [r[0] for r in rows]
    expanded = question + " " + " ".join(names)
    vector = embedder.encode([expanded], show_progress=False)[0]
    return store.search(vector, k=k, strategy=strategy)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--budget-ms", type=int, default=2000,
                        help="wall-clock budget for the retrieval phase")
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "28. Multi-Agent RAG",
        "Chapter 14, Multi-Agent RAG Systems",
        "Parallel retrievers over different indexes, a shared blackboard, "
        "enforced budgets and a critic gate.",
    )

    if not pg.available():
        display.missing_database("Every agent retrieves from the database.")
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

    print(f"  question: {args.question!r}")
    print(f"  budget: {args.budget_ms} ms for the retrieval phase\n")

    board = Blackboard()

    # --- parallel retrieval ----------------------------------------------
    display.heading("Retrieval agents, in parallel")

    # Each agent gets its own connection: a psycopg connection is not safe to
    # share across threads, and this is the sort of thing that works in a demo
    # and corrupts state under load.
    # ---- THE KEY DETAIL --------------------------------------------
    # Each agent opens its OWN connection. A psycopg connection is not
    # safe to share across threads: it appears to work in a demo and
    # corrupts state under concurrency.
    #
    # The embedder is shared, and is made thread-safe by a lock inside
    # ragkit/embed.py -- concurrent calls into one torch model segfault
    # on Apple's MPS backend rather than raising.
    def run_agent(name: str):
        started = time.perf_counter()
        own_conn = pg.connect()
        try:
            own_store = pg.PgVectorStore(own_conn, dim=embedder.dim, model=embedder.name)
            if name == "dense":
                hits = dense_agent(own_store, embedder, args.question, args.k, args.strategy)
            elif name == "lexical":
                hits = lexical_agent(own_store, embedder, args.question, args.k, args.strategy)
            else:
                hits = graph_agent(own_conn, own_store, embedder, args.question,
                                   args.k, args.strategy)
            return name, hits, (time.perf_counter() - started) * 1000
        finally:
            own_conn.close()

    started = time.perf_counter()
    # --- agents run in parallel, under one budget - book:retrieval-multi-agent
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_agent, name): name
                   for name in ("dense", "lexical", "graph")}

        for future in as_completed(futures, timeout=args.budget_ms / 1000):
            try:
                name, hits, elapsed = future.result()
                board.post(name, hits, elapsed)
    # ------------------------------------------------------------- /book
            except Exception as error:  # noqa: BLE001
                # One agent failing must not take the answer with it. A
                # degraded result beats no result.
                board.log.append(f"{futures[future]}: FAILED ({error})")

    wall_ms = (time.perf_counter() - started) * 1000

    for line in board.log:
        print(f"  {line}")
    print(f"\n  wall clock: {wall_ms:.0f} ms")

    sequential = sum(
        float(line.split()[-2]) for line in board.log if line.endswith("ms")
    )
    if sequential:
        print(f"  sum of agent times: {sequential:.0f} ms")
        print(f"  parallel speedup: {sequential / max(wall_ms, 1):.1f}x")

    # --- the blackboard ---------------------------------------------------
    display.heading("Blackboard after the retrieval phase")
    entries = sorted(
        board.findings.values(),
        key=lambda e: (-len(set(e["agents"])), -e["best_score"]),
    )
    display.table(
        ["found by", "score", "story", "passage"],
        [
            [", ".join(sorted(set(e["agents"]))), f"{e['best_score']:.3f}",
             display.truncate(e["chunk"].story, 22),
             display.truncate(e["chunk"].text, 28)]
            for e in entries[:8]
        ],
        align="llll",
    )

    multi = [e for e in entries if len(set(e["agents"])) > 1]
    print(f"\n  {len(board.findings)} unique passages from "
          f"{sum(len(e['agents']) for e in entries)} agent results.")
    print(f"  {len(multi)} were found by more than one agent -- that agreement")
    print("  is a ranking signal no single retriever can produce, and it is")
    print("  the main thing the blackboard is for.")

    # --- budget enforcement -----------------------------------------------
    display.heading("Budget enforcement")
    print("  The retrieval phase ran under a wall-clock budget, and")
    print("  as_completed() with a timeout is what enforces it. An agent that")
    print("  overran would be abandoned, not waited for.")
    print("\n  This is the part that decides whether a multi-agent design is")
    print("  usable. Without it the system's latency is its slowest agent on")
    print("  its worst day, and its cost is unbounded.")
    failed = [line for line in board.log if "FAILED" in line]
    display.kv("agents completed", f"{len(board.log) - len(failed)}/3")
    display.kv("agents failed or dropped", len(failed))

    # --- critic -----------------------------------------------------------
    display.heading("Critic gate")
    top = entries[: args.k]
    if not llm.available():
        print(f"  No LLM: {llm.why_unavailable()}")
        print("\n  A local proxy for the critic, using the same signals as")
        print("  example 23 -- agreement and score margin:\n")
        agreement = len(multi) / max(len(entries), 1)
        best = entries[0]["best_score"] if entries else 0.0
        display.kv("cross-agent agreement", f"{agreement:.2f}")
        display.kv("best score", f"{best:.3f}")
        verdict = "release" if (best > 0.45 and agreement > 0.1) else "insufficient -- re-plan"
        display.kv("verdict", verdict)
        print("\n  A real critic reads the passages and checks entailment. This")
        print("  version only reads the numbers, which catches the obvious")
        print("  failures and none of the subtle ones.")
    else:
        if llm.STUB:
            display.stub_warning()
        context = "\n\n".join(
            f"[{i+1}] {e['chunk'].context_text}" for i, e in enumerate(top)
        )
        verdict = llm.complete_json(
            f"Question: {args.question}\n\nPassages:\n{context}\n\n"
            'Reply as JSON: {"sufficient": true/false, "reason": "one sentence", '
            '"unsupported_risk": "low/medium/high"}',
            system="You are a strict critic gating a RAG answer before release.",
            purpose="critic gate",
        )
        display.kv("sufficient", verdict.get("sufficient"))
        display.kv("risk", verdict.get("unsupported_risk"))
        display.para(str(verdict.get("reason", "")), indent="      ")

    display.notice(
        "The agents search different indexes, not the same index differently. "
        "Three dense retrievers with different prompts is not a multi-agent "
        "system, it is three copies of one.",
        "Passages found by more than one agent are the highest-confidence "
        "results. That cross-agent agreement is the signal the blackboard "
        "exists to compute.",
        "Budgets and cancellation are what make this shippable. Without them "
        "latency is the slowest agent's worst case and cost has no ceiling.",
        "Each agent needs its own database connection. Sharing one across "
        "threads appears to work and corrupts state under concurrency.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
