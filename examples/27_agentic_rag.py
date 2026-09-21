#!/usr/bin/env python3
"""
27. Agentic RAG
Book: Chapter 13, Agentic RAG
WHAT THIS SHOWS: Retrieval as a loop instead of a step.

    plan      break the question into sub-questions
    retrieve  gather evidence for each
    reflect   score what came back; is it enough?
    re-plan   if not, reformulate toward the gap and go again
    answer    synthesise once the evidence holds up

The baseline in example 17 retrieves once and hopes. That works for "what was
the speckled band?" and fails for anything needing two facts from two places,
because a single embedding of a two-part question is near neither part.

The interesting property here is that the loop is *bounded*. Unbounded
self-reflection is how agentic systems burn a hundred calls on a question
whose answer is not in the corpus, so every stopping condition is explicit.

HOW THIS SCRIPT PROCEEDS
    1. Run the one-shot baseline for comparison
    2. Plan: decompose into sub-questions     <-- THE TECHNIQUE, the load-bearing step
    3. Retrieve for each sub-question
    4. Criticise: is the evidence sufficient?
    5. If not, re-plan toward the gap and repeat
    6. Stop -- and note there are THREE stopping conditions

Step 6 matters more than it sounds. Unbounded self-reflection is how agentic
systems spend a hundred calls on a question whose answer is not in the corpus.


WHAT CHANGED SINCE EXAMPLE 23
    Example 23 adapted a single retrieval: more passages for harder queries.
    This makes retrieval a LOOP, which is a different thing -- it can go back
    and look for something specific it noticed was missing.

    The cost is real: this uses several model calls against the baseline's
    one. It is a tool for questions the one-shot pipeline cannot answer at any
    k, and it should be ROUTED to -- by something like example 23's classifier
    -- rather than used by default.


REQUIRES: Postgres. OPENAI_API_KEY for planning and reflection -- without it,
the retrieval half still runs and the prompts are printed.
RUN: python examples/27_agentic_rag.py ["your question"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import config, display, llm, pg, tokens
from ragkit.embed import get_embedder

DEFAULT_QUESTION = (
    "How did Holmes connect a hat and a goose to a stolen gem, "
    "and why did he let the thief go?"
)

MAX_ROUNDS = 3
CONFIDENCE_TARGET = 0.45

PLANNER_SYSTEM = (
    "You decompose questions into independent retrieval sub-questions. "
    "You reply with JSON."
)

PLANNER_PROMPT = """Break this question into 2-4 sub-questions that can each be
answered by searching a collection of Sherlock Holmes stories independently.

Each sub-question must be self-contained: name people and objects explicitly
rather than using pronouns, because each one is sent to the search engine on
its own.

Reply as JSON: {{"sub_questions": ["...", "..."]}}

Question: {question}"""

CRITIC_SYSTEM = (
    "You judge whether retrieved passages are sufficient to answer a question. "
    "You reply with JSON and you are strict."
)

CRITIC_PROMPT = """Question: {question}

Retrieved passages:
{context}

Is this enough to answer the question completely?

Reply as JSON:
{{"sufficient": true/false,
  "missing": "what specifically is still missing, or empty string",
  "follow_up": "a search query that would find it, or empty string"}}"""


def retrieve(store, embedder, query: str, k: int, strategy: str):
    vector = embedder.encode([query], show_progress=False)[0]
    return store.search(vector, k=k, strategy=strategy)


def evidence_confidence(hits) -> float:
    """A cheap, local stand-in for the critic.

    Used when no model is available, and as a pre-filter when one is: if the
    scores are obviously poor there is no point paying for a critic call to
    confirm it. Same score-distribution reasoning as example 23.
    """
    if not hits:
        return 0.0
    scores = np.array([h.score for h in hits])
    margin = float(scores[0] - scores[1:].mean()) if len(scores) > 1 else 0.0
    return float(min(1.0, max(0.0, scores[0])) * (0.6 + min(0.4, margin * 3)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "27. Agentic RAG",
        "Chapter 13, Agentic RAG",
        "Plan, retrieve, criticise, re-retrieve. A bounded loop instead of a "
        "single shot.",
    )

    if not pg.available():
        display.missing_database("Every round of the loop retrieves from the index.")
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
    have_model = llm.available()
    if not have_model:
        print(f"  No LLM: {llm.why_unavailable()}")
    elif llm.STUB:
        display.stub_warning()
    print()

    # --- baseline for comparison -----------------------------------------
    display.heading("Baseline: one shot, as in example 17")
    baseline = retrieve(store, embedder, args.question, args.k, args.strategy)
    display.table(
        ["rank", "score", "story", "passage"],
        [[h.rank, f"{h.score:.3f}", display.truncate(h.chunk.story, 24),
          display.truncate(h.chunk.text, 30)] for h in baseline],
        align="rrll",
    )
    baseline_conf = evidence_confidence(baseline)
    display.kv("evidence confidence", f"{baseline_conf:.2f}")
    print("\n  A compound question embedded whole produces a vector that sits")
    print("  between its parts and near neither of them.")

    # --- planning ---------------------------------------------------------
    display.heading("Step 1: plan")
    if have_model:
        # --- the model decides the sub-questions -- book:retrieval-agentic
        plan = llm.complete_json(
            PLANNER_PROMPT.format(question=args.question),
            system=PLANNER_SYSTEM,
            purpose="query planning",
        )
        # ------------------------------------------------------------- /book
        sub_questions = [
            s for s in plan.get("sub_questions", []) if isinstance(s, str) and s.strip()
        ]
    else:
        display.prompt_block("PLANNER SYSTEM", PLANNER_SYSTEM)
        display.prompt_block("PLANNER USER", PLANNER_PROMPT.format(question=args.question))
        # Hand-written, so the retrieval half of the loop still runs and is
        # still a real measurement.
        sub_questions = [
            "How did Holmes trace the hat to Henry Baker?",
            "Where was the blue carbuncle found inside the goose?",
            "Why did Holmes decide not to hand James Ryder to the police?",
        ]
        display.illustrative(*[f'"{s}"' for s in sub_questions])
        print("\n  Using those hand-written sub-questions below so the retrieval")
        print("  comparison is still real.")

    print()
    for index, sub in enumerate(sub_questions, start=1):
        print(f"  {index}. {sub}")

    # --- the loop ---------------------------------------------------------
    display.heading("Step 2: retrieve per sub-question, then criticise")

    gathered: dict[int, list] = {}
    rounds_used = 0
    calls = 1 if have_model else 0

    for round_number in range(1, args.rounds + 1):
        rounds_used = round_number
        print(f"\n  --- round {round_number} ---")

        for sub in sub_questions:
            hits = retrieve(store, embedder, sub, args.k, args.strategy)
            for hit in hits:
                gathered.setdefault(hit.chunk.index, []).append(hit)
            confidence = evidence_confidence(hits)
            print(f"    {display.truncate(sub, 52):<54} conf {confidence:.2f}  "
                  f"{len(hits)} hits")

        unique = list(gathered)
        combined = [max(v, key=lambda h: h.score) for v in gathered.values()]
        combined.sort(key=lambda h: -h.score)
        overall = evidence_confidence(combined[: args.k])

        print(f"\n    unique passages gathered: {len(unique)}")
        print(f"    combined confidence: {overall:.2f}")

        # Stopping conditions, all explicit.
        # ---- THE THREE STOPPING CONDITIONS -------------------------
        # All explicit, because any one of them missing turns this into
        # an unbounded spend: (1) confidence cleared, (2) round limit,
        # (3) the critic can find no actionable gap.
        if overall >= CONFIDENCE_TARGET:
            print(f"    -> confidence cleared {CONFIDENCE_TARGET}; stopping")
            break
        if round_number == args.rounds:
            print(f"    -> round limit ({args.rounds}) reached; stopping")
            break

        # Re-plan toward the gap.
        if have_model:
            context = "\n\n".join(
                f"[{i+1}] {h.chunk.context_text}" for i, h in enumerate(combined[:6])
            )
            verdict = llm.complete_json(
                CRITIC_PROMPT.format(question=args.question, context=context),
                system=CRITIC_SYSTEM,
                purpose="self-reflection",
            )
            calls += 1
            if verdict.get("sufficient"):
                print("    -> critic says the evidence is sufficient; stopping")
                break
            follow_up = str(verdict.get("follow_up", "")).strip()
            missing = str(verdict.get("missing", "")).strip()
            if not follow_up:
                print("    -> critic found no actionable gap; stopping")
                break
            print(f"    critic: missing {missing!r}")
            print(f"    -> re-retrieving with {follow_up!r}")
            sub_questions = [follow_up]
        else:
            print("    -> no critic available; stopping after one round")
            break

    # --- results ----------------------------------------------------------
    display.heading("What the loop gathered")
    combined = [max(v, key=lambda h: h.score) for v in gathered.values()]
    combined.sort(key=lambda h: -h.score)

    baseline_ids = {h.chunk.index for h in baseline}
    agentic_ids = {h.chunk.index for h in combined}

    display.table(
        ["", "passages", "stories covered", "new vs baseline"],
        [
            ["baseline (one shot)", len(baseline),
             len({h.chunk.story for h in baseline}), "-"],
            ["agentic loop", len(combined),
             len({h.chunk.story for h in combined}),
             len(agentic_ids - baseline_ids)],
        ],
        align="lrrr",
    )
    print(f"\n  rounds used: {rounds_used} of {args.rounds}")
    print(f"  model calls: {calls}")
    print(f"  retrieval calls: {rounds_used * len(sub_questions)}+")

    print("\n  Top passages after merging:")
    display.table(
        ["score", "story", "passage"],
        [[f"{h.score:.3f}", display.truncate(h.chunk.story, 24),
          display.truncate(h.chunk.text, 36)] for h in combined[:6]],
        align="rll",
    )

    context_cost = sum(tokens.count_tokens(h.chunk.context_text) for h in combined[:6])
    baseline_cost = sum(tokens.count_tokens(h.chunk.context_text) for h in baseline)
    display.heading("What it cost")
    display.table(
        ["", "context tokens", "model calls"],
        [["baseline", f"{baseline_cost:,}", 1],
         ["agentic", f"{context_cost:,}", calls + 1]],
    )
    print("\n  Agentic RAG is not cheaper. It is for questions the one-shot")
    print("  pipeline cannot answer at any k, and it should be routed to, not")
    print("  used by default -- which is what example 23's classifier is for.")

    display.notice(
        "Decomposition is the load-bearing step. Each sub-question is a clean "
        "retrieval target; the compound original is a vector near neither of "
        "its halves.",
        "Every stopping condition is explicit: a confidence target, a round "
        "limit, and a critic that can find no actionable gap. Without all "
        "three, a question with no answer in the corpus loops until the budget "
        "is gone.",
        "The critic reads the retrieved context, not the model's own answer. "
        "Asking a model to check its own output invites it to agree with "
        "itself.",
        f"This used {calls + 1} model calls against the baseline's 1. Agentic "
        "retrieval is a tool for hard questions, not a default setting.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
