#!/usr/bin/env python3
"""
33. Synthetic Data Generation for Evaluation
Book: Chapter 20, Synthetic Data Generation
WHAT THIS SHOWS: Generating question-answer-context triples so you have an
evaluation set at all, and the filtering that decides whether it is worth
having.

The hand-written gold set in data/eval_questions.json took real effort for
thirty questions. That does not scale to a corpus of ten thousand documents,
and it is the reason most RAG systems ship with no evaluation set.

Generation is the easy half. The book's four steps are passage selection,
question generation, answer synthesis, and *filtering for factual correctness
and coverage* -- and the fourth is where the value is. Ungated generation
produces questions that are unanswerable, trivially answerable from the
question itself, or duplicates, and an evaluation set full of those reports
numbers that mean nothing.

This example implements all four steps plus three filters, and reports what
each filter removed.

HOW THIS SCRIPT PROCEEDS
    1. Select passages                  stratification matters, see below
    2. Generate questions and answers
    3. Generate the supporting evidence quote
    4. FILTER                           <-- THE TECHNIQUE; steps 1-3 are the easy half
    5. Cross-check with a second pass

Three filters run in step 4, and each catches a failure mode that would
otherwise become a permanent, invisible distortion in every score.


WHY STEP 4 IS THE WHOLE POINT
    Generating questions is easy and every tutorial stops there. Ungated
    generation produces questions that are unanswerable, trivially answerable
    from the question itself, or duplicates.

    The worst of the three is the self-answering question: "What was the swamp
    adder that Holmes identified as the speckled band?" is answerable by a
    system that retrieves nothing at all, so it inflates every score it
    appears in -- and it looks entirely reasonable to a human skimming the set.


REQUIRES: OPENAI_API_KEY (falls back to showing the prompts and running the
filters against the hand-written gold set).
RUN: python examples/33_synthetic_eval_data.py [--passages 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import config, display, llm, pg
from ragkit.embed import get_embedder

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"

GEN_SYSTEM = (
    "You write evaluation questions for a retrieval system over fiction. "
    "You reply with JSON."
)

GEN_PROMPT = """Write {n} evaluation questions answerable ONLY from this passage.

Rules:
- The passage must contain the complete answer.
- Name people and places explicitly; the question is read without the passage.
- Do not give the answer away inside the question.
- For each, also give the exact sentence from the passage that answers it.

Reply as JSON:
{{"items": [{{"question": "...", "answer": "...", "evidence": "..."}}]}}

PASSAGE:
{passage}"""

VERIFY_SYSTEM = (
    "You verify that a question is answerable from a passage. Strict. JSON only."
)

VERIFY_PROMPT = """Passage: {passage}

Question: {question}
Proposed answer: {answer}

Reply as JSON:
{{"answerable_from_passage": true/false,
  "answer_correct": true/false,
  "self_answering": true/false,
  "note": "one sentence"}}

"self_answering" means the question contains its own answer."""


# ---------------------------------------------------------------------------
# Filters -- these run with or without a model
# ---------------------------------------------------------------------------


def filter_evidence_present(item: dict, passage: str) -> tuple[bool, str]:
    """The cited evidence must actually be in the passage.

    A model asked to quote will sometimes paraphrase instead, and a gold item
    whose evidence string does not appear in the corpus can never be scored --
    it will count as a retrieval failure no matter what is retrieved.
    """
    evidence = " ".join(item.get("evidence", "").split())
    if not evidence:
        return False, "no evidence quoted"
    if evidence not in " ".join(passage.split()):
        return False, "evidence not found in the passage"
    return True, ""


def filter_self_answering(item: dict) -> tuple[bool, str]:
    """Reject questions that contain their own answer.

    "What was the swamp adder that Holmes identified as the speckled band?"
    is answerable by a system that retrieves nothing at all, so it inflates
    every score it appears in.
    """
    question = item.get("question", "").lower()
    answer = item.get("answer", "").lower()
    if not question or not answer:
        return False, "missing question or answer"

    stop = {"the", "a", "an", "was", "is", "of", "in", "to", "and", "that", "it"}
    answer_words = {w for w in answer.split() if w not in stop and len(w) > 3}
    if not answer_words:
        return True, ""
    overlap = sum(1 for w in answer_words if w in question) / len(answer_words)
    if overlap > 0.6:
        return False, f"question contains {overlap:.0%} of the answer"
    return True, ""


def filter_duplicates(items: list[dict], embedder, threshold: float = 0.85):
    """Drop near-duplicate questions.

    Generating over adjacent passages produces the same question repeatedly.
    Duplicates do not merely waste calls -- they silently weight the
    evaluation toward whatever topic happened to be duplicated.
    """
    if not items:
        return [], 0
    vectors = embedder.encode([i["question"] for i in items], show_progress=False)
    keep, dropped = [], 0
    kept_vectors: list[np.ndarray] = []
    for item, vector in zip(items, vectors):
        if kept_vectors and max(float(vector @ k) for k in kept_vectors) > threshold:
            dropped += 1
            continue
        keep.append(item)
        kept_vectors.append(vector)
    return keep, dropped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=int, default=10)
    parser.add_argument("--per-passage", type=int, default=2)
    parser.add_argument("--out", default="")
    parser.add_argument("--strategy", default="contextual_header")
    args = parser.parse_args()

    display.banner(
        "33. Synthetic Data Generation",
        "Chapter 20, Synthetic Data Generation",
        "Generating evaluation triples, and the filtering that decides "
        "whether the result is an evaluation set or noise.",
    )

    if not pg.available():
        display.missing_database("Passages are sampled from the indexed corpus.")
        return 0

    try:
        conn = pg.connect()
        pg.check_schema(conn)
    except pg.DatabaseUnavailable as error:
        print(f"\n{error}\n\n{error.remedy}\n")
        return 1

    embedder = get_embedder()

    # --- step 1: passage selection ----------------------------------------
    display.heading("Step 1: select passages")
    rows = conn.execute(
        """
        SELECT id, body, story FROM chunks
        WHERE strategy = %s AND length(body) > 600
        ORDER BY random() LIMIT %s
        """,
        [args.strategy, args.passages],
    ).fetchall()

    print(f"  {len(rows)} passages sampled, from "
          f"{len({r[2] for r in rows})} stories")
    print("\n  Sampling matters more than it looks. Take the first N chunks and")
    print("  the evaluation set covers one story; sample by length and it")
    print("  covers only the descriptive passages. Stratifying by document is")
    print("  usually the right default.")

    # --- steps 2 and 3: generate ------------------------------------------
    display.heading("Steps 2 and 3: generate questions and answers")

    if not llm.available():
        print(f"  No LLM: {llm.why_unavailable()}\n")
        display.prompt_block("SYSTEM", GEN_SYSTEM)
        display.prompt_block(
            "USER",
            GEN_PROMPT.format(n=args.per_passage,
                              passage=display.truncate(rows[0][1], 500)),
        )
        display.illustrative(
            '{"items": [',
            '  {"question": "What did Holmes deduce from the hat?",',
            '   "answer": "That its owner was intellectual and had fallen on hard times.",',
            '   "evidence": "He is a man who leads a sedentary life..."}',
            ']}',
        )
        print()
        display.para(
            "The filters below are the interesting part and they need no "
            "model, so they are run against the hand-written gold set instead "
            "-- which also checks that the gold set would survive its own "
            "quality gate."
        )
        gold = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["questions"]
        candidates = [
            {"question": g["question"], "answer": g["answer"],
             "evidence": g["evidence"], "passage": ""}
            for g in gold
        ]

        # Three deliberately malformed items, so each filter visibly fires.
        # Written by hand to match the failure modes a generator actually
        # produces -- these are not hypothetical, they are what comes back
        # when you generate a few hundred questions without a quality gate.
        candidates += [
            {
                # Self-answering: retrievable by a system that retrieves nothing.
                "question": "What was the swamp adder that Holmes identified as the speckled band?",
                "answer": "The swamp adder was the speckled band.",
                "evidence": "It is a swamp adder",
                "passage": "",
            },
            {
                # Quoted evidence that is a paraphrase, so it appears nowhere
                # in the corpus and can never be scored.
                "question": "Which snake did Holmes name?",
                "answer": "A swamp adder.",
                "evidence": "Holmes declared that the creature was a swamp adder from India",
                "passage": "It is a swamp adder! cried Holmes; the deadliest snake in India.",
            },
            {
                # A near-duplicate of gold question q19.
                "question": "What exactly was the speckled band?",
                "answer": "A swamp adder.",
                "evidence": "It is a swamp adder",
                "passage": "",
            },
        ]
        print(f"\n  Added 3 deliberately malformed items so each filter fires.")
        source = "hand-written gold set + 3 planted failures"
    else:
        if llm.STUB:
            display.stub_warning()
        else:
            print(f"  model: {config.OPENAI_CHAT_MODEL}")
        candidates = []
        for index, (chunk_id, body, story) in enumerate(rows, start=1):
            print(f"    generating {index}/{len(rows)}", end="\r", flush=True)
            reply = llm.complete_json(
                GEN_PROMPT.format(n=args.per_passage, passage=body),
                system=GEN_SYSTEM,
                purpose="synthetic evaluation data",
            )
            for item in reply.get("items", []):
                if isinstance(item, dict):
                    candidates.append({**item, "passage": body, "story": story})
        print(" " * 40, end="\r")
        source = "generated"

    print(f"\n  {len(candidates)} candidate items ({source})")

    # --- step 4: filter ----------------------------------------------------
    display.heading("Step 4: filter")

    surviving = []
    reasons: dict[str, int] = {}

    for item in candidates:
        passage = item.get("passage", "")

        if passage:
            ok, reason = filter_evidence_present(item, passage)
            if not ok:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue

        ok, reason = filter_self_answering(item)
        if not ok:
            reasons["self-answering"] = reasons.get("self-answering", 0) + 1
            continue

        surviving.append(item)

    deduped, duplicates = filter_duplicates(surviving, embedder)
    if duplicates:
        reasons["near-duplicate"] = duplicates

    display.table(
        ["filter", "removed"],
        [[reason, count] for reason, count in sorted(reasons.items())] or [["-", 0]],
    )
    display.kv("candidates in", len(candidates))
    display.kv("surviving", len(deduped))
    if candidates:
        display.kv("pass rate", f"{100 * len(deduped) / len(candidates):.0f}%")

    if reasons:
        print("\n  Every filtered item would have become a permanent, invisible")
        print("  distortion in every score computed against this set.")

    # --- cross-model consistency ------------------------------------------
    display.heading("Cross-checking (the book's consistency step)")
    if llm.available() and not llm.STUB and deduped:
        checked = 0
        rejected = 0
        for item in deduped[:5]:
            verdict = llm.complete_json(
                VERIFY_PROMPT.format(
                    passage=item.get("passage", "")[:1500],
                    question=item["question"],
                    answer=item["answer"],
                ),
                system=VERIFY_SYSTEM,
                purpose="synthetic data verification",
            )
            checked += 1
            if not verdict.get("answerable_from_passage") or verdict.get("self_answering"):
                rejected += 1
        display.kv("verified", checked)
        display.kv("rejected on second pass", rejected)
        print("\n  A second model pass catches what the structural filters")
        print("  cannot -- questions that are well-formed and still not")
        print("  answerable from the passage they were generated from.")
    else:
        display.prompt_block("VERIFIER SYSTEM", VERIFY_SYSTEM)
        print("\n  The generator and the verifier should ideally be different")
        print("  models. Asking one model to check its own output gets you")
        print("  agreement rather than verification -- the same reason example")
        print("  27's critic reads the retrieved context rather than the")
        print("  model's own answer.")

    # --- sample ------------------------------------------------------------
    if deduped:
        display.heading("Surviving items")
        display.table(
            ["question", "answer"],
            [[display.truncate(i["question"], 40), display.truncate(i["answer"], 34)]
             for i in deduped[:6]],
            align="ll",
        )

    if args.out and deduped:
        Path(args.out).write_text(
            json.dumps({"questions": deduped}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  Wrote {len(deduped)} items to {args.out}")

    display.notice(
        "Generation is the easy half. Filtering is what separates an "
        "evaluation set from a pile of plausible strings, and the book lists "
        "it as a step for good reason.",
        "A self-answering question is the worst failure here, because it "
        "inflates every score it appears in and looks entirely reasonable to "
        "a human skimming the set.",
        "Quoted evidence must be verified against the source. A paraphrased "
        "quote produces a gold item that no retrieval can ever satisfy.",
        "Synthetic questions inherit the generator's blind spots. They are "
        "good for breadth and regression testing; a hand-written set is still "
        "what you calibrate against.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
