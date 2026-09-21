#!/usr/bin/env python3
"""
32. Generation Metrics
Book: Chapter 19, Evaluation Metrics
WHAT THIS SHOWS: Measuring the answer rather than the retrieval, and why the
classic text-overlap metrics are close to useless for RAG.

BLEU and ROUGE ask "does this answer share wording with a reference answer?"
That was a reasonable proxy for translation and summarisation. For RAG it is
actively misleading: a correct answer phrased differently scores near zero,
and a fluent hallucination that reuses the question's vocabulary scores well.

What actually matters is **groundedness** -- is every claim in the answer
supported by the retrieved context? That can be approximated without a model
at all, and this example does both so you can see how far the cheap version
gets.

HOW THIS SCRIPT PROCEEDS
    1. Take four answers with KNOWN properties    including one hallucination
    2. Score them with BLEU and ROUGE-L           both written out
    3. Score them for groundedness against the context
    4. Show WHICH SENTENCE the hallucination lives in
    5. Show what the cheap metric misses
    6. The LLM-as-judge alternative

Canned answers, deliberately. The point is to see how each metric responds to
a known failure, which you cannot do with whatever a live model happens to say.


THE RESULT THIS FILE EXISTS TO SHOW
    BLEU and ROUGE rank the fluent hallucination ABOVE the correct paraphrase.
    Not marginally -- ROUGE-L gives the hallucination 0.545 and the correct
    answer 0.129, because the hallucination reuses the reference's sentence
    structure while the correct one uses different words.

    Any metric that does that is not measuring correctness.


REQUIRES: nothing (works on canned answers). Postgres + OPENAI_API_KEY to
score a live pipeline.
RUN: python examples/32_generation_metrics.py
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit import display, llm
from ragkit.corpus import split_sentences

# Four answers to one question, with known properties. Canned deliberately:
# the point is to see how each metric responds to a known failure, which you
# cannot do with whatever a live model happens to produce.
QUESTION = "What was the speckled band?"

CONTEXT = (
    "It is a swamp adder! cried Holmes; the deadliest snake in India. "
    "He has died within ten seconds of being bitten. Violence does, in "
    "truth, recoil upon the violent, and the schemer falls into the pit "
    "which he digs for another. The snake had been trained to return to "
    "the room through the ventilator when the whistle sounded."
)

REFERENCE = "The speckled band was a swamp adder, the deadliest snake in India."

ANSWERS = {
    "correct, same wording": (
        "The speckled band was a swamp adder, the deadliest snake in India."
    ),
    "correct, different wording": (
        "It turned out to be a venomous Indian serpent that Dr. Roylott had "
        "trained to pass through the ventilator."
    ),
    "hallucinated but fluent": (
        "The speckled band was a rare tropical spider, the deadliest arachnid "
        "in India, which Dr. Roylott kept in a locked case."
    ),
    "correct but evasive": (
        "The retrieved passages do not fully specify the creature's exact "
        "species, though a snake is mentioned."
    ),
}

_WORD = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# ---------------------------------------------------------------------------
# The classic overlap metrics
# ---------------------------------------------------------------------------


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """BLEU with a brevity penalty.

    Geometric mean of clipped n-gram precisions. "Clipped" means a candidate
    cannot score by repeating one correct word many times -- each n-gram
    counts at most as often as it appears in the reference.
    """
    cand = tokenize(candidate)
    ref = tokenize(reference)
    if not cand or not ref:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(tuple(cand[i:i + n]) for i in range(len(cand) - n + 1))
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum(min(c, ref_ngrams[g]) for g, c in cand_ngrams.items())
        precisions.append(overlap / sum(cand_ngrams.values()))

    if min(precisions) == 0:
        return 0.0

    # Brevity penalty: without it, a one-word answer that happens to be right
    # scores a perfect 1.0.
    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L: F-measure over the longest common subsequence.

    Subsequence, not substring -- the matched words need not be adjacent, so
    this tolerates insertions that BLEU's n-grams do not.
    """
    cand = tokenize(candidate)
    ref = tokenize(reference)
    if not cand or not ref:
        return 0.0

    # Standard LCS dynamic program.
    table = [[0] * (len(ref) + 1) for _ in range(len(cand) + 1)]
    for i in range(1, len(cand) + 1):
        for j in range(1, len(ref) + 1):
            if cand[i - 1] == ref[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    lcs = table[len(cand)][len(ref)]
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------


def lexical_groundedness(answer: str, context: str) -> float:
    """What fraction of the answer's content words appear in the context?

    A crude proxy for "is this supported by the evidence", and a surprisingly
    useful one: a hallucinated answer introduces vocabulary that is not in the
    context, and this notices.

    It is fooled by paraphrase in the other direction -- a correct answer
    using synonyms scores low -- so it works as a cheap alarm, not a verdict.
    """
    stop = {
        "the", "a", "an", "was", "is", "were", "are", "of", "in", "to", "and",
        "that", "it", "which", "this", "be", "been", "had", "has", "have",
        "as", "for", "on", "with", "by", "from", "at", "or", "but", "not",
    }
    answer_words = [w for w in tokenize(answer) if w not in stop]
    context_words = set(tokenize(context))
    if not answer_words:
        return 0.0
    return sum(1 for w in answer_words if w in context_words) / len(answer_words)


def claim_support(answer: str, context: str) -> tuple[float, list[tuple[str, float]]]:
    """Per-sentence groundedness.

    Averaging over the whole answer hides the failure mode that matters: an
    answer that is 90% grounded and contains one invented sentence. Scoring
    each sentence separately surfaces exactly that sentence.
    """
    sentences = split_sentences(answer) or [answer]
    scored = [(s, lexical_groundedness(s, context)) for s in sentences]
    return (sum(v for _, v in scored) / len(scored), scored)


JUDGE_SYSTEM = (
    "You assess whether an answer is supported by the provided context. "
    "You reply with JSON and you do not use outside knowledge."
)

JUDGE_PROMPT = """Context:
{context}

Answer: {answer}

Is every factual claim in the answer supported by the context?

Reply as JSON:
{{"faithfulness": 0.0 to 1.0,
  "unsupported_claims": ["..."],
  "verdict": "one short sentence"}}"""


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    display.banner(
        "32. Generation Metrics",
        "Chapter 19, Evaluation Metrics",
        "BLEU, ROUGE-L and groundedness on four answers with known "
        "properties -- including one confident hallucination.",
    )

    print(f"  question: {QUESTION!r}")
    print(f"  reference: {REFERENCE!r}\n")

    display.heading("The four answers")
    for label, answer in ANSWERS.items():
        print(f"  {label}:")
        display.preview(answer, limit=150)
    print()

    display.heading("Overlap metrics vs groundedness")
    rows = []
    for label, answer in ANSWERS.items():
        grounded, _ = claim_support(answer, CONTEXT)
        rows.append([
            label,
            f"{bleu(answer, REFERENCE):.3f}",
            f"{rouge_l(answer, REFERENCE):.3f}",
            f"{grounded:.3f}",
        ])
    display.table(["answer", "BLEU", "ROUGE-L", "groundedness"], rows, align="lrrr")

    print("\n  Read the two middle rows against each other. The correct answer")
    print("  phrased differently scores near zero on BLEU and ROUGE, because it")
    print("  shares almost no wording with the reference. The hallucination")
    print("  reuses the reference's sentence structure and scores well.")
    print("\n  Any metric that ranks a fluent hallucination above a correct")
    print("  paraphrase is not measuring correctness. That is the case against")
    print("  overlap metrics for RAG, and it is not a subtle one.")

    display.heading("Where the hallucination actually shows up")
    hallucination = ANSWERS["hallucinated but fluent"]
    _, per_sentence = claim_support(hallucination, CONTEXT)
    display.table(
        ["sentence", "grounded"],
        [[display.truncate(s, 56), f"{v:.2f}"] for s, v in per_sentence],
        align="lr",
    )
    print("\n  'spider' and 'arachnid' are nowhere in the context, and the")
    print("  groundedness score notices without any model being involved.")
    print("  Per-sentence scoring matters here: an answer that is mostly")
    print("  grounded with one invented sentence averages out to 'fine'.")

    display.heading("What the cheap metric misses")
    paraphrase = ANSWERS["correct, different wording"]
    grounded, _ = claim_support(paraphrase, CONTEXT)
    print(f"  The correct paraphrase scores {grounded:.2f} on lexical")
    print("  groundedness -- low, despite being right. 'Venomous Indian")
    print("  serpent' is supported by 'swamp adder, the deadliest snake in")
    print("  India', but not one content word matches.")
    print("\n  That is the ceiling on string-matching. Catching entailment")
    print("  rather than overlap needs either an NLI model or an LLM judge.")

    display.heading("LLM-as-judge")
    if not llm.available():
        print(f"  No LLM: {llm.why_unavailable()}")
        print("\n  Everything above ran without one. The judge prompt would be:")
        display.prompt_block("SYSTEM", JUDGE_SYSTEM)
        display.prompt_block(
            "USER",
            JUDGE_PROMPT.format(
                context=display.truncate(CONTEXT, 300), answer=paraphrase
            ),
        )
        print("\n  Worth noting the cost shape: a judge call per answer, on every")
        print("  evaluation run. That is why the book recommends running it")
        print("  asynchronously over sampled traffic rather than inline.")
    else:
        if llm.STUB:
            display.stub_warning()
        for label, answer in ANSWERS.items():
            verdict = llm.complete_json(
                JUDGE_PROMPT.format(context=CONTEXT, answer=answer),
                system=JUDGE_SYSTEM,
                purpose="faithfulness judging",
            )
            print(f"\n  {label}:")
            display.kv("faithfulness", verdict.get("faithfulness"))
            display.kv("verdict", display.truncate(str(verdict.get("verdict", "")), 60))

    display.notice(
        "BLEU and ROUGE measure wording, not truth. In this example they rank "
        "a confident hallucination above a correct paraphrase, which is "
        "exactly backwards.",
        "Lexical groundedness catches invented vocabulary for free and is "
        "worth computing on every answer. It cannot recognise a correct "
        "paraphrase, so it is an alarm rather than a verdict.",
        "Score sentences, not answers. An answer that is 90% grounded with one "
        "fabricated sentence is a serious failure and a reassuring average.",
        "An LLM judge handles entailment but costs a call per answer and "
        "carries its own biases -- it prefers long, confident, familiar-"
        "sounding text. Calibrate it against human labels before trusting it.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
