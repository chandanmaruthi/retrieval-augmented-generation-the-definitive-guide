#!/usr/bin/env python3
"""
Build and verify data/eval_questions.json.

Thirty hand-written questions about the twelve stories. Each one carries an
``evidence`` string: a literal phrase from the corpus that a retrieved chunk
must contain for the retrieval to count as correct.

That design is what makes the chunking comparison in example 15 possible.
Scoring on chunk ids would be useless, because every strategy produces
different chunks -- there is no shared id space. Scoring on a substring works
for all thirteen strategies without favouring any of them.

This script does not just write the file, it *checks* it: every evidence
string must appear in the corpus, and it must appear in the story the question
says it does. Run it after any change to the corpus.

    python scripts/build_eval_set.py          # verify and write
    python scripts/build_eval_set.py --check  # verify only, non-zero on failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit.corpus import load_book

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"

# (id, question, story key, answer, evidence)
#
# `evidence` is matched case-sensitively as a substring of the normalised
# chunk text. Phrases were chosen to be distinctive: a chunk containing
# "It is a swamp adder" is unambiguously the passage that answers q19, while
# one containing "Mary" could be almost anywhere in the book.
QUESTIONS = [
    ("q01", "What was hidden behind the sliding panel in Irene Adler's sitting room?",
     "Scandal", "The photograph of the King with Irene Adler.",
     "photograph is in a recess behind a sliding panel"),
    ("q02", "How did Holmes trick Irene Adler into revealing the photograph's hiding place?",
     "Scandal", "He staged a fire alarm, knowing she would run to save it.",
     "woman thinks that her house is on fire"),
    ("q03", "Who did Irene Adler marry?",
     "Scandal", "Godfrey Norton, a lawyer of the Inner Temple.",
     "Godfrey Norton"),
    ("q04", "What does Watson say Holmes always calls Irene Adler?",
     "Scandal", "The woman.",
     "she is always the woman"),
    ("q05", "What work was Jabez Wilson paid to do for the Red-Headed League?",
     "Red-Headed", "Copying out the Encyclopaedia Britannica by hand.",
     "copy out the"),
    ("q06", "Why was the Red-Headed League created?",
     "Red-Headed", "To get Wilson out of his shop for hours each day so a tunnel could be dug.",
     "out of the way for a number of hours"),
    ("q07", "What were the thieves tunnelling towards?",
     "Red-Headed", "A bank vault holding French gold.",
     "French gold"),
    ("q08", "Who was the criminal behind the Red-Headed League?",
     "Red-Headed", "John Clay.",
     "John Clay"),
    ("q09", "Who was Hosmer Angel really?",
     "Case of Identity", "Mary Sutherland's stepfather, James Windibank, in disguise.",
     "James Windibank"),
    ("q10", "Why did Mary Sutherland's stepfather invent a suitor for her?",
     "Case of Identity", "To stop her marrying, so her income stayed in the household.",
     "hundred a year"),
    ("q11", "What did the murdered man say as he died at Boscombe Pool?",
     "Boscombe", "He made a dying reference to a rat.",
     "dying reference to a rat"),
    ("q12", "Who actually killed Charles McCarthy?",
     "Boscombe", "John Turner, whom McCarthy had been blackmailing.",
     "John Turner"),
    ("q13", "What did the letters K. K. K. stand for?",
     "Five Orange Pips", "The Ku Klux Klan.",
     "the Ku Klux Klan"),
    ("q14", "What was sent to the victims before each death?",
     "Five Orange Pips", "Five dried orange pips in an envelope.",
     "five dried orange pips"),
    ("q15", "What had become of Neville St. Clair?",
     "Twisted Lip", "He was living a second life as the beggar Hugh Boone.",
     "Hugh Boone"),
    ("q16", "How did Neville St. Clair make his money?",
     "Twisted Lip", "By begging in the City, which paid far better than his profession.",
     "begging"),
    ("q17", "Where was the blue carbuncle found?",
     "Blue Carbuncle", "In the crop of a goose.",
     "crop of a goose in Tottenham Court Road"),
    ("q18", "Who stole the blue carbuncle?",
     "Blue Carbuncle", "James Ryder, the hotel's upper-attendant.",
     "James Ryder, upper-attendant"),
    ("q19", "What was the speckled band?",
     "Speckled Band", "A swamp adder.",
     "It is a swamp adder"),
    ("q20", "What kind of snake killed Dr. Roylott?",
     "Speckled Band", "A swamp adder, the deadliest snake in India.",
     "the deadliest snake in India"),
    ("q21", "What did Dr. Roylott stand to lose if his stepdaughters married?",
     "Speckled Band", "Each daughter could claim an income on marriage.",
     "daughter can claim an income"),
    ("q22", "What machine was Victor Hatherley hired to examine?",
     "Engineer", "A hydraulic press.",
     "hydraulic press"),
    ("q23", "What was the hydraulic press really used for?",
     "Engineer", "Making counterfeit coins.",
     "coiners"),
    ("q24", "Who was the American gentleman staying at the hotel in the Noble Bachelor?",
     "Noble Bachelor", "Francis H. Moulton.",
     "Francis H. Moulton"),
    ("q25", "Who did Lord St. Simon marry?",
     "Noble Bachelor", "Hatty Doran, an American heiress.",
     "Hatty Doran"),
    ("q26", "Who took the beryl coronet?",
     "Beryl Coronet", "Sir George Burnwell.",
     "Sir George Burnwell"),
    ("q27", "Who was the niece living in the Holder household?",
     "Beryl Coronet", "Mary Holder.",
     "Mary Holder"),
    ("q28", "Why was Violet Hunter asked to cut her hair?",
     "Copper Beeches", "So she could impersonate Alice Rucastle.",
     "cut your hair"),
    ("q29", "Who was imprisoned at the Copper Beeches?",
     "Copper Beeches", "Alice Rucastle, by her father.",
     "Alice Rucastle"),
    ("q30", "What animal guarded the grounds at the Copper Beeches?",
     "Copper Beeches", "A mastiff, loosed at night.",
     "mastiff"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()

    book = load_book()
    # Normalise whitespace, because chunkers join wrapped lines and an
    # evidence phrase must match across what was once a line break.
    corpus = {story.title: " ".join(story.text.split()) for story in book.stories}

    records = []
    failures = []

    for qid, question, story_key, answer, evidence in QUESTIONS:
        matches = [title for title in corpus if story_key.lower() in title.lower()]
        if len(matches) != 1:
            failures.append(f"{qid}: story key {story_key!r} matched {len(matches)} stories")
            continue
        story = matches[0]

        in_gold = corpus[story].count(evidence)
        elsewhere = sum(
            count
            for title, count in ((t, corpus[t].count(evidence)) for t in corpus)
            if title != story and count
        )

        if in_gold == 0:
            failures.append(f"{qid}: evidence {evidence!r} not found in {story!r}")
            continue

        records.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "story": story,
                "evidence": evidence,
                "occurrences_in_story": in_gold,
                "occurrences_elsewhere": elsewhere,
            }
        )

    print(f"Verified {len(records)}/{len(QUESTIONS)} questions against the corpus")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    # Evidence that also appears outside its gold story is not wrong, but it
    # weakens the question, so report it rather than hide it.
    leaky = [r for r in records if r["occurrences_elsewhere"]]
    if leaky:
        print(f"\n{len(leaky)} question(s) whose evidence also appears in other stories:")
        for record in leaky:
            print(
                f"  {record['id']}: {record['evidence']!r} "
                f"(+{record['occurrences_elsewhere']} elsewhere)"
            )
        print("  These still score correctly -- scoring checks the phrase, not the story.")

    by_story: dict[str, int] = {}
    for record in records:
        by_story[record["story"]] = by_story.get(record["story"], 0) + 1
    print(f"\nCoverage: {len(by_story)} of {len(book.stories)} stories")

    if args.check:
        return 0

    OUT_PATH.write_text(
        json.dumps({"questions": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_PATH.relative_to(OUT_PATH.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
