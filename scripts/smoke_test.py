#!/usr/bin/env python3
"""
Run every example and report which ones work.

Three things are being checked, and the third is the one that matters:

1. Examples exit 0.
2. Examples that need something missing (a database, a key, torch) say so and
   exit 0 anyway -- a reader who has installed only the core requirements
   should get an explanation, never a traceback.
3. No example crashes.

    python scripts/smoke_test.py              # everything
    python scripts/smoke_test.py --no-db      # pretend DATABASE_URL is unset
    python scripts/smoke_test.py --stub-llm   # exercise the LLM code paths

A skip is a pass, which is right for a reader and wrong for CI: a run where
everything declines would be green while proving nothing. --fail-on-skip turns
skips back into failures, for the CI job that has a database and a stub LLM and
therefore expects all 38 to actually run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((REPO / "examples").glob("[0-9][0-9]_*.py"))

# Arguments that keep the slow examples quick without changing what they test.
FAST_ARGS = {
    "10_chunk_context_buffered": ["--limit", "4"],
    "11_chunk_question_derived": ["--limit", "4"],
    "12_chunk_question_anchored": ["--limit", "2"],
    "13_chunk_qa_context_buffered": ["--limit", "2"],
    "14_chunk_dual_index": ["--limit", "10"],
    "26_knowledge_graph_rag": [],
    "33_synthetic_eval_data": ["--passages", "3"],
    "36_monitoring": ["--queries", "12"],
}

# Phrases that mean "this example declined to run, on purpose".
SKIP_MARKERS = (
    "which is not installed",
    "which is not set",
    "DATABASE_URL is not set",
    "No LLM configured",
    "Nothing indexed",
    "Corpus not found",
    "Gold set missing",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--stub-llm", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--only", default="", help="substring filter")
    parser.add_argument(
        "--fail-on-skip", action="store_true",
        help="treat a clean decline as a failure (for CI, where everything should run)",
    )
    parser.add_argument(
        "--python", default="",
        help="interpreter to run the examples with (default: the repo .venv, else this one)",
    )
    args = parser.parse_args()

    env = dict(os.environ)
    if args.no_db:
        env["DATABASE_URL"] = ""
    if args.stub_llm:
        env["RAGKIT_STUB_LLM"] = "1"

    python = args.python or str(REPO / ".venv" / "bin" / "python")
    if not Path(python).exists():
        python = sys.executable

    targets = [p for p in EXAMPLES if args.only in p.stem]

    print(f"\nRunning {len(targets)} examples")
    print(f"  database: {'disabled' if args.no_db else 'as configured'}")
    print(f"  llm: {'stubbed' if args.stub_llm else 'as configured'}\n")

    passed, skipped, failed = [], [], []

    for path in targets:
        name = path.stem
        command = [python, str(path)] + FAST_ARGS.get(name, [])
        started = time.time()
        try:
            result = subprocess.run(
                command, capture_output=True, text=True,
                timeout=args.timeout, env=env, cwd=REPO,
            )
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT  {name}")
            failed.append((name, "timed out"))
            continue

        elapsed = time.time() - started
        output = result.stdout + result.stderr

        if result.returncode != 0:
            tail = [l for l in output.strip().split("\n") if l.strip()][-1:]
            print(f"  FAIL     {name}  ({elapsed:.0f}s)")
            failed.append((name, tail[0] if tail else f"exit {result.returncode}"))
        elif any(marker in output for marker in SKIP_MARKERS):
            reason = next(m for m in SKIP_MARKERS if m in output)
            print(f"  skip     {name}  ({reason})")
            skipped.append((name, reason))
        else:
            print(f"  ok       {name}  ({elapsed:.0f}s)")
            passed.append(name)

    skip_note = "counted as a failure" if args.fail_on_skip else "declined cleanly -- this is a pass"

    print(f"\n{'=' * 60}")
    print(f"  ran      {len(passed)}")
    print(f"  skipped  {len(skipped)}   ({skip_note})")
    print(f"  failed   {len(failed)}")

    if failed:
        print("\nFailures:")
        for name, reason in failed:
            print(f"  {name}: {reason[:120]}")
        return 1

    if args.fail_on_skip and skipped:
        print("\nSkipped, but --fail-on-skip was set:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
        print("\nThis configuration is meant to run every example. Something it "
              "needs -- the database, the corpus index, the stub LLM -- is missing.\n")
        return 1

    print("\nAll examples either ran or declined cleanly.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
