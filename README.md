<div align="center">

<a href="https://www.amazon.com/dp/B0HFH2JNMK">
  <img src="docs/book-cover.jpg" width="190" alt="Retrieval-Augmented Generation: The Definitive Guide">
</a>

# RAG: The Definitive Guide — Runnable Examples

### The official code companion to the book<br>*Retrieval-Augmented Generation: The Definitive Guide*

**Every retrieval technique in the book, as a script you can run.**

[![Get the book on Amazon](https://img.shields.io/badge/Get%20the%20book-Amazon-black?style=for-the-badge&logo=amazon&logoColor=white)](https://www.amazon.com/dp/B0HFH2JNMK)

<sub>Revised and Expanded **Third Edition** &nbsp;·&nbsp; chapter numbers on this page follow it</sub>

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-black.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/chandanmaruthi/retrieval-augmented-generation-the-definitive-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/chandanmaruthi/retrieval-augmented-generation-the-definitive-guide/actions/workflows/ci.yml)
[![Stars](https://img.shields.io/github/stars/chandanmaruthi/retrieval-augmented-generation-the-definitive-guide?style=flat&color=black)](https://github.com/chandanmaruthi/retrieval-augmented-generation-the-definitive-guide/stargazers)

</div>

---

Thirty-eight examples over one real corpus — **The Adventures of Sherlock Holmes**,
104,000 words, shipped as a 189-page PDF — so that the strategies are comparable to each
other rather than each demonstrated on its own toy input. Thirty hand-written gold
questions turn "this technique is better" into a number you can reproduce.

```
$ python examples/19_hybrid_fusion.py

The scale problem
-----------------
  top dense score            0.6507
  top lexical score          0.0010
  ratio                      645x

Variant 1: linear weighting, raw scores (the wrong way)
-------------------------------------------------------
  rank     vec     txt   total  passage
  ----  ------  ------  ------  --------------------------------
     1  0.6507  0.0000  0.3254  It was a quarter-past nine when…
     2  0.6468  0.0010  0.3239  "Hum! So much for the police-co…

  The lexical arm contributed 0.03% of the total score.
  That is the bug, and it is invisible unless you look.
```

That 0.03% is the point of the repo. Most RAG code you will read does exactly what
Variant 1 does above, and calls it hybrid search.

---

## Start here

**Two ways in, depending on how you got here.**

<table>
<tr>
<td width="50%" valign="top">

### 📖 You're reading the book

Go straight to your chapter.

**[→ Index by book chapter](#index-by-book-chapter)**

Script numbers do **not** match chapter numbers, and never did — the scripts are ordered
by what they build on. Use the table rather than guessing.

Then run [Quickstart](#quickstart) once and come back.

</td>
<td width="50%" valign="top">

### 🔍 You found this on GitHub

Read the numbers first — they are the argument.

**[→ Results](#results)**, then
**[→ Known limits](#known-limits)**

The short version: the top-ranked chunking strategy drops from **1st to last** when you
control for context budget, and naive hybrid fusion is pure vector search wearing a hat.

Then [Quickstart](#quickstart) — eight of the examples run on three dependencies, with no
database, no API key and no model download.

</td>
</tr>
</table>

<details>
<summary><b>Full table of contents</b></summary>

- [Quickstart](#quickstart)
- [The book](#the-book)
- [Index by book chapter](#index-by-book-chapter)
- [The corpus](#the-corpus)
- [Results](#results)
- [The method index](#the-method-index)
- [Why Postgres](#why-postgres)
- [How it is laid out](#how-it-is-laid-out)
- [Cost](#cost)
- [Verification](#verification)
- [Known limits](#known-limits)
- [Fork this](#fork-this)
- [Attribution](#attribution)

</details>

---

## Quickstart

Python 3.10+ required.

```bash
git clone https://github.com/chandanmaruthi/retrieval-augmented-generation-the-definitive-guide.git
cd retrieval-augmented-generation-the-definitive-guide
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python examples/02_chunk_fixed_token.py
```

That works immediately. No database, no API key, no model download: **eight examples run
on the committed corpus** with the three core dependencies — 01–05, 07, 08 and 32. The
other thirty tell you what they need and exit cleanly.

**For the retrieval examples**, add local embeddings and Postgres:

```bash
pip install -r requirements-local.txt -r requirements-db.txt

cp .env.example .env             # already points at the local database
docker compose up -d             # Postgres 17 + pgvector, on port 5433
python scripts/init_db.py        # schema, indexes, row-level security
python scripts/index_corpus.py   # chunk, embed, load

python examples/19_hybrid_fusion.py
```

That is the whole setup. The database is stock `pgvector/pgvector:pg17` — nothing in this
repo depends on a hosted provider, and everything goes through `DATABASE_URL`, psycopg and
raw SQL. No ORM, because in the examples that need a database the SQL *is* the lesson.

**Optional, for the 5 examples marked ★:** put an `OPENAI_API_KEY` in `.env` and
`pip install -r requirements-openai.txt`. Everything else runs without it.

If you have `make`, the same thing in four lines:

```bash
make setup     # venv + all requirements
make db        # docker compose up, init schema
make index     # chunk, embed, load the corpus
make smoke     # run all 38 and report
```

<details>
<summary>Using a Postgres you already have</summary>

Any Postgres 15+ with pgvector available works — point `DATABASE_URL` at it and run
`scripts/init_db.py`. Postgres 15 is the floor because the `edges` table uses
`UNIQUE NULLS NOT DISTINCT`.

If the password contains punctuation, percent-encode it: `@`→`%40`, `&`→`%26`, `#`→`%23`.
An unencoded `#` starts a URL fragment and silently truncates the rest of the string.
`ragkit/pg.py` checks for this and says so rather than letting psycopg report a confusing
host.
</details>

---

## The book

<a href="https://www.amazon.com/dp/B0HFH2JNMK">
<img src="docs/book-cover.jpg" alt="Book cover" width="150" align="right">
</a>

### [Retrieval-Augmented Generation: The Definitive Guide](https://www.amazon.com/dp/B0HFH2JNMK)

**Chandan Maruthi** · Twig AI · Revised and Expanded Third Edition ·
[Read it on Amazon →](https://www.amazon.com/dp/B0HFH2JNMK)

Written from two years of building enterprise RAG systems in production. From the
preface:

> Over the past two years, the Twig team has been building enterprise-grade RAG systems
> for some of the most demanding production environments. Through this journey, we
> discovered a hard truth: the gap between a hackathon demo and a true enterprise
> deployment is vast. What looks impressive in a demo often breaks in production — due to
> missing steps in robust data ingestion, intelligent chunking, context retrieval, and
> agentic orchestration.

**This repo is the code. The book is the reasoning.** The scripts here show you *that*
parent-child chunking wins at fixed `k` and loses at fixed budget; the book covers the
six-block reference architecture the whole thing hangs off, when each of the ten retrieval
architectures earns its cost, and the production concerns — privacy, compliance,
monitoring, human-in-the-loop — that a benchmark script cannot express.

<details>
<summary><b>Every part and chapter</b></summary>

<!-- book-parts:begin -->
**26 chapters in 11 parts.**

| Part | Chapters |
|---|---|
| **I — RAG and the Reference Architecture** | 1 The Evolution of RAG · 2 Foundations of RAG Systems · 3 Reference Architecture |
| **II — Data Extraction** | 4 Data Extraction |
| **III — Chunking** | 5 Chunking Strategies |
| **IV — RAG Strategies** | 6 Baseline RAG Pipeline · 7 Context-Aware RAG · 8 Dynamic RAG · 9 Hybrid RAG · 10 Multi-Stage Retrieval · 11 Graph-Based RAG · 12 Hierarchical RAG · 13 Agentic RAG · 14 Multi-Agent RAG Systems · 15 Streaming RAG · 16 Choosing a Retrieval Architecture |
| **V — Memory and Content Management** | 17 Memory-Augmented RAG · 18 Knowledge Graph Integration |
| **VI — Evaluation** | 19 Evaluation Metrics · 20 Synthetic Data Generation |
| **VII — Fine-Tuning** | 21 Domain-Specific Fine-Tuning |
| **VIII — Security** | 22 Privacy & Compliance in RAG |
| **IX — Production** | 23 Real-Time Evaluation & Monitoring · 24 Human-in-the-Loop RAG |
| **X — Twig RAG Strategies** | 25 RAG Strategies in Twig |
| **XI — Conclusion** | 26 Conclusion & Future Directions |
<!-- book-parts:end -->

</details>

**The repo stands on its own.** Every example explains the technique in its own comments
and ends with a `WHAT TO NOTICE` block. You do not need the book to run any of this — but
the examples deliberately do not re-argue the book's case, so where a script says *"this
is why the naive version is wrong,"* the chapter is where the argument lives.

---

## Index by book chapter

Reading a chapter? This is your lookup.

**Legend:** ○ nothing · ◐ local model (`requirements-local.txt`) · ▣ Postgres · ★ `OPENAI_API_KEY`

<!-- book-index:begin -->
| Book chapter | Run this | Needs |
|---|---|---|
| **3** — Reference Architecture | [`16_index_pgvector.py`](examples/16_index_pgvector.py) · [`38_reference_architecture.py`](examples/38_reference_architecture.py) | ▣ |
| **4** — Data Extraction | [`01_extract_pdf.py`](examples/01_extract_pdf.py) | ○ |
| **5** — Chunking Strategies | [`02_chunk_fixed_token.py`](examples/02_chunk_fixed_token.py) · [`03_chunk_sentence.py`](examples/03_chunk_sentence.py) · [`04_chunk_paragraph.py`](examples/04_chunk_paragraph.py) · [`05_chunk_heading.py`](examples/05_chunk_heading.py) · [`06_chunk_semantic.py`](examples/06_chunk_semantic.py) · [`07_chunk_sentence_window.py`](examples/07_chunk_sentence_window.py) · [`08_chunk_parent_child.py`](examples/08_chunk_parent_child.py) · [`09_chunk_contextual_header.py`](examples/09_chunk_contextual_header.py) · [`10_chunk_context_buffered.py`](examples/10_chunk_context_buffered.py) · [`11_chunk_question_derived.py`](examples/11_chunk_question_derived.py) · [`12_chunk_question_anchored.py`](examples/12_chunk_question_anchored.py) · [`13_chunk_qa_context_buffered.py`](examples/13_chunk_qa_context_buffered.py) · [`14_chunk_dual_index.py`](examples/14_chunk_dual_index.py) · [`15_compare_chunking.py`](examples/15_compare_chunking.py) | ○ ◐ ★ |
| **6** — Baseline RAG Pipeline | [`17_baseline_rag.py`](examples/17_baseline_rag.py) | ▣ ★ |
| **7** — Context-Aware RAG | [`22_context_aware_rag.py`](examples/22_context_aware_rag.py) | ▣ ★ |
| **8** — Dynamic RAG | [`23_dynamic_rag.py`](examples/23_dynamic_rag.py) | ▣ ★ |
| **9** — Hybrid RAG | [`18_sparse_lexical.py`](examples/18_sparse_lexical.py) · [`19_hybrid_fusion.py`](examples/19_hybrid_fusion.py) | ▣ |
| **10** — Multi-Stage Retrieval | [`20_multistage_rerank.py`](examples/20_multistage_rerank.py) · [`21_late_interaction.py`](examples/21_late_interaction.py) | ▣ ◐ |
| **11** — Graph-Based RAG | [`25_graph_rag.py`](examples/25_graph_rag.py) | ▣ |
| **12** — Hierarchical RAG | [`24_hierarchical_rag.py`](examples/24_hierarchical_rag.py) | ▣ |
| **13** — Agentic RAG | [`27_agentic_rag.py`](examples/27_agentic_rag.py) | ▣ ★ |
| **14** — Multi-Agent RAG Systems | [`28_multi_agent_rag.py`](examples/28_multi_agent_rag.py) | ▣ ★ |
| **15** — Streaming RAG | [`29_streaming_rag.py`](examples/29_streaming_rag.py) | ▣ |
| **17** — Memory-Augmented RAG | [`30_memory_rag.py`](examples/30_memory_rag.py) | ▣ |
| **18** — Knowledge Graph Integration | [`26_knowledge_graph_rag.py`](examples/26_knowledge_graph_rag.py) | ▣ ★ |
| **19** — Evaluation Metrics | [`31_retrieval_metrics.py`](examples/31_retrieval_metrics.py) · [`32_generation_metrics.py`](examples/32_generation_metrics.py) | ▣ ▣ ★ |
| **20** — Synthetic Data Generation | [`33_synthetic_eval_data.py`](examples/33_synthetic_eval_data.py) | ★ |
| **21** — Domain-Specific Fine-Tuning | [`34_finetuning_data_prep.py`](examples/34_finetuning_data_prep.py) | ▣ |
| **22** — Privacy & Compliance in RAG | [`35_privacy_acl.py`](examples/35_privacy_acl.py) | ▣ |
| **23** — Real-Time Evaluation & Monitoring | [`36_monitoring.py`](examples/36_monitoring.py) | ▣ |
| **24** — Human-in-the-Loop RAG | [`37_human_in_the_loop.py`](examples/37_human_in_the_loop.py) | ▣ |
<!-- book-index:end -->

Chapters **1–3**, **25** and **26** — the author, the history of RAG, the foundations, the
Twig product chapter and the conclusion — are narrative and have no script.

---

## The corpus

**The Adventures of Sherlock Holmes**, Arthur Conan Doyle, from
[Project Gutenberg #1661](https://www.gutenberg.org/ebooks/1661). Public domain.

| | |
|---|---|
| Words | 104,346 |
| Documents | 12 stories |
| PDF | 189 pages, committed to the repo |
| Structure | book → story → part → paragraph → sentence |

It was chosen for reasons that matter to the examples:

- **Real nested structure.** The book's Hierarchical RAG chapter describes a
  `Topic → Doc → Section → Sentence` hierarchy. This corpus has one, so hierarchical and
  parent-child chunking operate on the document's own boundaries rather than on clusters
  invented by k-means.
- **A known cast.** Thirty-five recurring characters make the graph examples verifiable —
  you can check whether an extracted edge is true.
- **Genuine extraction problems.** Page numbers, wrapped lines, paragraphs split across
  page boundaries, nested quotation. Example 01 solves real ones.
- **Answerable questions with unambiguous answers**, which is what the gold set needs.

`data/eval_questions.json` holds 30 questions covering all 12 stories. Each carries a
literal phrase from the corpus that a retrieved chunk must contain for the retrieval to
count — so every strategy is scored the same way, with no model in the loop.
`scripts/build_eval_set.py` verifies all 30 against the source text.

---

## Results

Real numbers, reproduced by `python examples/15_compare_chunking.py`.

### All eight free chunking strategies, 30 gold questions

| Strategy | Book § | Chunks | Recall@5 | MRR | Context tokens |
|---|---|---|---|---|---|
| parent_child | 3.7 | 391 | **0.57** | 0.347 | 5,418 |
| contextual_header | 3.8 | 413 | 0.50 | 0.245 | 1,471 |
| heading | 3.4 | 413 | 0.43 | 0.248 | 1,480 |
| sentence | 3.2 | 589 | 0.40 | 0.207 | 1,201 |
| fixed_token | 3.1 | 581 | 0.37 | 0.202 | 1,264 |
| semantic | 3.5 | 782 | 0.27 | 0.149 | 854 |
| sentence_window | 3.6 | 6,665 | 0.27 | 0.142 | 793 |
| paragraph | 3.3 | 1,215 | 0.13 | 0.065 | 614 |

`parent_child` wins. It also delivers **8.8× more text per query** than the strategy at the
bottom, so some of that win is simply volume.

### The same strategies at an equal 2,000-token context budget

| Strategy | Recall | Rank change |
|---|---|---|
| contextual_header | **0.50** | +1 |
| fixed_token | 0.50 | +3 |
| heading | 0.47 | — |
| semantic | 0.43 | +2 |
| sentence_window | 0.43 | +2 |
| sentence | 0.40 | −2 |
| paragraph | 0.27 | +1 |
| parent_child | 0.23 | **−7** |

`parent_child` goes from first to last. At fixed `k` it was being rewarded for returning
more text; once every strategy may spend the same context budget, its large chunks mean it
fits only 2.1 of them and it loses.

Comparing retrieval strategies at fixed `k` without checking how much text each one
delivers is one of the easiest ways to reach a confident wrong conclusion. That is why
both tables are here.

### Other measured findings

| | |
|---|---|
| Naive linear hybrid fusion | lexical arm contributes **0.03%** of the score — the ranking is identical to pure vector search |
| Cross-encoder reranking | promotes results from ranks **43, 38 and 29** into the top 5 |
| ANN indexes at 413 vectors | exact scan 0.43 ms / recall 1.00; IVFFlat 0.27 ms / recall **0.68** |
| Contextual headers | **+0.199** cosine similarity for matching chunks, +0.001 for unrelated ones |
| Failure attribution | **100%** of failures are retrieval — the evidence is indexed, just not in the top 5 |

> **If the rank reversal above was news to you, ⭐ the repo** — it is the cheapest way to
> find it again when you are benchmarking your own chunker at 2am. Better still,
> [run it on your own corpus](#fork-this) and tell us whether the ordering holds.

---

## The method index

Every technique in the book, the script that implements it, and what it needs. Sorted by
example number — for the reverse lookup, see [index by book chapter](#index-by-book-chapter).

**Legend:** ○ nothing · ◐ local model (`requirements-local.txt`) · ▣ Postgres · ★ `OPENAI_API_KEY`

### Extraction and chunking

| # | Script | Book | Needs |
|---|---|---|---|
| 01 | [`01_extract_pdf.py`](examples/01_extract_pdf.py) — reading order, page furniture, normalisation, dedup | Data Extraction | ○ |
| 02 | [`02_chunk_fixed_token.py`](examples/02_chunk_fixed_token.py) | §3.1 | ○ |
| 03 | [`03_chunk_sentence.py`](examples/03_chunk_sentence.py) | §3.2 | ○ |
| 04 | [`04_chunk_paragraph.py`](examples/04_chunk_paragraph.py) | §3.3 | ○ |
| 05 | [`05_chunk_heading.py`](examples/05_chunk_heading.py) | §3.4 | ○ |
| 06 | [`06_chunk_semantic.py`](examples/06_chunk_semantic.py) — embedding-similarity boundaries | §3.5 | ◐ |
| 07 | [`07_chunk_sentence_window.py`](examples/07_chunk_sentence_window.py) | §3.6 | ○ |
| 08 | [`08_chunk_parent_child.py`](examples/08_chunk_parent_child.py) | §3.7 | ○ |
| 09 | [`09_chunk_contextual_header.py`](examples/09_chunk_contextual_header.py) — *measures* the benefit | §3.8 | ◐ |
| 10 | [`10_chunk_context_buffered.py`](examples/10_chunk_context_buffered.py) | §3.9 | ★ |
| 11 | [`11_chunk_question_derived.py`](examples/11_chunk_question_derived.py) | §3.10 | ★ |
| 12 | [`12_chunk_question_anchored.py`](examples/12_chunk_question_anchored.py) | §3.11 | ★ |
| 13 | [`13_chunk_qa_context_buffered.py`](examples/13_chunk_qa_context_buffered.py) — the canonical chunk | §3.12 | ★ |
| 14 | [`14_chunk_dual_index.py`](examples/14_chunk_dual_index.py) | §3.13 | ★ |
| 15 | [`15_compare_chunking.py`](examples/15_compare_chunking.py) — **all of them, scored** | Chunking Strategies | ◐ |

### Indexing and retrieval

| # | Script | Book | Needs |
|---|---|---|---|
| 16 | [`16_index_pgvector.py`](examples/16_index_pgvector.py) — HNSW vs IVFFlat vs no index, measured | Reference Architecture (block 3, Indexing) | ▣ |
| 17 | [`17_baseline_rag.py`](examples/17_baseline_rag.py) — the five-step pipeline | Baseline RAG Pipeline | ▣ |
| 18 | [`18_sparse_lexical.py`](examples/18_sparse_lexical.py) — and why `ts_rank` is not BM25 | Hybrid RAG | ▣ |
| 19 | [`19_hybrid_fusion.py`](examples/19_hybrid_fusion.py) — **linear vs RRF, in SQL** | Hybrid RAG | ▣ |
| 20 | [`20_multistage_rerank.py`](examples/20_multistage_rerank.py) — cross-encoder + MMR | Multi-Stage Retrieval | ▣ ◐ |
| 21 | [`21_late_interaction.py`](examples/21_late_interaction.py) — ColBERT-style MaxSim | Multi-Stage Retrieval | ▣ ◐ |
| 22 | [`22_context_aware_rag.py`](examples/22_context_aware_rag.py) — conversational query rewriting | Context-Aware RAG | ▣ |
| 23 | [`23_dynamic_rag.py`](examples/23_dynamic_rag.py) — adaptive k, confidence gating | Dynamic RAG | ▣ |
| 24 | [`24_hierarchical_rag.py`](examples/24_hierarchical_rag.py) — coarse to fine | Hierarchical RAG | ▣ |
| 25 | [`25_graph_rag.py`](examples/25_graph_rag.py) — k-hop traversal in a recursive CTE | Graph-Based RAG | ▣ |
| 26 | [`26_knowledge_graph_rag.py`](examples/26_knowledge_graph_rag.py) — typed triples, three fusion modes | Knowledge Graph Integration | ▣ |
| 27 | [`27_agentic_rag.py`](examples/27_agentic_rag.py) — plan, retrieve, criticise, repeat | Agentic RAG | ▣ |
| 28 | [`28_multi_agent_rag.py`](examples/28_multi_agent_rag.py) — parallel agents, blackboard, budgets | Multi-Agent RAG Systems | ▣ |
| 29 | [`29_streaming_rag.py`](examples/29_streaming_rag.py) — upserts, TTL, recency prior | Streaming RAG | ▣ |
| 30 | [`30_memory_rag.py`](examples/30_memory_rag.py) — short-term buffer + long-term store | Memory-Augmented RAG | ▣ |

### Evaluation and operations

| # | Script | Book | Needs |
|---|---|---|---|
| 31 | [`31_retrieval_metrics.py`](examples/31_retrieval_metrics.py) — Recall@k, MRR, NDCG from scratch | Evaluation Metrics | ▣ |
| 32 | [`32_generation_metrics.py`](examples/32_generation_metrics.py) — BLEU, ROUGE, groundedness | Evaluation Metrics | ○ |
| 33 | [`33_synthetic_eval_data.py`](examples/33_synthetic_eval_data.py) — generation **and filtering** | Synthetic Data Generation | ▣ |
| 34 | [`34_finetuning_data_prep.py`](examples/34_finetuning_data_prep.py) — hard negative mining | Domain-Specific Fine-Tuning | ▣ |
| 35 | [`35_privacy_acl.py`](examples/35_privacy_acl.py) — **RLS, redaction, cascading deletion** | Privacy & Compliance in RAG | ▣ |
| 36 | [`36_monitoring.py`](examples/36_monitoring.py) — grounding rate, drift, the 15% alert | Real-Time Evaluation & Monitoring | ▣ |
| 37 | [`37_human_in_the_loop.py`](examples/37_human_in_the_loop.py) — uncertainty sampling | Human-in-the-Loop RAG Systems | ▣ |
| 38 | [`38_reference_architecture.py`](examples/38_reference_architecture.py) — **failure attribution** | Reference Architecture | ▣ |

Most examples take arguments — `--help` works on 36 of the 38. The common shape is a
positional query plus `--k` and `--strategy`:

```bash
python examples/17_baseline_rag.py "who was the king of Bohemia" --k 8
python examples/19_hybrid_fusion.py --strategy parent_child
python examples/15_compare_chunking.py --budget 2000
```

---

## Why Postgres

One database covers what would otherwise need four separate libraries — a vector index, a
full-text engine, a graph store and an access-control layer. Every chapter maps onto
something Postgres already has:

| Book chapter | Postgres feature |
|---|---|
| Hybrid RAG | `pgvector` cosine + `ts_rank` full text, fused in one query |
| Multi-Stage Retrieval | HNSW and IVFFlat — the exact index types Chapter 11 names |
| Privacy & Compliance | Row-Level Security — ACLs enforced *in the index*, which is what the book demands |
| Streaming RAG | incremental `UPSERT`, TTL, `score = sim − λ·age_hours` as SQL |
| Graph-Based RAG | recursive CTEs for k-hop traversal |

The queries live in [`sql/queries/`](sql/queries/) as readable `.sql` files so you can run
them in `psql` and `EXPLAIN ANALYZE` them.

---

## How it is laid out

```
ragkit/        small shared core — corpus loading, embedding, Postgres, output
examples/      the techniques, one per file, heavily commented
sql/           schema, indexes, RLS policies, the fusion queries
scripts/       corpus build, db init, indexing, eval set, smoke test
data/          the PDF, the text, the gold questions
```

`ragkit/` deliberately contains **no retrieval techniques**. It loads the corpus, embeds
text, talks to the database and formats tables — the plumbing that would otherwise be
copy-pasted 38 times. Every method lives in `examples/`, in the open. The rule: if you have
to open `ragkit/` to understand how a technique works, the technique is in the wrong place.

Each example follows one shape: a header saying which chapter it comes from and what it
needs, then the technique with its reasoning in the comments, then a **WHAT TO NOTICE**
block tying the output back to the book's claim.

---

## Cost

| | |
|---|---|
| Free, offline, no account | 33 of the 38 examples — everything except the ★ rows |
| One-time ~2GB download | PyTorch, for local embeddings (`requirements-local.txt`) |
| Local Postgres | free, via Docker |
| Costs money | the 5 examples marked ★ (the LLM-assisted chunking strategies, 10–14) |

Several other examples use a model when one is configured and do something useful without
one — example 17 still runs the full retrieval and prints the prompt it assembled; example
27 runs its loop with hand-written sub-questions.

LLM responses are cached to `.cache/llm.db`, so re-running a script while reading costs
nothing the second time.

Examples that need something you have not installed **explain what to install and exit
cleanly**. They never raise a traceback at you, and the ones needing an API key print the
exact prompt they would have sent — which for the LLM-assisted chunking strategies is most
of the technique.

---

## Verification

```bash
python scripts/smoke_test.py             # everything you have configured
python scripts/smoke_test.py --no-db     # what a reader without Postgres sees
python scripts/smoke_test.py --stub-llm  # exercises every LLM code path
```

Zero failures in every configuration. "Declined cleanly" means the example detected
something missing, explained what to install, and exited 0 — which is the behaviour being
tested, not a gap in coverage.

| Configuration | Ran | Declined cleanly | Failed |
|---|---|---|---|
| **Postgres + live OpenAI key** | **38** | 0 | **0** |
| Postgres + stubbed LLM | 38 | 0 | **0** |
| Local embeddings, no Postgres | 16 | 22 | **0** |
| Core requirements only, no Postgres, no key | 8 | 30 | **0** |

The first row is the one that costs money: every example run end to end against Docker
Postgres and a live `gpt-4o-mini` key, from a cold response cache, so none of the LLM
calls were replayed. The last row is what a reader gets thirty seconds after cloning, and
is what CI enforces on every push.

`RAGKIT_STUB_LLM=1` makes every model call return canned filler. It is a testing facility,
not a way to read the examples — the mechanics are exercised faithfully and the generated
text is meaningless, so every run says so loudly.

CI runs the database-free configuration on every push, and the full 38 against a
`pgvector/pgvector:pg17` service container with the LLM stubbed. `--fail-on-skip` is set on
the full job, so a run where everything quietly declines is a red build rather than a
misleading green one.

---

## Known limits

Stated plainly, because a tutorial that hides these teaches the wrong habits.

- **The published numbers come from one model and one embedder** (`gpt-4o-mini`,
  `all-MiniLM-L6-v2`). The LLM-dependent examples are verified against a live API, but a
  different model will produce different questions, summaries and triples — and therefore
  different chunk boundaries.
- **Thirty questions on one book of Victorian detective fiction.** The rankings above are
  evidence about this corpus, not a general law. Technical documentation, with deep heading
  structure and no dialogue, would likely reorder them — which is exactly why the book
  tells you to measure your own.
- **`ts_rank` is not BM25.** There is no IDF term. Example 18 implements real BM25
  alongside it and shows where the two rankings disagree; this repo calls the Postgres arm
  "lexical" throughout.
- **The ANN index numbers are for 413 vectors.** At that size the exact scan wins and
  example 16 says so. Do not read those latencies as guidance for a corpus a thousand times
  larger.
- **Entity extraction in example 25 is a gazetteer**, not a model. It works because the
  cast is known and fixed, and it does not generalise.
- **Uncertainty sampling in example 37 separates its two groups by only a few points** —
  noise at n=30. The script reports this rather than quoting the flattering number.
- **The corpus is currently hardcoded.** `ragkit/corpus.py` reads Sherlock Holmes from a
  fixed path, so pointing the examples at your own documents means editing it. A proper
  seam for this is the next thing on the list — see [Fork this](#fork-this).

---

## Fork this

The repo is deliberately small and unclever so that it is easy to take apart. Three forks
worth making:

**1. Run it on your own corpus.** The most valuable thing you can do, and the honest
caveat is above: `ragkit/corpus.py` hardcodes the path and the heading grammar, and the 30
gold questions are Holmes-specific, so your recall numbers need your own eval set —
`scripts/build_eval_set.py` is the template to copy. If the chunking ranking comes out
different on technical documentation, that is a genuinely interesting result and
[an issue](https://github.com/chandanmaruthi/retrieval-augmented-generation-the-definitive-guide/issues/new/choose)
we want to read.

**2. Add a strategy.** One new file in `examples/`, following the same shape — header,
technique, `WHAT TO NOTICE`. The house rule is in [CONTRIBUTING.md](CONTRIBUTING.md): no
retrieval technique may hide in `ragkit/`.

**3. Break a claim.** Every number here came from one run on one machine. If
`15_compare_chunking.py` gives you a different ordering, open an issue with your output —
the Known Limits section exists because that is expected, not embarrassing.

⭐ **Star it** if the benchmark tables saved you an afternoon. 🍴 **Fork it** if you are
about to build the real thing.

---

## Attribution

Code: MIT, see [LICENSE](LICENSE).

Corpus: *The Adventures of Sherlock Holmes* by Arthur Conan Doyle, from Project Gutenberg.
Public domain in the US and most other countries. The Gutenberg header and footer are
stripped from `data/sherlock_holmes.txt`, so that file carries no Project Gutenberg
trademark or licence restriction.

Companion to **[Retrieval-Augmented Generation: The Definitive Guide](https://www.amazon.com/dp/B0HFH2JNMK)**
by Chandan Maruthi — *Technical Foundations, Architectures, and Future Directions*,
Revised and Expanded Third Edition, published by Twig AI.

<div align="center">
<br>

**[Get the book on Amazon →](https://www.amazon.com/dp/B0HFH2JNMK)**

<sub>If the code here saved you time, the book is where the reasoning behind it lives.</sub>

</div>
